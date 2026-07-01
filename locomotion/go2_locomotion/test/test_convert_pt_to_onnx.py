import sys
import os
import importlib.util

import torch
import pytest

_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools")
_MODULE_PATH = os.path.join(_TOOLS_DIR, "convert_pt_to_onnx.py")
_spec = importlib.util.spec_from_file_location("convert_pt_to_onnx", _MODULE_PATH)
convert_pt_to_onnx = importlib.util.module_from_spec(_spec)
sys.modules["convert_pt_to_onnx"] = convert_pt_to_onnx
_spec.loader.exec_module(convert_pt_to_onnx)


def _gru_state_dict(input_size=45, hidden_size=256, num_layers=1):
    sd = {}
    in_dim = input_size
    for layer in range(num_layers):
        sd[f"memory_a.rnn.weight_ih_l{layer}"] = torch.zeros(3 * hidden_size, in_dim)
        sd[f"memory_a.rnn.weight_hh_l{layer}"] = torch.zeros(3 * hidden_size, hidden_size)
        sd[f"memory_a.rnn.bias_ih_l{layer}"] = torch.zeros(3 * hidden_size)
        sd[f"memory_a.rnn.bias_hh_l{layer}"] = torch.zeros(3 * hidden_size)
        in_dim = hidden_size
    return sd


def _lstm_state_dict(input_size=45, hidden_size=128, num_layers=2):
    sd = {}
    in_dim = input_size
    for layer in range(num_layers):
        sd[f"memory_a.rnn.weight_ih_l{layer}"] = torch.zeros(4 * hidden_size, in_dim)
        sd[f"memory_a.rnn.weight_hh_l{layer}"] = torch.zeros(4 * hidden_size, hidden_size)
        sd[f"memory_a.rnn.bias_ih_l{layer}"] = torch.zeros(4 * hidden_size)
        sd[f"memory_a.rnn.bias_hh_l{layer}"] = torch.zeros(4 * hidden_size)
        in_dim = hidden_size
    return sd


def test_infer_gru_single_layer():
    sd = _gru_state_dict(input_size=45, hidden_size=256, num_layers=1)
    arch = convert_pt_to_onnx.infer_rnn_arch(sd)
    assert arch == {"rnn_type": "gru", "hidden_size": 256, "num_layers": 1, "input_size": 45}


def test_infer_lstm_two_layers():
    sd = _lstm_state_dict(input_size=45, hidden_size=128, num_layers=2)
    arch = convert_pt_to_onnx.infer_rnn_arch(sd)
    assert arch == {"rnn_type": "lstm", "hidden_size": 128, "num_layers": 2, "input_size": 45}


def test_infer_rnn_arch_missing_key_raises():
    with pytest.raises(ValueError):
        convert_pt_to_onnx.infer_rnn_arch({"actor.0.weight": torch.zeros(1, 1)})


def test_infer_rnn_arch_bad_gate_ratio_raises():
    sd = {
        "memory_a.rnn.weight_ih_l0": torch.zeros(500, 45),  # 500/256 is not 3 or 4
        "memory_a.rnn.weight_hh_l0": torch.zeros(500, 256),
    }
    with pytest.raises(ValueError):
        convert_pt_to_onnx.infer_rnn_arch(sd)


def test_recurrent_actor_gru_forward_shapes():
    model = convert_pt_to_onnx.RecurrentActor(
        rnn_type="gru", input_size=45, hidden_size=256, num_layers=1,
        action_dim=12, actor_hidden_dims=[512, 256, 128],
    )
    model.eval()
    obs = torch.zeros(1, 45)
    h_in = torch.zeros(1, 1, 256)
    actions, h_out = model.forward_gru(obs, h_in)
    assert actions.shape == (1, 12)
    assert h_out.shape == (1, 1, 256)


def test_recurrent_actor_lstm_forward_shapes():
    model = convert_pt_to_onnx.RecurrentActor(
        rnn_type="lstm", input_size=45, hidden_size=128, num_layers=2,
        action_dim=12, actor_hidden_dims=[512, 256, 128],
    )
    model.eval()
    obs = torch.zeros(1, 45)
    h_in = torch.zeros(2, 1, 128)
    c_in = torch.zeros(2, 1, 128)
    actions, h_out, c_out = model.forward_lstm(obs, h_in, c_in)
    assert actions.shape == (1, 12)
    assert h_out.shape == (2, 1, 128)
    assert c_out.shape == (2, 1, 128)


def test_load_recurrent_weights_matches_manual_rnn_step():
    torch.manual_seed(0)
    sd = _gru_state_dict(input_size=45, hidden_size=256, num_layers=1)
    # give actor head real weights too (matches build_actor([512,256,128]) for hidden_size=256 in, 12 out)
    sd["actor.0.weight"] = torch.randn(512, 256)
    sd["actor.0.bias"] = torch.randn(512)
    sd["actor.2.weight"] = torch.randn(256, 512)
    sd["actor.2.bias"] = torch.randn(256)
    sd["actor.4.weight"] = torch.randn(128, 256)
    sd["actor.4.bias"] = torch.randn(128)
    sd["actor.6.weight"] = torch.randn(12, 128)
    sd["actor.6.bias"] = torch.randn(12)
    for k in list(sd.keys()):
        if k.startswith("memory_a.rnn."):
            sd[k] = torch.randn_like(sd[k])

    model = convert_pt_to_onnx.RecurrentActor(
        rnn_type="gru", input_size=45, hidden_size=256, num_layers=1,
        action_dim=12, actor_hidden_dims=[512, 256, 128],
    )
    convert_pt_to_onnx.load_recurrent_weights(model, sd)
    model.eval()

    obs = torch.randn(1, 45)
    h_in = torch.zeros(1, 1, 256)
    actions, h_out = model.forward_gru(obs, h_in)

    # reference: run torch.nn.GRU directly with the same weights
    ref_rnn = torch.nn.GRU(input_size=45, hidden_size=256, num_layers=1)
    ref_rnn.load_state_dict(
        {k[len("memory_a.rnn."):]: v for k, v in sd.items() if k.startswith("memory_a.rnn.")}
    )
    ref_rnn.eval()
    ref_out, ref_h = ref_rnn(obs.unsqueeze(0), h_in)
    assert torch.allclose(h_out, ref_h, atol=1e-6)
    assert actions.shape == (1, 12)
