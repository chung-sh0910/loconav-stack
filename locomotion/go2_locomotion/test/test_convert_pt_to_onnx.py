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
