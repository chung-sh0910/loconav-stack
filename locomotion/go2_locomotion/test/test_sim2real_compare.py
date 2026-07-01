import importlib.util, os, sys
import numpy as np

_MOD = os.path.join(os.path.dirname(__file__), "..", "tools", "sim2real_compare.py")
_spec = importlib.util.spec_from_file_location("sim2real_compare", _MOD)
s2r = importlib.util.module_from_spec(_spec)
sys.modules["sim2real_compare"] = s2r
_spec.loader.exec_module(s2r)


def test_rmse_basic():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert abs(s2r.rmse(a, b) - np.sqrt((9 + 16 + 0) / 3)) < 1e-9


def test_rmse_ignores_nan():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, 5.0, 3.0])
    assert s2r.rmse(a, b) == 0.0   # only finite pairs (indices 0,2) count


def test_rmse_all_nan_returns_nan():
    a = np.array([np.nan, np.nan])
    b = np.array([1.0, 2.0])
    assert np.isnan(s2r.rmse(a, b))


def test_align_lengths_truncates_to_min():
    real = {"step": np.arange(10.0)}
    sim = {"step": np.arange(7.0)}
    assert s2r.align_lengths(real, sim) == 7


def test_gap_table_offset_case():
    n = 50
    real = {f"q_{i}": np.zeros(n) for i in range(12)}
    sim = {f"q_{i}": np.zeros(n) for i in range(12)}
    real.update({f"tau_{i}": np.zeros(n) for i in range(12)})
    sim.update({f"tau_{i}": np.zeros(n) for i in range(12)})
    real["base_roll"] = np.zeros(n); sim["base_roll"] = np.zeros(n)
    real["base_pitch"] = np.zeros(n); sim["base_pitch"] = np.zeros(n)
    # constant 0.1 offset on q_3
    sim["q_3"] = np.full(n, 0.1)
    rows = s2r.gap_table(real, sim, n_common=n, steady_start=20)
    by_sig = {r["signal"]: r for r in rows}
    assert abs(by_sig["q_3"]["rmse_full"] - 0.1) < 1e-9
    assert abs(by_sig["q_0"]["rmse_full"]) < 1e-12


def test_validate_metadata_ok():
    m = {"joint_sdk_names": ["j%d" % i for i in range(12)], "version": 1}
    s2r.validate_metadata(dict(m), dict(m))  # must not raise


def test_validate_metadata_joint_mismatch_raises():
    import pytest
    a = {"joint_sdk_names": ["a"] * 12, "version": 1}
    b = {"joint_sdk_names": ["b"] * 12, "version": 1}
    with pytest.raises(ValueError):
        s2r.validate_metadata(a, b)


def test_validate_metadata_version_mismatch_warns(capsys):
    m1 = {"joint_sdk_names": ["j"] * 12, "version": 1}
    m2 = {"joint_sdk_names": ["j"] * 12, "version": 2}
    s2r.validate_metadata(m1, m2)  # must not raise
    assert "version" in capsys.readouterr().out.lower()
