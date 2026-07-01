import math
import numpy as np
from go2_locomotion.utils import rollout_log as rl


def test_columns_shape():
    # 5 scalars + 12*5 joint cols + 13 base cols
    assert rl.COLUMNS[:5] == ["step", "t", "cmd_vx", "cmd_vy", "cmd_vyaw"]
    assert "raw_action_11" in rl.COLUMNS
    assert "target_q_0" in rl.COLUMNS and "tau_11" in rl.COLUMNS
    assert rl.COLUMNS[-4:] == ["base_height", "base_vx", "base_vy", "base_vyaw"]
    assert len(rl.COLUMNS) == 5 + 12 * 5 + 13


def test_quat_to_roll_pitch_level():
    roll, pitch = rl.quat_to_roll_pitch(1.0, 0.0, 0.0, 0.0)
    assert abs(roll) < 1e-6 and abs(pitch) < 1e-6


def test_quat_to_roll_pitch_known_roll():
    # 90 deg roll about x: q = (cos45, sin45, 0, 0)
    c = math.cos(math.pi / 4)
    roll, pitch = rl.quat_to_roll_pitch(c, c, 0.0, 0.0)
    assert abs(roll - math.pi / 2) < 1e-6


def test_make_row_keys_and_values():
    row = rl.make_row(
        step=3, t=0.06, cmd=(0.5, 0.0, 0.1),
        raw_action=list(range(12)), target_q=[0.1] * 12, q=[0.2] * 12,
        dq=[0.3] * 12, tau=[0.4] * 12, quat=(1.0, 0.0, 0.0, 0.0), gyro=(0.01, 0.02, 0.03),
    )
    assert set(row.keys()) == set(rl.COLUMNS)
    assert row["step"] == 3 and row["cmd_vx"] == 0.5
    assert row["raw_action_5"] == 5 and row["q_0"] == 0.2 and row["tau_11"] == 0.4
    assert math.isnan(row["base_vx"]) and math.isnan(row["base_height"])


def test_write_read_roundtrip(tmp_path):
    rows = [
        rl.make_row(step=i, t=i * 0.02, cmd=(0.5, 0.0, 0.0),
                    raw_action=[i + 0.1 * j for j in range(12)],
                    target_q=[0.1] * 12, q=[0.2] * 12, dq=[0.3] * 12, tau=[0.4] * 12,
                    quat=(1.0, 0.0, 0.0, 0.0), gyro=(0.0, 0.0, 0.0))
        for i in range(4)
    ]
    meta = {"source": "real", "version": rl.SCHEMA_VERSION, "dt": 0.02,
            "joint_sdk_names": ["j%d" % i for i in range(12)]}
    base = str(tmp_path / "log")
    rl.write_log(base, rows, meta)

    meta2, cols = rl.read_log(base)
    assert meta2["source"] == "real" and meta2["version"] == rl.SCHEMA_VERSION
    assert cols["step"].tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(cols["raw_action_5"], [0.5, 1.5, 2.5, 3.5])


def test_rollout_logger_append_flush(tmp_path):
    base = str(tmp_path / "roll")
    logger = rl.RolloutLogger(base, {"source": "real", "version": rl.SCHEMA_VERSION})
    for i in range(3):
        logger.append(rl.make_row(step=i, t=i * 0.02, cmd=(0.0, 0.0, 0.0),
                                   raw_action=[0.0] * 12, target_q=[0.0] * 12, q=[0.0] * 12,
                                   dq=[0.0] * 12, tau=[0.0] * 12, quat=(1.0, 0.0, 0.0, 0.0),
                                   gyro=(0.0, 0.0, 0.0)))
    logger.set_meta("initial_q", [0.1] * 12)
    logger.flush()
    meta, cols = rl.read_log(base)
    assert len(cols["step"]) == 3
    assert meta["n_steps"] == 3
    assert meta["initial_q"] == [0.1] * 12
