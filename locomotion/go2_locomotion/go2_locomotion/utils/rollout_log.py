"""Shared rollout log schema for sim2real comparison (dependency-light: csv/json + numpy)."""
import csv
import json
import math

import numpy as np

SCHEMA_VERSION = 1

_JOINT_GROUPS = ["raw_action", "target_q", "q", "dq", "tau"]
COLUMNS = (
    ["step", "t", "cmd_vx", "cmd_vy", "cmd_vyaw"]
    + [f"{g}_{i}" for g in _JOINT_GROUPS for i in range(12)]
    + ["base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z",
       "base_roll", "base_pitch",
       "base_gyro_x", "base_gyro_y", "base_gyro_z",
       "base_height", "base_vx", "base_vy", "base_vyaw"]
)


def quat_to_roll_pitch(w, x, y, z):
    """Roll (about x) and pitch (about y) from a (w,x,y,z) quaternion, radians."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return roll, pitch


def make_row(step, t, cmd, raw_action, target_q, q, dq, tau, quat, gyro,
             base_height=float("nan"), base_vx=float("nan"),
             base_vy=float("nan"), base_vyaw=float("nan")):
    roll, pitch = quat_to_roll_pitch(quat[0], quat[1], quat[2], quat[3])
    row = {
        "step": step, "t": t,
        "cmd_vx": cmd[0], "cmd_vy": cmd[1], "cmd_vyaw": cmd[2],
        "base_quat_w": quat[0], "base_quat_x": quat[1],
        "base_quat_y": quat[2], "base_quat_z": quat[3],
        "base_roll": roll, "base_pitch": pitch,
        "base_gyro_x": gyro[0], "base_gyro_y": gyro[1], "base_gyro_z": gyro[2],
        "base_height": base_height, "base_vx": base_vx,
        "base_vy": base_vy, "base_vyaw": base_vyaw,
    }
    for name, arr in zip(_JOINT_GROUPS, [raw_action, target_q, q, dq, tau]):
        for i in range(12):
            row[f"{name}_{i}"] = float(arr[i])
    return row


def write_log(base_path, rows, metadata):
    with open(base_path + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    meta = dict(metadata)
    meta.setdefault("version", SCHEMA_VERSION)
    meta["n_steps"] = len(rows)
    with open(base_path + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def read_log(base_path):
    with open(base_path + ".json") as f:
        metadata = json.load(f)
    data = {name: [] for name in COLUMNS}
    with open(base_path + ".csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name in COLUMNS:
                data[name].append(float(row[name]))
    cols = {name: np.array(vals, dtype=float) for name, vals in data.items()}
    return metadata, cols


class RolloutLogger:
    """In-memory buffer; O(1) append, single write on flush (keeps the 50 Hz loop clean)."""

    def __init__(self, base_path, metadata):
        self._base_path = base_path
        self._metadata = dict(metadata)
        self._rows = []

    def append(self, row):
        self._rows.append(row)

    def set_meta(self, key, value):
        self._metadata[key] = value

    def flush(self):
        write_log(self._base_path, self._rows, self._metadata)
