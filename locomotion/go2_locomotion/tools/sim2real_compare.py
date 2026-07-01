#!/usr/bin/env python3
"""Compare a real rollout log against an Isaac Lab replay log (dynamics-gap analysis).

Usage:
    python3 tools/sim2real_compare.py --real real_log --sim sim_log --out out_dir
(paths are the base name without .csv/.json)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from go2_locomotion.utils import rollout_log as rl


def rmse(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def align_lengths(real_cols, sim_cols):
    return int(min(len(real_cols["step"]), len(sim_cols["step"])))


def validate_metadata(real_meta, sim_meta):
    """Guard against comparing logs with mismatched joint order or schema version."""
    r = real_meta.get("joint_sdk_names")
    s = sim_meta.get("joint_sdk_names")
    if r != s:
        raise ValueError(
            f"joint_sdk_names mismatch between real and sim logs — joints would be "
            f"misaligned.\n  real: {r}\n  sim:  {s}"
        )
    if real_meta.get("version") != sim_meta.get("version"):
        print(f"warning: schema version mismatch (real={real_meta.get('version')} "
              f"sim={sim_meta.get('version')}) — columns may differ")


def _signals():
    joints = [f"q_{i}" for i in range(12)] + [f"tau_{i}" for i in range(12)]
    return joints + ["base_roll", "base_pitch"]


def gap_table(real_cols, sim_cols, n_common, steady_start=20):
    rows = []
    for sig in _signals():
        r = real_cols[sig][:n_common]
        s = sim_cols[sig][:n_common]
        rows.append({
            "signal": sig,
            "rmse_full": rmse(r, s),
            "rmse_steady": rmse(r[steady_start:], s[steady_start:]),
        })
    return rows


def _plot_group(names, titles, real_cols, sim_cols, n, out_path, extra=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 3
    nrow = (len(names) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    t = np.arange(n)
    for idx, (name, title) in enumerate(zip(names, titles)):
        ax = axes[idx // ncol][idx % ncol]
        if extra and name in extra:
            ax.plot(t, extra[name][:n], "k--", lw=0.8, label=extra["_label"])
        ax.plot(t, real_cols[name][:n], "b", lw=0.9, label="real")
        ax.plot(t, sim_cols[name][:n], "r", lw=0.9, label="sim")
        ax.set_title(title); ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True)
    p.add_argument("--sim", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steady-start", type=int, default=20)
    args = p.parse_args()

    real_meta, real_cols = rl.read_log(args.real)
    sim_meta, sim_cols = rl.read_log(args.sim)
    validate_metadata(real_meta, sim_meta)
    n = align_lengths(real_cols, sim_cols)
    if len(real_cols["step"]) != len(sim_cols["step"]):
        print(f"경고: 길이 불일치 real={len(real_cols['step'])} sim={len(sim_cols['step'])} → {n}로 truncate")

    os.makedirs(args.out, exist_ok=True)

    jn = [f"q_{i}" for i in range(12)]
    tgt = {f"q_{i}": real_cols[f"target_q_{i}"] for i in range(12)}
    tgt["_label"] = "target"
    _plot_group(jn, jn, real_cols, sim_cols, n, os.path.join(args.out, "joints.png"), extra=tgt)

    tn = [f"tau_{i}" for i in range(12)]
    _plot_group(tn, tn, real_cols, sim_cols, n, os.path.join(args.out, "torque.png"))

    bn = ["base_roll", "base_pitch", "base_gyro_x", "base_gyro_y", "base_gyro_z"]
    _plot_group(bn, bn, real_cols, sim_cols, n, os.path.join(args.out, "base.png"))

    rows = gap_table(real_cols, sim_cols, n, args.steady_start)
    print(f"{'signal':<14}{'rmse_full':>12}{'rmse_steady':>14}")
    with open(os.path.join(args.out, "gap_summary.csv"), "w") as f:
        f.write("signal,rmse_full,rmse_steady\n")
        for r in rows:
            print(f"{r['signal']:<14}{r['rmse_full']:>12.5f}{r['rmse_steady']:>14.5f}")
            f.write(f"{r['signal']},{r['rmse_full']},{r['rmse_steady']}\n")


if __name__ == "__main__":
    main()
