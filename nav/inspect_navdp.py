"""Stage 1 data check: per-trajectory statistics for NavDP scene tarballs.

Reads each <scene>.tar directly (no extraction) and reports, per trajectory:
frame counts (RGB vs parquet rows), per-frame step length, per-frame yaw change,
and block displacement / yaw change for candidate world-model strides.

Usage:
    python nav/inspect_navdp.py --raw /scratch/yf2685/lewm-nav/raw/matterport3d_d435i \
        --out /scratch/yf2685/lewm-nav/reports/inspect --workers 16
"""

import argparse
import io
import json
import os
import re
import tarfile
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

STRIDES = (1, 4, 8)
TRAJ_RE = re.compile(r"^(?P<scene>[^/]+)/(?P<traj>[^/]+)/(?P<rest>.+)$")


def camera_poses(df):
    """(T, 4, 4) camera-to-world poses from NavDP's misnamed `action` column."""
    return np.array([np.stack(f) for f in df["action"]], dtype=np.float64).reshape(-1, 4, 4)


def heading(poses):
    """Yaw of the camera optical axis (OpenCV z-forward) projected onto the world xy-plane."""
    fwd = poses[:, :3, 2]
    return np.arctan2(fwd[:, 1], fwd[:, 0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def traj_stats(df, n_rgb):
    poses = camera_poses(df)
    xy, yaw = poses[:, :2, 3], heading(poses)
    out = {
        "n_rows": len(df),
        "n_rgb": n_rgb,
        "z_std": float(poses[:, 2, 3].std()),
        "path_len_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
    }
    for s in STRIDES:
        if len(poses) <= s:
            continue
        idx = np.arange(0, len(poses), s)
        out[f"step_m_s{s}"] = np.linalg.norm(np.diff(xy[idx], axis=0), axis=1).tolist()
        out[f"dyaw_s{s}"] = wrap(np.diff(yaw[idx])).tolist()
    return out


def scan_tar(path):
    scene = os.path.basename(path)[: -len(".tar")]
    rgb_count = defaultdict(int)
    parquets, img_size = {}, None
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            g = TRAJ_RE.match(m.name)
            if not g:
                continue
            traj, rest = g["traj"], g["rest"]
            if "observation.images.rgb/" in rest and rest.endswith(".jpg"):
                rgb_count[traj] += 1
                if img_size is None:
                    img_size = Image.open(io.BytesIO(tf.extractfile(m).read())).size
            elif rest.endswith(".parquet"):
                parquets[traj] = pd.read_parquet(io.BytesIO(tf.extractfile(m).read()))
    trajs = {}
    for traj, df in parquets.items():
        try:
            trajs[traj] = traj_stats(df, rgb_count.get(traj, 0))
        except Exception as e:  # keep scanning; report the bad trajectory
            trajs[traj] = {"error": repr(e)}
    cols = list(next(iter(parquets.values())).columns) if parquets else []
    return {"scene": scene, "img_size": img_size, "columns": cols, "trajs": trajs}


def pct(x, qs=(0, 5, 25, 50, 75, 95, 100)):
    x = np.asarray(x)
    return {f"p{q}": round(float(np.percentile(x, q)), 4) for q in qs} if len(x) else {}


def summarize(scenes):
    T = [t for s in scenes for t in s["trajs"].values() if "error" not in t]
    errors = [(s["scene"], k, t["error"]) for s in scenes for k, t in s["trajs"].items() if "error" in t]
    summary = {
        "n_scenes": len(scenes),
        "n_trajs": len(T),
        "n_errors": len(errors),
        "errors": errors[:20],
        "img_sizes": sorted({str(s["img_size"]) for s in scenes}),
        "parquet_columns": sorted({c for s in scenes for c in s["columns"]}),
        "frames_per_traj": pct([t["n_rows"] for t in T]),
        "total_frames": int(sum(t["n_rows"] for t in T)),
        "rgb_mismatch_trajs": sum(t["n_rows"] != t["n_rgb"] for t in T),
        "path_len_m": pct([t["path_len_m"] for t in T]),
        "max_z_std_m": round(max(t["z_std"] for t in T), 5),
    }
    for s in STRIDES:
        steps = np.concatenate([t.get(f"step_m_s{s}", []) for t in T])
        dyaw = np.degrees(np.abs(np.concatenate([t.get(f"dyaw_s{s}", []) for t in T])))
        summary[f"stride{s}"] = {
            "n_transitions": int(len(steps)),
            "step_m": pct(steps),
            "abs_dyaw_deg": pct(dyaw),
            "frac_abs_dyaw_gt_5deg": round(float((dyaw > 5).mean()), 4),
            "frac_abs_dyaw_gt_15deg": round(float((dyaw > 15).mean()), 4),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-scenes", type=int, default=None)
    args = ap.parse_args()

    tars = sorted(os.path.join(args.raw, f) for f in os.listdir(args.raw) if f.endswith(".tar"))
    tars = tars[: args.max_scenes]
    with Pool(args.workers) as pool:
        scenes = pool.map(scan_tar, tars)

    os.makedirs(args.out, exist_ok=True)
    summary = summarize(scenes)
    per_traj = [
        {"scene": s["scene"], "traj": k, **{f: t[f] for f in ("n_rows", "n_rgb", "path_len_m") if f in t}}
        for s in scenes for k, t in s["trajs"].items()
    ]
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(per_traj).to_csv(os.path.join(args.out, "per_traj.csv"), index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
