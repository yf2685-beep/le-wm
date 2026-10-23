"""Stage 2: convert NavDP (InternData-N1 vln_n1, LeRobot-style) trajectories to LeWM HDF5 files.

One HDF5 file per split (train / val / test), split by *scene*. Layout matches
stable_worldmodel's HDF5Dataset (per-column datasets + ep_len / ep_offset), one row per raw frame:

  pixels    (224, 224, 3) uint8   RGB, longer side resized to 224 and zero-padded (as in NavDP)
  act_f     (3,) float32          frame i -> i+1 local [dx, dy, dtheta]   (NaN on the last frame)
  act_s4    (3,) float32          frame i -> i+4 local [dx, dy, dtheta]   (NaN on the last 4 frames)
  act_s8    (3,) float32          frame i -> i+8 local [dx, dy, dtheta]
  pose      (3,) float32          world [x, y, yaw] of the robot base (debugging / probes)
  scene_idx, traj_idx  int32      provenance

Action representations (xy / xyt / xysc / disc) are derived from act_s* at load time
(nav/navdp_dataset.py), so thresholds can change without re-converting.

Sources: either a directory of extracted scenes (<src>/<scene>/<trajectory_k>/...) as on 127,
or a directory of <scene>.tar files as on Torch.

Usage:
    python nav/convert_navdp.py --src <raw dir> --out <h5 dir> --workers 12 [--max-scenes 2]
"""

import argparse
import io
import json
import os
import random
import re
import tarfile
import time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from navdp_geometry import base_poses, local_displacement, world_xyyaw

IMG = 224
STRIDES = (4, 8)
RGB_DIR = "videos/chunk-000/observation.images.rgb"
PARQUET = "data/chunk-000/episode_000000.parquet"


# ---------------------------------------------------------------- reading

def list_scenes(src):
    names = sorted(os.listdir(src))
    tars = [n[:-4] for n in names if n.endswith(".tar")]
    return tars if tars else [n for n in names if os.path.isdir(os.path.join(src, n))]


def iter_trajectories(src, scene):
    """Yield (traj_name, parquet DataFrame, [jpeg bytes sorted by frame index])."""
    tar_path = os.path.join(src, scene + ".tar")
    if os.path.isfile(tar_path):
        parquets, rgbs = {}, {}
        with tarfile.open(tar_path) as tf:
            for m in tf:
                if not m.isfile():
                    continue
                parts = m.name.split("/", 2)
                if len(parts) < 3:
                    continue
                traj, rest = parts[1], parts[2]
                if rest == PARQUET:
                    parquets[traj] = pd.read_parquet(io.BytesIO(tf.extractfile(m).read()))
                elif rest.startswith(RGB_DIR + "/") and rest.endswith(".jpg"):
                    rgbs.setdefault(traj, {})[int(os.path.basename(rest)[:-4])] = tf.extractfile(m).read()
        for traj in sorted(parquets, key=natural_key):
            frames = rgbs.get(traj, {})
            yield traj, parquets[traj], [frames[k] for k in sorted(frames)]
    else:
        scene_dir = os.path.join(src, scene)
        for traj in sorted(os.listdir(scene_dir), key=natural_key):
            pq = os.path.join(scene_dir, traj, PARQUET)
            rgb_dir = os.path.join(scene_dir, traj, RGB_DIR)
            if not (os.path.isfile(pq) and os.path.isdir(rgb_dir)):
                continue
            files = sorted((f for f in os.listdir(rgb_dir) if f.endswith(".jpg")), key=lambda f: int(f[:-4]))
            frames = []
            for f in files:
                with open(os.path.join(rgb_dir, f), "rb") as fh:
                    frames.append(fh.read())
            yield traj, pd.read_parquet(pq), frames


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def letterbox(jpeg_bytes):
    """Resize the longer side to IMG and zero-pad to IMG x IMG (same as NavDP's process_image)."""
    im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    w, h = im.size
    s = IMG / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.BILINEAR)
    out = np.zeros((IMG, IMG, 3), dtype=np.uint8)
    top, left = (IMG - nh) // 2, (IMG - nw) // 2
    out[top: top + nh, left: left + nw] = np.asarray(im)
    return out


# ---------------------------------------------------------------- per-scene work

def convert_scene(job):
    src, scene, scene_idx, max_trajs = job
    episodes, skipped = [], []
    for traj_idx, (traj, df, frames) in enumerate(iter_trajectories(src, scene)):
        if max_trajs is not None and traj_idx >= max_trajs:
            break
        n = len(df)
        if n != len(frames) or n < 2:
            skipped.append((traj, n, len(frames)))
            continue
        cam = np.array([np.stack(f) for f in df["action"]], dtype=np.float64).reshape(-1, 4, 4)
        ext = np.vstack(np.array(df["observation.camera_extrinsic"].tolist()[0])).reshape(4, 4)
        poses = base_poses(cam, ext)
        ep = {
            "pixels": np.stack([letterbox(b) for b in frames]),
            "act_f": local_displacement(poses, 1).astype(np.float32),
            "pose": world_xyyaw(poses).astype(np.float32),
            "scene_idx": np.full(n, scene_idx, dtype=np.int32),
            "traj_idx": np.full(n, traj_idx, dtype=np.int32),
        }
        for s in STRIDES:
            ep[f"act_s{s}"] = local_displacement(poses, s).astype(np.float32)
        episodes.append((traj, ep))
    return scene, episodes, skipped


# ---------------------------------------------------------------- writing

class H5Writer:
    """HDF5 writer compatible with stable_worldmodel.HDF5Dataset, with lzf compression on pixels."""

    def __init__(self, path):
        self.f = h5py.File(path, "w", libver="latest")
        self.ptr, self.n_eps = 0, 0
        self.f.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32)
        self.f.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64)

    def write(self, ep):
        n = len(ep["pixels"])
        for col, arr in ep.items():
            if col not in self.f:
                big = arr.ndim == 4
                self.f.create_dataset(
                    col, shape=(0, *arr.shape[1:]), maxshape=(None, *arr.shape[1:]), dtype=arr.dtype,
                    chunks=(1, *arr.shape[1:]) if big else (4096, *arr.shape[1:]),
                    compression="lzf" if big else None,
                )
            ds = self.f[col]
            ds.resize(self.ptr + n, axis=0)
            ds[self.ptr: self.ptr + n] = arr
        for name, val in (("ep_len", n), ("ep_offset", self.ptr)):
            self.f[name].resize(self.n_eps + 1, axis=0)
            self.f[name][self.n_eps] = val
        self.ptr += n
        self.n_eps += 1

    def close(self):
        self.f.close()


def make_splits(scenes, seed=0, frac=(0.8, 0.1, 0.1)):
    order = scenes[:]
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(order) * frac[1]))
    n_test = max(1, round(len(order) * frac[2])) if len(order) > 2 else 0
    n_train = len(order) - n_val - n_test
    return {
        "seed": seed,
        "train": sorted(order[:n_train]),
        "val": sorted(order[n_train: n_train + n_val]),
        "test": sorted(order[n_train + n_val:]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=None, help="splits.json (created in --out if missing)")
    ap.add_argument("--only", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-scenes", type=int, default=None, help="debug: first N scenes of each split")
    ap.add_argument("--max-trajs", type=int, default=None, help="debug: first N trajectories per scene")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scenes = list_scenes(args.src)
    scene_idx = {s: i for i, s in enumerate(scenes)}
    splits_path = args.splits or os.path.join(args.out, "splits.json")
    if os.path.isfile(splits_path):
        splits = json.load(open(splits_path))
    else:
        splits = make_splits(scenes)
        json.dump(splits, open(splits_path, "w"), indent=2)
    print(f"{len(scenes)} scenes; split sizes", {k: len(splits[k]) for k in ("train", "val", "test")}, flush=True)

    for split in args.only:
        names = splits[split][: args.max_scenes]
        tmp, final = os.path.join(args.out, f"{split}.h5.part"), os.path.join(args.out, f"{split}.h5")
        writer, t0 = H5Writer(tmp), time.time()
        skipped_all, act_stats = [], {f"act_s{s}": [] for s in STRIDES}
        jobs = [(args.src, s, scene_idx[s], args.max_trajs) for s in names]
        with Pool(args.workers) as pool:
            for k, (scene, episodes, skipped) in enumerate(pool.imap(convert_scene, jobs)):
                for _, ep in episodes:
                    writer.write(ep)
                    for key in act_stats:
                        act_stats[key].append(ep[key][~np.isnan(ep[key]).any(1)])
                skipped_all += [(scene, *s) for s in skipped]
                print(f"[{split}] {k + 1}/{len(names)} {scene}: {len(episodes)} trajs, "
                      f"{writer.ptr} frames total, {time.time() - t0:.0f}s", flush=True)
        n_eps, n_frames = writer.n_eps, writer.ptr
        writer.close()
        os.replace(tmp, final)

        meta = {"split": split, "scenes": names, "n_episodes": n_eps, "n_frames": n_frames,
                "skipped_trajs": skipped_all, "seconds": round(time.time() - t0)}
        for key, chunks in act_stats.items():
            a = np.concatenate(chunks) if chunks else np.zeros((0, 3))
            meta[key] = {
                "mean": a.mean(0).round(5).tolist(), "std": a.std(0).round(5).tolist(),
                "abs_dtheta_deg_pct": {p: round(float(np.degrees(np.percentile(np.abs(a[:, 2]), p))), 3)
                                       for p in (50, 75, 90, 95, 99)} if len(a) else {},
                "step_m_pct": {p: round(float(np.percentile(np.linalg.norm(a[:, :2], axis=1), p)), 4)
                               for p in (1, 5, 50, 95)} if len(a) else {},
            }
        json.dump(meta, open(os.path.join(args.out, f"{split}_meta.json"), "w"), indent=2)
        print(f"[{split}] wrote {final}: {n_eps} episodes, {n_frames} frames, skipped {len(skipped_all)}", flush=True)


if __name__ == "__main__":
    main()
