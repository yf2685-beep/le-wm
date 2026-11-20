"""Stage 2 check: plot training samples (history + future frames) next to a top-down view of the
actions, to verify image/action alignment by eye. A left turn (dtheta > 0) should make the scene
slide to the right across consecutive frames.

Usage:
    python nav/viz_samples.py --h5 <dir>/train.h5 --out <reports dir> --frameskip 4 --n 12
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from navdp_dataset import NavDPDataset  # noqa: E402
from navdp_geometry import compose  # noqa: E402


def pick(ds, n, rng):
    """Half straight, half sharpest-turning clips."""
    turn = np.array([np.abs(np.nan_to_num(ds[i]["action"][:-1, 2].numpy())).sum()
                     for i in range(0, len(ds), max(1, len(ds) // 400))])
    idx = np.arange(0, len(ds), max(1, len(ds) // 400))
    order = np.argsort(turn)
    return list(rng.choice(idx[order[: len(order) // 2]], n // 2, replace=False)) + list(idx[order[-(n - n // 2):]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frameskip", type=int, default=4)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = NavDPDataset(args.h5, frameskip=args.frameskip, num_steps=4, action_rep="xyt", extra_keys=("pose",))
    rng = np.random.default_rng(0)
    for k, i in enumerate(pick(ds, args.n, rng)):
        s = ds[int(i)]
        px, act, pose = s["pixels"].numpy(), s["action"].numpy(), s["pose"].numpy()
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
        for t in range(4):
            axes[t].imshow(px[t].transpose(1, 2, 0))
            label = "target" if t == 3 else f"ctx {t}"
            a = act[t]
            txt = "" if t == 3 else f"\na=({a[0]:.3f}m, {a[1]:+.3f}m, {np.degrees(a[2]):+.1f}°)"
            axes[t].set_title(label + txt, fontsize=9)
            axes[t].axis("off")
        traj = compose(act[:3])
        ax = axes[4]
        ax.plot(-traj[:, 1], traj[:, 0], "o-", color="C0", label="composed actions")
        rel = pose - pose[0]
        c, s_ = np.cos(-pose[0, 2]), np.sin(-pose[0, 2])
        gt = np.stack([c * rel[:, 0] - s_ * rel[:, 1], s_ * rel[:, 0] + c * rel[:, 1]], -1)
        ax.plot(-gt[:, 1], gt[:, 0], "x--", color="C3", label="world pose (check)")
        for p in traj:
            ax.arrow(-p[1], p[0], -0.03 * np.sin(p[2]), 0.03 * np.cos(p[2]), head_width=0.01, color="C0")
        ax.set_aspect("equal")
        ax.set_xlabel("right (m)")
        ax.set_ylabel("forward (m)")
        ax.legend(fontsize=7)
        ax.set_title(f"clip {i}: top-down, start at origin facing up", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"sample_{k:02d}.png"), dpi=80)
        plt.close(fig)
    print("saved", args.n, "figures to", args.out)


if __name__ == "__main__":
    main()
