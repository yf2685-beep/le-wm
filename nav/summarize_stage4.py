"""Stage 4 summary: aggregate eval_val.json of all stage-4 runs into tables and rollout plots.

Groups runs by (action_rep, frameskip, history_size); L1-loss runs (s4l1_*) appear as "<rep>+l1" and reports mean +- std over seeds of:
  * model MSE at k=1 and at matched distances (~0.3 m, ~0.6 m, ~1.2 m), with copy-last / zero / shuffled baselines
  * action reliance: (zero-action MSE - model MSE) / model MSE at k=1   (> 0 means actions help)
  * collapse: effective rank, min per-dim std
  * linear probe R^2 of [dx, dy, dtheta]

Usage:
    python nav/summarize_stage4.py --runs /scratch/yf2685/lewm-nav/runs --out /scratch/yf2685/lewm-nav/reports/stage4
"""

import argparse
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAME = re.compile(r"^s4(?P<tag>diag|l1)?_(?P<rep>[a-z]+)_fs(?P<fs>\d+)_h(?P<h>\d+)_seed(?P<seed>\d+)$")
TARGET_M = (0.3, 0.6, 1.2)


def at_metres(res, series, m):
    """Value of a per-k series at the step closest to m metres."""
    k = int(np.argmin(np.abs(np.array(res["metres_at_k"]) - m)))
    return series[k], res["metres_at_k"][k]


def load(runs):
    groups = defaultdict(list)
    for name in sorted(os.listdir(runs)):
        g = NAME.match(name)
        path = os.path.join(runs, name, "eval_val.json")
        if g and os.path.isfile(path):
            rep = g["rep"] + ("+l1" if g["tag"] == "l1" else "")  # s4l1_*: trained with L1 prediction loss
            groups[(rep, int(g["fs"]), int(g["h"]))].append((int(g["seed"]), json.load(open(path))))
    return groups


def ms(xs):
    xs = np.asarray(xs, dtype=float)
    return f"{xs.mean():.4f} ± {xs.std():.4f}" if len(xs) > 1 else f"{xs.mean():.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    groups = load(args.runs)

    rows, summary = [], {}
    header = ["rep", "fs", "H", "n_seeds", "mse@k1", "copylast@k1", "zero_act@k1", "action_gain@k1"]
    header += [f"mse@{m}m" for m in TARGET_M] + [f"copylast@{m}m" for m in TARGET_M]
    header += ["eff_rank", "min_std", "probe_dx", "probe_dy", "probe_dth"]
    for (rep, fs, h), runs in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        R = [r for _, r in runs]
        mod = [r["rollout"]["model"]["mse"] for r in R]
        cl = [r["rollout"]["copylast"]["mse"] for r in R]
        zero = [r["rollout"]["zero_action"]["mse"] for r in R]
        row = [rep, fs, h, len(R), ms([m[0] for m in mod]), ms([c[0] for c in cl]), ms([z[0] for z in zero]),
               ms([(z[0] - m[0]) / m[0] for z, m in zip(zero, mod)])]
        row += [ms([at_metres(r, m, t)[0] for r, m in zip(R, mod)]) for t in TARGET_M]
        row += [ms([at_metres(r, c, t)[0] for r, c in zip(R, cl)]) for t in TARGET_M]
        row += [ms([r["collapse"]["effective_rank"] for r in R]), ms([r["collapse"]["emb_std_min"] for r in R])]
        row += [ms([r["probe_r2_dx_dy_dtheta"][i] for r in R]) for i in range(3)]
        rows.append(row)
        summary[f"{rep}_fs{fs}_h{h}"] = {
            "seeds": [s for s, _ in runs],
            "metres_at_k": R[0]["metres_at_k"],
            **{n: np.mean([r["rollout"][n]["mse"] for r in R], 0).tolist()
               for n in ("model", "copylast", "zero_action", "shuffled_action")},
        }

    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    open(os.path.join(args.out, "stage4_table.md"), "w").write("\n".join(md) + "\n")
    json.dump(summary, open(os.path.join(args.out, "stage4_curves.json"), "w"), indent=1)

    # rollout error vs metres, one panel per history size
    for h in sorted({k[2] for k in groups}):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, (key, s) in enumerate(sorted((k, v) for k, v in summary.items() if k.endswith(f"_h{h}"))):
            style = "-" if "_fs4_" in key else "--"
            ax.plot(s["metres_at_k"], s["model"], style, marker="o", ms=3, color=f"C{i % 10}", label=key)
        for key, s in summary.items():
            if key.endswith(f"_h{h}"):
                ax.plot(s["metres_at_k"], s["copylast"], ":", color="grey", lw=1)
        ax.set_xlabel("prediction distance (m)")
        ax.set_ylabel("latent MSE (val)")
        ax.set_title(f"Stage 4 rollout error, history={h} (grey dotted: copy-last)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"stage4_rollout_h{h}.png"), dpi=110)
        plt.close(fig)
    print("\n".join(md))
    print("saved to", args.out)


if __name__ == "__main__":
    main()
