"""Stage 4 evaluation: latent prediction quality of a trained nav LeWM run.

For held-out clips (default: val split) it reports, per rollout step k = 1..K:
  * model rollout error vs. the true future embedding (MSE, L1, cosine distance, normalised MSE)
  * baselines: copy-last (z_hat_{t+k} = z_t), zero actions, actions shuffled within the batch
and once per run:
  * collapse checks on encoder embeddings: per-dim std, effective rank, SIGReg statistic
  * a linear (ridge) probe predicting the true block action [dx, dy, dtheta] from (z_t, z_{t+1})

Rollout: the context holds the last `history_size` (predicted or true) embeddings; step k uses actions
a_{t-H+k .. t+k-1}, exactly as JEPA.rollout. Physical distance per step is reported so strides can be
compared at equal metres.

Usage:
    python nav/eval_latent.py --run <run_dir> [--split val] [--max-k 8] [--max-clips 20000]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nav")]

from module import SIGReg  # noqa: E402
from train_nav import build_datasets  # noqa: E402  (also registers the `eval` resolver)


def effective_rank(z):
    z = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(z.double())
    p = s ** 2 / (s ** 2).sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))


def ridge_r2(x, y, lam=1e-2):
    """Fit on the first half, report per-dimension R^2 on the second half."""
    n = len(x) // 2
    xm, ym = x[:n].mean(0), y[:n].mean(0)
    xs = x[:n].std(0) + 1e-6
    a, b = (x[:n] - xm) / xs, y[:n] - ym
    w = torch.linalg.solve(a.T @ a + lam * n * torch.eye(a.shape[1], dtype=a.dtype), a.T @ b)
    pred = ((x[n:] - xm) / xs) @ w + ym
    ss_res = ((y[n:] - pred) ** 2).sum(0)
    ss_tot = ((y[n:] - y[n:].mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).tolist()


@torch.no_grad()
def rollout(model, emb_ctx, act_emb, H, K):
    """emb_ctx: (B, H, D) true context; act_emb: (B, H+K-1, Da). Returns (B, K, D) predictions."""
    seq, preds = emb_ctx, []
    for k in range(K):
        pred = model.predict(seq[:, -H:], act_emb[:, k: k + H])[:, -1:]
        preds.append(pred)
        seq = torch.cat([seq, pred], dim=1)
    return torch.cat(preds, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--weights", default=None, help="default: last weights_epoch_*.pt")
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    H, K = cfg.history_size, args.max_k
    cfg.num_preds = K  # clips of H + K observations
    cfg.nav.val_file = f"{args.split}.h5"
    cfg.nav.max_val_episodes = None
    _, ds = build_datasets(cfg)  # normaliser fitted on train, applied to this split

    wpath = Path(args.weights) if args.weights else max(
        (run / "weights").glob("weights_epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model = model.cuda().eval()
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).cuda()

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[: args.max_clips].tolist()
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=args.batch_size,
                                         num_workers=8, shuffle=False)
    raw_src = ds._cache[ds.src_col]  # raw [dx, dy, dtheta] for distances / probe targets

    names = ["model", "copylast", "zero_action", "shuffled_action"]
    sums = {n: {m: torch.zeros(K) for m in ("mse", "l1", "cos")} for n in names}
    tgt_sq, tgt_sum, n_seen = 0.0, 0.0, 0  # per-(k, dim) sums for the target variance
    all_emb, probe_x, sig_vals = [], [], []

    for batch in loader:
        pixels = batch["pixels"].cuda()
        action = torch.nan_to_num(batch["action"], 0.0).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.encode({"pixels": pixels, "action": action})
            emb, act_emb = out["emb"].float(), out["act_emb"].float()
            zero_emb = model.action_encoder(torch.zeros_like(action)).float()
            shuf_emb = act_emb[torch.randperm(len(act_emb), device=act_emb.device)]
            ctx, tgt = emb[:, :H], emb[:, H: H + K]
            preds = {
                "model": rollout(model, ctx, act_emb, H, K).float(),
                "zero_action": rollout(model, ctx, zero_emb, H, K).float(),
                "shuffled_action": rollout(model, ctx, shuf_emb, H, K).float(),
                "copylast": emb[:, H - 1: H].expand_as(tgt),
            }
        for n, p in preds.items():
            sums[n]["mse"] += (p - tgt).pow(2).mean(-1).sum(0).cpu()
            sums[n]["l1"] += (p - tgt).abs().mean(-1).sum(0).cpu()
            sums[n]["cos"] += (1 - F.cosine_similarity(p, tgt, dim=-1)).sum(0).cpu()
        tgt_sum = tgt_sum + tgt.sum(0).cpu()
        tgt_sq = tgt_sq + tgt.pow(2).sum(0).cpu()
        n_seen += len(tgt)
        all_emb.append(emb[:, H - 1].cpu())
        probe_x.append(torch.cat([emb[:, H - 1], emb[:, H]], -1).cpu())
        sig_vals.append(float(sigreg(emb.transpose(0, 1))))

    # probe targets: the raw block action between obs H-1 and H of each clip
    fs = ds.frameskip
    starts = [ds.offsets[ds.clip_indices[i][0]] + ds.clip_indices[i][1] + (H - 1) * fs for i in idx]
    if ds.action_rep == "concat":
        y = np.stack([raw_src[s: s + fs] for s in starts])  # per-frame steps -> compose approximately by sum
        y = y.sum(1)
    else:
        y = raw_src[starts]
    y = torch.from_numpy(np.nan_to_num(y)).double()

    Z = torch.cat(all_emb)
    step_m = float(np.nanmean(np.linalg.norm(ds._cache[ds.src_col][:, :2], axis=1)))
    if ds.action_rep == "concat":
        step_m *= fs
    var = (tgt_sq / n_seen - (tgt_sum / n_seen) ** 2).mean(-1).clamp_min(1e-8)  # (K,) mean per-dim variance
    result = {
        "run": str(run), "weights": wpath.name, "split": args.split, "n_clips": n_seen,
        "action_rep": ds.action_rep, "frameskip": fs, "loss": cfg.loss.pred,
        "metres_per_step": round(step_m, 4),
        "metres_at_k": [round(step_m * (k + 1), 3) for k in range(K)],
        "rollout": {n: {m: (v / n_seen).tolist() for m, v in d.items()} for n, d in sums.items()},
        "collapse": {
            "emb_std_mean": float(Z.std(0).mean()), "emb_std_min": float(Z.std(0).min()),
            "effective_rank": effective_rank(Z), "embed_dim": Z.shape[1],
            "sigreg_mean": float(np.mean(sig_vals)),
        },
        "probe_r2_dx_dy_dtheta": ridge_r2(torch.cat(probe_x).double(), y),
    }
    for n in names:  # MSE normalised by the target variance (1.0 = predicting the mean embedding)
        result["rollout"][n]["nmse"] = (sums[n]["mse"] / n_seen / var).tolist()
    out = run / f"eval_{args.split}.json"
    json.dump(result, open(out, "w"), indent=2)
    print(json.dumps({k: result[k] for k in ("n_clips", "metres_at_k", "collapse", "probe_r2_dx_dy_dtheta")}, indent=1))
    for n in names:
        print(f"{n:16s} mse@k:", " ".join(f"{v:.4f}" for v in result["rollout"][n]["mse"]))
    print("saved", out)


if __name__ == "__main__":
    main()
