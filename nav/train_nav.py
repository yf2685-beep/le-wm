"""Stage 3/4: train LeWM on converted NavDP data.

Same objective as train.py (prediction loss + lambda * SIGReg) with nav-specific plumbing:
  * scene-level train/val files instead of random_split (no leakage across windows of one trajectory)
  * selectable action representation / stride (nav.action_rep, nav.frameskip)
  * prediction loss "mse" (LeWM default) or "l1"
  * crash-safe: Lightning last.ckpt every `ckpt_every_n_steps` + auto-resume, per-epoch weights,
    CSV metrics written to the run directory (WandB optional)

Usage (from repo root):
    python nav/train_nav.py nav.h5_dir=/path/to/h5 nav.action_rep=xyt nav.frameskip=4 run_root=/path/to/runs
"""

import os
import sys
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nav")]

from module import SIGReg  # noqa: E402
from navdp_dataset import NavDPDataset  # noqa: E402
from utils import ZScoreNormalizer, get_img_preprocessor  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)


def nav_forward(self, batch, stage, cfg):
    """LeWM forward (as train.lejepa_forward) + optional L1 and cheap monitoring metrics."""
    ctx_len, n_preds = cfg.history_size, cfg.num_preds
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb, act_emb = output["emb"], output["act_emb"]
    pred_emb = self.model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
    tgt_emb = emb[:, n_preds:]

    diff = pred_emb - tgt_emb
    output["pred_loss"] = diff.abs().mean() if cfg.loss.pred == "l1" else diff.pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]

    with torch.no_grad():  # monitoring only
        output["mse_metric"] = diff.pow(2).mean()
        output["copylast_mse_metric"] = (emb[:, ctx_len - 1] - emb[:, ctx_len]).pow(2).mean()
        output["emb_std_metric"] = emb.reshape(-1, emb.size(-1)).std(0).mean()

    logs = {f"{stage}/{k}": v.detach() for k, v in output.items() if k.endswith(("loss", "metric"))}
    self.log_dict(logs, on_step=(stage == "train"), on_epoch=True, sync_dist=True,
                  batch_size=emb.size(0))
    return output


class SaveWeights(Callback):
    """Per-epoch model weights (small, eval-ready), independent of Lightning's resume checkpoints."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(pl_module.model.state_dict(), self.out_dir / f"weights_epoch_{trainer.current_epoch + 1}.pt")


def build_datasets(cfg):
    n = cfg.nav
    kw = dict(frameskip=n.frameskip, num_steps=cfg.history_size + cfg.num_preds, action_rep=n.action_rep,
              disc_tau_deg=n.disc_tau_deg, disc_eps_m=n.disc_eps_m)
    train = NavDPDataset(os.path.join(n.h5_dir, n.train_file), **kw)
    val = NavDPDataset(os.path.join(n.h5_dir, n.val_file), **kw)
    for ds, k in ((train, n.max_train_episodes), (val, n.max_val_episodes)):
        if k is not None:
            ds.clip_indices = [c for c in ds.clip_indices if c[0] < k]
        if n.get("clip_stride", 1) > 1:  # neighbouring clips overlap almost fully; keep every clip_stride-th start
            ds.clip_indices = [c for c in ds.clip_indices if c[1] % n.clip_stride == 0]

    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    if n.action_rep != "disc":  # one-hot actions are left as-is (a never-seen class has zero std)
        a = torch.from_numpy(np.asarray(train.get_col_data("action"), dtype=np.float32))
        a = a[~torch.isnan(a).any(1)]
        mean, std = a.mean(0, keepdim=True), a.std(0, keepdim=True).clamp_min(1e-6)
        transforms.append(spt.data.transforms.WrapTorchTransform(
            ZScoreNormalizer(mean, std), source="action", target="action"))
    transform = spt.data.transforms.Compose(*transforms)
    train.transform = val.transform = transform
    return train, val


@hydra.main(version_base=None, config_path="../config/train", config_name="lewm_nav")
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    run_dir = Path(cfg.run_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set = build_datasets(cfg)
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = train_set.action_input_dim
    OmegaConf.save(cfg, run_dir / "config.yaml")
    print(f"run_dir={run_dir} train_clips={len(train_set)} val_clips={len(val_set)} "
          f"action_dim={train_set.action_input_dim}", flush=True)

    gen = torch.Generator().manual_seed(cfg.seed)
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True, drop_last=True, generator=gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    module = spt.Module(
        model=hydra.utils.instantiate(cfg.model),
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(nav_forward, cfg=cfg),
        optim={"model_opt": {"modules": "model", "optimizer": dict(cfg.optimizer),
                             "scheduler": {"type": "LinearWarmupCosineAnnealingLR"}, "interval": "epoch"}},
    )

    loggers = [CSVLogger(str(run_dir), name="csv", version="")]
    if cfg.wandb.enabled:
        loggers.append(WandbLogger(**cfg.wandb.config, save_dir=str(run_dir)))
        loggers[-1].log_hyperparams(OmegaConf.to_container(cfg))

    ckpt_dir = run_dir / "ckpt"
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=loggers,
        callbacks=[
            ModelCheckpoint(dirpath=str(ckpt_dir), save_last=True, save_top_k=0,
                            every_n_train_steps=cfg.ckpt_every_n_steps),
            SaveWeights(run_dir / "weights"),
        ],
        default_root_dir=str(run_dir),
        num_sanity_val_steps=1,
    )
    last = ckpt_dir / "last.ckpt"
    print(f"resume_from={last if last.exists() else None}", flush=True)
    trainer.fit(module, datamodule=spt.data.DataModule(train=train, val=val),
                ckpt_path=str(last) if last.exists() else None)
    (run_dir / "DONE").touch()


if __name__ == "__main__":
    run()
