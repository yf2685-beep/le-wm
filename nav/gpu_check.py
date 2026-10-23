"""Stage 1 check: build LeWM from the training config and run one forward/backward
pass with navigation-shaped inputs (frameskip=4, action_dim=3) on GPU, timing it."""

import os
import sys
import time

import hydra
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from module import SIGReg  # noqa: E402

FRAMESKIP, ACTION_DIM, BATCH = 4, 3, 128

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available with this torch build"
print("device", torch.cuda.get_device_name(0))

OmegaConf.register_new_resolver("eval", eval, replace=True)
with hydra.initialize(version_base=None, config_path="../config/train"):
    cfg = hydra.compose(config_name="lewm", overrides=["data=tworoom"])
cfg.model.action_encoder.input_dim = FRAMESKIP * ACTION_DIM

model = hydra.utils.instantiate(cfg.model).cuda()
sigreg = SIGReg(**cfg.loss.sigreg.kwargs).cuda()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"params {n_params:.1f}M")

T = cfg.history_size + cfg.num_preds
batch = {
    "pixels": torch.randn(BATCH, T, 3, cfg.img_size, cfg.img_size, device="cuda"),
    "action": torch.randn(BATCH, T, FRAMESKIP * ACTION_DIM, device="cuda"),
}
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

for it in range(6):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.encode(dict(batch))
        emb, act = out["emb"], out["act_emb"]
        pred = model.predict(emb[:, : cfg.history_size], act[:, : cfg.history_size])
        pred_loss = (pred - emb[:, cfg.num_preds:]).pow(2).mean()
        loss = pred_loss + cfg.loss.sigreg.weight * sigreg(emb.transpose(0, 1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    print(f"iter {it} loss {loss.item():.4f} time {time.time() - t0:.3f}s")

print(f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB at batch {BATCH}")
print("GPU_CHECK_OK")
