"""HDF5 reader for converted NavDP data that exposes one chosen action representation as `action`.

The world-model step is `frameskip` raw frames (~3.2 cm each). Observations are taken every
`frameskip` rows, as in stable_worldmodel. The `action` attached to observation k is:

  rep in {xy, xyt, xysc, disc}: the block displacement from obs k to obs k+1 (column act_s{frameskip}),
                                converted to that representation -> (num_steps, A)
  rep == "concat":              the frameskip per-frame displacements inside the block, each in its own
                                local frame (column act_f), concatenated -> (num_steps, frameskip * 3).
                                This is LeWM's native action chunking.
"""

import numpy as np
import torch
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset

from navdp_geometry import ACTION_DIMS, to_representation


class NavDPDataset(HDF5Dataset):
    def __init__(self, path, frameskip, num_steps, action_rep="xyt", disc_tau_deg=None, disc_eps_m=None,
                 extra_keys=(), transform=None):
        self.action_rep = action_rep
        self.tau = None if disc_tau_deg is None else np.deg2rad(disc_tau_deg)
        self.eps = disc_eps_m
        self.src_col = "act_f" if action_rep == "concat" else f"act_s{frameskip}"
        super().__init__(
            path=path, frameskip=frameskip, num_steps=num_steps, transform=transform,
            keys_to_load=["pixels", self.src_col, *extra_keys], keys_to_cache=[self.src_col],
        )
        self.action_input_dim = 3 * frameskip if action_rep == "concat" else ACTION_DIMS[action_rep]

    @property
    def column_names(self):
        return ["action" if k == self.src_col else k for k in self._keys]

    def _convert(self, a):
        if self.action_rep == "concat":
            return np.asarray(a, dtype=np.float32)
        return to_representation(a, self.action_rep, self.tau, self.eps)

    def _load_slice(self, ep_idx, start, end):
        self._open()
        g0, g1 = self.offsets[ep_idx] + start, self.offsets[ep_idx] + end
        steps = {}
        for col in self._keys:
            src = self._cache if col in self._cache else self.h5_file
            data = src[col][g0:g1]
            if col == self.src_col:
                if self.action_rep == "concat":
                    data = data.reshape((g1 - g0) // self.frameskip, -1)
                else:
                    data = data[:: self.frameskip]
                steps["action"] = torch.from_numpy(self._convert(data))
                continue
            data = data[:: self.frameskip]
            t = torch.from_numpy(np.ascontiguousarray(data))
            if data.ndim == 4 and data.shape[-1] in (1, 3):
                t = t.permute(0, 3, 1, 2)
            steps[col] = t
        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col):
        if col != "action":
            return super().get_col_data(col)
        a = self._cache[self.src_col]
        if self.action_rep == "concat":
            return np.tile(a, (1, self.frameskip))  # per-frame stats, tiled to the chunk width
        return self._convert(a)

    def get_dim(self, col):
        return self.action_input_dim if col == "action" else super().get_dim(col)
