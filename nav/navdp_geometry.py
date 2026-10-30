"""Pose -> local action conversion for NavDP trajectories.

Conventions follow InternNav `navdp_dataset_lerobot.py` (relative_pose):
  * NavDP's parquet `action` column is a per-frame 4x4 camera-to-world pose (Z-up world, metres).
  * Robot base orientation: R_base = R_cam @ inv(R_ext[:3, :3]); base origin = camera position.
  * In the base frame, axis 1 points forward and axis 0 points right, so the local
    (x forward, y left) coordinates of a world point p are
        p_b = R_base^T (p - T_base);  x = p_b[1];  y = -p_b[0].
  * Heading yaw = angle of the forward axis (column 1 of R_base) in the world xy-plane;
    dtheta > 0 is a left (counter-clockwise) turn, matching y-left.
"""

import numpy as np


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def base_poses(cam_poses, extrinsic):
    """(T, 4, 4) camera-to-world poses + (4, 4) base-to-camera extrinsic -> (T, 4, 4) base-to-world."""
    out = cam_poses.copy()
    out[:, :3, :3] = cam_poses[:, :3, :3] @ np.linalg.inv(extrinsic[:3, :3])
    return out


def world_xyyaw(poses):
    """(T, 4, 4) base poses -> (T, 3) world [x, y, yaw]."""
    fwd = poses[:, :3, 1]
    return np.stack([poses[:, 0, 3], poses[:, 1, 3], np.arctan2(fwd[:, 1], fwd[:, 0])], axis=-1)


def local_displacement(poses, stride):
    """Displacement from frame i to frame i+stride, expressed in frame i's local (x fwd, y left) frame.

    Returns (T, 3) float [dx, dy, dtheta]; rows without a frame i+stride are NaN.
    """
    T = len(poses)
    out = np.full((T, 3), np.nan)
    if T <= stride:
        return out
    R, t = poses[:-stride, :3, :3], poses[:-stride, :3, 3]
    p_b = np.einsum("nji,nj->ni", R, poses[stride:, :3, 3] - t)  # R^T (p - t)
    yaw = world_xyyaw(poses)[:, 2]
    out[:-stride, 0] = p_b[:, 1]
    out[:-stride, 1] = -p_b[:, 0]
    out[:-stride, 2] = wrap_angle(yaw[stride:] - yaw[:-stride])
    return out


def compose(actions):
    """Integrate per-step local [dx, dy, dtheta] into poses [x, y, yaw] starting at the origin."""
    poses = [np.zeros(3)]
    for dx, dy, dth in actions:
        x, y, yaw = poses[-1]
        c, s = np.cos(yaw), np.sin(yaw)
        poses.append(np.array([x + c * dx - s * dy, y + s * dx + c * dy, wrap_angle(yaw + dth)]))
    return np.stack(poses)


# ---- action representations (applied at load time to [dx, dy, dtheta]) ----

ACTION_DIMS = {"xy": 2, "xyt": 3, "xysc": 4, "disc": 4}  # dbin: sum(len(e) + 1 for e in bin_edges)


def quantile_bin_edges(xyt, bins):
    """Inner bin edges per dimension of [dx, dy, dtheta] at equal-mass quantiles of (N, 3) data (NaN rows ignored)."""
    x = np.asarray(xyt, dtype=np.float64)
    x = x[~np.isnan(x).any(axis=-1)]
    return [np.quantile(x[:, d], np.arange(1, b) / b).astype(np.float32) for d, b in enumerate(bins)]


def to_representation(xyt, rep, tau_rad=None, eps_m=None, bin_edges=None):
    """Convert (N, 3) [dx, dy, dtheta] to the requested representation; NaN rows stay NaN.

    xy:   [dx, dy]
    xyt:  [dx, dy, dtheta]
    xysc: [dx, dy, sin dtheta, cos dtheta]
    disc: one-hot over [stop, forward, left, right] using |dtheta| >= tau for turns and
          planar distance < eps for stop.
    dbin: magnitude-preserving discretisation: each of dx, dy, dtheta is binned with `bin_edges`
          (see quantile_bin_edges) and the per-dimension one-hots are concatenated (no stop class).
    """
    xyt = np.asarray(xyt, dtype=np.float32)
    if rep == "xy":
        return xyt[..., :2]
    if rep == "xyt":
        return xyt
    if rep == "xysc":
        return np.concatenate([xyt[..., :2], np.sin(xyt[..., 2:]), np.cos(xyt[..., 2:])], axis=-1)
    if rep == "disc":
        assert tau_rad is not None and eps_m is not None, "disc needs tau_rad and eps_m"
        dist = np.linalg.norm(xyt[..., :2], axis=-1)
        th = xyt[..., 2]
        cls = np.where(th >= tau_rad, 2, np.where(th <= -tau_rad, 3, np.where(dist < eps_m, 0, 1)))
        onehot = np.eye(4, dtype=np.float32)[cls]
        onehot[np.isnan(xyt).any(axis=-1)] = np.nan
        return onehot
    if rep == "dbin":
        assert bin_edges is not None, "dbin needs bin_edges"
        parts = [np.eye(len(e) + 1, dtype=np.float32)[np.digitize(np.nan_to_num(xyt[..., d]), e)]
                 for d, e in enumerate(bin_edges)]
        out = np.concatenate(parts, axis=-1)
        out[np.isnan(xyt).any(axis=-1)] = np.nan
        return out
    raise ValueError(f"unknown action representation {rep!r}")
