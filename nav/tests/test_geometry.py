"""Unit tests for nav/navdp_geometry.py. Run: python -m pytest nav/tests -q"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from navdp_geometry import (  # noqa: E402
    base_poses,
    compose,
    local_displacement,
    to_representation,
    world_xyyaw,
)


def make_base_pose(x, y, yaw, z=0.0):
    """Base-to-world pose whose forward axis (column 1) points at `yaw` in the world xy-plane."""
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)  # column 0 = right, so (fwd, left=-right, up) is right-handed
    P = np.eye(4)
    P[:3, 0], P[:3, 1], P[:3, 2], P[:3, 3] = right, fwd, up, (x, y, z)
    return P


def traj(xyyaw):
    return np.stack([make_base_pose(*p) for p in xyyaw])


def test_straight_line_forward():
    poses = traj([(0.1 * i, 0.0, 0.0) for i in range(10)])
    d = local_displacement(poses, 1)[:-1]
    np.testing.assert_allclose(d, np.tile([0.1, 0.0, 0.0], (9, 1)), atol=1e-9)


def test_straight_line_any_heading():
    yaw = 2.3
    poses = traj([(0.1 * i * np.cos(yaw), 0.1 * i * np.sin(yaw), yaw) for i in range(6)])
    np.testing.assert_allclose(local_displacement(poses, 1)[:-1], np.tile([0.1, 0, 0], (5, 1)), atol=1e-9)


def test_left_is_positive_y_and_theta():
    poses = traj([(0, 0, 0), (0, 0.2, 0)])  # world +y is to the left when facing +x
    np.testing.assert_allclose(local_displacement(poses, 1)[0], [0, 0.2, 0], atol=1e-9)
    poses = traj([(0, 0, 0), (0, 0, np.pi / 2)])  # rotate in place, counter-clockwise
    np.testing.assert_allclose(local_displacement(poses, 1)[0], [0, 0, np.pi / 2], atol=1e-9)


def test_wraparound():
    poses = traj([(0, 0, np.pi - 0.05), (0, 0, -np.pi + 0.05)])
    assert abs(local_displacement(poses, 1)[0, 2] - 0.1) < 1e-9


def test_stride_and_nan_tail():
    poses = traj([(0.05 * i, 0, 0) for i in range(10)])
    d = local_displacement(poses, 4)
    np.testing.assert_allclose(d[:6], np.tile([0.2, 0, 0], (6, 1)), atol=1e-9)
    assert np.isnan(d[6:]).all()


def test_compose_reconstructs_arc():
    # quarter circle of radius 1, 50 steps; integrating per-frame actions must recover the world path
    th = np.linspace(0, np.pi / 2, 51)
    xyyaw = np.stack([np.sin(th), 1 - np.cos(th), th], axis=-1)
    poses = traj(xyyaw)
    rec = compose(local_displacement(poses, 1)[:-1])
    np.testing.assert_allclose(rec, xyyaw, atol=1e-9)
    # stride-4 displacement equals composition of the 4 per-frame steps
    d1, d4 = local_displacement(poses, 1), local_displacement(poses, 4)
    np.testing.assert_allclose(compose(d1[:4])[-1], d4[0], atol=1e-9)


def test_extrinsic_pitch_does_not_change_base_motion():
    # camera pitched down 20 deg about its own x axis; base motion must be unaffected
    a = np.deg2rad(20)
    ext = np.eye(4)
    ext[:3, :3] = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    base = traj([(0.1 * i, 0.02 * i, 0.1 * i) for i in range(5)])
    cam = base.copy()
    cam[:, :3, :3] = base[:, :3, :3] @ ext[:3, :3]
    rec = base_poses(cam, ext)
    np.testing.assert_allclose(world_xyyaw(rec), world_xyyaw(base), atol=1e-9)


def test_representations():
    x = np.array([[0.1, 0.0, 0.0], [0.1, 0.01, 0.3], [0.1, -0.01, -0.3], [0.001, 0, 0], [np.nan] * 3])
    assert to_representation(x, "xy").shape == (5, 2)
    sc = to_representation(x, "xysc")
    np.testing.assert_allclose(sc[1, 2:], [np.sin(0.3), np.cos(0.3)], rtol=1e-6)
    d = to_representation(x, "disc", tau_rad=0.2, eps_m=0.01)
    np.testing.assert_array_equal(d[:4].argmax(-1), [1, 2, 3, 0])  # forward, left, right, stop
    assert np.isnan(d[4]).all()
