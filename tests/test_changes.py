"""Synthetic check of the evidence logic: ray-cast box scenes with known changes.

Before: chair at A, box R, static box S, object H.
After : chair moved to B, R removed, S unchanged, a board hides H (H is still there),
        object U added in a corner the before video never looked at.

Expected: chair-at-A and R removed; chair-at-B and board added; S unchanged;
H not removed (hidden, not gone); U unknown (never observed before), not added.
"""

import numpy as np
import pytest
import torch
import yaml

from wcitr import changes, objects
from wcitr.geometry import camera_centers, look_at

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H_IMG, W_IMG, F = 120, 160, 120.0


def box(cx, cy, sx, sy, sz, z0=0.0):
    return np.array([cx - sx / 2, cy - sy / 2, z0]), np.array([cx + sx / 2, cy + sy / 2, z0 + sz])


ROOM = [box(0, 2, 6, 6, 0.1, z0=-0.1), box(0, 5.05, 6, 0.1, 2.5), box(-3.05, 2, 0.1, 6, 2.5), box(3.05, 2, 0.1, 6, 2.5)]
CHAIR_A, CHAIR_B = box(-1.0, 2.5, 0.4, 0.4, 0.8), box(1.0, 2.5, 0.4, 0.4, 0.8)
R_BOX, S_BOX = box(0.0, 3.6, 0.3, 0.3, 0.3), box(-1.6, 1.4, 0.35, 0.35, 0.5)
H_OBJ, BOARD = box(2.0, 3.6, 0.3, 0.3, 0.4), box(2.0, 2.9, 1.6, 0.05, 1.2)
U_OBJ = box(-2.3, 1.5, 0.35, 0.35, 0.45)  # inside every after view, outside every before view

BEFORE = ROOM + [CHAIR_A, R_BOX, S_BOX, H_OBJ]
AFTER = ROOM + [CHAIR_B, S_BOX, H_OBJ, BOARD, U_OBJ]


def render(boxes, E):
    K = np.array([[F, 0, W_IMG / 2], [0, F, H_IMG / 2], [0, 0, 1]])
    v, u = np.mgrid[0:H_IMG, 0:W_IMG]
    rays_c = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u, float)], -1)
    R, t = E[:, :3], E[:, 3]
    o = -R.T @ t
    d = rays_c @ R  # world directions, camera-z component = 1, so ray parameter = depth
    depth = np.full((H_IMG, W_IMG), np.inf)
    ident = np.zeros((H_IMG, W_IMG), int)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, (lo, hi) in enumerate(boxes):
            t1, t2 = (lo - o) / d, (hi - o) / d
            tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
            hit = (tn <= tf) & (tn > 0) & (tn < depth)
            depth[hit], ident[hit] = tn[hit], k
    valid = np.isfinite(depth)
    depth[~valid] = 0
    rgb = (np.stack([ident * 37 % 255, ident * 91 % 255, ident * 53 % 255], -1)).astype(np.uint8)
    return depth, valid, rgb, K


def cameras(xs, y=-0.5, z=1.5, target=(0.0, 2.6, 0.4)):
    return [look_at((x, y, z), target) for x in xs]


@pytest.fixture(scope="module")
def scene():
    cams_a = cameras(np.linspace(-1.0, 1.0, 14))
    cams_b = cameras(np.linspace(-2.5, 0.2, 14), target=(-0.6, 2.6, 0.4))
    data = [render(BEFORE, E) for E in cams_a] + [render(AFTER, E) for E in cams_b]
    depth = np.stack([d[0] for d in data]).astype(np.float32)
    valid = np.stack([d[1] for d in data])
    images = np.stack([d[2] for d in data])
    K = np.stack([d[3] for d in data]).astype(np.float32)
    E = np.stack(cams_a + cams_b).astype(np.float32)
    video = np.array([0] * len(cams_a) + [1] * len(cams_b))
    cfg = yaml.safe_load(open("config.yaml"))["change"]
    cfg.update(stride=1, voxel=0.04)
    res = changes.detect_changes(depth, np.full_like(depth, 5.0), valid, E, K, video, images, cfg, device=DEV)
    cams = {}
    for vid in (0, 1):
        m = video == vid
        cams[vid] = {"E": E[m].astype(float), "K": K[m].astype(float), "hw": (H_IMG, W_IMG),
                     "centers": camera_centers(E[m].astype(float))}
    return res, cfg, cams


def inside(xyz, b, pad=0.03):
    lo, hi = b
    return np.all((xyz >= lo - pad) & (xyz <= hi + pad), axis=1)


def state_fractions(res, b):
    m = inside(res["xyz"], b)
    s = res["state"][m]
    assert m.sum() > 5, "region not observed at all in the synthetic scene"
    return {k: float((s == v).mean()) for k, v in
            (("changed", changes.CHANGED), ("unchanged", changes.UNCHANGED), ("unknown", changes.UNKNOWN))}


def test_moved_chair_is_removed_then_added(scene):
    res = scene[0]
    assert state_fractions(res[0], CHAIR_A)["changed"] > 0.6
    assert state_fractions(res[1], CHAIR_B)["changed"] > 0.6


def test_removed_box(scene):
    res = scene[0]
    assert state_fractions(res[0], R_BOX)["changed"] > 0.6


def test_static_box_unchanged(scene):
    res = scene[0]
    assert state_fractions(res[0], S_BOX)["changed"] < 0.05
    assert state_fractions(res[1], S_BOX)["changed"] < 0.05


def test_hidden_object_is_not_removed(scene):
    res = scene[0]
    assert state_fractions(res[0], H_OBJ)["changed"] < 0.05


def test_board_added(scene):
    res = scene[0]
    assert state_fractions(res[1], BOARD)["changed"] > 0.5


def test_unseen_corner_is_unknown_not_added(scene):
    res = scene[0]
    f = state_fractions(res[1], U_OBJ)
    assert f["changed"] < 0.05
    assert f["unknown"] > 0.8


def test_room_has_no_false_changes(scene):
    res = scene[0]
    for r in res.values():
        walls = np.zeros(len(r["xyz"]), bool)
        for b in ROOM:
            walls |= inside(r["xyz"], b, pad=0.0)
        for b in (CHAIR_A, CHAIR_B, R_BOX, BOARD, H_OBJ, U_OBJ, S_BOX):
            walls &= ~inside(r["xyz"], b, pad=0.08)
        assert (r["state"][walls] == changes.CHANGED).mean() < 0.01


def test_objects_and_moved_matching(scene):
    res, cfg, cams = scene
    ocfg = yaml.safe_load(open("config.yaml"))["objects"]
    removed = objects.cluster_changes(res[0], ocfg, cfg["voxel"])
    added = objects.cluster_changes(res[1], ocfg, cfg["voxel"])
    for o in removed + added:  # no images here: give each object an identity-like embedding by colour
        o["embedding"] = np.ones(4) / 2
    events = objects.match_events(removed, added, ocfg, cfg["voxel"], cams)
    moved = [e for e in events if e["type"] == "moved"]
    assert len(moved) == 1, events
    e = moved[0]
    # Within 2 voxels (8 cm here; this synthetic scene is only 160x120 px with 4 cm voxels).
    tol = 2 * cfg["voxel"]
    assert abs(e["displacement"][0] - 2.0) < tol and abs(e["displacement"][1]) < tol, e["displacement"]
    assert abs(e["rotation_deg"]) < 10, e["rotation_deg"]
    assert any(ev["type"] == "added" for ev in events)  # the board
