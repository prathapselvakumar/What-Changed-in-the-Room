"""Step 3: evidence-based change detection (the core idea).

Every confident depth pixel of one video is lifted to a 3D point and checked
against the depth maps of the OTHER video:

    other camera saw farther than the point  -> it looked through empty space: "free"
    other camera saw the same surface        -> "consistent"
    other camera saw something in front      -> occluded, no evidence
    point outside the other camera's view    -> no evidence

Points of the before video that the after video sees through are REMOVED.
Points of the after video that the before video saw through are ADDED.
Points without enough evidence are UNKNOWN, never "changed".

Points are first checked against frames of their own video. A point its own
video cannot confirm (noise) or sees through (a person walking by) is dropped.
Votes are fused in voxels.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import backproject, cam_to_world, camera_centers, erode, min_max_pool, normals_from_points, project

UNCHANGED, CHANGED, UNKNOWN = 0, 1, 2


def flying_pixels(depth, rel_threshold, surface_tol=0.02):
    """Pixels that float between a foreground and a background surface at a depth edge.

    Edges: inverse depth is linear in pixel coordinates on any plane, so its second difference is
    ~0 on flat surfaces (even at grazing angles) and large at depth discontinuities.
    Near an edge, pixels lying on the near or far surface are kept; only pixels with a depth
    strictly in between are dropped. Dropping whole edge bands would shave every object's
    silhouette and bias its measured position.
    """
    inv = torch.where(depth > 0, 1.0 / depth.clamp_min(1e-6), torch.zeros_like(depth))
    p = F.pad(inv[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    d2x = (p[:, 1:-1, 2:] + p[:, 1:-1, :-2] - 2 * inv).abs()
    d2y = (p[:, 2:, 1:-1] + p[:, :-2, 1:-1] - 2 * inv).abs()
    edge = torch.maximum(d2x, d2y) > rel_threshold * inv
    near_edge = F.max_pool2d(edge[:, None].float(), 3, 1, 1)[:, 0] > 0
    mn, mx = min_max_pool(depth)
    between = (depth > mn * (1 + surface_tol)) & (depth < mx * (1 - surface_tol))
    return near_edge & between


def reliable_pixels(depth, conf, valid, cfg):
    """Confident pixels away from depth edges (edges create 'flying' points)."""
    flat = conf[valid]
    if flat.numel() > 2_000_000:
        flat = flat[torch.randint(0, flat.numel(), (2_000_000,), device=flat.device)]
    thr = torch.quantile(flat, cfg["conf_quantile"])
    return valid & (conf >= thr) & (depth > 0) & ~flying_pixels(depth, cfg["edge_rel_threshold"])


WINDOWS = (3, 5, 9, 17, 33)  # depth min/max window sizes (pixels) for registration tolerance


def window_maps(depth):
    """Min and max depth over growing windows: (levels, S, H, W) each."""
    mins, maxs = zip(*(min_max_pool(depth, k) for k in WINDOWS))
    return torch.stack(mins), torch.stack(maxs)


@torch.no_grad()
def vote(Xw, nw, tgt, maps, cfg):
    """Free / consistent votes of points Xw (P, 3) with normals nw against target frames `tgt`.

    The two videos never line up perfectly (registration error ~ cfg['registration_sigma']).
    A point therefore counts as seen-through only if the other camera saw farther across a window
    covering that error at the point's depth, and as consistent if the surface lies anywhere in it.
    Without this, every object outline shows up as a thin band of false changes.
    """
    E, K, ok, C = (maps[k][tgt] for k in ("E", "K", "ok", "C"))
    dmin, dmax = maps["dmin"][:, tgt], maps["dmax"][:, tgt]
    _, T, H, W = dmin.shape
    u, v, z = project(Xw, E, K)
    ui, vi = u.round().long(), v.round().long()
    inb = (z > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi = ui.clamp(0, W - 1), vi.clamp(0, H - 1)
    t_idx = torch.arange(T, device=Xw.device)[:, None].expand_as(ui)
    obs = inb & ok[t_idx, vi, ui]

    view = Xw[None] - C[:, None, :]
    view = view / view.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    obs &= (view * nw[None]).sum(-1).abs() >= cfg["min_cos"]

    # Window radius (pixels) that covers the registration error at this depth.
    radius = K[:, 0, 0][:, None] * cfg["registration_sigma"] / z.clamp_min(0.05)
    half = torch.as_tensor([k // 2 for k in WINDOWS], device=Xw.device, dtype=radius.dtype)
    level = torch.searchsorted(half, radius.contiguous()).clamp(max=len(WINDOWS) - 1)

    tau = torch.clamp(cfg["tau_rel"] * z, min=cfg["tau_abs"])
    lo, hi = dmin[level, t_idx, vi, ui], dmax[level, t_idx, vi, ui]
    free = obs & (lo > z + tau)
    cons = obs & (lo <= z + tau) & (hi >= z - tau)
    return free, cons


@torch.no_grad()
def detect_changes(depth, conf, valid, E, K, video, images, cfg, device="cuda"):
    """Returns per-video voxel results: {0: before, 1: after} -> dict of arrays."""
    t = lambda a, dt=torch.float32: torch.as_tensor(a, device=device, dtype=dt)
    depth, conf, valid = t(depth), t(conf), t(valid, torch.bool)
    E, K = t(E), t(K)
    video = np.asarray(video)
    S, H, W = depth.shape

    good = reliable_pixels(depth, conf, valid, cfg)
    dmin, dmax = window_maps(depth)
    maps = {
        "E": E, "K": K, "dmin": dmin, "dmax": dmax,
        "ok": erode(good),  # target pixel and its neighbours must be reliable
        "C": t(camera_centers(E.cpu().numpy())),
    }
    P_cam = backproject(depth, K)
    P_world = cam_to_world(P_cam, E)
    N_world = cam_to_world(normals_from_points(P_cam), torch.cat([E[:, :, :3], torch.zeros_like(E[:, :, 3:])], -1))
    rgb = t(images, torch.uint8)

    stride = torch.zeros_like(good)
    stride[:, ::cfg["stride"], ::cfg["stride"]] = True

    results = {}
    for vid in (0, 1):
        own = np.nonzero(video == vid)[0]
        other = torch.as_tensor(np.nonzero(video != vid)[0], device=device)
        pts, cols, fr, nf, nc = [], [], [], [], []
        for s in own:
            m = good[s] & stride[s]
            Xw, nw = P_world[s][m], N_world[s][m]
            if len(Xw) == 0:
                continue
            own_t = torch.as_tensor(own[own != s], device=device)
            sf, sc = vote(Xw, nw, own_t, maps, cfg)
            sf, sc = sf.sum(0), sc.sum(0)
            keep = (sc >= cfg["min_self_support"]) & (sf <= cfg["max_self_free_ratio"] * (sf + sc))
            of, oc = vote(Xw[keep], nw[keep], other, maps, cfg)
            pts.append(Xw[keep])
            cols.append(rgb[s][m][keep])
            fr.append(torch.full((int(keep.sum()),), int(s), device=device))
            nf.append(of.sum(0))
            nc.append(oc.sum(0))
        results[vid] = fuse_voxels(torch.cat(pts), torch.cat(cols), torch.cat(fr), torch.cat(nf), torch.cat(nc), cfg)
    return results


def fuse_voxels(pts, cols, frames, n_free, n_cons, cfg):
    vs = cfg["voxel"]
    keys = torch.floor(pts / vs).long()
    uniq, inv = torch.unique(keys, dim=0, return_inverse=True)
    V = len(uniq)
    dev = pts.device

    def ssum(x):
        out = torch.zeros((V,) + x.shape[1:], device=dev, dtype=torch.float64)
        return out.index_add_(0, inv, x.double())

    count = ssum(torch.ones(len(pts), device=dev))
    center = ssum(pts) / count[:, None]
    color = ssum(cols.float()) / count[:, None]
    F, C = ssum(n_free.float()), ssum(n_cons.float())
    pair = torch.unique(torch.stack([inv, frames]), dim=1)
    n_frames = torch.zeros(V, device=dev, dtype=torch.long).index_add_(0, pair[0], torch.ones_like(pair[0]))

    obs = F + C
    ratio = torch.where(obs > 0, F / obs.clamp_min(1), torch.zeros_like(F))
    state = torch.full((V,), UNKNOWN, device=dev, dtype=torch.int8)
    enough = obs >= cfg["min_obs"]
    state[enough & (ratio >= cfg["changed_ratio"])] = CHANGED
    state[enough & (ratio <= cfg["unchanged_ratio"])] = UNCHANGED

    keep = n_frames >= cfg["min_voxel_frames"]
    # Which frames saw each changed voxel: needed later to count supporting views per object.
    return {
        "xyz": center[keep].float().cpu().numpy(),
        "rgb": color[keep].clamp(0, 255).byte().cpu().numpy(),
        "state": state[keep].cpu().numpy(),
        "free_ratio": ratio[keep].float().cpu().numpy(),
        "n_obs": obs[keep].float().cpu().numpy(),
        "n_frames": n_frames[keep].cpu().numpy(),
        "voxel_frames": _voxel_frame_lists(pair, keep),
    }


def _voxel_frame_lists(pair, keep):
    """Sparse (voxel_index, frame) pairs re-indexed to kept voxels, as an (M, 2) int array."""
    keep_np = keep.cpu().numpy()
    remap = -np.ones(len(keep_np), np.int64)
    remap[keep_np] = np.arange(keep_np.sum())
    p = pair.cpu().numpy()
    vox = remap[p[0]]
    m = vox >= 0
    return np.stack([vox[m], p[1][m]], axis=1)


def summarize(results):
    out = {}
    for vid, name in ((0, "before"), (1, "after")):
        s = results[vid]["state"]
        out[name] = {
            "voxels": int(len(s)),
            "unchanged": int((s == UNCHANGED).sum()),
            "changed": int((s == CHANGED).sum()),
            "unknown": int((s == UNKNOWN).sum()),
        }
    return out
