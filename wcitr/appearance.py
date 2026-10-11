"""Appearance channel: changes that depth cannot see (transparent items, small or flat objects).

Depth networks look straight through a clear bottle and blur a pen into the desk, but both
change how the scene LOOKS. For every frame of one video:

  1. warp the most similar frames of the OTHER video into this camera, using this frame's
     depth (a reference pixel is used only if that camera could see the spot: not occluded);
  2. compare this frame with each warped view using DINOv2 patch features, which tolerate
     small misalignment and lighting differences;
  3. keep the SMALLEST difference over the warped views: a pixel is changed only if it
     disagrees with every view of the other video that saw the spot.

A transparent object's pixels have the depth of the wall behind it, so the changed pixels do
not tell where the object is in 3D. `visual_hull` finds that: a voxel belongs to the object if,
in every frame that sees it, it projects inside the changed pixels (and is not hidden behind
a surface). Rays from different viewpoints only agree where the object really is.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import backproject, cam_to_world, camera_centers, project

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Features:
    """DINOv2 patch features computed at twice the model resolution (finer patches)."""

    def __init__(self, model_name, device="cuda", upscale=2):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device, self.upscale = device, upscale
        self.patch = self.model.config.patch_size

    @torch.no_grad()
    def __call__(self, imgs):
        """imgs (B, 3, H, W) in [0, 1] -> unit-norm features on the patch grid (B, C, gh, gw)."""
        _, _, H, W = imgs.shape
        h = (H * self.upscale) // self.patch * self.patch
        w = (W * self.upscale) // self.patch * self.patch
        x = F.interpolate(imgs, size=(h, w), mode="bilinear", align_corners=False)
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        with torch.autocast("cuda", dtype=torch.float16):
            tok = self.model(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS
        f = tok.float().transpose(1, 2).reshape(len(imgs), -1, h // self.patch, w // self.patch)
        return F.normalize(f, dim=1)


def nearest_views(E, src, candidates, k):
    """The k candidate cameras most similar to camera `src` (viewing direction + position)."""
    C = camera_centers(E)
    fwd = E[:, 2, :3]  # optical axis in world coordinates (third row of R)
    ang = np.degrees(np.arccos(np.clip(fwd[candidates] @ fwd[src], -1, 1)))
    dist = np.linalg.norm(C[candidates] - C[src], axis=1)
    order = np.argsort(ang + 40.0 * dist)
    return candidates[order[:k]]


@torch.no_grad()
def warp(src, refs, depth, K, E, images, valid, tau_abs, tau_rel):
    """Warp reference frames `refs` into camera `src` using src's depth.

    Returns warped images (R, 3, H, W) and masks (R, H, W) of pixels the reference camera
    could see (inside its image, not hidden behind something nearer)."""
    H, W = depth.shape[1:]
    Xw = cam_to_world(backproject(depth[src:src + 1], K[src:src + 1]), E[src:src + 1]).reshape(-1, 3)
    u, v, z = project(Xw, E[refs], K[refs])
    grid = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], -1).view(len(refs), H, W, 2)
    warped = F.grid_sample(images[refs], grid, mode="bilinear", align_corners=True)
    d_ref = F.grid_sample(depth[refs][:, None], grid, mode="nearest", align_corners=True)[:, 0]
    ok_ref = F.grid_sample(valid[refs][:, None].float(), grid, mode="nearest", align_corners=True)[:, 0] > 0.5
    z = z.view(len(refs), H, W)
    tau = torch.clamp(tau_rel * z, min=tau_abs)
    inside = (u >= 0).view_as(z) & (u <= W - 1).view_as(z) & (v >= 0).view_as(z) & (v <= H - 1).view_as(z)
    seen = inside & ok_ref & (z > 0.05) & (d_ref >= z - tau) & valid[src][None] & (depth[src][None] > 0)
    return warped, seen


@torch.no_grad()
def change_maps(recon, cfg, device="cuda"):
    """Per-frame appearance difference to the other video.

    Returns diff (S, H, W) in [0, 2] (NaN where fewer than `min_views` warped views exist)."""
    t = lambda a: torch.as_tensor(a, device=device)
    depth, K, E = t(recon["depth"]).float(), t(recon["intrinsic"]).float(), t(recon["extrinsic"]).float()
    valid = t(recon["valid"]).bool()
    images = t(recon["images"]).permute(0, 3, 1, 2).float() / 255.0
    video = recon["video"]
    feats = Features(cfg["model"], device)

    S, H, W = depth.shape
    diff = torch.full((S, H, W), float("nan"), device=device)
    for s in range(S):
        refs = nearest_views(recon["extrinsic"], s, np.nonzero(video != video[s])[0], cfg["views"])
        warped, seen = warp(s, torch.as_tensor(refs, device=device), depth, K, E, images, valid,
                            cfg["tau_abs"], cfg["tau_rel"])
        # Unseen pixels: copy this frame's own pixels so they add no difference at patch level.
        warped = torch.where(seen[:, None], warped, images[s][None])
        f = feats(torch.cat([images[s][None], warped]))
        d = 1 - (f[1:] * f[:1]).sum(1)  # cosine distance per warped view, on the patch grid
        d = F.interpolate(d[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        d = torch.where(seen, d, torch.full_like(d, float("inf")))
        n_seen = seen.sum(0)
        best = d.min(0).values
        diff[s] = torch.where(n_seen >= cfg["min_views"], best, torch.full_like(best, float("nan")))
    del feats
    torch.cuda.empty_cache()
    return diff.cpu().numpy()


def threshold(diff, cfg):
    """Changed-pixel masks. Threshold adapts to this pair: robust z-score with an absolute floor."""
    vals = diff[np.isfinite(diff)]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826
    thr = max(cfg["abs_threshold"], med + cfg["mad_k"] * mad)
    return np.isfinite(diff) & (diff > thr), float(thr)


@torch.no_grad()
def visual_hull(masks, frames, recon, bounds, voxel, cfg, device="cuda"):
    """Voxels that project inside the changed pixels of (nearly) every frame that sees them.

    Returns voxel centres (N, 3), their in-mask ratio and the number of frames that saw them."""
    lo, hi = bounds
    axes = [torch.arange(lo[i], hi[i] + voxel, voxel, device=device) for i in range(3)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).float()
    E = torch.as_tensor(recon["extrinsic"][frames], device=device).float()
    K = torch.as_tensor(recon["intrinsic"][frames], device=device).float()
    D = torch.as_tensor(recon["depth"][frames], device=device).float()
    M = torch.as_tensor(masks[frames], device=device)
    T, H, W = D.shape
    inside = torch.zeros(len(grid), device=device)
    seen = torch.zeros(len(grid), device=device)
    for i in range(0, len(grid), 400_000):
        X = grid[i:i + 400_000]
        u, v, z = project(X, E, K)
        ui, vi = u.round().long(), v.round().long()
        inb = (z > 0.05) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi = ui.clamp(0, W - 1), vi.clamp(0, H - 1)
        t_idx = torch.arange(T, device=device)[:, None].expand_as(ui)
        d = D[t_idx, vi, ui]
        # A voxel hidden behind a nearer surface in this frame says nothing about it.
        visible = inb & ((d <= 0) | (z <= d + torch.clamp(cfg["tau_rel"] * z, min=cfg["tau_abs"])))
        inside[i:i + 400_000] = (visible & M[t_idx, vi, ui]).sum(0).float()
        seen[i:i + 400_000] = visible.sum(0).float()
    ratio = inside / seen.clamp_min(1)
    keep = (seen >= cfg["hull_min_views"]) & (ratio >= cfg["hull_ratio"])
    return grid[keep].cpu().numpy(), ratio[keep].cpu().numpy(), seen[keep].cpu().numpy()


# ---------------------------------------------------------------------------------------------
# From changed pixels to changed objects
# ---------------------------------------------------------------------------------------------

def clean_masks(masks, min_area=30):
    """Drop tiny speckles from each frame's changed-pixel mask."""
    import cv2
    out = np.zeros_like(masks)
    k = np.ones((3, 3), np.uint8)
    for i, m in enumerate(masks):
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        out[i] = keep[lab]
    return out


def hull_clusters(X, seen, voxel, cfg):
    """DBSCAN on hull voxels, each cluster trimmed to the voxels most views agree on.

    Both videos move only a little, so a visual hull stretches along the viewing direction;
    the stretched ends are seen by fewer frames than the object itself."""
    from sklearn.cluster import DBSCAN
    if len(X) == 0:
        return []
    lab = DBSCAN(eps=voxel * 2.05, min_samples=3).fit_predict(X)
    clusters = []
    for l in set(lab) - {-1}:
        m = lab == l
        if m.sum() < cfg["hull_min_voxels"]:
            continue
        core = m & (seen >= cfg["hull_core_views"] * seen[m].max())
        if core.sum() >= cfg["hull_min_voxels"]:
            clusters.append(X[core])
    return clusters


def scene_bounds(recon):
    """Axis-aligned box around the reconstructed scene (1st-99th percentile, +10 cm)."""
    pts = []
    for s in range(len(recon["depth"])):
        m = recon["valid"][s] & (recon["depth"][s] > 0)
        P = cam_to_world(backproject(torch.as_tensor(recon["depth"][s:s + 1]),
                                     torch.as_tensor(recon["intrinsic"][s:s + 1])),
                         torch.as_tensor(recon["extrinsic"][s:s + 1]))[0].numpy()
        pts.append(P[m][::7])
    pts = np.concatenate(pts)
    return np.percentile(pts, 1, 0) - 0.1, np.percentile(pts, 99, 0) + 0.1
