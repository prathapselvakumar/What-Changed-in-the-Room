"""Exact object outlines with Segment Anything (SAM), prompted by the projected 3D box.

The 3D detection says roughly where an object is; SAM cuts out its exact silhouette in an
image. Used twice:
  * refine_extent: silhouettes from many keyframes carve the object's true 3D extent
    (space carving), which tightens boxes stretched by partial depth or by a visual hull;
  * the comparison video draws each object's real contour in every frame.
"""

import cv2
import numpy as np
import torch
from PIL import Image

from .geometry import rotation_about_z


class Segmenter:
    def __init__(self, model_name="facebook/sam-vit-large", device="cuda"):
        from transformers import SamModel, SamProcessor
        self.model = SamModel.from_pretrained(model_name).to(device).eval()
        self.proc = SamProcessor.from_pretrained(model_name)
        self.device = device

    @torch.no_grad()
    def mask(self, image, box):
        """Best SAM mask (H, W bool) for the object inside `box` (x0, y0, x1, y1) of an RGB image.

        Of SAM's three candidate masks, keeps the most confident one that fills a plausible part
        of the box (an object, not the whole box background and not a speck)."""
        H, W = image.shape[:2]
        x0, y0, x1, y1 = [float(v) for v in box]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        inputs = self.proc(images=Image.fromarray(image), input_boxes=[[[x0, y0, x1, y1]]],
                           return_tensors="pt").to(self.device)
        out = self.model(**inputs, multimask_output=True)
        masks = self.proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                             inputs["reshaped_input_sizes"].cpu())[0][0].numpy()  # (3, H, W)
        scores = out.iou_scores[0, 0].float().cpu().numpy()
        box_area = (x1 - x0) * (y1 - y0)
        best, best_score = None, -1.0
        for m, sc in zip(masks, scores):
            inside = m[int(y0):int(y1) + 1, int(x0):int(x1) + 1].sum()
            fill, spill = inside / box_area, 1 - inside / max(m.sum(), 1)
            if 0.08 <= fill <= 0.95 and spill < 0.3 and sc > best_score:
                best, best_score = m, sc
        return best


def box_corners(center, half, yaw):
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return np.asarray(center) + (signs * np.asarray(half)) @ rotation_about_z(yaw).T


def project_box(corners, E, K, H, W, pad=0.15):
    """2D bounding box of projected 3D corners, padded; None if behind the camera / off-image."""
    Xc = corners @ E[:, :3].T + E[:, 3]
    if (Xc[:, 2] <= 0.05).any():
        return None
    uv = (Xc[:, :2] / Xc[:, 2:]) @ K[:2, :2].T + K[:2, 2]
    (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
    w, h = x1 - x0, y1 - y0
    x0, y0, x1, y1 = x0 - pad * w, y0 - pad * h, x1 + pad * w, y1 + pad * h
    if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
        return None
    inside = (min(x1, W) - max(x0, 0)) * (min(y1, H) - max(y0, 0)) / max(w * h * (1 + 2 * pad) ** 2, 1)
    return (x0, y0, x1, y1) if inside > 0.6 else None


def refine_extent(obj, vid, recon, segmenter, voxel, max_views=12, keep_ratio=0.75):
    """Carve the object's 3D extent from SAM silhouettes in its own video's keyframes.

    A point of the (enlarged) current box is kept if it projects inside the silhouette in at
    least `keep_ratio` of the views that see it. For opaque objects it must also not lie in front
    of the surface the camera observed there (that space is visibly empty). Objects the depth map
    sees through (transparent) are detected automatically and carved by silhouettes only.
    Updates the object's box in place; returns the number of views used (0 = unchanged)."""
    frames = np.nonzero(recon["video"] == vid)[0]
    H, W = recon["depth"].shape[1:]
    half0 = obj["half_size"] * 1.3 + voxel
    corners = box_corners(obj["center"], half0, obj["yaw_deg"])
    views = [(f, b) for f in frames if (b := project_box(corners, recon["extrinsic"][f], recon["intrinsic"][f], H, W))]
    if len(views) < 3:
        return 0
    views = [views[i] for i in np.linspace(0, len(views) - 1, min(max_views, len(views))).astype(int)]

    masks, see_through = [], 0
    for f, b in views:
        m = segmenter.mask(recon["images"][f], b)
        if m is None or m.sum() < 20:
            continue
        E = recon["extrinsic"][f]
        zc = (obj["center"] @ E[:, :3].T + E[:, 3])[2]
        d = recon["depth"][f][m]
        d = d[d > 0]
        if len(d) and np.median(d) > zc + obj["half_size"].max() + 0.05:
            see_through += 1
        masks.append((f, m))
    if len(masks) < 3:
        return 0
    transparent = obj.get("source") == "appearance" or see_through >= 0.5 * len(masks)

    step = voxel / 2
    axes = [np.arange(-h, h + step, step) for h in half0]
    local = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    pts = obj["center"] + local @ rotation_about_z(obj["yaw_deg"]).T
    inside, seen = np.zeros(len(pts)), np.zeros(len(pts))
    for f, m in masks:
        E, K = recon["extrinsic"][f], recon["intrinsic"][f]
        Xc = pts @ E[:, :3].T + E[:, 3]
        z = Xc[:, 2]
        uv = np.round((Xc[:, :2] / np.maximum(z, 1e-6)[:, None]) @ K[:2, :2].T + K[:2, 2]).astype(int)
        ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        u, v = uv[ok, 0], uv[ok, 1]
        hit = m[v, u]
        d = recon["depth"][f][v, u]
        tau = max(0.03, 0.03 * float(np.median(z[ok])))
        if transparent:  # a see-through object stands in front of what is seen through it
            hit &= (d <= 0) | (z[ok] <= d + tau)
        else:            # space in front of an opaque object's visible surface is empty
            hit &= (d <= 0) | (z[ok] >= d - tau)
        inside[ok] += hit
        seen[ok] += 1
    keep = (seen >= 3) & (inside >= keep_ratio * seen)
    if keep.sum() < 20:
        return 0
    from .objects import oriented_box
    X = pts[keep]
    if transparent:
        X = _trim_view_stretch(X)
    center, half, yaw = oriented_box(X)
    obj.update(center=center, half_size=np.maximum(half, voxel / 2), yaw_deg=yaw,
               bottom=float(np.percentile(X[:, 2], 2)), top=float(np.percentile(X[:, 2], 98)),
               refined_views=len(masks), transparent=bool(transparent))
    return len(masks)


def _trim_view_stretch(X, max_ratio=1.2):
    """Silhouettes from two short videos filmed from similar directions cannot fix how deep a
    see-through object is: the carved shape stretches along the viewing direction. Assume the
    object is about as deep as it is wide, and that it rests on something: of all object-sized
    sections along the stretched horizontal axis, keep the one whose bottom is lowest (where the
    downward viewing rays meet the supporting surface)."""
    xy = X[:, :2] - np.median(X[:, :2], axis=0)
    w, vecs = np.linalg.eigh(np.cov(xy.T))
    a, b = xy @ vecs[:, 1], xy @ vecs[:, 0]
    half_len = max_ratio * (np.percentile(b, 98) - np.percentile(b, 2)) / 2
    if np.percentile(a, 98) - np.percentile(a, 2) <= 2 * half_len:
        return X
    best, best_bottom = None, np.inf
    for c in np.linspace(a.min() + half_len, a.max() - half_len, 25):
        keep = np.abs(a - c) <= half_len
        if keep.sum() < 20:
            continue
        bottom = np.percentile(X[keep, 2], 2)
        if bottom < best_bottom:
            best, best_bottom = keep, bottom
    return X[best] if best is not None else X


def contour(mask, img, color, thickness=3):
    """Draw the outline of a mask on img (in place)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) > 30]
    cv2.drawContours(img, cs, -1, color, thickness, cv2.LINE_AA)
    return cs
