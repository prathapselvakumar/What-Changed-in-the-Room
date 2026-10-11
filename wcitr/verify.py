"""Appearance verification: do the two videos really look different at this 3D place?

Every candidate change (from the depth check or from the appearance channel) is projected into
pairs of matched views: an after frame and the before frame with the closest camera pose. The
two crops are compared with DINOv2 (mean patch embedding). A static object looks the same in
both crops; a removed or added object does not.

On the recorded data, real changes scored 0.40-0.71 and static places 0.76-0.89 (threshold 0.72).
The score must also be clearly below that of nearby control places seen through the same view
pairs, because a large viewpoint difference lowers every score (config: verify_gap).

For candidates found only by the appearance channel (fully transparent items the depth map
ignores), `which_side` decides removed vs added with differential zero-shot labelling: for every
object name, CLIP scores how much better it describes the before crops than the after crops.
The name with the largest difference is the label, and its sign says which video holds it.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

from . import appearance
from .semantics import VOCAB

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class CropComparer:
    def __init__(self, model_name, device="cuda"):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, crops):
        x = torch.stack([torch.as_tensor(np.asarray(Image.fromarray(c).resize((224, 224)))).permute(2, 0, 1)
                         for c in crops]).float().to(self.device) / 255
        x = (x - MEAN.to(self.device)) / STD.to(self.device)
        tok = self.model(pixel_values=x).last_hidden_state[:, 1:]
        return F.normalize(tok.mean(1), dim=-1)


def _box(X, E, K, H, W, pad=0.6, min_px=24):
    Xc = X @ E[:, :3].T + E[:, 3]
    if (Xc[:, 2] <= 0.05).any():
        return None
    uv = (Xc[:, :2] / Xc[:, 2:]) @ K[:2, :2].T + K[:2, 2]
    (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
    w, h = max(x1 - x0, min_px) * (1 + pad), max(y1 - y0, min_px) * (1 + pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    b = [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]
    return b if b[0] >= 0 and b[1] >= 0 and b[2] <= W and b[3] <= H else None


def matched_crops(X, recon, step=3):
    """Crop pairs (before, after) of the place X from matched views (closest camera poses)."""
    H, W = recon["depth"].shape[1:]
    before = np.nonzero(recon["video"] == 0)[0]
    after = np.nonzero(recon["video"] == 1)[0]
    pairs = []
    for a in after[::step]:
        b = appearance.nearest_views(recon["extrinsic"], a, before, 1)[0]
        ba = _box(X, recon["extrinsic"][a], recon["intrinsic"][a], H, W)
        bb = _box(X, recon["extrinsic"][b], recon["intrinsic"][b], H, W)
        if ba is not None and bb is not None:
            pairs.append((recon["images"][b][bb[1]:bb[3], bb[0]:bb[2]],
                          recon["images"][a][ba[1]:ba[3], ba[0]:ba[2]]))
    return pairs


def _pair_similarity(X, recon, comparer):
    pairs = matched_crops(X, recon)
    if not pairs:
        return float("nan"), pairs
    e = comparer.embed([c for p in pairs for c in p]).view(len(pairs), 2, -1)
    return float(torch.median((e[:, 0] * e[:, 1]).sum(-1))), pairs


def similarity(X, recon, comparer):
    """Before/after appearance similarity at X, and the same measure at control places.

    Controls are X shifted sideways by one object width (4 directions) and seen through the same
    kind of view pairs, so they share the viewpoint difference between the videos. A real change
    must look clearly less alike than its surroundings: from a very different angle everything
    looks less alike, and that alone is not a change. Returns (similarity, control median, pairs)."""
    sim, pairs = _pair_similarity(X, recon, comparer)
    size = X.max(0) - X.min(0)
    controls = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shift = np.array([dx * (size[0] + 0.05), dy * (size[1] + 0.05), 0.0])
        c, _ = _pair_similarity(X + shift, recon, comparer)
        if np.isfinite(c):
            controls.append(c)
    return sim, (float(np.median(controls)) if controls else float("nan")), pairs


def is_changed(sim, control, cfg):
    if not np.isfinite(sim):
        return False
    below_control = (not np.isfinite(control)) or sim <= control - cfg["verify_gap"]
    return sim < cfg["verify_threshold"] and below_control


@torch.no_grad()
def which_side(pairs, labeler):
    """(vid holding the object: 0 before / 1 after, label, margin) by differential zero-shot labelling."""
    eb = torch.stack([labeler.embed(Image.fromarray(b)) for b, _ in pairs]).mean(0)
    ea = torch.stack([labeler.embed(Image.fromarray(a)) for _, a in pairs]).mean(0)
    d = labeler.text_emb @ eb - labeler.text_emb @ ea
    k = int(d.abs().argmax())
    return (0 if d[k] > 0 else 1), VOCAB[k], float(d[k].abs())


def appearance_objects(recon, cfg, voxel, comparer, labeler, device="cuda"):
    """Transparent / depth-invisible changes: ({0: [(voxels, label)], 1: [...]}, info)."""
    diff = appearance.change_maps(recon, cfg, device)
    masks, thr = appearance.threshold(diff, cfg)
    masks = appearance.clean_masks(masks, cfg["min_mask_area"])
    bounds = appearance.scene_bounds(recon)

    cands = []  # the same place found from both videos is one candidate
    for vid in (0, 1):
        frames = np.nonzero(recon["video"] == vid)[0]
        X, _, seen = appearance.visual_hull(masks, frames, recon, bounds, voxel, cfg, device)
        for c in appearance.hull_clusters(X, seen, voxel, cfg):
            for k, other in enumerate(cands):
                if (cKDTree(other).query(c)[0] <= 2 * voxel).mean() > 0.3:
                    cands[k] = np.unique(np.concatenate([other, c]), axis=0)
                    break
            else:
                cands.append(c)

    out, verdicts = {0: [], 1: []}, []
    for c in cands:
        sim, control, pairs = similarity(c, recon, comparer)
        v = {"center": c.mean(0).round(3).tolist(), "voxels": int(len(c)),
             "similarity": round(sim, 3), "control": round(control, 3)}
        if is_changed(sim, control, cfg):
            side, label, margin = which_side(pairs, labeler)
            v.update(verdict="removed" if side == 0 else "added", label=label, margin=round(margin, 4))
            if margin >= cfg["side_margin"]:
                out[side].append((c, label))
        else:
            v["verdict"] = "static"
        verdicts.append(v)
    return out, {"threshold": round(thr, 3), "candidates": verdicts}
