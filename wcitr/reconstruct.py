"""Step 2: one shared 3D scene from both videos.

Keyframes from the before AND after video go through VGGT in a single forward
pass, so all cameras and depth maps come out in one coordinate frame by
construction (no separate registration step that a moved sofa could break).

Then:
  * metric scale: VGGT's scale is arbitrary; a metric depth network
    (Depth Anything V2, indoor) gives the factor to metres (median ratio).
  * room frame: z points up (from camera orientation + floor plane),
    the floor is z = 0, the origin is below the first before-camera and
    +y is that camera's horizontal viewing direction.
"""

from collections import Counter

import numpy as np
import open3d as o3d
import torch
from PIL import Image

from .geometry import apply_world_transform, backproject, cam_to_world, camera_centers


def preprocess(frames, resolution=518):
    """Resize RGB frames into one common size (multiple of 14), padding if aspect differs.

    Returns images (S, 3, H, W) in [0, 1], valid-pixel masks (S, H, W) and, per frame,
    (scale, offset_x, offset_y) so that original_pixel = (pixel - offset) / scale.
    """
    sizes = [(f.shape[1], f.shape[0]) for f in frames]
    w0, h0 = Counter(sizes).most_common(1)[0][0]
    if w0 >= h0:
        W, H = resolution, max(14, round(h0 * resolution / w0 / 14) * 14)
    else:
        H, W = resolution, max(14, round(w0 * resolution / h0 / 14) * 14)

    images = np.ones((len(frames), H, W, 3), np.float32)
    valid = np.zeros((len(frames), H, W), bool)
    mapping = np.zeros((len(frames), 3), np.float32)
    for i, f in enumerate(frames):
        img = Image.fromarray(f)
        w, h = img.size
        s = min(W / w, H / h)
        nw, nh = min(W, round(w * s)), min(H, round(h * s))
        ox, oy = (W - nw) // 2, (H - nh) // 2
        images[i, oy:oy + nh, ox:ox + nw] = np.asarray(img.resize((nw, nh), Image.Resampling.BICUBIC)) / 255.0
        valid[i, oy:oy + nh, ox:ox + nw] = True
        mapping[i] = (nw / w, ox, oy)
    return torch.from_numpy(images).permute(0, 3, 1, 2), valid, mapping


@torch.no_grad()
def run_vggt(images, model_name, device="cuda"):
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    model = VGGT.from_pretrained(model_name)
    model.point_head = None  # depth + cameras are more accurate than the point head; saves memory
    model.track_head = None
    model = model.to(device).eval()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(images.to(device))
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"VGGT ran out of GPU memory with {len(images)} images. "
            "Lower frames.per_video in config.yaml."
        ) from e
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    out = {
        "extrinsic": extri[0].float().cpu().numpy(),
        "intrinsic": intri[0].float().cpu().numpy(),
        "depth": pred["depth"][0, ..., 0].float().cpu().numpy(),
        "conf": pred["depth_conf"][0].float().cpu().numpy(),
    }
    del model, pred
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def metric_scale(images, valid, depth, conf, model_name, device="cuda"):
    """Median ratio between metric monocular depth and VGGT depth over confident pixels."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()
    ratios = []
    for i in range(len(images)):
        ys, xs = np.nonzero(valid[i])
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgb = (images[i].permute(1, 2, 0).numpy()[y0:y1, x0:x1] * 255).astype(np.uint8)
        inputs = processor(images=Image.fromarray(rgb), return_tensors="pt").to(device)
        pred = model(**inputs).predicted_depth  # (1, h, w), metres
        metric = torch.nn.functional.interpolate(pred[:, None], size=rgb.shape[:2], mode="bilinear")[0, 0]
        metric = metric.cpu().numpy()
        d, c = depth[i, y0:y1, x0:x1], conf[i, y0:y1, x0:x1]
        m = (c >= np.quantile(c, 0.5)) & (d > 0) & (metric > 0.1) & (metric < 10)
        if m.sum() > 100:
            ratios.append(np.median(metric[m] / d[m]))
    del model
    torch.cuda.empty_cache()
    if not ratios:
        raise RuntimeError("Metric depth gave no usable pixels")
    ratios = np.array(ratios)
    spread = (np.percentile(ratios, 75) - np.percentile(ratios, 25)) / np.median(ratios)
    return float(np.median(ratios)), float(spread)


def room_frame(points, E, video):
    """4x4 transform from VGGT world to the room frame (z up, floor at z = 0)."""
    R = E[:, :3, :3]
    # Phones are held roughly upright: the camera's y axis (down) points at the floor.
    up0 = -R[:, 1, :].mean(axis=0)
    up0 /= np.linalg.norm(up0)

    h = points @ up0
    low = points[h < np.percentile(h, 30)]
    up, floor_h = up0, np.percentile(h, 2)
    if len(low) > 500:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(low))
        plane, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=2000)
        n = np.array(plane[:3]) / np.linalg.norm(plane[:3])
        if n @ up0 < 0:
            n, plane = -n, -np.array(plane)
        if n @ up0 > np.cos(np.deg2rad(20)) and len(inliers) > 0.15 * len(low):
            up, floor_h = n, -plane[3] / np.linalg.norm(plane[:3])

    c0 = camera_centers(E[:1])[0]
    fwd = R[0, 2, :] - (R[0, 2, :] @ up) * up
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    Rw = np.stack([right, fwd, up])  # rows: room x, y, z expressed in VGGT world
    origin = c0 - (c0 @ up - floor_h) * up
    M = np.eye(4)
    M[:3, :3] = Rw
    M[:3, 3] = -Rw @ origin
    return M


def reconstruct(keyframes, cfg, device="cuda"):
    """keyframes: {"before": (frame indices, RGB frames), "after": (...)}."""
    idx_b, fr_b = keyframes["before"]
    idx_a, fr_a = keyframes["after"]
    frames = list(fr_b) + list(fr_a)
    video = np.array([0] * len(fr_b) + [1] * len(fr_a), np.int8)
    images, valid, mapping = preprocess(frames, cfg["reconstruction"]["resolution"])

    out = run_vggt(images, cfg["reconstruction"]["model"], device)
    depth, conf, E, K = out["depth"], out["conf"], out["extrinsic"], out["intrinsic"]
    depth[~valid] = 0.0
    conf[~valid] = 0.0

    if cfg["scale"]["method"] == "metric_depth":
        s, spread = metric_scale(images, valid, depth, conf, cfg["scale"]["model"], device)
        scale_info = {"method": "metric_depth", "factor": s, "relative_spread": spread}
    else:
        good = depth[conf >= np.quantile(conf[valid], 0.5)]
        s = cfg["scale"]["fallback_median_depth"] / np.median(good)
        scale_info = {"method": "assumed_median_depth", "factor": float(s), "relative_spread": None}
    depth = depth * s
    E = E.copy()
    E[:, :3, 3] *= s

    # Confident points for floor fitting.
    with torch.no_grad():
        P = cam_to_world(backproject(torch.from_numpy(depth), torch.from_numpy(K)), torch.from_numpy(E)).numpy()
    m = valid & (conf >= np.quantile(conf[valid], 0.5))
    pts = P[m]
    pts = pts[np.random.default_rng(0).choice(len(pts), min(len(pts), 300_000), replace=False)]
    M = room_frame(pts, E, video)
    E = apply_world_transform(E, M).astype(np.float32)

    return {
        "images": (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8),
        "valid": valid,
        "depth": depth.astype(np.float32),
        "conf": conf.astype(np.float32),
        "extrinsic": E,
        "intrinsic": K.astype(np.float32),
        "video": video,
        "mapping": mapping,
        "frame_idx": np.concatenate([idx_b, idx_a]),
        "key_hw": np.array([f.shape[:2] for f in frames]),
        "scale_info": scale_info,
    }
