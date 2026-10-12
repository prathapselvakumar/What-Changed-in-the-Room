"""Side-by-side comparison video: reference (before) | test (after), with the 3D changes drawn in.

Camera poses are known only for keyframes, so poses for every other frame are interpolated
(rotation: spherical interpolation, position: linear). Each detected 3D box is then projected
into every frame of both videos:

    reference side: REMOVED objects (red, solid), the OLD position of moved objects (blue),
                    and dashed outlines where NEW objects will appear (the spot was empty)
    test side:      NEW objects (green), MISSING items as a dashed red box on the now-empty
                    spot, and moved objects at their NEW position with an arrow from the old one
    both sides:     a light yellow tint on surfaces only that video saw (unknown, not checked)
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .changes import UNKNOWN
from .geometry import rotation_about_z

COLORS = {"removed": (40, 39, 214), "added": (44, 160, 44), "moved": (180, 119, 31), "rotated": (189, 103, 148)}
TITLES = {"removed": "REMOVED", "added": "NEW", "moved": "MOVED", "rotated": "ROTATED"}
PANEL_W, PANEL_H = 960, 540
FOOTER_H = 170
EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return frames, fps


def frame_cameras(recon, vid, n_frames, frame_shape):
    """Projection data (K, R, t in original video pixels) for every frame of one video."""
    idx = np.nonzero(recon["video"] == vid)[0]
    fidx = recon["frame_idx"][idx]
    order = np.argsort(fidx)
    idx, fidx = idx[order], fidx[order]
    E = recon["extrinsic"][idx]
    R_c2w = np.transpose(E[:, :3, :3], (0, 2, 1))
    C = -np.einsum("sij,sj->si", R_c2w, E[:, :3, 3])

    # Model pixels -> keyframe pixels -> video pixels (keyframes may be stored smaller than the video).
    s, ox, oy = recon["mapping"][idx[0]]
    scale = frame_shape[1] / recon["key_hw"][idx[0]][1]
    K = recon["intrinsic"][idx[0]].astype(float).copy()
    K[0, 0], K[1, 1] = K[0, 0] / s * scale, K[1, 1] / s * scale
    K[0, 2], K[1, 2] = (K[0, 2] - ox) / s * scale, (K[1, 2] - oy) / s * scale

    t = np.clip(np.arange(n_frames), fidx[0], fidx[-1])
    if len(fidx) > 1:
        rots = Slerp(fidx, Rotation.from_matrix(R_c2w))(t).as_matrix()
        cens = np.stack([np.interp(t, fidx, C[:, k]) for k in range(3)], 1)
    else:
        rots, cens = np.repeat(R_c2w, n_frames, 0), np.repeat(C, n_frames, 0)
    return [(K, r.T, -r.T @ c) for r, c in zip(rots, cens)]


def box_corners(center, size, yaw):
    h = np.array(size) / 2
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    return np.array(center) + (signs * h) @ rotation_about_z(yaw).T


def project(K, R, t, X):
    Xc = X @ R.T + t
    z = Xc[:, 2]
    uv = (Xc[:, :2] / np.maximum(z, 1e-6)[:, None]) @ K[:2, :2].T + K[:2, 2]
    return uv, z


def draw_box(img, cam, obj, color, label=None, dashed=False, thickness=3):
    K, R, t = cam
    uv, z = project(K, R, t, box_corners(obj["center_m"], obj["size_m"], obj["yaw_deg"]))
    H, W = img.shape[:2]
    if (z <= 0.05).all():
        return None
    inside = (uv[:, 0] > -W) & (uv[:, 0] < 2 * W) & (uv[:, 1] > -H) & (uv[:, 1] < 2 * H) & (z > 0.05)
    if not inside.any():
        return None
    for a, b in EDGES:
        if z[a] > 0.05 and z[b] > 0.05:
            p, q = uv[a], uv[b]
            if dashed:
                for k in range(0, 10, 2):
                    p1 = p + (q - p) * k / 10
                    p2 = p + (q - p) * (k + 1) / 10
                    cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)), color, max(1, thickness - 1), cv2.LINE_AA)
            else:
                cv2.line(img, tuple(map(int, p)), tuple(map(int, q)), color, thickness, cv2.LINE_AA)
    vis = uv[z > 0.05]
    top = vis[np.argmin(vis[:, 1])]
    if label and 0 <= top[0] < W and -20 <= top[1] < H:
        put_label(img, label, (int(top[0]), int(max(top[1], 30))), color)
    center_uv, cz = project(K, R, t, np.array([obj["center_m"]]))
    return center_uv[0] if cz[0] > 0.05 else None


def draw_object(img, cam, obj, color, label, seg=None):
    """Exact silhouette of an object that IS in this frame: SAM mask from the projected 3D box,
    drawn as a contour with a light fill. Falls back to the 3D box if SAM finds nothing."""
    if seg is None:
        return draw_box(img, cam, obj, color, label)
    from .segment import box_corners as corners_of, project_box
    K, R, t = cam
    H, W = img.shape[:2]
    corners = corners_of(obj["center_m"], np.array(obj["size_m"]) / 2, obj["yaw_deg"])
    box = project_box(corners, np.hstack([R, t[:, None]]), K, H, W, pad=0.2)
    mask = seg.mask(np.ascontiguousarray(img[..., ::-1]), box) if box is not None else None
    if mask is None:
        return draw_box(img, cam, obj, color, label)
    overlay = img.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, color, 3, cv2.LINE_AA)
    ys, xs = np.nonzero(mask)
    if label:
        put_label(img, label, (int(xs.mean()), int(max(ys.min(), 30))), color)
    return np.array([xs.mean(), ys.mean()])


def put_label(img, text, org, color, scale=0.8):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    x = int(np.clip(x - w // 2, 0, img.shape[1] - w - 6))
    y = int(np.clip(y - 10, h + 6, img.shape[0] - 4))
    cv2.rectangle(img, (x - 4, y - h - 6), (x + w + 4, y + base - 2), color, -1)
    cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def annotate(frame, cam, events, side, seg=None):
    img = frame.copy()
    for e in events:
        col = COLORS[e["type"]]
        name = f"{e['id']}. {TITLES[e['type']]}: {e['label']}"
        if side == "before":
            if e["type"] == "removed":
                draw_object(img, cam, e["before"], col, name, seg)
            elif e["type"] == "added":  # the spot where something will appear: empty in the reference
                draw_box(img, cam, e["after"], col, f"{e['id']}. empty here before", dashed=True, thickness=2)
            elif e["type"] in ("moved", "rotated"):
                draw_object(img, cam, e["before"], col, f"{e['id']}. {e['label']} (old position)", seg)
        else:
            if e["type"] == "added":
                draw_object(img, cam, e["after"], col, name, seg)
            elif e["type"] == "removed":  # mark the now-empty spot where the item used to be
                draw_box(img, cam, e["before"], col, f"{e['id']}. MISSING: {e['label']}", dashed=True)
            elif e["type"] in ("moved", "rotated"):
                old = draw_box(img, cam, e["before"], col, None, dashed=True, thickness=2)
                new = draw_object(img, cam, e["after"], col, f"{name} {e.get('distance_m', 0):.2f} m", seg)
                if old is not None and new is not None:
                    cv2.arrowedLine(img, tuple(map(int, old)), tuple(map(int, new)), col, 3, cv2.LINE_AA, tipLength=0.15)
    return img


def header(img, text, color):
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (20, 20, 20), -1)
    cv2.putText(img, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


UNKNOWN_BGR = (138, 213, 242)  # same light yellow as the top-down map


def unchecked_tint(img, cam, xyz_all, unknown_mask, scale=4, alpha=0.35):
    """Tint the surfaces this video saw that the other video never saw (state UNKNOWN).

    Voxels are splatted into a quarter-resolution depth buffer so only surfaces facing this
    camera are tinted, not the unknown voxels hidden behind nearer geometry."""
    K, R, t = cam
    H, W = img.shape[:2]
    h, w = H // scale, W // scale
    Xc = xyz_all @ R.T + t
    z = Xc[:, 2]
    ok = z > 0.05
    uv = (Xc[ok, :2] / z[ok, None]) @ K[:2, :2].T + K[:2, 2]
    u, v = (uv[:, 0] / scale).astype(int), (uv[:, 1] / scale).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, zz, unk = u[inside], v[inside], z[ok][inside], unknown_mask[ok][inside]
    if not unk.any():
        return img
    zbuf = np.full((h, w), np.inf, np.float32)
    np.minimum.at(zbuf, (v, u), zz)
    zbuf = cv2.erode(zbuf, np.ones((3, 3), np.uint8))  # min over a 3x3 neighbourhood
    visible = unk & (zz <= zbuf[v, u] + 0.04)
    mask = np.zeros((h, w), np.uint8)
    mask[v[visible], u[visible]] = 255
    mask = cv2.morphologyEx(cv2.dilate(mask, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR) > 127
    if not mask.any():
        return img
    overlay = img.copy()
    overlay[mask] = UNKNOWN_BGR
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, [c for c in cs if cv2.contourArea(c) > 400], -1, UNKNOWN_BGR, 1, cv2.LINE_AA)
    return out


def footer(events, width, coverage):
    f = np.full((FOOTER_H, width, 3), 250, np.uint8)
    if not events:
        cv2.putText(f, "No changes found in the areas both videos could see.", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
    for k, e in enumerate(events[:5]):
        text = f"{e['id']}. {e.get('caption', e['text'])}"
        if e.get("verified_by_vlm") is False:
            text += "  (unconfirmed)"
        cv2.rectangle(f, (14, 16 + 28 * k), (30, 32 + 28 * k), COLORS[e["type"]], -1)
        cv2.putText(f, text[:150], (40, 31 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    if coverage:
        cv2.rectangle(f, (14, FOOTER_H - 26), (30, FOOTER_H - 10), UNKNOWN_BGR, -1)
        note = (f"Yellow tint: seen in only one video, so not checked ({coverage['before']['unknown_pct']:.0f}% of the "
                f"reference, {coverage['after']['unknown_pct']:.0f}% of the test). Changes there are not reported.")
        cv2.putText(f, note, (40, FOOTER_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    return f


def summary_card(out_dir, width, height):
    """Last seconds of the video: the top-down map."""
    card = np.full((height, width, 3), 255, np.uint8)
    top = cv2.imread(str(out_dir / "topdown.png"))
    if top is not None:
        s = min((height - 20) / top.shape[0], (width - 20) / top.shape[1])
        top = cv2.resize(top, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        y, x = (height - top.shape[0]) // 2, (width - top.shape[1]) // 2
        card[y:y + top.shape[0], x:x + top.shape[1]] = top
    return card


def make_comparison_video(out_dir, recon, report_data, before_path, after_path, out_name="comparison.mp4",
                          playback_fps=15, summary_seconds=4, segmenter=None, results=None):
    events = report_data["events"]
    fb, _ = read_frames(before_path)
    fa, _ = read_frames(after_path)
    cams_b = frame_cameras(recon, 0, len(fb), fb[0].shape)
    cams_a = frame_cameras(recon, 1, len(fa), fa[0].shape)

    width, height = 2 * PANEL_W, PANEL_H + FOOTER_H
    foot = footer(events, width, report_data.get("coverage"))
    n = max(len(fb), len(fa))
    raw = Path(out_dir) / ("_raw_" + out_name)
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), playback_fps, (width, height))
    for i in range(n):
        ib = round(i * (len(fb) - 1) / max(n - 1, 1))
        ia = round(i * (len(fa) - 1) / max(n - 1, 1))
        fb_i, fa_i = fb[ib], fa[ia]
        if results is not None:  # tint first, so outlines and labels stay on top
            fb_i = unchecked_tint(fb_i, cams_b[ib], results[0]["xyz"], results[0]["state"] == UNKNOWN)
            fa_i = unchecked_tint(fa_i, cams_a[ia], results[1]["xyz"], results[1]["state"] == UNKNOWN)
        left = cv2.resize(annotate(fb_i, cams_b[ib], events, "before", segmenter), (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
        right = cv2.resize(annotate(fa_i, cams_a[ia], events, "after", segmenter), (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
        header(left, "REFERENCE (before)", (255, 255, 255))
        header(right, "TEST (after)", (255, 255, 255))
        writer.write(np.vstack([np.hstack([left, right]), foot]))
    card = summary_card(Path(out_dir), width, height)
    for _ in range(int(summary_seconds * playback_fps)):
        writer.write(card)
    writer.release()

    final = Path(out_dir) / out_name
    ffmpeg = shutil.which("ffmpeg") or _winget_ffmpeg()
    if ffmpeg:  # re-encode to H.264 so it plays in browsers and on GitHub
        r = subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)])
        if r.returncode == 0:
            raw.unlink()
            return final
    raw.replace(final)
    return final


def _winget_ffmpeg():
    base = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    hits = sorted(base.glob("Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe")) if base.exists() else []
    return str(hits[0]) if hits else None
