"""Step 1: pick sharp, evenly spread keyframes from a video (kept in memory, never written to disk).

The video is cut into equal time slots and the sharpest frame (variance of the
Laplacian) is kept from each slot. This keeps coverage of the whole walk while
avoiding motion-blurred frames, which give poor depth.
"""

import cv2
import numpy as np


def sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = 640.0 / max(gray.shape)
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_keyframes(video_path, n_keyframes, candidates_per_keyframe=6, max_side=1920):
    """Returns (frame indices, RGB frames) of `n_keyframes` sharp frames in time order."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    n_candidates = min(total, n_keyframes * candidates_per_keyframe)
    candidate_ids = np.unique(np.linspace(0, total - 1, n_candidates).round().astype(int))
    slot_lookup = dict(zip(candidate_ids.tolist(),
                           np.minimum((candidate_ids * n_keyframes) // total, n_keyframes - 1).tolist()))

    best = {}  # slot -> (score, frame index, frame)
    idx = 0
    while True:
        ok = cap.grab()  # grab every frame: seeking is unreliable on phone codecs
        if not ok:
            break
        if idx in slot_lookup:
            ok, frame = cap.retrieve()
            if ok:
                s, slot = sharpness(frame), slot_lookup[idx]
                if slot not in best or s > best[slot][0]:
                    best[slot] = (s, idx, frame)
        idx += 1
    cap.release()

    indices, frames = [], []
    for slot in sorted(best):
        _, frame_idx, frame = best[slot]
        scale = max_side / max(frame.shape[:2])
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        indices.append(frame_idx)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.array(indices), frames
