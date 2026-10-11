"""Step 4a: turn changed voxels into objects, then into events.

  * cluster changed voxels per video (DBSCAN)
  * complete each object with adjacent voxels that were never checkable
    (e.g. the back of a removed box the after video never looked at)
  * match removed objects (before) with added objects (after) by appearance,
    colour, size and label; a confident match is a MOVED object and ICP gives
    its displacement and rotation
"""

import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from .changes import CHANGED, UNKNOWN
from .geometry import rotation_about_z


def cluster_changes(res, cfg, voxel):
    """List of objects (dicts) from one video's voxel result."""
    changed = np.nonzero(res["state"] == CHANGED)[0]
    if len(changed) == 0:
        return []
    eps = cfg["dbscan_eps_voxels"] * voxel * 1.01
    labels = DBSCAN(eps=eps, min_samples=cfg["dbscan_min_samples"]).fit_predict(res["xyz"][changed])

    unknown = np.nonzero((res["state"] == UNKNOWN) & (res["xyz"][:, 2] > cfg["floor_clearance"]))[0]
    unknown_tree = cKDTree(res["xyz"][unknown]) if len(unknown) else None
    vf = res["voxel_frames"]

    labels = merge_vertical_parts(res["xyz"][changed], labels, cfg)

    objects = []
    for lab in sorted(set(labels) - {-1}):
        core = changed[labels == lab]
        frames = np.unique(vf[np.isin(vf[:, 0], core), 1])
        xyz = res["xyz"][core]
        extent = xyz.max(0) - xyz.min(0)
        if len(core) < cfg["min_voxels"] or extent.max() < cfg["min_extent"] or len(frames) < cfg["min_frames"]:
            continue
        members = _grow(core, res["xyz"], unknown, unknown_tree, eps, cfg["grow_iterations"])
        objects.append(_describe(members, core, res, frames))
    return objects


def merge_vertical_parts(xyz, labels, cfg):
    """Join clusters stacked above each other with overlapping footprints into one object.

    Depth networks see through transparent or very thin parts (a clear jug's body, chair legs),
    so one object can come out as separate pieces at different heights (lid and base).
    """
    ids = sorted(set(labels) - {-1})
    if len(ids) < 2:
        return labels
    info = {}
    for i in ids:
        p = xyz[labels == i]
        c = p[:, :2].mean(0)
        r = np.percentile(np.linalg.norm(p[:, :2] - c, axis=1), 90)
        info[i] = (c, r, p[:, 2].min(), p[:, 2].max())
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in ids:
        for b in ids:
            if a >= b:
                continue
            ca, ra, lo_a, hi_a = info[a]
            cb, rb, lo_b, hi_b = info[b]
            overlap_xy = np.linalg.norm(ca - cb) <= max(ra, rb) + cfg["stack_xy_margin"]
            gap_z = max(lo_a, lo_b) - min(hi_a, hi_b)
            if overlap_xy and gap_z <= cfg["stack_max_gap"]:
                parent[find(b)] = find(a)
    return np.array([find(l) if l != -1 else -1 for l in labels])


def _grow(core, xyz, unknown, tree, eps, iterations):
    if tree is None or iterations == 0:
        return core
    members, frontier = set(core.tolist()), xyz[core]
    for _ in range(iterations):
        hits = {int(unknown[j]) for lst in tree.query_ball_point(frontier, eps) for j in lst}
        new = hits - members
        if not new:
            break
        members |= new
        frontier = xyz[list(new)]
    return np.array(sorted(members))


def oriented_box(xyz):
    """Gravity-aligned box: yaw from horizontal PCA. Returns center, half sizes, yaw (deg)."""
    xy = xyz[:, :2] - xyz[:, :2].mean(0)
    if len(xy) >= 3:
        w, vecs = np.linalg.eigh(np.cov(xy.T))
        major = vecs[:, np.argmax(w)]
        yaw = np.degrees(np.arctan2(major[1], major[0]))
    else:
        yaw = 0.0
    R = rotation_about_z(yaw)
    local = xyz @ R  # into the box frame
    lo, hi = np.percentile(local, 2, axis=0), np.percentile(local, 98, axis=0)
    center = R @ ((lo + hi) / 2)
    return center, (hi - lo) / 2, float(yaw)


def _describe(members, core, res, frames):
    xyz, rgb = res["xyz"][members], res["rgb"][members]
    center, half, yaw = oriented_box(xyz)
    return {
        "voxels": members,
        "xyz": xyz,
        "rgb": rgb,
        "center": center,
        "half_size": half,
        "yaw_deg": yaw,
        "bottom": float(np.percentile(xyz[:, 2], 2)),
        "top": float(np.percentile(xyz[:, 2], 98)),
        "n_core_voxels": int(len(core)),
        "n_frames": int(len(frames)),
        "free_ratio": float(res["free_ratio"][core].mean()),
    }


def color_similarity(rgb_a, rgb_b, bins=8):
    """Bhattacharyya coefficient between coarse RGB histograms."""
    def hist(rgb):
        q = (rgb.astype(int) * bins) // 256
        h = np.bincount(q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2], minlength=bins ** 3).astype(float)
        return h / h.sum()
    return float(np.sqrt(hist(rgb_a) * hist(rgb_b)).sum())


def size_similarity(half_a, half_b):
    a, b = np.sort(half_a)[::-1], np.sort(half_b)[::-1]
    return float(np.mean(np.minimum(a, b) / np.maximum(a, b).clip(1e-6)))


def pair_score(a, b):
    cos = float(np.dot(a["embedding"], b["embedding"])) if "embedding" in a and "embedding" in b else 0.6
    s_app = np.clip((cos - 0.6) / 0.35, 0, 1)
    s_col = color_similarity(a["rgb"], b["rgb"])
    s_size = size_similarity(a["half_size"], b["half_size"])
    same_label = float(a.get("label") is not None and a.get("label", "").lower() == b.get("label", "").lower())
    return 0.5 * s_app + 0.2 * s_col + 0.15 * s_size + 0.15 * same_label


def _normals(xyz, cams, voxel):
    """Normals oriented towards the cameras that observed the points."""
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
    n = np.asarray(pcd.normals)
    C = cams["centers"]
    to_cam = C[np.argmin(((xyz[:, None] - C[None]) ** 2).sum(-1), axis=1)] - xyz
    return np.where(((n * to_cam).sum(1) < 0)[:, None], -n, n)


def expected_visible(xyz, normals, cams, min_views=2):
    """Points facing at least `min_views` cameras and inside their image (occlusion ignored)."""
    E, K, (H, W) = cams["E"], cams["K"], cams["hw"]
    Xc = np.einsum("sij,pj->spi", E[:, :, :3], xyz) + E[:, None, :, 3]
    z = Xc[..., 2]
    u = K[:, None, 0, 0] * Xc[..., 0] / np.maximum(z, 1e-6) + K[:, None, 0, 2]
    v = K[:, None, 1, 1] * Xc[..., 1] / np.maximum(z, 1e-6) + K[:, None, 1, 2]
    view = cams["centers"][:, None] - xyz[None]
    facing = (view * normals[None]).sum(-1) > 0.2 * np.linalg.norm(view, axis=-1)
    seen = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & facing
    return seen.sum(0) >= min_views


def _apply(T, xyz):
    return xyz @ T[:3, :3].T + T[:3, 3]


def visible_fitness(T, a, b, cams, voxel):
    """Symmetric overlap, counted only on the points the OTHER video could have seen.

    Each video usually sees different sides of an object; a side the other video never faced
    must not count against a pose.
    """
    scores = []
    for src, dst, T_, other in ((a, b, T, cams[1]), (b, a, np.linalg.inv(T), cams[0])):
        p = _apply(T_, src["xyz"])
        n = src["normals"] @ T_[:3, :3].T
        m = expected_visible(p, n, other)
        if m.sum() < 0.2 * len(p):
            m = np.ones(len(p), bool)  # too few expected points: fall back to plain overlap
        d, _ = cKDTree(dst["xyz"]).query(p[m])
        scores.append(float((d <= 2 * voxel).mean()))
    return min(scores)


def _translation_only(a, b, voxel, cams=None, iterations=10):
    """Translation for the 'not rotated' hypothesis.

    Restricted to surfaces BOTH videos could see, the two point sets describe the same physical
    patch, so their centroids correspond. (Trimmed ICP would discard exactly the boundary
    correspondences that carry the remaining offset and stall near its start.)
    """
    t = b["center"] - a["center"]
    if cams is None or "normals" not in a or "normals" not in b:
        return _icp_translation(a, b, t)
    for _ in range(iterations):
        ma = expected_visible(a["xyz"] + t, a["normals"], cams[1])
        mb = expected_visible(b["xyz"] - t, b["normals"], cams[0])
        if ma.sum() < 0.2 * len(ma) or mb.sum() < 0.2 * len(mb):
            return _icp_translation(a, b, t)
        t_new = b["xyz"][mb].mean(0) - a["xyz"][ma].mean(0)
        done = np.linalg.norm(t_new - t) < 1e-4
        t = t_new
        if done:
            break
    T = np.eye(4)
    T[:3, 3] = t
    return T


def _icp_translation(a, b, t, iterations=40):
    """Plain symmetric nearest-neighbour translation ICP (fallback without cameras)."""
    tree_a, tree_b = cKDTree(a["xyz"]), cKDTree(b["xyz"])
    for _ in range(iterations):
        _, j1 = tree_b.query(a["xyz"] + t)
        _, j2 = tree_a.query(b["xyz"] - t)
        step = np.concatenate([b["xyz"][j1] - (a["xyz"] + t), (b["xyz"] - t) - a["xyz"][j2]]).mean(0)
        t = t + step
        if np.linalg.norm(step) < 1e-4:
            break
    T = np.eye(4)
    T[:3, 3] = t
    return T


def register(a, b, voxel, cams=None, rotation_margin=0.2):
    """Rigid transform taking object a (before) onto object b (after), and its fitness.

    Two hypotheses: 'not rotated' (translation only) and 'rotated' (ICP from 12 yaw seeds).
    Both are scored by visibility-aware overlap. The rotated pose wins only if it fits better
    by `rotation_margin`. Partial views of symmetric objects (boxes, round bins) fit several
    rotations; a phantom rotation is worse than reporting none.
    """
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a["xyz"].astype(np.float64)))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b["xyz"].astype(np.float64)))
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    def fitness(T):
        if cams is not None and "normals" in a and "normals" in b:
            return visible_fitness(T, a, b, cams, voxel)
        fwd = o3d.pipelines.registration.evaluate_registration(src, dst, 2 * voxel, T).fitness
        bwd = o3d.pipelines.registration.evaluate_registration(dst, src, 2 * voxel, np.linalg.inv(T)).fitness
        return min(fwd, bwd)

    T_still = _translation_only(a, b, voxel, cams)
    fit_still = fitness(T_still)
    best_rot = (-1.0, None)
    for yaw in range(0, 360, 30):
        T0 = np.eye(4)
        T0[:3, :3] = rotation_about_z(yaw)
        T0[:3, 3] = b["center"] - T0[:3, :3] @ a["center"]
        reg = o3d.pipelines.registration.registration_icp(src, dst, 5 * voxel, T0, p2p)
        reg = o3d.pipelines.registration.registration_icp(src, dst, 2 * voxel, reg.transformation, p2p)
        f = fitness(reg.transformation)
        if f > best_rot[0]:
            best_rot = (f, reg.transformation)
    if best_rot[0] >= fit_still + rotation_margin:
        return best_rot[1], float(best_rot[0])
    return T_still, float(fit_still)


def camera_sets(recon):
    """Per-video camera arrays used for visibility reasoning."""
    from .geometry import camera_centers
    out = {}
    for vid in (0, 1):
        m = recon["video"] == vid
        out[vid] = {"E": recon["extrinsic"][m], "K": recon["intrinsic"][m],
                    "hw": recon["depth"].shape[1:3], "centers": camera_centers(recon["extrinsic"][m])}
    return out


def match_events(removed, added, cfg, voxel, cams=None):
    """Removed + added objects -> list of events (moved / rotated / removed / added).

    `cams` (from camera_sets) enables visibility-aware pose scoring."""
    events, used_r, used_a = [], set(), set()
    if removed and added:
        if cams is not None:
            for vid, group in ((0, removed), (1, added)):
                for o in group:
                    o["normals"] = _normals(o["xyz"], cams[vid], voxel)
        S = np.array([[pair_score(r, a) for a in added] for r in removed])
        rows, cols = linear_sum_assignment(-S)
        for i, j in sorted(zip(rows, cols), key=lambda p: -S[p]):
            if S[i, j] < cfg["match_threshold"]:
                continue
            r, a = removed[i], added[j]
            T, fit = register(r, a, voxel, cams, cfg.get("rotation_margin", 0.2))
            strong = S[i, j] >= cfg["match_threshold"] + 0.2
            if fit < cfg["min_icp_fitness"] and not strong:
                continue
            if fit >= cfg["min_icp_fitness"]:
                new_center = T[:3, :3] @ r["center"] + T[:3, 3]
                yaw = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
            else:  # appearance match but shapes too partial for ICP: use box centres, no rotation
                new_center, yaw = a["center"], None
            disp = new_center - r["center"]
            moved = np.linalg.norm(disp) >= cfg["min_move"]
            turned = yaw is not None and abs(yaw) >= cfg["min_rotation_deg"]
            used_r.add(i)
            used_a.add(j)
            if not moved and not turned:
                continue  # same object in the same place: not a change
            events.append({
                "type": "moved" if moved else "rotated",
                "before": r, "after": a,
                "displacement": disp, "rotation_deg": yaw,
                "match_score": float(S[i, j]), "icp_fitness": fit,
            })
    events += [{"type": "removed", "before": r} for i, r in enumerate(removed) if i not in used_r]
    events += [{"type": "added", "after": a} for j, a in enumerate(added) if j not in used_a]
    return events


def objects_from_hull(clusters, vid, recon, voxel):
    """Object records (same fields as cluster_changes) for appearance-channel voxel clusters."""
    frames = np.nonzero(recon["video"] == vid)[0]
    out = []
    for X in clusters:
        rgb, views = _sample_colors(X, frames, recon)
        center, half, yaw = oriented_box(X)
        out.append({
            "voxels": np.arange(len(X)), "xyz": X, "rgb": rgb,
            "center": center, "half_size": np.maximum(half, voxel / 2), "yaw_deg": yaw,
            "bottom": float(np.percentile(X[:, 2], 2)), "top": float(np.percentile(X[:, 2], 98)),
            "n_core_voxels": int(len(X)), "n_frames": int(views), "free_ratio": 1.0,
            "source": "appearance",
        })
    return out


def _sample_colors(X, frames, recon):
    """Mean image colour of the voxels over the frames that see them (no depth test: the
    appearance channel exists for objects the depth map does not contain)."""
    cols, views = np.zeros((len(X), 3)), 0
    count = np.zeros(len(X))
    H, W = recon["depth"].shape[1:]
    for f in frames:
        e, k = recon["extrinsic"][f], recon["intrinsic"][f]
        Xc = X @ e[:, :3].T + e[:, 3]
        uv = (Xc[:, :2] / np.maximum(Xc[:, 2:], 1e-6)) @ k[:2, :2].T + k[:2, 2]
        ok = (Xc[:, 2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if ok.mean() > 0.5:
            views += 1
        cols[ok] += recon["images"][f][uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
        count[ok] += 1
    return (cols / np.maximum(count, 1)[:, None]).astype(np.uint8), views


def same_object(a, b, dist):
    """True if two detections describe one object: their points overlap, or their footprints
    coincide in plan view at overlapping heights (a stretched appearance hull vs a depth cluster)."""
    d, _ = cKDTree(b["xyz"]).query(a["xyz"])
    if (d <= dist).mean() > 0.2:
        return True
    xy = np.linalg.norm(a["center"][:2] - b["center"][:2])
    reach = max(a["half_size"][:2].min(), b["half_size"][:2].min()) + dist
    z_overlap = min(a["top"], b["top"]) - max(a["bottom"], b["bottom"]) > 0
    return bool(xy <= reach and z_overlap)
