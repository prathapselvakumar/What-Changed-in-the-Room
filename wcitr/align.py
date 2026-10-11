"""Refine before/after alignment using only geometry that did NOT change.

VGGT puts both videos in one frame already; this removes any small residual
offset between the two halves. Changed and unknown voxels are excluded, so a
moved object cannot pull the alignment towards itself.
"""

import numpy as np
import open3d as o3d

from .changes import UNCHANGED


def _cloud(xyz, radius):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return pcd


def refine_alignment(results, cfg_align, voxel):
    """Rigid 4x4 T mapping after-video geometry onto before-video geometry, or None."""
    a = results[0]["xyz"][results[0]["state"] == UNCHANGED]
    b = results[1]["xyz"][results[1]["state"] == UNCHANGED]
    if len(a) < 500 or len(b) < 500:
        return None, {"applied": False, "reason": "too little unchanged geometry"}

    src, dst = _cloud(b, 3 * voxel), _cloud(a, 3 * voxel)
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
    before = o3d.pipelines.registration.evaluate_registration(src, dst, cfg_align["max_corr"], np.eye(4))
    reg = o3d.pipelines.registration.registration_icp(
        src, dst, cfg_align["max_corr"], np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=cfg_align["max_corr"] / 2)),
        crit,
    )
    T = reg.transformation
    shift = float(np.linalg.norm(T[:3, 3]))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    info = {
        "translation_m": shift, "rotation_deg": angle,
        "rmse_before": float(before.inlier_rmse), "rmse_after": float(reg.inlier_rmse),
        "fitness_before": float(before.fitness), "fitness_after": float(reg.fitness),
    }
    better = (reg.fitness > before.fitness + 1e-3 or
              (reg.fitness >= before.fitness - 1e-3 and reg.inlier_rmse < 0.98 * before.inlier_rmse))
    ok = shift <= cfg_align["max_translation"] and angle <= cfg_align["max_rotation_deg"] and better
    info["applied"] = bool(ok)
    return (T if ok else None), info
