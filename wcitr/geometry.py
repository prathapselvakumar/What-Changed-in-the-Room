"""Camera geometry shared by every stage.

Conventions (same as VGGT): OpenCV cameras (x right, y down, z forward),
extrinsics are 3x4 camera-from-world, pixel centres at integer coordinates.
"""

import numpy as np
import torch
import torch.nn.functional as F


def to_4x4(E):
    """(..., 3, 4) -> (..., 4, 4)."""
    out = np.zeros(E.shape[:-2] + (4, 4), dtype=E.dtype)
    out[..., :3, :] = E
    out[..., 3, 3] = 1.0
    return out


def camera_centers(E):
    """World positions of cameras, (S, 3)."""
    R, t = E[:, :3, :3], E[:, :3, 3]
    return -np.einsum("sji,sj->si", R, t)


def apply_world_transform(E, M):
    """Re-express cameras after world points are mapped by X' = M X (M is 4x4)."""
    return (to_4x4(E) @ np.linalg.inv(M))[:, :3, :]


def pixel_grid(H, W, device):
    v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    return u.float(), v.float()


def backproject(depth, K):
    """Depth (S, H, W) + intrinsics (S, 3, 3) -> camera-frame points (S, H, W, 3)."""
    S, H, W = depth.shape
    u, v = pixel_grid(H, W, depth.device)
    fx, fy = K[:, 0, 0].view(S, 1, 1), K[:, 1, 1].view(S, 1, 1)
    cx, cy = K[:, 0, 2].view(S, 1, 1), K[:, 1, 2].view(S, 1, 1)
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    return torch.stack([x, y, depth], dim=-1)


def cam_to_world(Xc, E):
    """Camera points (S, ..., 3) -> world, with E (S, 3, 4)."""
    R, t = E[:, :3, :3], E[:, :3, 3]
    shape = Xc.shape
    Xc = Xc.reshape(shape[0], -1, 3)
    Xw = torch.einsum("sji,snj->sni", R, Xc - t[:, None, :])
    return Xw.reshape(shape)


def normals_from_points(P):
    """Surface normals from an organised point map (S, H, W, 3), central differences."""
    Pp = F.pad(P.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate").permute(0, 2, 3, 1)
    dx = Pp[:, 1:-1, 2:] - Pp[:, 1:-1, :-2]
    dy = Pp[:, 2:, 1:-1] - Pp[:, :-2, 1:-1]
    n = torch.cross(dx, dy, dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def min_max_pool(x, k=3):
    """Per-pixel min and max over a k x k window, same size output. x: (S, H, W)."""
    p = k // 2
    xp = F.pad(x[:, None], (p, p, p, p), mode="replicate")
    mx = F.max_pool2d(xp, k, 1)[:, 0]
    mn = -F.max_pool2d(-xp, k, 1)[:, 0]
    return mn, mx


def erode(mask, k=3):
    """Binary erosion: a pixel survives only if its whole k x k window is True."""
    p = k // 2
    m = F.pad(mask[:, None].float(), (p, p, p, p), value=0.0)
    return (-F.max_pool2d(-m, k, 1)[:, 0]) > 0.5


def project(Xw, E, K):
    """World points (P, 3) into many cameras -> pixel u, v and depth z, each (S, P)."""
    R, t = E[:, :3, :3], E[:, :3, 3]
    Xc = torch.einsum("sij,pj->spi", R, Xw) + t[:, None, :]
    z = Xc[..., 2]
    zs = z.clamp_min(1e-6)
    u = K[:, None, 0, 0] * Xc[..., 0] / zs + K[:, None, 0, 2]
    v = K[:, None, 1, 1] * Xc[..., 1] / zs + K[:, None, 1, 2]
    return u, v, z


def look_at(center, target, up=(0.0, 0.0, 1.0)):
    """OpenCV camera-from-world extrinsics for a camera at `center` looking at `target`."""
    c, f = np.asarray(center, float), np.asarray(target, float) - np.asarray(center, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return np.concatenate([R, (-R @ c)[:, None]], axis=1)


def rotation_about_z(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
