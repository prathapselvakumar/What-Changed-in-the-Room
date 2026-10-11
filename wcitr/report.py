"""Step 5: changes.json, a plain-language report, a top-down map and a 3D Rerun recording.

Every sentence is built from the measured numbers in changes.json, so the text,
the map and the 3D view always say the same thing.

Room frame: origin on the floor below where the before video starts,
+y = that camera's viewing direction, +x = its right, +z = up.
"""

import json

import matplotlib
import numpy as np

from .changes import CHANGED, UNCHANGED, UNKNOWN
from .geometry import camera_centers, rotation_about_z

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------- language ----------

# The room frame's z = 0 plane is the lowest large horizontal surface. In close-up videos of a
# desk that is the desk top, not the floor; camera height above it tells the two apart.
SUPPORT = {"name": "floor"}


def local_support(obj, results, margin=0.05):
    """Height of the surface right under an object, from unchanged geometry of both videos.
    Objects are described relative to what they stand on, not to one global plane (a desk is
    rarely perfectly level with the fitted plane, and a shelf is not the floor)."""
    from .changes import UNCHANGED
    reach = np.linalg.norm(obj["half_size"][:2]) + margin
    zs = []
    for r in results.values():
        m = (r["state"] == UNCHANGED) & (np.linalg.norm(r["xyz"][:, :2] - obj["center"][:2], axis=1) <= reach)
        z = r["xyz"][m, 2]
        zs.append(z[(z <= obj["bottom"] + 0.03) & (z >= obj["bottom"] - 0.4)])
    z = np.concatenate(zs)
    return float(np.percentile(z, 90)) if len(z) >= 10 else 0.0


def set_support_surface(E):
    cam_height = float(np.median(camera_centers(E)[:, 2]))
    SUPPORT["name"] = "floor" if cam_height >= 1.0 else "table surface"
    SUPPORT["camera_height_m"] = round(cam_height, 2)


def where(obj):
    x, y, _ = obj["center"]
    dist = float(np.hypot(x, y))
    a = np.degrees(np.arctan2(x, y))  # 0 = straight ahead of the starting view, + = right
    side = "right" if a > 0 else "left"
    a = abs(a)
    if a < 25:
        direction = "straight ahead"
    elif a < 65:
        direction = f"ahead to the {side}"
    elif a < 115:
        direction = f"to the {side}"
    elif a < 155:
        direction = f"behind, to the {side}"
    else:
        direction = "behind"
    lift = obj["bottom"] - obj.get("support_z", 0.0)
    if lift < 0.06:
        height = f"on the {SUPPORT['name']}"
    else:
        height = f"about {lift:.2f} m above the {SUPPORT['name']}"
    return f"{height}, {dist:.1f} m {direction} of where the first video starts"


def direction_phrase(d):
    parts = []
    horiz = np.hypot(d[0], d[1])
    for value, pos, neg in ((d[0], "to the right", "to the left"), (d[1], "forward", "backward")):
        if abs(value) >= 0.3 * max(horiz, 1e-6) and abs(value) >= 0.03:
            parts.append(f"{abs(value):.2f} m {pos if value > 0 else neg}")
    if abs(d[2]) >= 0.1:
        parts.append(f"{abs(d[2]):.2f} m {'up' if d[2] > 0 else 'down'}")
    return " and ".join(parts) if parts else "slightly"


def obj_confidence(o):
    geo = o["free_ratio"] * min(1.0, o["n_frames"] / 6.0)
    return float(np.clip(geo, 0, 1))


def obj_json(o):
    return {
        "label": o.get("label", "object"),
        "description": o.get("description"),
        "label_source": o.get("label_source"),
        "found_by": o.get("source", "geometry"),
        "appearance_similarity": None if o.get("appearance_similarity") is None else round(float(o["appearance_similarity"]), 3),
        "control_similarity": None if o.get("appearance_control") is None else round(float(o["appearance_control"]), 3),
        "center_m": np.round(o["center"], 3).tolist(),
        "size_m": np.round(2 * o["half_size"], 3).tolist(),
        "yaw_deg": round(o["yaw_deg"], 1),
        "bottom_height_m": round(o["bottom"], 3),
        "height_above_support_m": round(o["bottom"] - o.get("support_z", 0.0), 3),
        "location": where(o),
        "evidence": {"voxels": o["n_core_voxels"], "views": o["n_frames"], "free_space_ratio": round(o["free_ratio"], 3)},
        "verified_by_vlm": o.get("verified"),
    }


def build_events(events):
    out = []
    for e in events:
        objs = [o for o in (e.get("before"), e.get("after")) if o is not None]
        label = next((o["label"] for o in objs if o.get("label")), "object")
        conf = float(np.mean([obj_confidence(o) for o in objs]))
        verified = [o.get("verified") for o in objs if o.get("verified") is not None]
        if e["type"] == "moved":
            text = f"{label.capitalize()} moved {np.linalg.norm(e['displacement']):.2f} m ({direction_phrase(e['displacement'])})"
            if e["rotation_deg"] is not None and abs(e["rotation_deg"]) >= 15:
                text += f" and turned about {abs(e['rotation_deg']):.0f} degrees"
            text += f". It is now {where(e['after'])}."
            conf *= min(1.0, e["match_score"] / 0.7)
        elif e["type"] == "rotated":
            text = f"{label.capitalize()} turned about {abs(e['rotation_deg']):.0f} degrees in place ({where(e['after'])})."
        elif e["type"] == "removed":
            text = f"{label.capitalize()} was removed. It was {where(e['before'])}."
        else:
            text = f"{label.capitalize()} was added {where(e['after'])}."
        item = {
            "type": e["type"],
            "label": label,
            "text": text,
            "confidence": round(conf, 2),
            "verified_by_vlm": (all(verified) if verified else None),
        }
        if "before" in e:
            item["before"] = obj_json(e["before"])
        if "after" in e:
            item["after"] = obj_json(e["after"])
        if e["type"] in ("moved", "rotated"):
            item["displacement_m"] = np.round(e["displacement"], 3).tolist()
            item["distance_m"] = round(float(np.linalg.norm(e["displacement"])), 3)
            item["rotation_deg"] = None if e["rotation_deg"] is None else round(e["rotation_deg"], 1)
            item["match_score"] = round(e["match_score"], 3)
            item["icp_fitness"] = round(e["icp_fitness"], 3)
        out.append(item)
    out.sort(key=lambda x: (x["verified_by_vlm"] is False, -x["confidence"]))
    for i, item in enumerate(out, 1):
        item["id"] = i
    return out


def coverage(results):
    cov = {}
    for vid, name in ((0, "before"), (1, "after")):
        s = results[vid]["state"]
        cov[name] = {
            "unchanged_pct": round(100 * float((s == UNCHANGED).mean()), 1),
            "changed_pct": round(100 * float((s == CHANGED).mean()), 1),
            "unknown_pct": round(100 * float((s == UNKNOWN).mean()), 1),
        }
    return cov


def write_markdown(path, events, cov, meta):
    lines = ["# What changed in the room", ""]
    confirmed = [e for e in events if e["verified_by_vlm"] is not False]
    doubtful = [e for e in events if e["verified_by_vlm"] is False]
    if not events:
        lines.append("No changes found in the areas both videos could see.")
    for e in confirmed:
        lines.append(f"{e['id']}. **{e['type'].capitalize()}:** {e['text']} (confidence {e['confidence']:.2f})")
    if doubtful:
        lines += ["", "Possible changes the image check could not confirm:"]
        lines += [f"{e['id']}. {e['text']} (confidence {e['confidence']:.2f})" for e in doubtful]
    lines += [
        "",
        f"**Not checked:** {cov['after']['unknown_pct']:.0f}% of the surfaces seen in the after video and "
        f"{cov['before']['unknown_pct']:.0f}% of those in the before video were not visible (or not clear enough) "
        "in the other video. Changes there are not reported.",
        "",
        "![Top-down map](topdown.png)",
        "",
    ]
    lines.append("Video: [comparison.mp4](comparison.mp4) — reference and test side by side with every change marked.")
    lines += ["", f"Scale: {meta['scale']['method']} (factor {meta['scale']['factor']:.3f}). "
              f"Alignment refinement: {meta.get('alignment', {}).get('applied', False)}."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- top-down map ----------

COLORS = {"removed": "#d62728", "added": "#2ca02c", "moved": "#1f77b4", "rotated": "#9467bd"}


def _footprint(o):
    c, h, yaw = np.array(o["center_m"]), np.array(o["size_m"]) / 2, o["yaw_deg"]
    R = rotation_about_z(yaw)[:2, :2]
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * h[:2]
    return corners @ R.T + c[:2]


def topdown(path, results, events, E, video):
    fig, ax = plt.subplots(figsize=(9, 9))
    rng = np.random.default_rng(0)
    for vid in (0, 1):
        r = results[vid]
        for state, color, size in ((UNCHANGED, "#9e9e9e", 1.5), (UNKNOWN, "#f2d58a", 1.5)):
            p = r["xyz"][r["state"] == state]
            if len(p) > 60000:
                p = p[rng.choice(len(p), 60000, replace=False)]
            ax.scatter(p[:, 0], p[:, 1], s=size, c=color, linewidths=0, zorder=1)
    for vid, color in ((0, "#d62728"), (1, "#2ca02c")):
        p = results[vid]["xyz"][results[vid]["state"] == CHANGED]
        ax.scatter(p[:, 0], p[:, 1], s=3.0, c=color, linewidths=0, zorder=2)
    C = camera_centers(E)
    ax.plot(*C[video == 0, :2].T, "-o", ms=2, color="#1565c0", label="before camera path")
    ax.plot(*C[video == 1, :2].T, "-o", ms=2, color="#ef6c00", label="after camera path")
    for e in events:
        col = COLORS[e["type"]]
        for key, ls in (("before", "--"), ("after", "-")):
            if key in e:
                fp = _footprint(e[key])
                ax.plot(fp[:, 0], fp[:, 1], ls, color=col, lw=2, zorder=3)
        anchor = e.get("after", e.get("before"))["center_m"]
        if e["type"] == "moved":
            b = e["before"]["center_m"]
            ax.annotate("", xy=anchor[:2], xytext=b[:2], arrowprops=dict(arrowstyle="->", color=col, lw=2.5), zorder=4)
        ax.text(anchor[0], anchor[1], f" {e['id']}. {e['label']} ({e['type']})", color=col, fontsize=10,
                weight="bold", zorder=5)
    for color, text in (("#9e9e9e", "unchanged"), ("#f2d58a", "not visible in the other video"),
                        ("#d62728", "removed / old position"), ("#2ca02c", "added / new position")):
        ax.plot([], [], "s", color=color, ms=6, label=text)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m) — right of the starting view")
    ax.set_ylabel("y (m) — forward from the starting view")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("What changed (top-down view)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- 3D (Rerun) ----------

def rerun_recording(path, results, events, recon):
    import rerun as rr

    rr.init("what-changed-in-the-room", spawn=False)
    rr.save(str(path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    for vid, name in ((0, "before"), (1, "after")):
        r = results[vid]
        for state, sname in ((UNCHANGED, "unchanged"), (UNKNOWN, "unknown")):
            m = r["state"] == state
            col = r["rgb"][m] if state == UNCHANGED else np.tile([[242, 213, 138]], (int(m.sum()), 1))
            rr.log(f"world/{name}/{sname}", rr.Points3D(r["xyz"][m], colors=col, radii=0.006), static=True)
        m = r["state"] == CHANGED
        col = [214, 39, 40] if vid == 0 else [44, 160, 44]
        rr.log(f"world/{name}/changed", rr.Points3D(r["xyz"][m], colors=col, radii=0.009), static=True)

    for e in events:
        col = [int(COLORS[e["type"]][i:i + 2], 16) for i in (1, 3, 5)]
        for key in ("before", "after"):
            if key in e:
                o = e[key]
                q = _quat_z(o["yaw_deg"])
                rr.log(f"world/events/{e['id']}/{key}", rr.Boxes3D(
                    centers=[o["center_m"]], half_sizes=[np.array(o["size_m"]) / 2],
                    rotations=[q], colors=[col], labels=[f"{e['id']}. {e['label']} ({e['type']}, {key})"]),
                    static=True)
        if e["type"] == "moved":
            b = np.array(e["before"]["center_m"])
            rr.log(f"world/events/{e['id']}/arrow", rr.Arrows3D(
                origins=[b], vectors=[np.array(e["after"]["center_m"]) - b], colors=[col]), static=True)

    E, K, images, video = recon["extrinsic"], recon["intrinsic"], recon["images"], recon["video"]
    H, W = images.shape[1:3]
    for i in range(len(E)):
        name = "before" if video[i] == 0 else "after"
        R, t = E[i, :3, :3], E[i, :3, 3]
        rr.log(f"world/cameras/{name}/{i:03d}", rr.Transform3D(translation=-R.T @ t, mat3x3=R.T), static=True)
        rr.log(f"world/cameras/{name}/{i:03d}/image", rr.Pinhole(image_from_camera=K[i], resolution=[W, H],
                                                                image_plane_distance=0.15), static=True)
        if i % 4 == 0:
            rr.log(f"world/cameras/{name}/{i:03d}/image", rr.Image(images[i]), static=True)


def _quat_z(yaw_deg):
    a = np.radians(yaw_deg) / 2
    return rr_quat([0.0, 0.0, np.sin(a), np.cos(a)])


def rr_quat(xyzw):
    import rerun as rr
    return rr.Quaternion(xyzw=xyzw)


def write_all(out_dir, results, events_raw, recon, meta):
    set_support_surface(recon["extrinsic"])
    meta = {**meta, "support_surface": dict(SUPPORT)}
    for e in events_raw:
        for key in ("before", "after"):
            if key in e:
                e[key]["support_z"] = local_support(e[key], results)
    events = build_events(events_raw)
    cov = coverage(results)
    topdown(out_dir / "topdown.png", results, events, recon["extrinsic"], recon["video"])
    report = {"events": events, "coverage": cov, **meta}
    (out_dir / "changes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(out_dir / "report.md", events, cov, meta)
    rerun_recording(out_dir / "scene.rrd", results, events, recon)
    return events
