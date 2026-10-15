"""What Changed in the Room? — one command, two videos in, a 3D change report out.

    python run.py --group cool_white                # every test in a lighting group
    python run.py --scene test4                     # one test
    python run.py --before A.mp4 --after B.mp4      # any two videos

Each step caches its result in the output folder; use --force to recompute.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import yaml

from wcitr import align, changes, frames, objects, reconstruct, report, segment, semantics, verify, video
from wcitr.geometry import apply_world_transform

DATA = Path("data")


def step(name):
    print(f"\n== {name} ==", flush=True)
    return time.time()


def load_groups(path=DATA / "scenes.yaml"):
    """group -> {test name: (reference video, test video, info)} from data/scenes.yaml."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups = {}
    for g, grp in spec["groups"].items():
        groups[g] = {name: (DATA / grp["reference"], DATA / t["video"], {**t, "group": g})
                     for name, t in grp["tests"].items()}
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", help="lighting group from data/scenes.yaml (e.g. cool_white), or 'all'")
    ap.add_argument("--scene", help="one test from data/scenes.yaml (e.g. test4)")
    ap.add_argument("--before", help="video recorded first")
    ap.add_argument("--after", help="video recorded after the changes")
    ap.add_argument("--out", help="output folder (default: results/...)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-claude", action="store_true", help="label with CLIP only, no API calls")
    ap.add_argument("--no-appearance", action="store_true", help="skip the appearance channel (depth only)")
    ap.add_argument("--force", action="store_true", help="recompute every step")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.no_claude:
        cfg["semantics"]["use_claude"] = False
    if args.no_appearance:
        cfg["appearance"]["enabled"] = False
    root = Path(args.out or "results")

    if args.group or args.scene:
        groups = load_groups()
        if args.group:
            names = list(groups) if args.group == "all" else [args.group]
            bad = [n for n in names if n not in groups]
            if bad:
                ap.error(f"unknown group {bad[0]!r}; choose from: {', '.join(groups)}")
            jobs = [(g, t) for g in names for t in groups[g]]
        else:
            jobs = [(g, args.scene) for g in groups if args.scene in groups[g]]
            if not jobs:
                ap.error(f"unknown scene {args.scene!r}; choose from: "
                         + ", ".join(t for g in groups.values() for t in g))
        for g, name in jobs:
            before, after, info = groups[g][name]
            print(f"\n######## {g}/{name}: {before.name} -> {after.name} ########", flush=True)
            run_pipeline(before, after, root / g / name, cfg, args.force, info)
    elif args.before and args.after:
        run_pipeline(Path(args.before), Path(args.after), root / Path(args.after).stem, cfg, args.force)
    else:
        ap.error("give --group, --scene, or both --before and --after")


def run_pipeline(before, after, out, cfg, force=False, scene_info=None):
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    # 1. keyframes (read into memory; nothing is written to disk)
    t0 = step("1/5 keyframes")
    kf = {}
    for name, path in (("before", before), ("after", after)):
        kf[name] = frames.extract_keyframes(path, cfg["frames"]["per_video"],
                                            cfg["frames"]["candidates_per_keyframe"], cfg["frames"]["max_side"])
        print(f"{name}: {len(kf[name][0])} keyframes")
    keyframes = list(kf["before"][1]) + list(kf["after"][1])
    frame_idx = np.concatenate([kf["before"][0], kf["after"][0]])
    timings["keyframes_s"] = time.time() - t0

    # 2. shared 3D reconstruction
    t0 = step("2/5 joint 3D reconstruction (VGGT) + metric scale + room frame")
    recon_path = out / "recon.npz"
    recon = None
    if recon_path.exists() and not force:
        z = np.load(recon_path, allow_pickle=True)
        recon = {k: z[k] for k in z.files}
        recon["scale_info"] = recon["scale_info"].item()
        if "frame_idx" not in recon or not np.array_equal(recon["frame_idx"], frame_idx):
            recon, force = None, True  # keyframes changed (new video or settings): recompute everything
    if recon is None:
        recon = reconstruct.reconstruct(kf, cfg)
        np.savez_compressed(recon_path, **{k: (np.array(v, dtype=object) if k == "scale_info" else v)
                                           for k, v in recon.items()})
    print(f"scale: {recon['scale_info']}")
    timings["reconstruction_s"] = time.time() - t0

    # 3. evidence-based change detection (+ alignment refinement on unchanged geometry)
    t0 = step("3/5 change detection")
    ch_path = out / "changes.pkl"
    settings = {"change": cfg["change"], "align": cfg["align"]}
    cached = pickle.loads(ch_path.read_bytes()) if ch_path.exists() and not force else None
    if cached is not None and len(cached) == 4 and cached[3] == settings:
        results, align_info, recon["extrinsic"], _ = cached
    else:
        args_cd = lambda E, c: (recon["depth"], recon["conf"], recon["valid"], E, recon["intrinsic"],
                                recon["video"], recon["images"], c)
        # Pass 1: conservative tolerance, to find the geometry that certainly did not change.
        results = changes.detect_changes(*args_cd(recon["extrinsic"], cfg["change"]))
        align_info = {"applied": False, "reason": "disabled"}
        rmse = None
        if cfg["align"]["enabled"]:
            T, align_info = align.refine_alignment(results, cfg["align"], cfg["change"]["voxel"])
            print(f"alignment refinement: {align_info}")
            if T is not None:
                E = recon["extrinsic"].copy()
                b = recon["video"] == 1
                E[b] = apply_world_transform(E[b], T)
                recon["extrinsic"] = E.astype(np.float32)
            rmse = align_info.get("rmse_after" if align_info.get("applied") else "rmse_before")
        # Pass 2: tolerance matched to how well THIS pair of videos lines up.
        c = cfg["change"]
        sigma = c["registration_sigma"] if rmse is None else float(
            np.clip(c["sigma_from_alignment"] * rmse, c["sigma_min"], c["sigma_max"]))
        align_info["registration_sigma_m"] = round(sigma, 4)
        print(f"registration tolerance: {sigma * 100:.1f} cm")
        results = changes.detect_changes(*args_cd(recon["extrinsic"], {**c, "registration_sigma": sigma}))
        ch_path.write_bytes(pickle.dumps((results, align_info, recon["extrinsic"], settings)))
    print(json.dumps(changes.summarize(results)))
    timings["change_detection_s"] = time.time() - t0

    # 4. objects, labels, moved matching
    t0 = step("4/5 objects + labels")
    voxel, acfg = cfg["change"]["voxel"], cfg["appearance"]
    removed = objects.cluster_changes(results[0], cfg["objects"], voxel)
    added = objects.cluster_changes(results[1], cfg["objects"], voxel)
    print(f"geometry: {len(removed)} removed, {len(added)} added")
    labeler = semantics.Labeler(cfg["semantics"])
    comparer = verify.CropComparer(acfg["model"])
    if acfg["enabled"]:
        app_path = out / "appearance.pkl"
        key = (acfg, cfg["change"], recon["extrinsic"].tobytes())
        cached = pickle.loads(app_path.read_bytes()) if app_path.exists() and not force else None
        if cached is not None and cached[0] == key:
            found, app_info = cached[1], cached[2]
        else:
            found, app_info = verify.appearance_objects(recon, acfg, voxel, comparer, labeler)
            app_path.write_bytes(pickle.dumps((key, found, app_info)))
        print(f"appearance: {app_info}")
        for vid, group in ((0, removed), (1, added)):
            for (c, label), o in zip(found[vid], objects.objects_from_hull([c for c, _ in found[vid]], vid, recon, voxel)):
                o.update(label=label, label_fixed=True, label_source="clip-differential")
                if not any(objects.same_object(o, g, acfg["merge_distance"]) for g in group):
                    group.append(o)
    # Appearance-only candidates at the place of a depth-found object add nothing: depth already
    # gave the direction there (the appearance channel cannot tell added from removed as well).
    # (Lists are filtered by rebuilding them: list.remove compares records with ==, which fails on
    # the NumPy arrays inside them.)
    geo = [g for g in removed + added if g.get("source") != "appearance"]
    removed, added = ([o for o in group if o.get("source") != "appearance"
                       or not any(objects.same_object(o, g, acfg["merge_distance"]) for g in geo)]
                      for group in (removed, added))
    # Every detection must also LOOK different between the videos at its place, clearly more than
    # nearby control places (drops static objects whose depth disagreed between the videos).
    def looks_changed(o, comparer=comparer):
        sim, control, _ = verify.similarity(o["xyz"], recon, comparer)
        o["appearance_similarity"], o["appearance_control"] = sim, control
        return not np.isfinite(sim) or verify.is_changed(sim, control, acfg)

    removed = [o for o in removed if looks_changed(o)]
    added = [o for o in added if looks_changed(o)]
    del comparer
    print(f"verified objects: {len(removed)} removed, {len(added)} added")
    semantics.label_objects(removed, "removed", recon, keyframes, labeler)
    semantics.label_objects(added, "added", recon, keyframes, labeler)
    events = objects.match_events(removed, added, cfg["objects"], cfg["change"]["voxel"],
                                  objects.camera_sets(recon))
    # Exact extents: carve each changed object's 3D box from SAM silhouettes in its own video.
    segmenter = segment.Segmenter(cfg["segment"]["model"]) if cfg["segment"]["enabled"] else None
    if segmenter is not None:
        for e in events:
            for key, vid in (("before", 0), ("after", 1)):
                if key in e:
                    segment.refine_extent(e[key], vid, recon, segmenter, voxel)
            if e["type"] in ("moved", "rotated"):
                # Distance and direction from the refined boxes, so the text, arrow and boxes agree.
                e["displacement"] = e["after"]["center"] - e["before"]["center"]
    timings["objects_s"] = time.time() - t0

    # 5. report
    t0 = step("5/5 report")
    meta = {"scale": recon["scale_info"], "alignment": align_info,
            "inputs": {"before": str(before), "after": str(after)},
            "scene": scene_info}
    final = report.write_all(out, results, events, recon, meta)
    timings["report_s"] = time.time() - t0
    meta["timings"] = {k: round(v, 1) for k, v in timings.items()}
    data = json.loads((out / "changes.json").read_text(encoding="utf-8"))
    data["timings"] = meta["timings"]
    (out / "changes.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    step("comparison video")
    clip = video.make_comparison_video(out, recon, data, before, after, segmenter=segmenter, results=results)
    print(f"video: {clip}")

    print()
    print((out / "report.md").read_text(encoding="utf-8").split("![")[0].strip())
    print(f"\nResults in {out.resolve()}\n  video: {clip.name}   3D view: rerun {out / 'scene.rrd'}")
    return final


if __name__ == "__main__":
    main()
