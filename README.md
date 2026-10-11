# What changed in the room?

Two phone videos of the same desk, filmed at different times and from different paths. This system builds one shared 3D scene from both, finds the objects that were added, removed or moved, outlines them in a side-by-side video and describes each change in a sentence.

The main design rule: a change is reported only when there is evidence for it. If the first video saw empty space where the second video now sees an object, the object was added. If a spot was hidden or never filmed in one of the videos, the system reports it as unknown instead of guessing. This is what separates real changes from differences in viewpoint, occlusion and incomplete coverage.

![test7: a drinking glass was removed](docs/assets/test7.gif)

*test7. Left: reference video, the glass next to the black mug is outlined. Right: test video, the dashed box marks where it used to be. The depth model sees straight through the glass, so this change was found by the appearance channel described below.*

![test4: a clear jug was added](docs/assets/test4.gif)

*test4. The new jug is outlined in every frame of the test video; the dashed box on the reference side marks the spot where it will appear.*

## Results

Eight test videos were compared with a reference video filmed under the same light. The ground truth comes from comparing matched views of each pair by eye (recorded in [data/scenes.yaml](data/scenes.yaml)); I am confident about test4 and test7 and less certain that the other six contain no change at all.

| Test | Light, viewpoint | What changed | System output |
|---|---|---|---|
| test4 | cool white | clear plastic jug added | "Jug was added on the table surface" |
| test7 | cool white | drinking glass removed | "Drinking glass was removed. It was on the table surface" |
| test6 | cool white | nothing (control) | no changes |
| test3 | cool white, high angle | nothing | no changes |
| test5 | cool white, high angle | nothing | no changes |
| test8 | cool white, high angle | nothing | no changes |
| test1 | warm, very dark, high angle | nothing visible | no changes |
| test9 | warm, high angle | nothing visible | no changes |

All eight outputs match the ground truth. Some numbers from the runs:

- The carved 3D boxes are 16 × 11 × 32 cm for the jug and 10 × 9 × 19 cm for the glass.
- Before the alignment step, the two videos were 2.6 to 6.0 mm apart for pairs filmed from similar heights, and 38 to 61 mm apart for the high-angle pairs. The change test widens its tolerance to match: 1.1 to 1.5 cm for the first group, 3.4 to 4.0 cm for the second.
- Surfaces seen in only one of the two videos ranged from 4% to 59% depending on the pair. Those areas are reported as unchecked, not as changes.

These results need two caveats. The thresholds were tuned on these same eight pairs, so they show that the method can separate these cases, not how it does on new recordings. None of the recordings contains a moved object; move detection is tested only on a synthetic scene (a chair moved 2 m, measured within 8 cm, with no false rotation).

## How it works

```
before.mp4 + after.mp4
 1. keyframes        40 sharp frames per video, kept in memory
 2. shared 3D        both videos through VGGT in one pass, metric scale, floor-aligned room frame
 3. evidence check   per voxel: changed / unchanged / unknown, alignment refined on unchanged geometry
 4. objects          depth changes + appearance changes, verified, matched into moves, outlined with SAM, labelled
 5. output           comparison video, changes.json, report.md, top-down map, 3D Rerun recording
```

1. Keyframes ([wcitr/frames.py](wcitr/frames.py)). Each video is cut into 40 time slots and the sharpest frame (variance of the Laplacian) is kept from each.
2. Shared 3D scene ([wcitr/reconstruct.py](wcitr/reconstruct.py)). All 80 keyframes go through [VGGT](https://github.com/facebookresearch/vggt) in one forward pass, so cameras and depth maps of both videos come out in one coordinate frame without a separate registration step. Depth Anything V2 (indoor, metric) converts the scale to metres. The floor or desk plane becomes z = 0.
3. Evidence check ([wcitr/changes.py](wcitr/changes.py)). Every confident depth pixel of one video becomes a 3D point and is checked against the other video's depth maps. If the other camera saw farther than the point, it looked through empty space there. If it saw the same surface, the point is unchanged. If something was in front, or the point was out of view, there is no evidence. Votes are fused in 2 cm voxels. Points that their own video cannot confirm (noise, a person walking through) are dropped first. The tolerance for this test comes from how well the two videos line up on unchanged geometry, measured per pair.
4. Objects and labels ([wcitr/objects.py](wcitr/objects.py), [wcitr/appearance.py](wcitr/appearance.py), [wcitr/verify.py](wcitr/verify.py), [wcitr/segment.py](wcitr/segment.py), [wcitr/semantics.py](wcitr/semantics.py)).
   - Changed voxels are clustered into objects. A removed object and an added object that look alike become one moved event, with distance, direction and rotation from ICP.
   - A second channel finds changes depth cannot see, such as a clear glass. It compares DINOv2 features of each frame with warped views of the other video, then locates the changed pixels in 3D with a visual hull.
   - Every candidate, from either channel, must look clearly less alike in matched before and after views than nearby control places do. This removed every false alarm from the high-angle videos.
   - Segment Anything outlines each object. The outlines from many keyframes carve its 3D box.
   - CLIP names the object (Claude can do this instead if an API key is set; that path has not been tested).
5. Output ([wcitr/report.py](wcitr/report.py), [wcitr/video.py](wcitr/video.py)). Every sentence, box and arrow is generated from `changes.json`, so the text, the video and the 3D view always agree.

The reasoning behind these choices, including two approaches I tried and dropped, is in [docs/DESIGN.md](docs/DESIGN.md).

## Running it

Tested on Windows 11 with Python 3.11 and an RTX 4090 Laptop GPU (16 GB). VGGT peaks at about 12 GB with 80 frames; lower `frames.per_video` in [config.yaml](config.yaml) for a smaller GPU.

```bash
python -m venv .venv
.venv\Scripts\activate                    # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The first run downloads the models from Hugging Face: VGGT-1B (about 5 GB), Depth Anything V2, DINOv2, CLIP and SAM. After that, `HF_HUB_OFFLINE=1` skips the network checks.

```bash
python run.py --scene test7                      # one test from data/scenes.yaml
python run.py --group cool_white                 # every test in a lighting group
python run.py --before A.mp4 --after B.mp4       # any two videos
```

Options: `--no-claude` labels with CLIP only, `--no-appearance` uses the depth channel alone, `--force` recomputes every cached step.

On the GPU above, a new pair takes about 70 s for the 3D reconstruction, 2 to 3 s for the evidence check and about 30 s for objects and labels. Rendering the comparison video with a SAM outline in every frame adds several minutes; I did not time it separately. Cached steps are reused on later runs.

## Outputs

Each pair writes to `results/<group>/<test>/`:

| File | Content |
|---|---|
| `comparison.mp4` | reference and test side by side, changes outlined, change list underneath, top-down map at the end |
| `changes.json` | every event: type, label, position and size in metres, location sentence, evidence counts, verification scores |
| `report.md` | the change list in plain language, plus how much of each video could not be checked |
| `topdown.png` | map of both camera paths, unchanged, unchecked and changed geometry |
| `scene.rrd` | 3D recording: open with `rerun results/<group>/<test>/scene.rrd` |
| `recon.npz`, `changes.pkl`, `appearance.pkl` | caches |

Positions use a room frame with the origin below where the first video starts: +y forward, +x right, +z up, in metres. Heights in the text are measured from the surface under each object.

![top-down map for test7](docs/assets/test7_topdown.png)

## Data

Ten phone videos of one desk, 1024 × 576 pixels, 3 to 5 seconds each: two references (cool white light, warm lamp light) and eight tests. Each test is paired with the reference filmed under the same light, chosen by average red/blue colour ratio. See [data/README.md](data/README.md) and [data/scenes.yaml](data/scenes.yaml).

## Limitations

- The thresholds were set on the same eight pairs the results come from.
- No real recording contains a moved object.
- A short move of a transparent object, where the old and new places overlap, may come out as a single added or removed event.
- Transparent objects are assumed to be about as deep as they are wide, and to rest on a surface. Both videos see them from similar directions, so their depth along the viewing direction cannot be measured.
- Identical objects swapped between two places cannot be told apart from no change.
- The videos are 3 to 5 seconds long, so each video moves only a short distance and some surfaces are seen by only one of them.
- The stack is heavy (VGGT, Depth Anything V2, DINOv2, CLIP, SAM) and needs a CUDA GPU with about 12 GB.

## Repository structure

```
run.py                  one command: two videos in, results out
config.yaml             every threshold, with comments
wcitr/
  frames.py             keyframe selection
  reconstruct.py        VGGT, metric scale, room frame
  changes.py            evidence check (changed / unchanged / unknown)
  align.py              alignment refinement on unchanged geometry
  objects.py            clustering, move matching, ICP
  appearance.py         appearance channel: DINOv2 change maps, visual hull
  verify.py             before/after verification against control places, removed vs added
  segment.py            SAM outlines and 3D carving
  semantics.py          CLIP / Claude labels
  report.py             changes.json, report.md, top-down map, Rerun recording
  video.py              side-by-side comparison video
  geometry.py           camera maths
tests/test_changes.py   synthetic scenes with known changes
data/                   videos and scene list
docs/                   DESIGN.md (design note), CAPTURE_GUIDE.md, RESEARCH_PLAN.md (initial plan), assets
```

The synthetic tests ray-cast box scenes with a moved chair, a removed box, an added board, an object hidden behind the board and an object in a corner the first video never saw. They check that each is classified correctly: `python -m pytest tests`.
#   W h a t - C h a n g e d - i n - t h e - R o o m  
 