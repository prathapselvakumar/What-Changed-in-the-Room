# Design note

This note explains why the system works the way it does, what each choice costs, and what I tried and dropped. The README covers how to run it.

## The problem behind the problem

Finding differences between two reconstructions is easy; most of them are not changes. Two reconstructions of an unchanged room still differ because:

- the second video sees surfaces the first never saw (the back of a bottle, the desk behind a laptop);
- an object is hidden in one video, not gone;
- depth networks are unreliable on glass, dark screens and thin objects;
- the two reconstructions never line up perfectly.

Comparing point clouds by distance reports all of these as changes. The challenge text names exactly these confusions, so the system is built around them.

## 1. A change needs evidence

Each 3D point of one video is checked against the other video's depth maps from that camera's point of view:

| The other camera saw... | Meaning |
|---|---|
| farther than the point | it looked through empty space there, so the point is absent in the other video: changed |
| the same surface | unchanged |
| something nearer | the point was hidden: no evidence |
| nothing (out of view) | no evidence |

Points of the before video that the after video sees through are removed; points of the after video that the before video saw through are added. Points without enough votes are **unknown**. This idea comes from free-space carving in occupancy mapping, where a ray that passes through a cell is evidence that the cell is empty.

The cost is that unknown areas produce no report at all, even when something did change there. For a robot that is the right failure: "I have not checked the corner behind the door" is safe to act on, while a guessed addition is not.

Two details mattered more than expected. The edge filter first removed every pixel at a depth jump; that shaved a strip off each object's silhouette and shifted its measured position by about one voxel. It now removes only pixels whose depth lies between the near and far surfaces (the "flying pixels"). The filter also used depth differences, which flagged floors seen at a shallow angle as edges and hid exactly the empty space the method needs. Inverse depth is linear across any plane, so its second difference is used instead.

## 2. One shared scene by construction

Both videos go through VGGT in a single forward pass, so their cameras share one coordinate frame without a separate registration step that a moved object could pull off course. A refinement step then aligns the two halves using only voxels already classified as unchanged.

The measured misalignment differs per pair: a few millimetres when both videos are filmed from similar heights, 4 to 6 cm when one looks down from a high angle. A fixed tolerance either floods the high-angle pairs with false changes along every object outline or misses small objects in the good pairs. The tolerance is therefore 2.5 times the measured alignment error of that pair, clamped to 1 to 5 cm. The free-space test must hold over a window that covers this error at the point's depth.

Cost: VGGT handles about 80 frames in 16 GB, so each video contributes 40 keyframes. Longer videos are subsampled, not used in full.

## 3. Changes that depth cannot see

A clear glass was removed in test7 and the depth channel found nothing: VGGT sees straight through it, in both videos. A second channel compares appearance:

1. For every frame, the most similar frames of the other video are warped into its view using its depth, only where those cameras could see the spot.
2. DINOv2 patch features of the frame and each warped view are compared; the smallest difference over the warped views is kept, so a pixel counts as changed only if no view of the other video agrees with it.
3. Changed pixels from many frames are intersected in 3D (a visual hull) to locate the object, because a transparent object's pixels carry the depth of the wall behind it.

Locating the candidate turned out to be the easy part. Deciding what it means was hard, and two approaches failed:

- *Symmetric cancellation.* Mismatches that appear in both videos at the same place looked like artefacts (a static glass whose depth is wrong in both). But a removed transparent glass also produces mismatches in both videos at its location, because rays from the after video pass through the spot where it stood. This rule cancelled the real change.
- *Plane-sweep presence.* Each video was asked separately whether something stands at the candidate location, by testing which depth explains its own frames best. Near object edges the background hypothesis was often poor, and the test said "present" in both videos for the removed glass.

What worked is simpler. Every candidate, from either channel, is cropped from matched views (an after frame and the before frame with the closest camera pose), and the two crops are compared with DINOv2. On the recorded data, real changes scored 0.41 to 0.71 and static places 0.76 to 0.89. A fixed threshold still failed on the high-angle videos, where every crop pair looks different because of the viewpoint, so each candidate is also compared with control places: the same region shifted sideways by one object width and seen through the same view pairs. A change must score at least 0.10 below its surroundings. This removed the last false alarm from the depth channel (a bottle seen from very different angles) as well.

Which video holds a transparent object is decided by differential zero-shot labelling: CLIP scores every object name on the before crops and on the after crops, and the name with the largest difference is the label; its sign gives the side. For test7 the top name was "drinking glass", on the before side. This works poorly when depth already found the object (it named background textures for the jug), so appearance candidates that coincide with a depth detection are dropped; depth gives the direction there.

Cost: about 30 seconds per pair, and the thresholds (0.72 similarity, 0.10 gap, 0.05 label margin) were tuned on the same eight pairs used for the results.

## 4. Exact outlines

The first boxes were rough. The glass came out 88 cm long and floating 10 cm above the desk, because both videos see it from similar directions and nothing fixes its depth along the viewing direction. Segment Anything, prompted with the projected box, outlines the object in each keyframe; the outlines from up to 12 keyframes then carve the 3D extent:

- a point is kept if it falls inside the outline in at least 75% of the views that see it;
- for an opaque object, space in front of its visible surface is empty;
- for a transparent object (detected when the depth behind the outline is clearly farther than the object), the object must stand in front of what is seen through it, and is assumed to be about as deep as it is wide and to rest on a surface.

The glass is now 10 × 9 × 19 cm and stands on the desk. The comparison video draws the outline in every frame on the side where the object exists, and a dashed box on the side where it is missing.

## 5. Text that cannot contradict the geometry

Every sentence is assembled from the numbers in `changes.json`: event type, label, size, distance and direction of a move, and position relative to the start of the first video. Heights are measured from the surface found under each object, not from one global plane; the desk in these videos sits about 10 cm higher at the back than the fitted plane. A language model (Claude) can name objects from crops, but it never writes the numbers.

## What I would do next

- Record pairs with moved objects, both opaque and transparent, and with several changes at once. Move matching is tested only on a synthetic scene.
- Describe locations relative to nearby objects ("left of the keyboard") instead of relative to the camera's starting point.
- Show the unknown areas in the comparison video, not only in the top-down map and the 3D recording.
- Hold out recordings that were not used for tuning, and report results on those.
