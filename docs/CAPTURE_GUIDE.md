# Capture Guide

## Rules for every recording

- 30–60 s per video, 1080p, 30 fps, main (1×) lens. Lock focus/exposure if your phone allows it.
- Walk slowly in an arc. Move your feet. Don't just turn on the spot.
- Film each changed area from at least 3 directions.
- Start the **after** video from a different side than the **before** video.
- Tape an A4 sheet to something that never moves (scale check).
- Write down every change in `data/<scene>/truth.yaml`: object, what happened, rough distance from a tape measure.

## Challenging scenarios

Priority: **must** = needed for the submission, **good** = shows robustness, **extra** = known limit, film if time allows.

### A. No-change controls (false alarms)

| # | Scenario | What it tests | Priority |
|---|---|---|---|
| A1 | Nothing changed, different walking path | Viewpoint differences alone must give zero changes | must |
| A2 | Nothing changed, lighting changed (lamp on/off, curtains, day vs night) | Shadows and brightness are not changes | must |
| A3 | Nothing changed, same path filmed twice | Noise floor of the system | good |

### B. Basic changes

| # | Scenario | What it tests | Priority |
|---|---|---|---|
| B1 | Chair moved ~1 m | Moved + distance | must |
| B2 | Backpack added in open floor space | Added | must |
| B3 | Box removed from a table | Removed | must |
| B4 | 3–5 changes at once | Separating several events | must |

### C. Hard kinds of change

| # | Scenario | What it tests | Priority |
|---|---|---|---|
| C1 | Small move: 5 cm, 10 cm, 20 cm | Smallest detectable change | must |
| C2 | Chair rotated 90° in place | Rotation with no movement | must |
| C3 | Two different objects swap places | Two crossed moves | good |
| C4 | Two identical objects swap (two same mugs) | Impossible to see; should not invent moves | good |
| C5 | One of three identical chairs moved | Matching among look-alikes | good |
| C6 | Big object moved (table, sofa) | Alignment when much of the room changed | must |
| C7 | Book placed on top of a box | Object stacked on another object | good |
| C8 | Item put inside a drawer or box | Object disappears with no new location | extra |

### D. Hidden and unseen areas

| # | Scenario | What it tests | Priority |
|---|---|---|---|
| D1 | Board placed in front of an object (object still there) | Hidden is not removed | must |
| D2 | Object added in a corner the before video never filmed | Unseen is "unknown", not "added" | must |
| D3 | Object removed, revealing a wall never seen before | Newly visible wall is not "added" | good |
| D4 | Object moved to a spot that was hidden in the before video | Weak evidence at the destination | good |
| D5 | Object only partly in frame in one video | Partial views | extra |
| D6 | After video from the opposite side, little overlap | Limit of joint reconstruction | good |

### E. Hard materials and shapes

| # | Scenario | What it tests | Priority |
|---|---|---|---|
| E1 | Flat objects: book on table, sheet of paper, closed laptop | Height change below depth noise | must |
| E2 | Thin objects: cable, umbrella, broom | Hard to reconstruct | good |
| E3 | Glass bottle, mirror, monitor, window | Bad depth gives ghost changes | good |
| E4 | White box against a white wall | No texture | good |
| E5 | Door/drawer opened, laptop lid opened | Part of an object moves | good |
| E6 | Same-shape object, different colour (red mug → blue mug) | Change without geometry change | good |
| E7 | Blanket refolded, jacket moved, pillow squashed | Soft objects change shape | extra |
| E8 | Screen shows a different image | Change that should be ignored | extra |
| E9 | Small items on a messy desk (keys, phone, pen) | Small objects in clutter | extra |

### F. Recording problems

| # | Scenario | What it tests | Priority |
|---|---|---|---|
| F1 | A person walks through one video | Moving things inside one video are ignored | good |
| F2 | Low light, fast motion, blur | Frame selection and depth quality | good |
| F3 | Before in landscape, after in portrait or 0.5× lens | Different camera settings | extra |
| F4 | Repetitive patterns (bookshelf, tiles) | Alignment confusion | extra |

## Suggested scenes (combine scenarios to save time)

| Scene | Scenarios | Purpose |
|---|---|---|
| scene1_null | A1 | Zero false alarms |
| scene2_basic | B1 + B2 + B3 + E1 | Main demo |
| scene3_hidden | D1 + D2 + D3 | Unknown vs changed |
| scene4_subtle | C1 (10 cm) + C2 | Sensitivity |
| scene5_big | C6 + C5 | Large change |
| scene6_lighting | A2 | Lighting robustness |
| scene7_swap | C3 + C4 | Matching ambiguity |
| scene8_materials | E2 + E3 + E5 + E6 | Honest limits |
| scene9_chaos | B4 + F1 + F2 | Everything at once |

Film scenes 1–3 first. They are enough for a complete submission.
