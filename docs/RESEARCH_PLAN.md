# What Changed in the Room? — Plan

**Deadline: Friday 16 October 2026** (today: Saturday 10 October)

## 1. The idea

Two phone videos of a room: **before** and **after**. Find what was **added**, **removed** or **moved**, show it in 3D, and describe it in words.

**Key rule: only report a change when there is evidence for it.**

- The before video saw empty space at that spot, and now there's an object → **added**
- There was an object, and now the after video sees empty space there → **removed**
- A removed object matches an added one → **moved**
- One of the videos never saw that spot → **unknown** (not a change)

That rule is what stops camera angle and hidden areas from showing up as fake changes.

## 2. How it works

```
before.mp4 + after.mp4
   → 1. pick sharp frames
   → 2. build ONE 3D scene from both videos (VGGT)
   → 3. compare with the evidence rule
   → 4. group changes into objects, name them (Claude vision)
   → 5. 3D view (Rerun) + text report
```

Run it with one command:

```
python run.py --before data/scene2/before.mp4 --after data/scene2/after.mp4
```

Example output:
> Moved: chair, 1.1 m toward the window.
> Added: backpack, on the floor by the bed.
> Removed: box, from the desk.
> Not checked: corner behind the door (not visible in the first video).

## 3. What to film (3 scenes)

Rules: 30–60 s each, walk slowly, start the second video from a different side, tape an A4 sheet somewhere fixed (for real-world scale), write down what you changed.

| Scene | Change between videos | Shows |
|---|---|---|
| 1. No change | nothing | no false alarms |
| 2. Basic | move a chair, add a backpack, remove a box | main demo |
| 3. Hidden | block an object with a board (object still there); add something in a corner the first video never saw | "hidden" and "unseen" are not reported as changes |

## 4. Files

```
run.py            # one command, runs everything
wcitr/
  frames.py       # step 1
  reconstruct.py  # step 2
  changes.py      # step 3
  objects.py      # step 4
  report.py       # step 5
data/             # videos + what I changed
results/          # changes.json, report.md, 3D view, images
README.md
```

## 5. Day-by-day schedule

| Day | Task | Done when |
|---|---|---|
| **Sat 10 Oct** | Install Python, PyTorch (CUDA), ffmpeg, VGGT. Film the 3 scenes. | VGGT runs on the GPU; 6 videos recorded |
| **Sun 11 Oct** | Steps 1–2: frames + shared 3D scene | Both videos appear as one aligned 3D scene in Rerun |
| **Mon 12 Oct** | Step 3: evidence rule | Scene 1 → no changes; scene 3 → "unknown" in the right place |
| **Tue 13 Oct** | Step 4: objects, moved matching, names | Scene 2 → correct chair / backpack / box |
| **Wed 14 Oct** | Step 5: coloured 3D view + text report | One command gives the full result for every scene |
| **Thu 15 Oct** | Fix problems, README with screenshots/GIF, results table, design note | A clean clone runs from the README |
| **Fri 16 Oct** | Final check, push to public GitHub, submit application | **Submitted** |

**If behind schedule:** skip Claude naming (use the detector labels) and skip scene 3's extra cases. Steps 1–3 + a simple 3D view + text still meet all the requirements.

## 6. Risks

| Risk | Fix |
|---|---|
| VGGT runs out of GPU memory | fewer frames or lower resolution |
| Small changes lost in noise | report the smallest change it can detect |
| Big change breaks alignment | re-align using only unchanged parts |
