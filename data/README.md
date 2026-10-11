# Data

Phone recordings of the same desk area, 1024×576, 3–5 s each (85–129 frames).

```
data/
├── scenes.yaml            # which reference each test is compared with + ground-truth changes
├── reference/             # "before" recordings
│   ├── ref1_light1.mp4    # cool white light
│   └── ref2_light2.mp4    # warm lamp light
└── test/                  # "after" recordings (objects moved / added / removed)
    ├── test1.mp4          # warm, very dark          -> compared with ref2_light2
    ├── test3.mp4          # cool, high camera angle  -> ref1_light1
    ├── test4.mp4          # cool                     -> ref1_light1
    ├── test5.mp4          # cool, high camera angle  -> ref1_light1
    ├── test6.mp4          # cool                     -> ref1_light1
    ├── test7.mp4          # cool                     -> ref1_light1
    ├── test8.mp4          # cool, high camera angle  -> ref1_light1
    └── test9.mp4          # warm                     -> ref2_light2
```

Each test is paired with the reference filmed under the same light, measured from the average
red/blue colour ratio of the video (light 1 ≈ 1.1, light 2 ≈ 1.7). The `lighting_null` scene
compares the two references, which tests robustness to lighting.

Original file names: `Reference-1 ligthing condition-1.mp4`, `Reference-2 ligthing condition-2.mp4`,
`Test-N.mp4`. There is no Test-2.
