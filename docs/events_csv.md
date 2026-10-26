# `events.csv` column reference

Each session has one `events.csv`, at
`<data root>/<experiment>/events/<animal>/<session>/events.csv`. The same 34
columns appear in every experiment (cbm-odor, cbm-vision, ...).

**One row = one video frame.** Rows are already aligned to the session video
(`frame_idx` increases by 1 each row), so you can line up behavior, task
events, and pose tracking without merging anything. The video runs at about
30 fps, so consecutive rows are about 33 ms apart. A few frames at the very
start of the video may have no row.

Example (load with pandas):

```python
import pandas as pd
df = pd.read_csv("events.csv").set_index("frame_idx")
```

## Trial structure

A trial moves through these phases, in this order:

1. **Search**: the mouse searches the arena for the hidden `target_cell`.
2. **Reward**: the mouse found the target and pokes the water port.
   `reward_state` is True.
3. **Drinking**: `drinking` is True.
4. **ITI** (inter-trial interval): `iti` is True. Then the next trial starts.

Trial 0 is the warm-up period before the first real trial.

## Timing and indexing

| Column | Type | Description |
|---|---|---|
| `frame_idx` | int | Video frame number. Use this to look up a frame in the `.avi` file. |
| `trial_number` | int | Trial this frame belongs to (0, 1, 2, ...). Never missing. |
| `bonsai_ts` | int | Bonsai's own clock, in **milliseconds**. Use differences between rows for durations (e.g. how long a trial lasted). |
| `timestamp_ms` | float | The same moment on the **ephys clock**, in ms, produced by the alignment pipeline. Use this when you are lining up with neural data. It is close to `bonsai_ts` but not identical, so don't mix the two. |

## Task state and events (True/False)

| Column | Description |
|---|---|
| `target_cell` | Hidden target location for the current trial: a number from 0 to 26 that indexes the 3×9 grid of arena cells, numbered row by row from the top left (see `cbmcore.arena`). It is stored as a float because it is **blank (NaN)** once the search is over (during reward, drinking, and ITI). |
| `flip_state` | Integer 0, 1, or 2, constant within a trial. In cbm-vision it says which visual stimulus is used (0 and 1 are the two monitors). Ask before relying on it in other experiments. |
| `iti` | True during the inter-trial interval. |
| `reward_state` | True from the moment the target is found until the next recorded poke (median 1-1.6 s). If a trial has any True rows, it was rewarded. When a poke goes unrecorded (about 8% of found trials) it stays True into the next trial, so don't mark the end of the search with it: use `target_cell` going blank (see `cbmcore.epochs`). |
| `water_left` / `water_right` | True while the left/right water valve is open. These mostly overlap with `reward_state`. Left is used far more than right. |
| `poke_left` / `poke_right` | True while the mouse's snout is in the left/right port. This can fire many times per trial, including pokes that don't earn a reward. |
| `drinking` | True while the mouse is drinking after a reward. Looser than the `consumption` epoch of `cbmcore.epochs`, which keeps only frames within 0.5 s of a poke. |
| `click` | Brief audio click cue, True for only a few frames per trial. *(Confirm what triggers it.)* |

## Bonsai online tracking

These columns come from Bonsai's real-time tracker. They are coarser than the
SLEAP columns below.

| Column | Type | Description |
|---|---|---|
| `bonsai_centroid_x`, `bonsai_centroid_y` | int (px) | ⚠️ **Despite the name, this tracks the nose, not the body centroid.** `0, 0` means tracking was lost. |
| `speed` | float | Frame-to-frame movement of the Bonsai-tracked point, in pixels per frame. |
| `direction` | float (rad) | Coarse movement angle, limited to the range 0 to π/2 (about 1.57) and mostly 0, π/4, or π/2. **Don't use this as head direction.** Calculate heading from the SLEAP `nose` and `centroid` points instead, as the viewer does. |

## SLEAP pose tracking

Body-part positions from SLEAP. Each part has an `_x` and a `_y` column (float,
pixels in raw camera coordinates, origin at the top-left of the image). A value
is NaN on any frame where that part wasn't tracked.

| Prefix | Body part |
|---|---|
| `nose` | Nose tip |
| `centroid` | Body center |
| `tailbase` | Base of the tail |
| `paw_l`, `paw_r` | Left / right paw |
| `hs_l`, `hs_r`, `hs_t` | Headstage markers: left, right, top. *(Confirm the naming.)* |

## Related files

- `vision_events.csv` is in the same folder for cbm-vision and cbm-multisensory
  sessions. It records, for each frame, whether the stimulus is on screen and
  whether it is in the mouse's field of view. It is produced by
  `clickbait-preprocess/6a_sync_vision_stimuli.ipynb`.
- `cbmcore.events.load_events` loads the file (bool columns cleaned, sorted by clock);
  `cbmcore.epochs.label_epochs` labels each row's behavioral epoch.
