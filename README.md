# cbm-core

The definitions every cbm project shares: where the data is, how to read `events.csv`, the behavioral
epochs, the arena geometry, place fields and phy units. Projects import these instead of keeping their own
copies, so a definition changes in one place.

Pure numpy / pandas / scipy, no plotting or GUI code. Python 3.10+.

## Install

Editable, into every environment that uses it:

    pip install -e d:/code/cbm-core[dev]     # [dev] adds pytest
    pytest                                   # from d:/code/cbm-core; real-data tests skip without E:/

## Modules

| Module | Contents |
|---|---|
| `cbmcore.paths` | `DATA_ROOT` (E:/, or the `CBM_DATA_ROOT` environment variable), experiments and regions; `Session(experiment, mouse, session)` with the path of every file; `list_sessions()` |
| `cbmcore.events` | `load_events()`: events.csv sorted by the ephys or Bonsai clock, bool columns cleaned; column names; `frame_durations_s()` |
| `cbmcore.epochs` | `label_epochs()`: search / retrieval / consumption (and unsplit) for every frame |
| `cbmcore.arena` | camera frame, the 5 x 11 target grid (`cell_bounds`, `cell_of`), long-axis zones, `PLACE_BOX`, `bin_edges`, `bin_index` |
| `cbmcore.place` | `PlaceMaps`: occupancy, rate maps, Skaggs SI and circular-shuffle nulls on fixed bins; `ssi_stats`; `upsample_rate_map` for display |
| `cbmcore.units` | `load_units()`: good/mua units of one region with spike times in ms |
| `docs/events_csv.md` | what every column of events.csv means |

## Example

```python
from cbmcore import Session, load_events, label_epochs, load_units, PlaceMaps, epochs, events

sess = Session('cbm-odor', '9005', 'x3')
ev = load_events(sess)                       # sorted by timestamp_ms (ephys clock)
ev['epoch'] = label_epochs(ev)               # label before dropping any frames
units, spike_ms = load_units(sess, 'hc')     # good + mua, spike times in ms

pm = PlaceMaps(ev['timestamp_ms'], *(ev[c] for c in events.xy('nose')))
search = (ev['epoch'] == epochs.SEARCH).to_numpy()
rate = pm.grid(pm.rate_map(spike_ms[0], search))   # (n_x, n_y) Hz; .T to plot
si = pm.si(spike_ms[0], search)
```

## Definitions and why

- **Epochs** (`epochs.py`): search, retrieval, consumption. (Named appetitive, consummatory and feeding
  until Oct 2026; only the names changed.) Retrieval starts when `target_cell` goes blank, not on `reward_state`'s
  rising edge: `reward_state` stays on until the next recorded poke, so an edge-triggered machine let
  retrieval run into the next search after every unrecorded poke (7.8% of found trials). A find with no
  recorded poke before the ITI is `UNSPLIT` and belongs to no epoch. Consumption keeps only frames within
  0.5 s of a poke, so it is stricter than events.csv's own `drinking` column, which is left as it is.
  Labels use `bonsai_ts`, so they are the same with or without ephys sync.
- **Place-map box** (`arena.PLACE_BOX`): x 0-888, y 25-1933 px for the SLEAP nose. Outside it fall a median
  of 0.01% of nose frames per session (at most 1.6%). The grid and zones use the full 888 x 1968 frame.
- **Bin size**: no single global value; `arena.DEFAULT_BIN_PX` = 129 px (about 2.5 cm at 50.7 px/cm, not
  1 cm) is the default for place maps, and each analysis can pass its own.
- **Occupancy** weights each frame by the interval to the next one (`events.frame_durations_s`), not by one
  interval for the whole session.
- **Units** (`units.py`): Kilosort's label overridden by phy curation; `cluster_info.tsv` not required.
  Selects exactly the units the viewers and the `cluster_info.tsv` rule (sniff-raster, position-decoding)
  did, in all 420 session-regions checked. `best_channel` is phy's `ch` (peak-to-peak of the unwhitened
  template), not the viewers' whitened-template channel, which differed for 14% of units.
- **Not yet checked**: `arena.PX_PER_CM` (50.7, from the clickbait rig) for the cbm rig.

## What does not belong here

Plotting, GUIs, decoders, sniff detection, MUA spike detection, and the Bonsai/ephys alignment (that stays
in clickbait-preprocess). Settings particular to one analysis (bin size, minimum spikes, number of shuffles,
colormaps) stay in that project's config.
# cbm-core
