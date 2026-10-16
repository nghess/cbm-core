"""Behavioral epoch of every frame of a session.

  appetitive   - from leaving the port after one reward until the next target is found. Includes the ITI.
  consummatory - target found: until the mouse pokes a port.
  feeding      - at the port: from that poke until the ITI begins, keeping only frames within
                 FEEDING_GAP_S of a poke pulse. The rest of that window (the mouse has wandered off
                 before the ITI starts) goes back to appetitive.

Feeding is stricter than events.csv's own `drinking` column, which is left as it is.

The consummatory epoch starts on the frame target_cell goes blank (the task consumed the target), not
on reward_state's rising edge. reward_state is not a one-frame pulse: it stays True from the click until
the next recorded poke, so when a poke goes unrecorded it stays True into the next trial. Across
cbm-odor and cbm-vision (Oct 2026) 7.8% of found trials have no poke recorded before the next ITI, and
an edge-triggered state machine (the earlier cluster-states-viewer version, copied into sniff-raster and
position-decoding) labelled the following search consummatory: 12-14% of frames consummatory instead of
about 6%, and up to a quarter of a session's frames mislabelled.

When a target is found and no poke is recorded before the next ITI, the stretch in between holds an
unlogged visit to the port and cannot be split into consummatory and feeding. It is labelled UNSPLIT and
belongs to no epoch, instead of letting the consummatory epoch run on into the next search.

Ported from task-behavior/src/epochs.py (drinking renamed feeding).
"""
import numpy as np

from .events import as_bool

APPETITIVE, CONSUMMATORY, FEEDING, UNSPLIT = range(4)
EPOCHS = ('appetitive', 'consummatory', 'feeding')  # label i is EPOCHS[i]; UNSPLIT belongs to none
COLUMNS = ('bonsai_ts', 'target_cell', 'iti', 'poke_left', 'poke_right')
FEEDING_GAP_S = 0.5  # licking re-pokes every ~100-300 ms; a longer gap means the mouse left the port


def _onsets(flag):
    """Indices where a boolean array turns True."""
    return np.flatnonzero(flag[1:] & ~flag[:-1]) + 1


def label_epochs(events, feeding_gap_s=FEEDING_GAP_S):
    """int8 epoch label (APPETITIVE / CONSUMMATORY / FEEDING / UNSPLIT) of every row of an events table
    in time order (it needs COLUMNS). Label the full table before dropping untracked frames, so that no
    transition is missed. Labels use bonsai_ts, so they are the same whether or not a session has been
    synced to ephys."""
    ts = events['bonsai_ts'].to_numpy(float) / 1000
    no_target = events['target_cell'].isna().to_numpy()
    poke = as_bool(events['poke_left']) | as_bool(events['poke_right'])
    iti = as_bool(events['iti'])

    # walk the three kinds of event in time order: target found, poke onset, ITI onset
    marks = sorted([(i, 'found') for i in _onsets(no_target)] + [(i, 'poke') for i in _onsets(poke)] +
                   [(i, 'iti') for i in _onsets(iti)])
    epoch = np.full(len(ts), APPETITIVE, dtype=np.int8)
    state, since = (CONSUMMATORY if len(ts) and no_target[0] else APPETITIVE), 0  # a session opens with a free reward
    for i, kind in marks:
        new = state
        if state == APPETITIVE and kind == 'found':
            new = CONSUMMATORY
        elif state == CONSUMMATORY and kind == 'poke':
            new = FEEDING
        elif state == CONSUMMATORY and kind == 'iti':
            state, new = UNSPLIT, APPETITIVE  # no poke was recorded for this reward
        elif state == FEEDING and kind == 'iti':
            new = APPETITIVE
        if new != state or state == UNSPLIT:
            epoch[since:i] = state
            state, since = new, i
    epoch[since:] = state

    # feeding is only the time actually spent at the port
    feeding = np.flatnonzero(epoch == FEEDING)
    poke_t = ts[poke]
    if len(feeding) and len(poke_t):
        k = np.clip(np.searchsorted(poke_t, ts[feeding]), 1, len(poke_t) - 1)
        gap = np.minimum(np.abs(ts[feeding] - poke_t[k - 1]), np.abs(poke_t[k] - ts[feeding]))
        epoch[feeding[gap > feeding_gap_s]] = APPETITIVE
    return epoch
