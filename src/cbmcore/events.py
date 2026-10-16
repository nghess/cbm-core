"""Loading events.csv: one row per video frame of a session (column reference: docs/events_csv.md).

Two clocks, never to be mixed:
  bonsai_ts     Bonsai's own clock (ms). Always present; use it for behavior on its own.
  timestamp_ms  the same frame on the ephys clock (ms), written by the alignment in clickbait-preprocess.
                Use it whenever spikes or other ephys signals are involved. Missing (NaN) in sessions not
                yet synced to ephys (cbm-vision's events_temp/).
"""
from pathlib import Path

import numpy as np
import pandas as pd

EPHYS_CLOCK = 'timestamp_ms'
BONSAI_CLOCK = 'bonsai_ts'
CLOCKS = {'ephys': EPHYS_CLOCK, 'bonsai': BONSAI_CLOCK}

BOOL_COLUMNS = ('poke_left', 'poke_right', 'iti', 'water_left', 'water_right', 'reward_state', 'click',
                'drinking')
SLEAP_POINTS = ('nose', 'centroid', 'tailbase', 'paw_l', 'paw_r', 'hs_l', 'hs_r', 'hs_t')
BONSAI_POINT = 'bonsai_centroid'  # Bonsai's online tracking; despite the name it tracks the nose


def xy(point):
    """Column names of a tracked point: ('nose_x', 'nose_y') for 'nose'."""
    return f'{point}_x', f'{point}_y'


def as_bool(values):
    """Boolean array of an events column, with blanks read as False."""
    s = pd.Series(values)
    if s.dtype == bool:
        return s.to_numpy()
    return s.isin([True, 'True']).to_numpy()


def load_events(source, columns=None, clock='ephys'):
    """events.csv of a session (a paths.Session or a file path), sorted by `clock`.

    columns: the columns to read (default all); the clock column is always read. The BOOL_COLUMNS read
    as bool, blanks as False.
    clock: 'ephys' (timestamp_ms; raises ValueError if the session has not been synced to ephys),
    'bonsai' (bonsai_ts) or None (file order). Rows without a clock value are dropped.
    """
    path = Path(source.events_file if hasattr(source, 'events_file') else source)
    clock_col = CLOCKS[clock] if clock else None
    usecols = None
    if columns is not None:
        wanted = set(columns) | ({clock_col} if clock_col else set())
        usecols = lambda c: c in wanted  # noqa: E731
    df = pd.read_csv(path, usecols=usecols)
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f'{path} has no column(s) {missing}')
    for c in BOOL_COLUMNS:
        if c in df.columns:
            df[c] = as_bool(df[c])
    if clock_col:
        if clock_col not in df.columns or df[clock_col].isna().all():
            raise ValueError(f'{path} has no {clock_col} values: the session has not been synced to ephys. '
                             f"Use clock='bonsai' for behavior alone.")
        df = df.dropna(subset=[clock_col]).sort_values(clock_col, kind='stable').reset_index(drop=True)
    return df


def frame_durations_s(frame_ms, max_gap_ms=100.0):
    """Time each frame stands for (s): the interval to the next frame, as task-behavior's occupancy maps
    use. The last frame, and frames followed by a gap longer than max_gap_ms (a lost stretch of video or
    tracking), get the median interval instead, so the gap is not credited to one position.
    max_gap_ms=None keeps every interval as it is (the last frame then gets 0, as in task-behavior).

    Prefer this to a single frame interval: the median of the first 100 intervals, which the viewers
    used, runs 31-34 ms against a true mean of 33.35 ms."""
    t = np.asarray(frame_ms, dtype=float)
    if t.size < 2:
        return np.zeros(t.size)
    dt = np.diff(t)
    if max_gap_ms is None:
        return np.r_[dt, 0.0] / 1000.0
    median = np.median(dt)
    dt = np.where(dt > max_gap_ms, median, dt)
    return np.r_[dt, median] / 1000.0
