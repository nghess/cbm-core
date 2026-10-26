"""Checks on one real session; skipped when the recordings are not under paths.DATA_ROOT."""
import numpy as np
import pytest

from cbmcore import epochs, events, paths
from cbmcore.place import PlaceMaps
from cbmcore.units import load_units

SESSION = paths.Session('cbm-odor', '9005', 'x3')
pytestmark = [pytest.mark.data,
              pytest.mark.skipif(not SESSION.events_file.exists(), reason='recordings not found')]


def test_events_and_epochs():
    ev = events.load_events(SESSION)
    assert ev['timestamp_ms'].is_monotonic_increasing
    ep = epochs.label_epochs(ev)
    frac = np.bincount(ep, minlength=4) / len(ep)
    assert frac[epochs.SEARCH] > 0.5 and 0.01 < frac[epochs.RETRIEVAL] < 0.15


def test_units_and_place_maps():
    ev = events.load_events(SESSION)
    units, spike_ms = load_units(SESSION, 'hc')
    assert len(units) > 0 and set(units['label']) <= {'good', 'mua'}
    pm = PlaceMaps(ev['timestamp_ms'], *(ev[c] for c in events.xy('nose')))
    assert pm.occupancy().sum() == pytest.approx((ev['timestamp_ms'].iloc[-1] - ev['timestamp_ms'].iloc[0]) / 1000,
                                                 rel=0.02)
    si = pm.si(spike_ms[0])
    assert np.isfinite(si) and si >= 0


def test_vision_sessions_listed_once():
    sessions = paths.list_sessions('cbm-vision')
    assert len({s.key for s in sessions}) == len(sessions)
