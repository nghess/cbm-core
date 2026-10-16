import numpy as np
import pytest

from cbmcore import arena
from cbmcore.place import PlaceMaps, nearest_frame, skaggs_si, ssi_stats, upsample_rate_map


def test_cells_round_trip():
    for cell in range(arena.N_CELLS):
        x0, y0, x1, y1 = arena.cell_bounds(cell)
        assert arena.cell_of((x0 + x1) / 2, (y0 + y1) / 2) == cell
    assert arena.cell_bounds(0) == (177, 178, 354, 356)
    assert arena.cell_bounds(np.nan) is None and arena.cell_bounds(27) is None


def test_cell_of_border_and_untracked():
    x = np.array([10, 444, 444, np.nan, 444])
    y = np.array([500, 100, 1900, 500, 500])
    assert list(arena.cell_of(x, y)) == [-1, -1, -1, -1, 4]  # inner row 1, inner col 1


def test_zones():
    assert list(arena.zone_of([889, 890, 1067, 1068])) == ['low', 'mid', 'mid', 'high']


def test_default_bins():
    xe, ye = arena.bin_edges()
    assert (len(xe) - 1, len(ye) - 1) == (7, 15)
    assert xe[0] == 0 and xe[-1] == 888 and ye[0] == 25 and ye[-1] == 1933


def test_bin_index_matches_histogram2d():
    rng = np.random.default_rng(0)
    xe, ye = arena.bin_edges(100, ((0, 400), (0, 300)))
    x, y = rng.uniform(-20, 420, 2000), rng.uniform(-20, 320, 2000)
    b = arena.bin_index(x, y, xe, ye)
    expected = np.histogram2d(x, y, bins=[xe, ye])[0].ravel()
    assert np.array_equal(np.bincount(b[b >= 0], minlength=12), expected)
    assert arena.bin_index([400.0], [300.0], xe, ye)[0] == 11  # far corner is in the last bin
    assert arena.bin_index([np.nan], [10.0], xe, ye)[0] == -1


def test_nearest_frame():
    frame_ms = np.array([0.0, 10.0, 20.0])
    assert list(nearest_frame([-5, 4, 5, 6, 26], frame_ms)) == [0, 0, 1, 1, 2]


def two_bin_maps():
    # 1000 frames at 10 ms, alternating halves of a 2 x 1 grid
    frame_ms = np.arange(1000) * 10.0
    x = np.where((np.arange(1000) // 100) % 2 == 0, 50.0, 150.0)
    return PlaceMaps(frame_ms, x, np.full(1000, 50.0), bin_px=100, box=((0, 200), (0, 100)), frame_s=0.01)


def test_rate_map_and_si():
    pm = two_bin_maps()
    assert np.allclose(pm.occupancy(), [5.0, 5.0])
    spikes = pm.frame_ms[pm.x == 50.0][::10]  # 50 spikes, all in the left bin
    assert np.allclose(pm.rate_map(spikes), [10.0, 0.0])
    assert pm.si(spikes) == pytest.approx(1.0)  # p = 0.5 each, all spikes in one bin -> log2(2)


def test_frames_subset_and_min_occupancy():
    pm = two_bin_maps()
    first = pm.frame_ms < 1000  # the first 100 frames, all in the left bin
    assert np.allclose(pm.occupancy(first), [1.0, 0.0])
    rate = pm.rate_map(pm.frame_ms[::5], first)
    assert rate[0] == pytest.approx(20.0) and np.isnan(rate[1])


def test_shuffle_keeps_spike_count_and_is_seeded():
    pm = two_bin_maps()
    spikes = pm.frame_ms[pm.x == 50.0][::10]
    a = pm.shuffle_si(spikes, 50, np.random.default_rng(1))
    b = pm.shuffle_si(spikes, 50, np.random.default_rng(1))
    assert np.array_equal(a, b) and a.mean() < pm.si(spikes)
    stats = ssi_stats(pm.si(spikes), a)
    assert stats['ssi'] > 0 and 0 <= stats['p_empirical'] <= 1


def test_skaggs_ignores_nan_bins():
    assert skaggs_si(np.array([1.0, 1.0, np.nan]), np.array([0.5, 0.5, 0.0])) == 0.0


def test_upsample_keeps_orientation():
    up = upsample_rate_map(np.ones((3, 5)), factor=4)
    assert up.shape == (12, 20)
