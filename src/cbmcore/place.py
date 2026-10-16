"""Place fields: occupancy, firing-rate maps, Skaggs spatial information and its shuffle test.

A spike is placed at its nearest tracked frame (frame and spike times on the same clock: the ephys clock,
events.csv timestamp_ms). Bins tile a fixed box (arena.PLACE_BOX by default), so they are the same in every
session. Maps are flat arrays in np.histogram2d(x, y).ravel() order; PlaceMaps.grid() reshapes one to
(n_x, n_y), and .T of that is an image with y down the rows.

Combines cluster-viewer / cluster-states-viewer place_field.py and sniff-raster spatial_info.py.
"""
import numpy as np

from . import arena
from .events import frame_durations_s

MIN_OCCUPANCY_S = 0.1  # bins visited for less than this are left out (NaN) of rate maps


def nearest_frame(t, frame_ms):
    """Index of the nearest frame of each time (ties go to the later frame)."""
    t = np.asarray(t, dtype=float)
    idx = np.clip(np.searchsorted(frame_ms, t), 1, len(frame_ms) - 1)
    left = np.abs(t - frame_ms[idx - 1])
    right = np.abs(t - frame_ms[idx])
    return np.where(left < right, idx - 1, idx)


def skaggs_si(rate, p_i):
    """Skaggs spatial information (bits/spike; Skaggs et al. 1993) of a flat rate map (NaN = left out),
    given each bin's occupancy probability:  SI = sum_i p_i (r_i / r) log2(r_i / r)."""
    valid = ~np.isnan(rate)
    denom = p_i[valid].sum()
    if denom == 0:
        return 0.0
    r_mean = np.sum(rate[valid] * p_i[valid]) / denom
    if r_mean == 0:
        return 0.0
    active = valid & (rate > 0)
    ratio = rate[active] / r_mean
    return float(np.sum(p_i[active] * ratio * np.log2(ratio)))


def ssi_stats(si, null):
    """Compare an SI with its shuffle null: ssi (z-score, population sd), p_empirical (fraction of the null
    >= SI), p_gaussian (1 - Phi(ssi)), null_mean, null_sd."""
    from scipy.stats import norm

    null = np.asarray(null, dtype=float)
    sd = float(null.std())
    z = (si - null.mean()) / sd if sd > 0 else 0.0
    return dict(ssi=float(z), p_empirical=float(np.mean(null >= si)), p_gaussian=float(1 - norm.cdf(z)),
                null_mean=float(null.mean()), null_sd=sd)


def upsample_rate_map(rate_map, factor=8, order=3):
    """Smooth zoom of a 2-D rate map, for display only. Takes and returns (n_x, n_y); NaN regions stay NaN."""
    from scipy.ndimage import zoom

    valid = ~np.isnan(rate_map)
    up = zoom(np.nan_to_num(rate_map, nan=0.0), factor, order=order)
    up[zoom(valid.astype(float), factor, order=order) < 0.5] = np.nan
    return up


class PlaceMaps:
    """A session's tracked positions on a fixed grid of spatial bins, for rate maps of any number of units.

    frame_ms, x, y: every frame in time order (untracked frames as NaN positions are fine: they count
    toward no bin). frame_s: the time each frame stands for, a scalar or one value per frame (default
    events.frame_durations_s(frame_ms)).

    Methods taking `frames` restrict to a subset of frames (a bool mask over frames, e.g. one epoch or
    flip state): the subset's own occupancy, and the spikes whose nearest frame is in it.
    """

    def __init__(self, frame_ms, x, y, bin_px=arena.DEFAULT_BIN_PX, box=arena.PLACE_BOX, frame_s=None,
                 min_occupancy_s=MIN_OCCUPANCY_S):
        self.frame_ms = np.asarray(frame_ms, dtype=float)
        self.x, self.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        frame_s = frame_durations_s(self.frame_ms) if frame_s is None else frame_s
        self.frame_s = np.broadcast_to(np.asarray(frame_s, dtype=float), self.frame_ms.shape)
        self.min_occupancy_s = min_occupancy_s
        self.x_edges, self.y_edges = arena.bin_edges(bin_px, box)
        self.shape = (len(self.x_edges) - 1, len(self.y_edges) - 1)
        self.n_bins = self.shape[0] * self.shape[1]
        self.frame_bin = arena.bin_index(self.x, self.y, self.x_edges, self.y_edges)

    def _frames(self, frames):
        return np.ones(len(self.frame_ms), bool) if frames is None else np.asarray(frames, dtype=bool)

    def grid(self, flat):
        """A flat map as (n_x, n_y)."""
        return np.asarray(flat).reshape(self.shape)

    def occupancy(self, frames=None):
        """Seconds spent in each bin."""
        keep = self._frames(frames) & (self.frame_bin >= 0)
        return np.bincount(self.frame_bin[keep], weights=self.frame_s[keep], minlength=self.n_bins)

    def p_i(self, frames=None):
        occ = self.occupancy(frames)
        return occ / occ.sum() if occ.sum() > 0 else occ

    def spike_frames(self, spike_ms, frames=None):
        """Nearest frame of each spike within the tracked window [first frame, last frame], keeping only
        those whose nearest frame is in `frames`."""
        t = np.asarray(spike_ms, dtype=float)
        t = t[(t >= self.frame_ms[0]) & (t <= self.frame_ms[-1])]
        f = nearest_frame(t, self.frame_ms)
        return f[self._frames(frames)[f]]

    def _rate(self, spike_bins, occ):
        counts = np.bincount(spike_bins[spike_bins >= 0], minlength=self.n_bins)
        rate = np.full(self.n_bins, np.nan)
        visited = occ >= self.min_occupancy_s
        rate[visited] = counts[visited] / occ[visited]
        return rate

    def spike_counts(self, spike_ms, frames=None):
        b = self.frame_bin[self.spike_frames(spike_ms, frames)]
        return np.bincount(b[b >= 0], minlength=self.n_bins)

    def rate_map(self, spike_ms, frames=None):
        """Firing rate (Hz) in each bin; NaN in bins with less than min_occupancy_s."""
        return self._rate(self.frame_bin[self.spike_frames(spike_ms, frames)], self.occupancy(frames))

    def si(self, spike_ms, frames=None):
        """Skaggs spatial information (bits/spike)."""
        return skaggs_si(self.rate_map(spike_ms, frames), self.p_i(frames))

    def shuffle_si(self, spike_ms, n_shuffles, rng, frames=None):
        """SI of n_shuffles circular shifts of the spikes against the unshifted trajectory. The spikes'
        frames are shifted by one random amount, wrapping within the selected frames concatenated in time,
        so each shuffle keeps the spike count and the occupancy exactly. rng: np.random.Generator (seed it
        per unit for reproducible nulls)."""
        keep = np.flatnonzero(self._frames(frames))
        if keep.size == 0:
            return np.full(n_shuffles, np.nan)
        j = np.searchsorted(keep, self.spike_frames(spike_ms, frames))  # position of each spike in keep
        bins = self.frame_bin[keep]
        occ, p_i = self.occupancy(frames), self.p_i(frames)
        return np.array([skaggs_si(self._rate(bins[(j + s) % keep.size], occ), p_i)
                         for s in rng.integers(0, keep.size, n_shuffles)])
