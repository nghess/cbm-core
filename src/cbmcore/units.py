"""Kilosort/phy units of one session and brain region.

Labels: cluster_KSLabel.tsv (Kilosort's automatic good/mua; always there) overridden cluster by cluster
by cluster_group.tsv when that file is headed 'group', i.e. once someone has curated in phy. So every
cluster is labelled good, mua or noise without needing cluster_info.tsv, which phy only writes when a
session is opened in it (three cbm-odor hc sessions never were). This is the cluster viewers' rule.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FS_HZ = 30000.0  # Kilosort sample rate of spike_times.npy
INCLUDE_LABELS = ('good', 'mua')


def _labels(path):
    """{cluster_id: label} of a phy cluster_*.tsv, and whether it is headed 'group' (curated)."""
    df = pd.read_csv(path, sep='\t')
    col = 'group' if 'group' in df.columns else 'KSLabel'
    return dict(zip(df['cluster_id'].astype(int), df[col].fillna('').astype(str))), col == 'group'


def cluster_labels(ks_dir):
    """{cluster_id: label} and the set of ids whose label comes from phy curation."""
    ks_dir = Path(ks_dir)
    labels, _ = _labels(ks_dir / 'cluster_KSLabel.tsv')
    curated = set()
    group_file = ks_dir / 'cluster_group.tsv'
    if group_file.exists():
        group, is_group = _labels(group_file)
        if is_group:
            labels.update(group)
            curated = set(group)
    return labels, curated


def _best_channels(ks_dir, ids):
    """phy's best channel ('ch' in cluster_info.tsv) of each cluster: the channel with the largest
    peak-to-peak amplitude of its unwhitened template (templates.npy @ whitening_mat_inv.npy). Computed
    from the templates, so it is there without cluster_info.tsv; that file's 'ch' is used for clusters
    without their own template (made by merging or splitting in phy), else -1. This rule reproduces phy's
    'ch' for all 12,077 units checked (Oct 2026); the viewers' largest absolute value of the whitened
    template disagreed for 14% of them."""
    ks_dir = Path(ks_dir)
    templates = np.load(ks_dir / 'templates.npy', mmap_mode='r') if (ks_dir / 'templates.npy').exists() else None
    unwhiten = np.load(ks_dir / 'whitening_mat_inv.npy') if (ks_dir / 'whitening_mat_inv.npy').exists() else None
    info_ch = {}
    if (ks_dir / 'cluster_info.tsv').exists():
        info = pd.read_csv(ks_dir / 'cluster_info.tsv', sep='\t')
        if 'ch' in info.columns:
            info_ch = dict(zip(info['cluster_id'].astype(int), info['ch'].astype(int)))
    out = []
    for cid in ids:
        if templates is not None and cid < len(templates):
            t = np.asarray(templates[cid], dtype=float)
            if unwhiten is not None:
                t = t @ unwhiten
            out.append(int(np.argmax(np.ptp(t, axis=0))))
        else:
            out.append(info_ch.get(cid, -1))
    return out


def load_units(source, region=None, include=INCLUDE_LABELS, min_spikes=0):
    """(units, spike_ms) of one region: a DataFrame with a row per unit (cluster_id, region, label,
    curated, best_channel, n_spikes) and a list of each unit's spike times (ms, ephys clock), in the same
    order.

    source: a paths.Session (with `region`) or a Kilosort/phy folder. Units are the clusters with spikes
    whose label is in `include` (clusters missing from both label files have label '' and are left out
    unless '' is included) and with at least min_spikes spikes over the whole session.
    """
    ks_dir = Path(source.ks_dir(region) if hasattr(source, 'ks_dir') else source)
    spike_times = np.load(ks_dir / 'spike_times.npy').ravel()
    spike_clusters = np.load(ks_dir / 'spike_clusters.npy').ravel()
    labels, curated = cluster_labels(ks_dir)

    order = np.argsort(spike_clusters, kind='stable')  # spike times stay sorted within a cluster
    ids, starts, counts = np.unique(spike_clusters[order], return_index=True, return_counts=True)
    units = pd.DataFrame(dict(cluster_id=ids.astype(int), start=starts, n_spikes=counts))
    units['label'] = [labels.get(int(c), '') for c in units['cluster_id']]
    units = units[units['label'].isin(include) & (units['n_spikes'] >= min_spikes)].reset_index(drop=True)

    spike_ms = [spike_times[order[a:a + n]] * (1000.0 / FS_HZ) for a, n in zip(units['start'], units['n_spikes'])]
    units['region'] = region if region is not None else ks_dir.parts[-3].removesuffix('-ks')
    units['curated'] = units['cluster_id'].isin(curated)
    units['best_channel'] = _best_channels(ks_dir, units['cluster_id'])
    units = units[['cluster_id', 'region', 'label', 'curated', 'best_channel', 'n_spikes']]
    return units, spike_ms
