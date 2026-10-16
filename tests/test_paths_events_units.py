import numpy as np
import pandas as pd
import pytest

from cbmcore import events, paths
from cbmcore.paths import Session, list_sessions, parse_session_name
from cbmcore.units import load_units


def test_parse_session_name():
    assert parse_session_name('x10') == ('x', 10, '')
    assert parse_session_name('v0b') == ('v', 0, 'b')
    with pytest.raises(ValueError):
        parse_session_name('m0_old')


def write_events(path, ts_ms=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(dict(bonsai_ts=[30, 0, 60], poke_left=[True, None, False], iti=[None, None, True],
                           timestamp_ms=[1030.0, 1000.0, 1060.0] if ts_ms else np.nan, nose_x=[1.0, 2.0, 3.0]))
    df.to_csv(path, index=False)


def test_list_sessions_first_dir_wins_and_natural_order(tmp_path):
    exp = tmp_path / 'cbm-vision'
    for d, s in [('events', 'v10'), ('events', 'v2'), ('events_temp', 'v2'), ('events_temp', 'v3')]:
        write_events(exp / d / '9000' / s / 'events.csv', ts_ms=d == 'events')
    (exp / 'hc-ks' / '9000' / 'v3').mkdir(parents=True)
    (exp / 'hc-ks' / '9000' / 'v3' / 'spike_times.npy').touch()

    sessions = list_sessions('cbm-vision', root=tmp_path)
    assert [s.session for s in sessions] == ['v2', 'v3', 'v10']
    assert sessions[0].events_file == exp / 'events' / '9000' / 'v2' / 'events.csv'
    assert sessions[1].events_file == exp / 'events_temp' / '9000' / 'v3' / 'events.csv'
    assert [s.session for s in list_sessions('cbm-vision', regions=['hc', 'ob'], root=tmp_path)] == ['v3']
    assert list_sessions('cbm-vision', regions=['hc', 'ob'], require='all', root=tmp_path) == []


def test_session_paths(tmp_path):
    s = Session.from_key('cbm-odor', '9005/x3', root=tmp_path)
    assert s.mouse == '9005' and s.session_type == 'x' and s.session_number == 3
    assert s.ks_dir('hc') == tmp_path / 'cbm-odor' / 'hc-ks' / '9005' / 'x3'
    assert s.out_dir('decoding') == tmp_path / 'cbm-odor' / 'decoding' / '9005' / 'x3'
    assert Session('cbm-odor', 9005, 'x3').mouse == '9005'
    assert str(s) == 'cbm-odor 9005/x3'


def test_load_events(tmp_path):
    path = tmp_path / 'events.csv'
    write_events(path)
    ev = events.load_events(path)
    assert list(ev['timestamp_ms']) == [1000.0, 1030.0, 1060.0]
    assert ev['poke_left'].dtype == bool and list(ev['poke_left']) == [False, True, False]
    assert list(ev['iti']) == [False, False, True]
    assert list(events.load_events(path, columns=['nose_x'], clock='bonsai').columns) == ['bonsai_ts', 'nose_x']
    with pytest.raises(KeyError):
        events.load_events(path, columns=['speed'])

    write_events(path, ts_ms=False)
    with pytest.raises(ValueError, match='synced to ephys'):
        events.load_events(path)
    assert len(events.load_events(path, clock='bonsai')) == 3


def test_frame_durations():
    t = np.array([0, 33, 66, 100, 1000, 1033], dtype=float)
    d = events.frame_durations_s(t)
    assert np.allclose(d, [0.033, 0.033, 0.034, 0.033, 0.033, 0.033])  # the 900 ms gap gets the median
    assert np.allclose(events.frame_durations_s(t, max_gap_ms=None)[3:], [0.9, 0.033, 0.0])


def make_ks(tmp_path, group_header='group'):
    ks = tmp_path / 'cbm-odor' / 'hc-ks' / '9000' / 'o1'
    ks.mkdir(parents=True)
    clusters = np.array([0, 1, 2, 1, 0, 1, 5, 5])  # cluster 3 has a label but no spikes; 5 is a merge
    np.save(ks / 'spike_times.npy', (np.arange(8) * 30).astype(np.uint64)[:, None])
    np.save(ks / 'spike_clusters.npy', clusters)
    templates = np.zeros((4, 3, 2))
    templates[:, 1, 1] = 1.0  # every template peaks on channel 1
    np.save(ks / 'templates.npy', templates)
    pd.DataFrame(dict(cluster_id=[0, 1, 2, 3], KSLabel=['good', 'mua', 'good', 'good'])).to_csv(
        ks / 'cluster_KSLabel.tsv', sep='\t', index=False)
    pd.DataFrame({'cluster_id': [2, 5], group_header: ['noise', 'good']}).to_csv(
        ks / 'cluster_group.tsv', sep='\t', index=False)
    return ks


def test_load_units_labels_and_spike_times(tmp_path):
    make_ks(tmp_path)
    s = Session('cbm-odor', '9000', 'o1', root=tmp_path)
    units, spike_ms = load_units(s, 'hc')
    assert list(units['cluster_id']) == [0, 1, 5]
    assert list(units['label']) == ['good', 'mua', 'good']
    assert list(units['curated']) == [False, False, True]
    assert list(units['n_spikes']) == [2, 3, 2]
    assert list(units['best_channel']) == [1, 1, -1]  # 5 has no template and no cluster_info.tsv
    assert set(units['region']) == {'hc'}
    assert np.allclose(spike_ms[1], [1.0, 3.0, 5.0])  # samples 30, 90, 150 at 30 kHz
    assert list(load_units(s.ks_dir('hc'), min_spikes=3)[0]['cluster_id']) == [1]
    assert load_units(s.ks_dir('hc'))[0]['region'].iloc[0] == 'hc'


def test_best_channel_uses_unwhitened_peak_to_peak(tmp_path):
    ks = make_ks(tmp_path)
    np.save(ks / 'whitening_mat_inv.npy', np.array([[0.0, 1.0], [1.0, 0.0]]))  # swaps the two channels
    pd.DataFrame(dict(cluster_id=[0, 1, 5], ch=[9, 9, 0])).to_csv(ks / 'cluster_info.tsv', sep='\t', index=False)
    units, _ = load_units(ks, 'hc')
    assert list(units['best_channel']) == [0, 0, 0]  # templates for 0 and 1, cluster_info for merged 5


def test_uncurated_group_file_is_ignored(tmp_path):
    ks = make_ks(tmp_path, group_header='KSLabel')
    units, _ = load_units(ks, 'hc')
    assert list(units['cluster_id']) == [0, 1, 2] and not units['curated'].any()


def test_data_root_is_overridable(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, 'DATA_ROOT', tmp_path)
    assert Session('cbm-odor', '9000', 'o1').exp_dir == tmp_path / 'cbm-odor'
