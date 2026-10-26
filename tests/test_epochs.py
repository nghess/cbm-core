import numpy as np
import pandas as pd

from cbmcore.epochs import SEARCH, RETRIEVAL, CONSUMPTION, UNSPLIT, label_epochs

S, R, C, U = SEARCH, RETRIEVAL, CONSUMPTION, UNSPLIT


def frames(spec):
    """An events table from a frame-by-frame spec: '.' search (target set), 'f' target found (blank),
    'p' poke while blank, 'i' ITI. 100 ms per frame."""
    n = len(spec)
    return pd.DataFrame(dict(
        bonsai_ts=np.arange(n) * 100,
        target_cell=[5.0 if c == '.' else np.nan for c in spec],
        poke_left=[c == 'p' for c in spec],
        poke_right=False,
        iti=[c == 'i' for c in spec],
    ))


def test_poked_trial():
    #           search  found  poke cons  iti   search
    spec = '...' + 'ff' + 'pp' + 'ii' + '..'
    assert list(label_epochs(frames(spec))) == [S] * 3 + [R] * 2 + [C] * 2 + [S] * 4


def test_unpoked_trial_is_unsplit_and_the_next_trial_starts_in_search():
    spec = '...' + 'fff' + 'ii' + '...' + 'f' + 'p' + 'i'
    expected = [S] * 3 + [U] * 3 + [S] * 2 + [S] * 3 + [R] + [C] + [S]
    assert list(label_epochs(frames(spec))) == expected


def test_consumption_far_from_a_poke_goes_back_to_search():
    # 100 ms frames, gap tolerance 0.5 s: frames > 500 ms from the nearest poke onset leave consumption
    spec = 'f' + 'p' + 'f' * 14 + 'p' + 'i'  # pokes at 0.1 s and 1.6 s
    ep = label_epochs(frames(spec))
    assert ep[0] == R and ep[1] == C and ep[16] == C and ep[17] == S
    assert ep[5] == C and ep[12] == C  # 0.4 s from a poke
    assert ep[7] == S and ep[10] == S  # 0.6 s from the nearest poke


def test_session_opening_on_free_reward():
    ep = label_epochs(frames('ff' + 'p' + 'i' + '..'))
    assert list(ep) == [R, R, C, S, S, S]


def legacy_epochs(df):
    """The reward_state edge-triggered machine of the earlier copies (cluster-states-viewer, sniff-raster,
    position-decoding), without its consumption-gap pruning."""
    rs, poke, iti = df['reward_state'].to_numpy(), df['poke_left'].to_numpy(), df['iti'].to_numpy()
    out, state, prev = [], S, (False, False, False)
    for r, p, i in zip(rs, poke, iti):
        if state == S and r and not prev[0]:
            state = R
        elif state == R and p and not prev[1]:
            state = C
        elif state == C and i and not prev[2]:
            state = S
        out.append(state)
        prev = (r, p, i)
    return np.array(out)


def test_legacy_machine_runs_retrieval_into_the_next_search():
    # reward_state stays True from the find until the next recorded poke, as in events.csv
    spec = '.' + 'ff' + 'ii' + '....' + 'f' + 'p' + 'i'
    df = frames(spec)
    rs, on = [], False
    for c in spec:
        on = (on or c == 'f') and c != 'p'
        rs.append(on or c == 'p')
    df['reward_state'] = rs
    search = slice(5, 9)
    assert (legacy_epochs(df)[search] == R).all()
    assert (label_epochs(df)[search] == S).all()
