"""Where the cbm data lives on disk, and how to find sessions.

Layout under <data root>/<experiment>/ (data root E:/ by default, or the CBM_DATA_ROOT environment
variable):

    bonsai/<mouse>/<session>/          raw Bonsai CSVs and the session video
    events/<mouse>/<session>/          events.csv on the ephys clock, plus vision_events.csv for vision experiments
    events_temp/<mouse>/<session>/     events.csv not yet synced to ephys (no timestamp_ms); cbm-vision only
    preprocessed/<mouse>/<session>/    ephys extracts (sniff.npy, reward_state.npy, <region>_mua.bin, ...)
    <region>-ks/<mouse>/<session>/     Kilosort/phy output of one brain region (hc, ob)

cbm-multisensory does not follow this layout yet (its sessions sit directly under <mouse>/).
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path(os.environ.get('CBM_DATA_ROOT', 'E:/'))
EXPERIMENTS = ('cbm-odor', 'cbm-vision', 'cbm-multisensory', 'cbm-null')
REGIONS = ('hc', 'ob')  # hippocampal CA1, olfactory bulb

# Searched in order; the first holding a session's events.csv wins. Every cbm-vision session in
# events/ is also in events_temp/, so the synced copy is the one found.
EVENTS_DIRS = ('events', 'events_temp')
EVENTS_FILE = 'events.csv'
VISION_EVENTS_FILE = 'vision_events.csv'

_SESSION_NAME = re.compile(r'([a-z]+)(\d+)([a-z]?)')


def parse_session_name(name):
    """(type, number, repeat) of a session name: ('x', 10, '') for 'x10', ('v', 0, 'b') for 'v0b',
    a second run of v0."""
    m = _SESSION_NAME.fullmatch(name)
    if m is None:
        raise ValueError(f'not a session name: {name!r}')
    return m.group(1), int(m.group(2)), m.group(3)


@dataclass(frozen=True)
class Session:
    """One recording session and the paths of its files. `root` defaults to DATA_ROOT."""
    experiment: str
    mouse: str
    session: str
    root: Path | None = None

    def __post_init__(self):
        object.__setattr__(self, 'mouse', str(self.mouse))

    @classmethod
    def from_key(cls, experiment, key, root=None):
        """Session from a '<mouse>/<session>' key, e.g. Session.from_key('cbm-odor', '9005/x3')."""
        mouse, session = key.split('/')
        return cls(experiment, mouse, session, root)

    def __str__(self):
        return f'{self.experiment} {self.key}'

    @property
    def key(self):
        return f'{self.mouse}/{self.session}'

    @property
    def session_type(self):
        return parse_session_name(self.session)[0]

    @property
    def session_number(self):
        return parse_session_name(self.session)[1]

    @property
    def exp_dir(self):
        return Path(self.root if self.root is not None else DATA_ROOT) / self.experiment

    def _subdir(self, name):
        return self.exp_dir / name / self.mouse / self.session

    @property
    def events_dir(self):
        """The first of EVENTS_DIRS holding this session's events.csv (events/ if none does)."""
        for name in EVENTS_DIRS:
            if (self._subdir(name) / EVENTS_FILE).exists():
                return self._subdir(name)
        return self._subdir(EVENTS_DIRS[0])

    @property
    def events_file(self):
        return self.events_dir / EVENTS_FILE

    @property
    def vision_events_file(self):
        return self.events_dir / VISION_EVENTS_FILE

    @property
    def bonsai_dir(self):
        return self._subdir('bonsai')

    @property
    def preprocessed_dir(self):
        return self._subdir('preprocessed')

    def ks_dir(self, region):
        """Kilosort/phy output of one region."""
        return self._subdir(f'{region}-ks')

    def has_units(self, region):
        return (self.ks_dir(region) / 'spike_times.npy').exists()

    def out_dir(self, name):
        """<experiment>/<name>/<mouse>/<session>/, for a project's own outputs (e.g. name='decoding')."""
        return self._subdir(name)


def _sort_key(s):
    try:
        kind, number, repeat = parse_session_name(s.session)
    except ValueError:
        kind, number, repeat = s.session, -1, ''
    return s.mouse, kind, number, repeat


def list_sessions(experiment, regions=None, require='any', events_dirs=EVENTS_DIRS, root=None):
    """Every session of an experiment with an events.csv, sorted by mouse, session type and number
    (so x2 comes before x10).

    regions: keep only sessions with Kilosort output (spike_times.npy) for 'any' or 'all' (`require`)
    of these regions. events_dirs: which events folders to search, in order; a session found in more
    than one is listed once. (Session.events_file itself always resolves through EVENTS_DIRS.)
    """
    exp_dir = Path(root if root is not None else DATA_ROOT) / experiment
    found = {}
    for name in events_dirs:
        for path in sorted((exp_dir / name).glob(f'*/*/{EVENTS_FILE}')):
            mouse, session = path.parts[-3], path.parts[-2]
            found.setdefault((mouse, session), Session(experiment, mouse, session, root))
    sessions = list(found.values())
    if regions is not None:
        test = any if require == 'any' else all
        sessions = [s for s in sessions if test(s.has_units(r) for r in regions)]
    return sorted(sessions, key=_sort_key)
