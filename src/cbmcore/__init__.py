"""Shared definitions for the cbm experiments. See README.md."""
from . import arena, epochs, events, paths, place, units
from .epochs import label_epochs
from .events import load_events
from .paths import Session, list_sessions
from .place import PlaceMaps
from .units import load_units

__version__ = '0.1.0'
