"""Arena geometry: camera frame, target grid, long-axis zones, and the spatial bins of place maps.

Positions are camera pixels, origin at the top left of the image, x across the arena's short axis and y
along its long axis.
"""
import numpy as np

FRAME_W, FRAME_H = 888, 1968  # px; events.csv positions are in this full-resolution frame
FPS = 30

# Pixels per cm, from sl-data (the clickbait rig): 888 x 1968 px would be 17.5 x 38.8 cm.
# Not yet checked for the cbm rig.
PX_PER_CM = 50.7

# Target grid, from the task script (clickbait-place/clickbait_odor/clickbait_odor_task.py, GridMaze +
# get_grid_location): a GRID_COLS x GRID_ROWS grid over the whole frame whose outer ring is never a
# target, leaving INNER_COLS x INNER_ROWS cells. target_cell ids number those row by row from the top left
# (the low-y end): id = inner_row * INNER_COLS + inner_col.
GRID_COLS, GRID_ROWS = 5, 11
INNER_COLS, INNER_ROWS = GRID_COLS - 2, GRID_ROWS - 2
N_CELLS = INNER_COLS * INNER_ROWS  # 27
CELL_W, CELL_H = FRAME_W // GRID_COLS, FRAME_H // GRID_ROWS  # 177 x 178 px

# Long-axis zones. flip_state 0 targets and cues are confined to inner rows 0-3 (the low-y end), flip_state
# 1 to inner rows 5-8; inner row 4 is the neutral band holding both water ports. Zones go by y only, so
# the border ring belongs to its nearest end.
LOW_MAX_Y = (1 + 4) * CELL_H  # y below this -> 'low' (890 px)
HIGH_MIN_Y = (1 + 5) * CELL_H  # y at or above this -> 'high' (1068 px)
CUE_ZONE = {0: 'low', 1: 'high'}  # flip_state 2 has no cue

# Box tiled by place-map bins, ((x0, x1), (y0, y1)) px, for the SLEAP nose. The nose rarely reaches the
# top and bottom walls: across 309 sessions (Oct 2026) a median of 0.01% of its frames, and at most 1.6%,
# fall outside y 25-1933, so the box is cropped there to keep the end bins from being mostly empty.
# The body centroid stays within y 87-1860 and x 5-880.
PLACE_BOX = ((0, FRAME_W), (25, 1933))
FRAME_BOX = ((0, FRAME_W), (0, FRAME_H))
DEFAULT_BIN_PX = 129  # about 2.5 cm (1 inch) at PX_PER_CM; 7 x 15 bins over PLACE_BOX


def cell_bounds(cell):
    """(x0, y0, x1, y1) px of a target_cell id, or None if it is not one (e.g. NaN after the search)."""
    if cell is None or not (0 <= cell < N_CELLS):
        return None
    col, row = int(cell) % INNER_COLS, int(cell) // INNER_COLS
    x0, y0 = (col + 1) * CELL_W, (row + 1) * CELL_H
    return x0, y0, x0 + CELL_W, y0 + CELL_H


def cell_of(x, y):
    """target_cell id of each position, -1 in the border ring, outside the frame or untracked (NaN)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    with np.errstate(invalid='ignore'):
        gx, gy = np.floor(x / CELL_W), np.floor(y / CELL_H)
        inside = (gx >= 1) & (gx <= INNER_COLS) & (gy >= 1) & (gy <= INNER_ROWS)
    cell = np.where(inside, (np.nan_to_num(gy) - 1) * INNER_COLS + (np.nan_to_num(gx) - 1), -1)
    return cell.astype(int)


def dist_to_cell(x, y, cell):
    """Shortest distance (px) from a point to a target cell; 0 inside it."""
    x0, y0, x1, y1 = cell_bounds(cell)
    return float(np.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1)))


def zone_of(y):
    """'low' / 'mid' / 'high' long-axis zone of y (scalar or array)."""
    y = np.asarray(y, dtype=float)
    return np.where(y < LOW_MAX_Y, 'low', np.where(y >= HIGH_MIN_Y, 'high', 'mid'))


def bin_edges(bin_px=DEFAULT_BIN_PX, box=PLACE_BOX):
    """(x_edges, y_edges) tiling the box exactly, with bins as close to bin_px as fits."""
    def edges(lo, hi):
        return np.linspace(lo, hi, max(1, round((hi - lo) / bin_px)) + 1)

    (x0, x1), (y0, y1) = box
    return edges(x0, x1), edges(y0, y1)


def bin_index(x, y, x_edges, y_edges):
    """Flat bin index (ix * ny + iy, np.histogram2d(x, y).ravel() order) of each position; -1 outside the
    edges or untracked."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    ix = np.searchsorted(x_edges, x, side='right') - 1
    iy = np.searchsorted(y_edges, y, side='right') - 1
    ix[x == x_edges[-1]] = nx - 1  # the last bin is closed on the right
    iy[y == y_edges[-1]] = ny - 1
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return np.where(ok, ix * ny + iy, -1)
