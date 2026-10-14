# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Generate upper-excursion-set prominence fixtures using NumPy and a vertex activation oracle.

Definition: 8-connected finite cells, closed boundary, nodata barrier. Activate vertices in
order (-height, cell index). Each component carries its elder (highest, then lowest index).
At a merge at level h, the younger elder dies with prominence height-h. The final elder in
each connected component has infinite prominence. Zero-persistence plateau/nonpeak cells
remain in the dense result. This is 0-dimensional superlevel persistence, not sea-level
prominence. Independent of the GPU's sorted edge sweep; no reference package is required.
Run this file with the pinned pysal-venv Python; it writes the adjacent JSON fixture.
"""
import json
from pathlib import Path
import numpy as np


def prominence(surface):
    height, width = surface.shape
    values = surface.ravel()
    parents = np.arange(values.size)
    elders = np.arange(values.size)
    active = np.zeros(values.size, dtype=bool)
    drops = np.full(values.size, np.nan)
    saddles = np.full(values.size, np.nan)

    def root(cell):
        while parents[cell] != cell:
            parents[cell] = parents[parents[cell]]
            cell = parents[cell]
        return cell

    for cell in sorted(np.flatnonzero(np.isfinite(values)), key=lambda cell: (-float(values[cell]), int(cell))):
        active[cell] = True
        drops[cell], saddles[cell] = np.inf, -np.inf
        row, column = divmod(int(cell), width)
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                next_row, next_column = row + row_offset, column + column_offset
                if not (0 <= next_row < height and 0 <= next_column < width):
                    continue
                neighbor = next_row * width + next_column
                if not active[neighbor]:
                    continue
                first, second = root(cell), root(neighbor)
                if first == second:
                    continue
                first_peak, second_peak = elders[first], elders[second]
                if (-float(values[first_peak]), int(first_peak)) > (-float(values[second_peak]), int(second_peak)):
                    first, second = second, first
                    first_peak, second_peak = second_peak, first_peak
                drops[second_peak] = np.float32(values[second_peak] - values[cell])
                saddles[second_peak] = values[cell]
                parents[second] = first
                elders[first] = first_peak
    return drops, saddles


def encode(values):
    return [float(value) if np.isfinite(value) else 'NaN' if np.isnan(value) else 'Infinity' if value > 0 else '-Infinity' for value in values]


random = np.random.default_rng(20261014)
fixtures = [
    ('two-peaks', [[0, 9, 3, 7, 0]]),
    ('equal-peaks', [[8, 1, 8, 4, 8]]),
    ('plateau', [[1, 5, 5, 1], [1, 5, 5, 1], [0, 0, 0, 0]]),
    ('disconnected', [[7, 2, np.nan, 5, 4], [1, 0, np.nan, 0, 3]]),
    ('single', [[1e6]]),
    ('invalid', [[np.nan, np.inf], [-np.inf, np.nan]]),
    ('negative', [[-12, -3, -7, -1, -9]]),
    ('high-offset', [[1000000, 1000012, 1000003, 1000009, 1000000]]),
    ('nested-ridges', [[9, 8, 7, 8, 9], [8, 2, 1, 3, 8], [7, 1, 6, 2, 7], [8, 3, 2, 4, 8], [10, 8, 7, 8, 11]]),
    ('multi-batch', random.integers(-20, 40, size=(35, 40)))
]
result = []
for name, data in fixtures:
    surface = np.array(data, dtype=np.float32)
    drops, saddles = prominence(surface)
    result.append(dict(name=name, width=surface.shape[1], height=surface.shape[0], elevation=encode(surface.ravel()), prominence=encode(drops), keySaddle=encode(saddles)))
Path(__file__).with_suffix('.json').write_text(json.dumps({'definition': __doc__, 'numpy': np.__version__, 'cases': result}, indent=2) + '\n')
