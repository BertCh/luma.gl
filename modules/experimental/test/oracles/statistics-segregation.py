# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Reference segregation indices from the PySAL `segregation` package.

Writes fixtures/statistics-segregation.json next to this script, consumed (through
test/gpu-spatial-analysis/segregation/segregation-reference.ts) by gpu-segregation-pysal.spec.ts.
Run from anywhere with
    <venv>/bin/python -I modules/experimental/test/oracles/statistics-segregation.py
in a venv with numpy, pandas, geopandas, shapely, libpysal and segregation (versions are recorded
in the fixture).

Scene: a 6 x 5 grid of unit squares (6 columns, 5 rows, row-major unit ids, EPSG:3857 so the
package does not warn about geographic coordinates) with the fixed counts of 3 groups below
(unit-major: unit * 3 + group). The counts are the scene definition; they are not drawn from a
generator.

References (segregation 2.6, B = 0.5):
- `aspatial`: `MultiInfoTheory` (`entropy`), `MultiDissim` (`multiD`), and per group `Dissim`,
  `Isolation` and `Atkinson(b=0.5)` of that group against the total population;
- `queen`: the same indices as PySAL's spatially implicit indices, `w=libpysal Queen` weights
  (binary, self weight 1: `fill_diagonal`), which equal `spatialForm: 'smoothed-population'`.

Tolerances (stored in `tolerances`): the GPU computes in float32 and the specs compare to 2e-4
relative (+ 1e-6 absolute) for entropy, dissimilarity and isolation, and 5e-3 relative (+ 1e-5
absolute) for Atkinson, whose square-root terms amplify the float32 rounding of shares near zero.
"""

import json
import math
import os
import platform
import warnings

import geopandas
import libpysal
import numpy as np
import pandas
import segregation
import shapely
from libpysal.weights import Queen
from segregation.multigroup import MultiDissim, MultiInfoTheory
from segregation.singlegroup import Atkinson, Dissim, Isolation

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, 'fixtures', 'statistics-segregation.json')
COLUMNS = 6
ROWS = 5
GROUPS = 3
ATKINSON_B = 0.5
# prettier-ignore
COUNTS = [
    14, 10, 15, 44, 21, 18, 28, 11, 11, 21, 34, 25, 14, 13, 12, 10, 22, 16, 65, 10, 20, 55, 12, 18,
    42, 13, 18, 28, 34, 16, 12, 33, 14, 10, 31, 25, 56, 10, 19, 23, 15, 13, 33, 18, 16, 27, 24, 27,
    12, 45, 15, 10, 48, 18, 34, 10, 7, 44, 16, 22, 44, 26, 6, 17, 31, 10, 21, 55, 26, 10, 60, 8, 28,
    10, 23, 33, 11, 13, 21, 17, 9, 19, 26, 24, 19, 35, 16, 10, 63, 21,
]


def encode(value):
    """JSON-safe copy: NaN -> null, infinities -> strings (decoded by test/oracles/oracle-fixture.ts)."""
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return value
    return value


counts = np.array(COUNTS, dtype=float).reshape(COLUMNS * ROWS, GROUPS)
group_columns = [f'group{index}' for index in range(GROUPS)]
frame = geopandas.GeoDataFrame(
    pandas.DataFrame(counts, columns=group_columns),
    geometry=[
        shapely.box(unit % COLUMNS, unit // COLUMNS, unit % COLUMNS + 1, unit // COLUMNS + 1)
        for unit in range(COLUMNS * ROWS)
    ],
    crs='EPSG:3857',
)
frame['total'] = counts.sum(axis=1)


def get_indices(weights):
    keywords = {} if weights is None else {'w': weights}
    return {
        'entropy': float(MultiInfoTheory(frame, group_columns, **keywords).statistic),
        'multiD': float(MultiDissim(frame, group_columns, **keywords).statistic),
        'D': [float(Dissim(frame, group, 'total', **keywords).statistic) for group in group_columns],
        'isolation': [float(Isolation(frame, group, 'total', **keywords).statistic) for group in group_columns],
        'atkinson': [
            float(Atkinson(frame, group, 'total', b=ATKINSON_B, **keywords).statistic) for group in group_columns
        ],
    }


fixture = {
    'generator': 'modules/experimental/test/oracles/statistics-segregation.py',
    'versions': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pandas': pandas.__version__,
        'geopandas': geopandas.__version__,
        'shapely': shapely.__version__,
        'libpysal': libpysal.__version__,
        'segregation': segregation.__version__,
    },
    'tolerances': {
        'entropy': {'absolute': 1e-6, 'relative': 2e-4},
        'multiD': {'absolute': 1e-6, 'relative': 2e-4},
        'D': {'absolute': 1e-6, 'relative': 2e-4},
        'isolation': {'absolute': 1e-6, 'relative': 2e-4},
        'atkinson': {'absolute': 1e-5, 'relative': 5e-3},
    },
    'reference': {
        'columns': COLUMNS,
        'rows': ROWS,
        'groups': GROUPS,
        'counts': COUNTS,
        'aspatial': get_indices(None),
        'queen': get_indices(Queen.from_dataframe(frame, use_index=False)),
    },
}
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as handle:
    json.dump(encode(fixture), handle, separators=(',', ':'))
    handle.write('\n')
print('wrote', OUTPUT)
