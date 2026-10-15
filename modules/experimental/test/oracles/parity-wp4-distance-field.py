# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""SciPy exact Euclidean distance transform, full 750x750 Dixie burn mask.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
The round-2 fixture is reused ONLY as a source-data mask, never as a distance
oracle. Its original raw-source SHA256 records are carried forward. Solve the
whole raster before serialization; no cropped-tile approximation. Source row
order retained, planar cells explicitly 20x20 or 12x16 units (the latter is an
anisotropic algorithm test, not the original CRS). Inputs exactly f32.

SciPy tie-selected indices need not match GPU ties. Full squared optimal
distances are integer-exact with these cell spacings, stored losslessly as gzip/base64 RLE;
spec verifies that EVERY reported nearest cell is a seed with exactly that
squared distance. Full mask uses common label 0, so all tied indices are valid.
Small labelled masks use scipy.spatial.distance.cdist to enumerate all exact
ties and require the documented smallest label. No near-tie relaxation.
Empty mask is explicitly infinity (SciPy otherwise invents an outside seed).
Point seeds test snapping, duplicates, disabled IDs and active-count truncation.
"""
import json
import gzip
import base64
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
HERE = Path(__file__).resolve().parent
NONE = 4294967295


def runs(values):
    flat = np.asarray(values).ravel()
    boundaries = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1])+1, len(flat)]
    values = np.stack([np.diff(boundaries), flat[boundaries[:-1]]], axis=-1).astype(np.uint64).ravel().tolist()
    return base64.b64encode(gzip.compress(json.dumps(values, separators=(',', ':')).encode(), mtime=0)).decode()


def decode(values, shape):
    return np.repeat(np.array(values[1::2], dtype=np.uint32), values[::2]).reshape(shape)


def make_case(name, mask, cells, limit=1e20, points=None, ids=None, count=0, labelled=False):
    height, width = mask.shape
    effective = mask.copy()
    if points:
        for point, label in zip(points[:count], ids[:count]):
            column, row = np.floor(np.array(point)/cells).astype(int)
            if 0 <= column < width and 0 <= row < height and label != NONE:
                effective[row, column] = min(int(effective[row, column]) or NONE, label+1)
    if effective.any():
        distance, indices = distance_transform_edt(effective == 0, sampling=cells[::-1], return_indices=True)
        rows, columns = np.indices(mask.shape)
        squared = ((indices[1]-columns)*cells[0])**2 + ((indices[0]-rows)*cells[1])**2
        np.testing.assert_allclose(distance**2, squared, rtol=1e-14, atol=1e-10)
    else:
        squared = np.full(mask.shape, NONE, dtype=np.uint32)
    allocation = []
    if labelled:
        rows, columns = np.indices(mask.shape)
        targets = np.column_stack([columns.ravel(), rows.ravel()])*cells
        seed_rows, seed_columns = np.nonzero(effective)
        distances = cdist(targets, np.column_stack([seed_columns, seed_rows])*cells, metric='sqeuclidean')
        labels = effective[seed_rows, seed_columns]-1
        allocation = np.min(np.where(distances == squared.ravel()[:, None], labels[None], NONE), axis=1).tolist()
    return dict(name=name, reference='scipy.ndimage.distance_transform_edt(return_indices=True); scipy.spatial.distance.cdist ties',
                width=width, height=height, cellSize=cells, maximumDistance=limit,
                mask=runs(mask), effectiveMask=runs(effective), squared=runs(squared), allocation=allocation,
                positions=np.array(points or []).ravel().tolist(), ids=ids or [], count=count)


source = json.loads((HERE/'fixtures/parity-wp4-components-dixie-4.json').read_text())
original = source['cases'][0]
mask = decode(original['values'], (750, 750)) * decode(original['expectedValidity'], (750, 750))
versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


def write(suffix, cases, sources):
    output = HERE/f'fixtures/parity-wp4-distance-{suffix}.json'
    output.write_text(json.dumps(dict(versions=versions, sources=sources, cases=cases), separators=(',', ':'))+'\n')
    assert output.stat().st_size <= 1_500_000, output.stat().st_size
    print(output.name, output.stat().st_size)


for suffix, cells in [('isotropic', [20, 20]), ('anisotropic', [12, 16])]:
    write(suffix, [make_case(f'dixie-full-{suffix}', mask, cells)], source['sources'])
small = np.zeros((9, 17), dtype=np.uint32)
small[1, 1] = 10; small[1, 9] = 3; small[7, 1] = 8; small[7, 9] = 2
cases = [make_case('labelled-exact-ties', small, [3, 4], labelled=True),
         make_case('cutoff-equality', small, [3, 4], limit=12, labelled=True),
         make_case('empty', np.zeros((3, 5), dtype=np.uint32), [1, 1]),
         make_case('single-cell', np.array([[8]], dtype=np.uint32), [1, 1], labelled=True),
         make_case('points-mask-duplicates-disabled-count', small, [3, 4], labelled=True,
                   points=[[3.25, 4.25], [3.75, 4.75], [-1, 0], [18.25, 12.25], [15.25, 16.25], [30.25, 24.25]],
                   ids=[4, 1, 0, NONE, 5, 0], count=5)]
write('synthetic', cases, [])
