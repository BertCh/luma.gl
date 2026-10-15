# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Full-raster component reference: scipy.ndimage.label, independently checked
against skimage.measure.label on binary foreground for both 4 and 8 connectivity.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label

Decode all four 750x750 Sentinel bands for the Dixie showcase. NBR uses f64
arithmetic on exact f32-rounded uint16 DN inputs; dNBR > .66 defines binary high
burn severity. Classification is fixture preparation, NOT a test of GPU NBR.
Zero band DN is invalid; cloud/shadow SCL {0,1,3,8,9,10,11} is independently
invalid. Image row order north->south, EPSG:32610 at 20m; no spatial resampling.
Full global components are solved BEFORE encoding, never independent crops.
RLE is lossless transport for the complete input and output arrays, NOT a
summary: GPU specs decode and compare all 562,500 cells, including rows/columns
511 and 512 and every workgroup seam. Minimum row-major member + 1 is the sparse
root; dense IDs are ranked roots starting at 1. Nonzero uint32 input values all
mean foreground (not categorical equal-value components). Valid zero background
is distinct from invalid zero output. Capacity truncates components by dense ID;
background retains validity. The synthetic snake includes a missing connector,
diagonal-only contact, extreme uint32 nodata, and unequal adjacent values.
"""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import skimage
import rasterio
from scipy.ndimage import label, generate_binary_structure
from skimage.measure import label as skimage_label

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/dixie-fire'
VERSIONS = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                skimage=skimage.__version__, rasterio=rasterio.__version__)


def encode_runs(values):
    flat = np.asarray(values).ravel()
    boundaries = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1, len(flat)]
    return np.stack([np.diff(boundaries), flat[boundaries[:-1]]], axis=-1).astype(np.uint64).ravel().tolist()


def make_case(name, values, validity, connectivity, no_data=4294967295):
    valid = (validity != 0) & (values != no_data)
    foreground = (values != 0) & valid
    labels, count = label(foreground, generate_binary_structure(2, 1 if connectivity == 4 else 2))
    other = skimage_label(foreground, connectivity=1 if connectivity == 4 else 2, background=0)
    # Both libraries assign by first encounter; assert full agreement, not just counts.
    np.testing.assert_array_equal(labels, other)
    minimum_members = np.full(count+1, values.size+1, dtype=np.uint32)
    np.minimum.at(minimum_members, labels.ravel(), np.arange(values.size, dtype=np.uint32) + 1)
    minimum_members[0] = 0
    sparse = minimum_members[labels]
    # Explicit root-order adapter, independent of a library's arbitrary label numbering.
    order = np.argsort(minimum_members[1:]) + 1
    rank = np.zeros(count+1, dtype=np.uint32)
    rank[order] = np.arange(1, count+1)
    dense = rank[labels]
    height, width = values.shape
    return dict(name=name, width=width, height=height, connectivity=connectivity, noDataValue=no_data,
                values=encode_runs(values), validity=encode_runs(validity),
                expectedValidity=encode_runs(valid.astype(np.uint32)), sparse=encode_runs(sparse),
                dense=encode_runs(dense), componentCount=int(count),
                foregroundCount=int(foreground.sum()), encoding='rle-count-value',
                reference='scipy.ndimage.label + skimage.measure.label; minimum-member canonicalization')


def write_fixture(suffix, cases, sources):
    fixture = dict(versions=VERSIONS, sources=sources, cases=cases)
    output = HERE / f'fixtures/parity-wp4-components-{suffix}.json'
    output.write_text(json.dumps(fixture, separators=(',', ':')) + '\n')
    assert output.stat().st_size <= 1_500_000, output.stat().st_size
    print(output.name, output.stat().st_size, [(case['name'], case['componentCount']) for case in cases])


sources = []

def read_source(name):
    path = DATA / name
    sources.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if path.suffix == '.bin':
        return np.frombuffer(path.read_bytes(), dtype='<u2').reshape(750, 750).astype(np.float32).astype(float)
    with rasterio.open(path) as dataset:
        return dataset.read(1)


near_before = read_source('B08_before.bin')
short_before = read_source('B12_before.bin')
near_after = read_source('B08_after.bin')
short_after = read_source('B12_after.bin')
classification_before = read_source('scl_before.png')
classification_after = read_source('scl_after.png')
valid = ((near_before > 0) & (short_before > 0) & (near_after > 0) & (short_after > 0) &
         ~np.isin(classification_before, [0, 1, 3, 8, 9, 10, 11]) &
         ~np.isin(classification_after, [0, 1, 3, 8, 9, 10, 11]))
with np.errstate(divide='ignore', invalid='ignore'):
    change = (near_before-short_before)/(near_before+short_before) - (near_after-short_after)/(near_after+short_after)
values = (change > .66).astype(np.uint32)
for connectivity in [4, 8]:
    write_fixture(f'dixie-{connectivity}',
                  [make_case(f'dixie-full-high-severity-{connectivity}', values, valid.astype(np.uint32), connectivity)], sources)

rows, columns = np.indices((35, 67))
snake = ((rows % 2 == 0) | (columns == np.where((rows//2) % 2 == 0, 66, 0))).astype(np.uint32)
snake *= np.where(columns % 2, 7, 4294967294).astype(np.uint32)
snake_validity = np.ones_like(snake)
snake_validity[9, 66] = 0
snake[23, 0] = 4294967295
edge = np.array([[7, 0, 0, 0, 9, 2], [0, 9, 0, 0, 0, 0],
                 [0, 0, 4294967295, 0, 0, 0], [0, 0, 0, 11, 0, 0]], dtype=np.uint32)
edge_validity = np.ones_like(edge); edge_validity[3, 3] = 0
cases = []
for connectivity in [4, 8]:
    cases.append(make_case(f'snake-seams-masked-connectors-{connectivity}', snake, snake_validity, connectivity))
    cases.append(make_case(f'diagonal-unequal-values-nodata-{connectivity}', edge, edge_validity, connectivity))
cases.append(make_case('empty-with-invalid-background', np.zeros((9, 17), dtype=np.uint32),
                       (np.indices((9, 17)).sum(axis=0) % 3 != 0).astype(np.uint32), 4))
cases.append(make_case('all-foreground', np.ones((17, 19), dtype=np.uint32), np.ones((17, 19), dtype=np.uint32), 8))
write_fixture('synthetic', cases, [])
