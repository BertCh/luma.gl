# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Full-raster shortest costs and paths from skimage.graph.MCP_Geometric.

Run <reference-venv>/bin/python -I this-file. Fully connected D8, edge length
(hypot of anisotropic spacing) times mean endpoint friction, source cost zero,
NaN/negative friction -> barriers, diagonal corner cutting allowed. Every cell
is compared, including infinity/unreachable masks. The complete minimizer set
of the independent MCP cost surface is encoded as a D8 bitmask, not one arbitrary
skimage traceback. Each GPU backlink must be in that set and every walk must
reach a source without a cycle; all full walk costs are compared to MCP.
MCP tracebacks and route_through_array also supply selected complete paths.
Zero-cost plateaus deliberately allow alternative forests, never cycles.

Decode the FULL Grand Canyon Terrarium raster, derive a scalar friction from
numpy.gradient on elevations (full raster, so the crop uses real halo values),
then solve the fixed 128x128 window rows 960:1088, columns 960:1088 as its own
bounded domain. This is not a full-DEM global cost-distance claim. Quantize
friction to 1+round(10*slope), an explicit test convention. Use an anisotropic
12x16 metre grid: its D8 edge lengths are 12,16,20 and the integer friction
makes every cost exactly representable below 2**24. This intentionally removes
ambiguous f32 near-ties from the realistic surface while retaining true ties.
The synthetic unit-grid case separately checks irrational sqrt(2) diagonals;
other cases exercise barriers, NaN/negative friction, two sources, and zeros.
All inputs/spacing round to f32 before f64 MCP. No random inputs or source costs.

https://scikit-image.org/docs/stable/api/skimage.graph.html#skimage.graph.MCP_Geometric
Profile L: elementwise atol=1e-6*scale/rtol=1e-4 and scaled residual <=1e-4.
Unreachable masks, convergence, predecessor membership and path structure exact.
"""
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import rasterio
import skimage
from skimage.graph import MCP_Geometric, route_through_array

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/grand-canyon-dem/dem.png'
OFFSETS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 'Infinity' if value > 0 else '-Infinity'
        return float(value)
    return value


def make_case(name, friction, spacing, sources, targets):
    friction = np.asarray(friction, dtype=np.float32).astype(float)
    spacing = np.asarray(spacing, dtype=np.float32).astype(float)
    height, width = friction.shape
    costs_input = np.where(np.isfinite(friction) & (friction >= 0), friction, np.inf)
    solver = MCP_Geometric(costs_input, fully_connected=True, sampling=tuple(spacing[::-1]))
    costs, _ = solver.find_costs([divmod(source, width) for source in sources])
    predecessor_masks = np.zeros((height, width), dtype=np.uint32)
    for row, column in np.ndindex(costs.shape):
        if not np.isfinite(costs[row, column]) or row * width + column in sources:
            continue
        for direction, (row_offset, column_offset) in enumerate(OFFSETS):
            other_row, other_column = row + row_offset, column + column_offset
            if 0 <= other_row < height and 0 <= other_column < width:
                length = np.hypot(row_offset * spacing[1], column_offset * spacing[0])
                candidate = costs[other_row, other_column] + length * (costs_input[row, column] + costs_input[other_row, other_column]) / 2
                # Only f64 evaluation roundoff, never an f32/global topology slack.
                if abs(candidate - costs[row, column]) <= 64 * np.finfo(float).eps * max(1, costs[row, column]):
                    predecessor_masks[row, column] |= 1 << direction
        assert predecessor_masks[row, column] != 0
    paths = []
    for target in targets:
        row, column = divmod(target, width)
        indices = [] if not np.isfinite(costs[row, column]) else [point[0] * width + point[1] for point in solver.traceback((row, column))][::-1]
        unique = bool(indices) and all(int(predecessor_masks.flat[index]).bit_count() == 1 for index in indices[:-1])
        paths.append(dict(target=target, indices=indices, cost=costs[row, column], unique=unique))
        if len(sources) == 1 and spacing.tolist() == [1., 1.] and indices:
            route, route_cost = route_through_array(costs_input, divmod(sources[0], width), (row, column), fully_connected=True, geometric=True)
            np.testing.assert_allclose(route_cost, costs[row, column], rtol=1e-14)
            assert len(route) >= 1
    return dict(name=name, width=width, height=height, spacing=spacing,
                friction=friction.ravel(), sources=sources, costs=costs.ravel(),
                unreachable=(~np.isfinite(costs)).astype(int).ravel(),
                predecessorMasks=predecessor_masks.ravel(), paths=paths)


with rasterio.open(SOURCE) as dataset:
    pixels = dataset.read()[:3].astype(float)
elevations = pixels[0] * 256 + pixels[1] + pixels[2] / 256 - 32768
row_gradient, column_gradient = np.gradient(elevations, 15.440107977971211)
friction = 1 + np.rint(10 * np.hypot(row_gradient, column_gradient))
window = friction[960:1088, 960:1088]
cases = [make_case('grand-canyon-window128-integer-friction', window, [12, 16], [0, 127], [16383, 16256, 8256])]
assert np.max(cases[0]['costs']) < 2 ** 24
cases.append(make_case('unit-diagonals', np.ones((9, 11)), [1, 1], [0], [98, 55, 10]))
barriers = np.ones((7, 9))
barriers[:, 5] = np.nan
barriers[2, 2] = -1
barriers[0, 1] = np.nan
barriers[1, 0] = np.nan  # the source still escapes diagonally through (1,1)
barriers[4, 1:4] = 3
cases.append(make_case('barriers-corner-cutting-multiple-sources', barriers, [3, 4], [0, 54], [40, 62, 1]))
plateau = np.ones((5, 7))
plateau[1:4, 1:6] = 0
cases.append(make_case('zero-cost-plateau', plateau, [3, 4], [16], [34, 8, 18]))
cases.append(make_case('unique-corridor', np.arange(1, 18)[None, :], [3, 4], [0], [16, 8, 0]))
fixture = dict(versions=dict(python=platform.python_version(), numpy=np.__version__, rasterio=rasterio.__version__, skimage=skimage.__version__),
    source=dict(path=str(SOURCE.relative_to(ROOT)), sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                originalShape=[2048, 2048], window=[960, 1088, 960, 1088],
                sourceCRS='EPSG:3857', crs='local planar metres; explicit 12x16 spacing',
                friction='1+round(10*hypot(numpy.gradient(elevation,15.440107977971211)))',
                nodata='NaN/negative -> impassable', inputPrecision='float32'),
    reference='skimage.graph.MCP_Geometric / route_through_array',
    tolerance=dict(profile='L', absoluteFactor=1e-6, relative=1e-4, scaledResidual=1e-4), cases=cases)
output = HERE / 'fixtures/parity-wp4-cost-distance.json'
output.write_text(json.dumps(encode(fixture), separators=(',', ':'), allow_nan=False) + '\n')
assert output.stat().st_size <= 1_500_000
print(f'{output.name}: {output.stat().st_size} bytes; {len(cases)} cases')
