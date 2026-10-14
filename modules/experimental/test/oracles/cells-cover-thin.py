#!/usr/bin/env python3
"""Oracle generator for GPUCellCover on thin, sparse and large polygons (scanline enumeration).

Reference: `h3.h3shape_to_cells` (alias `h3.polygon_to_cells`) of h3-py 4.x, which includes a cell
when its center is inside the polygon (planar in latitude/longitude, even-odd over shell and holes;
a multipolygon is the union of its polygons). That is the containment GPUCellCover uses for H3.

Cases (resolutions 5, 7 and 9): a thin diagonal strip, a river-like random-walk polyline buffered by
shapely, a comb with many teeth, a polygon with holes, a multipolygon with far-apart parts, and a
strip along a row of the H3 latitude-limit band. Every vertex is rounded to float32 first (the GPU
input), so h3-py and the GPU see the same polygon.

Writes fixtures/cell-cover-thin.json next to the cell cover specs, consumed by
test/gpu-spatial-analysis/cell-cover/gpu-cell-cover-scanline.spec.ts.

Tolerance (stored): the GPU forms cell centers in float32 (ulp up to 1.5e-5 degrees at longitude 180,
7.6e-6 below 128), so a cell whose center is within `centerMargin` = 1e-4 degrees (about 7 ulp) of
the polygon boundary may differ from h3-py. The fixture lists those cells per case as `ambiguous`
(centers of cells found by h3 within a buffer of the polygon, nearer to the boundary than the
margin); the spec requires the GPU result to equal `cells` outside them. Cells farther than the
margin must match exactly, in both directions. `boundsCandidates` is the lattice point count of the
bounding-box enumeration (same spacing constants as the kernel, see cell-cover-h3-constants.ts),
used by the spec to size `candidateCapacity`.

Usage, from the repository root:
  <venv>/bin/python -I modules/experimental/test/oracles/cells-cover-thin.py
Needs numpy, shapely >= 2.0 and h3 >= 4 (versions are recorded in the fixture).
"""
import json
import math
import os
import sys
from importlib.metadata import version

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, '..', 'gpu-spatial-analysis', 'cell-cover', 'fixtures', 'cell-cover-thin.json')
GENERATOR = 'modules/experimental/test/oracles/cells-cover-thin.py'
SEED = 20261014
CENTER_MARGIN = 1e-4
# Half the minimum inradius per resolution (cell-cover-h3-constants.ts, CELL_COVER_H3_MINIMUM_INRADIUS_DEGREES).
LATTICE_SPACING = {5: 0.5 * 0.046, 7: 0.5 * 0.0066, 9: 0.5 * 0.00094}
MAXIMUM_LATITUDE = 89.0


def f32(values):
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def ring_coordinates(ring):
    """Open float32 ring [[lng, lat], ...] of a shapely ring (closing vertex dropped)."""
    return f32(np.array(ring.coords)[:-1]).tolist()


def polygon_record(polygon):
    return {
        'shell': ring_coordinates(polygon.exterior),
        'holes': [ring_coordinates(hole) for hole in polygon.interiors],
    }


def to_f32_polygon(polygon):
    from shapely.geometry import Polygon
    shell = f32(np.array(polygon.exterior.coords)[:-1])
    holes = [f32(np.array(hole.coords)[:-1]) for hole in polygon.interiors]
    return Polygon(shell, holes)


def lattice_count(polygons, resolution):
    """Lattice points of the bounding-box enumeration (kernel formulas, float64)."""
    spacing = LATTICE_SPACING[resolution]
    xs = [x for polygon in polygons for x, _ in polygon.exterior.coords]
    ys = [y for polygon in polygons for _, y in polygon.exterior.coords]
    origin_lat = max(min(ys) - spacing, -MAXIMUM_LATITUDE)
    end_lat = min(max(ys) + spacing, MAXIMUM_LATITUDE)
    equatorward = 0.0 if origin_lat <= 0 <= end_lat else min(abs(origin_lat), abs(end_lat))
    spacing_lng = spacing / math.cos(math.radians(equatorward))
    width = math.ceil((max(xs) + spacing_lng - (min(xs) - spacing_lng)) / spacing_lng) + 1
    height = math.ceil((end_lat - origin_lat) / spacing) + 1
    return width * height


def to_h3_polygon(polygon):
    import h3
    return h3.LatLngPoly(
        [(lat, lng) for lng, lat in ring_coordinates(polygon.exterior)],
        *[[(lat, lng) for lng, lat in ring_coordinates(hole)] for hole in polygon.interiors])


def cover_case(name, resolution, polygons):
    """h3 cells of the union of float32 polygons, and the cells too close to the boundary to compare."""
    import h3
    from shapely.geometry import Point
    from shapely.ops import unary_union
    polygons = [to_f32_polygon(polygon) for polygon in polygons]
    shape = (h3.LatLngMultiPoly(*[to_h3_polygon(polygon) for polygon in polygons])
             if len(polygons) > 1 else to_h3_polygon(polygons[0]))
    cells = set(h3.h3shape_to_cells(shape, resolution))
    union = unary_union(polygons)
    boundary = union.boundary
    # Cells that could be affected by a float32 center: centers of cells h3 finds within a few cell
    # sizes of the polygon. Growing the polygon by a cell edge catches every center within the margin.
    edge = h3.average_hexagon_edge_length(resolution, unit='m') / 111_000.0
    grown = union.buffer(2.0 * edge)
    candidates = set()
    for polygon in ([grown] if grown.geom_type == 'Polygon' else list(grown.geoms)):
        candidates |= set(h3.h3shape_to_cells(to_h3_polygon(to_f32_polygon(polygon)), resolution))
    ambiguous = sorted(
        (cell for cell in candidates
         if Point(h3.cell_to_latlng(cell)[1], h3.cell_to_latlng(cell)[0]).distance(boundary) < CENTER_MARGIN),
        key=lambda cell: int(cell, 16))
    return {
        'name': name,
        'resolution': resolution,
        'polygons': [polygon_record(polygon) for polygon in polygons],
        'cells': sorted(cells, key=lambda cell: int(cell, 16)),
        'ambiguous': ambiguous,
        'boundsCandidates': lattice_count(polygons, resolution),
        'pentagonCells': sum(1 for cell in cells if h3.is_pentagon(cell)),
    }


def build_cases():
    from shapely.affinity import translate
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union
    rng = np.random.default_rng(SEED)
    cases = []
    # Per resolution: (cell edge scale in degrees, strip length in degrees). Lengths keep the bounding
    # box lattice below about one million points so the GPU spec stays quick.
    scale = {5: (0.08, 8.0), 7: (0.012, 2.0), 9: (0.0017, 0.4)}
    for resolution, (edge, length) in scale.items():
        origin = (12.3 + resolution * 0.1, 41.2 + resolution * 0.07)
        # Thin diagonal strip, about two cell edges wide.
        strip = LineString([origin, (origin[0] + length, origin[1] + length)]).buffer(
            edge, cap_style='flat')
        cases.append(cover_case('thin diagonal strip', resolution, [strip]))
        # River: a damped random walk, buffered (rounded joins give many vertices).
        steps = rng.normal(0.0, 0.5, size=(60, 2)) * length / 20.0 + np.array([0.6, 0.35]) * length / 40.0
        walk = np.cumsum(steps, axis=0) + np.array(origin) + np.array([-8.0, -9.0])
        river = LineString(walk).buffer(edge * 1.5, join_style='round', resolution=2).simplify(edge / 20.0)
        cases.append(cover_case('river', resolution, [river]))
        # Comb: a bar with 14 teeth, tooth width about two cell edges, pitch six.
        pitch = 6.0 * edge
        height = min(length, 14.0 * pitch)
        teeth = [box(index * pitch, 0.18 * height, index * pitch + 2.0 * edge, height) for index in range(14)]
        bar = box(0.0, 0.0, 13 * pitch + 2.0 * edge, 0.18 * height)
        comb = translate(unary_union([bar, *teeth]), -70.0, 30.0)
        cases.append(cover_case('comb', resolution, [comb]))
        # Large polygon with two lakes, one with an island: a multipolygon whose first part has holes.
        width = min(length, 40.0 * edge)
        outer = box(0.0, 0.0, width, width)
        lakes = [box(0.2 * width, 0.2 * width, 0.45 * width, 0.5 * width),
                 box(0.55 * width, 0.4 * width, 0.9 * width, 0.8 * width)]
        island = box(0.28 * width, 0.28 * width, 0.36 * width, 0.4 * width)
        holed = outer.difference(unary_union(lakes)).union(island)
        parts = list(holed.geoms) if holed.geom_type == 'MultiPolygon' else [holed]
        cases.append(cover_case('polygon with holes and an island', resolution,
                                [translate(part, 100.0, -40.0) for part in parts]))
        # Multipolygon: four far-apart parts (one thin strip among small blocks). The spread shrinks
        # with the resolution so the bounding box has far more cells than the bounding-box
        # enumeration can hold, while the row count stays within the default scanline capacity.
        k = {5: 60.0, 7: 12.0, 9: 5.0}[resolution] / 60.0
        parts = [
            box(-20.0 * k, 10.0 * k, -20.0 * k + 6 * edge, 10.0 * k + 4 * edge),
            box(30.0 * k, -12.0 * k, 30.0 * k + 8 * edge, -12.0 * k + 3 * edge),
            LineString([(55.0 * k, 20.0 * k), (55.0 * k + 20 * edge, 20.0 * k + 12 * edge)]).buffer(
                edge, cap_style='flat'),
            box(-60.0 * k, -30.0 * k, -60.0 * k + 5 * edge, -30.0 * k + 5 * edge),
        ]
        cases.append(cover_case('multipolygon far apart', resolution, parts))
    # Strip inside the H3 latitude-limit band (the lattice is clamped to 89 degrees).
    for resolution in (5, 7):
        edge = scale[resolution][0]
        polar = LineString([(5.0, 88.2), (9.0, 88.7)]).buffer(edge, cap_style='flat')
        cases.append(cover_case('strip near the latitude limit', resolution, [polar]))
    return cases


def main():
    import h3
    import shapely
    cases = build_cases()
    fixture = {
        'generator': GENERATOR,
        'seed': SEED,
        'versions': {
            'python': sys.version.split()[0],
            'numpy': version('numpy'),
            'shapely': version('shapely'),
            'geos': '.'.join(str(part) for part in shapely.geos_version),
            'h3': h3.__version__,
        },
        'oracle': 'h3.h3shape_to_cells (cell center inside the polygon; multipolygon = union of its polygons)',
        'tolerances': {
            'cells': 'exact set equality per case, except the cells listed as ambiguous',
            'centerMargin': CENTER_MARGIN,
            'derivation': (
                'the GPU forms cell centers in float32 (ulp up to 1.5e-5 degrees at longitude 180, 7.6e-6 below '
                '128), so a center within 1e-4 degrees (about 7 ulp) of the polygon boundary may fall on the other '
                'side. Those cells are listed as ambiguous; every other cell must match h3 in both directions. '
                'h3 sees the same float32 vertices as the GPU.'),
        },
        'cases': cases,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as handle:
        json.dump(fixture, handle, separators=(',', ':'))
        handle.write('\n')
    print('wrote', os.path.relpath(OUTPUT), os.path.getsize(OUTPUT), 'bytes')
    for case in cases:
        print(f"{case['name']:32s} res {case['resolution']} cells {len(case['cells']):6d} "
              f"ambiguous {len(case['ambiguous']):4d} bounds {case['boundsCandidates']}")


if __name__ == '__main__':
    main()
