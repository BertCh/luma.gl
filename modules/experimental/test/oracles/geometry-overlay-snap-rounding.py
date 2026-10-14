#!/usr/bin/env python3
"""Oracle generator for GPUPolygonOverlay on the snap-rounding lattice (shapely 2.2 / GEOS 3.14).

Reference (contract: the GPU overlay snaps every vertex to the lattice first, then nodes the rounded
segments; GEOS nodes the unrounded input, and the two differ by a pixel when input is off the lattice):
`op(set_precision(a, g, mode='pointwise'), set_precision(b, g, mode='pointwise'), grid_size=g)` and
`union_all(set_precision(polygons, g, mode='pointwise'), grid_size=g)`: OverlayNG with a fixed precision
model on input that already lies on the lattice (noding by snap rounding with hot pixels that are
half-open squares). The fixture keeps the RAW inputs (the GPU input) and records
`oracleInputs: 'pointwise-rounded'`. The winding-rule "faces" route snaps vertices pointwise as well.
Cases where GEOS raises on the rounded input are skipped and listed in `skipped`.
The crowded-pixel family's successful GEOS results on invalid rounded operands are retained as
`shapelyExpected`, but its contract oracle is `snap-rounded-faces`: exact Fraction snap rounding followed by polygonization
and per-operand even-odd winding of the rerouted segments. `oracleDifference` records the reason.

Every case is built in pixel units (multiples of 1/16, never a half-integer: no vertex on a pixel
corner) and scaled by `grid_size` in {0.25, 1, 1/64}; all coordinates are exact in float32 (asserted).
Exact crossing ties (a proper crossing at a half-integer pixel coordinate) are rejected while
generating, because GEOS and the GPU's closed/open convention agree everywhere else.

Case families: near-parallel crossings, many crossings inside one pixel, the classic snap-rounding
reroute (a vertex within half a pixel of another edge), slivers that collapse, coincident and partly
overlapping edges of both operands, nested holes and an island, a hole touching its shell, and dissolve.
Self-overlapping input (bow tie, pentagram, doubled ring, same-orientation inner ring) has no shapely
equivalent for the winding rules: its reference is "faces": the linework of the snapped rings is noded by
`unary_union(lines, grid_size=1)` in lattice space, polygonized, each face gets the exact integer winding
number of its representative point, faces inside by the fill rule are united with
`union_all(grid_size=1)`. `method` records which route made each case.

Per case: operation, grid size, lattice `{origin, cell}`, inputs (world coordinates), and `expected`:
`rings` (lattice integers, normalized: no closing vertex, start at the smallest (x, y), exterior CCW and
holes CW, polygons and holes sorted), `corners` (rings with collinear vertices removed), `area2` (twice
the area, exact integer in lattice units squared), `ringCount`, `polygonCount`, `edges` (directed edges of
the corner rings with the result on the left, sorted; maximal straight runs, so comparable to any noding
of the same boundary after merging collinear chains) and `rawEdges` (same for `rings`).

Usage, from the repository root:
  <venv>/bin/python -I modules/experimental/test/oracles/geometry-overlay-snap-rounding.py
Writes test/gpu-spatial-analysis/polygon-overlay/fixtures/overlay-snap-rounding.json.
"""
import json
import math
import os
import random
import sys
from fractions import Fraction
from importlib.metadata import version

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(
    HERE, '..', 'gpu-spatial-analysis', 'polygon-overlay', 'fixtures', 'overlay-snap-rounding.json'
)
GENERATOR = 'modules/experimental/test/oracles/geometry-overlay-snap-rounding.py'
SEED = 20261014
GRIDS = [0.25, 1.0, 1.0 / 64.0]
OPERATIONS = ['intersection', 'union', 'difference', 'symmetric-difference']
SHAPELY_OPERATIONS = {
    'intersection': shapely.intersection,
    'union': shapely.union,
    'difference': shapely.difference,
    'symmetric-difference': shapely.symmetric_difference,
}
RULES = ['even-odd', 'nonzero', 'positive', 'negative']


# ---------------------------------------------------------------- scene description (pixel units)
# A polygon is [shell, hole, ...]; a feature is a list of polygons; an operand is a list of features.

def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def star(cx, cy, inner, outer, spikes, phase):
    points = []
    for i in range(2 * spikes):
        radius = outer if i % 2 == 0 else inner
        angle = phase + math.pi * i / spikes
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def quantize(points):
    """Round to 1/16 and nudge off half-integers so no vertex sits on a pixel corner or edge."""
    result = []
    for x, y in points:
        qx, qy = round(x * 16) / 16, round(y * 16) / 16
        if qx - math.floor(qx) == 0.5:
            qx += 1 / 16
        if qy - math.floor(qy) == 0.5:
            qy += 1 / 16
        result.append((qx, qy))
    return result


def build_templates():
    """name -> (operand A, operand B or None, operations or fill rules)."""
    templates = {}
    templates['near-parallel'] = (
        [[[(2, 2.125), (62, 3.3125), (62, 9.125), (2, 9.125)]]],
        [[[(2, 3.0), (62, 2.625), (62, 12.1875), (2, 12.1875)]]],
        OPERATIONS,
    )
    # Two 12-spike stars whose spikes start inside one pixel: every spike pair crosses near the centre.
    star_a = quantize(star(20.3125, 20.1875, 0.6, 30, 12, 0.0))
    star_b = quantize(star(20.6875, 19.8125, 0.6, 30, 12, math.pi / 24))
    templates['crowded-pixel'] = ([[[star_a]]], [[[star_b]]], OPERATIONS)
    # Classic reroute: B's vertex (20.2, 4.9) lies within half a pixel of A's long slanted edge.
    templates['reroute-vertex'] = (
        [[[(0.125, 0.0625), (40.0625, 9.0625), (0.125, 20.0625)]]],
        [[[(20.1875, 4.9375), (30.0625, 30.0625), (10.0625, 30.0625)]]],
        OPERATIONS,
    )
    templates['reroute-chain'] = (
        [[[(1.0625, 1.0625), (45.0625, 13.0625), (1.0625, 26.0625)]]],
        [[[(10.125, 3.9375), (20.125, 6.5625), (30.125, 9.4375), (30.0625, 40.0625), (10.0625, 40.0625)]]],
        OPERATIONS,
    )
    templates['collapsing-sliver'] = (
        [[[(0.0625, 0.0625), (30.0625, 0.0625), (30.0625, 30.0625), (0.0625, 30.0625)]]],
        [[[(5.0625, 5.0625), (25.0625, 5.4375), (25.0625, 5.1875)],
          ]],
        OPERATIONS,
    )
    templates['thin-wedge'] = (
        [[[(0.0625, 0.0625), (30.0625, 0.0625), (30.0625, 30.0625), (0.0625, 30.0625)]]],
        [[[(-5.0625, 10.0625), (35.0625, 10.3125), (35.0625, 10.1875)]]],
        OPERATIONS,
    )
    templates['coincident-edges'] = (
        [[rect(0.0625, 0.0625, 20.0625, 20.0625)], [rect(40.0625, 0.0625, 60.0625, 20.0625)]],
        [[rect(20.0625, 5.0625, 40.0625, 15.0625)], [rect(10.0625, 0.0625, 30.0625, 10.0625)]],
        OPERATIONS,
    )
    templates['partly-overlapping-edges'] = (
        [[rect(0.0625, 0.0625, 20.0625, 20.0625)]],
        [[rect(10.0625, 0.0625, 30.0625, 12.0625)], [rect(0.0625, 20.0625, 15.0625, 30.0625)]],
        OPERATIONS,
    )
    templates['nested-holes'] = (
        [
            [rect(0.0625, 0.0625, 40.0625, 40.0625), rect(8.0625, 8.0625, 32.0625, 32.0625)],
            [rect(14.0625, 14.0625, 26.0625, 26.0625), rect(18.0625, 18.0625, 22.0625, 22.0625)],
        ],
        [[rect(4.0625, 16.0625, 36.0625, 24.0625)]],
        OPERATIONS,
    )
    templates['hole-touches-shell'] = (
        [[
            [(0.0625, 0.0625), (30.0625, 0.0625), (30.0625, 30.0625), (15.0625, 30.0625), (0.0625, 30.0625)],
            [(10.0625, 10.0625), (20.0625, 10.0625), (15.0625, 30.0625)],
        ]],
        [[quantize(star(15.1, 15.1, 5, 14, 8, 0.1))]],
        OPERATIONS,
    )
    # Dissolve of overlapping and adjacent rectangles, random but seeded.
    rng = random.Random(SEED)
    rects = []
    for _ in range(12):
        x0, y0 = rng.randrange(0, 40) + 0.0625, rng.randrange(0, 40) + 0.0625
        rects.append([rect(x0, y0, x0 + rng.randrange(4, 24), y0 + rng.randrange(4, 24))])
    templates['dissolve-random-rectangles'] = (rects, None, ['dissolve'])
    templates['dissolve-adjacent-squares'] = (
        [[rect(8.0625 * i, 8.0625 * j, 8.0625 * i + 8.0625, 8.0625 * j + 8.0625)] for i in range(3) for j in range(3)],
        None,
        ['dissolve'],
    )
    templates['dissolve-nested-holes'] = (
        [[rect(0.0625, 0.0625, 40.0625, 40.0625), rect(8.0625, 8.0625, 32.0625, 32.0625)],
         [rect(14.0625, 14.0625, 26.0625, 26.0625)],
         [rect(30.0625, 30.0625, 50.0625, 36.0625)]],
        None,
        ['dissolve'],
    )
    # Self-overlapping input, dissolve with a fill rule.
    templates['bow-tie'] = ([[[[(0.0625, 0.0625), (20.0625, 20.0625), (20.0625, 0.0625), (0.0625, 20.0625)]]]], None, RULES)
    pentagram = [(10.0625 + 10 * math.sin(4 * math.pi * i / 5), 10.0625 + 10 * math.cos(4 * math.pi * i / 5)) for i in range(5)]
    templates['pentagram'] = ([[[quantize(pentagram)]]], None, RULES)
    templates['doubled-ring'] = (
        [[[rect(0.0625, 0.0625, 20.0625, 20.0625) + rect(0.0625, 0.0625, 20.0625, 20.0625)]]], None, RULES
    )
    templates['same-orientation-hole'] = (
        [[[rect(0.0625, 0.0625, 30.0625, 30.0625), rect(8.0625, 8.0625, 22.0625, 22.0625)]]], None, RULES
    )
    templates['clockwise-inner-ring'] = (
        [[[rect(0.0625, 0.0625, 30.0625, 30.0625), rect(8.0625, 8.0625, 22.0625, 22.0625)[::-1]]]], None, RULES
    )
    return templates


# ---------------------------------------------------------------- helpers

def to_world(operand, grid):
    """Pixel units -> world coordinates, with the polygon list kept as [feature][polygon][ring]."""
    scaled = []
    for feature in operand:
        polygons = []
        for polygon in feature:
            polygons.append([[(x * grid, y * grid) for x, y in ring] for ring in polygon])
        scaled.append(polygons)
    return scaled


def normalize_feature(feature):
    """Accepts a feature as list of rings (one polygon) or list of polygons; returns list of polygons."""
    if feature and feature[0] and isinstance(feature[0][0][0], (int, float)):
        return [feature]
    return feature


def wrap_operand(operand):
    """Pixel-unit templates list polygons per feature as [[ring, ...]]; normalize to [feature][polygon][ring]."""
    result = []
    for feature in operand:
        result.append(normalize_polygons(feature))
    return result


def normalize_polygons(feature):
    # A feature is either [ring] (one polygon: first item is a ring: list of points) or a list of polygons.
    first = feature[0]
    if isinstance(first[0][0], (int, float)):
        return [feature]
    return feature


def assert_float32_exact(points):
    for x, y in points:
        assert float(np.float32(x)) == x and float(np.float32(y)) == y, (x, y)


def make_shapely(operand, grid=None):
    """Polygons of an operand; with `grid`, each rounded pointwise (vertices only, no repair)."""
    polygons = []
    for feature in operand:
        for polygon in feature:
            geometry = Polygon(polygon[0], polygon[1:])
            if grid is not None:
                geometry = shapely.set_precision(geometry, grid, mode='pointwise')
            polygons.append(geometry)
    return polygons


def lattice_of(operands, grid):
    xs, ys = [], []
    for operand in operands:
        for feature in operand:
            for polygon in feature:
                for ring in polygon:
                    for x, y in ring:
                        xs.append(x)
                        ys.append(y)
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    power = 2.0 ** math.ceil(math.log2(extent)) if extent > 0 else 2.0 ** -126
    cell = 2.0 ** math.ceil(math.log2(max(grid, power * 2.0 ** -22)))
    origin = (math.floor(min(xs) / cell) * cell, math.floor(min(ys) / cell) * cell)
    return {'origin': list(origin), 'cell': cell, 'extent': extent}


def to_lattice(point, lattice):
    ox, oy = lattice['origin']
    kx = Fraction(point[0]) - Fraction(ox)
    ky = Fraction(point[1]) - Fraction(oy)
    cell = Fraction(lattice['cell'])
    assert (kx / cell).denominator == 1 and (ky / cell).denominator == 1, point
    return (int(kx / cell), int(ky / cell))


def signed_area2(ring):
    return sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1] for i in range(len(ring)))


def rotate_to_minimum(ring):
    start = min(range(len(ring)), key=lambda i: ring[i])
    return ring[start:] + ring[:start]


def remove_collinear(ring):
    changed = True
    while changed and len(ring) > 3:
        changed = False
        for i in range(len(ring)):
            a, b, c = ring[i - 1], ring[i], ring[(i + 1) % len(ring)]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
            if cross == 0 and dot > 0:
                ring = ring[:i] + ring[i + 1:]
                changed = True
                break
    return ring


def directed_edges(rings):
    edges = []
    for ring in rings:
        for i in range(len(ring)):
            edges.append([ring[i][0], ring[i][1], ring[(i + 1) % len(ring)][0], ring[(i + 1) % len(ring)][1]])
    return sorted(edges)


def summarize(geometry, lattice):
    """Normalized lattice rings of the polygon parts of a shapely result."""
    polygons = [part for part in shapely.get_parts(geometry) if part.geom_type == 'Polygon'] if not geometry.is_empty else []
    if geometry.geom_type == 'GeometryCollection':
        polygons = [p for part in shapely.get_parts(geometry) for p in shapely.get_parts(part) if p.geom_type == 'Polygon']
    rings, corner_rings = [], []
    polygon_count = 0
    for polygon in sorted(polygons, key=lambda p: p.exterior.coords[0]):
        if polygon.is_empty:
            continue
        polygon_count += 1
        for index, ring in enumerate([polygon.exterior] + list(polygon.interiors)):
            points = [to_lattice(p, lattice) for p in ring.coords[:-1]]
            counter_clockwise = signed_area2(points) > 0
            if counter_clockwise != (index == 0):
                points = points[::-1]
            rings.append(rotate_to_minimum(points))
            corner_rings.append(rotate_to_minimum(remove_collinear(points)))
    order = sorted(range(len(rings)), key=lambda i: (rings[i][0], len(rings[i])))
    rings = [rings[i] for i in order]
    corner_rings = [corner_rings[i] for i in order]
    return {
        'rings': rings,
        'corners': corner_rings,
        'area2': sum(signed_area2(r) for r in rings),
        'ringCount': len(rings),
        'polygonCount': polygon_count,
        'edges': directed_edges(corner_rings),
        'rawEdges': directed_edges(rings),
    }


def has_exact_tie(operands, grid):
    """True when a proper crossing of two input segments lies exactly on a pixel half-integer."""
    segments = []
    for operand in operands:
        for feature in operand:
            for polygon in feature:
                for ring in polygon:
                    for i in range(len(ring)):
                        a, b = ring[i], ring[(i + 1) % len(ring)]
                        if a != b:
                            segments.append(tuple(Fraction(v) / Fraction(grid) for v in (*a, *b)))
    for i in range(len(segments)):
        ax, ay, bx, by = segments[i]
        for j in range(i + 1, len(segments)):
            cx, cy, dx, dy = segments[j]
            denominator = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
            if denominator == 0:
                continue
            t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denominator
            u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denominator
            if 0 < t < 1 and 0 < u < 1:
                x, y = ax + t * (bx - ax), ay + t * (by - ay)
                if (x - math.floor(x)) == Fraction(1, 2) or (y - math.floor(y)) == Fraction(1, 2):
                    return True
    return False


# ---------------------------------------------------------------- fill-rule reference ("faces")

def snap_half_up(value, lattice, axis):
    k = (Fraction(value) - Fraction(lattice['origin'][axis])) / Fraction(lattice['cell'])
    return math.floor(k + Fraction(1, 2))


def winding_number(point, rings):
    total = 0
    px, py = point
    for ring in rings:
        for i in range(len(ring)):
            (ax, ay), (bx, by) = ring[i], ring[(i + 1) % len(ring)]
            side = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            if ay <= py < by and side > 0:
                total += 1
            elif by <= py < ay and side < 0:
                total -= 1
    return total


def inside_by_rule(winding, rule):
    return {'even-odd': winding % 2 != 0, 'nonzero': winding != 0, 'positive': winding > 0, 'negative': winding < 0}[rule]


def faces_reference(operand, lattice, rule):
    """Dissolve under a fill rule, computed in lattice space (see the module docstring)."""
    snapped_polygons = []
    lines = []
    for feature in operand:
        for polygon in feature:
            rings = []
            for ring in polygon:
                points = [(snap_half_up(x, lattice, 0), snap_half_up(y, lattice, 1)) for x, y in ring]
                rings.append(points)
                for i in range(len(points)):
                    a, b = points[i], points[(i + 1) % len(points)]
                    if a != b:
                        lines.append(shapely.LineString([a, b]))
            snapped_polygons.append(rings)
    noded = shapely.unary_union(shapely.MultiLineString(lines), grid_size=1.0)
    faces = list(shapely.get_parts(shapely.polygonize([noded])))
    kept = []
    for face in faces:
        point = shapely.point_on_surface(face)
        probe = (Fraction(point.x), Fraction(point.y))
        # A representative point within a quarter pixel of an input edge could be misclassified.
        distance = min(
            shapely.distance(point, shapely.LineString([ring[i], ring[(i + 1) % len(ring)]]))
            for rings in snapped_polygons for ring in rings for i in range(len(ring)) if ring[i] != ring[(i + 1) % len(ring)]
        )
        assert distance > 0.2, ('face representative too close to an edge', distance)
        if any(inside_by_rule(winding_number(probe, rings), rule) for rings in snapped_polygons):
            kept.append(face)
    return shapely.union_all(kept, grid_size=1.0) if kept else shapely.Polygon()


def lattice_geometry_summary(geometry):
    """Summary of a geometry that already lives in lattice space (identity frame)."""
    identity = {'origin': [0.0, 0.0], 'cell': 1.0}
    return summarize(geometry, identity)


def snap_rounded_faces_reference(operands, lattice, operation):
    """Independent exact rational noder plus GEOS polygonizer; no polygon overlay on invalid inputs."""
    segments = []
    pixels = set()
    for operand_index, operand in enumerate(operands):
        polygon_index = 0
        for feature in operand:
            for polygon in feature:
                for ring in polygon:
                    points = [tuple(snap_half_up(value, lattice, axis) for axis, value in enumerate(point)) for point in ring]
                    for start, end in zip(points, points[1:] + points[:1]):
                        if start != end:
                            segments.append((start, end, operand_index, polygon_index))
                            pixels.update((start, end))
                polygon_index += 1
    def cross(first, second):
        return first[0] * second[1] - first[1] * second[0]
    def subtract(first, second):
        return (first[0] - second[0], first[1] - second[1])
    for first_index, (start, end, *_rest) in enumerate(segments):
        direction = subtract(end, start)
        for other_start, other_end, *_rest in segments[first_index + 1:]:
            other_direction = subtract(other_end, other_start)
            denominator = cross(direction, other_direction)
            if denominator == 0:
                continue
            displacement = subtract(other_start, start)
            parameter = Fraction(cross(displacement, other_direction), denominator)
            other_parameter = Fraction(cross(displacement, direction), denominator)
            if 0 < parameter < 1 and 0 < other_parameter < 1:
                pixels.add(tuple(math.floor(start[axis] + parameter * direction[axis] + Fraction(1, 2)) for axis in range(2)))
    def pixel_parameter(start, end, pixel):
        lower, upper = Fraction(0), Fraction(1)
        lower_closed = upper_closed = True
        for axis in range(2):
            direction = end[axis] - start[axis]
            low = Fraction(2 * pixel[axis] - 1, 2) - start[axis]
            high = low + 1
            if direction == 0:
                if not low <= 0 < high:
                    return None
                continue
            bounds = [(low / direction, True), (high / direction, False)]
            if direction < 0:
                bounds.reverse()
            (candidate_lower, candidate_lower_closed), (candidate_upper, candidate_upper_closed) = bounds
            if candidate_lower > lower:
                lower, lower_closed = candidate_lower, candidate_lower_closed
            elif candidate_lower == lower:
                lower_closed = lower_closed and candidate_lower_closed
            if candidate_upper < upper:
                upper, upper_closed = candidate_upper, candidate_upper_closed
            elif candidate_upper == upper:
                upper_closed = upper_closed and candidate_upper_closed
        if lower > upper or (lower == upper and not (lower_closed and upper_closed)):
            return None
        return (lower + upper) / 2
    pieces = []
    for start, end, operand_index, polygon_index in segments:
        visits = [(pixel_parameter(start, end, pixel), pixel) for pixel in pixels]
        ordered = [pixel for parameter, pixel in sorted((parameter, pixel) for parameter, pixel in visits if parameter is not None)]
        pieces.extend((first, second, operand_index, polygon_index) for first, second in zip(ordered, ordered[1:]))
    def membership(point):
        winding = {}
        for start, end, operand_index, polygon_index in pieces:
            key = (operand_index, polygon_index)
            winding.setdefault(key, 0)
            side = cross(subtract(end, start), subtract(point, start))
            if start[1] <= point[1] < end[1] and side > 0:
                winding[key] += 1
            elif end[1] <= point[1] < start[1] and side < 0:
                winding[key] -= 1
        return [any(value % 2 != 0 for (owner, _polygon), value in winding.items() if owner == operand_index) for operand_index in range(2)]
    linework = shapely.unary_union([shapely.LineString([start, end]) for start, end, *_rest in pieces])
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(linework)))
    kept = []
    for face in faces:
        point = face.representative_point()
        left, right = membership((Fraction(point.x), Fraction(point.y)))
        selected = {'intersection': left and right, 'union': left or right,
                    'difference': left and not right, 'symmetric-difference': left != right}[operation]
        if selected:
            kept.append(face)
    return shapely.union_all(kept), membership


# ---------------------------------------------------------------- generation

def main():
    templates = build_templates()
    cases = []
    skipped = []
    for name, (operand_a, operand_b, modes) in templates.items():
        operand_a = wrap_operand(operand_a)
        operand_b = wrap_operand(operand_b) if operand_b is not None else None
        for grid in GRIDS:
            world_a = to_world(operand_a, grid)
            world_b = to_world(operand_b, grid) if operand_b is not None else None
            operands = [world_a] + ([world_b] if world_b is not None else [])
            for operand in operands:
                for feature in operand:
                    for polygon in feature:
                        for ring in polygon:
                            assert_float32_exact(ring)
            if has_exact_tie(operands, grid):
                print('skip (exact crossing tie):', name, grid, file=sys.stderr)
                continue
            lattice = lattice_of(operands, grid)
            for mode in modes:
                record = {
                    'name': f'{name}@{grid}' + (f'/{mode}' if mode != 'dissolve' else '/dissolve'),
                    'template': name,
                    'gridSize': grid,
                    'lattice': lattice,
                    'left': world_a,
                    'right': world_b,
                }
                if mode in RULES:
                    record['operation'] = 'dissolve'
                    record['fillRule'] = mode
                    record['method'] = 'faces'
                    record['name'] = f'{name}@{grid}/dissolve-{mode}'
                    record['expected'] = lattice_geometry_summary(faces_reference(world_a, lattice, mode))
                elif mode == 'dissolve':
                    record['operation'] = 'dissolve'
                    record['fillRule'] = 'even-odd'
                    record['method'] = 'shapely'
                    try:
                        result = shapely.union_all(make_shapely(world_a, grid), grid_size=grid)
                    except shapely.errors.GEOSException as error:
                        skipped.append({'name': record['name'], 'error': str(error)})
                        continue
                    record['expected'] = summarize(result, lattice)
                else:
                    record['operation'] = mode
                    record['fillRule'] = 'even-odd'
                    record['method'] = 'shapely'
                    try:
                        sides = []
                        for operand in (world_a, world_b):
                            parts = make_shapely(operand, grid)
                            sides.append(parts[0] if len(parts) == 1 else shapely.union_all(parts))
                        result = SHAPELY_OPERATIONS[mode](sides[0], sides[1], grid_size=grid)
                    except shapely.errors.GEOSException as error:
                        skipped.append({'name': record['name'], 'error': str(error)})
                        continue
                    record['expected'] = summarize(result, lattice)
                    if name == 'crowded-pixel':
                        assert mode == 'intersection' and not any(shapely.is_valid(side) for side in sides)
                        reference, membership = snap_rounded_faces_reference(operands, lattice, mode)
                        witness = (Fraction(125, 4), Fraction(59, 2))
                        world_witness = shapely.Point(*(float(value * lattice['cell'] + lattice['origin'][axis]) for axis, value in enumerate(witness)))
                        assert result.contains(world_witness) and membership(witness) == [False, True]
                        record['shapelyExpected'] = record['expected']
                        record['expected'] = lattice_geometry_summary(reference)
                        record['method'] = 'snap-rounded-faces'
                        record['oracleDifference'] = {
                            'reason': 'Pointwise rounding produces invalid self-touching rings with retraced spikes. GEOS polygon overlay has no even-odd guarantee for invalid operands; classify the exact snap-rounded linework instead.',
                            'roundedInputValidity': [str(shapely.is_valid_reason(side)) for side in sides],
                            'witness': {'latticePoint': list(map(float, witness)), 'insideOperands': membership(witness), 'insideShapelyIntersection': bool(result.contains(world_witness))},
                        }
                record['oracleInputs'] = 'pointwise-rounded'
                cases.append(record)
    payload = {
        'generator': GENERATOR,
        'seed': SEED,
        'versions': {
            'shapely': version('shapely'),
            'geos': '.'.join(map(str, shapely.geos_version)),
            'numpy': version('numpy'),
            'python': sys.version.split()[0],
        },
        'gridSizes': GRIDS,
        'skipped': skipped,
        'cases': cases,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as handle:
        json.dump(payload, handle, separators=(',', ':'))
    print('skipped:', skipped)
    print(f'wrote {len(cases)} cases to {os.path.normpath(OUTPUT)}')


if __name__ == '__main__':
    main()
