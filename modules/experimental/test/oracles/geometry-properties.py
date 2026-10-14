#!/usr/bin/env python3
"""Seeded property-fixture generator for the luma.gl geometry and cell contributors.

Every fixture is produced from a fixed seed with shapely, scipy, numpy and h3-py; the fixtures record
the package versions, the seed and the tolerances (with their derivation) so a GPU spec can compare
against truth without re-deriving it. Nothing here reads the current output of a GPU kernel.

Fixtures written (relative to modules/experimental/test/gpu-spatial-analysis):
  cell-aggregation/fixtures/cell-rollup.json         GPUCellRollup vs H3 parents (h3 cell_to_parent)
  geometry-topology/fixtures/line-noding.json        GPULineNoding vs shapely.node / unary_union
  geometry-topology/fixtures/make-valid.json         GPUMakeValid vs shapely.make_valid(method='linework')
  delaunay-tessellation/fixtures/delaunay-properties.json   GPUDelaunayTessellation vs scipy.spatial
  polygon-buffer/fixtures/buffer-properties.json     GPUBufferSurface vs shapely.buffer
  cell-cover/fixtures/cell-cover-h3.json             GPUCellCover (H3, center) vs h3.polygon_to_cells
  trajectory-zones/fixtures/zone-events-grid.json    GPUZoneEvents vs an exact perturbed-point rule

Usage, from the repository root (every fixture, or the named ones):
  python3 -I modules/experimental/test/oracles/geometry-properties.py [rollup noding makevalid delaunay buffer cover zones]

Needs numpy, scipy, shapely >= 2.0 and h3 >= 4 (versions are recorded in each fixture).
"""
import json
import math
import os
import sys
from fractions import Fraction
from importlib.metadata import version

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_ROOT = os.path.join(HERE, '..', 'gpu-spatial-analysis')
GENERATOR = 'modules/experimental/test/oracles/geometry-properties.py'
SEED = 20261014
F32 = np.float32


def f32(values):
    """Rounds to float32 and returns python floats (exact in float64)."""
    return np.asarray(values, dtype=F32).astype(np.float64)


def versions():
    import shapely
    return {
        'python': sys.version.split()[0],
        'numpy': version('numpy'),
        'scipy': version('scipy'),
        'shapely': version('shapely'),
        'geos': '.'.join(str(part) for part in shapely.geos_version),
        'h3': version('h3'),
    }


def write_fixture(directory, name, payload):
    folder = os.path.join(TEST_ROOT, directory, 'fixtures')
    os.makedirs(folder, exist_ok=True)
    payload = {'generator': GENERATOR, 'seed': SEED, 'versions': versions(), **payload}
    path = os.path.join(folder, name)
    with open(path, 'w') as handle:
        json.dump(payload, handle, separators=(',', ':'))
        handle.write('\n')
    print('wrote', os.path.relpath(path, os.path.join(HERE, '..', '..', '..', '..')))


def make_rng(label):
    """Independent stream per fixture so adding a section does not change the others."""
    return np.random.default_rng([SEED, sum(ord(ch) * 31 ** i for i, ch in enumerate(label)) % (2 ** 31)])


# ---------------------------------------------------------------------------------------------
# GPUCellRollup
# ---------------------------------------------------------------------------------------------

def generate_rollup():
    import h3
    rng = make_rng('rollup')
    source_resolution = 8
    centers = [h3.latlng_to_cell(47.2, 8.5, source_resolution),
               h3.latlng_to_cell(-33.9, 150.3, source_resolution),
               h3.latlng_to_cell(70.2, 20.3, source_resolution),
               h3.get_pentagons(source_resolution)[0]]
    cells = set()
    for center in centers:
        for cell in h3.grid_disk(center, 7):
            if rng.random() < 0.6:
                cells.add(cell)
    ordered = sorted(cells, key=lambda cell: int(cell, 16))
    rows = []
    for cell in ordered:
        count = int(rng.integers(1, 60))
        # Fixed-point sums at scale 65536; magnitudes up to 2^40 so sums cross the 32-bit word boundary.
        total = int(rng.integers(-2 ** 40, 2 ** 40))
        low, high = sorted(f32(rng.normal(0, 500, 2)))
        rows.append({'cell': cell, 'count': count, 'sum': total, 'minimum': float(low), 'maximum': float(high)})
    cases = []
    for resolution in (6, 3):
        groups = {}
        for row in rows:
            parent = h3.cell_to_parent(row['cell'], resolution)
            group = groups.setdefault(parent, {'cell': parent, 'count': 0, 'sum': 0,
                                               'minimum': math.inf, 'maximum': -math.inf})
            group['count'] += row['count']
            group['sum'] += row['sum']
            group['minimum'] = min(group['minimum'], row['minimum'])
            group['maximum'] = max(group['maximum'], row['maximum'])
        parents = sorted(groups.values(), key=lambda group: int(group['cell'], 16))
        assert all(abs(group['sum']) < 2 ** 62 for group in parents)
        cases.append({'resolution': resolution, 'parents': [
            {**group, 'sum': str(group['sum'])} for group in parents]})
    write_fixture('cell-aggregation', 'cell-rollup.json', {
        'oracle': 'h3.cell_to_parent; counts, fixed-point sums and extremes are folded per parent in python ints/floats',
        'sumScale': 65536,
        'sourceResolution': source_resolution,
        'pentagonSourceCells': sum(1 for row in rows if h3.is_pentagon(row['cell'])),
        'tolerances': {
            'counts': 'exact', 'sums': 'exact (int64)', 'extremes': 'exact (float32 values)',
            'sumValues': 'relative 1e-6 + 1e-6 absolute against sum / sumScale (float32 conversion)',
        },
        'source': [{**row, 'sum': str(row['sum'])} for row in rows],
        'cases': cases,
    })


# ---------------------------------------------------------------------------------------------
# GPULineNoding
# ---------------------------------------------------------------------------------------------

def segment_angle(first, second):
    a = np.array(first[1]) - np.array(first[0])
    b = np.array(second[1]) - np.array(second[0])
    sine = abs(a[0] * b[1] - a[1] * b[0]) / (np.hypot(*a) * np.hypot(*b))
    return math.asin(min(1.0, sine))


def noding_case(rng, line_count):
    import shapely
    from shapely.geometry import LineString, MultiLineString
    for _ in range(2000):
        lines = []
        for _ in range(line_count):
            vertex_count = int(rng.integers(2, 6))
            lines.append([tuple(point) for point in f32(rng.uniform(0, 100, (vertex_count, 2)))])
        segments = [(line[i], line[i + 1]) for line in lines for i in range(len(line) - 1)]
        # Reject near-parallel crossings (the f32 crossing error grows as 1 / sin(angle)).
        crossing_count = 0
        crossings = []
        bad = False
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                if set(segments[i]) & set(segments[j]):
                    # Consecutive segments of one line share a vertex; that is not a crossing.
                    continue
                a = LineString(segments[i])
                b = LineString(segments[j])
                if not a.intersects(b):
                    continue
                shared = a.intersection(b)
                if shared.geom_type != 'Point' or segment_angle(segments[i], segments[j]) < math.radians(20):
                    bad = True
                    break
                crossing_count += 1
                crossings.append(shared.coords[0])
            if bad:
                break
        if bad or crossing_count < 3:
            continue
        noded = shapely.node(MultiLineString(lines))
        union = shapely.unary_union(MultiLineString(lines))
        pieces = list(noded.geoms)
        noded_segments = [(piece.coords[i], piece.coords[i + 1])
                          for piece in pieces for i in range(len(piece.coords) - 1)]
        union_segments = [(piece.coords[i], piece.coords[i + 1])
                          for piece in union.geoms for i in range(len(piece.coords) - 1)]
        # Count identity for linework in general position: every crossing splits two segments.
        assert len(noded_segments) == len(segments) + 2 * crossing_count, (len(noded_segments), len(segments), crossing_count)
        assert len(union_segments) == len(noded_segments)
        vertices = sorted({point for pair in noded_segments for point in pair})
        distinct = np.array(vertices)
        spread = min(np.hypot(*(distinct[i] - distinct[j])) for i in range(len(distinct))
                     for j in range(i + 1, len(distinct)))
        if spread < 0.05:
            continue
        return lines, noded_segments, crossing_count, spread
    raise RuntimeError('no noding case found')


def generate_noding():
    rng = make_rng('noding')
    cases = []
    for line_count in (4, 6, 8, 10):
        lines, segments, crossing_count, spread = noding_case(rng, line_count)
        cases.append({
            'lines': [[list(point) for point in line] for line in lines],
            'inputSegmentCount': sum(len(line) - 1 for line in lines),
            'crossingCount': crossing_count,
            'nodedSegmentCount': len(segments),
            'minimumVertexSpacing': spread,
            'segments': [[list(a), list(b)] for a, b in segments],
        })
    write_fixture('geometry-topology', 'line-noding.json', {
        'oracle': 'shapely.node(MultiLineString) segments (consecutive vertex pairs); unary_union segment count cross-checked',
        'tolerances': {
            'counts': 'exact: noded segment count = input segments + 2 * crossings (general position, no overlaps)',
            'endpointDistance': 1e-4,
            'derivation': (
                'coordinates in [0, 100], float32 ulp there is 2^-17 = 7.6e-6. A crossing is computed in float32 '
                'with an error of at most about 3 ulp, amplified by 1 / sin(angle) <= 1 / sin(20 deg) = 2.9 '
                '(crossings below 20 degrees are rejected): 3 * 7.6e-6 * 2.9 = 6.6e-5 < 1e-4. Input vertices '
                'are float32 and must match exactly (same tolerance applies). Noded vertices are at least '
                '0.05 apart so the greedy match is unambiguous.'),
        },
        'cases': cases,
    })


# ---------------------------------------------------------------------------------------------
# GPUMakeValid
# ---------------------------------------------------------------------------------------------

MAKE_VALID_DELTA = 2e-5
MAKE_VALID_MARGIN = 1e-4


def ring_list(polygon):
    rings = [list(polygon.exterior.coords)]
    rings += [list(interior.coords) for interior in polygon.interiors]
    return rings


def minimum_crossing_angle(rings):
    from shapely.geometry import LineString
    segments = [(ring[i], ring[i + 1]) for ring in rings for i in range(len(ring) - 1)]
    smallest = math.pi
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            a, b = LineString(segments[i]), LineString(segments[j])
            if a.intersects(b) and not (set(segments[i]) & set(segments[j])):
                smallest = min(smallest, segment_angle(segments[i], segments[j]))
    return smallest


def make_valid_case(rng, kind):
    import shapely
    from shapely.geometry import Point, Polygon
    for _ in range(5000):
        if kind == 'star':
            count = int(rng.integers(7, 12))
            points = f32(rng.uniform(0, 10, (count, 2)))
            rings = [[tuple(p) for p in points] + [tuple(points[0])]]
        else:
            # Convex-ish shell and a hole ring that crosses it.
            angles = np.sort(rng.uniform(0, 2 * math.pi, 9))
            radii = rng.uniform(3.5, 4.5, 9)
            shell = f32(np.column_stack([5 + radii * np.cos(angles), 5 + radii * np.sin(angles)]))
            anchor = shell[int(rng.integers(0, 9))]
            hole_angles = np.sort(rng.uniform(0, 2 * math.pi, 6))
            hole_radii = rng.uniform(1.2, 2.2, 6)
            hole = f32(np.column_stack([anchor[0] + hole_radii * np.cos(hole_angles),
                                        anchor[1] + hole_radii * np.sin(hole_angles)]))
            rings = [[tuple(p) for p in shell] + [tuple(shell[0])], [tuple(p) for p in hole] + [tuple(hole[0])]]
        if minimum_crossing_angle(rings) < 0.2:
            continue
        source = Polygon(rings[0], rings[1:])
        if source.is_valid:
            continue
        repaired = shapely.make_valid(source, method='linework')
        parts = list(repaired.geoms) if hasattr(repaired, 'geoms') else [repaired]
        if any(part.geom_type != 'Polygon' for part in parts):
            continue
        if any(part.area < 0.05 for part in parts):
            continue
        return rings, repaired, parts
    raise RuntimeError('no make-valid case found')


def generate_make_valid():
    import shapely
    from shapely.geometry import Point
    rng = make_rng('makevalid')
    cases = []
    for index in range(10):
        kind = 'star' if index < 6 else 'hole-crossing'
        rings, repaired, parts = make_valid_case(rng, kind)
        boundary = repaired.boundary
        samples = []
        for point in rng.uniform(-0.5, 10.5, (3000, 2)):
            geometry = Point(point)
            if geometry.distance(boundary) < MAKE_VALID_MARGIN:
                continue
            samples.append([float(point[0]), float(point[1]), 1 if repaired.contains(geometry) else 0])
        samples = samples[:800]
        cases.append({
            'kind': kind,
            'rings': [[list(p) for p in ring] for ring in rings],
            'area': repaired.area,
            'polygonCount': len(parts),
            'holeCount': sum(len(part.interiors) for part in parts),
            'boundaryLength': boundary.length,
            'areaTolerance': MAKE_VALID_DELTA * boundary.length,
            'samples': samples,
        })
    write_fixture('geometry-topology', 'make-valid.json', {
        'oracle': "shapely.make_valid(Polygon(shell, holes), method='linework')",
        'tolerances': {
            'polygonCount': 'exact',
            'area': 'absolute, areaTolerance = delta * boundaryLength of the repaired geometry',
            'membership': (
                'sample points farther than margin from the repaired boundary must have the same inside state; '
                'a symmetric difference of area S would contain about S / bboxArea of the samples, and all '
                'mismatching samples would lie within delta of the boundary, so zero mismatches certify '
                'symmetric difference <= delta * boundaryLength up to sampling'),
            'delta': MAKE_VALID_DELTA,
            'margin': MAKE_VALID_MARGIN,
            'derivation': (
                'coordinates in [0, 11], float32 ulp <= 2^-20 = 9.5e-7. Crossing points have error <= about 4 ulp '
                'amplified by 1 / sin(0.2 rad) = 5.0 (crossings below 0.2 rad are rejected): 1.9e-5 <= delta = '
                '2e-5. The area error is at most delta times the boundary length; margin = 5 delta.'),
        },
        'cases': cases,
    })


# ---------------------------------------------------------------------------------------------
# GPUDelaunayTessellation
# ---------------------------------------------------------------------------------------------

def exact_orient(a, b, c):
    a, b, c = (tuple(Fraction(float(v)) for v in p) for p in (a, b, c))
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def exact_incircle(a, b, c, d):
    rows = []
    for p in (a, b, c):
        dx, dy = Fraction(float(p[0])) - Fraction(float(d[0])), Fraction(float(p[1])) - Fraction(float(d[1]))
        rows.append((dx, dy, dx * dx + dy * dy))
    (ax, ay, aw), (bx, by, bw), (cx, cy, cw) = rows
    return aw * (bx * cy - cx * by) - bw * (ax * cy - cx * ay) + cw * (ax * by - bx * ay)


def delaunay_set(name, points):
    from scipy.spatial import ConvexHull, Delaunay
    unique = []
    seen = set()
    for point in points:
        key = (float(point[0]), float(point[1]))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    array = np.array(unique, dtype=np.float64)
    count = len(array)
    tri = Delaunay(array)
    hull = ConvexHull(array)
    triangles = [sorted(int(i) for i in simplex) for simplex in tri.simplices]
    hull_count = len(hull.vertices)
    expected = 2 * count - 2 - hull_count
    coplanar = len(tri.coplanar)
    # Strict general position of scipy's own answer: every triangle non-degenerate and every adjacent
    # triangle's opposite vertex strictly outside the circumcircle, in exact rational arithmetic.
    exact_ok = coplanar == 0 and len(triangles) == expected
    if exact_ok:
        for index, simplex in enumerate(tri.simplices):
            a, b, c = (array[i] for i in simplex)
            orient = exact_orient(a, b, c)
            if orient == 0:
                exact_ok = False
                break
            if orient < 0:
                a, b = b, a
            for side in range(3):
                neighbor = tri.neighbors[index][side]
                if neighbor < 0:
                    continue
                opposite = [v for v in tri.simplices[neighbor] if v not in simplex]
                value = exact_incircle(a, b, c, array[opposite[0]])
                if value >= 0:
                    exact_ok = False
                    break
            if not exact_ok:
                break
    return {
        'name': name,
        'coordinates': array.reshape(-1).tolist(),
        'siteCount': count,
        'hullCount': hull_count,
        'triangleCount': expected if len(triangles) == expected else len(triangles),
        'scipyTriangleCount': len(triangles),
        'coplanarCount': coplanar,
        'generalPosition': bool(exact_ok),
        'triangles': sorted(triangles) if exact_ok else None,
    }


def generate_delaunay():
    rng = make_rng('delaunay')
    uniform = f32(rng.uniform(0, 100, (300, 2)))
    centers = rng.uniform(15, 85, (6, 2))
    clustered = f32(np.concatenate([center + rng.normal(0, 3, (50, 2)) for center in centers]))
    band = f32(np.column_stack([rng.uniform(0, 1000, 200), rng.uniform(0, 1, 200)]))
    sets = [delaunay_set('uniform-300', uniform), delaunay_set('clustered-300', clustered),
            delaunay_set('thin-band-200-aspect-1e3', band)]
    for entry in sets:
        assert entry['scipyTriangleCount'] == 2 * entry['siteCount'] - 2 - entry['hullCount'], entry['name']
    write_fixture('delaunay-tessellation', 'delaunay-properties.json', {
        'oracle': 'scipy.spatial.Delaunay / ConvexHull on the float32 sites (exact in float64)',
        'tolerances': {
            'triangleCount': 'exact: T = 2n - 2 - h, h = ConvexHull vertex count (collinear hull points excluded)',
            'triangleSet': (
                'exact set equality of sorted vertex-id triples when generalPosition is true: no coplanar '
                'points and every adjacent opposite vertex strictly outside the circumcircle, checked with '
                'fractions.Fraction; otherwise only the count is compared'),
            'emptyCircle': 'exact BigInt arithmetic on the float32 inputs in the spec (checkTriangulation)',
        },
        'sets': sets,
    })


# ---------------------------------------------------------------------------------------------
# GPUBufferSurface
# ---------------------------------------------------------------------------------------------

QUAD_SEGMENTS = 8
# GEOS and the GPU both use round(turn / (pi / 2 / q)) uniform steps per round join: one step spans at
# most 1.5 * pi / (2 q), so an arc chord has sagitta at most d * (1 - cos(0.75 * pi / (2 q))).
SAGITTA_RATIO = 1 - math.cos(0.75 * math.pi / (2 * QUAD_SEGMENTS))


def turn_sum(points, closed):
    total = 0.0
    count = len(points)
    for i in range(count if closed else count - 2):
        a = np.array(points[i - 1] if closed else points[i])
        b = np.array(points[i] if closed else points[i + 1])
        c = np.array(points[(i + 1) % count] if closed else points[i + 2])
        u, v = b - a, c - b
        total += abs(math.atan2(u[0] * v[1] - u[1] * v[0], u @ v))
    return total


def buffer_geometry(kind, points, distance):
    from shapely.geometry import LineString, Polygon
    shape = LineString(points) if kind == 'line' else Polygon(points)
    return shape, shape.buffer(distance, quad_segs=QUAD_SEGMENTS, cap_style='round', join_style='round')


def structure(geometry):
    parts = list(geometry.geoms) if hasattr(geometry, 'geoms') else [geometry]
    return len(parts), sum(len(part.interiors) for part in parts)


def buffer_case(rng, kind):
    from shapely.geometry import Point
    for _ in range(2000):
        distance = float(rng.choice([0.5, 0.75, 1.0, 1.25]))
        if kind == 'line':
            count = int(rng.integers(8, 14))
            # Switchbacks: successive rows advance by a gap below 2 d so neighbouring passes overlap.
            ys = np.cumsum(rng.uniform(0.4, 1.7, count) * distance)
            xs = np.where(np.arange(count) % 2 == 0, rng.uniform(0, 3, count), rng.uniform(8, 12, count))
            points = [tuple(p) for p in f32(np.column_stack([xs, ys]))]
        else:
            count = int(rng.integers(9, 16))
            angles = np.sort(rng.uniform(0, 2 * math.pi, count))
            radii = rng.uniform(2.0, 8.0, count)
            points = [tuple(p) for p in f32(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))]
            if rng.random() < 0.5:
                points.reverse()
        from shapely.geometry import LineString, Polygon
        shape = LineString(points) if kind == 'line' else Polygon(points)
        if not shape.is_valid or (kind == 'line' and not shape.is_simple):
            continue
        if kind == 'polygon' and shape.area < 10:
            continue
        shape, result = buffer_geometry(kind, points, distance)
        base = structure(result)
        stable = all(structure(buffer_geometry(kind, points, distance * (1 + delta))[1]) == base
                     for delta in (-0.02, 0.02))
        if not stable:
            continue
        parts = list(result.geoms) if hasattr(result, 'geoms') else [result]
        holes = [Polygon(interior).area for part in parts for interior in part.interiors]
        if holes and min(holes) < 0.5 * distance * distance:
            continue
        turns = turn_sum(points, closed=kind == 'polygon') + (2 * math.pi if kind == 'line' else 0.0)
        minimum_x, minimum_y, maximum_x, maximum_y = result.bounds
        boundary = result.boundary
        samples = []
        for point in rng.uniform([minimum_x - distance, minimum_y - distance],
                                 [maximum_x + distance, maximum_y + distance], (800, 2)):
            geometry = Point(point)
            if geometry.distance(boundary) < 0.03 * distance:
                continue
            samples.append([float(point[0]), float(point[1]), 1 if result.contains(geometry) else 0])
        return {
            'kind': kind,
            'distance': distance,
            'ring': [list(map(float, p)) for p in points] + ([list(map(float, points[0]))] if kind == 'polygon' else []),
            'area': result.area,
            'polygonCount': base[0],
            'holeCount': base[1],
            'turnSum': turns,
            # Each approximation lies below the true buffer by at most (arc length) * (sagitta) = d * turns * d * ratio.
            'areaTolerance': SAGITTA_RATIO * distance * distance * turns + 1e-4 * result.area,
            'samples': samples[:400],
        }
    raise RuntimeError('no buffer case found')


def generate_buffer():
    rng = make_rng('buffer')
    cases = [buffer_case(rng, 'line') for _ in range(6)] + [buffer_case(rng, 'polygon') for _ in range(4)]
    write_fixture('polygon-buffer', 'buffer-properties.json', {
        'oracle': "shapely.buffer(distance, quad_segs=8, cap_style='round', join_style='round')",
        'quadSegments': QUAD_SEGMENTS,
        'sagittaRatio': SAGITTA_RATIO,
        'tolerances': {
            'area': (
                'absolute areaTolerance = sagittaRatio * d^2 * turnSum + 1e-4 * area. Both polygonal round joins '
                '(GEOS and the GPU) use round(turn / (pi / 2 / quadSegments)) uniform inscribed chords, so each '
                'lies below the true buffer by at most (arc length d * turnSum) * (sagitta d * sagittaRatio), '
                'sagittaRatio = 1 - cos(0.75 * pi / (2 * quadSegments)) = 0.0108 (a step spans at most 1.5 nominal '
                'steps). turnSum is the sum of absolute turn angles, plus 2 pi for the two round caps of a line. '
                'The difference of the two areas is therefore at most that bound; 1e-4 relative covers float32.'),
            'membership': 'samples farther than 0.03 d from the shapely boundary (> 2 sagittas) must agree',
            'structure': 'polygonCount and holeCount exact; cases are stable under d * (1 +/- 0.02)',
            'vertices': 'every input vertex lies inside the buffer (distance to the boundary is >= d)',
        },
        'cases': cases,
    })


# ---------------------------------------------------------------------------------------------
# GPUCellCover (H3)
# ---------------------------------------------------------------------------------------------

COVER_CENTER_MARGIN = 1e-4


def cover_case(rng, index, resolution):
    import h3
    from shapely.geometry import Point, Polygon
    high_latitude = index < 3
    for _ in range(500):
        center_lat = float(rng.uniform(60, 80)) if high_latitude else float(rng.uniform(-55, 55))
        center_lng = float(rng.uniform(-170, 170))
        radius = float(rng.uniform(0.7, 1.3)) * {3: 3.0, 4: 1.2, 5: 0.5}[resolution]
        count = int(rng.integers(8, 17))
        angles = 2 * math.pi * (np.arange(count) + rng.uniform(0, 1, count) * 0.8) / count
        radii = radius * rng.uniform(0.45, 1.0, count)
        ring = f32(np.column_stack([center_lng + radii * np.cos(angles) / math.cos(math.radians(center_lat)) ** 0.5,
                                    center_lat + radii * np.sin(angles)]))
        polygon = Polygon(ring)
        if not polygon.is_valid or polygon.area < 0.02:
            continue
        points = [(float(lat), float(lng)) for lng, lat in ring]
        cells = h3.polygon_to_cells(h3.LatLngPoly(points), resolution)
        if len(cells) < 5:
            continue
        # The GPU computes centers in float32. Reject polygons with a center (of a cell near the boundary)
        # closer to the boundary than the margin so an exact set comparison is not a float tie.
        candidates = set(h3.polygon_to_cells(h3.LatLngPoly([(float(lat), float(lng)) for lng, lat in
                                                           np.array(polygon.buffer(1.5 * radius).exterior.coords)]),
                                            resolution))
        boundary = polygon.boundary
        margin = min(Point(h3.cell_to_latlng(cell)[1], h3.cell_to_latlng(cell)[0]).distance(boundary)
                     for cell in candidates)
        if margin < COVER_CENTER_MARGIN:
            continue
        return {
            'resolution': resolution,
            'center': [center_lng, center_lat],
            'ring': ring.tolist(),
            'cells': sorted(cells, key=lambda cell: int(cell, 16)),
            'minimumCenterDistance': margin,
            'pentagonCells': sum(1 for cell in cells if h3.is_pentagon(cell)),
        }
    raise RuntimeError('no cover case found')


def generate_cover():
    rng = make_rng('cover')
    resolutions = [3, 4, 5, 3, 4, 5, 4, 5]
    cases = [cover_case(rng, index, resolution) for index, resolution in enumerate(resolutions)]
    write_fixture('cell-cover', 'cell-cover-h3.json', {
        'oracle': 'h3.polygon_to_cells (cell center inside the polygon; polygon edges are straight in lat/lng)',
        'tolerances': {
            'cells': 'exact set equality per polygon',
            'centerMargin': COVER_CENTER_MARGIN,
            'derivation': (
                'the GPU forms cell centers in float32 (ulp up to 1.5e-5 degrees at longitude 180, 7.6e-6 below 128). '
                'Only polygons whose nearby cell centers are all at least 1e-4 degrees (about 7 ulp) from the '
                'polygon boundary are kept, so a float32 center cannot cross an edge; h3 sees the same float32 '
                'vertices as the GPU. The existing spec measured near-edge disagreements below 5e-5 degrees.'),
        },
        'cases': cases,
    })


# ---------------------------------------------------------------------------------------------
# GPUZoneEvents
# ---------------------------------------------------------------------------------------------

PERTURBATION = Fraction(1, 1000)


def perturbed(point):
    """Sample P = (x, y) read as P' = (x - e^2, y - e) with e a concrete small rational.

    With integer coordinates up to 12 every orientation determinant is an integer plus terms of order e
    and e^2 with integer coefficients bounded by 12, so e = 1/1000 decides every sign the way an
    infinitesimal would (e term >= 1e-3 whenever nonzero, e^2 term <= 1.2e-5)."""
    return (Fraction(point[0]) - PERTURBATION ** 2, Fraction(point[1]) - PERTURBATION)


def orient_fraction(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def sign(value):
    return (value > 0) - (value < 0)


def ray_crosses(point, a, b):
    """+x ray from the perturbed point against edge a-b (exact rational arithmetic)."""
    x, y = perturbed(point)
    ax, ay, bx, by = (Fraction(v) for v in (*a, *b))
    if (ay > y) == (by > y):
        return False
    at_y = ax + (y - ay) * (bx - ax) / (by - ay)
    return at_y > x


def segment_crosses(p, q, c, d):
    """Perturbed segment p'q' against edge c-d: opposite sides both ways (exact). Zero length never crosses."""
    if tuple(p) == tuple(q):
        return False
    pp, qq = perturbed(p), perturbed(q)
    cc, dd = tuple(Fraction(v) for v in c), tuple(Fraction(v) for v in d)
    side_p, side_q = sign(orient_fraction(cc, dd, pp)), sign(orient_fraction(cc, dd, qq))
    side_c, side_d = sign(orient_fraction(pp, qq, cc)), sign(orient_fraction(pp, qq, dd))
    assert 0 not in (side_p, side_q, side_c, side_d)
    return side_p != side_q and side_c != side_d


def random_zone(rng, center):
    from shapely.geometry import Polygon
    for _ in range(1000):
        count = int(rng.integers(4, 8))
        ring = [(int(center[0] + rng.integers(-5, 6)), int(center[1] + rng.integers(-5, 6))) for _ in range(count)]
        if len(set(ring)) < count:
            continue
        polygon = Polygon(ring)
        if polygon.is_valid and polygon.area >= 6 and not any(
                orient_fraction(ring[i - 1], ring[i], ring[(i + 1) % count]) == 0 for i in range(count)):
            return [list(p) for p in ring]
    raise RuntimeError('no zone found')


def generate_zones():
    from shapely.geometry import Point, Polygon
    rng = make_rng('zones')
    zones = [random_zone(rng, c) for c in ((4, 4), (8, 5), (6, 8))]
    vertices = [tuple(p) for zone in zones for p in zone]
    polygons = [Polygon(zone) for zone in zones]
    cross_checked = 0
    tracks = []
    for _ in range(20):
        count = int(rng.integers(3, 11))
        positions = []
        for _ in range(count):
            if rng.random() < 0.3:
                positions.append(vertices[int(rng.integers(0, len(vertices)))])
            elif positions and rng.random() < 0.1:
                positions.append(positions[-1])
            else:
                positions.append((int(rng.integers(0, 13)), int(rng.integers(0, 13))))
        per_zone = []
        for zone, polygon in zip(zones, polygons):
            edges = [(zone[i], zone[(i + 1) % len(zone)]) for i in range(len(zone))]

            def inside(point):
                return sum(ray_crosses(point, a, b) for a, b in edges) % 2 == 1

            for point in positions:
                # Cross-check the rule against shapely for samples that do not touch the boundary.
                if not polygon.boundary.intersects(Point(point)):
                    assert inside(point) == polygon.contains(Point(point)), (point, zone)
                    cross_checked += 1
            state = inside(positions[0])
            initial = state
            enters = exits = 0
            for p, q in zip(positions, positions[1:]):
                crossings = sum(segment_crosses(p, q, a, b) for a, b in edges)
                for _ in range(crossings):
                    state = not state
                    if state:
                        enters += 1
                    else:
                        exits += 1
            assert state == inside(positions[-1]), 'toggle parity must equal the final sample state'
            per_zone.append({'initialInside': initial, 'enters': enters, 'exits': exits, 'finalInside': state})
        tracks.append({'positions': [list(p) for p in positions], 'zones': per_zone})
    total_events = sum(z['enters'] + z['exits'] for t in tracks for z in t['zones'])
    assert total_events > 30
    write_fixture('trajectory-zones', 'zone-events-grid.json', {
        'oracle': (
            'exact rational evaluation of the documented rule (ZONE_TIE_RULE_WGSL): every sample (x, y) is the '
            'point (x - e^2, y - e), e = 1/1000 as fractions.Fraction; edge crossings of the perturbed segment '
            'toggle the zone state; the initial state is the +x ray parity of the perturbed first sample'),
        'perturbation': str(PERTURBATION),
        'shapelyCrossChecks': cross_checked,
        'tolerances': {
            'counts': 'exact enter and exit events per (track, zone) and the final inside state',
            'derivation': (
                'integer coordinates in [0, 12]: orientation determinants are integers plus terms c1 e + c2 e^2 '
                'with |c1|, |c2| <= 12, so a concrete e = 1e-3 reproduces the sign of the infinitesimal limit '
                '(e term >= 1e-3 when nonzero, e^2 term <= 1.2e-5). Enter/exit counts depend only on the number of '
                'toggles and the initial state, not on the order of simultaneous crossings.'),
        },
        'zones': zones,
        'tracks': tracks,
    })


SECTIONS = {
    'rollup': generate_rollup,
    'noding': generate_noding,
    'makevalid': generate_make_valid,
    'delaunay': generate_delaunay,
    'buffer': generate_buffer,
    'cover': generate_cover,
    'zones': generate_zones,
}

if __name__ == '__main__':
    requested = sys.argv[1:] or list(SECTIONS)
    for section in requested:
        SECTIONS[section]()
