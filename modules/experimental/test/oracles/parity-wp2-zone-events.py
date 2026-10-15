# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Shapely segment/polygon intersections lifted onto a linear trajectory clock.

Reference components: shapely.STRtree.query(predicate='intersects'), intersection,
line_locate_point(normalized=True), contains; pyproj.Transformer(always_xy=True).
https://shapely.readthedocs.io/en/stable/reference/shapely.intersection.html
GEOS computes spatial inside pieces; the adapter maps each endpoint's arclength
fraction to the sample's linear time and merges adjacent pieces along track order.
This is external spatial parity plus an explicit time adapter, NOT MobilityDB.
No TS/WGSL crossing predicates or local oracle are used.
Event labels follow the API's (track, time, polygon edge-row) toggle order.
At equal times this differs from spatial entry/exit direction. GEOS supplies the
crossings and the nearest boundary edge; only the documented ordering/label
adapter is project-specific. Maximal time intervals are unaffected by reordering
zero-duration toggles. The original path endpoint direction is not an event label.

Real data: every second complete US AIS track touching the fixed NYC window,
all 63 showcase AIS zones, WGS84 AEQD centered -74.05,40.6 in meters. All projected
positions/polygons and seconds are rounded to f32 before reference evaluation.
No spatial simplification or temporal resampling. Islands/no visits retained.
Dwell is sum of maximal inside intervals; initial/final open intervals count as
visits without fabricated boundary events. Event times are relative to first
sample. Holes and separate polygon parts are preserved. No overlap union between
zones. Stationary segments use GEOS contains. Real boundary points are counted
and rejected rather than silently excluded.

Synthetic boundary ownership uses GEOS on translated tracks (-epsilon^2,-epsilon),
the documented south-then-west ownership, at epsilon=2^-15 and 2^-16. This is a
finite approximation to the symbolic convention, explicitly not a general exact
perturbation oracle. Both runs must have identical event identities/visit counts
and differ <1e-3 in every event coordinate/time and dwell value. Stored limits
use epsilon=2^-16 with conservative 1e-3 D-profile floor; no tolerance is widened.
"""
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pyproj
import shapely
from shapely import LineString, Point, Polygon, STRtree
from shapely.ops import transform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data'


def round_geometry(geometry):
    return transform(lambda longitude, latitude: (np.asarray(longitude, dtype=np.float32),
                                                  np.asarray(latitude, dtype=np.float32)), geometry)


def get_lines(geometry):
    if geometry.geom_type == 'LineString':
        return [geometry] if geometry.length > 0 else []
    if hasattr(geometry, 'geoms'):
        return [line for part in geometry.geoms for line in get_lines(part)]
    return []


def pack_polygons(zones):
    result = dict(positions=[], featureOffsets=[0], polygonOffsets=[0], ringOffsets=[0])
    for zone in zones:
        for polygon in ([zone] if isinstance(zone, Polygon) else zone.geoms):
            for ring in [polygon.exterior, *polygon.interiors]:
                result['positions'].extend(np.asarray(ring.coords)[:-1].reshape(-1).tolist())
                result['ringOffsets'].append(len(result['positions']) // 2)
            result['polygonOffsets'].append(len(result['ringOffsets']) - 1)
        result['featureOffsets'].append(len(result['polygonOffsets']) - 1)
    return result


def run_reference(tracks, zones, epsilon=0):
    tree = STRtree(zones)
    zone_count = len(zones)
    boundary_edges = []
    for zone in zones:
        for polygon in ([zone] if isinstance(zone, Polygon) else zone.geoms):
            for ring in [polygon.exterior, *polygon.interiors]:
                boundary_edges.extend(LineString([first, second])
                                      for first, second in zip(ring.coords[:-1], ring.coords[1:]))
    edge_tree = STRtree(boundary_edges)
    events, intervals, table = [], [], []
    dwell = [0.] * (len(tracks) * zone_count)
    visits = [0] * len(dwell)
    track_counts = [0] * len(tracks)
    row_offset = 0
    boundary_samples = 0
    for track_index, track in enumerate(tracks):
        if not len(track):
            continue
        positions = track[:, :2]
        shifted = positions - [epsilon**2, epsilon]
        times = track[:, 2] - track[0, 2]
        pieces = [[] for _ in zones]
        if not epsilon:
            boundary_samples += sum(any(zone.boundary.intersects(Point(position))
                                        for zone in zones) for position in shifted)
        if len(track) == 1:
            for zone_index in tree.query(Point(shifted[0]), predicate='within'):
                pieces[zone_index].append((0., 0.))
        for segment_index in range(len(track) - 1):
            first, second = shifted[segment_index:segment_index+2]
            if np.array_equal(first, second):
                for zone_index in tree.query(Point(first), predicate='within'):
                    pieces[zone_index].append((float(segment_index), float(segment_index+1)))
                continue
            segment = LineString([first, second])
            for zone_index in tree.query(segment, predicate='intersects'):
                for line in get_lines(shapely.intersection(segment, zones[zone_index])):
                    fractions = sorted(float(shapely.line_locate_point(segment, Point(coordinate), normalized=True))
                                       for coordinate in (line.coords[0], line.coords[-1]))
                    pieces[zone_index].append((segment_index + fractions[0], segment_index + fractions[1]))
        def interpolate(parameter):
            segment = min(int(math.floor(parameter)), max(0, len(track) - 2))
            fraction = parameter - segment
            if len(track) == 1:
                return 0., positions[0], row_offset
            position = positions[segment] + fraction * (positions[segment+1] - positions[segment])
            time = times[segment] + fraction * (times[segment+1] - times[segment])
            return float(time), position, row_offset + segment + 1
        for zone_index, zone_pieces in enumerate(pieces):
            merged = []
            for start, stop in sorted(zone_pieces):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], stop)
                else:
                    merged.append([start, stop])
            cell = track_index * zone_count + zone_index
            for start, stop in merged:
                entry_time, entry_position, entry_row = interpolate(start)
                exit_time, exit_position, exit_row = interpolate(stop)
                intervals.append([track_index, zone_index, entry_time, exit_time])
                dwell[cell] += exit_time - entry_time
                visits[cell] += 1
                if start > 0:
                    events.append([track_index, zone_index, 0, entry_time, *entry_position, entry_row])
                    track_counts[track_index] += 1
                if stop < len(track) - 1:
                    events.append([track_index, zone_index, 1, exit_time, *exit_position, exit_row])
                    track_counts[track_index] += 1
            if merged:
                table.append([track_index, zone_index, len(merged), dwell[cell],
                              interpolate(merged[0][0])[0], interpolate(merged[-1][1])[0]])
        row_offset += len(track)
    # Preserve canonical path order for comparing crossing coordinates, but assign
    # event types in the documented public order. This is not a crossing oracle.
    ordered_events = sorted(events, key=lambda event: (
        event[0], event[3], int(edge_tree.nearest(Point(np.asarray(event[4:6]) - [epsilon**2, epsilon])))))
    inside = {(track_index, zone_index): zone.contains(Point(track[0, :2] - [epsilon**2, epsilon]))
              for track_index, track in enumerate(tracks) if len(track)
              for zone_index, zone in enumerate(zones)}
    for event in ordered_events:
        cell = (event[0], event[1])
        inside[cell] = not inside[cell]
        event[2] = 0 if inside[cell] else 1
    return dict(events=events, intervals=intervals, table=table, dwellTimes=dwell,
                visitCounts=visits, trackEventCounts=track_counts, boundarySamples=boundary_samples)


def make_case(name, tracks, zones, epsilon=0):
    tracks = [np.asarray(track, dtype=np.float32).astype(float).reshape(-1, 3) for track in tracks]
    zones = [round_geometry(zone) for zone in zones]
    assert all(zone.is_valid for zone in zones)
    reference = run_reference(tracks, zones, epsilon)
    if not epsilon:
        assert reference['boundarySamples'] == 0
    magnitude = max(1., max(abs(coordinate) for zone in zones for coordinate in zone.bounds),
                    max(np.abs(track[:, :2]).max() for track in tracks if len(track)))
    time_scale = max(track[-1, 2] - track[0, 2] for track in tracks if len(track))
    return dict(name=name, tracks=[track.tolist() for track in tracks], zoneCount=len(zones),
                polygons=pack_polygons(zones), expected=reference,
                positionAbsolute=max(1e-3, 8 * float(np.spacing(np.float32(magnitude)))),
                timeAbsolute=max(1e-3, 8 * float(np.spacing(np.float32(time_scale)))),
                epsilon=epsilon)


source = DATA / 'poopdeck-ais-us'
offsets = np.fromfile(source / 'pathOffsets.bin', '<u4')
vertices = np.fromfile(source / 'vertices.bin', '<f4').reshape(-1, 2)
times = np.fromfile(source / 'time.bin', '<u4')
window = [-74.4, 40.3, -73.7, 40.9]
mask = ((vertices[:, 0] > window[0]) & (vertices[:, 0] < window[2])
        & (vertices[:, 1] > window[1]) & (vertices[:, 1] < window[3]))
window_tracks = [index for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:]))
                 if np.any(mask[start:stop])]
selected = window_tracks[::2]
crs = '+proj=aeqd +lat_0=40.6 +lon_0=-74.05 +datum=WGS84 +units=m'
transformer = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy=True)
tracks = []
for index in selected:
    start, stop = offsets[index:index+2]
    projected = np.asarray(transformer.transform(vertices[start:stop, 0], vertices[start:stop, 1])).T
    tracks.append(np.column_stack([projected, times[start:stop]]))
zone_source = DATA / 'ais-zones/zones.geojson'
zones = [transform(transformer.transform, shapely.geometry.shape(feature['geometry']))
         for feature in json.loads(zone_source.read_text())['features']]
real = make_case('nyc-us-ais-spatial-window-stride-2', tracks, zones)
assert len(real['expected']['events']) > 100

synthetic_zones = [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                           [[(3, 3), (3, 7), (7, 7), (7, 3)]]),
                   Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])]
synthetic_tracks = [
    [[-2, 5, 100], [22, 5, 124]],  # hole and shared boundary
    [[1, 1, 10], [2, 2, 20]],  # open both ends
    [[-2, 1, 0], [2, 1, 4], [-2, 1, 8], [2, 1, 12]],  # repeated visits
    [[-2, 10, 0], [22, 10, 24]],  # upper edge belongs inside
    [[-2, 0, 0], [22, 0, 24]],  # lower edge belongs outside
    [[10, -2, 0], [10, 12, 14]],  # shared edge belongs to west zone
    [[0, -2, 0], [0, 12, 14]],  # west edge outside
    [[-2, 2, 0], [2, -2, 4]],  # exterior vertex tangency
    [[-2, -2, 0], [12, 12, 14]],  # through corners and hole
    [[1, 1, 0], [1, 1, 10], [12, 1, 21]],  # stationary dwell
    [[1, 1, 7]], [],  # singleton and empty
    [[-2, 2, 0], [12, 2, 0], [22, 2, 10]],  # duplicate time, spatial crossings
]
edge = make_case('holes-open-visits-boundary-ownership', synthetic_tracks, synthetic_zones, 2**-16)
coarser = make_case('boundary-convergence', synthetic_tracks, synthetic_zones, 2**-15)
for field in ['visitCounts', 'trackEventCounts']:
    assert edge['expected'][field] == coarser['expected'][field]
for field in ['events', 'intervals', 'table', 'dwellTimes']:
    finer_values = np.asarray(edge['expected'][field])
    coarser_values = np.asarray(coarser['expected'][field])
    assert finer_values.shape == coarser_values.shape
    assert np.max(np.abs(finer_values - coarser_values)) < 1e-3

fixture = dict(generator=Path(__file__).name, reference='shapely.STRtree.query/intersection/line_locate_point + linear-time adapter',
               versions=dict(python=platform.python_version(), numpy=np.__version__, shapely=shapely.__version__,
                             geos=shapely.geos_version_string, pyproj=pyproj.__version__),
               source=dict(dataset='poopdeck-ais-us + ais-zones', fullTrackCount=len(offsets)-1,
                           window=window, windowTrackCount=len(window_tracks), sourceTrackIds=selected,
                           selection='all complete tracks touching window, in source order, stride 2; all 63 zones',
                           hashes={str(path.relative_to(DATA)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [source/'manifest.json', source/'vertices.bin', source/'time.bin',
                                                source/'pathOffsets.bin', zone_source]}),
               conventions=dict(crs=crs, units='meters and relative seconds', inputPrecision='float32',
                                boundary='real samples off-boundary; synthetic GEOS finite south-west translation',
                                intervalAdapter='segment arclength fraction -> linear time; merge adjacent pieces',
                                eventLabels='toggle in (track,time,polygon edge row) order, including tied times'),
               tolerance=dict(profile='D', relative=2e-6,
                              position='max(1e-3 m, 8*ulp32(max coordinate))',
                              time='max(1e-3 seconds, 8*ulp32(track duration)); same float32 budget in time units',
                              idsCountsFlags='exact'), cases=[real, edge])
output = HERE / 'fixtures/parity-wp2-zone-events.json'
output.write_text(json.dumps(fixture, separators=(',', ':'), allow_nan=False) + '\n')
assert output.stat().st_size <= 1500000
print(output.name, output.stat().st_size, [(case['name'], len(case['tracks']), sum(map(len, case['tracks'])), len(case['expected']['events'])) for case in fixture['cases']])
