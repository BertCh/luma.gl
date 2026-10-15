#!/usr/bin/env python3
# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""
Extracts the docs-sized real open data used by the stream-S3 (spatial geometry) lens scenarios
from the spatial-analysis showcase datasets (read-only) into
modules/experimental/src/computation-lenses/data/s3-*.json.

Every file carries lon/lat coordinates (EPSG:4326) and, where a planar operator needs them,
projected coordinates in metres: UTM (EPSG:326xx) minus a stated false origin, so float32 keeps
centimetre precision. The UTM forward projection below is the Krueger n-series (Karney 2011,
eq. 35) to n^6, accurate to well under a millimetre inside the zone.

Run from the repository root:  python3 scripts/lens-data/s3-geometry-data.py
"""

import csv
import json
import math
import os
import struct
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SOURCE = os.path.join(ROOT, 'examples/deck/spatial-analysis-showcase/public/data')
TARGET = os.path.join(ROOT, 'modules/experimental/src/computation-lenses/data')
MAXIMUM_BYTES = 150 * 1024

# --- UTM (Krueger series) ---------------------------------------------------------------------

A = 6378137.0
F = 1 / 298.257223563
N = F / (2 - F)
A_HAT = A / (1 + N) * (1 + N**2 / 4 + N**4 / 64 + N**6 / 256)
ALPHA = [
    N / 2 - 2 * N**2 / 3 + 5 * N**3 / 16 + 41 * N**4 / 180 - 127 * N**5 / 288 + 7891 * N**6 / 37800,
    13 * N**2 / 48 - 3 * N**3 / 5 + 557 * N**4 / 1440 + 281 * N**5 / 630 - 1983433 * N**6 / 1935360,
    61 * N**3 / 240 - 103 * N**4 / 140 + 15061 * N**5 / 26880 + 167603 * N**6 / 181440,
    49561 * N**4 / 161280 - 179 * N**5 / 168 + 6601661 * N**6 / 7257600,
    34729 * N**5 / 80640 - 3418889 * N**6 / 1995840,
    212378941 * N**6 / 319334400,
]
E = math.sqrt(F * (2 - F))


def utm(lon, lat, zone):
    """Forward UTM (north hemisphere) in metres."""
    lon0 = math.radians(-183 + 6 * zone)
    phi = math.radians(lat)
    lam = math.radians(lon) - lon0
    tau = math.tan(phi)
    sigma = math.sinh(E * math.atanh(E * tau / math.sqrt(1 + tau * tau)))
    tau_prime = tau * math.sqrt(1 + sigma * sigma) - sigma * math.sqrt(1 + tau * tau)
    xi_prime = math.atan2(tau_prime, math.cos(lam))
    eta_prime = math.asinh(math.sin(lam) / math.sqrt(tau_prime * tau_prime + math.cos(lam) ** 2))
    xi = xi_prime
    eta = eta_prime
    for j, alpha in enumerate(ALPHA, start=1):
        xi += alpha * math.sin(2 * j * xi_prime) * math.cosh(2 * j * eta_prime)
        eta += alpha * math.cos(2 * j * xi_prime) * math.sinh(2 * j * eta_prime)
    k0 = 0.9996
    return 500000 + k0 * A_HAT * eta, k0 * A_HAT * xi


def project_all(lonlat, zone, origin):
    out = []
    for i in range(0, len(lonlat), 2):
        x, y = utm(lonlat[i], lonlat[i + 1], zone)
        out += [round(x - origin[0], 2), round(y - origin[1], 2)]
    return out


def projected_meta(zone, origin):
    return {
        'crs': f'EPSG:326{zone:02d}',
        'unit': 'm',
        'falseOrigin': origin,
        'note': f'UTM zone {zone}N easting/northing minus falseOrigin, so float32 keeps cm precision',
    }


# --- binary readers ----------------------------------------------------------------------------


def read(dataset, file, fmt):
    with open(os.path.join(SOURCE, dataset, file), 'rb') as handle:
        data = handle.read()
    size = struct.calcsize(fmt)
    return list(struct.unpack(f'<{len(data) // size}{fmt}', data))


def r6(values):
    return [round(v, 6) for v in values]


def write(name, payload):
    path = os.path.join(TARGET, name)
    text = json.dumps(payload, separators=(',', ':'))
    if len(text) > MAXIMUM_BYTES:
        sys.exit(f'{name} is {len(text)} bytes, over the {MAXIMUM_BYTES} byte budget')
    with open(path, 'w') as handle:
        handle.write(text + '\n')
    print(f'{name}: {len(text)} bytes')


def in_box(lon, lat, box):
    return box[0] <= lon <= box[2] and box[1] <= lat <= box[3]


# --- polygons (GeoArrow four-level layout) ------------------------------------------------------


def extract_polygons(dataset, box, keep, maximum_features, name_of=None):
    vertices = read(dataset, 'vertices.bin', 'f')
    ring_offsets = read(dataset, 'ringOffsets.bin', 'I')
    polygon_ring_offsets = read(dataset, 'polygonRingOffsets.bin', 'I')
    part_feature = read(dataset, 'partFeature.bin', 'I')
    feature_parts = {}
    for part, feature in enumerate(part_feature):
        feature_parts.setdefault(feature, []).append(part)
    selected = []
    for feature in sorted(feature_parts):
        parts = feature_parts[feature]
        inside = True
        for part in parts:
            for ring in range(polygon_ring_offsets[part], polygon_ring_offsets[part + 1]):
                for vertex in range(ring_offsets[ring], ring_offsets[ring + 1]):
                    if not in_box(vertices[2 * vertex], vertices[2 * vertex + 1], box):
                        inside = False
        if inside and keep(feature):
            selected.append(feature)
    selected = selected[:maximum_features]
    positions, rings, polygons, features, ids = [], [0], [0], [0], []
    for feature in selected:
        for part in feature_parts[feature]:
            for ring in range(polygon_ring_offsets[part], polygon_ring_offsets[part + 1]):
                start, end = ring_offsets[ring], ring_offsets[ring + 1]
                positions += vertices[2 * start : 2 * end]
                rings.append(len(positions) // 2)
            polygons.append(len(rings) - 1)
        features.append(len(polygons) - 1)
        ids.append(feature)
    return {
        'sourceFeatureIds': ids,
        'positionsLonLat': r6(positions),
        'ringOffsets': rings,
        'polygonOffsets': polygons,
        'featureOffsets': features,
    }


def parks():
    zone, origin = 16, [440000, 4630000]
    names = read('chicago-parks', 'name.bin', 'H')
    area = read('chicago-parks', 'areaKm2.bin', 'f')
    manifest = json.load(open(os.path.join(SOURCE, 'chicago-parks/manifest.json')))
    name_table = manifest.get('names', [])
    box = [-87.690, 41.895, -87.615, 41.985]
    data = extract_polygons('chicago-parks', box, lambda f: area[f] > 0.002, 60)
    data['positionsProjected'] = project_all(data['positionsLonLat'], zone, origin)
    data['names'] = [
        name_table[names[f]] if isinstance(name_table, list) and names[f] < len(name_table) else ''
        for f in data['sourceFeatureIds']
    ]
    write(
        's3-chicago-parks.json',
        {
            'id': 's3-chicago-parks',
            'description': 'Parks and green space around Lincoln Park, Chicago (polygons, 4-level offsets)',
            'source': 'showcase public/data/chicago-parks (OpenStreetMap via Overpass, 2026-10-07)',
            'licence': 'ODbL 1.0, (c) OpenStreetMap contributors',
            'box': box,
            'featureCount': len(data['featureOffsets']) - 1,
            'projected': projected_meta(zone, origin),
            **data,
        },
    )


def community_areas():
    zone, origin = 16, [440000, 4630000]
    area_id = read('chicago-community-areas', 'areaId.bin', 'B')
    box = [-87.76, 41.84, -87.60, 41.99]
    data = extract_polygons('chicago-community-areas', box, lambda f: True, 20)
    data['positionsProjected'] = project_all(data['positionsLonLat'], zone, origin)
    data['areaIds'] = [area_id[f] for f in data['sourceFeatureIds']]
    write(
        's3-chicago-community-areas.json',
        {
            'id': 's3-chicago-community-areas',
            'description': 'Adjacent central Chicago community areas: a polygon coverage with shared edges',
            'source': 'showcase public/data/chicago-community-areas (City of Chicago Data Portal)',
            'licence': 'City of Chicago Data Portal Terms of Use',
            'box': box,
            'featureCount': len(data['featureOffsets']) - 1,
            'projected': projected_meta(zone, origin),
            **data,
        },
    )


# --- lines -------------------------------------------------------------------------------------


def roads(
    box=None,
    name='s3-chicago-roads',
    description='Street centrelines in Wicker Park, Chicago (one line per undirected edge)',
):
    zone, origin = 16, [440000, 4630000]
    dataset = 'chicago-roads'
    source = read(dataset, 'edgeSource.bin', 'I')
    target = read(dataset, 'edgeTarget.bin', 'I')
    reverse = read(dataset, 'edgeReverse.bin', 'I')
    edge_class = read(dataset, 'edgeClass.bin', 'B')
    path_offsets = read(dataset, 'edgePathOffsets.bin', 'I')
    path_vertices = read(dataset, 'edgeVertices.bin', 'f')
    box = box or [-87.6800, 41.9030, -87.6620, 41.9140]
    positions, line_offsets, classes, ids = [], [0], [], []
    for edge in range(len(source)):
        # Undirected: keep one direction of every two-way street.
        if reverse[edge] != 0xFFFFFFFF and reverse[edge] < edge:
            continue
        start, end = path_offsets[edge], path_offsets[edge + 1]
        coordinates = path_vertices[2 * start : 2 * end]
        if not all(
            in_box(coordinates[i], coordinates[i + 1], box) for i in range(0, len(coordinates), 2)
        ):
            continue
        positions += coordinates
        line_offsets.append(len(positions) // 2)
        classes.append(edge_class[edge])
        ids.append(edge)
    write(
        f'{name}.json',
        {
            'id': name,
            'description': description,
            'source': 'showcase public/data/chicago-roads (OpenStreetMap)',
            'licence': 'ODbL 1.0, (c) OpenStreetMap contributors',
            'box': box,
            'lineCount': len(line_offsets) - 1,
            'classNames': ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service/other'],
            'projected': projected_meta(zone, origin),
            'sourceEdgeIds': ids,
            'edgeClass': classes,
            'positionsLonLat': r6(positions),
            'positionsProjected': project_all(r6(positions), zone, origin),
            'lineOffsets': line_offsets,
        },
    )


# --- points ------------------------------------------------------------------------------------


def places():
    zone, origin = 16, [440000, 4630000]
    dataset = 'chicago-places'
    position = read(dataset, 'position.bin', 'f')
    category = read(dataset, 'category.bin', 'B')
    confidence = read(dataset, 'confidence.bin', 'B')
    manifest = json.load(open(os.path.join(SOURCE, dataset, 'manifest.json')))
    box = [-87.6800, 41.9030, -87.6620, 41.9140]
    positions, categories = [], []
    for row in range(len(category)):
        lon, lat = position[2 * row], position[2 * row + 1]
        if in_box(lon, lat, box) and confidence[row] >= 80:
            positions += [lon, lat]
            categories.append(category[row])
    write(
        's3-chicago-places.json',
        {
            'id': 's3-chicago-places',
            'description': 'Overture places (confidence >= 0.8) in Wicker Park, Chicago',
            'source': 'showcase public/data/chicago-places (Overture Maps release 2026-09-23.1)',
            'licence': 'CDLA-Permissive-2.0 (most sources), Apache-2.0 (Foursquare), CC0 (AllThePlaces)',
            'box': box,
            'pointCount': len(categories),
            'categoryNames': manifest['columns']['category']['categories'],
            'projected': projected_meta(zone, origin),
            'category': categories,
            'positionsLonLat': r6(positions),
            'positionsProjected': project_all(r6(positions), zone, origin),
        },
    )


# --- trajectories ------------------------------------------------------------------------------


def ais():
    zone, origin = 18, [570000, 4490000]
    dataset = 'ais-vessels'
    path_offsets = read(dataset, 'pathOffsets.bin', 'I')
    vertices = read(dataset, 'vertices.bin', 'f')
    timestamp = read(dataset, 'timestamp.bin', 'I')
    speed = read(dataset, 'speed.bin', 'f')
    mmsi = read(dataset, 'mmsi.bin', 'I')
    category = read(dataset, 'category.bin', 'B')
    manifest = json.load(open(os.path.join(SOURCE, dataset, 'manifest.json')))
    # Upper Bay, 14:00-16:00 UTC: ferries and tugs crossing, the encounter-rich window.
    box = [-74.075, 40.62, -73.99, 40.71]
    window = [14 * 3600, 16 * 3600]
    tracks = []
    for path in range(len(path_offsets) - 1):
        fixes = [
            i
            for i in range(path_offsets[path], path_offsets[path + 1])
            if window[0] <= timestamp[i] < window[1]
            and in_box(vertices[2 * i], vertices[2 * i + 1], box)
        ]
        if len(fixes) < 20:
            continue
        moved = math.hypot(
            vertices[2 * fixes[-1]] - vertices[2 * fixes[0]],
            vertices[2 * fixes[-1] + 1] - vertices[2 * fixes[0] + 1],
        )
        if moved < 0.01:
            continue
        tracks.append((path, fixes))
    tracks = sorted(tracks, key=lambda item: -len(item[1]))[:12]
    tracks.sort()
    positions, offsets, times, speeds, ids, categories = [], [0], [], [], [], []
    for path, fixes in tracks:
        for i in fixes:
            positions += [vertices[2 * i], vertices[2 * i + 1]]
            times.append(timestamp[i] - window[0])
            speeds.append(round(speed[i], 2))
        offsets.append(len(times))
        ids.append(mmsi[path])
        categories.append(category[path])
    write(
        's3-ny-harbor-ais.json',
        {
            'id': 's3-ny-harbor-ais',
            'description': 'AIS tracks of 12 moving vessels in Upper New York Bay, 2024-06-12 14:00-16:00 UTC',
            'source': 'showcase public/data/ais-vessels (NOAA / USCG Nationwide AIS 2024)',
            'licence': 'CC0 1.0',
            'box': box,
            'trackCount': len(offsets) - 1,
            'timestampUnit': 'seconds since 2024-06-12T14:00:00Z',
            'speedUnit': 'knots (SOG)',
            'categoryNames': manifest['columns']['category'].get('categories', []),
            'projected': projected_meta(zone, origin),
            'mmsi': ids,
            'category': categories,
            'positionsLonLat': r6(positions),
            'positionsProjected': project_all(r6(positions), zone, origin),
            'trackOffsets': offsets,
            'timestamps': times,
            'speeds': speeds,
        },
    )


def ais_zones():
    zone, origin = 18, [570000, 4490000]
    collection = json.load(open(os.path.join(SOURCE, 'ais-zones/zones.geojson')))
    box = [-74.075, 40.62, -73.99, 40.71]
    positions, rings, polygons, features, names = [], [0], [0], [0], []
    for feature in collection['features']:
        properties = feature['properties']
        if not str(properties.get('source', '')).startswith('NOAA'):
            continue  # skip the hand-drawn approximations
        geometry = feature['geometry']
        parts = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
        coordinates = [c for part in parts for ring in part for c in ring]
        if not all(in_box(lon, lat, box) for lon, lat in coordinates):
            continue
        for part in parts:
            for ring in part:
                for lon, lat in ring[:-1]:
                    positions += [lon, lat]
                rings.append(len(positions) // 2)
            polygons.append(len(rings) - 1)
        features.append(len(polygons) - 1)
        names.append(properties.get('name', ''))
    write(
        's3-ny-harbor-zones.json',
        {
            'id': 's3-ny-harbor-zones',
            'description': 'NOAA ENC anchorage areas and channels inside Upper New York Bay',
            'source': 'showcase public/data/ais-zones (NOAA Coast Survey ENCDirect)',
            'licence': 'Public domain (NOAA)',
            'box': box,
            'featureCount': len(features) - 1,
            'projected': projected_meta(zone, origin),
            'names': names,
            'positionsLonLat': r6(positions),
            'positionsProjected': project_all(r6(positions), zone, origin),
            'ringOffsets': rings,
            'polygonOffsets': polygons,
            'featureOffsets': features,
        },
    )


# --- great-circle pairs ------------------------------------------------------------------------


def flights():
    dataset = 'openflights'
    location = read(dataset, 'position.bin', 'f')
    origin = read(dataset, 'origin.bin', 'I')
    destination = read(dataset, 'destination.bin', 'I')
    count = read(dataset, 'count.bin', 'H')
    with open(os.path.join(SOURCE, dataset, 'airports.csv')) as handle:
        airports = {int(row['index']): row for row in csv.DictReader(handle)}
    hub = next(index for index, row in airports.items() if row['iata'] == 'ORD')
    routes = []
    for pair in range(len(origin)):
        if hub in (origin[pair], destination[pair]):
            other = destination[pair] if origin[pair] == hub else origin[pair]
            routes.append((count[pair], other))
    routes = sorted(routes, reverse=True)[:40]
    targets = sorted(other for _, other in routes)
    write(
        's3-openflights-ord.json',
        {
            'id': 's3-openflights-ord',
            'description': "The 40 busiest airport pairs touching Chicago O'Hare (ORD)",
            'source': 'showcase public/data/openflights (OpenFlights.org airports.dat, routes.dat)',
            'licence': 'ODbL 1.0, OpenFlights.org',
            'hub': {'iata': 'ORD', 'lonLat': r6([location[2 * hub], location[2 * hub + 1]])},
            'destinationCount': len(targets),
            'iata': [airports[other]['iata'] for other in targets],
            'destinationsLonLat': r6([v for other in targets for v in (location[2 * other], location[2 * other + 1])]),
        },
    )


if __name__ == '__main__':
    os.makedirs(TARGET, exist_ok=True)
    # A separate extract keeps other cards' Wicker Park inputs unchanged.
    roads(
        box=[-87.685, 41.908, -87.658, 41.925],
        name='s3-chicago-area-roads',
        description='Street centrelines across Logan Square, Lincoln Park and West Town (one line per undirected edge)',
    )
    if '--line-length-per-polygon-only' in sys.argv:
        sys.exit(0)
    parks()
    community_areas()
    roads()
    places()
    ais()
    ais_zones()
    flights()
