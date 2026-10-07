"""Shared helpers for the Chicago dataset builders (import via sys.path.insert of this folder)."""
import json, os
import numpy as np

SCRATCH = '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase'
RAW = SCRATCH + '/raw'
APP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
CHICAGO_BBOX = [-87.95, 41.64, -87.52, 42.03]


def out_dir(dataset_id):
    d = os.path.join(APP, 'public', 'data', dataset_id)
    os.makedirs(d, exist_ok=True)
    return d


def write_col(d, manifest, name, arr, dtype, file, components=1, **extra):
    a = np.ascontiguousarray(np.asarray(arr).astype('<' + np.dtype(dtype).str[1:]))
    a.tofile(os.path.join(d, file))
    manifest['columns'][name] = {'file': file, 'dtype': dtype, 'components': components,
                                 'length': int(a.size // components), **extra}


def write_manifest(d, manifest):
    with open(os.path.join(d, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    tot = 0
    for fn in sorted(os.listdir(d)):
        s = os.path.getsize(os.path.join(d, fn)); tot += s
        print(f'  {fn:28s}{s:>10,d}')
    print(f'  TOTAL {tot:,d} bytes')
    return tot


def polygons_to_binary(geoms):
    """GeoArrow-style: vertices (float32x2), ringOffsets (vertex index per ring, n+1),
    polygonRingOffsets (ring index per polygon, n+1). Multipolygons are exploded: one entry per part,
    with `partFeature` mapping part -> feature index. Rings keep GeoJSON orientation, closed (first==last)."""
    verts = []; ringOff = [0]; polyRing = [0]; part_feature = []; nv = 0; nr = 0
    for i, g in enumerate(geoms):
        parts = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
        for p in parts:
            for ring in [p.exterior] + list(p.interiors):
                c = np.asarray(ring.coords)[:, :2]
                verts.append(c); nv += len(c); ringOff.append(nv); nr += 1
            polyRing.append(nr); part_feature.append(i)
    return (np.vstack(verts).astype('<f4'), np.array(ringOff, '<u4'), np.array(polyRing, '<u4'),
            np.array(part_feature, '<u4'))


def assign_tract(lon, lat):
    """Index (into tracts.geojson / tract columns) of the tract containing each point; 65535 if none.
    Uses the full-resolution tract file written by chicago-tracts/build.py."""
    import geopandas as gpd, shapely
    t = gpd.read_file(f'{RAW}/chicago-tracts/tracts_full.gpkg').reset_index(drop=True)
    pts = shapely.points(np.asarray(lon, 'f8'), np.asarray(lat, 'f8'))
    ip, it = t.sindex.query(pts, predicate='within')
    out = np.full(len(pts), 65535, np.uint16)
    out[ip] = it  # a point on a shared edge would match 2; last wins (measure-zero)
    return out, t


NATURE_GROUPS = {'Plantae': 'Plants', 'Insecta': 'Insects', 'Aves': 'Birds', 'Fungi': 'Fungi', 'Arachnida': 'Spiders and kin',
                 'Mammalia': 'Mammals', 'Mollusca': 'Snails and mussels', 'Amphibia': 'Amphibians and reptiles',
                 'Reptilia': 'Amphibians and reptiles', 'Actinopterygii': 'Fish'}


def nature_group(iconic):
    """Display group of an iNaturalist iconic taxon name; anything unlisted is 'Other life'."""
    return iconic.map(NATURE_GROUPS).fillna('Other life')


def load_nature_observations():
    """Wild, precisely located, timed 2023 iNaturalist observations from RAW/chicago-nature/inat2023.json.
    Obscured points are randomised over ~20 km cells, captive/cultivated ones are planted or kept, and
    >250 m accuracy blurs block-level patterns. Returns (frame with latitude/longitude, fetched row count)."""
    import pandas as pd
    r = pd.DataFrame(json.load(open(f'{RAW}/chicago-nature/inat2023.json')))
    n_raw = len(r)
    r = r[r.location.notna() & ~r.obscured.astype(bool) & ~r.captive.astype(bool) & r.iconic.notna()]
    r = r[(r.accuracy.isna() | (r.accuracy <= 250)) & r.time_observed_at.notna()].copy()
    lat_lng = r.location.str.split(',', expand=True).astype('f8')
    r['latitude'], r['longitude'] = lat_lng[0], lat_lng[1]
    # Local wall-clock time: the ISO offset is Chicago's (-05:00/-06:00), so drop it rather than convert.
    local = pd.to_datetime(r.time_observed_at.str.slice(0, 19))
    r = r[(local >= '2023-01-01') & (local < '2024-01-01')].copy()
    r['ts'] = (pd.to_datetime(r.time_observed_at.str.slice(0, 19)) - pd.Timestamp('2023-01-01')).dt.total_seconds().astype('int64')
    return r, n_raw
