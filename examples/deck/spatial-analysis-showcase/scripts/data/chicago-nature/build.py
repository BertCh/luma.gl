import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chicago-tracts'))
from common import *
import pandas as pd, numpy as np, geopandas as gpd, shapely

ID = 'chicago-nature'
r, n_raw = load_nature_observations()
tract, tr = assign_tract(r.longitude, r.latitude)
r['tract'] = tract
r = r[r.tract != 65535].sort_values(['ts', 'id']).reset_index(drop=True)

r['group'] = nature_group(r.iconic)
order = r.group.value_counts().index.tolist()
order = [g for g in order if g != 'Other life'] + ['Other life']
catid = r.group.map({k: i for i, k in enumerate(order)}).to_numpy('u1')

ca_gdf = gpd.read_file(f'{RAW}/chicago-community-areas/ca.geojson').reset_index(drop=True)
pts = shapely.points(r.longitude.to_numpy(), r.latitude.to_numpy())
ip, ic = ca_gdf.sindex.query(pts, predicate='within')
ca = np.zeros(len(r), 'u1'); ca[ip] = ca_gdf.area_numbe.astype(int).to_numpy()[ic]

n = len(r)
d = out_dir(ID)
for f in os.listdir(d):
    os.remove(os.path.join(d, f))
m = {'id': ID, 'version': 1, 'kind': 'points', 'count': n, 'bbox': [round(float(r.longitude.min()), 5), round(float(r.latitude.min()), 5), round(float(r.longitude.max()), 5), round(float(r.latitude.max()), 5)], 'crs': 'EPSG:4326', 'columns': {}}
write_col(d, m, 'position', np.c_[r.longitude, r.latitude], 'float32', 'position.bin', 2)
write_col(d, m, 'timestamp', r.ts, 'uint32', 'time.bin', unit='seconds since 2023-01-01T00:00:00 (Chicago local clock time, stored as if UTC)')
write_col(d, m, 'category', catid, 'uint8', 'category.bin', categories=order, note='iNaturalist iconic taxon, grouped')
write_col(d, m, 'researchGrade', (r.quality_grade == 'research').astype('u1'), 'uint8', 'researchGrade.bin', note='1 = community-confirmed identification (iNaturalist research grade)')
write_col(d, m, 'introduced', (r.introduced == True).astype('u1'), 'uint8', 'introduced.bin', note='1 = taxon is introduced (non-native) in Chicago according to iNaturalist establishment means')
write_col(d, m, 'species', pd.factorize(r.taxon_id)[0].astype('u4'), 'uint32', 'species.bin', note='dense taxon index (distinct taxa observed), for richness counts')
write_col(d, m, 'communityArea', ca, 'uint8', 'communityArea.bin', note='1-77, 0 = missing')
write_col(d, m, 'tract', r.tract, 'uint16', 'tract.bin', note='index into chicago-tracts features (precomputed point-in-polygon join, validation column)')

dt = pd.to_datetime(r.time_observed_at.str.slice(0, 19))
mon = dt.dt.month.value_counts().sort_index(); hr = dt.dt.hour.value_counts().sort_index()
day = dt.dt.date.value_counts()
ca_names = {int(x.area_numbe): x.community.title() for x in ca_gdf.itertuples()}
top_ca = pd.Series(ca[ca > 0]).value_counts().head(3)
cell = (np.floor(r.longitude / 0.005).astype(int).astype(str) + '_' + np.floor(r.latitude / 0.005).astype(int).astype(str)).value_counts()
cx, cy = [int(v) for v in cell.index[0].split('_')]
top_species = r.common.fillna(r.name).value_counts().head(5)
m['properties'] = {
  'description': 'iNaturalist observations of wild plants, animals and fungi inside the City of Chicago, calendar year 2023: openly licensed (CC0, CC BY, CC BY-NC), not captive or cultivated, not location-obscured, positional accuracy 250 m or better, with an observation time. Sorted by time.',
  'source': 'https://api.inaturalist.org/v1/observations?place_id=49906&d1=2023-01-01&d2=2023-12-31&licensed=true (queried 2026-10-06)',
  'license': 'Each observation keeps its own licence (CC0, CC BY 4.0 or CC BY-NC 4.0); attribute "iNaturalist contributors". Non-commercial use.',
  'timeNote': 'Timestamps are local wall-clock times; no timezone conversion.',
  'filtering': f'{n_raw:,} licensed observations fetched; {n:,} kept after the wild/precise/timed/inside-city filters.',
  'storyNotes': [
    f'{n:,} wild observations of {r.taxon_id.nunique():,} taxa in 2023; by group: ' + ', '.join(f'{k} ({v:,})' for k, v in r.group.value_counts().head(5).items()) + '.',
    f'Seasonality: peak month {mon.idxmax()} ({mon.max():,}), quietest month {mon.idxmin()} ({mon.min():,}); hourly peak at {hr.idxmax()}:00. Busiest days {day.index[0]} ({day.iloc[0]:,}) and {day.index[1]} ({day.iloc[1]:,}); 2023-04-28..05-01 was the City Nature Challenge weekend; the median day has {int(day.median())}.',
    f'Research grade {(r.quality_grade == "research").mean()*100:.1f}%; introduced (non-native) taxa {(r.introduced == True).mean()*100:.1f}% of observations.',
    'Most observed: ' + ', '.join(f'{k} ({v:,})' for k, v in top_species.items()) + '.',
    'Busiest community areas: ' + ', '.join(f'{ca_names.get(int(k), k)} ({v:,})' for k, v in top_ca.items()) + '.',
    f'Densest ~500 m cell centred near {(cy+0.5)*0.005:.4f}N, {(cx+0.5)*0.005:.4f}E holds {cell.iloc[0]:,} observations.',
    'Observations follow observers: parks, lakefront and nature areas dominate, so counts measure observer effort as much as biodiversity.']}
write_manifest(d, m)
print(json.dumps(m['properties'], indent=1))
