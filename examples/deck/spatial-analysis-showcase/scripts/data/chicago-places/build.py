import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chicago-tracts'))
from common import *
import pandas as pd, numpy as np, geopandas as gpd, shapely

CATS = ['restaurant_cafe', 'bar_nightlife', 'grocery', 'health', 'school_education', 'park_recreation', 'transit',
        'retail', 'finance_business', 'personal_services', 'arts_culture', 'worship_community', 'lodging', 'other']
def classify(hier, cat):
    h = hier if isinstance(hier, str) else ''; c = cat if isinstance(cat, str) else ''
    l1 = h.split('>')[0]
    if re.search(r'transit|train_station|bus_station|metro_station|subway|light_rail', c): return 'transit'
    if c in ('grocery_store', 'supermarket', 'farmers_market', 'convenience_store') or 'grocery' in c: return 'grocery'
    if 'alcoholic_beverage' in h or c in ('bar', 'nightclub', 'pub', 'dance_club'): return 'bar_nightlife'
    if l1 == 'food_and_drink': return 'restaurant_cafe'
    if l1 == 'health_care': return 'health'
    if 'pharmacy' in c: return 'health'
    if l1 == 'education': return 'school_education'
    if l1 == 'sports_and_recreation': return 'park_recreation'
    if l1 == 'shopping': return 'retail'
    if 'place_of_worship' in h or 'religious' in h or l1 == 'community_and_government': return 'worship_community'
    if l1 in ('arts_and_entertainment', 'cultural_and_historic'): return 'arts_culture'
    if l1 == 'lodging': return 'lodging'
    if l1 == 'lifestyle_services': return 'personal_services'
    if l1 == 'services_and_business': return 'finance_business'
    return 'other'

p = pd.read_parquet(f'{RAW}/chicago-places/places.parquet')
p = p[(p.operating_status == 'open') & (p.confidence >= 0.5)].copy()
ca = gpd.read_file(f'{RAW}/chicago-community-areas/ca.geojson').to_crs(4326)
city = ca.union_all(); shapely.prepare(city)
pts = shapely.points(p.lon.to_numpy(), p.lat.to_numpy())
p = p[shapely.contains(city, pts)].reset_index(drop=True)
p['category'] = [classify(h, c) for h, c in zip(p.hier, p.cat)]
cid = p.category.map({k: i for i, k in enumerate(CATS)}).to_numpy('u1')
tract, _ = assign_tract(p.lon, p.lat)
ID = 'chicago-places'; d = out_dir(ID)
n = len(p)
m = {'id': ID, 'version': 1, 'kind': 'points', 'count': n, 'bbox': [round(float(p.lon.min()), 5), round(float(p.lat.min()), 5), round(float(p.lon.max()), 5), round(float(p.lat.max()), 5)], 'crs': 'EPSG:4326', 'columns': {}}
write_col(d, m, 'position', np.c_[p.lon, p.lat], 'float32', 'position.bin', 2)
write_col(d, m, 'category', cid, 'uint8', 'category.bin', categories=CATS)
write_col(d, m, 'confidence', (p.confidence * 100).round(), 'uint8', 'confidence.bin', note='percent')
write_col(d, m, 'tract', tract, 'uint16', 'tract.bin', note='index into chicago-tracts features; 65535 = none')
vc = p.category.value_counts()
# record category counts in transit/grocery
m['properties'] = {
  'description': 'Overture Maps Foundation places (release 2026-09-23.1) inside the City of Chicago, open, confidence >= 0.5, mapped to 14 broad categories from Overture taxonomy.',
  'license': 'Overture Places: CDLA-Permissive-2.0 (most sources), Apache-2.0 (Foursquare), CC0 (AllThePlaces) per docs.overturemaps.org/attribution/places',
  'storyNotes': [
    f'{n:,} open places; by category: ' + ', '.join(f'{k} {v:,}' for k, v in vc.head(8).items()) + '.',
    f'{vc.get("grocery", 0):,} grocery/convenience places vs {vc.get("restaurant_cafe", 0):,} restaurants/cafes: food-access (food desert) stories by tract.',
    f'{vc.get("transit", 0):,} transit-tagged places (bus/train stations) are noisy relative to the CTA GTFS; treat as a proxy.',
    'Places are denser downtown and along commercial corridors: strong spatial clustering for Ripley K / cross-K.',
    'Overture merges several commercial sources, so POI counts follow business-data coverage, not ground truth: weak in some South/West side tracts.']}
write_manifest(d, m)
print(m['properties']['storyNotes'])

# ---- facilities
FID = 'chicago-facilities'; fd = out_dir(FID)
rows = []
h = pd.read_parquet(f'{RAW}/chicago-places/places.parquet')
h = h[(h.cat == 'hospital') & (h.operating_status == 'open') & (h.confidence >= 0.7)].copy()
h = h[h.name.fillna('').str.contains(r'hospital|medical center|medical centre|health system', case=False, regex=True) & ~h.name.fillna('').str.contains(r'animal|veterin|pet |urgent|clinic|surgery|cancer center|rehab', case=False, regex=True)]
h = h[shapely.contains(city, shapely.points(h.lon.to_numpy(), h.lat.to_numpy()))]
hm = gpd.GeoSeries(gpd.points_from_xy(h.lon, h.lat), crs=4326).to_crs(26916)
keep = []; kx = []
for i, g in enumerate(hm):
    if all(g.distance(k) > 300 for k in kx): keep.append(i); kx.append(g)
h = h.iloc[keep]
for r in h.itertuples(): rows.append((0, r.lon, r.lat, r.name or ''))
lib = pd.read_csv(f'{RAW}/chicago-places/lib.csv')
for r in lib.itertuples():
    mm = re.search(r'\(([-\d.]+), ([-\d.]+)\)', r.location)
    if mm: rows.append((1, float(mm.group(2)), float(mm.group(1)), 'Chicago Public Library - ' + str(r.branch_)))
cps = pd.read_csv(f'{RAW}/chicago-places/cps.csv')
for r in cps.itertuples(): rows.append((2, r.long, r.lat, f'{r.short_name} ({r.grade_cat})'))
fire = pd.read_csv(f'{RAW}/chicago-places/fire.csv')
for r in fire.itertuples():
    mm = re.search(r'\(([-\d.]+), ([-\d.]+)\)', r.location)
    if mm: rows.append((3, float(mm.group(2)), float(mm.group(1)), f'Fire station {r.name}'))
f = pd.DataFrame(rows, columns=['type', 'lon', 'lat', 'name'])
f = f[shapely.contains(city, shapely.points(f.lon.to_numpy(), f.lat.to_numpy()))].reset_index(drop=True)
FT = ['hospital', 'library', 'cps_school', 'fire_station']
ft, _ = assign_tract(f.lon, f.lat)
fm = {'id': FID, 'version': 1, 'kind': 'points', 'count': len(f), 'bbox': [round(float(f.lon.min()), 5), round(float(f.lat.min()), 5), round(float(f.lon.max()), 5), round(float(f.lat.max()), 5)], 'crs': 'EPSG:4326', 'columns': {}}
write_col(fd, fm, 'position', np.c_[f.lon, f.lat], 'float32', 'position.bin', 2)
write_col(fd, fm, 'category', f.type, 'uint8', 'category.bin', categories=FT)
write_col(fd, fm, 'tract', ft, 'uint16', 'tract.bin')
f[['type', 'name']].assign(type=f.type.map(dict(enumerate(FT)))).to_csv(f'{fd}/names.csv', index=False)
cnt = f.type.value_counts().sort_index()
fm['namesCsv'] = 'names.csv'
fm['properties'] = {
  'description': 'Public facilities in Chicago for location-allocation / coverage: hospitals (Overture places, de-duplicated within 300 m, approximate), Chicago Public Library branches, CPS schools (SY2024-25 locations), CFD fire stations. names.csv rows align with the columns.',
  'storyNotes': [', '.join(f'{FT[k]} {v}' for k, v in cnt.items()) + ' facilities.',
    'Hospitals come from Overture (the city hospital list is a 2011 file) so the count is approximate; verify before citing.',
    'CPS schools (~650 incl. charters) far outnumber fire stations (~90): good for contrasting 5-minute vs 15-minute coverage standards.',
    'Small, well-defined point sets suited to coverage-gap, p-median and nearest-facility scenes.']}
fm['license'] = 'City of Chicago Data Portal Terms of Use (libraries x8fc-8rcq, CPS hexd-c4gn, fire stations 28km-gtjn); hospitals from Overture Maps (CDLA-Permissive-2.0)'
write_manifest(fd, fm)
print(fm['properties']['storyNotes'])
