import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import geopandas as gpd, pandas as pd, numpy as np, shapely

ID = 'chicago-tracts'
R = f'{RAW}/chicago-tracts'
t = gpd.read_file(f'{R}/shp/cb_2022_17_tract_500k.shp').to_crs(4326)
t = t[t.COUNTYFP == '031'].copy()
ca = gpd.read_file(f'{RAW}/chicago-community-areas/ca.geojson').to_crs(4326)
city = ca.union_all()
# keep tracts whose representative point lies in the City (whole tracts are kept so shared edges stay exact)
rp = t.geometry.representative_point()
t = t[rp.within(city)].copy()
t['GEOID'] = t.GEOID.astype(str)
t = t.sort_values('GEOID').reset_index(drop=True)
print('tracts', len(t))

# --- SVI 2022
s = pd.read_csv(f'{R}/svi_il.csv', encoding='utf-8-sig', dtype={'FIPS': str}).replace(-999, np.nan)
s = s.set_index('FIPS')
s = s.reindex(t.GEOID)
col = pd.DataFrame(index=range(len(t)))
def put(name, series):
    col[name] = pd.to_numeric(series, errors='coerce').to_numpy('f4')
put('population', s.E_TOTPOP); put('households', s.E_HH); put('housingUnits', s.E_HU)
put('poverty150', s.E_POV150); put('poverty150Pct', s.EP_POV150)
put('unemployed', s.E_UNEMP); put('unemployedPct', s.EP_UNEMP)
put('noVehicle', s.E_NOVEH); put('noVehiclePct', s.EP_NOVEH)
put('age65', s.E_AGE65); put('age65Pct', s.EP_AGE65)
put('age17', s.E_AGE17)
put('disability', s.E_DISABL); put('disabilityPct', s.EP_DISABL)
put('uninsured', s.E_UNINSUR); put('uninsuredPct', s.EP_UNINSUR)
put('noHighSchool', s.E_NOHSDP); put('housingBurdened', s.E_HBURD); put('limitedEnglish', s.E_LIMENG)
tot = s.E_TOTPOP.to_numpy('f8')
hisp = s.E_HISP.to_numpy('f8'); black = s.E_AFAM.to_numpy('f8'); asian = s.E_ASIAN.to_numpy('f8')
other = (s.E_AIAN + s.E_NHPI + s.E_TWOMORE + s.E_OTHERRACE).to_numpy('f8')
white = np.clip(tot - hisp - black - asian - other, 0, None)
col['hispanic'] = hisp.astype('f4'); col['nhWhite'] = white.astype('f4'); col['nhBlack'] = black.astype('f4')
col['nhAsian'] = asian.astype('f4'); col['nhOther'] = other.astype('f4')

# --- ACS 2020-2024 income via Census Reporter
acs = json.load(open(f'{R}/acs.json'))['data']
def acsv(table, var):
    out = []
    for gid in t.GEOID:
        v = acs.get('14000US' + gid, {}).get(table, {}).get('estimate', {}).get(var)
        out.append(np.nan if v is None or v < 0 else v)
    return np.array(out, 'f4')
col['perCapitaIncome'] = acsv('B19301', 'B19301001')
col['medianHouseholdIncome'] = acsv('B19013', 'B19013001')

# --- CDC PLACES
p = pd.read_csv(f'{R}/places.csv', dtype={'locationid': str})
pv = p.pivot_table(index='locationid', columns='measureid', values='data_value', aggfunc='first').reindex(t.GEOID)
names = {'OBESITY': 'obesity', 'DIABETES': 'diabetes', 'CASTHMA': 'asthma', 'DEPRESSION': 'depression', 'LPA': 'noLeisureActivity',
         'BPHIGH': 'highBloodPressure', 'CHD': 'heartDisease', 'MHLTH': 'poorMentalHealth', 'PHLTH': 'poorPhysicalHealth',
         'ACCESS2': 'noHealthInsurance1864', 'CSMOKING': 'smoking', 'BINGE': 'bingeDrinking'}
for k, v in names.items():
    col[v] = pv[k].to_numpy('f4')

# --- iNaturalist nature observation counts (2023)
c, _ = load_nature_observations()
t.to_file(f'{R}/tracts_full.gpkg', driver='GPKG')
ti, _ = assign_tract(c.longitude, c.latitude)
c['t'] = ti
c['group'] = nature_group(c.iconic)
def cnt(mask):
    return np.bincount(c.t[mask & (c.t != 65535)], minlength=len(t)).astype('f4')
allm = pd.Series(True, index=c.index)
col['natureObs2023'] = cnt(allm)
col['birdObs2023'] = cnt(c.group == 'Birds'); col['plantObs2023'] = cnt(c.group == 'Plants')
col['insectObs2023'] = cnt(c.group == 'Insects'); col['fungiObs2023'] = cnt(c.group == 'Fungi')
col['introducedObs2023'] = cnt(c.introduced == True)
inside = c[c.t != 65535]
col['speciesRichness2023'] = inside.groupby('t').taxon_id.nunique().reindex(range(len(t)), fill_value=0).to_numpy('f4')

wb = f'{RAW}/chicago-lodes-od/wac_by_tract.csv'
if os.path.exists(wb):
    w = pd.read_csv(wb, dtype={'GEOID': str}).set_index('GEOID').reindex(t.GEOID)
    col['jobsWac2021'] = w.wacJobs.to_numpy('f4'); col['residentWorkers2021'] = w.residentWorkers.to_numpy('f4')
# --- derived geography
tm = t.to_crs(26916)
col['areaKm2'] = (tm.area / 1e6).to_numpy('f4')
cam = ca.assign(id=ca.area_numbe.astype(int)).to_crs(26916)
j = gpd.sjoin(gpd.GeoDataFrame(geometry=tm.representative_point()), cam[['id', 'geometry']], predicate='within', how='left')
j = j[~j.index.duplicated()]
col['communityArea'] = j['id'].fillna(0).to_numpy('u1')

# --- write
d = out_dir(ID)
tg = t.copy(); tg['geometry'] = shapely.set_precision(tg.geometry, 1e-6)
verts, ringOff, polyRing, partFeat = polygons_to_binary(tg.geometry)
m = {'id': ID, 'version': 1, 'kind': 'polygons', 'count': len(t), 'bbox': [round(float(x), 5) for x in t.total_bounds], 'crs': 'EPSG:4326', 'columns': {}}
write_col(d, m, 'vertices', verts, 'float32', 'vertices.bin', 2)
write_col(d, m, 'ringOffsets', ringOff, 'uint32', 'ringOffsets.bin')
write_col(d, m, 'polygonRingOffsets', polyRing, 'uint32', 'polygonRingOffsets.bin')
write_col(d, m, 'partFeature', partFeat, 'uint32', 'partFeature.bin')
for k in col.columns:
    if k == 'communityArea':
        write_col(d, m, k, col[k], 'uint8', f'{k}.bin')
    else:
        write_col(d, m, k, col[k], 'float32', f'{k}.bin')
gj = tg[['GEOID', 'geometry']].copy()
for k in ['population', 'perCapitaIncome', 'natureObs2023', 'communityArea']:
    gj[k] = col[k].round(0).astype('Int64') if k != 'perCapitaIncome' else col[k].round(0).astype('Int64')
gj['index'] = range(len(gj))
gj.to_file(f'{d}/tracts.geojson', driver='GeoJSON', COORDINATE_PRECISION=6)
tab = col.copy(); tab.insert(0, 'GEOID', t.GEOID)
tab.to_csv(f'{d}/attributes.csv', index=False, float_format='%.6g', na_rep='')
# neighbour sanity: count shared-edge queen/rook neighbours
from libpysal import weights
q = weights.Queen.from_dataframe(tg, use_index=False, silence_warnings=True)
r = weights.Rook.from_dataframe(tg, use_index=False, silence_warnings=True)
print('queen mean nbrs', q.mean_neighbors, 'rook', r.mean_neighbors, 'islands q/r', len(q.islands), len(r.islands))
topc = tab.sort_values('natureObs2023', ascending=False).head(3)[['GEOID', 'natureObs2023', 'speciesRichness2023']]
print(topc)
m['geometry'] = {'type': 'polygons', 'file': 'tracts.geojson'}
m['geoid'] = list(t.GEOID)
m['attributesCsv'] = 'attributes.csv'
m['properties'] = {
  'description': '2020 Census tracts of Cook County whose representative point is inside the City of Chicago (whole tracts, not clipped, so shared edges are exact). Feature order = sorted GEOID; `geoid` array lists them. Polygon parts via partFeature. Each attribute is a float32 column (NaN = suppressed/not available); communityArea is uint8 (0 = none).',
  'columnNotes': {
    'population..limitedEnglish': 'CDC/ATSDR SVI 2022 (ACS 2018-2022 5-yr); *Pct are percentages; poverty150 = persons below 150% poverty line',
    'hispanic,nhWhite,nhBlack,nhAsian,nhOther': 'SVI 2022 race/ethnicity counts; nhWhite derived = population minus the other groups; nhOther = AIAN + NHPI + two-or-more + other',
    'perCapitaIncome,medianHouseholdIncome': 'ACS 2020-2024 5-yr tables B19301/B19013 (US dollars) via Census Reporter; NaN where suppressed',
    'obesity..bingeDrinking': 'CDC PLACES (release 2025, data year 2023 BRFSS model-based) crude prevalence in percent of adults',
    'natureObs2023..speciesRichness2023': 'Counts of 2023 wild iNaturalist observations in the tract (all, birds, plants, insects, fungi, introduced taxa) and distinct taxa observed',
    'jobsWac2021,residentWorkers2021': 'LEHD LODES8 2021: jobs by workplace tract (WAC) and workers living in tract (OD)',
    'communityArea': 'community area id (1-77) containing the tract representative point'},
  'licenses': {'boundaries': 'US Census Bureau cartographic boundary file cb_2022_17_tract_500k, public domain', 'svi': 'CDC/ATSDR SVI 2022, public domain', 'places': 'CDC PLACES, public domain', 'acs': 'US Census Bureau ACS, public domain', 'nature': 'iNaturalist observations, CC0 / CC BY 4.0 / CC BY-NC 4.0 per observation (iNaturalist contributors)'}}
sd = {'queenMeanNeighbours': q.mean_neighbors, 'rookMeanNeighbours': r.mean_neighbors}
m['properties']['topology'] = sd
m['properties']['storyNotes'] = []
json.dump(m, open('/dev/null', 'w'))
STORY = []
caname = dict(zip(ca.area_numbe.astype(int), ca.community.str.title()))
cname = lambda i: caname.get(int(col.communityArea[i]), 'n/a')
pop = col.population
STORY.append(f'{len(t)} tracts; city population in tracts = {int(np.nansum(pop)):,} (SVI 2022).')
hi = col.perCapitaIncome.idxmax(); lo = col.perCapitaIncome.idxmin()
STORY.append(f'Per-capita income ranges from ${col.perCapitaIncome.min():,.0f} (tract {t.GEOID[lo]}) to ${col.perCapitaIncome.max():,.0f} (tract {t.GEOID[hi]}): the highest tract is in {cname(hi)}, lowest in {cname(lo)}.')
P = tot
seg = lambda a: float(0.5 * np.nansum(np.abs(a / np.nansum(a) - (P - a) / (np.nansum(P) - np.nansum(a)))))

STORY.append(f'Citywide dissimilarity index (tract level): Black vs rest {seg(col.nhBlack.to_numpy("f8")):.2f}, Hispanic vs rest {seg(col.hispanic.to_numpy("f8")):.2f}, White vs rest {seg(col.nhWhite.to_numpy("f8")):.2f}: Chicago remains one of the most segregated big cities.')
i0 = int(topc.index[0])
STORY.append(f'Most observed tract 2023: {topc.GEOID.iloc[0]} ({cname(i0)}) with {int(topc.natureObs2023.iloc[0])} wild observations of {int(topc.speciesRichness2023.iloc[0])} taxa. Counts follow observer effort (parks, lakefront), so compare per-tract richness with empirical-Bayes smoothing rather than raw counts.')
isl = [int(q.islands[0])] if len(q.islands) else []
if isl: STORY.append(f'Queen-contiguity island: tract {t.GEOID[isl[0]]} ({cname(isl[0])}) has no neighbour; use distance/kNN weights or handle islands explicitly.')
corr = np.corrcoef(*[np.log1p(col[k].fillna(col[k].median())) for k in ['diabetes', 'perCapitaIncome']])[0, 1]
STORY.append(f'PLACES diabetes prevalence vs log per-capita income correlation = {corr:.2f}: a strong spatially structured health gradient suited to Moran/GWR.')
STORY.append(f'Queen contiguity averages {q.mean_neighbors:.1f} neighbours (rook {r.mean_neighbors:.1f}); islands: queen {len(q.islands)}, rook {len(r.islands)}.')
m['properties']['storyNotes'] = STORY
for s_ in STORY: print('-', s_)
write_manifest(d, m)
