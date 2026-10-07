#!/usr/bin/env python
"""Build the `us-counties` dataset (contiguous US counties + socioeconomic/health attributes).

Run:  <geo-venv>/bin/python -I build.py   (downloads cached in RAW; re-runnable)
"""
import json, os, sys, urllib.request, zipfile, io
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import mapping, Polygon, MultiPolygon
from shapely.geometry.polygon import orient

SC = '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase'
RAW = f'{SC}/raw/us-counties'
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../public/data/us-counties'))
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)

URLS = {
  'cb_2022_us_county_20m.zip': 'https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip',
  'SVI_2022_US_county.csv': 'https://svi.cdc.gov/Documents/Data/2022/csv/states_counties/SVI_2022_US_county.csv',
  'rucc.csv': 'https://www.ers.usda.gov/media/5768/2023-rural-urban-continuum-codes.csv',
  'unemp.xlsx': 'https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/48747/Unemployment2023.xlsx',
}
PLACES_URL = ('https://data.cdc.gov/resource/fu4u-a9bh.csv?$limit=300000&$where=data_value_type%20in'
  "('Age-adjusted%20prevalence','Crude%20prevalence')&$select=year,stateabbr,locationid,measureid,data_value_type,data_value,totalpopulation")

def fetch(name, url):
    p = f'{RAW}/{name}'
    if not os.path.exists(p):
        print('download', url); urllib.request.urlretrieve(url, p)
    return p

for n, u in URLS.items(): fetch(n, u)
fetch('places2024.csv', PLACES_URL)
if not os.path.exists(f'{RAW}/shp20'):
    zipfile.ZipFile(f'{RAW}/cb_2022_us_county_20m.zip').extractall(f'{RAW}/shp20')

# ---------------------------------------------------------------- geometry
g = gpd.read_file(f'{RAW}/shp20/cb_2022_us_county_20m.shp')
g = g[~g.STATEFP.isin(['02', '15', '60', '66', '69', '72', '78'])].sort_values('GEOID').reset_index(drop=True)
n = len(g); fips = g.GEOID.values
print('counties (contiguous US + DC):', n)

def polys(geom):
    return list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]

verts, ring_off, poly_ring_off, geom_poly_off = [], [0], [0], [0]
q_geoms = []
for geom in g.geometry:
    qpolys = []
    for pg in polys(geom):
        pg = orient(pg, 1.0)  # CCW exterior, CW holes
        rings = [pg.exterior] + list(pg.interiors)
        for r in rings:
            c = np.asarray(r.coords, dtype=np.float64)
            c = np.round(c, 5)  # shared vertices stay identical after quantisation
            verts.append(c.astype(np.float32))
            ring_off.append(ring_off[-1] + len(c))
        poly_ring_off.append(poly_ring_off[-1] + len(rings))
        qpolys.append(Polygon(np.round(np.asarray(pg.exterior.coords), 5),
                              [np.round(np.asarray(i.coords), 5) for i in pg.interiors]))
    geom_poly_off.append(geom_poly_off[-1] + len(qpolys))
    q_geoms.append(MultiPolygon(qpolys) if len(qpolys) > 1 else qpolys[0])
vertices = np.concatenate(verts)
print('vertices', len(vertices), 'rings', len(ring_off) - 1, 'polygons', len(poly_ring_off) - 1)

bounds = g.total_bounds.tolist()
def wr(name, arr):
    arr.tofile(f'{OUT}/{name}'); return os.path.getsize(f'{OUT}/{name}')

cols = {}
def add(name, arr, dtype='float32', unit=None, desc=None, categories=None):
    arr = np.asarray(arr)
    assert len(arr) == n, (name, len(arr))
    arr = arr.astype(dtype)
    fn = f'{name}.bin'
    wr(fn, arr)
    c = {'file': fn, 'dtype': dtype, 'components': 1, 'length': n}
    if unit: c['unit'] = unit
    if desc: c['description'] = desc
    if categories: c['categories'] = categories
    cols[name] = c

# geometry columns (variable length)
wr('vertices.bin', vertices.astype(np.float32))
wr('ringOffsets.bin', np.array(ring_off, dtype=np.uint32))
wr('polygonRingOffsets.bin', np.array(poly_ring_off, dtype=np.uint32))
wr('countyPolygonOffsets.bin', np.array(geom_poly_off, dtype=np.uint32))
cols['vertices'] = {'file': 'vertices.bin', 'dtype': 'float32', 'components': 2, 'length': len(vertices)}
cols['ringOffsets'] = {'file': 'ringOffsets.bin', 'dtype': 'uint32', 'components': 1, 'length': len(ring_off)}
cols['polygonRingOffsets'] = {'file': 'polygonRingOffsets.bin', 'dtype': 'uint32', 'components': 1, 'length': len(poly_ring_off)}
cols['countyPolygonOffsets'] = {'file': 'countyPolygonOffsets.bin', 'dtype': 'uint32', 'components': 1, 'length': len(geom_poly_off)}

ix = pd.Index(fips)
def align(df, key, col):
    s = df.set_index(key)[col]; s = s[~s.index.duplicated()]
    return s.reindex(ix).values.astype(float)

# ---------------------------------------------------------------- identity / basic
add('fips', fips.astype(np.int64), 'uint32', desc='Five-digit county FIPS as integer (e.g. 1001, 48201)')
add('stateFips', np.array([int(f[:2]) for f in fips]), 'uint8', desc='State FIPS')
cent = g.to_crs(5070).geometry.centroid.to_crs(4326)
cxy = np.column_stack([cent.x.values, cent.y.values]).astype(np.float32)
cxy.tofile(f'{OUT}/centroid.bin')
cols['centroid'] = {'file': 'centroid.bin', 'dtype': 'float32', 'components': 2, 'length': n,
                    'description': 'Area-weighted centroid lon/lat (computed in EPSG:5070)'}
add('landArea', g.ALAND.values / 1e6, unit='km2', desc='Land area (Census cartographic boundary ALAND)')

# ---------------------------------------------------------------- SVI 2022
svi = pd.read_csv(f'{RAW}/SVI_2022_US_county.csv', dtype={'FIPS': str, 'STCNTY': str}, encoding='utf-8-sig')
svi['FIPS'] = svi.STCNTY.str.zfill(5) if 'STCNTY' in svi else svi.FIPS
svi = svi.replace(-999, np.nan)
add('population', align(svi, 'FIPS', 'E_TOTPOP'), desc='Total population, ACS 2018-2022 (via CDC/ATSDR SVI 2022)')
add('popDensity', align(svi, 'FIPS', 'E_TOTPOP') / (g.ALAND.values / 1e6), unit='per km2')
SVI_PCT = {
  'poverty150': ('EP_POV150', 'Percent persons below 150% poverty'),
  'unemployment': ('EP_UNEMP', 'Percent civilian (16+) unemployed, ACS 2018-2022'),
  'housingBurden': ('EP_HBURD', 'Percent housing cost-burdened households'),
  'noHighSchool': ('EP_NOHSDP', 'Percent persons (25+) with no high school diploma'),
  'noHealthInsurance': ('EP_UNINSUR', 'Percent uninsured in civilian noninstitutionalised population'),
  'age65plus': ('EP_AGE65', 'Percent persons aged 65+'),
  'age17under': ('EP_AGE17', 'Percent persons aged 17 and younger'),
  'disability': ('EP_DISABL', 'Percent civilians with a disability'),
  'singleParent': ('EP_SNGPNT', 'Percent single-parent households with children under 18'),
  'limitedEnglish': ('EP_LIMENG', 'Percent persons (5+) with limited English'),
  'minorityShare': ('EP_MINRTY', 'Percent racial or ethnic minority (all but non-Hispanic white alone)'),
  'multiUnit': ('EP_MUNIT', 'Percent housing in structures with 10+ units'),
  'mobileHomes': ('EP_MOBILE', 'Percent mobile homes'),
  'crowding': ('EP_CROWD', 'Percent occupied housing units with more people than rooms'),
  'noVehicle': ('EP_NOVEH', 'Percent households with no vehicle'),
  'groupQuarters': ('EP_GROUPQ', 'Percent persons in group quarters'),
  'noInternet': ('EP_NOINT', 'Percent households without internet subscription'),
}
for name, (c, d) in SVI_PCT.items(): add(name, align(svi, 'FIPS', c), unit='percent', desc=d)
for name, c, d in [('sviOverall', 'RPL_THEMES', 'SVI overall percentile ranking (0-1)'),
                   ('sviTheme1', 'RPL_THEME1', 'SVI theme 1 socioeconomic status percentile'),
                   ('sviTheme2', 'RPL_THEME2', 'SVI theme 2 household characteristics percentile'),
                   ('sviTheme3', 'RPL_THEME3', 'SVI theme 3 racial & ethnic minority status percentile'),
                   ('sviTheme4', 'RPL_THEME4', 'SVI theme 4 housing type & transportation percentile')]:
    add(name, align(svi, 'FIPS', c), unit='percentile 0-1', desc=d)
tot = align(svi, 'FIPS', 'E_TOTPOP')
minority = align(svi, 'FIPS', 'E_MINRTY')
add('popWhiteNH', tot - minority, desc='Non-Hispanic white alone = population - minority (SVI definition)')
for name, c, d in [('popBlack', 'E_AFAM', 'Black/African American non-Hispanic'), ('popHispanic', 'E_HISP', 'Hispanic or Latino'),
                   ('popAsian', 'E_ASIAN', 'Asian non-Hispanic'), ('popAIAN', 'E_AIAN', 'American Indian/Alaska Native non-Hispanic'),
                   ('popNHPI', 'E_NHPI', 'Native Hawaiian/Pacific Islander non-Hispanic'),
                   ('popTwoMore', 'E_TWOMORE', 'Two or more races non-Hispanic'), ('popOtherRace', 'E_OTHERRACE', 'Other race non-Hispanic')]:
    add(name, align(svi, 'FIPS', c), desc=d + ' (counts, ACS 2018-2022)')
add('popMinority', minority, desc='All non-(non-Hispanic white alone) persons')

# ---------------------------------------------------------------- USDA ERS
rucc = pd.read_csv(f'{RAW}/rucc.csv', encoding='latin1', dtype={'FIPS': str})
rucc = rucc[rucc.Attribute == 'RUCC_2023']; rucc['Value'] = pd.to_numeric(rucc.Value, errors='coerce')
add('rucc2023', align(rucc, 'FIPS', 'Value'), 'float32',
    desc='USDA ERS Rural-Urban Continuum Code 2023: 1-3 metro, 4-9 nonmetro (1 = most urban, 9 = most rural)')
pop20 = pd.read_csv(f'{RAW}/rucc.csv', encoding='latin1', dtype={'FIPS': str}); pop20 = pop20[pop20.Attribute == 'Population_2020']
pop20['Value'] = pd.to_numeric(pop20.Value, errors='coerce')
add('population2020', align(pop20, 'FIPS', 'Value'), desc='2020 Census resident population (via USDA ERS)')
ers = pd.read_excel(f'{RAW}/unemp.xlsx', sheet_name=0, header=4, dtype={'FIPS_Code': str})
add('uic2013', align(ers, 'FIPS_Code', 'Urban_Influence_Code_2013'), desc='USDA ERS Urban Influence Code 2013 (1-12)')
add('medianHouseholdIncome', align(ers, 'FIPS_Code', 'Median_Household_Income_2022'), unit='USD 2022',
    desc='Median household income 2022 (Census SAIPE via USDA ERS)')
add('laborForce2023', align(ers, 'FIPS_Code', 'Civilian_labor_force_2023'), desc='Civilian labour force 2023 (BLS LAUS)')
for y in range(2000, 2024):
    add(f'unemploymentRate{y}', align(ers, 'FIPS_Code', f'Unemployment_rate_{y}'), unit='percent',
        desc=f'Annual average unemployment rate {y} (BLS LAUS via USDA ERS)')

# ---------------------------------------------------------------- CDC PLACES
pl = pd.read_csv(f'{RAW}/places2024.csv', dtype={'locationid': str})
pl['data_value'] = pd.to_numeric(pl.data_value, errors='coerce')
KEY = ['DIABETES', 'OBESITY', 'LPA', 'CSMOKING', 'DEPRESSION', 'MHLTH']
LABEL = {'DIABETES': 'diagnosed diabetes', 'OBESITY': 'obesity', 'LPA': 'no leisure-time physical activity',
         'CSMOKING': 'current cigarette smoking', 'DEPRESSION': 'depression', 'MHLTH': 'mental health not good for >=14 days',
         'BPHIGH': 'high blood pressure', 'CHD': 'coronary heart disease', 'COPD': 'COPD', 'STROKE': 'stroke',
         'CANCER': 'cancer (non-skin)', 'CASTHMA': 'current asthma', 'ARTHRITIS': 'arthritis', 'BINGE': 'binge drinking',
         'ACCESS2': 'no health insurance (18-64)', 'CHECKUP': 'annual checkup', 'DENTAL': 'dental visit', 'SLEEP': 'short sleep (<7h)',
         'PHLTH': 'physical health not good for >=14 days', 'GHLTH': 'fair or poor health', 'HIGHCHOL': 'high cholesterol',
         'TEETHLOST': 'all teeth lost (65+)', 'DISABILITY': 'any disability', 'COGNITION': 'cognitive disability',
         'MOBILITY': 'mobility disability', 'VISION': 'vision disability', 'HEARING': 'hearing disability',
         'SELFCARE': 'self-care disability', 'INDEPLIVE': 'independent living disability', 'BPMED': 'taking BP medication',
         'CHOLSCREEN': 'cholesterol screening', 'COLON_SCREEN': 'colorectal cancer screening', 'MAMMOUSE': 'mammography'}
pl_cols = []
for m in sorted(pl.measureid.unique()):
    if m not in LABEL: continue
    aa = pl[(pl.measureid == m) & (pl.data_value_type == 'Age-adjusted prevalence')]
    cr = pl[(pl.measureid == m) & (pl.data_value_type == 'Crude prevalence')]
    if len(aa): add(f'places_{m.lower()}_ageAdj', align(aa, 'locationid', 'data_value'), unit='percent',
                    desc=f'CDC PLACES age-adjusted prevalence: {LABEL[m]} (adults)'); pl_cols.append(f'places_{m.lower()}_ageAdj')
    if m in KEY and len(cr): add(f'places_{m.lower()}_crude', align(cr, 'locationid', 'data_value'), unit='percent',
                    desc=f'CDC PLACES crude prevalence: {LABEL[m]} (adults)')

# ---------------------------------------------------------------- GeoJSON
feat = []
for i in range(n):
    feat.append({'type': 'Feature', 'id': int(fips[i]),
                 'properties': {'fips': fips[i], 'name': g.NAME[i], 'state': g.STUSPS[i]},
                 'geometry': mapping(q_geoms[i])})
gj = json.dumps({'type': 'FeatureCollection', 'features': feat}, separators=(',', ':'))
open(f'{OUT}/counties.geojson', 'w').write(gj)

# ---------------------------------------------------------------- manifest
names = [f'{a}' for a in g.NAME.values]
open(f'{OUT}/names.json', 'w').write(json.dumps({'fips': list(fips), 'name': list(g.NAME.values), 'state': list(g.STUSPS.values)}, separators=(',', ':')))
manifest = {
  'id': 'us-counties', 'version': 1, 'kind': 'polygons', 'count': n,
  'bbox': [round(b, 4) for b in bounds], 'crs': 'EPSG:4326', 'columns': cols,
  'geometry': {'type': 'polygons', 'file': 'counties.geojson',
               'binary': {'countyPolygonOffsets': 'countyPolygonOffsets', 'polygonRingOffsets': 'polygonRingOffsets',
                          'ringOffsets': 'ringOffsets', 'vertices': 'vertices',
                          'note': 'GeoArrow-style: county i owns polygons [cPO[i], cPO[i+1]); polygon p owns rings [pRO[p], pRO[p+1]) (first = exterior, CCW; rest = holes, CW); ring r owns vertices [rO[r], rO[r+1]). Rings are closed (first vertex repeated). Adjacent counties share identical vertices (queen/rook contiguity can be built by exact vertex match).'}},
  'properties': {
    'order': 'sorted by 5-digit FIPS; every other county dataset (us-elections, us-county-mortality) is aligned to this order',
    'names': 'names.json holds fips/name/state arrays in the same order',
    'coverage': 'Contiguous US + DC (3109 counties). Alaska, Hawaii and territories dropped.',
    'boundaries': 'Census cartographic boundary file cb_2022_us_county_20m (1:20,000,000), coordinates rounded to 5 decimals',
    'connecticut': 'Connecticut appears as its 9 planning regions (2022 Census vintage; SVI 2022, PLACES and ERS use them). MIT election returns are reported for the 8 legacy counties, so us-elections columns are NaN for CT.',
    'nan': 'Missing values are NaN (float32).',
    'missingIncomePerCapita': 'ACS per-capita income needs a Census API key; medianHouseholdIncome (SAIPE 2022) shipped instead.',
    'sources': {'boundaries': 'US Census Bureau cb_2022_us_county_20m', 'svi': 'CDC/ATSDR SVI 2022 county (ACS 2018-2022)',
                'places': 'CDC PLACES county data, 2024 release (model-based estimates from BRFSS 2022; a few measures BRFSS 2021)',
                'ers': 'USDA ERS Rural-Urban Continuum Codes 2023; Unemployment & median household income file (BLS LAUS, Census SAIPE)'},
  }
}
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
print('wrote', sum(os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)) / 1e6, 'MB')

import subprocess
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'story.py')])
