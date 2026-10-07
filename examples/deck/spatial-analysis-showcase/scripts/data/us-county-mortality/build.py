#!/usr/bin/env python
"""Build `us-county-mortality`: motor-vehicle crash deaths per county, 2017-2023 (NHTSA FARS), aligned to us-counties.

Rare, over-dispersed events (median county has a few dozen deaths in 7 years, many have <10), so raw rates are
unstable in small rural counties: a textbook empirical-Bayes / spatial-rate-smoothing target.
Source: NHTSA Fatality Analysis Reporting System (FARS) annual national CSV, ACCIDENT file (public domain US Gov).
Run build us-counties first.
"""
import glob, json, os, subprocess, zipfile
import numpy as np, pandas as pd, geopandas as gpd

SC = '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase'
RAW = f'{SC}/raw/us-county-mortality'
APP = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
OUT = f'{APP}/public/data/us-county-mortality'; CTY = f'{APP}/public/data/us-counties'
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
YEARS = list(range(2017, 2024))

frames = []
for y in YEARS:
    zp = f'{RAW}/fars{y}.zip'
    if not os.path.exists(zp):
        subprocess.check_call(['curl', '-sSL', '-o', zp, f'https://static.nhtsa.gov/nhtsa/downloads/FARS/{y}/National/FARS{y}NationalCSV.zip'])
    d = f'{RAW}/x/{y}'
    if not os.path.exists(d):
        os.makedirs(d)
        with zipfile.ZipFile(zp) as z:
            for nm in z.namelist():
                if nm.lower().endswith('accident.csv'):
                    open(f'{d}/accident.csv', 'wb').write(z.read(nm))
    a = pd.read_csv(f'{d}/accident.csv', encoding='utf-8-sig', encoding_errors='replace',
                    usecols=['STATE', 'COUNTY', 'FATALS', 'RUR_URB', 'LATITUDE', 'LONGITUD', 'YEAR'])
    a['year'] = y
    frames.append(a)
a = pd.concat(frames, ignore_index=True)
a['fips'] = a.STATE.astype(int).astype(str).str.zfill(2) + a.COUNTY.astype(int).astype(str).str.zfill(3)
print('crashes', len(a), 'deaths', int(a.FATALS.sum()))

names = json.load(open(f'{CTY}/names.json')); fips = np.array(names['fips']); n = len(fips); ix = pd.Index(fips)

# Connecticut: FARS uses the 8 legacy counties; us-counties carries the 9 planning regions -> assign crashes by lat/lon.
ct = a.STATE == 9
SHP5 = f'{SC}/raw/us-counties/shp5/cb_2022_us_county_5m.shp'
if ct.any() and not os.path.exists(SHP5):
    os.makedirs(f'{SC}/raw/us-counties/shp5', exist_ok=True)
    zp5 = f'{SC}/raw/us-counties/cb_2022_us_county_5m.zip'
    if not os.path.exists(zp5):
        subprocess.check_call(['curl', '-sSL', '-o', zp5, 'https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_5m.zip'])
    zipfile.ZipFile(zp5).extractall(f'{SC}/raw/us-counties/shp5')
if ct.any():
    g5 = gpd.read_file(f'{SC}/raw/us-counties/shp5/cb_2022_us_county_5m.shp') if os.path.exists(f'{SC}/raw/us-counties/shp5/cb_2022_us_county_5m.shp') else None
    if g5 is not None:
        g5 = g5[g5.STATEFP == '09'][['GEOID', 'geometry']].to_crs(4326)
        pts = gpd.GeoDataFrame(a.loc[ct, ['LATITUDE', 'LONGITUD']].copy(),
                               geometry=gpd.points_from_xy(a.loc[ct, 'LONGITUD'], a.loc[ct, 'LATITUDE']), crs=4326)
        pts = pts[(pts.LATITUDE < 90) & (pts.LONGITUD > -180) & (pts.LONGITUD < 0)]
        j = gpd.sjoin(pts, g5, how='left', predicate='within')
        j = j[~j.index.duplicated()]
        a.loc[j.index, 'fips'] = j.GEOID.fillna('09000')
        a.loc[ct & ~a.index.isin(j.index), 'fips'] = '09000'
        print('CT crashes assigned by location:', int(j.GEOID.notna().sum()), 'of', int(ct.sum()))

a = a[a.fips.isin(ix)]
by = a.groupby('fips')
deaths_year = a.pivot_table(index='fips', columns='year', values='FATALS', aggfunc='sum').reindex(ix).fillna(0)
deaths = deaths_year.sum(axis=1).values
rural = a[a.RUR_URB == 1].groupby('fips').FATALS.sum().reindex(ix).fillna(0).values
crashes = by.size().reindex(ix).fillna(0).values
pop = np.fromfile(f'{CTY}/population2020.bin', dtype=np.float32).astype(float)
if np.isnan(pop).any(): print('NaN population for', int(np.isnan(pop).sum()), 'counties')

cols = {}
def add(name, arr, desc, unit=None, dtype='float32'):
    arr = np.asarray(arr).astype(dtype); assert len(arr) == n
    arr.tofile(f'{OUT}/{name}.bin')
    c = {'file': f'{name}.bin', 'dtype': dtype, 'components': 1, 'length': n, 'description': desc}
    if unit: c['unit'] = unit
    cols[name] = c
add('deaths', deaths, f'Motor-vehicle crash deaths {YEARS[0]}-{YEARS[-1]} (sum of FARS FATALS, county of crash)', 'deaths')
add('population', pop, '2020 Census resident population (exposure; person-years = 7 x population)', 'persons')
add('personYears', pop * len(YEARS), 'Population at risk, person-years over the 7 years', 'person-years')
add('fatalCrashes', crashes, 'Number of fatal crashes in the 7 years', 'crashes')
add('ruralDeaths', rural, 'Deaths in crashes coded rural (FARS RUR_URB=1)', 'deaths')
add('rawRate', deaths / (pop * len(YEARS)) * 1e5, 'Raw annual deaths per 100,000 residents (unsmoothed: unstable where population is small)', 'per 100k per year')
for y in YEARS:
    add(f'deaths{y}', deaths_year[y].values, f'Motor-vehicle crash deaths in {y}', 'deaths')

# ---- story
nm = lambda i: f"{names['name'][i]}, {names['state'][i]}"
raw = deaths / (pop * len(YEARS)) * 1e5
valid = ~np.isnan(raw) & (pop > 0)
small = valid & (pop < 5000); big = valid & (pop > 100000)
state_nat = deaths[valid].sum() / (pop[valid].sum() * len(YEARS)) * 1e5
hi = np.where(valid)[0][np.argsort(-raw[valid])][:5]
hi_big = np.where(valid & (pop > 100000))[0][np.argsort(-raw[valid & (pop > 100000)])][:4]
dsum = deaths_year.sum(axis=0)
story = [
  f'{int(deaths[valid].sum()):,} deaths in {len(YEARS)} years; national raw rate {state_nat:.1f} per 100k per year.',
  f'Small counties (<5,000 residents, n={int(small.sum())}): raw rates range {np.nanpercentile(raw[small],5):.0f}-{np.nanpercentile(raw[small],95):.0f} per 100k (5th-95th pct) vs {np.nanpercentile(raw[big],5):.0f}-{np.nanpercentile(raw[big],95):.0f} for counties over 100,000 (n={int(big.sum())}); the spread is mostly sampling noise, ideal for empirical Bayes shrinkage.',
  'Highest raw rates (mostly tiny counties, unreliable): ' + '; '.join(f'{nm(i)} {raw[i]:.0f}' for i in hi) + '.',
  'Highest raw rates among counties over 100,000 residents: ' + '; '.join(f'{nm(i)} {raw[i]:.1f}' for i in hi_big) + '.',
  f'{int((deaths[valid]==0).sum())} counties had zero deaths in 7 years (rate exactly 0, a classic EB artefact); {int((deaths[valid]<10).sum())} had fewer than 10.',
  'Annual deaths in the covered counties: ' + ', '.join(f'{y}: {int(dsum[y]):,}' for y in YEARS) + ' (2020-2021 jump despite less driving).',
]
manifest = {
  'id': 'us-county-mortality', 'version': 1, 'kind': 'table', 'count': n, 'crs': 'EPSG:4326',
  'bbox': json.load(open(f'{CTY}/manifest.json'))['bbox'], 'columns': cols,
  'properties': {
    'alignedTo': 'us-counties (same row order, sorted by FIPS)',
    'event': 'Motor-vehicle traffic crash deaths (FARS), county where the crash occurred',
    'usage': 'Empirical Bayes: counts = deaths, population at risk = personYears (or population). Rate = deaths / personYears.',
    'why': 'CDC WONDER cannot be fetched programmatically and CDC county overdose data suppress counts of 1-9; FARS has exact unsuppressed counts for every county.',
    'caveats': ['Counts are deaths by county of crash, not residence; population is 2020 Census residents, so tourist/transit counties are inflated.',
                'Connecticut crashes are assigned to the 9 planning regions by crash latitude/longitude (raw 5m boundaries); Alaska/Hawaii excluded.',
                'Crashes with a county code that does not match a us-counties FIPS are dropped.'],
    'sources': 'NHTSA FARS 2017-2023 final annual files (static.nhtsa.gov/nhtsa/downloads/FARS), public domain; population: US Census 2020 via USDA ERS',
    'storyNotes': story,
  }
}
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
print('\n'.join(story)); print('dropped crashes outside set:', len(frames) and int(sum(len(f) for f in frames) - len(a)), 'MB', sum(os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)) / 1e6)
