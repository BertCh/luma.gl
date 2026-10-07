#!/usr/bin/env python
"""Build `us-county-births`: births per county 2021-2023 and the women aged 15-44 at risk, aligned to us-counties.

The general fertility rate (births per 1,000 women aged 15-44 per year) is a rate of counted events over a
population at risk. Most counties have hundreds of births, but a few hundred small counties have a handful,
so their raw rates swing widely: a wholesome small-number target for empirical-Bayes rate smoothing (it
replaces the traffic-death counts of `us-county-mortality` in statistics/rate-smoothing).

Source: US Census Bureau Population Estimates Program, Vintage 2023 (public domain):
  co-est2023-alldata.csv          components of change per county: BIRTHS2021..2023 (births by mother's residence)
  cc-est2023-agesex-all.csv       county characteristics by age and sex: AGE1544_FEM per July 1 estimate
Vintage 2023 already uses the 9 Connecticut planning regions, the same units as us-counties.
Run from any directory with the geo venv: `python -I build.py` then `python -I validate.py`.
Downloads are cached in RAW (outside the app tree). The HTTP User-Agent is a generic project string.
"""
import json, os, subprocess
import numpy as np, pandas as pd

RAW = os.environ.get('US_COUNTY_BIRTHS_RAW', '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/14421614-4041-4f9e-a118-5d838928ca9c/scratchpad/fidelity/raw/us-county-births')
APP = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
OUT = f'{APP}/public/data/us-county-births'; CTY = f'{APP}/public/data/us-counties'
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
UA = 'luma-showcase-data-build/1.0 (https://github.com/visgl/luma.gl)'
BASE = 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties'
FILES = {'co-est2023-alldata.csv': f'{BASE}/totals/co-est2023-alldata.csv',
         'cc-est2023-agesex-all.csv': f'{BASE}/asrh/cc-est2023-agesex-all.csv'}
for name, url in FILES.items():
    if not os.path.exists(f'{RAW}/{name}'):
        subprocess.check_call(['curl', '-sSL', '-A', UA, '-o', f'{RAW}/{name}', url])

YEARS = [2021, 2022, 2023]
YEAR_CODE = {2021: 3, 2022: 4, 2023: 5}  # cc-est YEAR codes: 3..5 = July 1 2021..2023

names = json.load(open(f'{CTY}/names.json')); fips = names['fips']; n = len(fips)
totals = pd.read_csv(f'{RAW}/co-est2023-alldata.csv', dtype={'STATE': str, 'COUNTY': str}, encoding='latin-1')
totals = totals[totals.SUMLEV == 50].copy(); totals['fips'] = totals.STATE + totals.COUNTY
totals = totals.set_index('fips').reindex(fips)
agesex = pd.read_csv(f'{RAW}/cc-est2023-agesex-all.csv', dtype={'STATE': str, 'COUNTY': str}, encoding='latin-1')
agesex['fips'] = agesex.STATE + agesex.COUNTY
women = {y: agesex[agesex.YEAR == YEAR_CODE[y]].set_index('fips').reindex(fips).AGE1544_FEM.astype(float) for y in YEARS}
missing = int(totals.BIRTHS2023.isna().sum()) + int(sum(women[y].isna().sum() for y in YEARS))
assert missing == 0, f'{missing} us-counties rows have no PEP match'

births_year = {y: totals[f'BIRTHS{y}'].astype(float).values for y in YEARS}
births = sum(births_year.values())
women_years = sum(women[y].values for y in YEARS)
population = totals[[f'POPESTIMATE{y}' for y in YEARS]].astype(float).mean(axis=1).values

cols = {}
def add(name, arr, desc, unit=None):
    arr = np.asarray(arr, dtype=np.float32); assert len(arr) == n
    arr.tofile(f'{OUT}/{name}.bin')
    c = {'file': f'{name}.bin', 'dtype': 'float32', 'components': 1, 'length': n, 'description': desc}
    if unit: c['unit'] = unit
    cols[name] = c

add('births', births, 'Live births 2021-2023 to mothers resident in the county (PEP components of change)', 'births')
add('womenYears', women_years, 'Women aged 15-44, summed over the July 1 estimates of 2021, 2022 and 2023 (person-years at risk)', 'woman-years')
add('population', population, 'Resident population, mean of the July 1 estimates 2021-2023', 'persons')
add('rawRate', births / women_years * 1000, 'General fertility rate: births per 1,000 women aged 15-44 per year, 2021-2023 (unsmoothed)', 'per 1,000 women per year')
for y in YEARS:
    add(f'births{y}', births_year[y], f'Live births in {y} (July 1 {y - 1} to June 30 {y} estimate year)', 'births')
    add(f'women{y}', women[y].values, f'Women aged 15-44 on July 1 {y}', 'women')

# ---- story notes (measured, for the coordinator; the scene reads live readouts instead)
nm = lambda i: f"{names['name'][i]}, {names['state'][i]}"
raw = births / women_years * 1000
pooled = births.sum() / women_years.sum() * 1000
top = raw >= np.quantile(raw, 0.95); bottom = raw <= np.quantile(raw, 0.05)
small = population < 5000
order = np.argsort(-raw)
story = [
  f'{int(births.sum()):,} births in 2021-2023; pooled general fertility rate {pooled:.1f} per 1,000 women aged 15-44 per year.',
  f'Median county population {np.median(population):,.0f}; median population of the top 5 % raw rates {np.median(population[top]):,.0f}, of the bottom 5 % {np.median(population[bottom]):,.0f} (the low end is pulled by college towns, whose students inflate the denominator).',
  f'Counties under 5,000 residents (n={int(small.sum())}): raw rates {np.percentile(raw[small], 5):.0f}-{np.percentile(raw[small], 95):.0f} (5th-95th pct) vs {np.percentile(raw[population > 100000], 5):.0f}-{np.percentile(raw[population > 100000], 95):.0f} over 100,000.',
  'Highest raw rates: ' + '; '.join(f'{nm(i)} {raw[i]:.0f} ({int(births[i])} births)' for i in order[:5]) + '.',
  'Lowest raw rates: ' + '; '.join(f'{nm(i)} {raw[i]:.0f} ({int(births[i])} births)' for i in order[::-1][:5]) + '.',
  f'{int((births == 0).sum())} counties recorded no births in three years; {int((births < 30).sum())} fewer than 30.',
]
manifest = {
  'id': 'us-county-births', 'version': 1, 'kind': 'table', 'count': n, 'crs': 'EPSG:4326',
  'bbox': json.load(open(f'{CTY}/manifest.json'))['bbox'], 'columns': cols,
  'properties': {
    'alignedTo': 'us-counties (same row order, sorted by FIPS)',
    'event': 'Live births by county of the mother\'s residence (Census PEP Vintage 2023 components of change)',
    'usage': 'Empirical Bayes: events = births, population at risk = womenYears. Rate = births / womenYears x 1,000 = general fertility rate per year.',
    'caveats': ['PEP births come from NCHS birth records; the most recent year is partly estimated.',
                'Women 15-44 is a model-based estimate; in college towns students inflate it and lower the rate.',
                'Contiguous US + DC; Connecticut as its 9 planning regions (Vintage 2023).'],
    'sources': 'US Census Bureau, Population Estimates Program Vintage 2023 (co-est2023-alldata, cc-est2023-agesex-all), public domain',
    'storyNotes': story,
  }
}
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
print('\n'.join(story)); print('MB', sum(os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)) / 1e6)
