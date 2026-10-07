#!/usr/bin/env python
"""Build `us-elections`: county presidential returns 2000-2024, aligned to the us-counties order.

2000-2020: MIT Election Data + Science Lab, "County Presidential Election Returns 2000-2020"
           (doi:10.7910/DVN/VOQCHQ v14, CC0). Retrieved from a verbatim copy in the stiles/presidential-elections
           repo (data/raw/countypres_2000-2020.csv), because the Dataverse download needs a guestbook form.
2024:      rebuilt from MEDSL precinct returns (medsl2024_aggregate.py), CC0. Only counties whose state aggregate
           reconciles with MEDSL's own state totals are kept; others are NaN.
Run build us-counties first.
"""
import json, os, re, subprocess, sys
import numpy as np, pandas as pd

SC = '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase'
RAW = f'{SC}/raw/us-elections'
APP = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
OUT = f'{APP}/public/data/us-elections'; CTY = f'{APP}/public/data/us-counties'
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
MIT = f'{RAW}/countypres_2000-2020.csv'
if not os.path.exists(MIT):
    subprocess.check_call(['curl', '-sSL', '-o', MIT,
        'https://raw.githubusercontent.com/stiles/presidential-elections/main/data/raw/countypres_2000-2020.csv'])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(__file__), 'medsl2024_aggregate.py')])

names = json.load(open(f'{CTY}/names.json')); fips = np.array(names['fips']); n = len(fips)
ix = pd.Index(fips)
YEARS = [2000, 2004, 2008, 2012, 2016, 2020, 2024]
FIPS_FIX = {'46113': '46102', '51515': '51019'}  # Shannon->Oglala Lakota; Bedford city folded into Bedford county

# ----------------------------------------------------------------- 2000-2020 (MIT)
m = pd.read_csv(MIT, dtype={'county_fips': str})
m['f'] = m.county_fips.fillna('').str.zfill(5).replace(FIPS_FIX)
res = {}  # year -> DataFrame(fips -> d, r, t)
for y, g in m.groupby('year'):
    rows = {}
    for f, h in g.groupby('f'):
        if f not in ix: continue
        if (h['mode'] == 'TOTAL').any(): h = h[h['mode'] == 'TOTAL']
        t = h.totalvotes.max()  # totalvotes repeats the county total on every mode row
        d = h[h.party == 'DEMOCRAT'].candidatevotes.sum(); r = h[h.party == 'REPUBLICAN'].candidatevotes.sum()
        if f in rows:  # merged counties (Bedford city + county)
            d0, r0, t0 = rows[f]; d, r, t = d + d0, r + r0, t + t0
        rows[f] = (d, r, t)
    res[y] = pd.DataFrame.from_dict(rows, orient='index', columns=['d', 'r', 't'])
    print(y, 'counties matched', len(rows), 'unmatched in geometry:',
          sorted(set(g.f) - set(ix) - {'00000'})[:6], '...', len(set(g.f) - set(ix)))

# ----------------------------------------------------------------- 2024 (MEDSL precincts)
p = pd.read_csv(f'{RAW}/pres2024_cand_mode.csv', dtype={'county_fips': str}, keep_default_na=False, na_values=[''])
for c in ['county_fips']: p[c] = p[c].fillna('NA')
META = re.compile(r'UNDER|OVER ?VOTE|TOTAL|CAST|BLANK|WRITE|SCATTER|REGISTERED|BALLOT|TURNOUT|^NA$|NONE OF|N/A', re.I)
p = p[~p.candidate.str.contains(META)].copy()
p['cls'] = np.where(p.candidate.str.contains('HARRIS', case=False), 'D', np.where(p.candidate.str.contains('TRUMP', case=False), 'R', 'O'))
sv = pd.read_csv(f'{RAW}/medsl2024/pres-state.csv')
has_total = sv.groupby('state_po')['mode'].transform(lambda s: (s == 'TOTAL').any())
sv = sv[(sv['mode'] == 'TOTAL') | ~has_total]  # states without a TOTAL row report modes only
sv['cls'] = sv.candidate.str.upper().map(lambda x: 'D' if 'HARRIS' in x else 'R' if 'TRUMP' in x else 'O')
svt = sv.groupby(['state_po', 'cls']).votes.sum().unstack(); svt.index = svt.index.str.lower()

def county_option(g, opt):
    """opt A: TOTAL rows only; B: all modes summed; C: non-TOTAL modes summed."""
    if opt == 'A':
        g = g[g['mode'] == 'TOTAL']
    elif opt == 'C':
        g = g[g['mode'] != 'TOTAL']
    s = g.groupby('cls').v.sum()
    return s.get('D', 0.0), s.get('R', 0.0), s.sum(), g.nstar.sum()

rows24, status = {}, {}
for st, sg in p.groupby('st'):
    best = None
    for opt in 'ABC':
        tot = {'D': 0, 'R': 0}; cand = {}
        for f, g in sg.groupby('county_fips'):
            d, r, t, ns = county_option(g, opt); tot['D'] += d; tot['R'] += r
            if t > 0: cand[f] = (d, r, t, ns)
        err = max(abs(tot['D'] / svt.loc[st, 'D'] - 1), abs(tot['R'] / svt.loc[st, 'R'] - 1))
        if best is None or err < best[0]: best = (err, opt, cand)
    err, opt, cand = best
    ok = err < 0.015 and st != 'ak'
    status[st] = (opt, round(err, 4), ok)
    if ok:
        for f, (d, r, t, ns) in cand.items():
            if f in ix: rows24[f] = (d, r, t)
print('2024 per-state option/err/accepted:', {k: v for k, v in status.items() if not v[2] or v[0] != 'A'})
print('2024 states rejected:', [k for k, v in status.items() if not v[2]])
rows24.pop('29095', None)  # Jackson MO: Kansas City board of elections reports separately, precinct data undercounts by ~50%
res[2024] = pd.DataFrame.from_dict(rows24, orient='index', columns=['d', 'r', 't'])

# ----------------------------------------------------------------- columns
pop = np.fromfile(f'{CTY}/population2020.bin', dtype=np.float32)
cols = {}
def add(name, arr, desc, unit=None):
    arr = np.asarray(arr, dtype=np.float32); assert len(arr) == n
    arr.tofile(f'{OUT}/{name}.bin')
    c = {'file': f'{name}.bin', 'dtype': 'float32', 'components': 1, 'length': n, 'description': desc}
    if unit: c['unit'] = unit
    cols[name] = c
D, R, T = {}, {}, {}
for y in YEARS:
    df = res[y].reindex(ix)
    d, r, t = df.d.values.astype(float), df.r.values.astype(float), df.t.values.astype(float)
    t = np.where(t > 0, t, np.nan)
    D[y], R[y], T[y] = d, r, t
    add(f'demShare{y}', d / t, f'Democratic share of all votes cast for president, {y}', 'fraction')
    add(f'repShare{y}', r / t, f'Republican share of all votes cast for president, {y}', 'fraction')
    add(f'demTwoParty{y}', d / (d + r), f'Democratic / (Democratic + Republican) votes, {y}', 'fraction')
    add(f'totalVotes{y}', t, f'Total presidential votes cast in the county, {y}', 'votes')
    add(f'turnoutProxy{y}', t / pop, f'Votes cast / 2020 Census resident population (not voting-age population; only comparable across counties within a year, '
        'population growth inflates later years)', 'ratio')
for a, b in [(2012, 2016), (2016, 2020), (2020, 2024), (2000, 2024)]:
    sa = R[a] / (R[a] + D[a]); sb = R[b] / (R[b] + D[b])
    add(f'repSwing{a}_{b}', (sb - sa) * 100, f'Change in Republican two-party share {a}->{b}, percentage points (positive = shift to Republican)', 'percentage points')

nanc = {k: int(np.isnan(np.fromfile(f'{OUT}/{c["file"]}', dtype=np.float32)).sum()) for k, c in cols.items()}
print('NaN per column:', {k: v for k, v in nanc.items() if k.endswith(('2000', '2020', '2024')) and k.startswith('dem')})

# ----------------------------------------------------------------- story notes
def top(arr, k=3, hi=True, minvotes=20000, year=2024):
    a = np.where(T[year] >= minvotes, arr, np.nan); o = np.argsort(np.where(np.isnan(a), -np.inf if hi else np.inf, a))
    o = o[::-1] if not hi else o[::-1]
    return a, o
nm = lambda i: f"{names['name'][i]}, {names['state'][i]}"
sw = np.fromfile(f'{OUT}/repSwing2020_2024.bin', dtype=np.float32)
big = (T[2024] >= 50000) & ~np.isnan(sw)
idx = np.where(big)[0]
lead_r = idx[np.argsort(-sw[idx])][:4]; lead_d = idx[np.argsort(sw[idx])][:4]
sw1620 = np.fromfile(f'{OUT}/repSwing2016_2020.bin', dtype=np.float32)
n_flip = {}
for a, b in [(2012, 2016), (2016, 2020), (2020, 2024)]:
    ok = ~np.isnan(D[a]) & ~np.isnan(D[b]) & (T[a] > 0) & (T[b] > 0)
    fl = ok & ((D[a] > R[a]) != (D[b] > R[b])); n_flip[(a, b)] = (int(fl.sum()), int(ok.sum()))
cov24 = int((~np.isnan(T[2024])).sum())
story = [
  f'{cov24} of {n} counties carry 2024 returns (states rejected because MEDSL precinct files are incomplete: ' + ', '.join(k.upper() for k, v in status.items() if not v[2]) + '); earlier years cover ~3,100.',
  f'County winner changes: 2012->2016 {n_flip[(2012,2016)][0]}, 2016->2020 {n_flip[(2016,2020)][0]}, 2020->2024 {n_flip[(2020,2024)][0]} of ~{n_flip[(2020,2024)][1]} comparable counties.',
  '2020->2024 largest pro-Republican swings among counties with >=50k votes: ' + '; '.join(f'{nm(i)} {sw[i]:+.1f} pp' for i in lead_r) + '.',
  '2020->2024 largest pro-Democratic swings among counties with >=50k votes: ' + '; '.join(f'{nm(i)} {sw[i]:+.1f} pp' for i in lead_d) + '.',
  f'Median Republican two-party swing 2020->2024 = {np.nanmedian(sw):+.1f} pp; 2016->2020 = {np.nanmedian(sw1620):+.1f} pp.',
  'Connecticut is NaN (election returns use 8 legacy counties; us-counties carries 9 planning regions); Broomfield CO is NaN for 2000 (county created 2001).',
]
manifest = {
  'id': 'us-elections', 'version': 1, 'kind': 'table', 'count': n, 'crs': 'EPSG:4326', 'bbox': json.load(open(f'{CTY}/manifest.json'))['bbox'],
  'columns': cols,
  'properties': {
    'alignedTo': 'us-counties (same row order, sorted by FIPS)',
    'years': YEARS,
    'columnPattern': 'demShare<Y>, repShare<Y>, demTwoParty<Y>, totalVotes<Y>, turnoutProxy<Y> for each year, plus repSwing<A>_<B>',
    'sources': '2000-2020: MIT Election Data and Science Lab, County Presidential Election Returns 2000-2020, Harvard Dataverse, doi:10.7910/DVN/VOQCHQ (CC0 1.0). '
               '2024: MIT Election Data and Science Lab precinct-level returns 2024 (github.com/MEDSL/2024-elections-official; CC0), aggregated to county by scripts/data/us-elections.',
    'caveats': ['2024 is rebuilt from precinct data and is not the official MEDSL county file; states failing reconciliation against MEDSL state totals are NaN.',
                'Bedford city VA is folded into Bedford County (2000-2012); Shannon SD is mapped to Oglala Lakota. Alaska and Hawaii are not in us-counties.',
                'Jackson County MO (29095) is NaN for 2024 (Kansas City reported separately in the source).',
                'Missing values are NaN.'],
    'storyNotes': story,
  }
}
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
print('\n'.join(story)); print('total MB', sum(os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)) / 1e6)
