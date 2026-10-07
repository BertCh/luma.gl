import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chicago-tracts'))
from common import *
import pandas as pd, numpy as np, geopandas as gpd

ID = 'chicago-lodes-od'
R = f'{RAW}/chicago-lodes-od'
tm = json.load(open(f'{APP}/public/data/chicago-tracts/manifest.json'))
geoids = np.array([int(g) for g in tm['geoid']]); idx = {g: i for i, g in enumerate(geoids)}
t = gpd.read_file(f'{RAW}/chicago-tracts/tracts_full.gpkg').to_crs(26916)
cent = t.representative_point().to_crs(4326)
loc = np.c_[cent.x, cent.y]
cols = ['w_geocode', 'h_geocode', 'S000', 'SE01', 'SE02', 'SE03']
od = pd.read_csv(f'{R}/od.csv.gz', usecols=cols, dtype={'w_geocode': 'int64', 'h_geocode': 'int64'})
od['w'] = od.w_geocode // 10000; od['h'] = od.h_geocode // 10000
tot_jobs_il = od.S000.sum()
# resident workers per home tract (any workplace in IL) and jobs located in tract (OD side)
res = od.groupby('h')[['S000']].sum().S000
resident = np.array([res.get(g, 0) for g in geoids], 'f4')
wac = pd.read_csv(f'{R}/wac.csv.gz', usecols=['w_geocode', 'C000'], dtype={'w_geocode': 'int64'})
wac['w'] = wac.w_geocode // 10000
wt = wac.groupby('w').C000.sum()
wacJobs = np.array([wt.get(g, 0) for g in geoids], 'f4')
pd.DataFrame({'GEOID': geoids, 'wacJobs': wacJobs, 'residentWorkers': resident}).to_csv(f'{R}/wac_by_tract.csv', index=False)
inn = od[od.w.isin(idx) & od.h.isin(idx)]
g = inn.groupby(['h', 'w'])[['S000', 'SE01', 'SE02', 'SE03']].sum().reset_index()
g['o'] = g.h.map(idx); g['d'] = g.w.map(idx)
selfm = g.o == g.d
selfjobs = np.zeros(len(geoids), 'f4'); selfjobs[g.o[selfm]] = g.S000[selfm]
f = g[~selfm].sort_values('S000', ascending=False)
total_cc = f.S000.sum()
f = f.head(40000)
n = len(f)
d = out_dir(ID)
m = {'id': ID, 'version': 1, 'kind': 'flows', 'count': n, 'bbox': tm['bbox'], 'crs': 'EPSG:4326', 'columns': {}}
write_col(d, m, 'locations', loc, 'float32', 'locations.bin', 2, note='tract representative points, order = chicago-tracts features')
write_col(d, m, 'origin', f.o, 'uint32', 'origin.bin', note='HOME tract index into locations')
write_col(d, m, 'destination', f.d, 'uint32', 'destination.bin', note='WORK tract index into locations')
write_col(d, m, 'count', f.S000, 'uint32', 'count.bin', note='all jobs (S000)')
write_col(d, m, 'earningsLow', f.SE01, 'uint32', 'earningsLow.bin', note='SE01: <= $1,250/month')
write_col(d, m, 'earningsMid', f.SE02, 'uint32', 'earningsMid.bin', note='SE02: $1,251-$3,333/month')
write_col(d, m, 'earningsHigh', f.SE03, 'uint32', 'earningsHigh.bin', note='SE03: > $3,333/month')
write_col(d, m, 'wacJobs', wacJobs, 'float32', 'wacJobs.bin', note='jobs by workplace tract (LODES WAC C000, per tract, length = tracts)', length=len(geoids))
write_col(d, m, 'residentWorkers', resident, 'float32', 'residentWorkers.bin', note='workers living in tract (OD main, any IL workplace)', length=len(geoids))
write_col(d, m, 'sameTractJobs', selfjobs, 'float32', 'sameTractJobs.bin', note='home=work within tract (excluded from flows)', length=len(geoids))
loopi = int(np.argmax(wacJobs)); cst = tm['geoid'][loopi]
cov = f.S000.sum() / total_cc
top = f.iloc[0]
hi_share = f.SE03.sum() / f.S000.sum(); lo_share = f.SE01.sum() / f.S000.sum()
m['properties'] = {
  'description': 'LEHD LODES8 Illinois main OD (JT00, all jobs, 2021) aggregated from blocks to 2020 tract-to-tract flows between the Chicago tracts; the top 40,000 non-self flows by jobs. Origin = home, destination = work. Totals for all tracts are in the per-tract columns.',
  'licence': 'US Census Bureau LEHD LODES, public domain (attribute: U.S. Census Bureau, LEHD Origin-Destination Employment Statistics)',
  'storyNotes': [
    f'{len(g):,} tract pairs have commuters inside Chicago; the top {n:,} flows keep {cov*100:.0f}% of inter-tract Chicago-internal jobs.',
    f'Busiest workplace tract: {cst} with {int(wacJobs[loopi]):,} jobs (the Loop / downtown core); {int(wacJobs.sum()):,} jobs across city tracts.',
    f'Largest single flow: {int(top.S000):,} workers from tract {geoids[int(top.o)]} to {geoids[int(top.d)]}.',
    f'In kept flows {hi_share*100:.0f}% of jobs are high-earning (>$3,333/mo) and {lo_share*100:.0f}% low-earning; the earnings split is in the three earnings columns.',
    f'{int(selfjobs.sum()):,} workers live and work in the same tract (excluded from flows).',
    'LODES blocks are noise-infused for privacy, so small flows are fuzzed; aggregate to tracts (as here) for stable estimates.']}
write_manifest(d, m)
for s in m['properties']['storyNotes']: print('-', s)
