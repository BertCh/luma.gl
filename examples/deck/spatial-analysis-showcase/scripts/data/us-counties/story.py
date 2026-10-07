"""Compute story notes for us-counties (queen contiguity from exact shared vertices, Moran, LISA) and write them into manifest.json."""
import json, os, collections
import numpy as np
from collections import defaultdict
import esda
from libpysal.weights import W

D = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../public/data/us-counties'))
m = json.load(open(f'{D}/manifest.json')); n = m['count']; nm = json.load(open(f'{D}/names.json'))
DT = {'float32': np.float32, 'uint32': np.uint32, 'uint8': np.uint8}
col = lambda k: np.fromfile(f"{D}/{m['columns'][k]['file']}", dtype=DT[m['columns'][k]['dtype']])
v = col('vertices').reshape(-1, 2); cpo, pro, ro = col('countyPolygonOffsets'), col('polygonRingOffsets'), col('ringOffsets')
vx = defaultdict(set)
for i in range(n):
    for p in range(cpo[i], cpo[i + 1]):
        for r in range(pro[p], pro[p + 1]):
            for xy in map(tuple, v[ro[r]:ro[r + 1] - 1]): vx[xy].add(i)
nb = defaultdict(set)
for s in vx.values():
    if len(s) > 1:
        for a in s: nb[a] |= s - {a}
w = W({i: sorted(nb[i]) for i in range(n)}, silence_warnings=True); w.transform = 'r'
label = lambda i: f"{nm['name'][i]}, {nm['state'][i]}"
notes = []
dia = col('places_diabetes_ageAdj').astype(float); ok = ~np.isnan(dia)
y = np.where(ok, dia, np.nanmean(dia))
mi = esda.Moran(y, w, permutations=0); ll = esda.Moran_Local(y, w, permutations=499, seed=1)
hh = (ll.q == 1) & (ll.p_sim < 0.01) & ok; ll_ = (ll.q == 3) & (ll.p_sim < 0.01) & ok
st = collections.Counter(np.array(nm['state'])[hh]).most_common(6)
notes.append(f"Age-adjusted diabetes prevalence (CDC PLACES 2024, BRFSS 2022) is strongly clustered: global Moran's I = {mi.I:.2f} on queen contiguity. "
             f"{int(hh.sum())} high-high LISA counties (p<0.01), concentrated in " + ', '.join(f'{s} ({c})' for s, c in st) + " - the Deep South / Black Belt / Appalachian 'diabetes belt'.")
stl = collections.Counter(np.array(nm['state'])[ll_]).most_common(6)
notes.append(f"Diabetes range: {np.nanmin(dia):.1f}% to {np.nanmax(dia):.1f}% (max: {label(int(np.nanargmax(dia)))}); {int(ll_.sum())} low-low counties, led by " + ', '.join(f'{s} ({c})' for s, c in stl) + '.')
inc = col('medianHouseholdIncome').astype(float)
notes.append(f"Median household income 2022 ranges ${np.nanmin(inc):,.0f} ({label(int(np.nanargmin(inc)))}) to ${np.nanmax(inc):,.0f} ({label(int(np.nanargmax(inc)))}); Moran's I = {esda.Moran(np.where(np.isnan(inc), np.nanmean(inc), inc), w, permutations=0).I:.2f}.")
r = np.corrcoef(dia[ok & ~np.isnan(inc)], inc[ok & ~np.isnan(inc)])[0, 1]
notes.append(f"Diabetes vs median household income correlation r = {r:.2f} across counties: a good candidate for OLS residual maps, spatial lag/error and GWR (coefficient varies by region).")
mino = col('minorityShare').astype(float); pop = col('population').astype(float)
notes.append(f"Minority share is as spatially clustered as diabetes (Moran's I = {esda.Moran(np.where(np.isnan(mino), np.nanmean(mino), mino), w, permutations=0).I:.2f}): Black Belt, Texas-border and Native American reservation counties stand out; most populous county {label(int(np.nanargmax(pop)))} ({np.nanmax(pop):,.0f}).")
ru = col('rucc2023').astype(float)
notes.append(f"Rural-urban continuum: {int((ru<=3).sum())} metro counties (codes 1-3) and {int((ru>=4).sum())} nonmetro; mean diabetes prevalence {np.nanmean(dia[ru<=3]):.1f}% metro vs {np.nanmean(dia[ru>=4]):.1f}% nonmetro.")
notes.append(f"Contiguity: {sum(len(nb[i]) for i in range(n))//2:,} queen edges, mean {np.mean([len(nb[i]) for i in range(n)]):.1f} neighbours, max {max(len(nb[i]) for i in range(n))}; 2 true islands (Nantucket MA, San Juan WA) have no neighbours - exercise the island handling of weights.")
notes.append('Connecticut appears as 9 planning regions (2022 vintage); Alaska and Hawaii dropped (3,109 contiguous counties + DC).')
m['properties']['storyNotes'] = notes
json.dump(m, open(f'{D}/manifest.json', 'w'), indent=1)
print('\n'.join(notes))
