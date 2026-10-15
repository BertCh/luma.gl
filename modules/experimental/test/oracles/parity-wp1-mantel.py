# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent full-matrix Mantel reference: skbio.stats.distance.mantel and scipy.

Pearson, strict upper triangle, greater tail; no transforms of distances/times.
All 48636 Chicago rat reports are considered, then lexicographic geographic/time
strata select 2048 and 128 rows at fixed quantiles. EPSG:26916 km, local origin;
coordinates snapped to 1/1024 km and uploaded f32. Distances are scipy pdist
rounded f32 (the API consumes distances); TS reconstructs these INPUT distances
with Math.hypot, not any Mantel statistic. Times are integer days from source
local-clock epoch, exact f32. Full matrices, no truncated neighbor bands.

The API cannot accept supplied permutations: independently generated NumPy
assignments must NOT equal GPU same-seed draws. Store every independent reference
correlation (4095 draws) and use two-sample DKW CDF confidence bounds over all
reference and GPU values, plus tail/binomial uncertainty over 3 GPU seeds.
Observed r uses S. The 2048-row scale case has one draw and checks observed r;
the 128-row case tests inference. A six-row tied/duplicate case uses all 720
permutations (exact null), and constant times explicitly adapt skbio's NaN to
the documented GPU zero-variance r=0. No false claim of identical null draws.
https://scikit.bio/docs/latest/generated/skbio.stats.distance.mantel.html
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import numpy as numpy
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import skbio
from skbio.stats.distance import mantel
import pyproj

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT/'examples/deck/spatial-analysis-showcase/public/data/chicago-311-rats'


def makeCase(name, positions, times, sourceRows, draws, exhaustive=False):
    positions = numpy.asarray(positions, dtype=numpy.float32).astype(float)
    times = numpy.asarray(times, dtype=numpy.float32).astype(float)
    distances = pdist(positions).astype(numpy.float32).astype(float)
    timeDistances = pdist(times[:, None])
    observed, _, count = mantel(squareform(distances), squareform(timeDistances), permutations=0, alternative='greater')
    constant = not numpy.isfinite(observed)
    if constant:
        assert numpy.ptp(times) == 0
        observed = 0.
    random = numpy.random.default_rng(31015)
    assignments = itertools.permutations(range(len(times))) if exhaustive else (random.permutation(len(times)) for _ in range(draws))
    null = []
    for index, assignment in enumerate(assignments):
        permuted = times[list(assignment)]
        difference = pdist(permuted[:, None])
        statistic = 0. if constant else float(pearsonr(distances, difference).statistic)
        if index < 8 and not constant:
            check = mantel(squareform(distances), squareform(difference), permutations=0, alternative='greater')[0]
            assert abs(check-statistic) < 1e-12
        null.append(statistic)
    return {'name': name, 'reference': 'skbio.stats.distance.mantel; scipy.stats.pearsonr',
            'n': int(count), 'positions': positions.ravel().tolist(), 'times': times.tolist(),
            'sourceRows': numpy.asarray(sourceRows).tolist(), 'observed': float(observed), 'constant': constant,
            'null': null, 'exactNull': exhaustive or constant, 'permutationCount': 511 if draws else 1,
            'seeds': [7, 123, 901] if draws else [7],
            'inputFloat32Sha256': {key: hashlib.sha256(array.astype('<f4').tobytes()).hexdigest() for key,array in
                                  [('positions',positions),('times',times),('distances',distances)]}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'fixtures')
    arguments=parser.parse_args()
    positions=numpy.fromfile(DATA/'position.bin',dtype='<f4').reshape(-1,2)
    times=numpy.fromfile(DATA/'time.bin',dtype='<u4')//86400
    order=numpy.lexsort((times, positions[:,1], numpy.floor(positions[:,0]*100)))
    transform=pyproj.Transformer.from_crs(4326,26916,always_xy=True)
    projected=numpy.column_stack(transform.transform(positions[:,0],positions[:,1]))/1000-[440,4600]
    projected=numpy.rint(projected*1024)/1024
    cases=[]
    for count,draws in [(2048,0),(128,4095)]:
        selected=order[numpy.linspace(0,len(order)-1,count,dtype=int)]
        cases.append(makeCase(f'chicago-rats-{count}',projected[selected],times[selected],selected,draws))
    smallPositions=[[0,0],[0,0],[1,0],[3,1],[4,4],[7,2]]
    cases.append(makeCase('ties-duplicates-exact-null',smallPositions,[0,0,1,3,3,8],[],720,True))
    cases.append(makeCase('constant-time',smallPositions,[3]*6,[],511))
    fixture={'generator':Path(__file__).name,'versions':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'scikit-bio':skbio.__version__,'pyproj':pyproj.__version__},
             'provenance':{'source':str(DATA.relative_to(ROOT)),'sourceCount':len(times),'sourceSha256':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['manifest.json','position.bin','time.bin']},'crs':'EPSG:26916','units':'km; integer days since 2023-01-01 local clock','selection':'lexsort(floor(longitude*100),latitude,time), equally spaced ranks','referenceSeed':31015},'cases':cases}
    output=arguments.output_dir/'parity-wp1-mantel.json'
    serialized=json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n'
    assert len(serialized)<1500000
    output.write_text(serialized)
    print(output,len(serialized),[(case['name'],case['observed'],len(case['null'])) for case in cases])

if __name__=='__main__': main()
