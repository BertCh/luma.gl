# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""SciPy Pearson correlation on independent cKDTree distance-band pairs.

The partial Mantel contract is a Pearson correlation of the listed strict upper
triangle. skbio's full-matrix Mantel does not expose missing-pair weights, so the
independent reference is scipy.stats.pearsonr, with fixed NumPy permutation draws
and the existing DKW/binomial distribution contract. Do not compare equal seeds.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
path=Path(__file__).parent/'fixtures/parity-wp1-mantel.json';source=json.loads(path.read_text());dataset=source['cases'][1]
positions=np.array(dataset['positions']).reshape(-1,2);times=np.array(dataset['times']);cases=[]
for radius in [2.,5.,12.]:
 pairs=cKDTree(positions).query_pairs(radius,output_type='ndarray');pairs=pairs[np.lexsort((pairs[:,1],pairs[:,0]))]
 distances=np.linalg.norm(positions[pairs[:,0]]-positions[pairs[:,1]],axis=1).astype(np.float32).astype(float)
 offsets=np.r_[0,np.cumsum(np.bincount(pairs[:,0],minlength=len(times)))]
 random=np.random.default_rng(31015)
 null=[]
 for draw in range(4095):
  permuted=times[random.permutation(len(times))];null.append(float(pearsonr(distances,np.abs(permuted[pairs[:,0]]-permuted[pairs[:,1]])).statistic))
 observed=float(pearsonr(distances,np.abs(times[pairs[:,0]]-times[pairs[:,1]])).statistic)
 cases.append(dict(name=f'chicago-rats-128-band-{radius}km',reference='scipy.spatial.cKDTree.query_pairs; scipy.stats.pearsonr; numpy.random.Generator.permutation',n=len(times),positions=dataset['positions'],times=times.tolist(),radius=radius,offsets=offsets.tolist(),neighbors=pairs[:,1].tolist(),distances=distances.tolist(),observed=observed,constant=False,null=null,exactNull=False,permutationCount=511,seeds=[7,123,901]))
output=path.with_name('parity-wp1-mantel-band.json');output.write_text(json.dumps(dict(generator=Path(__file__).name,command='/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I modules/experimental/test/oracles/parity-wp1-mantel-band.py',versions={'numpy':np.__version__,'scipy':scipy.__version__},sourceSha256=hashlib.sha256(path.read_bytes()).hexdigest(),sourceProvenance=source['provenance'],cases=cases),separators=(',',':'))+'\n');print(output)
