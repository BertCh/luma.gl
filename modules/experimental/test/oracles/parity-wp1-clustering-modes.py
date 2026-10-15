# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""sklearn DBSCAN Euclidean/haversine plus independent group reductions.

Full-output synthetic edge cases exercise all optional summary outputs, bounded
cluster capacity, sorted/atomic centroids and planar dense-box modes. Geographic
fixtures use sklearn haversine [lat,lon] radians on identical uploaded f32 points,
R=6371008.8m, no antimeridian crossing (explicitly unsupported by this API).
Boundary separation margin is recorded; no threshold tolerance or allowed errors.
"""
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[4]

def makeCase(name,coordinates,geographic=False):
    positions=np.array(coordinates,dtype=np.float32).astype(float); epsilon=2000 if geographic else 1.25
    distances=haversine_distances(np.radians(positions[:,::-1]))*6371008.8 if geographic else cdist(positions,positions)
    model=DBSCAN(eps=epsilon,min_samples=3,metric='precomputed').fit(distances)
    core=np.zeros(len(positions),dtype=int);core[model.core_sample_indices_]=1
    roots=sorted(min(np.flatnonzero((model.labels_==label)&(core>0))) for label in set(model.labels_) if label>=0)
    labels=np.full(len(positions),4294967295,dtype=np.uint32); rootRows=labels.copy()
    for label,root in enumerate(roots):
        selected=model.labels_==model.labels_[root];labels[selected]=label;rootRows[selected]=root
    return dict(name=name,reference='sklearn.cluster.DBSCAN metric=precomputed; sklearn.metrics.haversine_distances; numpy.mean',positions=positions.ravel().tolist(),geographic=geographic,epsilon=epsilon,minimumPoints=3,bounds=[float(positions[:,0].min()-1),float(positions[:,1].min()-1),float(positions[:,0].max()+1),float(positions[:,1].max()+1)],labels=labels.tolist(),coreMask=core.tolist(),rootRows=rootRows.tolist(),roots=[int(root) for root in roots],sizes=[int((labels==label).sum()) for label in range(len(roots))],centroids=[positions[labels==label].mean(axis=0).tolist() for label in range(len(roots))],thresholdMargin=float(np.abs(distances-epsilon).min()))
positions=[[0,0],[.25,0],[0,.25],[.5,0],[10,0],[10,.25],[10.5,0],[50,50]]
cases=[makeCase('planar-summaries',positions),makeCase('great-circle-summaries',[[-85,40],[-85.005,40],[-85,40.005],[-85.01,40],[-90,45],[-90.005,45],[-90.01,45],[-70,30]],True)]
output=Path(__file__).parent/'fixtures/parity-wp1-clustering-modes.json';output.write_text(json.dumps(dict(generator=Path(__file__).name,command=f'/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I {Path(__file__).relative_to(ROOT)}',versions={'numpy':np.__version__,'sklearn':sklearn.__version__},cases=cases),separators=(',',':'))+'\n'); print(output)
