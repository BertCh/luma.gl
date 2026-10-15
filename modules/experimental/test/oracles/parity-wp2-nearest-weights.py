# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent KNN/CSR reference: scipy.spatial.cKDTree + scipy.sparse.csr_matrix.

All 876 Chicago facilities self join; all 876 targets for the prior reproducible
256-crash query subset. WGS84 AEQD -87.63,41.88, explicit 1/16m snap, then f32.
KDTree finds the kth radius, query_ball_point expands all boundary ties; scipy
cdist computes double distances. Adapter breaks exact ties by feature row,
excludes self before k truncation, then scipy CSR sorts neighbor columns.
An unbounded radius is OMITTED: decodeOracleFixture reserves JSON null for NaN.
Binary weights and inverse max(distance,floor)^power are explicit mathematical
adapters (libpysal KNN has arbitrary ties and no finite zero-distance floor).
Synthetic duplicates, equidistant neighbors, radius inclusion, empty rows and
self exclusion are retained. No local TS/WGSL oracle is used.
G distances: 16 coordinate ulps. Weights: float32 operations on squared distances
have relative error O(epsilon); use 2e-6 relative for p<=2, floor exact. Real input
subtraction may cancel but is exact by Sterbenz for nearby projected coordinates.
"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import pyproj

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[3]/'examples/deck/spatial-analysis-showcase/public/data'
CRS='+proj=aeqd +lon_0=-87.63 +lat_0=41.88 +datum=WGS84 +units=m'
transformer=pyproj.Transformer.from_crs(4326,CRS,always_xy=True)
source=DATA/'chicago-facilities/position.bin'
longitude_latitude=np.fromfile(source,'<f4').reshape(-1,2)
features=(np.round(np.asarray(transformer.transform(longitude_latitude[:,0],longitude_latitude[:,1])).T*16)/16).astype(np.float32).astype(float)
road_source=HERE/'fixtures/parity-wp2-prepared-nearest.json'
road_fixture=json.loads(road_source.read_text())
queries=np.array(road_fixture['cases'][0]['frames'][0]['queries']).reshape(-1,2)


def make_case(name, queries, features, self_join, neighbor_count, radius=None):
    queries=np.asarray(queries,dtype=np.float32).astype(float)
    features=np.asarray(features,dtype=np.float32).astype(float)
    tree=cKDTree(features)
    row_indices, columns, distances=[],[],[]
    for row, point in enumerate(queries):
        initial_distances,_=tree.query(point,k=min(len(features),neighbor_count+1))
        search_radius=float(np.max(initial_distances))
        if radius is not None: search_radius=min(search_radius,radius)
        candidates=tree.query_ball_point(point,np.nextafter(search_radius,np.inf))
        candidate_distances=cdist([point],features[candidates])[0]
        entries=sorted((float(distance),int(index)) for distance,index in zip(candidate_distances,candidates)
                       if not (self_join and index==row) and (radius is None or distance<=radius))[:neighbor_count]
        for distance,index in entries:
            row_indices.append(row); columns.append(index); distances.append(distance)
    matrix=csr_matrix((distances,(row_indices,columns)),shape=(len(queries),len(features)))
    matrix.sort_indices()
    # Explicit zero distances remain stored, preserving duplicate features.
    expected=dict(offsets=matrix.indptr.tolist(), neighbors=matrix.indices.tolist(), distances=matrix.data.tolist())
    scale=max(1.,np.abs(queries).max(),np.abs(features).max())
    return dict(name=name,queries=queries.reshape(-1).tolist(),features=features.reshape(-1).tolist(),selfJoin=self_join,k=neighbor_count,**({} if radius is None else dict(radius=radius)),
                absolute=16*float(np.spacing(np.float32(scale))),expected=expected,
                rules=[dict(weightType='binary',power=1,floor=.25,weights=np.ones(matrix.nnz).tolist()),
                       dict(weightType='inverse-distance',power=1,floor=.25,weights=(1/np.maximum(matrix.data,.25)).tolist()),
                       dict(weightType='inverse-distance',power=2,floor=.25,weights=(1/np.maximum(matrix.data,.25)**2).tolist())])

cases=[make_case('all-876-facilities-self-k8',features,features,True,8),
       make_case('256-crashes-to-all-876-facilities-radius1500',queries,features,False,4,1500),
       make_case('duplicates-radius-ties-islands',[[0,0],[1,0],[100,100],[0,3]],[[0,0],[0,0],[1,0],[-1,0],[0,1],[0,-1]],False,3,1),
       make_case('self-duplicate-ties',[[0,0],[0,0],[1,0],[-1,0],[0,1],[100,100]],[[0,0],[0,0],[1,0],[-1,0],[0,1],[100,100]],True,3,1)]
fixture=dict(generator=Path(__file__).name,reference='scipy.spatial.cKDTree.query/query_ball_point; scipy.spatial.distance.cdist; scipy.sparse.csr_matrix',
             versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pyproj=pyproj.__version__),
             source=dict(dataset='all chicago-facilities + prior Chicago crash subset',fullFeatureCount=len(features),queryIds=road_fixture['source']['queryIds'],
                         hashes={'chicago-facilities/position.bin':hashlib.sha256(source.read_bytes()).hexdigest(),road_source.name:hashlib.sha256(road_source.read_bytes()).hexdigest()}),
             conventions=dict(crs=CRS,snapGrid=1/16,inputPrecision='float32',ties='distance then feature row',csr='ascending neighbor IDs; explicit zero-distance entries retained',weights='unstandardized directed; finite distance floor'),
             tolerance=dict(profile='G',distance='16 coordinate ulps',weightsRelative=2e-6,structure='exact'),cases=cases)
path=HERE/'fixtures/parity-wp2-nearest-weights.json'
path.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n')
assert path.stat().st_size<=1500000
print(path.name,path.stat().st_size,[(entry['name'],len(entry['expected']['neighbors'])) for entry in cases])
