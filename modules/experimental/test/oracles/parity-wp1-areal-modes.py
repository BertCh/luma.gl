# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Dasymetric/categorical area shares via independent GEOS and Tobler.

Each raster cell becomes a disjoint rectangle with area equal to its uploaded
positive finite weight (unit height). Intersections of unions preserve precisely
the source/target cell incidence and effective measure. Tobler receives those
polygons and performs its own intersection/allocation. Zero/negative/nonfinite
weights have empty support, per the documented API. This is a measure adapter,
not a new interpolation implementation. All dyadic areas are exactly representable.
"""
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import csr_matrix
import shapely
from shapely import box, union_all
import geopandas as gpd
import tobler
from tobler.area_weighted import area_interpolate
ROOT=Path(__file__).resolve().parents[4]
source=np.array([0,0,0,1,1,1,0,0,2,2,1,1,0,0,2,2,9,9,9,9,2,2,9,9])
target=np.array([0,0,1,1,2,2,0,0,1,1,2,2,0,9,1,1,2,2,3,3,3,3,3,3])
categories=[0,1,0,1]; values=[10.,30.,50.,70.];cases=[]
for weighted in [False,True]:
 weights=np.array([.25,2,0,1,4,8,1,.5,2,4,0,1,1,2,4,8,1,1,2,2,4,4,0,1] if weighted else [1]*24,dtype=float)
 cells=[box(index*16,0,index*16+weight,1) if weight>0 else shapely.Polygon() for index,weight in enumerate(weights)]
 sources=[union_all([cells[index] for index in np.flatnonzero(source==zone)]) for zone in range(4)]
 targets=[union_all([cells[index] for index in np.flatnonzero(target==zone)]) for zone in range(4)]
 matrix=csr_matrix([[a.intersection(b).area for a in sources] for b in targets]);matrix.eliminate_zeros();matrix.sort_indices()
 sourceAreas=np.array([geometry.area for geometry in sources]);targetAreas=np.array([geometry.area for geometry in targets]);sourceOverlap=np.asarray(matrix.sum(axis=0)).ravel();targetOverlap=np.asarray(matrix.sum(axis=1)).ravel()
 references={}
 for extensive in ['zone','overlap']:
  sourceDenominator=sourceAreas if extensive=='zone' else sourceOverlap
  estimates=area_interpolate(gpd.GeoDataFrame({'mass':values,'rate':values},geometry=sources,crs='EPSG:3857'),gpd.GeoDataFrame(geometry=targets,crs='EPSG:3857'),extensive_variables=['mass'],intensive_variables=['rate'],allocate_total=extensive=='overlap')
  for intensive in ['zone','overlap']:
   targetDenominator=targetAreas if intensive=='zone' else targetOverlap
   extensiveWeights=matrix.multiply(np.divide(1,sourceDenominator,out=np.zeros(4),where=sourceDenominator>0)).tocsr().data
   intensiveWeights=matrix.multiply(np.divide(1,targetDenominator,out=np.zeros(4),where=targetDenominator>0)[:,None]).tocsr().data
   np.testing.assert_allclose(np.array(estimates['mass']),np.asarray(matrix.multiply(np.divide(1,sourceDenominator,out=np.zeros(4),where=sourceDenominator>0)).dot(values)),atol=1e-10)
   fractions=np.divide(sourceOverlap,sourceDenominator,out=np.zeros(4),where=sourceDenominator>0)
   references[extensive+'-'+intensive]=dict(extensiveWeights=extensiveWeights.tolist(),intensiveWeights=intensiveWeights.tolist(),fractions=[float(value) if sourceAreas[index]>0 else None for index,value in enumerate(fractions)])
 shares=np.asarray(matrix.dot(np.eye(2)[categories]))/targetOverlap[:,None]
 cases.append(dict(name='dasymetric-categories' if weighted else 'uniform-categories',reference='tobler.area_weighted.area_interpolate; shapely intersection/area; scipy.sparse category aggregation',width=6,height=4,sourceZones=source.tolist(),targetZones=target.tolist(),cellWeights=weights.tolist() if weighted else [],sourceCount=4,targetCount=4,categories=categories,offsets=matrix.indptr.tolist(),neighbors=matrix.indices.tolist(),areas=matrix.data.tolist(),sourceAreas=sourceAreas.tolist(),categoryShares=shares.ravel().tolist(),references=references))
path=Path(__file__).parent/'fixtures/parity-wp1-areal-modes.json';path.write_text(json.dumps(dict(generator=Path(__file__).name,command=f'/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python -I {Path(__file__).relative_to(ROOT)}',versions={'numpy':np.__version__,'scipy':scipy.__version__,'shapely':shapely.__version__,'tobler':tobler.__version__},cases=cases),separators=(',',':'))+'\n');print(path)
