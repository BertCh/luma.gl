# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Geographic zonal moments: scipy.integrate.quad spherical area, NumPy cov.

Spherical surface element R^2*cos(latitude) dlatitude dlongitude, WGS84 authalic
R=6371007.180918475 m (Snyder USGS PP1395). Reliability-weight sample variance
uses numpy.cov(aweights=area,ddof=1), not population variance or frequency weights.
F32 source samples, dyadic calibration, both signed row directions, unweighted
counts/extrema, empty zones and nodata. Full arrays; unchanged S/E tolerances.
"""
import json,platform
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import quad
HERE=Path(__file__).resolve().parent
cases=[]
for direction in ['north','south']:
 width,height=8,9;capacity=6; radius=6371007.180918475
 rows,columns=np.indices((height,width));zones=(rows+columns)%5
 raw=((rows*3+columns*5)%17+1).astype(float);raw[1,2]=-9999
 validity=np.ones_like(zones);validity[5,3]=0
 origin=[0,40 if direction=='north' else 76]
 edges=origin[1]+(1 if direction=='north' else -1)*np.arange(height+1)*4
 areas=np.array([abs(quad(np.cos,np.deg2rad(first),np.deg2rad(last),epsabs=1e-13)[0])*radius**2*np.deg2rad(2) for first,last in zip(edges,edges[1:])])
 weights=np.broadcast_to(areas[:,None],raw.shape)
 valid=(raw!=-9999)&(validity!=0);values=raw*.5+2
 expected={name:[] for name in ['cellCounts','valueCounts','sums','means','minimums','maximums','weightSums','variances','standardDeviations']}
 for zone in range(capacity):
  selected=(zones==zone)&valid;observed=values[selected];weight=weights[selected]
  expected['cellCounts'].append(int(np.count_nonzero(zones==zone)));expected['valueCounts'].append(len(observed))
  expected['sums'].append(float(np.dot(observed,weight)));expected['weightSums'].append(float(np.sum(weight)))
  for name,function in [('means',lambda:np.average(observed,weights=weight)),('minimums',lambda:np.min(observed)),('maximums',lambda:np.max(observed)),('variances',lambda:np.cov(observed,aweights=weight,ddof=1))]:
   expected[name].append(float(function()) if len(observed) else None)
  variance=expected['variances'][-1];expected['standardDeviations'].append(float(np.sqrt(variance)) if variance is not None else None)
 cases.append(dict(name=f'area-weighted-{direction}',width=width,height=height,rowDirection=direction,origin=origin,cellSize=[2,4],zoneCapacity=capacity,
  values=raw.ravel().tolist(),validity=validity.ravel().tolist(),zones=zones.ravel().tolist(),expected=expected))
fixture=dict(reference='scipy.integrate.quad spherical surface element; numpy.average/cov(aweights,ddof=1)/sum/min/max',
 versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),cases=cases)
output=HERE/'fixtures/parity-wp4-zonal-modes.json';output.write_text(json.dumps(fixture,separators=(',',':'))+'\n');print(output.name,output.stat().st_size)
