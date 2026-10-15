# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""COMPOSITE differential-geometry reference for all 15 local curvature kinds.

Published polynomial fits reuse the previously independently checked NumPy
pseudoinverse/SciPy convolution reference (not the library CPU oracle). Ground
spacing follows Snyder WGS84 radii at the f32 grid's row centre. For each cell,
SciPy eigh solves the generalized symmetric eigenproblem II v = k I v for
the surface's first and second fundamental forms. Mean/Gaussian/principal
curvatures follow from those eigenvalues, independently of the WGSL formulas.
Directional normal curvatures are quadratic forms along/across the gradient;
Minar et al. 2020 define the excess, accumulation, difference and ring products.
https://doi.org/10.1016/j.earscirev.2020.103414
https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
Snyder, Map Projections--A Working Manual (USGS PP 1395), ellipsoid radii.

Convex positive, north-up, WGS84 Web Mercator per-row local derivatives, default
zFactor=1 and flatGradient=1e-6. Clamped and invalid borders; f64 on f32 input
and georeference. Two real complete finite 33x33 domains, no full-DEM claim.
The separate multi-radius ring operator and Pade squash are NOT these local
ring products and have no independent reference here. Profile R unchanged.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy
from scipy.linalg import eigh

FOLDER=Path(__file__).parent

def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,FOLDER/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

curvature=load('independentCurvature','parity-wp5-terrain-curvature.py')
composites=load('independentComposites','parity-wp5-round8-composites.py')


def generate():
    for name in ['alps-dem','grand-canyon-dem']:
        base=json.loads((FOLDER/'fixtures'/f'parity-wp5-round8-{name}-vrm.json').read_text())['input']
        height,width=base['height'],base['width']
        elevation=numpy.asarray(base['elevation']).reshape(height,width)
        validity=numpy.asarray(base['validity']).reshape(height,width)
        north,south=[float(numpy.float32(base[key])) for key in ['north','south']]
        ordinate=north+(south-north)*(numpy.arange(height)+.5)/height
        latitude=numpy.arctan(numpy.sinh(ordinate/6378137))
        eccentricity=(1/298.257223563)*(2-1/298.257223563)
        denominator=1-eccentricity*numpy.sin(latitude)**2
        eastSize=(base['cellSize']*numpy.cos(latitude)/numpy.sqrt(denominator))[:,None]
        northSize=(base['cellSize']*numpy.cos(latitude)*(1-eccentricity)/denominator**1.5)[:,None]
        for method in ['zevenbergen-thorne','evans-young','florinsky']:
            for border in ['clamp','nodata']:
                partials,valid=curvature.getPartials(elevation,method,[1,1],'south',border,validity)
                east=partials['east']/eastSize;northward=partials['north']/northSize
                hessian=numpy.empty((height,width,2,2))
                hessian[:,:,0,0]=partials['eastEast']/eastSize**2
                hessian[:,:,1,1]=partials['northNorth']/northSize**2
                hessian[:,:,0,1]=hessian[:,:,1,0]=partials['eastNorth']/(eastSize*northSize)
                gradient=numpy.stack([east,northward],axis=-1)
                magnitude=numpy.linalg.norm(gradient,axis=-1)
                unit=numpy.divide(gradient,magnitude[:,:,None],out=numpy.zeros_like(gradient),where=magnitude[:,:,None]>1e-6)
                transverse=numpy.stack([-unit[:,:,1],unit[:,:,0]],axis=-1)
                weight=1+magnitude*magnitude
                along=numpy.einsum('...i,...ij,...j->...',unit,hessian,unit)
                across=numpy.einsum('...i,...ij,...j->...',transverse,hessian,transverse)
                mixed=numpy.einsum('...i,...ij,...j->...',unit,hessian,transverse)
                outputs=dict(profile=-along/weight**1.5,tangential=-across/numpy.sqrt(weight),
                             plan=numpy.divide(-across,magnitude,out=numpy.zeros_like(across),where=magnitude>1e-6),
                             rotor=numpy.divide(mixed,magnitude,out=numpy.zeros_like(across),where=magnitude>1e-6),
                             laplacian=numpy.trace(hessian,axis1=-2,axis2=-1))
                principal=numpy.full((height,width,2),numpy.nan)
                for row,column in zip(*numpy.where(valid)):
                    first=numpy.eye(2)+numpy.outer(gradient[row,column],gradient[row,column])
                    second=-hessian[row,column]/numpy.sqrt(weight[row,column])
                    principal[row,column]=eigh(second,first,eigvals_only=True)
                outputs['minimal']=principal[:,:,0];outputs['maximal']=principal[:,:,1]
                outputs['mean']=principal.mean(axis=-1);outputs['gaussian']=principal.prod(axis=-1)
                outputs['unsphericity']=(principal[:,:,1]-principal[:,:,0])/2
                outputs['difference']=(outputs['profile']-outputs['tangential'])/2
                outputs['horizontal-excess']=outputs['tangential']-outputs['minimal']
                outputs['vertical-excess']=outputs['profile']-outputs['minimal']
                outputs['accumulation']=outputs['tangential']*outputs['profile']
                outputs['ring']=outputs['horizontal-excess']*outputs['vertical-excess']
                case=dict(name=f'{name}-{method}-{border}',method=method,borderMode=border,
                          expected={key:numpy.where(valid,value,numpy.nan).ravel() for key,value in outputs.items()},validity=valid.astype(int).ravel())
                payload=dict(input=base,cases=[case],versions=curvature.VERSIONS,conventions=__doc__,
                             generatorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             dependencySha256=hashlib.sha256((FOLDER/'parity-wp5-terrain-curvature.py').read_bytes()).hexdigest(),
                             command=['/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python','-I',str(Path(__file__).relative_to(curvature.ROOT))])
                path=FOLDER/'fixtures'/f'parity-wp5-round8-curvature-{case["name"]}.json'
                path.write_text(json.dumps(composites.encode(payload),separators=(',',':'),allow_nan=False)+'\n')
                assert path.stat().st_size<1500000
                print(path.name,path.stat().st_size,flush=True)


if __name__=='__main__':generate()
