# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""PROJ sphere arc vertices for ALL 18930 OpenFlights routes, in contiguous batches.

https://pyproj4.github.io/pyproj/stable/api/geod.html: Geod.inv determines length;
Geod.npts supplies evenly spaced interior vertices. Explicit sphere R=6371008.8,
f32 input degrees. Longitude branch is within 180 degrees of the source, as
publicly documented (not a dateline split). Source and unwrapped target retained.
Segments=clamp(max(ceil(distance/maximumLength),minimum),1,cap), an API adapter;
all integer offsets/owners/counts exact. Distances at all vertices are along-arc
fraction*PROJ length, not planar chords. No custom spherical interpolator.
D bounds: .001m + 2e-6 relative for measures. Coordinates: 8 coordinate ulps,
plus angular conditioning. A unit-vector error <=8 epsilon32 leads to latitude
error <=8 epsilon32 radians and longitude error <=8 epsilon32/abs(cos(lat)).
Near antipodes, slerp additionally amplifies by 1/abs(sin(sigma)); short arcs do
not need that factor since the numerator cancels the small-angle denominator.
Use that factor only sigma>3 radians. Bounds are derived before any GPU run.
Exact antipodes are excluded from vertex equality because no unique arc exists;
none in the source routes. Explicit near-antipode is outside the documented
1e-3 rad exclusion. Synthetic includes dateline, poles, coincident, cap and
minimum-segment controls; exact polar-crossing longitude is deliberately absent.
"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pyproj
HERE=Path(__file__).resolve().parent
DATA=HERE.parents[3]/'examples/deck/spatial-analysis-showcase/public/data/openflights'
RADIUS=6371008.8
SPHERE=pyproj.Geod(a=RADIUS,f=0)
versions=dict(python=platform.python_version(),numpy=np.__version__,pyproj=pyproj.__version__,proj=pyproj.proj_version_str)
hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['position.bin','origin.bin','destination.bin','manifest.json']}

def make_case(name,sources,targets,settings):
    sources=np.asarray(sources,dtype=np.float32).astype(float)
    targets=np.asarray(targets,dtype=np.float32).astype(float)
    frames=[]
    for maximum_length,minimum,cap in settings:
        offsets=[0]; vertices=[]; measures=[]; owners=[]; bounds=[]; segment_margins=[]
        for row,(source,target) in enumerate(zip(sources,targets)):
            _,_,distance=SPHERE.inv(*source,*target)
            angle=distance/RADIUS
            assert np.pi-angle>1e-3
            segments=min(cap,max(minimum,int(np.ceil(distance/maximum_length)) if maximum_length else 1))
            # Track distance-to-ceil boundaries so exact count assertions are auditable.
            if maximum_length:
                ratio=distance/maximum_length
                segment_margins.append(abs(ratio-round(ratio)))
            interior=SPHERE.npts(*source,*target,segments-1) if segments>1 and distance else [source]*(segments-1)
            points=np.array([source,*interior,target])
            points[:,0]=source[0]+(points[:,0]-source[0]+180)%360-180
            points[0]=source
            scale=max(1.,np.abs(points).max())
            quantization=8*float(np.spacing(np.float32(scale)))
            condition=1/abs(np.sin(angle)) if angle>3 else 1.
            for vertex,point in enumerate(points):
                vertices.extend(point.tolist()); owners.append(row); measures.append(distance*vertex/segments)
                cosine=max(abs(np.cos(np.radians(point[1]))),1e-12)
                error=float(np.degrees(8*float(np.finfo(np.float32).eps)))*condition
                bounds.extend([quantization+error/cosine,quantization+error])
            offsets.append(len(owners))
        frames.append(dict(maximumSegmentLength=maximum_length,minimumSegments=minimum,maximumSegmentCount=cap,
                           positions=vertices,lineOffsets=offsets,sourceRows=owners,measures=measures,positionBounds=bounds,
                           minimumCountBoundaryMargin=min(segment_margins,default=0.)))
    return dict(name=name,sources=sources.reshape(-1).tolist(),targets=targets.reshape(-1).tolist(),frames=frames)

def write(suffix,entry,source):
    path=HERE/f'fixtures/parity-wp2-great-circle-arcs-{suffix}.json'
    fixture=dict(generator=Path(__file__).name,reference='pyproj.Geod.inv/npts (explicit sphere)',versions=versions,source=source,
                 conventions=dict(sphereRadius=RADIUS,inputPrecision='float32 degrees',longitude='source-relative shortest branch',units='meters'),
                 tolerance=dict(profile='D',positions='8 coordinate ulps + 8 epsilon32 angular vector error propagated through latitude/longitude; antipode factor for sigma>3',measureAbsolute=.001,measureRelative=2e-6),cases=[entry])
    path.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n'); assert path.stat().st_size<1500000
    print(path.name,path.stat().st_size,len(entry['sources'])//2,[len(frame['sourceRows']) for frame in entry['frames']])

positions=np.fromfile(DATA/'position.bin','<f4').reshape(-1,2)
origins=np.fromfile(DATA/'origin.bin','<u4'); targets=np.fromfile(DATA/'destination.bin','<u4')
for start in range(0,len(origins),1024):
    stop=min(start+1024,len(origins))
    write(f'flights-{start//1024:02}',make_case(f'flights-{start}-{stop}',positions[origins[start:stop]],positions[targets[start:stop]],[(2000000.,4,16)]),
          dict(dataset='openflights',hashes=hashes,fullRouteCount=len(origins),sourceRange=[start,stop],selection='every route, contiguous complete pair batches'))
write('edges',make_case('dateline-pole-zero-near-antipode',[[179,20],[-179,-30],[10,89],[20,10],[0,0],[10,20]],
                       [[-179,25],[179,-25],[100,88],[20,10],[179.5,.5],[11,20]],[(100000.,3,16),(0.,5,16)]),dict(dataset='synthetic'))
# Round-5 regression: the two failed source routes in both directions, plus short
# seam separations. 4 versus 64 segments distinguishes endpoint error from a scan
# accumulation error. All frames share the compile-time cap; tolerance unchanged.
regression_sources=[[177.443,-17.7554],[179.341,-16.8028],[-179.877,-16.6906],[-179.877,-16.6906],[179.99998,20],[-179.99998,20]]
regression_targets=[[-179.877,-16.6906],[-179.877,-16.6906],[177.443,-17.7554],[179.341,-16.8028],[-179.99998,20],[179.99998,20]]
write('seam-regression',make_case('short-seam-round5',regression_sources,regression_targets,[(0.,4,64),(0.,64,64)]),dict(dataset='OpenFlights routes 16092,18094 plus reversals and synthetic near seam',hashes=hashes))
