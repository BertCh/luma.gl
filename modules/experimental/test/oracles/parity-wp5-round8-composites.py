# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""COMPOSITE REFERENCES, independently assembled from published definitions.

TPI = centre minus annular mean (Weiss 2001); DEV = centre deviation / population
standard deviation of the full square (Lindsay, Cockburn & Russell 2015,
doi:10.1016/j.geomorph.2015.07.012). scipy.ndimage.uniform_filter computes box
statistics with zero-padded validity counts; windows clip, never replicate.
Input is rounded to the documented power-of-two quantum with numpy.rint.
Box sums are restored to their exact integer-quantum lattice after SciPy's
normalized filtering; all exact integer sums/squares here fit in 53 bits.
This removes filter normalization roundoff from exact DEVmax ties.
Weiss global standardizes TPI over cells with valid TPI AND Horn slope;
local uses DEV. Both position planes are nodata wherever classification is
invalid, as specified by the contributor's output-validity contract.
The published 3x3 low/mid/high classification matrix splits mid/mid at 5 degrees.
https://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf
VRM = 1 - norm(sum(unit normals))/count (Sappington et al. 2007,
doi:10.2193/2005-449). scipy.ndimage.sobel independently provides Horn derivatives;
all nine samples must be valid, clamp/nodata are explicit, normals clip at edges.
WGS84 row metrics use Snyder's ellipsoid radii and inverse spherical Mercator,
with float32-rounded cell sizes and row edges. This covers actual showcase
Web Mercator ground units as well as planar units, without a TS CPU oracle.

Texture is a COMPONENT comparison, not equivalence to Brown's FFT fractional
Laplacian. scipy.ndimage.gaussian_filter supplies the documented Gaussian
cascade, nearest borders, radius ceil(3 sigma), and normalized convolution
(Knutsson/Westin 1993, doi:10.1109/CVPR.1993.341081); differences of Gaussians
are combined at explicit f32 weights, gain and detail. Burt/Adelson 1983,
doi:10.1109/TCOM.1983.1095851. Full-resolution only: decimated cascades need a
separate independently specified resampling contract. No TS/WGSL is translated.

Complete 33x33 finite domains from BOTH showcase DEMs, no resampling, plus a
quantized saddle with a validity hole. This does not prove full-raster or
full-halo behavior for the landscape/broad presets; large windows deliberately
clip to the finite domain. Full output arrays, not selected summary cells.
Float64 reference arithmetic, f32 inputs/calibration, R for terrain, L for
Gaussian operator, exact integer outputs/masks. No tolerance widened.
"""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy
import rasterio
from scipy.ndimage import uniform_filter, sobel, minimum_filter, gaussian_filter

ROOT=Path(__file__).resolve().parents[4]
OUTPUT=Path(__file__).parent/'fixtures'
VERSIONS={name:importlib.metadata.version(name) for name in ['numpy','scipy','rasterio']}
VERSIONS.update(python=platform.python_version(),gdal=rasterio.__gdal_version__)


def encode(value):
    if isinstance(value,dict): return {key:encode(item) for key,item in value.items()}
    if isinstance(value,(list,tuple,numpy.ndarray)): return [encode(item) for item in value]
    if isinstance(value,(numpy.integer,numpy.bool_)): return int(value)
    if isinstance(value,(float,numpy.floating)): return float(value) if numpy.isfinite(value) else None
    return value


def write(name,base,cases):
    payload=dict(input=base,cases=cases,versions=VERSIONS,conventions=__doc__,
                 generatorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 command=['/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval/venv/bin/python','-I',str(Path(__file__).relative_to(ROOT))])
    path=OUTPUT/f'parity-wp5-round8-{name}.json'
    path.write_text(json.dumps(encode(payload),separators=(',',':'),allow_nan=False)+'\n')
    assert path.stat().st_size<1500000,(path,path.stat().st_size)
    print(path.name,len(cases),path.stat().st_size,flush=True)


def box(values,radius): return uniform_filter(values.astype(float),size=2*radius+1,mode='constant',cval=0)*(2*radius+1)**2


def positions(elevation,validity,scales,quantum):
    rounded=numpy.rint(elevation/quantum)*quantum
    anchor=rounded[validity].min()
    centered=numpy.where(validity,rounded-anchor,0)
    indices=[];deviations=[]
    for scale in scales:
        radius=scale['radius'];inner=scale.get('innerRadius',0)
        counts=numpy.rint(box(validity,radius));sums=numpy.rint(box(centered,radius)/quantum)*quantum
        annularCounts=counts-numpy.rint(box(validity,inner))
        annularSums=sums-numpy.rint(box(centered,inner)/quantum)*quantum
        with numpy.errstate(invalid='ignore',divide='ignore'):
            index=centered-annularSums/annularCounts
            mean=sums/counts
            variance=numpy.maximum(0,(numpy.rint(box(centered*centered,radius)/(quantum*quantum))*(quantum*quantum))/counts-mean*mean)
            deviation=numpy.divide(centered-mean,numpy.sqrt(variance),out=numpy.zeros_like(mean),where=variance>0)
        indices.append(numpy.where(validity & (annularCounts>.5),index,numpy.nan))
        deviations.append(numpy.where(validity,deviation,numpy.nan))
    indices=numpy.asarray(indices);deviations=numpy.asarray(deviations)
    winners=numpy.argmax(numpy.where(numpy.isfinite(deviations),numpy.abs(deviations),-1),axis=0)
    maximum=numpy.take_along_axis(deviations,winners[None,:,:],axis=0)[0]
    radii=numpy.asarray([scale['radius'] for scale in scales])[winners]
    return dict(topographicPositionIndex=indices.ravel(),deviationFromMean=deviations.ravel(),
                maximumDeviation=maximum.ravel(),maximumDeviationRadius=numpy.where(validity,radii,0).ravel(),
                outputValidity=validity.astype(int).ravel())


def normals(elevation,validity,base,mode,border,zFactor=1):
    valid=minimum_filter(validity.astype(int),size=3,mode='nearest' if border=='clamp' else 'constant',cval=0).astype(bool)
    cellEast=numpy.full((elevation.shape[0],1),base['cellSize']);cellNorth=cellEast.copy()
    if mode=='web-mercator':
        fraction=(numpy.arange(elevation.shape[0])+.5)/elevation.shape[0]
        north,south=map(lambda value:float(numpy.float32(value)),[base['north'],base['south']])
        latitude=numpy.arctan(numpy.sinh((north+(south-north)*fraction)/6378137))[:,None]
        eccentricity=(1/298.257223563)*(2-1/298.257223563)
        denominator=1-eccentricity*numpy.sin(latitude)**2
        cellEast*=numpy.cos(latitude)/numpy.sqrt(denominator)
        cellNorth*=numpy.cos(latitude)*(1-eccentricity)/denominator**1.5
    values=numpy.where(validity,elevation,0)
    east=sobel(values,axis=1,mode='nearest')*zFactor/(8*cellEast)
    north=-sobel(values,axis=0,mode='nearest')*zFactor/(8*cellNorth)
    vectors=numpy.stack([-east,-north,numpy.ones_like(elevation)])
    vectors/=numpy.linalg.norm(vectors,axis=0)
    vectors[:,~valid]=0
    return vectors,valid,numpy.degrees(numpy.arctan(numpy.hypot(east,north)))


def generate(name,elevation,validity,base):
    base.update(name=name,width=elevation.shape[1],height=elevation.shape[0],elevation=elevation.ravel(),validity=validity.astype(int).ravel(),elevationSha256=hashlib.sha256(elevation.astype('<f4').tobytes()).hexdigest())
    presets={'fine':[1,2,3,4,6,8,12,16],'landscape':[2,4,8,16,32,64,128,256],'broad':[8,16,32,64,128,256,384,512]}
    # Split fixtures by preset to respect the 1.5 MB limit, retain every cell/plane.
    for preset,radii in presets.items():
        cases=[]
        for quantum in [1/4,1/64,1/256]:
            for innerFraction in [0,.5]:
                scales=[dict(radius=radius,innerRadius=int(radius*innerFraction)) for radius in radii]
                cases.append(dict(name=f'{name}-{preset}-{quantum}-{innerFraction}',quantum=quantum,scales=scales,expected=positions(elevation,validity,scales,quantum)))
        # Each case is independent; no duplicated huge fixture batch.
        for index,case in enumerate(cases): write(f'{name}-tpi-{preset}-{index}',base,[case])
    rugged=[];weiss=[]
    for mode in ['uniform','web-mercator']:
        for border in ['clamp','nodata']:
            vectors,valid,slope=normals(elevation,validity,base,mode,border)
            for radius in range(1,9):
                counts=box(valid,radius)
                sums=numpy.stack([box(component,radius) for component in vectors])
                with numpy.errstate(invalid='ignore',divide='ignore'):
                    result=numpy.clip(1-numpy.linalg.norm(sums,axis=0)/counts,0,1)
                rugged.append(dict(name=f'{name}-{mode}-{border}-{radius}',cellSizeMode=mode,borderMode=border,radius=radius,
                                   expected=dict(vectorRuggedness=numpy.where(valid,result,numpy.nan).ravel(),outputValidity=valid.astype(int).ravel())))
        vectors,valid,slope=normals(elevation,validity,base,mode,'clamp')
        for scales in [[dict(radius=3),dict(radius=15)],[dict(radius=4),dict(radius=40)]]:
            result=positions(elevation,validity,scales,1/256)
            for standardization in ['global','local']:
                position=numpy.asarray(result['topographicPositionIndex' if standardization=='global' else 'deviationFromMean']).reshape(2,*elevation.shape)
                if standardization=='global':
                    for plane in position:
                        population=plane[valid & numpy.isfinite(plane)]
                        deviation=numpy.std(population)
                        plane[:]=(plane-numpy.mean(population))/deviation if deviation else 0
                small=numpy.where(position[0]<=-1,0,numpy.where(position[0]>=1,2,1))
                large=numpy.where(position[1]<=-1,0,numpy.where(position[1]>=1,2,1))
                classes=numpy.asarray([[1,2,3],[4,5,7],[8,9,10]])[small,large]
                classes=numpy.where((small==1)&(large==1)&(slope>5),6,classes)
                usable=valid&numpy.isfinite(position).all(axis=0)
                weiss.append(dict(name=f'{name}-{mode}-{standardization}-{scales[0]["radius"]}-{scales[1]["radius"]}',cellSizeMode=mode,standardization=standardization,smallScale=scales[0],largeScale=scales[1],
                                  expected=dict(landforms=numpy.where(usable,classes,0).ravel(),standardizedPosition=numpy.where(usable[None,:,:],position,numpy.nan).ravel(),outputValidity=usable.astype(int).ravel())))
    write(name+'-vrm',base,rugged);write(name+'-weiss',base,weiss)
    textures=[]
    for levels,baseSigma in [(1,1),(3,1),(6,.5),(6,1)]:
        for detail in [0,.5,1]:
            weights=numpy.asarray([2**(-level*detail) for level in range(levels)],dtype='float32').astype(float)
            numerator=numpy.where(validity,elevation,0).astype(float);denominator=validity.astype(float)
            previous=numpy.where(validity,elevation,0).astype(float);shade=numpy.zeros_like(previous)
            for level in range(levels):
                sigma=baseSigma*2**level
                incremental=numpy.sqrt(sigma**2-(baseSigma*2**(level-1))**2) if level else sigma
                numerator=gaussian_filter(numerator,incremental,radius=int(numpy.ceil(3*incremental)),mode='nearest')
                denominator=gaussian_filter(denominator,incremental,radius=int(numpy.ceil(3*incremental)),mode='nearest')
                blurred=numerator/denominator
                shade+=weights[level]*(previous-blurred)
                previous=blurred
            textures.append(dict(name=f'{name}-{levels}-{baseSigma}-{detail}',levelCount=levels,baseSigma=baseSigma,detail=detail,gain=float(numpy.float32(.01)),
                                 expected=dict(textureShade=numpy.where(validity,shade*float(numpy.float32(.01)),numpy.nan).ravel(),outputValidity=validity.astype(int).ravel())))
    write(name+'-texture',base,textures)


if __name__=='__main__':
    for name in ['alps-dem','grand-canyon-dem']:
        folder=ROOT/'examples/deck/spatial-analysis-showcase/public/data'/name
        manifest=json.loads((folder/'manifest.json').read_text())
        with rasterio.open(folder/'dem.png') as dataset: channels=dataset.read().astype(float)
        full=channels[0]*256+channels[1]+channels[2]/256-32768
        spacing=manifest['raster']['cellSizeMercatorM'];north=manifest['raster']['boundsMercator'][3]-960*spacing
        base=dict(cellSize=float(numpy.float32(spacing)),north=north,south=north-33*spacing,
                  source=dict(path=str(folder.relative_to(ROOT)),window=[960,960,33,33],pngSha256=hashlib.sha256((folder/'dem.png').read_bytes()).hexdigest(),manifestSha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()))
        generate(name,full[960:993,960:993],numpy.ones((33,33),bool),base)
    rows,columns=numpy.indices((17,17));validity=numpy.ones((17,17),bool);validity[8,8]=False
    generate('saddle',(columns**2-rows**2)/4,validity,dict(cellSize=8.,north=5780000.,south=5780000.-17*8,source=dict(synthetic='quantized saddle, centre hole')))
