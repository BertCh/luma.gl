# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent GRASS 8.5 / GDAL 3.12 references, not a port of TS/WGSL.

Real Terrarium cells are decoded without resampling from both showcase DEMs.
The 33x33 windows are COMPLETE finite test domains, not full-DEM visibility.
Each is assigned a local square metric grid using f32 manifest ground spacing;
this is NOT Web Mercator parity. Sight-line endpoints are cell centres, with
explicit observer/target heights, common sphere radius 6371008.8 m, zero or
matched refraction, zero tolerance/ignore bands, and explicit range masks.
GDAL's cropped output is placed back by its affine transform. Both references'
full arrays and exact disagreement indices are retained; neither wins by fiat.
Algorithms differ (GDAL Wang reference planes, GRASS sweep, GPU bilinear march).
Exact comparisons may FAIL: model differences are not f32 error budgets.

r.horizon has no eye-height/refraction switch, uses R=6371000 m and clips
negative horizons to zero. Compare the nonnegative projection only, cardinal
rays, distance=1, explicit maxdistance; do not claim negative skyline parity.
r.sunmask -z is hard sunlight, positive altitude, no soft solar disk; unlike
GPU bounded casts its search continues to the edge of this finite domain.
Geomorphon outputs are direct GRASS forms + canonical ternary, all 3 comparisons;
border NULLs are encoded as 0. GRASS dist is in cells without -m; the
GPU flatDistance in metres is divided by cellSize for the reference command. Raw pattern codes are not canonical ternary codes.

https://grass.osgeo.org/grass85/manuals/r.viewshed.html
https://gdal.org/en/stable/programs/gdal_viewshed.html
https://grass.osgeo.org/grass85/manuals/r.horizon.html
https://grass.osgeo.org/grass85/manuals/r.geomorphon.html
https://grass.osgeo.org/grass85/manuals/r.sunmask.html
r.vector.ruggedness exponent=0 uses unweighted square normal averages; its
FCELL slope/aspect intermediates are retained (not replaced by our normal
reference). Nodata centre normals mask the outer row/column explicitly.
Exact subprocess argument arrays, software versions, hashes in every fixture.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import tempfile
import numpy
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[4]
EVAL = Path('/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval')
BIN = EVAL / 'refenv/bin'
OUTPUT = Path(__file__).parent / 'fixtures'
SKYVIEW = Path('/private/tmp/parity-wp5-grass-addons/scripts/r.skyview')
RUGGEDNESS = SKYVIEW.with_name('r.vector.ruggedness')
ENV = dict(os.environ, PATH=str(BIN)+os.pathsep+os.environ['PATH'], GRASS_MESSAGE_FORMAT='plain', LC_ALL='C')
COMMANDS = []


def execute(arguments):
    arguments = [str(value) for value in arguments]
    COMMANDS.append(arguments)
    result = subprocess.run(arguments, env=ENV, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{shlex.join(arguments)}\n{result.stdout}\n{result.stderr}')
    return result.stdout


VERSIONS = dict(python=platform.python_version(), numpy=numpy.__version__,
                rasterio=rasterio.__version__, rasterioGDAL=rasterio.__gdal_version__,
                grass=execute([BIN/'grass', '--config', 'version']).strip(),
                gdal=execute([BIN/'gdalinfo', '--version']).strip())
for package in ['grass','libgdal-core']:
    for metadata in sorted((EVAL/'refenv/conda-meta').glob(package+'-*.json')):
        record=json.loads(metadata.read_text())
        VERSIONS[package+'Build']=record['version']+'-'+record['build']


VERSIONS['r.skyviewSha256']=hashlib.sha256(SKYVIEW.read_bytes()).hexdigest()
VERSIONS['r.vector.ruggednessSha256']=hashlib.sha256(RUGGEDNESS.read_bytes()).hexdigest()

def encode(value):
    if isinstance(value, dict): return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)): return [encode(item) for item in value]
    if isinstance(value, (numpy.integer, numpy.bool_)): return int(value)
    if isinstance(value, (float, numpy.floating)): return float(value) if numpy.isfinite(value) else None
    return value


def readRaster(path, fill=0):
    with rasterio.open(path) as dataset:
        return dataset.read(1, masked=True).filled(fill)


def writeFixture(name, base, cases):
    payload = dict(versions=VERSIONS, conventions=__doc__, input=base, cases=cases,
                   commands=COMMANDS.copy(), generatorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = OUTPUT / f'parity-wp5-round8-{name}.json'
    path.write_text(json.dumps(encode(payload), separators=(',', ':'), allow_nan=False)+'\n')
    assert path.stat().st_size < 1500000, path
    print(path.name, len(cases), path.stat().st_size, flush=True)


def generate(name, elevation, cellSize, source):
    COMMANDS.clear()
    height, width = elevation.shape
    base = dict(name=name, width=width, height=height, cellSize=cellSize,
                elevation=elevation.ravel(), source=source,
                elevationSha256=hashlib.sha256(elevation.astype('<f4').tobytes()).hexdigest())
    with tempfile.TemporaryDirectory(prefix='parity-wp5-r8-') as temporary:
        folder = Path(temporary)
        raster = folder/'input.tif'
        transform = from_origin(0, height*cellSize, cellSize, cellSize)
        with rasterio.open(raster, 'w', driver='GTiff', width=width, height=height, count=1,
                           dtype='float32', transform=transform,
                           crs='+proj=eqc +R=6371008.8 +units=m +no_defs') as dataset:
            dataset.write(elevation.astype('float32'), 1)
        script = ['set -eu']
        def grass(*arguments):
            COMMANDS.append(list(map(str, arguments)))
            script.append(shlex.join(list(map(str, arguments))))
        def export(mapName):
            grass('r.out.gdal', 'input='+mapName, 'output='+str(folder/(mapName+'.tif')), 'type=Float64', 'nodata=-9999', '--overwrite', '--quiet')
        grass('r.in.gdal', 'input='+str(raster), 'output=elevation', '--quiet')
        grass('g.region', 'raster=elevation')
        visibilityCases = []
        for settingIndex, (observerHeight, targetHeight, refraction, maximumDistance) in enumerate([
            (1.7, 0, None, 0), (1.7, 0, 0.13, 0), (12, 2, 1/7, cellSize*12.25)]):
            case = dict(name=f'{name}-setting-{settingIndex}', observerHeight=float(numpy.float32(observerHeight)),
                        targetHeight=targetHeight, curvatureCoefficient=0 if refraction is None else (1-refraction)/(2*6371008.8),
                        maximumDistance=float(numpy.float32(maximumDistance)), observers=[[8,16],[24,16]])
            case['grass'] = []; case['gdal'] = []
            for observerIndex, (column,row) in enumerate(case['observers']):
                east, north = transform*(column+.5,row+.5)
                outputName = f'view_{settingIndex}_{observerIndex}'
                grass('r.viewshed', '-b' if refraction is None else '-bcr', 'input=elevation', 'output='+outputName,
                      f'coordinates={east},{north}', f'observer_elevation={case["observerHeight"]}',
                      f'target_elevation={targetHeight}', f'max_distance={case["maximumDistance"] or -1}',
                      f'refraction_coeff={refraction or 0}', '--quiet')
                export(outputName)
                case['grass'].append(outputName)
                gdalOutput = folder/(outputName+'-gdal.tif')
                execute([BIN/'gdal_viewshed','--quiet','-ox',east,'-oy',north,'-oz',case['observerHeight'],
                         '-tz',targetHeight,'-md',case['maximumDistance'],'-cc',0 if refraction is None else 1-refraction,
                         '-vv',1,'-iv',0,'-ov',2,'-a_nodata',3,raster,gdalOutput])
                reference = numpy.full((height,width),2,dtype=int)
                with rasterio.open(gdalOutput) as dataset:
                    left,top = ~transform*(dataset.transform.c,dataset.transform.f)
                    left,top = round(left),round(top)
                    reference[top:top+dataset.height,left:left+dataset.width] = dataset.read(1)
                case['gdal'].append(reference)
            visibilityCases.append(case)
        geomorphonCases = []
        for comparison in ['anglev1','anglev2','anglev2-distance']:
            for skip,flatDistance in [(0,0),(1,cellSize*4)]:
                outputName = f'geom_{len(geomorphonCases)}'
                grass('r.geomorphon','elevation=elevation','forms='+outputName,'ternary='+outputName+'_ternary',
                      'search=8',f'skip={skip}','flat=1',f'dist={flatDistance/cellSize}',
                      'comparison='+comparison.replace('-','_'),'--quiet')
                export(outputName); export(outputName+'_ternary')
                geomorphonCases.append(dict(name=outputName,comparison=comparison,searchRadius=8,skipRadius=skip,
                                            flatThresholdDegrees=1,flatDistance=flatDistance))
        grass('r.horizon','-d','elevation=elevation','step=90','distance=1',f'maxdistance={cellSize*16}',
              'output=horizon','nprocs=1','--quiet')
        for angle in [0,90,180,270]: export(f'horizon_{angle:03d}')
        grass(str(SKYVIEW),'input=elevation','output=skyview','ndir=4',f'maxdistance={cellSize*16}','nprocs=1','--quiet')
        export('skyview')
        grass(str(RUGGEDNESS),'elevation=elevation','output=ruggedness','size=3,7','exponent=0','nprocs=1','--quiet')
        export('ruggedness_3'); export('ruggedness_7')
        grass('r.horizon','-dcl','elevation=elevation','step=90','distance=1',f'maxdistance={cellSize*16}',
              f'coordinates={16.5*cellSize},{16.5*cellSize}','format=json','file='+str(folder/'point.json'),'--quiet')
        for azimuth in [0,90,180,270]:
            grass('r.sunmask','-z','elevation=elevation',f'output=shadow_{azimuth}','altitude=20',f'azimuth={azimuth}','--quiet')
            export(f'shadow_{azimuth}')
        scriptPath = folder/'reference.sh'
        scriptPath.write_text('\n'.join(script)+'\n')
        execute([BIN/'grass','--tmp-project',raster,'--exec','sh',scriptPath])
        rows,columns = numpy.indices(elevation.shape)
        for case in visibilityCases:
            case['grass'] = [readRaster(folder/(output+'.tif')).astype(int) for output in case['grass']]
            case['referenceDisagreements'] = []
            for observerIndex,(column,row) in enumerate(case['observers']):
                distance = numpy.hypot(columns-column,rows-row)*cellSize
                outside = (distance > case['maximumDistance']) if case['maximumDistance'] else numpy.zeros(elevation.shape,bool)
                case['grass'][observerIndex][outside] = 2
                difference = numpy.flatnonzero(case['grass'][observerIndex].ravel()!=case['gdal'][observerIndex].ravel())
                case['referenceDisagreements'].append(difference)
                print(name, case['name'], observerIndex, 'GRASS/GDAL mismatches',len(difference), flush=True)
            case['grassCounts'] = numpy.sum(numpy.asarray(case['grass'])==1,axis=0).ravel()
            case['gdalCounts'] = numpy.sum(numpy.asarray(case['gdal'])==1,axis=0).ravel()
            for reference in ['grass','gdal']: case[reference] = [value.ravel() for value in case[reference]]
        writeFixture(name+'-visibility',base,visibilityCases)
        for case in geomorphonCases:
            case['forms'] = readRaster(folder/(case['name']+'.tif')).astype(int).ravel()
            case['ternary'] = readRaster(folder/(case['name']+'_ternary.tif')).astype(int).ravel()
        writeFixture(name+'-geomorphons',base,geomorphonCases)
        horizon = numpy.stack([readRaster(folder/f'horizon_{angle:03d}.tif') for angle in [90,0,270,180]],axis=-1)
        writeFixture(name+'-horizon',base,[dict(name=name+'-cardinal',maximumRadius=16,curvatureCoefficient=1/(2*6371000),
                     horizon=horizon.ravel(), skyViewFactor=readRaster(folder/'skyview.tif').ravel(), point=json.loads((folder/'point.json').read_text()))])
        validity=numpy.ones((height,width),dtype=int);validity[[0,-1],:]=0;validity[:,[0,-1]]=0
        writeFixture(name+'-ruggedness-addon',base,[dict(name=f'{name}-addon-{radius}',radius=radius,
                     expected=numpy.where(validity,readRaster(folder/f'ruggedness_{2*radius+1}.tif',fill=numpy.nan),numpy.nan).ravel(),
                     validity=validity.ravel()) for radius in [1,3]])
        writeFixture(name+'-shadow',base,[dict(name=f'{name}-{azimuth}',azimuthDegrees=azimuth,altitudeDegrees=20,
                     sunVisibility=(readRaster(folder/f'shadow_{azimuth}.tif')!=1).astype(int).ravel()) for azimuth in [0,90,180,270]])


if __name__ == '__main__':
    for name in ['alps-dem','grand-canyon-dem']:
        folder = ROOT/'examples/deck/spatial-analysis-showcase/public/data'/name
        manifest = json.loads((folder/'manifest.json').read_text())
        with rasterio.open(folder/'dem.png') as dataset:
            channels = dataset.read().astype(float)
        full = channels[0]*256+channels[1]+channels[2]/256-32768
        source = dict(path=str(folder.relative_to(ROOT)),window=[960,960,33,33],
                      pngSha256=hashlib.sha256((folder/'dem.png').read_bytes()).hexdigest(),
                      manifestSha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest())
        generate(name,full[960:993,960:993],float(numpy.float32(manifest['raster']['cellSizeGroundM'])),source)
    rows,columns = numpy.indices((33,33))
    generate('ridge',numpy.where(columns==16,40.,10.),8.,dict(synthetic='40 m north-south wall over 10 m ground'))
