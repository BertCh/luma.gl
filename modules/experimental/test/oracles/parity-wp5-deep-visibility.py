# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Sampler-matched external visibility reference (SciPy + GDAL 3.12).

GRASS r.viewshed uses angular sweep events; GDAL uses Wang reference planes.
Neither offers a switch selecting our finite ray sample set. This adapter
changes the reference INPUT, not its answer or the equality tolerance:
SciPy RegularGridInterpolator evaluates the bilinear DEM at the documented
sample distances. Subtract eye and curvature, then map sample i at distance d
to height i * (z(d) - eye - c*d*d)/d on a unit-spaced radial raster. Its slope
from the zero-height origin is unchanged. Put the target last, including its
height. On a single row Wang's recurrence is h[i] = max(z[i], i*h[i-1]/(i-1));
therefore GDAL tests exactly the same maximum apparent slope. Equality is
visible. Curvature is pre-applied, so GDAL receives -cc 0 and -oz/-tz 0.
This is sampler-matched parity, NOT native 2D GRASS/GDAL algorithm parity.

Sources: https://gdal.org/en/stable/programs/gdal_viewshed.html (Wang et al.
2000); https://grass.osgeo.org/grass-stable/manuals/r.viewshed.html.
Inputs/parameters are rounded to f32; interpolation and reference are f64.
Coarse samples: k/n, k=1..n-1, n=ceil(max(abs(delta))). Point samples: public
dyadic lattice, d>=1 metre and d<target distance, zero ignore/tolerance bands.
Anisotropic distances use hypot(dx*cellX,dy*cellY); no curvature on the eye.
Range and coincident endpoints are mapped explicitly to codes 2 and 1.
Every output cell is retained; no mismatch allowance or masked threshold ties.

Run with the designated venv Python -I. A child refenv Python uses GDAL's
in-memory driver (no installation or shared environment modification).
"""
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

EVAL = Path('/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval')
OUTPUT = Path(__file__).parent / 'fixtures'


def run_gdal_batch(source, destination):
    from osgeo import gdal
    import numpy
    gdal.UseExceptions()
    results = []
    for heights in json.loads(Path(source).read_text()):
        raster = gdal.GetDriverByName('MEM').Create('', len(heights), 1, 1, gdal.GDT_Float64)
        raster.SetGeoTransform((0, 1, 0, 1, 0, -1))
        raster.GetRasterBand(1).WriteArray(numpy.asarray([heights]))
        result = gdal.ViewshedGenerate(raster.GetRasterBand(1), 'MEM', '', [],
                                      .5, .5, 0, 0, 1, 0, 2, 3, 0, gdal.GVM_Edge, 0)
        results.append(int(result.ReadAsArray()[0, -1]))
    Path(destination).write_text(json.dumps(dict(version=gdal.VersionInfo('--version'), results=results)))


def run_gdal_grid(source, destination):
    from osgeo import gdal
    import numpy
    gdal.UseExceptions()
    fixture = json.loads(Path(source).read_text())
    base = fixture['input']
    raster = gdal.GetDriverByName('MEM').Create('', base['width'], base['height'], 1, gdal.GDT_Float64)
    raster.SetGeoTransform((0, base['cellSize'], 0, base['height']*base['cellSize'], 0, -base['cellSize']))
    raster.GetRasterBand(1).WriteArray(numpy.array(base['elevation']).reshape(base['height'], base['width']))
    settings = fixture['cases'][0]
    output = []
    for column, row in settings['observers']:
        result = gdal.ViewshedGenerate(raster.GetRasterBand(1), 'MEM', '', [],
            (column+.5)*base['cellSize'], (base['height']-row-.5)*base['cellSize'],
            settings['observerHeight'], settings['targetHeight'], 1, 0, 2, 3, 0, gdal.GVM_Edge, 0)
        output.append(result.ReadAsArray().ravel().tolist())
    Path(destination).write_text(json.dumps(output))


def make_minimal():
    import numpy
    import rasterio
    from rasterio.transform import from_origin
    import shlex
    # A two-by-two saddle embedded in a complete nine-by-nine finite DEM.
    terrain = numpy.zeros((9, 9), dtype=numpy.float32)
    terrain[4, 5] = terrain[5, 4] = 8
    fixture = dict(input=dict(name='saddle', width=9, height=9, cellSize=8,
        elevation=terrain.ravel().tolist(), elevationSha256=hashlib.sha256(terrain.astype('<f4').tobytes()).hexdigest()),
        cases=[dict(name='saddle-setting-0', observerHeight=1, targetHeight=0,
                    curvatureCoefficient=0, maximumDistance=0, observers=[[4,4],[2,4]])])
    commands = []
    with tempfile.TemporaryDirectory(prefix='parity-wp5-minimal-') as temporary:
        folder = Path(temporary)
        source, destination = folder/'source.json', folder/'gdal.json'
        source.write_text(json.dumps(fixture))
        subprocess.run([str(EVAL/'refenv/bin/python'), '-I', __file__, '--gdal-grid',
                        str(source), str(destination)], check=True)
        fixture['cases'][0]['gdal'] = json.loads(destination.read_text())
        raster = folder/'dem.tif'
        with rasterio.open(raster, 'w', driver='GTiff', width=9, height=9, count=1,
                           dtype='float32', transform=from_origin(0, 72, 8, 8), crs='EPSG:3857') as dataset:
            dataset.write(terrain, 1)
        commands.extend([['r.in.gdal', f'input={raster}', 'output=dem', '--quiet'], ['g.region','raster=dem']])
        for index, (column,row) in enumerate(fixture['cases'][0]['observers']):
            commands.append(['r.viewshed','-b','input=dem',f'output=view{index}',
                f'coordinates={(column+.5)*8},{(9-row-.5)*8}', 'observer_elevation=1','target_elevation=0','--quiet'])
            commands.append(['r.out.gdal',f'input=view{index}', f'output={folder}/view{index}.tif','--quiet'])
        # Independent cardinal hard-shadow reference on the same small DEM.
        for azimuth in [0,90,180,270]:
            commands.append(['r.sunmask','-z','elevation=dem',f'output=shadow{azimuth}',
                             'altitude=20',f'azimuth={azimuth}','--quiet'])
            commands.append(['r.out.gdal',f'input=shadow{azimuth}', f'output={folder}/shadow{azimuth}.tif','--quiet'])
        script = folder/'reference.sh'
        script.write_text('set -eu\n'+'\n'.join(shlex.join(command) for command in commands)+'\n')
        environment = dict(os.environ, PATH=str(EVAL/'refenv/bin')+os.pathsep+os.environ['PATH'], LC_ALL='C')
        subprocess.run([str(EVAL/'refenv/bin/grass'), '--tmp-project', str(raster), '--exec','sh',str(script)],
                       env=environment, text=True, capture_output=True, check=True)
        fixture['cases'][0]['grass'] = []
        for index in range(2):
            with rasterio.open(folder/f'view{index}.tif') as dataset:
                fixture['cases'][0]['grass'].append(dataset.read(1).ravel().tolist())
        fixture['shadows'] = []
        for azimuth in [0,90,180,270]:
            with rasterio.open(folder/f'shadow{azimuth}.tif') as dataset:
                shadow = dataset.read(1, masked=True).filled(0)
                fixture['shadows'].append(dict(azimuthDegrees=azimuth, altitudeDegrees=20,
                    sunVisibility=(shadow != 1).astype(int).ravel().tolist()))
    settings = fixture['cases'][0]
    for reference in ['grass','gdal']:
        settings[reference+'Counts'] = numpy.sum(numpy.asarray(settings[reference]) == 1, axis=0).tolist()
    fixture['commands'] = commands
    return fixture


def point_distances(maximum, cell_size):
    # Enumerate the published power-of-two lattice as octave sets, independently
    # of traversal, pyramid, shader code, and the production lattice helper.
    import numpy
    octaves = []
    for exponent in range(math.floor(math.log2(maximum)) + 1):
        base = 2.0 ** exponent
        desired = max(3.5e-4 * base, min(max(.01 * base, .25), .5 * cell_size))
        step = 2.0 ** min(exponent, max(-8, math.floor(math.log2(desired))))
        values = numpy.arange(base, 2 * base, step)
        octaves.append(values[values <= maximum])
    return numpy.concatenate(octaves)


def make_rays(base, settings, sampler, observer):
    import numpy
    from scipy.interpolate import RegularGridInterpolator
    width, height = base['width'], base['height']
    terrain = numpy.asarray(base['elevation'], dtype=numpy.float32).astype(float).reshape(height, width)
    spacing = numpy.broadcast_to(base['cellSize'], (2,)).astype(numpy.float32).astype(float)
    interpolate = RegularGridInterpolator((numpy.arange(height), numpy.arange(width)), terrain)
    eye = terrain[observer[1], observer[0]] + float(numpy.float32(settings['observerHeight']))
    curvature = float(numpy.float32(settings['curvatureCoefficient']))
    target_height = float(numpy.float32(settings['targetHeight']))
    maximum = float(numpy.float32(settings['maximumDistance']))
    lattice = point_distances(math.ceil(math.hypot(width, height) * max(spacing)), max(spacing))
    rays, codes, margins, witnesses = [], [], [], []
    for row in range(height):
        for column in range(width):
            delta = numpy.array([column, row]) - observer
            distance = numpy.linalg.norm(delta * spacing)
            if distance == 0 or (maximum and distance > maximum):
                codes.append(1 if distance == 0 else 2)
                margins.append(None)
                continue
            count = int(max(abs(delta)))
            distances = (numpy.arange(1, count) * distance / count if sampler == 'coarse'
                         else lattice[lattice < distance])
            positions = numpy.asarray(observer) + distances[:, None] / distance * delta
            elevations = interpolate(positions[:, ::-1])
            slopes = (elevations - eye) / distances - curvature * distances
            target_slope = (terrain[row, column] + target_height - eye) / distance - curvature * distance
            transformed = numpy.r_[0., slopes * numpy.arange(1, len(slopes) + 1), target_slope * (len(slopes) + 1)]
            rays.append(transformed.tolist())
            expected = int(not len(slopes) or target_slope >= max(slopes))
            codes.append(expected)
            margin = float(target_slope - max(slopes)) if len(slopes) else None
            margins.append(margin)
            if len(slopes):
                blocker = int(numpy.argmax(slopes))
                witnesses.append(dict(index=row * width + column, observer=list(observer),
                    target=[column, row], distance=float(distance), targetSlope=float(target_slope),
                    maximumSlope=float(slopes[blocker]), sampleDistance=float(distances[blocker]),
                    samplePosition=positions[blocker].tolist(), sampleElevation=float(elevations[blocker]),
                    expected=expected))
    return rays, codes, margins, witnesses


def generate():
    import numpy
    import scipy
    versions = dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
        grass=subprocess.run([str(EVAL/'refenv/bin/grass'), '--config', 'version'],
                             text=True, capture_output=True, check=True).stdout.strip())
    for name in ['alps-dem', 'grand-canyon-dem', 'ridge', 'saddle']:
        original = (make_minimal() if name == 'saddle' else
                    json.loads((OUTPUT / f'parity-wp5-round8-{name}-visibility.json').read_text()))
        cases = []
        for settings in original['cases']:
            case = dict(**settings, coarse=[], point=[], margins={}, witnesses={})
            for sampler in ['coarse', 'point']:
                all_rays, all_codes, all_margins, all_witnesses = [], [], [], []
                for observer in settings['observers']:
                    rays, codes, margins, witnesses = make_rays(original['input'], settings, sampler, observer)
                    all_rays.extend(rays)
                    all_codes.append(codes)
                    all_margins.append(margins)
                    differences = set(index for reference in ['grass', 'gdal'] for index, (actual, expected) in
                        enumerate(zip(codes, settings[reference][len(all_codes)-1])) if actual != expected)
                    all_witnesses.append([witness for witness in witnesses if witness['index'] in differences])
                with tempfile.TemporaryDirectory(prefix='parity-wp5-deep-') as temporary:
                    source, destination = Path(temporary)/'rays.json', Path(temporary)/'results.json'
                    source.write_text(json.dumps(all_rays))
                    subprocess.run([str(EVAL/'refenv/bin/python'), '-I', __file__, '--gdal-batch',
                                    str(source), str(destination)], check=True)
                    external = json.loads(destination.read_text())
                versions['gdal'] = external['version']
                reference = iter(external['results'])
                for observer_index, observer in enumerate(settings['observers']):
                    for index, code in enumerate(all_codes[observer_index]):
                        if code == 2 or index == observer[1]*original['input']['width']+observer[0]:
                            continue
                        assert next(reference) == code, (name, sampler, index)
                assert next(reference, None) is None
                case[sampler] = all_codes
                case['margins'][sampler] = all_margins
                case['witnesses'][sampler] = all_witnesses
                for reference in ['grass', 'gdal']:
                    print(settings['name'], sampler, reference,
                          [sum(value != wanted for value, wanted in zip(actual, expected)) for actual,expected in
                           zip(all_codes, settings[reference])], flush=True)
            case['coarseCounts'] = numpy.sum(numpy.asarray(case['coarse']) == 1, axis=0).tolist()
            cases.append(case)
        result = dict(versions=versions, conventions=__doc__, input=original['input'],
            commands=original.get('commands', []), shadows=original.get('shadows', []),
            generatorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=cases)
        path = OUTPUT / f'parity-wp5-deep-{name}-visibility.json'
        path.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
        assert path.stat().st_size < 1500000
        print(path.name, path.stat().st_size, flush=True)
    write_audit()


def write_audit():
    """Check every recorded failure's two native-reference mismatch counts.

    r9 retained stdout only for failed tests, not full GPU arrays. Matching
    counts is explanatory evidence, not proof of per-pixel post-change parity.
    """
    cases = {case['name']: case for path in OUTPUT.glob('parity-wp5-deep-*-visibility.json')
             for case in json.loads(path.read_text())['cases']}
    records = []
    classes = ['GPUTerrainViewshed', 'GPUTerrainLineOfSight',
               'GPUTerrainCumulativeViewshed', 'GPUPointHorizonVisibility']
    for line in (EVAL/'gpu/r9.log').read_text().splitlines():
        if not line.startswith('PARITY '):
            continue
        record = json.loads(line[7:])
        case_name = record['case'].rsplit('/', 1)[0]
        if case_name not in cases or record['class'] not in classes:
            continue
        settings = cases[case_name]
        reference = 'grass' if record['reference'].startswith('GRASS') else 'gdal'
        counts = record['class'] == 'GPUTerrainCumulativeViewshed'
        predicted = (settings['coarseCounts'] if counts else
                     settings['point' if record['class'] == 'GPUPointHorizonVisibility' else 'coarse'][0])
        expected = settings[reference+'Counts'] if counts else settings[reference][0]
        mismatches = sum(actual != wanted for actual, wanted in zip(predicted, expected))
        records.append(dict(**record, derivedMismatches=mismatches,
                            agreesWithLog=mismatches == record['mismatches']))
    assert records and all(record['agreesWithLog'] for record in records)
    (EVAL/'handoffs/parity-wp5-deep-visibility-audit.json').write_text(json.dumps(records, indent=2)+'\n')
    print('All', len(records), 'logged comparison counts explained; post-change GPU run pending.', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--gdal-batch':
        run_gdal_batch(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == '--gdal-grid':
        run_gdal_grid(sys.argv[2], sys.argv[3])
    else:
        generate()
