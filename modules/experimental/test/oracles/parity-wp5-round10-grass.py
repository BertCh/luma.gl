# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Round 10 reference adapters, preserving the raw Round 8 GRASS outputs.

No terrain is added: extend the GRASS region by nine NODATA cells on each side,
then crop back to the original 33x33 finite domain. This keeps r.geomorphon's
rolling-row buffer boundary and r.horizon's initial (0,0) search sentinel outside
real terrain. Apply the documented skip+1 invalid output border explicitly.
Use a strict metric geomorphon search radius just inside eight cells: the
nearest excluded cardinal sample is exactly eight cells, the farthest included
one is seven (diagonal five). The 1e-5 guard is far smaller than either gap.
Skip metres are placed inside the integer-cell bin to avoid floor roundoff.
GRASS tests horizon maxdistance AFTER sampling. A cap of 15.5 cells therefore
includes exactly sixteen cardinal samples for every f32 spacing, rather than
occasionally seventeen. No angular, distance, or integer tolerance changes.

The raw references remain available, including all real-data tie discrepancies.
Source analysis: https://github.com/OSGeo/grass/blob/main/raster/r.geomorphon/pattern.c
and https://github.com/OSGeo/grass/blob/main/raster/r.horizon/main.c .
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import numpy
import rasterio
from rasterio.transform import from_origin

spec = importlib.util.spec_from_file_location('round8', Path(__file__).with_name('parity-wp5-round8-grass.py'))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def generate(name):
    original = json.loads((reference.OUTPUT / f'parity-wp5-round8-{name}-geomorphons.json').read_text())
    original_horizon = json.loads((reference.OUTPUT / f'parity-wp5-round8-{name}-horizon.json').read_text())
    base = original['input']
    cell = base['cellSize']
    width, height = base['width'], base['height']
    padding = 9
    reference.COMMANDS.clear()
    with tempfile.TemporaryDirectory(prefix='parity-wp5-r10-') as temporary:
        folder = Path(temporary)
        raster = folder / 'input.tif'
        with rasterio.open(raster, 'w', driver='GTiff', width=width, height=height, count=1,
                           dtype='float32', transform=from_origin(0, height*cell, cell, cell),
                           crs='+proj=eqc +R=6371008.8 +units=m +no_defs') as dataset:
            dataset.write(numpy.asarray(base['elevation'], dtype='float32').reshape(height, width), 1)
        commands = [['r.in.gdal', 'input='+str(raster), 'output=elevation', '--quiet'],
                    ['g.region', 'raster=elevation'],
                    ['g.region', f'n={(height+padding)*cell}', f's={-padding*cell}',
                     f'e={(width+padding)*cell}', f'w={-padding*cell}']]
        def export(map_name):
            commands.append(['r.out.gdal', 'input='+map_name, 'output='+str(folder/(map_name+'.tif')),
                             'type=Float64', 'nodata=-9999', '--quiet'])
        for case in original['cases']:
            commands.append(['r.geomorphon', '-m', 'elevation=elevation', 'forms='+case['name'],
                             'ternary='+case['name']+'_ternary', f'search={8*cell*(1-1e-5)}',
                             f'skip={(case["skipRadius"]+.25)*cell}', 'flat=1',
                             f'dist={case["flatDistance"]}',
                             'comparison='+case['comparison'].replace('-', '_'), '--quiet'])
            export(case['name']); export(case['name']+'_ternary')
        commands.append(['r.horizon', '-d', 'elevation=elevation', 'step=90', 'distance=1',
                         f'maxdistance={15.5*cell}', 'output=horizon', 'nprocs=1', '--quiet'])
        for angle in [0, 90, 180, 270]: export(f'horizon_{angle:03d}')
        commands.append([str(reference.SKYVIEW), 'input=elevation', 'output=skyview', 'ndir=4',
                         f'maxdistance={15.5*cell}', 'nprocs=1', '--quiet'])
        export('skyview')
        script = folder / 'reference.sh'
        script.write_text('set -eu\n'+'\n'.join(map(shlex.join, commands))+'\n')
        reference.execute([reference.BIN/'grass', '--tmp-project', raster, '--exec', 'sh', script])
        def read(map_name):
            return reference.readRaster(folder/(map_name+'.tif'))[padding:padding+height, padding:padding+width]
        cases = []
        for old_case in original['cases']:
            case = dict(old_case)
            case['rawForms'], case['rawTernary'] = case['forms'], case['ternary']
            border = case['skipRadius']+1
            for metric, suffix in [('forms', ''), ('ternary', '_ternary')]:
                values = read(case['name']+suffix).astype(int)
                values[:border] = 0; values[-border:] = 0
                values[:, :border] = 0; values[:, -border:] = 0
                case[metric] = values.ravel().tolist()
            cases.append(case)
        horizon_case = dict(original_horizon['cases'][0])
        horizon_case['rawHorizon'] = horizon_case['horizon']
        horizon_case['rawSkyViewFactor'] = horizon_case['skyViewFactor']
        horizon_case['horizon'] = numpy.stack([read(f'horizon_{angle:03d}') for angle in [90,0,270,180]], axis=-1).ravel().tolist()
        horizon_case['skyViewFactor'] = read('skyview').ravel().tolist()
        for topic, output_cases in [('geomorphons', cases), ('horizon', [horizon_case])]:
            payload = dict(input=base, cases=output_cases, versions=reference.VERSIONS,
                           conventions=__doc__, commands=commands+reference.COMMANDS,
                           generatorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           rawReferenceGeneratorSha256=original['generatorSha256'])
            path = reference.OUTPUT/f'parity-wp5-round10-{name}-{topic}.json'
            path.write_text(json.dumps(payload, separators=(',', ':'))+'\n')
            assert path.stat().st_size < 1500000
            print(path.name, path.stat().st_size, flush=True)


if __name__ == '__main__':
    for dataset_name in ['alps-dem', 'grand-canyon-dem', 'ridge']: generate(dataset_name)
