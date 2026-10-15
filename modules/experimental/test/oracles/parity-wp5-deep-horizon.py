# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Small independent r.horizon/SciPy examples for WP5's cap, sentinel and lattice cases.

GRASS 8.5 r.horizon: cardinal distance=1, R=6371000, zero eye height,
negative angles clipped to zero. Matched cap (radius-.5)*cell includes exactly
radius samples because GRASS stops AFTER sampling. Nodata padding moves the
initial southwest-cell sentinel outside valid terrain; no elevations are added.
Point reference: scipy.interpolate.interp1d on the declared sample set, followed
by the definition max((height-eye)/distance - distance/(2R)). Fractional case:
1.25,1.5,...,3.25 m; matched power-of-two case: 2,4,6 m. No GPU traversal port.
All input cells, runtime parameters and curvature are first f32-rounded. Exact
GRASS full arrays are retained and checked against the cardinal definition.
https://grass.osgeo.org/grass-stable/manuals/r.horizon.html
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import tempfile

import numpy
import rasterio
import scipy
from scipy.interpolate import interp1d
from rasterio.transform import from_origin

REFERENCE_SPEC = importlib.util.spec_from_file_location('reference', Path(__file__).with_name('parity-wp5-round8-grass.py'))
REFERENCE = importlib.util.module_from_spec(REFERENCE_SPEC)
REFERENCE_SPEC.loader.exec_module(REFERENCE)


def generate():
    cases = []
    REFERENCE.COMMANDS.clear()
    for name, cellSize in [('cap', float(numpy.float32(1.1))), ('sentinel', 2),
                            ('point-fractional', float(numpy.float32(1.1))), ('point-matched', 2)]:
        elevation = numpy.zeros((9, 9), dtype='float32')
        if name == 'cap':
            elevation[4, 8] = 10
        elif name == 'sentinel':
            elevation[8, 0] = 10
        else:
            elevation[4, 5] = 1
        curvature = float(numpy.float32(1 / (2 * 6371000)))
        with tempfile.TemporaryDirectory(prefix='parity-wp5-deep-horizon-') as temporary:
            folder = Path(temporary)
            raster = folder / 'input.tif'
            with rasterio.open(raster, 'w', driver='GTiff', width=9, height=9, count=1,
                               dtype='float32', transform=from_origin(0, 9*cellSize, cellSize, cellSize),
                               crs='+proj=eqc +R=6371008.8 +units=m +no_defs') as dataset:
                dataset.write(elevation, 1)
            commands = [['r.in.gdal', 'input='+str(raster), 'output=elevation', '--quiet']]
            for label, padding, cap in [('raw', 0, 3*cellSize), ('matched', 4, 2.5*cellSize)]:
                commands.append(['g.region', f'n={(9+padding)*cellSize}', f's={-padding*cellSize}',
                                 f'e={(9+padding)*cellSize}', f'w={-padding*cellSize}', f'res={cellSize}'])
                commands.append(['r.horizon', '-d', 'elevation=elevation', 'step=90', 'distance=1',
                                 f'maxdistance={cap}', 'output='+label, 'nprocs=1', '--quiet'])
                for angle in [0, 90, 180, 270]:
                    commands.append(['r.out.gdal', f'input={label}_{angle:03}',
                                     f'output={folder}/{label}-{angle}.tif', 'type=Float64', 'nodata=-9999', '--quiet'])
            script = folder/'reference.sh'
            script.write_text('set -eu\n'+'\n'.join(map(shlex.join, commands))+'\n')
            REFERENCE.COMMANDS.extend(commands)
            REFERENCE.execute([REFERENCE.BIN/'grass', '--tmp-project', raster, '--exec', 'sh', script])
            output = {}
            for label, padding in [('raw', 0), ('matched', 4)]:
                output[label] = numpy.stack([REFERENCE.readRaster(folder/f'{label}-{angle}.tif')[padding:padding+9,padding:padding+9]
                                             for angle in [90,0,270,180]], axis=-1).ravel().tolist()
        definition = []
        for row in range(9):
            for column in range(9):
                for rowStep, columnStep in [(-1,0),(0,1),(1,0),(0,-1)]:
                    tangents = [0]
                    for count in range(1,4):
                        sampleRow, sampleColumn = row+count*rowStep, column+count*columnStep
                        if not (0 <= sampleRow < 9 and 0 <= sampleColumn < 9):
                            break
                        distance = count*cellSize
                        tangents.append((float(elevation[sampleRow,sampleColumn])-float(elevation[row,column]))/distance-curvature*distance)
                    definition.append(math.degrees(math.atan(max(tangents))))
        assert numpy.max(numpy.abs(numpy.array(definition)-output['matched'])) < 1e-4
        case = dict(name=name, width=9, height=9, cellSize=cellSize, elevation=elevation.ravel().tolist(),
                    maximumRadius=3, curvatureCoefficient=curvature, horizon=output['matched'], rawHorizon=output['raw'],
                    cardinalDefinition=definition)
        if name.startswith('point'):
            distances = numpy.arange(1.25,3.26,.25) if name == 'point-fractional' else numpy.array([2.,4.,6.])
            pointAngles, pointDistances = [], []
            for rowStep, columnStep in [(-1,0),(0,1),(1,0),(0,-1)]:
                values = [elevation[4+count*rowStep,4+count*columnStep] for count in range(5)]
                heights = interp1d(numpy.arange(5)*cellSize, values)(distances)
                tangents = heights/distances-curvature*distances
                winningIndex = int(numpy.argmax(tangents))
                pointAngles.append(max(0,math.degrees(math.atan(tangents[winningIndex]))))
                pointDistances.append(float(distances[winningIndex]) if tangents[winningIndex]>0 else 0)
            case.update(pointHorizon=pointAngles, pointDistance=pointDistances, sampleDistances=distances.tolist(),
                        minimumDistance=cellSize, maximumDistance=3*cellSize,
                        latticeOptions={} if name=='point-fractional' else dict(minimumStep=2,cellSteps=1,stepFactor=1/3))
        cases.append(case)
        index = (4*9+4)*4+1 if name!='sentinel' else (7*9)*4+2
        print(name,'raw/matched/point degrees',output['raw'][index],output['matched'][index],case.get('pointHorizon'),flush=True)
    payload = dict(cases=cases, conventions=__doc__, versions={**REFERENCE.VERSIONS,'scipy':scipy.__version__},
                   commands=REFERENCE.COMMANDS,generatorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (REFERENCE.OUTPUT/'parity-wp5-deep-horizon.json').write_text(json.dumps(payload,separators=(',',':'))+'\n')


if __name__ == '__main__':
    generate()
