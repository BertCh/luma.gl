# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Round 10 diagnostic bounds, NOT acceptance tolerances or GPU parity claims.

Visibility: Hamming distance is a metric, hence for any GPU output U and two
references G,D, H(U,G)+H(U,D)>=H(G,D) and max(H(U,G),H(U,D))>=ceil(H(G,D)/2).
The positive lower bound proves simultaneous exact equality is impossible.
It is not an upper error budget; retain strict E assertions until sampler
contracts are aligned. Cumulative masks use the same metric on integer counts.

Cardinal horizon: h(d) is piecewise linear between DEM centres, so apparent
slope t(d)=(h(d)-eye)/d-c*d has derivative -(intercept-eye)/d**2-c.
A Lipschitz bound on angle in degrees is (180/pi)*max |t'(d)|, because atan
is 1-Lipschitz. For two finite sample sets whose Hausdorff distance is delta,
their maximum angles differ by at most L*delta. This derives a SAMPLING bound,
not float32 error. Distances of maximizing samples have no corresponding
small bound without an angular separation condition: they can span the ray.
Report the unconditioned ray-span bound; never use it as an acceptance tolerance.
The separate SciPy fixture evaluates the exact public sample set and keeps D
for angles and distances. This is explicitly composite component parity, not
GRASS equality. No established external visibility tool exposes this dyadic
sampler. interp1d supplies independently implemented linear interpolation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
The apparent-angle definition is documented at
https://grass.osgeo.org/grass-stable/manuals/r.horizon.html .
"""
import hashlib
import json
import math
from pathlib import Path
import platform
import numpy
from scipy.interpolate import interp1d
import scipy

ROOT = Path(__file__).resolve().parents[4]
FOLDER = Path(__file__).parent/'fixtures'
EVAL = Path('/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/82c646e8-3e98-44d4-84b8-212745483f4a/scratchpad/eval')


def lattice(minimum, maximum, cell):
    # Public dyadic sample-set definition, independent of traversal or WGSL.
    distances = []
    for exponent in range(math.floor(math.log2(minimum)), math.floor(math.log2(maximum))+1):
        base = 2.**exponent
        requested = max(.00035*base, min(max(.01*base, .25), .5*cell))
        step = 2.**min(exponent, max(-8, math.floor(math.log2(requested))))
        values = base + numpy.arange(round(base/step))*step
        distances.extend(values[(values >= minimum) & (values <= maximum)])
    return numpy.asarray(distances)


def read(name, topic, round_number=8):
    return json.loads((FOLDER/f'parity-wp5-round{round_number}-{name}-{topic}.json').read_text())


def run():
    records = []
    active = False
    failed_cases = []
    for line in (EVAL/'gpu/r9.log').read_text().splitlines():
        if line.startswith('stdout |'): active = 'parity-wp5-' in line
        if active and 'PARITY {' in line: records.append(json.loads(line[line.index('{'):]))
        if ' FAIL ' in line and 'parity-wp5-' in line:
            failed_cases.append(line.split(' > ', 1)[1])
    visibility = []
    horizons = []
    for name in ['alps-dem', 'grand-canyon-dem', 'ridge']:
        fixture = read(name, 'visibility')
        for case in fixture['cases']:
            for observer_index in range(len(case['observers'])):
                disagreements = sum(first != second for first, second in zip(case['grass'][observer_index], case['gdal'][observer_index]))
                visibility.append(dict(case=case['name'], metric=f'observer-{observer_index}',
                                       referenceDisagreements=disagreements, unavoidableWorstMismatchCount=math.ceil(disagreements/2)))
            for class_name in ['GPUTerrainViewshed','GPUTerrainLineOfSight','GPUTerrainCumulativeViewshed','GPUPointHorizonVisibility']:
                counts = class_name == 'GPUTerrainCumulativeViewshed'
                first = case['grassCounts'] if counts else case['grass'][0]
                second = case['gdalCounts'] if counts else case['gdal'][0]
                disagreements = sum(left != right for left, right in zip(first, second))
                for traversal in ['march','pyramid']:
                    measured = [record for record in records if record['class']==class_name and record['case']==case['name']+'/'+traversal]
                    if not measured:
                        # Vitest only retained stdout for failures in this log.
                        assert disagreements == 0
                        continue
                    assert len(measured)==2, (class_name, case['name'], traversal, len(measured))
                    total = sum(record['mismatches'] for record in measured)
                    assert total >= disagreements
                    visibility.append(dict(case=case['name']+'/'+traversal, className=class_name,
                                           referenceDisagreements=disagreements, gpuMismatchSum=total,
                                           derivedLowerBound=disagreements, satisfiesNecessaryBound=total>=disagreements))
        fixture = read(name, 'horizon', 10)
        cell = fixture['input']['cellSize']
        elevation = numpy.asarray(fixture['input']['elevation']).reshape(33,33)
        case = fixture['cases'][0]
        grid_angles = numpy.asarray(case['horizon']).reshape(33,33,4)[16,16]
        point = sorted(case['point'][0]['horizons'], key=lambda value:value['azimuth'])
        for direction,(row_step,column_step) in enumerate([(-1,0),(0,1),(1,0),(0,-1)]):
            distances = numpy.arange(17)*cell
            heights = elevation[16+numpy.arange(17)*row_step,16+numpy.arange(17)*column_step]
            samples = lattice(cell,16*cell,cell)
            height_function = interp1d(distances,heights,kind='linear')
            tangents = (height_function(samples)-heights[0])/samples-samples/(2*6371000)
            angles = numpy.maximum(0,numpy.degrees(numpy.arctan(tangents)))
            discrete = distances[1:]
            delta = max(numpy.abs(discrete[:,None]-samples[None,:]).min(axis=0).max(),numpy.abs(discrete[:,None]-samples[None,:]).min(axis=1).max())
            segment_slopes = numpy.diff(heights)/cell
            intercepts = heights[:-1]-segment_slopes*distances[:-1]
            derivative_bound = (numpy.abs(intercepts[1:]-heights[0])/distances[1:-1]**2+1/(2*6371000)).max()
            angle_bound = float(numpy.degrees(derivative_bound)*delta)
            independent_angle = float(angles.max())
            disagreement = abs(independent_angle-point[direction]['angle'])
            # GRASS point JSON rounds to six decimals; the grid output is FCELL.
            assert disagreement <= angle_bound+1e-4
            horizons.append(dict(case=case['name'],azimuth=point[direction]['azimuth'],
                                 grassPointAngle=point[direction]['angle'],grassGridAngle=float(grid_angles[direction]),
                                 pointVsGridAbs=abs(point[direction]['angle']-float(grid_angles[direction])),
                                 scipyLatticeAngle=independent_angle,grassVsScipyAbs=disagreement,
                                 maximumSlopeDerivative=float(derivative_bound),sampleHausdorffMeters=float(delta),
                                 derivedAngleBoundDegrees=angle_bound,
                                 grassDistance=point[direction]['distance'],scipyDistance=float(samples[numpy.argmax(angles)]) if independent_angle>0 else 0,
                                 derivedUnconditionedDistanceBoundMeters=float(16*cell)))
    result = dict(versions=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__),
                  generatorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),derivation=__doc__,
                  failingCases=failed_cases,visibility=visibility,horizons=horizons,
                  gpuBefore=[record for record in records if record['mismatches']])
    output = EVAL/'handoffs/parity-wp5-round10-audit.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    point_cases = []
    for name in ['alps-dem', 'grand-canyon-dem', 'ridge']:
        selected = [value for value in horizons if value['case'] == name+'-cardinal']
        source = read(name, 'horizon', 10)['input']
        point_cases.append(dict(name=name+'-cardinal', elevationSha256=source['elevationSha256'],
                                cellSize=source['cellSize'], minimumDistance=source['cellSize'],
                                maximumDistance=16*source['cellSize'], observer=[16,16,0],
                                curvatureCoefficient=1/(2*6371000), horizon=[value['scipyLatticeAngle'] for value in selected],
                                distance=[value['scipyDistance'] for value in selected]))
    point_fixture = dict(versions=result['versions'], generatorSha256=result['generatorSha256'],
                         conventions=__doc__, cases=point_cases)
    (FOLDER/'parity-wp5-round10-point-horizon.json').write_text(json.dumps(point_fixture,indent=2)+'\n')
    print(output)
    print('Failing cases:',len(failed_cases))
    for name in ['alps-dem','grand-canyon-dem','ridge']:
        selected=[value for value in horizons if value['case']==name+'-cardinal']
        print(name,'GRASS point/grid max',max(value['pointVsGridAbs'] for value in selected),
              'GRASS/SciPy max',max(value['grassVsScipyAbs'] for value in selected),
              'sampling bound max',max(value['derivedAngleBoundDegrees'] for value in selected))


if __name__=='__main__': run()
