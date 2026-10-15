# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""GRASS 8.5 with exact-ratio tie adjudication; no empirical mismatch allowance.

Reference: Jasiewicz & Stepinski (2013), Geomorphology 182, 147-156,
https://doi.org/10.1016/j.geomorph.2012.11.005, with GRASS's ANGLEV1/2 variants:
https://github.com/OSGeo/grass/blob/main/raster/r.geomorphon/pattern.c .
ANGLEV1 is NOT the paper's original signed zenith/nadir difference test.

Run with the PARITY-BRIEF reference Python, -I. The original finite 33x33
terrains are unchanged. Nodata padding and strict radius follow Round 10.
First run GRASS in the original metric coordinates, then in exact unit-grid
coordinates with tan(flat')=cellSize*tan(flat), dist'=dist/cellSize. Heights
are unchanged FCELL values. This preserves every ideal angle ordering and
threshold decision while removing GRASS region-coordinate serialization error.
GRASS has no exact-ratio tie switch. Its remaining libm atan2 ties are adapted
by a COMPOSITE reference: fractions.Fraction certifies equality of the input
height/step ratios. No angle epsilon is used to label a tie. Every original
and normalized disagreement must have a GRASS one-off profile proving the
same samples and a tied extremum; otherwise generation fails. Untied GRASS
outputs remain authoritative, with full arrays and all certificates retained.

Tie convention: first sample attains each signed extremum. Equal absolute
extrema yield 0 (anglev1), +1 (anglev2), or the farther extremum's sign then
+1 (anglev2-distance). This is the contributor's documented convention.
The algebraic slope reference is independent of the TS/WGSL implementation.
Integer output tolerance remains exactly zero. No input cell is excluded.
"""
import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path
import shlex
import tempfile

import numpy
import rasterio
from rasterio.transform import from_origin

REFERENCE_SPEC = importlib.util.spec_from_file_location(
    'grass_reference', Path(__file__).with_name('parity-wp5-round8-grass.py'))
REFERENCE = importlib.util.module_from_spec(REFERENCE_SPEC)
REFERENCE_SPEC.loader.exec_module(REFERENCE)
DIRECTIONS = [('NE', -1, 1), ('N', -1, 0), ('NW', -1, -1), ('W', 0, -1),
              ('SW', 1, -1), ('S', 1, 0), ('SE', 1, 1), ('E', 0, 1)]
# Published GRASS ten-form lookup (negative count, positive count).
FORM_TABLE = [[1,1,1,8,8,9,9,9,10], [1,1,8,8,8,9,9,9,0],
              [1,4,6,6,7,7,9,0,0], [4,4,6,6,6,7,0,0,0],
              [4,4,5,6,6,0,0,0,0], [3,3,5,5,0,0,0,0,0],
              [3,3,3,0,0,0,0,0,0], [3,3,0,0,0,0,0,0,0],
              [2,0,0,0,0,0,0,0,0]]


def encodePattern(digits):
    values = [digit + 1 for digit in digits]
    return min(sum(sequence[(position + rotation) % 8] * 3**position
                   for position in range(8))
               for sequence in [values, values[::-1]] for rotation in range(8))


def classifyDirection(samples, case, cellSize, diagonal):
    if not samples:
        return 0
    maximum = max(samples, key=lambda sample: sample['ratio'])
    minimum = min(samples, key=lambda sample: sample['ratio'])
    step = cellSize * (math.sqrt(2) if diagonal else 1)
    flatDistance = float(numpy.float32(case['flatDistance']))
    if not case['skipRadius'] * cellSize < flatDistance < case['searchRadius'] * cellSize:
        flatDistance = 0
    threshold = math.tan(math.radians(case['flatThresholdDegrees']))

    def exceeds(sample):
        distance = sample['count'] * step
        referenceDistance = min(distance, flatDistance) if flatDistance else distance
        # An explicit separation check prevents a float threshold boundary becoming a tie patch.
        difference = abs(float(sample['height'])) - threshold * referenceDistance
        assert abs(difference) > 2**-18 * max(1, abs(float(sample['height'])), threshold * referenceDistance)
        return difference > 0

    maximumOver, minimumOver = exceeds(maximum), exceeds(minimum)
    if not (maximumOver or minimumOver):
        return 0
    if case['comparison'] != 'anglev1' and maximumOver != minimumOver:
        return 1 if maximumOver else -1
    if abs(maximum['ratio']) != abs(minimum['ratio']):
        return 1 if abs(maximum['ratio']) > abs(minimum['ratio']) else -1
    if case['comparison'] == 'anglev1':
        return 0
    if case['comparison'] == 'anglev2-distance':
        return 1 if maximum['count'] >= minimum['count'] else -1
    return 1


def calculateDefinition(base, case):
    width, height, cellSize = base['width'], base['height'], base['cellSize']
    elevation = base['elevation']
    results = dict(forms=[], ternary=[], pattern=[])
    profiles = {}
    for row in range(height):
        for column in range(width):
            index = row * width + column
            if min(row, column, height - 1 - row, width - 1 - column) <= case['skipRadius']:
                for values in results.values():
                    values.append(0)
                continue
            digits, directions = [], {}
            for direction, rowStep, columnStep in DIRECTIONS:
                samples = []
                for count in range(case['skipRadius'] + 1, case['searchRadius']):
                    sampleRow, sampleColumn = row + count * rowStep, column + count * columnStep
                    # Squared integer distances make the strict circular domain exact.
                    if count**2 * (rowStep**2 + columnStep**2) >= case['searchRadius']**2:
                        break
                    if not (0 <= sampleRow < height and 0 <= sampleColumn < width):
                        break
                    difference = Fraction(elevation[sampleRow * width + sampleColumn]) - Fraction(elevation[index])
                    # These datasets have exact f32 differences and products through step 7.
                    assert Fraction(float(numpy.float32(float(difference)))) == difference
                    for multiplier in range(1, case['searchRadius']):
                        assert Fraction(float(numpy.float32(float(difference * multiplier)))) == difference * multiplier
                    samples.append(dict(count=count, height=difference, ratio=difference / count))
                digit = classifyDirection(samples, case, cellSize, bool(rowStep and columnStep))
                digits.append(digit)
                directions[direction] = dict(digit=digit, samples=samples)
            results['forms'].append(FORM_TABLE[digits.count(-1)][digits.count(1)])
            results['ternary'].append(encodePattern(digits))
            results['pattern'].append(sum((digit + 1) * 3**position for position, digit in enumerate(digits)))
            profiles[index] = directions
    return results, profiles


def getArguments(case, cellSize, normalized):
    referenceCell = 1 if normalized else cellSize
    threshold = case['flatThresholdDegrees']
    if normalized:
        threshold = math.degrees(math.atan(cellSize * math.tan(math.radians(threshold))))
    return ['r.geomorphon', '-m', 'elevation=elevation',
            f'search={case["searchRadius"] * referenceCell * .99999}',
            f'skip={(case["skipRadius"] + .25) * referenceCell}',
            f'flat={threshold}', f'dist={case["flatDistance"] / (cellSize if normalized else 1)}',
            'comparison=' + case['comparison'].replace('-', '_'), '--quiet']


def runScript(raster, folder, commands):
    script = folder / 'reference.sh'
    script.write_text('set -eu\n' + '\n'.join(map(shlex.join, commands)) + '\n')
    REFERENCE.COMMANDS.extend(commands)
    REFERENCE.execute([REFERENCE.BIN / 'grass', '--tmp-project', raster, '--exec', 'sh', script])


def runReference(base, cases, normalized):
    width, height, cellSize = base['width'], base['height'], base['cellSize']
    referenceCell = 1 if normalized else cellSize
    padding = 9
    results = []
    with tempfile.TemporaryDirectory(prefix='parity-wp5-deep-') as temporary:
        folder = Path(temporary)
        raster = folder / 'input.tif'
        with rasterio.open(raster, 'w', driver='GTiff', width=width, height=height, count=1,
                           dtype='float32', transform=from_origin(0, height * referenceCell, referenceCell, referenceCell),
                           crs='+proj=eqc +R=6371008.8 +units=m +no_defs') as dataset:
            dataset.write(numpy.asarray(base['elevation'], dtype='float32').reshape(height, width), 1)
        setup = [['r.in.gdal', 'input=' + str(raster), 'output=elevation', '--quiet'],
                 ['g.region', f'n={(height + padding) * referenceCell}', f's={-padding * referenceCell}',
                  f'e={(width + padding) * referenceCell}', f'w={-padding * referenceCell}', f'res={referenceCell}']]
        commands = list(setup)
        for case in cases:
            commands.append(getArguments(case, cellSize, normalized) +
                            ['forms=' + case['name'], 'ternary=' + case['name'] + '_ternary'])
            for metric, suffix in [('forms', ''), ('ternary', '_ternary')]:
                commands.append(['r.out.gdal', 'input=' + case['name'] + suffix,
                                 'output=' + str(folder / (case['name'] + metric + '.tif')),
                                 'type=Float64', 'nodata=-9999', '--quiet'])
        runScript(raster, folder, commands)
        commands = list(setup)
        for case in cases:
            definition, profiles = calculateDefinition(base, case)
            raw = {}
            for metric in ['forms', 'ternary']:
                values = REFERENCE.readRaster(folder / (case['name'] + metric + '.tif'))[padding:padding+height, padding:padding+width].astype(int)
                border = case['skipRadius'] + 1
                values[:border] = 0
                values[-border:] = 0
                values[:, :border] = 0
                values[:, -border:] = 0
                raw[metric] = values.ravel().tolist()
            differences = [index for index in range(width * height)
                           if any(raw[metric][index] != definition[metric][index] for metric in raw)]
            for index in differences:
                row, column = divmod(index, width)
                commands.append(getArguments(case, cellSize, normalized) +
                                [f'coordinates={(column+.5)*referenceCell},{(height-row-.5)*referenceCell}',
                                 f'profiledata={folder}/{case["name"]}-{index}.json', 'profileformat=json'])
            results.append(dict(raw=raw, definition=definition, profiles=profiles, differences=differences))
        runScript(raster, folder, commands)
        for case, result in zip(cases, results):
            certificates = []
            for index in result['differences']:
                external = json.loads((folder / f'{case["name"]}-{index}.json').read_text())
                assert external['intermediate_data']['ternary_498'] == result['raw']['ternary'][index]
                directions = []
                for direction, _, _ in DIRECTIONS:
                    expected = result['profiles'][index][direction]
                    actualDigit = external['intermediate_data']['pattern'][direction]
                    if actualDigit == expected['digit']:
                        continue
                    samples = expected['samples']
                    rawSamples = external['search_rel_elevation_m'][direction]
                    assert {f'step_{sample["count"]}': float(sample['height']) for sample in samples} == {
                        step: value for step, value in rawSamples.items() if value is not None}
                    maximum = max(samples, key=lambda sample: sample['ratio'])
                    minimum = min(samples, key=lambda sample: sample['ratio'])
                    # A differing digit needs an EXACT extremum tie, never merely a small angle.
                    assert abs(maximum['ratio']) == abs(minimum['ratio']), (base['name'], case['name'], index, direction)
                    directions.append(dict(direction=direction, grassDigit=actualDigit, matchedDigit=expected['digit'],
                                           maximumRatio=str(maximum['ratio']), minimumRatio=str(minimum['ratio']),
                                           samples=[dict(count=sample['count'], height=str(sample['height']), ratio=str(sample['ratio']))
                                                    for sample in samples]))
                assert directions
                certificates.append(dict(index=index, row=index // width, column=index % width, directions=directions))
            result['certificates'] = certificates
            del result['profiles']
    return results


def generate(name):
    original = json.loads((REFERENCE.OUTPUT / f'parity-wp5-round10-{name}-geomorphons.json').read_text()) if name != 'plane' else None
    if original:
        base, cases = original['input'], original['cases']
    else:
        base = dict(name='plane-9x9', width=9, height=9, cellSize=float(numpy.float32(6.639009)),
                    elevation=[100 + column * .5 + row * .25 for row in range(9) for column in range(9)])
        base['elevationSha256'] = hashlib.sha256(numpy.asarray(base['elevation'], dtype='<f4').tobytes()).hexdigest()
        cases = [dict(name=f'geom_{position}', searchRadius=4, skipRadius=0,
                      comparison=comparison, flatThresholdDegrees=1, flatDistance=0)
                 for position, comparison in enumerate(['anglev1', 'anglev2', 'anglev2-distance'])]
    REFERENCE.COMMANDS.clear()
    rawResults = runReference(base, cases, False)
    normalizedResults = runReference(base, cases, True)
    outputCases = []
    for case, raw, normalized in zip(cases, rawResults, normalizedResults):
        if original:
            for metric in ['forms', 'ternary']:
                assert raw['raw'][metric] == case[metric], (name, case['name'], metric)
        # Keep the external normalized reference unless a certified exact tie changed it.
        matched = {metric: list(values) for metric, values in normalized['raw'].items()}
        for certificate in normalized['certificates']:
            for metric in matched:
                matched[metric][certificate['index']] = normalized['definition'][metric][certificate['index']]
        assert all(matched[metric] == normalized['definition'][metric] for metric in matched)
        outputCase = {key: value for key, value in case.items() if key not in ['forms', 'ternary', 'rawForms', 'rawTernary']}
        outputCase.update(matched, pattern=normalized['definition']['pattern'], rawForms=raw['raw']['forms'], rawTernary=raw['raw']['ternary'],
                           normalizedForms=normalized['raw']['forms'], normalizedTernary=normalized['raw']['ternary'],
                           originalTieCertificates=raw['certificates'], normalizedTieCertificates=normalized['certificates'])
        outputCases.append(outputCase)
        print(name, case['name'], 'original/normalized certified pixels', len(raw['certificates']), len(normalized['certificates']), flush=True)
    payload = dict(input=base, cases=outputCases, versions=REFERENCE.VERSIONS, conventions=__doc__, commands=REFERENCE.COMMANDS,
                   generatorSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output = REFERENCE.OUTPUT / f'parity-wp5-deep-{name}-geomorphons.json'
    output.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
    assert output.stat().st_size < 1500000


if __name__ == '__main__':
    for datasetName in ['alps-dem', 'grand-canyon-dem', 'ridge', 'plane']:
        generate(datasetName)
