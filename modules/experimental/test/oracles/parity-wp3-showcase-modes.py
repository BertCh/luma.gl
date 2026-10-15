# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Round 8 showcase variants using the existing independent Shapely/NetworkX/SciPy
and Spaghetti generators. Native planar metres; f32 inputs; no GPU outputs.
Map matching: all 4 candidate counts x 4 route budgets x 4 cell sizes, on eight
parallel directed roads with distinct candidate distances. Every route fits the
smallest budget. Runtime slider endpoints/defaults are separate comparisons,
NOT exhaustive numeric Cartesian coverage. This supplements the real-data HMM
fixture, which supplies the non-nearest route-continuity cases.
K: all four scene band counts, with external length-weighted null simulations.
The GPU spec crosses these with all three block sizes and both Morton settings.
"""
import importlib.util
import itertools
import json
from pathlib import Path
import numpy

HERE = Path(__file__).resolve().parent

def loadModule(name):
    specification = importlib.util.spec_from_file_location(name, HERE / f'parity-wp3-{name}.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def makeMapCases():
    reference = loadModule('map-matching')
    positions = reference.roundInputs([[x, y] for y in [0, 12, 22, 33, 45, 58, 70, 84] for x in [0, 300]])
    edges = numpy.array([(index, index + 1) for index in range(0, 16, 2)])
    tracks = [reference.roundInputs([[25, 7], [150, 8], [275, 6]])]
    defaults = dict(sigma=20, beta=60, searchRadius=60, routeFactor=3, routeSlack=100)
    settingsCases = [('defaults', defaults)]
    for key, values in dict(sigma=[2, 80], beta=[5, 300], searchRadius=[10, 200], routeFactor=[1, 6], routeSlack=[0, 400]).items():
        for value in values:
            settingsCases.append((f'{key}-{value}', dict(defaults, **{key: value})))
    # Use the established generator's precision derivation and payload shape.
    temporaryName = 'showcase-template'
    reference.writeFixture(temporaryName, positions, edges, tracks, dict(description=__doc__), defaults)
    path = HERE / 'fixtures' / f'parity-wp3-map-matching-{temporaryName}.json'
    template = json.loads(path.read_text())
    path.unlink()
    fixtures = []
    for settingsName, settings in settingsCases:
        configurations = itertools.product([2, 4, 6, 8], [16, 32, 64, 128], [40, 60, 80, 100]) if settingsName == 'defaults' else [(8, 64, 60)]
        for candidateCount, routeNodeBudget, cellSize in configurations:
            actualSettings = dict(settings, searchRadius=min(settings['searchRadius'], 2 * cellSize))
            graph, lines, candidates, cutoffs, searches = reference.prepareQueries(positions, edges, tracks, actualSettings)
            candidates = [[choices[:candidateCount] for choices in track] for track in candidates]
            expected = [reference.solveTrack(track, choices, edges, lines, searches, actualSettings) for track, choices in zip(tracks, candidates)]
            fixture = dict(template, case=f'showcase/{settingsName}/candidates-{candidateCount}/budget-{routeNodeBudget}/cell-{cellSize}', settings=actualSettings,
                           candidateCount=candidateCount, routeNodeBudget=routeNodeBudget, cellSize=cellSize,
                           entryCapacity=4096, bounds=dict(minimum=[-cellSize, -cellSize], maximum=[400, 200]),
                           candidates=candidates, expected=expected, generator=Path(__file__).name)
            fixture['conventions'] = dict(template['conventions'], candidateCount=candidateCount, routeNodeBudget=routeNodeBudget)
            fixture['tolerances'] = dict(template['tolerances'], likelihood=dict(absolute=1e-6*max(1, *(abs(item['logLikelihood']) for item in expected)), relative=1e-4))
            assert max(len(search[0]) for search in searches.values()) < routeNodeBudget
            fixtures.append(fixture)
    output = HERE / 'fixtures/parity-wp3-map-matching-showcase.json'
    output.write_text(json.dumps(fixtures, separators=(',', ':'), allow_nan=False)+'\n')
    assert output.stat().st_size < 1_500_000
    print(output.name, len(fixtures), output.stat().st_size)


def makeKCases():
    reference = loadModule('network-k-function')
    for bandCount in [12, 24, 48, 96]:
        reference.generate(f'showcase-{bandCount}', [[0, 0], [400, 0], [400, 400], [800, 400]], [(0, 1), (1, 2), (2, 3)],
                           [[25, 0], [100, 0], [250, 0], [400, 50], [400, 175], [400, 350], [500, 400], [725, 400]],
                           1200, dict(description=__doc__), 1024, bandCount=bandCount)

if __name__ == '__main__':
    makeMapCases()
    makeKCases()
