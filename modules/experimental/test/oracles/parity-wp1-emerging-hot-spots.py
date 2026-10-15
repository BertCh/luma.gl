# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Expanded-cube esda.G_Local, SciPy Kendall tau-b, statsmodels FDR reference.

Run with the supplied Python -I. All Chicago nature events are binned into an
8x8 lon/lat lattice and 12 calendar months (local-clock timestamps as supplied).
This is parity on the count cube, not a distance/projection or aggregation test.
Binary closed radius-1 lattice neighborhoods include self and current/previous
slices. One directed W over ALL finite cube bins preserves the whole-cube null
population; separate per-slice esda calls would test a different statistic.
Masked cells and NaN bins are removed from both the population and every W row.
FDR correction family is all finite Gi* bins, two-sided normal p, no permutations.

Composite reference: Kendall (1945), The treatment of ties in ranking problems,
Biometrika 33:239-251, doi:10.1093/biomet/33.3.239. scipy.stats.kendalltau gives
tau-b; recover integer S from its tie denominator, then apply the published
Var(S) and the documented continuity correction (S-sign(S))/sqrt(Var(S)).
Trends use f32-rounded reference Gi* (the public inter-pass storage format).
No approximate tie tolerance is applied. Classification is a separate decision
table from https://doc.esri.com/en/arcgis-pro/latest/tool-reference/space-time-pattern-mining/learnmoreemerging.html
with EXPLICIT API adaptations: skip missing slices; finalSlice=require or
last-valid; minimum trend length; New has priority over mixed history; a neutral
final bin with only hot/cold history is Sporadic (the current ESRI page instead
requires a significant final bin). Not a claim of full ArcGIS equivalence.
The classification decision table is self-tested on all 18 category codes.
Profile P observed/S: atol=1e-6*output scale, rtol=1e-4; integer/missing outputs exact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import warnings

import esda
import libpysal
import numpy as numpy
import scipy
from scipy.stats import kendalltau, norm
import statsmodels
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data/chicago-nature'


def encode(value):
    if isinstance(value, dict):
        return {key: encode(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [encode(entry) for entry in value]
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, (float, numpy.floating)):
        return float(value) if numpy.isfinite(value) else None
    return value


def classify(states, trend, probability, trendKnown=True, finalMissing=False):
    """Published category predicates, with the API adaptations listed above."""
    states = numpy.asarray(states)
    if len(states) == 0:
        return 0
    if finalMissing:
        return 17
    final = states[-1]
    for direction, base in [(1, 0), (-1, 8)]:
        selected = numpy.flatnonzero(states == direction)
        count = len(selected)
        opposite = numpy.count_nonzero(states == -direction)
        persistent = count >= float(numpy.float32(.9)) * len(states)
        consecutive = count >= 2 and numpy.array_equal(selected, numpy.arange(len(states)-count, len(states)))
        rules = [
            (final == direction and count == 1, 1),
            (final == direction and consecutive and not persistent, 2),
            (final == direction and persistent and not trendKnown, 17),
            (final == direction and persistent and probability <= .05 and direction*trend > 0, 3),
            (final == direction and persistent and probability <= .05 and direction*trend < 0, 5),
            (final == direction and persistent, 4),
            (final == direction and opposite > 0, 7),
            (final == 0 and persistent, 8),
            (count > 0 and opposite == 0, 6),
        ]
        for matches, code in rules:
            if matches:
                return code if code == 17 else base + code
    return 0


def validateClassification():
    cases = [([0]*10, 0, 1, 0), ([0]*9+[1], 0, 1, 1),
             ([0]*8+[1]*2, 0, 1, 2), ([1]*10, 3, .01, 3),
             ([1]*10, 0, 1, 4), ([1]*10, -3, .01, 5),
             ([1, 0]*4+[0, 1], 0, 1, 6), ([-1, 1]+[0]*8+[1], 0, 1, 7),
             ([1]*9+[0], 0, 1, 8)]
    for states, trend, probability, expected in cases:
        assert classify(states, trend, probability) == expected
        if expected:
            assert classify(-numpy.array(states), -trend, probability) == expected+8
    assert classify([1]*4, numpy.nan, numpy.nan, False) == 17
    assert classify([0, 1], 0, 1, finalMissing=True) == 17
    assert classify([], 0, 1) == 0
    assert classify([1, 0], 0, 1) == 6  # explicit neutral-final API extension
    assert classify([-1]*9+[1], 0, 1) == 1  # explicit precedence
    return 22


def makeCase(name, cube, width, height, window, mask, method, finalSlice, provenance):
    cube = numpy.asarray(cube, dtype=numpy.float32).reshape(width*height, -1)
    sliceCount = cube.shape[1]
    present = numpy.isfinite(cube) & numpy.asarray(mask, dtype=bool)[:, None]
    keep = numpy.flatnonzero(present.ravel())
    remap = {int(original): index for index, original in enumerate(keep)}
    neighbors = {}
    for index, original in enumerate(keep):
        cell, time = divmod(int(original), sliceCount)
        column, row = cell % width, cell // width
        cells = [otherRow*width+otherColumn for otherRow in range(height)
                 for otherColumn in range(width)
                 if (otherColumn-column)**2+(otherRow-row)**2 <= 1]
        neighbors[index] = [remap[other*sliceCount+otherTime] for other in cells
                            for otherTime in range(max(0, time-window), time+1)
                            if other*sliceCount+otherTime in remap]
    weights = libpysal.weights.W(neighbors, silence_warnings=True)
    population = cube.ravel()[keep].astype(numpy.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = esda.G_Local(population, weights, transform='B', star=True, permutations=0)
    scores = numpy.full(cube.size, numpy.nan)
    scores[keep] = model.Zs
    probabilities = 2*norm.sf(numpy.abs(scores))
    adjusted = probabilities.copy()
    finite = numpy.isfinite(scores)
    if method != 'none':
        adjusted[finite] = multipletests(probabilities[finite], method={'benjamini-hochberg': 'fdr_bh', 'benjamini-yekutieli': 'fdr_by'}[method])[1]
    scoreCube = scores.astype(numpy.float32).reshape(cube.shape)
    adjustedCube = adjusted.reshape(cube.shape)
    expected = {'giZScores': scores, 'pValues': probabilities, 'adjustedPValues': adjusted,
                'globalStatistics': [len(population), population.mean(), population.var(), population.std()],
                'trendS': [], 'trendZ': [], 'trendP': [], 'category': [], 'hotSliceCount': [],
                'coldSliceCount': [], 'lastValidSlice': []}
    for cell in range(len(cube)):
        valid = numpy.flatnonzero(numpy.isfinite(scoreCube[cell]))
        series = scoreCube[cell, valid].astype(numpy.float64)
        length = len(series)
        pairs = length*(length-1)//2
        counts = numpy.unique(series, return_counts=True)[1]
        tiedPairs = numpy.sum(counts*(counts-1)//2)
        statistic = 0
        if pairs > tiedPairs:
            tau = kendalltau(numpy.arange(length), series, method='asymptotic').statistic
            statistic = int(round(tau*numpy.sqrt(pairs*(pairs-tiedPairs))))
        variance = (length*(length-1)*(2*length+5)-numpy.sum(counts*(counts-1)*(2*counts+5)))/18
        trend = (statistic-numpy.sign(statistic))/numpy.sqrt(variance) if variance > 0 else 0.
        probability = 2*norm.sf(abs(trend))
        if length < 10:
            trend = probability = numpy.nan
        significant = adjustedCube[cell, valid] <= float(numpy.float32(.05))
        states = numpy.where(significant, numpy.sign(series), 0).astype(int)
        finalMissing = finalSlice == 'require' and length > 0 and valid[-1] != sliceCount-1
        expected['trendS'].append(statistic)
        expected['trendZ'].append(trend)
        expected['trendP'].append(probability)
        expected['category'].append(classify(states, trend, probability, length >= 10, finalMissing))
        expected['hotSliceCount'].append(numpy.count_nonzero(states == 1))
        expected['coldSliceCount'].append(numpy.count_nonzero(states == -1))
        expected['lastValidSlice'].append(int(valid[-1]) if length else 4294967295)
    return {'name': name, 'reference': 'esda.G_Local expanded cube; scipy.stats.kendalltau; statsmodels.multipletests; composite classification',
            'n': cube.size, 'width': width, 'height': height, 'sliceCount': sliceCount,
            'values': cube.ravel(), 'mask': mask, 'temporalWindow': window,
            'multipleTesting': method, 'finalSlice': finalSlice, 'expected': expected,
            'inputFloat32Sha256': hashlib.sha256(cube.astype('<f4').tobytes()).hexdigest(),
            'provenance': provenance}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='outputDirectory', type=Path, default=Path(__file__).parent/'fixtures')
    arguments = parser.parse_args()
    positions = numpy.fromfile(DATA/'position.bin', dtype='<f4').reshape(-1, 2)
    times = numpy.fromfile(DATA/'time.bin', dtype='<u4')
    edges = numpy.array([numpy.datetime64(f'2023-{month:02d}-01') for month in range(1, 13)] + [numpy.datetime64('2024-01-01')])
    seconds = (edges-numpy.datetime64('2023-01-01')).astype('timedelta64[s]').astype(numpy.int64)
    histogram, _ = numpy.histogramdd(numpy.column_stack([positions[:, 1], positions[:, 0], times]),
                                   bins=[numpy.linspace(41.64, 42.03, 9), numpy.linspace(-87.94, -87.52, 9), seconds])
    assert histogram.sum() == len(positions) == 43557
    provenance = {'source': str(DATA.relative_to(ROOT)), 'sourceCount': len(positions),
                  'selection': 'all rows; 8x8 whole-extent bins and all 12 calendar months',
                  'sourceSha256': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['manifest.json', 'position.bin', 'time.bin']},
                  'attribution': 'iNaturalist contributors; CC0/CC BY/CC BY-NC, non-commercial use'}
    cases = [makeCase('chicago-nature-full-cube', histogram, 8, 8, 1, [1]*64,
                      'benjamini-hochberg', 'require', provenance)]
    # Repeated values exercise exact tied trend S; missing final/interior bins and one masked cell.
    cube = numpy.array([[10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40],
                        [40, 40, 40, 30, 30, 30, 20, 20, 20, 10, 10, 10],
                        [0]*12, [100]*12, [5, 8, 9, 3, 7, 8, 1, 2, 4, 6, 8, 9],
                        [200]*12], dtype=float)
    cube[2, [3, 11]] = numpy.nan
    cube[3, 2] = numpy.nan
    for method in ['none', 'benjamini-yekutieli']:
        for finalSlice in ['require', 'last-valid']:
            cases.append(makeCase(f'ties-missing-mask/{method}/{finalSlice}', cube, 3, 2, 0,
                                  [1, 1, 1, 1, 1, 0], method, finalSlice, {'source': 'synthetic'}))
    # Independent designed classification coverage: sparse three-cell plateaus
    # in a long baseline lattice. Their centers have identical spatial support,
    # so repeated bin values produce exact ties without order-sensitive sums.
    designed = numpy.full((192, 12), 100., dtype=float)
    patterns = [numpy.array([100]*11+[200]), numpy.array([100]*10+[200]*2),
                numpy.linspace(170, 230, 12), numpy.full(12, 200.),
                numpy.linspace(230, 170, 12), numpy.array([200, 100]*5+[100, 200]),
                numpy.array([0, 200]+[100]*9+[200]), numpy.array([200]*11+[100])]
    for directionIndex, direction in enumerate([1, -1]):
        for patternIndex, pattern in enumerate(patterns):
            center = (directionIndex*8+patternIndex)*10+3
            designed[center-1:center+2] = 100+direction*(pattern-100)
    designedCase = makeCase('designed-category-plateaus', designed, 192, 1, 0, [1]*192,
                           'none', 'require', {'source': 'synthetic; isolated three-cell temporal patterns'})
    for directionIndex in range(2):
        for patternIndex in range(8):
            center = (directionIndex*8+patternIndex)*10+3
            assert designedCase['expected']['category'][center] == directionIndex*8+patternIndex+1, (center, designedCase['expected']['category'][center])
    cases.append(designedCase)
    fixture = {'generator': Path(__file__).name,
               'versions': {'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__,
                            'esda': esda.__version__, 'libpysal': libpysal.__version__, 'statsmodels': statsmodels.__version__},
               'classificationChecks': validateClassification(), 'cases': cases}
    output = arguments.outputDirectory/'parity-wp1-emerging-hot-spots.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(encode(fixture), separators=(',', ':'), allow_nan=False)+'\n'
    assert len(serialized.encode()) <= 1500000
    output.write_text(serialized)
    print(output, len(serialized), 'bytes')


if __name__ == '__main__':
    main()
