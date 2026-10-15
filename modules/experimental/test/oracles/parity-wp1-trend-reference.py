# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Conditional SciPy Kendall references for the actual f32 Gi* stage output.

Mann-Kendall is discontinuous at ties. Rounding f64 reference Gi* once is not
identical to computing Gi* in f32; it cannot prescribe exact ranks at tied bins.
Keep the existing Gi* S bound e_i=1e-6*max(1,max(abs(z)))+1e-4*abs(z_i).
Only intervals [z_i-e_i,z_i+e_i] that overlap can exchange ranks. Enumerate every
weak ordering within each connected interval group using scipy.stats.kendalltau.
Cross-group pair signs cannot change, by the triangle inequality. The browser
selects the table entry using ACTUAL Gi* ranks, never the observed trend output;
S still has zero tolerance. Kendall's additive pair statistic and tied-group
variance compose the groups. NumPy evaluates the published continuity-corrected
variance, scipy.stats.norm.sf supplies p. No GPU arithmetic is replicated.

Reference: Kendall (1975), Rank Correlation Methods, 4th ed.; the same variance
and continuity convention as parity-wp1-emerging-hot-spots.py. The transport
stores exact table lookups, not a fitted error budget or an allowed S interval.
"""
import base64
import hashlib
import itertools
import json
import sys
from pathlib import Path
import zlib
import numpy
import scipy
from scipy.stats import kendalltau, norm

DIRECTORY = Path(__file__).parent


def getStatistic(series):
    length = len(series)
    counts = numpy.unique(series, return_counts=True)[1]
    pairs = length * (length - 1) // 2
    tiedPairs = int(numpy.sum(counts * (counts - 1) // 2))
    statistic = 0 if pairs == tiedPairs else int(round(
        kendalltau(numpy.arange(length), series).statistic * numpy.sqrt(pairs * (pairs - tiedPairs))))
    tieTerm = int(numpy.sum(counts * (counts - 1) * (2 * counts + 5)))
    return statistic, tieTerm


def makeCell(scores, absolute):
    length = len(scores)
    errors = absolute + 1e-4 * abs(scores)
    lower, upper = scores - errors, scores + errors
    groups = []
    maximum = -numpy.inf
    for index in numpy.argsort(lower):
        if lower[index] > maximum:
            groups.append([])
        groups[-1].append(int(index))
        maximum = max(maximum, upper[index])
    groups = [sorted(group) for group in groups if len(group) > 1]
    baseStatistic = getStatistic(scores)[0]
    records = []
    outcomes = {(0, 0)}
    for indices in groups:
        size = len(indices)
        assert size <= 5, 'Bounded enumeration; do not silently omit larger tie groups'
        baseStatistic -= getStatistic(scores[indices])[0]
        variants = {}
        for ranks in itertools.product(range(size), repeat=size):
            if set(ranks) != set(range(max(ranks) + 1)):
                continue
            variants[','.join(map(str, ranks))] = getStatistic(ranks)
        changes = set(variants.values())
        outcomes = {(statistic + change, tieTerm + correction)
                    for statistic, tieTerm in outcomes for change, correction in changes}
        records.append(dict(indices=indices, variants=variants))
    lookup = {}
    for change, tieTerm in sorted(outcomes):
        statistic = baseStatistic + change
        variance = (length * (length - 1) * (2 * length + 5) - tieTerm) / 18
        score = (statistic - numpy.sign(statistic)) / numpy.sqrt(variance) if variance > 0 else 0.
        lookup[f'{statistic},{tieTerm}'] = [float(score), float(2 * norm.sf(abs(score)))]
    # Certify table assembly against SciPy on the unchanged reference series.
    statistic, tieTerm = getStatistic(scores)
    assert f'{statistic},{tieTerm}' in lookup
    return dict(baseStatistic=baseStatistic, groups=records, lookup=lookup)


def main():
    cases, templates, sources, probes = {}, [], {}, []
    random = numpy.random.default_rng(10101)
    identities = {}
    for kind in ['lengths', 'weighted']:
        path = DIRECTORY / f'fixtures/parity-wp1-emerging-{kind}.json'
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        source = json.loads(path.read_text())
        if 'data' in source:
            source = json.loads(zlib.decompress(base64.b64decode(source['data'])))
        for case in source['cases']:
            scores = numpy.array(case['expected']['giZScores'])
            identity = hashlib.sha256(scores.tobytes()).hexdigest()
            if identity not in identities:
                identities[identity] = len(templates)
                absolute = 1e-6 * max(1, max(abs(scores)))
                templates.append([makeCell(series, absolute)
                                  for series in scores.reshape(-1, case['sliceCount'])])
                for sample in range(4):
                    perturbed = scores if sample == 0 else scores + random.uniform(-.25, .25, len(scores)) * (absolute + 1e-4 * abs(scores))
                    uploaded = perturbed.astype(numpy.float32).astype(float)
                    expected = dict(trendS=[], trendZ=[], trendP=[])
                    for series in uploaded.reshape(-1, case['sliceCount']):
                        statistic, tieTerm = getStatistic(series)
                        length = len(series)
                        variance = (length * (length - 1) * (2 * length + 5) - tieTerm) / 18
                        score = (statistic - numpy.sign(statistic)) / numpy.sqrt(variance) if variance > 0 else 0.
                        expected['trendS'].append(statistic)
                        expected['trendZ'].append(float(score))
                        expected['trendP'].append(float(2 * norm.sf(abs(score))))
                    probes.append(dict(name=case['name'], sliceCount=case['sliceCount'], scores=uploaded.tolist(), expected=expected))
            cases[case['name']] = identities[identity]
    payload = json.dumps(dict(cases=cases, templates=templates, probes=probes), separators=(',', ':')).encode()
    output = dict(generator=Path(__file__).name,
                  command=f'{sys.executable} -I modules/experimental/test/oracles/parity-wp1-trend-reference.py',
                  versions={'numpy': numpy.__version__, 'scipy': scipy.__version__},
                  sourceSha256=sources, encoding='zlib+base64',
                  sha256=hashlib.sha256(payload).hexdigest(),
                  data=base64.b64encode(zlib.compress(payload, 9)).decode())
    path = DIRECTORY / 'fixtures/parity-wp1-trend-reference.json'
    path.write_text(json.dumps(output, separators=(',', ':')) + '\n')
    assert path.stat().st_size < 1500000
    print(path, path.stat().st_size, 'bytes;', len(templates), 'distinct cubes')


if __name__ == '__main__':
    main()
