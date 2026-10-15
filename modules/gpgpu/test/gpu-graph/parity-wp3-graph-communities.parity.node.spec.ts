// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {
  canonicalizePartition,
  communityFixtures,
  evaluatePartition,
  getAdjustedRandIndex
} from './parity-wp3-graph-communities-fixture';

for (const fixture of communityFixtures) {
  it(`WP3 graph fixture integrity only, no GPU: ${fixture.case}`, () => {
    expect(fixture.versions['networkx']).toBeTruthy();
    for (const scoring of fixture.scoring) {
      const actual = evaluatePartition(fixture, scoring.labels, scoring.resolution);
      const comparison = {
        ...fixture,
        case: `${fixture.case}/${scoring.name}/${scoring.resolution}`
      };
      compareParity(
        'GPUGraphModularity/CPU-fixture-integrity-only',
        comparison,
        'independent evaluator against every NetworkX contribution',
        actual,
        scoring.contributions,
        {absolute: 1e-12, relative: 1e-12}
      );
      compareParity(
        'GPUGraphModularity/CPU-fixture-integrity-only',
        comparison,
        'independent evaluator against NetworkX modularity function',
        [actual.reduce((sum, value) => sum + value, 0)],
        [scoring.score],
        {absolute: 1e-12, relative: 1e-12}
      );
    }
    for (let first = 0; first < fixture.runs.length; first++) {
      const run = fixture.runs[first];
      compareParity(
        'GPUGraphLabelPropagation/CPU-fixture-integrity-only',
        fixture,
        `reference score seed ${run.seed}`,
        [evaluatePartition(fixture, run.labels, 1, false).reduce((sum, value) => sum + value, 0)],
        [run.score],
        {absolute: 1e-12, relative: 1e-12}
      );
      compareParity(
        'GPUGraphLabelPropagation/CPU-fixture-integrity-only',
        fixture,
        `canonical partition seed ${run.seed}`,
        canonicalizePartition(run.labels),
        run.labels,
        EXACT
      );
      compareParity(
        'GPUGraphLabelPropagation/CPU-fixture-integrity-only',
        fixture,
        `permutation preserves all vertex IDs seed ${run.seed}`,
        [...run.permutation].sort((left, right) => left - right),
        Array.from({length: fixture.nodeCount}, (_, index) => index),
        EXACT
      );
      compareParity(
        'GPUGraphLabelPropagation/CPU-fixture-integrity-only',
        {...fixture, reference: 'sklearn.metrics.adjusted_rand_score'},
        `all reference stability pairs seed ${run.seed}`,
        fixture.runs.map(second => getAdjustedRandIndex(run.labels, second.labels)),
        fixture.stability[first],
        {absolute: 1e-12, relative: 1e-12}
      );
    }
  });
}
