// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUGraphPageRank} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {submitGraph} from '../../../experimental/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../experimental/test/utils/require-test-device';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {createParityResources} from '../../../experimental/test/gpu-network/parity-wp3-round2-resources';
import {
  createMetricsTopology,
  evaluateRankResidual,
  graphMetricsFixtures
} from './parity-wp3-graph-metrics-fixture';

for (const fixture of graphMetricsFixtures) {
  for (const run of fixture.rankCases) {
    it(`NetworkX PageRank ${fixture.case}/alpha-${run.damping}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources = createParityResources(device);
      const topology = createMetricsTopology(resources, fixture);
      new GPUGraphPageRank({
        topology,
        damping: run.damping,
        iterations: run.iterations,
        output: resources.vector('ranks', 'float32', new Array(fixture.nodeCount).fill(0)),
        residual: resources.vector('residual', 'float32', [0])
      }).addToGraph(resources.graph);
      const compiled = resources.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const comparison = {...fixture, case: `${fixture.case}/alpha-${run.damping}`};
        const ranks = await resources.readFloat('ranks');
        const tolerance = {absolute: 1e-6 * Math.max(...run.ranks), relative: 1e-4};
        compareParity(
          'GPUGraphPageRank',
          comparison,
          'every stationary rank',
          ranks,
          run.ranks,
          tolerance
        );
        compareParity(
          'GPUGraphPageRank',
          comparison,
          'nonnegative finite ranks',
          ranks.map(value => Number(!Number.isFinite(value) || value < 0)),
          ranks.map(() => 0),
          EXACT
        );
        compareParity(
          'GPUGraphPageRank',
          comparison,
          'probability sum',
          [ranks.reduce((sum, value) => sum + value, 0)],
          [1],
          {absolute: 1e-6, relative: 1e-4}
        );
        // ||T(x)-x||_1 <= (1+alpha)||x-pi||_1 for the stochastic contraction T.
        // Sum the unchanged elementwise S budgets; this is a derived residual bound,
        // not an independently loosened tolerance. 2*alpha^256 < 1.8e-18 truncation.
        const rankErrorBudget = run.ranks.reduce(
          (sum, rank) => sum + tolerance.absolute + tolerance.relative * rank,
          0
        );
        const residualBound = (1 + run.damping) * rankErrorBudget;
        const actualResidual = evaluateRankResidual(fixture, ranks, run.damping);
        compareParity(
          'GPUGraphPageRank',
          comparison,
          'independent L1 fixed-point residual',
          [actualResidual.reduce((sum, value) => sum + value, 0)],
          [0],
          {absolute: residualBound, relative: 0}
        );
        // Published residual is ||x_k-x_(k-1)||_1, not ||T(x_k)-x_k||_1.
        // At a stationary run both iterates have the same S budget: triangle inequality.
        compareParity(
          'GPUGraphPageRank',
          comparison,
          'published final-iteration L1 change',
          await resources.readFloat('residual'),
          [0],
          {absolute: 2 * rankErrorBudget, relative: 0}
        );
        const status = [
          ...(await resources.readUint('invalid')),
          ...(await resources.readUint('forward-overflow')),
          ...(fixture.directed ? await resources.readUint('reverse-overflow') : [])
        ];
        compareParity(
          'GPUGraphPageRank',
          comparison,
          'complete topology: invalid/overflow',
          status,
          status.map(() => 0),
          EXACT
        );
      } finally {
        compiled.destroy();
        resources.destroy();
      }
    }, 180000);
  }
}
