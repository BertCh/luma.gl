// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUNetworkAnalyticsColumns} from '../../../src/gpu-network/network-analysis/gpu-network-analytics-columns';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';
import {analysisFixtures} from './parity-wp3-network-analysis-fixture';

for (const fixture of analysisFixtures) {
  it(`external analytics columns ${fixture.input.case}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const resources = createParityResources(device);
    const {input, reverse} = fixture;
    const topology = {
      offsets: resources.input('offsets', 'uint32', input.offsets),
      neighbors: resources.input('neighbors', 'uint32', input.neighbors),
      // These nonunit/zero weights must have no effect on the unweighted API.
      weights: resources.input(
        'weights',
        'float32',
        input.neighbors.map((_, index) => index % 7)
      )
    };
    function column<Format extends 'uint32' | 'float32'>(name: string, format: Format) {
      return {
        output: resources.output(name, format, input.nodeCount),
        normalized: resources.output(`${name}-normalized`, 'float32', input.nodeCount),
        extent: resources.output(`${name}-extent`, format, 2)
      };
    }
    resources.graph.add(
      new GPUNetworkAnalyticsColumns({
        topology,
        reverseTopology: input.directed
          ? {
              offsets: resources.input('reverse-offsets', 'uint32', reverse.offsets),
              neighbors: resources.input('reverse-neighbors', 'uint32', reverse.neighbors)
            }
          : undefined,
        degree: column('degree', 'uint32'),
        inDegree: input.directed ? column('inDegree', 'uint32') : undefined,
        coreNumber: {
          ...column('coreNumber', 'uint32'),
          maximumIterations: 128,
          degeneracy: resources.output('degeneracy', 'uint32', 1),
          status: {converged: resources.output('core-converged', 'uint32', 1)}
        },
        pageRank: {
          ...column('pageRank', 'float32'),
          damping: Math.fround(0.85),
          iterationCount: 256,
          residual: resources.output('rank-residual', 'float32', 1)
        },
        components: {
          output: resources.output('components', 'uint32', input.nodeCount),
          maximumIterations: 512,
          status: {converged: resources.output('components-converged', 'uint32', 1)}
        }
      })
    );
    const compiled = resources.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      for (const metric of fixture.metrics) {
        const name = metric.metric;
        if (name === 'inDegree' && !input.directed) continue;
        const floating = name === 'pageRank';
        const tolerance = floating ? {absolute: 1e-6 * metric.extent[1], relative: 1e-4} : EXACT;
        const actual = floating ? await resources.readFloat(name) : await resources.readUint(name);
        compareParity(
          'GPUNetworkAnalyticsColumns',
          metric,
          `${name}: every original vertex row`,
          actual,
          metric.values,
          tolerance
        );
        if (name === 'components') continue;
        const extent = floating
          ? await resources.readFloat(`${name}-extent`)
          : await resources.readUint(`${name}-extent`);
        compareParity(
          'GPUNetworkAnalyticsColumns',
          metric,
          `${name}: min/max`,
          extent,
          metric.extent,
          tolerance
        );
        // S profile at normalized output scale one. Constant columns are exactly zero.
        compareParity(
          'GPUNetworkAnalyticsColumns',
          metric,
          `${name}: every normalized row`,
          await resources.readFloat(`${name}-normalized`),
          metric.normalized,
          metric.extent[0] === metric.extent[1] ? EXACT : {absolute: 1e-6, relative: 1e-4}
        );
      }
      const core = fixture.metrics.find(metric => metric.metric === 'coreNumber')!;
      compareParity(
        'GPUNetworkAnalyticsColumns',
        input,
        'core convergence, component convergence, degeneracy',
        [
          ...(await resources.readUint('core-converged')),
          ...(await resources.readUint('components-converged')),
          ...(await resources.readUint('degeneracy'))
        ],
        [1, 1, core.extent[1]],
        EXACT
      );
      const rank = fixture.metrics.find(metric => metric.metric === 'pageRank')!;
      const actualRanks = await resources.readFloat('pageRank');
      // Summing per-element S budgets bounds mass error. Contraction gives truncation
      // <=2*alpha^256 <1.8e-18; successive iterate difference <=2*sum(budgets).
      const totalBudget = rank.values.reduce(
        (sum, value) => sum + 1e-6 * rank.extent[1] + 1e-4 * value,
        0
      );
      compareParity(
        'GPUNetworkAnalyticsColumns',
        rank,
        'rank mass',
        [actualRanks.reduce((sum, value) => sum + value, 0)],
        [1],
        {absolute: totalBudget, relative: 0}
      );
      compareParity(
        'GPUNetworkAnalyticsColumns',
        rank,
        'published last iteration change',
        await resources.readFloat('rank-residual'),
        [0],
        {absolute: 2 * totalBudget, relative: 0}
      );
    } finally {
      compiled.destroy();
      resources.destroy();
    }
  }, 180000);
}
