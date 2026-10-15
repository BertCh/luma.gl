// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUNetworkReachability} from '../../../src/gpu-network/network-reachability';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';
import {routingFixtures} from '../network-analysis/parity-wp3-routing-fixture';

for (const {input, cases} of routingFixtures) {
  for (const fixture of cases) {
    it(`GPUNetworkReachability NetworkX full arrays ${fixture.case}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources = createParityResources(device);
      resources.graph.add(
        new GPUNetworkReachability({
          topology: {
            offsets: resources.input('offsets', 'uint32', input.offsets),
            neighbors: resources.input('neighbors', 'uint32', input.neighbors),
            weights: resources.input('weights', 'float32', input.weights)
          },
          startNodes: resources.input('starts', 'uint32', fixture.startNodes),
          startCosts: resources.input('initial', 'float32', fixture.startCosts),
          costLimit: resources.input('limit', 'float32', [fixture.limit]),
          costBreaks: resources.input('breaks', 'float32', fixture.breaks),
          maximumIterations: 128,
          maximumTieIterations: 16,
          localIterations: 16,
          output: {
            costs: resources.output('costs', 'float32', input.nodeCount),
            predecessors: resources.output('predecessors', 'uint32', input.nodeCount),
            bands: resources.output('bands', 'uint32', input.nodeCount),
            bandCounts: resources.output('band-counts', 'uint32', fixture.breaks.length),
            unresolvedCount: resources.output('unresolved', 'uint32', 1),
            status: {converged: resources.output('converged', 'uint32', 1)}
          }
        })
      );
      const compiled = resources.graph.compile();
      try {
        for (let encoding = 0; encoding < 2; encoding++) {
          submitGraph(device, compiled, undefined);
          const comparison = {...fixture, case: `${fixture.case}/encoding=${encoding}`};
          // All weights are multiples of 1/8 and all finite path costs <2^21.
          // Their positive sums are exactly representable in f32: E is stricter than L.
          compareParity(
            'GPUNetworkReachability',
            comparison,
            'all costs and unreachable masks',
            await resources.readFloat('costs'),
            fixture.costs,
            EXACT
          );
          compareParity(
            'GPUNetworkReachability',
            comparison,
            'every tie-aligned predecessor',
            await resources.readUint('predecessors'),
            fixture.predecessors,
            EXACT
          );
          compareParity(
            'GPUNetworkReachability',
            comparison,
            'every inclusive cost band',
            await resources.readUint('bands'),
            fixture.bands,
            EXACT
          );
          compareParity(
            'GPUNetworkReachability',
            comparison,
            'every band count',
            await resources.readUint('band-counts'),
            fixture.bandCounts,
            EXACT
          );
          compareParity(
            'GPUNetworkReachability',
            comparison,
            'converged and no unresolved chains',
            [
              ...(await resources.readUint('converged')),
              ...(await resources.readUint('unresolved'))
            ],
            [1, 0],
            EXACT
          );
        }
      } finally {
        compiled.destroy();
        resources.destroy();
      }
    }, 180000);
  }
}
