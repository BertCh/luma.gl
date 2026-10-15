// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUGraphCoreNumber, GPUGraphDegree} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {submitGraph} from '../../../experimental/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../experimental/test/utils/require-test-device';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {createParityResources} from '../../../experimental/test/gpu-network/parity-wp3-round2-resources';
import {createMetricsTopology, graphMetricsFixtures} from './parity-wp3-graph-metrics-fixture';

for (const {fixture, iterations} of graphMetricsFixtures.flatMap(fixture =>
  [32, fixture.coreIterations].map(iterations => ({fixture, iterations}))
)) {
  for (const weighted of [false, true]) {
    it(`NetworkX core/degree ${fixture.case}/iterations-${iterations}/weights-${weighted}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources = createParityResources(device);
      const topology = createMetricsTopology(resources, fixture, weighted);
      new GPUGraphCoreNumber({
        topology,
        iterations,
        output: resources.vector('cores', 'uint32', new Array(fixture.nodeCount).fill(0)),
        degeneracy: resources.vector('degeneracy', 'uint32', [0]),
        converged: resources.vector('converged', 'uint32', [0])
      }).addToGraph(resources.graph);
      for (const direction of ['incoming', 'outgoing'] as const) {
        new GPUGraphDegree({
          id: `degree-${direction}`,
          topology,
          direction,
          output: resources.vector(direction, 'uint32', new Array(fixture.nodeCount).fill(0))
        }).addToGraph(resources.graph);
      }
      const compiled = resources.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const comparison = {
          ...fixture,
          case: `${fixture.case}/iterations-${iterations}/weights-${weighted}`
        };
        for (const metric of ['cores', 'degeneracy', 'converged'] as const) {
          const expected =
            metric === 'cores' ? fixture.cores : [metric === 'degeneracy' ? fixture.degeneracy : 1];
          compareParity(
            'GPUGraphCoreNumber',
            comparison,
            metric,
            await resources.readUint(metric),
            expected,
            EXACT
          );
        }
        for (const direction of ['incoming', 'outgoing'] as const) {
          compareParity(
            'GPUGraphDegree',
            comparison,
            `every ${direction} degree (unweighted API)`,
            await resources.readUint(direction),
            fixture[direction],
            EXACT
          );
        }
        const status = [
          ...(await resources.readUint('invalid')),
          ...(await resources.readUint('forward-overflow')),
          ...(fixture.directed ? await resources.readUint('reverse-overflow') : [])
        ];
        compareParity(
          'GPUGraphCoreNumber',
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
