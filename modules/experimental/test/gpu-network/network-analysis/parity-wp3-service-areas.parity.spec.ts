// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUNetworkServiceAreas} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';
import {getPredecessorViolations, serviceFixtures} from './parity-wp3-service-areas-fixture';

for (const {input, cases} of serviceFixtures) {
  for (const {fixture, showcase} of cases.flatMap(fixture =>
    [false, true].map(showcase => ({fixture, showcase}))
  )) {
    it(`GPUNetworkServiceAreas NetworkX ${fixture.case}/showcase-${showcase}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources = createParityResources(device);
      const {graph} = resources;
      graph.add(
        new GPUNetworkServiceAreas({
          topology: {
            offsets: resources.input('offsets', 'uint32', input.offsets),
            neighbors: resources.input('neighbors', 'uint32', input.neighbors),
            weights: resources.input('weights', 'float32', input.weights)
          },
          facilities: resources.input('facilities', 'uint32', input.facilities),
          facilityCosts: resources.input('initial-costs', 'float32', input.initialCosts),
          costLimit: resources.input('limit', 'float32', [fixture.budget]),
          maximumIterations: showcase ? 64 : 128,
          maximumLabelIterations: showcase ? undefined : 128,
          maximumTieIterations: showcase ? undefined : 32,
          output: {
            assignments: resources.output('assignments', 'uint32', input.nodeCount),
            costs: resources.output('costs', 'float32', input.nodeCount),
            predecessors: resources.output('predecessors', 'uint32', input.nodeCount),
            facilityNodeCounts: resources.output('counts', 'uint32', input.facilities.length),
            facilityCostSums: resources.output('sums', 'float32', input.facilities.length),
            status: {converged: resources.output('converged', 'uint32', 1)}
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const comparison = {...fixture, case: `${fixture.case}/showcase-${showcase}`};
        const className = 'GPUNetworkServiceAreas';
        compareParity(
          className,
          comparison,
          'convergence',
          await resources.readUint('converged'),
          [1],
          EXACT
        );
        compareParity(
          className,
          comparison,
          'every node cost and infinite mask',
          await resources.readFloat('costs'),
          fixture.costs,
          EXACT
        );
        compareParity(
          className,
          comparison,
          'every winning facility row',
          await resources.readUint('assignments'),
          fixture.assignments,
          EXACT
        );
        compareParity(
          className,
          comparison,
          'facility node counts',
          await resources.readUint('counts'),
          fixture.counts,
          EXACT
        );
        compareParity(
          className,
          comparison,
          'facility cost sums',
          await resources.readFloat('sums'),
          fixture.sums,
          fixture.tolerance
        );
        compareParity(
          className,
          comparison,
          'every predecessor allowed and reaches assigned facility',
          getPredecessorViolations(input, fixture, await resources.readUint('predecessors')),
          new Array(input.nodeCount).fill(0),
          EXACT
        );
      } finally {
        compiled.destroy();
        resources.destroy();
      }
    }, 120000);
  }
}
