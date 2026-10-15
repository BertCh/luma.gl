// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUGraphDegree, GPUGraphTopologyView} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {submitGraph} from '../../../experimental/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../experimental/test/utils/require-test-device';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {createParityResources} from '../../../experimental/test/gpu-network/parity-wp3-round2-resources';
import {createMetricsTopology, graphMetricsFixtures} from './parity-wp3-graph-metrics-fixture';
import {showcaseGraphs} from './parity-wp3-graph-showcase-fixture';

for (const fixture of graphMetricsFixtures.filter(fixture => fixture.edges.length)) {
  it(`NetworkX degrees with bounded adjacency: ${fixture.case}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const resources = createParityResources(device);
    const topology = createMetricsTopology(resources, fixture, true, 1);
    for (const direction of ['incoming', 'outgoing'] as const)
      new GPUGraphDegree({
        id: direction,
        topology,
        direction,
        output: resources.vector(direction, 'uint32', new Array(fixture.nodeCount).fill(0))
      }).addToGraph(resources.graph);
    const compiled = resources.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      for (const direction of ['incoming', 'outgoing'] as const)
        compareParity(
          'GPUGraphDegree',
          fixture,
          `overflow: every ${direction} degree`,
          await resources.readUint(direction),
          fixture[direction],
          EXACT
        );
      compareParity(
        'GPUGraphDegree',
        {...fixture, reference: 'bounded adjacency capacity contract'},
        'overflow exposed',
        await resources.readUint('forward-overflow'),
        [1],
        EXACT
      );
    } finally {
      compiled.destroy();
      resources.destroy();
    }
  });
}
for (const fixture of showcaseGraphs) {
  it(`NetworkX degrees through graph views: ${fixture.case}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const resources = createParityResources(device);
    const topology = new GPUGraphTopologyView({
      vertexCount: fixture.nodeCount,
      directed: fixture.directed,
      forward: {
        offsets: resources.input('offsets', 'uint32', fixture.offsets),
        neighbors: resources.input('neighbors', 'uint32', fixture.neighbors)
      },
      reverse: fixture.directed
        ? {
            offsets: resources.input('reverse-offsets', 'uint32', fixture.reverseOffsets),
            neighbors: resources.input('reverse-neighbors', 'uint32', fixture.reverseNeighbors)
          }
        : undefined
    });
    for (const direction of ['incoming', 'outgoing'] as const)
      new GPUGraphDegree({
        id: direction,
        topology,
        direction,
        output: resources.output(direction, 'uint32', fixture.nodeCount)
      }).addToGraph(resources.graph);
    const compiled = resources.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      for (const direction of ['incoming', 'outgoing'] as const)
        compareParity(
          'GPUGraphDegree',
          {
            ...fixture,
            reference:
              'networkx.MultiDiGraph.in_degree/out_degree; MultiGraph.degree with single loop incidence'
          },
          `view: every ${direction} degree`,
          await resources.readUint(direction),
          fixture[direction],
          EXACT
        );
    } finally {
      compiled.destroy();
      resources.destroy();
    }
  });
}
