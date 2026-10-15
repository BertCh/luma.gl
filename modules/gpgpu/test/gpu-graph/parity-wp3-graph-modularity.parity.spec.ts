// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUGraph, GPUGraphModularity} from '@luma.gl/gpgpu/gpu-graph';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {submitGraph} from '../../../experimental/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../experimental/test/utils/require-test-device';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {createParityResources} from '../../../experimental/test/gpu-network/parity-wp3-round2-resources';
import {communityFixtures} from './parity-wp3-graph-communities-fixture';

for (const fixture of communityFixtures) {
  for (const scoring of fixture.scoring) {
    it(`GPUGraphModularity NetworkX ${fixture.case}/${scoring.name}/resolution-${scoring.resolution}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources = createParityResources(device);
      const graph = new GPUGraph({
        vertexCount: fixture.nodeCount,
        directed: fixture.directed,
        sourceVertices: resources.vector(
          'sources',
          'uint32',
          fixture.edges.map(edge => edge[0])
        ),
        targetVertices: resources.vector(
          'targets',
          'uint32',
          fixture.edges.map(edge => edge[1])
        ),
        edgeWeights: resources.vector(
          'weights',
          'float32',
          fixture.edges.map(edge => edge[2])
        )
      });
      new GPUGraphModularity({
        graph,
        resolution: scoring.resolution,
        communities: resources.vector('labels', 'uint32', scoring.labels),
        output: resources.vector('score', 'float32', [0]),
        communityContributions: resources.vector(
          'contributions',
          'float32',
          new Array(fixture.nodeCount).fill(0)
        ),
        valid: resources.vector('valid', 'uint32', [0])
      }).addToGraph(resources.graph);
      const compiled = resources.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const comparison = {
          ...fixture,
          case: `${fixture.case}/${scoring.name}/resolution-${scoring.resolution}`
        };
        compareParity(
          'GPUGraphModularity',
          comparison,
          'valid partition',
          await resources.readUint('valid'),
          [1],
          EXACT
        );
        compareParity(
          'GPUGraphModularity',
          comparison,
          'full partition modularity',
          await resources.readFloat('score'),
          [scoring.score],
          fixture.tolerance
        );
        compareParity(
          'GPUGraphModularity',
          {
            ...comparison,
            reference:
              'composite Newman definition using NetworkX subgraph.size and degrees; sum verified against networkx.community.modularity'
          },
          'every stable-label contribution',
          await resources.readFloat('contributions'),
          scoring.contributions,
          fixture.tolerance
        );
      } finally {
        compiled.destroy();
        resources.destroy();
      }
    }, 60000);
  }
}
