// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUNetworkNeighborhood} from '../../../src/gpu-network/network-analysis/gpu-network-neighborhood';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';
import {analysisFixtures} from './parity-wp3-network-analysis-fixture';

for (const fixture of analysisFixtures) {
  for (const mode of ['forward', 'reverse', 'bounded'] as const) {
    it(`external neighborhood ${fixture.input.case}/${mode}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const resources = createParityResources(device);
      const {input} = fixture;
      const scenarios = fixture.neighborhoods.filter(scenario =>
        mode === 'reverse'
          ? scenario.incoming
          : !scenario.incoming &&
            (mode === 'bounded'
              ? scenario.case.endsWith('/bounded')
              : !scenario.case.endsWith('/bounded'))
      );
      const first = scenarios[0];
      const topology = mode === 'reverse' ? fixture.reverse : input;
      const starts = new GPUParameterBuffer(device, {
        id: 'starts',
        format: 'uint32',
        length: 3,
        values: Uint32Array.from(first.starts)
      });
      const startCount = new GPUParameterBuffer(device, {
        id: 'start-count',
        format: 'uint32',
        length: 1,
        values: Uint32Array.of(first.startCount)
      });
      const hops = new GPUParameterBuffer(device, {
        id: 'hops',
        format: 'uint32',
        length: 1,
        values: Uint32Array.of(first.hops)
      });
      const compact = (name: string, capacity: number) => ({
        ids: resources.output(`${name}Ids`, 'uint32', capacity),
        status: {
          count: resources.output(`${name}Count`, 'uint32', 1),
          overflow: resources.output(`${name}Overflow`, 'uint32', 1)
        }
      });
      resources.graph.add(
        new GPUNetworkNeighborhood({
          topology: {
            offsets: resources.input('offsets', 'uint32', topology.offsets),
            neighbors: resources.input('neighbors', 'uint32', topology.neighbors),
            weights: resources.input(
              'weights',
              'float32',
              topology.neighbors.map((_, index) => index % 7)
            )
          },
          startNodes: starts.importToGraph(resources.graph),
          startCount: startCount.importToGraph(resources.graph),
          hops: hops.importToGraph(resources.graph),
          maximumHops: first.maximumHops,
          nodeIds: resources.input('original-node-ids', 'uint32', input.nodeIds),
          edgeIds: resources.input('original-edge-ids', 'uint32', topology.edgeIds),
          output: {
            hopDistances: resources.output('hopDistances', 'uint32', input.nodeCount),
            nodeMask: resources.output('nodeMask', 'uint32', input.nodeCount),
            edgeMask: resources.output('edgeMask', 'uint32', input.neighbors.length),
            nodes: compact('node', first.nodeCapacity),
            edges: compact('edge', first.edgeCapacity)
          }
        })
      );
      const compiled = resources.graph.compile();
      try {
        // Return to the first state after empty/clamped states: stale words must not survive.
        for (const scenario of [...scenarios, first]) {
          starts.write(Uint32Array.from(scenario.starts));
          startCount.write(Uint32Array.of(scenario.startCount));
          hops.write(Uint32Array.of(scenario.hops));
          submitGraph(device, compiled, undefined);
          for (const metric of ['hopDistances', 'nodeMask', 'edgeMask'] as const) {
            compareParity(
              'GPUNetworkNeighborhood',
              scenario,
              metric,
              await resources.readUint(metric),
              scenario[metric],
              EXACT
            );
          }
          for (const kind of ['node', 'edge'] as const) {
            for (const suffix of ['Count', 'Overflow'] as const) {
              const name = `${kind}${suffix}` as const;
              compareParity(
                'GPUNetworkNeighborhood',
                scenario,
                name,
                await resources.readUint(name),
                [scenario[name]],
                EXACT
              );
            }
            // Compare the entire published prefix; storage past the count is unspecified.
            const name = `${kind}Ids` as const;
            const count = scenario[`${kind}Count`];
            compareParity(
              'GPUNetworkNeighborhood',
              scenario,
              `${name}: stable CSR-order compact prefix`,
              (await resources.readUint(name)).slice(0, count),
              scenario[name],
              EXACT
            );
          }
        }
      } finally {
        compiled.destroy();
        starts.destroy();
        startCount.destroy();
        hops.destroy();
        resources.destroy();
      }
    }, 180000);
  }
}
