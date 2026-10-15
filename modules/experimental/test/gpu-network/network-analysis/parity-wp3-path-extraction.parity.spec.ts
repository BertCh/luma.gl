// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUNetworkPathExtraction} from '../../../src/gpu-network/network-analysis/gpu-network-path-extraction';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';
import {mapEdgeId, mapNodeId, routingFixtures} from './parity-wp3-routing-fixture';

for (const {input, cases} of routingFixtures) {
  for (const fixture of cases) {
    for (const mode of ['complete', 'bounded', 'short', 'showcase-nodes-only'] as const) {
      it(`GPUNetworkPathExtraction NetworkX ${fixture.case}/${mode}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const resources = createParityResources(device);
        const targetCount = new GPUParameterBuffer(device, {
          id: 'active-targets',
          format: 'uint32',
          length: 1,
          values: new Uint32Array([fixture.targets.length])
        });
        const complete = new GPUParameterBuffer(device, {
          id: 'complete',
          format: 'uint32',
          length: 1,
          values: new Uint32Array([1])
        });
        const maximumPathLength =
          mode === 'short' ? 2 : mode === 'showcase-nodes-only' ? 2048 : fixture.maximumPathLength;
        const nodeCapacity = mode === 'bounded' ? 3 : Math.max(1, fixture.paths.flat().length);
        const edgeCapacity = mode === 'bounded' ? 2 : Math.max(1, fixture.edgePaths.flat().length);
        const compact = (prefix: string, capacity: number) => ({
          ids: resources.output(`${prefix}-ids`, 'uint32', capacity),
          lineOffsets: resources.output(`${prefix}-offsets`, 'uint32', fixture.targets.length + 1),
          status: {
            count: resources.output(`${prefix}-count`, 'uint32', 1),
            requiredCount: resources.output(`${prefix}-required`, 'uint32', 1),
            overflow: resources.output(`${prefix}-overflow`, 'uint32', 1)
          }
        });
        resources.graph.add(
          new GPUNetworkPathExtraction({
            predecessors: resources.input('predecessors', 'uint32', fixture.predecessors),
            costs: resources.input('costs', 'float32', fixture.costs),
            targets: resources.input('targets', 'uint32', fixture.targets),
            targetCount: targetCount.importToGraph(resources.graph),
            predecessorsComplete: complete.importToGraph(resources.graph),
            maximumPathLength,
            nodeIds:
              mode === 'showcase-nodes-only'
                ? undefined
                : resources.input(
                    'node-mapping',
                    'uint32',
                    Array.from({length: input.nodeCount}, (_, row) => mapNodeId(input, row))
                  ),
            output: {
              ...compact('nodes', nodeCapacity),
              pathFound: resources.output('found', 'uint32', fixture.targets.length),
              pathCosts: resources.output('path-costs', 'float32', fixture.targets.length)
            },
            edges:
              mode === 'showcase-nodes-only'
                ? undefined
                : {
                    topology: {
                      offsets: resources.input('offsets', 'uint32', input.offsets),
                      neighbors: resources.input('neighbors', 'uint32', input.neighbors),
                      weights: resources.input('weights', 'float32', input.weights)
                    },
                    edgeIds: resources.input(
                      'edge-mapping',
                      'uint32',
                      input.neighbors.map((_, row) => mapEdgeId(input, row))
                    ),
                    output: compact('edges', edgeCapacity)
                  }
          })
        );
        const compiled = resources.graph.compile();
        try {
          const states = [
            {count: fixture.targets.length, complete: 1},
            {count: 0, complete: 1},
            {count: Math.floor(fixture.targets.length / 2), complete: 1},
            {count: fixture.targets.length, complete: 0},
            {count: fixture.targets.length, complete: 1}
          ];
          for (const [encoding, state] of states.entries()) {
            targetCount.write(new Uint32Array([state.count]));
            complete.write(new Uint32Array([state.complete]));
            submitGraph(device, compiled, undefined);
            const comparison = {...fixture, case: `${fixture.case}/${mode}/encoding=${encoding}`};
            // Expectations are packing of the complete NetworkX fixture paths, never a
            // traversal of GPU predecessors or a local routing oracle.
            const found = fixture.paths.map((path, row) =>
              Number(
                Boolean(
                  state.complete &&
                    row < state.count &&
                    path.length &&
                    path.length <= maximumPathLength
                )
              )
            );
            // The walk bound still diagnoses truncation when the search-complete flag
            // suppresses found paths; that flag does not suppress walking.
            const truncated = fixture.paths.some(
              (path, row) => row < state.count && path.length > maximumPathLength
            );
            compareParity(
              'GPUNetworkPathExtraction',
              comparison,
              'every found flag',
              await resources.readUint('found'),
              found,
              EXACT
            );
            compareParity(
              'GPUNetworkPathExtraction',
              comparison,
              'every path cost and infinite mask',
              await resources.readFloat('path-costs'),
              fixture.pathCosts.map((cost, row) => (found[row] ? cost : Infinity)),
              EXACT
            );
            for (const [prefix, paths, capacity, mapping] of [
              [
                'nodes',
                fixture.paths,
                nodeCapacity,
                (row: number) => (mode === 'showcase-nodes-only' ? row : mapNodeId(input, row))
              ],
              ['edges', fixture.edgePaths, edgeCapacity, (row: number) => mapEdgeId(input, row)]
            ] as const) {
              if (prefix === 'edges' && mode === 'showcase-nodes-only') continue;
              const selected = paths.map((path, row) => (found[row] ? path.map(mapping) : []));
              const values = selected.flat();
              const offsets = [0];
              for (const path of selected) offsets.push(offsets[offsets.length - 1] + path.length);
              compareParity(
                'GPUNetworkPathExtraction',
                comparison,
                `${prefix}: every published stable ID`,
                (await resources.readUint(`${prefix}-ids`)).slice(
                  0,
                  Math.min(values.length, capacity)
                ),
                values.slice(0, capacity),
                EXACT
              );
              compareParity(
                'GPUNetworkPathExtraction',
                comparison,
                `${prefix}: every unclamped path offset`,
                await resources.readUint(`${prefix}-offsets`),
                offsets,
                EXACT
              );
              compareParity(
                'GPUNetworkPathExtraction',
                comparison,
                `${prefix}: count, required, overflow`,
                [
                  ...(await resources.readUint(`${prefix}-count`)),
                  ...(await resources.readUint(`${prefix}-required`)),
                  ...(await resources.readUint(`${prefix}-overflow`))
                ],
                [
                  Math.min(values.length, capacity),
                  values.length,
                  Number(Boolean(truncated || values.length > capacity))
                ],
                EXACT
              );
            }
          }
        } finally {
          compiled.destroy();
          targetCount.destroy();
          complete.destroy();
          resources.destroy();
        }
      }, 180000);
    }
  }
}
