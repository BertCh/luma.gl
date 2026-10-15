// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUNetworkLineGraph} from '../../../src/gpu-network/network-line-graph';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';
import {lineGraphFixtures} from './parity-wp3-line-graph-fixture';

for (const {input, cases} of lineGraphFixtures) {
  for (const fixture of cases) {
    for (const bounded of [false, true]) {
      it(`GPUNetworkLineGraph NetworkX ${fixture.case}/bounded=${bounded}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const resources = createParityResources(device);
        const capacity = bounded ? Math.max(1, Math.floor(fixture.count / 3)) : fixture.count + 3;
        resources.graph.add(
          new GPUNetworkLineGraph({
            topology: {
              offsets: resources.input('offsets', 'uint32', input.offsets),
              neighbors: resources.input('neighbors', 'uint32', input.neighbors),
              weights: resources.input('weights', 'float32', input.weights),
              nodePositions: resources.input('positions', 'float32x2', input.positions)
            },
            parameters: resources.input('parameters', 'float32', fixture.parameters),
            bannedTurns: fixture.bans.length
              ? resources.input('bans', 'uint32', fixture.bans)
              : undefined,
            output: {
              topology: {
                offsets: resources.output('line-offsets', 'uint32', input.neighbors.length + 1),
                neighbors: resources.output('line-neighbors', 'uint32', capacity),
                weights: resources.output('line-weights', 'float32', capacity)
              },
              arcCount: resources.output('count', 'uint32', 1),
              status: {overflow: resources.output('overflow', 'uint32', 1)}
            }
          })
        );
        const compiled = resources.graph.compile();
        try {
          for (let encoding = 0; encoding < 2; encoding++) {
            submitGraph(device, compiled, undefined);
            const comparison = {
              ...fixture,
              case: `${fixture.case}/bounded=${bounded}/encoding=${encoding}`
            };
            compareParity(
              'GPUNetworkLineGraph',
              comparison,
              'every clamped CSR offset',
              await resources.readUint('line-offsets'),
              fixture.offsets.map(value => Math.min(value, capacity)),
              EXACT
            );
            const expectedNeighbors = fixture.neighbors.slice(0, capacity);
            const expectedWeights = fixture.weights.slice(0, capacity);
            while (expectedNeighbors.length < capacity) {
              expectedNeighbors.push(0xffffffff);
              expectedWeights.push(-1);
            }
            compareParity(
              'GPUNetworkLineGraph',
              comparison,
              'every line-node transition and unused sentinel',
              await resources.readUint('line-neighbors'),
              expectedNeighbors,
              EXACT
            );
            const actualWeights = await resources.readFloat('line-weights');
            compareParity(
              'GPUNetworkLineGraph',
              comparison,
              'every following-edge and turn cost',
              actualWeights,
              expectedWeights,
              fixture.exactCosts ? EXACT : {absolute: fixture.costBound, relative: 1e-4}
            );
            compareParity(
              'GPUNetworkLineGraph',
              comparison,
              'exact impassable and unused weight sentinels',
              actualWeights.map(value => (value < 0 ? value : 0)),
              expectedWeights.map(value => (value < 0 ? value : 0)),
              EXACT
            );
            compareParity(
              'GPUNetworkLineGraph',
              comparison,
              'arc count and overflow',
              [...(await resources.readUint('count')), ...(await resources.readUint('overflow'))],
              [Math.min(fixture.count, capacity), Number(fixture.count > capacity)],
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
}
