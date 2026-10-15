// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUNetworkCostMatrix} from '../../../src/gpu-network/network-accessibility';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {createParityResources} from '../parity-wp3-round2-resources';
import {
  costMatrixFixtures,
  type CostMatrixInput,
  type CostMatrixCase
} from './parity-wp3-cost-matrix-fixture';

function repeatRows(input: CostMatrixInput, cases: CostMatrixCase[], rowCount: number) {
  const indices = Array.from(
    {length: rowCount},
    (_, index) => index % Math.floor(input.startNodes.length / input.startsPerRow)
  );
  const repeat = (values: number[], width: number) =>
    indices.flatMap(index => values.slice(index * width, (index + 1) * width));
  return {
    input: {
      ...input,
      case: `showcase-${rowCount}`,
      rowCount,
      startNodes: repeat(input.startNodes, input.startsPerRow),
      startCosts: repeat(input.startCosts, input.startsPerRow),
      startRows: indices.flatMap((_, row) => new Array(input.startsPerRow).fill(row))
    },
    cases: cases.map(fixture => ({
      ...fixture,
      case: `${fixture.case}/showcase-rows-${rowCount}`,
      costs: repeat(fixture.costs, input.nodeCount),
      compact: repeat(fixture.compact, input.targets.length)
    }))
  };
}
const matrixGroups = [
  ...costMatrixFixtures,
  ...[128, 256, 384].map(rowCount =>
    repeatRows(costMatrixFixtures[0].input, costMatrixFixtures[0].cases, rowCount)
  )
];
for (const {input, cases} of matrixGroups) {
  for (const fixture of cases) {
    for (const mode of input.case.startsWith('showcase-')
      ? [8, 16, 32, 64].map(laneCount => ({
          name: `showcase-lanes-${laneCount}`,
          laneCount,
          compact: false,
          explicit: false
        }))
      : [
          {name: 'dense-one-lane-explicit-rows', laneCount: 1, compact: false, explicit: true},
          {name: 'dense-two-lanes-implicit-rows', laneCount: 2, compact: false, explicit: false},
          {name: 'compact-two-lanes-explicit-rows', laneCount: 2, compact: true, explicit: true}
        ]) {
      it(`GPUNetworkCostMatrix NetworkX ${fixture.case}/${mode.name}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const resources = createParityResources(device);
        const columnCount = mode.compact ? input.targets.length : input.nodeCount;
        // Explicit row IDs must work independently of input order.
        const orderStarts = (values: number[]) => (mode.explicit ? [...values].reverse() : values);
        resources.graph.add(
          new GPUNetworkCostMatrix({
            topology: {
              offsets: resources.input('offsets', 'uint32', input.offsets),
              neighbors: resources.input('neighbors', 'uint32', input.neighbors),
              weights: resources.input('weights', 'float32', input.weights)
            },
            startNodes: resources.input('starts', 'uint32', orderStarts(input.startNodes)),
            startCosts: resources.input('initial', 'float32', orderStarts(input.startCosts)),
            startRows: mode.explicit
              ? resources.input('rows', 'uint32', orderStarts(input.startRows))
              : undefined,
            startsPerRow: input.startsPerRow,
            targetNodes: mode.compact
              ? resources.input('targets', 'uint32', input.targets)
              : undefined,
            costLimit: resources.input('limit', 'float32', [fixture.budget]),
            laneCount: mode.laneCount,
            maximumIterations: input.case.startsWith('showcase-') ? 48 : 128,
            localIterations: 16,
            output: {
              costs: resources.output('costs', 'float32', input.rowCount * columnCount),
              status: {converged: resources.output('converged', 'uint32', 1)}
            }
          })
        );
        const compiled = resources.graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const comparison = {...fixture, case: `${fixture.case}/${mode.name}`};
          compareParity(
            'GPUNetworkCostMatrix',
            comparison,
            'every matrix entry and infinite mask',
            await resources.readFloat('costs'),
            mode.compact ? fixture.compact : fixture.costs,
            EXACT
          );
          compareParity(
            'GPUNetworkCostMatrix',
            comparison,
            'all lanes reached a fixed point',
            await resources.readUint('converged'),
            [1],
            EXACT
          );
        } finally {
          compiled.destroy();
          resources.destroy();
        }
      }, 180000);
    }
  }
}
