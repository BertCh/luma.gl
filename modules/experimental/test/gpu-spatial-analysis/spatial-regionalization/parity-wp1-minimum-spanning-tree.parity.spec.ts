// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUSpatialWeightsMinimumSpanningTree} from '../../../src/gpu-spatial-analysis/spatial-regionalization';
import fixture from '../../oracles/fixtures/parity-wp1-round6-forest.json';
import {decodeRound4Fixture} from '../../oracles/parity-wp1-round4-fixture';
import type {ForestCase} from '../../oracles/parity-wp1-round6-fixture';
import {createRound4Harness} from '../../oracles/parity-wp1-round4-harness';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';

const decoded = decodeRound4Fixture<{cases: ForestCase[]}>(fixture);

/** Generic graph feasibility only; the independent SciPy fixture supplies optimal costs. */
function inspectForest(rows: number, endpoints: number[]) {
  const parents = Array.from({length: rows}, (_, row) => row);
  const findRoot = (row: number): number => {
    while (parents[row] !== row) row = parents[row];
    return row;
  };
  let cycles = 0;
  for (let index = 0; index < endpoints.length; index += 2) {
    const first = findRoot(endpoints[index]);
    const second = findRoot(endpoints[index + 1]);
    if (first === second) cycles++;
    else parents[Math.max(first, second)] = Math.min(first, second);
  }
  return {cycles, labels: parents.map((_, row) => findRoot(row))};
}

for (let caseIndex = 0; caseIndex < 4; caseIndex++) {
  for (const truncated of [false, true]) {
    it(`GPUSpatialWeightsMinimumSpanningTree SciPy forest ${caseIndex} truncated=${truncated}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const testCase = (await decoded).cases[caseIndex];
      const capacity = truncated ? 2 : testCase.n;
      const slotCount = testCase.input.neighbors.length;
      const harness = createRound4Harness(device);
      try {
        harness.graph.add(
          new GPUSpatialWeightsMinimumSpanningTree({
            weights: {
              offsets: harness.input('offsets', testCase.input.offsets, 'uint32'),
              neighbors: harness.input(
                'neighbors',
                [...testCase.input.neighbors, 0, 0, 0],
                'uint32'
              ),
              weights: harness.input('weights', [...testCase.input.weights, 0, 0, 0], 'float32')
            },
            values: harness.input('values', testCase.values, 'float32'),
            columnCount: testCase.columnCount,
            standardize: testCase.standardize,
            output: {
              treeEdgeFlags: harness.output('flags', slotCount + 3, 'uint32'),
              componentLabels: harness.output('labels', testCase.n, 'uint32'),
              standardizedValues: testCase.standardize
                ? harness.output('standardized', testCase.values.length, 'float32')
                : undefined,
              edges: {
                ids: harness.output('ids', capacity, 'uint32'),
                status: {
                  count: harness.output('count', 1, 'uint32'),
                  overflow: harness.output('overflow', 1, 'uint32'),
                  requiredCount: harness.output('required', 1, 'uint32')
                }
              },
              edgeEndpoints: harness.output('endpoints', capacity * 2, 'uint32'),
              edgeDissimilarities: harness.output('costs', capacity, 'float32')
            }
          })
        );
        harness.run();
        const comparison = {...testCase, name: `${testCase.name}/capacity-${capacity}`};
        const compare = (metric: string, actual: number[], expected: number[], profile = 'E') =>
          compareParity(
            'GPUSpatialWeightsMinimumSpanningTree',
            comparison,
            metric,
            actual,
            expected,
            profile
          );
        const flags = await harness.read('flags');
        compare(
          'binary-flags',
          flags,
          flags.map(value => Number(value !== 0))
        );
        compare('unused-flags', flags.slice(slotCount), [0, 0, 0]);
        const sourceRows = testCase.input.neighbors.map(() => 0);
        for (let row = 0; row < testCase.n; row++) {
          for (
            let slot = testCase.input.offsets[row];
            slot < testCase.input.offsets[row + 1];
            slot++
          )
            sourceRows[slot] = row;
        }
        const selected = flags.flatMap((value, slot) =>
          value === 1 && slot < slotCount ? [slot] : []
        );
        const endpoints = selected.flatMap(slot => [
          sourceRows[slot],
          testCase.input.neighbors[slot]
        ]);
        compare(
          'lower-row-slots',
          selected.map(slot => Number(sourceRows[slot] < testCase.input.neighbors[slot])),
          selected.map(() => 1)
        );
        const connectivity = inspectForest(testCase.n, endpoints);
        compare('cycles', [connectivity.cycles], [0]);
        compare('selected-edge-components', connectivity.labels, testCase.expected.componentLabels);
        compare('componentLabels', await harness.read('labels'), testCase.expected.componentLabels);
        compare('edgeCount', [selected.length], [testCase.expected.edgeCount]);
        if (testCase.exactEdges)
          compare('distinct-cost-edge-flags', flags.slice(0, slotCount), testCase.expected.flags);
        const used = Math.min(capacity, testCase.expected.edgeCount);
        compare('compact-ids', (await harness.read('ids')).slice(0, used), selected.slice(0, used));
        compare(
          'compact-endpoints',
          (await harness.read('endpoints')).slice(0, used * 2),
          endpoints.slice(0, used * 2)
        );
        compare('count', await harness.read('count'), [used]);
        compare('requiredCount', await harness.read('required'), [testCase.expected.edgeCount]);
        compare('overflow', await harness.read('overflow'), [
          Number(testCase.expected.edgeCount > capacity)
        ]);
        compare(
          'every-compact-edge-cost',
          (await harness.read('costs')).slice(0, used),
          selected.slice(0, used).map(slot => testCase.expected.slotCosts[slot]),
          'S'
        );
        // Evaluate the entire GPU-selected forest using independent float64 costs,
        // even when the compact output is truncated. Equal-cost edges may differ.
        const total = selected.reduce((sum, slot) => sum + testCase.expected.slotCosts[slot], 0);
        compare('optimal-total-cost', [total], [testCase.expected.totalCost], 'S');
        compare('optimal-quality-ratio', [total / testCase.expected.totalCost], [1], 'H');
        if (testCase.standardize) {
          const standardized = await harness.read('standardized');
          for (let column = 0; column < testCase.columnCount; column++) {
            compare(
              `standardized-column-${column}`,
              standardized.filter((_, index) => index % testCase.columnCount === column),
              testCase.expected.standardizedValues.filter(
                (_, index) => index % testCase.columnCount === column
              ),
              'S'
            );
          }
        }
      } finally {
        harness.destroy();
      }
    }, 120000);
  }
}
