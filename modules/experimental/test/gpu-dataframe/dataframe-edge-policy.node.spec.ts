// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeClassBreaksOracle} from './column-classification/class-breaks-oracle';
import {computeColumnQuantilesOracle} from './column-classification/column-quantiles-oracle';
import {
  CLASS_BREAKS_EDGE_POLICY,
  ORDER_STATISTIC_EDGE_POLICY,
  VARIOGRAM_EDGE_POLICY
} from './dataframe-edge-policy';
import {interpolateQuantile} from './group-statistics/group-statistics-oracle';
import {getOrderStatisticSample} from './order-statistic-oracle';
import {computeVariogramOnCPU} from './pair-statistics/variogram-oracle';

/** Same value, with NaN equal to NaN and -0 distinct from +0. */
function expectSameValue(actual: number, expected: number, label: string): void {
  expect(Object.is(actual, expected), `${label}: ${actual} is not ${expected}`).toBe(true);
}

it('order-statistic edge policy: column quantiles and group statistics oracles agree (B-14)', () => {
  for (const row of ORDER_STATISTIC_EDGE_POLICY) {
    const column = computeColumnQuantilesOracle({values: row.values, quantiles: [row.probability]});
    expectSameValue(column.quantiles[0], row.expected, `column ${row.name}`);
    const group = interpolateQuantile(getOrderStatisticSample(row.values), row.probability);
    expectSameValue(Math.fround(group), row.expected, `group ${row.name}`);
  }
});

it('class breaks edge policy holds in the CPU mirror', () => {
  for (const row of CLASS_BREAKS_EDGE_POLICY) {
    const result = computeClassBreaksOracle({
      values: Float32Array.from(row.values),
      parameters: {
        method: row.method,
        classCount: row.classCount,
        customEdges: row.customEdges,
        standardDeviationInterval: row.standardDeviationInterval
      },
      maximumClassCount: 4,
      naturalBreaksBinCount: 16
    });
    expect(result.classCount, row.name).toBe(row.expectedClassCount);
    for (const [index, expected] of row.expectedBreaks.entries()) {
      expectSameValue(result.breaks[index], expected, `${row.name} e[${index}]`);
    }
  }
});

it('variogram edge policy holds in the CPU mirror', () => {
  for (const row of VARIOGRAM_EDGE_POLICY) {
    const [minX, minY, maxX, maxY, maximumDistance] = row.parameters;
    const result = computeVariogramOnCPU(
      {
        positions: Float32Array.from(row.positions),
        values: Float32Array.from(row.values),
        mask: row.mask ? Uint32Array.from(row.mask) : undefined
      },
      {bounds: [minX, minY, maxX, maxY], maximumDistance},
      2
    );
    expect(result.pairCounts, row.name).toEqual(row.expectedPairCounts);
    for (const [lag, expected] of row.expectedSemivariances.entries()) {
      expectSameValue(result.semivariances[lag], expected, `${row.name} lag ${lag}`);
    }
  }
});
