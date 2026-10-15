// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {cellFixtures} from '../../oracles/parity-wp2-cells-fixtures';
import {compareParity} from '../../oracles/parity-wp2-assert';
import type {CellTableFixture} from '../../oracles/parity-wp2-cells-harness';
import {aggregateCellsOnCPU, splitCellKey} from './cell-aggregation-oracle';
import {compareCellTablesOnCPU} from '../cell-table-compare/cell-table-compare-oracle';
import seam from '../../oracles/fixtures/parity-wp2-great-circle-arcs-seam-regression.json';

function joinWords(words: number[], index: number): bigint {
  return BigInt(words[2 * index]) | (BigInt(words[2 * index + 1]) << 32n);
}
function tableRows(table: CellTableFixture) {
  return table.counts.map((count, index) => ({
    key: joinWords(table.cells, index),
    count,
    sum: BigInt.asIntN(64, joinWords(table.sums, index))
  }));
}

for (const fixture of cellFixtures) {
  it(`H3/pandas input and aggregation convention bridge ${fixture.expected.name}`, () => {
    const {input, expected} = fixture;
    expect(input.keys.length).toBe(input.values.length * 2);
    if (expected.name === 'nature') expect(input.values.length).toBe(43557);
    for (const [frameIndex, frame] of expected.frames.entries()) {
      const actual = aggregateCellsOnCPU({
        family: 'h3',
        resolution: 9,
        keys: input.values.map((_, index) => joinWords(input.keys, index)),
        values: Float32Array.from(input.values),
        mask: Uint32Array.from(input.masks[frameIndex]),
        sumScale: 65536
      });
      const reference = frame.levels[0];
      for (const [metric, measured, wanted] of [
        ['keys', actual.flatMap(row => splitCellKey(row.key)), reference.cells],
        ['counts', actual.map(row => row.count), reference.counts],
        [
          'fixed sums',
          actual.flatMap(row => splitCellKey(BigInt.asUintN(64, row.sum))),
          reference.sums
        ],
        ['minimums', actual.map(row => row.minimum), reference.minimums],
        ['maximums', actual.map(row => row.maximum), reference.maximums]
      ] as const)
        compareParity(
          'GPUCellAggregation',
          `${expected.name}/${frame.name}/node bridge`,
          expected.reference,
          metric,
          measured,
          wanted
        );
      for (const level of frame.levels)
        expect(level.counts.reduce((sum, value) => sum + value, 0)).toBe(
          reference.counts.reduce((sum, value) => sum + value, 0)
        );
    }
  });
  it(`pandas outer-join and score convention bridge ${fixture.expected.name}`, () => {
    for (const entry of fixture.comparison.comparisons) {
      const actual = compareCellTablesOnCPU(
        tableRows(fixture.expected.frames[entry.before].levels[0]),
        tableRows(fixture.expected.frames[entry.after].levels[0]),
        {
          measure: entry.measure,
          sumScale: 65536,
          zScore: entry.measure === 'count' ? 'poisson' : 'standardized',
          capacity: entry.capacity
        }
      );
      const label = `${fixture.expected.name}/${entry.before}->${entry.after}/${entry.measure}/${entry.capacity}/node bridge`;
      compareParity(
        'GPUCellTableCompare',
        label,
        fixture.comparison.reference,
        'keys',
        actual.rows.flatMap(row => splitCellKey(row.key)),
        entry.expected.cells
      );
      compareParity(
        'GPUCellTableCompare',
        label,
        fixture.comparison.reference,
        'status',
        [actual.rows.length, actual.total, Number(actual.overflow)],
        [entry.expected.count, entry.expected.requiredCount, entry.expected.overflow]
      );
      for (const field of [
        'presence',
        'before',
        'after',
        'delta',
        'ratio',
        'percentChange',
        'zScore'
      ] as const)
        compareParity(
          'GPUCellTableCompare',
          label,
          fixture.comparison.reference,
          field,
          actual.rows.map(row => row[field]),
          entry.expected[field],
          field === 'presence' ? 0 : 1e-6,
          field === 'presence' ? 0 : 1e-4
        );
    }
  });
}

it('integer-scaled seam subtraction preserves all input bits under reassociation', () => {
  const entry = seam.cases[0];
  for (let index = 0; index < entry.sources.length; index += 2) {
    const source = entry.sources[index];
    const destination = entry.targets[index];
    const exact = destination - source - 360 * Math.round((destination - source) / 360);
    const sourceUnits = source * 131072;
    const destinationUnits = destination * 131072;
    expect(Number.isInteger(sourceUnits)).toBe(true);
    expect(Number.isInteger(destinationUnits)).toBe(true);
    const integerDifference = destinationUnits - sourceUnits + (source > 0 ? 47185920 : -47185920);
    expect(Math.fround(integerDifference) * 2 ** -17).toBe(exact);
    if (index < 8) {
      // This is a legal floating-point reassociation of the round-5 helper. It loses
      // the same longitude bits before the small seam difference can be recovered.
      const reassociated = Math.fround(
        Math.fround(destination - source) - 360 * Math.round((destination - source) / 360)
      );
      expect(Math.abs(reassociated - exact)).toBeGreaterThan(1e-5);
    }
  }
});
