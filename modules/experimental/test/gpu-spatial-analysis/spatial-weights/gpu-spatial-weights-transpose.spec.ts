// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {createSeededRandom, type OracleCSR} from './spatial-weights-oracle';

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

/** Random sparse CSR with `rows` rows over `columns` columns, empty rows allowed. */
function createRandomCSR(
  rows: number,
  columns: number,
  density: number,
  seed: number,
  options: {square?: boolean} = {}
): OracleCSR {
  const random = createSeededRandom(seed);
  const csr: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      if (options.square && column === row) continue;
      if (random() < density) {
        csr.neighbors.push(column);
        csr.weights.push(Math.fround(0.25 + Math.floor(random() * 8) * 0.25));
        csr.distances.push(Math.fround(random() * 10));
      }
    }
    csr.offsets.push(csr.neighbors.length);
  }
  return csr;
}

/** CPU transpose with ascending rows. */
export function computeTransposeOracle(csr: OracleCSR, columns: number): OracleCSR {
  const rows = csr.offsets.length - 1;
  const buckets: {row: number; weight: number; distance: number}[][] = Array.from(
    {length: columns},
    () => []
  );
  for (let row = 0; row < rows; row++) {
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      buckets[csr.neighbors[slot]].push({
        row,
        weight: csr.weights[slot],
        distance: csr.distances[slot] ?? 0
      });
    }
  }
  const result: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  for (const bucket of buckets) {
    for (const entry of bucket) {
      result.neighbors.push(entry.row);
      result.weights.push(entry.weight);
      result.distances.push(entry.distance);
    }
    result.offsets.push(result.neighbors.length);
  }
  return result;
}

async function runTranspose(
  device: Device,
  csr: OracleCSR,
  columns: number,
  options: {slack?: number; withDistances?: boolean; checkSymmetry?: boolean} = {}
): Promise<{result: OracleCSR; asymmetricSlots?: number}> {
  const rows = csr.offsets.length - 1;
  const rig = new WeightsRig(device);
  const input = rig.uploadWeights(
    options.withDistances ? csr : {...csr, distances: []},
    options.slack ?? 0
  );
  const output = rig.weightsOutput(columns, input.neighbors.length, options.withDistances);
  const asymmetric = options.checkSymmetry ? rig.output('uint32', 1) : undefined;
  rig.run(
    new GPUSpatialWeightsTranspose({
      weights: input,
      columnCount: rows === columns ? undefined : columns,
      output: output.spatialWeights,
      asymmetricSlots: asymmetric?.view
    })
  );
  const result = await output.read();
  const asymmetricSlots = asymmetric ? (await readUint32(asymmetric.buffer, 1))[0] : undefined;
  rig.destroy();
  return {result, asymmetricSlots};
}

function expectSameCSR(actual: OracleCSR, expected: OracleCSR, label: string): void {
  expect(actual.offsets, `${label} offsets`).toEqual(expected.offsets);
  expect(actual.neighbors, `${label} neighbors`).toEqual(expected.neighbors);
  expect(actual.weights, `${label} weights`).toEqual(expected.weights);
}

it('GPUSpatialWeightsTranspose matches the oracle for square asymmetric weights', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const [rows, density, seed] of [
    [12, 0.3, 1],
    [97, 0.08, 2],
    [300, 0.03, 3]
  ] as const) {
    const csr = createRandomCSR(rows, rows, density, seed, {square: true});
    expect(csr.neighbors.length).toBeGreaterThan(0);
    const {result} = await runTranspose(device, csr, rows, {slack: 5, withDistances: true});
    const expected = computeTransposeOracle(csr, rows);
    expectSameCSR(result, expected, `square ${rows}`);
    expect(result.distances, `square ${rows} distances`).toEqual(expected.distances);
  }
});

it('GPUSpatialWeightsTranspose handles rectangular cross weights, isolates and empty rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const [rows, columns, seed] of [
    [20, 7, 4],
    [7, 31, 5],
    [64, 64, 6]
  ] as const) {
    const csr = createRandomCSR(rows, columns, 0.12, seed);
    // Force an empty first row and an unreached last column.
    const trimmed: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
    for (let row = 0; row < rows; row++) {
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
        if (row === 0 || csr.neighbors[slot] === columns - 1) continue;
        trimmed.neighbors.push(csr.neighbors[slot]);
        trimmed.weights.push(csr.weights[slot]);
        trimmed.distances.push(csr.distances[slot]);
      }
      trimmed.offsets.push(trimmed.neighbors.length);
    }
    const {result} = await runTranspose(device, trimmed, columns, {slack: 3});
    expectSameCSR(result, computeTransposeOracle(trimmed, columns), `${rows}x${columns}`);
    expect(result.offsets[columns]).toBe(result.offsets[columns - 1]);
  }
});

it('GPUSpatialWeightsTranspose handles hub columns and column counts at radix-key boundaries', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Offsets come from a binary search of the sorted keys, so exercise a column that every row
  // lists (a hub), empty columns on both sides, and column counts just around powers of two,
  // where the sentinel key (columns) needs the last representable key value.
  for (const columns of [1, 2, 3, 15, 16, 17, 31, 32, 33]) {
    const rows = 40;
    const csr: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
    for (let row = 0; row < rows; row++) {
      const listed = new Set<number>([0, columns - 1]);
      if (columns > 4) {
        listed.add(1 + ((row * 7) % (columns - 2)));
      }
      for (const column of [...listed].sort((a, b) => a - b)) {
        csr.neighbors.push(column);
        csr.weights.push(Math.fround(1 + ((row + column) % 5) * 0.5));
        csr.distances.push(0);
      }
      csr.offsets.push(csr.neighbors.length);
    }
    const {result} = await runTranspose(device, csr, columns, {slack: 9});
    expectSameCSR(result, computeTransposeOracle(csr, columns), `hub ${columns}`);
  }
});

it('GPUSpatialWeightsTranspose is an involution and drops out-of-range neighbors', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const csr = createRandomCSR(40, 25, 0.15, 7);
  const {result: once} = await runTranspose(device, csr, 25);
  const {result: twice} = await runTranspose(device, once, 40);
  expectSameCSR(twice, csr, 'transpose of transpose');
  // The same weights read as 40 x 10 columns lose the entries pointing at columns >= 10.
  const {result: clipped} = await runTranspose(device, csr, 10);
  const kept: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  for (let row = 0; row < 40; row++) {
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      if (csr.neighbors[slot] < 10) {
        kept.neighbors.push(csr.neighbors[slot]);
        kept.weights.push(csr.weights[slot]);
        kept.distances.push(csr.distances[slot]);
      }
    }
    kept.offsets.push(kept.neighbors.length);
  }
  expectSameCSR(clipped, computeTransposeOracle(kept, 10), 'clipped columns');
});

it('GPUSpatialWeightsTranspose reports asymmetric slots', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Symmetric pattern and values: zero.
  const base = createRandomCSR(30, 30, 0.1, 8, {square: true});
  const symmetric = new Map<number, number>();
  for (let row = 0; row < 30; row++) {
    for (let slot = base.offsets[row]; slot < base.offsets[row + 1]; slot++) {
      const column = base.neighbors[slot];
      symmetric.set(row * 30 + column, base.weights[slot]);
      symmetric.set(column * 30 + row, base.weights[slot]);
    }
  }
  const toCSR = (entries: Map<number, number>): OracleCSR => {
    const csr: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
    for (let row = 0; row < 30; row++) {
      for (let column = 0; column < 30; column++) {
        const weight = entries.get(row * 30 + column);
        if (weight !== undefined) {
          csr.neighbors.push(column);
          csr.weights.push(weight);
        }
      }
      csr.offsets.push(csr.neighbors.length);
    }
    return csr;
  };
  const zero = await runTranspose(device, toCSR(symmetric), 30, {checkSymmetry: true});
  expect(zero.asymmetricSlots).toBe(0);
  // One weight changed: positions (a, b) and (b, a) both mismatch.
  const [firstKey] = [...symmetric.keys()];
  const changed = new Map(symmetric);
  changed.set(firstKey, changed.get(firstKey)! + 1);
  const valueMismatch = await runTranspose(device, toCSR(changed), 30, {checkSymmetry: true});
  expect(valueMismatch.asymmetricSlots).toBe(2);
  // One link removed one way: positions (a, b) and (b, a) both mismatch.
  const oneWay = new Map(symmetric);
  oneWay.delete(firstKey);
  const patternMismatch = await runTranspose(device, toCSR(oneWay), 30, {checkSymmetry: true});
  expect(patternMismatch.asymmetricSlots).toBe(2);
});
