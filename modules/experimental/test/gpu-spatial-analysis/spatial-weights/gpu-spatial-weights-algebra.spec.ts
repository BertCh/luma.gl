// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUSpatialWeightsAlgebra,
  GPUSpatialWeightsSummary,
  GPUSpatialWeightsTransform,
  type GPUSpatialWeightsAlgebraProps
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {
  computeBinaryOracle,
  computeBlockOracle,
  computeHigherOrderOracle,
  computeSelfWeightOracle,
  computeSubgraphOracle,
  computeSummaryOracle,
  createRandomWeights,
  truncateCSR,
  type OracleCombineRule
} from './spatial-weights-algebra-oracle';
import {
  assertValidCSR,
  computeLatticeOracle,
  computeTransformOracle,
  createSeededRandom,
  type OracleCSR
} from './spatial-weights-oracle';

function expectClose(actual: number[], expected: number[], label: string, relative = 2e-5): void {
  expect(actual.length, `${label} length`).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Math.abs(actual[index] - expected[index]) > 1e-6 + relative * Math.abs(expected[index])) {
      throw new Error(`${label}: [${index}] ${actual[index]} != ${expected[index]}`);
    }
  }
}

function expectSameCSR(actual: OracleCSR, expected: OracleCSR, label: string, distances = false) {
  expect(actual.offsets, `${label} offsets`).toEqual(expected.offsets);
  expect(actual.neighbors, `${label} neighbors`).toEqual(expected.neighbors);
  expectClose(actual.weights, expected.weights, `${label} weights`);
  if (distances) expectClose(actual.distances, expected.distances, `${label} distances`);
}

/** Runs one algebra operation and reads the CSR, overflow flag and total back. */
async function runAlgebra(
  rig: WeightsRig,
  rows: number,
  capacity: number,
  distances: boolean,
  create: (
    output: ReturnType<WeightsRig['weightsOutput']>['spatialWeights'],
    overflow: GraphDataView<'uint32'>,
    total: GraphDataView<'uint32'>
  ) => GPUSpatialWeightsAlgebraProps
) {
  const output = rig.weightsOutput(rows, capacity, distances);
  const overflow = rig.output('uint32', 1);
  const total = rig.output('uint32', 1);
  rig.run(new GPUSpatialWeightsAlgebra(create(output.spatialWeights, overflow.view, total.view)));
  return {
    csr: await output.read(),
    overflow: (await readUint32(overflow.buffer, 1))[0],
    total: (await readUint32(total.buffer, 1))[0]
  };
}

const ROWS = 48;

it('GPUSpatialWeightsAlgebra set operations match the oracle for every weight rule', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const left = createRandomWeights(ROWS, 0.12, 1);
  const right = createRandomWeights(ROWS, 0.12, 2);
  const rules: OracleCombineRule[] = ['left', 'right', 'sum', 'min', 'max', 'product', 'binary'];
  for (const operation of ['union', 'intersection', 'difference', 'symmetricDifference'] as const) {
    for (const rule of operation === 'union' || operation === 'intersection'
      ? rules
      : ['left' as const, 'binary' as const]) {
      const expected = computeBinaryOracle(operation, left, right, rule);
      expect(expected.neighbors.length, `${operation} nonempty`).toBeGreaterThan(0);
      const rig = new WeightsRig(device);
      const leftWeights = rig.uploadWeights(left, 2);
      const rightWeights = rig.uploadWeights(right, 1);
      const actual = await runAlgebra(
        rig,
        ROWS,
        expected.neighbors.length + 5,
        true,
        (output, overflow, total) => ({
          operation,
          left: leftWeights,
          right: rightWeights,
          weightRule: rule,
          output,
          overflow,
          totalNeighbors: total
        })
      );
      const label = `${operation} ${rule}`;
      assertValidCSR(actual.csr, ROWS, label);
      expectSameCSR(actual.csr, expected, label, true);
      expect(actual.overflow, `${label} overflow`).toBe(0);
      expect(actual.total, `${label} total`).toBe(expected.neighbors.length);
      rig.destroy();
    }
  }
});

it('GPUSpatialWeightsAlgebra identities hold on the GPU', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const left = createRandomWeights(ROWS, 0.15, 3);
  const right = createRandomWeights(ROWS, 0.15, 4);
  const union = computeBinaryOracle('union', left, right);
  const intersection = computeBinaryOracle('intersection', left, right);
  const run = async (operation: 'difference' | 'symmetricDifference', capacity: number) => {
    const rig = new WeightsRig(device);
    const leftWeights = rig.uploadWeights(left);
    const rightWeights = rig.uploadWeights(right);
    const result = await runAlgebra(rig, ROWS, capacity, false, (output, overflow) => ({
      operation,
      left: leftWeights,
      right: rightWeights,
      output,
      overflow
    }));
    rig.destroy();
    return result;
  };
  const difference = await run('difference', left.neighbors.length);
  const symmetric = await run('symmetricDifference', union.neighbors.length);
  // |A u B| = |A| + |B| - |A n B| and A xor B = (A u B) minus (A n B).
  expect(union.neighbors.length).toBe(
    left.neighbors.length + right.neighbors.length - intersection.neighbors.length
  );
  expect(symmetric.csr.neighbors.length).toBe(
    union.neighbors.length - intersection.neighbors.length
  );
  expect(difference.csr.neighbors.length).toBe(
    left.neighbors.length - intersection.neighbors.length
  );
  expect(intersection.neighbors.length).toBeGreaterThan(0);
});

it('GPUSpatialWeightsAlgebra reports overflow and keeps a valid truncated CSR', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const left = createRandomWeights(ROWS, 0.12, 5);
  const right = createRandomWeights(ROWS, 0.12, 6);
  const full = computeBinaryOracle('union', left, right);
  const capacity = Math.floor(full.neighbors.length / 2);
  const rig = new WeightsRig(device);
  const leftWeights = rig.uploadWeights(left);
  const rightWeights = rig.uploadWeights(right);
  const actual = await runAlgebra(rig, ROWS, capacity, false, (output, overflow, total) => ({
    operation: 'union',
    left: leftWeights,
    right: rightWeights,
    output,
    overflow,
    totalNeighbors: total
  }));
  expect(actual.overflow).toBe(1);
  expect(actual.total).toBe(full.neighbors.length);
  const expected = truncateCSR(full, capacity);
  assertValidCSR(actual.csr, ROWS, 'truncated union');
  expectSameCSR(actual.csr, expected, 'truncated union');
  rig.destroy();
});

it('GPUSpatialWeightsAlgebra higherOrder matches breadth-first search', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const bases = [
    createRandomWeights(ROWS, 0.05, 7),
    computeLatticeOracle({width: 7, height: 6, criterion: 'rook'})
  ];
  for (const base of bases) {
    const rows = base.offsets.length - 1;
    for (const order of [1, 2, 3, 4]) {
      for (const cumulative of [false, true]) {
        const expected = computeHigherOrderOracle(base, order, cumulative);
        expect(expected.neighbors.length, `order ${order} nonempty`).toBeGreaterThan(0);
        const rig = new WeightsRig(device);
        const weights = rig.uploadWeights(base);
        const actual = await runAlgebra(
          rig,
          rows,
          expected.neighbors.length + 4,
          false,
          (output, overflow) => ({
            operation: 'higherOrder',
            weights,
            order,
            cumulative,
            // Lower-order sets can exceed an exact-order output.
            workCapacity: computeHigherOrderOracle(base, order, true).neighbors.length + 4,
            output,
            overflow
          })
        );
        const label = `rows ${rows} order ${order} cumulative ${cumulative}`;
        assertValidCSR(actual.csr, rows, label);
        expectSameCSR(actual.csr, expected, label);
        expect(actual.overflow, `${label} overflow`).toBe(0);
        rig.destroy();
      }
    }
  }
});

it('GPUSpatialWeightsAlgebra selfWeight, subgraph and block match the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const base = createRandomWeights(ROWS, 0.12, 8);
  const random = createSeededRandom(9);
  const perRow = new Float32Array(ROWS).map(() => 0.25 + random() * 2);
  const mask = new Uint32Array(ROWS).map(() => (random() < 0.7 ? 1 : 0));
  const groupIds = new Uint32Array(ROWS).map(() => Math.floor(random() * 7));
  const groupCount = 6; // group ID 6 means no group

  for (const selfWeight of [0.5, 'perRow'] as const) {
    const rig = new WeightsRig(device);
    const weights = rig.uploadWeights(base);
    const expected = computeSelfWeightOracle(base, selfWeight === 'perRow' ? perRow : selfWeight);
    const actual = await runAlgebra(
      rig,
      ROWS,
      expected.neighbors.length,
      true,
      (output, overflow) => ({
        operation: 'selfWeight',
        weights,
        selfWeight: selfWeight === 'perRow' ? rig.input(perRow, 'float32', ROWS) : selfWeight,
        output,
        overflow
      })
    );
    expectSameCSR(actual.csr, expected, `selfWeight ${selfWeight}`, true);
    expect(actual.overflow).toBe(0);
    // Every row now lists itself.
    for (let row = 0; row < ROWS; row++) {
      const slots = actual.csr.neighbors.slice(
        actual.csr.offsets[row],
        actual.csr.offsets[row + 1]
      );
      expect(slots).toContain(row);
    }
    rig.destroy();
  }

  // A row that already lists itself has the weight replaced, not duplicated.
  {
    const rig = new WeightsRig(device);
    const withSelf = computeSelfWeightOracle(base, 9);
    const weights = rig.uploadWeights(withSelf);
    const expected = computeSelfWeightOracle(withSelf, 0.5);
    const actual = await runAlgebra(
      rig,
      ROWS,
      withSelf.neighbors.length,
      true,
      (output, overflow) => ({
        operation: 'selfWeight',
        weights,
        selfWeight: 0.5,
        output,
        overflow
      })
    );
    expectSameCSR(actual.csr, expected, 'selfWeight replace', true);
    rig.destroy();
  }

  {
    const rig = new WeightsRig(device);
    const weights = rig.uploadWeights(base);
    const expected = computeSubgraphOracle(base, mask);
    expect(expected.neighbors.length).toBeGreaterThan(0);
    expect(expected.neighbors.length).toBeLessThan(base.neighbors.length);
    const actual = await runAlgebra(
      rig,
      ROWS,
      expected.neighbors.length + 2,
      true,
      (output, overflow) => ({
        operation: 'subgraph',
        weights,
        mask: rig.input(mask, 'uint32', ROWS),
        output,
        overflow
      })
    );
    assertValidCSR(actual.csr, ROWS, 'subgraph');
    expectSameCSR(actual.csr, expected, 'subgraph', true);
    rig.destroy();
  }

  {
    const rig = new WeightsRig(device);
    const expected = computeBlockOracle(groupIds, groupCount);
    expect(expected.neighbors.length).toBeGreaterThan(0);
    const actual = await runAlgebra(
      rig,
      ROWS,
      expected.neighbors.length + 2,
      false,
      (output, overflow) => ({
        operation: 'block',
        groupIds: rig.input(groupIds, 'uint32', ROWS),
        groupCount,
        output,
        overflow
      })
    );
    assertValidCSR(actual.csr, ROWS, 'block');
    expectSameCSR(actual.csr, expected, 'block');
    expect(actual.overflow).toBe(0);
    rig.destroy();
  }
});

it('GPUSpatialWeightsSummary matches the oracle and the libpysal 3x3 rook values', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const lattice = computeLatticeOracle({width: 3, height: 3, criterion: 'rook'});
  const random = createSeededRandom(11);
  // Isolates, asymmetric weights and one-way links.
  const directed = createRandomWeights(ROWS, 0.06, 12);
  const asymmetricWeights: OracleCSR = {
    ...directed,
    weights: directed.weights.map(weight => weight * (random() < 0.3 ? 2 : 1))
  };
  for (const [name, csr] of [
    ['lattice', lattice],
    ['directed', asymmetricWeights],
    ['symmetric', createRandomWeights(ROWS, 0.1, 13, true)]
  ] as const) {
    const rows = csr.offsets.length - 1;
    const rig = new WeightsRig(device);
    const weights = rig.uploadWeights(csr, 3);
    const statistics = rig.output('float32', 3);
    const counts = rig.output('uint32', 5);
    const cardinality = rig.output('uint32', rows);
    rig.run(
      new GPUSpatialWeightsSummary({
        weights,
        statistics: statistics.view,
        counts: counts.view,
        cardinality: cardinality.view
      })
    );
    const expected = computeSummaryOracle(csr);
    const [s0, s1, s2] = await readFloat32(statistics.buffer, 3);
    expectClose([s0, s1, s2], [expected.s0, expected.s1, expected.s2], `${name} S0 S1 S2`, 1e-5);
    expect(await readUint32(counts.buffer, 5), `${name} counts`).toEqual([
      expected.slots,
      expected.asymmetricSlots,
      expected.isolates,
      expected.minimumCardinality,
      expected.maximumCardinality
    ]);
    expect(await readUint32(cardinality.buffer, rows)).toEqual(expected.cardinality);
    expect(s0).toBeGreaterThan(0);
    if (name === 'lattice') {
      // libpysal lat2W(3, 3) (rook): s0 = 24, s1 = 48, s2 = 272, symmetric, no isolates.
      expect([s0, s1, s2]).toEqual([24, 48, 272]);
    }
    if (name === 'directed') {
      expect(expected.asymmetricSlots).toBeGreaterThan(0);
      expect(expected.isolates).toBeGreaterThan(0);
    }
    rig.destroy();
  }
});

it('GPUSpatialWeightsTransform double and variance match the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const base = createRandomWeights(ROWS, 0.1, 14);
  for (const inPlace of [false, true]) {
    const cases = [
      {operation: 'double' as const, doubleSum: 'one' as const},
      {operation: 'double' as const, doubleSum: 'rows' as const},
      {operation: 'variance' as const, doubleSum: undefined}
    ];
    for (const {operation, doubleSum} of cases) {
      const rig = new WeightsRig(device);
      const upload = rig.uploadWeights(base, 3);
      const output = rig.output('float32', upload.neighbors.length);
      rig.run(
        new GPUSpatialWeightsTransform({
          operation,
          doubleSum,
          weights: upload,
          output: inPlace ? undefined : output.view
        })
      );
      const actual = await readFloat32(
        inPlace ? rig.bufferOf(upload.weights) : output.buffer,
        base.neighbors.length
      );
      const expected = computeTransformOracle(base, operation, {doubleSum});
      expectClose(actual, expected, `${operation} ${doubleSum} inPlace=${inPlace}`, 1e-4);
      expect(actual.some(value => value > 0)).toBe(true);
      const total = actual.reduce((sum, value) => sum + value, 0);
      if (operation === 'double') {
        // PySAL D: weights sum to 1 (or to n with doubleSum 'rows').
        expect(total).toBeCloseTo(doubleSum === 'rows' ? ROWS : 1, 3);
      } else {
        // PySAL V: s_ij * n / Q sums to n.
        expect(total).toBeCloseTo(ROWS, 2);
      }
      rig.destroy();
    }
  }
});
