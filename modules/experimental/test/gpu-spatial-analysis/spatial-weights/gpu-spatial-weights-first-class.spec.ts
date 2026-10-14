// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_SPATIAL_WEIGHTS_VIOLATION,
  GPUSpatialWeightsSummary,
  GPUSpatialWeightsTransform,
  GPUSpatialWeightsValidation,
  type GPUSpatialWeights,
  type GPUSpatialWeightsKernel
} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import type {OracleCSR} from './spatial-weights-oracle';
import {WEIGHTS_ORACLE} from './spatial-weights-u4-oracle';

const VIOLATION = GPU_SPATIAL_WEIGHTS_VIOLATION;

/** Four rows, two neighbors each: a valid symmetric CSR. */
function createValidCSR(): OracleCSR {
  return {
    offsets: [0, 2, 4, 6, 8],
    neighbors: [1, 2, 0, 3, 0, 3, 1, 2],
    weights: [1, 1, 1, 1, 1, 1, 1, 1],
    distances: []
  };
}

it('GPUSpatialWeightsValidation accepts a valid CSR and flags every kind of violation', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const valid = createValidCSR();
  const cases: {name: string; csr: OracleCSR; capacity?: number; flags: number[]}[] = [
    {name: 'valid', csr: valid, flags: [0, 0, 0, 0]},
    {
      name: 'unsorted',
      csr: {...valid, neighbors: [2, 1, 0, 3, 0, 3, 1, 2]},
      flags: [VIOLATION.order, 0, 0, 0]
    },
    {
      name: 'duplicate',
      csr: {...valid, neighbors: [1, 2, 3, 3, 0, 3, 1, 2]},
      flags: [0, VIOLATION.order, 0, 0]
    },
    {
      name: 'self',
      csr: {...valid, neighbors: [1, 2, 0, 3, 2, 3, 2, 1]},
      flags: [0, 0, VIOLATION.self, VIOLATION.order]
    },
    {
      name: 'range',
      csr: {...valid, neighbors: [1, 2, 0, 3, 0, 3, 1, 9]},
      flags: [0, 0, 0, VIOLATION.neighborRange]
    },
    {
      name: 'nan weight',
      csr: {...valid, weights: [1, Number.NaN, 1, 1, 1, 1, 1, 1]},
      flags: [VIOLATION.weight, 0, 0, 0]
    },
    {
      name: 'negative and infinite weights',
      csr: {...valid, weights: [1, 1, -1, 1, 1, Number.POSITIVE_INFINITY, 1, 1]},
      flags: [0, VIOLATION.weight, VIOLATION.weight, 0]
    },
    {
      name: 'offsets[0] != 0',
      csr: {...valid, offsets: [1, 2, 4, 6, 8]},
      flags: [VIOLATION.offsets, 0, 0, 0]
    },
    {
      name: 'decreasing offsets',
      // Moving offsets[1] to 3 also shifts the slots of the rows around it: row 0 then reads
      // neighbors [1, 2, 0] (descending, and itself) and row 2 reads [0, 3, 0, 3] (descending).
      csr: {...valid, offsets: [0, 3, 2, 6, 8]},
      flags: [VIOLATION.order | VIOLATION.self, VIOLATION.offsets, VIOLATION.order, 0]
    },
    {
      name: 'offsets[rows] above capacity',
      csr: {...valid, offsets: [0, 2, 4, 6, 9]},
      flags: [0, 0, 0, VIOLATION.offsets]
    }
  ];
  for (const {name, csr, flags} of cases) {
    const rig = new WeightsRig(device);
    const weights = rig.uploadWeights(csr, 0);
    const invalidCount = rig.output('uint32', 1);
    const rowFlags = rig.output('uint32', 4);
    rig.run(
      new GPUSpatialWeightsValidation({
        weights,
        invalidCount: invalidCount.view,
        flags: rowFlags.view
      })
    );
    expect(await readUint32(rowFlags.buffer, 4), `${name} flags`).toEqual(flags);
    expect(await readUint32(invalidCount.buffer, 1), `${name} invalidCount`).toEqual([
      flags.filter(flag => flag !== 0).length
    ]);
    rig.destroy();
  }
});

it('GPUSpatialWeightsSummary counts islands as rows without an active slot, as libpysal islands', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Oracle: fix/work/A/oracle-u4-weights.py (libpysal 4.15): W built without the zero-weight entry,
  // islands [3, 4, 5], cardinalities [2, 2, 2, 0, 0, 0], s0 = 7, s1 = 21, s2 = 70. Row 4 lists
  // one zero-weight slot (4 -> 5) and rows 3 and 5 are empty.
  const csr: OracleCSR = {
    offsets: [0, 2, 4, 6, 6, 7, 7],
    neighbors: [1, 2, 0, 2, 0, 1, 5],
    weights: [1, 2, 1, 0.5, 2, 0.5, 0],
    distances: []
  };
  const {islands} = WEIGHTS_ORACLE;
  const rig = new WeightsRig(device);
  const weights = rig.uploadWeights(csr, 0);
  const statistics = rig.output('float32', 3);
  const counts = rig.output('uint32', 6);
  const cardinality = rig.output('uint32', 6);
  rig.run(
    new GPUSpatialWeightsSummary({
      weights,
      statistics: statistics.view,
      counts: counts.view,
      cardinality: cardinality.view
    })
  );
  const [s0, s1, s2] = await readFloat32(statistics.buffer, 3);
  expect([s0, s1, s2]).toEqual([islands.s0, islands.s1, islands.s2]);
  expect(await readUint32(cardinality.buffer, 6)).toEqual(islands.cardinalities);
  const [slots, asymmetricSlots, isolates, minimum, maximum, emptyRows] = await readUint32(
    counts.buffer,
    6
  );
  expect(isolates).toBe(islands.islands.length);
  expect([minimum, maximum]).toEqual([islands.minCardinality, islands.maxCardinality]);
  // The zero-weight slot is listed (7 slots) but is not a neighbor, and rows 3 and 5 have no slot.
  expect([slots, asymmetricSlots, emptyRows]).toEqual([7, 1, 2]);
  rig.destroy();
});

it('GPUSpatialWeightsTransform adaptive kernel matches libpysal Kernel(fixed=False) on kNN rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Oracle: fix/work/A/oracle-u4-weights.py (libpysal 4.15), k = 3, self slot removed. The k-th
  // neighbor of each row keeps the eps tail weight (bandwidth = d_k * 1.0000001).
  const kernels: GPUSpatialWeightsKernel[] = [
    'triangular',
    'epanechnikov',
    'bisquare',
    'uniform',
    'gaussian'
  ];
  for (const kernel of kernels) {
    const {rows} = WEIGHTS_ORACLE.kernels[kernel];
    const csr: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
    for (const row of rows) {
      for (const [neighbor, distance] of row) {
        csr.neighbors.push(neighbor);
        csr.weights.push(1);
        csr.distances.push(distance);
      }
      csr.offsets.push(csr.neighbors.length);
    }
    const rig = new WeightsRig(device);
    const upload: GPUSpatialWeights = {
      ...rig.uploadWeights(csr, 0),
      info: {
        rowCount: rows.length,
        transform: 'raw',
        symmetric: false,
        provenance: {source: 'test', criterion: 'knn', k: WEIGHTS_ORACLE.k}
      }
    };
    const output = rig.output('float32', csr.neighbors.length);
    const transform = new GPUSpatialWeightsTransform({
      operation: 'kernel',
      kernel,
      bandwidth: 'adaptive',
      weights: upload,
      output: output.view
    });
    rig.run(transform);
    const actual = await readFloat32(output.buffer, csr.neighbors.length);
    const expected = rows.flatMap(row => row.map(([, , weight]) => weight));
    expected.forEach((weight, slot) => {
      expect(Math.abs(actual[slot] - weight), `${kernel} slot ${slot}`).toBeLessThan(1e-5);
    });
    if (kernel === 'gaussian') {
      // libpysal's gaussian is not compact: the k-th neighbor sits at z = 1 / 1.0000001, so it keeps
      // phi(1) = 0.24197 (the slot comparison above already matches the libpysal weights).
      csr.offsets.slice(0, -1).forEach((begin, row) => {
        const end = csr.offsets[row + 1];
        let farthest = begin;
        for (let slot = begin; slot < end; slot++) {
          if (csr.distances[slot] > csr.distances[farthest]) farthest = slot;
        }
        expect(actual[farthest], `${kernel} row ${row} tail`).toBeCloseTo(0.24197, 4);
      });
    } else if (kernel !== 'uniform') {
      // The k-th neighbor (largest distance) keeps a tiny positive weight, as in libpysal.
      csr.offsets.slice(0, -1).forEach((begin, row) => {
        const end = csr.offsets[row + 1];
        let farthest = begin;
        for (let slot = begin; slot < end; slot++) {
          if (csr.distances[slot] > csr.distances[farthest]) farthest = slot;
        }
        expect(actual[farthest], `${kernel} row ${row} tail`).toBeGreaterThan(0);
        expect(actual[farthest], `${kernel} row ${row} tail`).toBeLessThan(1e-5);
      });
    }
    expect(transform.outputWeights.info?.transform).toBe('kernel');
    rig.destroy();
  }
});
