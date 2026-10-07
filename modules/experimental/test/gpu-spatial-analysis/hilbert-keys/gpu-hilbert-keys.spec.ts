// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUHilbertInvalidKey,
  GPUHilbertKeys
} from '../../../src/gpu-spatial-analysis/hilbert-keys/index';
import {createGeometryFixture, createRandom} from '../outline-geometry/geometry-fixture';
import {
  getCellReference,
  getHilbertCellReference,
  getHilbertIndexReference
} from './hilbert-oracle';

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

function createPointsFixture(device: Device, points: Float32Array, order: number, bounds: boolean) {
  const count = points.length / 2;
  return createGeometryFixture(device, {
    inputs: {
      points: {values: points, format: 'float32x2'},
      ...(bounds ? {bounds: {values: Float32Array.of(0, 0, 1024, 1024), format: 'float32'}} : {})
    },
    outputs: {
      keys: {format: 'uint32', length: count},
      sortedRows: {format: 'uint32', length: count},
      sortedKeys: {format: 'uint32', length: count}
    },
    create: ({inputs, outputs}) =>
      new GPUHilbertKeys({
        order,
        points: inputs['points'] as never,
        bounds: bounds ? (inputs['bounds'] as never) : undefined,
        output: {
          keys: outputs['keys'] as never,
          sortedRows: outputs['sortedRows'] as never,
          sortedKeys: outputs['sortedKeys'] as never
        }
      })
  });
}

for (const order of [4, 10, 16]) {
  it(`GPUHilbertKeys order ${order} is bit-exact against the CPU xy2d and sorts rows`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) return;
    const random = createRandom(7 + order);
    const count = 3000;
    // Eighths inside a 1024 square: every f32 operation of the quantization is exact.
    const values = new Float32Array(count * 2);
    for (let i = 0; i < count * 2; i++) values[i] = Math.floor(random() * 8192) / 8;
    // A few invalid and out-of-bounds items.
    values[0] = Number.NaN;
    values[3] = Number.POSITIVE_INFINITY;
    values[10] = -50;
    values[11] = 2000;
    const fixture = createPointsFixture(device, values, order, true);
    const result = await fixture.run();
    const invalidRows = new Set<number>([0, 1]);
    for (let row = 0; row < count; row++) {
      const x = values[2 * row];
      const y = values[2 * row + 1];
      const expected =
        !Number.isFinite(x) || !Number.isFinite(y)
          ? getGPUHilbertInvalidKey(order)
          : getHilbertIndexReference(
              order,
              getCellReference(order, x, 0, 1024),
              getCellReference(order, y, 0, 1024)
            );
      if (expected === getGPUHilbertInvalidKey(order)) invalidRows.add(row);
      expect(result['keys'][row], `row ${row}`).toBe(expected);
    }
    expect(
      Math.max(...result['keys'].filter(key => key !== getGPUHilbertInvalidKey(order)))
    ).toBeGreaterThan(2 ** (2 * order - 2));
    // The permutation is a stable ascending sort of the keys.
    const expectedOrder = Array.from({length: count}, (_, row) => row).sort(
      (a, b) => result['keys'][a] - result['keys'][b] || a - b
    );
    expect(result['sortedRows']).toEqual(expectedOrder);
    expect(result['sortedKeys']).toEqual(expectedOrder.map(row => result['keys'][row]));
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  });
}

it('GPUHilbertKeys visits a full grid as an edge-connected curve and follows per-frame bounds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const order = 4;
  const size = 2 ** order;
  const points = new Float32Array(size * size * 2);
  for (let cell = 0; cell < size * size; cell++) {
    // Cell centers of a size x size grid over [0, 1024), shuffled by a stride permutation.
    const shuffled = (cell * 37) % (size * size);
    points[2 * cell] = ((shuffled % size) + 0.5) * (1024 / size);
    points[2 * cell + 1] = (Math.floor(shuffled / size) + 0.5) * (1024 / size);
  }
  const fixture = createPointsFixture(device, points, order, true);
  const result = await fixture.run();
  expect([...result['keys']].sort((a, b) => a - b)).toEqual(
    Array.from({length: size * size}, (_, key) => key)
  );
  const cells = result['sortedRows'].map(row => [
    Math.floor(points[2 * row] / (1024 / size)),
    Math.floor(points[2 * row + 1] / (1024 / size))
  ]);
  for (let key = 0; key < cells.length; key++) {
    expect(cells[key]).toEqual(getHilbertCellReference(order, key));
    if (key > 0) {
      const step =
        Math.abs(cells[key][0] - cells[key - 1][0]) + Math.abs(cells[key][1] - cells[key - 1][1]);
      expect(step).toBe(1);
    }
  }
  // Doubling the bounds per frame moves every point into the lower-left quadrant: no recompile.
  fixture.writeInput('bounds', Float32Array.of(0, 0, 2048, 2048));
  const wider = await fixture.run();
  expect(Math.max(...wider['keys'])).toBeLessThan((size * size) / 4);
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPUHilbertKeys keys feature-box centers and reduces bounds itself when none are given', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const random = createRandom(31);
  const count = 700;
  const minima = new Float32Array(count * 2);
  const maxima = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    const x = Math.floor(random() * 4000) / 4;
    const y = Math.floor(random() * 4000) / 4;
    minima.set([x, y], 2 * row);
    maxima.set([x + 2, y + 6], 2 * row);
  }
  minima.set([5, 5], 4);
  maxima.set([1, 1], 4); // Empty box: invalid.
  const fixture = createGeometryFixture(device, {
    inputs: {
      minima: {values: minima, format: 'float32x2'},
      maxima: {values: maxima, format: 'float32x2'}
    },
    outputs: {keys: {format: 'uint32', length: count}},
    create: ({inputs, outputs}) =>
      new GPUHilbertKeys({
        order: 12,
        minima: inputs['minima'] as never,
        maxima: inputs['maxima'] as never,
        output: {keys: outputs['keys'] as never}
      })
  });
  const result = await fixture.run();
  const centers: number[][] = [];
  for (let row = 0; row < count; row++) {
    centers.push([
      (minima[2 * row] + maxima[2 * row]) / 2,
      (minima[2 * row + 1] + maxima[2 * row + 1]) / 2
    ]);
  }
  const valid = centers.filter((_, row) => row !== 2);
  const boundsMinimum = [Math.min(...valid.map(c => c[0])), Math.min(...valid.map(c => c[1]))];
  const boundsMaximum = [Math.max(...valid.map(c => c[0])), Math.max(...valid.map(c => c[1]))];
  expect(result['keys'][2]).toBe(getGPUHilbertInvalidKey(12));
  let mismatches = 0;
  for (let row = 0; row < count; row++) {
    if (row === 2) continue;
    const expected = getHilbertIndexReference(
      12,
      getCellReference(12, centers[row][0], boundsMinimum[0], boundsMaximum[0]),
      getCellReference(12, centers[row][1], boundsMinimum[1], boundsMaximum[1])
    );
    // Non power-of-two extents leave f32 rounding at cell borders; allow a handful.
    if (result['keys'][row] !== expected) mismatches++;
  }
  expect(mismatches).toBeLessThan(count * 0.02);
  expect(new Set(result['keys']).size).toBeGreaterThan(count / 2);
  fixture.destroy();
});

it('GPUHilbertKeys reduces bounds across workgroups for large inputs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const random = createRandom(53);
  // More than 4096 items per reduction workgroup, so the two-level reduction runs. The extremes
  // (a power-of-two extent keeps the quantization exact) sit in the last blocks, and invalid
  // items must not affect the bounds.
  const count = 30000;
  const values = new Float32Array(count * 2);
  for (let i = 0; i < count * 2; i++) values[i] = 16 + Math.floor(random() * 7936) / 8;
  values.set([16, 1040], 2 * (count - 20));
  values.set([1040, 16], 2 * (count - 10));
  values.set([Number.NaN, 2000], 2 * (count - 5));
  values.set([Number.POSITIVE_INFINITY, -9], 2 * 3);
  const order = 13;
  const fixture = createGeometryFixture(device, {
    inputs: {points: {values, format: 'float32x2'}},
    outputs: {keys: {format: 'uint32', length: count}},
    create: ({inputs, outputs}) =>
      new GPUHilbertKeys({
        order,
        points: inputs['points'] as never,
        output: {keys: outputs['keys'] as never}
      })
  });
  const result = await fixture.run();
  // Valid x spans [16, 1040] and y spans [16, 1040] (extent 1024).
  let mismatches = 0;
  for (let row = 0; row < count; row++) {
    const x = values[2 * row];
    const y = values[2 * row + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      expect(result['keys'][row], `row ${row}`).toBe(getGPUHilbertInvalidKey(order));
      continue;
    }
    const expected = getHilbertIndexReference(
      order,
      getCellReference(order, x, 16, 1040),
      getCellReference(order, y, 16, 1040)
    );
    if (result['keys'][row] !== expected) mismatches++;
  }
  expect(mismatches).toBe(0);
  fixture.destroy();
});
