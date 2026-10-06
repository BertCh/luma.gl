// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPURasterSieveParameterValues,
  GPURasterPatchMetrics,
  GPURasterSieve
} from '../../../src/gpu-raster/raster-patches/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../../gpu-spatial-analysis/spatial-weights/spatial-weights-harness';
import {
  computePatchMetricsOracle,
  computeSieveOracle,
  createClumps,
  createRandom,
  createSegmentation,
  getAcceptedLabels
} from './raster-patches-oracle';

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

const AFFINE = [2, 0.5, 10, -0.25, -3, 20] as const;

type Guards = {
  validity?: Uint32Array;
  componentCount?: number;
  converged?: number;
  overflow?: number;
};

async function runMetrics(
  device: Device,
  labels: Uint32Array,
  width: number,
  height: number,
  capacity: number,
  countBorder: boolean,
  guards: Guards = {}
) {
  const rig = new WeightsRig(device);
  const outputs = {
    pixelCounts: rig.output('uint32', capacity),
    perimeterFaceCounts: rig.output('uint32', capacity),
    minColumns: rig.output('uint32', capacity),
    minRows: rig.output('uint32', capacity),
    maxColumns: rig.output('uint32', capacity),
    maxRows: rig.output('uint32', capacity),
    areas: rig.output('float32', capacity),
    perimeters: rig.output('float32', capacity)
  };
  rig.run(
    new GPURasterPatchMetrics({
      width,
      height,
      labels: rig.input(labels, 'uint32', labels.length),
      labelValidity: guards.validity && rig.input(guards.validity, 'uint32', labels.length),
      componentCount:
        guards.componentCount !== undefined
          ? rig.input(new Uint32Array([guards.componentCount]), 'uint32', 1)
          : undefined,
      converged:
        guards.converged !== undefined
          ? rig.input(new Uint32Array([guards.converged]), 'uint32', 1)
          : undefined,
      overflow:
        guards.overflow !== undefined
          ? rig.input(new Uint32Array([guards.overflow]), 'uint32', 1)
          : undefined,
      affine: AFFINE,
      countRasterBorder: countBorder,
      output: {
        pixelCounts: outputs.pixelCounts.view,
        perimeterFaceCounts: outputs.perimeterFaceCounts.view,
        minColumns: outputs.minColumns.view,
        minRows: outputs.minRows.view,
        maxColumns: outputs.maxColumns.view,
        maxRows: outputs.maxRows.view,
        areas: outputs.areas.view,
        perimeters: outputs.perimeters.view
      }
    })
  );
  const result = {
    pixelCounts: await readUint32(outputs.pixelCounts.buffer, capacity),
    faces: await readUint32(outputs.perimeterFaceCounts.buffer, capacity),
    minColumns: await readUint32(outputs.minColumns.buffer, capacity),
    minRows: await readUint32(outputs.minRows.buffer, capacity),
    maxColumns: await readUint32(outputs.maxColumns.buffer, capacity),
    maxRows: await readUint32(outputs.maxRows.buffer, capacity),
    areas: await readFloat32(outputs.areas.buffer, capacity),
    perimeters: await readFloat32(outputs.perimeters.buffer, capacity)
  };
  rig.destroy();
  return result;
}

function expectMetrics(
  result: Awaited<ReturnType<typeof runMetrics>>,
  accepted: Uint32Array,
  width: number,
  height: number,
  capacity: number,
  countBorder: boolean
): void {
  const expected = computePatchMetricsOracle(accepted, width, height, capacity, countBorder);
  const [a, b, , d, e] = AFFINE;
  const columnStep = Math.hypot(a, d);
  const rowStep = Math.hypot(b, e);
  const determinant = Math.abs(a * e - b * d);
  expect(result.pixelCounts).toEqual(expected.counts);
  expect(result.minColumns).toEqual(expected.minColumns);
  expect(result.minRows).toEqual(expected.minRows);
  expect(result.maxColumns).toEqual(expected.maxColumns);
  expect(result.maxRows).toEqual(expected.maxRows);
  expect(result.faces).toEqual(
    expected.rowFaces.map((faces, index) => faces + expected.columnFaces[index])
  );
  for (let patch = 0; patch < capacity; patch++) {
    const expectedPerimeter =
      expected.rowFaces[patch] * columnStep + expected.columnFaces[patch] * rowStep;
    expect(Math.abs(result.perimeters[patch] - expectedPerimeter)).toBeLessThanOrEqual(
      1e-4 * Math.max(1, expectedPerimeter)
    );
    expect(
      Math.abs(result.areas[patch] - expected.counts[patch] * determinant)
    ).toBeLessThanOrEqual(1e-4 * Math.max(1, expected.counts[patch] * determinant));
  }
}

it('GPURasterPatchMetrics matches the oracle on clumps, with and without raster border', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 23;
  const height = 17;
  const labels = createClumps(width, height, 0.5, 11);
  const capacity = Math.max(...labels) + 3;
  expect(Math.max(...labels)).toBeGreaterThan(5);
  for (const countBorder of [true, false]) {
    const result = await runMetrics(device, labels, width, height, capacity, countBorder);
    expectMetrics(
      result,
      getAcceptedLabels(labels, {capacity}),
      width,
      height,
      capacity,
      countBorder
    );
    // Nonzero assertions guard against silent compile failures.
    expect(result.pixelCounts.reduce((sum, count) => sum + count, 0)).toBeGreaterThan(50);
    expect(result.perimeters.some(value => value > 0)).toBe(true);
    // Rows past the last label are empty with zeroed extents.
    expect(result.pixelCounts.at(-1)).toBe(0);
    expect(result.maxColumns.at(-1)).toBe(0);
  }
});

it('GPURasterPatchMetrics honors validity, capacity and upstream guards', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 20;
  const height = 14;
  const labels = createClumps(width, height, 0.55, 5);
  const random = createRandom(9);
  const validity = Uint32Array.from(labels, () => (random() < 0.9 ? 1 : 0));
  const maximum = Math.max(...labels);
  const capacity = Math.floor(maximum / 2);
  const componentCount = capacity - 1;
  const accepted = getAcceptedLabels(labels, {validity, capacity, componentCount});
  const result = await runMetrics(device, labels, width, height, capacity, true, {
    validity,
    componentCount,
    converged: 1,
    overflow: 0
  });
  expectMetrics(result, accepted, width, height, capacity, true);
  expect(result.pixelCounts.at(-1)).toBe(0);
  expect(result.pixelCounts.some(count => count > 0)).toBe(true);
  for (const guards of [
    {converged: 0, overflow: 0},
    {converged: 1, overflow: 1}
  ]) {
    const blocked = await runMetrics(device, labels, width, height, capacity, true, {
      componentCount,
      ...guards
    });
    expect(blocked.pixelCounts.every(count => count === 0)).toBe(true);
    expect(blocked.perimeters.every(value => value === 0)).toBe(true);
  }
});

async function runSieve(
  device: Device,
  labels: Uint32Array,
  width: number,
  height: number,
  capacity: number,
  options: {
    minimumPixels: number;
    mode: 'remove' | 'merge';
    connectivity: 4 | 8;
    validity?: Uint32Array;
  }
) {
  const rig = new WeightsRig(device);
  const output = rig.output('uint32', labels.length);
  const patchTargets = rig.output('uint32', capacity);
  const sievedCount = rig.output('uint32', 1);
  const parameters = new GPUParameterBuffer(device, {
    id: 'sieve-parameters',
    format: 'uint32',
    length: 4,
    values: getGPURasterSieveParameterValues({minimumPixels: options.minimumPixels})
  });
  rig.run(
    new GPURasterSieve({
      width,
      height,
      labels: rig.input(labels, 'uint32', labels.length),
      labelValidity: options.validity && rig.input(options.validity, 'uint32', labels.length),
      patchCapacity: capacity,
      parameters: parameters.importToGraph(rig.graph),
      mode: options.mode,
      connectivity: options.connectivity,
      output: {labels: output.view, patchTargets: patchTargets.view, sievedCount: sievedCount.view}
    })
  );
  const result = {
    labels: await readUint32(output.buffer, labels.length),
    patchTargets: await readUint32(patchTargets.buffer, capacity),
    sievedCount: (await readUint32(sievedCount.buffer, 1))[0]
  };
  parameters.destroy();
  rig.destroy();
  return result;
}

it('GPURasterSieve removes small clumps like the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 26;
  const height = 18;
  const labels = createClumps(width, height, 0.45, 21);
  const capacity = Math.max(...labels);
  const accepted = getAcceptedLabels(labels, {capacity});
  for (const minimumPixels of [1, 3, 8]) {
    const result = await runSieve(device, labels, width, height, capacity, {
      minimumPixels,
      mode: 'remove',
      connectivity: 4
    });
    const expected = computeSieveOracle(
      accepted,
      width,
      height,
      capacity,
      minimumPixels,
      'remove',
      4
    );
    expect(result.labels).toEqual(Array.from(expected.labels));
    expect(result.patchTargets).toEqual(expected.patchTargets);
    expect(result.sievedCount).toBe(expected.sievedCount);
    if (minimumPixels === 1) expect(result.sievedCount).toBe(0);
    if (minimumPixels === 8) {
      expect(result.sievedCount).toBeGreaterThan(0);
      expect(result.labels.some(label => label !== 0)).toBe(true);
    }
  }
});

it('GPURasterSieve merges small segments into their largest large neighbor', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 30;
  const height = 20;
  const labels = createSegmentation(width, height, 24, 3);
  const capacity = 24;
  const random = createRandom(4);
  const validity = Uint32Array.from(labels, () => (random() < 0.97 ? 1 : 0));
  const accepted = getAcceptedLabels(labels, {capacity, validity});
  for (const connectivity of [4, 8] as const) {
    for (const minimumPixels of [15, 40]) {
      const result = await runSieve(device, labels, width, height, capacity, {
        minimumPixels,
        mode: 'merge',
        connectivity,
        validity
      });
      const expected = computeSieveOracle(
        accepted,
        width,
        height,
        capacity,
        minimumPixels,
        'merge',
        connectivity
      );
      expect(result.labels, `conn ${connectivity} min ${minimumPixels}`).toEqual(
        Array.from(expected.labels)
      );
      expect(result.patchTargets).toEqual(expected.patchTargets);
      expect(result.sievedCount).toBe(expected.sievedCount);
      // The test only means something when some patch merged into another label.
      const merged = expected.patchTargets.filter(
        (target, index) => target !== 0 && target !== index + 1
      );
      expect(merged.length).toBeGreaterThan(0);
      // Merging never removes pixels that had a large neighbor: surviving labels are all large.
      const survivors = new Set(result.labels.filter(label => label !== 0));
      for (const label of survivors) {
        expect(expected.counts[label - 1]).toBeGreaterThanOrEqual(minimumPixels);
      }
    }
  }
});
