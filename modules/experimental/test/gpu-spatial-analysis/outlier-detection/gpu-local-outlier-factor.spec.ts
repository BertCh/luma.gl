// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPULocalOutlierFactor,
  getGPULocalOutlierFactorParameterValues
} from '../../../src/gpu-spatial-analysis/outlier-detection/index';
import {GPUNeighborSearch} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {getGPUNeighborSearchParameterValues} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {
  DUPLICATES_FIXTURE,
  RANDOM_FIXTURE,
  type LocalOutlierFactorFixture
} from './outlier-detection-oracle';

type Device = NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;

function expectClose(actual: number[], expected: number[], label: string, relative: number): void {
  expect(actual.length, `${label} length`).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Math.abs(actual[index] - expected[index]) > 1e-6 + relative * Math.abs(expected[index])) {
      throw new Error(`${label}: [${index}] ${actual[index]} != ${expected[index]}`);
    }
  }
}

async function runLof(
  device: Device,
  fixture: LocalOutlierFactorFixture,
  threshold: number,
  useNeighborSearch: boolean
) {
  const rows = fixture.points.length / 2;
  const rig = new WeightsRig(device);
  const parameters = rig.input(getGPULocalOutlierFactorParameterValues({threshold}), 'float32', 4);
  const kDistance = rig.output('float32', rows);
  const density = rig.output('float32', rows);
  const lof = rig.output('float32', rows);
  const outlier = rig.output('uint32', rows);
  const outlierCount = rig.output('uint32', 1);
  const producers = [];
  let neighbors;
  if (useNeighborSearch) {
    const table = rig.weightsOutput(rows, rows * fixture.k, true);
    const overflow = rig.output('uint32', 1);
    producers.push(
      new GPUNeighborSearch({
        mode: 'knn',
        k: fixture.k,
        positions: rig.input(new Float32Array(fixture.points), 'float32x2', rows),
        parameters: rig.input(
          getGPUNeighborSearchParameterValues({bounds: [-100, -100, 300, 300]}),
          'float32',
          12
        ),
        gridSize: [4, 4],
        weights: table.spatialWeights,
        overflow: overflow.view
      })
    );
    neighbors = table.spatialWeights;
  } else {
    neighbors = rig.uploadWeights({
      offsets: fixture.offsets,
      neighbors: fixture.neighbors,
      weights: fixture.neighbors.map(() => 1),
      distances: fixture.distances
    });
  }
  producers.push(
    new GPULocalOutlierFactor({
      neighbors,
      parameters,
      kDistance: kDistance.view,
      localReachabilityDensity: density.view,
      lof: lof.view,
      outlier: outlier.view,
      outlierCount: outlierCount.view
    })
  );
  rig.run(...producers);
  const result = {
    kDistance: await readFloat32(kDistance.buffer, rows),
    lof: await readFloat32(lof.buffer, rows),
    outlier: await readUint32(outlier.buffer, rows),
    count: (await readUint32(outlierCount.buffer, 1))[0]
  };
  rig.destroy();
  return result;
}

it('GPULocalOutlierFactor matches scikit-learn on tie-free random points', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const useNeighborSearch of [false, true]) {
    const label = useNeighborSearch ? 'neighbor-search' : 'uploaded';
    const start = performance.now();
    const result = await runLof(device, RANDOM_FIXTURE, 1.5, useNeighborSearch);
    console.log(`lof ${label}: ${(performance.now() - start).toFixed(1)} ms`);
    expectClose(result.kDistance, RANDOM_FIXTURE.kDistance, `${label} kDistance`, 2e-6);
    expectClose(result.lof, RANDOM_FIXTURE.lof, `${label} lof`, 1e-4);
    const expectedMask = RANDOM_FIXTURE.lof.map(value => (value > 1.5 ? 1 : 0));
    expect(expectedMask.some(flag => flag === 1)).toBe(true);
    expect(result.outlier, `${label} mask`).toEqual(expectedMask);
    expect(result.count).toBe(expectedMask.reduce((sum, flag) => sum + flag, 0));
  }
});

it('GPULocalOutlierFactor keeps coincident points finite through the density floor', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runLof(device, DUPLICATES_FIXTURE, 2, false);
  expectClose(result.lof, DUPLICATES_FIXTURE.lof, 'duplicates lof', 1e-3);
  for (let row = 0; row < 4; row++) {
    expect(result.lof[row]).toBeCloseTo(1, 5);
  }
  expect(result.lof.every(Number.isFinite)).toBe(true);
});
