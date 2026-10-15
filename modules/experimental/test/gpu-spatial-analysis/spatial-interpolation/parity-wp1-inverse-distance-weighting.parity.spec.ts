// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUInverseDistanceWeighting,
  getGPUInverseDistanceWeightingParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-interpolation';
import fixtureJson from '../../oracles/fixtures/parity-wp1-inverse-distance-weighting.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {makeExtentGrid} from './extent-grid';

type InterpolationDataset = {
  positions: number[];
  values: number[];
  mask: number[];
  indexBounds: [number, number, number, number];
};
type InterpolationCase = {
  name: string;
  reference: string;
  n: number;
  dataset: string;
  width: number;
  height: number;
  extent: [number, number, number, number];
  searchRadius: number;
  power: number;
  neighborCount: number;
  minimumNeighborCount: number;
  rowDirection: 'north' | 'south';
  expected: Record<string, number[]>;
};
const fixture = decodeOracleFixture<{
  datasets: Record<string, InterpolationDataset>;
  cases: InterpolationCase[];
}>(fixtureJson);

function compareValues(
  testCase: InterpolationCase,
  metric: string,
  actual: number[],
  exact: boolean
): void {
  const expected = testCase.expected[metric];
  const scale = Math.max(1, ...expected.filter(Number.isFinite).map(Math.abs));
  const absolute = exact ? 0 : 1e-6 * scale;
  const relative = exact ? 0 : 1e-4;
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  expected.forEach((value, row) => {
    if (!Number.isFinite(value) || !Number.isFinite(actual[row])) {
      if (!Object.is(actual[row], value)) mismatches++;
      return;
    }
    const error = Math.abs(actual[row] - value);
    maxAbs = Math.max(maxAbs, error);
    maxRel = Math.max(maxRel, error / Math.max(Math.abs(value), Number.MIN_VALUE));
    if (error > absolute + relative * Math.abs(value)) mismatches++;
  });
  console.log(
    'PARITY',
    JSON.stringify({
      class: 'GPUInverseDistanceWeighting',
      case: testCase.name,
      reference: testCase.reference,
      n: testCase.n,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance: {profile: exact ? 'E' : 'L', absolute, relative}
    })
  );
  expect.soft(mismatches, `${testCase.name}: ${metric}`).toBe(0);
}

for (const testCase of fixture.cases) {
  it(`GPUInverseDistanceWeighting native GDAL parity: ${testCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new GraphRig(device, 'parity-wp1-idw');
    try {
      const dataset = fixture.datasets[testCase.dataset];
      const grid = makeExtentGrid(
        testCase.width,
        testCase.height,
        testCase.extent,
        testCase.rowDirection
      );
      const values = rig.output('float32', testCase.n);
      const counts = rig.output('uint32', testCase.n);
      rig.run(
        new GPUInverseDistanceWeighting({
          positions: rig.input(
            Float32Array.from(dataset.positions),
            'float32x2',
            dataset.values.length
          ),
          values: rig.input(Float32Array.from(dataset.values), 'float32'),
          mask: rig.input(Uint32Array.from(dataset.mask), 'uint32'),
          grid,
          indexBounds: dataset.indexBounds,
          indexGridSize: testCase.dataset === 'ghcn' ? [64, 64] : [4, 4],
          maximumNeighborCount: testCase.neighborCount,
          parameters: rig.input(
            getGPUInverseDistanceWeightingParameterValues(
              {
                searchRadius: testCase.searchRadius,
                power: testCase.power,
                neighborCount: testCase.neighborCount,
                minimumNeighborCount: testCase.minimumNeighborCount
              },
              grid
            ),
            'float32'
          ),
          output: {values: values.view, counts: counts.view}
        })
      );
      compareValues(testCase, 'values', await values.readFloat32(), false);
      compareValues(testCase, 'counts', await counts.readUint32(), true);
    } finally {
      rig.destroy();
    }
  });
}
