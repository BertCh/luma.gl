// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPURipley} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley';
import {GPURipleyDistanceFunctions} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions';
import {getGPURipleyParameterValues} from '../../../src/gpu-dataframe/pair-statistics/ripley-parameters';
import {getGPURipleyDistanceFunctionsParameterValues} from '../../../src/gpu-dataframe/pair-statistics/ripley-distance-parameters';
import {createPairStatisticsHarness} from './pair-statistics-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import ripleyJSON from '../../oracles/fixtures/parity-wp6-ripley.json';
import distanceJSON from '../../oracles/fixtures/parity-wp6-ripley-distance.json';
import positionsJSON from '../../oracles/fixtures/parity-wp6-point-patterns-positions.json';

type Scene<Column extends string> = {
  name: string;
  positions: number[];
  fullPositionsPayload: boolean;
  mask: number[];
  bounds: [number, number, number, number];
  maximumDistance: number;
  radiusCount: number;
  referenceGrid: [number, number];
  sourceCount: number;
  expected: Record<Column, number[]>;
};
type RipleyColumn = 'k' | 'l' | 'lMinusR' | 'pairCorrelation' | 'pairCounts' | 'radii';
type DistanceColumn = 'f' | 'g' | 'j' | 'radii';
const ripley = decodeOracleFixture<{cases: Scene<RipleyColumn>[]}>(ripleyJSON);
const distance = decodeOracleFixture<{cases: Scene<DistanceColumn>[]}>(distanceJSON);

function getPositions(scene: Scene<string>): Float32Array {
  if (!scene.fullPositionsPayload) {
    return new Float32Array(scene.positions);
  }
  const bytes = Uint8Array.from(atob(positionsJSON.positions.base64), character =>
    character.charCodeAt(0)
  );
  const view = new DataView(bytes.buffer);
  return Float32Array.from({length: positionsJSON.positions.length}, (_, index) =>
    view.getFloat32(index * 4, true)
  );
}

for (const scene of ripley.cases) {
  it(`GPURipley pointpats parity (rectangle/none): ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createPairStatisticsHarness(device, {
      positions: getPositions(scene),
      mask: new Uint32Array(scene.mask),
      parameterLength: 8,
      outputs: {
        k: {format: 'float32', length: scene.radiusCount},
        l: {format: 'float32', length: scene.radiusCount},
        lMinusR: {format: 'float32', length: scene.radiusCount},
        pairCorrelation: {format: 'float32', length: scene.radiusCount},
        pairCounts: {format: 'uint32', length: scene.radiusCount},
        radii: {format: 'float32', length: scene.radiusCount}
      },
      createContributor: views =>
        new GPURipley({
          positions: views.positions,
          mask: views.mask,
          parameters: views.parameters,
          gridSize: [32, 32],
          radiusCount: scene.radiusCount,
          output: {
            k: views.outputs.k as GraphDataView<'float32'>,
            l: views.outputs.l as GraphDataView<'float32'>,
            lMinusR: views.outputs.lMinusR as GraphDataView<'float32'>,
            pairCorrelation: views.outputs.pairCorrelation as GraphDataView<'float32'>,
            pairCounts: views.outputs.pairCounts as GraphDataView<'uint32'>,
            radii: views.outputs.radii as GraphDataView<'float32'>
          }
        })
    });
    try {
      const result = await harness.run(
        getGPURipleyParameterValues({
          bounds: scene.bounds,
          maximumDistance: scene.maximumDistance,
          edgeCorrection: 'none'
        })
      );
      for (const metric of [
        'k',
        'l',
        'lMinusR',
        'pairCorrelation',
        'pairCounts',
        'radii'
      ] as const) {
        compareParity(
          'GPURipley',
          `${scene.name}/none`,
          'pointpats.k/l (explicit window and n/(n-1) normalization); scipy.pdist; derived annulus differences',
          scene.sourceCount,
          metric,
          result[metric],
          scene.expected[metric],
          metric === 'pairCounts' || metric === 'radii'
        );
      }
    } finally {
      harness.destroy();
    }
  }, 120000);
}

for (const scene of distance.cases) {
  it(`GPURipleyDistanceFunctions pointpats parity (rectangle/none): ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createPairStatisticsHarness(device, {
      positions: getPositions(scene),
      mask: scene.fullPositionsPayload ? undefined : new Uint32Array(scene.mask),
      parameterLength: 8,
      outputs: {
        f: {format: 'float32', length: scene.radiusCount},
        g: {format: 'float32', length: scene.radiusCount},
        j: {format: 'float32', length: scene.radiusCount},
        radii: {format: 'float32', length: scene.radiusCount}
      },
      createContributor: views =>
        new GPURipleyDistanceFunctions({
          positions: views.positions,
          mask: views.mask,
          parameters: views.parameters,
          gridSize: [32, 32],
          referenceGrid: scene.referenceGrid,
          radiusCount: scene.radiusCount,
          output: {
            f: views.outputs.f as GraphDataView<'float32'>,
            g: views.outputs.g as GraphDataView<'float32'>,
            j: views.outputs.j as GraphDataView<'float32'>,
            radii: views.outputs.radii as GraphDataView<'float32'>
          }
        })
    });
    try {
      const result = await harness.run(
        getGPURipleyDistanceFunctionsParameterValues({
          bounds: scene.bounds,
          maximumDistance: scene.maximumDistance,
          edgeCorrection: 'none'
        })
      );
      for (const metric of ['f', 'g', 'j', 'radii'] as const) {
        compareParity(
          'GPURipleyDistanceFunctions',
          `${scene.name}/none`,
          'pointpats.f/g/j; scipy.cKDTree; identical fixed cell-center F locations and full-support denominator',
          scene.sourceCount,
          metric,
          result[metric],
          scene.expected[metric],
          metric === 'radii'
        );
      }
    } finally {
      harness.destroy();
    }
  }, 120000);
}
