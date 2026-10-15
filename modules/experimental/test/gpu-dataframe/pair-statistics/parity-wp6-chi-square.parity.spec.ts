// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUPointPatternIndices} from '../../../src/gpu-dataframe/pair-statistics/gpu-point-pattern-indices';
import {getGPUPointPatternIndicesParameterValues} from '../../../src/gpu-dataframe/pair-statistics/point-pattern-indices-parameters';
import {createPairStatisticsHarness} from './pair-statistics-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareParity} from '../../oracles/parity-wp6-compare';
import fixture from '../../oracles/fixtures/parity-wp6-chi-square.json';

for (const scene of fixture.cases) {
  it(`GPUPointPatternIndices SciPy gamma regression: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const [columns, rows] = scene.quadratGrid;
    const positions: number[] = [];
    for (let index = 0; index < scene.counts.length; index++) {
      for (let event = 0; event < scene.counts[index]; event++) {
        positions.push((index % columns) + 0.5, Math.floor(index / columns) + 0.5);
      }
    }
    const harness = createPairStatisticsHarness(device, {
      positions: new Float32Array(positions),
      parameterLength: 8,
      outputs: {
        counts: {format: 'uint32', length: scene.counts.length},
        statistics: {format: 'float32', length: 7}
      },
      createContributor: views =>
        new GPUPointPatternIndices({
          positions: views.positions,
          parameters: views.parameters,
          gridSize: [16, 16],
          quadratGrid: [columns, rows],
          output: {
            quadratCounts: views.outputs.counts as GraphDataView<'uint32'>,
            quadratStatistics: views.outputs.statistics as GraphDataView<'float32'>
          }
        })
    });
    try {
      for (const alternative of ['two-sided', 'lesser', 'greater'] as const) {
        const result = await harness.run(
          getGPUPointPatternIndicesParameterValues({
            bounds: [0, 0, columns, rows],
            maximumDistance: 1,
            clarkEvansCorrection: 'none',
            alternative
          })
        );
        compareParity(
          'GPUPointPatternIndices',
          `${scene.name}/${alternative}`,
          fixture.reference,
          positions.length / 2,
          'quadratCounts',
          result.counts,
          scene.counts,
          true
        );
        compareParity(
          'GPUPointPatternIndices',
          `${scene.name}/${alternative}`,
          fixture.reference,
          positions.length / 2,
          'quadratStatistics',
          result.statistics,
          [...scene.statistics, scene.tails[alternative]]
        );
        compareParity(
          'GPUPointPatternIndices',
          `${scene.name}/${alternative}`,
          fixture.reference,
          positions.length / 2,
          'quadratCount/degreesOfFreedom',
          [result.statistics[0], result.statistics[5]],
          [scene.counts.length, scene.counts.length - 1],
          true
        );
      }
    } finally {
      harness.destroy();
    }
  }, 120000);
}
