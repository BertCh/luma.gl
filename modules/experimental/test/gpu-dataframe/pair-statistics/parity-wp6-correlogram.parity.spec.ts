// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUSpatialCorrelogram} from '../../../src/gpu-dataframe/pair-statistics/gpu-spatial-correlogram';
import {getGPUSpatialCorrelogramParameterValues} from '../../../src/gpu-dataframe/pair-statistics/spatial-correlogram-parameters';
import {createPairStatisticsHarness} from './pair-statistics-harness';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import fixtureJSON from '../../oracles/fixtures/parity-wp6-correlogram.json';

type Column =
  | 'moransI'
  | 'expectedI'
  | 'varianceI'
  | 'zScores'
  | 'pValues'
  | 'pairCounts'
  | 'peakBands'
  | 'statistics'
  | 'overflow';
const fixture = decodeOracleFixture<{
  cases: {
    name: string;
    positions: number[];
    values: number[];
    mask: number[];
    bounds: [number, number, number, number];
    maximumDistance: number;
    bandCount: number;
    runs: {
      bandMode: 'annulus' | 'cumulative';
      varianceAssumption: 'normality' | 'randomization';
      expected: Record<Column, number[]>;
    }[];
  }[];
}>(fixtureJSON);

for (const scene of fixture.cases) {
  for (const bandMode of ['annulus', 'cumulative'] as const) {
    it(`GPUSpatialCorrelogram esda parity: ${scene.name}/${bandMode}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createPairStatisticsHarness(device, {
        positions: new Float32Array(scene.positions),
        values: new Float32Array(scene.values),
        mask: new Uint32Array(scene.mask),
        parameterLength: 8,
        outputs: {
          moransI: {format: 'float32', length: scene.bandCount},
          expectedI: {format: 'float32', length: scene.bandCount},
          varianceI: {format: 'float32', length: scene.bandCount},
          zScores: {format: 'float32', length: scene.bandCount},
          pValues: {format: 'float32', length: scene.bandCount},
          pairCounts: {format: 'uint32', length: scene.bandCount},
          peakBands: {format: 'uint32', length: 2},
          statistics: {format: 'float32', length: 5},
          overflow: {format: 'uint32', length: 1}
        },
        createContributor: views =>
          new GPUSpatialCorrelogram({
            positions: views.positions,
            values: views.values!,
            mask: views.mask,
            parameters: views.parameters,
            gridSize: [32, 32],
            bandCount: scene.bandCount,
            bandMode,
            output: {
              moransI: views.outputs.moransI as GraphDataView<'float32'>,
              expectedI: views.outputs.expectedI as GraphDataView<'float32'>,
              varianceI: views.outputs.varianceI as GraphDataView<'float32'>,
              zScores: views.outputs.zScores as GraphDataView<'float32'>,
              pValues: views.outputs.pValues as GraphDataView<'float32'>,
              pairCounts: views.outputs.pairCounts as GraphDataView<'uint32'>,
              peakBands: views.outputs.peakBands as GraphDataView<'uint32'>,
              statistics: views.outputs.statistics as GraphDataView<'float32'>,
              status: {overflow: views.outputs.overflow as GraphDataView<'uint32'>}
            }
          })
      });
      try {
        for (const run of scene.runs.filter(candidate => candidate.bandMode === bandMode)) {
          const result = await harness.run(
            getGPUSpatialCorrelogramParameterValues({
              bounds: scene.bounds,
              maximumDistance: scene.maximumDistance,
              varianceAssumption: run.varianceAssumption
            })
          );
          const caseName = `${scene.name}/${bandMode}/${run.varianceAssumption}`;
          for (const metric of [
            'moransI',
            'expectedI',
            'varianceI',
            'zScores',
            'pValues',
            'pairCounts',
            'peakBands',
            'statistics',
            'overflow'
          ] as const) {
            compareParity(
              'GPUSpatialCorrelogram',
              caseName,
              'esda.Moran(transformation=B, permutations=0); scipy.pdist/kurtosis/find_peaks',
              scene.values.length,
              metric,
              result[metric],
              run.expected[metric],
              ['pairCounts', 'peakBands', 'overflow'].includes(metric)
            );
          }
          compareParity(
            'GPUSpatialCorrelogram',
            caseName,
            'finite in-window unmasked row count',
            scene.values.length,
            'statistics.count',
            [result.statistics[0]],
            [run.expected.statistics[0]],
            true
          );
        }
      } finally {
        harness.destroy();
      }
    }, 120000);
  }
}
