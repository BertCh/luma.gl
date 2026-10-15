// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPURipley,
  GPURipleyDistanceFunctions,
  GPUPointPatternIndices,
  getGPURipleyParameterValues,
  getGPURipleyDistanceFunctionsParameterValues,
  getGPUPointPatternIndicesParameterValues,
  type GPURipleyEdgeCorrection,
  type GPURipleyDistanceEdgeCorrection,
  type GPUClarkEvansCorrection
} from '../../../src/gpu-dataframe/pair-statistics';
import {
  createPairStatisticsHarness,
  createPairStatisticsWindowHarness,
  type HarnessOutput,
  type HarnessViews
} from './pair-statistics-harness';
import type {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import indexJson from '../../oracles/fixtures/parity-wp6-round8-indices.json';
import ripleyJson from '../../oracles/fixtures/parity-wp6-round8-ripley.json';
import rectangleJson from '../../oracles/fixtures/parity-wp6-round8-distance-rectangle.json';
import cityJson from '../../oracles/fixtures/parity-wp6-round8-distance-city.json';
import viewJson from '../../oracles/fixtures/parity-wp6-round8-distance-view.json';

type Scene = {
  name: string;
  positions: number[];
  bounds: [number, number, number, number];
  window: null | {positions: number[]; ringOffsets: number[]; polygonOffsets: number[]};
  maximumDistance: number;
  radiusCount: number;
  referenceGrid: [number, number];
  quadratGrid: [number, number];
  guardDistance: number;
  alternative: 'two-sided' | 'greater' | 'lesser';
  correction: string;
  expected: Record<string, number[]>;
};
const groups = [
  ['GPUPointPatternIndices', indexJson],
  ['GPURipley', ripleyJson],
  ['GPURipleyDistanceFunctions', rectangleJson],
  ['GPURipleyDistanceFunctions', cityJson],
  ['GPURipleyDistanceFunctions', viewJson]
] as const;

for (const [className, json] of groups) {
  const fixture = decodeOracleFixture<{reference: string; cases: Scene[]}>(json);
  for (const scene of fixture.cases) {
    for (const masked of className === 'GPUPointPatternIndices' ? [true] : [false, true]) {
      it(`${className} corrected independent showcase parity: ${scene.name}/masked=${masked}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const exact = new Set(['nearestNeighborIds', 'quadratCounts', 'pairCounts']);
        const outputs: Record<string, HarnessOutput> = Object.fromEntries(
          Object.entries(scene.expected).map(([name, values]) => [
            name,
            {format: exact.has(name) ? 'uint32' : 'float32', length: values.length}
          ])
        );
        const createContributor = (
          views: HarnessViews,
          studyWindow?: GPUStudyWindowPreparation
        ) => {
          const float = (name: string) => views.outputs[name] as GraphDataView<'float32'>;
          const word = (name: string) => views.outputs[name] as GraphDataView<'uint32'>;
          const common = {
            mask: views.mask,
            positions: views.positions,
            parameters: views.parameters,
            studyWindow,
            gridSize: [96, 96] as const
          };
          if (className === 'GPURipley')
            return new GPURipley({
              ...common,
              radiusCount: scene.radiusCount,
              output: {
                k: float('k'),
                l: float('l'),
                lMinusR: float('lMinusR'),
                pairCorrelation: float('pairCorrelation'),
                radii: float('radii'),
                pairCounts: word('pairCounts')
              }
            });
          if (className === 'GPURipleyDistanceFunctions')
            return new GPURipleyDistanceFunctions({
              ...common,
              radiusCount: scene.radiusCount,
              referenceGrid: scene.referenceGrid,
              output: {g: float('g'), f: float('f'), j: float('j'), radii: float('radii')}
            });
          return new GPUPointPatternIndices({
            ...common,
            quadratGrid: scene.quadratGrid,
            output: {
              nearestNeighborDistances: float('nearestNeighborDistances'),
              nearestNeighborIds: word('nearestNeighborIds'),
              quadratCounts: word('quadratCounts'),
              quadratAreas: float('quadratAreas'),
              clarkEvans: float('clarkEvans'),
              quadratStatistics: float('quadratStatistics')
            }
          });
        };
        const props = {
          positions: Float32Array.from(scene.positions),
          mask: masked ? new Uint32Array(scene.positions.length / 2).fill(1) : undefined,
          parameterLength: 8,
          outputs,
          createContributor
        };
        const harness = scene.window
          ? createPairStatisticsWindowHarness(device, {
              ...props,
              window: {
                positions: Float64Array.from(scene.window.positions),
                ringOffsets: Uint32Array.from(scene.window.ringOffsets),
                polygonOffsets: Uint32Array.from(scene.window.polygonOffsets)
              }
            })
          : createPairStatisticsHarness(device, props);
        try {
          const common = {
            bounds: scene.window ? ('study-window' as const) : scene.bounds,
            maximumDistance: scene.maximumDistance
          };
          const parameters =
            className === 'GPURipley'
              ? getGPURipleyParameterValues({
                  ...common,
                  edgeCorrection: scene.correction as GPURipleyEdgeCorrection
                })
              : className === 'GPURipleyDistanceFunctions'
                ? getGPURipleyDistanceFunctionsParameterValues({
                    ...common,
                    edgeCorrection: scene.correction as GPURipleyDistanceEdgeCorrection
                  })
                : getGPUPointPatternIndicesParameterValues({
                    ...common,
                    clarkEvansCorrection: scene.correction as GPUClarkEvansCorrection,
                    guardDistance: scene.guardDistance,
                    alternative: scene.alternative
                  });
          const actual = await harness.run(parameters);
          if (className === 'GPUPointPatternIndices') {
            compareParity(
              className,
              scene.name,
              fixture.reference,
              scene.positions.length / 2,
              'summary-counts',
              [
                actual['clarkEvans'][0],
                actual['quadratStatistics'][0],
                actual['quadratStatistics'][5]
              ],
              [
                scene.expected['clarkEvans'][0],
                scene.expected['quadratStatistics'][0],
                scene.expected['quadratStatistics'][5]
              ],
              true
            );
          }
          for (const [metric, expected] of Object.entries(scene.expected)) {
            compareParity(
              className,
              `${scene.name}/masked=${masked}`,
              fixture.reference,
              scene.positions.length / 2,
              metric,
              actual[metric],
              expected,
              exact.has(metric)
            );
            if (className === 'GPURipleyDistanceFunctions' && ['g', 'f', 'j'].includes(metric)) {
              // Empty/full probability mass is exact. In particular, G = 1 implies J = 0
              // while F < 1; a rounded reciprocal must not manufacture a surviving tail.
              const endpoints = expected.flatMap((value, index) =>
                value === 0 || (metric !== 'j' && value === 1) ? [index] : []
              );
              compareParity(
                className,
                `${scene.name}/masked=${masked}`,
                fixture.reference,
                scene.positions.length / 2,
                `${metric}-exact-endpoints`,
                endpoints.map(index => actual[metric][index]),
                endpoints.map(index => expected[index]),
                true
              );
            }
          }
        } finally {
          harness.destroy();
        }
      }, 120000);
    }
  }
}
