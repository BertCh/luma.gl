// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import type {GPUGeographicallyWeightedRegressionProps} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters';
import {getStudentTWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-student-t';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

const ROWS = 40;
let serial = 0;

type Configuration = {
  name: string;
  predictorCount: number;
  props: Partial<GPUGeographicallyWeightedRegressionProps>;
  allOutputs: boolean;
};

function getKernelSources(configuration: Configuration): {id: string; source: string}[] {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `wgsl-${serial++}`, format, length);
  const {predictorCount} = configuration;
  const P = predictorCount + 1;
  const ladder = configuration.props.maximumBandwidthCount ?? 32;
  const passes =
    1 +
    (configuration.props.refinement === false
      ? 0
      : typeof configuration.props.refinement === 'number'
        ? configuration.props.refinement
        : 2);
  const output: GPUGeographicallyWeightedRegressionProps['output'] = {
    coefficients: view('float32', ROWS * P)
  };
  if (configuration.allOutputs) {
    Object.assign(output, {
      standardErrors: view('float32', ROWS * P),
      tValues: view('float32', ROWS * P),
      significant: view('uint32', ROWS * P),
      localR2: view('float32', ROWS),
      fitted: view('float32', ROWS),
      residuals: view('float32', ROWS),
      hatDiagonal: view('float32', ROWS),
      localStatus: view('uint32', ROWS),
      localConditionNumber: view('float32', ROWS),
      bandwidthScores: view('float32', ladder),
      bandwidthSingularRowCounts: view('uint32', ladder),
      candidateTable: view(
        'float32',
        passes * ladder * GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_CANDIDATE_TABLE_STRIDE
      ),
      selectedBandwidth: view('float32', 2),
      summary: view('float32', 12)
    });
  }
  const regression = new GPUGeographicallyWeightedRegression({
    positions: view('float32x2', ROWS),
    predictors: view('float32', ROWS * predictorCount),
    predictorCount,
    response: view('float32', ROWS),
    mask: view('uint32', ROWS),
    parameters: view('float32', getGPUGeographicallyWeightedRegressionParameterLength(ladder)),
    output,
    ...configuration.props
  });
  const nodes = regression.getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  device.destroy();
  expect(sources.length, `${configuration.name} kernel count`).toBe(nodes.length);
  return sources;
}

const CONFIGURATIONS: Configuration[] = [
  {
    name: 'planar scan, defaults',
    predictorCount: 2,
    props: {indexGridSize: false},
    allOutputs: false
  },
  {
    name: 'planar scan, every output',
    predictorCount: 2,
    props: {indexGridSize: false, maximumBandwidthCount: 8},
    allOutputs: true
  },
  {
    name: 'grid, every output, three refinement passes',
    predictorCount: 3,
    props: {indexGridSize: [4, 4], maximumBandwidthCount: 6, refinement: 3},
    allOutputs: true
  },
  {
    name: 'great circle, every output',
    predictorCount: 2,
    props: {
      maximumBandwidthCount: 8,
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters'
      }
    },
    allOutputs: true
  },
  {
    name: 'refinement off',
    predictorCount: 2,
    props: {indexGridSize: false, refinement: false, maximumBandwidthCount: 8},
    allOutputs: true
  },
  {name: 'eleven predictors', predictorCount: 11, props: {indexGridSize: false}, allOutputs: true}
];

it('GPUGeographicallyWeightedRegression kernels parse as WGSL and respect the binding limit', () => {
  for (const configuration of CONFIGURATIONS) {
    const sources = getKernelSources(configuration);
    for (const {id, source} of sources) {
      let reflection: WgslReflect | undefined;
      expect(() => {
        reflection = new WgslReflect(source);
      }, `${configuration.name}: ${id} parses`).not.toThrow();
      const storageBindings = reflection!.storage.length;
      expect(storageBindings, `${configuration.name}: ${id} storage bindings`).toBeLessThanOrEqual(
        8
      );
    }
  }
});

it('GPUGeographicallyWeightedRegression builds the documented node sequence', () => {
  const ids = (configuration: Configuration) =>
    getKernelSources(configuration).map(({id}) =>
      id.replace(/^geographically-weighted-regression/, 'g')
    );
  expect(ids(CONFIGURATIONS[0])).toEqual([
    'g-validate',
    'g-candidates',
    'g-tile-candidates',
    'g-select',
    'g-refine-1-ladder',
    'g-refine-1-candidates',
    'g-refine-1-tile-candidates',
    'g-refine-1-select',
    'g-refine-2-ladder',
    'g-refine-2-candidates',
    'g-refine-2-tile-candidates',
    'g-refine-2-select',
    'g-choose',
    'g-fit',
    'g-tile-statistics',
    'g-summarize'
  ]);
  expect(ids(CONFIGURATIONS[4])).toEqual([
    'g-validate',
    'g-candidates',
    'g-tile-candidates',
    'g-select',
    'g-choose',
    'g-fit',
    'g-tile-statistics',
    'g-summarize',
    'g-publish-candidates',
    'g-publish',
    'g-publish-inference'
  ]);
});

it('the Student t WGSL declares the functions the selection kernel calls', () => {
  const source = getStudentTWGSL();
  expect(source).toContain('fn gwrStudentTwoSidedTail(');
  expect(source).toContain('fn gwrStudentCriticalValue(');
});
