// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPU_RIPLEY_EDGE_CORRECTION} from '../../../gpu-dataframe/pair-statistics/ripley-parameters';
import {runExample} from './ripley-envelope.example';
import {
  capturePairStatisticsViews,
  createDataFrameChoice,
  createDataFrameLensGraph,
  createDataFrameLensRecord,
  getDataFrameElementFormulas,
  loadDataFrameLensData
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.ripley-envelope';
const RADIUS_COUNT = 10;
const MAXIMUM_DISTANCE = 2000;
const SIMULATION_COUNT = 39;
const SEED = 2024;
const STATISTICS_LENGTH = 6;

/**
 * Runs the CSR envelope once and reads the observed curves, bands and summary back from the GPU.
 * The per-simulation K curves live in a graph-internal transient and are not an observable
 * output, so they are not captured.
 */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-nature');
  const {x, y} = dataset.columns;
  const positions = new Float32Array(x.length * 2);
  x.forEach((value, index) => {
    positions[2 * index] = value;
    positions[2 * index + 1] = y[index];
  });
  const [minX, minY, maxX, maxY] = dataset.studyWindow;
  const studyWindow = [minX, minY, maxX, maxY] as const;
  const graph = createDataFrameLensGraph<undefined>(device, SCENARIO_ID, [
    'observedK',
    'observedLMinusR',
    'radii',
    'lower',
    'upper',
    'globalLower',
    'globalUpper',
    'statistics'
  ]);
  const example = await runExample(device, {graph, observations: {positions}, studyWindow});
  try {
    options?.signal?.throwIfAborted();
    const compiled = graph.compile();
    try {
      const graphId = `${SCENARIO_ID}-compiled`;
      // statistics rows are [n, validSimulationCount, observedMaximumDeviation,
      // criticalMaximumDeviation, pValue, rank]; the formula names two of them.
      const statisticsRow = (row: number) =>
        graph.createDataView(example.statistics.buffer, {
          format: 'float32',
          length: 1,
          byteOffset: row * Float32Array.BYTES_PER_ELEMENT
        });
      const captures = await capturePairStatisticsViews(compiled, graphId, {
        positions: {
          view: example.positions,
          role: 'input',
          keySpace: 'point',
          geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
        },
        radii: {
          view: example.radii,
          role: 'input',
          keySpace: 'radius',
          meaning: 'Metres; from the observed GPURipley pass.'
        },
        observedK: {
          view: example.observedK,
          role: 'input',
          keySpace: 'radius',
          meaning: 'Observed isotropic K in square metres; feeds the maximum-deviation test.'
        },
        observedLMinusR: {
          view: example.observedLMinusR,
          role: 'input',
          keySpace: 'radius',
          meaning: 'Observed L(r) - r from the same GPURipley pass.'
        },
        lower: {
          view: example.lower,
          role: 'output',
          keySpace: 'radius',
          meaning: 'Pointwise lower bound of L - r: smallest of 39 CSR simulations.'
        },
        upper: {
          view: example.upper,
          role: 'output',
          keySpace: 'radius',
          meaning: 'Pointwise upper bound of L - r: largest of 39 CSR simulations.'
        },
        globalLower: {
          view: example.globalLower,
          role: 'output',
          keySpace: 'radius',
          meaning: 'Simultaneous lower band: minus the critical maximum deviation.'
        },
        globalUpper: {
          view: example.globalUpper,
          role: 'output',
          keySpace: 'radius',
          meaning: 'Simultaneous upper band: plus the critical maximum deviation.'
        },
        statistics: {
          view: example.statistics,
          role: 'output',
          keySpace: 'index',
          meaning:
            'Rows: n, valid simulations, observed and critical maximum deviation, p-value, rank.'
        },
        validCount: {
          view: statisticsRow(1),
          role: 'intermediate',
          keySpace: 'index',
          meaning: 'Row 1 of statistics: simulations with a finite curve.'
        },
        pValue: {
          view: statisticsRow(4),
          role: 'intermediate',
          keySpace: 'index',
          meaning: 'Row 4 of statistics: Monte Carlo maximum-absolute-deviation p-value.'
        }
      });
      return createDataFrameLensRecord({
        scenarioId: SCENARIO_ID,
        device,
        graphs: [compiled.describe({graphId, domain: 'gpu-dataframe'})],
        formulas: getDataFrameElementFormulas('GPURipleyEnvelope'),
        keySpaces: [
          {id: 'point', label: 'Observation', size: x.length, indexSpace: '1d'},
          {id: 'radius', label: 'Radius (200 m)', size: RADIUS_COUNT, indexSpace: '1d'},
          {id: 'index', label: 'Summary row', size: STATISTICS_LENGTH, indexSpace: '1d'}
        ],
        choices: [
          createDataFrameChoice('filter', 'coordinateModel', 'projected', {
            alternatives: ['projected', 'geographic']
          }),
          createDataFrameChoice('filter', 'studyWindow', 'rectangle'),
          createDataFrameChoice('filter', 'bounds', studyWindow.join(',')),
          createDataFrameChoice('aggregate', 'maximumDistance', MAXIMUM_DISTANCE, {
            unit: 'm',
            sweepable: true
          }),
          createDataFrameChoice('aggregate', 'radiusCount', RADIUS_COUNT),
          createDataFrameChoice('aggregate', 'edgeCorrection', 'isotropic', {
            alternatives: Object.keys(GPU_RIPLEY_EDGE_CORRECTION)
          }),
          createDataFrameChoice('derive', 'statistic', 'l-minus-r', {
            alternatives: ['k', 'l', 'l-minus-r']
          }),
          createDataFrameChoice('infer', 'simulationCount', SIMULATION_COUNT, {
            alternatives: [39, 99],
            sweepable: true
          }),
          createDataFrameChoice('infer', 'rank', 1),
          createDataFrameChoice('infer', 'seed', SEED),
          createDataFrameChoice('infer', 'pCorrection', 'none')
        ],
        captures
      });
    } finally {
      compiled.destroy();
    }
  } finally {
    example.parameters.destroy();
    example.buffers.forEach(buffer => buffer.destroy());
  }
}
