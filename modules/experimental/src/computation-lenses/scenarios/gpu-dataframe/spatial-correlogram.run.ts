// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {runExample} from './spatial-correlogram.example';
import {
  capturePairStatisticsViews,
  createDataFrameChoice,
  createDataFrameLensGraph,
  createDataFrameLensRecord,
  getDataFrameElementFormulas,
  loadDataFrameLensData
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.spatial-correlogram';
const BAND_COUNT = 8;
const MAXIMUM_DISTANCE = 200_000;

/** Runs the rainfall correlogram once and reads the inputs and every band back from the GPU. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-ghcn-stations');
  const {x, y, prcp} = dataset.columns;
  const positions = new Float32Array(x.length * 2);
  x.forEach((value, index) => {
    positions[2 * index] = value;
    positions[2 * index + 1] = y[index];
  });
  // Inclusive extent of the stations, widened by 1 km so no station sits on the edge.
  const bounds = [
    Math.floor(Math.min(...x) / 1000) * 1000 - 1000,
    Math.floor(Math.min(...y) / 1000) * 1000 - 1000,
    Math.ceil(Math.max(...x) / 1000) * 1000 + 1000,
    Math.ceil(Math.max(...y) / 1000) * 1000 + 1000
  ] as const;
  const graph = createDataFrameLensGraph<undefined>(device, SCENARIO_ID, [
    'moransI',
    'expectedI',
    'zScores',
    'pValues',
    'pairCounts',
    'statistics'
  ]);
  const example = await runExample(device, {
    graph,
    stations: {positions, precipitation: new Float32Array(prcp)},
    bounds
  });
  try {
    options?.signal?.throwIfAborted();
    const compiled = graph.compile();
    try {
      const graphId = `${SCENARIO_ID}-compiled`;
      // statistics rows are [n, mean, variance, sumOfSquares, kurtosis]; the formula names two.
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
        prcp: {view: example.rainfall, role: 'input', keySpace: 'point'},
        moransI: {view: example.moransI, role: 'output', keySpace: 'band'},
        expectedI: {view: example.expectedI, role: 'output', keySpace: 'band'},
        zScores: {view: example.zScores, role: 'output', keySpace: 'band'},
        pValues: {
          view: example.pValues,
          role: 'output',
          keySpace: 'band',
          meaning: 'Two-sided normal p-value, uncorrected for the number of bands tested.'
        },
        pairCounts: {
          view: example.pairCounts,
          role: 'output',
          keySpace: 'band',
          meaning: 'Unordered station pairs per band.'
        },
        statistics: {
          view: example.statistics,
          role: 'output',
          keySpace: 'statistic',
          meaning: 'Rows: n, mean, variance, sum of squares, kurtosis of the included values.'
        },
        count: {
          view: statisticsRow(0),
          role: 'intermediate',
          keySpace: 'statistic',
          meaning: 'Row 0 of statistics: included station count n.'
        },
        sumOfSquares: {
          view: statisticsRow(3),
          role: 'intermediate',
          keySpace: 'statistic',
          meaning: 'Row 3 of statistics: centered sum of squares M2.'
        }
      });
      return createDataFrameLensRecord({
        scenarioId: SCENARIO_ID,
        device,
        graphs: [compiled.describe({graphId, domain: 'gpu-dataframe'})],
        formulas: getDataFrameElementFormulas('GPUSpatialCorrelogram'),
        keySpaces: [
          {id: 'point', label: 'Weather station', size: x.length, indexSpace: '1d'},
          {id: 'band', label: 'Distance band (25 km)', size: BAND_COUNT, indexSpace: '1d'},
          {id: 'statistic', label: 'Summary row', size: 5, indexSpace: '1d'}
        ],
        choices: [
          createDataFrameChoice('filter', 'coordinateModel', 'projected', {
            alternatives: ['projected', 'geographic']
          }),
          createDataFrameChoice('filter', 'bounds', bounds.join(',')),
          createDataFrameChoice('group', 'maximumDistance', MAXIMUM_DISTANCE, {
            unit: 'm',
            sweepable: true
          }),
          createDataFrameChoice('group', 'bandCount', BAND_COUNT, {sweepable: true}),
          createDataFrameChoice('group', 'bandMode', 'cumulative', {
            alternatives: ['cumulative', 'annulus']
          }),
          createDataFrameChoice('infer', 'varianceAssumption', 'randomization', {
            alternatives: ['randomization', 'normality'],
            sweepable: true
          }),
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
