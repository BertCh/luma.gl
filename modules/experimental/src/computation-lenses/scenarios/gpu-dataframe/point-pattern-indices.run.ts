// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPU_CLARK_EVANS_CORRECTION} from '../../../gpu-dataframe/pair-statistics/point-pattern-indices-parameters';
import {runExample} from './point-pattern-indices.example';
import {
  capturePairStatisticsViews,
  createDataFrameChoice,
  createDataFrameLensGraph,
  createDataFrameLensRecord,
  getDataFrameElementFormulas,
  loadDataFrameLensData
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.point-pattern-indices';
const SEARCH_CELL_SIZE = 500;
const CLARK_EVANS_LENGTH = 7;
const QUADRAT_STATISTICS_LENGTH = 7;

/** Runs the Clark-Evans and quadrat tests once and reads every output back from the GPU. */
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
    'nearestNeighborDistances',
    'nearestNeighborIds',
    'clarkEvans',
    'quadratCounts',
    'quadratStatistics'
  ]);
  const example = await runExample(device, {graph, observations: {positions}, studyWindow});
  try {
    options?.signal?.throwIfAborted();
    const compiled = graph.compile();
    try {
      const graphId = `${SCENARIO_ID}-compiled`;
      const row = (view: typeof example.clarkEvans, index: number) =>
        graph.createDataView(view.buffer, {
          format: 'float32',
          length: 1,
          byteOffset: index * Float32Array.BYTES_PER_ELEMENT
        });
      const captures = await capturePairStatisticsViews(compiled, graphId, {
        positions: {
          view: example.positions,
          role: 'input',
          keySpace: 'point',
          geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
        },
        nearestNeighborDistances: {
          view: example.nearestNeighborDistances,
          role: 'output',
          keySpace: 'point',
          meaning: 'Metres to the nearest other observation; zero for duplicated locations.'
        },
        nearestNeighborIds: {
          view: example.nearestNeighborIds,
          role: 'output',
          keySpace: 'point',
          meaning: 'Row of the nearest other observation; ties go to the smallest row.'
        },
        quadratCounts: {
          view: example.quadratCounts,
          role: 'output',
          keySpace: 'bin',
          meaning: 'Observations per quadrat, row-major over 8 columns and 10 rows.'
        },
        clarkEvans: {
          view: example.clarkEvans,
          role: 'output',
          keySpace: 'index',
          meaning: 'Rows: n, observed, expected, ratio, standard error, z, p-value.'
        },
        observed: {
          view: row(example.clarkEvans, 1),
          role: 'intermediate',
          keySpace: 'index',
          meaning: 'Row 1 of clarkEvans: mean nearest-neighbor distance.'
        },
        expected: {
          view: row(example.clarkEvans, 2),
          role: 'intermediate',
          keySpace: 'index',
          meaning: 'Row 2 of clarkEvans: Donnelly expected mean distance.'
        },
        ratio: {
          view: row(example.clarkEvans, 3),
          role: 'intermediate',
          keySpace: 'index',
          meaning: 'Row 3 of clarkEvans: observed over expected.'
        },
        quadratStatistics: {
          view: example.quadratStatistics,
          role: 'output',
          keySpace: 'quadratIndex',
          meaning:
            'Rows: quadrats, mean, variance, variance-to-mean ratio, chi-square, df, p-value.'
        },
        chiSquare: {
          view: row(example.quadratStatistics, 4),
          role: 'intermediate',
          keySpace: 'quadratIndex',
          meaning: 'Row 4 of quadratStatistics: sum of (count - expected)^2 / expected.'
        }
      });
      return createDataFrameLensRecord({
        scenarioId: SCENARIO_ID,
        device,
        graphs: [compiled.describe({graphId, domain: 'gpu-dataframe'})],
        formulas: getDataFrameElementFormulas('GPUPointPatternIndices'),
        keySpaces: [
          {id: 'point', label: 'Observation', size: x.length, indexSpace: '1d'},
          {id: 'bin', label: 'Quadrat (8 x 10)', size: 80, indexSpace: '1d'},
          {id: 'index', label: 'Clark-Evans row', size: CLARK_EVANS_LENGTH, indexSpace: '1d'},
          {
            id: 'quadratIndex',
            label: 'Quadrat statistic row',
            size: QUADRAT_STATISTICS_LENGTH,
            indexSpace: '1d'
          }
        ],
        choices: [
          createDataFrameChoice('filter', 'coordinateModel', 'projected', {
            alternatives: ['projected', 'geographic']
          }),
          createDataFrameChoice('filter', 'studyWindow', 'rectangle'),
          createDataFrameChoice('filter', 'bounds', studyWindow.join(',')),
          createDataFrameChoice('select', 'maximumDistance', SEARCH_CELL_SIZE, {unit: 'm'}),
          createDataFrameChoice('aggregate', 'clarkEvansCorrection', 'donnelly', {
            alternatives: Object.keys(GPU_CLARK_EVANS_CORRECTION),
            sweepable: true
          }),
          createDataFrameChoice('aggregate', 'quadratGrid', '8 x 10', {sweepable: true}),
          createDataFrameChoice('infer', 'alternative', 'two-sided', {
            alternatives: ['two-sided', 'greater', 'lesser']
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
