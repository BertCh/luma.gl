// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPU_RIPLEY_EDGE_CORRECTION} from '../../../gpu-dataframe/pair-statistics/ripley-parameters';
import {runExample} from './ripley.example';
import {
  capturePairStatisticsViews,
  createDataFrameChoice,
  createDataFrameLensGraph,
  createDataFrameLensRecord,
  getDataFrameElementFormulas,
  loadDataFrameLensData
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.ripley';
const RADIUS_COUNT = 10;
const MAXIMUM_DISTANCE = 2000;

/** Runs Ripley's K once over the Chicago observations and reads every radius back from the GPU. */
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
    'k',
    'l',
    'lMinusR',
    'radii',
    'pairCounts'
  ]);
  const example = await runExample(device, {graph, observations: {positions}, studyWindow});
  try {
    options?.signal?.throwIfAborted();
    const compiled = graph.compile();
    try {
      const graphId = `${SCENARIO_ID}-compiled`;
      const captures = await capturePairStatisticsViews(compiled, graphId, {
        positions: {
          view: example.positions,
          role: 'input',
          keySpace: 'point',
          geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
        },
        radii: {view: example.radii, role: 'output', keySpace: 'radius', meaning: 'Metres.'},
        k: {
          view: example.k,
          role: 'output',
          keySpace: 'radius',
          meaning: 'Edge-corrected K in square metres; NaN where undefined.'
        },
        l: {view: example.l, role: 'output', keySpace: 'radius', meaning: 'sqrt(K / pi), metres.'},
        lMinusR: {
          view: example.lMinusR,
          role: 'output',
          keySpace: 'radius',
          meaning: 'L(r) - r: zero under complete spatial randomness.'
        },
        pairCounts: {
          view: example.pairCounts,
          role: 'output',
          keySpace: 'radius',
          meaning: 'Uncorrected ordered pairs in the annulus ending at the radius.'
        }
      });
      return createDataFrameLensRecord({
        scenarioId: SCENARIO_ID,
        device,
        graphs: [compiled.describe({graphId, domain: 'gpu-dataframe'})],
        formulas: getDataFrameElementFormulas('GPURipley'),
        keySpaces: [
          {id: 'point', label: 'Observation', size: x.length, indexSpace: '1d'},
          {id: 'radius', label: 'Radius (200 m)', size: RADIUS_COUNT, indexSpace: '1d'}
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
          createDataFrameChoice('aggregate', 'radiusCount', RADIUS_COUNT, {sweepable: true}),
          createDataFrameChoice('aggregate', 'edgeCorrection', 'isotropic', {
            alternatives: Object.keys(GPU_RIPLEY_EDGE_CORRECTION),
            sweepable: true
          })
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
