// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPURipleyDistanceEdgeCorrection} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './ripley-distance-functions.example';
import {
  capturePairStatisticsViews,
  createDataFrameChoice,
  createDataFrameLensGraph,
  createDataFrameLensRecord,
  getDataFrameElementFormulas,
  loadDataFrameLensData
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.ripley-distance-functions';
const RADIUS_COUNT = 10;
const MAXIMUM_DISTANCE = 2000;
const REFERENCE_GRID = [53, 47] as const;
const EDGE_CORRECTIONS: readonly GPURipleyDistanceEdgeCorrection[] = [
  'none',
  'border',
  'kaplan-meier',
  'hanisch'
];

/** Runs G, F and J once over the Chicago observations and reads every radius back from the GPU. */
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
  const graph = createDataFrameLensGraph<undefined>(device, SCENARIO_ID, ['g', 'f', 'j', 'radii']);
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
        g: {
          view: example.g,
          role: 'output',
          keySpace: 'radius',
          meaning: 'Nearest-event distance function; NaN where undefined.'
        },
        f: {
          view: example.f,
          role: 'output',
          keySpace: 'radius',
          meaning: 'Empty-space function over the reference lattice; NaN where undefined.'
        },
        j: {
          view: example.j,
          role: 'output',
          keySpace: 'radius',
          meaning: 'J = (1 - G) / (1 - F): below 1 clustered, above 1 regular.'
        }
      });
      return createDataFrameLensRecord({
        scenarioId: SCENARIO_ID,
        device,
        graphs: [compiled.describe({graphId, domain: 'gpu-dataframe'})],
        formulas: getDataFrameElementFormulas('GPURipleyDistanceFunctions'),
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
          createDataFrameChoice('aggregate', 'referenceGrid', REFERENCE_GRID.join('x')),
          createDataFrameChoice('aggregate', 'edgeCorrection', 'border', {
            alternatives: EDGE_CORRECTIONS,
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
