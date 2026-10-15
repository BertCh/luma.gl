// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {runExample} from './variogram.example';
import {
  capturePairStatisticsViews,
  createDataFrameChoice,
  createDataFrameLensGraph,
  createDataFrameLensRecord,
  getDataFrameElementFormulas,
  loadDataFrameLensData
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.variogram';
const LAG_COUNT = 12;
const MAXIMUM_DISTANCE = 300_000;

/** Runs the station variogram once and reads the inputs and every lag bin back from the GPU. */
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
    'semivariances',
    'pairCounts',
    'meanDistances'
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
      const captures = await capturePairStatisticsViews(compiled, graphId, {
        positions: {
          view: example.positions,
          role: 'input',
          keySpace: 'point',
          geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
        },
        prcp: {view: example.rainfall, role: 'input', keySpace: 'point'},
        semivariances: {view: example.semivariances, role: 'output', keySpace: 'lag'},
        pairCounts: {
          view: example.pairCounts,
          role: 'output',
          keySpace: 'lag',
          meaning: 'Unordered station pairs in the lag.'
        },
        meanDistances: {
          view: example.meanDistances,
          role: 'output',
          keySpace: 'lag',
          meaning: 'Mean pair distance in metres.'
        }
      });
      return createDataFrameLensRecord({
        scenarioId: SCENARIO_ID,
        device,
        graphs: [compiled.describe({graphId, domain: 'gpu-dataframe'})],
        formulas: getDataFrameElementFormulas('GPUVariogram'),
        keySpaces: [
          {id: 'point', label: 'Weather station', size: x.length, indexSpace: '1d'},
          {id: 'lag', label: 'Lag (25 km)', size: LAG_COUNT, indexSpace: '1d'}
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
          createDataFrameChoice('group', 'lagCount', LAG_COUNT, {sweepable: true}),
          createDataFrameChoice('group', 'directionCount', 1)
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
