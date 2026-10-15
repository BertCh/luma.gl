// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {runExample} from './time-window-filter.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.time-window-filter';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['ids', 'count', 'overflow', 'outputMask'];

/** Executes the recorded time window once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-nature');
  const timestamp = Float64Array.from(dataset.columns.timestamp);
  const graph = createDataFrameLensGraph<undefined>(
    device,
    'gpu-dataframe-time-window-filter',
    RETAINED
  );
  const example = await runExample(device, {graph, observations: {timestamp}});
  try {
    options?.signal?.throwIfAborted();
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUTimeWindowFilter',
      graph,
      specs: {
        timestamps: {
          view: example.timestamps,
          role: 'input',
          keySpace: 'row',
          meaning:
            'High float32 part of the double-single timestamp, seconds since 2023-01-01 local clock.'
        },
        outputMask: {
          view: example.outputMask,
          role: 'output',
          keySpace: 'row',
          meaning: '1 when start <= timestamp <= end, both ends inclusive.'
        },
        count: {view: example.count, role: 'output', keySpace: 'summary'},
        ids: {
          view: example.ids,
          role: 'output',
          keySpace: 'compactSlot',
          meaning: 'Only the first count slots are accepted row ids; remaining slots are unused.'
        }
      },
      keySpaces: [
        {id: 'row', label: 'Observation', size: timestamp.length, indexSpace: '1d'},
        {id: 'invocation', label: 'Compute invocation', size: timestamp.length, indexSpace: '1d'},
        {id: 'summary', label: 'Whole-column summary', size: 1, indexSpace: '1d'},
        {id: 'compactSlot', label: 'Compacted row slot', size: timestamp.length, indexSpace: '1d'}
      ],
      relations: [
        {
          id: 'invocation-row',
          from: 'invocation',
          to: 'row',
          cardinality: '1:1',
          kind: 'function',
          ref: {function: 'identity'},
          timeDependent: false
        }
      ],
      datumKeySpace: 'row',
      choices: [
        createDataFrameChoice('filter', 'start', example.window.start, {
          unit: 's',
          sweepable: true
        }),
        createDataFrameChoice('filter', 'end', example.window.end, {unit: 's', sweepable: true}),
        createDataFrameChoice('filter', 'startFadeDuration', 0, {unit: 's'}),
        createDataFrameChoice('filter', 'endFadeDuration', 0, {unit: 's'}),
        createDataFrameChoice('filter', 'sorted', false, {alternatives: [false, true]})
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`
    });
  } finally {
    example.destroy();
  }
}
