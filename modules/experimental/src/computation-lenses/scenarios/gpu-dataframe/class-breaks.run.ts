// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPU_CLASS_BREAKS_METHODS} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './class-breaks.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.class-breaks';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['breaks', 'classCount', 'classCounts'];
const CLASS_COUNT = 5;

/** Executes the recorded class breaks once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-tracts');
  const poverty = Float32Array.from(dataset.columns.poverty150Pct, value => value ?? NaN);
  const graph = createDataFrameLensGraph<undefined>(device, 'gpu-dataframe-class-breaks', RETAINED);
  const example = await runExample(device, {graph, tracts: {poverty150Pct: poverty}});
  try {
    options?.signal?.throwIfAborted();
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUClassBreaks',
      graph,
      specs: {
        poverty150Pct: {
          view: example.poverty,
          role: 'input',
          keySpace: 'row',
          meaning:
            'Share of residents below 150% of the poverty line; suppressed estimates are NaN.'
        },
        breaks: {
          view: example.breaks,
          role: 'output',
          keySpace: 'edge',
          meaning: 'Class edges e[0..k]: smallest value, inner edges, largest value.'
        },
        classCount: {view: example.classCount, role: 'output', keySpace: 'summary'},
        classCounts: {view: example.classCounts, role: 'output', keySpace: 'class'}
      },
      keySpaces: [
        {id: 'row', label: 'Census tract', size: poverty.length, indexSpace: '1d'},
        {id: 'edge', label: 'Class edge', size: CLASS_COUNT + 1, indexSpace: '1d'},
        {id: 'class', label: 'Class', size: CLASS_COUNT, indexSpace: '1d'},
        {id: 'summary', label: 'Whole-column summary', size: 1, indexSpace: '1d'}
      ],
      datumKeySpace: 'edge',
      choices: [
        createDataFrameChoice('classify', 'method', 'equal-interval', {
          alternatives: GPU_CLASS_BREAKS_METHODS
        }),
        createDataFrameChoice('classify', 'methods', 'equal-interval, quantile'),
        createDataFrameChoice('classify', 'classCount', CLASS_COUNT, {sweepable: true}),
        createDataFrameChoice('classify', 'maximumClassCount', CLASS_COUNT)
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`
    });
  } finally {
    example.destroy();
  }
}
