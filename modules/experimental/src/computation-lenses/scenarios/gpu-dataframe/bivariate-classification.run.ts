// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {runExample} from './bivariate-classification.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.bivariate-classification';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['classIds', 'colors', 'classCounts'];

/** Executes the recorded bivariate classification once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-tracts');
  const poverty = Float32Array.from(dataset.columns.poverty150Pct, value => value ?? NaN);
  const uninsured = Float32Array.from(dataset.columns.uninsuredPct, value => value ?? NaN);
  const graph = createDataFrameLensGraph<undefined>(
    device,
    'gpu-dataframe-bivariate-classification',
    RETAINED
  );
  const example = await runExample(device, {
    graph,
    tracts: {poverty150Pct: poverty, uninsuredPct: uninsured}
  });
  try {
    options?.signal?.throwIfAborted();
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUBivariateClassification',
      graph,
      specs: {
        poverty150Pct: {
          view: example.poverty,
          role: 'input',
          keySpace: 'row',
          meaning: 'X axis: share of residents below 150% of the poverty line.'
        },
        uninsuredPct: {
          view: example.uninsured,
          role: 'input',
          keySpace: 'row',
          meaning: 'Y axis: share of residents without health insurance.'
        },
        classIds: {
          view: example.classIds,
          role: 'output',
          keySpace: 'row',
          meaning: 'Joint class uninsuredClass * 3 + povertyClass; 4294967295 marks no data.'
        },
        colors: {
          view: example.colors,
          role: 'output',
          keySpace: 'row',
          meaning: 'Packed rgba8 colour per tract: r | g << 8 | b << 16 | a << 24.'
        },
        classCounts: {view: example.classCounts, role: 'output', keySpace: 'class'}
      },
      keySpaces: [
        {id: 'row', label: 'Census tract', size: poverty.length, indexSpace: '1d'},
        {id: 'invocation', label: 'Compute invocation', size: poverty.length, indexSpace: '1d'},
        {
          id: 'class',
          label: 'Joint class (uninsured row, poverty column)',
          size: 9,
          indexSpace: '1d'
        }
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
        createDataFrameChoice('classify', 'classCountX', 3, {sweepable: true}),
        createDataFrameChoice('classify', 'classCountY', 3, {sweepable: true}),
        createDataFrameChoice('classify', 'maximumClassCount', 3),
        createDataFrameChoice('classify', 'breaksX', '0, 20, 35, 100'),
        createDataFrameChoice('classify', 'breaksY', '0, 8, 16, 100'),
        createDataFrameChoice('derive', 'valueByAlpha', false, {alternatives: [false, true]})
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`
    });
  } finally {
    example.destroy();
  }
}
