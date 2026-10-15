// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  GPU_COLUMN_QUANTILE_INTERPOLATION_CODES,
  GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './column-quantiles.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.column-quantiles';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['quantiles', 'validCount', 'outputMask', 'filterBounds'];

/** Executes the recorded quantile selection once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-tracts');
  const poverty = Float32Array.from(dataset.columns.poverty150Pct, value => value ?? NaN);
  const graph = createDataFrameLensGraph<undefined>(
    device,
    'gpu-dataframe-column-quantiles',
    RETAINED
  );
  const example = await runExample(device, {graph, tracts: {poverty150Pct: poverty}});
  try {
    options?.signal?.throwIfAborted();
    // A window over the packed probability slots of the parameter buffer, after its header.
    const probability = graph.createDataView(example.parameterView.buffer, {
      format: 'float32',
      length: example.quantiles.length,
      byteOffset: GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH * Float32Array.BYTES_PER_ELEMENT
    });
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUColumnQuantiles',
      graph,
      specs: {
        poverty150Pct: {
          view: example.poverty,
          role: 'input',
          keySpace: 'row',
          meaning:
            'Share of residents below 150% of the poverty line; suppressed estimates are NaN.'
        },
        probability: {view: probability, role: 'input', keySpace: 'quantile'},
        quantiles: {view: example.quantiles, role: 'output', keySpace: 'quantile'},
        validCount: {
          view: example.validCount,
          role: 'output',
          keySpace: 'summary',
          meaning: 'Rows that are finite and unmasked.'
        },
        outputMask: {
          view: example.outputMask,
          role: 'output',
          keySpace: 'row',
          meaning: '1 when the row lies inside the 5-95 percentile filter bounds.'
        },
        filterBounds: {
          view: example.filterBounds,
          role: 'output',
          keySpace: 'bound',
          meaning: 'Lower and upper value of the percentile filter.'
        }
      },
      keySpaces: [
        {id: 'row', label: 'Census tract', size: poverty.length, indexSpace: '1d'},
        {id: 'quantile', label: 'Quantile probability', size: 3, indexSpace: '1d'},
        {id: 'summary', label: 'Whole-column summary', size: 1, indexSpace: '1d'},
        {id: 'bound', label: 'Filter bound (lower, upper)', size: 2, indexSpace: '1d'}
      ],
      datumKeySpace: 'quantile',
      choices: [
        createDataFrameChoice('select', 'quantiles', '0.25, 0.5, 0.75'),
        createDataFrameChoice('select', 'quantileCount', 3),
        createDataFrameChoice('select', 'interpolation', 'linear', {
          alternatives: Object.keys(GPU_COLUMN_QUANTILE_INTERPOLATION_CODES)
        }),
        createDataFrameChoice('filter', 'filterRange', '0.05, 0.95', {sweepable: true})
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`
    });
  } finally {
    example.destroy();
  }
}
