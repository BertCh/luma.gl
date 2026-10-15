// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPU_COLOR_SCALE_CODES} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './color-scale.example';
import {
  createDataFrameChoice,
  createDataFrameLensGraph,
  loadDataFrameLensData,
  runDataFrameContributorLens
} from './dataframe-lens-support';

const SCENARIO_ID = 'gpu-dataframe.color-scale';
/** Transient outputs the capture reads; they must be retained because transients can alias. */
const RETAINED = ['classIndices', 'colors', 'classCounts'];
const PALETTE_COUNT = 5;

/** Executes the recorded colour scale once and reads inputs and outputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-tracts');
  const poverty = Float32Array.from(dataset.columns.poverty150Pct, value => value ?? NaN);
  const graph = createDataFrameLensGraph<undefined>(device, 'gpu-dataframe-color-scale', RETAINED);
  const example = await runExample(device, {graph, tracts: {poverty150Pct: poverty}});
  try {
    options?.signal?.throwIfAborted();
    return await runDataFrameContributorLens(device, {
      scenarioId: SCENARIO_ID,
      element: 'GPUColorScale',
      graph,
      specs: {
        values: {
          view: example.poverty,
          role: 'input',
          keySpace: 'row',
          meaning:
            'Share of residents below 150% of the poverty line; suppressed estimates are NaN.'
        },
        classIndices: {
          view: example.classIndices,
          role: 'output',
          keySpace: 'row',
          meaning: 'Palette class per tract; 4294967295 marks a no-data tract.'
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
        {id: 'class', label: 'Palette class', size: PALETTE_COUNT, indexSpace: '1d'}
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
        createDataFrameChoice('classify', 'scale', 'threshold', {
          alternatives: Object.keys(GPU_COLOR_SCALE_CODES)
        }),
        createDataFrameChoice('classify', 'domainCount', example.bands.length),
        createDataFrameChoice('classify', 'paletteCount', PALETTE_COUNT),
        createDataFrameChoice('derive', 'interpolation', 'step', {
          alternatives: ['step', 'linear']
        }),
        createDataFrameChoice('derive', 'clamp', false, {alternatives: [false, true]}),
        createDataFrameChoice('derive', 'noDataColor', 0),
        createDataFrameChoice('classify', 'domain', '0, 10, 20, 30, 40, 100')
      ],
      datasetVersion: `${dataset.id}:${dataset.rowCount}`
    });
  } finally {
    example.destroy();
  }
}
