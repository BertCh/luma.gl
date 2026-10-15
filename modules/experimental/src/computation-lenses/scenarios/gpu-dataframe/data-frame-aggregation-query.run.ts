// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './data-frame-aggregation-query.example';
import {
  loadDataFrameLensData,
  createDataFrameLensGraph,
  readGPUVectorValues,
  createDataFrameChoice,
  createTableCapture,
  createDataFrameQueryLensRecord,
  submitDataFrameQuery,
  applyValidity
} from './dataframe-lens-support';

/** Executes the global reduction once and reads the population column and every statistic from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const columns = (await loadDataFrameLensData('f-chicago-tracts')).columns;
  const graph = createDataFrameLensGraph<GPUDataFrameQueryParameters>(
    device,
    'gpu-dataframe.data-frame-aggregation-query'
  );
  const {query, tracts} = await runExample(device, {
    graph,
    tracts: {population: Float32Array.from(columns.population)}
  });
  try {
    options?.signal?.throwIfAborted();
    submitDataFrameQuery(device, query);
    // The input is named after the column, not the formula symbol, because the symbol is indexed by row
    // while the statistics have a single row.
    const captures = [
      createTableCapture({
        id: 'population',
        name: 'population',
        keySpace: 'row',
        role: 'input',
        values: await readGPUVectorValues(tracts.table.gpuVectors['population'], 'f32'),
        elementType: 'f32'
      })
    ];
    // An empty or fully rejected input leaves an invalid sidecar; its payload is read back as NaN.
    const validity = query.validity as Record<string, GPUVector | undefined>;
    for (const [name, elementType] of [
      ['rows', 'u32'],
      ['sum', 'f32'],
      ['minimum', 'f32'],
      ['maximum', 'f32'],
      ['mean', 'f32']
    ] as const) {
      const values = await readGPUVectorValues(
        (query.table.gpuVectors as Record<string, GPUVector>)[name],
        elementType
      );
      const statisticValidity = validity[name];
      captures.push(
        createTableCapture({
          id: name,
          name,
          keySpace: 'statistic',
          role: 'output',
          values: statisticValidity
            ? applyValidity(values, await readGPUVectorValues(statisticValidity, 'u32'))
            : values,
          elementType
        })
      );
    }
    const description = query.describe({
      graphId: 'gpu-dataframe.data-frame-aggregation-query-compiled'
    });
    return createDataFrameQueryLensRecord({
      scenarioId: 'gpu-dataframe.data-frame-aggregation-query',
      element: 'GPUDataFrameAggregationQuery',
      device,
      graphs: [description.intent, description.lowered, description.compiled],
      derivedFormulas: description.formulas,
      keySpaces: [
        ...description.keySpaces,
        {id: 'statistic', label: 'Statistics row', size: 1, indexSpace: '1d'}
      ],
      datumKeySpace: 'statistic',
      relations: description.relations,
      choices: [
        createDataFrameChoice('aggregate', 'operations', 'count, sum, min, max, mean', {
          alternatives: ['count', 'sum', 'min', 'max', 'mean']
        })
      ],
      captures
    });
  } finally {
    query.destroy();
    tracts.destroy();
  }
}
