// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './data-frame-grouped-aggregation-query.example';
import {
  loadDataFrameLensData,
  createDataFrameLensGraph,
  createDataFrameChoice,
  readGPUVectorValues,
  createTableCapture,
  createDataFrameQueryLensRecord,
  submitDataFrameQuery,
  applyValidity
} from './dataframe-lens-support';

const GROUP_COUNT = 10;

async function runGroupedAggregationLens(
  device: Device,
  scenarioId: string,
  options?: {signal?: AbortSignal}
) {
  options?.signal?.throwIfAborted();
  const columns = (await loadDataFrameLensData('f-chicago-nature')).columns;
  const graph = createDataFrameLensGraph<GPUDataFrameQueryParameters>(device, scenarioId);
  const {query, observations} = await runExample(device, {
    graph,
    observations: {
      category: Uint32Array.from(columns.category),
      researchGrade: Float32Array.from(columns.researchGrade)
    }
  });
  try {
    options?.signal?.throwIfAborted();
    submitDataFrameQuery(device, query);
    const captures = [
      createTableCapture({
        id: 'category',
        name: 'category',
        keySpace: 'row',
        role: 'input',
        values: await readGPUVectorValues(observations.table.gpuVectors['category'], 'u32'),
        elementType: 'u32'
      }),
      createTableCapture({
        id: 'researchGrade',
        name: 'researchGrade',
        keySpace: 'row',
        role: 'input',
        values: await readGPUVectorValues(observations.table.gpuVectors['researchGrade'], 'f32'),
        elementType: 'f32'
      })
    ];
    // Empty groups carry an invalid sidecar; their payload is read back as NaN.
    const validity = query.validity as Record<string, GPUVector | undefined>;
    for (const [name, captureName, elementType] of [
      ['category', 'groupKey', 'u32'],
      ['count', 'count', 'u32'],
      ['sum', 'sum', 'f32'],
      ['mean', 'mean', 'f32']
    ] as const) {
      const values = await readGPUVectorValues(
        (query.table.gpuVectors as Record<string, GPUVector>)[name],
        elementType
      );
      const groupValidity = validity[name];
      captures.push(
        createTableCapture({
          id: captureName,
          name: captureName,
          keySpace: 'group',
          role: 'output',
          values: groupValidity
            ? applyValidity(values, await readGPUVectorValues(groupValidity, 'u32'))
            : values,
          elementType
        })
      );
    }
    const description = query.describe({graphId: `${scenarioId}-compiled`});
    return createDataFrameQueryLensRecord({
      scenarioId,
      element: 'GPUDataFrameGroupedAggregationQuery',
      device,
      graphs: [description.intent, description.lowered, description.compiled],
      derivedFormulas: description.formulas,
      keySpaces: [
        ...description.keySpaces,
        {id: 'group', label: 'Category group', size: GROUP_COUNT, indexSpace: '1d'}
      ],
      datumKeySpace: 'group',
      relations: description.relations,
      choices: [
        createDataFrameChoice('group', 'groupCount', GROUP_COUNT),
        createDataFrameChoice('aggregate', 'operations', 'count, sum, mean', {
          alternatives: ['count', 'sum', 'min', 'max', 'mean']
        })
      ],
      captures
    });
  } finally {
    query.destroy();
    observations.destroy();
  }
}

/** Executes the grouped aggregation once and reads every displayed column from GPU buffers. */
export function run(device: Device, options?: {signal?: AbortSignal}) {
  return runGroupedAggregationLens(
    device,
    'gpu-dataframe.data-frame-grouped-aggregation-query',
    options
  );
}

/** The same run, recorded under the `GPUDataFrameGroupByQuery` scenario id. */
export function runGroupBy(device: Device, options?: {signal?: AbortSignal}) {
  return runGroupedAggregationLens(device, 'gpu-dataframe.data-frame-group-by-query', options);
}
