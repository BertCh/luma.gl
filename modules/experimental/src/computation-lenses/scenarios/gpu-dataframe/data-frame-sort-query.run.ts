// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './data-frame-sort-query.example';
import {
  loadDataFrameLensData,
  createDataFrameLensGraph,
  createDataFrameChoice,
  readGPUVectorValues,
  createTableCapture,
  createDataFrameQueryLensRecord,
  submitDataFrameQuery,
  applyValidity,
  gatherValues
} from './dataframe-lens-support';

/** Executes the per-batch top-K sort once and reads the ranking and its inputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const columns = (await loadDataFrameLensData('f-chicago-tracts')).columns;
  const graph = createDataFrameLensGraph<GPUDataFrameQueryParameters>(
    device,
    'gpu-dataframe.data-frame-sort-query'
  );
  const {query, tracts} = await runExample(device, {
    graph,
    tracts: {
      poverty150Pct: new Float32Array(columns.poverty150Pct.map(value => value ?? 0)),
      povertyValidity: new Uint32Array(columns.poverty150Pct.map(value => (value === null ? 0 : 1)))
    }
  });
  try {
    options?.signal?.throwIfAborted();
    submitDataFrameQuery(device, query);
    const poverty = applyValidity(
      await readGPUVectorValues(tracts.table.gpuVectors['poverty150Pct'], 'f32'),
      await readGPUVectorValues(tracts.validity.poverty150Pct!, 'u32')
    );
    const selectedCounts = await readGPUVectorValues(query.selectedCounts, 'u32');
    // Only the first selectedCounts[0] slots of rowIndices are published row ids.
    const rowIndices = (await readGPUVectorValues(query.rowIndices, 'u32')).slice(
      0,
      selectedCounts[0]
    );
    const captures = [
      createTableCapture({
        id: 'poverty150Pct',
        name: 'poverty150Pct',
        keySpace: 'row',
        role: 'input',
        values: poverty,
        elementType: 'f32',
        meaning: 'NaN marks a tract without a poverty estimate (invalid sidecar); it sorts last.'
      }),
      createTableCapture({
        id: 'rowIndices',
        name: 'rowIndices',
        keySpace: 'rank',
        role: 'output',
        values: rowIndices,
        elementType: 'u32',
        meaning: 'Source row of each rank: the published prefix of rowIndices.'
      }),
      createTableCapture({
        id: 'sortedPoverty150Pct',
        name: 'sortedPoverty150Pct',
        keySpace: 'rank',
        role: 'output',
        values: gatherValues(poverty, rowIndices),
        elementType: 'f32',
        meaning: 'poverty150Pct at rowIndices: an exact gather of two readbacks.'
      }),
      createTableCapture({
        id: 'selectedCounts',
        name: 'selectedCounts',
        keySpace: 'batch',
        role: 'output',
        values: selectedCounts,
        elementType: 'u32'
      })
    ];
    const description = query.describe({graphId: 'gpu-dataframe.data-frame-sort-query-compiled'});
    return createDataFrameQueryLensRecord({
      scenarioId: 'gpu-dataframe.data-frame-sort-query',
      element: 'GPUDataFrameSortQuery',
      device,
      graphs: [description.intent, description.lowered, description.compiled],
      derivedFormulas: description.formulas,
      keySpaces: [
        ...description.keySpaces,
        {id: 'rank', label: 'Rank in batch', size: rowIndices.length, indexSpace: '1d'},
        {id: 'batch', label: 'Source batch', size: selectedCounts.length, indexSpace: 'batch'}
      ],
      datumKeySpace: 'rank',
      relations: description.relations,
      choices: [
        createDataFrameChoice('order', 'direction', query.direction, {
          alternatives: ['ascending', 'descending']
        }),
        createDataFrameChoice('order', 'nulls', query.nulls, {alternatives: ['first', 'last']}),
        createDataFrameChoice('order', 'limit', query.limit ?? 'none', {sweepable: true}),
        createDataFrameChoice('order', 'algorithm', query.algorithm, {
          alternatives: ['auto', 'bitonic', 'radix']
        })
      ],
      captures
    });
  } finally {
    query.destroy();
    tracts.destroy();
  }
}
