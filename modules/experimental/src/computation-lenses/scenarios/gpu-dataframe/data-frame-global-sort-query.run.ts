// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './data-frame-global-sort-query.example';
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

/** The tracts are split here so the global ordering has to span two record batches. */
const FIRST_BATCH_ROW_COUNT = 396;

/** Executes the global top-K once and reads the ranking, batch counts and inputs from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const columns = (await loadDataFrameLensData('f-chicago-tracts')).columns;
  const graph = createDataFrameLensGraph<GPUDataFrameQueryParameters>(
    device,
    'gpu-dataframe.data-frame-global-sort-query'
  );
  const poverty150Pct = new Float32Array(columns.poverty150Pct.map(value => value ?? 0));
  const povertyValidity = new Uint32Array(
    columns.poverty150Pct.map(value => (value === null ? 0 : 1))
  );
  const {query, tracts} = await runExample(device, {
    graph,
    tractBatches: [
      {
        poverty150Pct: poverty150Pct.slice(0, FIRST_BATCH_ROW_COUNT),
        povertyValidity: povertyValidity.slice(0, FIRST_BATCH_ROW_COUNT)
      },
      {
        poverty150Pct: poverty150Pct.slice(FIRST_BATCH_ROW_COUNT),
        povertyValidity: povertyValidity.slice(FIRST_BATCH_ROW_COUNT)
      }
    ]
  });
  try {
    options?.signal?.throwIfAborted();
    submitDataFrameQuery(device, query);
    const poverty = applyValidity(
      await readGPUVectorValues(tracts.table.gpuVectors['poverty150Pct'], 'f32'),
      await readGPUVectorValues(tracts.validity.poverty150Pct!, 'u32')
    );
    const globalSelectedCount = await readGPUVectorValues(query.globalSelectedCount, 'u32');
    const selectedCounts = await readGPUVectorValues(query.selectedCounts, 'u32');
    // Only the first globalSelectedCount slots of globalRowIndices are published row ids.
    const globalRowIndices = (await readGPUVectorValues(query.globalRowIndices, 'u32')).slice(
      0,
      globalSelectedCount[0]
    );
    const captures = [
      createTableCapture({
        id: 'poverty150Pct',
        name: 'poverty150Pct',
        keySpace: 'row',
        role: 'input',
        values: poverty,
        elementType: 'f32',
        meaning: 'Both batches in source order; NaN marks a tract without an estimate (sorts last).'
      }),
      createTableCapture({
        id: 'globalRowIndices',
        name: 'globalRowIndices',
        keySpace: 'rank',
        role: 'output',
        values: globalRowIndices,
        elementType: 'u32',
        meaning: 'Source row of each global rank: the published prefix of globalRowIndices.'
      }),
      createTableCapture({
        id: 'sortedPoverty150Pct',
        name: 'sortedPoverty150Pct',
        keySpace: 'rank',
        role: 'output',
        values: gatherValues(poverty, globalRowIndices),
        elementType: 'f32',
        meaning: 'poverty150Pct at globalRowIndices: an exact gather of two readbacks.'
      }),
      createTableCapture({
        id: 'selectedCounts',
        name: 'selectedCounts',
        keySpace: 'batch',
        role: 'output',
        values: selectedCounts,
        elementType: 'u32',
        meaning: 'Rows of the global top-K that fall in each source batch.'
      })
    ];
    const description = query.describe({
      graphId: 'gpu-dataframe.data-frame-global-sort-query-compiled'
    });
    return createDataFrameQueryLensRecord({
      scenarioId: 'gpu-dataframe.data-frame-global-sort-query',
      element: 'GPUDataFrameGlobalSortQuery',
      device,
      graphs: [description.intent, description.lowered, description.compiled],
      derivedFormulas: description.formulas,
      keySpaces: [
        ...description.keySpaces,
        {id: 'rank', label: 'Global rank', size: globalRowIndices.length, indexSpace: '1d'},
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
        createDataFrameChoice('order', 'batchCount', selectedCounts.length)
      ],
      captures
    });
  } finally {
    query.destroy();
    tracts.destroy();
  }
}
