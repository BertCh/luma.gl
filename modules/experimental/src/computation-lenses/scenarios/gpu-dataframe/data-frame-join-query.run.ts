// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './data-frame-join-query.example';
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

/** Executes the inner join once and reads both inputs and the paired row ids from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const nature = (await loadDataFrameLensData('f-chicago-nature')).columns;
  const tractColumns = (await loadDataFrameLensData('f-chicago-tracts')).columns;
  const graph = createDataFrameLensGraph<GPUDataFrameQueryParameters>(
    device,
    'gpu-dataframe.data-frame-join-query'
  );
  const {query, observations, tracts} = await runExample(device, {
    graph,
    observations: {
      tract: Uint32Array.from(nature.tract),
      category: Uint32Array.from(nature.category)
    },
    tracts: {
      tract: Uint32Array.from(tractColumns.tract),
      poverty150Pct: new Float32Array(tractColumns.poverty150Pct.map(value => value ?? 0)),
      povertyValidity: new Uint32Array(
        tractColumns.poverty150Pct.map(value => (value === null ? 0 : 1))
      )
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
    const requiredCounts = await readGPUVectorValues(query.requiredCounts, 'u32');
    // Only the first selectedCounts[0] slots of the paired id vectors are published pairs.
    const readPairs = async (vector: typeof query.rowIndices) =>
      (await readGPUVectorValues(vector, 'u32')).slice(0, selectedCounts[0]);
    const rowIndices = await readPairs(query.rowIndices);
    const rightRowIndices = await readPairs(query.rightRowIndices);
    const input = (
      id: string,
      keySpace: string,
      values: number[],
      elementType: 'u32' | 'f32',
      meaning?: string
    ) =>
      createTableCapture({
        id,
        name: id,
        keySpace,
        role: 'input',
        values,
        elementType,
        ...(meaning ? {meaning} : {})
      });
    const captures = [
      input(
        'tract',
        'row',
        await readGPUVectorValues(observations.table.gpuVectors['tract'], 'u32'),
        'u32'
      ),
      input(
        'category',
        'row',
        await readGPUVectorValues(observations.table.gpuVectors['category'], 'u32'),
        'u32'
      ),
      input(
        'rightTract',
        'rightRow',
        await readGPUVectorValues(tracts.table.gpuVectors['tract'], 'u32'),
        'u32'
      ),
      input(
        'poverty150Pct',
        'rightRow',
        poverty,
        'f32',
        'NaN marks a tract without a poverty estimate (invalid sidecar).'
      ),
      createTableCapture({
        id: 'rowIndices',
        name: 'rowIndices',
        keySpace: 'pair',
        role: 'output',
        values: rowIndices,
        elementType: 'u32',
        meaning: 'Left (observation) row of each pair.'
      }),
      createTableCapture({
        id: 'rightRowIndices',
        name: 'rightRowIndices',
        keySpace: 'pair',
        role: 'output',
        values: rightRowIndices,
        elementType: 'u32',
        meaning: 'Right (tract) row of each pair.'
      }),
      createTableCapture({
        id: 'rightPoverty150Pct',
        name: 'rightPoverty150Pct',
        keySpace: 'pair',
        role: 'output',
        values: gatherValues(poverty, rightRowIndices),
        elementType: 'f32',
        meaning: 'poverty150Pct at rightRowIndices: an exact gather of two readbacks.'
      }),
      createTableCapture({
        id: 'selectedCounts',
        name: 'selectedCounts',
        keySpace: 'batch',
        role: 'output',
        values: selectedCounts,
        elementType: 'u32'
      }),
      createTableCapture({
        id: 'requiredCounts',
        name: 'requiredCounts',
        keySpace: 'batch',
        role: 'output',
        values: requiredCounts,
        elementType: 'u32',
        meaning: 'Pairs the join needs, before any capacity truncation.'
      })
    ];
    const description = query.describe({graphId: 'gpu-dataframe.data-frame-join-query-compiled'});
    return createDataFrameQueryLensRecord({
      scenarioId: 'gpu-dataframe.data-frame-join-query',
      element: 'GPUDataFrameJoinQuery',
      device,
      graphs: [description.intent, description.lowered, description.compiled],
      derivedFormulas: description.formulas,
      keySpaces: [
        ...description.keySpaces,
        {id: 'rightRow', label: 'Tract (right row)', size: poverty.length, indexSpace: '1d'},
        {id: 'pair', label: 'Joined pair', size: rowIndices.length, indexSpace: '1d'},
        {id: 'batch', label: 'Source batch', size: selectedCounts.length, indexSpace: 'batch'}
      ],
      datumKeySpace: 'pair',
      relations: description.relations,
      choices: [
        createDataFrameChoice('join', 'joinType', query.joinType, {
          alternatives: ['inner', 'left', 'semi', 'anti']
        }),
        createDataFrameChoice('join', 'leftOn', 'tract'),
        createDataFrameChoice('join', 'rightOn', 'tract')
      ],
      captures
    });
  } finally {
    query.destroy();
    observations.destroy();
    tracts.destroy();
  }
}
