// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './data-frame-lookup-query.example';
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

/** Executes the lookup once and reads both inputs and the source-aligned match columns from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const nature = (await loadDataFrameLensData('f-chicago-nature')).columns;
  const tractColumns = (await loadDataFrameLensData('f-chicago-tracts')).columns;
  const graph = createDataFrameLensGraph<GPUDataFrameQueryParameters>(
    device,
    'gpu-dataframe.data-frame-lookup-query'
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
    const rightRowIndices = await readGPUVectorValues(query.rightRowIndices, 'u32');
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
    const output = (id: string, values: number[], elementType: 'u32' | 'f32', meaning?: string) =>
      createTableCapture({
        id,
        name: id,
        keySpace: 'row',
        role: 'output',
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
      output(
        'rightRowIndices',
        rightRowIndices,
        'u32',
        'Right (tract) row per observation; 4294967295 marks no match.'
      ),
      output('matchMask', await readGPUVectorValues(query.matchMask, 'u32'), 'u32'),
      output('probeCounts', await readGPUVectorValues(query.probeCounts, 'u32'), 'u32'),
      output(
        'rightPoverty150Pct',
        gatherValues(poverty, rightRowIndices),
        'f32',
        'poverty150Pct at rightRowIndices: an exact gather of two readbacks.'
      )
    ];
    const description = query.describe({graphId: 'gpu-dataframe.data-frame-lookup-query-compiled'});
    return createDataFrameQueryLensRecord({
      scenarioId: 'gpu-dataframe.data-frame-lookup-query',
      element: 'GPUDataFrameLookupQuery',
      device,
      graphs: [description.intent, description.lowered, description.compiled],
      derivedFormulas: description.formulas,
      keySpaces: [
        ...description.keySpaces,
        {id: 'rightRow', label: 'Tract (right row)', size: poverty.length, indexSpace: '1d'}
      ],
      datumKeySpace: 'row',
      relations: description.relations,
      choices: [
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
