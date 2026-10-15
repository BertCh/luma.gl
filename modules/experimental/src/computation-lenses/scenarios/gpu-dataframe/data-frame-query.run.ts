// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './data-frame-query.example';
import {
  loadDataFrameLensData,
  createDataFrameLensGraph,
  readGPUVectorValues,
  createTableCapture,
  createDataFrameLensRecord
} from './dataframe-lens-support';

/** Executes the recorded tract query once and reads every displayed input and output from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const dataset = await loadDataFrameLensData('f-chicago-tracts');
  const columns = dataset.columns;
  const graph = createDataFrameLensGraph<GPUDataFrameQueryParameters>(
    device,
    'gpu-dataframe.data-frame-query'
  );
  const {query, tracts} = await runExample(device, {
    graph,
    tracts: {
      population: new Float32Array(columns.population),
      households: new Float32Array(columns.households),
      noVehicle: new Float32Array(columns.noVehicle),
      medianHouseholdIncome: new Float32Array(
        columns.medianHouseholdIncome.map(value => value ?? 0)
      ),
      incomeValidity: new Uint32Array(
        columns.medianHouseholdIncome.map(value => (value === null ? 0 : 1))
      )
    }
  });
  try {
    options?.signal?.throwIfAborted();
    const encoder = device.createCommandEncoder();
    try {
      query.encode(encoder, {minimumPopulation: 1000});
      device.submit(encoder.finish());
    } finally {
      encoder.destroy();
    }
    const captures = await Promise.all(
      ['population', 'households', 'noVehicle', 'medianHouseholdIncome'].map(async name => {
        const values = await readGPUVectorValues(tracts.table.gpuVectors[name], 'f32');
        if (name === 'medianHouseholdIncome') {
          const validity = await readGPUVectorValues(tracts.validity.medianHouseholdIncome!, 'u32');
          values.forEach((_, index) => {
            if (!validity[index]) values[index] = NaN;
          });
        }
        return createTableCapture({
          id: name,
          name,
          keySpace: 'row',
          role: 'input',
          values,
          elementType: 'f32'
        });
      })
    );
    for (const [name, vector, elementType, keySpace] of [
      ['noVehicleRate', query.table.gpuVectors['noVehicleRate'], 'f32', 'row'],
      ['keep', query.selectionMask, 'u32', 'row'],
      ['rowIndices', query.rowIndices, 'u32', 'compactSlot'],
      ['selectedCounts', query.selectedCounts, 'u32', 'batch']
    ] as const) {
      captures.push(
        createTableCapture({
          id: name,
          name,
          keySpace,
          role: 'output',
          values: await readGPUVectorValues(vector, elementType),
          elementType,
          ...(name === 'rowIndices'
            ? {
                meaning:
                  'Only the first selectedCounts[batch] slots are compact row ids; remaining slots are unused.'
              }
            : {})
        })
      );
    }
    const description = query.describe({graphId: 'gpu-dataframe.data-frame-query-compiled'});
    return createDataFrameLensRecord({
      scenarioId: 'gpu-dataframe.data-frame-query',
      device,
      graphs: [description.intent, description.lowered, description.compiled],
      formulas: description.formulas,
      keySpaces: [
        ...description.keySpaces,
        {
          id: 'compactSlot',
          label: 'Compacted row slot',
          size: query.rowIndices.length,
          indexSpace: '1d'
        },
        {id: 'batch', label: 'Source batch', size: query.selectedCounts.length, indexSpace: 'batch'}
      ],
      relations: description.relations,
      choices: [{stage: 'filter', name: 'minimumPopulation', value: 1000, sweepable: true}],
      captures
    });
  } finally {
    query.destroy();
    tracts.destroy();
  }
}
