// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {runExample} from './data-frame-histogram-query.example';
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

const BIN_COUNT = 8;
const DOMAIN = [0, 256_000] as const;

/** Executes the histogram once and reads the income column and every bin from GPU buffers. */
export async function run(device: Device, options?: {signal?: AbortSignal}) {
  options?.signal?.throwIfAborted();
  const columns = (await loadDataFrameLensData('f-chicago-tracts')).columns;
  const graph = createDataFrameLensGraph<GPUDataFrameQueryParameters>(
    device,
    'gpu-dataframe.data-frame-histogram-query'
  );
  const {query, tracts} = await runExample(device, {
    graph,
    tracts: {
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
    submitDataFrameQuery(device, query);
    const income = await readGPUVectorValues(
      tracts.table.gpuVectors['medianHouseholdIncome'],
      'f32'
    );
    const incomeValidity = await readGPUVectorValues(tracts.validity.medianHouseholdIncome!, 'u32');
    const captures = [
      createTableCapture({
        id: 'medianHouseholdIncome',
        name: 'medianHouseholdIncome',
        keySpace: 'row',
        role: 'input',
        values: applyValidity(income, incomeValidity),
        elementType: 'f32',
        meaning: 'NaN marks a suppressed Census estimate (invalid sidecar); histograms skip it.'
      })
    ];
    for (const [name, captureName] of [
      ['bin', 'binId'],
      ['count', 'count']
    ] as const) {
      captures.push(
        createTableCapture({
          id: captureName,
          name: captureName,
          keySpace: 'bin',
          role: 'output',
          values: await readGPUVectorValues(
            (query.table.gpuVectors as Record<string, GPUVector>)[name],
            'u32'
          ),
          elementType: 'u32'
        })
      );
    }
    const description = query.describe({
      graphId: 'gpu-dataframe.data-frame-histogram-query-compiled'
    });
    return createDataFrameQueryLensRecord({
      scenarioId: 'gpu-dataframe.data-frame-histogram-query',
      element: 'GPUDataFrameHistogramQuery',
      device,
      graphs: [description.intent, description.lowered, description.compiled],
      derivedFormulas: description.formulas,
      keySpaces: [
        ...description.keySpaces,
        {id: 'bin', label: 'Income bin', size: query.binCount, indexSpace: '1d'}
      ],
      datumKeySpace: 'bin',
      relations: description.relations,
      choices: [
        createDataFrameChoice('aggregate', 'bins', BIN_COUNT, {sweepable: true}),
        createDataFrameChoice('aggregate', 'domain', `${DOMAIN[0]} to ${DOMAIN[1]}`, {
          alternatives: ['edges: strictly increasing literal boundaries']
        })
      ],
      captures
    });
  } finally {
    query.destroy();
    tracts.destroy();
  }
}
