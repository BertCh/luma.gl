// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPURecordBatch, GPUTable} from '@luma.gl/experimental/gpu-tables';
import {GPUDataFrame, type GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';

/** One record batch of poverty rates, with a mask for the tracts that have no estimate. */
export type DataFrameGlobalSortQueryBatch = {
  poverty150Pct: Float32Array;
  povertyValidity: Uint32Array;
};

/** Typed source arrays: the tracts arrive as several record batches that are never concatenated. */
export type DataFrameGlobalSortQueryExampleInput = {
  graph: GPUCommandGraph<GPUDataFrameQueryParameters>;
  tractBatches: readonly DataFrameGlobalSortQueryBatch[];
};

type TractColumns = {poverty150Pct: 'float32'};

function uploadTracts(
  device: Device,
  tractBatches: readonly DataFrameGlobalSortQueryBatch[]
): GPUDataFrame<TractColumns> {
  const upload = <Format extends 'float32' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ): GPUData<Format> =>
    new GPUData({
      buffer: device.createBuffer({
        id: `tract-${name}`,
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      }),
      format,
      length: values.length,
      ownsBuffer: true
    });
  const batches = tractBatches.map(
    tracts =>
      new GPURecordBatch<TractColumns>({
        gpuData: {poverty150Pct: upload('poverty150Pct', tracts.poverty150Pct, 'float32')},
        fields: [{name: 'poverty150Pct', format: 'float32', nullable: true}]
      })
  );
  const validity = new GPUVector({
    name: 'povertyValidity',
    type: 'data',
    format: 'uint32',
    data: tractBatches.map(tracts => upload('povertyValidity', tracts.povertyValidity, 'uint32')),
    ownsData: true
  });
  return new GPUDataFrame({
    table: new GPUTable<TractColumns>({batches}),
    validity: {poverty150Pct: validity},
    ownership: 'owned'
  });
}

/** Uploads the tract batches and compiles one global top-K; the caller submits and reads it. */
export async function runExample(device: Device, input: DataFrameGlobalSortQueryExampleInput) {
  const tracts = uploadTracts(device, input.tractBatches);
  try {
    // #region example
    const query = tracts.topKGlobal('poverty150Pct', 10, {nulls: 'last'}).compile(input.graph);
    return {query, tracts};
    // #endregion example
  } catch (error) {
    tracts.destroy();
    throw error;
  }
}
