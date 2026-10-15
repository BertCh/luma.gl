// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPURecordBatch, GPUTable} from '@luma.gl/experimental/gpu-tables';
import {GPUDataFrame, type GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';

/** Typed source arrays, with an explicit mask for the tracts that have no poverty estimate. */
export type DataFrameSortQueryExampleInput = {
  graph: GPUCommandGraph<GPUDataFrameQueryParameters>;
  tracts: {poverty150Pct: Float32Array; povertyValidity: Uint32Array};
};

type TractColumns = {poverty150Pct: 'float32'};

function uploadTracts(
  device: Device,
  tracts: DataFrameSortQueryExampleInput['tracts']
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
  const batch = new GPURecordBatch<TractColumns>({
    gpuData: {poverty150Pct: upload('poverty150Pct', tracts.poverty150Pct, 'float32')},
    fields: [{name: 'poverty150Pct', format: 'float32', nullable: true}]
  });
  const validity = new GPUVector({
    name: 'povertyValidity',
    type: 'data',
    format: 'uint32',
    data: [upload('povertyValidity', tracts.povertyValidity, 'uint32')],
    ownsData: true
  });
  return new GPUDataFrame({
    table: new GPUTable<TractColumns>({batches: [batch]}),
    validity: {poverty150Pct: validity},
    ownership: 'owned'
  });
}

/** Uploads the poverty rates and compiles one stable top-K sort; the caller submits and reads it. */
export async function runExample(device: Device, input: DataFrameSortQueryExampleInput) {
  const tracts = uploadTracts(device, input.tracts);
  try {
    // #region example
    const query = tracts
      .sortBy('poverty150Pct', {direction: 'descending', nulls: 'last'})
      .topK(10)
      .compile(input.graph);
    return {query, tracts};
    // #endregion example
  } catch (error) {
    tracts.destroy();
    throw error;
  }
}
