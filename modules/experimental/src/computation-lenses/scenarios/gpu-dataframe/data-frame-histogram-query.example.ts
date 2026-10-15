// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPURecordBatch, GPUTable} from '@luma.gl/experimental/gpu-tables';
import {GPUDataFrame, type GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';

/** Typed source arrays, with an explicit mask for suppressed Census income estimates. */
export type DataFrameHistogramQueryExampleInput = {
  graph: GPUCommandGraph<GPUDataFrameQueryParameters>;
  tracts: {medianHouseholdIncome: Float32Array; incomeValidity: Uint32Array};
};

type TractColumns = {medianHouseholdIncome: 'float32'};

function uploadTracts(
  device: Device,
  tracts: DataFrameHistogramQueryExampleInput['tracts']
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
    gpuData: {
      medianHouseholdIncome: upload(
        'medianHouseholdIncome',
        tracts.medianHouseholdIncome,
        'float32'
      )
    },
    fields: [{name: 'medianHouseholdIncome', format: 'float32', nullable: true}]
  });
  const validity = new GPUVector({
    name: 'incomeValidity',
    type: 'data',
    format: 'uint32',
    data: [upload('incomeValidity', tracts.incomeValidity, 'uint32')],
    ownsData: true
  });
  return new GPUDataFrame({
    table: new GPUTable<TractColumns>({batches: [batch]}),
    validity: {medianHouseholdIncome: validity},
    ownership: 'owned'
  });
}

/** Uploads the incomes and compiles one fixed-domain histogram; the caller submits and reads it. */
export async function runExample(device: Device, input: DataFrameHistogramQueryExampleInput) {
  const tracts = uploadTracts(device, input.tracts);
  try {
    // #region example
    const query = tracts
      .histogram('medianHouseholdIncome', {bins: 8, domain: [0, 256_000]})
      .compile(input.graph);
    return {query, tracts};
    // #endregion example
  } catch (error) {
    tracts.destroy();
    throw error;
  }
}
