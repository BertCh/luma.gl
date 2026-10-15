// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPURecordBatch, GPUTable} from '@luma.gl/experimental/gpu-tables';
import {GPUDataFrame, type GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';

/** Typed source arrays: wildlife observations (left) and the census tracts they fall in (right). */
export type DataFrameJoinQueryExampleInput = {
  graph: GPUCommandGraph<GPUDataFrameQueryParameters>;
  observations: {tract: Uint32Array; category: Uint32Array};
  tracts: {tract: Uint32Array; poverty150Pct: Float32Array; povertyValidity: Uint32Array};
};

type ObservationColumns = {tract: 'uint32'; category: 'uint32'};
type TractColumns = {tract: 'uint32'; poverty150Pct: 'float32'};

function createUpload(device: Device) {
  return <Format extends 'float32' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ): GPUData<Format> =>
    new GPUData({
      buffer: device.createBuffer({
        id: name,
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      }),
      format,
      length: values.length,
      ownsBuffer: true
    });
}

function uploadObservations(
  device: Device,
  observations: DataFrameJoinQueryExampleInput['observations']
): GPUDataFrame<ObservationColumns> {
  const upload = createUpload(device);
  const batch = new GPURecordBatch<ObservationColumns>({
    gpuData: {
      tract: upload('observation-tract', observations.tract, 'uint32'),
      category: upload('observation-category', observations.category, 'uint32')
    },
    fields: [
      {name: 'tract', format: 'uint32', nullable: false},
      {name: 'category', format: 'uint32', nullable: false}
    ]
  });
  return new GPUDataFrame({
    table: new GPUTable<ObservationColumns>({batches: [batch]}),
    ownership: 'owned'
  });
}

function uploadTracts(
  device: Device,
  tracts: DataFrameJoinQueryExampleInput['tracts']
): GPUDataFrame<TractColumns> {
  const upload = createUpload(device);
  const batch = new GPURecordBatch<TractColumns>({
    gpuData: {
      tract: upload('tract-key', tracts.tract, 'uint32'),
      poverty150Pct: upload('tract-poverty150Pct', tracts.poverty150Pct, 'float32')
    },
    fields: [
      {name: 'tract', format: 'uint32', nullable: false},
      {name: 'poverty150Pct', format: 'float32', nullable: true}
    ]
  });
  const validity = new GPUVector({
    name: 'povertyValidity',
    type: 'data',
    format: 'uint32',
    data: [upload('tract-povertyValidity', tracts.povertyValidity, 'uint32')],
    ownsData: true
  });
  return new GPUDataFrame({
    table: new GPUTable<TractColumns>({batches: [batch]}),
    validity: {poverty150Pct: validity},
    ownership: 'owned'
  });
}

/** Uploads both tables and compiles one unique-right inner join; the caller submits and reads it. */
export async function runExample(device: Device, input: DataFrameJoinQueryExampleInput) {
  const observations = uploadObservations(device, input.observations);
  let tracts: GPUDataFrame<TractColumns> | undefined;
  try {
    tracts = uploadTracts(device, input.tracts);
    // #region example
    const query = observations
      .innerJoin(tracts, {leftOn: 'tract', rightOn: 'tract'})
      .compile(input.graph);
    return {query, observations, tracts};
    // #endregion example
  } catch (error) {
    observations.destroy();
    tracts?.destroy();
    throw error;
  }
}
