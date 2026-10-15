// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData} from '@luma.gl/gpgpu/gpu-data';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPURecordBatch, GPUTable} from '@luma.gl/experimental/gpu-tables';
import {GPUDataFrame, type GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';

/** Typed source array: the resident population of every Chicago tract. */
export type DataFrameAggregationQueryExampleInput = {
  graph: GPUCommandGraph<GPUDataFrameQueryParameters>;
  tracts: {population: Float32Array};
};

type TractColumns = {population: 'float32'};

function uploadTracts(
  device: Device,
  tracts: DataFrameAggregationQueryExampleInput['tracts']
): GPUDataFrame<TractColumns> {
  const batch = new GPURecordBatch<TractColumns>({
    gpuData: {
      population: new GPUData({
        buffer: device.createBuffer({
          id: 'tract-population',
          data: tracts.population,
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        }),
        format: 'float32',
        length: tracts.population.length,
        ownsBuffer: true
      })
    },
    fields: [{name: 'population', format: 'float32', nullable: false}]
  });
  return new GPUDataFrame({
    table: new GPUTable<TractColumns>({batches: [batch]}),
    ownership: 'owned'
  });
}

/** Uploads the populations and compiles one global reduction; the caller submits and reads it. */
export async function runExample(device: Device, input: DataFrameAggregationQueryExampleInput) {
  const tracts = uploadTracts(device, input.tracts);
  try {
    // #region example
    const query = tracts
      .aggregate({
        rows: 'count',
        sum: {sum: 'population'},
        minimum: {min: 'population'},
        maximum: {max: 'population'},
        mean: {mean: 'population'}
      })
      .compile(input.graph);
    return {query, tracts};
    // #endregion example
  } catch (error) {
    tracts.destroy();
    throw error;
  }
}
