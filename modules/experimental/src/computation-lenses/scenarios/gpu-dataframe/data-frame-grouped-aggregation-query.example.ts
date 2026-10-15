// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData} from '@luma.gl/gpgpu/gpu-data';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPURecordBatch, GPUTable} from '@luma.gl/experimental/gpu-tables';
import {GPUDataFrame, type GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';

/** Typed source arrays: the dense category of each observation and its research-grade flag. */
export type DataFrameGroupedAggregationQueryExampleInput = {
  graph: GPUCommandGraph<GPUDataFrameQueryParameters>;
  observations: {category: Uint32Array; researchGrade: Float32Array};
};

type ObservationColumns = {category: 'uint32'; researchGrade: 'float32'};

function uploadObservations(
  device: Device,
  observations: DataFrameGroupedAggregationQueryExampleInput['observations']
): GPUDataFrame<ObservationColumns> {
  const upload = <Format extends 'float32' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ): GPUData<Format> =>
    new GPUData({
      buffer: device.createBuffer({
        id: `observation-${name}`,
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      }),
      format,
      length: values.length,
      ownsBuffer: true
    });
  const batch = new GPURecordBatch<ObservationColumns>({
    gpuData: {
      category: upload('category', observations.category, 'uint32'),
      researchGrade: upload('researchGrade', observations.researchGrade, 'float32')
    },
    fields: [
      {name: 'category', format: 'uint32', nullable: false},
      {name: 'researchGrade', format: 'float32', nullable: false}
    ]
  });
  return new GPUDataFrame({
    table: new GPUTable<ObservationColumns>({batches: [batch]}),
    ownership: 'owned'
  });
}

/** Uploads the observations and compiles one grouped aggregation; the caller submits and reads it. */
export async function runExample(
  device: Device,
  input: DataFrameGroupedAggregationQueryExampleInput
) {
  const observations = uploadObservations(device, input.observations);
  try {
    // #region example
    const query = observations
      .groupBy('category', {groupCount: 10})
      .aggregate({
        count: 'count',
        sum: {sum: 'researchGrade'},
        mean: {mean: 'researchGrade'}
      })
      .compile(input.graph);
    return {query, observations};
    // #endregion example
  } catch (error) {
    observations.destroy();
    throw error;
  }
}
