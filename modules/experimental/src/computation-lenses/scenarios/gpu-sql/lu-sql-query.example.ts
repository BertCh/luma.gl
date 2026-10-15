// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData} from '@luma.gl/gpgpu/gpu-data';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPURecordBatch, GPUTable, type GPUTypeMap} from '@luma.gl/experimental/gpu-tables';
import {GPUDataFrame, type GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {LuSQLContext} from '@luma.gl/experimental/gpu-sql';

/** Typed source arrays: the Census counts of every Chicago tract. */
export type LuSQLQueryExampleInput = {
  graph: GPUCommandGraph<GPUDataFrameQueryParameters>;
  tracts: {population: Float32Array; households: Float32Array; noVehicle: Float32Array};
};

// LuSQLContext resolves column names at run time, so its tables are untyped dataframes.
function uploadTracts(device: Device, tracts: LuSQLQueryExampleInput['tracts']): GPUDataFrame {
  const upload = (name: string, values: Float32Array): GPUData<'float32'> =>
    new GPUData({
      buffer: device.createBuffer({
        id: `tract-${name}`,
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      }),
      format: 'float32',
      length: values.length,
      ownsBuffer: true
    });
  const batch = new GPURecordBatch<GPUTypeMap>({
    gpuData: {
      population: upload('population', tracts.population),
      households: upload('households', tracts.households),
      noVehicle: upload('noVehicle', tracts.noVehicle)
    },
    fields: [
      {name: 'population', format: 'float32', nullable: false},
      {name: 'households', format: 'float32', nullable: false},
      {name: 'noVehicle', format: 'float32', nullable: false}
    ]
  });
  return new GPUDataFrame({
    table: new GPUTable<GPUTypeMap>({batches: [batch]}),
    ownership: 'owned'
  });
}

/** Uploads the tracts and plans and compiles one SQL statement; the caller submits and reads it. */
export async function runExample(device: Device, input: LuSQLQueryExampleInput) {
  const tracts = uploadTracts(device, input.tracts);
  try {
    // #region example
    const query = new LuSQLContext({tracts}).query(
      `SELECT population, noVehicle / households AS noVehicleRate
       FROM tracts
       WHERE population > :minimumPopulation`,
      {parameters: {minimumPopulation: 1000}}
    );
    const compiled = query.compile(input.graph);
    return {query, compiled, tracts};
    // #endregion example
  } catch (error) {
    tracts.destroy();
    throw error;
  }
}
