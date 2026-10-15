// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPURecordBatch, GPUTable} from '@luma.gl/experimental/gpu-tables';
import {
  GPUDataFrame,
  and,
  column,
  parameter,
  type GPUDataFrameQueryParameters
} from '@luma.gl/experimental/gpu-dataframe';

/** Typed source arrays, with an explicit mask for suppressed Census income estimates. */
export type DataFrameQueryExampleInput = {
  graph: GPUCommandGraph<GPUDataFrameQueryParameters>;
  tracts: {
    population: Float32Array;
    households: Float32Array;
    noVehicle: Float32Array;
    medianHouseholdIncome: Float32Array;
    incomeValidity: Uint32Array;
  };
};

type TractColumns = {
  population: 'float32';
  households: 'float32';
  noVehicle: 'float32';
  medianHouseholdIncome: 'float32';
};

function uploadTracts(
  device: Device,
  tracts: DataFrameQueryExampleInput['tracts']
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
      population: upload('population', tracts.population, 'float32'),
      households: upload('households', tracts.households, 'float32'),
      noVehicle: upload('noVehicle', tracts.noVehicle, 'float32'),
      medianHouseholdIncome: upload(
        'medianHouseholdIncome',
        tracts.medianHouseholdIncome,
        'float32'
      )
    },
    fields: ['population', 'households', 'noVehicle', 'medianHouseholdIncome'].map(name => ({
      name,
      format: 'float32',
      nullable: name === 'medianHouseholdIncome'
    }))
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

/** Uploads Census inputs and compiles one immutable row query; the caller submits and reads it. */
export async function runExample(device: Device, input: DataFrameQueryExampleInput) {
  const tracts = uploadTracts(device, input.tracts);
  try {
    // #region example
    const query = tracts
      .withColumn('noVehicleRate', column('noVehicle').divide(column('households')))
      .filter(
        and(
          column('population').greaterThan(parameter('minimumPopulation', 1000)),
          column('medianHouseholdIncome').isValid()
        )
      )
      .select(['population', 'noVehicleRate', 'medianHouseholdIncome'])
      .compile(input.graph);
    return {query, tracts};
    // #endregion example
  } catch (error) {
    tracts.destroy();
    throw error;
  }
}
