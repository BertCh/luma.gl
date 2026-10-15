// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_COLUMN_PROFILE_STATISTIC_COUNT,
  GPUColumnProfile
} from '@luma.gl/experimental/gpu-dataframe';

/** Tract columns the lens scenario reads from `f-chicago-tracts.json`; suppressed estimates are NaN. */
export type ColumnProfileExampleInput = {
  graph: GPUCommandGraph<undefined>;
  tracts: {
    poverty150Pct: Float32Array;
    uninsuredPct: Float32Array;
    age65Pct: Float32Array;
    medianHouseholdIncome: Float32Array;
  };
};

/** A statistics panel for four tract columns: counts, nulls, extent, moments, histogram, distinct. */
export async function runExample(device: Device, {graph, tracts}: ColumnProfileExampleInput) {
  const buffers: Buffer[] = [];
  const importColumn = (id: string, data: Float32Array) => {
    const buffer = device.createBuffer({
      id,
      data,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format: 'float32', length: data.length});
  };
  const createOutput = <Format extends 'float32' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
  // #region example
  const columns = Object.entries(tracts).map(([name, values]) => ({
    name,
    values: importColumn(name, values)
  }));
  const histogramBinCount = 8;
  const output = {
    // GPU_COLUMN_PROFILE_STATISTIC_COUNT (11) floats per column, column-major
    statistics: createOutput(
      'statistics',
      'float32',
      columns.length * GPU_COLUMN_PROFILE_STATISTIC_COUNT
    ),
    histograms: createOutput('histograms', 'uint32', columns.length * histogramBinCount)
  };
  graph.add(
    new GPUColumnProfile({
      columns: columns.map(({values}) => ({values})), // NaN rows are nulls
      histogramBinCount,
      output
    })
  );
  const {statistics, histograms} = output;
  // #endregion example
  return {
    columns,
    histogramBinCount,
    statistics,
    histograms,
    destroy() {
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
