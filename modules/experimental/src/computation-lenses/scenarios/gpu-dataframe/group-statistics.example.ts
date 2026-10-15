// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUGroupStatistics} from '@luma.gl/experimental/gpu-dataframe';

/** Tract columns the lens scenario reads from `f-chicago-tracts.json`; suppressed estimates are NaN. */
export type GroupStatisticsExampleInput = {
  graph: GPUCommandGraph<undefined>;
  tracts: {
    /** Community area (1 to 77) of each tract, used as the group key. */
    communityArea: Uint32Array;
    /** Share of residents below 150% of the poverty line. */
    poverty150Pct: Float32Array;
  };
  /** Number of distinct community areas; it is the capacity of the group table. */
  groupCount: number;
};

/** Mean, sample variance and median of tract poverty for each of the 77 community areas. */
export async function runExample(
  device: Device,
  {graph, tracts, groupCount}: GroupStatisticsExampleInput
) {
  const buffers: Buffer[] = [];
  const importColumn = <Format extends 'float32' | 'uint32'>(
    id: string,
    data: Float32Array | Uint32Array,
    format: Format
  ) => {
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
    return graph.createDataView(handle, {format, length: data.length});
  };
  const view = <Format extends 'float32' | 'uint32'>(id: string, format: Format, length: number) =>
    createTransientView(graph, id, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
  // #region example
  const keys = importColumn('communityArea', tracts.communityArea, 'uint32');
  const poverty = importColumn('poverty150Pct', tracts.poverty150Pct, 'float32');
  const column = {
    counts: view('counts', 'uint32', groupCount), // finite values per group
    means: view('means', 'float32', groupCount),
    variances: view('variances', 'float32', groupCount),
    medians: view('medians', 'float32', groupCount)
  };
  const groupKeys = view('groupKeys', 'uint32', groupCount);
  const groupRows = view('groupRows', 'uint32', groupCount); // valid rows per group
  graph.add(
    new GPUGroupStatistics({
      keys,
      columns: [
        {values: poverty, statistics: ['count', 'mean', 'variance', 'median'], output: column}
      ],
      variance: 'sample', // n - 1; 'population' divides by n
      output: {
        keys: groupKeys,
        counts: groupRows,
        status: {count: view('groupStatus', 'uint32', 1), overflow: view('overflow', 'uint32', 1)}
      }
    })
  );
  // row g of every output describes the g-th smallest community area
  // #endregion example
  return {keys, poverty, groupKeys, groupRows, ...column, buffers};
}
