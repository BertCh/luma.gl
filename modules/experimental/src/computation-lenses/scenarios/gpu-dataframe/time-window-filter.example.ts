// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUTimeWindowFilterParameterValues,
  GPUParameterBuffer,
  GPUTimeWindowFilter,
  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
  splitTimestamps
} from '@luma.gl/experimental/gpu-dataframe';

/** Observation columns the lens scenario reads from `f-chicago-nature.json`. */
export type TimeWindowFilterExampleInput = {
  graph: GPUCommandGraph<undefined>;
  observations: {
    /** Seconds since 2023-01-01T00:00 on the Chicago local clock. */
    timestamp: Float64Array;
  };
};

/** Keeps the observations of a summer window, 2023-06-01 to 2023-09-01 inclusive. */
export async function runExample(
  device: Device,
  {graph, observations}: TimeWindowFilterExampleInput
) {
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
  const createOutput = (id: string, length: number) =>
    createTransientView(graph, id, 'uint32', length, Buffer.STORAGE | Buffer.COPY_SRC);
  // #region example
  const {high, low} = splitTimestamps(observations.timestamp); // double-single seconds
  const timestamps = importColumn('timestamps', high);
  const day = 86_400;
  const window = new GPUParameterBuffer(device, {
    id: 'time-window',
    format: 'float32',
    length: GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
    values: getGPUTimeWindowFilterParameterValues({start: 151 * day, end: 243 * day})
  });
  const output = {
    ids: createOutput('ids', timestamps.length),
    status: {count: createOutput('count', 1), overflow: createOutput('overflow', 1)},
    outputMask: createOutput('outputMask', timestamps.length) // 1 inside the window
  };
  graph.add(
    new GPUTimeWindowFilter({
      timestamps,
      timestampsLow: importColumn('timestampsLow', low),
      window: window.importToGraph(graph),
      output
    })
  );
  const {ids, status, outputMask} = output;
  // #endregion example
  return {
    timestamps,
    ids,
    count: status.count,
    outputMask,
    window: {start: 151 * day, end: 243 * day},
    destroy() {
      window.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
