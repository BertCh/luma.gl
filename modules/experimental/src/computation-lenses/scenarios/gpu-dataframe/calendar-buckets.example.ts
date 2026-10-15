// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUCalendarBucketsParameterValues,
  getInt64TimeWords,
  GPUCalendarBuckets,
  GPUParameterBuffer,
  GPU_CALENDAR_BUCKETS_MATRIX_LENGTH,
  GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

/** Observation columns the lens scenario reads from `f-chicago-nature.json`. */
export type CalendarBucketsExampleInput = {
  graph: GPUCommandGraph<undefined>;
  observations: {
    /** Chicago local clock time stored as epoch milliseconds, as if it were UTC. */
    epochMilliseconds: BigInt64Array;
  };
};

/** Month, weekday and hour of 1,500 observations, plus a weekday-by-hour count matrix. */
export async function runExample(
  device: Device,
  {graph, observations}: CalendarBucketsExampleInput
) {
  const buffers: Buffer[] = [];
  const importColumn = <Format extends 'uint32x2'>(
    id: string,
    data: Uint32Array,
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
    return graph.createDataView(handle, {format, length: data.length / 2});
  };
  const createOutput = (id: string, length: number) =>
    createTransientView(graph, id, 'uint32', length, Buffer.STORAGE | Buffer.COPY_SRC);
  // #region example
  const timestamps = importColumn(
    'timestamps',
    getInt64TimeWords(observations.epochMilliseconds), // (low, high) words per row
    'uint32x2'
  );
  const parameters = new GPUParameterBuffer(device, {
    id: 'calendar-parameters',
    format: 'sint32',
    length: GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH,
    values: getGPUCalendarBucketsParameterValues(0, 0) // already local clock; weeks start Monday
  });
  const output = {
    month: createOutput('month', timestamps.length),
    weekday: createOutput('weekday', timestamps.length), // 0 is Monday
    hour: createOutput('hour', timestamps.length),
    hourWeekdayCounts: createOutput('hourWeekdayCounts', GPU_CALENDAR_BUCKETS_MATRIX_LENGTH)
  };
  graph.add(
    new GPUCalendarBuckets({timestamps, parameters: parameters.importToGraph(graph), output})
  );
  const {month, weekday, hourWeekdayCounts} = output;
  // #endregion example
  return {
    timestamps,
    month,
    weekday,
    hour: output.hour,
    hourWeekdayCounts,
    destroy() {
      parameters.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}
