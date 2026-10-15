// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// biome-ignore-all format: the shown example is hand-formatted to fit the 25-line card limit

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUTemporalReductionParameterValues,
  GPUParameterBuffer,
  GPUTemporalReduction,
  GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

/** Observation columns the lens scenario reads from `f-chicago-nature.json`. */
export type TemporalReductionExampleInput = {
  graph: GPUCommandGraph<undefined>;
  observations: {
    /** Seconds since 2023-01-01T00:00 on the Chicago local clock. */
    timestamp: Float32Array;
    /** Iconic taxon group, 0 to 9. */
    category: Uint32Array;
    /** 1 when the community agreed on the species, else 0. */
    researchGrade: Float32Array;
  };
};

/** Weekly counts and research-grade extremes per taxon group, one slot per group and week. */
export async function runExample(
  device: Device,
  {graph, observations}: TemporalReductionExampleInput
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
  const createOutput = <Format extends 'float32' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
  // #region example
  const timestamps = importColumn('timestamp', observations.timestamp, 'float32');
  const cellIds = importColumn('category', observations.category, 'uint32');
  const values = importColumn('researchGrade', observations.researchGrade, 'float32');
  const cellCount = 10; // taxon groups
  const bucketCount = 53; // weeks from 2023-01-01
  const slotCount = cellCount * bucketCount;
  const parameters = new GPUParameterBuffer(device, {
    id: 'bucket-parameters',
    format: 'float32',
    length: GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
    values: getGPUTemporalReductionParameterValues(0, 7 * 86_400) // origin, width in seconds
  });
  const output = {
    counts: createOutput('counts', 'uint32', slotCount), // slot = category * 53 + week
    minimums: createOutput('minimums', 'float32', slotCount),
    maximums: createOutput('maximums', 'float32', slotCount),
    first: createOutput('first', 'float32', slotCount),
    last: createOutput('last', 'float32', slotCount),
    occupiedSlots: {
      ids: createOutput('occupiedIds', 'uint32', slotCount),
      status: {count: createOutput('occupiedCount', 'uint32', 1), overflow: createOutput('occupiedOverflow', 'uint32', 1)}
    }
  };
  graph.add(new GPUTemporalReduction({cellIds, timestamps, values, parameters: parameters.importToGraph(graph), cellCount, bucketCount, output}));
  const {counts, first, last} = output;
  // #endregion example
  return {
    timestamps,
    cellIds,
    values,
    slotCount,
    counts,
    minimums: output.minimums,
    maximums: output.maximums,
    first,
    last,
    occupiedIds: output.occupiedSlots.ids,
    occupiedCount: output.occupiedSlots.status.count,
    bucket: {cellCount, bucketCount, origin: 0, width: 7 * 86_400},
    destroy() {
      parameters.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}