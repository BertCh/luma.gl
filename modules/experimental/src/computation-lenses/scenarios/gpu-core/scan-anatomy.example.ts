// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUMask,
  GPUCompaction,
  GPUReduction,
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';

/** Uploaded station inputs and caller-owned destination handles. */
export type ScanAnatomyExampleInput = {
  stationRows: GraphDataView<'uint32'>;
  rainfallMm: GraphDataView<'float32'>;
  wet: GraphDataView<'uint32'>;
  high: GraphDataView<'uint32'>;
  idsBuffer: GraphBufferHandle;
  countBuffer: GraphBufferHandle;
  totalRainBuffer: GraphBufferHandle;
};

/** Adds the real station computation; the shared harness owns execution and readback. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {
    stationRows,
    rainfallMm,
    wet,
    high,
    idsBuffer,
    countBuffer,
    totalRainBuffer
  }: ScanAnatomyExampleInput
) {
  // #region example
  const maskId = 'heavyRainHighStations';
  const heavyRainHighStations = createTransientView(graph, maskId, 'uint32', wet.length);
  const ids = graph.createDataView(idsBuffer, {format: 'uint32', length: wet.length});
  const count = graph.createDataView(countBuffer, {format: 'uint32', length: 1});
  const totalRain = graph.createDataView(totalRainBuffer, {format: 'float32', length: 1});
  graph.add(new GPUMask({inputs: [wet, high], output: heavyRainHighStations, operation: 'and'}));
  const selection = {input: stationRows, flags: heavyRainHighStations, output: ids, count};
  graph.add(new GPUCompaction(selection));
  graph.add(
    new GPUReduction({
      input: rainfallMm,
      mask: heavyRainHighStations,
      output: totalRain,
      operation: 'sum'
    })
  );
  // ids[0..count) selects wet, high stations; totalRain sums their measured rainfall in mm.
  return {ids, count, totalRain};
  // #endregion example
}
