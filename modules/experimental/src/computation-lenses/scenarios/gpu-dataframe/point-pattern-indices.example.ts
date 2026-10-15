// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUPointPatternIndicesParameterValues,
  GPUParameterBuffer,
  GPUPointPatternIndices,
  GPU_CLARK_EVANS_LENGTH,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  GPU_QUADRAT_STATISTICS_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';

/** Observation columns the lens scenario reads from `f-chicago-nature.json`. */
export type PointPatternIndicesExampleInput = {
  graph: GPUCommandGraph<undefined>;
  observations: {
    /** Interleaved projected x, y in metres. */
    positions: Float32Array;
  };
  /** Study window `[minX, minY, maxX, maxY]` in metres, from the data file. */
  studyWindow: readonly [number, number, number, number];
};

/** Clark-Evans nearest-neighbor index and an 8 x 10 quadrat test of iNaturalist observations. */
export async function runExample(
  device: Device,
  {graph, observations, studyWindow}: PointPatternIndicesExampleInput
) {
  const buffers: Buffer[] = [];
  const buffer = device.createBuffer({
    id: 'observations',
    data: observations.positions,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  buffers.push(buffer);
  const rowCount = observations.positions.length / 2;
  const positions = graph.createDataView(
    graph.importBuffer(
      {id: 'observations', byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    ),
    {format: 'float32x2', length: rowCount}
  );
  const quadratGrid = [8, 10] as const;
  const view = <Format extends 'float32' | 'uint32'>(id: string, format: Format, length: number) =>
    createTransientView(graph, id, format, length, Buffer.STORAGE | Buffer.COPY_SRC);
  const nearestNeighborDistances = view('nearestNeighborDistances', 'float32', rowCount);
  const nearestNeighborIds = view('nearestNeighborIds', 'uint32', rowCount);
  const clarkEvans = view('clarkEvans', 'float32', GPU_CLARK_EVANS_LENGTH);
  const quadratCounts = view('quadratCounts', 'uint32', quadratGrid[0] * quadratGrid[1]);
  const quadratStatistics = view('quadratStatistics', 'float32', GPU_QUADRAT_STATISTICS_LENGTH);
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'point-pattern-parameters',
    format: 'float32',
    length: GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
    values: getGPUPointPatternIndicesParameterValues({bounds: studyWindow, maximumDistance: 500})
  });
  graph.add(
    new GPUPointPatternIndices({
      positions,
      parameters: parameters.importToGraph(graph),
      gridSize: [32, 32],
      quadratGrid, // 8 x 10 quadrats
      output: {
        nearestNeighborDistances,
        nearestNeighborIds,
        clarkEvans,
        quadratCounts,
        quadratStatistics
      }
    })
  );
  // clarkEvans: [n, observed, expected, ratio, standardError, z, p], Donnelly edge correction
  // #endregion example
  return {
    positions,
    nearestNeighborDistances,
    nearestNeighborIds,
    clarkEvans,
    quadratCounts,
    quadratStatistics,
    quadratGrid,
    parameters,
    buffers
  };
}
