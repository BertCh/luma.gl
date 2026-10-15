// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUNeighborSearchParameterValues,
  getGPUSpaceTimeParameterValues,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  GPUKnoxTest,
  GPUNeighborSearch,
  GPUParameterBuffer,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-nature.json`. */
export type KnoxTestExampleInput = {
  /** iNaturalist observations, projected to local-frame metres. */
  observations: GraphDataView<'float32x2'>;
  /** Observation day of 2023 (0 = 1 January), one per observation. */
  days: GraphDataView<'float32'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
  /** Padded bounds `[minX, minY, maxX, maxY]` of the observations, in frame metres. */
  bounds: readonly [number, number, number, number];
  /** Slots reserved for the close-in-space pairs (both directions of every pair). */
  pairCapacity: number;
};

/** Do observations that are within 250 m of each other also fall within 7 days of each other? */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {observations, days, spatialContext, bounds, pairCapacity}: KnoxTestExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const count = observations.length;
  const closeInSpace = {
    offsets: createTransientView(graph, 'pairOffsets', 'uint32', count + 1, usage),
    neighbors: createTransientView(graph, 'pairNeighbors', 'uint32', pairCapacity, usage),
    weights: createTransientView(graph, 'pairWeights', 'float32', pairCapacity, usage),
    status: {overflow: createTransientView(graph, 'pairOverflow', 'uint32', 1, usage)}
  };
  const statistics = createTransientView(graph, 'statistics', 'uint32', 200, usage);
  const summary = createTransientView(
    graph,
    'summary',
    'float32',
    GPU_SPACE_TIME_SUMMARY_LENGTH,
    usage
  );
  const parameterBuffers: GPUParameterBuffer[] = [];
  // Packs per-frame parameter values into a buffer the lens scenario destroys after the run.
  const createParameters = ((values: Float32Array | Uint32Array) => {
    const id = `parameters-${parameterBuffers.length}`;
    const buffer =
      values instanceof Float32Array
        ? new GPUParameterBuffer(graph.device, {
            id,
            format: 'float32',
            length: values.length,
            values
          })
        : new GPUParameterBuffer(graph.device, {
            id,
            format: 'uint32',
            length: values.length,
            values
          });
    parameterBuffers.push(buffer);
    return buffer.importToGraph(graph);
  }) as {
    (values: Float32Array): GraphDataView<'float32'>;
    (values: Uint32Array): GraphDataView<'uint32'>;
  };
  // #region example
  const searchValues = getGPUNeighborSearchParameterValues({bounds, radius: 250});
  const knoxValues = getGPUSpaceTimeParameterValues({
    seed: 1,
    permutationCount: 199,
    timeThreshold: 7
  });
  const search = new GPUNeighborSearch({
    spatialContext,
    mode: 'radius',
    positions: observations,
    gridSize: [32, 32],
    parameters: createParameters(searchValues),
    output: closeInSpace
  });
  graph.add(search);
  graph.add(
    new GPUKnoxTest({
      pairs: search.outputWeights,
      times: days,
      parameters: createParameters(knoxValues),
      maximumPermutations: 199,
      output: {statistics, summary}
    })
  );
  // statistics[0]: observed Knox count; statistics[1..199]: counts after permuting the days
  // #endregion example
  return {closeInSpace, statistics, summary, parameterBuffers};
}
