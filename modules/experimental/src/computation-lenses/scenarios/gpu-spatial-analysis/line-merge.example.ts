// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPULineMerge, type GPULineGeometryFields} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded Chicago geometry and parameters for GPULineMerge. */
export type LineMergeExampleInput = {
  roads: GPULineGeometryFields;
};

/** Runs GPULineMerge and returns the views drawn or inspected by its card. */
export async function runExample(graph: GPUCommandGraph<undefined>, input: LineMergeExampleInput) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {roads} = input;
  const roadCount = roads.lineOffsets.length - 1;
  // #region example
  const output = {
    positions: createTransientView(
      graph,
      'mergedRoads',
      'float32x2',
      roads.positions.length,
      usage
    ),
    chainOffsets: createTransientView(graph, 'chainOffsets', 'uint32', roadCount + 1, usage),
    lineChains: createTransientView(graph, 'lineChains', 'uint32', roadCount, usage),
    status: {count: createTransientView(graph, 'count', 'uint32', 1, usage)}
  };
  graph.add(new GPULineMerge({...roads, directed: false, output}));
  const {positions: mergedRoads, chainOffsets} = output;
  // #endregion example
  return {...output, mergedRoads, chainOffsets};
}
