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
  GPULinearReferencing,
  type GPULineGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded Chicago geometry and parameters for GPULinearReferencing. */
export type LinearReferencingExampleInput = {
  roads: GPULineGeometryFields;
  places: GraphDataView<'float32x2'>;
  radius: GraphDataView<'float32'>;
};

/** Runs GPULinearReferencing and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: LinearReferencingExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {roads, places, radius} = input;
  const placeCount = places.length;
  // #region example
  const output = {
    footPoints: createTransientView(graph, 'footPoints', 'float32x2', placeCount, usage),
    lineIndices: createTransientView(graph, 'lineIndices', 'uint32', placeCount, usage),
    measures: createTransientView(graph, 'measures', 'float32', placeCount, usage),
    distances: createTransientView(graph, 'distances', 'float32', placeCount, usage),
    status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage)}
  };
  // Search within 500 m of each of the 40 selected places.
  graph.add(
    new GPULinearReferencing({
      ...roads,
      queryPositions: places,
      radius,
      candidateCapacity: roads.positions.length * placeCount,
      output
    })
  );
  const {footPoints, measures, distances} = output;
  // #endregion example
  return {...output, footPoints, measures, distances};
}
