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
  GPUCoverageDissolve,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded Chicago community areas and per-area labels for GPUCoverageDissolve. */
export type CoverageDissolveExampleInput = {
  areas: GPUPolygonGeometryFields;
  labels: GraphDataView<'uint32'>;
  ringCapacity: number;
  vertexCapacity: number;
};

/** Runs GPUCoverageDissolve and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: CoverageDissolveExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {areas, labels, ringCapacity, vertexCapacity} = input;
  const output = {
    ringOffsets: createTransientView(graph, 'dissolvedOffsets', 'uint32', ringCapacity + 1, usage),
    positions: createTransientView(graph, 'dissolved', 'float32x2', vertexCapacity, usage),
    ringGroups: createTransientView(graph, 'ringLabels', 'uint32', ringCapacity, usage),
    status: {
      count: createTransientView(graph, 'ringCount', 'uint32', 1, usage),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage),
      boundarySegmentCount: createTransientView(graph, 'boundarySegmentCount', 'uint32', 1, usage)
    }
  };
  // #region example
  // One uint32 label per area; areas with equal labels merge. Coordinates are UTM 16N metres.
  graph.add(
    new GPUCoverageDissolve({
      ...areas,
      labels,
      splitTouchingRings: true,
      output
    })
  );
  const {positions: dissolved, ringOffsets} = output;
  // #endregion example
  return {...output, dissolved, ringOffsets};
}
