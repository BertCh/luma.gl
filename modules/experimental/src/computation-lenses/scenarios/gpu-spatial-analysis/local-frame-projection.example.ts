// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULocalFrameProjection,
  type GPULineGeometryFields,
  type GPULocalFrame
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded Chicago geometry and parameters for GPULocalFrameProjection. */
export type LocalFrameProjectionExampleInput = {
  roads: GPULineGeometryFields;
  frame: GPULocalFrame;
  bounds: [number, number, number, number];
};

/** Runs GPULocalFrameProjection and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: LocalFrameProjectionExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {roads, frame, bounds} = input;
  // #region example
  const output = createTransientView(
    graph,
    'projectedRoads',
    'float32x2',
    roads.positions.length,
    usage
  );
  const validity = createTransientView(graph, 'validity', 'uint32', roads.positions.length, usage);
  const projection = new GPULocalFrameProjection({
    frame,
    bounds,
    positions: roads.positions,
    output,
    validity,
    tolerance: 0.01
  });
  graph.add(projection);
  // Input is lon/lat degrees; output is local azimuthal-equidistant metres.
  const projectedRoads = output;
  // #endregion example
  return {projectedRoads, validity, projection};
}
