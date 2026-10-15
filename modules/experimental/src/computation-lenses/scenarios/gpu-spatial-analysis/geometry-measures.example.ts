// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGeometryMeasures,
  type GPULineGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Uploaded Chicago geometry and parameters for GPUGeometryMeasures. */
export type GeometryMeasuresExampleInput = {
  roads: GPULineGeometryFields;
};

/** Runs GPUGeometryMeasures and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: GeometryMeasuresExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {roads} = input;
  // #region example
  const lengths = createTransientView(
    graph,
    'lengths',
    'float32',
    roads.lineOffsets.length - 1,
    usage
  );
  graph.add(
    new GPUGeometryMeasures({
      ...roads,
      geometryType: 'lines',
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'ellipsoidal',
        units: 'meters'
      },
      ellipsoidalEdgeModel: 'geodesic',
      output: {lengths}
    })
  );
  // Draw the same road paths coloured by WGS84 geodesic length in metres.
  const measuredRoads = roads.positions;
  // #endregion example
  return {lengths, measuredRoads};
}
