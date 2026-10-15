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
  GPULonLatPolygonPreparation,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Harbour zones in lon/lat degrees, the degree step and the output vertex capacity. */
export type LonLatPolygonPreparationExampleInput = {
  zones: GPUPolygonGeometryFields;
  /** `[maximumSegmentDegrees, 0, 0, 0]` from `getGPULonLatPolygonPreparationParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Vertex capacity of the prepared positions, sized from the conservative required count. */
  vertexCapacity: number;
};

/** Densified, antimeridian-safe copies of the harbour zones, in the doubled main/east layout. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {zones, parameters, vertexCapacity}: LonLatPolygonPreparationExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const word = (id: string, length: number) =>
    createTransientView(graph, id, 'uint32', length, usage);
  // #region example
  const output = {
    positions: createTransientView(graph, 'prepared', 'float32x2', vertexCapacity, usage),
    featureOffsets: word('preparedFeatureOffsets', zones.featureOffsets.length),
    polygonOffsets: word('preparedPolygonOffsets', 2 * zones.polygonOffsets.length - 1),
    ringOffsets: word('preparedRingOffsets', 2 * zones.ringOffsets.length - 1),
    ringFlags: word('ringFlags', zones.ringOffsets.length - 1),
    status: {
      count: word('count', 1),
      overflow: word('overflow', 1),
      requiredCount: word('requiredCount', 1),
      invalidCount: word('invalidCount', 1),
      approximation: word('approximation', 1)
    }
  };
  graph.add(
    new GPULonLatPolygonPreparation({
      ...zones,
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none', edgeModel: 'linear'},
      parameters,
      maximumPieceCount: 64,
      output
    })
  );
  // ringOffsets: main rings then their east copies; ringFlags: crossing, capped, ambiguous, ...
  const {positions: prepared, ringOffsets, ringFlags} = output;
  // #endregion example
  return {...output, prepared, ringOffsets, ringFlags};
}
