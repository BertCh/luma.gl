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
  GPUBufferSurface,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Parks in UTM 16N metres, the packed distance (50 m) and the output capacities. */
export type BufferSurfaceExampleInput = {
  parks: GPUPolygonGeometryFields;
  parameters: GraphDataView<'float32'>;
  capacity: {
    intersections: number;
    nodedSegments: number;
    boundary: number;
    rings: number;
    vertices: number;
  };
};

/** Dissolved 50 m buffer polygons of the parks. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, parameters, capacity}: BufferSurfaceExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const scalar = (id: string) => createTransientView(graph, id, 'uint32', 1, usage);
  const view = <Format extends 'float32x2' | 'float32x4' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const output = {
    geometry: {
      kind: 'polygons' as const,
      positions: view('bufferPositions', 'float32x2', capacity.vertices),
      featureOffsets: view('bufferFeatureOffsets', 'uint32', capacity.rings + 1),
      polygonOffsets: view('bufferPolygonOffsets', 'uint32', capacity.rings + 1),
      ringOffsets: view('bufferRingOffsets', 'uint32', capacity.rings + 1),
      sourceIds: view('bufferSourceIds', 'uint32', capacity.rings)
    },
    status: {
      count: scalar('bufferCount'),
      requiredCount: scalar('bufferRequiredCount'),
      overflow: scalar('bufferOverflow'),
      candidateOverflow: scalar('bufferCandidateOverflow')
    },
    boundary: {
      endpoints: view('bufferBoundaryEndpoints', 'float32x4', capacity.boundary),
      operandIds: view('bufferBoundaryOperandIds', 'uint32', capacity.boundary),
      sourceFeatureIds: view('bufferBoundarySourceFeatureIds', 'uint32', capacity.boundary),
      status: {
        count: scalar('bufferBoundaryCount'),
        requiredCount: scalar('bufferBoundaryRequiredCount'),
        overflow: scalar('bufferBoundaryOverflow'),
        candidateOverflow: scalar('bufferBoundaryCandidateOverflow')
      }
    }
  };
  // #region example
  // Parameters: distance 50 m, mitre limit 5; parks are UTM 16N metres, so the context is planar.
  graph.add(
    new GPUBufferSurface({
      geometry: {kind: 'polygons', ...parks},
      parameters,
      joinStyle: 'round',
      quadSegments: 8,
      vertexTolerance: 0.05,
      capacity: {intersections: capacity.intersections, nodedSegments: capacity.nodedSegments},
      output
    })
  );
  // ringOffsets are padded past the used rings, so unused rings are empty
  const {geometry: buffers} = output;
  // #endregion example
  return {...output, buffers};
}
