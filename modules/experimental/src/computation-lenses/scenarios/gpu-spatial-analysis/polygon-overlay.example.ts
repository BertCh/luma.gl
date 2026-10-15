// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUPolygonOverlay,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Parks and one community area in the shared UTM 16N frame (metres), with the output capacities. */
export type PolygonOverlayExampleInput = {
  parks: GPUPolygonGeometryFields;
  area: GPUPolygonGeometryFields;
  capacity: {
    intersections: number;
    nodedSegments: number;
    boundary: number;
    rings: number;
    vertices: number;
  };
};

/** Intersection of the parks with one community area. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, area, capacity}: PolygonOverlayExampleInput
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
      positions: view('overlayPositions', 'float32x2', capacity.vertices),
      featureOffsets: view('overlayFeatureOffsets', 'uint32', capacity.rings + 1),
      polygonOffsets: view('overlayPolygonOffsets', 'uint32', capacity.rings + 1),
      ringOffsets: view('overlayRingOffsets', 'uint32', capacity.rings + 1),
      sourceIds: view('overlaySourceIds', 'uint32', capacity.rings)
    },
    status: {
      count: scalar('overlayCount'),
      requiredCount: scalar('overlayRequiredCount'),
      overflow: scalar('overlayOverflow'),
      candidateOverflow: scalar('overlayCandidateOverflow')
    },
    boundary: {
      endpoints: view('boundaryEndpoints', 'float32x4', capacity.boundary),
      operandIds: view('boundaryOperandIds', 'uint32', capacity.boundary),
      sourceFeatureIds: view('boundarySourceFeatureIds', 'uint32', capacity.boundary),
      status: {
        count: scalar('boundaryCount'),
        requiredCount: scalar('boundaryRequiredCount'),
        overflow: scalar('boundaryOverflow'),
        candidateOverflow: scalar('boundaryCandidateOverflow')
      }
    }
  };
  // #region example
  // Lattice step 0.05 m (rounded up to a power of two); both operands are UTM 16N metres.
  graph.add(
    new GPUPolygonOverlay({
      left: {kind: 'polygons', ...parks},
      right: {kind: 'polygons', ...area},
      operation: 'intersection',
      fillRule: 'even-odd',
      vertexTolerance: 0.05,
      capacity: {intersections: capacity.intersections, nodedSegments: capacity.nodedSegments},
      output
    })
  );
  // ringOffsets are padded past the used rings; boundary.sourceFeatureIds: park of each kept edge
  const {geometry: overlap} = output;
  // #endregion example
  return {...output, overlap};
}
