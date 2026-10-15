// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUMakeValid,
  type GPUPolygonGeometryFields,
  type GPUPolygonizeOutput
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Park polygons in UTM 16N metres and the capacities of the noded segments and faces. */
export type MakeValidExampleInput = {
  parks: GPUPolygonGeometryFields;
  segmentCapacity: number;
  ringCapacity: number;
  vertexCapacity: number;
};

/** Allocates every column of a polygonization result (transient ids are listed in the scenario's retain). */
function createPolygonizeOutput(
  graph: GPUCommandGraph<undefined>,
  {segmentCapacity, ringCapacity, vertexCapacity}: Omit<MakeValidExampleInput, 'parks'>
): GPUPolygonizeOutput {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const directedCapacity = 2 * segmentCapacity;
  const make = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const status = (prefix: string) => ({
    count: make(`${prefix}Count`, 'uint32', 1),
    requiredCount: make(`${prefix}RequiredCount`, 'uint32', 1),
    overflow: make(`${prefix}Overflow`, 'uint32', 1),
    candidateOverflow: make(`${prefix}CandidateOverflow`, 'uint32', 1)
  });
  return {
    nodedSegments: {
      endpoints: make('nodedEndpoints', 'float32x4', segmentCapacity),
      sourceFeatureIds: make('nodedFeatureIds', 'uint32', segmentCapacity),
      sourceRingIds: make('nodedRingIds', 'uint32', segmentCapacity),
      sourceEdgeIds: make('nodedEdgeIds', 'uint32', segmentCapacity),
      sourceStartParameters: make('nodedStartParameters', 'float32', segmentCapacity),
      sourceEndParameters: make('nodedEndParameters', 'float32', segmentCapacity),
      status: status('noded')
    },
    directedEdges: {
      endpoints: make('directedEndpoints', 'float32x4', directedCapacity),
      sourceSegmentIds: make('directedSegmentIds', 'uint32', directedCapacity),
      sourceFeatureIds: make('directedFeatureIds', 'uint32', directedCapacity),
      twinIds: make('directedTwinIds', 'uint32', directedCapacity),
      ringIds: make('directedRingIds', 'uint32', directedCapacity),
      flags: make('directedFlags', 'uint32', directedCapacity),
      status: status('directed')
    },
    polygons: {
      kind: 'polygons',
      positions: make('polygonPositions', 'float32x2', vertexCapacity),
      featureOffsets: make('polygonFeatureOffsets', 'uint32', ringCapacity + 1),
      polygonOffsets: make('polygonPolygonOffsets', 'uint32', ringCapacity + 1),
      ringOffsets: make('polygonRingOffsets', 'uint32', ringCapacity + 1),
      sourceIds: make('polygonSourceIds', 'uint32', ringCapacity)
    },
    diagnostics: {
      edgeClasses: make('edgeClasses', 'uint32', segmentCapacity),
      closedRingCount: make('closedRingCount', 'uint32', 1),
      openChainCount: make('openChainCount', 'uint32', 1),
      cutEdgeCount: make('cutEdgeCount', 'uint32', 1),
      dangleCount: make('dangleCount', 'uint32', 1),
      unusedEdgeCount: make('unusedEdgeCount', 'uint32', 1)
    },
    status: status('polygon')
  };
}

/** Rebuilds every park polygon into valid faces by noding it and keeping odd-parity faces. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, ...capacities}: MakeValidExampleInput
) {
  // #region example
  const output = createPolygonizeOutput(graph, capacities); // GPUPolygonizeOutput columns
  graph.add(
    new GPUMakeValid({
      polygons: {kind: 'polygons', ...parks}, // UTM 16N metres
      intersectionCapacity: 1024,
      precision: {vertexTolerance: 0.01, predicates: 'exact-or-uncertain', storage: 'float32'},
      output
    })
  );
  // polygons: the kept faces, one ring each; status.count: polygons written
  const {polygons, status} = output;
  // #endregion example
  return {...output, polygons, status};
}
