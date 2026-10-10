// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUBoundedResultStatusPort, GPUPolygonGeometryPort} from '../contracts/index';

/** Sentinel written to provenance and ownership slots with no result row. */
export const GPU_TOPOLOGY_NONE = 0xffffffff;

/** Per-edge polygonization classification bits. */
export const GPU_POLYGONIZE_EDGE_CLASS = {
  /** At least one directed side owns a bounded polygon ring. */
  closedRing: 1,
  /** The edge belongs to an open chain. */
  openChain: 2,
  /** The edge is connected at both ends but owns no bounded face. */
  cut: 4,
  /** At least one endpoint has degree one in the noded graph. */
  dangle: 8,
  /** The edge was not consumed by a bounded polygon. */
  unused: 16
} as const;

/** Explicit snap and arithmetic policy for geometry-topology operators. */
export type GPUTopologyPrecisionPolicy = {
  /** Chebyshev distance used only to identify equal noded vertices. */
  vertexTolerance: number;
  /** Predicate policy. Exact signs reuse the segment-intersection expansion predicates. */
  predicates: 'exact-or-uncertain';
  /** Coordinates and generated intersection points are stored as float32. */
  storage: 'float32';
};

/** Topology status always distinguishes final capacity from an incomplete candidate substrate. */
export type GPUTopologyStatusPort = GPUBoundedResultStatusPort & {
  requiredCount: GraphDataView<'uint32'>;
  candidateOverflow: GraphDataView<'uint32'>;
};

/**
 * Capacity-bounded atomic segments after all crossings, touches, overlaps and repeated vertices
 * have been resolved. Provenance is aligned with `endpoints` and identifies the original feature,
 * line/ring, edge and normalized parameter interval.
 */
export type GPUNodedSegmentPort = {
  endpoints: GraphDataView<'float32x4'>;
  sourceFeatureIds: GraphDataView<'uint32'>;
  sourceRingIds: GraphDataView<'uint32'>;
  sourceEdgeIds: GraphDataView<'uint32'>;
  sourceStartParameters: GraphDataView<'float32'>;
  sourceEndParameters: GraphDataView<'float32'>;
  status: GPUTopologyStatusPort;
};

/** Both orientations of every noded edge plus the ring/face owned by each directed side. */
export type GPUDirectedEdgePort = {
  endpoints: GraphDataView<'float32x4'>;
  sourceSegmentIds: GraphDataView<'uint32'>;
  sourceFeatureIds: GraphDataView<'uint32'>;
  /** Oppositely oriented directed edge for the same noded source segment. */
  twinIds: GraphDataView<'uint32'>;
  ringIds: GraphDataView<'uint32'>;
  flags: GraphDataView<'uint32'>;
  status: GPUTopologyStatusPort;
};

/** Scalar and per-source-edge diagnostics matching `polygonize_full` categories. */
export type GPUPolygonizeDiagnosticsPort = {
  edgeClasses: GraphDataView<'uint32'>;
  closedRingCount: GraphDataView<'uint32'>;
  openChainCount: GraphDataView<'uint32'>;
  cutEdgeCount: GraphDataView<'uint32'>;
  dangleCount: GraphDataView<'uint32'>;
  unusedEdgeCount: GraphDataView<'uint32'>;
};

/** Canonical output of general bounded polygonization. */
export type GPUPolygonizeOutput = {
  nodedSegments: GPUNodedSegmentPort;
  directedEdges: GPUDirectedEdgePort;
  polygons: GPUPolygonGeometryPort & {
    positions: GraphDataView<'float32x2'>;
    sourceIds: GraphDataView<'uint32'>;
  };
  diagnostics: GPUPolygonizeDiagnosticsPort;
  status: GPUTopologyStatusPort;
};
