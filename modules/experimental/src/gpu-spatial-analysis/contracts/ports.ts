// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GraphVectorView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUBoundedResultStatusPort, GPUCardinalityPort} from './status';

/** Packed or partition-preserving coordinate rows. */
export type GPUCoordinateRows = GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;

/** Packed or partition-preserving unsigned rows. */
export type GPUIdentifierRows = GraphDataView<'uint32'> | GraphVectorView<'uint32'>;

/** Point features with optional stable source identity. */
export type GPUPointGeometryPort = {
  kind: 'points';
  positions: GPUCoordinateRows;
  sourceIds?: GPUIdentifierRows;
};

/** Linestring features in a global offset layout. */
export type GPULineGeometryPort = {
  kind: 'lines';
  positions: GPUCoordinateRows;
  /** Feature-to-vertex offsets with one terminal entry. */
  lineOffsets: GraphDataView<'uint32'>;
  sourceIds?: GPUIdentifierRows;
};

/** Polygon or multipolygon features in a GeoArrow-compatible offset layout. */
export type GPUPolygonGeometryPort = {
  kind: 'polygons';
  positions: GPUCoordinateRows;
  /** Feature-to-polygon offsets with one terminal entry. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets with one terminal entry. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets with one terminal entry. */
  ringOffsets: GraphDataView<'uint32'>;
  sourceIds?: GPUIdentifierRows;
};

/** Supported physical formats for a trajectory timestamp column. */
export type GPUTrajectoryTimestampFormat = 'float32' | 'uint32';

/** Ordered trajectory coordinates, track offsets and aligned timestamps. */
export type GPUTrajectoryGeometryPort<
  TimestampFormat extends GPUTrajectoryTimestampFormat = GPUTrajectoryTimestampFormat
> = {
  kind: 'trajectories';
  positions: GPUCoordinateRows;
  /** Track-to-position offsets with one terminal entry. */
  trackOffsets: GraphDataView<'uint32'>;
  timestamps: GraphDataView<TimestampFormat> | GraphVectorView<TimestampFormat>;
  sourceIds?: GPUIdentifierRows;
};

/** Canonical logical feature geometry accepted by spatial-analysis contributors. */
export type GPUFeatureGeometryPort =
  | GPUPointGeometryPort
  | GPULineGeometryPort
  | GPUPolygonGeometryPort
  | GPUTrajectoryGeometryPort;

/** Capacity-bounded pairs with explicit final and candidate completeness. */
export type GPUCompactPairPort = GPUCardinalityPort & {
  leftIds: GraphDataView<'uint32'>;
  rightIds: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  candidateOverflow?: GraphDataView<'uint32'>;
};

/** CSR neighborhood graph with an optional aligned distance column. */
export type GPUSpatialWeightsPort = {
  offsets: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
  weights: GraphDataView<'float32'>;
  distances?: GraphDataView<'float32'>;
  status?: Partial<GPUBoundedResultStatusPort>;
};

/** Sparse capacity-bounded DGGS table. */
export type GPUCellTablePort = GPUCardinalityPort & {
  /** 64-bit cell identifiers stored as two uint32 words. */
  cells: GraphDataView<'uint32x2'>;
  /** Optional source-row occupancy per cell. */
  counts?: GraphDataView<'uint32'>;
  overflow?: GraphDataView<'uint32'>;
};

/** Regular sampled field with an explicit row-major shape and nodata convention. */
export type GPUSampledSurfacePort<Format extends 'float32' | 'uint32' = 'float32'> = {
  values: GraphDataView<Format> | GraphVectorView<Format>;
  columns: number;
  rows: number;
  mask?: GPUIdentifierRows;
};

/** Source-aligned class labels with an optional valid-row mask. */
export type GPUClassifiedValuesPort = {
  classes: GPUIdentifierRows;
  mask?: GPUIdentifierRows;
  /** GPU-resident class cardinality so dynamic classification feeds consumers without readback. */
  classCount: GraphDataView<'uint32'>;
};

/** Generated geometry plus source provenance and bounded-result status. */
export type GPUGeneratedGeometryPort<
  Geometry extends GPUFeatureGeometryPort = GPUFeatureGeometryPort
> = {
  geometry: Geometry;
  sourceIds: GPUIdentifierRows;
  status: GPUBoundedResultStatusPort;
};

function validateFormat(
  id: string,
  name: string,
  view: {format: string; length: number},
  format: string
): void {
  if (view.format !== format) {
    // Port fields use fixed formats so connected contributors agree without casts.
    throw new Error(`${id} ${name} must use ${format}`);
  }
}

function validateSourceIds(
  id: string,
  sourceIds: GPUIdentifierRows | undefined,
  length: number
): void {
  if (!sourceIds) {
    return;
  }
  validateFormat(id, 'sourceIds', sourceIds, 'uint32');
  if (sourceIds.length !== length) {
    // Source IDs are feature-aligned rather than vertex-aligned.
    throw new Error(`${id} sourceIds length must equal the feature count`);
  }
}

/** Validates the static formats and lengths visible from a feature-geometry port. */
export function validateGPUFeatureGeometryPort(id: string, geometry: GPUFeatureGeometryPort): void {
  validateFormat(id, 'positions', geometry.positions, 'float32x2');
  if (geometry.kind === 'points') {
    validateSourceIds(id, geometry.sourceIds, geometry.positions.length);
    return;
  }
  if (geometry.kind === 'lines') {
    validateFormat(id, 'lineOffsets', geometry.lineOffsets, 'uint32');
    if (geometry.lineOffsets.length < 1) {
      throw new Error(`${id} lineOffsets must contain a terminal entry`);
    }
    validateSourceIds(id, geometry.sourceIds, geometry.lineOffsets.length - 1);
    return;
  }
  if (geometry.kind === 'polygons') {
    validateFormat(id, 'featureOffsets', geometry.featureOffsets, 'uint32');
    validateFormat(id, 'polygonOffsets', geometry.polygonOffsets, 'uint32');
    validateFormat(id, 'ringOffsets', geometry.ringOffsets, 'uint32');
    if (
      geometry.featureOffsets.length < 1 ||
      geometry.polygonOffsets.length < 1 ||
      geometry.ringOffsets.length < 1
    ) {
      throw new Error(`${id} polygon offsets must contain terminal entries`);
    }
    validateSourceIds(id, geometry.sourceIds, geometry.featureOffsets.length - 1);
    return;
  }
  validateFormat(id, 'trackOffsets', geometry.trackOffsets, 'uint32');
  if (geometry.timestamps.format !== 'float32' && geometry.timestamps.format !== 'uint32') {
    throw new Error(`${id} timestamps must use float32 or uint32`);
  }
  if (geometry.trackOffsets.length < 1) {
    throw new Error(`${id} trackOffsets must contain a terminal entry`);
  }
  if (geometry.timestamps.length !== geometry.positions.length) {
    throw new Error(`${id} timestamps length must equal positions length`);
  }
  validateSourceIds(id, geometry.sourceIds, geometry.trackOffsets.length - 1);
}

/** Validates the static shape of a compact pair port. */
export function validateGPUCompactPairPort(id: string, pairs: GPUCompactPairPort): void {
  for (const [name, view] of Object.entries(pairs)) {
    if (view && typeof view === 'object' && 'format' in view) {
      validateFormat(id, name, view as {format: string; length: number}, 'uint32');
    }
  }
  if (pairs.leftIds.length !== pairs.rightIds.length) {
    throw new Error(`${id} pair ID capacities must match`);
  }
  for (const [name, scalar] of [
    ['count', pairs.count],
    ['requiredCount', pairs.requiredCount],
    ['overflow', pairs.overflow],
    ['candidateOverflow', pairs.candidateOverflow]
  ] as const) {
    if (scalar && scalar.length < 1) {
      throw new Error(`${id} ${name} must contain one uint32 row`);
    }
  }
}
