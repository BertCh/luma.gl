// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GraphVectorView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUFloat32Positions, GPUUint32Rows} from '../../utils/gpu-contributor-types';
import type {
  GPUBoundedResultStatusPort,
  GPUCandidateStatusPort,
  GPUCardinalityPort,
  GPUOverflowStatusPort
} from './status';
import {validateGPUStatusPort} from './status';

export type {
  GPUCompactOutput,
  GPUFloat32Positions,
  GPUUint32Rows
} from '../../utils/gpu-contributor-types';

/** Point features with optional stable source identity. */
export type GPUPointGeometryPort = {
  kind: 'points';
  positions: GPUFloat32Positions;
  sourceIds?: GPUUint32Rows;
};

/** Linestring features in a global offset layout. */
export type GPULineGeometryPort = {
  kind: 'lines';
  positions: GPUFloat32Positions;
  /** Feature-to-vertex offsets with one terminal entry. */
  lineOffsets: GraphDataView<'uint32'>;
  sourceIds?: GPUUint32Rows;
};

/** Polygon or multipolygon features in a GeoArrow-compatible offset layout. */
export type GPUPolygonGeometryPort = {
  kind: 'polygons';
  positions: GPUFloat32Positions;
  /** Feature-to-polygon offsets with one terminal entry. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets with one terminal entry. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets with one terminal entry. */
  ringOffsets: GraphDataView<'uint32'>;
  sourceIds?: GPUUint32Rows;
};

/**
 * Supported physical formats for a trajectory timestamp column: `float32` relative to an
 * application epoch, or `uint32x2` Int64 `(low, high)` words (for example an Arrow `Int64`
 * epoch-millisecond column).
 */
export type GPUTrajectoryTimestampFormat = 'float32' | 'uint32x2';

/** Ordered trajectory coordinates, track offsets and aligned timestamps. */
export type GPUTrajectoryGeometryPort<
  TimestampFormat extends GPUTrajectoryTimestampFormat = GPUTrajectoryTimestampFormat
> = {
  kind: 'trajectories';
  positions: GPUFloat32Positions;
  /** Track-to-position offsets with one terminal entry. */
  trackOffsets: GraphDataView<'uint32'>;
  timestamps: GraphDataView<TimestampFormat> | GraphVectorView<TimestampFormat>;
  sourceIds?: GPUUint32Rows;
};

/** Canonical logical feature geometry accepted by spatial-analysis contributors. */
export type GPUFeatureGeometryPort =
  | GPUPointGeometryPort
  | GPULineGeometryPort
  | GPUPolygonGeometryPort
  | GPUTrajectoryGeometryPort;

/**
 * A geometry port whose positions are one packed view, the form geometry kernels bind.
 *
 * Contributors that take a whole geometry (for example each side of a spatial join) type it with
 * this. Any port whose `positions` is a single `GraphDataView` is assignable, so a generated
 * geometry output feeds the next contributor directly. `sourceIds`, when a contributor writes
 * feature IDs, replace the feature rows in its output; contributors that write no feature IDs do not
 * read them.
 */
export type GPUPackedGeometryPort<Port extends GPUFeatureGeometryPort = GPUFeatureGeometryPort> =
  Port extends GPUTrajectoryGeometryPort<infer TimestampFormat>
    ? Omit<Port, 'positions' | 'sourceIds' | 'timestamps'> & {
        positions: GraphDataView<'float32x2'>;
        timestamps: PackedTimestamps<TimestampFormat>;
        sourceIds?: GraphDataView<'uint32'>;
      }
    : Port extends unknown
      ? Omit<Port, 'positions' | 'sourceIds'> & {
          positions: GraphDataView<'float32x2'>;
          sourceIds?: GraphDataView<'uint32'>;
        }
      : never;

/** One packed timestamp view per format of the union (a `float32` or an Int64-words view). */
type PackedTimestamps<TimestampFormat extends GPUTrajectoryTimestampFormat> =
  TimestampFormat extends unknown ? GraphDataView<TimestampFormat> : never;

/**
 * The four-level polygon layout as flat contributor props: `positions`, `featureOffsets`,
 * `polygonOffsets` and `ringOffsets` of {@link GPUPolygonGeometryPort}, one row per feature.
 *
 * The first ring of every polygon is its shell and later rings are holes, whatever their winding.
 */
export type GPUPolygonGeometryFields = Omit<
  GPUPackedGeometryPort<GPUPolygonGeometryPort>,
  'kind' | 'sourceIds'
>;

/** The line layout as flat contributor props: `positions` and `lineOffsets`, one row per line. */
export type GPULineGeometryFields = Omit<
  GPUPackedGeometryPort<GPULineGeometryPort>,
  'kind' | 'sourceIds'
>;

/**
 * The trajectory layout as flat contributor props: `positions`, `trackOffsets` and `timestamps`,
 * one row per track.
 */
export type GPUTrajectoryGeometryFields<
  TimestampFormat extends GPUTrajectoryTimestampFormat = GPUTrajectoryTimestampFormat
> = Omit<GPUPackedGeometryPort<GPUTrajectoryGeometryPort<TimestampFormat>>, 'kind' | 'sourceIds'>;

/** Capacity-bounded pairs with explicit final and candidate completeness. */
export type GPUCompactPairPort = {
  leftIds: GraphDataView<'uint32'>;
  rightIds: GraphDataView<'uint32'>;
  /** Pair count, final overflow and, when refinement is candidate-bounded, candidate overflow. */
  status: GPUCardinalityPort & GPUOverflowStatusPort & GPUCandidateStatusPort;
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
export type GPUCellTablePort = {
  /** 64-bit cell identifiers stored as two uint32 words. */
  cells: GraphDataView<'uint32x2'>;
  /** Optional source-row occupancy per cell. */
  counts?: GraphDataView<'uint32'>;
  /** Occupied row count, optional unclamped requirement and optional overflow. */
  status: GPUCardinalityPort & Partial<GPUOverflowStatusPort>;
};

/** Regular sampled field with an explicit row-major shape and nodata convention. */
export type GPUSampledSurfacePort<Format extends 'float32' | 'uint32' = 'float32'> = {
  values: GraphDataView<Format> | GraphVectorView<Format>;
  columns: number;
  rows: number;
  mask?: GPUUint32Rows;
};

/** Source-aligned class labels with an optional valid-row mask. */
export type GPUClassifiedValuesPort = {
  classes: GPUUint32Rows;
  mask?: GPUUint32Rows;
  /** GPU-resident class cardinality so dynamic classification feeds consumers without readback. */
  classCount: GraphDataView<'uint32'>;
};

/** Generated geometry plus source provenance and bounded-result status. */
export type GPUGeneratedGeometryPort<
  Geometry extends GPUFeatureGeometryPort = GPUFeatureGeometryPort
> = {
  geometry: Geometry;
  sourceIds: GPUUint32Rows;
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

function validateSourceIds(id: string, sourceIds: GPUUint32Rows | undefined, length: number): void {
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
  if (geometry.timestamps.format !== 'float32' && geometry.timestamps.format !== 'uint32x2') {
    throw new Error(`${id} timestamps must use float32 or uint32x2`);
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
  validateFormat(id, 'leftIds', pairs.leftIds, 'uint32');
  validateFormat(id, 'rightIds', pairs.rightIds, 'uint32');
  if (pairs.leftIds.length !== pairs.rightIds.length) {
    throw new Error(`${id} pair ID capacities must match`);
  }
  validateGPUStatusPort(`${id} status`, pairs.status);
}
