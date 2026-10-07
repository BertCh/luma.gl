// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GPU_GEODESIC_MEAN_EARTH_RADIUS} from '../geometry-measures/geodesic-wgsl';
import {GPUNearestFeatureJoin} from '../spatial-join/gpu-nearest-feature-join';
import {createPathPrefixNode} from '../line-segmentize/line-segmentize-kernels';
import {
  createPackedScatterNodes,
  createPathSegmentsNode,
  createPointProjectionNode,
  createSphericalProjectionNode,
  createSphericalSegmentTableNode,
  getPointProjectionScatterColumns,
  LINEAR_REFERENCING_RESULT_STRIDE,
  SPHERICAL_SEGMENT_TABLE_STRIDE,
  type PointProjectionColumns
} from './linear-referencing-kernels';

const OPERATION = 'GPULinearReferencing';

/** Coordinate interpretation of {@link GPULinearReferencing}. */
export type GPULinearReferencingCoordinateSystem = 'planar' | 'spherical';

/**
 * Per-point outputs of {@link GPULinearReferencing}; each optional, `points.length` rows. Points with
 * no path within the radius get `GPU_LINE_NO_SOURCE` indices, NaN fractions, foot points, measures,
 * normalized measures and signed offsets, distance `-1` and side `0`.
 *
 * With `coordinateSystem: 'spherical'`, foot points are longitude/latitude degrees, `fractions` is
 * the angular fraction along the great-circle arc, and distances, measures and signed offsets are
 * in sphere-radius units (meters by default).
 */
export type GPULinearReferencingOutput = {
  /** Index of the nearest path. */
  pathIndices?: GraphDataView<'uint32'>;
  /** Index of the nearest segment within its path (segment `k` joins vertices `k` and `k + 1`). */
  segmentIndices?: GraphDataView<'uint32'>;
  /** Position of the foot point along the segment, in `[0, 1]`. */
  fractions?: GraphDataView<'float32'>;
  /** Closest point on the path (turf `nearestPointOnLine`, PostGIS `ST_ClosestPoint`). */
  footPoints?: GraphDataView<'float32x2'>;
  /** Distance from the point to the foot point. */
  distances?: GraphDataView<'float32'>;
  /** Distance along the path from its first vertex to the foot point (`ST_LineLocatePoint` times length). */
  measures?: GraphDataView<'float32'>;
  /**
   * `measures` divided by the length of the nearest path, in `[0, 1]` (shapely
   * `line_locate_point(normalized=True)`); `0` for zero-length paths.
   */
  normalizedMeasures?: GraphDataView<'float32'>;
  /** `1` left of the segment direction, `-1` right, `0` on the line. */
  sides?: GraphDataView<'sint32'>;
  /** `side * distance`: signed lateral offset, positive to the left. */
  signedOffsets?: GraphDataView<'float32'>;
  /** Optional per-vertex cumulative path measure, `positions.length` rows. */
  vertexMeasures?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPULinearReferencing}.
 *
 * Per-frame (no recompile): `radius` and the contents of every input buffer. Compile-time: view
 * lengths, the path count, `coordinateSystem`, `sphereRadius`, `candidateCapacity`, `leafCapacity`,
 * `spatialSort`, and which outputs are present.
 */
export type GPULinearReferencingProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'linear-referencing'`. */
  id?: string;
  /**
   * `'planar'` (default): Euclidean segments in position units. `'spherical'`: positions are
   * longitude/latitude degrees, segments are great-circle arcs, and `radius`, distances, measures
   * and offsets are in sphere-radius units (meters by default).
   */
  coordinateSystem?: GPULinearReferencingCoordinateSystem;
  /** Sphere radius for `'spherical'`. Defaults to {@link GPU_GEODESIC_MEAN_EARTH_RADIUS} (meters). */
  sphereRadius?: number;
  /** Packed query points (longitude/latitude degrees for `'spherical'`). */
  points: GraphDataView<'float32x2'>;
  /**
   * Packed path vertices sorted by path (longitude/latitude degrees for `'spherical'`; project
   * geographic data first for `'planar'`).
   */
  positions: GraphDataView<'float32x2'>;
  /** `pathCount + 1` monotonic row offsets. */
  pathOffsets: GraphDataView<'uint32'>;
  /** Per-frame search radius, one float32 row. NaN, negative or infinite matches nothing. */
  radius: GraphDataView<'float32'>;
  /**
   * Maximum `(point, segment)` bounding-box candidates per encoding. Required for `'planar'`;
   * ignored by `'spherical'`, which scans every segment and never overflows.
   */
  candidateCapacity?: number;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of `positions.length`. Planar only. */
  leafCapacity?: number;
  /** Morton-sort segments before the BVH build; see `GPUNearestFeatureJoin`. Default false. Planar only. */
  spatialSort?: boolean;
  /** Per-point outputs. */
  output: GPULinearReferencingOutput;
  /** One-row flag: 1 when the BVH leaf or candidate capacity overflowed. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row unclamped candidate count. */
  candidateCount?: GraphDataView<'uint32'>;
};

/**
 * Snaps many points to the nearest of many polylines and reports linear-referencing coordinates:
 * path, segment, fraction, foot point, distance, measure along the path, side and signed offset
 * (turf `nearestPointOnLine`, PostGIS `ST_ClosestPoint` and `ST_LineLocatePoint`, route chainage,
 * map-matching candidates, lateral position against a corridor).
 *
 * Composition: one kernel turns every vertex row into a segment to the next row of the same path
 * (other rows become NaN segments that never match); `GPUNearestFeatureJoin` finds each point's
 * nearest segment within the radius through its BVH, with ties going to the smallest segment row
 * (so the earliest segment along the path, and the lowest path); a per-path sequential prefix
 * computes cumulative vertex measures (Neumaier-compensated, deterministic); one kernel projects
 * each point onto its segment in segment-local coordinates; scatter kernels copy the requested
 * columns.
 *
 * `coordinateSystem: 'spherical'` treats positions as longitude/latitude degrees and segments as
 * great-circle arcs (HaversineClosestPoint, cross-track and along-track distance): a per-row table
 * stores each arc's unit start vector, tangent and normal (built from the degree-difference
 * bearing, so meter-scale arcs stay well conditioned), and a brute-force kernel scans every arc per
 * point. Cost is `points x segments`, so use planar projection with the BVH join for large
 * networks. Distances, measures and offsets come out in sphere-radius units (meters by default);
 * the search radius is in the same units. Precision is f32: about 1 m for arcs in the full
 * longitude range, dominated by the f32 longitude/latitude inputs. Arcs spanning more than about
 * 179 degrees are unsupported.
 *
 * Otherwise planar: distances and measures are Euclidean in position units.
 */
export class GPULinearReferencing implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULinearReferencingProps;
  /** Number of paths. */
  readonly pathCount: number;
  /** Resolved coordinate system. */
  readonly coordinateSystem: GPULinearReferencingCoordinateSystem;
  /** Resolved sphere radius. */
  readonly sphereRadius: number;

  constructor(props: GPULinearReferencingProps) {
    this.id = props.id ?? 'linear-referencing';
    this.props = props;
    this.coordinateSystem = props.coordinateSystem ?? 'planar';
    this.sphereRadius = props.sphereRadius ?? GPU_GEODESIC_MEAN_EARTH_RADIUS;
    const {id} = this;
    const {output} = props;
    for (const [name, view] of Object.entries({
      points: props.points,
      positions: props.positions,
      pathOffsets: props.pathOffsets,
      ...output
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (this.coordinateSystem !== 'planar' && this.coordinateSystem !== 'spherical') {
      throw new Error(`${id} coordinateSystem must be 'planar' or 'spherical'`);
    }
    if (!Number.isFinite(this.sphereRadius) || this.sphereRadius <= 0) {
      throw new Error(`${id} sphereRadius must be a positive finite number`);
    }
    if (
      this.coordinateSystem === 'planar' &&
      !(Number.isInteger(props.candidateCapacity) && (props.candidateCapacity as number) > 0)
    ) {
      throw new Error(`${id} candidateCapacity must be a positive integer for planar projection`);
    }
    validatePackedView(props.points, ['float32x2'], `${id} points`);
    if (props.points.length < 1) {
      throw new Error(`${id} needs at least one point`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 2) {
      throw new Error(`${id} needs at least two positions`);
    }
    validatePackedUint32View(props.pathOffsets, `${id} pathOffsets`);
    if (props.pathOffsets.length < 2) {
      throw new Error(`${id} pathOffsets must contain at least two rows`);
    }
    this.pathCount = props.pathOffsets.length - 1;
    const pointCount = props.points.length;
    const formats = {
      pathIndices: 'uint32',
      segmentIndices: 'uint32',
      fractions: 'float32',
      footPoints: 'float32x2',
      distances: 'float32',
      measures: 'float32',
      normalizedMeasures: 'float32',
      sides: 'sint32',
      signedOffsets: 'float32',
      vertexMeasures: 'float32'
    } as const;
    let columnCount = 0;
    for (const [name, view] of Object.entries(output) as [
      keyof typeof formats,
      GraphDataView | undefined
    ][]) {
      if (!view) {
        continue;
      }
      if (!(name in formats)) {
        throw new Error(`${id} output.${name} is not a linear-referencing column`);
      }
      validatePackedView(view, [formats[name]], `${id} output.${name}`);
      const length = name === 'vertexMeasures' ? props.positions.length : pointCount;
      if (view.length !== length) {
        throw new Error(`${id} output.${name} must hold ${length} rows`);
      }
      columnCount++;
    }
    if (columnCount === 0) {
      throw new Error(`${id} requires at least one output column`);
    }
    validatePackedUint32View(props.overflow, `${id} overflow`);
    validateGraphOutputsDisjointFromInputs(
      id,
      [...Object.values(output), props.overflow, props.candidateCount],
      [props.points, props.positions, props.pathOffsets, props.radius]
    );
  }

  /**
   * Returns the segment (or spherical segment table) node, the measure prefix node, the
   * nearest-feature join nodes (planar), the projection node and the scatter nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.points,
      props.positions,
      props.pathOffsets,
      props.radius,
      props.overflow,
      props.candidateCount,
      ...Object.values(output)
    ]);
    const rowCount = props.positions.length;
    const pointCount = props.points.length;
    const spherical = this.coordinateSystem === 'spherical';
    const vertexMeasures =
      output.vertexMeasures ??
      createTransientView(graph, `${id}-vertex-measures`, 'float32', rowCount);
    const results = createTransientView(
      graph,
      `${id}-results`,
      'uint32',
      pointCount * LINEAR_REFERENCING_RESULT_STRIDE
    );
    const prefixNode = createPathPrefixNode<Parameters>(graph, {
      id: `${id}-vertex-measures`,
      operation: OPERATION,
      positions: props.positions,
      pathOffsets: props.pathOffsets,
      coordinateSystem: this.coordinateSystem,
      radius: spherical ? this.sphereRadius : 1,
      rowMeasures: vertexMeasures
    });
    const scatterNodes = createPackedScatterNodes<Parameters>(graph, {
      id: `${id}-output`,
      operation: OPERATION,
      rowCount: pointCount,
      stride: LINEAR_REFERENCING_RESULT_STRIDE,
      results,
      columns: getPointProjectionScatterColumns(output as PointProjectionColumns)
    });
    if (spherical) {
      const table = createTransientView(
        graph,
        `${id}-segment-table`,
        'float32',
        rowCount * SPHERICAL_SEGMENT_TABLE_STRIDE
      );
      return [
        createSphericalSegmentTableNode<Parameters>(graph, {
          id: `${id}-segment-table`,
          operation: OPERATION,
          positions: props.positions,
          pathOffsets: props.pathOffsets,
          table,
          overflow: props.overflow,
          candidateCount: props.candidateCount
        }),
        prefixNode,
        createSphericalProjectionNode<Parameters>(graph, {
          id: `${id}-project`,
          operation: OPERATION,
          points: props.points,
          positions: props.positions,
          pathOffsets: props.pathOffsets,
          table,
          vertexMeasures,
          radius: props.radius,
          sphereRadius: this.sphereRadius,
          results
        }),
        ...scatterNodes
      ];
    }
    const segmentStarts = createTransientView(graph, `${id}-segment-starts`, 'float32x2', rowCount);
    const segmentEnds = createTransientView(graph, `${id}-segment-ends`, 'float32x2', rowCount);
    const nearestSegments = createTransientView(
      graph,
      `${id}-nearest-segments`,
      'uint32',
      pointCount
    );
    return [
      createPathSegmentsNode<Parameters>(graph, {
        id: `${id}-segments`,
        operation: OPERATION,
        positions: props.positions,
        pathOffsets: props.pathOffsets,
        segmentStarts,
        segmentEnds
      }),
      prefixNode,
      ...new GPUNearestFeatureJoin({
        id: `${id}-join`,
        points: props.points,
        features: {kind: 'segments', starts: segmentStarts, ends: segmentEnds},
        radius: props.radius,
        candidateCapacity: props.candidateCapacity as number,
        leafCapacity: props.leafCapacity,
        spatialSort: props.spatialSort,
        nearestFeatureIds: nearestSegments,
        overflow: props.overflow,
        candidateCount: props.candidateCount
      }).getCommandNodes(graph),
      createPointProjectionNode<Parameters>(graph, {
        id: `${id}-project`,
        operation: OPERATION,
        points: props.points,
        positions: props.positions,
        pathOffsets: props.pathOffsets,
        nearestSegments,
        vertexMeasures,
        results
      }),
      ...scatterNodes
    ];
  }
}
