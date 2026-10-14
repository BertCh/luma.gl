// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  getStorageBindingElementLimit,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
  validatePowerOfTwo
} from '../../utils/gpu-contributor-utils';
import {
  getGPUSphereRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPULineGeometryFields} from '../../analysis-core/contracts/index';
import {
  getGPUNearestFeatureJoinMaximumLeafCapacity,
  GPUNearestFeatureJoin
} from '../spatial-join/gpu-nearest-feature-join';
import {
  getDefaultSpatialSort,
  getNextPowerOfTwo,
  getSortedFeatureBVHNodes
} from '../spatial-join/spatial-join-passes';
import {createLinePrefixNode} from '../line-segmentize/line-segmentize-kernels';
import {
  createPackedScatterNodes,
  createLineSegmentsNode,
  createPointProjectionNode,
  createSphericalArcBoundsNode,
  createSphericalNodeDataNode,
  createSphericalProjectionNode,
  createSphericalSegmentTableNode,
  SPHERICAL_NODE_WORDS,
  getPointProjectionScatterColumns,
  LINEAR_REFERENCING_RESULT_STRIDE,
  SPHERICAL_SEGMENT_TABLE_STRIDE,
  type PointProjectionColumns
} from './linear-referencing-kernels';

const OPERATION = 'GPULinearReferencing';

// Line measures and nearest-line offsets measure distance.
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
});

/** Kernel model of {@link GPULinearReferencing}, derived from its `spatialContext` (not a prop). */
export type GPULinearReferencingCoordinateSystem = 'planar' | 'spherical';

/**
 * Outputs of {@link GPULinearReferencing}; each column optional, `queryPositions.length` rows. Points with
 * no line within the radius get `GPU_LINE_NO_SOURCE` indices, NaN fractions, foot points, measures,
 * normalized measures and signed offsets, distance `-1` and side `0`.
 *
 * With a great-circle `spatialContext`, foot points are longitude/latitude degrees, `fractions` is
 * the angular fraction along the great-circle arc, and distances, measures and signed offsets are
 * in sphere-radius units (meters by default).
 */
export type GPULinearReferencingOutput = {
  /** Index of the nearest line. */
  lineIndices?: GraphDataView<'uint32'>;
  /** Index of the nearest segment within its line (segment `k` joins vertices `k` and `k + 1`). */
  segmentIndices?: GraphDataView<'uint32'>;
  /** Position of the foot point along the segment, in `[0, 1]`. */
  fractions?: GraphDataView<'float32'>;
  /** Closest point on the line (turf `nearestPointOnLine`, PostGIS `ST_ClosestPoint`). */
  footPoints?: GraphDataView<'float32x2'>;
  /** Distance from the point to the foot point. */
  distances?: GraphDataView<'float32'>;
  /** Distance along the line from its first vertex to the foot point (`ST_LineLocatePoint` times length). */
  measures?: GraphDataView<'float32'>;
  /**
   * `measures` divided by the length of the nearest line, in `[0, 1]` (shapely
   * `line_locate_point(normalized=True)`); `0` for zero-length lines.
   */
  normalizedMeasures?: GraphDataView<'float32'>;
  /** `1` left of the segment direction, `-1` right, `0` on the line. */
  sides?: GraphDataView<'sint32'>;
  /** `side * distance`: signed lateral offset, positive to the left. */
  signedOffsets?: GraphDataView<'float32'>;
  /** Optional per-vertex cumulative line measure, `positions.length` rows. */
  vertexMeasures?: GraphDataView<'float32'>;
  /** Completeness of the search (BVH leaf capacity; planar also candidate capacity). */
  status: {
    /** One-row flag: 1 when the BVH leaf or candidate capacity overflowed. */
    overflow: GraphDataView<'uint32'>;
    /** Optional one-row unclamped candidate count. */
    candidateCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Largest `queryPositions` row count of {@link GPULinearReferencing} on `device`: the packed result
 * table holds 9 `uint32` words (36 bytes) per query in one storage binding, so 3728270 on a default
 * device (128 MiB bindings).
 */
export function getGPULinearReferencingMaximumQueryCount(device: Device): number {
  return getStorageBindingElementLimit(device, LINEAR_REFERENCING_RESULT_STRIDE * 4);
}

/**
 * Largest `positions` row count of {@link GPULinearReferencing} with a planar context on `device`:
 * the BVH over the segments takes `nextPowerOfTwo(positions)` leaf slots by default, so this is the
 * largest power of two that {@link getGPUNearestFeatureJoinMaximumLeafCapacity} allows, 2^23 on a
 * default device (128 MiB bindings).
 */
export function getGPULinearReferencingMaximumPlanarPositionCount(device: Device): number {
  return getGPUNearestFeatureJoinMaximumLeafCapacity(device);
}

/**
 * Largest `positions` row count of {@link GPULinearReferencing} with a great-circle context on
 * `device`: the arc table holds 16 `float32` words (64 bytes) per row in one storage binding, so
 * 2097152 on a default device (128 MiB bindings). A planar context has no such table.
 */
export function getGPULinearReferencingMaximumSphericalPositionCount(device: Device): number {
  return getStorageBindingElementLimit(device, SPHERICAL_SEGMENT_TABLE_STRIDE * 4);
}

/**
 * Properties for {@link GPULinearReferencing}.
 *
 * Per-frame (no recompile): `radius` and the contents of every input buffer. Compile-time: view
 * lengths, the line count, `spatialContext`, `candidateCapacity`, `leafCapacity`,
 * `spatialSort`, and which outputs are present.
 */
export type GPULinearReferencingProps = GPULineGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'linear-referencing'`. */
  id?: string;
  /**
   * Planar (default): Euclidean segments in position units. `{coordinateSpace: 'longitude-latitude',
   * metric: 'great-circle'}`: positions are longitude/latitude degrees, segments are great-circle
   * arcs, and measures, distances and offsets are in the context unit on its `sphereRadius`
   * (meters on the mean Earth radius by default). Replaces the former `coordinateSystem` and
   * `sphereRadius` props.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Packed query points (longitude/latitude degrees for a great-circle context), at most
   * {@link getGPULinearReferencingMaximumQueryCount} rows (3728270 on a default device). Qualified as `queryPositions` because
   * `positions` and `lineOffsets` (the line layout) hold the lines. A great-circle context also
   * limits `positions` to {@link getGPULinearReferencingMaximumSphericalPositionCount} rows
   * (2097152 on a default device), a planar one to
   * {@link getGPULinearReferencingMaximumPlanarPositionCount} (8388608).
   */
  queryPositions: GraphDataView<'float32x2'>;
  /**
   * Per-frame search radius, one float32 row. NaN, negative or infinite matches nothing.
   * On longitude/latitude this distance arrives as a GPU view, so no short-distance warning is possible;
   * below about 17 m use createGPULocalFrame (see Precision of longitude/latitude positions).
   */
  radius: GraphDataView<'float32'>;
  /**
   * Maximum `(point, segment)` bounding-box candidates per encoding. Required for `'planar'`;
   * ignored by a great-circle context, whose branch-and-bound arc BVH traversal needs no candidate
   * list.
   */
  candidateCapacity?: number;
  /**
   * Power-of-two BVH leaf slots (over segments, or over great-circle arcs). Defaults to the next
   * power of two of `positions.length`; fewer slots than positions sets `output.status.overflow`.
   */
  leafCapacity?: number;
  /**
   * Hilbert-sort segments (arcs) before the BVH build; see `GPUNearestFeatureJoin`. Results are
   * identical either way. Planar: default false. Great-circle: defaults to on from 256 positions.
   */
  spatialSort?: boolean;
  /** Per-point outputs and status. */
  output: GPULinearReferencingOutput;
};

/**
 * Snaps many points to the nearest of many polylines and reports linear-referencing coordinates:
 * line, segment, fraction, foot point, distance, measure along the line, side and signed offset
 * (turf `nearestPointOnLine`, PostGIS `ST_ClosestPoint` and `ST_LineLocatePoint`, route chainage,
 * map-matching candidates, lateral position against a corridor).
 *
 * Composition: one kernel turns every vertex row into a segment to the next row of the same line
 * (other rows become NaN segments that never match); `GPUNearestFeatureJoin` finds each point's
 * nearest segment within the radius through its BVH, with ties going to the smallest segment row
 * (so the earliest segment along the line, and the lowest line); a per-line sequential prefix
 * computes cumulative vertex measures (Neumaier-compensated, deterministic); one kernel projects
 * each point onto its segment in segment-local coordinates; scatter kernels copy the requested
 * columns.
 *
 * A great-circle `spatialContext` treats positions as longitude/latitude degrees and segments as
 * great-circle arcs (HaversineClosestPoint, cross-track and along-track distance): a per-row table
 * stores each arc's unit start vector, tangent and normal (built from the degree-difference
 * bearing, so meter-scale arcs stay well conditioned), a `GPUBVH` over conservative
 * longitude/latitude boxes of the arcs (including the great circle's latitude peak, every
 * longitude for arcs across the antimeridian or near a pole) is Hilbert sorted by default, and
 * each point walks it branch-and-bound with the great-circle box lower bound, evaluating exact arc
 * distances at the leaves (ties to the lowest segment row). Cost is about `log(segments)` arcs per
 * point instead of every arc. Distances, measures and offsets come out in sphere-radius units (meters by default);
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
  /** Number of lines. */
  readonly lineCount: number;
  /** Resolved spatial context (planar unless given). */
  readonly spatialContext: GPUSpatialContext;
  /** Kernel model derived from `spatialContext`; not a prop. */
  private readonly coordinateSystem: GPULinearReferencingCoordinateSystem;
  /** Resolved sphere radius in the context's length unit (1 for planar). */
  private readonly sphereRadius: number;

  constructor(props: GPULinearReferencingProps) {
    this.id = props.id ?? 'linear-referencing';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.coordinateSystem =
      this.spatialContext.coordinateSpace === 'longitude-latitude' ? 'spherical' : 'planar';
    this.sphereRadius =
      this.coordinateSystem === 'spherical' ? getGPUSphereRadius(this.spatialContext) : 1;
    const {id} = this;
    const {output} = props;
    if (
      this.coordinateSystem === 'planar' &&
      !(Number.isInteger(props.candidateCapacity) && (props.candidateCapacity as number) > 0)
    ) {
      throw new Error(`${id} planar needs candidateCapacity`);
    }
    // Delegates are built later, in getCommandNodes: check forwarded options now, under this id.
    if (props.leafCapacity !== undefined) {
      validatePowerOfTwo(id, 'leafCapacity', props.leafCapacity);
    }
    validatePackedView(props.queryPositions, ['float32x2'], `${id} queryPositions`);
    if (props.queryPositions.length < 1) {
      throw new Error(`${id} needs at least one query`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 2) {
      throw new Error(`${id} needs at least two positions`);
    }
    validatePackedView(props.lineOffsets, ['uint32'], `${id} lineOffsets`);
    if (props.lineOffsets.length < 2) {
      throw new Error(`${id} lineOffsets needs two rows`);
    }
    this.lineCount = props.lineOffsets.length - 1;
    const pointCount = props.queryPositions.length;
    const formats = {
      lineIndices: 'uint32',
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
    const {status, ...columns} = output;
    for (const [name, view] of Object.entries(columns) as [
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
    validatePackedView(status.overflow, ['uint32'], `${id} output.status.overflow`);
    validateGraphOutputsDisjointFromInputs(
      id,
      [...Object.values(columns), status.overflow, status.candidateCount],
      [props.queryPositions, props.positions, props.lineOffsets, props.radius]
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
    const {status, ...columns} = output;
    validateGraphViewsBelongToGraph(id, graph, [
      props.queryPositions,
      props.positions,
      props.lineOffsets,
      props.radius,
      status.overflow,
      status.candidateCount,
      ...Object.values(columns)
    ]);
    validateMaximumCount(
      id,
      'queryPositions rows',
      props.queryPositions.length,
      getGPULinearReferencingMaximumQueryCount(graph.device)
    );
    validateMaximumCount(
      id,
      'positions rows',
      props.positions.length,
      this.coordinateSystem === 'spherical'
        ? getGPULinearReferencingMaximumSphericalPositionCount(graph.device)
        : getGPULinearReferencingMaximumPlanarPositionCount(graph.device)
    );
    const rowCount = props.positions.length;
    const pointCount = props.queryPositions.length;
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
    const prefixNode = createLinePrefixNode<Parameters>(graph, {
      id: `${id}-vertex-measures`,
      operation: OPERATION,
      positions: props.positions,
      lineOffsets: props.lineOffsets,
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
      columns: getPointProjectionScatterColumns(columns as PointProjectionColumns)
    });
    if (spherical) {
      const table = createTransientView(
        graph,
        `${id}-segment-table`,
        'float32',
        rowCount * SPHERICAL_SEGMENT_TABLE_STRIDE
      );
      const arcMinima = createTransientView(graph, `${id}-arc-minima`, 'float32x2', rowCount);
      const arcMaxima = createTransientView(graph, `${id}-arc-maxima`, 'float32x2', rowCount);
      const leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(rowCount);
      const {bvh, nodes: bvhNodes} = getSortedFeatureBVHNodes<Parameters>(
        graph,
        `${id}-arcs`,
        OPERATION,
        arcMinima,
        arcMaxima,
        leafCapacity,
        props.spatialSort ?? getDefaultSpatialSort(rowCount)
      );
      const nodeData = createTransientView(
        graph,
        `${id}-node-data`,
        'uint32',
        bvh.nodeCount * SPHERICAL_NODE_WORDS
      );
      return [
        createSphericalSegmentTableNode<Parameters>(graph, {
          id: `${id}-segment-table`,
          operation: OPERATION,
          positions: props.positions,
          lineOffsets: props.lineOffsets,
          table,
          overflow: status.overflow,
          candidateCount: status.candidateCount
        }),
        createSphericalArcBoundsNode<Parameters>(graph, {
          id: `${id}-arc-bounds`,
          operation: OPERATION,
          positions: props.positions,
          table,
          arcMinima,
          arcMaxima
        }),
        ...bvhNodes,
        createSphericalNodeDataNode<Parameters>(graph, {
          id: `${id}-node-data`,
          operation: OPERATION,
          nodeMinima: bvh.nodeMinima,
          nodeMaxima: bvh.nodeMaxima,
          leafIds: bvh.leafIds,
          bvhOverflow: bvh.overflow,
          internalNodeCount: bvh.internalNodeCount,
          nodeCount: bvh.nodeCount,
          nodeData,
          overflow: status.overflow
        }),
        prefixNode,
        createSphericalProjectionNode<Parameters>(graph, {
          id: `${id}-project`,
          operation: OPERATION,
          points: props.queryPositions,
          positions: props.positions,
          lineOffsets: props.lineOffsets,
          table,
          vertexMeasures,
          radius: props.radius,
          sphereRadius: this.sphereRadius,
          nodeData,
          internalNodeCount: bvh.internalNodeCount,
          levelCount: bvh.levelCount,
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
      createLineSegmentsNode<Parameters>(graph, {
        id: `${id}-segments`,
        operation: OPERATION,
        positions: props.positions,
        lineOffsets: props.lineOffsets,
        segmentStarts,
        segmentEnds
      }),
      prefixNode,
      ...new GPUNearestFeatureJoin({
        id: `${id}-join`,
        queries: {kind: 'points', positions: props.queryPositions},
        features: {kind: 'segments', starts: segmentStarts, ends: segmentEnds},
        radius: props.radius,
        candidateCapacity: props.candidateCapacity as number,
        leafCapacity: props.leafCapacity,
        spatialSort: props.spatialSort,
        output: {
          nearestFeatureIds: nearestSegments,
          status: {overflow: status.overflow, candidateCount: status.candidateCount}
        }
      }).getCommandNodes(graph),
      createPointProjectionNode<Parameters>(graph, {
        id: `${id}-project`,
        operation: OPERATION,
        points: props.queryPositions,
        positions: props.positions,
        lineOffsets: props.lineOffsets,
        nearestSegments,
        vertexMeasures,
        results
      }),
      ...scatterNodes
    ];
  }
}
