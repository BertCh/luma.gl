// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUVisibilityWorkflow,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  createPublishNode,
  createTransientUint32Rows
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput, GPUUint32Rows} from '../../utils/gpu-contributor-types';
import type {GPUPointGeometryPort} from '../../analysis-core/contracts/index';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getGraphViewChunks,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewTopology,
  validateGraphViewsBelongToGraph,
  validateCompactOutput,
  validateCapacity,
  validatePackedView,
  validatePowerOfTwo
} from '../../utils/gpu-contributor-utils';
import {GPUNearestFeatureJoin} from './gpu-nearest-feature-join';
import {validateStatusScalars} from './spatial-join-geometry';
import type {SpatialSortCurve} from './spatial-join-passes';
import {GPU_SPATIAL_JOIN_NO_FEATURE, type GPUNearestFeatureSource} from './spatial-join-types';

import {
  GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUBufferSelection';

/** Caller-owned outputs of {@link GPUBufferSelection}. */
export type GPUBufferSelectionOutput = {
  /** Optional per-point 0/1 mask, chunked like the points. Rewritten on every encoding. */
  outputMask?: GPUUint32Rows;
  /**
   * Optional bounded stable-ID result in ascending row order. `matches.ids.length` must be at
   * least one. `matches.overflow` is set when the selection exceeds the capacity or when the join
   * overflowed.
   */
  matches?: GPUCompactOutput;
  /**
   * Optional one-row draw count: the clamped selected count (`matches.count`), written by the same
   * publish node. Bind it as the instance count of an indirect draw, or as the source of an
   * indirect-args copy, so the selected points are drawn without a readback or a hand-copied count.
   * Requires `matches`.
   */
  drawInstanceCount?: GraphDataView<'uint32'>;
  /** Optional per-point distance to the nearest feature (planar, or meters when great-circle), or -1 outside. Chunked like the points. */
  distances?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Optional per-point nearest feature row, or `GPU_SPATIAL_JOIN_NO_FEATURE`. Chunked like the points. */
  nearestFeatureIds?: GPUUint32Rows;
  /** Completeness scalars. */
  status?: {
    /** Optional one-row flag: 1 when the join's BVH leaf or candidate capacity overflowed. */
    overflow?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUBufferSelection}.
 *
 * Per-frame: `distance` and the contents of every input buffer (points, features, source IDs).
 * Compile-time: view lengths and chunk topology, the feature kind, `candidateCapacity`,
 * `leafCapacity`, `spatialSort`, `output.matches.ids.length`, and which optional views exist.
 */
export type GPUBufferSelectionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'buffer-selection'`. */
  id?: string;
  /**
   * Coordinate space and metric of the points and features, forwarded to
   * {@link GPUNearestFeatureJoin}. Omitted means native planar coordinates, with `distance` and
   * `distances` in coordinate units. `{coordinateSpace: 'longitude-latitude', metric:
   * 'great-circle'}` selects points within a great-circle distance in meters (mean Earth sphere,
   * or `sphereRadius` of the context; kilometers are rejected), with segment features straight in
   * longitude/latitude; `candidateCapacity` is then ignored. Other longitude/latitude metrics
   * throw at construction.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Points to test, possibly chunked; contents are per-frame and at least one row is required.
   * Optional `sourceIds` (same chunk topology) are written to `output.matches.ids`; row indices
   * are used when omitted.
   */
  points: GPUPointGeometryPort;
  /**
   * Point features, or segment features. Express a polyline as consecutive segment rows.
   * Contents are per-frame; the feature count is compile-time.
   */
  features: GPUNearestFeatureSource;
  /**
   * Per-frame buffer distance, one float32 row. A point is selected when its planar distance to
   * the nearest feature is less than or equal to this value: the comparison is inclusive and is
   * made in float32, exactly as `GPUNearestFeatureJoin` does (`distance >= 0 && distance <= radius`).
   * NaN, negative, or infinite values select nothing.
   */
  distance: GraphDataView<'float32'>;
  /** Maximum `(point, feature)` bounding-box candidates per encoding. Compile-time. */
  candidateCapacity: number;
  /** Power-of-two BVH leaf slots forwarded to the join. Compile-time. */
  leafCapacity?: number;
  /**
   * Spatially sorts features before the BVH build, forwarded to `GPUNearestFeatureJoin` (Hilbert order by default, see `spatialSortCurve`). Compile-time.
   * Results are identical either way. Defaults to on from 256 features, as in
   * `GPUNearestFeatureJoin`; pass `false` to skip it.
   */
  spatialSort?: boolean;
  /**
   * Experimental, compile-time. Curve used by `spatialSort`: `'hilbert'` (default; 10 to 15% faster joins
   * than Morton in paired A/B runs) or `'morton'` (Z-order). Results are identical; only BVH
   * traversal cost changes.
   */
  spatialSortCurve?: SpatialSortCurve;
  /**
   * Processes the points in Hilbert order over the feature BVH bounds, forwarded to
   * `GPUNearestFeatureJoin`. Compile-time. Results are identical either way. Defaults to on from
   * 4096 points and 256 features.
   */
  querySort?: boolean;
  /** Caller-owned outputs; at least one of `outputMask` and `matches` is required. */
  output: GPUBufferSelectionOutput;
};

/**
 * Selects the points within a per-frame planar distance of point or polyline features.
 *
 * The contributor composes {@link GPUNearestFeatureJoin} (a feature BVH, candidate pairs, and exact
 * point-to-segment distances) with a mask kernel (`nearestFeatureIds != no feature`) and, when
 * `output.matches` is requested, `GPUVisibilityWorkflow` over that mask.
 *
 * This differs from the join's own `matches` output, which is unordered (atomic append). The
 * buffer selection publishes a source-aligned 0/1 mask and stable IDs in ascending row order, so
 * results are deterministic and can drive highlighting or draw-call instancing directly.
 *
 * Non-goals: polygon buffers and per-feature distances. Distances are planar in the units of the
 * input coordinates, or great-circle meters with a longitude/latitude `spatialContext`.
 */
export class GPUBufferSelection implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUBufferSelectionProps;
  /** Number of points. */
  readonly pointCount: number;
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUBufferSelectionProps) {
    this.id = props.id ?? 'buffer-selection';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT
    );
    const points = props.points.positions;
    const {sourceIds} = props.points;
    const {outputMask, matches, distances, nearestFeatureIds, drawInstanceCount} = props.output;
    if (!outputMask && !matches) {
      throw new Error(`${id} needs output.outputMask or output.matches`);
    }
    this.pointCount = points.length;
    if (this.pointCount === 0) {
      throw new Error(`${id} needs at least one point`);
    }
    for (const chunk of getGraphViewChunks(points)) {
      validatePackedView(chunk, ['float32x2'], `${id} points.positions`);
    }
    validatePackedView(props.distance, ['float32'], `${id} distance`);
    if (props.distance.length !== 1) {
      throw new Error(`${id} distance needs one float32 row`);
    }
    for (const [name, view] of [
      ['points.sourceIds', sourceIds],
      ['output.outputMask', outputMask],
      ['output.nearestFeatureIds', nearestFeatureIds]
    ] as const) {
      for (const chunk of view ? getGraphViewChunks(view) : []) {
        validatePackedView(chunk, ['uint32'], `${id} ${name}`);
      }
      validateGraphViewTopology(id, name, points, view);
    }
    for (const chunk of distances ? getGraphViewChunks(distances) : []) {
      validatePackedView(chunk, ['float32'], `${id} output.distances`);
    }
    validateGraphViewTopology(id, 'output.distances', points, distances);
    validateStatusScalars(id, {overflow: props.output.status?.overflow});
    if (drawInstanceCount) {
      if (!matches) {
        throw new Error(`${id} drawInstanceCount needs matches`);
      }
      validatePackedView(drawInstanceCount, ['uint32'], `${id} output.drawInstanceCount`);
      if (drawInstanceCount.length < 1) {
        throw new Error(`${id} output.drawInstanceCount needs one row`);
      }
    }
    if (matches) {
      validateCompactOutput(id, matches);
      if (matches.ids.length < 1) {
        throw new Error(`${id} output.matches.ids needs at least one row`);
      }
    }
    // Feature views are validated by the join when getCommandNodes builds it; forwarded scalar
    // options are checked here so the error names this contributor.
    validateCapacity(id, 'candidateCapacity', props.candidateCapacity);
    if (props.leafCapacity !== undefined) {
      validatePowerOfTwo(id, 'leafCapacity', props.leafCapacity);
    }
    validateGraphOutputsDisjointFromInputs(id, getOutputs(props), getInputs(props));
  }

  /** Returns join, mask, and (with `output`) stable compaction and publish nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, pointCount} = this;
    const points = props.points.positions;
    const {sourceIds} = props.points;
    const {outputMask, matches} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const nodes: GPUCommandNode<Parameters>[] = [];

    const nearestFeatureIds =
      props.output.nearestFeatureIds ??
      createTransientUint32Rows(graph, `${id}-nearest-feature-ids`, points);
    const joinOverflow =
      props.output.status?.overflow ??
      createTransientView(graph, `${id}-join-overflow`, 'uint32', 1);
    nodes.push(
      ...new GPUNearestFeatureJoin({
        id: `${id}-join`,
        spatialContext: this.spatialContext,
        queries: {kind: 'points', positions: points},
        features: props.features,
        radius: props.distance,
        candidateCapacity: props.candidateCapacity,
        leafCapacity: props.leafCapacity,
        spatialSort: props.spatialSort,
        spatialSortCurve: props.spatialSortCurve,
        querySort: props.querySort,
        output: {
          nearestFeatureIds,
          nearestDistances: props.output.distances,
          status: {overflow: joinOverflow}
        }
      }).getCommandNodes(graph)
    );

    const mask = outputMask ?? createTransientUint32Rows(graph, `${id}-mask`, points);
    const idChunks = getGraphViewChunks(nearestFeatureIds);
    const maskChunks = getGraphViewChunks(mask);
    for (const [chunkIndex, idChunk] of idChunks.entries()) {
      if (idChunk.length === 0) {
        continue;
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: idChunks.length > 1 ? `${id}-mask-chunk-${chunkIndex}` : `${id}-mask`,
          operation: OPERATION,
          variant: 'mask',
          bindings: [
            {name: 'featureIds', view: idChunk, type: 'u32', access: 'read'},
            {name: 'maskOut', view: maskChunks[chunkIndex], type: 'u32', access: 'read_write'}
          ],
          invocationCount: idChunk.length,
          declarations: `const NO_FEATURE: u32 = ${GPU_SPATIAL_JOIN_NO_FEATURE}u;`,
          body: `maskOut[maskOutOffset + index] =
    select(0u, 1u, featureIds[featureIdsOffset + index] != NO_FEATURE);`
        })
      );
    }

    if (matches) {
      const selectedIds = createTransientView(graph, `${id}-selected-ids`, 'uint32', pointCount);
      const selectedTotal = createTransientView(graph, `${id}-selected-total`, 'uint32', 1);
      nodes.push(
        ...new GPUVisibilityWorkflow({
          id: `${id}-visibility`,
          predicates: [{kind: 'selection', mask}],
          output: selectedIds,
          count: selectedTotal,
          sourceIds
        }).getCommandNodes(graph)
      );
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          requiredCount: selectedTotal,
          compactIds: selectedIds,
          output: matches,
          overflowSources: [joinOverflow],
          extraCounts: props.output.drawInstanceCount ? [props.output.drawInstanceCount] : []
        })
      );
    }
    return nodes;
  }
}

/** Returns every read-only view of a buffer selection. */
function getInputs(
  props: GPUBufferSelectionProps
): (GraphDataView | GraphVectorView | undefined)[] {
  const {features} = props;
  return [
    props.points.positions,
    props.points.sourceIds,
    props.distance,
    ...(features.kind === 'points' ? [features.positions] : [features.starts, features.ends]),
    features.sourceIds
  ];
}

/** Returns every caller-owned writable view of a buffer selection. */
function getOutputs(
  props: GPUBufferSelectionProps
): (GraphDataView | GraphVectorView | undefined)[] {
  const {output} = props;
  const {matches} = output;
  return [
    output.outputMask,
    output.distances,
    output.nearestFeatureIds,
    output.status?.overflow,
    output.drawInstanceCount,
    matches?.ids,
    matches?.status.count,
    matches?.status.overflow,
    matches?.status.requiredCount
  ];
}
