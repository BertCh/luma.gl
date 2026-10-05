// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUVisibilityWorkflow,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createMapGraphKernelNode,
  createMapGraphPublishNode,
  createTransientUint32Rows
} from '../map-graph-kernels';
import type {
  GPUMapGraphCompactOutput,
  GPUMapGraphPositions2D,
  GPUMapGraphRecipe,
  GPUMapGraphUint32Rows
} from '../map-graph-types';
import {
  getGraphViewChunks,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewTopology,
  validateGraphViewsBelongToGraph,
  validateMapGraphCompactOutput
} from '../map-graph-utils';
import {GPUNearestFeatureJoin} from './gpu-nearest-feature-join';
import {GPU_SPATIAL_JOIN_NO_FEATURE, type GPUNearestFeatureSource} from './spatial-join-types';

const OPERATION = 'GPUBufferSelection';

/**
 * Properties for {@link GPUBufferSelection}.
 *
 * Per-frame: `distance` and the contents of every input buffer (points, features, source IDs).
 * Compile-time: view lengths and chunk topology, the feature kind, `candidateCapacity`,
 * `leafCapacity`, `spatialSort`, `output.ids.length`, and which optional views exist.
 */
export type GPUBufferSelectionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'buffer-selection'`. */
  id?: string;
  /** Packed planar points to test. Contents are per-frame. At least one row is required. */
  points: GPUMapGraphPositions2D;
  /** Optional stable IDs written to `output.ids`, with the chunk topology of `points`. Row indices are used when omitted. */
  sourceIds?: GPUMapGraphUint32Rows;
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
   * Morton-sorts features before the BVH build, forwarded to `GPUNearestFeatureJoin`. Compile-time.
   * Results are identical either way; enable it for large feature sets that are not spatially
   * coherent in row order. Default false.
   */
  spatialSort?: boolean;
  /** Optional per-point 0/1 mask, chunked like `points`. Rewritten on every encoding. */
  outputMask?: GPUMapGraphUint32Rows;
  /**
   * Optional bounded stable-ID result in ascending row order. `output.ids.length` must be at
   * least one. `overflow` is set when the selection exceeds the capacity or when the join
   * overflowed.
   */
  output?: GPUMapGraphCompactOutput;
  /** Optional per-point planar distance to the nearest feature, or -1 outside. Chunked like `points`. */
  distances?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Optional per-point nearest feature row, or `GPU_SPATIAL_JOIN_NO_FEATURE`. Chunked like `points`. */
  nearestFeatureIds?: GPUMapGraphUint32Rows;
  /** Optional one-row flag: 1 when the join's BVH leaf or candidate capacity overflowed. */
  overflow?: GraphDataView<'uint32'>;
};

/**
 * Selects the points within a per-frame planar distance of point or polyline features.
 *
 * The recipe composes {@link GPUNearestFeatureJoin} (a feature BVH, candidate pairs, and exact
 * point-to-segment distances) with a mask kernel (`nearestFeatureIds != no feature`) and, when
 * `output` is requested, `GPUVisibilityWorkflow` over that mask.
 *
 * This differs from the join's own `matches` output, which is unordered (atomic append). The
 * buffer selection publishes a source-aligned 0/1 mask and stable IDs in ascending row order, so
 * results are deterministic and can drive highlighting or draw-call instancing directly.
 *
 * Non-goals: polygon buffers, per-feature distances, and geodesic distance. Distances are planar,
 * in the units of the input coordinates.
 */
export class GPUBufferSelection implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'buffer-selection';
  /** Validated properties. */
  readonly props: GPUBufferSelectionProps;
  /** Number of points. */
  readonly pointCount: number;

  constructor(props: GPUBufferSelectionProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const {points, sourceIds, outputMask, output, distances, nearestFeatureIds, overflow} = props;
    if (!outputMask && !output) {
      throw new Error(`${id} requires outputMask or output`);
    }
    this.pointCount = points.length;
    if (this.pointCount === 0) {
      throw new Error(`${id} requires at least one point`);
    }
    for (const chunk of getGraphViewChunks(points)) {
      validatePackedView(chunk, ['float32x2'], `${id} points`);
    }
    validatePackedView(props.distance, ['float32'], `${id} distance`);
    if (props.distance.length !== 1) {
      throw new Error(`${id} distance must contain one float32 row`);
    }
    for (const [name, view] of [
      ['sourceIds', sourceIds],
      ['outputMask', outputMask],
      ['nearestFeatureIds', nearestFeatureIds]
    ] as const) {
      for (const chunk of view ? getGraphViewChunks(view) : []) {
        validatePackedUint32View(chunk, `${id} ${name}`);
      }
      validateGraphViewTopology(id, name, points, view);
    }
    for (const chunk of distances ? getGraphViewChunks(distances) : []) {
      validatePackedView(chunk, ['float32'], `${id} distances`);
    }
    validateGraphViewTopology(id, 'distances', points, distances);
    if (overflow) {
      validatePackedUint32View(overflow, `${id} overflow`);
      if (overflow.length < 1) {
        throw new Error(`${id} overflow must contain one uint32 row`);
      }
    }
    if (output) {
      validateMapGraphCompactOutput(id, output);
      if (output.ids.length < 1) {
        throw new Error(`${id} output.ids must hold at least one row`);
      }
    }
    // Feature views and leafCapacity are validated by the join when getCommandNodes builds it.
    if (!Number.isSafeInteger(props.candidateCapacity) || props.candidateCapacity < 1) {
      throw new Error(`${id} candidateCapacity must be a positive integer`);
    }
    validateGraphOutputsDisjointFromInputs(id, getOutputs(props), getInputs(props));
  }

  /** Returns join, mask, and (with `output`) stable compaction and publish nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, pointCount} = this;
    const {points, sourceIds, outputMask, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const nodes: GPUCommandNode<Parameters>[] = [];

    const nearestFeatureIds =
      props.nearestFeatureIds ??
      createTransientUint32Rows(graph, `${id}-nearest-feature-ids`, points);
    const joinOverflow =
      props.overflow ?? createTransientView(graph, `${id}-join-overflow`, 'uint32', 1);
    nodes.push(
      ...new GPUNearestFeatureJoin({
        id: `${id}-join`,
        points,
        features: props.features,
        radius: props.distance,
        candidateCapacity: props.candidateCapacity,
        leafCapacity: props.leafCapacity,
        spatialSort: props.spatialSort,
        nearestFeatureIds,
        nearestDistances: props.distances,
        overflow: joinOverflow
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
        createMapGraphKernelNode<Parameters>(graph, {
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

    if (output) {
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
        createMapGraphPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          totalCount: selectedTotal,
          compactIds: selectedIds,
          output,
          overflowSources: [joinOverflow]
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
    props.points,
    props.sourceIds,
    props.distance,
    ...(features.kind === 'points' ? [features.positions] : [features.starts, features.ends])
  ];
}

/** Returns every caller-owned writable view of a buffer selection. */
function getOutputs(
  props: GPUBufferSelectionProps
): (GraphDataView | GraphVectorView | undefined)[] {
  const {output} = props;
  return [
    props.outputMask,
    props.distances,
    props.nearestFeatureIds,
    props.overflow,
    output?.ids,
    output?.count,
    output?.overflow,
    output?.totalCount
  ];
}
