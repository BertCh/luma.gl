// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  GPUFlagOffsets,
  GPUGridIndex,
  GPUGroupAggregation,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {getKeyGroupNodes, getKeyPairSortNodes} from '../spatial-weights/key-pair-grouping';
import {
  createSpatialClusteringBorderNode,
  createSpatialClusteringBoxCountsNode,
  createSpatialClusteringBoxKeysNode,
  createSpatialClusteringGridBoundsNode,
  createSpatialClusteringGridPositionsNode,
  createSpatialClusteringCentroidsNode,
  createSpatialClusteringCoreNode,
  createSpatialClusteringLabelsNode,
  createSpatialClusteringParentsInitNode,
  createSpatialClusteringResolveNode,
  createSpatialClusteringRootFlagsNode,
  createSpatialClusteringUnionNode
} from './spatial-clustering-kernels';
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import {GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH} from './spatial-clustering-parameters';

const OPERATION = 'GPUSpatialClustering';

/**
 * Properties for {@link GPUSpatialClustering}.
 *
 * Per-frame (no recompile): the contents of `positions`, `parameters` (bounds, epsilon, minimum
 * points) and `sourceIds`. Compile-time (needs a new graph): `positions.length`, `gridSize`,
 * `clusters.ids.length` (the cluster capacity), and which optional views are present.
 *
 * `gridSize` is only a maximum cell lattice. Every encoding derives the active lattice on the GPU
 * from the per-frame bounds and epsilon so each cell is at least epsilon wide. Results never
 * depend on `gridSize`, only the speed does: an epsilon that is small relative to
 * `bounds / gridSize` stays exact but puts more points in each cell, and a large epsilon simply
 * uses fewer cells.
 */
export type GPUSpatialClusteringProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-clustering'`. */
  id?: string;
  /** Packed planar points, one row per point. Compile-time length; per-frame contents. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Per-frame parameters: packed float32 view of at least
   * {@link GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH} elements written with
   * `getGPUSpatialClusteringParameterValues`. Invalid parameters (non-finite or non-positive
   * epsilon, non-finite or inverted bounds) exclude every point.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Maximum `[columns, rows]` of the neighbor-search cell lattice. Compile-time. Sizes the cell
   * buffers; `columns * rows` must be below `2^32 - 1`.
   */
  gridSize: readonly [number, number];
  /** Optional stable IDs, one per point, emitted in `clusters.ids`. Row indices otherwise. */
  sourceIds?: GraphDataView<'uint32'>;
  /** Caller-owned labels, one row per point: compact cluster ID `0..k-1`, or `0xffffffff` (noise). */
  labels: GraphDataView<'uint32'>;
  /**
   * Optional caller-owned root rows, one per point: the smallest core row of the point's cluster
   * (the row of its representative core point), or `0xffffffff` for noise. Border points report
   * the root of the cluster they were assigned to.
   */
  rootRows?: GraphDataView<'uint32'>;
  /** Optional caller-owned core flags, one per point: `1` for core points, else `0`. */
  coreFlags?: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of clusters `k`. */
  clusterCount?: GraphDataView<'uint32'>;
  /**
   * Optional bounded per-cluster output. `ids[labelId]` is the `sourceIds` entry (or row index) of
   * the cluster's root point, `count` is `min(k, ids.length)` and `overflow` is `1` when
   * `k > ids.length`. `ids.length` is the cluster capacity and must be at least one. Labels at or
   * above the capacity are still written to `labels` but excluded from sizes and centroids.
   */
  clusters?: GPUCompactOutput;
  /** Optional per-cluster member count (core and border points); requires `clusters`. Length equals `clusters.ids.length`. */
  clusterSizes?: GraphDataView<'uint32'>;
  /**
   * Optional per-cluster mean member position, `(0, 0)` for rows at or beyond the cluster count;
   * requires `clusters`. With `sumOrder: 'atomic'` sums use f32 atomics, so
   * the order of accumulation (and the last bits of the result) varies between runs. With the
   * default `sumOrder: 'sorted'` the sums are bitwise reproducible across repeated encodings on one
   * device: labels are canonical (independent of GPU scheduling),
   * so cluster slots and memberships are fixed, and members are summed in a fixed order.
   */
  clusterCentroids?: GraphDataView<'float32x2'>;
  /**
   * Accumulation order of the centroid sums. Compile-time; defaults to `'sorted'`.
   *
   * - `'sorted'`: stable sort of members by cluster slot, then a fixed-order segmented tree sum;
   *   bitwise reproducible on one device for identical inputs (adds a sort and two gathers).
   *   Reproducibility across different devices or drivers is not promised. Cost is flat whatever
   *   the cluster sizes.
   * - `'atomic'`: f32 compare-exchange atomic sums, last bits vary between runs. Members of one
   *   cluster retry on one address and serialize, so a few large clusters (typical for DBSCAN)
   *   make it slow; measured about 1.2x slower overall than `'sorted'` on 200k points in 4 blobs.
   *
   * Only affects `clusterCentroids`: `clusterSizes` are exact integer counts either way.
   */
  sumOrder?: 'atomic' | 'sorted';
  /**
   * FDBSCAN's dense-box shortcut. Compile-time; default `false`. Points are binned into a virtual
   * grid of boxes with side `0.7 * epsilon`, so every pair of points in one box is within epsilon.
   * A box holding at least `minimumPoints` points makes all of its points core without scanning
   * their neighbors. The result is identical to the plain algorithm (the shortcut only decides core
   * flags the scan would also reach), and the extra cost is two radix sorts of the points by box.
   * It pays off when epsilon is large relative to the point spacing so most points are core and
   * neighborhoods are crowded; sparse data gains nothing. Points whose box index exceeds 2^16
   * (extent over `0.7 * epsilon * 65536`) simply skip the shortcut.
   */
  denseBoxShortcut?: boolean;
};

/**
 * GPU DBSCAN density clustering of two-dimensional planar points.
 *
 * Canonical labeling, which the GPU result matches exactly:
 * - The neighborhood of a point is every valid point with `dx * dx + dy * dy <= epsilon * epsilon`
 *   (f32), the point itself included. Valid points are finite and inside the inclusive bounds;
 *   all others are noise and are never anyone's neighbor.
 * - A point is core when its neighborhood holds at least `minimumPoints` points.
 * - Clusters are the connected components of core points under the neighbor relation, and the
 *   root of a cluster is its smallest core row.
 * - A border point (valid, not core, with a core neighbor) joins the adjacent cluster with the
 *   smallest root. Every other point is noise.
 * - Compact cluster IDs rank the roots in ascending row order, so labels are deterministic and
 *   independent of GPU scheduling.
 *
 * Composition: grid-domain and grid-point kernels feeding `GPUGridIndex` (its domain is a buffer, so
 * bounds and epsilon never recompile; order within a cell is unspecified and nothing depends on
 * it), a core-point kernel, a lock-free CAS union-find kernel that hooks the
 * larger root under the smaller, resolve and border kernels, `GPUFlagOffsets` for compact IDs,
 * and optional `GPUCompaction`, `GPUGroupAggregation` and publish nodes for cluster outputs.
 *
 * Non-goals: OPTICS or HDBSCAN, geodesic distances, 3D points, chunked positions, incremental
 * clustering across frames, and cluster shapes or hulls (compose `GPUGroupGeometry` and
 * `GPUGroupConvexHull` over the cluster labels).
 */
export class GPUSpatialClustering implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialClusteringProps;

  constructor(props: GPUSpatialClusteringProps) {
    this.id = props.id ?? 'spatial-clustering';
    this.props = props;
    const id = this.id;
    const rows = props.positions.length;

    if (
      props.sumOrder !== undefined &&
      props.sumOrder !== 'atomic' &&
      props.sumOrder !== 'sorted'
    ) {
      throw new Error(`${id} sumOrder must be atomic or sorted`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH} float32 values`
      );
    }
    const [columns, gridRows] = props.gridSize;
    if (
      !Number.isInteger(columns) ||
      !Number.isInteger(gridRows) ||
      columns < 1 ||
      gridRows < 1 ||
      columns * gridRows >= 0xffffffff
    ) {
      throw new Error(
        `${id} gridSize must be two positive integers with columns * rows < 2^32 - 1`
      );
    }
    for (const [name, view] of [
      ['sourceIds', props.sourceIds],
      ['labels', props.labels],
      ['rootRows', props.rootRows],
      ['coreFlags', props.coreFlags]
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedUint32View(view, `${id} ${name}`);
      if (view.length !== rows) {
        throw new Error(`${id} ${name} length must equal positions length`);
      }
    }
    if (props.clusterCount) {
      validatePackedUint32View(props.clusterCount, `${id} clusterCount`);
      if (props.clusterCount.length < 1) {
        throw new Error(`${id} clusterCount must contain one uint32 row`);
      }
    }
    if (props.clusters) {
      validateCompactOutput(id, props.clusters);
      if (props.clusters.ids.length < 1) {
        throw new Error(`${id} clusters.ids must hold at least one row`);
      }
    }
    if ((props.clusterSizes || props.clusterCentroids) && !props.clusters) {
      throw new Error(`${id} clusterSizes and clusterCentroids require clusters`);
    }
    const capacity = props.clusters?.ids.length ?? 0;
    if (props.clusterSizes) {
      validatePackedUint32View(props.clusterSizes, `${id} clusterSizes`);
      if (props.clusterSizes.length !== capacity) {
        throw new Error(`${id} clusterSizes length must equal clusters.ids length`);
      }
    }
    if (props.clusterCentroids) {
      validatePackedView(props.clusterCentroids, ['float32x2'], `${id} clusterCentroids`);
      if (props.clusterCentroids.length !== capacity) {
        throw new Error(`${id} clusterCentroids length must equal clusters.ids length`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.labels,
        props.rootRows,
        props.coreFlags,
        props.clusterCount,
        props.clusters?.ids,
        props.clusters?.count,
        props.clusters?.overflow,
        props.clusters?.totalCount,
        props.clusterSizes,
        props.clusterCentroids
      ],
      [props.positions, props.parameters, props.sourceIds]
    );
  }

  /** Bins the points into dense boxes and writes each point's box population. */
  private _getBoxCountNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    boxCounts: GraphDataView<'uint32'>
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.positions.length;
    const boxHigh = createTransientView(graph, `${id}-box-high`, 'uint32', rows);
    const boxLow = createTransientView(graph, `${id}-box-low`, 'uint32', rows);
    const boxSort = getKeyPairSortNodes(graph, `${id}-box`, OPERATION, rows, boxHigh, boxLow);
    const boxGroups = getKeyGroupNodes(
      graph,
      `${id}-box`,
      OPERATION,
      rows,
      boxHigh,
      boxLow,
      boxSort.sortedItems
    );
    return [
      createSpatialClusteringBoxKeysNode<Parameters>(graph, {
        id: `${id}-box-keys`,
        positions: props.positions,
        parameters: props.parameters,
        gridSize: props.gridSize,
        boxHigh,
        boxLow
      }),
      ...boxSort.nodes,
      ...boxGroups.nodes,
      createSpatialClusteringBoxCountsNode<Parameters>(graph, {
        id: `${id}-box-counts`,
        sortedItems: boxSort.sortedItems,
        groupIndex: boxGroups.groupIndex,
        groupStarts: boxGroups.groupStarts,
        boxHigh,
        boxCounts
      })
    ];
  }

  /** Returns the clustering nodes in dependency order; the publish node, when any, is last. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {positions, parameters, gridSize, clusters} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      parameters,
      props.sourceIds,
      props.labels,
      props.rootRows,
      props.coreFlags,
      props.clusterCount,
      clusters?.ids,
      clusters?.count,
      clusters?.overflow,
      clusters?.totalCount,
      props.clusterSizes,
      props.clusterCentroids
    ]);
    const rows = positions.length;
    const capacity = clusters?.ids.length ?? 0;
    const cellCount = gridSize[0] * gridSize[1];
    const wantsCentroids = Boolean(props.clusterCentroids);
    const nodes: GPUCommandNode<Parameters>[] = [];

    const clusterCount =
      props.clusterCount ?? createTransientView(graph, `${id}-cluster-count`, 'uint32', 1);

    if (rows === 0) {
      // No points: every per-point output is empty, so only the scalar and cluster outputs remain.
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-cluster-count`,
          operation: OPERATION,
          view: clusterCount,
          type: 'u32',
          value: '0u',
          componentCount: 1
        })
      );
      if (props.clusterSizes) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-cluster-sizes`,
            operation: OPERATION,
            view: props.clusterSizes,
            type: 'u32',
            value: '0u'
          })
        );
      }
      if (props.clusterCentroids) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-cluster-centroids`,
            operation: OPERATION,
            view: props.clusterCentroids,
            type: 'f32',
            value: '0.0',
            componentCount: capacity * 2
          })
        );
      }
      if (clusters) {
        nodes.push(
          createPublishNode<Parameters>(graph, {
            id: `${id}-publish`,
            operation: OPERATION,
            totalCount: clusterCount,
            output: clusters
          })
        );
      }
      return nodes;
    }

    const gridBounds = createTransientView(graph, `${id}-grid-bounds`, 'float32', 4);
    const gridPositions = createTransientView(graph, `${id}-grid-positions`, 'float32x2', rows);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rows);
    const cellOffsets = createTransientView(graph, `${id}-cell-offsets`, 'uint32', cellCount + 1);
    const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rows);
    const gridCount = createTransientView(graph, `${id}-grid-count`, 'uint32', 1);
    const gridOverflow = createTransientView(graph, `${id}-grid-overflow`, 'uint32', 1);
    const coreFlags =
      props.coreFlags ?? createTransientView(graph, `${id}-core-flags`, 'uint32', rows);
    const parents = createTransientView(graph, `${id}-parents`, 'uint32', rows);
    const rootLabels =
      props.rootRows ?? createTransientView(graph, `${id}-root-labels`, 'uint32', rows);
    const rootFlags = createTransientView(graph, `${id}-root-flags`, 'uint32', rows);
    const clusterOffsets = createTransientView(graph, `${id}-cluster-offsets`, 'uint32', rows);
    const boxCounts = props.denseBoxShortcut
      ? createTransientView(graph, `${id}-box-counts`, 'uint32', rows)
      : undefined;
    const gridViews = {
      positions,
      parameters,
      sortedRows,
      gridCount,
      cellOffsets
    };

    nodes.push(
      createSpatialClusteringGridBoundsNode<Parameters>(graph, {
        id: `${id}-grid-bounds`,
        parameters,
        gridSize,
        gridBounds
      }),
      createSpatialClusteringGridPositionsNode<Parameters>(graph, {
        id: `${id}-grid-positions`,
        positions,
        parameters,
        gridSize,
        gridPositions,
        rowIds
      }),
      ...new GPUGridIndex({
        id: `${id}-grid-index`,
        positions: gridPositions,
        gridSize: [gridSize[0], gridSize[1]],
        bounds: [0, 0, 1, 1],
        boundsBuffer: gridBounds,
        cellOffsets,
        objectIds: sortedRows,
        count: gridCount,
        overflow: gridOverflow
      }).getCommandNodes(graph),
      ...(boxCounts ? this._getBoxCountNodes(graph, boxCounts) : []),
      // The core pass visits valid points only, so excluded rows need their zero flag first.
      createFillNode<Parameters>(graph, {
        id: `${id}-core-flags-clear`,
        operation: 'GPUSpatialClustering',
        view: coreFlags,
        type: 'u32',
        value: '0u'
      }),
      createSpatialClusteringCoreNode<Parameters>(graph, {
        id: `${id}-core`,
        ...gridViews,
        gridSize,
        coreFlags,
        boxCounts
      }),
      createSpatialClusteringParentsInitNode<Parameters>(graph, {
        id: `${id}-parents-init`,
        parents
      }),
      createSpatialClusteringUnionNode<Parameters>(graph, {
        id: `${id}-union`,
        ...gridViews,
        gridSize,
        coreFlags,
        parents
      }),
      createSpatialClusteringResolveNode<Parameters>(graph, {
        id: `${id}-resolve`,
        coreFlags,
        parents,
        rootLabels
      }),
      createSpatialClusteringBorderNode<Parameters>(graph, {
        id: `${id}-border`,
        ...gridViews,
        gridSize,
        coreFlags,
        rootLabels
      }),
      createSpatialClusteringRootFlagsNode<Parameters>(graph, {
        id: `${id}-root-flags`,
        coreFlags,
        rootLabels,
        rootFlags
      }),
      ...new GPUFlagOffsets({
        id: `${id}-cluster-offsets`,
        flags: rootFlags,
        offsets: clusterOffsets,
        count: clusterCount
      }).getCommandNodes(graph)
    );

    const memberXs = wantsCentroids
      ? createTransientView(graph, `${id}-member-xs`, 'float32', rows)
      : undefined;
    const memberYs = wantsCentroids
      ? createTransientView(graph, `${id}-member-ys`, 'float32', rows)
      : undefined;
    nodes.push(
      createSpatialClusteringLabelsNode<Parameters>(graph, {
        id: `${id}-labels`,
        positions,
        rootLabels,
        clusterOffsets,
        labels: props.labels,
        memberXs,
        memberYs
      })
    );

    if (!clusters) {
      return nodes;
    }

    const rootIds = createTransientView(graph, `${id}-root-ids`, 'uint32', rows);
    const compactCount = createTransientView(graph, `${id}-compact-count`, 'uint32', 1);
    const rowIdsForCompaction = props.sourceIds ?? rowIds;
    nodes.push(
      ...new GPUCompaction({
        id: `${id}-cluster-roots`,
        input: rowIdsForCompaction,
        flags: rootFlags,
        output: rootIds,
        count: compactCount
      }).getCommandNodes(graph)
    );

    const sizes =
      props.clusterSizes ??
      (wantsCentroids
        ? createTransientView(graph, `${id}-cluster-size-values`, 'uint32', capacity)
        : undefined);
    if (sizes) {
      nodes.push(
        ...new GPUGroupAggregation({
          id: `${id}-cluster-sizes`,
          keys: props.labels,
          output: sizes
        }).getCommandNodes(graph)
      );
    }
    if (props.clusterCentroids && sizes && memberXs && memberYs) {
      const sumsX = createTransientView(graph, `${id}-cluster-sum-values-x`, 'float32', capacity);
      const sumsY = createTransientView(graph, `${id}-cluster-sum-values-y`, 'float32', capacity);
      if (props.sumOrder !== 'atomic') {
        // Labels at or above the capacity (and noise) sort last and never contribute; `sizes`
        // counts exactly the labels below the capacity, as the segmented sum requires.
        nodes.push(
          ...getSortedSegmentSumNodes<Parameters>(graph, {
            id: `${id}-cluster-sums`,
            operation: OPERATION,
            segmentCount: capacity,
            segmentKeys: props.labels,
            segmentCounts: sizes,
            reductions: [
              {name: 'x', contributions: memberXs, output: sumsX},
              {name: 'y', contributions: memberYs, output: sumsY}
            ]
          })
        );
      } else {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-cluster-sums-x`,
            keys: props.labels,
            values: memberXs,
            output: sumsX,
            operation: 'sum'
          }).getCommandNodes(graph),
          ...new GPUGroupAggregation({
            id: `${id}-cluster-sums-y`,
            keys: props.labels,
            values: memberYs,
            output: sumsY,
            operation: 'sum'
          }).getCommandNodes(graph)
        );
      }
      nodes.push(
        createSpatialClusteringCentroidsNode<Parameters>(graph, {
          id: `${id}-cluster-centroids`,
          sizes,
          sumsX,
          sumsY,
          centroids: props.clusterCentroids
        })
      );
    }

    nodes.push(
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        totalCount: clusterCount,
        compactIds: rootIds,
        output: clusters
      })
    );
    return nodes;
  }
}
