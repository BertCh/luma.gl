// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUFlagOffsets,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLDispatchGate} from '../../utils/wgsl-kernel-nodes';
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import {
  getNextPowerOfTwo,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPUCardinalityPort} from '../../analysis-core/contracts/index';
import {GPUNearestFeatureJoin} from '../spatial-join/gpu-nearest-feature-join';
import {getSortedFeatureBVHNodes} from '../spatial-join/spatial-join-passes';
import {
  createHierarchicalDensityAddNode,
  createHierarchicalDensityClusterInfoNode,
  createHierarchicalDensityContributionsNode,
  createHierarchicalDensityEdgeKeysNode,
  createHierarchicalDensityGateNode,
  createHierarchicalDensityInitNode,
  createHierarchicalDensityLabelsNode,
  createHierarchicalDensityLeafFlagsNode,
  createHierarchicalDensityNodeDataNode,
  createHierarchicalDensityPointerInitNode,
  createHierarchicalDensityResolveNode,
  createHierarchicalDensitySearchNode,
  createHierarchicalDensityTieNode,
  HIERARCHICAL_DENSITY_CLUSTER_WORDS,
  HIERARCHICAL_DENSITY_DENDROGRAM_WORDS,
  HIERARCHICAL_DENSITY_EDGE_WORDS,
  HIERARCHICAL_DENSITY_NODE_WORDS,
  HIERARCHICAL_DENSITY_SELECTION_WORDS,
  HIERARCHICAL_DENSITY_STATE_WORDS,
  type HierarchicalDensityShape
} from './hierarchical-density-clustering-kernels';

import {
  getHierarchicalDensityDendrogramNodes,
  getHierarchicalDensityCondensingNodes,
  getHierarchicalDensitySelectionNodes
} from './hierarchical-density-tree';

/** Planar contexts suitable for distance measurements. */
const HIERARCHICAL_DENSITY_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

const OPERATION = 'GPUHierarchicalDensityClustering';

/** Largest `minimumSamples` of {@link GPUHierarchicalDensityClustering} (the k-nearest search keeps at most 32 neighbors). */
export const GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_SAMPLES = 32;

/**
 * Largest `positions` row count of {@link GPUHierarchicalDensityClustering}, bounding the
 * transient storage used by the MST, parallel dendrogram and condensed-tree contraction.
 */
export const GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_ROW_COUNT = 2 ** 20;

/** Caller-owned outputs of {@link GPUHierarchicalDensityClustering}. */
export type GPUHierarchicalDensityClusteringOutput = {
  /** One label per point: compact cluster ID `0..k-1`, or `0xffffffff` for noise. */
  labels: GraphDataView<'uint32'>;
  /**
   * Optional core distance per point: the distance to its `minimumSamples`-th nearest neighbor,
   * the point itself included (+infinity for non-finite points or fewer neighbors).
   */
  coreDistances?: GraphDataView<'float32'>;
  /** Optional one-row status; `count` receives the number of selected clusters. */
  status?: Partial<GPUCardinalityPort>;
};

/**
 * Properties for {@link GPUHierarchicalDensityClustering}.
 *
 * Per-frame (no recompile): the contents of `positions`. Compile-time: `positions.length`,
 * `minimumSamples`, `minimumClusterSize` and which optional outputs are present.
 */
export type GPUHierarchicalDensityClusteringProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'hierarchical-density-clustering'`. */
  id?: string;
  /** Planar only (the default); project longitude/latitude first. Other contexts throw. */
  spatialContext?: GPUSpatialContext;
  /**
   * Points, one row per point, at most {@link GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_ROW_COUNT}.
   * Non-finite points are noise and take no part in the hierarchy.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * Smallest condensed cluster, scikit-learn `min_cluster_size`, at least 2. Compile-time.
   */
  minimumClusterSize: number;
  /**
   * Neighbors of the core distance, scikit-learn `min_samples` (the point itself counts), from 1
   * to {@link GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_SAMPLES}. Defaults to
   * `min(minimumClusterSize, 32)`, scikit-learn's default when that is at most 32. Compile-time.
   */
  minimumSamples?: number;
  /** Caller-owned outputs. */
  output: GPUHierarchicalDensityClusteringOutput;
};

/**
 * HDBSCAN density clustering of planar points, fully on the GPU, matching scikit-learn's
 * `HDBSCAN` with its defaults (`cluster_selection_method='eom'`, `allow_single_cluster=False`,
 * `cluster_selection_epsilon=0`, Euclidean metric) up to a relabeling of the clusters.
 *
 * Composition:
 * 1. Core distances from a k-nearest search (`GPUNearestFeatureJoin` neighbors mode, the point
 *    itself as its first neighbor).
 * 2. The exact minimum spanning tree over the mutual reachability distance
 *    `max(core a, core b, |a - b|)` by Boruvka: at most `ceil(log2 rows) + 1` unrolled rounds of
 *    the same kernels (rule 12), gated on the GPU component count (rule 13). Each point walks a
 *    Hilbert-sorted point BVH for its nearest point of another component, pruning nodes by
 *    `max(core, box distance)` and nodes whose leaves all lie in its component (one inclusive scan
 *    of leaf component changes per round). Edges are totally ordered by `(weight, low row, high
 *    row)`, three integer atomic-min passes pick each component's minimum, and lock-free
 *    union-find hooks the components.
 * 3. A stable sort of the spanning tree by `(weight, low, high)` and the single-linkage dendrogram
 *    by concurrent atomic parent splicing, one invocation per edge. Euler-tour pointer doubling
 *    computes subtree sizes in logarithmic dispatch depth, including chain-shaped dendrograms.
 * 4. Condensing by pointer jumping (`ceil(log2(2 rows)) + 1` rounds of one kernel): each big node's
 *    cluster start and each point's lowest node of at least `minimumClusterSize` points, where it
 *    falls out of its cluster; compact cluster IDs with `GPUFlagOffsets`.
 * 5. Cluster stability as a fixed-order segmented sum of the condensed-tree rows, then the
 *    excess-of-mass selection by heavy-path tree contraction in logarithmic-squared dispatch
 *    depth. GPU cursors and indirect gates control the levels and convergence. Ancestor pointer
 *    doubling suppresses selected descendants; a flag scan assigns the compact labels.
 *
 * Ties: with equal mutual reachability distances several spanning trees exist; this one breaks
 * ties by point rows, which can differ from scikit-learn's order and, rarely, the selected
 * clusters. Duplicate points (zero distance) get the largest float32 density instead of
 * scikit-learn's infinity.
 *
 * Non-goals: `allow_single_cluster`, `cluster_selection_epsilon`, `'leaf'` selection, membership
 * probabilities, outlier scores, other metrics and 3D points.
 */
export class GPUHierarchicalDensityClustering implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUHierarchicalDensityClusteringProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved `minimumSamples`. */
  readonly minimumSamples: number;
  /** Number of unrolled Boruvka rounds. */
  readonly roundCount: number;

  constructor(props: GPUHierarchicalDensityClusteringProps) {
    this.id = props.id ?? 'hierarchical-density-clustering';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      HIERARCHICAL_DENSITY_SPATIAL_CONTEXT_SUPPORT
    );
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rows = props.positions.length;
    validateCapacity(
      id,
      'positions rows',
      rows,
      1,
      GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_ROW_COUNT
    );
    validateCapacity(id, 'minimumClusterSize', props.minimumClusterSize, 2);
    this.minimumSamples =
      props.minimumSamples ??
      Math.min(props.minimumClusterSize, GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_SAMPLES);
    validateCapacity(
      id,
      'minimumSamples',
      this.minimumSamples,
      1,
      GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_SAMPLES
    );
    const {output} = props;
    validatePackedView(output.labels, ['uint32'], `${id} output.labels`);
    if (output.labels.length !== rows) {
      throw new Error(`${id} output.labels length must equal positions length`);
    }
    if (output.coreDistances) {
      validatePackedView(output.coreDistances, ['float32'], `${id} output.coreDistances`);
      if (output.coreDistances.length !== rows) {
        throw new Error(`${id} output.coreDistances length must equal positions length`);
      }
    }
    if (output.status?.count) {
      validatePackedView(output.status.count, ['uint32'], `${id} output.status.count`);
      if (output.status.count.length < 1) {
        throw new Error(`${id} output.status.count must hold one row`);
      }
    }
    this.roundCount = Math.ceil(Math.log2(Math.max(rows, 2))) + 1;
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.labels, output.coreDistances, output.status?.count],
      [props.positions]
    );
  }

  /** Returns the k-nearest, Boruvka, dendrogram, condensing, stability, selection and label nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, minimumSamples} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      output.labels,
      output.coreDistances,
      output.status?.count
    ]);
    const rows = props.positions.length;
    // The k-nearest distances are one storage binding of `rows * minimumSamples` floats.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'neighbor distances',
      rows * minimumSamples * 4,
      'lower minimumSamples or the row count'
    );
    const leafCapacity = getNextPowerOfTwo(rows);
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', Math.max(length, 1));
    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Core distances.
    const neighborIds = u32('neighbor-ids', rows * minimumSamples);
    const neighborCounts = u32('neighbor-counts', rows);
    const neighborDistances = f32('neighbor-distances', rows * minimumSamples);
    const neighborOverflow = u32('neighbor-overflow', 1);
    nodes.push(
      ...new GPUNearestFeatureJoin({
        id: `${id}-neighbors`,
        queries: {kind: 'points', positions: props.positions},
        features: {kind: 'points', positions: props.positions},
        k: minimumSamples,
        output: {
          neighborIds,
          neighborCounts,
          neighborDistances,
          status: {overflow: neighborOverflow}
        }
      }).getCommandNodes(graph)
    );
    const state = u32('state', HIERARCHICAL_DENSITY_STATE_WORDS);
    const boxMinima = createTransientView(graph, `${id}-box-minima`, 'float32x2', rows);
    const boxMaxima = createTransientView(graph, `${id}-box-maxima`, 'float32x2', rows);
    const parents = u32('parents', rows);
    const coreDistances = output.coreDistances ?? f32('core-distances', rows);
    nodes.push(
      createFillStateNode(graph, `${id}-state-clear`, state),
      createHierarchicalDensityInitNode<Parameters>(graph, {
        id: `${id}-init`,
        shape: this.getShape(0, 0, 1),
        positions: props.positions,
        neighborDistances,
        neighborCounts,
        boxMinima,
        boxMaxima,
        parents,
        coreDistances,
        state
      })
    );
    const {bvh, nodes: bvhNodes} = getSortedFeatureBVHNodes<Parameters>(
      graph,
      `${id}-points`,
      OPERATION,
      boxMinima,
      boxMaxima,
      leafCapacity,
      rows >= 2
    );
    const shape = this.getShape(leafCapacity, bvh.internalNodeCount, bvh.levelCount);
    const nodeData = u32('node-data', bvh.nodeCount * HIERARCHICAL_DENSITY_NODE_WORDS);
    nodes.push(
      ...bvhNodes,
      createHierarchicalDensityNodeDataNode<Parameters>(graph, {
        id: `${id}-node-data`,
        shape,
        nodeMinima: bvh.nodeMinima,
        nodeMaxima: bvh.nodeMaxima,
        leafIds: bvh.leafIds,
        nodeData
      })
    );

    // 2. Boruvka rounds, gated on the GPU component count.
    const components = u32('components', rows);
    const componentBest = u32('component-best', rows * HIERARCHICAL_DENSITY_EDGE_WORDS);
    const candidates = u32('candidates', rows * HIERARCHICAL_DENSITY_EDGE_WORDS);
    const changeFlags = u32('change-flags', leafCapacity);
    const changeScan = u32('change-scan', leafCapacity);
    const edges = u32('edges', rows * HIERARCHICAL_DENSITY_EDGE_WORDS);
    const dispatchGate = createWGSLDispatchGate<Parameters>(graph, {
      id: `${id}-round-gate`,
      operation: OPERATION,
      shapes: [[rows], [leafCapacity]]
    });
    const rowGate = dispatchGate.getGate(0);
    const leafGate = dispatchGate.getGate(1);
    for (let round = 0; round < this.roundCount; round++) {
      const suffix = `-${round}`;
      nodes.push(
        createHierarchicalDensityGateNode<Parameters>(graph, {
          id: `${id}-gate${suffix}`,
          state,
          dispatchBinding: dispatchGate.binding,
          armWGSL: dispatchGate.armWGSL,
          disarmWGSL: dispatchGate.disarmWGSL
        }),
        createHierarchicalDensityResolveNode<Parameters>(graph, {
          id: `${id}-resolve${suffix}`,
          shape,
          positions: props.positions,
          parents,
          components,
          componentBest,
          gate: rowGate
        }),
        createHierarchicalDensityLeafFlagsNode<Parameters>(graph, {
          id: `${id}-leaf-flags${suffix}`,
          shape,
          nodeData,
          components,
          changeFlags,
          gate: leafGate
        }),
        ...new GPUScan({
          id: `${id}-leaf-scan${suffix}`,
          input: changeFlags,
          output: changeScan,
          mode: 'inclusive'
        }).getCommandNodes(graph),
        createHierarchicalDensitySearchNode<Parameters>(graph, {
          id: `${id}-search${suffix}`,
          shape,
          nodeData,
          changeScan,
          positions: props.positions,
          coreDistances,
          components,
          candidates,
          componentBest,
          gate: rowGate
        }),
        createHierarchicalDensityTieNode<Parameters>(graph, {
          id: `${id}-tie-low${suffix}`,
          shape,
          word: 1,
          components,
          candidates,
          componentBest,
          gate: rowGate
        }),
        createHierarchicalDensityTieNode<Parameters>(graph, {
          id: `${id}-tie-high${suffix}`,
          shape,
          word: 2,
          components,
          candidates,
          componentBest,
          gate: rowGate
        }),
        createHierarchicalDensityAddNode<Parameters>(graph, {
          id: `${id}-add${suffix}`,
          shape,
          components,
          componentBest,
          parents,
          edges,
          state,
          gate: rowGate
        })
      );
    }

    // 3. Spanning tree in (weight, low, high) order and the single-linkage dendrogram.
    const edgeIdentity = u32('edge-identity', rows);
    let edgeOrder: GraphDataView<'uint32'> | undefined;
    for (const word of [1, 0, 2] as const) {
      const keys = u32(`edge-keys-${word}`, rows);
      const sortedKeys = u32(`edge-sorted-keys-${word}`, rows);
      const nextOrder = u32(`edge-order-${word}`, rows);
      nodes.push(
        createHierarchicalDensityEdgeKeysNode<Parameters>(graph, {
          id: `${id}-edge-keys-${word}`,
          shape,
          word,
          state,
          edges,
          order: edgeOrder,
          identity: edgeOrder ? undefined : edgeIdentity,
          keys
        }),
        ...new GPUSort({
          id: `${id}-edge-sort-${word}`,
          keys,
          values: edgeOrder ?? edgeIdentity,
          outputKeys: sortedKeys,
          outputValues: nextOrder
        }).getCommandNodes(graph)
      );
      edgeOrder = nextOrder;
    }
    const sortedEdgeOrder = edgeOrder as GraphDataView<'uint32'>;
    const nodeCount = 2 * rows;
    const dendrogram = u32('dendrogram', nodeCount * HIERARCHICAL_DENSITY_DENDROGRAM_WORDS);
    nodes.push(
      ...getHierarchicalDensityDendrogramNodes(graph, {
        id: `${id}-dendrogram`,
        shape,
        state,
        edges,
        edgeOrder: sortedEdgeOrder,
        dendrogram
      })
    );

    // 4. Condensing by pointer jumping and compact cluster IDs.
    const pointers = u32('pointers', nodeCount * 2);
    const startFlags = u32('start-flags', nodeCount);
    const clusterOffsets = u32('cluster-offsets', nodeCount);
    const clusterCount = u32('cluster-count', 1);
    nodes.push(
      createHierarchicalDensityPointerInitNode<Parameters>(graph, {
        id: `${id}-pointer-init`,
        shape,
        dendrogram,
        pointers,
        startFlags
      })
    );
    nodes.push(
      ...getHierarchicalDensityCondensingNodes(graph, {
        id: `${id}-condensing`,
        rowCount: rows,
        pointers
      })
    );
    nodes.push(
      ...new GPUFlagOffsets({
        id: `${id}-cluster-offsets`,
        flags: startFlags,
        offsets: clusterOffsets,
        count: clusterCount
      }).getCommandNodes(graph)
    );

    // 5. Stability, excess-of-mass selection and labels.
    const {clusterCapacity} = shape;
    const clusterInfo = u32('cluster-info', clusterCapacity * HIERARCHICAL_DENSITY_CLUSTER_WORDS);
    const contributionKeys = u32('contribution-keys', nodeCount);
    const contributions = f32('contributions', nodeCount);
    const lowClusters = u32('low-clusters', rows);
    const stability = f32('stability', clusterCapacity);
    const selection = u32('selection', clusterCapacity * HIERARCHICAL_DENSITY_SELECTION_WORDS);
    const selectedCount = output.status?.count ?? u32('selected-count', 1);
    nodes.push(
      createHierarchicalDensityClusterInfoNode<Parameters>(graph, {
        id: `${id}-cluster-info`,
        shape,
        dendrogram,
        startFlags,
        clusterOffsets,
        pointers,
        clusterInfo
      }),
      createHierarchicalDensityContributionsNode<Parameters>(graph, {
        id: `${id}-contributions`,
        shape,
        dendrogram,
        startFlags,
        clusterOffsets,
        pointers,
        clusterInfo,
        contributionKeys,
        contributions,
        lowClusters
      }),
      ...getSortedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-stability`,
        operation: OPERATION,
        segmentCount: clusterCapacity,
        segmentKeys: contributionKeys,
        reductions: [{name: 'stability', contributions, output: stability}]
      }),
      ...getHierarchicalDensitySelectionNodes<Parameters>(graph, {
        id: `${id}-selection`,
        shape,
        clusterCount,
        stability,
        clusterInfo,
        selection,
        selectedCount
      }),
      createHierarchicalDensityLabelsNode<Parameters>(graph, {
        id: `${id}-labels`,
        shape,
        lowClusters,
        selection,
        labels: output.labels
      })
    );
    return nodes;
  }

  private getShape(
    leafCapacity: number,
    internalNodeCount: number,
    levelCount: number
  ): HierarchicalDensityShape {
    const rows = this.props.positions.length;
    return {
      rowCount: rows,
      minimumSamples: this.minimumSamples,
      minimumClusterSize: this.props.minimumClusterSize,
      clusterCapacity: Math.floor((2 * rows) / this.props.minimumClusterSize) + 1,
      leafCapacity,
      internalNodeCount,
      levelCount
    };
  }
}

function createFillStateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  state: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  return createFillNode<Parameters>(graph, {
    id,
    operation: OPERATION,
    view: state,
    type: 'u32',
    value: '0u'
  });
}
