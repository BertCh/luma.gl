// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  importGraphBuffer,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  createBoundsNode,
  createFeatureRingsNode,
  getSpatialJoinFeatureCount,
  getSpatialJoinGeometryViews,
  validateSpatialJoinGeometry
} from './spatial-join-geometry';
import {
  getNextPowerOfTwo,
  getSortedFeatureBVHNodes,
  isPowerOfTwo,
  type SpatialJoinBVHStorage
} from './spatial-join-passes';
import type {GPUSpatialJoinGeometry} from './spatial-join-types';

/**
 * Persistent output storage of a {@link GPUSpatialJoinPrepared} handle.
 *
 * These views must keep their contents between encodings, so they are imported buffers and never
 * graph transients (the graph may reuse transient memory once a transient's last reader has run).
 */
export type GPUSpatialJoinPreparedStorage = SpatialJoinBVHStorage & {
  /** `(ringStart, ringEnd)` per feature. Present for polygon geometry only. */
  featureRings?: GraphDataView<'uint32x2'>;
};

/** Properties for {@link GPUSpatialJoinPrepared}. */
export type GPUSpatialJoinPreparedProps = {
  /** Prefix for generated node, buffer and transient IDs. Defaults to `'spatial-join-prepared'`. */
  id?: string;
  /**
   * The right-hand side to index. Joins that use this handle must be given the same geometry
   * views as their `right` (and, for `GPUPointInPolygonJoin`, the same polygon views).
   */
  geometry: GPUSpatialJoinGeometry;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. */
  leafCapacity?: number;
  /**
   * Reorder features along a Morton curve before the build. Tightens the tree for large,
   * spatially incoherent right-hand sides. Candidate order is then unspecified, so it is only
   * accepted by joins that do not promise sorted output (`GPUPointInPolygonJoin`). Default false.
   */
  spatialSort?: boolean;
  /**
   * Caller-owned persistent storage. When omitted the handle allocates and imports its own buffers
   * when it first produces nodes, and {@link GPUSpatialJoinPrepared.destroy} frees them.
   */
  storage?: GPUSpatialJoinPreparedStorage;
  /**
   * Optional CPU predicate evaluated once per encoding, with the encoding's parameters. Return true
   * to rebuild the index in that encoding (for example when the right-hand side changed). Combine
   * with {@link GPUSpatialJoinPrepared.invalidate}.
   */
  // biome-ignore lint/suspicious/noExplicitAny: the handle is shared by graphs with any Parameters.
  rebuildWhen?: (parameters: any) => boolean;
};

/**
 * A prepared (static) right-hand side: the bounds and BVH of a feature set, built once and reused
 * by every encoding until it is invalidated. This is the GPU analogue of GEOS `PreparedGeometry`
 * for the index stage: animated left features, such as moving points, query the same polygon tree
 * without rebuilding it.
 *
 * It reuses a build. It never caches results: every join still evaluates every candidate against
 * the current left geometry on every encoding.
 *
 * Usage: add the handle to the graph before the joins that use it, and hand it to each join as
 * `prepared`. The first encoding builds the index. Later encodings skip all build nodes (CPU
 * conditions, so no GPU work is recorded) until {@link invalidate} is called or `rebuildWhen`
 * returns true; the next encoding then rebuilds. If the right-hand side changes and the index is
 * not invalidated, joins keep querying the stale index (by design: the contract of a static
 * right-hand side). Changing view lengths still requires a new handle and graph.
 *
 * One handle belongs to one graph. Several joins in that graph may share it.
 */
export class GPUSpatialJoinPrepared implements GPUCommandNodeProducer {
  /** Prefix for every node, buffer and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialJoinPreparedProps;
  /** Indexed geometry. */
  readonly geometry: GPUSpatialJoinGeometry;
  /** Number of indexed features. */
  readonly featureCount: number;
  /** Resolved power-of-two BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Whether features are Morton sorted before the build. */
  readonly spatialSort: boolean;
  /** Number of encodings that rebuilt the index so far (counted on the CPU at encode time). */
  encodedBuildCount = 0;

  private graph: unknown;
  private preparedStorage: GPUSpatialJoinPreparedStorage | undefined;
  private ownedBuffers: Buffer[] = [];
  private dirty = true;
  private internalNodeCount = 0;

  constructor(props: GPUSpatialJoinPreparedProps) {
    this.id = props.id ?? 'spatial-join-prepared';
    this.props = props;
    this.geometry = props.geometry;
    validateSpatialJoinGeometry(this.id, 'geometry', props.geometry);
    this.featureCount = getSpatialJoinFeatureCount(props.geometry);
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.featureCount, 1));
    if (!isPowerOfTwo(this.leafCapacity)) {
      throw new Error(`${this.id} leafCapacity must be a positive power of two`);
    }
    this.spatialSort = props.spatialSort ?? false;
    this.preparedStorage = props.storage;
    if (props.storage) {
      this.validateStorage(props.storage);
    }
  }

  /**
   * Persistent BVH storage. Throws until the handle has produced nodes for a graph, because
   * self-allocated storage is imported into that graph.
   */
  get storage(): GPUSpatialJoinPreparedStorage {
    if (!this.preparedStorage) {
      throw new Error(`${this.id} storage is available after the handle is added to a graph`);
    }
    return this.preparedStorage;
  }

  /** Number of BVH internal nodes. Available after the handle has produced nodes. */
  get bvhInternalNodeCount(): number {
    return this.internalNodeCount;
  }

  /** Returns whether `getCommandNodes` has declared this handle in `graph`. */
  isDeclaredIn(graph: unknown): boolean {
    return this.graph !== undefined && this.graph === graph;
  }

  /** Forces the next encoding to rebuild the index. Call it after the right-hand side changed. */
  invalidate(): void {
    this.dirty = true;
  }

  /** Destroys buffers the handle allocated itself. Destroy compiled graphs that use it first. */
  destroy(): void {
    for (const buffer of this.ownedBuffers) {
      buffer.destroy();
    }
    this.ownedBuffers = [];
  }

  private validateStorage(storage: GPUSpatialJoinPreparedStorage): void {
    const nodeCount = 2 * this.leafCapacity - 1;
    const expectations: [string, GraphDataView | undefined, number][] = [
      ['nodeMinima', storage.nodeMinima, nodeCount],
      ['nodeMaxima', storage.nodeMaxima, nodeCount],
      ['nodeChildren', storage.nodeChildren, nodeCount],
      ['leafIds', storage.leafIds, this.leafCapacity],
      ['count', storage.count, 1],
      ['overflow', storage.overflow, 1]
    ];
    if (this.geometry.kind === 'polygons') {
      expectations.push(['featureRings', storage.featureRings, this.featureCount]);
    }
    for (const [name, view, length] of expectations) {
      if (!view || view.length < length) {
        throw new Error(`${this.id} storage.${name} must hold at least ${length} rows`);
      }
    }
  }

  private allocateStorage<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUSpatialJoinPreparedStorage {
    const {device} = graph;
    const nodeCount = 2 * this.leafCapacity - 1;
    const create = (
      name: string,
      format: 'float32x2' | 'uint32x2' | 'uint32',
      length: number
    ): GraphDataView<typeof format> => {
      const rowBytes = format === 'uint32' ? 4 : 8;
      const buffer = device.createBuffer({
        id: `${this.id}-${name}`,
        byteLength: Math.max(length, 1) * rowBytes,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      this.ownedBuffers.push(buffer);
      return importGraphBuffer(graph, `${this.id}-${name}`, buffer, format, length);
    };
    return {
      nodeMinima: create('node-minima', 'float32x2', nodeCount) as GraphDataView<'float32x2'>,
      nodeMaxima: create('node-maxima', 'float32x2', nodeCount) as GraphDataView<'float32x2'>,
      nodeChildren: create('node-children', 'uint32x2', nodeCount) as GraphDataView<'uint32x2'>,
      leafIds: create('leaf-ids', 'uint32', this.leafCapacity) as GraphDataView<'uint32'>,
      count: create('count', 'uint32', 1) as GraphDataView<'uint32'>,
      overflow: create('overflow', 'uint32', 1) as GraphDataView<'uint32'>,
      featureRings:
        this.geometry.kind === 'polygons'
          ? (create('feature-rings', 'uint32x2', this.featureCount) as GraphDataView<'uint32x2'>)
          : undefined
    };
  }

  /**
   * Returns the bounds, feature-ring and BVH build nodes, each conditioned on the rebuild
   * decision of the current encoding.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, geometry, featureCount} = this;
    if (this.graph) {
      throw new Error(
        `${id} was already added to ${this.graph === graph ? 'this graph' : 'another graph'}; ` +
          'a prepared handle belongs to one graph and is added once'
      );
    }
    this.graph = graph;
    if (!this.preparedStorage) {
      this.preparedStorage = this.allocateStorage(graph);
    }
    const storage = this.preparedStorage;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getSpatialJoinGeometryViews(geometry),
      storage.nodeMinima,
      storage.nodeMaxima,
      storage.nodeChildren,
      storage.leafIds,
      storage.count,
      storage.overflow,
      storage.featureRings
    ]);

    const minima = createTransientView(graph, `${id}-minima`, 'float32x2', featureCount);
    const maxima = createTransientView(graph, `${id}-maxima`, 'float32x2', featureCount);
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (geometry.kind === 'polygons') {
      nodes.push(
        createFeatureRingsNode<Parameters>(graph, `${id}-rings`, {
          featureCount,
          geometry,
          featureRings: storage.featureRings as GraphDataView<'uint32x2'>
        })
      );
    }
    nodes.push(
      createBoundsNode<Parameters>(graph, `${id}-bounds`, {
        featureCount,
        geometry,
        featureRings: storage.featureRings,
        minima,
        maxima
      })
    );
    const {bvh, nodes: bvhNodes} = getSortedFeatureBVHNodes(
      graph,
      id,
      'GPUSpatialJoinPrepared',
      minima,
      maxima,
      this.leafCapacity,
      this.spatialSort,
      storage
    );
    this.internalNodeCount = bvh.internalNodeCount;
    nodes.push(...bvhNodes);

    // Every node of one encoding sees the same decision: it is latched when the first node of the
    // encoding is evaluated and released after the last one.
    const nodeCount = nodes.length;
    let evaluations = 0;
    let latched = true;
    const evaluate = (parameters: unknown): boolean => {
      if (evaluations === 0) {
        latched = this.dirty || (this.props.rebuildWhen?.(parameters) ?? false);
        if (latched) {
          this.encodedBuildCount++;
        }
      }
      evaluations++;
      if (evaluations === nodeCount) {
        evaluations = 0;
        if (latched) {
          this.dirty = false;
        }
      }
      return latched;
    };
    return nodes.map(node => {
      if (node.condition) {
        throw new Error(`${id} node ${node.id} already has a condition`);
      }
      return {
        ...node,
        condition: {id: `${id}-rebuild`, source: 'cpu', evaluate}
      } as GPUCommandNode<Parameters>;
    });
  }
}
