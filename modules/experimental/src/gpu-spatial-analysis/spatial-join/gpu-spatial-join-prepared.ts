// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  importGraphBuffer,
  validateGraphViewsBelongToGraph,
  validatePowerOfTwo
} from '../../utils/gpu-contributor-utils';
import {
  validateGPUPartitionDescriptor,
  type GPUPartitionDescriptor
} from '../../analysis-core/contracts/partitioning';
import {
  createBoundsNode,
  createFeatureRingsNode,
  getSpatialJoinFeatureCount,
  getSpatialJoinGeometryViews,
  validateSpatialJoinGeometry
} from './spatial-join-geometry';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  getDefaultSpatialSort,
  getNextPowerOfTwo,
  getPolygonEdgeLayoutLength,
  getPolygonEdgeLayoutNodes,
  getSortedFeatureBVHNodes,
  type SpatialJoinBVHStorage
} from './spatial-join-passes';
import {getPolygonEdgeSlabNodes, type PolygonEdgeSlabs} from './polygon-edge-slabs';
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
  /** Feature-row partitions retained by the prepared index for tiled and streaming execution. */
  partitioning?: GPUPartitionDescriptor;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. */
  leafCapacity?: number;
  /**
   * Reorder features along a Hilbert curve before the build. Tightens the tree for large,
   * spatially incoherent right-hand sides. Candidate order is then unspecified, so it is only
   * accepted by joins that do not promise sorted output. Defaults on from 256 features; pass false
   * when using the handle with a contributor that requires source-ordered candidates.
   */
  spatialSort?: boolean;
  /**
   * Polygon geometry only. Also builds, once per rebuild, the polygon edge-slab index and edge
   * layout that `GPUPointInPolygonJoin` uses for its exact refine (`edgeSlabs`), so repeated joins
   * reuse it like the BVH. A join given this handle defaults its `edgeSlabs` to the handle's
   * value. Costs memory of about `(vertexCount + 4 * vertexCount)` pairs of words in persistent
   * storage. Default false: other join kinds do not read it.
   */
  edgeSlabs?: boolean;
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
  /** Whether features are spatially sorted before the build. */
  readonly spatialSort: boolean;
  /** Whether the handle also builds the polygon edge-slab index (polygon geometry only). */
  readonly edgeSlabs: boolean;
  /** Feature-row partitions associated with this index, when it is built per partition. */
  readonly partitioning?: GPUPartitionDescriptor;
  /** Number of encodings that rebuilt the index so far (counted on the CPU at encode time). */
  encodedBuildCount = 0;

  private graph: unknown;
  private preparedStorage: GPUSpatialJoinPreparedStorage | undefined;
  private ownedBuffers: Buffer[] = [];
  private dirty = true;
  private internalNodeCount = 0;
  private slabStorage: {slabs: PolygonEdgeSlabs; layout: GraphDataView<'uint32'>} | undefined;

  constructor(props: GPUSpatialJoinPreparedProps) {
    this.id = props.id ?? 'spatial-join-prepared';
    this.props = props;
    this.geometry = props.geometry;
    validateSpatialJoinGeometry(this.id, 'geometry', props.geometry);
    this.featureCount = getSpatialJoinFeatureCount(props.geometry);
    this.partitioning = props.partitioning;
    if (this.partitioning) {
      validateGPUPartitionDescriptor(`${this.id} partitioning`, this.partitioning);
      if (this.partitioning.length !== this.featureCount) {
        throw new Error(`${this.id} partitioning needs one row per feature`);
      }
    }
    this.edgeSlabs = props.edgeSlabs ?? false;
    if (this.edgeSlabs && props.geometry.kind !== 'polygons') {
      throw new Error(`${this.id} edgeSlabs needs polygon geometry`);
    }
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.featureCount, 1));
    validatePowerOfTwo(this.id, 'leafCapacity', this.leafCapacity);
    this.spatialSort = props.spatialSort ?? getDefaultSpatialSort(this.featureCount);
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
      throw new Error(`${this.id} storage needs graph.add first`);
    }
    return this.preparedStorage;
  }

  /**
   * Persistent polygon edge-slab index and edge layout built by this handle (`edgeSlabs`), or
   * undefined. Available after the handle has produced nodes.
   */
  get edgeSlabStorage(): {slabs: PolygonEdgeSlabs; layout: GraphDataView<'uint32'>} | undefined {
    return this.slabStorage;
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
    if (this.ownedBuffers.length > 0) {
      // The storage views pointed at the buffers just destroyed.
      this.preparedStorage = undefined;
      this.slabStorage = undefined;
    }
    this.ownedBuffers = [];
    this.graph = undefined;
    this.dirty = true;
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
        throw new Error(`${this.id} storage.${name} needs ${length} rows`);
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
   * Builds the edge-slab index and layout into transients, then copies the slab views into
   * persistent buffers (graph transients may be aliased after their last use). The layout is
   * written to persistent storage directly.
   */
  private getEdgeSlabNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    geometry: Extract<GPUSpatialJoinGeometry, {kind: 'polygons'}>
  ): GPUCommandNode<Parameters>[] {
    const {id} = this;
    const create = <Format extends 'uint32' | 'uint32x2'>(
      name: string,
      format: Format,
      length: number
    ): GraphDataView<Format> => {
      const buffer = graph.device.createBuffer({
        id: `${id}-${name}`,
        byteLength: Math.max(length, 1) * (format === 'uint32' ? 4 : 8),
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      this.ownedBuffers.push(buffer);
      return importGraphBuffer(graph, `${id}-${name}`, buffer, format, length);
    };
    const built = getPolygonEdgeSlabNodes<Parameters>(graph, {
      id: `${id}-edges`,
      operation: 'GPUSpatialJoinPrepared',
      owner: 'polygon',
      positions: geometry.positions,
      polygonOffsets: geometry.polygonOffsets,
      ringOffsets: geometry.ringOffsets
    });
    const {slabs} = built;
    const persistent: PolygonEdgeSlabs = {
      ...slabs,
      params: create('edge-slab-params', 'uint32', slabs.params.length),
      slabOffsets: create('edge-slab-offsets', 'uint32', slabs.slabOffsets.length),
      slabEntries: create('edge-slab-entries', 'uint32x2', slabs.slabEntries.length)
    };
    const layoutNodes = getPolygonEdgeLayoutNodes<Parameters>(graph, {
      id: `${id}-edges`,
      operation: 'GPUSpatialJoinPrepared',
      polygons: geometry,
      layout: create(
        'edge-layout',
        'uint32',
        getPolygonEdgeLayoutLength(geometry.positions.length, this.featureCount)
      )
    });
    this.slabStorage = {slabs: persistent, layout: layoutNodes.layout};
    const copy = (
      name: string,
      source: GraphDataView<'uint32' | 'uint32x2'>,
      destination: GraphDataView<'uint32' | 'uint32x2'>,
      wordCount: number
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edges-copy-${name}`,
        operation: 'GPUSpatialJoinPrepared',
        variant: 'copy-words',
        bindings: [
          {name: 'source', view: source, type: 'u32', access: 'read'},
          {name: 'destination', view: destination, type: 'u32', access: 'read_write'}
        ],
        invocationCount: Math.max(wordCount, 1),
        declarations: `const WORD_COUNT: u32 = ${wordCount}u;`,
        body: `if (index >= WORD_COUNT) { return; }
  destination[destinationOffset + index] = source[sourceOffset + index];`
      });
    return [
      ...built.nodes,
      ...layoutNodes.nodes,
      copy('params', slabs.params, persistent.params, slabs.params.length),
      copy('offsets', slabs.slabOffsets, persistent.slabOffsets, slabs.slabOffsets.length),
      copy('entries', slabs.slabEntries, persistent.slabEntries, 2 * slabs.slabEntries.length)
    ];
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
      throw new Error(`${id} is already added to a graph`);
    }
    // Validate the caller's views before allocating, so a rejected call leaves nothing behind.
    validateGraphViewsBelongToGraph(id, graph, getSpatialJoinGeometryViews(geometry));
    if (this.preparedStorage) {
      const caller = this.preparedStorage;
      validateGraphViewsBelongToGraph(id, graph, [
        caller.nodeMinima,
        caller.nodeMaxima,
        caller.nodeChildren,
        caller.leafIds,
        caller.count,
        caller.overflow,
        caller.featureRings
      ]);
    }
    this.graph = graph;
    if (!this.preparedStorage) {
      this.preparedStorage = this.allocateStorage(graph);
    }
    const storage = this.preparedStorage;

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

    if (this.edgeSlabs && geometry.kind === 'polygons') {
      nodes.push(...this.getEdgeSlabNodes(graph, geometry));
    }

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
      // The bounds, ring and BVH build nodes are created above without conditions.
      assert(!node.condition);
      return {
        ...node,
        condition: {id: `${id}-rebuild`, source: 'cpu', evaluate}
      } as GPUCommandNode<Parameters>;
    });
  }
}
