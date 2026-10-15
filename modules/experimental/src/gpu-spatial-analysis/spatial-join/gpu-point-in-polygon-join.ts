// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPULonLatPolygonStatusPort} from '../../analysis-core/contracts/index';
import {
  createLonLatPolygonCheckNode,
  hasLonLatPolygonStatus
} from '../../analysis-core/lon-lat-ring-wgsl';
import {createFillNode} from '../../utils/wgsl-kernel-nodes';

import {
  createTransientView,
  type GPUBVH,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';
import {GPU_POINT_IN_POLYGON_CLASSIFICATION} from '../../geospatial/gpu-pairwise-point-in-polygon';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';
import {
  GPU_LINEAR_EDGE_MODELS,
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {
  getPolygonEdgeSlabBindings,
  getPolygonEdgeSlabLoopWGSL,
  getPolygonEdgeSlabNodes,
  getPolygonEdgeSlabQueryWGSL,
  type PolygonEdgeSlabs
} from './polygon-edge-slabs';
import {createWGSLActiveCountDispatch, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput, GPUUint32Rows} from '../../utils/gpu-contributor-types';
import type {
  GPUPackedGeometryPort,
  GPUPointGeometryPort,
  GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getGraphViewChunks,
  validateGraphViewsBelongToGraph,
  validateCapacity,
  validateCompactOutput,
  validatePackedView,
  validatePowerOfTwo
} from '../../utils/gpu-contributor-utils';
import {
  createSpatialJoinBoundsNode,
  createSpatialJoinClearNode,
  createSpatialJoinFinalizeNode,
  getSortedFeatureBVHNodes,
  type SpatialSortCurve,
  getDefaultSpatialSort,
  getNextPowerOfTwo,
  getSpatialJoinAssignNodes,
  getSpatialJoinCollectNodes,
  getDefaultEdgeSlabs,
  getDefaultQuerySort,
  getPolygonEdgeLayoutNodes,
  getSpatialJoinProbeNodes,
  validateDisjointOutputs,
  validateMatchingChunks
} from './spatial-join-passes';
import {
  getSpatialJoinGeometryViews,
  isSameSpatialJoinGeometry,
  validateSpatialJoinGeometry,
  validateStatusScalars
} from './spatial-join-geometry';
import type {GPUSpatialJoinPrepared} from './gpu-spatial-join-prepared';

const OPERATION = 'GPUPointInPolygonJoin';

/**
 * The spatial context to give a `GPUPointInPolygonJoin` for an operation resolved with `context`:
 * a longitude/latitude context becomes containment-only (`metric: 'none'`, polygon edges straight
 * in degrees). The declared edge model and datum are preserved so unsupported edges still fail
 * validation; any other context is passed through unchanged.
 */
export function getPointInPolygonContainmentContext(context: GPUSpatialContext): GPUSpatialContext {
  return context.coordinateSpace === 'longitude-latitude'
    ? {
        coordinateSpace: 'longitude-latitude',
        metric: 'none',
        ...(context.edgeModel === undefined ? {} : {edgeModel: context.edgeModel}),
        ...(context.datum === undefined ? {} : {datum: context.datum})
      }
    : context;
}

/** Planar coordinates, or longitude/latitude with no metric (topological containment). */
const SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  edgeModels: GPU_LINEAR_EDGE_MODELS,
  metrics: Object.freeze(['none', 'native'] as const)
});

/**
 * How {@link GPUPointInPolygonJoin} treats a point exactly on a polygon boundary: `'half-open'`
 * (each boundary point goes to exactly one feature of a valid coverage), `'closed'` (boundary counts
 * as contained, the smallest row wins) or `'open'` (boundary belongs to no feature).
 */
export type GPUPointInPolygonBoundaryRule = 'half-open' | 'closed' | 'open';

/** Bit of a classification word set when a boundary point is owned under the half-open rule. */
const HALF_OPEN_OWNER_BIT = 0x100;

/** Caller-owned outputs of {@link GPUPointInPolygonJoin}. */
export type GPUPointInPolygonJoinOutput = {
  /**
   * Per-point containing feature ID (`polygons.sourceIds`) or row, or `GPU_SPATIAL_JOIN_NO_FEATURE`.
   * Chunked like `points.positions`.
   */
  pointFeatureIds: GPUUint32Rows;
  /** Optional per-feature count of assigned points. */
  featureCounts?: GraphDataView<'uint32'>;
  /** Optional compact stable IDs of points that joined a feature (unspecified order). */
  matches?: GPUCompactOutput;
  /**
   * Completeness scalars. Optional `antimeridianUncertainCount` counts unprepared crossing rings;
   * optional `poleUncertainCount` counts unprepared pole-enclosing rings. Prepare these polygons
   * with `GPULonLatPolygonPreparation`. Both fields are left untouched in planar contexts.
   */
  status: GPULonLatPolygonStatusPort & {
    /** One-row flag: 1 when BVH leaf, candidate, or `matches` capacity overflowed. */
    overflow: GraphDataView<'uint32'>;
    /** Optional one-row unclamped candidate count, for sizing `candidateCapacity`. */
    candidateCount?: GraphDataView<'uint32'>;
    /**
     * Optional one-row count of refined (point, feature) candidates classified `uncertain`.
     * Uncertain pairs are never assigned: a point whose only candidates were uncertain gets no
     * feature, so a nonzero value means the assignments and `featureCounts` may be incomplete.
     */
    uncertainCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUPointInPolygonJoin}.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths and chunking,
 * `candidateCapacity`, `leafCapacity`, `boundaryRule`, `spatialSort`, and which optional views exist.
 */
export type GPUPointInPolygonJoinProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'point-in-polygon-join'`. */
  id?: string;
  /**
   * Planar points, possibly chunked. Optional `sourceIds` (same chunk topology) are the IDs
   * written to `output.matches`.
   */
  points: GPUPointGeometryPort;
  /**
   * Polygon or multipolygon features in the four-level layout (ring 0 of a polygon is its shell,
   * later rings are holes; rings close implicitly). Optional `sourceIds` are written to
   * `output.pointFeatureIds` instead of feature rows.
   */
  polygons: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
  /**
   * Optional prepared (static) polygon set over the same `polygons` views. Its bounds and BVH are reused across encodings until
   * the handle is invalidated, so animated points query a fixed polygon tree without rebuilding
   * it. The handle must be added to the graph before the join. Its `leafCapacity` and
   * `spatialSort` replace this join's. Reuses a build only; assignments are recomputed every encoding.
   */
  prepared?: GPUSpatialJoinPrepared;
  /** Maximum `(point, feature)` bounding-box candidates per encoding. */
  candidateCapacity: number;
  /** Power-of-two BVH leaf slots. Defaults to the next power of two of the feature count. */
  leafCapacity?: number;
  /**
   * Compile-time. How a point exactly on a polygon boundary is treated. The decision uses the exact
   * orientation predicate, so "exactly on" is exact for the f32 vertices.
   *
   * - `'half-open'` (default): the boundary is split by the crossing-number convention of a
   *   half-open region, so a point on a boundary shared by several features of a valid coverage
   *   belongs to exactly one of them, whatever the feature rows are. The owner is the feature that
   *   contains the point after an infinitesimal move toward +x and +y (+y dominates), which means
   *   an edge or vertex belongs to the feature whose interior is to its right/above: left and
   *   bottom boundaries are included, right and top boundaries are excluded. A horizontal edge
   *   belongs to the feature above it, a vertical edge to the feature on its right, a vertex to the
   *   feature whose interior touches the upper right of the vertex. Points on the outer right or
   *   top boundary of a coverage belong to no feature (they are not lost by a bug: the rule that
   *   makes interior boundaries unique makes those open).
   * - `'closed'`: every boundary point counts as contained by every feature whose boundary it
   *   touches, then the smallest feature row wins (the assignment keeps one feature per point).
   *   Nothing is lost on the outer boundary, but the owner of a shared-boundary point depends on
   *   the feature row order.
   * - `'open'`: boundary points belong to no feature (only strict interior counts).
   *
   * A point strictly inside is unaffected by the rule. Overlapping features (an invalid coverage)
   * still resolve to the smallest containing row. `matches` lists exactly the points that get a
   * feature under the same rule.
   */
  boundaryRule?: GPUPointInPolygonBoundaryRule;
  /**
   * Coordinate space and metric of the points and polygons. Omitted means native planar
   * coordinates. Containment is a topological test evaluated in the plane of the given
   * coordinates: polygon edges are straight lines in the coordinates (as in GeoJSON), not
   * geodesics. `{coordinateSpace: 'longitude-latitude', metric: 'none'}` is therefore accepted and
   * behaves exactly like planar. A longitude/latitude context with any metric (for example
   * `'great-circle'`) throws at construction because there are no geodesic edges; project to a
   * planar frame first. Prepare polygons that cross ±180 or enclose a pole with
   * `GPULonLatPolygonPreparation`; optional status counters report unprepared rings.
   * Longitude/latitude requires `edgeModel: 'linear'` (the default); great-circle edges require preparation.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `polygons` when declared separately from `spatialContext`. Must be compatible
   * with `spatialContext` (datum, coordinate space, metric, units, planar frame).
   * Longitude/latitude must use linear edges.
   * Omitted means `spatialContext` describes both inputs.
   */
  polygonsSpatialContext?: GPUSpatialContext;
  /**
   * Compile-time. When true, features are reordered along a Hilbert curve (see `spatialSortCurve`) of their bound
   * centers before the BVH build, so leaves that are close in space are close in the tree and
   * internal node bounds stay tight. Empty or invalid features sort last.
   *
   * Results are identical either way (assignment still keeps the smallest containing feature row); only traversal cost changes. Default false.
   *
   * Measured on an Apple M3 Pro with 14,400 small square
   * features and 250,000 points: shuffled rows drop from 218 ms to 6.7 ms per encoding (about 9,500
   * to 61 BVH nodes visited per point); row-major rows are 6.0 ms unsorted and 5.6 ms sorted.
   *
   * Defaults to on from 256 features and off below, where the BVH is too shallow for the order to
   * matter. On coherent data the sort costs about 1 ms; pass `false` to skip it. Candidate order is
   * then unspecified (the point-in-polygon result is not affected).
   */
  spatialSort?: boolean;
  /**
   * Experimental, compile-time. Curve used by `spatialSort`: `'hilbert'` (default; 10 to 15% faster joins
   * than Morton in paired A/B runs) or `'morton'` (Z-order). Results are identical; only BVH
   * traversal cost changes.
   */
  spatialSortCurve?: SpatialSortCurve;
  /**
   * Compile-time. When true, each chunk of points is processed in Hilbert order of the points
   * (quantized over the root bounds of the polygon BVH, non-finite points last) instead of input
   * order, so neighboring invocations walk neighboring BVH subtrees. Per-point results, candidate
   * slots and counts stay addressed by the original row, so `pointFeatureIds`, `matches`,
   * `featureCounts`, the candidate pairs and what a `candidateCapacity` overflow drops are
   * identical either way; only traversal cost changes. Costs one key pass and a radix sort per point
   * chunk, which is why chunks of fewer than two points are never sorted.
   *
   * Defaults to on from 4096 points and 256 features (provisional; see
   * `spatial-join-query-order.spec.ts` for the A/B harness), and off below, where the sort cannot
   * pay for itself.
   */
  querySort?: boolean;
  /**
   * Compile-time. When true, the exact refine of a candidate `(point, feature)` pair tests only
   * the edges of the feature's y-slab at `point.y` (a prepared edge-slab index over the polygon
   * parts) instead of every edge of the feature. The classification is a parity of edges that
   * straddle `point.y` plus an on-boundary test on edges whose closed y-range contains it, so every
   * edge that can change the result is in the slab: `pointFeatureIds`, `matches`, `featureCounts`,
   * `candidateCount` and `uncertainCount` are identical for all three boundary rules, including the
   * half-open owner and malformed offsets (UNCERTAIN). Parity is kept per polygon part and
   * combined with OR, exactly as the unslabbed walk does, because each part is its own slab owner.
   *
   * One difference, at the numeric edge: the exact orientation test answers UNCERTAIN when its
   * terms span more than 200 binary orders of magnitude. The unslabbed walk reports that for any
   * edge of the feature; the slab refine only sees edges of the slab and so classifies a pair that
   * the walk would have reported UNCERTAIN because of an irrelevant far edge. Features with a
   * non-finite or `|coordinate| >= 1e30` vertex are never slabbed, so they keep the walk's
   * behavior exactly.
   *
   * Features under 8 vertices per part are walked. Costs a few passes over the polygon vertices
   * (done once in a {@link GPUSpatialJoinPrepared} created with `edgeSlabs: true`, which this join
   * then reuses) and changes the candidate buffer layout (four words per candidate). Defaults to
   * on when `polygons.positions.length * candidateCapacity` is at least 4,000,000 and features
   * average 8 vertices (provisional; see `spatial-join-edge-slabs.spec.ts`), or when `prepared`
   * carries the index; pass `false` to refine by walking.
   */
  edgeSlabs?: boolean;
  /** Caller-owned outputs. */
  output: GPUPointInPolygonJoinOutput;
};

/**
 * Joins planar points to the polygon features that contain them.
 *
 * A `GPUBVH` over feature bounds produces bounding-box candidates, an exact even/odd classifier
 * (exact orientation signs, see `boundaryRule`) refines them, and each point keeps the containing
 * feature with the smallest row.
 *
 * **Boundary points.** By default (`boundaryRule: 'half-open'`) a point on an edge or vertex shared
 * by several features of a valid coverage is assigned to exactly one of them, independent of the
 * feature row order, so counts over a coverage partition the points (only points on the outer right
 * and top boundary are left out). `'closed'` and `'open'` select the previous behaviors; see
 * {@link GPUPointInPolygonJoinProps.boundaryRule}. Zonal statistics rely on the half-open default. Per-feature counts and an optional compact list of joined points are produced on
 * the GPU. Every capacity reports overflow on the GPU.
 *
 * Candidates whose containment cannot be proven (`uncertain`) are not assigned and are not
 * silently ignored: they are counted in `uncertainCount`. Read it to know whether assignments may
 * be incomplete.
 */
export class GPUPointInPolygonJoin implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointInPolygonJoinProps;
  /** Number of features, `featureOffsets.length - 1`. */
  readonly featureCount: number;
  /** Resolved BVH leaf capacity. */
  readonly leafCapacity: number;
  /** Candidate pair capacity. */
  readonly candidateCapacity: number;
  /** Resolved boundary rule. */
  readonly boundaryRule: GPUPointInPolygonBoundaryRule;
  /** Resolved spatial context: native planar coordinates. */
  readonly spatialContext: GPUSpatialContext;
  /** Whether features are spatially sorted before the BVH build. */
  readonly spatialSort: boolean;
  /** Whether point chunks are processed in Hilbert order. */
  readonly querySort: boolean;
  /** Whether the exact refine tests only the edges of the y-slab at the point. */
  readonly edgeSlabs: boolean;

  constructor(props: GPUPointInPolygonJoinProps) {
    this.id = props.id ?? 'point-in-polygon-join';
    this.props = props;
    const {id} = this;
    const {points, polygons, output} = props;
    const {status} = output;
    if (points.kind !== 'points' || polygons.kind !== 'polygons') {
      throw new Error(`${id} needs point and polygon geometry`);
    }
    for (const chunk of getGraphViewChunks(points.positions)) {
      validatePackedView(chunk, ['float32x2'], `${id} points.positions`);
    }
    validateSpatialJoinGeometry(id, 'polygons', polygons, {allowEmpty: true});
    this.featureCount = polygons.featureOffsets.length - 1;
    for (const [name, view] of [
      ['output.pointFeatureIds', output.pointFeatureIds],
      ['points.sourceIds', points.sourceIds]
    ] as const) {
      for (const chunk of view ? getGraphViewChunks(view) : []) {
        validatePackedView(chunk, ['uint32'], `${id} ${name}`);
      }
      validateMatchingChunks(id, name, points.positions, view);
    }
    for (const [name, view] of [
      ['polygons.sourceIds', polygons.sourceIds],
      ['output.featureCounts', output.featureCounts]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length !== this.featureCount) {
          throw new Error(`${id} ${name} needs one row per feature`);
        }
      }
    }
    validateStatusScalars(id, status);
    if (output.matches) {
      validateCompactOutput(id, output.matches);
    }
    this.candidateCapacity = props.candidateCapacity;
    // Candidate pairs are stored as two words each in a uint32-addressed buffer.
    validateCapacity(id, 'candidateCapacity', this.candidateCapacity, 1, 0x7ffffffe);
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(Math.max(this.featureCount, 1));
    validatePowerOfTwo(id, 'leafCapacity', this.leafCapacity);
    if (
      props.spatialContext?.coordinateSpace === 'longitude-latitude' &&
      props.spatialContext.metric !== 'none'
    ) {
      throw new Error(`${id} needs metric 'none' on longitude-latitude`);
    }
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, SUPPORT);
    resolveGPUSecondarySpatialContext(
      id,
      'polygons',
      props.polygonsSpatialContext,
      this.spatialContext,
      SUPPORT.edgeModels
    );
    if (
      props.boundaryRule !== undefined &&
      !['half-open', 'closed', 'open'].includes(props.boundaryRule)
    ) {
      throw new Error(`${id} boundaryRule must be 'half-open', 'closed' or 'open'`);
    }
    this.boundaryRule = props.boundaryRule ?? 'half-open';
    this.spatialSort =
      props.prepared?.spatialSort ?? props.spatialSort ?? getDefaultSpatialSort(this.featureCount);
    this.querySort =
      props.querySort ?? getDefaultQuerySort(props.points.positions.length, this.featureCount);
    this.edgeSlabs =
      props.edgeSlabs ??
      props.prepared?.edgeSlabs ??
      getDefaultEdgeSlabs(polygons.positions.length, this.candidateCapacity, this.featureCount);
    if (props.prepared) {
      this.leafCapacity = props.prepared.leafCapacity;
      if (!isSameSpatialJoinGeometry(props.prepared.geometry, polygons)) {
        throw new Error(`${id} prepared must index the same polygon views`);
      }
    }
    validateDisjointOutputs(id, getInputs(props), getOutputs(props));
  }

  /** Returns bounds, BVH, candidate, refinement, assignment, and finalize nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount, leafCapacity, candidateCapacity} = this;
    const {polygons, output} = props;
    const points = props.points.positions;
    const {matches, status} = output;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const pointCount = points.length;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (
      this.spatialContext.coordinateSpace === 'longitude-latitude' &&
      hasLonLatPolygonStatus(status)
    ) {
      for (const name of ['antimeridianUncertainCount', 'poleUncertainCount'] as const) {
        const view = status[name];
        if (view) {
          nodes.push(
            createFillNode(graph, {
              id: `${id}-clear-${name}`,
              operation: OPERATION,
              view,
              type: 'u32',
              value: '0u',
              componentCount: 1
            })
          );
        }
      }
      nodes.push(
        createLonLatPolygonCheckNode(graph, {
          id: `${id}-lon-lat-polygon-check-0`,
          operation: OPERATION,
          positions: polygons.positions,
          ringOffsets: polygons.ringOffsets,
          status
        })
      );
    }

    if (props.prepared && !props.prepared.isDeclaredIn(graph)) {
      throw new Error(`${id} requires prepared to be added to the graph first`);
    }
    let bvh: Pick<
      GPUBVH,
      'nodeMinima' | 'nodeMaxima' | 'leafIds' | 'overflow' | 'internalNodeCount'
    >;
    if (props.prepared) {
      const {storage} = props.prepared;
      bvh = {...storage, internalNodeCount: props.prepared.bvhInternalNodeCount};
    } else {
      const minima = createTransientView(graph, `${id}-feature-minima`, 'float32x2', featureCount);
      const maxima = createTransientView(graph, `${id}-feature-maxima`, 'float32x2', featureCount);
      if (featureCount > 0) {
        nodes.push(
          createSpatialJoinBoundsNode<Parameters>(graph, {
            id: `${id}-bounds`,
            operation: OPERATION,
            featureCount,
            source: polygons,
            minima,
            maxima
          })
        );
      }
      const built = getSortedFeatureBVHNodes(
        graph,
        id,
        OPERATION,
        minima,
        maxima,
        leafCapacity,
        this.spatialSort,
        undefined,
        this.props.spatialSortCurve
      );
      nodes.push(...built.nodes);
      bvh = built.bvh;
    }

    const state = createTransientView(graph, `${id}-state`, 'uint32', 4);
    const assignment = createTransientView(graph, `${id}-assignment`, 'uint32', pointCount);
    // With slabs a candidate is four words in one binding: row, feature, point x and y bits.
    const useSlabs = this.edgeSlabs;
    const pairStride = useSlabs ? 4 : 2;
    const candidatePairs = createTransientView(
      graph,
      `${id}-candidate-pairs`,
      'uint32x2',
      candidateCapacity * (pairStride / 2)
    );
    const candidatePoints = useSlabs
      ? undefined
      : createTransientView(graph, `${id}-candidate-points`, 'float32x2', candidateCapacity);
    let slabs: PolygonEdgeSlabs | undefined;
    let edgeLayout: GraphDataView<'uint32'> | undefined;
    if (useSlabs) {
      const preparedSlabs = props.prepared?.edgeSlabStorage;
      if (preparedSlabs) {
        ({slabs, layout: edgeLayout} = preparedSlabs);
      } else {
        const built = getPolygonEdgeSlabNodes<Parameters>(graph, {
          id: `${id}-edges`,
          operation: OPERATION,
          owner: 'polygon',
          positions: polygons.positions,
          polygonOffsets: polygons.polygonOffsets,
          ringOffsets: polygons.ringOffsets
        });
        const layoutNodes = getPolygonEdgeLayoutNodes<Parameters>(graph, {
          id: `${id}-edges`,
          operation: OPERATION,
          polygons
        });
        nodes.push(...built.nodes, ...layoutNodes.nodes);
        slabs = built.slabs;
        edgeLayout = layoutNodes.layout;
      }
    }
    const pairClassifications = createTransientView(
      graph,
      `${id}-pair-classifications`,
      'uint32',
      candidateCapacity
    );
    nodes.push(
      createSpatialJoinClearNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: OPERATION,
        state,
        assignment,
        pointCount,
        featureCounts: output.featureCounts
      })
    );
    nodes.push(
      ...getSpatialJoinProbeNodes<Parameters>(graph, {
        id: `${id}-probe`,
        operation: OPERATION,
        points,
        bvh,
        featureCount,
        candidateCapacity,
        state,
        candidatePairs,
        candidatePoints,
        packedCandidates: useSlabs,
        pointRowScale: 1,
        pointRowOffset: 0,
        querySort: this.querySort
      })
    );
    const candidateDispatch = createWGSLActiveCountDispatch(graph, {
      id: `${id}-candidate-dispatch`,
      operation: OPERATION,
      count: state,
      maximumItemCount: candidateCapacity
    });
    nodes.push(candidateDispatch.updateNode);
    if (slabs && edgeLayout) {
      nodes.push(
        this.createSlabClassifyNode(graph, {
          condition: candidateDispatch.condition,
          state,
          candidatePairs,
          pairClassifications,
          slabs,
          edgeLayout
        })
      );
    } else {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-classify`,
          operation: OPERATION,
          variant: 'classify-exact',
          record:
            this.boundaryRule === 'half-open'
              ? {formula: 'gpu-spatial-analysis.point-in-polygon-join'}
              : undefined,
          bindings: [
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
            {name: 'candidatePoints', view: candidatePoints!, type: 'f32', access: 'read'},
            {name: 'featureOffsets', view: polygons.featureOffsets, type: 'u32', access: 'read'},
            {name: 'polygonPositions', view: polygons.positions, type: 'f32', access: 'read'},
            {name: 'polygonOffsets', view: polygons.polygonOffsets, type: 'u32', access: 'read'},
            {name: 'ringOffsets', view: polygons.ringOffsets, type: 'u32', access: 'read'},
            {
              name: 'pairClassifications',
              view: pairClassifications,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: candidateCapacity,
          condition: candidateDispatch.condition,
          declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
  const POLYGON_COUNT: u32 = ${polygons.polygonOffsets.length - 1}u;
  const RING_COUNT: u32 = ${polygons.ringOffsets.length - 1}u;
  const VERTEX_COUNT: u32 = ${polygons.positions.length}u;
  const OUTSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.outside}u;
  const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
  const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
  const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;
  const HALF_OPEN_OWNER_BIT: u32 = ${HALF_OPEN_OWNER_BIT}u;
  ${EXACT_ORIENTATION_WGSL}
  fn readVertex(vertexIndex: u32) -> vec2f {
    return vec2f(
      polygonPositions[polygonPositionsOffset + vertexIndex * 2u],
      polygonPositions[polygonPositionsOffset + vertexIndex * 2u + 1u]
    );
  }
  // Exact even/odd classification, or UNCERTAIN when some orientation has no sign. A BOUNDARY result
  // also carries HALF_OPEN_OWNER_BIT when the point belongs to the feature under the half-open rule.
  //
  // Half-open rule: classify the point moved by (epsilon^2, epsilon) with epsilon infinitesimal. The
  // crossing test against an edge only changes when the point is on the edge's supporting line, where
  // the sign of the moved point is the sign of dx (the epsilon term), or of -dy for a vertical edge
  // (the epsilon^2 term). The y comparisons are unchanged because a vertex y is either equal to the
  // point's y or at least one ulp away.
  fn classifyExactly(point: vec2f, geometryStart: u32, geometryEnd: u32) -> u32 {
    var geometryInside = false;
    var geometryOwned = false;
    var onBoundary = false;
    for (var polygonIndex = geometryStart; polygonIndex < geometryEnd; polygonIndex++) {
      let polygonStart = polygonOffsets[polygonOffsetsOffset + polygonIndex];
      let polygonEnd = polygonOffsets[polygonOffsetsOffset + polygonIndex + 1u];
      // Malformed offsets cannot be classified; an empty part contains nothing (as in shapely) and
      // must not hide the other parts of the feature.
      if (polygonStart > polygonEnd || polygonEnd > RING_COUNT) { return UNCERTAIN; }
      var polygonInside = false;
      var polygonOwned = false;
      for (var ringIndex = polygonStart; ringIndex < polygonEnd; ringIndex++) {
        let ringStart = ringOffsets[ringOffsetsOffset + ringIndex];
        let ringEnd = ringOffsets[ringOffsetsOffset + ringIndex + 1u];
        if (ringEnd < ringStart || ringEnd > VERTEX_COUNT) { return UNCERTAIN; }
        // A ring with fewer than 3 vertices bounds no area: skip it, as shapely treats empty parts.
        if (ringEnd < ringStart + 3u) { continue; }
        var previous = readVertex(ringEnd - 1u);
        for (var vertexIndex = ringStart; vertexIndex < ringEnd; vertexIndex++) {
          let current = readVertex(vertexIndex);
          let sign = orientSign(previous, current, point);
          if (sign == 2) { return UNCERTAIN; }
          var side = sign;
          if (sign == 0) {
            if (point.x >= min(previous.x, current.x) && point.x <= max(previous.x, current.x) &&
                point.y >= min(previous.y, current.y) && point.y <= max(previous.y, current.y)) {
              onBoundary = true;
            }
            side = select(select(0, -1, current.x < previous.x), 1, current.x > previous.x);
            if (side == 0) {
              side = select(select(0, 1, current.y < previous.y), -1, current.y > previous.y);
            }
          }
          if ((previous.y > point.y) != (current.y > point.y)) {
            if (current.y > previous.y && sign > 0) { polygonInside = !polygonInside; }
            if (current.y < previous.y && sign < 0) { polygonInside = !polygonInside; }
            if ((current.y > previous.y && side > 0) || (current.y < previous.y && side < 0)) {
              polygonOwned = !polygonOwned;
            }
          }
          previous = current;
        }
      }
      geometryInside = geometryInside || polygonInside;
      geometryOwned = geometryOwned || polygonOwned;
    }
    if (onBoundary) { return BOUNDARY | select(0u, HALF_OPEN_OWNER_BIT, geometryOwned); }
    return select(OUTSIDE, INSIDE, geometryInside);
  }`,
          body: `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
    if (index >= activeCount) { return; }
    let featureRow = candidatePairs[candidatePairsOffset + index * 2u + 1u];
    let geometryStart = featureOffsets[featureOffsetsOffset + featureRow];
    let geometryEnd = featureOffsets[featureOffsetsOffset + featureRow + 1u];
    var classification = UNCERTAIN;
    if (geometryStart <= geometryEnd && geometryEnd <= POLYGON_COUNT) {
      let point = vec2f(candidatePoints[candidatePointsOffset + index * 2u], candidatePoints[candidatePointsOffset + index * 2u + 1u]);
      classification = classifyExactly(point, geometryStart, geometryEnd);
    }
    pairClassifications[pairClassificationsOffset + index] = classification;`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-resolve`,
        operation: OPERATION,
        variant: 'resolve',
        bindings: [
          {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read'},
          {name: 'pairClassifications', view: pairClassifications, type: 'u32', access: 'read'},
          {name: 'assignment', view: assignment, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        condition: candidateDispatch.condition,
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;
const HALF_OPEN_OWNER_BIT: u32 = ${HALF_OPEN_OWNER_BIT}u;
const BOUNDARY_CLOSED: bool = ${this.boundaryRule === 'closed'};
const BOUNDARY_HALF_OPEN: bool = ${this.boundaryRule === 'half-open'};`,
        body: `let activeCount = min(atomicLoad(&state[stateOffset]), CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let pointRow = candidatePairs[candidatePairsOffset + index * ${pairStride}u];
  let featureRow = candidatePairs[candidatePairsOffset + index * ${pairStride}u + 1u];
  let word = pairClassifications[pairClassificationsOffset + index];
  let classification = word & ~HALF_OPEN_OWNER_BIT;
  let owned = BOUNDARY_CLOSED || (BOUNDARY_HALF_OPEN && (word & HALF_OPEN_OWNER_BIT) != 0u);
  if (classification == INSIDE || (owned && classification == BOUNDARY)) {
    atomicMin(&assignment[assignmentOffset + pointRow], featureRow);
  } else if (classification == UNCERTAIN) {
    atomicAdd(&state[stateOffset + 1u], 1u);
  }`
      })
    );
    nodes.push(
      ...getSpatialJoinAssignNodes<Parameters>(graph, {
        id: `${id}-assign`,
        operation: OPERATION,
        assignment,
        pointFeatureIds: output.pointFeatureIds,
        featureIds: polygons.sourceIds,
        featureCounts: output.featureCounts
      })
    );
    if (matches) {
      nodes.push(
        ...getSpatialJoinCollectNodes<Parameters>(graph, {
          id: `${id}-collect-matches`,
          operation: OPERATION,
          points,
          assignment,
          state,
          sourceIds: props.points.sourceIds,
          matches
        })
      );
    }
    nodes.push(
      createSpatialJoinFinalizeNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        state,
        bvhOverflow: bvh.overflow,
        candidateCapacity,
        overflow: status.overflow,
        candidateCount: status.candidateCount,
        uncertainCount: status.uncertainCount,
        matches
      })
    );
    return nodes;
  }

  /**
   * Slab-based exact refine: one thread per candidate classifies the point against the edges of
   * the y-slab of each polygon part of the feature. Exactly eight bindings: state, the packed
   * candidates, classifications, polygon positions, the three slab views and the edge layout.
   */
  private createSlabClassifyNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: {
      condition: GPUCommandNode<Parameters>['condition'];
      state: GraphDataView<'uint32'>;
      candidatePairs: GraphDataView;
      pairClassifications: GraphDataView<'uint32'>;
      slabs: PolygonEdgeSlabs;
      edgeLayout: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters> {
    const {id, candidateCapacity} = this;
    const {polygons} = this.props;
    const edgeLoop = getPolygonEdgeSlabLoopWGSL({
      owner: 'polygonIndex',
      lowY: 'point.y',
      highY: 'point.y',
      getVertex: 'readVertex',
      pointNames: ['previous', 'current'],
      // One call site for the exact helpers. The bit is set when the edge starts in a ring of at
      // least three vertices, the rings the unslabbed walk uses.
      body: `let isUsableRing = ((edgeLayout[edgeLayoutOffset + (edgeStart >> 5u)] >> (edgeStart & 31u)) & 1u) != 0u;
        if (!isUsableRing) { continue; }
        let sign = orientSign(previous, current, point);
        if (sign == 2) { return UNCERTAIN; }
        var side = sign;
        if (sign == 0) {
          if (point.x >= min(previous.x, current.x) && point.x <= max(previous.x, current.x) &&
              point.y >= min(previous.y, current.y) && point.y <= max(previous.y, current.y)) {
            onBoundary = true;
          }
          side = select(select(0, -1, current.x < previous.x), 1, current.x > previous.x);
          if (side == 0) {
            side = select(select(0, 1, current.y < previous.y), -1, current.y > previous.y);
          }
        }
        if ((previous.y > point.y) != (current.y > point.y)) {
          if (current.y > previous.y && sign > 0) { polygonInside = !polygonInside; }
          if (current.y < previous.y && sign < 0) { polygonInside = !polygonInside; }
          if ((current.y > previous.y && side > 0) || (current.y < previous.y && side < 0)) {
            polygonOwned = !polygonOwned;
          }
        }`
    });
    return createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-classify`,
      operation: OPERATION,
      variant: 'classify-exact-slabs',
      bindings: [
        {name: 'state', view: props.state, type: 'u32', access: 'read'},
        {name: 'candidatePairs', view: props.candidatePairs, type: 'u32', access: 'read'},
        {name: 'polygonPositions', view: polygons.positions, type: 'f32', access: 'read'},
        ...getPolygonEdgeSlabBindings(props.slabs),
        {name: 'edgeLayout', view: props.edgeLayout, type: 'u32', access: 'read'},
        {
          name: 'pairClassifications',
          view: props.pairClassifications,
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: candidateCapacity,
      condition: props.condition,
      declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const OUTSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.outside}u;
const INSIDE: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.inside}u;
const BOUNDARY: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary}u;
const UNCERTAIN: u32 = ${GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain}u;
const HALF_OPEN_OWNER_BIT: u32 = ${HALF_OPEN_OWNER_BIT}u;
const NO_PART: u32 = 0xffffffffu;
const FEATURE_BASE: u32 = ${Math.max(Math.ceil(polygons.positions.length / 32), 1)}u;
${EXACT_ORIENTATION_WGSL}
${getPolygonEdgeSlabQueryWGSL(props.slabs)}
fn readVertex(vertexIndex: u32) -> vec2f {
  return vec2f(
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u],
    polygonPositions[polygonPositionsOffset + vertexIndex * 2u + 1u]
  );
}
// Same classification as the unslabbed classifyExactly, over the edges of the slab at point.y of
// each part: see the edgeSlabs prop. Parity is per part (each part is a slab owner) and combined
// with OR across parts.
fn classifySlabs(point: vec2f, partStart: u32, partEnd: u32) -> u32 {
  var geometryInside = false;
  var geometryOwned = false;
  var onBoundary = false;
  for (var polygonIndex = partStart; polygonIndex < partEnd; polygonIndex++) {
    var polygonInside = false;
    var polygonOwned = false;
    ${edgeLoop}
    geometryInside = geometryInside || polygonInside;
    geometryOwned = geometryOwned || polygonOwned;
  }
  if (onBoundary) { return BOUNDARY | select(0u, HALF_OPEN_OWNER_BIT, geometryOwned); }
  return select(OUTSIDE, INSIDE, geometryInside);
}`,
      body: `let activeCount = min(state[stateOffset], CANDIDATE_CAPACITY);
  if (index >= activeCount) { return; }
  let candidate = candidatePairsOffset + index * 4u;
  let featureRow = candidatePairs[candidate + 1u];
  let point = vec2f(bitcast<f32>(candidatePairs[candidate + 2u]), bitcast<f32>(candidatePairs[candidate + 3u]));
  let feature = edgeLayoutOffset + FEATURE_BASE + featureRow * 2u;
  var classification = UNCERTAIN;
  let partStart = edgeLayout[feature];
  // A malformed feature (partStart == NO_PART) is UNCERTAIN, as in the unslabbed walk.
  if (partStart != NO_PART) {
    classification = classifySlabs(point, partStart, edgeLayout[feature + 1u]);
  }
  pairClassifications[pairClassificationsOffset + index] = classification;`
    });
  }
}

function getInputs(
  props: GPUPointInPolygonJoinProps
): (GraphDataView | GraphVectorView | undefined)[] {
  const {points, polygons} = props;
  return [
    points.positions,
    points.sourceIds,
    ...getSpatialJoinGeometryViews(polygons),
    polygons.sourceIds
  ];
}

function getOutputs(
  props: GPUPointInPolygonJoinProps
): (GraphDataView | GraphVectorView | undefined)[] {
  const {output} = props;
  const {matches, status} = output;
  return [
    output.pointFeatureIds,
    output.featureCounts,
    status.overflow,
    status.candidateCount,
    status.uncertainCount,
    status.antimeridianUncertainCount,
    status.poleUncertainCount,
    matches?.ids,
    matches?.status.count,
    matches?.status.overflow,
    matches?.status.requiredCount
  ];
}
