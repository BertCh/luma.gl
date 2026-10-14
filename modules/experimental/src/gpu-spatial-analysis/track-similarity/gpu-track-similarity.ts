// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  getDynamicTimeWarpingBody,
  getDynamicTimeWarpingDeclarations,
  getFrechetDecisionBody,
  getFrechetDecisionDeclarations,
  TRACK_SIMILARITY_NO_BAND
} from './track-similarity-dynamic-programs';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import type {GPULineGeometryPort, GPUPackedGeometryPort} from '../../analysis-core/contracts/index';

/** Spatial contexts supported by this distance-measuring operation. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

const OPERATION = 'GPUTrackSimilarity';
const HAUSDORFF_WORKGROUP_SIZE = 64;
const FRECHET_LANES = 256;

/**
 * Largest pair capacity (`pairA.length`) of {@link GPUTrackSimilarity}: the Frechet kernel runs
 * 256 lanes per pair in one 1-D dispatch, which must stay below `2^32` invocations, so
 * `floor((2^32 - 1) / 256) = 2^24 - 1`.
 */
export const GPU_TRACK_SIMILARITY_MAXIMUM_PAIR_COUNT = Math.floor(0xffffffff / FRECHET_LANES);
/** Workgroup memory holds one f32 per vertex of the shorter Frechet track (8 KiB at the maximum). */
const MAXIMUM_FRECHET_VERTEX_COUNT = 2048;
const MAXIMUM_DENSIFY_SUBDIVISIONS = 4096;
/** Largest `dynamicTimeWarpingBandRadius`, so band sums of densified indices stay in 32 bits. */
const MAXIMUM_BAND_RADIUS = 2 ** 30;

/**
 * WGSL: metric value (squared planar distance, or great-circle distances in the context's length
 * unit) from a point to the polyline of one track (`side` is `'A'` or `'B'`), with
 * Taha-Hanbury early break against `worst` and a rotating start (`hint`, updated to the nearest
 * segment found). Returns a value at most `worst` as soon as the point cannot raise the maximum.
 */
function getDirectedPolylineSource(side: 'A' | 'B'): string {
  return `// Metric value (squared distance, or the context's length unit on the sphere) from a point to polyline ${side}; may stop early once it is at most worst.
fn distanceToPolyline${side}(point: vec2<f32>, start: u32, count: u32, worst: f32, hint: ptr<function, u32>) -> f32 {
  if (count == 1u) {
    return pointMetric(point, load${side}(start));
  }
  let segments = count - 1u;
  var nearest = 3.402823466e+38;
  var first = *hint;
  if (first >= segments) { first = 0u; }
  for (var step = 0u; step < segments; step++) {
    var j = first + step;
    if (j >= segments) { j -= segments; }
    let squared = pointSegmentMetric(point, load${side}(start + j), load${side}(start + j + 1u));
    if (squared < nearest) {
      nearest = squared;
      *hint = j;
      if (nearest <= worst) { break; }
    }
  }
  return nearest;
}
// Cooperative minimum over the segments of polyline ${side}: every lane takes a stride of segments.
fn cooperativeDistanceToPolyline${side}(point: vec2<f32>, start: u32, count: u32, lane: u32) -> f32 {
  var nearest = 3.402823466e+38;
  if (count == 1u) {
    return pointMetric(point, load${side}(start));
  }
  for (var j = lane; j + 1u < count; j += WORKGROUP_SIZE) {
    nearest = min(nearest, pointSegmentMetric(point, load${side}(start + j), load${side}(start + j + 1u)));
  }
  return nearest;
}`;
}

/** WGSL: the directed pass of points of track `from` against the polyline of track `to`. */
function getDirectedPassSource(from: 'A' | 'B', to: 'A' | 'B'): string {
  return `{
    let pointCount = uniformDense${from};
    if (pointCount * 8u <= WORKGROUP_SIZE) {
      for (var i = 0u; i < pointCount; i++) {
        let point = densifiedPoint${from}(start${from}, count${from}, i);
        partials[lane] = cooperativeDistanceToPolyline${to}(point, start${to}, count${to}, lane);
        workgroupBarrier();
        for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
          if (lane < stride) { partials[lane] = min(partials[lane], partials[lane + stride]); }
          workgroupBarrier();
        }
        worst = max(worst, partials[0]);
        workgroupBarrier();
      }
    } else {
      let chunk = (pointCount + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
      let chunkStart = min(lane * chunk, pointCount);
      let chunkEnd = min(chunkStart + chunk, pointCount);
      var hint = 0u;
      for (var i = chunkStart; i < chunkEnd; i++) {
        let nearest = distanceToPolyline${to}(densifiedPoint${from}(start${from}, count${from}, i), start${to}, count${to}, worst, &hint);
        worst = max(worst, nearest);
      }
    }
  }`;
}

/** Bits of the `output.pairStatus` column of {@link GPUTrackSimilarity}. */
export const GPU_TRACK_SIMILARITY_STATUS = {
  /** At least one of the two tracks has no vertices. All distances are NaN. */
  emptyTrack: 1,
  /**
   * The shorter of the two (densified) tracks has more than `maximumFrechetVertexCount` vertices.
   * The Frechet distance and dynamic time warping are NaN and `frechetWithin` is 0. The longer
   * track is not limited.
   */
  frechetCapExceeded: 2,
  /** The pair index is out of range or at or beyond `activePairCount`. All distances are NaN. */
  invalidPair: 4
} as const;

/** Caller-owned per-pair outputs of {@link GPUTrackSimilarity}. At least one measure is required. */
export type GPUTrackSimilarityOutput = {
  /** Optional per-pair symmetric discrete Hausdorff distance in the context's length unit, NaN when not computed. */
  hausdorff?: GraphDataView<'float32'>;
  /** Optional per-pair discrete Frechet distance in the context's length unit, NaN when not computed. */
  frechet?: GraphDataView<'float32'>;
  /**
   * Optional per-pair Frechet decision: 1 when the discrete Frechet distance is at most
   * `frechetEpsilon`, else 0 (also 0 for skipped pairs and an unusable epsilon). Requires
   * `frechetEpsilon`. Cheaper than `frechet`: endpoint and coupling bounds decide most pairs in
   * `O(n + m)`, and the free-space program stops once no monotone path can survive. The test is
   * made in the internal metric (squared planar distance against `epsilon^2`, or the context's
   * length unit), so an epsilon within an f32 rounding of the distance may decide either way.
   */
  frechetWithin?: GraphDataView<'uint32'>;
  /**
   * Optional per-pair dynamic time warping cost: the smallest sum of point distances in the
   * context's length unit over monotone warping paths between the (densified) vertex
   * sequences, the convention of `similaritymeasures.dtw` (not the square root of summed squares
   * of `tslearn` and `dtaidistance`). Banded by `dynamicTimeWarpingBandRadius`; NaN when not
   * computed. Summed in f32, so the relative error grows with the path length (about
   * `(n + m) * 2^-24`).
   */
  dynamicTimeWarping?: GraphDataView<'float32'>;
  /**
   * Optional per-pair maximum distance between any vertex of one track and any vertex of the other
   * (original vertices, never densified), in the context's length unit, NaN when not computed.
   * Matches Sedona `ST_MaxDistance`.
   */
  maximumDistance?: GraphDataView<'float32'>;
  /** Optional per-pair {@link GPU_TRACK_SIMILARITY_STATUS} bit set, 0 when computed. */
  pairStatus?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUTrackSimilarity}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths, the
 * pair capacity (`pairA.length`), `maximumFrechetVertexCount`, and which outputs are present.
 */
export type GPUTrackSimilarityProps = {
  /** Prefix for generated node IDs. Defaults to `'track-similarity'`. */
  id?: string;
  /**
   * Coordinate space of both vertex sets. Native planar (the default): distances are Euclidean in
   * the f32 input coordinates. `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`:
   * vertices are `[longitude, latitude]` degrees and every distance (Hausdorff, Frechet and
   * `maximumDistance`) is a great-circle distance in the context's length unit on a sphere of
   * `sphereRadius` (default
   * mean Earth radius in the context's length unit). Segments are straight in longitude/latitude
   * (the edge model of
   * the rest of the package): the point-to-segment distance is the haversine distance to the
   * closest point of the segment, with the closest point found in the local equirectangular frame
   * of the point, which is exact to first order for segments of kilometers; densify longer
   * segments (see `densify`). Densified points interpolate longitude and latitude linearly with the
   * longitude difference wrapped to +-180, so tracks may cross the antimeridian. Other metrics
   * throw.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * First set of tracks (lines or polygon boundaries): packed planar `positions` sorted by track
   * and `lineOffsets` with `trackCountA + 1` monotonic vertex offsets.
   */
  linesA: GPUPackedGeometryPort<GPULineGeometryPort>;
  /** Second set of tracks. Defaults to the first set (similarity inside one set). */
  linesB?: GPUPackedGeometryPort<GPULineGeometryPort>;
  /**
   * Track index into the first set for each pair. The length is the pair capacity, at most
   * {@link GPU_TRACK_SIMILARITY_MAXIMUM_PAIR_COUNT}.
   */
  pairA: GraphDataView<'uint32'>;
  /** Track index into the second set for each pair, aligned with `pairA`. */
  pairB: GraphDataView<'uint32'>;
  /**
   * Optional one-row GPU-written pair count, for example `GPUTrajectoryEncounters`
   * `output.pairs.count`. Pairs at and after it get `invalidPair`.
   */
  activePairCount?: GraphDataView<'uint32'>;
  /**
   * Densify fraction with the meaning of Shapely `hausdorff_distance(a, b, densify)` and
   * `frechet_distance(a, b, densify)`: every segment is split into `round(1 / densify)` equal
   * parts and the extra points join the vertices (Hausdorff samples them, Frechet couples them).
   * 0 or undefined disables it. Compile-time, because it sets the subdivision count of the kernels
   * and the densified track lengths; must be 0 or in `[1 / 4096, 1]`. In a great-circle context
   * the extra points are linear in longitude/latitude (wrapped across the antimeridian).
   */
  densify?: number;
  /**
   * Cap on the (densified) vertex count of the shorter track in the Frechet dynamic program, from
   * 1 to 2048 (workgroup memory holds one row of it). Defaults to 256. The longer track is
   * unlimited: it is swept in strips of 256 lanes. Pairs whose shorter track exceeds the cap report
   * `frechetCapExceeded` and a NaN Frechet distance; resample or simplify them first.
   */
  maximumFrechetVertexCount?: number;
  /**
   * Per-frame Frechet decision threshold in the context's length unit, one float32 row.
   * Required with `output.frechetWithin`. NaN,
   * negative or infinite values decide 0.
   * On longitude/latitude this distance arrives as a GPU view, so no short-distance warning is possible;
   * below about 17 m use createGPULocalFrame (see Precision of longitude/latitude positions).
   */
  frechetEpsilon?: GraphDataView<'float32'>;
  /**
   * Sakoe-Chiba band radius of `output.dynamicTimeWarping`, in (densified) vertices; omitted means
   * unbanded. Compile-time. Same band as `tslearn.metrics.sakoe_chiba_mask(n, m, radius)`: for
   * tracks of `L >= S` vertices, long vertex `l` may warp to short vertex `s` when
   * `s - radius <= l <= s + (L - S) + radius`, so radius 0 is the widened diagonal and a radius
   * of at least `S` is unbanded. Cells outside the band are never evaluated.
   */
  dynamicTimeWarpingBandRadius?: number;
  /** Caller-owned per-pair output views. */
  output: GPUTrackSimilarityOutput;
};

/**
 * Computes Hausdorff, discrete Frechet and maximum vertex distance between pairs of lines or
 * polygon boundaries (rings given as closed vertex sequences).
 *
 * Semantics follow Shapely 2.1 (GEOS 3.13). Hausdorff is `max(directed(A, B), directed(B, A))`
 * where `directed(A, B)` is the largest distance from a (densified) vertex of A to the polyline B
 * (to its nearest point on any segment, not only to its vertices). The Frechet distance is the
 * minimum over monotone couplings of the largest coupled distance between (densified) vertices.
 * `maximumDistance` is the largest vertex-to-vertex distance (Sedona `ST_MaxDistance`). Polygon
 * boundaries are the open ring sequences with the repeated closing vertex, so Frechet is
 * orientation and start sensitive. Distances are planar in the f32 input coordinates by default,
 * or great-circle distances in the context's length unit with a `{coordinateSpace:
 * 'longitude-latitude', metric: 'great-circle'}`
 * `spatialContext` (see {@link GPUTrackSimilarityProps.spatialContext}); the planar kernels contain
 * no geodesic code.
 *
 * Composition: one status kernel, then one workgroup per pair per requested measure. Hausdorff and
 * `maximumDistance` use 64 lanes that stride over vertices and reduce with a workgroup max tree
 * (`O(n * m)` per pair, no cap). Frechet runs the dynamic program on anti-diagonals with one lane
 * per vertex of the longer track, in strips of 256 lanes whose last row is carried to the next
 * strip in workgroup memory; only the shorter track is capped (`maximumFrechetVertexCount`, flagged in
 * `output.pairStatus`). Densified vertices are computed on the fly, nothing is materialised.
 *
 * The same anti-diagonal layout computes two more measures. `frechetWithin` is the Frechet
 * decision against a per-frame `frechetEpsilon`: endpoint distances (a lower bound) and one fixed
 * monotone coupling (an upper bound) decide most pairs in `O(n + m)`, and the remaining pairs run
 * the boolean free-space program, which stops as soon as no monotone path can survive.
 * `dynamicTimeWarping` is the dynamic time warping cost (`similaritymeasures.dtw`: sum of point
 * distances), optionally inside a Sakoe-Chiba band (`dynamicTimeWarpingBandRadius`, tslearn's
 * mask), whose outside cells are never evaluated. Both share the shorter-track cap.
 *
 * Pairs are the caller's choice, for example `GPUTrajectoryEncounters` output, cluster candidates,
 * or two vintages of one boundary. This contributor does not generate pairs.
 */
export class GPUTrackSimilarity implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTrackSimilarityProps;
  /** Resolved coordinate context: native planar, or longitude-latitude with the great-circle metric. */
  readonly spatialContext: GPUSpatialContext;
  /** Sphere radius in the context's length unit of a great-circle context, `undefined` for planar. */
  readonly sphereRadius: number | undefined;
  /** Resolved Frechet vertex cap. */
  readonly maximumFrechetVertexCount: number;
  /** Resolved subdivisions per segment, `round(1 / densify)`, 1 without densify. */
  readonly densifySubdivisions: number;

  constructor(props: GPUTrackSimilarityProps) {
    this.id = props.id ?? 'track-similarity';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.sphereRadius = getGPUGreatCircleRadius(this.spatialContext);
    const {linesA, linesB, output} = props;
    validatePackedView(linesA.positions, ['float32x2'], `${id} linesA.positions`);
    validatePackedView(linesA.lineOffsets, ['uint32'], `${id} linesA.lineOffsets`);
    if (linesA.lineOffsets.length < 2) {
      throw new Error(`${id} linesA.lineOffsets needs two rows`);
    }
    if (linesB) {
      validatePackedView(linesB.positions, ['float32x2'], `${id} linesB.positions`);
      validatePackedView(linesB.lineOffsets, ['uint32'], `${id} linesB.lineOffsets`);
      if (linesB.lineOffsets.length < 2) {
        throw new Error(`${id} linesB.lineOffsets needs two rows`);
      }
    }
    validatePackedView(props.pairA, ['uint32'], `${id} pairA`);
    validatePackedView(props.pairB, ['uint32'], `${id} pairB`);
    if (props.pairA.length < 1 || props.pairB.length !== props.pairA.length) {
      throw new Error(`${id} pairA and pairB must have the same positive length`);
    }
    validateCapacity(
      id,
      'pairA.length',
      props.pairA.length,
      1,
      GPU_TRACK_SIMILARITY_MAXIMUM_PAIR_COUNT
    );
    if (props.activePairCount) {
      validatePackedView(props.activePairCount, ['uint32'], `${id} activePairCount`);
      if (props.activePairCount.length < 1) {
        throw new Error(`${id} activePairCount must contain one uint32 row`);
      }
    }
    if (
      !output.hausdorff &&
      !output.frechet &&
      !output.maximumDistance &&
      !output.frechetWithin &&
      !output.dynamicTimeWarping
    ) {
      throw new Error(
        `${id} needs hausdorff, frechet, frechetWithin, dynamicTimeWarping or maximumDistance output`
      );
    }
    if (Boolean(output.frechetWithin) !== Boolean(props.frechetEpsilon)) {
      throw new Error(`${id} output.frechetWithin and frechetEpsilon go together`);
    }
    if (props.frechetEpsilon) {
      validatePackedView(props.frechetEpsilon, ['float32'], `${id} frechetEpsilon`);
      if (props.frechetEpsilon.length < 1) {
        throw new Error(`${id} frechetEpsilon must contain one float32 row`);
      }
    }
    if (props.dynamicTimeWarpingBandRadius !== undefined) {
      validateCapacity(
        id,
        'dynamicTimeWarpingBandRadius',
        props.dynamicTimeWarpingBandRadius,
        0,
        MAXIMUM_BAND_RADIUS
      );
    }
    for (const [name, view, format] of [
      ['hausdorff', output.hausdorff, 'float32'],
      ['frechet', output.frechet, 'float32'],
      ['frechetWithin', output.frechetWithin, 'uint32'],
      ['dynamicTimeWarping', output.dynamicTimeWarping, 'float32'],
      ['maximumDistance', output.maximumDistance, 'float32'],
      ['pairStatus', output.pairStatus, 'uint32']
    ] as const) {
      if (view) {
        validatePackedView(view, [format], `${id} ${name}`);
        if (view.length !== props.pairA.length) {
          throw new Error(`${id} ${name} length must equal the pair capacity`);
        }
      }
    }
    const densify = props.densify ?? 0;
    if (!(densify >= 0 && densify <= 1)) {
      throw new Error(`${id} densify must be a fraction from 0 to 1`);
    }
    this.densifySubdivisions = densify === 0 ? 1 : Math.round(1 / densify);
    if (this.densifySubdivisions > MAXIMUM_DENSIFY_SUBDIVISIONS) {
      throw new Error(`${id} densify must be at least 1 / ${MAXIMUM_DENSIFY_SUBDIVISIONS}`);
    }
    this.maximumFrechetVertexCount = props.maximumFrechetVertexCount ?? 256;
    validateCapacity(
      id,
      'maximumFrechetVertexCount',
      this.maximumFrechetVertexCount,
      1,
      MAXIMUM_FRECHET_VERTEX_COUNT
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.hausdorff,
        output.frechet,
        output.frechetWithin,
        output.dynamicTimeWarping,
        output.maximumDistance,
        output.pairStatus
      ],
      [
        linesA.positions,
        linesA.lineOffsets,
        linesB?.positions,
        linesB?.lineOffsets,
        props.pairA,
        props.pairB,
        props.activePairCount,
        props.frechetEpsilon
      ]
    );
  }

  /** Returns the status kernel and the requested distance kernels. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {linesA, linesB, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      linesA.positions,
      linesA.lineOffsets,
      linesB?.positions,
      linesB?.lineOffsets,
      props.pairA,
      props.pairB,
      props.activePairCount,
      props.frechetEpsilon,
      output.hausdorff,
      output.frechet,
      output.frechetWithin,
      output.dynamicTimeWarping,
      output.maximumDistance,
      output.pairStatus
    ]);
    const positionsB = linesB?.positions ?? linesA.positions;
    const offsetsB = linesB?.lineOffsets ?? linesA.lineOffsets;
    const pairCount = props.pairA.length;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const constants = `const PAIR_COUNT: u32 = ${pairCount}u;
const TRACK_COUNT_A: u32 = ${linesA.lineOffsets.length - 1}u;
const TRACK_COUNT_B: u32 = ${offsetsB.length - 1}u;
const FRECHET_CAP: u32 = ${this.maximumFrechetVertexCount}u;
const DENSIFY: u32 = ${this.densifySubdivisions}u;
const STATUS_EMPTY: u32 = ${GPU_TRACK_SIMILARITY_STATUS.emptyTrack}u;
const STATUS_TOO_LONG: u32 = ${GPU_TRACK_SIMILARITY_STATUS.frechetCapExceeded}u;
const STATUS_INVALID: u32 = ${GPU_TRACK_SIMILARITY_STATUS.invalidPair}u;`;

    // The status buffer is needed by the distance kernels even when the caller does not want it.
    const status =
      output.pairStatus ?? createTransientView(graph, `${id}-status`, 'uint32', pairCount);
    const statusBindings: WGSLKernelBinding[] = [
      {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
      {name: 'pairB', view: props.pairB, type: 'u32', access: 'read'},
      {name: 'offsetsA', view: linesA.lineOffsets, type: 'u32', access: 'read'},
      {name: 'offsetsB', view: offsetsB, type: 'u32', access: 'read'}
    ];
    if (props.activePairCount) {
      statusBindings.push({
        name: 'activePairCount',
        view: props.activePairCount,
        type: 'u32',
        access: 'read'
      });
    }
    statusBindings.push({name: 'status', view: status, type: 'u32', access: 'read_write'});
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-status`,
        operation: OPERATION,
        variant: 'status',
        bindings: statusBindings,
        invocationCount: pairCount,
        declarations: constants,
        body: `let a = pairA[pairAOffset + index];
  let b = pairB[pairBOffset + index];
  var flags = 0u;
  ${props.activePairCount ? 'if (index >= activePairCount[activePairCountOffset]) { flags = STATUS_INVALID; }' : ''}
  if (a >= TRACK_COUNT_A || b >= TRACK_COUNT_B) {
    flags = STATUS_INVALID;
  } else if (flags == 0u) {
    let countA = offsetsA[offsetsAOffset + a + 1u] - offsetsA[offsetsAOffset + a];
    let countB = offsetsB[offsetsBOffset + b + 1u] - offsetsB[offsetsBOffset + b];
    if (countA == 0u || countB == 0u) { flags = STATUS_EMPTY; }
    else if (min((countA - 1u) * DENSIFY, (countB - 1u) * DENSIFY) + 1u > FRECHET_CAP) { flags = STATUS_TOO_LONG; }
  }
  status[statusOffset + index] = flags;`
      })
    );

    const distanceBindings = (output: GraphDataView<'float32'>): WGSLKernelBinding[] => [
      {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
      {name: 'pairB', view: props.pairB, type: 'u32', access: 'read'},
      {name: 'positionsA', view: linesA.positions, type: 'f32', access: 'read'},
      {name: 'offsetsA', view: linesA.lineOffsets, type: 'u32', access: 'read'},
      {name: 'positionsB', view: positionsB, type: 'f32', access: 'read'},
      {name: 'offsetsB', view: offsetsB, type: 'u32', access: 'read'},
      {name: 'status', view: status, type: 'u32', access: 'read'},
      {name: 'distances', view: output, type: 'f32', access: 'read_write'}
    ];
    // Shared prologue: skipped pairs get empty tracks so every lane still reaches the barriers.
    const prologue = (skipMask: string) => `let pair = index / WORKGROUP_SIZE;
  let lane = localInvocationIndex;
  let flags = status[statusOffset + min(pair, PAIR_COUNT - 1u)];
  let skipped = pair >= PAIR_COUNT || (flags & ${skipMask}) != 0u;
  var startA = 0u;
  var countA = 0u;
  var startB = 0u;
  var countB = 0u;
  if (!skipped) {
    let a = pairA[pairAOffset + pair];
    let b = pairB[pairBOffset + pair];
    startA = offsetsA[offsetsAOffset + a];
    countA = offsetsA[offsetsAOffset + a + 1u] - startA;
    startB = offsetsB[offsetsBOffset + b];
    countB = offsetsB[offsetsBOffset + b + 1u] - startB;
  }
  // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
  let notANumber = bitcast<f32>(0x7fc00000u | (index & 0u));`;
    // Vertex loaders and the virtual (densified) vertex of a track. Virtual vertex `k` of a track
    // with `count` vertices lies on segment `k / DENSIFY` at fraction `(k % DENSIFY) / DENSIFY`,
    // the sampling GEOS uses for its densify fraction.
    const trackHelpers = `fn loadA(row: u32) -> vec2<f32> {
  return vec2<f32>(positionsA[positionsAOffset + row * 2u], positionsA[positionsAOffset + row * 2u + 1u]);
}
fn loadB(row: u32) -> vec2<f32> {
  return vec2<f32>(positionsB[positionsBOffset + row * 2u], positionsB[positionsBOffset + row * 2u + 1u]);
}
fn densifiedCount(count: u32) -> u32 {
  return select(0u, (count - 1u) * DENSIFY + 1u, count > 0u);
}
fn densifiedPointA(start: u32, count: u32, k: u32) -> vec2<f32> {
  let segment = k / DENSIFY;
  let subStep = k - segment * DENSIFY;
  if (subStep == 0u || segment + 1u >= count) { return loadA(start + min(segment, count - 1u)); }
  return interpolatePoint(loadA(start + segment), loadA(start + segment + 1u), f32(subStep) / f32(DENSIFY));
}
fn densifiedPointB(start: u32, count: u32, k: u32) -> vec2<f32> {
  let segment = k / DENSIFY;
  let subStep = k - segment * DENSIFY;
  if (subStep == 0u || segment + 1u >= count) { return loadB(start + min(segment, count - 1u)); }
  return interpolatePoint(loadB(start + segment), loadB(start + segment + 1u), f32(subStep) / f32(DENSIFY));
}
${
  this.sphereRadius === undefined
    ? `fn pointMetric(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let offset = a - b;
  return dot(offset, offset);
}
fn finishMetric(value: f32) -> f32 {
  return sqrt(value);
}
fn interpolatePoint(from0: vec2<f32>, to0: vec2<f32>, fraction: f32) -> vec2<f32> {
  return from0 + (to0 - from0) * fraction;
}
fn pointSegmentMetric(point: vec2<f32>, from0: vec2<f32>, to0: vec2<f32>) -> f32 {
  let edge = to0 - from0;
  let lengthSquared = dot(edge, edge);
  var t = 0.0;
  if (lengthSquared > 0.0) { t = clamp(dot(point - from0, edge) / lengthSquared, 0.0, 1.0); }
  let offset = point - (from0 + edge * t);
  return dot(offset, offset);
}`
    : `${GEODESIC_WGSL}
const SPHERE_RADIUS: f32 = ${this.sphereRadius};
fn pointMetric(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return geodesicHaversineDistance(a, b, SPHERE_RADIUS);
}
fn finishMetric(value: f32) -> f32 {
  return value;
}
// Linear in longitude/latitude with the longitude difference wrapped across the antimeridian.
fn interpolatePoint(from0: vec2<f32>, to0: vec2<f32>, fraction: f32) -> vec2<f32> {
  return from0 + vec2<f32>(geodesicWrapLongitudeDelta(to0.x - from0.x), to0.y - from0.y) * fraction;
}
fn pointSegmentMetric(point: vec2<f32>, from0: vec2<f32>, to0: vec2<f32>) -> f32 {
  let t = geodesicSegmentFraction(point, from0, to0);
  return geodesicSegmentDistance(point, from0, to0, t, SPHERE_RADIUS);
}`
}`;

    if (output.hausdorff) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-hausdorff`,
          operation: OPERATION,
          variant: 'hausdorff',
          bindings: distanceBindings(output.hausdorff),
          workgroupSize: HAUSDORFF_WORKGROUP_SIZE,
          invocationCount: pairCount * HAUSDORFF_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${constants}
const WORKGROUP_SIZE: u32 = ${HAUSDORFF_WORKGROUP_SIZE}u;
var<workgroup> partials: array<f32, ${HAUSDORFF_WORKGROUP_SIZE}>;
var<workgroup> sharedDenseA: u32;
var<workgroup> sharedDenseB: u32;
${trackHelpers}
${getDirectedPolylineSource('A')}
${getDirectedPolylineSource('B')}`,
          // Directed distances use Taha and Hanbury's early break (a point whose running nearest
          // distance drops to the lane's current maximum cannot raise it, so its scan stops) with
          // the scan starting at the previous point's nearest segment, which is where the next
          // point of a track usually finds its nearest. The maximum is unchanged: the break only
          // skips points that cannot be the maximum. Lanes own contiguous runs of points so the
          // hint stays coherent. A side with fewer than 8 lanes' worth of points would idle most
          // lanes while each scans the whole other track, so there the lanes split the segments of
          // the other track instead and reduce the nearest distance per point.
          body: `${prologue('(STATUS_EMPTY | STATUS_INVALID)')}
  var worst = 0.0;
  let denseA = densifiedCount(countA);
  let denseB = densifiedCount(countB);
  if (lane == 0u) {
    sharedDenseA = denseA;
    sharedDenseB = denseB;
  }
  workgroupBarrier();
  let uniformDenseA = workgroupUniformLoad(&sharedDenseA);
  let uniformDenseB = workgroupUniformLoad(&sharedDenseB);
  ${getDirectedPassSource('A', 'B')}
  ${getDirectedPassSource('B', 'A')}
  partials[lane] = worst;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = max(partials[lane], partials[lane + stride]); }
    workgroupBarrier();
  }
  if (lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = select(finishMetric(partials[0]), notANumber, skipped);
  }`
        })
      );
    }

    if (output.maximumDistance) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-max-distance`,
          operation: OPERATION,
          variant: 'max-distance',
          bindings: distanceBindings(output.maximumDistance),
          workgroupSize: HAUSDORFF_WORKGROUP_SIZE,
          invocationCount: pairCount * HAUSDORFF_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${constants}
const WORKGROUP_SIZE: u32 = ${HAUSDORFF_WORKGROUP_SIZE}u;
var<workgroup> partials: array<f32, ${HAUSDORFF_WORKGROUP_SIZE}>;
${trackHelpers}`,
          body: `${prologue('(STATUS_EMPTY | STATUS_INVALID)')}
  var farthest = 0.0;
  // Lanes stride over the longer track and scan the shorter one, so a short track next to a long
  // one does not leave most lanes idle. The maximum is order independent.
  if (countA >= countB) {
    for (var i = lane; i < countA; i += WORKGROUP_SIZE) {
      let point = loadA(startA + i);
      for (var j = 0u; j < countB; j++) {
        farthest = max(farthest, pointMetric(point, loadB(startB + j)));
      }
    }
  } else {
    for (var j = lane; j < countB; j += WORKGROUP_SIZE) {
      let point = loadB(startB + j);
      for (var i = 0u; i < countA; i++) {
        farthest = max(farthest, pointMetric(point, loadA(startA + i)));
      }
    }
  }
  partials[lane] = farthest;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = max(partials[lane], partials[lane + stride]); }
    workgroupBarrier();
  }
  if (lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = select(finishMetric(partials[0]), notANumber, skipped);
  }`
        })
      );
    }

    if (output.frechet) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-frechet`,
          operation: OPERATION,
          variant: 'frechet',
          bindings: distanceBindings(output.frechet),
          workgroupSize: FRECHET_LANES,
          invocationCount: pairCount * FRECHET_LANES,
          guardIndex: false,
          declarations: `${constants}
const WORKGROUP_SIZE: u32 = ${FRECHET_LANES}u;
const LANES: u32 = ${FRECHET_LANES}u;
var<workgroup> olderDiagonal: array<f32, ${FRECHET_LANES}>;
var<workgroup> previousDiagonal: array<f32, ${FRECHET_LANES}>;
var<workgroup> boundaryRow: array<f32, ${this.maximumFrechetVertexCount}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
${trackHelpers}
// Metric value between densified vertex rowLong of the longer track and rowShort of the other.
fn coupledSquared(swapped: bool, startA: u32, countA: u32, startB: u32, countB: u32, rowLong: u32, rowShort: u32) -> f32 {
  if (swapped) {
    return pointMetric(densifiedPointB(startB, countB, rowLong), densifiedPointA(startA, countA, rowShort));
  }
  return pointMetric(densifiedPointA(startA, countA, rowLong), densifiedPointB(startB, countB, rowShort));
}`,
          // The longer (densified) track runs along the lanes in strips of LANES rows; the dynamic
          // program sweeps each strip by anti-diagonals and hands its last row to the next strip
          // through workgroup memory, so only the shorter track is capped.
          body: `${prologue('(STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)')}
  let infinity = 3.402823466e+38;
  let denseA = densifiedCount(countA);
  let denseB = densifiedCount(countB);
  let swapped = denseB > denseA;
  if (lane == 0u) {
    sharedLongCount = max(denseA, denseB);
    sharedShortCount = min(denseA, denseB);
  }
  // Barrier-bearing loops need bounds the compiler can prove uniform across the workgroup.
  let longCount = workgroupUniformLoad(&sharedLongCount);
  let shortCount = workgroupUniformLoad(&sharedShortCount);
  var finalValue = infinity;
  let stripCount = (longCount + LANES - 1u) / LANES;
  for (var strip = 0u; strip < stripCount; strip++) {
    let firstRow = strip * LANES;
    let rows = min(LANES, longCount - firstRow);
    let diagonalCount = rows + shortCount - 1u;
    olderDiagonal[lane] = infinity;
    previousDiagonal[lane] = infinity;
    workgroupBarrier();
    for (var diagonal = 0u; diagonal < diagonalCount; diagonal++) {
      var value = infinity;
      var column = 0u;
      let isActive = lane < rows && diagonal >= lane && diagonal - lane < shortCount;
      if (isActive) {
        column = diagonal - lane;
        let cost = coupledSquared(swapped, startA, countA, startB, countB, firstRow + lane, column);
        let hasUp = lane > 0u || strip > 0u;
        let hasLeft = column > 0u;
        var up = infinity;
        var upLeft = infinity;
        var left = infinity;
        if (lane > 0u) {
          up = previousDiagonal[lane - 1u];
          upLeft = olderDiagonal[lane - 1u];
        } else if (strip > 0u) {
          up = boundaryRow[column];
          if (hasLeft) { upLeft = boundaryRow[column - 1u]; }
        }
        if (hasLeft) { left = previousDiagonal[lane]; }
        if (!hasUp && !hasLeft) {
          value = cost;
        } else if (!hasUp) {
          value = max(left, cost);
        } else if (!hasLeft) {
          value = max(up, cost);
        } else {
          value = max(min(min(up, upLeft), left), cost);
        }
        if (firstRow + lane == longCount - 1u && column == shortCount - 1u) { finalValue = value; }
      }
      workgroupBarrier();
      olderDiagonal[lane] = previousDiagonal[lane];
      previousDiagonal[lane] = value;
      if (isActive && lane == LANES - 1u) { boundaryRow[column] = value; }
      workgroupBarrier();
    }
  }
  if (!skipped && longCount > 0u && lane == (longCount - 1u) % LANES) {
    distances[distancesOffset + pair] = finishMetric(finalValue);
  } else if (skipped && lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = notANumber;
  }`
        })
      );
    }
    if (output.dynamicTimeWarping) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-dynamic-time-warping`,
          operation: OPERATION,
          variant: 'dynamic-time-warping',
          bindings: distanceBindings(output.dynamicTimeWarping),
          workgroupSize: FRECHET_LANES,
          invocationCount: pairCount * FRECHET_LANES,
          guardIndex: false,
          declarations: `${constants}
${getDynamicTimeWarpingDeclarations(
  FRECHET_LANES,
  this.maximumFrechetVertexCount,
  props.dynamicTimeWarpingBandRadius ?? TRACK_SIMILARITY_NO_BAND
)}
${trackHelpers}`,
          body: getDynamicTimeWarpingBody(
            prologue('(STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)')
          )
        })
      );
    }

    if (output.frechetWithin && props.frechetEpsilon) {
      // The decision kernel binds the epsilon too, so the pair ranges are resolved first into one
      // view to stay within eight storage bindings.
      const pairRanges = createTransientView(graph, `${id}-pair-ranges`, 'uint32', pairCount * 4);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-pair-ranges`,
          operation: OPERATION,
          variant: 'pair-ranges',
          bindings: [
            {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
            {name: 'pairB', view: props.pairB, type: 'u32', access: 'read'},
            {name: 'offsetsA', view: linesA.lineOffsets, type: 'u32', access: 'read'},
            {name: 'offsetsB', view: offsetsB, type: 'u32', access: 'read'},
            {name: 'status', view: status, type: 'u32', access: 'read'},
            {name: 'pairRanges', view: pairRanges, type: 'u32', access: 'read_write'}
          ],
          invocationCount: pairCount,
          declarations: constants,
          body: `var range = vec4<u32>(0u);
  if (status[statusOffset + index] == 0u) {
    let a = pairA[pairAOffset + index];
    let b = pairB[pairBOffset + index];
    let startA = offsetsA[offsetsAOffset + a];
    let startB = offsetsB[offsetsBOffset + b];
    range = vec4<u32>(startA, offsetsA[offsetsAOffset + a + 1u] - startA,
      startB, offsetsB[offsetsBOffset + b + 1u] - startB);
  }
  pairRanges[pairRangesOffset + index * 4u] = range.x;
  pairRanges[pairRangesOffset + index * 4u + 1u] = range.y;
  pairRanges[pairRangesOffset + index * 4u + 2u] = range.z;
  pairRanges[pairRangesOffset + index * 4u + 3u] = range.w;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-frechet-decision`,
          operation: OPERATION,
          variant: 'frechet-decision',
          bindings: [
            {name: 'pairRanges', view: pairRanges, type: 'u32', access: 'read'},
            {name: 'positionsA', view: linesA.positions, type: 'f32', access: 'read'},
            {name: 'positionsB', view: positionsB, type: 'f32', access: 'read'},
            {name: 'status', view: status, type: 'u32', access: 'read'},
            {name: 'frechetEpsilon', view: props.frechetEpsilon, type: 'f32', access: 'read'},
            {name: 'frechetWithin', view: output.frechetWithin, type: 'u32', access: 'read_write'}
          ],
          workgroupSize: FRECHET_LANES,
          invocationCount: pairCount * FRECHET_LANES,
          guardIndex: false,
          declarations: `${constants}
${getFrechetDecisionDeclarations(
  FRECHET_LANES,
  this.maximumFrechetVertexCount,
  this.sphereRadius !== undefined
)}
${trackHelpers}`,
          body: getFrechetDecisionBody(`let pair = index / WORKGROUP_SIZE;
  let lane = localInvocationIndex;
  let flags = status[statusOffset + min(pair, PAIR_COUNT - 1u)];
  let skipped = pair >= PAIR_COUNT ||
    (flags & (STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)) != 0u;
  let rangeBase = pairRangesOffset + min(pair, PAIR_COUNT - 1u) * 4u;
  let startA = pairRanges[rangeBase];
  let countA = select(pairRanges[rangeBase + 1u], 0u, skipped);
  let startB = pairRanges[rangeBase + 2u];
  let countB = select(pairRanges[rangeBase + 3u], 0u, skipped);`)
        })
      );
    }

    return nodes;
  }
}
