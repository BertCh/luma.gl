// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

const OPERATION = 'GPUTrackSimilarity';
const HAUSDORFF_WORKGROUP_SIZE = 64;
const FRECHET_LANES = 256;
/** Workgroup memory holds one f32 per vertex of the shorter Frechet track (8 KiB at the maximum). */
const MAXIMUM_FRECHET_VERTICES = 2048;
const MAXIMUM_DENSIFY_SUBDIVISIONS = 4096;

/**
 * WGSL: squared distance from a point to the polyline of one track (`side` is `'A'` or `'B'`), with
 * Taha-Hanbury early break against `worst` and a rotating start (`hint`, updated to the nearest
 * segment found). Returns a value at most `worst` as soon as the point cannot raise the maximum.
 */
function getDirectedPolylineSource(side: 'A' | 'B'): string {
  return `// Squared distance from a point to polyline ${side}; may stop early once it is at most worst.
fn distanceToPolyline${side}(point: vec2<f32>, start: u32, count: u32, worst: f32, hint: ptr<function, u32>) -> f32 {
  if (count == 1u) {
    let offset = point - load${side}(start);
    return dot(offset, offset);
  }
  let segments = count - 1u;
  var nearest = 3.402823466e+38;
  var first = *hint;
  if (first >= segments) { first = 0u; }
  for (var step = 0u; step < segments; step++) {
    var j = first + step;
    if (j >= segments) { j -= segments; }
    let squared = pointSegmentSquared(point, load${side}(start + j), load${side}(start + j + 1u));
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
    let offset = point - load${side}(start);
    return dot(offset, offset);
  }
  for (var j = lane; j + 1u < count; j += WORKGROUP_SIZE) {
    nearest = min(nearest, pointSegmentSquared(point, load${side}(start + j), load${side}(start + j + 1u)));
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

/** Bits of the `status` column of {@link GPUTrackSimilarity}. */
export const GPU_TRACK_SIMILARITY_STATUS = {
  /** At least one of the two tracks has no vertices. All distances are NaN. */
  emptyTrack: 1,
  /**
   * The shorter of the two (densified) tracks has more than `maxFrechetVertices` vertices. The
   * Frechet distance is NaN. The longer track is not limited.
   */
  frechetCapExceeded: 2,
  /** The pair index is out of range or at or beyond `activePairCount`. All distances are NaN. */
  invalidPair: 4
} as const;

/**
 * Properties for {@link GPUTrackSimilarity}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths, the
 * pair capacity (`pairA.length`), `maxFrechetVertices`, and which outputs are present.
 */
export type GPUTrackSimilarityProps = {
  /** Prefix for generated node IDs. Defaults to `'track-similarity'`. */
  id?: string;
  /** Packed planar vertices of the first track set, sorted by track. */
  positionsA: GraphDataView<'float32x2'>;
  /** `trackCountA + 1` monotonic vertex offsets of the first set. */
  offsetsA: GraphDataView<'uint32'>;
  /** Vertices of the second set. Defaults to the first set (similarity inside one set). */
  positionsB?: GraphDataView<'float32x2'>;
  /** Offsets of the second set. Required with `positionsB`. */
  offsetsB?: GraphDataView<'uint32'>;
  /** Track index into the first set for each pair. */
  pairA: GraphDataView<'uint32'>;
  /** Track index into the second set for each pair, aligned with `pairA`. */
  pairB: GraphDataView<'uint32'>;
  /**
   * Optional one-row GPU-written pair count, for example `GPUTrajectoryEncounters`
   * `pairs.output.count`. Pairs at and after it get `invalidPair`.
   */
  activePairCount?: GraphDataView<'uint32'>;
  /** Optional per-pair symmetric discrete Hausdorff distance, NaN when not computed. */
  hausdorff?: GraphDataView<'float32'>;
  /** Optional per-pair discrete Frechet distance, NaN when not computed. */
  frechet?: GraphDataView<'float32'>;
  /**
   * Optional per-pair maximum distance between any vertex of one track and any vertex of the other
   * (original vertices, never densified), NaN when not computed. Matches Sedona `ST_MaxDistance`.
   */
  maxDistance?: GraphDataView<'float32'>;
  /** Optional per-pair {@link GPU_TRACK_SIMILARITY_STATUS} bit set, 0 when computed. */
  status?: GraphDataView<'uint32'>;
  /**
   * Densify fraction with the meaning of Shapely `hausdorff_distance(a, b, densify)` and
   * `frechet_distance(a, b, densify)`: every segment is split into `round(1 / densify)` equal
   * parts and the extra points join the vertices (Hausdorff samples them, Frechet couples them).
   * 0 or undefined disables it. Compile-time, because it sets the subdivision count of the kernels
   * and the densified track lengths; must be 0 or in `[1 / 4096, 1]`.
   */
  densify?: number;
  /**
   * Cap on the (densified) vertex count of the shorter track in the Frechet dynamic program, from
   * 1 to 2048 (workgroup memory holds one row of it). Defaults to 256. The longer track is
   * unlimited: it is swept in strips of 256 lanes. Pairs whose shorter track exceeds the cap report
   * `frechetCapExceeded` and a NaN Frechet distance; resample or simplify them first.
   */
  maxFrechetVertices?: number;
};

/**
 * Computes Hausdorff, discrete Frechet and maximum vertex distance between pairs of lines or
 * polygon boundaries (rings given as closed vertex sequences).
 *
 * Semantics follow Shapely 2.1 (GEOS 3.13). Hausdorff is `max(directed(A, B), directed(B, A))`
 * where `directed(A, B)` is the largest distance from a (densified) vertex of A to the polyline B
 * (to its nearest point on any segment, not only to its vertices). The Frechet distance is the
 * minimum over monotone couplings of the largest coupled distance between (densified) vertices.
 * `maxDistance` is the largest vertex-to-vertex distance (Sedona `ST_MaxDistance`). Polygon
 * boundaries are the open ring sequences with the repeated closing vertex, so Frechet is
 * orientation and start sensitive. Planar distances only, computed in the f32 input coordinates.
 *
 * Composition: one status kernel, then one workgroup per pair per requested measure. Hausdorff and
 * `maxDistance` use 64 lanes that stride over vertices and reduce with a workgroup max tree
 * (`O(n * m)` per pair, no cap). Frechet runs the dynamic program on anti-diagonals with one lane
 * per vertex of the longer track, in strips of 256 lanes whose last row is carried to the next
 * strip in workgroup memory; only the shorter track is capped (`maxFrechetVertices`, flagged in
 * `status`). Densified vertices are computed on the fly, nothing is materialised.
 *
 * Pairs are the caller's choice, for example `GPUTrajectoryEncounters` output, cluster candidates,
 * or two vintages of one boundary. This contributor does not generate pairs.
 */
export class GPUTrackSimilarity implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTrackSimilarityProps;
  /** Resolved Frechet vertex cap. */
  readonly maxFrechetVertices: number;
  /** Resolved subdivisions per segment, `round(1 / densify)`, 1 without densify. */
  readonly densifySubdivisions: number;

  constructor(props: GPUTrackSimilarityProps) {
    this.id = props.id ?? 'track-similarity';
    this.props = props;
    const {id} = this;
    validatePackedView(props.positionsA, ['float32x2'], `${id} positionsA`);
    validatePackedUint32View(props.offsetsA, `${id} offsetsA`);
    if (props.offsetsA.length < 2) {
      throw new Error(`${id} offsetsA must contain at least two rows`);
    }
    if (Boolean(props.positionsB) !== Boolean(props.offsetsB)) {
      throw new Error(`${id} positionsB and offsetsB must be given together`);
    }
    if (props.positionsB && props.offsetsB) {
      validatePackedView(props.positionsB, ['float32x2'], `${id} positionsB`);
      validatePackedUint32View(props.offsetsB, `${id} offsetsB`);
      if (props.offsetsB.length < 2) {
        throw new Error(`${id} offsetsB must contain at least two rows`);
      }
    }
    validatePackedUint32View(props.pairA, `${id} pairA`);
    validatePackedUint32View(props.pairB, `${id} pairB`);
    if (props.pairA.length < 1 || props.pairB.length !== props.pairA.length) {
      throw new Error(`${id} pairA and pairB must have the same positive length`);
    }
    if (props.activePairCount) {
      validatePackedUint32View(props.activePairCount, `${id} activePairCount`);
      if (props.activePairCount.length < 1) {
        throw new Error(`${id} activePairCount must contain one uint32 row`);
      }
    }
    if (!props.hausdorff && !props.frechet && !props.maxDistance) {
      throw new Error(`${id} needs hausdorff, frechet or maxDistance output`);
    }
    for (const [name, view, format] of [
      ['hausdorff', props.hausdorff, 'float32'],
      ['frechet', props.frechet, 'float32'],
      ['maxDistance', props.maxDistance, 'float32'],
      ['status', props.status, 'uint32']
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
    this.maxFrechetVertices = props.maxFrechetVertices ?? 256;
    if (
      !Number.isInteger(this.maxFrechetVertices) ||
      this.maxFrechetVertices < 1 ||
      this.maxFrechetVertices > MAXIMUM_FRECHET_VERTICES
    ) {
      throw new Error(
        `${id} maxFrechetVertices must be an integer from 1 to ${MAXIMUM_FRECHET_VERTICES}`
      );
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.hausdorff, props.frechet, props.maxDistance, props.status],
      [
        props.positionsA,
        props.offsetsA,
        props.positionsB,
        props.offsetsB,
        props.pairA,
        props.pairB,
        props.activePairCount
      ]
    );
  }

  /** Returns the status kernel and the requested distance kernels. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positionsA,
      props.offsetsA,
      props.positionsB,
      props.offsetsB,
      props.pairA,
      props.pairB,
      props.activePairCount,
      props.hausdorff,
      props.frechet,
      props.maxDistance,
      props.status
    ]);
    const positionsB = props.positionsB ?? props.positionsA;
    const offsetsB = props.offsetsB ?? props.offsetsA;
    const pairCount = props.pairA.length;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const constants = `const PAIR_COUNT: u32 = ${pairCount}u;
const TRACK_COUNT_A: u32 = ${props.offsetsA.length - 1}u;
const TRACK_COUNT_B: u32 = ${offsetsB.length - 1}u;
const FRECHET_CAP: u32 = ${this.maxFrechetVertices}u;
const DENSIFY: u32 = ${this.densifySubdivisions}u;
const STATUS_EMPTY: u32 = ${GPU_TRACK_SIMILARITY_STATUS.emptyTrack}u;
const STATUS_TOO_LONG: u32 = ${GPU_TRACK_SIMILARITY_STATUS.frechetCapExceeded}u;
const STATUS_INVALID: u32 = ${GPU_TRACK_SIMILARITY_STATUS.invalidPair}u;`;

    // The status buffer is needed by the distance kernels even when the caller does not want it.
    const status = props.status ?? createTransientView(graph, `${id}-status`, 'uint32', pairCount);
    const statusBindings: WGSLKernelBinding[] = [
      {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
      {name: 'pairB', view: props.pairB, type: 'u32', access: 'read'},
      {name: 'offsetsA', view: props.offsetsA, type: 'u32', access: 'read'},
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
      {name: 'positionsA', view: props.positionsA, type: 'f32', access: 'read'},
      {name: 'offsetsA', view: props.offsetsA, type: 'u32', access: 'read'},
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
  let from0 = loadA(start + segment);
  return from0 + (loadA(start + segment + 1u) - from0) * (f32(subStep) / f32(DENSIFY));
}
fn densifiedPointB(start: u32, count: u32, k: u32) -> vec2<f32> {
  let segment = k / DENSIFY;
  let subStep = k - segment * DENSIFY;
  if (subStep == 0u || segment + 1u >= count) { return loadB(start + min(segment, count - 1u)); }
  let from0 = loadB(start + segment);
  return from0 + (loadB(start + segment + 1u) - from0) * (f32(subStep) / f32(DENSIFY));
}
fn pointSegmentSquared(point: vec2<f32>, from0: vec2<f32>, to0: vec2<f32>) -> f32 {
  let edge = to0 - from0;
  let lengthSquared = dot(edge, edge);
  var t = 0.0;
  if (lengthSquared > 0.0) { t = clamp(dot(point - from0, edge) / lengthSquared, 0.0, 1.0); }
  let offset = point - (from0 + edge * t);
  return dot(offset, offset);
}`;

    if (props.hausdorff) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-hausdorff`,
          operation: OPERATION,
          variant: 'hausdorff',
          bindings: distanceBindings(props.hausdorff),
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
    distances[distancesOffset + pair] = select(sqrt(partials[0]), notANumber, skipped);
  }`
        })
      );
    }

    if (props.maxDistance) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-max-distance`,
          operation: OPERATION,
          variant: 'max-distance',
          bindings: distanceBindings(props.maxDistance),
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
        let offset = point - loadB(startB + j);
        farthest = max(farthest, dot(offset, offset));
      }
    }
  } else {
    for (var j = lane; j < countB; j += WORKGROUP_SIZE) {
      let point = loadB(startB + j);
      for (var i = 0u; i < countA; i++) {
        let offset = point - loadA(startA + i);
        farthest = max(farthest, dot(offset, offset));
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
    distances[distancesOffset + pair] = select(sqrt(partials[0]), notANumber, skipped);
  }`
        })
      );
    }

    if (props.frechet) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-frechet`,
          operation: OPERATION,
          variant: 'frechet',
          bindings: distanceBindings(props.frechet),
          workgroupSize: FRECHET_LANES,
          invocationCount: pairCount * FRECHET_LANES,
          guardIndex: false,
          declarations: `${constants}
const WORKGROUP_SIZE: u32 = ${FRECHET_LANES}u;
const LANES: u32 = ${FRECHET_LANES}u;
var<workgroup> olderDiagonal: array<f32, ${FRECHET_LANES}>;
var<workgroup> previousDiagonal: array<f32, ${FRECHET_LANES}>;
var<workgroup> boundaryRow: array<f32, ${this.maxFrechetVertices}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
${trackHelpers}
// Squared distance between densified vertex rowLong of the longer track and rowShort of the other.
fn coupledSquared(swapped: bool, startA: u32, countA: u32, startB: u32, countB: u32, rowLong: u32, rowShort: u32) -> f32 {
  var offset = vec2<f32>(0.0);
  if (swapped) {
    offset = densifiedPointB(startB, countB, rowLong) - densifiedPointA(startA, countA, rowShort);
  } else {
    offset = densifiedPointA(startA, countA, rowLong) - densifiedPointB(startB, countB, rowShort);
  }
  return dot(offset, offset);
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
    distances[distancesOffset + pair] = sqrt(finalValue);
  } else if (skipped && lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = notANumber;
  }`
        })
      );
    }
    return nodes;
  }
}
