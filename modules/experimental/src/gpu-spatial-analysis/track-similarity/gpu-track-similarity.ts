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
const MAXIMUM_FRECHET_WORKGROUP_SIZE = 256;

/** Bits of the `status` column of {@link GPUTrackSimilarity}. */
export const GPU_TRACK_SIMILARITY_STATUS = {
  /** At least one of the two tracks has no vertices. Both distances are NaN. */
  emptyTrack: 1,
  /** A track has more than `maxFrechetVertices` vertices. The Frechet distance is NaN. */
  frechetCapExceeded: 2,
  /** The pair index is out of range or at or beyond `activePairCount`. Both distances are NaN. */
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
  /** Optional per-pair {@link GPU_TRACK_SIMILARITY_STATUS} bit set, 0 when computed. */
  status?: GraphDataView<'uint32'>;
  /**
   * Vertex cap of the Frechet dynamic program, from 1 to 256 (the invocation limit of one
   * workgroup). Defaults to 128. Pairs with a longer track report `frechetCapExceeded` and a NaN
   * Frechet distance; resample long tracks with `GPUTrajectoryResample` or simplify them first.
   */
  maxFrechetVertices?: number;
};

/**
 * Computes the discrete Hausdorff and discrete Frechet distance between pairs of tracks or rings.
 *
 * Both are vertex-based, matching Shapely `hausdorff_distance` and `frechet_distance` without
 * densification: Hausdorff is `max(directed(A, B), directed(B, A))` with
 * `directed(A, B) = max over a of min over b of |a - b|`, and the Frechet distance is the minimum
 * over monotone couplings of the largest coupled vertex distance. Rings are treated as open vertex
 * sequences (a repeated closing vertex is just another vertex), so Frechet is orientation and start
 * sensitive. Planar distances only.
 *
 * Composition: one status kernel, then one workgroup per pair. Hausdorff uses 64 lanes that stride
 * over the vertices of one track, take the minimum over the other track, and reduce with a
 * workgroup max tree (no cap, `O(n * m)` per pair). Frechet runs the dynamic program on
 * anti-diagonals, one lane per vertex of track A, with three diagonals in workgroup memory. It is
 * limited by `maxFrechetVertices` for both tracks (flagged in `status`).
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
  /** Frechet workgroup size, the smallest power of two of at least 32 holding the cap. */
  readonly frechetWorkgroupSize: number;

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
    if (!props.hausdorff && !props.frechet) {
      throw new Error(`${id} needs hausdorff or frechet output`);
    }
    for (const [name, view, format] of [
      ['hausdorff', props.hausdorff, 'float32'],
      ['frechet', props.frechet, 'float32'],
      ['status', props.status, 'uint32']
    ] as const) {
      if (view) {
        validatePackedView(view, [format], `${id} ${name}`);
        if (view.length !== props.pairA.length) {
          throw new Error(`${id} ${name} length must equal the pair capacity`);
        }
      }
    }
    this.maxFrechetVertices = props.maxFrechetVertices ?? 128;
    if (
      !Number.isInteger(this.maxFrechetVertices) ||
      this.maxFrechetVertices < 1 ||
      this.maxFrechetVertices > MAXIMUM_FRECHET_WORKGROUP_SIZE
    ) {
      throw new Error(
        `${id} maxFrechetVertices must be an integer from 1 to ${MAXIMUM_FRECHET_WORKGROUP_SIZE}`
      );
    }
    this.frechetWorkgroupSize = Math.max(32, 2 ** Math.ceil(Math.log2(this.maxFrechetVertices)));
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.hausdorff, props.frechet, props.status],
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
    else if (countA > FRECHET_CAP || countB > FRECHET_CAP) { flags = STATUS_TOO_LONG; }
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
    const squaredDistance = `fn squaredDistance(rowA: u32, rowB: u32) -> f32 {
  let dx = positionsA[positionsAOffset + rowA * 2u] - positionsB[positionsBOffset + rowB * 2u];
  let dy = positionsA[positionsAOffset + rowA * 2u + 1u] - positionsB[positionsBOffset + rowB * 2u + 1u];
  return dx * dx + dy * dy;
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
${squaredDistance}`,
          body: `${prologue('(STATUS_EMPTY | STATUS_INVALID)')}
  var worst = 0.0;
  for (var i = lane; i < countA; i += WORKGROUP_SIZE) {
    var nearest = 3.402823466e+38;
    for (var j = 0u; j < countB; j++) { nearest = min(nearest, squaredDistance(startA + i, startB + j)); }
    worst = max(worst, nearest);
  }
  for (var j = lane; j < countB; j += WORKGROUP_SIZE) {
    var nearest = 3.402823466e+38;
    for (var i = 0u; i < countA; i++) { nearest = min(nearest, squaredDistance(startA + i, startB + j)); }
    worst = max(worst, nearest);
  }
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

    if (props.frechet) {
      const size = this.frechetWorkgroupSize;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-frechet`,
          operation: OPERATION,
          variant: 'frechet',
          bindings: distanceBindings(props.frechet),
          workgroupSize: size,
          invocationCount: pairCount * size,
          guardIndex: false,
          declarations: `${constants}
const WORKGROUP_SIZE: u32 = ${size}u;
const MAXIMUM_DIAGONALS: u32 = ${2 * this.maxFrechetVertices - 1}u;
var<workgroup> olderDiagonal: array<f32, ${size}>;
var<workgroup> previousDiagonal: array<f32, ${size}>;
${squaredDistance}`,
          body: `${prologue('(STATUS_EMPTY | STATUS_INVALID | STATUS_TOO_LONG)')}
  let infinity = 3.402823466e+38;
  olderDiagonal[lane] = infinity;
  previousDiagonal[lane] = infinity;
  workgroupBarrier();
  var finalValue = infinity;
  let diagonalCount = select(0u, countA + countB - 1u, countA > 0u && countB > 0u);
  // A constant trip count keeps the barriers in uniform control flow.
  for (var diagonal = 0u; diagonal < MAXIMUM_DIAGONALS; diagonal++) {
    var value = infinity;
    if (diagonal < diagonalCount && lane < countA && diagonal >= lane && diagonal - lane < countB) {
      let column = diagonal - lane;
      let cost = squaredDistance(startA + lane, startB + column);
      if (lane == 0u && column == 0u) {
        value = cost;
      } else if (lane == 0u) {
        value = max(previousDiagonal[0], cost);
      } else if (column == 0u) {
        value = max(previousDiagonal[lane - 1u], cost);
      } else {
        value = max(min(min(previousDiagonal[lane - 1u], previousDiagonal[lane]), olderDiagonal[lane - 1u]), cost);
      }
      if (lane == countA - 1u && column == countB - 1u) { finalValue = value; }
    }
    workgroupBarrier();
    olderDiagonal[lane] = previousDiagonal[lane];
    previousDiagonal[lane] = value;
    workgroupBarrier();
  }
  if (!skipped && lane == countA - 1u) {
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
