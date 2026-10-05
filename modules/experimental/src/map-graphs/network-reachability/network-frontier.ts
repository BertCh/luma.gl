// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';

/**
 * Compact frontier queues with a round-stamped visited set, shared by the phases of
 * {@link GPUNetworkReachability}. @internal
 *
 * Round `i` reads the queue region `(i % 2) * nodeCount` and pushes into region
 * `((i + 1) % 2) * nodeCount`. A node is pushed for round `r` only when
 * `atomicMax(queuedRound[node], stampBase + r + 1) < stampBase + r + 1`, so each round's queue holds
 * at most `nodeCount` unique nodes and stamps never need clearing within one encoding. Every push
 * that opens a new 256-entry chunk grows the next round's indirect `x`, so no gate node is needed:
 * a round that received no pushes dispatches zero workgroups, and so does every later round.
 *
 * A phase is a contiguous run of rounds with its own control region (`truncated`, `limit`, two
 * parameter words, and `roundCounts`), its own dispatch slots, and a stamp base. Phases may share one
 * {@link FrontierState}: a later phase continues the stamps above the earlier phase's maximum
 * (`stampBase + maxRounds`), and re-initializes only its own control words and dispatch slots.
 */

/** Queue chunk and workgroup size of frontier rounds. */
export const FRONTIER_WORKGROUP_SIZE = 256;

/** Words before `roundCounts` in a phase control region. */
export const FRONTIER_PHASE_HEADER_WORDS = 4;

/** Phase control word: 1 when a push targeted a round at or beyond the limit. */
export const FRONTIER_CONTROL_TRUNCATED = 0;
/** Phase control word: resolved per-frame round limit. */
export const FRONTIER_CONTROL_LIMIT = 1;
/** Phase control word: first phase-defined parameter (the cost phase stores `costLimit` bits). */
export const FRONTIER_CONTROL_PARAMETER_0 = 2;
/** Phase control word: second phase-defined parameter. */
export const FRONTIER_CONTROL_PARAMETER_1 = 3;

/** Graph-owned buffers of a frontier. @internal */
export type FrontierState = {
  nodeCount: number;
  /** `2 * nodeCount` node IDs, two alternating round regions. */
  queues: GraphDataView<'uint32'>;
  /** Per-node last stamp, zeroed by the first phase's initialization. */
  queuedRound: GraphDataView<'uint32'>;
  /** Phase control regions. */
  control: GraphDataView<'uint32'>;
  /** Indirect `[x, y, z]` slots of even rounds. */
  dispatchEven: GraphDataView<'uint32'>;
  /** Indirect `[x, y, z]` slots of odd rounds. */
  dispatchOdd: GraphDataView<'uint32'>;
};

/** One run of rounds inside a {@link FrontierState}. @internal */
export type FrontierPhase = {
  /** Number of round nodes the phase unrolls. */
  maxRounds: number;
  /** Largest stamp already used by earlier phases of this state, 0 for the first phase. */
  stampBase: number;
  /** Word offset of this phase's control region inside `FrontierState.control`. */
  controlWordOffset: number;
  /** Slot offset of this phase inside each dispatch buffer. */
  dispatchSlotOffset: number;
};

/** Control words and dispatch slots one phase needs. @internal */
export function getFrontierPhaseLayout(maxRounds: number): {
  controlWordCount: number;
  dispatchSlotCount: number;
} {
  return {
    controlWordCount: FRONTIER_PHASE_HEADER_WORDS + maxRounds + 1,
    dispatchSlotCount: Math.floor(maxRounds / 2) + 1
  };
}

/**
 * Creates the frontier transients. `controlWordCount` and `dispatchSlotCount` must cover every phase
 * placed in the state (sum of {@link getFrontierPhaseLayout}, or the maximum of the slot counts when
 * phases re-initialize shared slots).
 *
 * @internal
 */
export function createFrontierState<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: {
    nodeCount: number;
    controlWordCount: number;
    dispatchSlotCount: number;
  }
): FrontierState {
  const dispatchUsage = Buffer.STORAGE | Buffer.INDIRECT;
  return {
    nodeCount: props.nodeCount,
    queues: createTransientView(graph, `${id}-queues`, 'uint32', 2 * props.nodeCount),
    queuedRound: createTransientView(graph, `${id}-queued-round`, 'uint32', props.nodeCount),
    control: createTransientView(graph, `${id}-control`, 'uint32', props.controlWordCount),
    dispatchEven: createTransientView(
      graph,
      `${id}-dispatch-even`,
      'uint32',
      props.dispatchSlotCount * 3,
      dispatchUsage
    ),
    dispatchOdd: createTransientView(
      graph,
      `${id}-dispatch-odd`,
      'uint32',
      props.dispatchSlotCount * 3,
      dispatchUsage
    )
  };
}

/** Largest stamp a phase can write; the next phase on the same state uses it as `stampBase`. */
export function getFrontierPhaseMaximumStamp(phase: FrontierPhase): number {
  return phase.stampBase + phase.maxRounds + 1;
}

/**
 * Returns the frontier storage bindings in `@binding` order: `queues` (u32), `queuedRound`,
 * `control` and `nextDispatch` (atomic<u32>). `nextDispatch` is the dispatch buffer of the round
 * the kernel pushes into, which is the other parity from the round it reads.
 *
 * @internal
 */
export function getFrontierBindings(
  state: FrontierState,
  nextRoundIsOdd: boolean
): MapGraphKernelBinding[] {
  return [
    {name: 'queues', view: state.queues, type: 'u32', access: 'read_write'},
    {
      name: 'queuedRound',
      view: state.queuedRound,
      type: 'atomic<u32>',
      access: 'read_write'
    },
    {
      name: 'control',
      view: state.control,
      type: 'atomic<u32>',
      access: 'read_write'
    },
    {
      name: 'nextDispatch',
      view: nextRoundIsOdd ? state.dispatchOdd : state.dispatchEven,
      type: 'atomic<u32>',
      access: 'read_write'
    }
  ];
}

/**
 * WGSL for the constants and `frontierPush(targetNode: u32)` of a kernel that pushes into
 * `nextRound`. Needs the {@link getFrontierBindings} bindings. `frontierPush` sets the truncated flag
 * instead of pushing when `nextRound` is at or beyond the phase limit.
 *
 * @internal
 */
export function getFrontierPushSource(
  phase: FrontierPhase,
  nodeCount: number,
  nextRound: number,
  maximumWorkgroupsPerDimension: number
): string {
  const nextWord = 3 * (phase.dispatchSlotOffset + Math.floor(nextRound / 2));
  return /* wgsl */ `
const FRONTIER_MAXIMUM_X: u32 = ${maximumWorkgroupsPerDimension}u;
const FRONTIER_TRUNCATED_WORD: u32 = controlOffset + ${phase.controlWordOffset + FRONTIER_CONTROL_TRUNCATED}u;
const FRONTIER_LIMIT_WORD: u32 = controlOffset + ${phase.controlWordOffset + FRONTIER_CONTROL_LIMIT}u;
const FRONTIER_PARAMETER_0_WORD: u32 = controlOffset + ${phase.controlWordOffset + FRONTIER_CONTROL_PARAMETER_0}u;
const FRONTIER_PARAMETER_1_WORD: u32 = controlOffset + ${phase.controlWordOffset + FRONTIER_CONTROL_PARAMETER_1}u;
const FRONTIER_ROUND_COUNTS_WORD: u32 = controlOffset + ${phase.controlWordOffset + FRONTIER_PHASE_HEADER_WORDS}u;
const FRONTIER_NEXT_ROUND: u32 = ${nextRound}u;
const FRONTIER_NEXT_STAMP: u32 = ${phase.stampBase + nextRound + 1}u;
const FRONTIER_NEXT_QUEUE_BASE: u32 = ${(nextRound % 2) * nodeCount}u;
const FRONTIER_NEXT_DISPATCH_WORD: u32 = ${nextWord}u;

fn frontierPush(targetNode: u32) {
  if (FRONTIER_NEXT_ROUND >= atomicLoad(&control[FRONTIER_LIMIT_WORD])) {
    atomicStore(&control[FRONTIER_TRUNCATED_WORD], 1u);
    return;
  }
  if (atomicMax(&queuedRound[queuedRoundOffset + targetNode], FRONTIER_NEXT_STAMP) >= FRONTIER_NEXT_STAMP) {
    return;
  }
  let slot = atomicAdd(&control[FRONTIER_ROUND_COUNTS_WORD + FRONTIER_NEXT_ROUND], 1u);
  queues[queuesOffset + FRONTIER_NEXT_QUEUE_BASE + slot] = targetNode;
  if (slot % ${FRONTIER_WORKGROUP_SIZE}u == 0u && slot / ${FRONTIER_WORKGROUP_SIZE}u < FRONTIER_MAXIMUM_X) {
    atomicAdd(&nextDispatch[nextDispatchOffset + FRONTIER_NEXT_DISPATCH_WORD], 1u);
  }
}`;
}

/**
 * WGSL statements for an initialize kernel's `index` that reset one phase: control words (zeroed,
 * with `limit` and the parameter words set from the given WGSL `u32` expressions) and the phase's
 * dispatch slots (`x = 0, y = 1, z = 1`). Needs `control`, `dispatchEven` and `dispatchOdd` bound as
 * plain `u32` read_write arrays. The kernel needs at least {@link getFrontierResetInvocationCount}
 * invocations. Zeroing `queuedRound` is the first phase's responsibility.
 *
 * @internal
 */
export function getFrontierResetSource(
  phase: FrontierPhase,
  expressions: {limit: string; parameter0?: string; parameter1?: string}
): string {
  const {controlWordCount, dispatchSlotCount} = getFrontierPhaseLayout(phase.maxRounds);
  return `if (index < ${controlWordCount}u) {
    var controlValue = 0u;
    if (index == ${FRONTIER_CONTROL_LIMIT}u) { controlValue = ${expressions.limit}; }
    if (index == ${FRONTIER_CONTROL_PARAMETER_0}u) { controlValue = ${expressions.parameter0 ?? '0u'}; }
    if (index == ${FRONTIER_CONTROL_PARAMETER_1}u) { controlValue = ${expressions.parameter1 ?? '0u'}; }
    control[controlOffset + ${phase.controlWordOffset}u + index] = controlValue;
  }
  if (index < ${dispatchSlotCount * 3}u) {
    let dispatchValue = select(1u, 0u, index % 3u == 0u);
    dispatchEven[dispatchEvenOffset + ${phase.dispatchSlotOffset * 3}u + index] = dispatchValue;
    dispatchOdd[dispatchOddOffset + ${phase.dispatchSlotOffset * 3}u + index] = dispatchValue;
  }`;
}

/** Minimum invocation count of a kernel using {@link getFrontierResetSource}. @internal */
export function getFrontierResetInvocationCount(phase: FrontierPhase): number {
  const {controlWordCount, dispatchSlotCount} = getFrontierPhaseLayout(phase.maxRounds);
  return Math.max(controlWordCount, dispatchSlotCount * 3);
}

/**
 * WGSL statements (one thread, `control` bound as a plain `u32` array) that define
 * `frontierRoundCount` (rounds `i < limit` with a non-empty queue, at least `min(limit, 1)`) and
 * `frontierConverged` (1 when nothing was truncated and `limit > 0`).
 *
 * @internal
 */
export function getFrontierFinalizeSource(phase: FrontierPhase): string {
  const base = `controlOffset + ${phase.controlWordOffset}u`;
  return `let frontierLimit = control[${base} + ${FRONTIER_CONTROL_LIMIT}u];
  var frontierRoundCount = 0u;
  for (var roundIndex = 0u; roundIndex < ${phase.maxRounds}u; roundIndex++) {
    if (roundIndex >= frontierLimit || control[${base} + ${FRONTIER_PHASE_HEADER_WORDS}u + roundIndex] == 0u) {
      break;
    }
    frontierRoundCount++;
  }
  frontierRoundCount = max(frontierRoundCount, min(frontierLimit, 1u));
  let frontierConverged = select(
    0u,
    1u,
    frontierLimit > 0u && control[${base} + ${FRONTIER_CONTROL_TRUNCATED}u] == 0u
  );`;
}

/** Properties for {@link createFrontierRoundNode}. @internal */
export type FrontierRoundProps = {
  /** Graph-wide node ID. */
  id: string;
  /** Operation name reported in the workload estimate. */
  operation: string;
  /** Implementation variant reported in the workload estimate. */
  variant?: string;
  state: FrontierState;
  phase: FrontierPhase;
  /** Round index inside the phase, `0 <= round < phase.maxRounds`. */
  round: number;
  /**
   * Phase-specific bindings placed before the frontier bindings. At most 4, since a kernel has 8
   * storage bindings on default WebGPU limits.
   */
  bindings: readonly MapGraphKernelBinding[];
  /** Phase-specific module-scope WGSL, such as constants. */
  declarations?: string;
  /**
   * WGSL `fn visit(node: u32)` that processes one queued node and calls `push(targetNode: u32)` for
   * each node that must be processed again. It may use the phase bindings and the
   * `FRONTIER_*_WORD` constants. Every successful improvement of a node must be followed by a
   * `push` of that node.
   */
  visitSource: string;
  /** Hops one workgroup chains inside a dispatch, 1 to 64. */
  localIterations: number;
};

/**
 * Builds the node of one round: an indirect-dispatched kernel whose workgroups each process
 * 256-entry chunks of the round's queue (grid-stride) and chain up to `localIterations` hops on the
 * nodes they improve through a workgroup-local queue (capacity 256), flushing overflow and the
 * leftovers of the last hop to the next round with the stamped push.
 *
 * @internal
 */
export function createFrontierRoundNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: FrontierRoundProps
): GPUCommandNode<Parameters> {
  const {state, phase, round, localIterations} = props;
  const {nodeCount} = state;
  const frontierBindings = getFrontierBindings(state, (round + 1) % 2 === 1);
  const bindings = [...props.bindings, ...frontierBindings];
  if (bindings.length > 8) {
    throw new Error(`${props.id} needs more than 8 storage bindings`);
  }
  const maximumX = graph.device.limits.maxComputeWorkgroupsPerDimension;
  const currentDispatch = round % 2 === 0 ? state.dispatchEven : state.dispatchOdd;
  const slotByteOffset = 12 * (phase.dispatchSlotOffset + Math.floor(round / 2));
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: props.variant,
    bindings,
    invocationCount: nodeCount,
    guardIndex: false,
    condition: {
      id: `${props.id}-gate`,
      source: 'gpu',
      mode: 'indirect',
      buffer: currentDispatch.buffer,
      byteOffset: currentDispatch.byteOffset + slotByteOffset
    },
    extraResources: [{buffer: currentDispatch, usage: 'indirect'}],
    declarations: `${props.declarations ?? ''}
${getFrontierPushSource(phase, nodeCount, round + 1, maximumX)}
const FRONTIER_LOCAL_ITERATIONS: u32 = ${localIterations}u;
const FRONTIER_CURRENT_ROUND: u32 = ${round}u;
const FRONTIER_CURRENT_QUEUE_BASE: u32 = ${(round % 2) * nodeCount}u;

var<workgroup> frontierLocalQueue: array<u32, ${2 * FRONTIER_WORKGROUP_SIZE}>;
var<workgroup> frontierLocalCount: array<atomic<u32>, 2>;
var<workgroup> frontierSharedCount: u32;
var<private> frontierPushToLocal: bool;
var<private> frontierLocalWriteHalf: u32;

fn push(targetNode: u32) {
  if (frontierPushToLocal) {
    let slot = atomicAdd(&frontierLocalCount[frontierLocalWriteHalf], 1u);
    if (slot < ${FRONTIER_WORKGROUP_SIZE}u) {
      frontierLocalQueue[frontierLocalWriteHalf * ${FRONTIER_WORKGROUP_SIZE}u + slot] = targetNode;
      return;
    }
  }
  frontierPush(targetNode);
}

${props.visitSource}`,
    body: `if (localInvocationIndex == 0u) {
    frontierSharedCount = atomicLoad(&control[FRONTIER_ROUND_COUNTS_WORD + FRONTIER_CURRENT_ROUND]);
  }
  let queueLength = workgroupUniformLoad(&frontierSharedCount);
  let dispatchedGroupCount = min(
    (queueLength + ${FRONTIER_WORKGROUP_SIZE - 1}u) / ${FRONTIER_WORKGROUP_SIZE}u,
    FRONTIER_MAXIMUM_X
  );
  for (var chunk = workgroupId.x; chunk * ${FRONTIER_WORKGROUP_SIZE}u < queueLength; chunk += dispatchedGroupCount) {
    // Hop 0 reads the global queue chunk; hop k reads the local queue written by hop k - 1.
    frontierPushToLocal = FRONTIER_LOCAL_ITERATIONS > 1u;
    frontierLocalWriteHalf = 1u;
    let queueIndex = chunk * ${FRONTIER_WORKGROUP_SIZE}u + localInvocationIndex;
    if (queueIndex < queueLength) {
      visit(queues[queuesOffset + FRONTIER_CURRENT_QUEUE_BASE + queueIndex]);
    }
    for (var hop = 1u; hop < FRONTIER_LOCAL_ITERATIONS; hop++) {
      workgroupBarrier();
      let readHalf = hop & 1u;
      if (localInvocationIndex == 0u) {
        frontierSharedCount = min(atomicLoad(&frontierLocalCount[readHalf]), ${FRONTIER_WORKGROUP_SIZE}u);
        atomicStore(&frontierLocalCount[readHalf], 0u);
      }
      let localLength = workgroupUniformLoad(&frontierSharedCount);
      if (localLength == 0u) { break; }
      // The last hop pushes straight to the next global round.
      frontierPushToLocal = hop + 1u < FRONTIER_LOCAL_ITERATIONS;
      frontierLocalWriteHalf = (hop + 1u) & 1u;
      if (localInvocationIndex < localLength) {
        visit(frontierLocalQueue[readHalf * ${FRONTIER_WORKGROUP_SIZE}u + localInvocationIndex]);
      }
    }
    workgroupBarrier();
  }`
  });
}
