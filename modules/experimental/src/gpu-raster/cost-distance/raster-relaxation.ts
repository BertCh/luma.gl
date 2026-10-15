// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUComputationNodeAnnotation} from '@luma.gl/gpgpu';
import {
  getBoundedDispatchLayout,
  type GPUBoundedDispatchLayout,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {validatePackedView} from '../../utils/gpu-contributor-utils';
import type {GPURasterCellSizeMode} from '../../analysis-core/raster-grid';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLDispatchGate,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {createContributorTransientView, getRasterGridWGSL} from './raster-grid-utils';

/** Largest accepted compile-time iteration count for GPU-gated raster iterations. */
export const GPU_RASTER_MAXIMUM_ITERATIONS = 1024;

/** Edge length of one square relaxation tile; one 256-invocation workgroup owns one tile. @internal */
export const RASTER_RELAXATION_TILE_SIZE = 16;

// Grid-implicit relaxation, not GPUGraphSingleSourceShortestPath: that needs explicit CSR (8N edges) and has no anisotropic cell-size, tie or tile-wise rules.
const WORKGROUP_SIZE = 256;

/** Throws unless `iterations` is an integer in `[1, GPU_RASTER_MAXIMUM_ITERATIONS]`. @internal */
export function validateRasterIterations(id: string, name: string, iterations: number): void {
  if (
    !Number.isSafeInteger(iterations) ||
    iterations < 1 ||
    iterations > GPU_RASTER_MAXIMUM_ITERATIONS
  ) {
    throw new Error(`${id} ${name} must be an integer in [1, ${GPU_RASTER_MAXIMUM_ITERATIONS}]`);
  }
}

/**
 * Graph-owned state of one GPU-gated iteration loop.
 *
 * `status` holds `[changed, enabled, iterationCount, converged]`. `dispatch` is the indirect
 * workgroup count of the gated kernel, kept in its own buffer because one buffer must not be both
 * an indirect argument and storage-written in one dispatch.
 *
 * @internal
 */
export type RasterIterationState<Parameters> = {
  status: GraphDataView<'uint32'>;
  /** Indirect dispatch triple of the gated kernel, the view of {@link gate}. */
  dispatch: GraphDataView<'uint32'>;
  /** Dispatch gate that owns `dispatch`: its one slot holds `layout`. */
  gate: WGSLDispatchGate<Parameters>;
  /** Bounded dispatch layout of the gated kernel. */
  layout: GPUBoundedDispatchLayout;
  /** Linear invocation count of the gated kernel. */
  invocationCount: number;
  /**
   * Status word that holds the iteration count at which the current execution stops. Set only for
   * continued relaxations, whose cumulative `iterationCount` carries over between executions, so
   * `maxIterations` is a budget per execution and not a bound on the count.
   */
  limitWord?: number;
};

/**
 * Creates status and indirect-dispatch transients for a gated kernel with `invocationCount`
 * linear invocations in 256-wide workgroups.
 *
 * @internal
 */
export function createRasterIterationState<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  invocationCount: number,
  contributorId: string = id,
  options: {
    /** Caller-owned status view (at least 4 words) used instead of a graph transient. */
    status?: GraphDataView<'uint32'>;
    /** See {@link RasterIterationState.limitWord}. */
    limitWord?: number;
  } = {}
): RasterIterationState<Parameters> {
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-dispatch`,
    operation,
    shapes: [[invocationCount, WORKGROUP_SIZE]]
  });
  return {
    status:
      options.status ??
      createContributorTransientView(graph, operation, contributorId, `${id}-status`, 'uint32', 4),
    limitWord: options.limitWord,
    dispatch: gate.view,
    gate,
    layout: getBoundedDispatchLayout(
      operation,
      Math.max(invocationCount, 1),
      WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    ),
    invocationCount
  };
}

/** Returns the indirect GPU condition and resource use that gate one kernel on `state`. @internal */
export function getRasterIterationCondition<Parameters>(
  state: RasterIterationState<Parameters>,
  nodeId: string
): {condition: GPUCommandGraphNodeCondition<Parameters>; extraResources: GraphBufferUse[]} {
  const {condition, extraResources} = state.gate.getGate(0);
  return {condition: {...condition, id: `${nodeId}-gate`}, extraResources};
}

/** Returns a one-thread node that enables the loop and arms the indirect dispatch. @internal */
export function createRasterIterationResetNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; state: RasterIterationState<Parameters>}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'iteration-reset',
    bindings: [
      {name: 'status', view: props.state.status, type: 'u32', access: 'read_write'},
      props.state.gate.binding
    ],
    invocationCount: 1,
    body: `status[statusOffset + 0u] = 0u;
  status[statusOffset + 1u] = 1u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;
  ${props.state.gate.armWGSL}`
  });
}

/**
 * Returns a one-thread gate that runs after each gated iteration: it counts the iteration, clears
 * the change flag, and zeroes the indirect dispatch once an iteration changed nothing (converged)
 * or `maxIterations` iterations ran. The WGSL source does not depend on the iteration index, so
 * every gate shares one shader.
 *
 * @internal
 */
export function createRasterIterationGateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    state: RasterIterationState<Parameters>;
    maxIterations: number;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'iteration-gate',
    bindings: [
      {name: 'status', view: props.state.status, type: 'u32', access: 'read_write'},
      props.state.gate.binding
    ],
    invocationCount: 1,
    body: `if (status[statusOffset + 1u] == 0u) { return; }
  let changed = status[statusOffset + 0u];
  status[statusOffset + 0u] = 0u;
  let iterationCount = status[statusOffset + 2u] + 1u;
  status[statusOffset + 2u] = iterationCount;
  if (changed == 0u) {
    status[statusOffset + 1u] = 0u;
    status[statusOffset + 3u] = 1u;
    ${props.state.gate.disarmWGSL}
    return;
  }
  if (iterationCount >= ${props.maxIterations}u) {
    status[statusOffset + 1u] = 0u;
    ${props.state.gate.disarmWGSL}
  }`
  });
}

/** Default number of unrolled iterations that share one gate node. @internal */
export const RASTER_GATE_INTERVAL = 4;

/** Largest gate group: one status bit per iteration of the group. @internal */
export const RASTER_GATE_MAXIMUM_GROUP_SIZE = 32;

/**
 * Elements between consecutive words of a {@link RasterGateSlotTable}: 256 bytes, the storage
 * binding alignment, so every slot sub-view binds at element offset 0 and the WGSL of every round
 * is the same text.
 */
const RASTER_GATE_SLOT_STRIDE = 64;

/**
 * Graph-owned table of gate-group positions, written once per encoding.
 *
 * Word `k` (at element `k * 64`) holds `k`. Round `k` of a gate group binds the one-word sub-view
 * {@link getSlot}`(k)` and reads its position from it, and the gate of a group of `n` rounds binds
 * `getSlot(n)` to read its size. So the position is data, never WGSL text (library rule 12), and
 * every round of a loop shares one pipeline.
 *
 * @internal
 */
export type RasterGateSlotTable<Parameters> = {
  /** Largest group size the table covers. */
  groupSize: number;
  /** One-thread-per-word node that writes the table; schedule it before the first round. */
  writeNode: GPUCommandNode<Parameters>;
  /** Returns the one-word view holding `slot`, for `slot` in `[0, groupSize]`. */
  getSlot: (slot: number) => GraphDataView<'uint32'>;
};

/** Creates a {@link RasterGateSlotTable} for groups of up to `groupSize` rounds. @internal */
export function createRasterGateSlotTable<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; contributorId?: string; groupSize: number}
): RasterGateSlotTable<Parameters> {
  const {groupSize} = props;
  if (
    !Number.isSafeInteger(groupSize) ||
    groupSize < 1 ||
    groupSize > RASTER_GATE_MAXIMUM_GROUP_SIZE
  ) {
    throw new Error(
      `${props.operation}: gate group size must be an integer in [1, ${RASTER_GATE_MAXIMUM_GROUP_SIZE}]`
    );
  }
  const table = createContributorTransientView(
    graph,
    props.operation,
    props.contributorId ?? props.id,
    `${props.id}-slots`,
    'uint32',
    (groupSize + 1) * RASTER_GATE_SLOT_STRIDE
  );
  const writeNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-slots-write`,
    operation: props.operation,
    variant: 'gate-slot-table',
    bindings: [{name: 'slotTable', view: table, type: 'u32', access: 'read_write'}],
    invocationCount: groupSize + 1,
    body: `slotTable[slotTableOffset + index * ${RASTER_GATE_SLOT_STRIDE}u] = index;`
  });
  const getSlot = (slot: number): GraphDataView<'uint32'> => {
    // Callers only ask for positions inside a group or for a group size, both within the table.
    assert(Number.isSafeInteger(slot) && slot >= 0 && slot <= groupSize);
    return graph.createDataView(table.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: table.byteOffset + slot * RASTER_GATE_SLOT_STRIDE * 4
    });
  };
  return {groupSize, writeNode, getSlot};
}

/**
 * Returns a one-thread gate that closes a group of consecutive gated iterations.
 *
 * Iteration `k` of the group records a change with `atomicOr(&status[0], 1u << k)`, reading `k`
 * from its slot word. The gate reads the group size from `groupSize` (a slot-table word), counts
 * the iterations up to and including the first unchanged one (or the whole group), publishes
 * convergence, and zeroes the indirect dispatch once converged or `maxIterations` ran. Iterations
 * of a group that follow a converged one only re-run an idempotent fixpoint step, so results and
 * counts match one gate per iteration while the graph needs one gate node per group. The WGSL
 * depends on neither the group index nor its size, so every gate of a loop shares one pipeline.
 *
 * @internal
 */
export function createRasterIterationGroupGateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    state: RasterIterationState<Parameters>;
    maxIterations: number;
    /** One-word view holding the number of iterations in this group, see {@link RasterGateSlotTable}. */
    groupSize: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'iteration-group-gate',
    bindings: [
      {name: 'status', view: props.state.status, type: 'u32', access: 'read_write'},
      props.state.gate.binding,
      {name: 'groupSize', view: props.groupSize, type: 'u32', access: 'read'}
    ],
    invocationCount: 1,
    body: `if (status[statusOffset + 1u] == 0u) { return; }
  let changedBits = status[statusOffset + 0u];
  status[statusOffset + 0u] = 0u;
  let size = groupSize[groupSizeOffset];
  var executed = size;
  var converged = false;
  for (var round = 0u; round < size; round++) {
    if (((changedBits >> round) & 1u) == 0u) {
      executed = round + 1u;
      converged = true;
      break;
    }
  }
  let iterationCount = status[statusOffset + 2u] + executed;
  status[statusOffset + 2u] = iterationCount;
  if (converged) {
    status[statusOffset + 1u] = 0u;
    status[statusOffset + 3u] = 1u;
    ${props.state.gate.disarmWGSL}
    return;
  }
  if (iterationCount >= ${
    props.state.limitWord === undefined
      ? `${props.maxIterations}u`
      : `status[statusOffset + ${props.state.limitWord}u]`
  }) {
    status[statusOffset + 1u] = 0u;
    ${props.state.gate.disarmWGSL}
  }`
  });
}

/** Per-iteration context passed to the node factory of {@link createRasterGatedLoopNodes}. @internal */
export type RasterGatedRound<Parameters> = {
  /** Unique node ID of this iteration's kernel. */
  nodeId: string;
  /**
   * Zero-based unrolled iteration index, for choosing among a fixed set of kernel variants only.
   * It must never reach WGSL text or an override constant (library rule 12).
   */
  iteration: number;
  /**
   * Binding of this iteration's slot word (its position inside the gate group). The round kernel
   * must declare it; {@link markChangedWGSL} and {@link roundSlotWGSL} read it.
   */
  slotBinding: WGSLKernelBinding;
  /** WGSL `u32` expression of this iteration's position inside its gate group. */
  roundSlotWGSL: string;
  /** WGSL statement that records "something changed" for this iteration. */
  markChangedWGSL: string;
  /** Indirect GPU condition gating the kernel. */
  condition: GPUCommandGraphNodeCondition<Parameters>;
  /** Resource uses the kernel must declare. */
  extraResources: GraphBufferUse[];
};

function annotateLoopNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  node: GPUCommandNode<Parameters>,
  loop: GPUComputationNodeAnnotation['loop'],
  loopRole: 'setup' | 'round' | 'gate'
): GPUCommandNode<Parameters> {
  if (graph.recording) {
    node.annotation = {
      ...node.annotation,
      ...(loop ? {loop: node.annotation?.loop ?? loop} : {}),
      attrs: {...node.annotation?.attrs, loopRole}
    };
  }
  return node;
}

/**
 * Unrolls `maxIterations` gated iterations with one gate node per `groupSize` iterations.
 *
 * `createRound` builds each iteration's kernel, must declare `slotBinding`, and must run
 * `markChangedWGSL` whenever the iteration changed anything. The first node,
 * `${gateId}-slots-write`, writes the gate slot table, so the node count is `1 + maxIterations + ceil(maxIterations / groupSize)` instead of
 * `2 * maxIterations`. The group position comes from a slot word, so rounds with the same kernel
 * variant share one pipeline whatever their index.
 *
 * @internal
 */
export function createRasterGatedLoopNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    /** Prefix of the round node IDs; round `i` is `${id}-${i}`. */
    id: string;
    /** Prefix of the gate node IDs; the gate after round `i` is `${gateId}-${i}`. Defaults to `${id}-gate`. */
    gateId?: string;
    operation: string;
    state: RasterIterationState<Parameters>;
    maxIterations: number;
    groupSize?: number;
    createRound: (round: RasterGatedRound<Parameters>) => GPUCommandNode<Parameters>;
  }
): GPUCommandNode<Parameters>[] {
  const groupSize = Math.min(
    props.groupSize ?? RASTER_GATE_INTERVAL,
    RASTER_GATE_MAXIMUM_GROUP_SIZE,
    Math.max(props.maxIterations, 1)
  );
  const gateId = props.gateId ?? `${props.id}-gate`;
  const slotTable = createRasterGateSlotTable<Parameters>(graph, {
    id: gateId,
    operation: props.operation,
    groupSize
  });
  const nodes: GPUCommandNode<Parameters>[] = [
    annotateLoopNode(graph, slotTable.writeNode, undefined, 'setup')
  ];
  for (let start = 0; start < props.maxIterations; start += groupSize) {
    const size = Math.min(groupSize, props.maxIterations - start);
    for (let groupOffset = 0; groupOffset < size; groupOffset++) {
      const iteration = start + groupOffset;
      const nodeId = `${props.id}-${iteration}`;
      const {condition, extraResources} = getRasterIterationCondition<Parameters>(
        props.state,
        nodeId
      );
      nodes.push(
        annotateLoopNode(
          graph,
          props.createRound({
            nodeId,
            iteration,
            slotBinding: {
              name: 'roundSlot',
              view: slotTable.getSlot(groupOffset),
              type: 'u32',
              access: 'read'
            },
            roundSlotWGSL: 'roundSlot[roundSlotOffset]',
            markChangedWGSL: 'atomicOr(&status[statusOffset], 1u << roundSlot[roundSlotOffset]);',
            condition,
            extraResources
          }),
          {id: props.id, iteration, phase: 'gated', maximumIterations: props.maxIterations},
          'round'
        )
      );
    }
    nodes.push(
      annotateLoopNode(
        graph,
        createRasterIterationGroupGateNode<Parameters>(graph, {
          id: `${gateId}-${start}`,
          operation: props.operation,
          state: props.state,
          maxIterations: props.maxIterations,
          groupSize: slotTable.getSlot(size)
        }),
        {
          id: props.id,
          iteration: start + size - 1,
          phase: 'gated',
          maximumIterations: props.maxIterations
        },
        'gate'
      )
    );
  }
  return nodes;
}

/** Publishes the converged flag and executed iteration count of one loop. @internal */
export function createRasterIterationFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    state: RasterIterationState<Parameters>;
    converged?: GraphDataView<'uint32'>;
    iterationCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'status', view: props.state.status, type: 'u32', access: 'read'}
  ];
  if (props.converged) {
    bindings.push({name: 'converged', view: props.converged, type: 'u32', access: 'read_write'});
  }
  if (props.iterationCount) {
    bindings.push({
      name: 'iterationCount',
      view: props.iterationCount,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'iteration-finalize',
    bindings,
    invocationCount: 1,
    body: `${props.converged ? 'converged[convergedOffset] = status[statusOffset + 3u];' : ''}
  ${props.iterationCount ? 'iterationCount[iterationCountOffset] = status[statusOffset + 2u];' : ''}`
  });
}

/** Element offset of the tile stamps inside a continuation state: 256 bytes, the binding alignment. */
const CONTINUATION_STAMP_OFFSET = 64;

/** State word holding the iteration count at which the current execution stops. */
const CONTINUATION_LIMIT_WORD = 4;

/**
 * Index of the restart word inside a continuation state. A caller sets it to a nonzero value
 * before an execution that must start from scratch (inputs changed); the GPU clears it when the
 * execution has consumed it.
 */
export const GPU_RASTER_RELAXATION_RESTART_WORD = 5;

/**
 * Returns the word count of the caller-owned state that lets a tiled relaxation of a `grid`-sized
 * raster continue across executions: `[changed, enabled, iterationCount, converged, limit,
 * restart]` padded to 256 bytes, then one stamp per 16x16 tile. A zero-filled state means "no
 * previous execution", so the first execution starts from scratch.
 */
export function getGPURasterRelaxationContinuationStateLength(grid: {
  width: number;
  height: number;
}): number {
  const tileCount =
    Math.ceil(grid.width / RASTER_RELAXATION_TILE_SIZE) *
    Math.ceil(grid.height / RASTER_RELAXATION_TILE_SIZE);
  return CONTINUATION_STAMP_OFFSET + tileCount;
}

/**
 * Continuation of a tiled relaxation across executions, see {@link RasterTiledRelaxationProps}.
 *
 * @internal
 */
export type RasterTiledRelaxationContinuation = {
  /**
   * Caller-owned state of {@link getGPURasterRelaxationContinuationStateLength} words. It must
   * outlive an execution (an imported buffer, never a graph transient, which the lifetime planner
   * may alias).
   */
  state: GraphDataView<'uint32'>;
  /** Keep a completed stage frozen until an explicit restart (dependent multi-stage consumers). */
  retainConverged?: boolean;
  /** Optional preceding relaxation state; run only after its converged word is set. */
  prerequisite?: GraphDataView<'uint32'>;

  /**
   * `[invocationCount, workgroupSize?]` of the consumer kernels that must run only on a restart
   * (value initialization, seeding). Shape `i` is gated by `getRestartGate(i, ...)`.
   */
  restartShapes?: readonly (readonly [invocationCount: number, workgroupSize?: number])[];
};

/** Graph-owned state of one tiled min-relaxation. @internal */
export type RasterTiledRelaxation<Parameters> = {
  /** Gated-loop state; the gated kernel runs one workgroup per tile. */
  state: RasterIterationState<Parameters>;
  /**
   * Per-tile activity stamps. A tile is processed in iteration `i` when it or one of its eight
   * neighbor tiles holds a stamp `>= i + 1`. Seeds mark a tile with `atomicMax(&stamp, 1u)`.
   */
  tileStamps: GraphDataView<'uint32'>;
  /** Tiles per row. */
  tilesX: number;
  /** Tile count. */
  tileCount: number;
  /** Present for a continued relaxation. */
  continuation?: {
    /**
     * Returns the indirect condition and resource use that run a consumer kernel only when this
     * execution restarts (shape `shapeIndex` of `restartShapes`, named `nodeId`'s gate).
     */
    getRestartGate: (
      shapeIndex: number,
      nodeId: string
    ) => {condition: GPUCommandGraphNodeCondition<Parameters>; extraResources: GraphBufferUse[]};
  };
};

/** Properties for {@link createRasterTiledRelaxation}. @internal */
export type RasterTiledRelaxationProps = {
  /** Prefix for relaxation node and transient IDs. */
  id: string;
  /** Operation name for workload estimates. */
  operation: string;
  width: number;
  height: number;
  cellSizeMode: GPURasterCellSizeMode;
  /**
   * Unrolled iterations per execution. With `continuation` this is a budget per submission and
   * the iteration count accumulates across executions.
   */
  maxIterations: number;
  /**
   * Opt-in continuation. An execution then resumes the previous one (no reset, no consumer
   * initialization) when the state says that one stopped unconverged and its restart word is 0,
   * and otherwise restarts from scratch. The working values must be caller-owned too: they carry
   * the progress. Callers that change any input must set the restart word. The schedule (plain
   * tiled passes, then sweeps from `sweepAfterIteration`) follows the cumulative iteration count,
   * so a continued run takes the trajectory of one long execution.
   */
  continuation?: RasterTiledRelaxationContinuation;
};

/**
 * Creates the state for a tiled min-relaxation and returns it with its reset nodes.
 *
 * Schedule order: `resetNodes`, then the contributor's own value initialization and seed nodes (which
 * may mark tiles), then {@link createRasterTiledRelaxationNodes}, then an optional
 * {@link createRasterIterationFinalizeNode}.
 *
 * @param activateAllTiles When true every tile starts active; otherwise only seeded tiles.
 * @internal
 */
export function createRasterTiledRelaxation<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: RasterTiledRelaxationProps,
  activateAllTiles: boolean
): {relaxation: RasterTiledRelaxation<Parameters>; resetNodes: GPUCommandNode<Parameters>[]} {
  const tilesX = Math.ceil(props.width / RASTER_RELAXATION_TILE_SIZE);
  const tilesY = Math.ceil(props.height / RASTER_RELAXATION_TILE_SIZE);
  const tileCount = tilesX * tilesY;
  const {continuation} = props;
  if (continuation) {
    return createContinuedRelaxation(graph, props, continuation, activateAllTiles);
  }
  const state = createRasterIterationState(
    graph,
    `${props.id}-relax`,
    props.operation,
    tileCount * WORKGROUP_SIZE,
    props.id
  );
  const tileStamps = createContributorTransientView(
    graph,
    props.operation,
    props.id,
    `${props.id}-tile-stamps`,
    'uint32',
    tileCount
  );
  const resetNodes = [
    createRasterIterationResetNode<Parameters>(graph, {
      id: `${props.id}-relax-reset`,
      operation: props.operation,
      state
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-tile-reset`,
      operation: props.operation,
      variant: 'tile-reset',
      bindings: [{name: 'tileStamps', view: tileStamps, type: 'u32', access: 'read_write'}],
      invocationCount: tileCount,
      body: `tileStamps[tileStampsOffset + index] = ${activateAllTiles ? 1 : 0}u;`
    })
  ];
  return {relaxation: {state, tileStamps, tilesX, tileCount}, resetNodes};
}

/** WGSL that decides whether this execution resumes; shared by the nodes that read the decision. */
const CONTINUATION_RESUME_WGSL = `// Resume only a previous execution that stopped unconverged, unless the caller asked to restart.
  let canResume = state[stateOffset + ${GPU_RASTER_RELAXATION_RESTART_WORD}u] == 0u &&
    state[stateOffset + 3u] == 0u && state[stateOffset + 2u] > 0u;`;

function createContinuedRelaxation<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: RasterTiledRelaxationProps,
  continuation: RasterTiledRelaxationContinuation,
  activateAllTiles: boolean
): {relaxation: RasterTiledRelaxation<Parameters>; resetNodes: GPUCommandNode<Parameters>[]} {
  const tilesX = Math.ceil(props.width / RASTER_RELAXATION_TILE_SIZE);
  const tilesY = Math.ceil(props.height / RASTER_RELAXATION_TILE_SIZE);
  const tileCount = tilesX * tilesY;
  const {state: stateView} = continuation;
  if (stateView.length !== getGPURasterRelaxationContinuationStateLength(props)) {
    throw new Error(`${props.operation} "${props.id}": continuation state has the wrong length`);
  }
  const state = createRasterIterationState(
    graph,
    `${props.id}-relax`,
    props.operation,
    tileCount * WORKGROUP_SIZE,
    props.id,
    {status: stateView, limitWord: CONTINUATION_LIMIT_WORD}
  );
  // A sub-view of the caller's state; the relaxation kernels address it through `status`.
  const tileStamps = graph.createDataView(stateView.buffer, {
    format: 'uint32',
    length: tileCount,
    byteOffset: stateView.byteOffset + CONTINUATION_STAMP_OFFSET * 4
  });
  const restartGate = createWGSLDispatchGate(graph, {
    id: `${props.id}-restart`,
    operation: props.operation,
    shapes: [[tileCount], ...(continuation.restartShapes ?? [])]
  });
  const getRestartGate = (shapeIndex: number, nodeId: string) => {
    const {condition, extraResources} = restartGate.getGate(shapeIndex + 1);
    return {condition: {...condition, id: `${nodeId}-gate`}, extraResources};
  };
  const tileReset = getRestartGate(-1, `${props.id}-tile-reset`);
  const prerequisiteBindings: WGSLKernelBinding[] = continuation.prerequisite
    ? [{name: 'prerequisite', view: continuation.prerequisite, type: 'u32', access: 'read'}]
    : [];
  const skipWGSL = `let skipStage = ${continuation.prerequisite ? 'prerequisite[prerequisiteOffset + 3u] == 0u || ' : ''}${continuation.retainConverged ? `(state[stateOffset + 3u] != 0u && state[stateOffset + ${GPU_RASTER_RELAXATION_RESTART_WORD}u] == 0u)` : 'false'};`;
  const resetNodes = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-restart-gate`,
      operation: props.operation,
      variant: 'continuation-gate',
      bindings: [
        ...prerequisiteBindings,
        {name: 'state', view: stateView, type: 'u32', access: 'read'},
        restartGate.binding
      ],
      invocationCount: 1,
      body: `${CONTINUATION_RESUME_WGSL}
  ${skipWGSL}
  if (canResume || skipStage) {
    ${restartGate.disarmWGSL}
  } else {
    ${restartGate.armWGSL}
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-relax-reset`,
      operation: props.operation,
      variant: 'continuation-arm',
      bindings: [
        ...prerequisiteBindings,
        {name: 'state', view: stateView, type: 'u32', access: 'read_write'},
        state.gate.binding
      ],
      invocationCount: 1,
      body: `${CONTINUATION_RESUME_WGSL}
  ${skipWGSL}
  if (skipStage) { ${state.gate.disarmWGSL} return; }
  state[stateOffset + ${GPU_RASTER_RELAXATION_RESTART_WORD}u] = 0u;
  state[stateOffset + 0u] = 0u;
  state[stateOffset + 1u] = 1u;
  state[stateOffset + 3u] = 0u;
  if (!canResume) { state[stateOffset + 2u] = 0u; }
  state[stateOffset + ${CONTINUATION_LIMIT_WORD}u] = state[stateOffset + 2u] + ${props.maxIterations}u;
  ${state.gate.armWGSL}`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-tile-reset`,
      operation: props.operation,
      variant: 'tile-reset',
      bindings: [{name: 'tileStamps', view: tileStamps, type: 'u32', access: 'read_write'}],
      invocationCount: tileCount,
      body: `tileStamps[tileStampsOffset + index] = ${activateAllTiles ? 1 : 0}u;`,
      condition: tileReset.condition,
      extraResources: tileReset.extraResources
    })
  ];
  return {
    relaxation: {state, tileStamps, tilesX, tileCount, continuation: {getRestartGate}},
    resetNodes
  };
}

/** Returns WGSL `getRelaxationTile(cell) -> u32` for seed kernels that mark tiles. @internal */
export function getRasterRelaxationTileWGSL(width: number): string {
  const tilesX = Math.ceil(width / RASTER_RELAXATION_TILE_SIZE);
  return /* wgsl */ `
fn getRelaxationTile(cell: u32) -> u32 {
  return (cell / ${width}u / ${RASTER_RELAXATION_TILE_SIZE}u) * ${tilesX}u +
    (cell % ${width}u) / ${RASTER_RELAXATION_TILE_SIZE}u;
}`;
}

/**
 * Returns `maxIterations` pairs of GPU-gated tiled relaxation and gate nodes.
 *
 * Each relaxation computes, for every valid cell `c` of an active tile,
 * `value[c] = min(value[c], min over valid D8 neighbors n of getRelaxationCandidate(...))`.
 * A workgroup loads its 16x16 tile plus a one-cell halo of `values` and `auxiliary` into
 * workgroup memory and repeats the update up to `innerIterations` times with barriers, so a
 * global iteration propagates across a whole tile. Only the owning workgroup writes a cell, so the
 * result never depends on scheduling beyond float rounding of equal-cost alternatives, and values
 * may be any finite or infinite float (no unsigned bit-order trick). A cell or neighbor whose
 * auxiliary value is NaN is excluded.
 *
 * Unless `additiveEdgeCosts` is set, `declarations` must define
 * `fn getRelaxationCandidate(neighborValue: f32, neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32`
 * and may use the {@link getRasterGridWGSL} helpers and the `settings` binding.
 *
 * @internal
 */
export function createRasterTiledRelaxationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: RasterTiledRelaxationProps & {
    relaxation: RasterTiledRelaxation<Parameters>;
    /** Working values, bound as `atomic<u32>` float bits and updated in place. */
    values: GraphDataView<'float32'>;
    /** Per-cell 1-based round of the last decrease, or 0; unsupported with continuation. */
    settledAt?: GraphDataView<'uint32'>;
    /** Per-cell auxiliary input such as friction or elevation; NaN excludes a cell. */
    auxiliary: GraphDataView<'float32'>;
    /** Settings starting with the shared grid prefix. */
    settings: GraphDataView<'float32'>;
    /** Contributor WGSL defining `getRelaxationCandidate`. */
    declarations: string;
    /** In-tile repetitions per global iteration. Defaults to `2 * RASTER_RELAXATION_TILE_SIZE`. */
    innerIterations?: number;
    /**
     * Enables directional tile sweeps after this many plain tiled iterations. From then on the
     * unrolled iterations cycle through tiled, right, left, down, and up passes. A sweep pass runs
     * one workgroup per tile row (or column) that walks its tiles in order, so a front travels along
     * a whole row or column of tiles in one pass instead of one tile. That removes the
     * one-tile-per-iteration limit on long corridors and spirals, while open terrain (which
     * converges in the plain phase) pays nothing. Undefined disables sweeps.
     */
    sweepAfterIteration?: number;
    /** Iterations that share one gate node. Defaults to {@link RASTER_GATE_INTERVAL}. */
    gateInterval?: number;
    /**
     * Opts into the additive-edge contract: `declarations` define
     * `fn getRelaxationEdgeCost(neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32`
     * and `fn getRelaxationLimit() -> f32` instead of `getRelaxationCandidate`. The candidate of a
     * neighbor is then `neighborValue + edgeCost`, ignored when it exceeds the limit. Edge costs do
     * not depend on the evolving values, so each invocation evaluates its eight edge costs once per
     * tile visit instead of once per inner iteration (they can involve global settings reads,
     * `cos`, `cosh` and `sqrt`). The result is bit-identical to the equivalent
     * `getRelaxationCandidate`.
     */
    additiveEdgeCosts?: boolean;
  }
): GPUCommandNode<Parameters>[] {
  const {relaxation} = props;
  const {state} = relaxation;
  if (props.settledAt) {
    if (props.continuation || relaxation.continuation) {
      throw new Error(`${props.id} settledAt is not supported with continuation`);
    }
    validatePackedView(props.settledAt, ['uint32'], `${props.id} settledAt`);
    if (props.settledAt.length !== props.width * props.height) {
      throw new Error(`${props.id} settledAt must contain one value per cell`);
    }
  }
  const tileSize = RASTER_RELAXATION_TILE_SIZE;
  const haloSize = tileSize + 2;
  const tilesX = relaxation.tilesX;
  const innerIterations = props.innerIterations ?? 2 * tileSize;
  const additiveEdgeCosts = props.additiveEdgeCosts ?? false;
  const continued = Boolean(relaxation.continuation);
  // A continued relaxation addresses the stamps through the one binding of the caller's state.
  const getStampWGSL = (index: string) =>
    continued
      ? `status[statusOffset + ${CONTINUATION_STAMP_OFFSET}u + ${index}]`
      : `tileStamps[tileStampsOffset + ${index}]`;
  const declarations = `${getRasterGridWGSL(props)}
const TILE_SIZE: u32 = ${tileSize}u;
const HALO_SIZE: u32 = ${haloSize}u;
const HALO_CELL_COUNT: u32 = ${haloSize * haloSize}u;
const TILES_X: u32 = ${tilesX}u;
const TILES_Y: u32 = ${Math.ceil(props.height / tileSize)}u;
const TILE_COUNT: u32 = ${relaxation.tileCount}u;
const INNER_ITERATIONS: u32 = ${innerIterations}u;
var<workgroup> haloValues: array<f32, ${haloSize * haloSize}>;
var<workgroup> haloAuxiliary: array<f32, ${haloSize * haloSize}>;
var<workgroup> tileActive: u32;
var<workgroup> tileChanged: u32;
var<workgroup> tileChangedAtomic: atomic<u32>;
${continued ? 'var<workgroup> roundPhase: u32;' : ''}
${props.declarations}`;
  const relaxTileFunction = /* wgsl */ `
fn relaxTile(tileIndex: u32, localInvocationIndex: u32) {
  let tileColumn = tileIndex % TILES_X;
  let tileRow = tileIndex / TILES_X;
  // The gate adds each group's executed iterations to status[2]; roundSlot is this round's place in the group.
  let roundSlotValue = roundSlot[roundSlotOffset];
  let iteration = atomicLoad(&status[statusOffset + 2u]) + roundSlotValue;
  if (localInvocationIndex == 0u) {
    var anyActive = 0u;
    for (var rowDelta = -1; rowDelta <= 1; rowDelta++) {
      for (var columnDelta = -1; columnDelta <= 1; columnDelta++) {
        let neighborColumn = i32(tileColumn) + columnDelta;
        let neighborRow = i32(tileRow) + rowDelta;
        if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(TILES_X) || neighborRow >= i32(TILES_Y)) {
          continue;
        }
        let neighborTile = u32(neighborRow) * TILES_X + u32(neighborColumn);
        if (atomicLoad(&${getStampWGSL('neighborTile')}) >= iteration + 1u) {
          anyActive = 1u;
        }
      }
    }
    tileActive = anyActive;
    atomicStore(&tileChangedAtomic, 0u);
  }
  if (workgroupUniformLoad(&tileActive) == 0u) { return; }

  let originColumn = i32(tileColumn * TILE_SIZE) - 1;
  let originRow = i32(tileRow * TILE_SIZE) - 1;
  for (var haloIndex = localInvocationIndex; haloIndex < HALO_CELL_COUNT; haloIndex += ${WORKGROUP_SIZE}u) {
    let column = originColumn + i32(haloIndex % HALO_SIZE);
    let row = originRow + i32(haloIndex / HALO_SIZE);
    var haloValue = getInfinity();
    var haloAux = getQuietNaN();
    if (column >= 0 && row >= 0 && column < i32(GRID_WIDTH) && row < i32(GRID_HEIGHT)) {
      let cell = u32(row) * GRID_WIDTH + u32(column);
      haloValue = bitcast<f32>(atomicLoad(&values[valuesOffset + cell]));
      haloAux = auxiliary[auxiliaryOffset + cell];
    }
    haloValues[haloIndex] = haloValue;
    haloAuxiliary[haloIndex] = haloAux;
  }
  workgroupBarrier();

  let localColumn = localInvocationIndex % TILE_SIZE;
  let localRow = localInvocationIndex / TILE_SIZE;
  let column = tileColumn * TILE_SIZE + localColumn;
  let row = tileRow * TILE_SIZE + localRow;
  let haloCenter = (localRow + 1u) * HALO_SIZE + localColumn + 1u;
  let centerAuxiliary = haloAuxiliary[haloCenter];
  let originalValue = haloValues[haloCenter];
  let participates = column < GRID_WIDTH && row < GRID_HEIGHT && !isNaNValue(centerAuxiliary);

  ${
    additiveEdgeCosts
      ? `// Loop-invariant: the edge costs depend only on the auxiliary values, never on the relaxed values.
  var edgeCosts: array<f32, 8>;
  if (participates) {
    for (var direction = 0u; direction < 8u; direction++) {
      let neighborHalo = u32(i32(haloCenter) + getD8RowOffset(direction) * i32(HALO_SIZE) + getD8ColumnOffset(direction));
      let neighborAuxiliary = haloAuxiliary[neighborHalo];
      edgeCosts[direction] = select(getRelaxationEdgeCost(neighborAuxiliary, centerAuxiliary, row, direction), getInfinity(), isNaNValue(neighborAuxiliary));
    }
  }
  let relaxationLimit = getRelaxationLimit();`
      : ''
  }
  for (var inner = 0u; inner < INNER_ITERATIONS; inner++) {
    var best = haloValues[haloCenter];
    if (participates) {
      for (var direction = 0u; direction < 8u; direction++) {
        let neighborHalo = u32(i32(haloCenter) + getD8RowOffset(direction) * i32(HALO_SIZE) + getD8ColumnOffset(direction));
        ${
          additiveEdgeCosts
            ? `let candidate = haloValues[neighborHalo] + edgeCosts[direction];
        if (candidate <= relaxationLimit && candidate < best) { best = candidate; }`
            : `let neighborAuxiliary = haloAuxiliary[neighborHalo];
        if (isNaNValue(neighborAuxiliary)) { continue; }
        let candidate = getRelaxationCandidate(haloValues[neighborHalo], neighborAuxiliary, centerAuxiliary, row, direction);
        if (candidate < best) { best = candidate; }`
        }
      }
    }
    workgroupBarrier();
    if (best < haloValues[haloCenter]) {
      haloValues[haloCenter] = best;
      atomicStore(&tileChangedAtomic, 1u);
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) {
      tileChanged = atomicExchange(&tileChangedAtomic, 0u);
    }
    if (workgroupUniformLoad(&tileChanged) == 0u) { break; }
  }

  let finalValue = haloValues[haloCenter];
  if (participates && finalValue < originalValue) {
    atomicStore(&values[valuesOffset + row * GRID_WIDTH + column], bitcast<u32>(finalValue));${props.settledAt ? '\n    settledAt[settledAtOffset + row * GRID_WIDTH + column] = iteration + 1u;' : ''}
    atomicStore(&tileChangedAtomic, 1u);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u && atomicLoad(&tileChangedAtomic) != 0u) {
    atomicMax(&${getStampWGSL('tileIndex')}, iteration + 2u);
    atomicOr(&status[statusOffset], 1u << roundSlotValue);
  }
}`;
  const tiledBody = /* wgsl */ `
  if (workgroupIndex >= TILE_COUNT) { return; }
  relaxTile(workgroupIndex, localInvocationIndex);`;
  const getSweepBody = (vertical: boolean, reverse: boolean) => /* wgsl */ `
  let lineCount = ${vertical ? relaxation.tilesX : Math.ceil(props.height / tileSize)}u;
  let stepCount = ${vertical ? Math.ceil(props.height / tileSize) : relaxation.tilesX}u;
  if (workgroupIndex >= lineCount) { return; }
  for (var step = 0u; step < stepCount; step++) {
    let position = ${reverse ? 'stepCount - 1u - step' : 'step'};
    let tileIndex = ${vertical ? 'position * TILES_X + workgroupIndex' : 'workgroupIndex * TILES_X + position'};
    relaxTile(tileIndex, localInvocationIndex);
    // Publish this tile's cells to the next tile's halo load within the workgroup.
    storageBarrier();
    workgroupBarrier();
  }`;
  // One kernel for every pass: the pass of a round comes from the cumulative iteration count in the
  // GPU state, so a resumed execution continues the schedule instead of restarting it.
  const sweepAfter = props.sweepAfterIteration;
  const continuedBody = /* wgsl */ `
  if (localInvocationIndex == 0u) {
    let iteration = atomicLoad(&status[statusOffset + 2u]) + roundSlot[roundSlotOffset];
    roundPhase = ${
      sweepAfter === undefined
        ? '0u'
        : `select(0u, (iteration - ${sweepAfter}u) % 5u, iteration >= ${sweepAfter}u)`
    };
  }
  // 0 tiled; 1 right, 2 left, 3 down, 4 up.
  let phase = workgroupUniformLoad(&roundPhase);
  if (phase == 0u) {
    if (workgroupIndex >= TILE_COUNT) { return; }
    relaxTile(workgroupIndex, localInvocationIndex);
    return;
  }
  let vertical = phase >= 3u;
  let reverse = (phase & 1u) == 0u;
  let lineCount = select(TILES_Y, TILES_X, vertical);
  let stepCount = select(TILES_X, TILES_Y, vertical);
  if (workgroupIndex >= lineCount) { return; }
  for (var step = 0u; step < stepCount; step++) {
    let position = select(step, stepCount - 1u - step, reverse);
    let tileIndex = select(workgroupIndex * TILES_X + position, position * TILES_X + workgroupIndex, vertical);
    relaxTile(tileIndex, localInvocationIndex);
    // Publish this tile's cells to the next tile's halo load within the workgroup.
    storageBarrier();
    workgroupBarrier();
  }`;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const settledAtBindings: WGSLKernelBinding[] = props.settledAt
    ? [{name: 'settledAt', view: props.settledAt, type: 'u32', access: 'read_write'}]
    : [];
  if (props.settledAt) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-settled-at-reset`,
        operation: props.operation,
        variant: 'settled-at-reset',
        bindings: settledAtBindings,
        invocationCount: props.width * props.height,
        body: 'settledAt[settledAtOffset + index] = 0u;'
      })
    );
  }
  const sweepPasses = [
    {variant: 'sweep-right', body: getSweepBody(false, false)},
    {variant: 'sweep-left', body: getSweepBody(false, true)},
    {variant: 'sweep-down', body: getSweepBody(true, false)},
    {variant: 'sweep-up', body: getSweepBody(true, true)}
  ];
  nodes.push(
    ...createRasterGatedLoopNodes<Parameters>(graph, {
      id: `${props.id}-relax`,
      gateId: `${props.id}-relax-gate`,
      operation: props.operation,
      state,
      maxIterations: props.maxIterations,
      groupSize: props.gateInterval,
      createRound: ({nodeId, iteration, slotBinding, condition, extraResources}) => {
        // After the plain phase the schedule cycles tiled, right, left, down, up.
        const cycleIndex =
          props.sweepAfterIteration === undefined || iteration < props.sweepAfterIteration
            ? 0
            : (iteration - props.sweepAfterIteration) % (sweepPasses.length + 1);
        const pass = continued
          ? {variant: 'continued-relax', body: continuedBody}
          : cycleIndex === 0
            ? {variant: 'tiled-relax', body: tiledBody}
            : sweepPasses[cycleIndex - 1];
        return createWGSLKernelNode<Parameters>(graph, {
          id: nodeId,
          operation: props.operation,
          variant: pass.variant,
          bindings: [
            {name: 'values', view: props.values, type: 'atomic<u32>', access: 'read_write'},
            {name: 'auxiliary', view: props.auxiliary, type: 'f32', access: 'read'},
            {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
            ...(continued
              ? []
              : [
                  {
                    name: 'tileStamps',
                    view: relaxation.tileStamps,
                    type: 'atomic<u32>' as const,
                    access: 'read_write' as const
                  }
                ]),
            {name: 'status', view: state.status, type: 'atomic<u32>', access: 'read_write'},
            slotBinding,
            ...settledAtBindings
          ],
          invocationCount: state.invocationCount,
          workgroupSize: WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${declarations}\n${relaxTileFunction}`,
          body: pass.body,
          condition,
          extraResources
        });
      }
    })
  );
  return nodes;
}
