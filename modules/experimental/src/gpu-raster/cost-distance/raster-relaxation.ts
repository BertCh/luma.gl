// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  getBoundedDispatchLayout,
  type GPUBoundedDispatchLayout,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUTerrainCellSizeMode} from '../../gpu-terrain/terrain-analysis/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {createContributorTransientView, getRasterGridWGSL} from './raster-grid-utils';

/** Largest accepted compile-time iteration count for GPU-gated raster iterations. */
export const GPU_RASTER_MAXIMUM_ITERATIONS = 1024;

/** Edge length of one square relaxation tile; one 256-invocation workgroup owns one tile. @internal */
export const RASTER_RELAXATION_TILE_SIZE = 16;

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
export type RasterIterationState = {
  status: GraphDataView<'uint32'>;
  dispatch: GraphDataView<'uint32'>;
  /** Bounded dispatch layout of the gated kernel. */
  layout: GPUBoundedDispatchLayout;
  /** Linear invocation count of the gated kernel. */
  invocationCount: number;
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
  contributorId: string = id
): RasterIterationState {
  return {
    status: createContributorTransientView(
      graph,
      operation,
      contributorId,
      `${id}-status`,
      'uint32',
      4
    ),
    dispatch: createContributorTransientView(
      graph,
      operation,
      contributorId,
      `${id}-dispatch`,
      'uint32',
      3,
      Buffer.STORAGE | Buffer.INDIRECT
    ),
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
  state: RasterIterationState,
  nodeId: string
): {condition: GPUCommandGraphNodeCondition<Parameters>; extraResources: GraphBufferUse[]} {
  return {
    condition: {
      id: `${nodeId}-gate`,
      source: 'gpu',
      mode: 'indirect',
      buffer: state.dispatch.buffer,
      byteOffset: state.dispatch.byteOffset
    },
    extraResources: [{buffer: state.dispatch, usage: 'indirect'}]
  };
}

/** Returns a one-thread node that enables the loop and arms the indirect dispatch. @internal */
export function createRasterIterationResetNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; state: RasterIterationState}
): GPUCommandNode<Parameters> {
  const {x, y, z} = props.state.layout;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'iteration-reset',
    bindings: [
      {name: 'status', view: props.state.status, type: 'u32', access: 'read_write'},
      {name: 'dispatch', view: props.state.dispatch, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    body: `status[statusOffset + 0u] = 0u;
  status[statusOffset + 1u] = 1u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;
  dispatch[dispatchOffset + 0u] = ${x}u;
  dispatch[dispatchOffset + 1u] = ${y}u;
  dispatch[dispatchOffset + 2u] = ${z}u;`
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
  props: {id: string; operation: string; state: RasterIterationState; maxIterations: number}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'iteration-gate',
    bindings: [
      {name: 'status', view: props.state.status, type: 'u32', access: 'read_write'},
      {name: 'dispatch', view: props.state.dispatch, type: 'u32', access: 'read_write'}
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
    dispatch[dispatchOffset] = 0u;
    return;
  }
  if (iterationCount >= ${props.maxIterations}u) {
    status[statusOffset + 1u] = 0u;
    dispatch[dispatchOffset] = 0u;
  }`
  });
}

/** Default number of unrolled iterations that share one gate node. @internal */
export const RASTER_GATE_INTERVAL = 4;

/**
 * Returns a one-thread gate that closes `groupSize` consecutive gated iterations.
 *
 * Iteration `k` of the group records a change with `atomicOr(&status[0], 1u << k)`. The gate counts
 * the iterations up to and including the first unchanged one (or the whole group), publishes
 * convergence, and zeroes the indirect dispatch once converged or `maxIterations` ran. Iterations
 * of a group that follow a converged one only re-run an idempotent fixpoint step, so results and
 * counts match one gate per iteration while the graph needs one gate node per group.
 *
 * @internal
 */
export function createRasterIterationGroupGateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    state: RasterIterationState;
    maxIterations: number;
    groupSize: number;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'iteration-group-gate',
    bindings: [
      {name: 'status', view: props.state.status, type: 'u32', access: 'read_write'},
      {name: 'dispatch', view: props.state.dispatch, type: 'u32', access: 'read_write'}
    ],
    invocationCount: 1,
    body: `if (status[statusOffset + 1u] == 0u) { return; }
  let changedBits = status[statusOffset + 0u];
  status[statusOffset + 0u] = 0u;
  var executed = ${props.groupSize}u;
  var converged = false;
  for (var round = 0u; round < ${props.groupSize}u; round++) {
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
    dispatch[dispatchOffset] = 0u;
    return;
  }
  if (iterationCount >= ${props.maxIterations}u) {
    status[statusOffset + 1u] = 0u;
    dispatch[dispatchOffset] = 0u;
  }`
  });
}

/** Per-iteration context passed to the node factory of {@link createRasterGatedLoopNodes}. @internal */
export type RasterGatedRound<Parameters> = {
  /** Unique node ID of this iteration's kernel. */
  nodeId: string;
  /** Zero-based unrolled iteration index. */
  iteration: number;
  /** Position inside the gate group; also the status bit this iteration owns. */
  groupOffset: number;
  /** WGSL statement that records "something changed" for this iteration. */
  markChangedWGSL: string;
  /** Indirect GPU condition gating the kernel. */
  condition: GPUCommandGraphNodeCondition<Parameters>;
  /** Resource uses the kernel must declare. */
  extraResources: GraphBufferUse[];
};

/**
 * Unrolls `maxIterations` gated iterations with one gate node per `groupSize` iterations.
 *
 * `createRound` builds each iteration's kernel and must call `markChangedWGSL` whenever the
 * iteration changed anything. The node count is `maxIterations + ceil(maxIterations / groupSize)`
 * instead of `2 * maxIterations`.
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
    state: RasterIterationState;
    maxIterations: number;
    groupSize?: number;
    createRound: (round: RasterGatedRound<Parameters>) => GPUCommandNode<Parameters>;
  }
): GPUCommandNode<Parameters>[] {
  const groupSize = Math.min(props.groupSize ?? RASTER_GATE_INTERVAL, 32);
  const nodes: GPUCommandNode<Parameters>[] = [];
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
        props.createRound({
          nodeId,
          iteration,
          groupOffset,
          markChangedWGSL: `atomicOr(&status[statusOffset], ${1 << groupOffset}u);`,
          condition,
          extraResources
        })
      );
    }
    nodes.push(
      createRasterIterationGroupGateNode<Parameters>(graph, {
        id: `${props.gateId ?? `${props.id}-gate`}-${start}`,
        operation: props.operation,
        state: props.state,
        maxIterations: props.maxIterations,
        groupSize: size
      })
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
    state: RasterIterationState;
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

/** Graph-owned state of one tiled min-relaxation. @internal */
export type RasterTiledRelaxation = {
  /** Gated-loop state; the gated kernel runs one workgroup per tile. */
  state: RasterIterationState;
  /**
   * Per-tile activity stamps. A tile is processed in iteration `i` when it or one of its eight
   * neighbor tiles holds a stamp `>= i + 1`. Seeds mark a tile with `atomicMax(&stamp, 1u)`.
   */
  tileStamps: GraphDataView<'uint32'>;
  /** Tiles per row. */
  tilesX: number;
  /** Tile count. */
  tileCount: number;
};

/** Properties for {@link createRasterTiledRelaxation}. @internal */
export type RasterTiledRelaxationProps = {
  /** Prefix for relaxation node and transient IDs. */
  id: string;
  /** Operation name for workload estimates. */
  operation: string;
  width: number;
  height: number;
  cellSizeMode: GPUTerrainCellSizeMode;
  maxIterations: number;
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
): {relaxation: RasterTiledRelaxation; resetNodes: GPUCommandNode<Parameters>[]} {
  const tilesX = Math.ceil(props.width / RASTER_RELAXATION_TILE_SIZE);
  const tilesY = Math.ceil(props.height / RASTER_RELAXATION_TILE_SIZE);
  const tileCount = tilesX * tilesY;
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
    relaxation: RasterTiledRelaxation;
    /** Working values, bound as `atomic<u32>` float bits and updated in place. */
    values: GraphDataView<'float32'>;
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
  const tileSize = RASTER_RELAXATION_TILE_SIZE;
  const haloSize = tileSize + 2;
  const tilesX = relaxation.tilesX;
  const innerIterations = props.innerIterations ?? 2 * tileSize;
  const additiveEdgeCosts = props.additiveEdgeCosts ?? false;
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
${props.declarations}`;
  const relaxTileFunction = /* wgsl */ `
fn relaxTile(tileIndex: u32, localInvocationIndex: u32) {
  let tileColumn = tileIndex % TILES_X;
  let tileRow = tileIndex / TILES_X;
  // The gate adds each group's executed iterations to status[2]; GROUP_OFFSET is this round's place in the group.
  let iteration = atomicLoad(&status[statusOffset + 2u]) + GROUP_OFFSET;
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
        if (atomicLoad(&tileStamps[tileStampsOffset + neighborTile]) >= iteration + 1u) {
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
    atomicStore(&values[valuesOffset + row * GRID_WIDTH + column], bitcast<u32>(finalValue));
    atomicStore(&tileChangedAtomic, 1u);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u && atomicLoad(&tileChangedAtomic) != 0u) {
    atomicMax(&tileStamps[tileStampsOffset + tileIndex], iteration + 2u);
    atomicOr(&status[statusOffset], 1u << GROUP_OFFSET);
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
  const nodes: GPUCommandNode<Parameters>[] = [];
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
      createRound: ({nodeId, iteration, groupOffset, condition, extraResources}) => {
        // After the plain phase the schedule cycles tiled, right, left, down, up.
        const cycleIndex =
          props.sweepAfterIteration === undefined || iteration < props.sweepAfterIteration
            ? 0
            : (iteration - props.sweepAfterIteration) % (sweepPasses.length + 1);
        const pass =
          cycleIndex === 0
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
            {
              name: 'tileStamps',
              view: relaxation.tileStamps,
              type: 'atomic<u32>',
              access: 'read_write'
            },
            {name: 'status', view: state.status, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: state.invocationCount,
          workgroupSize: WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `const GROUP_OFFSET: u32 = ${groupOffset}u;\n${declarations}\n${relaxTileFunction}`,
          body: pass.body,
          condition,
          extraResources
        });
      }
    })
  );
  return nodes;
}
