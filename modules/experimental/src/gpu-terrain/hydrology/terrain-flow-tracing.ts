// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  createRasterIterationFinalizeNode,
  createRasterIterationGateNode,
  createRasterIterationResetNode,
  createRasterIterationState,
  getRasterIterationCondition
} from '../../gpu-raster/cost-distance/raster-relaxation';

/** Receiver index, label, and drainage value meaning "none" in the D8 tracing recipes. @internal */
export const TERRAIN_FLOW_TRACING_NONE = 0xffffffff;

/**
 * WGSL constants and helpers shared by the D8 tracing kernels: `GRID_WIDTH`, `GRID_HEIGHT`,
 * `NO_CELL`, and `getFlowReceiver(cell, code)`.
 *
 * A code has a receiver only when it is a single bit in `1..128` (ESRI D8: `1 << direction`, east
 * first, clockwise) and the neighbor lies inside the grid. Terminal (0), invalid
 * (`0xffffffff`), multi-bit, and out-of-grid codes are all terminal. The helpers do not read the
 * grid `settings`, so kernels that use them need no settings binding.
 *
 * @internal
 */
export function getTerrainFlowTracingWGSL(width: number, height: number): string {
  return /* wgsl */ `
const GRID_WIDTH: u32 = ${width}u;
const GRID_HEIGHT: u32 = ${height}u;
const NO_CELL: u32 = 0xffffffffu;

fn getFlowReceiver(cell: u32, code: u32) -> u32 {
  if (code == 0u || code > 128u || (code & (code - 1u)) != 0u) { return NO_CELL; }
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let direction = firstTrailingBit(code);
  let column = i32(cell % GRID_WIDTH) + columnOffsets[direction];
  let row = i32(cell / GRID_WIDTH) + rowOffsets[direction];
  if (column < 0 || row < 0 || column >= i32(GRID_WIDTH) || row >= i32(GRID_HEIGHT)) {
    return NO_CELL;
  }
  return u32(row) * GRID_WIDTH + u32(column);
}`;
}

/**
 * Which cells stop pointer jumping.
 *
 * - `'none'`: only cells without a receiver stop, so the fixpoint is the terminal cell.
 * - `'streams'`: valid cells (direction code other than `0xffffffff`) with a non-zero `streams` word.
 * - `'markers'`: cells whose `markers` word is not `0xffffffff`.
 *
 * @internal
 */
export type TerrainFlowStopRule =
  | {kind: 'none'}
  | {kind: 'streams'; streams: GraphDataView<'uint32'>}
  | {kind: 'markers'; markers: GraphDataView<'uint32'>};

/** Returns the WGSL `isStopCell(cell, code)` function and its bindings for a stop rule. @internal */
function getStopRuleSource(stops: TerrainFlowStopRule): {
  binding?: WGSLKernelBinding;
  declarations: string;
} {
  switch (stops.kind) {
    case 'none':
      return {declarations: 'fn isStopCell(cell: u32, code: u32) -> bool { return false; }'};
    case 'streams':
      return {
        binding: {name: 'stops', view: stops.streams, type: 'u32', access: 'read'},
        declarations: `fn isStopCell(cell: u32, code: u32) -> bool {
  return code != NO_CELL && stops[stopsOffset + cell] != 0u;
}`
      };
    case 'markers':
      return {
        binding: {name: 'stops', view: stops.markers, type: 'u32', access: 'read'},
        declarations: `fn isStopCell(cell: u32, code: u32) -> bool {
  return stops[stopsOffset + cell] != NO_CELL;
}`
      };
  }
}

/**
 * Returns the pointer-jumping nodes: initialization, loop reset, `maxIterations` gated doubling
 * rounds, and the optional converged-flag publication.
 *
 * `pointer[c]` starts as `c` for stop cells and cells without a receiver, otherwise as the D8
 * receiver. Each round sets `pointer[c] = pointer[pointer[c]]` in place. A read of a partially
 * updated pointer is still a cell on the same downstream path, so the fixpoint (the first stop cell
 * or the terminal cell downstream) is unique and independent of the schedule. Path lengths up to
 * `2^k` need `k` doubling rounds plus one round that confirms nothing changed. Receiver cycles
 * (possible only in caller-supplied directions) never reach a fixpoint and leave `converged` at 0.
 *
 * @internal
 */
export function createTerrainFlowPointerNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    recipeId: string;
    operation: string;
    width: number;
    height: number;
    maxIterations: number;
    flowDirections: GraphDataView<'uint32'>;
    pointer: GraphDataView<'uint32'>;
    stops: TerrainFlowStopRule;
    converged?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const cellCount = props.width * props.height;
  const state = createRasterIterationState(
    graph,
    `${props.id}-pointer`,
    props.operation,
    cellCount,
    props.recipeId
  );
  const stopRule = getStopRuleSource(props.stops);
  const tracing = getTerrainFlowTracingWGSL(props.width, props.height);
  const initBindings: WGSLKernelBinding[] = [
    {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
    ...(stopRule.binding ? [stopRule.binding] : []),
    {name: 'pointer', view: props.pointer, type: 'u32', access: 'read_write'}
  ];
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-pointer-init`,
      operation: props.operation,
      variant: 'pointer-init',
      bindings: initBindings,
      invocationCount: cellCount,
      declarations: `${tracing}\n${stopRule.declarations}`,
      body: `let code = flowDirections[flowDirectionsOffset + index];
  var jumpTarget = index;
  if (!isStopCell(index, code)) {
    let receiver = getFlowReceiver(index, code);
    if (receiver != NO_CELL) { jumpTarget = receiver; }
  }
  pointer[pointerOffset + index] = jumpTarget;`
    }),
    createRasterIterationResetNode<Parameters>(graph, {
      id: `${props.id}-pointer-reset`,
      operation: props.operation,
      state
    })
  ];
  const roundBindings: WGSLKernelBinding[] = [
    {name: 'pointer', view: props.pointer, type: 'atomic<u32>', access: 'read_write'},
    {name: 'status', view: state.status, type: 'atomic<u32>', access: 'read_write'}
  ];
  const roundBody = `let current = atomicLoad(&pointer[pointerOffset + index]);
  let next = atomicLoad(&pointer[pointerOffset + current]);
  if (next != current) {
    atomicStore(&pointer[pointerOffset + index], next);
    atomicStore(&status[statusOffset], 1u);
  }`;
  for (let iteration = 0; iteration < props.maxIterations; iteration++) {
    const nodeId = `${props.id}-pointer-round-${iteration}`;
    const {condition, extraResources} = getRasterIterationCondition<Parameters>(state, nodeId);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: nodeId,
        operation: props.operation,
        variant: 'pointer-round',
        bindings: roundBindings,
        invocationCount: cellCount,
        body: roundBody,
        condition,
        extraResources
      }),
      createRasterIterationGateNode<Parameters>(graph, {
        id: `${props.id}-pointer-gate-${iteration}`,
        operation: props.operation,
        state,
        maxIterations: props.maxIterations
      })
    );
  }
  if (props.converged) {
    nodes.push(
      createRasterIterationFinalizeNode<Parameters>(graph, {
        id: `${props.id}-pointer-finalize`,
        operation: props.operation,
        state,
        converged: props.converged
      })
    );
  }
  return nodes;
}

/** Throws unless `view` is a packed view of `format` with one value per cell. @internal */
export function validateTerrainTracingCellView(
  id: string,
  name: string,
  view: GraphDataView,
  format: 'uint32' | 'float32',
  cellCount: number
): void {
  validatePackedView(view, [format], `${id} ${name}`);
  if (view.length !== cellCount) {
    throw new Error(`${id} ${name} must contain one value per cell`);
  }
}

/** Throws unless `view` is a packed one-row `uint32` view. @internal */
export function validateTerrainTracingConvergedView(
  id: string,
  view: GraphDataView<'uint32'> | undefined
): void {
  if (!view) {
    return;
  }
  validatePackedUint32View(view, `${id} converged`);
  if (view.length !== 1) {
    throw new Error(`${id} converged must contain one row`);
  }
}
