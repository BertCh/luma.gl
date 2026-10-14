// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getBoundedDispatchLayout,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GraphBufferUse
} from '@luma.gl/gpgpu/gpu-core';
import {
  createRasterIterationState,
  RASTER_GATE_INTERVAL,
  type RasterIterationState
} from '../../gpu-raster/cost-distance/raster-relaxation';
import {createContributorTransientView} from '../../gpu-raster/cost-distance/raster-grid-utils';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLDispatchGate,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';

/** Workgroup size of every spill fill kernel; gated kernels must match their gate shape. @internal */
export const SPILL_WORKGROUP_SIZE = 256;

/** `[invocationCount, workgroupSize]` of one gated kernel shape. @internal */
export type SpillGateShape = readonly [invocationCount: number, workgroupSize: number];

/** Number of bits of `value`: `floor(log2(value)) + 1` for `value >= 1`, 0 for 0. @internal */
export function getSpillBitLength(value: number): number {
  return 32 - Math.clz32(value);
}

/**
 * Unrolled rounds of a pointer-jumping loop over a list or forest path of at most `length`
 * elements: `ceil(log2(length))` doublings plus one that confirms that nothing changed, rounded up
 * to the gate group size so that ping-pong buffers end in the first pair. @internal
 */
export function getSpillRoundCount(length: number): number {
  const doublings = getSpillBitLength(Math.max(length, 2) - 1);
  return Math.ceil((doublings + 1) / RASTER_GATE_INTERVAL) * RASTER_GATE_INTERVAL;
}

/** Condition and resource uses that gate one kernel on one slot of a dispatch gate. @internal */
export type SpillGateUse<Parameters> = {
  condition: GPUCommandGraphNodeCondition<Parameters>;
  extraResources: GraphBufferUse[];
};

/** Returns the gate use of slot `slot` of `gate` for the node `nodeId`. @internal */
export function getSpillGateUse<Parameters>(
  gate: WGSLDispatchGate<Parameters>,
  slot: number,
  nodeId: string
): SpillGateUse<Parameters> {
  const {condition, extraResources} = gate.getGate(slot);
  return {condition: {...condition, id: `${nodeId}-gate`}, extraResources};
}

/**
 * Creates the state of a gated loop whose rounds contain kernels of several shapes. Slot `i` of the
 * dispatch gate holds `shapes[i]`; `status` has `statusLength` words (at least four, see
 * {@link RasterIterationState}). The loop helpers of `raster-relaxation` gate their rounds on slot 0.
 *
 * @internal
 */
export function createSpillLoopState<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    contributorId: string;
    shapes: readonly SpillGateShape[];
    statusLength?: number;
  }
): RasterIterationState<Parameters> {
  if (props.shapes.length === 1 && (props.statusLength ?? 4) === 4) {
    return createRasterIterationState(
      graph,
      props.id,
      props.operation,
      props.shapes[0][0],
      props.contributorId
    );
  }
  const gate = createWGSLDispatchGate(graph, {
    id: `${props.id}-dispatch`,
    operation: props.operation,
    shapes: props.shapes
  });
  return {
    status: createContributorTransientView(
      graph,
      props.operation,
      props.contributorId,
      `${props.id}-status`,
      'uint32',
      props.statusLength ?? 4
    ),
    dispatch: gate.view,
    gate,
    layout: getBoundedDispatchLayout(
      props.operation,
      Math.max(props.shapes[0][0], 1),
      props.shapes[0][1],
      graph.device.limits.maxComputeWorkgroupsPerDimension
    ),
    invocationCount: props.shapes[0][0]
  };
}

/**
 * Returns a one-thread node that initializes `state` for a loop that may be switched off: when the
 * WGSL boolean expression `enabledExpression` is true it writes `[0, 1, 0, 0, ...]` and arms the
 * gate, otherwise it writes `[0, 0, 0, 0, ...]` and disarms every slot, so a disabled loop launches
 * nothing even though the dispatch buffer starts with undefined contents.
 *
 * @param enabledBindings Read bindings that `enabledExpression` uses.
 * @internal
 */
export function createSpillResetNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    state: RasterIterationState<Parameters>;
    statusLength?: number;
    enabledBindings: readonly WGSLKernelBinding[];
    enabledExpression: string;
  }
): GPUCommandNode<Parameters> {
  const words: string[] = [];
  for (let word = 0; word < (props.statusLength ?? 4); word++) {
    words.push(
      word === 1
        ? 'status[statusOffset + 1u] = select(0u, 1u, enabled);'
        : `status[statusOffset + ${word}u] = 0u;`
    );
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'spill-reset',
    bindings: [
      ...props.enabledBindings,
      {name: 'status', view: props.state.status, type: 'u32', access: 'read_write'},
      props.state.gate.binding
    ],
    invocationCount: 1,
    body: `let enabled = ${props.enabledExpression};
  ${words.join('\n  ')}
  if (enabled) {
    ${props.state.gate.armWGSL}
  } else {
    ${props.state.gate.disarmWGSL}
  }`
  });
}

/**
 * Order-preserving `u32` keys of `f32` values and the shared grid helpers of the spill fill WGSL.
 * `getSpillKey(a) < getSpillKey(b)` exactly when `a < b` for finite values (`-0` sorts below `+0`).
 * Defines `SPILL_*` constants for a `width x height` grid and `getSpillNeighbor`, whose first four
 * directions (E, SE, S, SW) visit every unordered neighbor pair once. @internal
 */
export function getSpillCommonWGSL(width: number, height: number): string {
  return /* wgsl */ `
const SPILL_WIDTH: u32 = ${width}u;
const SPILL_HEIGHT: u32 = ${height}u;
const SPILL_NONE: u32 = 0xffffffffu;
const SPILL_BOUNDARY_FLAG: u32 = 0x80000000u;
const SPILL_ID_MASK: u32 = 0x7fffffffu;
const SPILL_INFINITE_KEY: u32 = 0xffffffffu;

fn getSpillNeighbor(cell: u32, direction: u32) -> u32 {
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let column = i32(cell % SPILL_WIDTH) + columnOffsets[direction];
  let row = i32(cell / SPILL_WIDTH) + rowOffsets[direction];
  if (column < 0 || row < 0 || column >= i32(SPILL_WIDTH) || row >= i32(SPILL_HEIGHT)) {
    return SPILL_NONE;
  }
  return u32(row) * SPILL_WIDTH + u32(column);
}

fn isSpillFinite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
}

fn getSpillKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, bits >= 0x80000000u);
}

fn getSpillValueBits(key: u32) -> u32 {
  return select(~key, key & 0x7fffffffu, key >= 0x80000000u);
}`;
}
