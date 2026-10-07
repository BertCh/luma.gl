// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandGraphNodeCondition,
  GPUCommandNode,
  GraphBufferUse,
  GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  LINE_ENDPOINT_IMPORTANCE_BITS,
  LINE_SIMPLIFICATION_METRIC_WGSL
} from './line-simplification-kernels';

const OPERATION = 'GPULineSimplification';

/** Graph-owned scratch columns of the Visvalingam-Whyatt loop. @internal */
export type VisvalingamScratch = {
  /** Previous surviving row of each undecided row; the row itself once decided. */
  previousRows: GraphDataView<'uint32'>;
  /** Next surviving row of each undecided row; the row itself once decided. */
  nextRows: GraphDataView<'uint32'>;
  /** f32 bits of the triangle area of each undecided row in the current round. */
  areaKeys: GraphDataView<'uint32'>;
  /** f32 bits of the largest effective area among removed neighbors (atomic max). */
  floorKeys: GraphDataView<'uint32'>;
  /** `1` for rows removed in the current round until they are unlinked. */
  pendingFlags: GraphDataView<'uint32'>;
};

/** Gate shared by every node of one round. @internal */
export type VisvalingamRoundGate<Parameters> = {
  condition: GPUCommandGraphNodeCondition<Parameters>;
  extraResources: GraphBufferUse[];
};

const IS_UNDECIDED_WGSL = /* wgsl */ `let previousRow = previousRows[previousRowsOffset + index];
  let nextRow = nextRows[nextRowsOffset + index];
  if (!(previousRow < index && index < nextRow)) {
    return;
  }`;

/**
 * Initializes the linked list: endpoints get `+Infinity`, interior rows link to their neighbors,
 * and rows outside every line are decided with importance 0.
 *
 * @internal
 */
export function createVisvalingamInitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    trackOffsets: GraphDataView<'uint32'>;
    scratch: VisvalingamScratch;
    importance: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const lineCount = props.trackOffsets.length - 1;
  const {scratch} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'visvalingam-init',
    bindings: [
      {name: 'trackOffsets', view: props.trackOffsets, type: 'u32', access: 'read'},
      {name: 'previousRows', view: scratch.previousRows, type: 'u32', access: 'read_write'},
      {name: 'nextRows', view: scratch.nextRows, type: 'u32', access: 'read_write'},
      {name: 'areaKeys', view: scratch.areaKeys, type: 'u32', access: 'read_write'},
      {name: 'floorKeys', view: scratch.floorKeys, type: 'u32', access: 'read_write'},
      {name: 'pendingFlags', view: scratch.pendingFlags, type: 'u32', access: 'read_write'},
      {name: 'importance', view: props.importance, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.importance.length,
    declarations: `const LINE_COUNT: u32 = ${lineCount}u;
const ENDPOINT_IMPORTANCE: u32 = ${LINE_ENDPOINT_IMPORTANCE_BITS}u;`,
    body: /* wgsl */ `areaKeys[areaKeysOffset + index] = 0u;
  floorKeys[floorKeysOffset + index] = 0u;
  pendingFlags[pendingFlagsOffset + index] = 0u;
  previousRows[previousRowsOffset + index] = index;
  nextRows[nextRowsOffset + index] = index;
  importance[importanceOffset + index] = 0u;
  let firstRow = trackOffsets[trackOffsetsOffset];
  let endRow = trackOffsets[trackOffsetsOffset + LINE_COUNT];
  if (index < firstRow || index >= endRow) {
    return;
  }
  // Last line whose start is <= index (empty lines share a start with the next line).
  var low = 0u;
  var high = LINE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (trackOffsets[trackOffsetsOffset + middle] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let line = low - 1u;
  let lineStart = trackOffsets[trackOffsetsOffset + line];
  let lineLast = trackOffsets[trackOffsetsOffset + line + 1u] - 1u;
  if (index == lineStart || index >= lineLast) {
    importance[importanceOffset + index] = ENDPOINT_IMPORTANCE;
    return;
  }
  previousRows[previousRowsOffset + index] = index - 1u;
  nextRows[nextRowsOffset + index] = index + 1u;`
  });
}

/**
 * Returns the three gated nodes of one Visvalingam-Whyatt round:
 *
 * 1. `area`: every surviving interior row computes the area of the triangle formed with its
 *    surviving neighbors, `0.5 * |cross|`, from f32 `+`, `-`, and `*` only (so it is reproducible
 *    bit for bit);
 * 2. `mark`: a row whose `(area, row)` is smaller than that of every surviving row within
 *    `neighborhoodRadius` steps on both sides (line endpoints stop the walk) is removed with
 *    effective area `max(area, floor)`; other rows raise the "rows remain" flag. The order is
 *    total, so two adjacent rows are never removed together;
 * 3. `unlink`: removed rows are unlinked and raise the floor of both neighbors to their effective
 *    area with `atomicMax`.
 *
 * @internal
 */
export function createVisvalingamRoundNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    neighborhoodRadius: number;
    positions: GraphDataView<'float32x2'>;
    scratch: VisvalingamScratch;
    importance: GraphDataView<'float32'>;
    status: GraphDataView<'uint32'>;
    gate: VisvalingamRoundGate<Parameters>;
  }
): GPUCommandNode<Parameters>[] {
  const {scratch, gate, neighborhoodRadius} = props;
  const gated = {
    operation: OPERATION,
    invocationCount: props.importance.length,
    condition: gate.condition,
    extraResources: gate.extraResources
  };
  const links = (access: 'read' | 'read_write'): WGSLKernelBinding[] => [
    {name: 'previousRows', view: scratch.previousRows, type: 'u32', access},
    {name: 'nextRows', view: scratch.nextRows, type: 'u32', access}
  ];
  const floorKeys: WGSLKernelBinding = {
    name: 'floorKeys',
    view: scratch.floorKeys,
    type: 'atomic<u32>',
    access: 'read_write'
  };
  const position = (row: string, component: 0 | 1) =>
    `positions[positionsOffset + 2u * ${row} + ${component}u]`;
  return [
    createWGSLKernelNode<Parameters>(graph, {
      ...gated,
      id: `${props.id}-area`,
      variant: 'visvalingam-area',
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        ...links('read'),
        {name: 'areaKeys', view: scratch.areaKeys, type: 'u32', access: 'read_write'}
      ],
      declarations: LINE_SIMPLIFICATION_METRIC_WGSL,
      body: `${IS_UNDECIDED_WGSL}
  lineOpaqueZero = previousRow >> 31u;
  let originX = ${position('previousRow', 0)};
  let originY = ${position('previousRow', 1)};
  let toMiddleX = ${position('index', 0)} - originX;
  let toMiddleY = ${position('index', 1)} - originY;
  let toNextX = ${position('nextRow', 0)} - originX;
  let toNextY = ${position('nextRow', 1)} - originY;
  let cross = lineProduct(toMiddleX, toNextY) - lineProduct(toMiddleY, toNextX);
  // abs() makes the key a canonical non-negative f32 whose bits order like its value.
  areaKeys[areaKeysOffset + index] = bitcast<u32>(abs(cross) * 0.5);`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      ...gated,
      id: `${props.id}-mark`,
      variant: `visvalingam-mark-${neighborhoodRadius}`,
      bindings: [
        ...links('read'),
        {name: 'areaKeys', view: scratch.areaKeys, type: 'u32', access: 'read'},
        floorKeys,
        {name: 'pendingFlags', view: scratch.pendingFlags, type: 'u32', access: 'read_write'},
        {name: 'importance', view: props.importance, type: 'u32', access: 'read_write'},
        {name: 'status', view: props.status, type: 'atomic<u32>', access: 'read_write'}
      ],
      declarations: `const NEIGHBORHOOD_RADIUS: u32 = ${neighborhoodRadius}u;

// True when a surviving row within the neighborhood orders before (key, row).
fn hasSmallerNeighbor(row: u32, key: u32, forward: bool) -> bool {
  var cursor = row;
  for (var step = 0u; step < NEIGHBORHOOD_RADIUS; step++) {
    cursor = select(previousRows[previousRowsOffset + cursor], nextRows[nextRowsOffset + cursor], forward);
    // Endpoints and rows outside a line are not undecided interior rows: stop the walk.
    if (!(previousRows[previousRowsOffset + cursor] < cursor &&
        cursor < nextRows[nextRowsOffset + cursor])) {
      return false;
    }
    let neighborKey = areaKeys[areaKeysOffset + cursor];
    if (neighborKey < key || (neighborKey == key && cursor < row)) {
      return true;
    }
  }
  return false;
}`,
      body: `${IS_UNDECIDED_WGSL}
  let key = areaKeys[areaKeysOffset + index];
  if (hasSmallerNeighbor(index, key, false) || hasSmallerNeighbor(index, key, true)) {
    atomicStore(&status[statusOffset], 1u);
    return;
  }
  importance[importanceOffset + index] = max(key, atomicLoad(&floorKeys[floorKeysOffset + index]));
  pendingFlags[pendingFlagsOffset + index] = 1u;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      ...gated,
      id: `${props.id}-unlink`,
      variant: 'visvalingam-unlink',
      bindings: [
        ...links('read_write'),
        floorKeys,
        {name: 'pendingFlags', view: scratch.pendingFlags, type: 'u32', access: 'read_write'},
        {name: 'importance', view: props.importance, type: 'u32', access: 'read'}
      ],
      body: `if (pendingFlags[pendingFlagsOffset + index] == 0u) {
    return;
  }
  let previousRow = previousRows[previousRowsOffset + index];
  let nextRow = nextRows[nextRowsOffset + index];
  let effectiveArea = importance[importanceOffset + index];
  nextRows[nextRowsOffset + previousRow] = nextRow;
  previousRows[previousRowsOffset + nextRow] = previousRow;
  atomicMax(&floorKeys[floorKeysOffset + previousRow], effectiveArea);
  atomicMax(&floorKeys[floorKeysOffset + nextRow], effectiveArea);
  previousRows[previousRowsOffset + index] = index;
  nextRows[nextRowsOffset + index] = index;
  pendingFlags[pendingFlagsOffset + index] = 0u;`
    })
  ];
}

/**
 * Gives every row still undecided after the round cap importance `+Infinity`, so the kept set at
 * any tolerance is a superset of the converged one.
 *
 * @internal
 */
export function createVisvalingamUnresolvedNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    scratch: VisvalingamScratch;
    importance: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'visvalingam-unresolved',
    bindings: [
      {name: 'previousRows', view: props.scratch.previousRows, type: 'u32', access: 'read'},
      {name: 'nextRows', view: props.scratch.nextRows, type: 'u32', access: 'read'},
      {name: 'importance', view: props.importance, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.importance.length,
    declarations: `const ENDPOINT_IMPORTANCE: u32 = ${LINE_ENDPOINT_IMPORTANCE_BITS}u;`,
    body: `${IS_UNDECIDED_WGSL}
  importance[importanceOffset + index] = ENDPOINT_IMPORTANCE;`
  });
}
