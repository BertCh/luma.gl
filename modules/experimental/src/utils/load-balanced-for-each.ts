// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from './wgsl-kernel-nodes';

/**
 * Shared load-balanced CSR loops (algorithms roadmap decision 3): merge-path partitioning in the
 * style of Merrill and Garland ("Merge-based parallel sparse matrix-vector multiplication", SC16)
 * and moderngpu's load-balancing search.
 *
 * The rows' end offsets and the item indices form one merged sequence of `rowCount + itemCount`
 * steps. Every workgroup takes {@link LOAD_BALANCED_TILE_SIZE} consecutive steps and every thread
 * {@link LOAD_BALANCED_ITEMS_PER_THREAD}, whatever the row lengths, so a hub row with 1e5 items
 * spreads over about a hundred workgroups instead of serializing one thread (the thread-per-row
 * loop) or one workgroup (the 64-item block hybrids). A workgroup finds its two tile bounds with a
 * binary search over the global offsets, stages the at most `TILE_SIZE + 1` row ends of its tile in
 * workgroup memory, and every thread finds its start with a binary search there.
 *
 * The offsets are CSR row offsets (`rowCount + 1` non-decreasing `uint32`). Items are the slots
 * `[offsets[0], offsets[rowCount])`, clamped to `itemCapacity` (library rule 11: bounded by the
 * offsets, never by the capacity alone). Every size is a pipeline-overridable constant and no loop
 * index reaches WGSL text, so every call with the same body shares one shader module (rule 12).
 *
 * @internal
 */
export const LOAD_BALANCED_WORKGROUP_SIZE = 256;

/** Merge steps (row ends plus items) per thread. @internal */
export const LOAD_BALANCED_ITEMS_PER_THREAD = 4;

/** Merge steps per workgroup tile. @internal */
export const LOAD_BALANCED_TILE_SIZE =
  LOAD_BALANCED_WORKGROUP_SIZE * LOAD_BALANCED_ITEMS_PER_THREAD;

/** Accumulator types of {@link createLoadBalancedSegmentSumNodes}. @internal */
export type LoadBalancedAccumulatorType =
  | 'f32'
  | 'vec2<f32>'
  | 'vec4<f32>'
  | 'u32'
  | 'vec2<u32>'
  | 'vec4<u32>';

/** Properties shared by the load-balanced CSR loops. @internal */
export type LoadBalancedCSRProps = {
  /** Graph-wide node ID prefix. */
  id: string;
  /** Operation name reported in workload estimates. */
  operation: string;
  /** Workload variant of the main kernel. */
  variant?: string;
  /** CSR row offsets, `rowCount + 1` non-decreasing entries. */
  offsets: GraphDataView<'uint32'>;
  /**
   * Allocated item slots (for example `neighbors.length`). The loop visits the slots
   * `[offsets[0], offsets[rowCount])` clamped to it; it also sizes the dispatch.
   */
  itemCapacity: number;
  /**
   * Independent passes over the same CSR (for example value columns); each sees its `lane`.
   * Defaults to 1. `laneCount * (rowCount + itemCapacity)` must fit in `uint32` (rule 16).
   */
  laneCount?: number;
  /**
   * Caller storage bindings, bound after the utility's own (`loadBalancedOffsets` and, for the
   * segment sum, the output and the tile carries), within the 8-binding limit.
   */
  bindings: readonly WGSLKernelBinding[];
  /**
   * Module-scope WGSL. Must declare the per-item function named by the loop:
   * `visitLoadBalancedItem` or `getLoadBalancedContribution`.
   */
  declarations: string;
};

/** Number of row offsets (`rowCount + 1`) and tiles per lane of a load-balanced loop. @internal */
export function getLoadBalancedTileCount(rowCount: number, itemCapacity: number): number {
  return Math.max(1, Math.ceil((rowCount + itemCapacity) / LOAD_BALANCED_TILE_SIZE));
}

/**
 * Runs `fn visitLoadBalancedItem(row: u32, item: u32, lane: u32)` (from `declarations`) once per
 * CSR item, with its row, across evenly loaded threads. `item` is the absolute slot index. One
 * node, one dispatch of `laneCount * tileCount` workgroups. The visit order is unspecified, so
 * visits that combine results need atomics or {@link createLoadBalancedSegmentSumNodes}.
 *
 * @internal
 */
export function createLoadBalancedForEachNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: LoadBalancedCSRProps
): GPUCommandNode<Parameters> {
  const shape = getLoadBalancedShape(props);
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: props.variant ?? 'load-balanced-for-each',
    bindings: [getOffsetsBinding(props), ...props.bindings],
    workgroupSize: LOAD_BALANCED_WORKGROUP_SIZE,
    invocationCount: shape.workgroupCount * LOAD_BALANCED_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `${getLoadBalancedDeclarations()}
${props.declarations}`,
    constants: shape.constants,
    body: `${getLoadBalancedWalkWGSL({
      consume: 'visitLoadBalancedItem(loadBalancedRow, loadBalancedSlot, loadBalancedLane);',
      emit: ''
    })}`
  });
}

/**
 * Per-row sums `output[(row * laneCount + lane) * C + c]` of
 * `fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> T` over the row's items,
 * where `T` is `accumulatorType` with `C` components. Every row of every lane is written, empty
 * rows with zero (rule 11).
 *
 * Two nodes: the load-balanced walk, where every thread sums its run of a row and a segmented
 * workgroup scan joins the runs of rows that cross threads; then a fixup with one thread per
 * tile that adds the carries of a row crossing tiles, in tile order (a hub of `d` items is one
 * serial loop of about `d / 1024` carries). The association depends only on the offsets, so
 * float sums are bitwise reproducible; they differ from a sequential sum by f32 rounding.
 *
 * @internal
 */
export function createLoadBalancedSegmentSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: LoadBalancedCSRProps & {
    accumulatorType: LoadBalancedAccumulatorType;
    /** `rowCount * laneCount * C` entries, `float32` or `uint32` matching `accumulatorType`. */
    output: GraphDataView;
  }
): GPUCommandNode<Parameters>[] {
  const shape = getLoadBalancedShape(props);
  const {accumulatorType} = props;
  const componentCount = getComponentCount(accumulatorType);
  const scalarType = accumulatorType.includes('f32') ? 'f32' : 'u32';
  const rowCount = props.offsets.length - 1;
  // The walk and the fixup write `(row * laneCount + lane) * C + c` for every row and lane.
  assert(props.output.length >= rowCount * shape.laneCount * componentCount);
  const carryStride = 1 + componentCount;
  const carries = createTransientView(
    graph,
    `${props.id}-tile-carries`,
    'uint32',
    shape.workgroupCount * carryStride
  );
  const component = (value: string, index: number) =>
    componentCount === 1 ? value : `${value}[${index}]`;
  const toWord = (value: string) => (scalarType === 'f32' ? `bitcast<u32>(${value})` : value);
  const fromWord = (value: string) => (scalarType === 'f32' ? `bitcast<f32>(${value})` : value);
  const writeRow = (row: string, value: string, operator: '=' | '+=') =>
    Array.from(
      {length: componentCount},
      (_, index) =>
        `loadBalancedOutput[loadBalancedOutputOffset + (${row} * LOAD_BALANCED_LANE_COUNT + loadBalancedLane) * ${componentCount}u + ${index}u] ${operator} ${component(value, index)};`
    ).join('\n      ');
  const readCarry = (record: string) =>
    componentCount === 1
      ? fromWord(
          `loadBalancedCarries[loadBalancedCarriesOffset + ${record} * ${carryStride}u + 1u]`
        )
      : `${accumulatorType}(${Array.from({length: componentCount}, (_, index) =>
          fromWord(
            `loadBalancedCarries[loadBalancedCarriesOffset + ${record} * ${carryStride}u + ${index + 1}u]`
          )
        ).join(', ')})`;
  const outputBinding: WGSLKernelBinding = {
    name: 'loadBalancedOutput',
    view: props.output,
    type: scalarType,
    access: 'read_write'
  };
  const carriesBinding: WGSLKernelBinding = {
    name: 'loadBalancedCarries',
    view: carries,
    type: 'u32',
    access: 'read_write'
  };
  const walkNode = createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: props.variant ?? 'load-balanced-segment-sum',
    bindings: [getOffsetsBinding(props), outputBinding, carriesBinding, ...props.bindings],
    workgroupSize: LOAD_BALANCED_WORKGROUP_SIZE,
    invocationCount: shape.workgroupCount * LOAD_BALANCED_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `${getLoadBalancedDeclarations()}
var<workgroup> loadBalancedCarryRows: array<u32, ${LOAD_BALANCED_WORKGROUP_SIZE}>;
var<workgroup> loadBalancedCarryValues: array<${accumulatorType}, ${LOAD_BALANCED_WORKGROUP_SIZE}>;
${props.declarations}`,
    constants: shape.constants,
    body: `var loadBalancedRunning = ${accumulatorType}();
  var loadBalancedFirstValue = ${accumulatorType}();
  var loadBalancedHasEmitted = false;
  ${getLoadBalancedWalkWGSL({
    consume: `loadBalancedRunning = loadBalancedRunning +
          getLoadBalancedContribution(loadBalancedRow, loadBalancedSlot, loadBalancedLane);`,
    emit: `if (loadBalancedHasEmitted) {
            ${writeRow('loadBalancedRow', 'loadBalancedRunning', '=')}
          } else {
            loadBalancedFirstValue = loadBalancedRunning;
            loadBalancedHasEmitted = true;
          }
          loadBalancedRunning = ${accumulatorType}();`
  })}
  // Inclusive segmented scan of the thread runs, keyed by row (rows never decrease by thread).
  loadBalancedCarryRows[localInvocationIndex] = loadBalancedRow;
  loadBalancedCarryValues[localInvocationIndex] = loadBalancedRunning;
  workgroupBarrier();
  for (var stride = 1u; stride < ${LOAD_BALANCED_WORKGROUP_SIZE}u; stride = stride * 2u) {
    var previousRow = 0xffffffffu;
    var previousValue = ${accumulatorType}();
    if (localInvocationIndex >= stride) {
      previousRow = loadBalancedCarryRows[localInvocationIndex - stride];
      previousValue = loadBalancedCarryValues[localInvocationIndex - stride];
    }
    workgroupBarrier();
    if (previousRow == loadBalancedCarryRows[localInvocationIndex]) {
      loadBalancedCarryValues[localInvocationIndex] =
        previousValue + loadBalancedCarryValues[localInvocationIndex];
    }
    workgroupBarrier();
  }
  if (loadBalancedIsActive) {
    if (loadBalancedHasEmitted) {
      // Thread t starts in the row where thread t - 1 stopped; its scan holds the earlier runs.
      var carryIn = ${accumulatorType}();
      if (localInvocationIndex > 0u &&
          loadBalancedCarryRows[localInvocationIndex - 1u] == loadBalancedStartRow) {
        carryIn = loadBalancedCarryValues[localInvocationIndex - 1u];
      }
      let rowValue = carryIn + loadBalancedFirstValue;
      ${writeRow('loadBalancedStartRow', 'rowValue', '=')}
    }
    if (localInvocationIndex == ${LOAD_BALANCED_WORKGROUP_SIZE - 1}u) {
      let carryRow = loadBalancedCarryRows[localInvocationIndex];
      let carryValue = loadBalancedCarryValues[localInvocationIndex];
      let record = loadBalancedCarriesOffset + workgroupIndex * ${carryStride}u;
      loadBalancedCarries[record] = select(LOAD_BALANCED_ROW_COUNT, carryRow, loadBalancedTileLength > 0u);
      ${Array.from(
        {length: componentCount},
        (_, index) =>
          `loadBalancedCarries[record + ${index + 1}u] = ${toWord(component('carryValue', index))};`
      ).join('\n      ')}
    }
  }`
  });
  const fixupNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-fixup`,
    operation: props.operation,
    variant: 'load-balanced-segment-sum-fixup',
    bindings: [carriesBinding, outputBinding],
    invocationCount: shape.workgroupCount,
    // A fixed size keeps one module for every tile count (rule 12).
    workgroupSize: 64,
    declarations: `override LOAD_BALANCED_ROW_COUNT: u32;
override LOAD_BALANCED_LANE_COUNT: u32;
override LOAD_BALANCED_TILES_PER_LANE: u32;`,
    constants: {
      LOAD_BALANCED_ROW_COUNT: rowCount,
      LOAD_BALANCED_LANE_COUNT: shape.laneCount,
      LOAD_BALANCED_TILES_PER_LANE: shape.tileCount
    },
    body: `let loadBalancedLane = index / LOAD_BALANCED_TILES_PER_LANE;
  let tile = index % LOAD_BALANCED_TILES_PER_LANE;
  let row = loadBalancedCarries[loadBalancedCarriesOffset + index * ${carryStride}u];
  if (row >= LOAD_BALANCED_ROW_COUNT) {
    return;
  }
  // Only the first tile of a run of tiles carrying the same row adds the run, in tile order.
  if (tile > 0u && loadBalancedCarries[loadBalancedCarriesOffset + (index - 1u) * ${carryStride}u] == row) {
    return;
  }
  var sum = ${readCarry('index')};
  let laneEnd = (loadBalancedLane + 1u) * LOAD_BALANCED_TILES_PER_LANE;
  var next = index + 1u;
  loop {
    if (next >= laneEnd ||
        loadBalancedCarries[loadBalancedCarriesOffset + next * ${carryStride}u] != row) {
      break;
    }
    sum = sum + ${readCarry('next')};
    next = next + 1u;
  }
  ${writeRow('row', 'sum', '+=')}`
  });
  return [walkNode, fixupNode];
}

function getComponentCount(type: LoadBalancedAccumulatorType): number {
  return type.startsWith('vec4') ? 4 : type.startsWith('vec2') ? 2 : 1;
}

function getOffsetsBinding(props: LoadBalancedCSRProps): WGSLKernelBinding {
  return {name: 'loadBalancedOffsets', view: props.offsets, type: 'u32', access: 'read'};
}

function getLoadBalancedShape(props: LoadBalancedCSRProps): {
  laneCount: number;
  tileCount: number;
  workgroupCount: number;
  constants: Record<string, number>;
} {
  const rowCount = props.offsets.length - 1;
  const laneCount = props.laneCount ?? 1;
  if (
    rowCount < 0 ||
    !Number.isSafeInteger(props.itemCapacity) ||
    props.itemCapacity < 0 ||
    !Number.isSafeInteger(laneCount) ||
    laneCount < 1 ||
    laneCount * (rowCount + props.itemCapacity) > 0xffffffff
  ) {
    throw new Error(`${props.operation}: invalid load-balanced topology for ${props.id}`);
  }
  const tileCount = getLoadBalancedTileCount(rowCount, props.itemCapacity);
  const workgroupCount = laneCount * tileCount;
  if (workgroupCount * LOAD_BALANCED_WORKGROUP_SIZE > 0xffffffff) {
    throw new Error(`${props.operation}: load-balanced dispatch of ${props.id} is too large`);
  }
  return {
    laneCount,
    tileCount,
    workgroupCount,
    constants: {
      LOAD_BALANCED_ROW_COUNT: rowCount,
      LOAD_BALANCED_ITEM_CAPACITY: props.itemCapacity,
      LOAD_BALANCED_LANE_COUNT: laneCount,
      LOAD_BALANCED_TILES_PER_LANE: tileCount
    }
  };
}

/** Module-scope declarations of the merge-path walk. */
function getLoadBalancedDeclarations(): string {
  return /* wgsl */ `override LOAD_BALANCED_ROW_COUNT: u32;
override LOAD_BALANCED_ITEM_CAPACITY: u32;
override LOAD_BALANCED_LANE_COUNT: u32;
override LOAD_BALANCED_TILES_PER_LANE: u32;
var<workgroup> loadBalancedTileRowEnds: array<u32, ${LOAD_BALANCED_TILE_SIZE + 1}>;
var<workgroup> loadBalancedTileBounds: array<u32, 2>;
/** End of row \`row\` relative to \`itemBase\`, clamped to the items \`[itemBase, itemLimit)\`. */
fn getLoadBalancedRowEnd(row: u32, itemBase: u32, itemLimit: u32) -> u32 {
  return clamp(loadBalancedOffsets[loadBalancedOffsetsOffset + row + 1u], itemBase, itemLimit) - itemBase;
}
/** Rows consumed at merge step \`diagonal\` (Merrill-Garland merge-path search). */
fn searchLoadBalancedPath(diagonal: u32, itemBase: u32, itemLimit: u32) -> u32 {
  let itemCount = itemLimit - itemBase;
  var low = select(0u, diagonal - itemCount, diagonal > itemCount);
  var high = min(diagonal, LOAD_BALANCED_ROW_COUNT);
  loop {
    if (low >= high) { break; }
    let pivot = (low + high) / 2u;
    if (getLoadBalancedRowEnd(pivot, itemBase, itemLimit) <= diagonal - pivot - 1u) {
      low = pivot + 1u;
    } else {
      high = pivot;
    }
  }
  return low;
}`;
}

/**
 * The merge-path walk. Defines `loadBalancedLane`, `loadBalancedIsActive`,
 * `loadBalancedTileLength`, `loadBalancedStartRow` and leaves `loadBalancedRow` at the row the
 * thread stopped in. `consume` sees `loadBalancedRow` and the absolute `loadBalancedSlot`; `emit`
 * runs when the thread passes the end of `loadBalancedRow`, before the row advances. Every barrier
 * is in uniform control flow.
 */
function getLoadBalancedWalkWGSL(props: {consume: string; emit: string}): string {
  return /* wgsl */ `let loadBalancedIsActive =
    workgroupIndex < LOAD_BALANCED_LANE_COUNT * LOAD_BALANCED_TILES_PER_LANE;
  let loadBalancedLane = workgroupIndex / LOAD_BALANCED_TILES_PER_LANE;
  let loadBalancedTile = workgroupIndex % LOAD_BALANCED_TILES_PER_LANE;
  let loadBalancedItemBase = min(
    loadBalancedOffsets[loadBalancedOffsetsOffset],
    LOAD_BALANCED_ITEM_CAPACITY
  );
  let loadBalancedItemLimit = clamp(
    loadBalancedOffsets[loadBalancedOffsetsOffset + LOAD_BALANCED_ROW_COUNT],
    loadBalancedItemBase,
    LOAD_BALANCED_ITEM_CAPACITY
  );
  let loadBalancedMergeCount =
    LOAD_BALANCED_ROW_COUNT + loadBalancedItemLimit - loadBalancedItemBase;
  let loadBalancedTileBegin = select(
    loadBalancedMergeCount,
    min(loadBalancedTile * ${LOAD_BALANCED_TILE_SIZE}u, loadBalancedMergeCount),
    loadBalancedIsActive
  );
  let loadBalancedTileEnd = min(loadBalancedTileBegin + ${LOAD_BALANCED_TILE_SIZE}u, loadBalancedMergeCount);
  let loadBalancedTileLength = loadBalancedTileEnd - loadBalancedTileBegin;
  if (localInvocationIndex < 2u) {
    loadBalancedTileBounds[localInvocationIndex] = searchLoadBalancedPath(
      select(loadBalancedTileBegin, loadBalancedTileEnd, localInvocationIndex == 1u),
      loadBalancedItemBase,
      loadBalancedItemLimit
    );
  }
  workgroupBarrier();
  let loadBalancedTileRowBegin = loadBalancedTileBounds[0];
  let loadBalancedTileRowCount = loadBalancedTileBounds[1] - loadBalancedTileRowBegin;
  // Item coordinates are relative to loadBalancedItemBase; rows at or past ROW_COUNT never end.
  let loadBalancedTileItemBegin = loadBalancedTileBegin - loadBalancedTileRowBegin;
  let loadBalancedTileItemCount =
    loadBalancedTileEnd - loadBalancedTileBounds[1] - loadBalancedTileItemBegin;
  for (var entry = localInvocationIndex; entry <= loadBalancedTileRowCount; entry += ${LOAD_BALANCED_WORKGROUP_SIZE}u) {
    let entryRow = loadBalancedTileRowBegin + entry;
    var rowEnd = 0xffffffffu;
    if (entryRow < LOAD_BALANCED_ROW_COUNT) {
      rowEnd = getLoadBalancedRowEnd(entryRow, loadBalancedItemBase, loadBalancedItemLimit);
    }
    loadBalancedTileRowEnds[entry] = rowEnd;
  }
  workgroupBarrier();
  let loadBalancedDiagonal = min(
    localInvocationIndex * ${LOAD_BALANCED_ITEMS_PER_THREAD}u,
    loadBalancedTileLength
  );
  var loadBalancedLow = select(
    0u,
    loadBalancedDiagonal - loadBalancedTileItemCount,
    loadBalancedDiagonal > loadBalancedTileItemCount
  );
  var loadBalancedHigh = min(loadBalancedDiagonal, loadBalancedTileRowCount);
  loop {
    if (loadBalancedLow >= loadBalancedHigh) { break; }
    let pivot = (loadBalancedLow + loadBalancedHigh) / 2u;
    if (loadBalancedTileRowEnds[pivot] <= loadBalancedTileItemBegin + loadBalancedDiagonal - pivot - 1u) {
      loadBalancedLow = pivot + 1u;
    } else {
      loadBalancedHigh = pivot;
    }
  }
  var loadBalancedRow = loadBalancedTileRowBegin + loadBalancedLow;
  let loadBalancedStartRow = loadBalancedRow;
  var loadBalancedItem = loadBalancedTileItemBegin + loadBalancedDiagonal - loadBalancedLow;
  let loadBalancedStepCount = min(
    ${LOAD_BALANCED_ITEMS_PER_THREAD}u,
    loadBalancedTileLength - loadBalancedDiagonal
  );
  for (var walkStep = 0u; walkStep < ${LOAD_BALANCED_ITEMS_PER_THREAD}u; walkStep = walkStep + 1u) {
    if (walkStep < loadBalancedStepCount) {
      if (loadBalancedItem < loadBalancedTileRowEnds[loadBalancedRow - loadBalancedTileRowBegin]) {
        let loadBalancedSlot = loadBalancedItemBase + loadBalancedItem;
        ${props.consume}
        loadBalancedItem = loadBalancedItem + 1u;
      } else {
        ${props.emit}
        loadBalancedRow = loadBalancedRow + 1u;
      }
    }
  }`;
}
