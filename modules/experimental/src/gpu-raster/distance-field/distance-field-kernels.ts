// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {createRasterGateSlotTable} from '../cost-distance/raster-relaxation';
import {GPU_DISTANCE_FIELD_NONE} from './distance-field-parameters';

/** Operation name reported in workload estimates. @internal */
export const DISTANCE_FIELD_OPERATION = 'GPUDistanceField';

/**
 * Correctly rounded f32 square root of `mantissa * 2^exponent` (a positive integer key or a
 * normal f32), independent of the backend's `sqrt` precision.
 *
 * WGSL only bounds `sqrt` error (Metal returns 7.0000005 for 49), which would break the 1-ULP
 * contract. The hardware result seeds a search over adjacent floats; each step compares the
 * square of the midpoint to the next float against the input exactly, using 64-bit products
 * emulated with u32 pairs. A midpoint square is odd and wider than 24 bits, so it never equals
 * the input and no tie rule is needed.
 */
const ROUNDED_SQUARE_ROOT_WGSL = /* wgsl */ `
fn multiplyWide(left: u32, right: u32) -> vec2<u32> {
  let leftLow = left & 0xffffu;
  let leftHigh = left >> 16u;
  let rightLow = right & 0xffffu;
  let rightHigh = right >> 16u;
  let lowLow = leftLow * rightLow;
  let lowHigh = leftLow * rightHigh;
  let highLow = leftHigh * rightLow;
  let middle = (lowLow >> 16u) + (lowHigh & 0xffffu) + (highLow & 0xffffu);
  let low = (lowLow & 0xffffu) | (middle << 16u);
  let high = leftHigh * rightHigh + (lowHigh >> 16u) + (highLow >> 16u) + (middle >> 16u);
  return vec2<u32>(low, high);
}

/** Whether (candidate + ulp / 2)^2 > mantissa * 2^exponent, for a positive normal candidate. */
fn isUpperMidpointAbove(candidate: f32, mantissa: u32, exponent: i32) -> bool {
  let bits = bitcast<u32>(candidate);
  let candidateExponent = i32((bits >> 23u) & 0xffu);
  let odd = (((bits & 0x7fffffu) | 0x800000u) << 1u) | 1u;
  // Midpoint^2 = odd^2 * 2^(2 * candidateExponent - 302); compare odd^2 with mantissa * 2^shift.
  let square = multiplyWide(odd, odd);
  let shift = exponent - 2 * candidateExponent + 302;
  if (shift < 0) { return true; }
  if (shift >= 64) { return false; }
  var scaled: vec2<u32>;
  if (shift == 0) {
    scaled = vec2<u32>(mantissa, 0u);
  } else if (shift < 32) {
    scaled = vec2<u32>(mantissa << u32(shift), mantissa >> u32(32 - shift));
  } else {
    let highShift = u32(shift - 32);
    if (highShift > 0u && (mantissa >> (32u - highShift)) != 0u) { return false; }
    scaled = vec2<u32>(0u, mantissa << highShift);
  }
  return square.y > scaled.y || (square.y == scaled.y && square.x > scaled.x);
}

fn getRoundedSquareRoot(mantissa: u32, exponent: i32) -> f32 {
  if (mantissa == 0u) { return 0.0; }
  var root = sqrt(f32(mantissa) * exp2(f32(exponent)));
  for (var step = 0u; step < 8u; step++) {
    if (!isUpperMidpointAbove(root, mantissa, exponent)) {
      root = bitcast<f32>(bitcast<u32>(root) + 1u);
    } else if (isUpperMidpointAbove(bitcast<f32>(bitcast<u32>(root) - 1u), mantissa, exponent)) {
      root = bitcast<f32>(bitcast<u32>(root) - 1u);
    } else {
      break;
    }
  }
  return root;
}`;

/** Workgroup size of the one-invocation-per-row and per-column carry kernels. */
const LINE_WORKGROUP_SIZE = 64;

/** Grid dimensions shared by every distance-field kernel. @internal */
export type DistanceFieldGrid = {width: number; height: number};

/**
 * WGSL grid constants and the candidate order shared by the exact and jump-flood kernels.
 *
 * `isCloser` compares two candidates given as absolute cell offsets `(dx, dy)` and seed IDs. With
 * equal cell sizes it compares exact `u32` squared offsets; otherwise it compares the f32 squared
 * ground distance `(cellSizeX * dx)^2 + (cellSizeY * dy)^2`, the same expression the finalize
 * kernel takes the square root of. Equal keys break on the smaller seed ID. The comparison is a
 * total order, which the lower-envelope pass relies on.
 */
function getDistanceFieldGridWGSL(grid: DistanceFieldGrid, withMetric: boolean): string {
  return /* wgsl */ `
const GRID_WIDTH: u32 = ${grid.width}u;
const GRID_HEIGHT: u32 = ${grid.height}u;
const NONE: u32 = ${GPU_DISTANCE_FIELD_NONE}u;

fn getAbsoluteDifference(left: u32, right: u32) -> u32 {
  return select(right - left, left - right, left > right);
}
${
  withMetric
    ? /* wgsl */ `
fn isIsotropic() -> bool {
  return settings[settingsOffset + 2u] == settings[settingsOffset + 3u];
}

fn getAnisotropicKey(dx: u32, dy: u32) -> f32 {
  let groundX = settings[settingsOffset + 2u] * f32(dx);
  let groundY = settings[settingsOffset + 3u] * f32(dy);
  return groundX * groundX + groundY * groundY;
}

fn isCloser(dxA: u32, dyA: u32, idA: u32, dxB: u32, dyB: u32, idB: u32) -> bool {
  if (isIsotropic()) {
    let keyA = dxA * dxA + dyA * dyA;
    let keyB = dxB * dxB + dyB * dyB;
    if (keyA != keyB) { return keyA < keyB; }
  } else {
    let keyA = getAnisotropicKey(dxA, dyA);
    let keyB = getAnisotropicKey(dxB, dyB);
    if (keyA != keyB) { return keyA < keyB; }
  }
  return idA < idB;
}`
    : ''
}`;
}

/**
 * Writes the smallest seed ID per cell: first the mask seeds (`mask value - 1`) or
 * {@link GPU_DISTANCE_FIELD_NONE}, then point seeds folded in with `atomicMin`.
 *
 * @internal
 */
export function createDistanceFieldSeedNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    grid: DistanceFieldGrid;
    settings: GraphDataView<'float32'>;
    cellSeeds: GraphDataView<'uint32'>;
    seedMask?: GraphDataView<'uint32'>;
    seedPositions?: GraphDataView;
    seedIds?: GraphDataView<'uint32'>;
    seedCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, grid} = props;
  const cellCount = grid.width * grid.height;
  const clearBindings: WGSLKernelBinding[] = [
    {
      name: 'cellSeeds',
      view: props.cellSeeds,
      type: 'u32',
      access: 'read_write'
    }
  ];
  if (props.seedMask) {
    clearBindings.unshift({
      name: 'seedMask',
      view: props.seedMask,
      type: 'u32',
      access: 'read'
    });
  }
  const nodes = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-seed-clear`,
      operation: DISTANCE_FIELD_OPERATION,
      variant: 'seed-clear',
      bindings: clearBindings,
      invocationCount: cellCount,
      body: props.seedMask
        ? // Mask value v marks a seed with ID v - 1, so 0 is "no seed" and 0xffffffff wraps to NONE.
          'cellSeeds[cellSeedsOffset + index] = seedMask[seedMaskOffset + index] - 1u;'
        : `cellSeeds[cellSeedsOffset + index] = ${GPU_DISTANCE_FIELD_NONE}u;`
    })
  ];
  const positions = props.seedPositions;
  if (positions && positions.length > 0) {
    const bindings: WGSLKernelBinding[] = [
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'positions', view: positions, type: 'f32', access: 'read'}
    ];
    if (props.seedIds) {
      bindings.push({
        name: 'seedIds',
        view: props.seedIds,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.seedCount) {
      bindings.push({
        name: 'seedCount',
        view: props.seedCount,
        type: 'u32',
        access: 'read'
      });
    }
    bindings.push({
      name: 'cellSeedBits',
      view: props.cellSeeds,
      type: 'atomic<u32>',
      access: 'read_write'
    });
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-seed-scatter`,
        operation: DISTANCE_FIELD_OPERATION,
        variant: 'seed-scatter',
        bindings,
        invocationCount: positions.length,
        declarations: getDistanceFieldGridWGSL(grid, false),
        body: /* wgsl */ `${props.seedCount ? 'if (index >= seedCount[seedCountOffset]) { return; }' : ''}
  let seedId = ${props.seedIds ? 'seedIds[seedIdsOffset + index]' : 'index'};
  if (seedId == NONE) { return; }
  let column = floor((positions[positionsOffset + index * 2u] - settings[settingsOffset]) * settings[settingsOffset + 4u]);
  let row = floor((positions[positionsOffset + index * 2u + 1u] - settings[settingsOffset + 1u]) * settings[settingsOffset + 5u]);
  // Negated comparisons also reject NaN and infinite coordinates.
  if (!(column >= 0.0 && column < f32(GRID_WIDTH) && row >= 0.0 && row < f32(GRID_HEIGHT))) { return; }
  atomicMin(&cellSeedBits[cellSeedBitsOffset + u32(row) * GRID_WIDTH + u32(column)], seedId);`
      })
    );
  }
  return nodes;
}

/** Rows per band of the banded column pass and columns per band of the banded row passes. @internal */
export const DISTANCE_FIELD_BAND_SIZE = 32;

/** Band counts of the banded exact transform. @internal */
export type DistanceFieldBandLayout = {
  /** Row bands per column: `ceil(height / DISTANCE_FIELD_BAND_SIZE)`. */
  columnBandCount: number;
  /** Column bands per row: `ceil(width / DISTANCE_FIELD_BAND_SIZE)`. */
  rowBandCount: number;
  /** Pairwise envelope merge rounds: `ceil(log2(rowBandCount))`, 0 for one band. */
  mergeRoundCount: number;
};

/** Returns the band layout of the banded exact transform for `grid`. @internal */
export function getDistanceFieldBandLayout(grid: DistanceFieldGrid): DistanceFieldBandLayout {
  const columnBandCount = Math.ceil(grid.height / DISTANCE_FIELD_BAND_SIZE);
  const rowBandCount = Math.ceil(grid.width / DISTANCE_FIELD_BAND_SIZE);
  return {
    columnBandCount,
    rowBandCount,
    mergeRoundCount: rowBandCount > 1 ? Math.ceil(Math.log2(rowBandCount)) : 0
  };
}

/** Scratch of {@link createDistanceFieldColumnNodes}. @internal */
export type DistanceFieldColumnScratch = {
  /** `4 * columnBandCount * width` words: per (band, column) top row, top ID, bottom row, bottom ID. */
  bandSeeds: GraphDataView<'uint32'>;
  /** `4 * columnBandCount * width` words: per (band, column) seed above (row, ID) and below (row, ID). */
  bandCarries: GraphDataView<'uint32'>;
};

/**
 * Exact separable pass 1 with row banding (the first phase of the Parallel Banding Algorithm, Cao,
 * Tang, Mitra and Tan 2010): every cell receives the nearest seed row of its column, the smaller
 * ID on equal row distance.
 *
 * 1. `column-bands`, one invocation per (row band, column): the band's first and last seed.
 * 2. `column-carry`, one invocation per column: the last seed above each band and the first seed
 *    below it, from the band summaries.
 * 3. `column-fill`, one invocation per (row band, column): a forward scan seeded with the seed above
 *    and a backward scan seeded with the seed below, with the tie rule of the unbanded scans.
 *
 * The forward value of a cell is the last seed at or above it and the backward value the first
 * seed at or below it, whatever the banding, so the result is bit-identical to one forward and one
 * backward scan per column. Work per invocation is `O(DISTANCE_FIELD_BAND_SIZE)` for the band
 * passes and `O(columnBandCount)` for the carry.
 *
 * @internal
 */
export function createDistanceFieldColumnNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    grid: DistanceFieldGrid;
    cellSeeds: GraphDataView<'uint32'>;
    columnRows: GraphDataView<'uint32'>;
    columnIds: GraphDataView<'uint32'>;
    scratch: DistanceFieldColumnScratch;
  }
): GPUCommandNode<Parameters>[] {
  const {grid, scratch} = props;
  const {columnBandCount} = getDistanceFieldBandLayout(grid);
  const declarations = /* wgsl */ `${getDistanceFieldGridWGSL(grid, false)}
const BAND_SIZE: u32 = ${DISTANCE_FIELD_BAND_SIZE}u;
const COLUMN_BAND_COUNT: u32 = ${columnBandCount}u;`;
  const bandInvocationCount = columnBandCount * grid.width;
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-column-bands`,
      operation: DISTANCE_FIELD_OPERATION,
      variant: 'exact-column-bands',
      bindings: [
        {name: 'cellSeeds', view: props.cellSeeds, type: 'u32', access: 'read'},
        {name: 'bandSeeds', view: scratch.bandSeeds, type: 'u32', access: 'read_write'}
      ],
      invocationCount: bandInvocationCount,
      declarations,
      body: /* wgsl */ `let column = index % GRID_WIDTH;
  let rowStart = (index / GRID_WIDTH) * BAND_SIZE;
  let rowEnd = min(rowStart + BAND_SIZE, GRID_HEIGHT);
  var topRow = NONE;
  var topId = NONE;
  var bottomRow = NONE;
  var bottomId = NONE;
  for (var row = rowStart; row < rowEnd; row++) {
    let seedId = cellSeeds[cellSeedsOffset + row * GRID_WIDTH + column];
    if (seedId != NONE) {
      if (topRow == NONE) {
        topRow = row;
        topId = seedId;
      }
      bottomRow = row;
      bottomId = seedId;
    }
  }
  let base = bandSeedsOffset + 4u * index;
  bandSeeds[base] = topRow;
  bandSeeds[base + 1u] = topId;
  bandSeeds[base + 2u] = bottomRow;
  bandSeeds[base + 3u] = bottomId;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-column-carry`,
      operation: DISTANCE_FIELD_OPERATION,
      variant: 'exact-column-carry',
      bindings: [
        {name: 'bandSeeds', view: scratch.bandSeeds, type: 'u32', access: 'read'},
        {name: 'bandCarries', view: scratch.bandCarries, type: 'u32', access: 'read_write'}
      ],
      invocationCount: grid.width,
      workgroupSize: LINE_WORKGROUP_SIZE,
      declarations,
      body: /* wgsl */ `let column = index;
  var lastRow = NONE;
  var lastId = NONE;
  for (var band = 0u; band < COLUMN_BAND_COUNT; band++) {
    let base = 4u * (band * GRID_WIDTH + column);
    bandCarries[bandCarriesOffset + base] = lastRow;
    bandCarries[bandCarriesOffset + base + 1u] = lastId;
    let bottomRow = bandSeeds[bandSeedsOffset + base + 2u];
    if (bottomRow != NONE) {
      lastRow = bottomRow;
      lastId = bandSeeds[bandSeedsOffset + base + 3u];
    }
  }
  lastRow = NONE;
  lastId = NONE;
  for (var step = 0u; step < COLUMN_BAND_COUNT; step++) {
    let base = 4u * ((COLUMN_BAND_COUNT - 1u - step) * GRID_WIDTH + column);
    bandCarries[bandCarriesOffset + base + 2u] = lastRow;
    bandCarries[bandCarriesOffset + base + 3u] = lastId;
    let topRow = bandSeeds[bandSeedsOffset + base];
    if (topRow != NONE) {
      lastRow = topRow;
      lastId = bandSeeds[bandSeedsOffset + base + 1u];
    }
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-column-fill`,
      operation: DISTANCE_FIELD_OPERATION,
      variant: 'exact-column-fill',
      bindings: [
        {name: 'cellSeeds', view: props.cellSeeds, type: 'u32', access: 'read'},
        {name: 'bandCarries', view: scratch.bandCarries, type: 'u32', access: 'read'},
        {name: 'columnRows', view: props.columnRows, type: 'u32', access: 'read_write'},
        {name: 'columnIds', view: props.columnIds, type: 'u32', access: 'read_write'}
      ],
      invocationCount: bandInvocationCount,
      declarations,
      body: /* wgsl */ `let column = index % GRID_WIDTH;
  let rowStart = (index / GRID_WIDTH) * BAND_SIZE;
  let rowEnd = min(rowStart + BAND_SIZE, GRID_HEIGHT);
  let base = bandCarriesOffset + 4u * index;
  var lastRow = bandCarries[base];
  var lastId = bandCarries[base + 1u];
  for (var row = rowStart; row < rowEnd; row++) {
    let cell = row * GRID_WIDTH + column;
    let seedId = cellSeeds[cellSeedsOffset + cell];
    if (seedId != NONE) {
      lastRow = row;
      lastId = seedId;
    }
    columnRows[columnRowsOffset + cell] = lastRow;
    columnIds[columnIdsOffset + cell] = lastId;
  }
  lastRow = bandCarries[base + 2u];
  lastId = bandCarries[base + 3u];
  for (var row = rowEnd; row > rowStart; row--) {
    let cell = (row - 1u) * GRID_WIDTH + column;
    let seedId = cellSeeds[cellSeedsOffset + cell];
    if (seedId != NONE) {
      lastRow = row - 1u;
      lastId = seedId;
    }
    if (lastRow == NONE) { continue; }
    let forwardRow = columnRows[columnRowsOffset + cell];
    var useBackward = forwardRow == NONE;
    if (!useBackward) {
      let forwardDistance = row - 1u - forwardRow;
      let backwardDistance = lastRow - (row - 1u);
      useBackward = backwardDistance < forwardDistance ||
        (backwardDistance == forwardDistance && lastId < columnIds[columnIdsOffset + cell]);
    }
    if (useBackward) {
      columnRows[columnRowsOffset + cell] = lastRow;
      columnIds[columnIdsOffset + cell] = lastId;
    }
  }`
    })
  ];
}

/** Scratch of {@link createDistanceFieldRowNodes}; entry arrays are laid out `column * height + row`. @internal */
export type DistanceFieldRowScratch = {
  /** Next envelope entry (a column) of each entry, `NONE` at the tail. */
  entryNext: GraphDataView<'uint32'>;
  /** Previous envelope entry of each entry, `NONE` at the head. */
  entryPrevious: GraphDataView<'uint32'>;
  /** First cell an entry wins, `NONE` for a column that is not (or no longer) on the envelope. */
  entryStarts: GraphDataView<'uint32'>;
  /** `2 * rowBandCount * height` words: head and tail column of the envelope of the block starting at each band. */
  blockEnds: GraphDataView<'uint32'>;
  /** Per cell `1 + column` of the entry starting there, 0 elsewhere; laid out like the grid. */
  startMarkers: GraphDataView<'uint32'>;
  /** `rowBandCount * height` words: per (band, row) maximum marker, then the exclusive running maximum. */
  bandMarkers: GraphDataView<'uint32'>;
};

/** WGSL shared by the envelope passes: candidate lookup, `beatsAt` and the start search. */
const DISTANCE_FIELD_ENVELOPE_WGSL = /* wgsl */ `
/** Whether the candidate in column q beats the candidate in column v at cell x of this row. */
fn beatsAt(q: u32, qDy: u32, qId: u32, v: u32, vDy: u32, vId: u32, x: u32) -> bool {
  return isCloser(getAbsoluteDifference(x, q), qDy, qId, getAbsoluteDifference(x, v), vDy, vId);
}

fn getEntrySlot(column: u32, row: u32) -> u32 {
  return column * GRID_HEIGHT + row;
}

/**
 * Felzenszwalb-Huttenlocher insertion of the candidate in column q into the envelope whose tail is
 * 'tail': pops every tail entry q beats at its start (marking it off the envelope) and returns
 * vec2(new tail, start of q), start GRID_WIDTH when q never wins. The start is the first cell
 * after the tail's start where q wins, found by binary search with isCloser, so ties break exactly
 * on the smallest seed ID.
 */
fn insertEntry(q: u32, row: u32, tailIn: u32) -> vec2<u32> {
  let rowBase = row * GRID_WIDTH;
  let qDy = getAbsoluteDifference(columnRows[columnRowsOffset + rowBase + q], row);
  let qId = columnIds[columnIdsOffset + rowBase + q];
  var tail = tailIn;
  loop {
    if (tail == NONE) { return vec2<u32>(NONE, 0u); }
    let tailSlot = getEntrySlot(tail, row);
    let vStart = entryStarts[entryStartsOffset + tailSlot];
    let vDy = getAbsoluteDifference(columnRows[columnRowsOffset + rowBase + tail], row);
    let vId = columnIds[columnIdsOffset + rowBase + tail];
    if (beatsAt(q, qDy, qId, tail, vDy, vId, vStart)) {
      entryStarts[entryStartsOffset + tailSlot] = NONE;
      tail = entryPrevious[entryPreviousOffset + tailSlot];
      continue;
    }
    // First cell in (vStart, GRID_WIDTH) where q wins, or GRID_WIDTH when q never wins.
    var low = vStart + 1u;
    var high = GRID_WIDTH;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (beatsAt(q, qDy, qId, tail, vDy, vId, middle)) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    return vec2<u32>(tail, low);
  }
}`;

/**
 * Exact separable pass 2 with column banding (phases 2 and 3 of the Parallel Banding Algorithm,
 * Cao, Tang, Mitra and Tan 2010): every cell receives the winner of the discrete lower envelope of
 * the per-column candidates of its row (Felzenszwalb-Huttenlocher), bit-identical to one
 * sequential envelope per row.
 *
 * 1. `row-bands`, one invocation per (column band, row): the sequential envelope of the band's
 *    candidates, as a doubly linked list over column slots with each entry's start (the first cell
 *    it wins, searched over the whole row).
 * 2. `row-merge`, `mergeRoundCount` rounds sharing one pipeline (the round index is a slot word,
 *    library rule 12): blocks of `2^round` bands merge pairwise. The right block's entries are
 *    inserted in order into the left envelope exactly as the sequential sweep would insert them.
 *    That is enough because a column that loses at every cell inside its block loses at every
 *    cell overall (`isCloser` is a total order), and the merge stops at the first right entry
 *    whose predecessor and start are unchanged: every later right entry then sees the same
 *    predecessor and start as in its block, so its block-local state is already the merged state.
 * 3. `row-start-markers`, then `row-marker-bands`, `row-marker-carry` and `row-color`: each
 *    surviving entry marks its start cell with `1 + column`. Starts and columns both increase along
 *    the envelope, so the winner of a cell is the running maximum of the markers to its left, a
 *    banded row scan like pass 1.
 *
 * Work per invocation is `O(DISTANCE_FIELD_BAND_SIZE * log width)` for the band envelopes and the
 * coloring; a merge touches only the entries it removes plus one start search, `O(width)` per row
 * over all rounds.
 *
 * @internal
 */
export function createDistanceFieldRowNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    grid: DistanceFieldGrid;
    settings: GraphDataView<'float32'>;
    columnRows: GraphDataView<'uint32'>;
    columnIds: GraphDataView<'uint32'>;
    scratch: DistanceFieldRowScratch;
    nearestCells: GraphDataView<'uint32'>;
    nearestIds: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {grid, scratch} = props;
  const {rowBandCount, mergeRoundCount} = getDistanceFieldBandLayout(grid);
  const cellCount = grid.width * grid.height;
  const bandInvocationCount = rowBandCount * grid.height;
  const declarations = /* wgsl */ `${getDistanceFieldGridWGSL(grid, true)}
const BAND_SIZE: u32 = ${DISTANCE_FIELD_BAND_SIZE}u;
const ROW_BAND_COUNT: u32 = ${rowBandCount}u;`;
  const envelopeBindings: WGSLKernelBinding[] = [
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
    {name: 'columnRows', view: props.columnRows, type: 'u32', access: 'read'},
    {name: 'columnIds', view: props.columnIds, type: 'u32', access: 'read'},
    {name: 'entryNext', view: scratch.entryNext, type: 'u32', access: 'read_write'},
    {name: 'entryPrevious', view: scratch.entryPrevious, type: 'u32', access: 'read_write'},
    {name: 'entryStarts', view: scratch.entryStarts, type: 'u32', access: 'read_write'},
    {name: 'blockEnds', view: scratch.blockEnds, type: 'u32', access: 'read_write'}
  ];
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-row-bands`,
      operation: DISTANCE_FIELD_OPERATION,
      variant: 'exact-row-bands',
      bindings: envelopeBindings,
      invocationCount: bandInvocationCount,
      declarations: `${declarations}\n${DISTANCE_FIELD_ENVELOPE_WGSL}`,
      body: /* wgsl */ `let row = index % GRID_HEIGHT;
  let band = index / GRID_HEIGHT;
  let rowBase = row * GRID_WIDTH;
  let columnStart = band * BAND_SIZE;
  let columnEnd = min(columnStart + BAND_SIZE, GRID_WIDTH);
  var head = NONE;
  var tail = NONE;
  for (var q = columnStart; q < columnEnd; q++) {
    let slot = getEntrySlot(q, row);
    entryStarts[entryStartsOffset + slot] = NONE;
    if (columnRows[columnRowsOffset + rowBase + q] == NONE) { continue; }
    let inserted = insertEntry(q, row, tail);
    tail = inserted.x;
    if (inserted.y < GRID_WIDTH) {
      entryStarts[entryStartsOffset + slot] = inserted.y;
      entryPrevious[entryPreviousOffset + slot] = tail;
      entryNext[entryNextOffset + slot] = NONE;
      if (tail == NONE) {
        head = q;
      } else {
        entryNext[entryNextOffset + getEntrySlot(tail, row)] = q;
      }
      tail = q;
    }
  }
  blockEnds[blockEndsOffset + 2u * index] = head;
  blockEnds[blockEndsOffset + 2u * index + 1u] = tail;`
    })
  ];
  if (mergeRoundCount > 0) {
    const roundSlots = createRasterGateSlotTable<Parameters>(graph, {
      id: `${props.id}-row-merge-rounds`,
      operation: DISTANCE_FIELD_OPERATION,
      groupSize: mergeRoundCount
    });
    nodes.push(roundSlots.writeNode);
    // Every round dispatches the first round's pair count, so all rounds share one pipeline.
    const pairInvocationCount = Math.ceil(rowBandCount / 2) * grid.height;
    for (let round = 0; round < mergeRoundCount; round++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${props.id}-row-merge-${round}`,
          operation: DISTANCE_FIELD_OPERATION,
          variant: 'exact-row-merge',
          bindings: [
            ...envelopeBindings,
            {name: 'mergeRound', view: roundSlots.getSlot(round), type: 'u32', access: 'read'}
          ],
          invocationCount: pairInvocationCount,
          declarations: `${declarations}\n${DISTANCE_FIELD_ENVELOPE_WGSL}`,
          body: /* wgsl */ `let blockBands = 1u << mergeRound[mergeRoundOffset];
  let row = index % GRID_HEIGHT;
  let leftBand = (index / GRID_HEIGHT) * 2u * blockBands;
  let rightBand = leftBand + blockBands;
  if (rightBand >= ROW_BAND_COUNT) { return; }
  let leftEnds = blockEndsOffset + 2u * (leftBand * GRID_HEIGHT + row);
  let rightEnds = blockEndsOffset + 2u * (rightBand * GRID_HEIGHT + row);
  let leftHead = blockEnds[leftEnds];
  let rightTail = blockEnds[rightEnds + 1u];
  var tail = blockEnds[leftEnds + 1u];
  var current = blockEnds[rightEnds];
  var mergedHead = NONE;
  var reachedLocalState = false;
  loop {
    if (current == NONE) { break; }
    let slot = getEntrySlot(current, row);
    let localPrevious = entryPrevious[entryPreviousOffset + slot];
    let localStart = entryStarts[entryStartsOffset + slot];
    let localNext = entryNext[entryNextOffset + slot];
    let inserted = insertEntry(current, row, tail);
    tail = inserted.x;
    if (inserted.y >= GRID_WIDTH) {
      entryStarts[entryStartsOffset + slot] = NONE;
      current = localNext;
      continue;
    }
    entryStarts[entryStartsOffset + slot] = inserted.y;
    entryPrevious[entryPreviousOffset + slot] = tail;
    if (tail == NONE) {
      mergedHead = current;
    } else {
      entryNext[entryNextOffset + getEntrySlot(tail, row)] = current;
    }
    if (tail == localPrevious && inserted.y == localStart) {
      reachedLocalState = true;
      break;
    }
    tail = current;
    current = localNext;
  }
  // The left head survives unless the merge popped the whole left envelope.
  if (leftHead != NONE && entryStarts[entryStartsOffset + getEntrySlot(leftHead, row)] != NONE) {
    mergedHead = leftHead;
  }
  blockEnds[leftEnds] = mergedHead;
  blockEnds[leftEnds + 1u] = select(tail, rightTail, reachedLocalState);`
        })
      );
    }
  }
  const markerDeclarations = `const GRID_WIDTH: u32 = ${grid.width}u;
const GRID_HEIGHT: u32 = ${grid.height}u;
const NONE: u32 = ${GPU_DISTANCE_FIELD_NONE}u;
const BAND_SIZE: u32 = ${DISTANCE_FIELD_BAND_SIZE}u;
const ROW_BAND_COUNT: u32 = ${rowBandCount}u;`;
  nodes.push(
    createFillNode<Parameters>(graph, {
      id: `${props.id}-row-marker-clear`,
      operation: DISTANCE_FIELD_OPERATION,
      view: scratch.startMarkers,
      type: 'u32',
      value: '0u'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-row-start-markers`,
      operation: DISTANCE_FIELD_OPERATION,
      variant: 'exact-row-start-markers',
      bindings: [
        {name: 'entryStarts', view: scratch.entryStarts, type: 'u32', access: 'read'},
        {name: 'startMarkers', view: scratch.startMarkers, type: 'u32', access: 'read_write'}
      ],
      invocationCount: cellCount,
      declarations: markerDeclarations,
      // Slots are column * height + row; envelope starts are unique within a row.
      body: /* wgsl */ `let start = entryStarts[entryStartsOffset + index];
  if (start == NONE) { return; }
  let column = index / GRID_HEIGHT;
  let row = index % GRID_HEIGHT;
  startMarkers[startMarkersOffset + row * GRID_WIDTH + start] = column + 1u;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-row-marker-bands`,
      operation: DISTANCE_FIELD_OPERATION,
      variant: 'exact-row-marker-bands',
      bindings: [
        {name: 'startMarkers', view: scratch.startMarkers, type: 'u32', access: 'read'},
        {name: 'bandMarkers', view: scratch.bandMarkers, type: 'u32', access: 'read_write'}
      ],
      invocationCount: bandInvocationCount,
      declarations: markerDeclarations,
      body: /* wgsl */ `let row = index % GRID_HEIGHT;
  let columnStart = (index / GRID_HEIGHT) * BAND_SIZE;
  let columnEnd = min(columnStart + BAND_SIZE, GRID_WIDTH);
  var maximum = 0u;
  for (var x = columnStart; x < columnEnd; x++) {
    maximum = max(maximum, startMarkers[startMarkersOffset + row * GRID_WIDTH + x]);
  }
  bandMarkers[bandMarkersOffset + index] = maximum;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-row-marker-carry`,
      operation: DISTANCE_FIELD_OPERATION,
      variant: 'exact-row-marker-carry',
      bindings: [
        {name: 'bandMarkers', view: scratch.bandMarkers, type: 'u32', access: 'read_write'}
      ],
      invocationCount: grid.height,
      workgroupSize: LINE_WORKGROUP_SIZE,
      declarations: markerDeclarations,
      // In place: each band receives the maximum marker of the bands before it.
      body: /* wgsl */ `var running = 0u;
  for (var band = 0u; band < ROW_BAND_COUNT; band++) {
    let slot = bandMarkersOffset + band * GRID_HEIGHT + index;
    let bandMaximum = bandMarkers[slot];
    bandMarkers[slot] = running;
    running = max(running, bandMaximum);
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-row-color`,
      operation: DISTANCE_FIELD_OPERATION,
      variant: 'exact-row-color',
      bindings: [
        {name: 'columnRows', view: props.columnRows, type: 'u32', access: 'read'},
        {name: 'columnIds', view: props.columnIds, type: 'u32', access: 'read'},
        {name: 'startMarkers', view: scratch.startMarkers, type: 'u32', access: 'read'},
        {name: 'bandMarkers', view: scratch.bandMarkers, type: 'u32', access: 'read'},
        {name: 'nearestCells', view: props.nearestCells, type: 'u32', access: 'read_write'},
        {name: 'nearestIds', view: props.nearestIds, type: 'u32', access: 'read_write'}
      ],
      invocationCount: bandInvocationCount,
      declarations: markerDeclarations,
      body: /* wgsl */ `let row = index % GRID_HEIGHT;
  let rowBase = row * GRID_WIDTH;
  let columnStart = (index / GRID_HEIGHT) * BAND_SIZE;
  let columnEnd = min(columnStart + BAND_SIZE, GRID_WIDTH);
  var running = bandMarkers[bandMarkersOffset + index];
  for (var x = columnStart; x < columnEnd; x++) {
    let cell = rowBase + x;
    running = max(running, startMarkers[startMarkersOffset + cell]);
    if (running == 0u) {
      nearestCells[nearestCellsOffset + cell] = NONE;
      nearestIds[nearestIdsOffset + cell] = NONE;
      continue;
    }
    let column = running - 1u;
    nearestCells[nearestCellsOffset + cell] = columnRows[columnRowsOffset + rowBase + column] * GRID_WIDTH + column;
    nearestIds[nearestIdsOffset + cell] = columnIds[columnIdsOffset + rowBase + column];
  }`
    })
  );
  return nodes;
}

/**
 * Returns jump-flood step lengths: `N/2, N/4, ..., 1` with `N` the next power of two of the larger
 * grid side, followed by `2, 1` or `1` refinement passes.
 *
 * @internal
 */
export function getDistanceFieldJumpFloodSteps(
  grid: DistanceFieldGrid,
  refinementIterationCount: number
): number[] {
  const steps: number[] = [];
  let size = 1;
  while (size < Math.max(grid.width, grid.height)) {
    size *= 2;
  }
  for (let step = size / 2; step >= 1; step /= 2) {
    steps.push(step);
  }
  if (refinementIterationCount === 2) {
    steps.push(2, 1);
  } else if (refinementIterationCount === 1) {
    steps.push(1);
  }
  return steps;
}

/**
 * Builds the jump-flood initialization and step nodes, ping-ponging between two transient pairs.
 * Returns the pair that holds the final result.
 *
 * @internal
 */
export function createDistanceFieldJumpFloodNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    grid: DistanceFieldGrid;
    settings: GraphDataView<'float32'>;
    cellSeeds: GraphDataView<'uint32'>;
    buffers: readonly [
      {cells: GraphDataView<'uint32'>; ids: GraphDataView<'uint32'>},
      {cells: GraphDataView<'uint32'>; ids: GraphDataView<'uint32'>}
    ];
    refinementIterationCount: number;
  }
): {
  nodes: GPUCommandNode<Parameters>[];
  result: {cells: GraphDataView<'uint32'>; ids: GraphDataView<'uint32'>};
} {
  const {id, grid, buffers} = props;
  const cellCount = grid.width * grid.height;
  const nodes = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-jump-flood-initialize`,
      operation: DISTANCE_FIELD_OPERATION,
      variant: 'jump-flood-initialize',
      bindings: [
        {
          name: 'cellSeeds',
          view: props.cellSeeds,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'targetCells',
          view: buffers[0].cells,
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'targetIds',
          view: buffers[0].ids,
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: cellCount,
      body: /* wgsl */ `let seedId = cellSeeds[cellSeedsOffset + index];
  targetCells[targetCellsOffset + index] = select(${GPU_DISTANCE_FIELD_NONE}u, index, seedId != ${GPU_DISTANCE_FIELD_NONE}u);
  targetIds[targetIdsOffset + index] = seedId;`
    })
  ];
  const steps = getDistanceFieldJumpFloodSteps(grid, props.refinementIterationCount);
  // Step k's offset lives at element k * 64 (256 bytes, so every step binds at element offset 0):
  // all jump-flood steps share one WGSL source (library rule 12).
  const stepWords = createTransientView(
    graph,
    `${id}-jump-flood-step-words`,
    'uint32',
    steps.length * 64
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-jump-flood-steps`,
      operation: DISTANCE_FIELD_OPERATION,
      variant: 'jump-flood-steps',
      bindings: [{name: 'stepWords', view: stepWords, type: 'u32', access: 'read_write'}],
      invocationCount: 1,
      declarations: `const STEPS = array<u32, ${steps.length}>(${steps.map(step => `${step}u`).join(', ')});`,
      body: `for (var stepIndex = 0u; stepIndex < ${steps.length}u; stepIndex++) {
    stepWords[stepWordsOffset + stepIndex * 64u] = STEPS[stepIndex];
  }`
    })
  );
  let source = buffers[0];
  for (const [stepIndex] of steps.entries()) {
    const target = source === buffers[0] ? buffers[1] : buffers[0];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-jump-flood-${stepIndex}`,
        operation: DISTANCE_FIELD_OPERATION,
        variant: 'jump-flood-step',
        bindings: [
          {
            name: 'settings',
            view: props.settings,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'sourceCells',
            view: source.cells,
            type: 'u32',
            access: 'read'
          },
          {name: 'sourceIds', view: source.ids, type: 'u32', access: 'read'},
          {
            name: 'targetCells',
            view: target.cells,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'targetIds',
            view: target.ids,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'jumpStep',
            view: graph.createDataView(stepWords.buffer, {
              format: 'uint32',
              length: 1,
              byteOffset: stepWords.byteOffset + stepIndex * 64 * 4
            }),
            type: 'u32',
            access: 'read'
          }
        ],
        invocationCount: cellCount,
        declarations: getDistanceFieldGridWGSL(grid, true),
        body: /* wgsl */ `let step = i32(jumpStep[jumpStepOffset]);
  let x = index % GRID_WIDTH;
  let y = index / GRID_WIDTH;
  var bestCell = sourceCells[sourceCellsOffset + index];
  var bestId = sourceIds[sourceIdsOffset + index];
  for (var offsetY = -1; offsetY <= 1; offsetY++) {
    for (var offsetX = -1; offsetX <= 1; offsetX++) {
      let neighborX = i32(x) + offsetX * step;
      let neighborY = i32(y) + offsetY * step;
      if ((offsetX == 0 && offsetY == 0) || neighborX < 0 || neighborY < 0 ||
          neighborX >= i32(GRID_WIDTH) || neighborY >= i32(GRID_HEIGHT)) { continue; }
      let neighbor = u32(neighborY) * GRID_WIDTH + u32(neighborX);
      let candidateCell = sourceCells[sourceCellsOffset + neighbor];
      if (candidateCell == NONE) { continue; }
      let candidateId = sourceIds[sourceIdsOffset + neighbor];
      let candidateDx = getAbsoluteDifference(x, candidateCell % GRID_WIDTH);
      let candidateDy = getAbsoluteDifference(y, candidateCell / GRID_WIDTH);
      if (bestCell == NONE || isCloser(
        candidateDx, candidateDy, candidateId,
        getAbsoluteDifference(x, bestCell % GRID_WIDTH), getAbsoluteDifference(y, bestCell / GRID_WIDTH), bestId
      )) {
        bestCell = candidateCell;
        bestId = candidateId;
      }
    }
  }
  targetCells[targetCellsOffset + index] = bestCell;
  targetIds[targetIdsOffset + index] = bestId;`
      })
    );
    source = target;
  }
  return {nodes, result: source};
}

/**
 * Converts per-cell nearest seed cells and IDs into ground distances, applies the per-frame
 * `maximumDistance`, and writes the caller's outputs.
 *
 * @internal
 */
export function createDistanceFieldFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    grid: DistanceFieldGrid;
    settings: GraphDataView<'float32'>;
    nearestCells: GraphDataView<'uint32'>;
    nearestIds: GraphDataView<'uint32'>;
    distances: GraphDataView<'float32'>;
    allocation?: GraphDataView<'uint32'>;
    nearestCellsOutput?: GraphDataView<'uint32'>;
    withinDistance?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {grid} = props;
  const bindings: WGSLKernelBinding[] = [
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
    {
      name: 'nearestCells',
      view: props.nearestCells,
      type: 'u32',
      access: 'read'
    },
    {name: 'nearestIds', view: props.nearestIds, type: 'u32', access: 'read'},
    {
      name: 'distanceBits',
      view: props.distances,
      type: 'u32',
      access: 'read_write'
    }
  ];
  if (props.allocation) {
    bindings.push({
      name: 'allocation',
      view: props.allocation,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.nearestCellsOutput) {
    bindings.push({
      name: 'nearestCellsOutput',
      view: props.nearestCellsOutput,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.withinDistance) {
    bindings.push({
      name: 'withinDistance',
      view: props.withinDistance,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-finalize`,
    operation: DISTANCE_FIELD_OPERATION,
    variant: 'finalize',
    bindings,
    invocationCount: grid.width * grid.height,
    declarations: `${getDistanceFieldGridWGSL(grid, true)}
${ROUNDED_SQUARE_ROOT_WGSL}`,
    body: /* wgsl */ `var nearestCell = nearestCells[nearestCellsOffset + index];
  var seedId = nearestIds[nearestIdsOffset + index];
  var bits = 0x7f800000u;
  if (nearestCell != NONE) {
    let dx = getAbsoluteDifference(index % GRID_WIDTH, nearestCell % GRID_WIDTH);
    let dy = getAbsoluteDifference(index / GRID_WIDTH, nearestCell / GRID_WIDTH);
    var distance: f32;
    if (isIsotropic()) {
      // Exact integer key, so the only roundings are the square root and the scale.
      distance = settings[settingsOffset + 2u] * getRoundedSquareRoot(dx * dx + dy * dy, 0);
    } else {
      let key = getAnisotropicKey(dx, dy);
      let keyBits = bitcast<u32>(key);
      let keyExponent = (keyBits >> 23u) & 0xffu;
      if (keyExponent == 0u || keyExponent == 0xffu) {
        distance = sqrt(key);
      } else {
        distance = getRoundedSquareRoot((keyBits & 0x7fffffu) | 0x800000u, i32(keyExponent) - 150);
      }
    }
    if (distance > settings[settingsOffset + 6u]) {
      nearestCell = NONE;
      seedId = NONE;
    } else {
      bits = bitcast<u32>(distance);
    }
  }
  if (nearestCell == NONE) { seedId = NONE; }
  distanceBits[distanceBitsOffset + index] = bits;
  ${props.allocation ? 'allocation[allocationOffset + index] = seedId;' : ''}
  ${props.nearestCellsOutput ? 'nearestCellsOutput[nearestCellsOutputOffset + index] = nearestCell;' : ''}
  ${props.withinDistance ? 'withinDistance[withinDistanceOffset + index] = select(0u, 1u, nearestCell != NONE);' : ''}`
  });
}
