// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
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

/** Workgroup size of the one-invocation-per-row and per-column envelope kernels. */
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
  const clearBindings: MapGraphKernelBinding[] = [
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
    createMapGraphKernelNode<Parameters>(graph, {
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
    const bindings: MapGraphKernelBinding[] = [
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
      createMapGraphKernelNode<Parameters>(graph, {
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

/**
 * Exact separable pass 1: one invocation per column finds the nearest seed row in the column
 * (smaller ID on equal row distance) with one forward and one backward scan.
 *
 * @internal
 */
export function createDistanceFieldColumnNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    grid: DistanceFieldGrid;
    cellSeeds: GraphDataView<'uint32'>;
    columnRows: GraphDataView<'uint32'>;
    columnIds: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: `${props.id}-columns`,
    operation: DISTANCE_FIELD_OPERATION,
    variant: 'exact-columns',
    bindings: [
      {name: 'cellSeeds', view: props.cellSeeds, type: 'u32', access: 'read'},
      {
        name: 'columnRows',
        view: props.columnRows,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'columnIds',
        view: props.columnIds,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: props.grid.width,
    workgroupSize: LINE_WORKGROUP_SIZE,
    declarations: getDistanceFieldGridWGSL(props.grid, false),
    body: /* wgsl */ `let column = index;
  var lastRow = NONE;
  var lastId = NONE;
  for (var row = 0u; row < GRID_HEIGHT; row++) {
    let cell = row * GRID_WIDTH + column;
    let seedId = cellSeeds[cellSeedsOffset + cell];
    if (seedId != NONE) {
      lastRow = row;
      lastId = seedId;
    }
    columnRows[columnRowsOffset + cell] = lastRow;
    columnIds[columnIdsOffset + cell] = lastId;
  }
  lastRow = NONE;
  lastId = NONE;
  for (var step = 0u; step < GRID_HEIGHT; step++) {
    let row = GRID_HEIGHT - 1u - step;
    let cell = row * GRID_WIDTH + column;
    let seedId = cellSeeds[cellSeedsOffset + cell];
    if (seedId != NONE) {
      lastRow = row;
      lastId = seedId;
    }
    if (lastRow == NONE) { continue; }
    let forwardRow = columnRows[columnRowsOffset + cell];
    var useBackward = forwardRow == NONE;
    if (!useBackward) {
      let forwardDistance = row - forwardRow;
      let backwardDistance = lastRow - row;
      useBackward = backwardDistance < forwardDistance ||
        (backwardDistance == forwardDistance && lastId < columnIds[columnIdsOffset + cell]);
    }
    if (useBackward) {
      columnRows[columnRowsOffset + cell] = lastRow;
      columnIds[columnIdsOffset + cell] = lastId;
    }
  }`
  });
}

/**
 * Exact separable pass 2: one invocation per row builds the discrete lower envelope of the
 * per-column candidates (Felzenszwalb-Huttenlocher) and assigns every cell its winner.
 *
 * Each stack entry stores a column and the first cell where it wins. Because a candidate further
 * right beats one further left on a suffix of the row (the key difference is linear in x), the
 * pop test is one comparison at the top entry's start, and the new start is found by binary
 * search with `isCloser`, so ties break exactly on the smallest seed ID and no division is used.
 * The stack lives in global scratch laid out `entry * height + row` for coalesced access.
 *
 * @internal
 */
export function createDistanceFieldRowNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    grid: DistanceFieldGrid;
    settings: GraphDataView<'float32'>;
    columnRows: GraphDataView<'uint32'>;
    columnIds: GraphDataView<'uint32'>;
    stackColumns: GraphDataView<'uint32'>;
    stackStarts: GraphDataView<'uint32'>;
    nearestCells: GraphDataView<'uint32'>;
    nearestIds: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: `${props.id}-rows`,
    operation: DISTANCE_FIELD_OPERATION,
    variant: 'exact-rows',
    bindings: [
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {
        name: 'columnRows',
        view: props.columnRows,
        type: 'u32',
        access: 'read'
      },
      {name: 'columnIds', view: props.columnIds, type: 'u32', access: 'read'},
      {
        name: 'stackColumns',
        view: props.stackColumns,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'stackStarts',
        view: props.stackStarts,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'nearestCells',
        view: props.nearestCells,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'nearestIds',
        view: props.nearestIds,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: props.grid.height,
    workgroupSize: LINE_WORKGROUP_SIZE,
    declarations: /* wgsl */ `${getDistanceFieldGridWGSL(props.grid, true)}

/** Whether the candidate in column q beats the candidate in column v at cell x of this row. */
fn beatsAt(q: u32, qDy: u32, qId: u32, v: u32, vDy: u32, vId: u32, x: u32) -> bool {
  return isCloser(getAbsoluteDifference(x, q), qDy, qId, getAbsoluteDifference(x, v), vDy, vId);
}`,
    body: /* wgsl */ `let row = index;
  let rowBase = row * GRID_WIDTH;
  var count = 0u;
  for (var q = 0u; q < GRID_WIDTH; q++) {
    let qRow = columnRows[columnRowsOffset + rowBase + q];
    if (qRow == NONE) { continue; }
    let qDy = getAbsoluteDifference(qRow, row);
    let qId = columnIds[columnIdsOffset + rowBase + q];
    var start = 0u;
    loop {
      if (count == 0u) {
        start = 0u;
        break;
      }
      let top = (count - 1u) * GRID_HEIGHT + row;
      let v = stackColumns[stackColumnsOffset + top];
      let vStart = stackStarts[stackStartsOffset + top];
      let vDy = getAbsoluteDifference(columnRows[columnRowsOffset + rowBase + v], row);
      let vId = columnIds[columnIdsOffset + rowBase + v];
      if (beatsAt(q, qDy, qId, v, vDy, vId, vStart)) {
        count = count - 1u;
        continue;
      }
      // First cell in (vStart, GRID_WIDTH) where q wins, or GRID_WIDTH when q never wins.
      var low = vStart + 1u;
      var high = GRID_WIDTH;
      while (low < high) {
        let middle = (low + high) / 2u;
        if (beatsAt(q, qDy, qId, v, vDy, vId, middle)) {
          high = middle;
        } else {
          low = middle + 1u;
        }
      }
      start = low;
      break;
    }
    if (start < GRID_WIDTH) {
      stackColumns[stackColumnsOffset + count * GRID_HEIGHT + row] = q;
      stackStarts[stackStartsOffset + count * GRID_HEIGHT + row] = start;
      count = count + 1u;
    }
  }
  var entry = 0u;
  for (var x = 0u; x < GRID_WIDTH; x++) {
    let cell = rowBase + x;
    if (count == 0u) {
      nearestCells[nearestCellsOffset + cell] = NONE;
      nearestIds[nearestIdsOffset + cell] = NONE;
      continue;
    }
    while (entry + 1u < count && stackStarts[stackStartsOffset + (entry + 1u) * GRID_HEIGHT + row] <= x) {
      entry = entry + 1u;
    }
    let column = stackColumns[stackColumnsOffset + entry * GRID_HEIGHT + row];
    nearestCells[nearestCellsOffset + cell] = columnRows[columnRowsOffset + rowBase + column] * GRID_WIDTH + column;
    nearestIds[nearestIdsOffset + cell] = columnIds[columnIdsOffset + rowBase + column];
  }`
  });
}

/**
 * Returns jump-flood step lengths: `N/2, N/4, ..., 1` with `N` the next power of two of the larger
 * grid side, followed by `2, 1` or `1` refinement passes.
 *
 * @internal
 */
export function getDistanceFieldJumpFloodSteps(
  grid: DistanceFieldGrid,
  refinementPasses: number
): number[] {
  const steps: number[] = [];
  let size = 1;
  while (size < Math.max(grid.width, grid.height)) {
    size *= 2;
  }
  for (let step = size / 2; step >= 1; step /= 2) {
    steps.push(step);
  }
  if (refinementPasses === 2) {
    steps.push(2, 1);
  } else if (refinementPasses === 1) {
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
    refinementPasses: number;
  }
): {
  nodes: GPUCommandNode<Parameters>[];
  result: {cells: GraphDataView<'uint32'>; ids: GraphDataView<'uint32'>};
} {
  const {id, grid, buffers} = props;
  const cellCount = grid.width * grid.height;
  const nodes = [
    createMapGraphKernelNode<Parameters>(graph, {
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
  const steps = getDistanceFieldJumpFloodSteps(grid, props.refinementPasses);
  let source = buffers[0];
  for (const [stepIndex, step] of steps.entries()) {
    const target = source === buffers[0] ? buffers[1] : buffers[0];
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
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
          }
        ],
        invocationCount: cellCount,
        declarations: `${getDistanceFieldGridWGSL(grid, true)}
const STEP: i32 = ${step};`,
        body: /* wgsl */ `let x = index % GRID_WIDTH;
  let y = index / GRID_WIDTH;
  var bestCell = sourceCells[sourceCellsOffset + index];
  var bestId = sourceIds[sourceIdsOffset + index];
  for (var offsetY = -1; offsetY <= 1; offsetY++) {
    for (var offsetX = -1; offsetX <= 1; offsetX++) {
      let neighborX = i32(x) + offsetX * STEP;
      let neighborY = i32(y) + offsetY * STEP;
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
 * `maxDistance`, and writes the caller's outputs.
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
  const bindings: MapGraphKernelBinding[] = [
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
  return createMapGraphKernelNode<Parameters>(graph, {
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
