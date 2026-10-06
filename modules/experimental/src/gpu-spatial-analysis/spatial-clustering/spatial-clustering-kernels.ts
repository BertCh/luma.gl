// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPUSpatialClustering';

/** Views shared by every kernel that walks the cell grid. @internal */
export type SpatialClusteringGridViews = {
  positions: GraphDataView<'float32x2'>;
  parameters: GraphDataView<'float32'>;
  sortedRows: GraphDataView<'uint32'>;
  /** One-row count of valid points: the prefix of `sortedRows` that holds rows. */
  gridCount: GraphDataView<'uint32'>;
  /** `cellCount + 1` exclusive offsets from `GPUGridIndex` over the full `COLUMNS x ROWS` lattice. */
  cellOffsets: GraphDataView<'uint32'>;
};

/**
 * WGSL shared by every spatial-clustering kernel: the lattice derivation, validity test, and cell
 * lookup. Kernels that include it must bind `positions` and `parameters` as `array<f32>`.
 *
 * Each encoding derives the active lattice from the per-frame bounds and epsilon so every cell is
 * at least epsilon wide (with a 2^-10 relative margin against rounding), which makes the 3x3 cell
 * neighborhood of a point always contain all of its epsilon-neighbors.
 *
 * @internal
 */
export function getSpatialClusteringSharedWGSL(columns: number, rows: number): string {
  return /* wgsl */ `
const COLUMNS: u32 = ${columns}u;
const ROWS: u32 = ${rows}u;
const NOISE: u32 = 0xffffffffu;
const CELL_MARGIN: f32 = 1.0009765625;

struct Lattice {
  valid: bool,
  minimumX: f32,
  minimumY: f32,
  maximumX: f32,
  maximumY: f32,
  cellWidth: f32,
  cellHeight: f32,
  columns: u32,
  rows: u32,
  epsilonSquared: f32,
  minimumPoints: u32
}

fn isFiniteFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn readLattice() -> Lattice {
  var lattice: Lattice;
  lattice.cellWidth = 1.0;
  lattice.cellHeight = 1.0;
  lattice.columns = 1u;
  lattice.rows = 1u;
  lattice.epsilonSquared = 0.0;
  lattice.minimumPoints = 1u;
  let minimumX = parameters[parametersOffset];
  let minimumY = parameters[parametersOffset + 1u];
  let maximumX = parameters[parametersOffset + 2u];
  let maximumY = parameters[parametersOffset + 3u];
  let epsilon = parameters[parametersOffset + 4u];
  let minimumPointsValue = parameters[parametersOffset + 5u];
  let width = maximumX - minimumX;
  let height = maximumY - minimumY;
  let epsilonSquared = epsilon * epsilon;
  let cellEpsilon = epsilon * CELL_MARGIN;
  lattice.valid =
    isFiniteFloat(minimumX) && isFiniteFloat(minimumY) && isFiniteFloat(maximumX) &&
    isFiniteFloat(maximumY) && isFiniteFloat(epsilon) && isFiniteFloat(minimumPointsValue) &&
    isFiniteFloat(width) && isFiniteFloat(height) && isFiniteFloat(epsilonSquared) &&
    isFiniteFloat(cellEpsilon) && epsilon > 0.0 && width >= 0.0 && height >= 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  lattice.minimumX = minimumX;
  lattice.minimumY = minimumY;
  lattice.maximumX = maximumX;
  lattice.maximumY = maximumY;
  lattice.cellWidth = max(width / f32(COLUMNS), cellEpsilon);
  lattice.cellHeight = max(height / f32(ROWS), cellEpsilon);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
  lattice.epsilonSquared = epsilonSquared;
  lattice.minimumPoints = u32(max(minimumPointsValue, 0.0));
  return lattice;
}

fn isPointValid(lattice: Lattice, x: f32, y: f32) -> bool {
  return lattice.valid && isFiniteFloat(x) && isFiniteFloat(y) &&
    x >= lattice.minimumX && x <= lattice.maximumX &&
    y >= lattice.minimumY && y <= lattice.maximumY;
}

fn getCellColumn(lattice: Lattice, x: f32) -> u32 {
  return min(u32(floor((x - lattice.minimumX) / lattice.cellWidth)), lattice.columns - 1u);
}

fn getCellRow(lattice: Lattice, y: f32) -> u32 {
  return min(u32(floor((y - lattice.minimumY) / lattice.cellHeight)), lattice.rows - 1u);
}
`;
}

/**
 * WGSL statements that visit every valid point within epsilon of `(x, y)` (the point itself
 * included) through the 3x3 cell neighborhood and run `action` with `neighbor` (row index) in
 * scope. Requires bindings `positions`, `sortedRows`, `cellOffsets` and a `lattice`
 * local. `keepGoing` is an extra loop condition that lets the caller stop early.
 */
function getNeighborLoopWGSL(action: string, keepGoing: string = 'true'): string {
  return /* wgsl */ `
  let column = getCellColumn(lattice, x);
  let row = getCellRow(lattice, y);
  let firstColumn = max(column, 1u) - 1u;
  let lastColumn = min(column + 1u, lattice.columns - 1u);
  let firstRow = max(row, 1u) - 1u;
  let lastRow = min(row + 1u, lattice.rows - 1u);
  for (var cellRow = firstRow; cellRow <= lastRow && ${keepGoing}; cellRow++) {
    for (var cellColumn = firstColumn; cellColumn <= lastColumn && ${keepGoing}; cellColumn++) {
      let cell = cellRow * COLUMNS + cellColumn;
      let start = cellOffsets[cellOffsetsOffset + cell];
      let end = cellOffsets[cellOffsetsOffset + cell + 1u];
      for (var slot = start; slot < end && ${keepGoing}; slot++) {
        let neighbor = sortedRows[sortedRowsOffset + slot];
        let deltaX = positions[positionsOffset + neighbor * 2u] - x;
        let deltaY = positions[positionsOffset + neighbor * 2u + 1u] - y;
        if (deltaX * deltaX + deltaY * deltaY <= lattice.epsilonSquared) {
          ${action}
        }
      }
    }
  }`;
}

/**
 * Maps the thread to a point: threads walk the valid points in grid-cell order (the order of
 * `sortedRows`), so neighboring threads search neighboring cells whatever the input order. Defines
 * `point`, the row the thread owns; surplus threads (excluded points) return. Results are written
 * by row, so they do not depend on the mapping.
 */
const THREAD_TO_POINT_WGSL = `if (index >= gridCount[gridCountOffset]) {
    return;
  }
  let point = sortedRows[sortedRowsOffset + index];`;

function getGridBindings(views: SpatialClusteringGridViews): WGSLKernelBinding[] {
  return [
    {name: 'positions', view: views.positions, type: 'f32', access: 'read'},
    {name: 'parameters', view: views.parameters, type: 'f32', access: 'read'},
    {name: 'sortedRows', view: views.sortedRows, type: 'u32', access: 'read'},
    {name: 'cellOffsets', view: views.cellOffsets, type: 'u32', access: 'read'},
    {name: 'gridCount', view: views.gridCount, type: 'u32', access: 'read'}
  ];
}

/**
 * Writes the `[minX, minY, maxX, maxY]` domain `GPUGridIndex` reads at run time: the per-frame
 * lattice extended to `COLUMNS x ROWS` whole cells, or NaN (so nothing is indexed) when the
 * parameters are invalid. @internal
 */
export function createSpatialClusteringGridBoundsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    parameters: GraphDataView<'float32'>;
    gridSize: readonly [number, number];
    gridBounds: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'grid-bounds',
    bindings: [
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'gridBounds', view: props.gridBounds, type: 'f32', access: 'read_write'}
    ],
    invocationCount: 1,
    declarations: getSpatialClusteringSharedWGSL(props.gridSize[0], props.gridSize[1]),
    body: `let lattice = readLattice();
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridBounds[gridBoundsOffset] = select(invalid, lattice.minimumX, lattice.valid);
  gridBounds[gridBoundsOffset + 1u] = select(invalid, lattice.minimumY, lattice.valid);
  gridBounds[gridBoundsOffset + 2u] = select(invalid,
    max(lattice.maximumX, lattice.minimumX + lattice.cellWidth * f32(COLUMNS)), lattice.valid);
  gridBounds[gridBoundsOffset + 3u] = select(invalid,
    max(lattice.maximumY, lattice.minimumY + lattice.cellHeight * f32(ROWS)), lattice.valid);`
  });
}

/**
 * Copies every valid point and writes NaN for excluded ones, which `GPUGridIndex` ignores. The
 * index domain is wider than the valid extent, so it cannot reject them itself. Also writes the
 * identity row IDs used when compacting. @internal
 */
export function createSpatialClusteringGridPositionsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    parameters: GraphDataView<'float32'>;
    gridSize: readonly [number, number];
    gridPositions: GraphDataView<'float32x2'>;
    rowIds: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'grid-positions',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'gridPositions', view: props.gridPositions, type: 'f32', access: 'read_write'},
      {name: 'rowIds', view: props.rowIds, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: getSpatialClusteringSharedWGSL(props.gridSize[0], props.gridSize[1]),
    body: `let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let valid = isPointValid(lattice, x, y);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridPositions[gridPositionsOffset + index * 2u] = select(invalid, x, valid);
  gridPositions[gridPositionsOffset + index * 2u + 1u] = select(invalid, y, valid);
  rowIds[rowIdsOffset + index] = index;`
  });
}

/**
 * Writes the core flag of every point: valid and at least `minimumPoints` neighbors. With
 * `boxCounts`, a point whose dense box already holds `minimumPoints` points is core without a
 * neighbor scan (every point of a box is within epsilon of every other). @internal
 */
export function createSpatialClusteringCoreNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SpatialClusteringGridViews & {
    id: string;
    gridSize: readonly [number, number];
    coreFlags: GraphDataView<'uint32'>;
    boxCounts?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {boxCounts} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: boxCounts ? 'core-dense-box' : 'core',
    bindings: [
      ...getGridBindings(props),
      {name: 'coreFlags', view: props.coreFlags, type: 'u32', access: 'read_write'},
      ...(boxCounts
        ? [{name: 'boxCounts', view: boxCounts, type: 'u32' as const, access: 'read' as const}]
        : [])
    ],
    invocationCount: props.positions.length,
    declarations: getSpatialClusteringSharedWGSL(props.gridSize[0], props.gridSize[1]),
    body: `${THREAD_TO_POINT_WGSL}
  let lattice = readLattice();
  let x = positions[positionsOffset + point * 2u];
  let y = positions[positionsOffset + point * 2u + 1u];
  var isCore = 0u;
  if (isPointValid(lattice, x, y)) {
    ${
      boxCounts
        ? `if (boxCounts[boxCountsOffset + point] >= lattice.minimumPoints) {
      isCore = 1u;
    } else {`
        : '{'
    }
      var neighborCount = 0u;
      ${getNeighborLoopWGSL('neighborCount++;', 'neighborCount < lattice.minimumPoints')}
      isCore = select(0u, 1u, neighborCount >= lattice.minimumPoints);
    }
  }
  coreFlags[coreFlagsOffset + point] = isCore;`
  });
}

/**
 * Writes the dense-box key `(boxX, boxY)` of every valid point, a virtual grid with side
 * `0.7 * epsilon` (diagonal `0.99 * epsilon`), or the invalid key `(0xffffffff, 0xffffffff)` for
 * excluded points and for points whose box index would exceed 2^16 (where the f32 quotient could
 * misplace a point by a noticeable fraction of a box). @internal
 */
export function createSpatialClusteringBoxKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    parameters: GraphDataView<'float32'>;
    gridSize: readonly [number, number];
    boxHigh: GraphDataView<'uint32'>;
    boxLow: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'box-keys',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'boxHigh', view: props.boxHigh, type: 'u32', access: 'read_write'},
      {name: 'boxLow', view: props.boxLow, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: `${getSpatialClusteringSharedWGSL(props.gridSize[0], props.gridSize[1])}
const BOX_SIDE_FACTOR: f32 = 0.7;
const MAXIMUM_BOX_INDEX: f32 = 65536.0;`,
    body: `let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var high = 0xffffffffu;
  var low = 0xffffffffu;
  if (isPointValid(lattice, x, y)) {
    let side = parameters[parametersOffset + 4u] * BOX_SIDE_FACTOR;
    let boxX = floor((x - lattice.minimumX) / side);
    let boxY = floor((y - lattice.minimumY) / side);
    if (boxX >= 0.0 && boxX < MAXIMUM_BOX_INDEX && boxY >= 0.0 && boxY < MAXIMUM_BOX_INDEX) {
      high = u32(boxX);
      low = u32(boxY);
    }
  }
  boxHigh[boxHighOffset + index] = high;
  boxLow[boxLowOffset + index] = low;`
  });
}

/**
 * Writes, per point, the number of points in its dense box (0 for points without a box). Runs over
 * the sorted box order of `getKeyPairSortNodes` and `getKeyGroupNodes`. @internal
 */
export function createSpatialClusteringBoxCountsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    sortedItems: GraphDataView<'uint32'>;
    groupIndex: GraphDataView<'uint32'>;
    groupStarts: GraphDataView<'uint32'>;
    boxHigh: GraphDataView<'uint32'>;
    boxCounts: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'box-counts',
    bindings: [
      {name: 'sortedItems', view: props.sortedItems, type: 'u32', access: 'read'},
      {name: 'groupIndex', view: props.groupIndex, type: 'u32', access: 'read'},
      {name: 'groupStarts', view: props.groupStarts, type: 'u32', access: 'read'},
      {name: 'boxHigh', view: props.boxHigh, type: 'u32', access: 'read'},
      {name: 'boxCounts', view: props.boxCounts, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.sortedItems.length,
    body: `let item = sortedItems[sortedItemsOffset + index];
  let group = groupIndex[groupIndexOffset + index] - 1u;
  let count = groupStarts[groupStartsOffset + group + 1u] - groupStarts[groupStartsOffset + group];
  boxCounts[boxCountsOffset + item] = select(0u, count, boxHigh[boxHighOffset + item] != 0xffffffffu);`
  });
}

/** Sets `parents[i] = i`. @internal */
export function createSpatialClusteringParentsInitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; parents: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'parents-init',
    bindings: [{name: 'parents', view: props.parents, type: 'u32', access: 'read_write'}],
    invocationCount: props.parents.length,
    body: 'parents[parentsOffset + index] = index;'
  });
}

/**
 * Lock-free union-find over core points. Every core point unites with its core neighbors of
 * smaller row. A root is only ever hooked under a smaller root, so the final root of each
 * component is its smallest core row.
 *
 * @internal
 */
export function createSpatialClusteringUnionNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SpatialClusteringGridViews & {
    id: string;
    gridSize: readonly [number, number];
    coreFlags: GraphDataView<'uint32'>;
    parents: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'union',
    bindings: [
      ...getGridBindings(props),
      {name: 'coreFlags', view: props.coreFlags, type: 'u32', access: 'read'},
      {name: 'parents', view: props.parents, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: `${getSpatialClusteringSharedWGSL(props.gridSize[0], props.gridSize[1])}
// Path halving: a visited node is a non-root forever, so any ancestor is a valid parent for it.
fn findRoot(start: u32) -> u32 {
  var current = start;
  loop {
    let parent = atomicLoad(&parents[parentsOffset + current]);
    if (parent == current) {
      return current;
    }
    let grandparent = atomicLoad(&parents[parentsOffset + parent]);
    if (grandparent != parent) {
      atomicStore(&parents[parentsOffset + current], grandparent);
    }
    current = parent;
  }
}

// Hooks the larger root under the smaller one; retries with fresh roots when another invocation
// hooked the larger root first (or the weak compare-exchange failed spuriously).
fn unite(first: u32, second: u32) {
  var a = findRoot(first);
  var b = findRoot(second);
  loop {
    if (a == b) {
      return;
    }
    let low = min(a, b);
    let high = max(a, b);
    let result = atomicCompareExchangeWeak(&parents[parentsOffset + high], high, low);
    if (result.exchanged) {
      return;
    }
    a = findRoot(low);
    b = findRoot(high);
  }
}`,
    body: `${THREAD_TO_POINT_WGSL}
  if (coreFlags[coreFlagsOffset + point] == 0u) {
    return;
  }
  let lattice = readLattice();
  let x = positions[positionsOffset + point * 2u];
  let y = positions[positionsOffset + point * 2u + 1u];
  ${getNeighborLoopWGSL(`if (neighbor < point && coreFlags[coreFlagsOffset + neighbor] != 0u) {
            unite(point, neighbor);
          }`)}`
  });
}

/** Writes each core point's final union-find root and NOISE for every other point. @internal */
export function createSpatialClusteringResolveNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    coreFlags: GraphDataView<'uint32'>;
    parents: GraphDataView<'uint32'>;
    rootLabels: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'resolve',
    bindings: [
      {name: 'coreFlags', view: props.coreFlags, type: 'u32', access: 'read'},
      {name: 'parents', view: props.parents, type: 'u32', access: 'read'},
      {name: 'rootLabels', view: props.rootLabels, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.rootLabels.length,
    body: `var root = 0xffffffffu;
  if (coreFlags[coreFlagsOffset + index] != 0u) {
    root = index;
    loop {
      let parent = parents[parentsOffset + root];
      if (parent == root) {
        break;
      }
      root = parent;
    }
  }
  rootLabels[rootLabelsOffset + index] = root;`
  });
}

/**
 * Assigns every valid non-core point the smallest root among its core neighbors, or NOISE. Reads
 * only core entries of `rootLabels` and writes only its own non-core entry, so it is race free.
 *
 * @internal
 */
export function createSpatialClusteringBorderNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SpatialClusteringGridViews & {
    id: string;
    gridSize: readonly [number, number];
    coreFlags: GraphDataView<'uint32'>;
    rootLabels: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'border',
    bindings: [
      ...getGridBindings(props),
      {name: 'coreFlags', view: props.coreFlags, type: 'u32', access: 'read'},
      {name: 'rootLabels', view: props.rootLabels, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: getSpatialClusteringSharedWGSL(props.gridSize[0], props.gridSize[1]),
    body: `${THREAD_TO_POINT_WGSL}
  if (coreFlags[coreFlagsOffset + point] != 0u) {
    return;
  }
  let lattice = readLattice();
  let x = positions[positionsOffset + point * 2u];
  let y = positions[positionsOffset + point * 2u + 1u];
  if (!isPointValid(lattice, x, y)) {
    return;
  }
  var bestRoot = NOISE;
  ${getNeighborLoopWGSL(`if (coreFlags[coreFlagsOffset + neighbor] != 0u) {
            bestRoot = min(bestRoot, rootLabels[rootLabelsOffset + neighbor]);
          }`)}
  rootLabels[rootLabelsOffset + point] = bestRoot;`
  });
}

/** Writes `rootFlags[i] = 1` when point `i` is core and the root of its own cluster. @internal */
export function createSpatialClusteringRootFlagsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    coreFlags: GraphDataView<'uint32'>;
    rootLabels: GraphDataView<'uint32'>;
    rootFlags: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'root-flags',
    bindings: [
      {name: 'coreFlags', view: props.coreFlags, type: 'u32', access: 'read'},
      {name: 'rootLabels', view: props.rootLabels, type: 'u32', access: 'read'},
      {name: 'rootFlags', view: props.rootFlags, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.rootLabels.length,
    body: `let isRoot = coreFlags[coreFlagsOffset + index] != 0u &&
    rootLabels[rootLabelsOffset + index] == index;
  rootFlags[rootFlagsOffset + index] = select(0u, 1u, isRoot);`
  });
}

/** Writes compact labels and, when requested, the member coordinate columns. @internal */
export function createSpatialClusteringLabelsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    rootLabels: GraphDataView<'uint32'>;
    clusterOffsets: GraphDataView<'uint32'>;
    labels: GraphDataView<'uint32'>;
    memberXs?: GraphDataView<'float32'>;
    memberYs?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'rootLabels', view: props.rootLabels, type: 'u32', access: 'read'},
    {name: 'clusterOffsets', view: props.clusterOffsets, type: 'u32', access: 'read'},
    {name: 'labels', view: props.labels, type: 'u32', access: 'read_write'}
  ];
  const columns = props.memberXs && props.memberYs;
  if (props.memberXs && props.memberYs) {
    bindings.push(
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'memberXs', view: props.memberXs, type: 'f32', access: 'read_write'},
      {name: 'memberYs', view: props.memberYs, type: 'f32', access: 'read_write'}
    );
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'labels',
    bindings,
    invocationCount: props.rootLabels.length,
    body: `let root = rootLabels[rootLabelsOffset + index];
  labels[labelsOffset + index] = select(clusterOffsets[clusterOffsetsOffset + min(root, INVOCATION_COUNT - 1u)], 0xffffffffu, root == 0xffffffffu);
  ${
    columns
      ? `memberXs[memberXsOffset + index] = positions[positionsOffset + index * 2u];
  memberYs[memberYsOffset + index] = positions[positionsOffset + index * 2u + 1u];`
      : ''
  }`
  });
}

/** Divides coordinate sums by member counts; empty clusters get `(0, 0)`. @internal */
export function createSpatialClusteringCentroidsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    sizes: GraphDataView<'uint32'>;
    sumsX: GraphDataView<'float32'>;
    sumsY: GraphDataView<'float32'>;
    centroids: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'centroids',
    bindings: [
      {name: 'sizes', view: props.sizes, type: 'u32', access: 'read'},
      {name: 'sumsX', view: props.sumsX, type: 'f32', access: 'read'},
      {name: 'sumsY', view: props.sumsY, type: 'f32', access: 'read'},
      {name: 'centroids', view: props.centroids, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.sizes.length,
    body: `let size = sizes[sizesOffset + index];
  let divisor = f32(max(size, 1u));
  centroids[centroidsOffset + index * 2u] = select(0.0, sumsX[sumsXOffset + index] / divisor, size > 0u);
  centroids[centroidsOffset + index * 2u + 1u] = select(0.0, sumsY[sumsYOffset + index] / divisor, size > 0u);`
  });
}
