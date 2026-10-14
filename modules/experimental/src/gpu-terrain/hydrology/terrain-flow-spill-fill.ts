// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLDispatchGate, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  createContributorTransientView,
  getRasterGridWGSL
} from '../../gpu-raster/cost-distance/raster-grid-utils';
import {
  createRasterGatedLoopNodes,
  createRasterIterationGateNode,
  createRasterTiledRelaxation,
  createRasterTiledRelaxationNodes,
  getRasterRelaxationTileWGSL
} from '../../gpu-raster/cost-distance/raster-relaxation';
import {
  createTerrainFlowFillNodes,
  TERRAIN_FLOW_SWEEP_AFTER_ITERATION,
  type TerrainFlowGrid
} from './terrain-flow-passes';
import {createTerrainFlowSpillTreeNodes} from './terrain-flow-spill-tree';
import {
  createSpillLoopState,
  createSpillResetNode,
  getSpillBitLength,
  getSpillCommonWGSL,
  getSpillGateUse,
  getSpillRoundCount,
  SPILL_WORKGROUP_SIZE,
  type SpillGateUse
} from './terrain-flow-spill-utils';

const OPERATION = 'GPUTerrainFlow';

/*
 * Depression filling by a spill graph (Barnes 2016, "Parallel priority-flood depression filling
 * for trillion cell digital elevation models on desktops or clusters", tile-wise scheme).
 *
 * The filled surface is the minimax (bottleneck) elevation to the outlets: F(c) is the smallest
 * possible maximum elevation along a path of cells from c to a boundary cell, both ends included.
 * It is the unique fixpoint of F(c) = max(z(c), min over neighbors n of F(n)) with F = z on the
 * boundary, so any exact algorithm gives the same float32 values as the tiled Planchon-Darboux
 * relaxation (`createTerrainFlowFillNodes`). This module computes it in four stages without a
 * length-dependent loop:
 *
 * 1. Tile fill. One workgroup per T x T tile (T = 32, or 16 when the device offers less than 12 KiB
 *    of workgroup memory) fills the tile on its own, with every tile perimeter cell and every true
 *    boundary cell (grid edge, or next to an invalid cell) as an outlet that keeps its elevation.
 *    This yields L(c), the minimax elevation to an outlet of the tile. Then each cell takes as label
 *    the smallest outlet cell index that it can reach by a path whose cells all have L <= L(c).
 *    Both loops are monotone atomic-min fixpoints in workgroup memory, bounded by T * T passes.
 * 2. Spill edges. Every pair of adjacent valid cells with different labels is an edge between the
 *    two labels with weight max(L(a), L(b)); every boundary outlet has an edge of weight z to the
 *    ocean node. The edges are implicit (computed from the cell arrays on demand).
 * 3. Bottleneck solve. A Boruvka minimum spanning tree (at most floor(log2 n) + 2 rounds), then the
 *    tree is rooted at the ocean by an Euler tour ranked with Wyllie pointer jumping, and G, the
 *    maximum edge weight on the tree path to the ocean, is computed by pointer jumping with max.
 *    The path in a minimum spanning tree minimizes the maximum edge weight among all paths
 *    (the bottleneck property: Hu 1961; Gabow and Tarjan 1988), so G(label) is the minimax weight
 *    from the label to the ocean in the spill graph.
 * 4. F(c) = max(L(c), G(label(c))).
 *
 * Proof of stage 4. Upper bound: c reaches label(c) through cells whose L is at most L(c), hence
 * whose elevation is at most L(c) (the label chain); a spill edge (a, b) joins label(a) and label(b)
 * through a, b with elevation at most max(L(a), L(b)) (chains to both labels, and the pair itself);
 * so following the optimal label-graph path to the ocean gives a path of cells with maximum at
 * most max(L(c), G(label(c))). Lower bound: take a best path from c to a boundary cell b with
 * maximum M. Every cell x on it has L(x) <= M, because the part of the path inside x's tile up to
 * its first outlet is a tile path. Consecutive cells with different labels give a spill edge of
 * weight at most M, and b gives the ocean edge of weight z(b) <= M, so G(label(c)) <= M, and
 * L(c) <= M. Hence max(L(c), G(label(c))) <= M = F(c).
 *
 * Every label of a valid cell exists: take the cell with the smallest (L, tile-fill depth) that
 * has none; its tile-fill predecessor has L at most L(c) and a smaller depth (or a smaller L), so it
 * has a label that propagates to c. The label loop is a monotone minimum, so its fixpoint does not
 * depend on the order of the atomic updates, and neither does anything else: the spill fill is
 * deterministic and bit-identical to the relaxation (signed zeros aside: `-0` orders below `+0`).
 */

/** Edge length of the tile of stage 1 when workgroup memory allows it. @internal */
const LARGE_TILE_SIZE = 32;
/** Workgroup bytes of the 32 x 32 tile: key, label and category per cell plus two flags. @internal */
const LARGE_TILE_WORKGROUP_BYTES = 3 * 4 * LARGE_TILE_SIZE * LARGE_TILE_SIZE + 8;

/** Gate slots of the mode gate. */
const MODE_SLOT_TILES = 0;
const MODE_SLOT_NODES = 1;
const MODE_SLOT_HALF_EDGES = 2;
const MODE_SLOT_CELLS = 3;
const MODE_SLOT_RELAXATION_INIT = 4;

/** Gate slots of the Boruvka loop. */
const BORUVKA_SLOT_NODES = 0;
const BORUVKA_SLOT_EDGES = 1;

/** Returns the tile edge length (32 or 16) that fits the device's workgroup memory. @internal */
export function getTerrainFlowSpillTileSize<Parameters>(
  graph: GPUCommandGraph<Parameters>
): number {
  return graph.device.limits.maxComputeWorkgroupStorageSize >= LARGE_TILE_WORKGROUP_BYTES
    ? LARGE_TILE_SIZE
    : 16;
}

/**
 * Number of unrolled Boruvka rounds for `cellCount` cells: every round at least halves the number
 * of components minus the ocean's, so `floor(log2 n) + 1` rounds merge everything and one more
 * confirms that no component hooks. @internal
 */
export function getTerrainFlowSpillBoruvkaRoundCount(cellCount: number): number {
  return getSpillBitLength(Math.max(cellCount, 1)) + 1;
}

/**
 * Whether the spill fill fits this grid: its largest array is the Euler tour with `2 * cellCount`
 * words, which must fit one storage binding. Larger grids use the tiled relaxation alone.
 *
 * @internal
 */
export function canUseTerrainFlowSpillFill<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {width: number; height: number}
): boolean {
  return 8 * (props.width * props.height + 1) <= graph.device.limits.maxStorageBufferBindingSize;
}

/**
 * Returns the depression filling nodes of the spill graph fill, with the shape of
 * {@link createTerrainFlowFillNodes}: nodes, and a four-word status view whose word 2 is the
 * iteration count and word 3 is 1 once the fill is exact.
 *
 * The result is bit-identical to the tiled relaxation for `fillEpsilon` 0. A `fillEpsilon` above 0
 * (read from `settings[4]` on the GPU, no readback) selects the tiled relaxation, which is the only
 * path that supports a strict-descent increment; both paths are in the graph and the unselected
 * one launches nothing. Status semantics: for the spill path the iteration count is the number of
 * Boruvka rounds executed (including the confirming round) and `converged` is 1 whenever the tree
 * spans the ocean's component; for the relaxation path they are the relaxation's. `maxIterations`
 * only bounds the relaxation path. Grids too large for one storage binding of `2 * cellCount` words
 * (see {@link canUseTerrainFlowSpillFill}) use the relaxation path alone, exactly as before.
 *
 * @internal
 */
export function createTerrainFlowSpillFillNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TerrainFlowGrid & {
    id: string;
    maxIterations: number;
    elevation: GraphDataView<'float32'>;
    filled: GraphDataView<'float32'>;
    settings: GraphDataView<'float32'>;
    converged?: GraphDataView<'uint32'>;
    iterationCount?: GraphDataView<'uint32'>;
  }
): {nodes: GPUCommandNode<Parameters>[]; status: GraphDataView<'uint32'>} {
  if (!canUseTerrainFlowSpillFill(graph, props)) {
    return createTerrainFlowFillNodes(graph, props);
  }
  const {id, width, height} = props;
  const cellCount = width * height;
  const nodeCount = cellCount + 1;
  const edgeSlotCount = 5 * cellCount;
  const tileSize = getTerrainFlowSpillTileSize(graph);
  const tilesX = Math.ceil(width / tileSize);
  const tilesY = Math.ceil(height / tileSize);
  const tileCount = tilesX * tilesY;
  const tileThreads = tileCount * SPILL_WORKGROUP_SIZE;
  const create = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createContributorTransientView(graph, OPERATION, id, `${id}-spill-${name}`, format, length);

  const mode = create('mode', 'uint32', 1);
  const localFill = create('local-fill', 'float32', cellCount);
  const labels = create('labels', 'uint32', cellCount);
  const comp = create('comp', 'uint32', nodeCount);
  const bestKey = create('best-key', 'uint32', nodeCount);
  const bestEdge = create('best-edge', 'uint32', nodeCount);
  const hookPointer = create('hook-pointer', 'uint32', nodeCount);
  const hookPointerScratch = create('hook-pointer-scratch', 'uint32', nodeCount);
  const mstFirst = create('mst-first', 'uint32', cellCount);
  const mstSecond = create('mst-second', 'uint32', cellCount);
  const mstKey = create('mst-key', 'uint32', cellCount);
  const head = create('head', 'uint32', nodeCount);
  const parent = create('parent', 'uint32', nodeCount);
  const parentKey = create('parent-key', 'uint32', nodeCount);
  const combinedStatus = create('status', 'uint32', 4);

  const modeBinding: WGSLKernelBinding = {
    name: 'mode',
    view: mode,
    type: 'u32',
    access: 'read'
  };
  // mode word: 1 selects the epsilon relaxation, 0 the spill graph.
  const spillEnabled = 'mode[modeOffset] == 0u';

  const modeGate = createWGSLDispatchGate(graph, {
    id: `${id}-spill-mode-dispatch`,
    operation: OPERATION,
    shapes: [
      [tileThreads, SPILL_WORKGROUP_SIZE],
      [nodeCount, SPILL_WORKGROUP_SIZE],
      [2 * cellCount, SPILL_WORKGROUP_SIZE],
      [cellCount, SPILL_WORKGROUP_SIZE],
      [cellCount, SPILL_WORKGROUP_SIZE]
    ]
  });
  const getModeGate = (slot: number) => (nodeId: string) => getSpillGateUse(modeGate, slot, nodeId);

  const kernel = (
    name: string,
    invocationCount: number,
    bindings: WGSLKernelBinding[],
    declarations: string,
    body: string,
    gate: SpillGateUse<Parameters>,
    guardIndex = true
  ) =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-spill-${name}`,
      operation: OPERATION,
      variant: `spill-${name}`,
      bindings,
      invocationCount,
      workgroupSize: SPILL_WORKGROUP_SIZE,
      guardIndex,
      declarations,
      body,
      ...gate
    });

  // Selects the path from fillEpsilon, arms the matching kernels and disarms the others.
  const disarmSlots = (slots: number[]) =>
    slots.map(slot => `dispatch[dispatchOffset + ${3 * slot}u] = 0u;`).join('\n    ');
  const selectNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-spill-select`,
    operation: OPERATION,
    variant: 'spill-select',
    bindings: [
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'mode', view: mode, type: 'u32', access: 'read_write'},
      modeGate.binding
    ],
    invocationCount: 1,
    body: `let epsilon = settings[settingsOffset + 4u];
  // Invalid, negative and non-finite epsilon mean 0, as in the relaxation.
  let useRelaxation = (bitcast<u32>(epsilon) & 0x7fffffffu) < 0x7f800000u && epsilon > 0.0;
  mode[modeOffset] = select(0u, 1u, useRelaxation);
  ${modeGate.armWGSL}
  if (useRelaxation) {
    ${disarmSlots([MODE_SLOT_TILES, MODE_SLOT_NODES, MODE_SLOT_HALF_EDGES, MODE_SLOT_CELLS])}
  } else {
    ${disarmSlots([MODE_SLOT_RELAXATION_INIT])}
  }`
  });

  // Relaxation path (fillEpsilon > 0): the tiled Planchon-Darboux fill, gated on the mode word.
  const relaxationProps = {
    id: `${id}-fill`,
    operation: OPERATION,
    width,
    height,
    cellSizeMode: props.cellSizeMode,
    maxIterations: props.maxIterations
  };
  const {relaxation, resetNodes} = createRasterTiledRelaxation<Parameters>(
    graph,
    relaxationProps,
    false
  );
  const tileReset = resetNodes.find(node => node.id === `${id}-fill-tile-reset`);
  if (!tileReset) {
    throw new Error(`${id} relaxation tile reset node is missing`);
  }
  const relaxationReset = createSpillResetNode<Parameters>(graph, {
    id: `${id}-fill-relax-reset`,
    operation: OPERATION,
    state: relaxation.state,
    enabledBindings: [modeBinding],
    enabledExpression: 'mode[modeOffset] == 1u'
  });
  const relaxationInit = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-fill-init`,
    operation: OPERATION,
    variant: 'fill-init',
    bindings: [
      {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'filled', view: props.filled, type: 'u32', access: 'read_write'},
      {
        name: 'tileStamps',
        view: relaxation.tileStamps,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ],
    invocationCount: cellCount,
    workgroupSize: SPILL_WORKGROUP_SIZE,
    declarations: `${getRasterGridWGSL(props)}\n${getRasterRelaxationTileWGSL(width)}`,
    // Boundary cells keep their elevation, other valid cells start at +infinity and are lowered.
    // The phony assignment keeps 'settings' in the pipeline layout (grid helpers may not read it).
    body: `_ = settings[settingsOffset];
  let elevationValue = elevation[elevationOffset + index];
  if (!isFiniteValue(elevationValue)) {
    filled[filledOffset + index] = bitcast<u32>(getQuietNaN());
    return;
  }
  var boundary = false;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE) {
      boundary = true;
      break;
    }
    if (!isFiniteValue(elevation[elevationOffset + neighbor])) {
      boundary = true;
      break;
    }
  }
  filled[filledOffset + index] = bitcast<u32>(select(getInfinity(), elevationValue, boundary));
  if (boundary) {
    atomicMax(&tileStamps[tileStampsOffset + getRelaxationTile(index)], 1u);
  }`,
    ...getSpillGateUse(modeGate, MODE_SLOT_RELAXATION_INIT, `${id}-fill-init`)
  });
  const relaxationNodes = createRasterTiledRelaxationNodes<Parameters>(graph, {
    ...relaxationProps,
    relaxation,
    sweepAfterIteration: TERRAIN_FLOW_SWEEP_AFTER_ITERATION,
    values: props.filled,
    auxiliary: props.elevation,
    settings: props.settings,
    declarations: `
fn getRelaxationCandidate(neighborValue: f32, neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  var epsilon = settings[settingsOffset + 4u];
  if (!(epsilon >= 0.0) || !isFiniteValue(epsilon)) { epsilon = 0.0; }
  return max(centerAuxiliary, neighborValue + epsilon);
}`
  });

  // Stage 1: tile fill and labels.
  const common = getSpillCommonWGSL(width, height);
  const tileCells = tileSize * tileSize;
  const tileFill = kernel(
    'tile-fill',
    tileThreads,
    [
      {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
      {name: 'localFill', view: localFill, type: 'u32', access: 'read_write'},
      {name: 'labels', view: labels, type: 'u32', access: 'read_write'}
    ],
    `${common}
const TILE: u32 = ${tileSize}u;
const TILE_CELLS: u32 = ${tileCells}u;
const CELLS_PER_THREAD: u32 = ${tileCells / SPILL_WORKGROUP_SIZE}u;
const TILES_X: u32 = ${tilesX}u;
const TILE_COUNT: u32 = ${tileCount}u;
var<workgroup> tileKey: array<atomic<u32>, ${tileCells}>;
var<workgroup> tileLabel: array<atomic<u32>, ${tileCells}>;
var<workgroup> tileCategory: array<u32, ${tileCells}>;
var<workgroup> tileChanged: atomic<u32>;
var<workgroup> tileContinue: u32;`,
    // Categories: 0 no cell (invalid or outside the grid), 1 interior, 2 tile perimeter outlet,
    // 3 boundary outlet. Interior cells never touch an invalid cell or the grid edge (they would be
    // boundary outlets) and never leave the tile (they would be perimeter outlets), so all eight
    // neighbors of an interior cell are valid cells of the tile.
    `if (workgroupIndex >= TILE_COUNT) { return; }
  let tileColumn = workgroupIndex % TILES_X;
  let tileRow = workgroupIndex / TILES_X;
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);

  for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
    let local = localInvocationIndex + slot * ${SPILL_WORKGROUP_SIZE}u;
    let localColumn = local % TILE;
    let localRow = local / TILE;
    let column = tileColumn * TILE + localColumn;
    let row = tileRow * TILE + localRow;
    var category = 0u;
    var key = SPILL_INFINITE_KEY;
    var label = SPILL_NONE;
    if (column < SPILL_WIDTH && row < SPILL_HEIGHT) {
      let cell = row * SPILL_WIDTH + column;
      let value = elevation[elevationOffset + cell];
      if (isSpillFinite(value)) {
        var boundary = false;
        for (var direction = 0u; direction < 8u; direction++) {
          let neighbor = getSpillNeighbor(cell, direction);
          if (neighbor == SPILL_NONE) {
            boundary = true;
            break;
          }
          if (!isSpillFinite(elevation[elevationOffset + neighbor])) {
            boundary = true;
            break;
          }
        }
        let perimeter = localColumn == 0u || localRow == 0u || localColumn == TILE - 1u || localRow == TILE - 1u;
        if (boundary) {
          category = 3u;
          key = getSpillKey(value);
          label = cell | SPILL_BOUNDARY_FLAG;
        } else if (perimeter) {
          category = 2u;
          key = getSpillKey(value);
          label = cell;
        } else {
          category = 1u;
        }
      }
    }
    tileCategory[local] = category;
    atomicStore(&tileKey[local], key);
    atomicStore(&tileLabel[local], label);
  }
  workgroupBarrier();

  // Fill: key = max(own, lowest neighbor key) to the fixpoint. Keys only decrease (atomicMin), so
  // the update order does not matter; the cap T * T + 1 passes is the longest simple path plus one
  // pass that confirms the fixpoint, and each changing pass lowers a key, so it always exits.
  for (var fillPass = 0u; fillPass <= TILE_CELLS; fillPass++) {
    if (localInvocationIndex == 0u) { atomicStore(&tileChanged, 0u); }
    workgroupBarrier();
    for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
      let local = localInvocationIndex + slot * ${SPILL_WORKGROUP_SIZE}u;
      if (tileCategory[local] != 1u) { continue; }
      let column = tileColumn * TILE + local % TILE;
      let row = tileRow * TILE + local / TILE;
      let ownKey = getSpillKey(elevation[elevationOffset + row * SPILL_WIDTH + column]);
      var lowest = SPILL_INFINITE_KEY;
      for (var direction = 0u; direction < 8u; direction++) {
        let neighborLocal = u32(i32(local) + rowOffsets[direction] * i32(TILE) + columnOffsets[direction]);
        lowest = min(lowest, atomicLoad(&tileKey[neighborLocal]));
      }
      let candidate = max(ownKey, lowest);
      if (candidate < atomicLoad(&tileKey[local])) {
        atomicMin(&tileKey[local], candidate);
        atomicStore(&tileChanged, 1u);
      }
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) { tileContinue = atomicLoad(&tileChanged); }
    if (workgroupUniformLoad(&tileContinue) == 0u) { break; }
  }

  // Labels: the smallest outlet index reachable through cells of non-increasing key. A neighbor
  // passes its label on when its key is at most the own key (the chain stays within the own key).
  for (var labelPass = 0u; labelPass <= TILE_CELLS; labelPass++) {
    if (localInvocationIndex == 0u) { atomicStore(&tileChanged, 0u); }
    workgroupBarrier();
    for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
      let local = localInvocationIndex + slot * ${SPILL_WORKGROUP_SIZE}u;
      if (tileCategory[local] != 1u) { continue; }
      let ownKey = atomicLoad(&tileKey[local]);
      var best = SPILL_NONE;
      for (var direction = 0u; direction < 8u; direction++) {
        let neighborLocal = u32(i32(local) + rowOffsets[direction] * i32(TILE) + columnOffsets[direction]);
        let neighborLabel = atomicLoad(&tileLabel[neighborLocal]);
        if (neighborLabel != SPILL_NONE && atomicLoad(&tileKey[neighborLocal]) <= ownKey) {
          best = min(best, neighborLabel & SPILL_ID_MASK);
        }
      }
      if (best < atomicLoad(&tileLabel[local])) {
        atomicMin(&tileLabel[local], best);
        atomicStore(&tileChanged, 1u);
      }
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) { tileContinue = atomicLoad(&tileChanged); }
    if (workgroupUniformLoad(&tileContinue) == 0u) { break; }
  }

  for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
    let local = localInvocationIndex + slot * ${SPILL_WORKGROUP_SIZE}u;
    let column = tileColumn * TILE + local % TILE;
    let row = tileRow * TILE + local / TILE;
    if (column >= SPILL_WIDTH || row >= SPILL_HEIGHT) { continue; }
    let cell = row * SPILL_WIDTH + column;
    if (tileCategory[local] == 0u) {
      localFill[localFillOffset + cell] = 0x7fc00000u;
      labels[labelsOffset + cell] = SPILL_NONE;
    } else {
      localFill[localFillOffset + cell] = getSpillValueBits(atomicLoad(&tileKey[local]));
      labels[labelsOffset + cell] = atomicLoad(&tileLabel[local]);
    }
  }`,
    getSpillGateUse(modeGate, MODE_SLOT_TILES, `${id}-spill-tile-fill`),
    false
  );

  // Stage 2 and 3: implicit spill edges and the Boruvka spanning tree.
  const edgeWGSL = `${common}
const SPILL_OCEAN: u32 = ${cellCount}u;
const SPILL_FORWARD_SLOTS: u32 = ${4 * cellCount}u;

// The decoded edge lives in module-scope private variables (no struct return value).
var<private> edgeValid: bool;
var<private> edgeFirst: u32;
var<private> edgeSecond: u32;
var<private> edgeKey: u32;

// Edge slots [0, 4 n): cell = slot / 4 and one of the four forward directions (E, SE, S, SW),
// an edge between the labels of two adjacent cells. Slots [4 n, 5 n): the ocean edge of cell
// slot - 4 n, present when that cell is a boundary outlet. The slot is the tie-break of equal keys.
fn loadSpillEdge(slot: u32) {
  edgeValid = false;
  if (slot < SPILL_FORWARD_SLOTS) {
    let cell = slot >> 2u;
    let neighbor = getSpillNeighbor(cell, slot & 3u);
    if (neighbor == SPILL_NONE) { return; }
    let firstWord = labels[labelsOffset + cell];
    let secondWord = labels[labelsOffset + neighbor];
    if (firstWord == SPILL_NONE || secondWord == SPILL_NONE) { return; }
    edgeFirst = firstWord & SPILL_ID_MASK;
    edgeSecond = secondWord & SPILL_ID_MASK;
    if (edgeFirst == edgeSecond) { return; }
    edgeKey = max(
      getSpillKey(bitcast<f32>(localFill[localFillOffset + cell])),
      getSpillKey(bitcast<f32>(localFill[localFillOffset + neighbor]))
    );
    edgeValid = true;
    return;
  }
  let cell = slot - SPILL_FORWARD_SLOTS;
  let word = labels[labelsOffset + cell];
  if (word == SPILL_NONE || (word & SPILL_BOUNDARY_FLAG) == 0u) { return; }
  edgeFirst = cell;
  edgeSecond = SPILL_OCEAN;
  edgeKey = getSpillKey(bitcast<f32>(localFill[localFillOffset + cell]));
  edgeValid = true;
}`;
  const labelBindings: WGSLKernelBinding[] = [
    {name: 'labels', view: labels, type: 'u32', access: 'read'},
    {name: 'localFill', view: localFill, type: 'u32', access: 'read'}
  ];
  const rounds = getTerrainFlowSpillBoruvkaRoundCount(cellCount);
  const jumpRounds = getSpillRoundCount(nodeCount);
  const outerState = createSpillLoopState<Parameters>(graph, {
    id: `${id}-spill-boruvka`,
    operation: OPERATION,
    contributorId: id,
    shapes: [
      [nodeCount, SPILL_WORKGROUP_SIZE],
      [edgeSlotCount, SPILL_WORKGROUP_SIZE]
    ],
    statusLength: 8
  });
  const outerReset = createSpillResetNode<Parameters>(graph, {
    id: `${id}-spill-boruvka-reset`,
    operation: OPERATION,
    state: outerState,
    statusLength: 8,
    enabledBindings: [modeBinding],
    enabledExpression: spillEnabled
  });
  const initialize = kernel(
    'init',
    nodeCount,
    [
      {name: 'comp', view: comp, type: 'u32', access: 'read_write'},
      {name: 'head', view: head, type: 'u32', access: 'read_write'},
      {name: 'parent', view: parent, type: 'u32', access: 'read_write'},
      {name: 'parentKey', view: parentKey, type: 'u32', access: 'read_write'}
    ],
    '',
    `comp[compOffset + index] = index;
  head[headOffset + index] = 0xffffffffu;
  parent[parentOffset + index] = index;
  parentKey[parentKeyOffset + index] = 0u;`,
    getSpillGateUse(modeGate, MODE_SLOT_NODES, `${id}-spill-init`)
  );
  const innerState = createSpillLoopState<Parameters>(graph, {
    id: `${id}-spill-boruvka-jump`,
    operation: OPERATION,
    contributorId: id,
    shapes: [[nodeCount, SPILL_WORKGROUP_SIZE]]
  });

  const boruvkaNodes: GPUCommandNode<Parameters>[] = [];
  for (let round = 0; round < rounds; round++) {
    const prefix = `boruvka-${round}`;
    const nodesGate = (name: string) =>
      getSpillGateUse(outerState.gate, BORUVKA_SLOT_NODES, `${id}-spill-${prefix}-${name}`);
    const edgesGate = (name: string) =>
      getSpillGateUse(outerState.gate, BORUVKA_SLOT_EDGES, `${id}-spill-${prefix}-${name}`);
    boruvkaNodes.push(
      kernel(
        `${prefix}-clear`,
        nodeCount,
        [
          {name: 'bestKey', view: bestKey, type: 'u32', access: 'read_write'},
          {name: 'bestEdge', view: bestEdge, type: 'u32', access: 'read_write'}
        ],
        '',
        `bestKey[bestKeyOffset + index] = 0xffffffffu;
  bestEdge[bestEdgeOffset + index] = 0xffffffffu;`,
        nodesGate('clear')
      ),
      // Phase 1: the smallest key of every component.
      kernel(
        `${prefix}-pick-key`,
        edgeSlotCount,
        [
          ...labelBindings,
          {name: 'comp', view: comp, type: 'u32', access: 'read'},
          {name: 'bestKey', view: bestKey, type: 'atomic<u32>', access: 'read_write'}
        ],
        edgeWGSL,
        `loadSpillEdge(index);
  if (!edgeValid) { return; }
  let firstComponent = comp[compOffset + edgeFirst];
  let secondComponent = comp[compOffset + edgeSecond];
  if (firstComponent == secondComponent) { return; }
  atomicMin(&bestKey[bestKeyOffset + firstComponent], edgeKey);
  atomicMin(&bestKey[bestKeyOffset + secondComponent], edgeKey);`,
        edgesGate('pick-key')
      ),
      // Phase 2: the smallest slot among the edges that have that key. (key, slot) is a strict
      // total order of the edges, so the picks form a forest plus two-cycles of mutual picks.
      kernel(
        `${prefix}-pick-edge`,
        edgeSlotCount,
        [
          ...labelBindings,
          {name: 'comp', view: comp, type: 'u32', access: 'read'},
          {name: 'bestKey', view: bestKey, type: 'u32', access: 'read'},
          {name: 'bestEdge', view: bestEdge, type: 'atomic<u32>', access: 'read_write'}
        ],
        edgeWGSL,
        `loadSpillEdge(index);
  if (!edgeValid) { return; }
  let firstComponent = comp[compOffset + edgeFirst];
  let secondComponent = comp[compOffset + edgeSecond];
  if (firstComponent == secondComponent) { return; }
  if (edgeKey == bestKey[bestKeyOffset + firstComponent]) {
    atomicMin(&bestEdge[bestEdgeOffset + firstComponent], index);
  }
  if (edgeKey == bestKey[bestKeyOffset + secondComponent]) {
    atomicMin(&bestEdge[bestEdgeOffset + secondComponent], index);
  }`,
        edgesGate('pick-edge')
      ),
      // Hook: a component joins the component at the other end of its picked edge. In a mutual
      // pick the larger id stays a root; the ocean's component never hooks, so it is the root of
      // its tree and (being the largest id) of every mutual pick it takes part in.
      kernel(
        `${prefix}-hook`,
        nodeCount,
        [
          ...labelBindings,
          {name: 'comp', view: comp, type: 'u32', access: 'read'},
          {name: 'bestEdge', view: bestEdge, type: 'u32', access: 'read'},
          {name: 'hookPointer', view: hookPointer, type: 'u32', access: 'read_write'},
          {name: 'status', view: outerState.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        edgeWGSL,
        `var hookTarget = index;
  if (comp[compOffset + index] == index && index != SPILL_OCEAN) {
    let slot = bestEdge[bestEdgeOffset + index];
    if (slot != SPILL_NONE) {
      loadSpillEdge(slot);
      let firstComponent = comp[compOffset + edgeFirst];
      let secondComponent = comp[compOffset + edgeSecond];
      let other = select(firstComponent, secondComponent, firstComponent == index);
      let mutual = bestEdge[bestEdgeOffset + other] == slot;
      if (!(mutual && index > other)) {
        hookTarget = other;
        atomicStore(&status[statusOffset + 0u], 1u);
      }
    }
  }
  hookPointer[hookPointerOffset + index] = hookTarget;`,
        nodesGate('hook')
      ),
      // Record the hooked edges as tree edges (every hook is a spanning tree edge).
      kernel(
        `${prefix}-record`,
        nodeCount,
        [
          ...labelBindings,
          {name: 'bestEdge', view: bestEdge, type: 'u32', access: 'read'},
          {name: 'hookPointer', view: hookPointer, type: 'u32', access: 'read'},
          {name: 'mstFirst', view: mstFirst, type: 'u32', access: 'read_write'},
          {name: 'mstSecond', view: mstSecond, type: 'u32', access: 'read_write'},
          {name: 'mstKey', view: mstKey, type: 'u32', access: 'read_write'},
          {name: 'status', view: outerState.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        edgeWGSL,
        `if (hookPointer[hookPointerOffset + index] == index) { return; }
  loadSpillEdge(bestEdge[bestEdgeOffset + index]);
  let position = atomicAdd(&status[statusOffset + 4u], 1u);
  // A spanning tree of at most n + 1 nodes has at most n edges.
  if (position < ${cellCount}u) {
    mstFirst[mstFirstOffset + position] = edgeFirst;
    mstSecond[mstSecondOffset + position] = edgeSecond;
    mstKey[mstKeyOffset + position] = edgeKey;
  }`,
        nodesGate('record')
      ),
      createSpillResetNode<Parameters>(graph, {
        id: `${id}-spill-${prefix}-jump-reset`,
        operation: OPERATION,
        state: innerState,
        enabledBindings: [{name: 'outer', view: outerState.status, type: 'u32', access: 'read'}],
        // Jump only while the outer loop runs and something hooked.
        enabledExpression: 'outer[outerOffset + 1u] == 1u && outer[outerOffset + 0u] != 0u'
      }),
      // Flatten the hook forests: after the doubling rounds every hook pointer is a root.
      ...createRasterGatedLoopNodes<Parameters>(graph, {
        id: `${id}-spill-${prefix}-jump`,
        gateId: `${id}-spill-${prefix}-jump-gate`,
        operation: OPERATION,
        state: innerState,
        maxIterations: jumpRounds,
        createRound: ({
          nodeId,
          iteration,
          slotBinding,
          markChangedWGSL,
          condition,
          extraResources
        }) => {
          const [from, to] =
            iteration % 2 === 0
              ? [hookPointer, hookPointerScratch]
              : [hookPointerScratch, hookPointer];
          return createWGSLKernelNode<Parameters>(graph, {
            id: nodeId,
            operation: OPERATION,
            variant: 'spill-jump',
            bindings: [
              {name: 'pointerIn', view: from, type: 'u32', access: 'read'},
              {name: 'pointerOut', view: to, type: 'u32', access: 'read_write'},
              {name: 'status', view: innerState.status, type: 'atomic<u32>', access: 'read_write'},
              slotBinding
            ],
            invocationCount: nodeCount,
            workgroupSize: SPILL_WORKGROUP_SIZE,
            body: `let next = pointerIn[pointerInOffset + index];
  let jumped = pointerIn[pointerInOffset + next];
  pointerOut[pointerOutOffset + index] = jumped;
  if (jumped != next) {
    ${markChangedWGSL}
  }`,
            condition,
            extraResources
          });
        }
      }),
      kernel(
        `${prefix}-apply`,
        nodeCount,
        [
          {name: 'comp', view: comp, type: 'u32', access: 'read_write'},
          {name: 'hookPointer', view: hookPointer, type: 'u32', access: 'read'}
        ],
        '',
        `comp[compOffset + index] = hookPointer[hookPointerOffset + comp[compOffset + index]];`,
        nodesGate('apply')
      ),
      createRasterIterationGateNode<Parameters>(graph, {
        id: `${id}-spill-${prefix}-gate`,
        operation: OPERATION,
        state: outerState,
        maxIterations: rounds
      })
    );
  }

  // Stage 3b: root the tree at the ocean and take the path maximum.
  const tree = createTerrainFlowSpillTreeNodes<Parameters>(graph, {
    id,
    operation: OPERATION,
    contributorId: id,
    cellCount,
    mstFirst,
    mstSecond,
    mstKey,
    head,
    mstStatus: outerState.status,
    parent,
    parentKey,
    enabledBindings: [modeBinding],
    enabledExpression: spillEnabled,
    getHalfEdgeGate: getModeGate(MODE_SLOT_HALF_EDGES),
    getEdgeGate: getModeGate(MODE_SLOT_CELLS)
  });

  // Stage 4: F = max(L, G(label)). A valid cell without a label, or whose label does not reach
  // the ocean, is unresolved: the output is NaN and the converged flag clears.
  const finalize = kernel(
    'final',
    cellCount,
    [
      {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
      ...labelBindings,
      {name: 'parent', view: parent, type: 'u32', access: 'read'},
      {name: 'parentKey', view: parentKey, type: 'u32', access: 'read'},
      {name: 'filled', view: props.filled, type: 'u32', access: 'read_write'},
      {name: 'status', view: outerState.status, type: 'atomic<u32>', access: 'read_write'}
    ],
    `${common}
const SPILL_OCEAN: u32 = ${cellCount}u;`,
    `let value = elevation[elevationOffset + index];
  if (!isSpillFinite(value)) {
    filled[filledOffset + index] = 0x7fc00000u;
    return;
  }
  let word = labels[labelsOffset + index];
  if (word == SPILL_NONE || parent[parentOffset + (word & SPILL_ID_MASK)] != SPILL_OCEAN) {
    filled[filledOffset + index] = 0x7fc00000u;
    atomicStore(&status[statusOffset + 3u], 0u);
    return;
  }
  let key = max(
    getSpillKey(bitcast<f32>(localFill[localFillOffset + index])),
    parentKey[parentKeyOffset + (word & SPILL_ID_MASK)]
  );
  filled[filledOffset + index] = getSpillValueBits(key);`,
    getSpillGateUse(modeGate, MODE_SLOT_CELLS, `${id}-spill-final`)
  );

  // The status of the selected path and the optional outputs.
  const statusBindings: WGSLKernelBinding[] = [
    modeBinding,
    {name: 'relaxationStatus', view: relaxation.state.status, type: 'u32', access: 'read'},
    {name: 'spillStatus', view: outerState.status, type: 'u32', access: 'read'},
    {name: 'combined', view: combinedStatus, type: 'u32', access: 'read_write'}
  ];
  if (props.converged) {
    statusBindings.push({
      name: 'converged',
      view: props.converged,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.iterationCount) {
    statusBindings.push({
      name: 'iterationCount',
      view: props.iterationCount,
      type: 'u32',
      access: 'read_write'
    });
  }
  const publish = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-spill-publish`,
    operation: OPERATION,
    variant: 'spill-publish',
    bindings: statusBindings,
    invocationCount: 1,
    body: `let useRelaxation = mode[modeOffset] == 1u;
  let roundCount = select(spillStatus[spillStatusOffset + 2u], relaxationStatus[relaxationStatusOffset + 2u], useRelaxation);
  let isConverged = select(spillStatus[spillStatusOffset + 3u], relaxationStatus[relaxationStatusOffset + 3u], useRelaxation);
  combined[combinedOffset + 0u] = 0u;
  combined[combinedOffset + 1u] = 0u;
  combined[combinedOffset + 2u] = roundCount;
  combined[combinedOffset + 3u] = isConverged;
  ${props.converged ? 'converged[convergedOffset] = isConverged;' : ''}
  ${props.iterationCount ? 'iterationCount[iterationCountOffset] = roundCount;' : ''}`
  });

  const nodes: GPUCommandNode<Parameters>[] = [
    selectNode,
    relaxationReset,
    tileReset,
    relaxationInit,
    ...relaxationNodes,
    outerReset,
    tileFill,
    initialize,
    ...boruvkaNodes,
    ...tree.nodes,
    finalize,
    publish
  ];
  return {nodes, status: combinedStatus};
}
