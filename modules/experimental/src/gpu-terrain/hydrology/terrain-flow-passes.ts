// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPURasterCellSizeMode} from '../../analysis-core/raster-grid';
import {getRasterGridWGSL} from '../../gpu-raster/cost-distance/raster-grid-utils';
import {
  createRasterGatedLoopNodes,
  createRasterIterationResetNode,
  createRasterIterationState,
  createRasterIterationFinalizeNode,
  createRasterTiledRelaxation,
  createRasterTiledRelaxationNodes,
  getRasterRelaxationTileWGSL
} from '../../gpu-raster/cost-distance/raster-relaxation';

import {
  createTerrainFlowEulerTourNodes,
  getTerrainFlowEulerTourWGSL,
  TERRAIN_FLOW_TOUR_RESOLVED_WGSL
} from './terrain-flow-euler-tour';

const OPERATION = 'GPUTerrainFlow';

/** Compile-time grid description shared by the terrain flow kernels. @internal */
export type TerrainFlowGrid = {
  width: number;
  height: number;
  cellSizeMode: GPURasterCellSizeMode;
};

/**
 * Plain tiled iterations before fill and flat resolution switch to directional row and column sweeps.
 * Open terrain converges earlier and never sweeps; long corridors and spirals then cross a whole tile
 * row or column per pass instead of one tile.
 *
 * @internal
 */
export const TERRAIN_FLOW_SWEEP_AFTER_ITERATION = 96;

/** Longest downstream walk of one thread within one accumulation round. @internal */
export const TERRAIN_FLOW_MAXIMUM_WALK_LENGTH = 1024;

/**
 * Returns the depression filling nodes (initialization, tiled Planchon-Darboux relaxation, and the
 * optional converged-flag publication) and the relaxation status view, whose word 3 is 1 once the
 * fill reached its fixpoint.
 *
 * @internal
 */
export function createTerrainFlowFillNodes<Parameters>(
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
  const relaxationProps = {
    id: `${props.id}-fill`,
    operation: OPERATION,
    width: props.width,
    height: props.height,
    cellSizeMode: props.cellSizeMode,
    maxIterations: props.maxIterations
  };
  const {relaxation, resetNodes} = createRasterTiledRelaxation<Parameters>(
    graph,
    relaxationProps,
    // Only tiles holding a boundary cell are seeded: interior cells start at +infinity and can only
    // be lowered once a finite neighbor tile exists, so iteration 0 skips the untouched interior.
    false
  );
  const initialize = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-fill-init`,
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
    invocationCount: props.width * props.height,
    declarations: `${getRasterGridWGSL(props)}\n${getRasterRelaxationTileWGSL(props.width)}`,
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
    if (neighbor == GRID_NONE || !isFiniteValue(elevation[elevationOffset + neighbor])) {
      boundary = true;
      break;
    }
  }
  filled[filledOffset + index] = bitcast<u32>(select(getInfinity(), elevationValue, boundary));
  if (boundary) {
    atomicMax(&tileStamps[tileStampsOffset + getRelaxationTile(index)], 1u);
  }`
  });
  const relaxNodes = createRasterTiledRelaxationNodes<Parameters>(graph, {
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
  const nodes = [...resetNodes, initialize, ...relaxNodes];
  if (props.converged || props.iterationCount) {
    nodes.push(
      createRasterIterationFinalizeNode<Parameters>(graph, {
        id: `${props.id}-fill-finalize`,
        operation: OPERATION,
        state: relaxation.state,
        converged: props.converged,
        iterationCount: props.iterationCount
      })
    );
  }
  return {nodes, status: relaxation.state.status};
}

/**
 * Returns the D8 flow direction kernel. Writes whichever of `directions`, `classes`, and
 * `receivers` are provided.
 *
 * @internal
 */
export function createTerrainFlowDirectionNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TerrainFlowGrid & {
    id: string;
    elevation: GraphDataView<'float32'>;
    surface: GraphDataView<'float32'>;
    settings: GraphDataView<'float32'>;
    directions?: GraphDataView<'uint32'>;
    classes?: GraphDataView<'uint32'>;
    receivers?: GraphDataView<'uint32'>;
    cellClass: {draining: number; flat: number; pit: number; outlet: number; invalid: number};
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
    {name: 'surface', view: props.surface, type: 'f32', access: 'read'},
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
  ];
  for (const [name, view] of [
    ['directions', props.directions],
    ['classes', props.classes],
    ['receivers', props.receivers]
  ] as const) {
    if (view) {
      bindings.push({name, view, type: 'u32', access: 'read_write'});
    }
  }
  const write = (directionCode: string, cellClass: string, receiver: string) =>
    [
      props.directions ? `directions[directionsOffset + index] = ${directionCode};` : '',
      props.classes ? `classes[classesOffset + index] = ${cellClass};` : '',
      props.receivers ? `receivers[receiversOffset + index] = ${receiver};` : ''
    ].join('\n  ');
  const {cellClass} = props;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'flow-direction',
    bindings,
    invocationCount: props.width * props.height,
    declarations: getRasterGridWGSL(props),
    body: `let row = index / GRID_WIDTH;
  if (!isFiniteValue(elevation[elevationOffset + index])) {
    ${write('GRID_NONE', `${cellClass.invalid}u`, 'GRID_NONE')}
    return;
  }
  let centerSurface = surface[surfaceOffset + index];
  var bestDrop = 0.0;
  var bestDirection = 8u;
  var bestNeighbor = GRID_NONE;
  var boundary = false;
  var hasEqualNeighbor = false;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE || !isFiniteValue(elevation[elevationOffset + neighbor])) {
      boundary = true;
      continue;
    }
    let neighborSurface = surface[surfaceOffset + neighbor];
    if (neighborSurface == centerSurface) { hasEqualNeighbor = true; }
    let drop = (centerSurface - neighborSurface) / getD8Distance(direction, row);
    // Strictly greater: ties keep the lowest direction index.
    if (drop > bestDrop) {
      bestDrop = drop;
      bestDirection = direction;
      bestNeighbor = neighbor;
    }
  }
  if (bestDirection < 8u) {
    ${write('getD8Code(bestDirection)', `${cellClass.draining}u`, 'bestNeighbor')}
    return;
  }
  var terminalClass = ${cellClass.pit}u;
  if (boundary) { terminalClass = ${cellClass.outlet}u; }
  else if (hasEqualNeighbor) { terminalClass = ${cellClass.flat}u; }
  ${write('0u', 'terminalClass', 'GRID_NONE')}`
  });
}

/**
 * Whether D8 accumulation of this configuration runs on the Euler tour (exact integer subtree
 * sizes, `O(log n)` rounds) instead of the pull rounds: unit weights only (no `runoff`, no area),
 * and the tour arrays (`2 * cellCount` words) must fit one storage binding.
 *
 * @internal
 */
export function canUseTerrainFlowTourAccumulation<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {width: number; height: number; runoff?: unknown; area: boolean}
): boolean {
  return (
    !props.runoff &&
    !props.area &&
    8 * props.width * props.height <= graph.device.limits.maxStorageBufferBindingSize
  );
}

/**
 * Returns the D8 'cells' accumulation nodes on the Euler tour of the receiver forest: tour link,
 * gated Wyllie ranking rounds, and one kernel that turns ranks into subtree sizes. The count of a
 * cell is `(distance(E) - distance(X) + 1) / 2`, an exact `u32` converted once to float32, so the
 * result does not depend on the forest height or on any iteration limit. Cells whose list ends at
 * no root (receiver cycles) read NaN and clear the converged flag; invalid cells read NaN.
 *
 * @internal
 */
function createTerrainFlowTourAccumulationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TerrainFlowGrid & {
    id: string;
    elevation: GraphDataView<'float32'>;
    receivers: GraphDataView<'uint32'>;
    accumulation: GraphDataView<'float32'>;
    converged?: GraphDataView<'uint32'>;
    iterationCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const cellCount = props.width * props.height;
  const tour = createTerrainFlowEulerTourNodes<Parameters>(graph, {
    id: `${props.id}-accumulate`,
    operation: OPERATION,
    contributorId: props.id,
    width: props.width,
    height: props.height,
    declarations: `fn getTourParent(cell: u32) -> u32 { return receivers[receiversOffset + cell]; }`,
    parentBindings: [{name: 'receivers', view: props.receivers, type: 'u32', access: 'read'}]
  });
  const count = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-accumulate-count`,
    operation: OPERATION,
    variant: 'tour-accumulation',
    bindings: [
      {name: 'acc', view: props.accumulation, type: 'u32', access: 'read_write'},
      {name: 'receivers', view: props.receivers, type: 'u32', access: 'read'},
      {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
      {name: 'tourSucc', view: tour.successor, type: 'u32', access: 'read'},
      {name: 'tourDist', view: tour.distance, type: 'u32', access: 'read'},
      {name: 'status', view: tour.state.status, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: cellCount,
    declarations: `${getTerrainFlowEulerTourWGSL(props.width, props.height)}
fn getTourParent(cell: u32) -> u32 { return receivers[receiversOffset + cell]; }
${TERRAIN_FLOW_TOUR_RESOLVED_WGSL}`,
    body: `let isValid = (bitcast<u32>(elevation[elevationOffset + index]) & 0x7fffffffu) < 0x7f800000u;
  if (!isValid) { acc[accOffset + index] = 0xffffffffu; return; }
  if (!isTourEdgeResolved(2u * index)) {
    // The list never reached a root: a receiver cycle. Not converged.
    acc[accOffset + index] = 0xffffffffu;
    atomicStore(&status[statusOffset + 3u], 0u);
    return;
  }
  // The edges from E(v) to X(v) inclusive are the tour of v's subtree: 2 * size(v) of them.
  let size = (tourDist[tourDistOffset + 2u * index] - tourDist[tourDistOffset + 2u * index + 1u] + 1u) / 2u;
  acc[accOffset + index] = bitcast<u32>(f32(size));`
  });
  const nodes = [...tour.nodes, count];
  if (props.converged || props.iterationCount) {
    nodes.push(
      createRasterIterationFinalizeNode<Parameters>(graph, {
        id: `${props.id}-accumulate-finalize`,
        operation: OPERATION,
        state: tour.state,
        converged: props.converged,
        iterationCount: props.iterationCount
      })
    );
  }
  return nodes;
}

/**
 * Returns the D8 accumulation nodes. Unit-weight accumulation ('cells' without `runoff`) runs on
 * the Euler tour of the receiver forest and ignores `maxIterations`; weighted accumulation (`runoff`
 * or area) is the deterministic pull: sentinel fill, loop reset, and `maxIterations` gated rounds.
 *
 * @internal
 */
export function createTerrainFlowAccumulationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TerrainFlowGrid & {
    id: string;
    maxIterations: number;
    elevation: GraphDataView<'float32'>;
    receivers: GraphDataView<'uint32'>;
    accumulation: GraphDataView<'float32'>;
    settings: GraphDataView<'float32'>;
    runoff?: GraphDataView<'float32'>;
    area: boolean;
    converged?: GraphDataView<'uint32'>;
    iterationCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  if (canUseTerrainFlowTourAccumulation(graph, props)) {
    return createTerrainFlowTourAccumulationNodes(graph, props);
  }
  const cellCount = props.width * props.height;
  const state = createRasterIterationState(graph, `${props.id}-accumulate`, OPERATION, cellCount);
  const nodes: GPUCommandNode<Parameters>[] = [
    createFillNode<Parameters>(graph, {
      id: `${props.id}-accumulate-init`,
      operation: OPERATION,
      view: props.accumulation,
      type: 'u32',
      value: '0xffffffffu'
    }),
    createRasterIterationResetNode<Parameters>(graph, {
      id: `${props.id}-accumulate-reset`,
      operation: OPERATION,
      state
    })
  ];
  const bindings: WGSLKernelBinding[] = [
    {name: 'acc', view: props.accumulation, type: 'atomic<u32>', access: 'read_write'},
    {name: 'receivers', view: props.receivers, type: 'u32', access: 'read'},
    {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
  ];
  if (props.runoff) {
    bindings.push({name: 'runoff', view: props.runoff, type: 'f32', access: 'read'});
  }
  bindings.push({name: 'status', view: state.status, type: 'atomic<u32>', access: 'read_write'});
  const declarations = `${getRasterGridWGSL(props)}
const SENTINEL: u32 = 0xffffffffu;
const MAX_WALK_LENGTH: u32 = ${TERRAIN_FLOW_MAXIMUM_WALK_LENGTH}u;

fn getWeight(cell: u32) -> f32 {
  var weight: f32 = 1.0;
  ${
    props.runoff
      ? `let runoffValue = runoff[runoffOffset + cell];
  weight = select(0.0, runoffValue, isFiniteValue(runoffValue) && runoffValue >= 0.0);`
      : ''
  }
  ${
    props.area
      ? `let ground = getGroundCellSize(f32(cell / GRID_WIDTH) + 0.5);
  weight = weight * (ground.x * ground.y);`
      : ''
  }
  return weight;
}

// Returns true when this thread wrote the final value of 'cell'.
fn tryFinalize(cell: u32) -> bool {
  if (atomicLoad(&acc[accOffset + cell]) != SENTINEL) { return false; }
  var sum = getWeight(cell);
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(cell, direction);
    if (neighbor != GRID_NONE && receivers[receiversOffset + neighbor] == cell) {
      let donorBits = atomicLoad(&acc[accOffset + neighbor]);
      if (donorBits == SENTINEL) { return false; }
      sum = sum + bitcast<f32>(donorBits);
    }
  }
  return atomicCompareExchangeWeak(&acc[accOffset + cell], SENTINEL, bitcast<u32>(sum)).exchanged;
}`;
  // The phony assignment keeps 'settings' in the pipeline layout when weights ignore ground area.
  const getBody = (markChangedWGSL: string) => `_ = settings[settingsOffset];
  if (!isFiniteValue(elevation[elevationOffset + index])) { return; }
  if (atomicLoad(&acc[accOffset + index]) != SENTINEL) { return; }
  if (tryFinalize(index)) {
    var walk = receivers[receiversOffset + index];
    for (var step = 0u; step < MAX_WALK_LENGTH; step++) {
      if (walk == GRID_NONE || !tryFinalize(walk)) { break; }
      walk = receivers[receiversOffset + walk];
    }
  }
  if (atomicLoad(&acc[accOffset + index]) == SENTINEL) {
    ${markChangedWGSL}
  }`;
  nodes.push(
    ...createRasterGatedLoopNodes<Parameters>(graph, {
      id: `${props.id}-accumulate-round`,
      gateId: `${props.id}-accumulate-gate`,
      operation: OPERATION,
      state,
      maxIterations: props.maxIterations,
      createRound: ({nodeId, slotBinding, markChangedWGSL, condition, extraResources}) =>
        createWGSLKernelNode<Parameters>(graph, {
          id: nodeId,
          operation: OPERATION,
          variant: 'accumulate-round',
          bindings: [...bindings, slotBinding],
          invocationCount: cellCount,
          declarations,
          body: getBody(markChangedWGSL),
          condition,
          extraResources
        })
    })
  );
  if (props.converged || props.iterationCount) {
    nodes.push(
      createRasterIterationFinalizeNode<Parameters>(graph, {
        id: `${props.id}-accumulate-finalize`,
        operation: OPERATION,
        state,
        converged: props.converged,
        iterationCount: props.iterationCount
      })
    );
  }
  return nodes;
}

/**
 * Returns the stream mask kernel: 1 where accumulation is finite and at least the threshold, 0
 * where it is below the threshold and on invalid cells, and `unresolvedValue` on valid cells whose
 * accumulation is still NaN (the accumulation round limit was hit). @internal
 */
export function createTerrainFlowStreamsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    /** Canonical elevation, NaN for invalid cells. */
    elevation: GraphDataView<'float32'>;
    accumulation: GraphDataView<'float32'>;
    streams: GraphDataView<'uint32'>;
    settings: GraphDataView<'float32'>;
    unresolvedValue: number;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'streams',
    bindings: [
      {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
      {name: 'accumulation', view: props.accumulation, type: 'f32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'streams', view: props.streams, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    body: `let value = accumulation[accumulationOffset + index];
  let isResolved = (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
  let isStream = isResolved && value >= settings[settingsOffset + 5u];
  var stream = select(0u, 1u, isStream);
  // A valid cell without a value was cut off by the accumulation round limit. It is usually the
  // downstream trunk, so 0 ("not a stream") would erase the main river.
  let isValid = (bitcast<u32>(elevation[elevationOffset + index]) & 0x7fffffffu) < 0x7f800000u;
  if (isValid && !isResolved) { stream = ${props.unresolvedValue}u; }
  streams[streamsOffset + index] = stream;`
  });
}
