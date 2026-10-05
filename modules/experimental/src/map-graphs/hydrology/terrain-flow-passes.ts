// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createMapGraphFillNode,
  createMapGraphKernelNode,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import type {GPUTerrainCellSizeMode} from '../terrain-analysis/index';
import {getRasterGridWGSL} from '../cost-distance/raster-grid-utils';
import {
  createRasterIterationGateNode,
  createRasterIterationResetNode,
  createRasterIterationState,
  createRasterIterationFinalizeNode,
  createRasterTiledRelaxation,
  createRasterTiledRelaxationNodes,
  getRasterIterationCondition
} from '../cost-distance/raster-relaxation';

const OPERATION = 'GPUTerrainFlow';

/** Compile-time grid description shared by the terrain flow kernels. @internal */
export type TerrainFlowGrid = {
  width: number;
  height: number;
  cellSizeMode: GPUTerrainCellSizeMode;
};

/** Longest downstream walk of one thread within one accumulation round. @internal */
export const TERRAIN_FLOW_MAXIMUM_WALK_LENGTH = 1024;

/**
 * Returns the depression filling nodes: initialization, tiled Planchon-Darboux relaxation, and the
 * optional converged-flag publication.
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
  }
): GPUCommandNode<Parameters>[] {
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
    true
  );
  const initialize = createMapGraphKernelNode<Parameters>(graph, {
    id: `${props.id}-fill-init`,
    operation: OPERATION,
    variant: 'fill-init',
    bindings: [
      {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'filled', view: props.filled, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.width * props.height,
    declarations: getRasterGridWGSL(props),
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
  filled[filledOffset + index] = bitcast<u32>(select(getInfinity(), elevationValue, boundary));`
  });
  const relaxNodes = createRasterTiledRelaxationNodes<Parameters>(graph, {
    ...relaxationProps,
    relaxation,
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
  if (props.converged) {
    nodes.push(
      createRasterIterationFinalizeNode<Parameters>(graph, {
        id: `${props.id}-fill-finalize`,
        operation: OPERATION,
        state: relaxation.state,
        converged: props.converged
      })
    );
  }
  return nodes;
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
  const bindings: MapGraphKernelBinding[] = [
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
  return createMapGraphKernelNode<Parameters>(graph, {
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
 * Returns the deterministic pull-based accumulation nodes: sentinel fill, loop reset, and
 * `maxIterations` gated rounds.
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
  }
): GPUCommandNode<Parameters>[] {
  const cellCount = props.width * props.height;
  const state = createRasterIterationState(graph, `${props.id}-accumulate`, OPERATION, cellCount);
  const nodes: GPUCommandNode<Parameters>[] = [
    createMapGraphFillNode<Parameters>(graph, {
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
  const bindings: MapGraphKernelBinding[] = [
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
  const body = `_ = settings[settingsOffset];
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
    atomicStore(&status[statusOffset], 1u);
  }`;
  for (let iteration = 0; iteration < props.maxIterations; iteration++) {
    const nodeId = `${props.id}-accumulate-round-${iteration}`;
    const {condition, extraResources} = getRasterIterationCondition<Parameters>(state, nodeId);
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: nodeId,
        operation: OPERATION,
        variant: 'accumulate-round',
        bindings,
        invocationCount: cellCount,
        declarations,
        body,
        condition,
        extraResources
      }),
      createRasterIterationGateNode<Parameters>(graph, {
        id: `${props.id}-accumulate-gate-${iteration}`,
        operation: OPERATION,
        state,
        maxIterations: props.maxIterations
      })
    );
  }
  if (props.converged) {
    nodes.push(
      createRasterIterationFinalizeNode<Parameters>(graph, {
        id: `${props.id}-accumulate-finalize`,
        operation: OPERATION,
        state,
        converged: props.converged
      })
    );
  }
  return nodes;
}

/** Returns the stream mask kernel: 1 where accumulation is finite and at least the threshold. @internal */
export function createTerrainFlowStreamsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    accumulation: GraphDataView<'float32'>;
    streams: GraphDataView<'uint32'>;
    settings: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'streams',
    bindings: [
      {name: 'accumulation', view: props.accumulation, type: 'f32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'streams', view: props.streams, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    body: `let value = accumulation[accumulationOffset + index];
  let isStream = (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u &&
    value >= settings[settingsOffset + 5u];
  streams[streamsOffset + index] = select(0u, 1u, isStream);`
  });
}
