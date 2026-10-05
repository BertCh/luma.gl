// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createMapGraphFillNode,
  createMapGraphKernelNode,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import {getRasterGridWGSL} from '../cost-distance/raster-grid-utils';
import {
  createRasterIterationFinalizeNode,
  createRasterIterationGateNode,
  createRasterIterationResetNode,
  createRasterIterationState,
  getRasterIterationCondition
} from '../cost-distance/raster-relaxation';
import {TERRAIN_FLOW_MAXIMUM_WALK_LENGTH, type TerrainFlowGrid} from './terrain-flow-passes';

const OPERATION = 'GPUTerrainFlow';

/** Non-D8 routings handled by {@link createTerrainFlowRoutingAccumulationNodes}. @internal */
export type TerrainFlowMultipleRouting = 'd-infinity' | 'mfd-freeman' | 'mfd-quinn';

/**
 * Returns the WGSL that defines `getFlowFraction(cell, direction)`: the fraction of the flow of
 * `cell` that leaves toward its neighbor in D8 `direction`, for the given routing.
 *
 * Expects the kernel to bind `surface` (f32), `receivers` (u32) and `settings` (f32) and to
 * include the raster grid WGSL. Fractions of one cell sum to 1 when it has a lower neighbor (or a
 * fallback receiver) and to 0 otherwise.
 *
 * Geometry (all ground lengths, evaluated like `getD8Distance`, so `cellSizeMode` is honored):
 * - D-infinity facet = (cardinal neighbor e1, diagonal neighbor e2). d1 is the cardinal distance,
 *   d2 the other axis' ground cell size at the diagonal move's midpoint row, so for north-south
 *   cardinals `d1^2 + d2^2` is exactly the squared diagonal distance `dd` and for east-west
 *   cardinals it differs only by the latitude change of the east-west spacing between the cell row
 *   and the midpoint row. Facets are enumerated cardinal by cardinal in direction order E, S, W, N,
 *   each with its two adjacent diagonals, lower diagonal index first: (E,SE) (E,NE) (S,SE) (S,SW)
 *   (W,SW) (W,NW) (N,NW) (N,NE). The strictly steepest facet wins; ties keep the first.
 * - MFD: slope `tan(beta) = drop / distance`; Quinn contour length `L` is half the ground side
 *   perpendicular to a cardinal move at the cell's row center and a quarter of the diagonal length
 *   for a diagonal move. Weights are normalized by the largest score before the power so large
 *   exponents do not underflow (the fractions are mathematically unchanged).
 *
 * @internal
 */
function getFlowFractionWGSL(routing: TerrainFlowMultipleRouting): string {
  const common = /* wgsl */ `
// Fraction of 'cell' flow that goes to the neighbor in 'direction' when no lower neighbor exists.
fn getFallbackFraction(cell: u32, direction: u32) -> f32 {
  let receiver = receivers[receiversOffset + cell];
  return select(0.0, 1.0, receiver != GRID_NONE && receiver == getD8Neighbor(cell, direction));
}`;
  if (routing === 'd-infinity') {
    return `${common}
// Tarboton (1997) facets in the documented order: index -> (cardinal, diagonal).
fn getFacetCardinal(facet: u32) -> u32 { return (facet / 2u) * 2u; }
fn getFacetDiagonal(facet: u32) -> u32 {
  let cardinal = getFacetCardinal(facet);
  let lower = select(cardinal - 1u, 1u, cardinal == 0u);
  let upper = select(cardinal + 1u, 7u, cardinal == 0u);
  return select(lower, upper, (facet & 1u) == 1u);
}

fn getFlowFraction(cell: u32, direction: u32) -> f32 {
  let center = surface[surfaceOffset + cell];
  if (!isFiniteValue(center)) { return 0.0; }
  let row = cell / GRID_WIDTH;
  var bestSlope = 0.0;
  var bestCardinal = 8u;
  var bestDiagonal = 8u;
  var bestDiagonalFraction = 0.0;
  for (var facet = 0u; facet < 8u; facet++) {
    let cardinal = getFacetCardinal(facet);
    let diagonal = getFacetDiagonal(facet);
    let cardinalNeighbor = getD8Neighbor(cell, cardinal);
    let diagonalNeighbor = getD8Neighbor(cell, diagonal);
    var cardinalUsable = false;
    var diagonalUsable = false;
    var e1 = 0.0;
    var e2 = 0.0;
    if (cardinalNeighbor != GRID_NONE) {
      e1 = surface[surfaceOffset + cardinalNeighbor];
      cardinalUsable = isFiniteValue(e1);
    }
    if (diagonalNeighbor != GRID_NONE) {
      e2 = surface[surfaceOffset + diagonalNeighbor];
      diagonalUsable = isFiniteValue(e2);
    }
    if (!cardinalUsable && !diagonalUsable) { continue; }
    let d1 = getD8Distance(cardinal, row);
    let diagonalDistance = getD8Distance(diagonal, row);
    let diagonalRow = f32(row) + f32(getD8RowOffset(diagonal));
    let ground = getGroundCellSize((f32(row) + diagonalRow + 1.0) * 0.5);
    let d2 = select(ground.x, ground.y, cardinal == 0u || cardinal == 4u);
    var slope = 0.0;
    var fractionDiagonal = 0.0;
    if (cardinalUsable && diagonalUsable) {
      let s1 = (center - e1) / d1;
      let s2 = (e1 - e2) / d2;
      if (s1 > 0.0 && s2 == 0.0) {
        // Exactly cardinal: no sqrt/atan, so equal-height ties compare bit-exactly.
        slope = s1;
      } else if (s1 > 0.0 && s2 > 0.0) {
        let angle = atan2(s2, s1);
        let maximumAngle = atan2(d2, d1);
        if (angle <= maximumAngle) {
          slope = sqrt(s1 * s1 + s2 * s2);
          fractionDiagonal = angle / maximumAngle;
        } else {
          slope = (center - e2) / diagonalDistance;
          fractionDiagonal = 1.0;
        }
      } else if (s1 > 0.0) {
        slope = s1;
      } else if (s2 > 0.0) {
        slope = (center - e2) / diagonalDistance;
        fractionDiagonal = 1.0;
      }
    } else if (cardinalUsable) {
      slope = (center - e1) / d1;
    } else {
      slope = (center - e2) / diagonalDistance;
      fractionDiagonal = 1.0;
    }
    // Strictly greater: ties keep the first facet.
    if (slope > bestSlope) {
      bestSlope = slope;
      bestCardinal = cardinal;
      bestDiagonal = diagonal;
      bestDiagonalFraction = fractionDiagonal;
    }
  }
  if (bestCardinal == 8u) { return getFallbackFraction(cell, direction); }
  if (direction == bestDiagonal) { return bestDiagonalFraction; }
  if (direction == bestCardinal) { return 1.0 - bestDiagonalFraction; }
  return 0.0;
}`;
  }
  const quinn = routing === 'mfd-quinn';
  return `${common}
fn getFlowExponent() -> f32 {
  let exponent = settings[settingsOffset + 6u];
  return select(${quinn ? '1.0' : '1.1'}, exponent, isFiniteValue(exponent) && exponent > 0.0);
}

// Slope (Freeman) or slope times contour length (Quinn) of one lower neighbor.
fn getMultipleFlowScore(drop: f32, direction: u32, row: u32) -> f32 {
  let tangent = drop / getD8Distance(direction, row);
  ${
    quinn
      ? `let ground = getGroundCellSize(f32(row) + 0.5);
  var contourLength = 0.25 * getD8Distance(direction, row);
  if (direction == 0u || direction == 4u) { contourLength = 0.5 * ground.y; }
  if (direction == 2u || direction == 6u) { contourLength = 0.5 * ground.x; }
  return tangent * contourLength;`
      : 'return tangent;'
  }
}

fn getFlowFraction(cell: u32, direction: u32) -> f32 {
  let center = surface[surfaceOffset + cell];
  if (!isFiniteValue(center)) { return 0.0; }
  let row = cell / GRID_WIDTH;
  var hasLower = false;
  var maximumScore = 0.0;
  for (var other = 0u; other < 8u; other++) {
    let neighbor = getD8Neighbor(cell, other);
    if (neighbor == GRID_NONE) { continue; }
    let drop = center - surface[surfaceOffset + neighbor];
    if (isFiniteValue(drop) && drop > 0.0) {
      hasLower = true;
      maximumScore = max(maximumScore, getMultipleFlowScore(drop, other, row));
    }
  }
  if (!hasLower) { return getFallbackFraction(cell, direction); }
  let exponent = getFlowExponent();
  var total = 0.0;
  var targetWeight = 0.0;
  for (var other = 0u; other < 8u; other++) {
    let neighbor = getD8Neighbor(cell, other);
    if (neighbor == GRID_NONE) { continue; }
    let drop = center - surface[surfaceOffset + neighbor];
    if (isFiniteValue(drop) && drop > 0.0) {
      let score = getMultipleFlowScore(drop, other, row);
      let ratio = select(1.0, score / maximumScore, maximumScore > 0.0);
      let weight = select(0.0, pow(ratio, exponent), ratio > 0.0);
      total = total + weight;
      if (other == direction) { targetWeight = weight; }
    }
  }
  return select(0.0, targetWeight / total, total > 0.0);
}`;
}

/**
 * Returns D-infinity (Tarboton 1997) or multiple-flow-direction (Freeman 1991; Quinn et al. 1991)
 * accumulation nodes: sentinel fill, loop reset, and `maxIterations` gated rounds.
 *
 * It is a deterministic pull like the D8 accumulation. The accumulation word holds a float32 bit
 * pattern or the sentinel `0xffffffff` (not yet final). A cell finalizes only when every donor
 * with a positive flow fraction toward it is final; its value is `weight + sum(fraction * donor)`
 * summed in fixed direction order, so the bits never depend on scheduling. Each donor's fraction
 * is recomputed from `surface` (and `receivers` for the no-lower-neighbor fallback) instead of
 * being stored, which needs no per-edge buffer. After finalizing a cell a thread walks downstream
 * through the first receiver it also finalizes (up to {@link TERRAIN_FLOW_MAXIMUM_WALK_LENGTH}
 * steps) to keep the round count low. Cell sizes follow `cellSizeMode`; weights are runoff and
 * optional ground area exactly as in the D8 accumulation.
 *
 * `elevation` is accepted for symmetry with the D8 accumulation; invalid cells are recognized by
 * their NaN `surface`.
 *
 * @internal
 */
export function createTerrainFlowRoutingAccumulationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TerrainFlowGrid & {
    id: string;
    routing: TerrainFlowMultipleRouting;
    maxIterations: number;
    /** Canonical elevation, NaN for invalid cells. */
    elevation: GraphDataView<'float32'>;
    /** Routing surface (filled elevation or `elevation`), NaN for invalid cells. */
    surface: GraphDataView<'float32'>;
    /** D8 receivers after optional flat resolution; the fallback for cells with no lower neighbor. */
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
    {name: 'surface', view: props.surface, type: 'f32', access: 'read'},
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
  ];
  if (props.runoff) {
    bindings.push({name: 'runoff', view: props.runoff, type: 'f32', access: 'read'});
  }
  bindings.push({name: 'status', view: state.status, type: 'atomic<u32>', access: 'read_write'});
  const declarations = `${getRasterGridWGSL(props)}
const SENTINEL: u32 = 0xffffffffu;
const MAX_WALK_LENGTH: u32 = ${TERRAIN_FLOW_MAXIMUM_WALK_LENGTH}u;
${getFlowFractionWGSL(props.routing)}

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
    if (neighbor == GRID_NONE) { continue; }
    let fraction = getFlowFraction(neighbor, (direction + 4u) & 7u);
    if (fraction > 0.0) {
      let donorBits = atomicLoad(&acc[accOffset + neighbor]);
      if (donorBits == SENTINEL) { return false; }
      sum = sum + fraction * bitcast<f32>(donorBits);
    }
  }
  return atomicCompareExchangeWeak(&acc[accOffset + cell], SENTINEL, bitcast<u32>(sum)).exchanged;
}`;
  const body = `if (!isFiniteValue(surface[surfaceOffset + index])) { return; }
  if (atomicLoad(&acc[accOffset + index]) != SENTINEL) { return; }
  var walk = index;
  var finalized = tryFinalize(index);
  for (var step = 0u; step < MAX_WALK_LENGTH && finalized; step++) {
    // Try every receiver in direction order; continue from the first one this thread finalized.
    var next = GRID_NONE;
    for (var direction = 0u; direction < 8u; direction++) {
      let receiver = getD8Neighbor(walk, direction);
      if (receiver != GRID_NONE && getFlowFraction(walk, direction) > 0.0) {
        if (tryFinalize(receiver) && next == GRID_NONE) { next = receiver; }
      }
    }
    walk = next;
    finalized = next != GRID_NONE;
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
        variant: `accumulate-round-${props.routing}`,
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
