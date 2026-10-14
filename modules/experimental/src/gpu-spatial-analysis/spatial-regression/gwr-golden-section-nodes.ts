// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLDispatchGate,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_DELTA,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_RELATIVE_TOLERANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_TOLERANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE
} from './geographically-weighted-regression-parameters';

/** Inputs of {@link createGoldenSectionSearchNodes}. @internal */
export type GoldenSectionSearchProps = {
  /** Prefix of the node and transient IDs. */
  id: string;
  /** Operation name of the generated kernels. */
  operation: string;
  rowCount: number;
  /** Rows summed by one thread of a tile pass. */
  tileRows: number;
  tileCount: number;
  /** Candidates evaluated together by pass 0 (2: the first iteration evaluates both interior points). */
  candidateWidth: number;
  /** Compile-time `maximumSearchIterations`. */
  maximumIterations: number;
  /** Rows of the candidate table, `maximumIterations + 1`. */
  tableRowCount: number;
  /** Declarations shared by every GWR kernel (the constants and the NaN helpers). */
  common: string;
  /** Candidate declarations: the pass parameter reader and the local fit helpers. */
  candidateDeclarations: string;
  /** WGSL that maps the invocation index to the focal row (grid order or row order). */
  focalRowWGSL: string;
  /** True when a `GPUGridIndex` is bound (its bindings are in `candidateBindings`). */
  hasGrid: boolean;
  /** Bindings of the candidate-evaluation kernel except `candidateScratch`, in its order. */
  candidateBindings: WGSLKernelBinding[];
  /** Row validity flags (the head of the shared row-state buffer). */
  rowValid: GraphDataView<'uint32'>;
  /** `[cursor, block 0, block 1, ...]` ladders: block `i` holds the candidates of pass `i`. */
  ladders: GraphDataView<'float32'>;
  /** The caller's parameters (bounds in the first two ladder values). */
  parameters: GraphDataView<'float32'>;
  candidateScratch: GraphDataView<'float32'>;
  tileCandidates: GraphDataView<'float32'>;
  candidateTable: GraphDataView<'float32'>;
  searchState: GraphDataView<'float32'>;
};

/** Nodes and gate of the golden-section search. @internal */
export type GoldenSectionSearchNodes<Parameters> = {
  /** Bound computation and search initialization, in order, before the first candidate pass. */
  setupNodes: GPUCommandNode<Parameters>[];
  /**
   * Gate for the candidate passes after the first: slot 0 candidates (`rowCount` invocations),
   * slot 1 tile sums (`tileCount * candidateWidth`), slot 2 selection (`candidateWidth`).
   */
  gate: WGSLDispatchGate<Parameters>;
  /** The one-thread kernel that compares the two interior scores after pass `iteration - 1`. */
  createIterationNode: (iteration: number) => GPUCommandNode<Parameters>;
};

const STATE = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE;

/**
 * Creates the golden-section bandwidth search (mgwr `golden_section`) as GPU nodes: the default
 * bound computation, the initialization, and the per-iteration state update. The caller wires the
 * candidate passes between them (pass `i` reads block `i` of `ladders`), gated by `gate` after
 * pass 0.
 *
 * State words (float32, see `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SEARCH_STATE`): the bracket
 * `a, b, c, d`, the table rows of the scores of `b` and `d`, the iteration count, the armed flag,
 * the table row of the last winner, the evaluation count and the converged flag. The loop index
 * is the cursor word of `ladders`, never WGSL text (rule 12), and a converged search zeroes the
 * dispatch triples of the later passes (rule 13).
 *
 * @internal
 */
export function createGoldenSectionSearchNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GoldenSectionSearchProps
): GoldenSectionSearchNodes<Parameters> {
  const {id, operation, rowCount, tileCount, candidateWidth} = props;
  const gate = createWGSLDispatchGate<Parameters>(graph, {
    id: `${id}-search-dispatch`,
    operation,
    shapes: [[rowCount], [tileCount * candidateWidth], [candidateWidth]]
  });
  const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
    name,
    view,
    type,
    access: 'read'
  });
  const write = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
    name,
    view,
    type,
    access: 'read_write'
  });
  const stateConstants = Object.entries(STATE)
    .map(([name, slot]) => `const STATE_${name}: u32 = ${slot}u;`)
    .join('\n');
  const searchCommon = `${props.common}
const MAXIMUM_ITERATIONS: u32 = ${props.maximumIterations}u;
const GOLDEN_DELTA: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_DELTA};
const TOLERANCE_ABSOLUTE: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_TOLERANCE};
const TOLERANCE_RELATIVE: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_GOLDEN_SECTION_RELATIVE_TOLERANCE};
${stateConstants}`;
  // Scratch per row: [nearest positive distance, farthest distance]; SENTINEL / 0 when unused.
  const boundsRowsBody = `${props.focalRowWGSL}
  let slot = candidateScratchOffset + focal * 2u;
  candidateScratch[slot] = SENTINEL;
  candidateScratch[slot + 1u] = 0.0;
  if (getIsAdaptive() || rowValid[rowValidOffset + focal] == 0u) {
    return;
  }
  let origin = getPosition(focal);
  var nearest = SENTINEL;
  var farthest = 0.0;
  ${
    props.hasGrid
      ? `// The two smallest squared distances: the row itself, then its nearest neighbour. A
  // coincident row gives 0, which is no positive distance (as for the scan below).
  var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
  selectNeighborDistances(focal, 2u, &neighbors);
  if (neighbors[1] > 0.0 && neighbors[1] < SENTINEL) {
    nearest = sqrt(neighbors[1]);
  }
  // The farthest row is no farther than the farthest corner of the grid extent.
  let boundsBase = sortedIdsOffset + ROW_COUNT;
  let minimum = vec2f(bitcast<f32>(sortedIds[boundsBase]), bitcast<f32>(sortedIds[boundsBase + 1u]));
  let maximum = vec2f(bitcast<f32>(sortedIds[boundsBase + 2u]), bitcast<f32>(sortedIds[boundsBase + 3u]));
  farthest = max(
    max(getDistance(origin, minimum), getDistance(origin, maximum)),
    max(getDistance(origin, vec2f(minimum.x, maximum.y)), getDistance(origin, vec2f(maximum.x, minimum.y)))
  );`
      : `for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    let distance = getDistance(origin, getPosition(row));
    if (distance > 0.0) {
      nearest = min(nearest, distance);
    }
    farthest = max(farthest, distance);
  }`
  }
  candidateScratch[slot] = nearest;
  candidateScratch[slot + 1u] = farthest;`;
  const setupNodes: GPUCommandNode<Parameters>[] = [
    // Per row: nearest positive distance and farthest distance (fixed bandwidths only).
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-search-bounds-rows`,
      operation,
      variant: 'search-bounds-rows',
      bindings: [
        ...props.candidateBindings,
        write('candidateScratch', props.candidateScratch, 'f32')
      ],
      invocationCount: rowCount,
      declarations: props.candidateDeclarations,
      body: boundsRowsBody
    }),
    // Per tile: smallest nearest distance, largest farthest distance, included rows.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-search-bounds-tiles`,
      operation,
      variant: 'search-bounds-tiles',
      bindings: [
        read('candidateScratch', props.candidateScratch, 'f32'),
        read('rowValid', props.rowValid, 'u32'),
        write('tileCandidates', props.tileCandidates, 'f32')
      ],
      invocationCount: tileCount,
      declarations: searchCommon,
      body: `let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var nearest = SENTINEL;
  var farthest = 0.0;
  var included = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    included = included + 1.0;
    nearest = min(nearest, candidateScratch[candidateScratchOffset + row * 2u]);
    farthest = max(farthest, candidateScratch[candidateScratchOffset + row * 2u + 1u]);
  }
  let base = tileCandidatesOffset + index * 3u;
  tileCandidates[base] = nearest;
  tileCandidates[base + 1u] = farthest;
  tileCandidates[base + 2u] = included;`
    }),
    // Bounds, interior points, block 0 of the ladders, the state and the gate.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-search-init`,
      operation,
      variant: 'search-init',
      bindings: [
        read('params', props.parameters, 'f32'),
        write('ladders', props.ladders, 'f32'),
        read('tileCandidates', props.tileCandidates, 'f32'),
        write('table', props.candidateTable, 'f32'),
        write('state', props.searchState, 'f32'),
        gate.binding
      ],
      invocationCount: 1,
      declarations: searchCommon,
      body: `let nan = getNaN();
  for (var row = 0u; row < TABLE_ROWS; row++) {
    for (var field = 0u; field < TABLE_STRIDE; field++) {
      table[tableOffset + row * TABLE_STRIDE + field] = nan;
    }
  }
  let isAdaptive = params[paramsOffset + 1u] > 0.5;
  let callerCount = u32(max(params[paramsOffset + 2u], 0.0));
  var nearest = SENTINEL;
  var farthest = 0.0;
  var included = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let base = tileCandidatesOffset + tile * 3u;
    nearest = min(nearest, tileCandidates[base]);
    farthest = max(farthest, tileCandidates[base + 1u]);
    included = included + tileCandidates[base + 2u];
  }
  // mgwr _init_section: adaptive [40 + 2 P, n] (n capped at the neighbour list), fixed
  // [min_dist / 2, 2 max_dist]; a positive caller bound replaces its default.
  var lower = 0.0;
  var upper = 0.0;
  if (isAdaptive) {
    lower = f32(40u + 2u * P);
    upper = min(included, f32(MAXIMUM_NEIGHBORS));
  } else if (included > 0.0 && nearest < SENTINEL) {
    lower = 0.5 * nearest;
    upper = 2.0 * farthest;
  }
  let isSearch = callerCount >= 2u;
  if (isSearch) {
    let givenLower = params[paramsOffset + HEADER];
    let givenUpper = params[paramsOffset + HEADER + 1u];
    if (givenLower > 0.0 && isFiniteBits(givenLower)) {
      lower = givenLower;
    }
    if (givenUpper > 0.0 && isFiniteBits(givenUpper)) {
      upper = givenUpper;
    }
  }
  let low = min(lower, upper);
  let high = max(lower, upper);
  var first = low + GOLDEN_DELTA * (high - low);
  var second = high - GOLDEN_DELTA * (high - low);
  if (isAdaptive) {
    // numpy round: half to even, as WGSL round.
    first = round(first);
    second = round(second);
  }
  if (isSearch) {
    // Block 0 holds the caller header; it now carries the two interior points of iteration 1.
    ladders[laddersOffset + 3u] = 2.0;
    ladders[laddersOffset + 1u + HEADER] = first;
    ladders[laddersOffset + 2u + HEADER] = second;
  }
  state[stateOffset + STATE_LOWER] = low;
  state[stateOffset + STATE_LOWER_INTERIOR] = first;
  state[stateOffset + STATE_UPPER] = high;
  state[stateOffset + STATE_UPPER_INTERIOR] = second;
  state[stateOffset + STATE_LOWER_INTERIOR_ROW] = 0.0;
  state[stateOffset + STATE_UPPER_INTERIOR_ROW] = 1.0;
  state[stateOffset + STATE_ITERATION_COUNT] = 0.0;
  state[stateOffset + STATE_ACTIVE] = select(0.0, 1.0, isSearch);
  // A single caller value is a fixed bandwidth: its evaluation is table row 0.
  state[stateOffset + STATE_OPTIMUM_ROW] = select(0.0, -1.0, isSearch);
  state[stateOffset + STATE_EVALUATION_COUNT] = select(1.0, 2.0, isSearch);
  state[stateOffset + STATE_CONVERGED] = 0.0;
  state[stateOffset + STATE_RESERVED] = 0.0;
  if (isSearch) {
    ${gate.armWGSL}
  } else {
    ${gate.disarmWGSL}
  }`
    })
  ];

  const createIterationNode = (iteration: number): GPUCommandNode<Parameters> =>
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-search-${iteration}-iterate`,
      operation,
      variant: 'search-iterate',
      bindings: [
        write('ladders', props.ladders, 'f32'),
        read('table', props.candidateTable, 'f32'),
        write('state', props.searchState, 'f32'),
        gate.binding
      ],
      invocationCount: 1,
      declarations: `${searchCommon}
fn getRowScore(row: u32) -> f32 {
  return table[tableOffset + row * TABLE_STRIDE + 1u];
}
fn getRowSingular(row: u32) -> f32 {
  return table[tableOffset + row * TABLE_STRIDE + 2u];
}`,
      // The iteration index is the state word, not this node's position: one pipeline for all.
      body: `if (state[stateOffset + STATE_ACTIVE] < 0.5) {
    return;
  }
  let iteration = u32(state[stateOffset + STATE_ITERATION_COUNT]) + 1u;
  let firstRow = u32(state[stateOffset + STATE_LOWER_INTERIOR_ROW]);
  let secondRow = u32(state[stateOffset + STATE_UPPER_INTERIOR_ROW]);
  let firstScore = getRowScore(firstRow);
  let secondScore = getRowScore(secondRow);
  let firstSingular = getRowSingular(firstRow);
  let secondSingular = getRowSingular(secondRow);
  let isFirstValid = isFiniteBits(firstScore);
  let isSecondValid = isFiniteBits(secondScore);
  // mgwr: score_b <= score_d keeps the lower interval. Fewest singular rows first; an invalid
  // score loses to a valid one; with two invalid scores the comparison is false, as for NaN.
  var firstWins = isFirstValid;
  if (isFirstValid && isSecondValid) {
    firstWins = firstSingular < secondSingular || (firstSingular == secondSingular && firstScore <= secondScore);
  }
  let isAdaptive = ladders[laddersOffset + 2u] > 0.5;
  var lower = state[stateOffset + STATE_LOWER];
  var first = state[stateOffset + STATE_LOWER_INTERIOR];
  var upper = state[stateOffset + STATE_UPPER];
  var second = state[stateOffset + STATE_UPPER_INTERIOR];
  let newRow = iteration + 1u;
  var optimumRow = secondRow;
  var nextFirstRow = firstRow;
  var nextSecondRow = secondRow;
  var newValue = 0.0;
  if (firstWins) {
    // c = d; d = b; b = a + delta |c - a|
    optimumRow = firstRow;
    upper = second;
    second = first;
    nextSecondRow = firstRow;
    first = lower + GOLDEN_DELTA * abs(upper - lower);
    if (isAdaptive) {
      first = round(first);
    }
    newValue = first;
    nextFirstRow = newRow;
  } else {
    // a = b; b = d; d = c - delta |c - a|
    lower = first;
    first = second;
    nextFirstRow = secondRow;
    second = upper - GOLDEN_DELTA * abs(upper - lower);
    if (isAdaptive) {
      second = round(second);
    }
    newValue = second;
    nextSecondRow = newRow;
  }
  // mgwr loops while |score_b - score_d| > tol; the difference is of the pair just compared.
  var keepSearching = true;
  if (isFirstValid && isSecondValid && firstSingular == secondSingular) {
    let tolerance = max(TOLERANCE_ABSOLUTE, TOLERANCE_RELATIVE * max(abs(firstScore), abs(secondScore)));
    keepSearching = abs(firstScore - secondScore) > tolerance;
  } else if (!isFirstValid && !isSecondValid) {
    keepSearching = false;
  }
  state[stateOffset + STATE_LOWER] = lower;
  state[stateOffset + STATE_LOWER_INTERIOR] = first;
  state[stateOffset + STATE_UPPER] = upper;
  state[stateOffset + STATE_UPPER_INTERIOR] = second;
  state[stateOffset + STATE_LOWER_INTERIOR_ROW] = f32(nextFirstRow);
  state[stateOffset + STATE_UPPER_INTERIOR_ROW] = f32(nextSecondRow);
  state[stateOffset + STATE_ITERATION_COUNT] = f32(iteration);
  state[stateOffset + STATE_OPTIMUM_ROW] = f32(optimumRow);
  if (!keepSearching || iteration >= MAXIMUM_ITERATIONS) {
    // Converged or out of iterations: no new candidate, and the later passes launch nothing.
    state[stateOffset + STATE_ACTIVE] = 0.0;
    state[stateOffset + STATE_CONVERGED] = select(0.0, 1.0, !keepSearching);
    ${gate.disarmWGSL}
    return;
  }
  // Schedule the one new candidate as block iteration of the ladders and advance the GPU cursor.
  ladders[laddersOffset] = f32(iteration);
  let blockBase = laddersOffset + 1u + iteration * PARAMETER_LENGTH;
  ladders[blockBase] = ladders[laddersOffset + 1u];
  ladders[blockBase + 1u] = ladders[laddersOffset + 2u];
  ladders[blockBase + 2u] = 1.0;
  ladders[blockBase + 3u] = ladders[laddersOffset + 4u];
  ladders[blockBase + HEADER] = newValue;
  state[stateOffset + STATE_EVALUATION_COUNT] = f32(newRow + 1u);`
    });

  return {setupNodes, gate, createIterationNode};
}
