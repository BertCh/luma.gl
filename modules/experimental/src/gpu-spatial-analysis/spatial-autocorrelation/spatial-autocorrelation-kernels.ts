// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {getGPUSpatialWeightsNeighborLoopWGSL} from '../spatial-weights/spatial-weights';
import {
  resolveGPUIslandPolicy,
  validateGPUSpatialWeights,
  type GPUIslandPolicy,
  type GPUSpatialWeights
} from '../spatial-weights/index';
import {
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH
} from './spatial-autocorrelation-parameters';
import {GET_QUIET_NAN_SEEDED_WGSL, IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';
import {validateCapacity, validatePackedView} from '../../utils/gpu-contributor-utils';
import {
  createFloat32ValuesNodes,
  type GPUFloat32OrUint32Values,
  validateFloat32OrUint32Values
} from '../spatial-weights/float32-values';

/** Rows reduced by one workgroup in the first level of the deterministic global sums. */
const BLOCK_ROWS = 4096;

/** Threads of the moment-sum workgroups. */
const MOMENT_WORKGROUP_SIZE = 256;

/** Row state written by the island pass: not part of the analysis (masked or non-finite). */
const ROW_STATE_MISSING = 0;
/** Row state: included and has at least one active neighbor. */
const ROW_STATE_TESTED = 1;
/** Row state: included, but no active neighbor (an island). */
const ROW_STATE_ISLAND = 2;

/** Inputs shared by both spatial-autocorrelation contributors. @internal */
export type SpatialAutocorrelationInputProps = {
  id: string;
  weights: GPUSpatialWeights;
  /** Analysis values, `float32` or `uint32` (converted to f32 by one node, exact below 2^24). */
  values: GPUFloat32OrUint32Values;
  parameters: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  globalStatistics?: GraphDataView<'float32'>;
  /**
   * Opt-in island pass. When omitted, no island pass runs, no row is classified as an island and
   * the island count is 0 (the behavior of external callers that predate islands). When given,
   * `'exclude'` removes islands from `n`, the mean and the variance and writes NaN for them;
   * `'keep'` keeps them in the moments and only counts them.
   */
  islands?: GPUIslandPolicy;
};

/** Moment views produced by {@link getSpatialAutocorrelationInputNodes}. @internal */
export type SpatialAutocorrelationInputs<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /**
   * `rows + 5` floats: centered values `x - mean` (quiet NaN for rows outside the analysis)
   * followed by the moments `[n, mean, variance, sumOfSquares, islandCount]`.
   */
  statistics: GraphDataView<'float32'>;
};

/** Index of the island count inside the moments that follow the centered values. @internal */
export const SPATIAL_AUTOCORRELATION_ISLAND_COUNT_SLOT = 4;

/**
 * Validates an optional minimum active-neighbor count and returns it (default 1).
 *
 * @internal
 */
export function resolveSpatialAutocorrelationMinimumNeighbors(
  id: string,
  minimumNeighbors: number | undefined
): number {
  const minimum = minimumNeighbors ?? 1;
  validateCapacity(id, 'minimumNeighbors', minimum);
  return minimum;
}

/**
 * Validates the props shared by both contributors and returns the row count.
 *
 * @internal
 */
export function validateSpatialAutocorrelationInputs(
  props: SpatialAutocorrelationInputProps
): number {
  const {id} = props;
  const rows = validateGPUSpatialWeights(id, props.weights);
  validateFloat32OrUint32Values(props.values, `${id} values`);
  validatePackedView(props.parameters, ['float32'], `${id} parameters`);
  if (props.islands !== undefined) {
    resolveGPUIslandPolicy(id, props.islands);
  }
  if (props.weights.status?.overflow) {
    validatePackedUint32View(props.weights.status.overflow, `${id} weights.status.overflow`);
  }
  // Counts and ranks are compared as f32 inside the kernels.
  validateCapacity(id, 'weights row count', rows, 1, 2 ** 24 - 1);
  if (props.values.length !== rows) {
    throw new Error(`${id} values length must equal the weights row count`);
  }
  validateCapacity(
    id,
    'parameters length',
    props.parameters.length,
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );
  if (props.mask) {
    validatePackedUint32View(props.mask, `${id} mask`);
    if (props.mask.length !== rows) {
      throw new Error(`${id} mask length must equal the weights row count`);
    }
  }
  if (props.globalStatistics) {
    validatePackedView(props.globalStatistics, ['float32'], `${id} output.globalStatistics`);
    validateCapacity(
      id,
      'output.globalStatistics length',
      props.globalStatistics.length,
      GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH
    );
  }
  return rows;
}

/**
 * Parameter-free WGSL helpers: finiteness test, runtime quiet NaN, and the two-sided normal
 * p-value. Included by {@link getSpatialAutocorrelationSharedWGSL}.
 *
 * @internal
 */
export const SPATIAL_AUTOCORRELATION_FLOAT_WGSL = /* wgsl */ `
${IS_FINITE_FLOAT_WGSL}

// A runtime operand keeps the quiet-NaN bit pattern out of constant evaluation.
${GET_QUIET_NAN_SEEDED_WGSL}

// Two-sided normal p-value erfc(|z| / sqrt(2)) with the Numerical Recipes erfcc Chebyshev fit
// (fractional error below 1.2e-7 in exact arithmetic), so tiny p-values keep relative accuracy.
fn getTwoSidedPValue(z: f32) -> f32 {
  let x = abs(z) * 0.70710678118654752;
  let t = 1.0 / (1.0 + 0.5 * x);
  let exponent = -x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277))))))));
  return min(t * exp(exponent), 1.0);
}
`;

/**
 * WGSL shared by every kernel that reads the parameters: `readParameter`, quiet NaN, finiteness
 * and the two-sided normal p-value. Kernels that include it must bind `parameters` as
 * `array<f32>`.
 *
 * @internal
 */
export function getSpatialAutocorrelationSharedWGSL(): string {
  return /* wgsl */ `
fn readParameter(slot: u32) -> f32 {
  return parameters[parametersOffset + slot];
}

${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
`;
}

/**
 * WGSL constants every neighbor kernel of the weights needs: `MOMENTS` (offset of the moments
 * inside `statistics`), `ROWS` and `CAPACITY` (the slot capacity, `neighbors.length`).
 *
 * @internal
 */
export function getSpatialAutocorrelationConstantsWGSL(rows: number, capacity: number): string {
  return `const MOMENTS: u32 = ${rows}u;
const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${capacity}u;`;
}

/**
 * WGSL statements that visit, in CSR slot order (ascending neighbor ID), every active neighbor of
 * row `index` (see `getGPUSpatialWeightsNeighborLoopWGSL`: a valid neighbor other than `index`
 * with a positive weight) that is part of the analysis, and run `action` with `neighbor` (row
 * index) and `weight` (`w_ij`) in scope. Rows outside the analysis (masked, missing, and islands
 * under `'exclude'`) hold NaN in `statistics`. Requires bindings `offsets`, `neighbors`,
 * `weights` and `statistics`, and the constants `ROWS` and `CAPACITY`.
 *
 * The order is fixed by the weights, so per-row sums are reproducible.
 *
 * @internal
 */
export function getSpatialWeightsNeighborLoopWGSL(action: string): string {
  return getGPUSpatialWeightsNeighborLoopWGSL(
    'isFiniteFloat(statistics[statisticsOffset + neighbor])',
    action
  );
}

/**
 * Builds the shared front end of both contributors: per-row validity, the optional island pass and
 * the global moments.
 *
 * Moments come from fixed-order two-level workgroup tree sums over the rows (no float atomics):
 * first the count (u32) and a provisional mean, then a second pass that adds the mean of the
 * deviations from it (`mean += sum(x - mean) / n`, which removes the rounding error of a large
 * offset), then the sum of squared centered values, so the variance never subtracts two large
 * numbers. Fixed moments in the parameters replace the count, mean and variance. When
 * `weights.status.overflow` is nonzero the weights are truncated and the mean becomes NaN, which
 * makes every output NaN.
 *
 * With `props.islands` given, a row-state kernel marks islands (included rows without an active
 * neighbor) first. `'exclude'` leaves them out of the moments (their `statistics` entry is NaN);
 * `'keep'` keeps them in. The moments carry the island count either way.
 *
 * @internal
 */
export function getSpatialAutocorrelationInputNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SpatialAutocorrelationInputProps & {operation: string}
): SpatialAutocorrelationInputs<Parameters> {
  const {id, operation, parameters, mask, weights} = props;
  const valuesInput = createFloat32ValuesNodes(graph, `${id}-values`, props.values);
  const values = valuesInput.view;
  const rows = values.length;
  const blockCount = Math.ceil(rows / BLOCK_ROWS);
  const sharedWGSL = getSpatialAutocorrelationSharedWGSL();
  const nodes: GPUCommandNode<Parameters>[] = [...valuesInput.nodes];
  const hasIslandPass = props.islands !== undefined;
  const keepIslands = props.islands === 'keep';
  const overflow = weights.status?.overflow;

  const valuePartials = createTransientView(graph, `${id}-value-partials`, 'float32', blockCount);
  const countPartials = createTransientView(graph, `${id}-count-partials`, 'uint32', blockCount);
  const islandPartials = hasIslandPass
    ? createTransientView(graph, `${id}-island-partials`, 'uint32', blockCount)
    : undefined;
  const deviationPartials = createTransientView(
    graph,
    `${id}-deviation-partials`,
    'float32',
    blockCount
  );
  const squarePartials = createTransientView(graph, `${id}-square-partials`, 'float32', blockCount);
  const statistics = createTransientView(graph, `${id}-statistics`, 'float32', rows + 5);
  const rowStates = hasIslandPass
    ? createTransientView(graph, `${id}-row-states`, 'uint32', rows)
    : undefined;
  const maskBindings: WGSLKernelBinding[] = mask
    ? [{name: 'mask', view: mask, type: 'u32', access: 'read'}]
    : [];
  const isRawRowValidWGSL = `fn isRawRowValid(row: u32) -> bool {
  return ${mask ? 'mask[maskOffset + row] != 0u && ' : ''}isFiniteFloat(values[valuesOffset + row]);
}`;
  // Rows that enter the moments: valid rows, minus the islands under 'exclude'.
  const momentRowBindings: WGSLKernelBinding[] = rowStates
    ? [{name: 'rowStates', view: rowStates, type: 'u32', access: 'read'}]
    : maskBindings;
  const blockDeclarations = `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
const MOMENTS: u32 = ${rows}u;
const ROW_COUNT: u32 = ${rows}u;
const BLOCK_ROWS: u32 = ${BLOCK_ROWS}u;
const WORKGROUP_SIZE: u32 = ${MOMENT_WORKGROUP_SIZE}u;
${
  rowStates
    ? `fn isRowValid(row: u32) -> bool {
  let state = rowStates[rowStatesOffset + row];
  return state == ${ROW_STATE_TESTED}u || (state == ${ROW_STATE_ISLAND}u && ${keepIslands});
}`
    : isRawRowValidWGSL.replace('isRawRowValid', 'isRowValid')
}`;

  if (rowStates) {
    // Row states: 0 missing, 1 tested, 2 island. A neighbor counts when it is a valid row.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-row-states`,
        operation,
        variant: 'row-states',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'values', view: values, type: 'f32', access: 'read'},
          ...maskBindings,
          {name: 'rowStates', view: rowStates, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${weights.neighbors.length}u;
${isRawRowValidWGSL}`,
        body: `if (!isRawRowValid(index)) {
    rowStates[rowStatesOffset + index] = ${ROW_STATE_MISSING}u;
    return;
  }
  var hasNeighbor = false;
  ${getGPUSpatialWeightsNeighborLoopWGSL('isRawRowValid(neighbor)', 'hasNeighbor = true; break;')}
  rowStates[rowStatesOffset + index] = select(${ROW_STATE_ISLAND}u, ${ROW_STATE_TESTED}u, hasNeighbor);`
      })
    );
  }

  // Each level-1 workgroup reduces one block of rows: thread t sums rows block + t, + 256, ... and a
  // fixed binary tree combines the threads, so the order depends only on the shape and the sums
  // are bitwise reproducible. Validity is evaluated in place (no per-row validity or term
  // buffers).
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-blocks`,
      operation,
      variant: 'moments-blocks',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...momentRowBindings,
        {name: 'valuePartials', view: valuePartials, type: 'f32', access: 'read_write'},
        {name: 'countPartials', view: countPartials, type: 'u32', access: 'read_write'},
        ...(islandPartials
          ? [
              {
                name: 'islandPartials',
                view: islandPartials,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      workgroupSize: MOMENT_WORKGROUP_SIZE,
      invocationCount: blockCount * MOMENT_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `${blockDeclarations}
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;
var<workgroup> partialCounts: array<u32, ${MOMENT_WORKGROUP_SIZE}>;
var<workgroup> partialIslands: array<u32, ${MOMENT_WORKGROUP_SIZE}>;`,
      // No early return: every invocation of a workgroup must reach the barriers.
      body: `let block = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  var count = 0u;
  var islands = 0u;
  if (isInRange) {
    let end = min((block + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      if (isRowValid(row)) {
        sum += values[valuesOffset + row];
        count++;
      }
      ${rowStates ? `islands += select(0u, 1u, rowStates[rowStatesOffset + row] == ${ROW_STATE_ISLAND}u);` : ''}
    }
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  partialIslands[localInvocationIndex] = islands;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
      partialIslands[localInvocationIndex] += partialIslands[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    valuePartials[valuePartialsOffset + block] = partialSums[0];
    countPartials[countPartialsOffset + block] = partialCounts[0];
    ${islandPartials ? 'islandPartials[islandPartialsOffset + block] = partialIslands[0];' : ''}
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-mean`,
      operation,
      variant: 'moments-mean',
      bindings: [
        {name: 'valuePartials', view: valuePartials, type: 'f32', access: 'read'},
        {name: 'countPartials', view: countPartials, type: 'u32', access: 'read'},
        ...(islandPartials
          ? [
              {
                name: 'islandPartials',
                view: islandPartials,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'}
      ],
      workgroupSize: MOMENT_WORKGROUP_SIZE,
      invocationCount: MOMENT_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
const MOMENTS: u32 = ${rows}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${MOMENT_WORKGROUP_SIZE}u;
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;
var<workgroup> partialCounts: array<u32, ${MOMENT_WORKGROUP_SIZE}>;
var<workgroup> partialIslands: array<u32, ${MOMENT_WORKGROUP_SIZE}>;`,
      body: `var sum = 0.0;
  var count = 0u;
  var islands = 0u;
  for (var block = localInvocationIndex; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    sum += valuePartials[valuePartialsOffset + block];
    count += countPartials[countPartialsOffset + block];
    ${islandPartials ? 'islands += islandPartials[islandPartialsOffset + block];' : ''}
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  partialIslands[localInvocationIndex] = islands;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
      partialIslands[localInvocationIndex] += partialIslands[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    // The count is exact in u32 and converted to f32 only here. The mean is provisional: the
    // deviation pass corrects it.
    let total = f32(partialCounts[0]);
    statistics[statisticsOffset + MOMENTS] = total;
    statistics[statisticsOffset + MOMENTS + 1u] = select(getQuietNaN(index), partialSums[0] / total, total >= 1.0);
    statistics[statisticsOffset + MOMENTS + ${SPATIAL_AUTOCORRELATION_ISLAND_COUNT_SLOT}u] = f32(partialIslands[0]);
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-deviations`,
      operation,
      variant: 'moments-deviations',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...momentRowBindings,
        {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
        {name: 'deviationPartials', view: deviationPartials, type: 'f32', access: 'read_write'}
      ],
      workgroupSize: MOMENT_WORKGROUP_SIZE,
      invocationCount: blockCount * MOMENT_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `${blockDeclarations}
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;`,
      // Sum of the deviations from the provisional mean, whose rounding error it measures.
      body: `let block = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  if (isInRange) {
    let provisionalMean = statistics[statisticsOffset + MOMENTS + 1u];
    let end = min((block + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      if (isRowValid(row)) {
        sum += values[valuesOffset + row] - provisionalMean;
      }
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    deviationPartials[deviationPartialsOffset + block] = partialSums[0];
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-correct`,
      operation,
      variant: 'moments-correct',
      bindings: [
        {name: 'deviationPartials', view: deviationPartials, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'},
        ...(overflow
          ? [{name: 'overflow', view: overflow, type: 'u32' as const, access: 'read' as const}]
          : [])
      ],
      workgroupSize: MOMENT_WORKGROUP_SIZE,
      invocationCount: MOMENT_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `${sharedWGSL}
const MOMENTS: u32 = ${rows}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${MOMENT_WORKGROUP_SIZE}u;
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;`,
      body: `var sum = 0.0;
  for (var block = localInvocationIndex; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    sum += deviationPartials[deviationPartialsOffset + block];
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    var total = statistics[statisticsOffset + MOMENTS];
    var mean = statistics[statisticsOffset + MOMENTS + 1u];
    if (total >= 1.0) {
      mean += partialSums[0] / total;
    }
    if (readParameter(1u) != 0.0) {
      total = readParameter(2u);
      mean = readParameter(3u);
    }
    ${
      overflow
        ? `// Truncated weights: no statistic from them is trustworthy, so every output becomes NaN.
    if (overflow[overflowOffset] != 0u) {
      mean = getQuietNaN(index);
    }`
        : ''
    }
    statistics[statisticsOffset + MOMENTS] = total;
    statistics[statisticsOffset + MOMENTS + 1u] = mean;
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-center`,
      operation,
      variant: 'center',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...momentRowBindings,
        {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'},
        {name: 'squarePartials', view: squarePartials, type: 'f32', access: 'read_write'}
      ],
      workgroupSize: MOMENT_WORKGROUP_SIZE,
      invocationCount: blockCount * MOMENT_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `${blockDeclarations}
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;`,
      // Centers each row and reduces the squared deviations of its block in the same pass.
      body: `let block = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  if (isInRange) {
    let mean = statistics[statisticsOffset + MOMENTS + 1u];
    let end = min((block + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      let valid = isRowValid(row);
      let centered = values[valuesOffset + row] - mean;
      // Rows outside the analysis store quiet NaN, which every later kernel reads as "not a focus
      // row and not a neighbor".
      statistics[statisticsOffset + row] = select(getQuietNaN(row), centered, valid);
      sum += select(0.0, centered * centered, valid);
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    squarePartials[squarePartialsOffset + block] = partialSums[0];
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-variance`,
      operation,
      variant: 'moments-variance',
      bindings: [
        {name: 'squarePartials', view: squarePartials, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'},
        ...(props.globalStatistics
          ? [
              {
                name: 'globalStatistics',
                view: props.globalStatistics,
                type: 'f32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      workgroupSize: MOMENT_WORKGROUP_SIZE,
      invocationCount: MOMENT_WORKGROUP_SIZE,
      guardIndex: false,
      declarations: `${sharedWGSL}
const MOMENTS: u32 = ${rows}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${MOMENT_WORKGROUP_SIZE}u;
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;`,
      body: `var sum = 0.0;
  for (var block = localInvocationIndex; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    sum += squarePartials[squarePartialsOffset + block];
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let count = statistics[statisticsOffset + MOMENTS];
    let mean = statistics[statisticsOffset + MOMENTS + 1u];
    var sumOfSquares = partialSums[0];
    var variance = select(getQuietNaN(index), sumOfSquares / count, count >= 1.0);
    if (readParameter(1u) != 0.0) {
      variance = readParameter(4u);
      sumOfSquares = variance * count;
    }
    statistics[statisticsOffset + MOMENTS + 2u] = variance;
    statistics[statisticsOffset + MOMENTS + 3u] = sumOfSquares;
    ${
      props.globalStatistics
        ? `globalStatistics[globalStatisticsOffset] = count;
    globalStatistics[globalStatisticsOffset + 1u] = mean;
    globalStatistics[globalStatisticsOffset + 2u] = variance;
    globalStatistics[globalStatisticsOffset + 3u] = sqrt(variance);
    globalStatistics[globalStatisticsOffset + ${SPATIAL_AUTOCORRELATION_ISLAND_COUNT_SLOT}u] = statistics[statisticsOffset + MOMENTS + ${SPATIAL_AUTOCORRELATION_ISLAND_COUNT_SLOT}u];`
        : ''
    }
  }`
    })
  );
  return {nodes, statistics};
}
