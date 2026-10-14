// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {createSegmentSumNode} from '../../utils/sorted-segment-sums';
import {
  getGPUSpatialWeightsNeighborLoopWGSL,
  isGPUSpatialWeightsRowStandardized,
  type GPUIslandPolicy,
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';
import {GPU_PERMUTATION_PARAMETER_LENGTH} from './permutation-parameters';
import {validateCapacity} from '../../utils/gpu-contributor-utils';
import {
  createFloat32ValuesNodes,
  type GPUFloat32OrUint32Values,
  validateFloat32OrUint32Values
} from '../spatial-weights/float32-values';

/** Rows reduced by one workgroup in the first level of the deterministic column sums. @internal */
export const BLOCK_ROWS = 4096;

/** `rowPositions` value of an excluded row. @internal */
export const PERMUTATION_INVALID_POSITION = 0xffffffff;

/**
 * Slots of the `totals` view produced by {@link getPermutationInputNodes}.
 *
 * @internal
 */
export const PERMUTATION_TOTAL = {
  count: 0,
  sumX: 1,
  sumY: 2,
  sumSquares: 3,
  sumSquaresY: 4,
  s0: 5,
  mean: 6,
  meanY: 7,
  /** Sum over analysis rows of the squared active row weight sum, `1^T W^T W 1` (Lee's L denominator). */
  s0Squared: 8
} as const;

/** Inputs shared by both permutation tests. @internal */
export type PermutationInputProps = {
  id: string;
  operation: string;
  weights: GPUSpatialWeights;
  values: GPUFloat32OrUint32Values;
  secondValues?: GPUFloat32OrUint32Values;
  mask?: GraphDataView<'uint32'>;
  parameters: GraphDataView<'uint32'>;
  maximumPermutations: number;
  /** Island policy; defaults to `'exclude'`. */
  islands?: GPUIslandPolicy;
  /**
   * Whether `compactY` holds the mean-centered second values (default) or the raw ones (binary
   * join-count variables).
   */
  centerY?: boolean;
  /** Optional caller-owned one-row uint32 output: the number of islands among the included rows. */
  islandCount?: GraphDataView<'uint32'>;
};

/** Views produced by {@link getPermutationInputNodes}. @internal */
export type PermutationInputs<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /**
   * `rows + 1` entries: the position of each analysis row in the compacted order (ascending row
   * order), or `PERMUTATION_INVALID_POSITION`; entry `rows` holds the analysis count `n` as an
   * exact uint32.
   */
  rowPositions: GraphDataView<'uint32'>;
  /** Compacted `x` values (centered when `centerX`), `n` valid entries. */
  compactX: GraphDataView<'float32'>;
  /** Compacted `y` values, centered unless `centerY` is false (zero without `secondValues`). */
  compactY: GraphDataView<'float32'>;
  /** `PERMUTATION_TOTAL` slots. */
  totals: GraphDataView<'float32'>;
  /** One uint32: the number of islands among the included rows (the caller's view when given). */
  islandCount: GraphDataView<'uint32'>;
  /**
   * The weights the statistics must read, one per slot of `weights.neighbors`: the caller's
   * `weights.weights` itself, or, for row-standardized weights, a transient copy in which every
   * row is divided by the sum of its active weights (re-standardization over the analysis rows).
   */
  weights: GraphDataView<'float32'>;
};

/**
 * Validates the props shared by both permutation tests and returns the row count.
 *
 * @internal
 */
export function validatePermutationInputs(props: PermutationInputProps): number {
  const {id} = props;
  const rows = validateGPUSpatialWeights(id, props.weights);
  validateCapacity(id, 'weights row count', rows, 2, 2 ** 24 - 1);
  if (validateFloat32OrUint32Values(props.values, `${id} values`) !== rows) {
    throw new Error(`${id} values length must equal the weights row count`);
  }
  if (
    props.secondValues &&
    validateFloat32OrUint32Values(props.secondValues, `${id} secondValues`) !== rows
  ) {
    throw new Error(`${id} secondValues length must equal the weights row count`);
  }
  if (props.mask) {
    validatePackedUint32View(props.mask, `${id} mask`);
    if (props.mask.length !== rows) {
      throw new Error(`${id} mask length must equal the weights row count`);
    }
  }
  if (props.islandCount) {
    validatePackedUint32View(props.islandCount, `${id} output.islandCount`);
    validateCapacity(id, 'output.islandCount length', props.islandCount.length);
  }
  if (props.weights.status?.overflow) {
    validatePackedUint32View(props.weights.status.overflow, `${id} weights.status.overflow`);
  }
  validatePackedUint32View(props.parameters, `${id} parameters`);
  validateCapacity(
    id,
    'parameters length',
    props.parameters.length,
    GPU_PERMUTATION_PARAMETER_LENGTH
  );
  validateCapacity(id, 'maximumPermutations', props.maximumPermutations, 1, 2 ** 20);
  return rows;
}

/**
 * WGSL reading the per-frame permutation parameters. Requires a `parameters` binding of `u32`.
 *
 * - `readPermutationCount()` clamps the requested count to `[1, maximum]`; kept for contributors
 *   that clamp by design.
 * - `isPermutationCountValid()` is true when the requested count lies in `[1, maximum]`.
 * - `readValidPermutationCount()` is the requested count, or 0 when it is not valid. The
 *   permutation tests use it: they never clamp, they write NaN and report the problem.
 *
 * @internal
 */
export function getPermutationParameterWGSL(maximumPermutations: number): string {
  return /* wgsl */ `
const MAXIMUM_PERMUTATIONS: u32 = ${maximumPermutations}u;
fn readSeedKey() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}
fn readPermutationCount() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_PERMUTATIONS);
}
fn isPermutationCountValid() -> bool {
  let requested = parameters[parametersOffset + 2u];
  return requested >= 1u && requested <= MAXIMUM_PERMUTATIONS;
}
fn readValidPermutationCount() -> u32 {
  return select(0u, parameters[parametersOffset + 2u], isPermutationCountValid());
}
fn readSignificanceLevel() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}
`;
}

/** Float helpers (finiteness, quiet NaN, two-sided p) shared with the autocorrelation contributors. @internal */
export const PERMUTATION_FLOAT_WGSL = SPATIAL_AUTOCORRELATION_FLOAT_WGSL;

/**
 * Builds the shared front end.
 *
 * Definitions (one rule for both tests):
 * - A row is *included* when its mask is nonzero and its value(s) are finite.
 * - An *island* is an included row without an active slot to another included row (an active
 *   slot has a valid neighbor other than the row, a positive weight and an included neighbor; see
 *   `getGPUSpatialWeightsNeighborLoopWGSL`).
 * - The *analysis rows* are the included rows, minus the islands under `islands: 'exclude'`. They
 *   are compacted in ascending row order; `n` (exact uint32), the mean, the sums of squares and
 *   `S0` are over them. Active slots whose neighbor is not an analysis row are dropped; there is
 *   no cascade, so a row whose only neighbors were islands keeps its place in `n` with no active
 *   slot (it is not tested and contributes 0 to `S0`).
 * - When the weights are row standardized, each analysis row's active weights are re-standardized
 *   to sum to 1 (see the returned `weights`), so masked or missing neighbors drop out of the lag
 *   instead of shrinking it toward 0.
 * - If `weights.status.overflow` is nonzero the weights are truncated and no row is analyzed (all
 *   statistics are NaN).
 * - The mean uses a fixed-order f32 tree sum followed by a second pass on the deviations from that
 *   first mean, so the centered values sum to zero to f32 accuracy for data with a large offset.
 *
 * @internal
 */
export function getPermutationInputNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PermutationInputProps & {centerX: boolean}
): PermutationInputs<Parameters> {
  const {id, operation, weights, mask} = props;
  const valuesInput = createFloat32ValuesNodes(graph, `${id}-values`, props.values);
  const secondInput = props.secondValues
    ? createFloat32ValuesNodes(graph, `${id}-second-values`, props.secondValues)
    : undefined;
  const values = valuesInput.view;
  const secondValues = secondInput?.view;
  const keepIslands = (props.islands ?? 'exclude') === 'keep';
  const standardize = isGPUSpatialWeightsRowStandardized(weights);
  const weightsOverflow = weights.status?.overflow;
  const rows = values.length;
  const blockCount = Math.ceil(rows / BLOCK_ROWS);
  const columnCount = 8;
  const matrix = createTransientView(graph, `${id}-matrix`, 'float32', columnCount * rows);
  const flags = createTransientView(graph, `${id}-flags`, 'uint32', rows);
  const analysisFlags = createTransientView(graph, `${id}-analysis-flags`, 'uint32', rows);
  const scanned = createTransientView(graph, `${id}-scanned`, 'uint32', rows);
  const rowPositions = createTransientView(graph, `${id}-row-positions`, 'uint32', rows + 1);
  const compactX = createTransientView(graph, `${id}-compact-x`, 'float32', rows);
  const compactY = createTransientView(graph, `${id}-compact-y`, 'float32', rows);
  const islandCount =
    props.islandCount ?? createTransientView(graph, `${id}-island-count`, 'uint32', 1);
  const effectiveWeights = standardize
    ? createTransientView(graph, `${id}-scaled-weights`, 'float32', weights.neighbors.length)
    : weights.weights;
  const totalsA = createTransientView(graph, `${id}-totals-a`, 'float32', 2);
  const totalsB = createTransientView(graph, `${id}-totals-b`, 'float32', 2);
  const totalsC = createTransientView(graph, `${id}-totals-c`, 'float32', 4);
  const totals = createTransientView(graph, `${id}-totals`, 'float32', 9);
  const constantsWGSL = `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${weights.neighbors.length}u;
const INVALID_POSITION: u32 = ${PERMUTATION_INVALID_POSITION}u;
fn getMatrixIndex(column: u32, row: u32) -> u32 {
  return column * ROWS + row;
}
${PERMUTATION_FLOAT_WGSL}`;
  const weightBindings: WGSLKernelBinding[] = [
    {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
    {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
    {name: 'weights', view: weights.weights, type: 'f32', access: 'read'}
  ];
  const secondBinding: WGSLKernelBinding[] = secondValues
    ? [{name: 'secondValues', view: secondValues, type: 'f32', access: 'read'}]
    : [];
  const nodes: GPUCommandNode<Parameters>[] = [
    ...valuesInput.nodes,
    ...(secondInput?.nodes ?? []),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-flags`,
      operation,
      variant: 'flags',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...secondBinding,
        ...(mask
          ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: constantsWGSL,
      body: `let x = values[valuesOffset + index];
  let y = ${secondValues ? 'secondValues[secondValuesOffset + index]' : '0.0'};
  let included = ${mask ? 'mask[maskOffset + index] != 0u &&' : ''} isFiniteFloat(x) && isFiniteFloat(y);
  flags[flagsOffset + index] = select(0u, 1u, included);`
    }),
    createFillNode<Parameters>(graph, {
      id: `${id}-island-clear`,
      operation,
      view: islandCount,
      type: 'u32',
      value: '0u',
      componentCount: 1
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-analysis`,
      operation,
      variant: keepIslands ? 'analysis-keep' : 'analysis-exclude',
      bindings: [
        ...weightBindings,
        {name: 'flags', view: flags, type: 'u32', access: 'read'},
        {name: 'analysisFlags', view: analysisFlags, type: 'u32', access: 'read_write'},
        {name: 'islandCount', view: islandCount, type: 'atomic<u32>', access: 'read_write'},
        ...(weightsOverflow
          ? [
              {
                name: 'weightsOverflow',
                view: weightsOverflow,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : [])
      ],
      invocationCount: rows,
      declarations: constantsWGSL,
      body: `let included = flags[flagsOffset + index] != 0u;
  var hasNeighbor = false;
  if (included) {
    ${getGPUSpatialWeightsNeighborLoopWGSL('flags[flagsOffset + neighbor] != 0u', 'hasNeighbor = true; break;')}
  }
  if (included && !hasNeighbor) {
    atomicAdd(&islandCount[islandCountOffset], 1u);
  }
  let truncated = ${weightsOverflow ? 'weightsOverflow[weightsOverflowOffset] != 0u' : 'false'};
  let analyzed = included && !truncated && ${keepIslands ? 'true' : 'hasNeighbor'};
  analysisFlags[analysisFlagsOffset + index] = select(0u, 1u, analyzed);`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moment-inputs`,
      operation,
      variant: 'moment-inputs',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...secondBinding,
        {name: 'analysisFlags', view: analysisFlags, type: 'u32', access: 'read'},
        {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: constantsWGSL,
      body: `let analyzed = analysisFlags[analysisFlagsOffset + index] != 0u;
  matrix[matrixOffset + getMatrixIndex(0u, index)] =
    select(0.0, values[valuesOffset + index], analyzed);
  matrix[matrixOffset + getMatrixIndex(1u, index)] =
    select(0.0, ${secondValues ? 'secondValues[secondValuesOffset + index]' : '0.0'}, analyzed);`
    }),
    ...getColumnSumNodes<Parameters>(graph, {
      id: `${id}-sum-a`,
      operation,
      matrix,
      rows,
      blockCount,
      first: 0,
      count: 2,
      totals: totalsA
    }),
    ...new GPUScan({
      id: `${id}-scan`,
      input: analysisFlags,
      output: scanned,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-positions`,
      operation,
      variant: 'positions',
      bindings: [
        {name: 'analysisFlags', view: analysisFlags, type: 'u32', access: 'read'},
        {name: 'scanned', view: scanned, type: 'u32', access: 'read'},
        {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows + 1,
      declarations: constantsWGSL,
      body: `if (index < ROWS) {
    rowPositions[rowPositionsOffset + index] =
      select(INVALID_POSITION, scanned[scannedOffset + index], analysisFlags[analysisFlagsOffset + index] != 0u);
  } else {
    rowPositions[rowPositionsOffset + ROWS] =
      scanned[scannedOffset + ROWS - 1u] + analysisFlags[analysisFlagsOffset + ROWS - 1u];
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-deviations`,
      operation,
      variant: 'deviations',
      bindings: [
        {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
        {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
        {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
      ],
      // First-pass deviations from the tree-sum mean; their sum corrects that mean below.
      invocationCount: rows,
      declarations: constantsWGSL,
      body: `let included = rowPositions[rowPositionsOffset + index] != INVALID_POSITION;
  let count = f32(rowPositions[rowPositionsOffset + ROWS]);
  matrix[matrixOffset + getMatrixIndex(2u, index)] =
    select(0.0, matrix[matrixOffset + getMatrixIndex(0u, index)] - totalsA[totalsAOffset] / count, included);
  matrix[matrixOffset + getMatrixIndex(3u, index)] =
    select(0.0, matrix[matrixOffset + getMatrixIndex(1u, index)] - totalsA[totalsAOffset + 1u] / count, included);`
    }),
    ...getColumnSumNodes<Parameters>(graph, {
      id: `${id}-sum-b`,
      operation,
      matrix,
      rows,
      blockCount,
      first: 2,
      count: 2,
      totals: totalsB
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-compact`,
      operation,
      variant: 'compact',
      bindings: [
        {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
        {name: 'totalsB', view: totalsB, type: 'f32', access: 'read'},
        {name: 'compactX', view: compactX, type: 'f32', access: 'read_write'},
        {name: 'compactY', view: compactY, type: 'f32', access: 'read_write'},
        {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: constantsWGSL,
      body: `let position = rowPositions[rowPositionsOffset + index];
  let included = position != INVALID_POSITION;
  let count = f32(rowPositions[rowPositionsOffset + ROWS]);
  // Second-pass mean correction: the deviations from the first mean sum to count * (mean - mean0).
  let centeredX = matrix[matrixOffset + getMatrixIndex(2u, index)] - totalsB[totalsBOffset] / count;
  let centeredY = matrix[matrixOffset + getMatrixIndex(3u, index)] - totalsB[totalsBOffset + 1u] / count;
  if (included) {
    compactX[compactXOffset + position] = ${props.centerX ? 'centeredX' : 'matrix[matrixOffset + getMatrixIndex(0u, index)]'};
    compactY[compactYOffset + position] = ${props.centerY === false ? 'matrix[matrixOffset + getMatrixIndex(1u, index)]' : 'centeredY'};
  }
  matrix[matrixOffset + getMatrixIndex(4u, index)] = select(0.0, centeredX * centeredX, included);
  matrix[matrixOffset + getMatrixIndex(5u, index)] = select(0.0, centeredY * centeredY, included);`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-row-weights`,
      operation,
      variant: standardize ? 'row-weights-standardize' : 'row-weights',
      bindings: [
        ...weightBindings,
        {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
        {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'},
        ...(standardize
          ? [
              {
                name: 'scaledWeights',
                view: effectiveWeights,
                type: 'f32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: rows,
      declarations: constantsWGSL,
      body: `var rowWeight = 0.0;
  if (rowPositions[rowPositionsOffset + index] != INVALID_POSITION) {
    ${getGPUSpatialWeightsNeighborLoopWGSL('rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION', 'rowWeight += weight;')}
  }
  // Row-standardized weights are re-standardized over the active slots: a row with any active slot sums to 1.
  matrix[matrixOffset + getMatrixIndex(6u, index)] = ${standardize ? 'select(0.0, 1.0, rowWeight > 0.0)' : 'rowWeight'};
  matrix[matrixOffset + getMatrixIndex(7u, index)] = ${standardize ? 'select(0.0, 1.0, rowWeight > 0.0)' : 'rowWeight * rowWeight'};
  ${
    standardize
      ? `let scale = select(0.0, 1.0 / rowWeight, rowWeight > 0.0);
  let begin = min(offsets[offsetsOffset + index], CAPACITY);
  let end = min(offsets[offsetsOffset + index + 1u], CAPACITY);
  for (var slot = begin; slot < end; slot++) {
    scaledWeights[scaledWeightsOffset + slot] = weights[weightsOffset + slot] * scale;
  }`
      : ''
  }`
    }),
    ...getColumnSumNodes<Parameters>(graph, {
      id: `${id}-sum-c`,
      operation,
      matrix,
      rows,
      blockCount,
      first: 4,
      count: 4,
      totals: totalsC
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-totals`,
      operation,
      variant: 'totals',
      bindings: [
        {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
        {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
        {name: 'totalsB', view: totalsB, type: 'f32', access: 'read'},
        {name: 'totalsC', view: totalsC, type: 'f32', access: 'read'},
        {name: 'totals', view: totals, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `const ROWS: u32 = ${rows}u;`,
      // The count is an exact uint32 (the scan total) and is converted to f32 only here.
      body: `let count = f32(rowPositions[rowPositionsOffset + ROWS]);
  let meanX = totalsA[totalsAOffset] / count + totalsB[totalsBOffset] / count;
  let meanY = totalsA[totalsAOffset + 1u] / count + totalsB[totalsBOffset + 1u] / count;
  totals[totalsOffset] = count;
  totals[totalsOffset + 1u] = meanX * count;
  totals[totalsOffset + 2u] = meanY * count;
  totals[totalsOffset + 3u] = totalsC[totalsCOffset];
  totals[totalsOffset + 4u] = totalsC[totalsCOffset + 1u];
  totals[totalsOffset + 5u] = totalsC[totalsCOffset + 2u];
  totals[totalsOffset + 6u] = meanX;
  totals[totalsOffset + 7u] = meanY;
  totals[totalsOffset + 8u] = totalsC[totalsCOffset + 3u];`
    })
  ];
  return {nodes, rowPositions, compactX, compactY, totals, islandCount, weights: effectiveWeights};
}

/**
 * Sums `count` columns of a column-major `rows`-row matrix, starting at `first`, into `totals` with
 * two fixed-order workgroup tree levels (per column and block of rows, then per column).
 *
 * @internal
 */
export function getColumnSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    matrix: GraphDataView<'float32'>;
    rows: number;
    blockCount: number;
    first: number;
    count: number;
    totals: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, rows, blockCount, first, count} = props;
  const segmentCount = count * blockCount;
  const blockOffsets = createTransientView(
    graph,
    `${id}-block-offsets`,
    'uint32',
    segmentCount + 1
  );
  const columnOffsets = createTransientView(graph, `${id}-column-offsets`, 'uint32', count + 1);
  const partials = createTransientView(graph, `${id}-partials`, 'float32', segmentCount);
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-offsets`,
      operation,
      variant: 'sum-offsets',
      bindings: [
        {name: 'blockOffsets', view: blockOffsets, type: 'u32', access: 'read_write'},
        {name: 'columnOffsets', view: columnOffsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: segmentCount + 1,
      declarations: `const ROWS: u32 = ${rows}u;
const BLOCK_ROWS: u32 = ${BLOCK_ROWS}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const FIRST_COLUMN: u32 = ${first}u;
const COLUMN_COUNT: u32 = ${count}u;`,
      body: `let column = index / BLOCK_COUNT;
  let block = index % BLOCK_COUNT;
  blockOffsets[blockOffsetsOffset + index] = (FIRST_COLUMN + column) * ROWS + min(block * BLOCK_ROWS, ROWS);
  if (index <= COLUMN_COUNT) {
    columnOffsets[columnOffsetsOffset + index] = index * BLOCK_COUNT;
  }`
    }),
    createSegmentSumNode<Parameters>(graph, {
      id: `${id}-blocks`,
      operation,
      segmentCount,
      input: props.matrix,
      segmentOffsets: blockOffsets,
      output: partials,
      // The largest offset is `(first + count) * rows`; checked against the matrix at build time.
      offsetsWithinInput: (first + count) * rows <= props.matrix.length
    }),
    createSegmentSumNode<Parameters>(graph, {
      id: `${id}-columns`,
      operation,
      segmentCount: count,
      input: partials,
      segmentOffsets: columnOffsets,
      output: props.totals,
      // Column offsets are `column * blockCount <= count * blockCount`, the rows of `partials`.
      offsetsWithinInput: true
    })
  ];
}
