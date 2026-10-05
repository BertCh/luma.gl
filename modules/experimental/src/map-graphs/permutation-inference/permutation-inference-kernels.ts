// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import {createMapGraphSegmentSumNode} from '../map-graph-sorted-sums';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../neighbor-search/spatial-weights';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';
import {GPU_PERMUTATION_PARAMETER_LENGTH} from './permutation-parameters';

/** Rows reduced by one workgroup in the first level of the deterministic column sums. */
const BLOCK_ROWS = 4096;

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
  meanY: 7
} as const;

/** Inputs shared by both permutation tests. @internal */
export type PermutationInputProps = {
  id: string;
  operation: string;
  weights: GPUSpatialWeights;
  values: GraphDataView<'float32'>;
  secondValues?: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  parameters: GraphDataView<'uint32'>;
  maximumPermutations: number;
};

/** Views produced by {@link getPermutationInputNodes}. @internal */
export type PermutationInputs<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /**
   * `rows + 1` entries: the position of each included row in the compacted order (ascending row
   * order), or `PERMUTATION_INVALID_POSITION`; entry `rows` holds the included count `m`.
   */
  rowPositions: GraphDataView<'uint32'>;
  /** Compacted `x` values (centered when `centerX`), `m` valid entries. */
  compactX: GraphDataView<'float32'>;
  /** Compacted centered `y` values (zero without `secondValues`). */
  compactY: GraphDataView<'float32'>;
  /** `PERMUTATION_TOTAL` slots. */
  totals: GraphDataView<'float32'>;
};

/**
 * Validates the props shared by both permutation tests and returns the row count.
 *
 * @internal
 */
export function validatePermutationInputs(props: PermutationInputProps): number {
  const {id} = props;
  const rows = validateGPUSpatialWeights(id, props.weights);
  if (rows < 2 || rows >= 2 ** 24) {
    throw new Error(`${id} weights must hold between 2 and 2^24 - 1 rows`);
  }
  validatePackedView(props.values, ['float32'], `${id} values`);
  if (props.values.length !== rows) {
    throw new Error(`${id} values length must equal the weights row count`);
  }
  if (props.secondValues) {
    validatePackedView(props.secondValues, ['float32'], `${id} secondValues`);
    if (props.secondValues.length !== rows) {
      throw new Error(`${id} secondValues length must equal the weights row count`);
    }
  }
  if (props.mask) {
    validatePackedUint32View(props.mask, `${id} mask`);
    if (props.mask.length !== rows) {
      throw new Error(`${id} mask length must equal the weights row count`);
    }
  }
  validatePackedUint32View(props.parameters, `${id} parameters`);
  if (props.parameters.length < GPU_PERMUTATION_PARAMETER_LENGTH) {
    throw new Error(`${id} parameters must hold ${GPU_PERMUTATION_PARAMETER_LENGTH} uint32 values`);
  }
  if (
    !Number.isInteger(props.maximumPermutations) ||
    props.maximumPermutations < 1 ||
    props.maximumPermutations > 2 ** 20
  ) {
    throw new Error(`${id} maximumPermutations must be an integer in [1, 2^20]`);
  }
  return rows;
}

/**
 * WGSL reading the per-frame permutation parameters, clamped to the compile-time maximum. Requires
 * a `parameters` binding of `u32`.
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
fn readSignificanceLevel() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}
`;
}

/** Float helpers (finiteness, quiet NaN, two-sided p) shared with the autocorrelation recipes. @internal */
export const PERMUTATION_FLOAT_WGSL = SPATIAL_AUTOCORRELATION_FLOAT_WGSL;

/**
 * Builds the shared front end: inclusion flags, compaction of the included rows (ascending row
 * order), centered or raw compacted values, and the fixed-order totals `PERMUTATION_TOTAL`.
 *
 * @internal
 */
export function getPermutationInputNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PermutationInputProps & {centerX: boolean}
): PermutationInputs<Parameters> {
  const {id, operation, weights, values, secondValues, mask} = props;
  const rows = values.length;
  const blockCount = Math.ceil(rows / BLOCK_ROWS);
  const columnCount = 6;
  const matrix = createTransientView(graph, `${id}-matrix`, 'float32', columnCount * rows);
  const flags = createTransientView(graph, `${id}-flags`, 'uint32', rows);
  const scanned = createTransientView(graph, `${id}-scanned`, 'uint32', rows);
  const rowPositions = createTransientView(graph, `${id}-row-positions`, 'uint32', rows + 1);
  const compactX = createTransientView(graph, `${id}-compact-x`, 'float32', rows);
  const compactY = createTransientView(graph, `${id}-compact-y`, 'float32', rows);
  const totalsA = createTransientView(graph, `${id}-totals-a`, 'float32', 3);
  const totalsB = createTransientView(graph, `${id}-totals-b`, 'float32', 3);
  const totals = createTransientView(graph, `${id}-totals`, 'float32', 8);
  const constantsWGSL = `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${weights.neighbors.length}u;
const INVALID_POSITION: u32 = ${PERMUTATION_INVALID_POSITION}u;
fn getMatrixIndex(column: u32, row: u32) -> u32 {
  return column * ROWS + row;
}
${PERMUTATION_FLOAT_WGSL}`;
  const secondBinding: MapGraphKernelBinding[] = secondValues
    ? [{name: 'secondValues', view: secondValues, type: 'f32', access: 'read'}]
    : [];
  const nodes: GPUCommandNode<Parameters>[] = [
    createMapGraphKernelNode<Parameters>(graph, {
      id: `${id}-flags`,
      operation,
      variant: 'flags',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...secondBinding,
        ...(mask
          ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
        {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: constantsWGSL,
      body: `let x = values[valuesOffset + index];
  let y = ${secondValues ? 'secondValues[secondValuesOffset + index]' : '0.0'};
  let included = ${mask ? 'mask[maskOffset + index] != 0u &&' : ''} isFiniteFloat(x) && isFiniteFloat(y);
  flags[flagsOffset + index] = select(0u, 1u, included);
  matrix[matrixOffset + getMatrixIndex(0u, index)] = select(0.0, 1.0, included);
  matrix[matrixOffset + getMatrixIndex(1u, index)] = select(0.0, x, included);
  matrix[matrixOffset + getMatrixIndex(2u, index)] = select(0.0, y, included);`
    }),
    ...getColumnSumNodes<Parameters>(graph, {
      id: `${id}-sum-a`,
      operation,
      matrix,
      rows,
      blockCount,
      first: 0,
      count: 3,
      totals: totalsA
    }),
    ...new GPUScan({
      id: `${id}-scan`,
      input: flags,
      output: scanned,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createMapGraphKernelNode<Parameters>(graph, {
      id: `${id}-positions`,
      operation,
      variant: 'positions',
      bindings: [
        {name: 'flags', view: flags, type: 'u32', access: 'read'},
        {name: 'scanned', view: scanned, type: 'u32', access: 'read'},
        {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows + 1,
      declarations: constantsWGSL,
      body: `if (index < ROWS) {
    rowPositions[rowPositionsOffset + index] =
      select(INVALID_POSITION, scanned[scannedOffset + index], flags[flagsOffset + index] != 0u);
  } else {
    rowPositions[rowPositionsOffset + ROWS] =
      scanned[scannedOffset + ROWS - 1u] + flags[flagsOffset + ROWS - 1u];
  }`
    }),
    createMapGraphKernelNode<Parameters>(graph, {
      id: `${id}-compact`,
      operation,
      variant: 'compact',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...secondBinding,
        {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
        {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
        {name: 'compactX', view: compactX, type: 'f32', access: 'read_write'},
        {name: 'compactY', view: compactY, type: 'f32', access: 'read_write'},
        {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: constantsWGSL,
      body: `let position = rowPositions[rowPositionsOffset + index];
  let included = position != INVALID_POSITION;
  let count = totalsA[totalsAOffset];
  let x = values[valuesOffset + index];
  let centeredX = x - totalsA[totalsAOffset + 1u] / count;
  let centeredY = ${secondValues ? 'secondValues[secondValuesOffset + index] - totalsA[totalsAOffset + 2u] / count' : '0.0'};
  if (included) {
    compactX[compactXOffset + position] = ${props.centerX ? 'centeredX' : 'x'};
    compactY[compactYOffset + position] = centeredY;
  }
  matrix[matrixOffset + getMatrixIndex(3u, index)] = select(0.0, centeredX * centeredX, included);
  matrix[matrixOffset + getMatrixIndex(4u, index)] = select(0.0, centeredY * centeredY, included);`
    }),
    createMapGraphKernelNode<Parameters>(graph, {
      id: `${id}-row-weights`,
      operation,
      variant: 'row-weights',
      bindings: [
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
        {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
        {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
        {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: constantsWGSL,
      body: `var rowWeight = 0.0;
  if (rowPositions[rowPositionsOffset + index] != INVALID_POSITION) {
    let begin = min(offsets[offsetsOffset + index], CAPACITY);
    let end = min(offsets[offsetsOffset + index + 1u], CAPACITY);
    for (var slot = begin; slot < end; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor < ROWS && neighbor != index &&
          rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION) {
        rowWeight += weights[weightsOffset + slot];
      }
    }
  }
  matrix[matrixOffset + getMatrixIndex(5u, index)] = rowWeight;`
    }),
    ...getColumnSumNodes<Parameters>(graph, {
      id: `${id}-sum-b`,
      operation,
      matrix,
      rows,
      blockCount,
      first: 3,
      count: 3,
      totals: totalsB
    }),
    createMapGraphKernelNode<Parameters>(graph, {
      id: `${id}-totals`,
      operation,
      variant: 'totals',
      bindings: [
        {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
        {name: 'totalsB', view: totalsB, type: 'f32', access: 'read'},
        {name: 'totals', view: totals, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      body: `let count = totalsA[totalsAOffset];
  totals[totalsOffset] = count;
  totals[totalsOffset + 1u] = totalsA[totalsAOffset + 1u];
  totals[totalsOffset + 2u] = totalsA[totalsAOffset + 2u];
  totals[totalsOffset + 3u] = totalsB[totalsBOffset];
  totals[totalsOffset + 4u] = totalsB[totalsBOffset + 1u];
  totals[totalsOffset + 5u] = totalsB[totalsBOffset + 2u];
  totals[totalsOffset + 6u] = totalsA[totalsAOffset + 1u] / count;
  totals[totalsOffset + 7u] = totalsA[totalsAOffset + 2u] / count;`
    })
  ];
  return {nodes, rowPositions, compactX, compactY, totals};
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
    createMapGraphKernelNode<Parameters>(graph, {
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
    createMapGraphSegmentSumNode<Parameters>(graph, {
      id: `${id}-blocks`,
      operation,
      segmentCount,
      input: props.matrix,
      segmentOffsets: blockOffsets,
      output: partials
    }),
    createMapGraphSegmentSumNode<Parameters>(graph, {
      id: `${id}-columns`,
      operation,
      segmentCount: count,
      input: partials,
      segmentOffsets: columnOffsets,
      output: props.totals
    })
  ];
}
