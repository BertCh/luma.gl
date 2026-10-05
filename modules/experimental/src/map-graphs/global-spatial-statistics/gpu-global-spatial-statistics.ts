// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGroupAggregation,
  GPUScan,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  createMapGraphFillNode,
  createMapGraphKernelNode,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import {createMapGraphSegmentSumNode, getMapGraphSortKeyBits} from '../map-graph-sorted-sums';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../neighbor-search/spatial-weights';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';
import {
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  type GPUGlobalSpatialStatistic
} from './global-spatial-statistics-layout';

const OPERATION = 'GPUGlobalSpatialStatistics';

/** Rows reduced by one workgroup in the first level of the deterministic column sums. */
const BLOCK_ROWS = 4096;

const STATISTICS: readonly GPUGlobalSpatialStatistic[] = [
  'moran',
  'geary',
  'getisOrdG',
  'bivariateMoran',
  'joinCount'
];

/**
 * Per-row partial columns of the internal column-major matrix, summed in three fixed-order
 * batches: A (inclusion and raw sums) before centering, B after the pair and column passes, C
 * (deviations from the batch-B means) last.
 */
const COLUMN = {
  included: 0,
  x: 1,
  y: 2,
  centeredSquare: 3,
  centeredCube: 4,
  centeredFourth: 5,
  centeredSquareY: 6,
  black: 7,
  rowWeight: 8,
  s1: 9,
  moranCross: 10,
  gearySum: 11,
  gCross: 12,
  bivariateCross: 13,
  binaryOutDegree: 14,
  binaryS1: 15,
  degree: 16,
  binaryDegree: 17,
  columnLag: 18,
  island: 19,
  degreeSquare: 20,
  binaryDegreeSquare: 21,
  degreeDeviation: 22,
  binaryDegreeDeviation: 23,
  columnLagDeviation: 24
} as const;
const COLUMN_COUNT = 25;
const BATCH_A = {first: 0, count: 3} as const;
const BATCH_B = {first: 3, count: 19} as const;
const BATCH_C = {first: 22, count: 3} as const;

/**
 * Properties for {@link GPUGlobalSpatialStatistics}.
 *
 * Per-frame (no rebuild or recompile): the contents of `weights`, `values`, `secondValues` and
 * `mask`. Compile-time: the row count, the slot capacity, `statistics` and which optional views
 * exist.
 */
export type GPUGlobalSpatialStatisticsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'global-spatial-statistics'`. */
  id?: string;
  /** Square spatial weights: neighbor IDs index the same rows as `offsets`. */
  weights: GPUSpatialWeights;
  /** Analysis values, one per row. Join counts treat nonzero values as black. */
  values: GraphDataView<'float32'>;
  /** Second variable `y` of bivariate Moran's I, one per row. Required for `'bivariateMoran'`. */
  secondValues?: GraphDataView<'float32'>;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Statistics to report. Compile-time; other result blocks hold quiet NaN. */
  statistics: readonly GPUGlobalSpatialStatistic[];
  /**
   * Caller-owned results, at least `GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length` float32 values,
   * laid out by `GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT`.
   */
  results: GraphDataView<'float32'>;
  /**
   * Optional caller-owned exact ordered-pair join counts `[BB2, BW2, WW2]` (twice the join counts
   * for symmetric weights) as uint32.
   */
  joinCounts?: GraphDataView<'uint32'>;
};

/**
 * Global spatial autocorrelation from a {@link GPUSpatialWeights} CSR and a value column: Moran's
 * I, Geary's C, Getis-Ord General G, bivariate Moran's I and join counts, each with its analytic
 * expectation, variance, z-score and two-sided normal p-value. Weights are used as given (apply
 * row standardization in `GPUNeighborSearch` if wanted).
 *
 * Definitions (PySAL esda), over the `n` included rows (mask nonzero and every provided value
 * finite; excluded rows are neither foci nor neighbors), with `z = x - mean(x)`,
 * `S0 = sum w_ij`, `S1 = 1/2 sum (w_ij + w_ji)^2`, `S2 = sum_i (w_i. + w_.i)^2`:
 * - Moran: `I = (n / S0) sum w_ij z_i z_j / sum z_i^2`, `E = -1/(n-1)`; `esda.Moran` variances
 *   `VI_norm` and `VI_rand` (with the sample kurtosis).
 * - Geary: `C = (n-1) sum w_ij (x_i - x_j)^2 / (2 S0 sum z_i^2)`, `E = 1`; `esda.Geary`
 *   `VC_norm` and `VC_rand`.
 * - General G: `G = sum_{i != j} w_ij x_i x_j / sum_{i != j} x_i x_j`, `E = S0 / (n(n-1))`; the
 *   randomization variance equals `esda.G.VG` but is evaluated in centered form,
 *   `Var(x'Wx) = mean^2 Var(d'z) + 2 mean Cov(d'z, z'Wz) + Var(z'Wz)` with `d_i = w_i. + w_.i`,
 *   which avoids the catastrophic cancellation of `E[G^2] - E[G]^2` in f32. Values should be
 *   non-negative, as in esda. Only randomization fields are filled.
 * - Bivariate Moran (`esda.Moran_BV`, which permutes `y`):
 *   `I_BV = (n / S0) sum_ij w_ij zx_i zy_j / sqrt(sum zx^2 sum zy^2)`, which equals esda's value for
 *   row-standardized weights (`S0 = n`). Under randomization of `y`, `E = 0` and the exact
 *   variance is `(n / S0)^2 sum_j (b_j - mean b)^2 / ((n-1) sum zx^2)` with the column lag
 *   `b_j = sum_i w_ij zx_i`. esda itself reports only permutation inference.
 * - Join counts: weights binarized (`w > 0`), black `x != 0`; `BB = 1/2 sum b_ij x_i x_j`,
 *   `BW = 1/2 sum b_ij (x_i - x_j)^2`, `WW` likewise, as exact integer ordered-pair counts halved.
 *   Cliff-Ord randomization (nonfree sampling) moments of BB and BW, evaluated with the same
 *   centered quadratic-form identities (BB is General G's numerator of the indicator, BW is
 *   Geary's numerator). Asymmetric weights are handled through `S1` and `S2`, and counts may be
 *   half-integers.
 *
 * Variances are NaN when `n < 4` (the randomization formulas divide by `n - 3`), `S0 = 0`, or the
 * values are constant; z and p are NaN where the variance is not positive.
 *
 * Algorithm: one gather pass per row over its CSR slots (binary search of `w_ji` in row `j`), a
 * deterministic transpose (stable sort of slots by neighbor) for column sums, and per-row partials
 * summed by fixed-order two-level workgroup trees. Join counts use u32 atomics. No float atomics,
 * so results are bitwise reproducible. Integer-valued sums (counts, degrees) are exact below 2^24.
 */
export class GPUGlobalSpatialStatistics implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'global-spatial-statistics';
  /** Validated properties. */
  readonly props: GPUGlobalSpatialStatisticsProps;

  constructor(props: GPUGlobalSpatialStatisticsProps) {
    const id = props.id ?? 'global-spatial-statistics';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeights(id, props.weights);
    if (rows >= 2 ** 31) {
      throw new Error(`${id} weights must hold fewer than 2^31 rows`);
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
    if (props.statistics.length === 0) {
      throw new Error(`${id} statistics must name at least one statistic`);
    }
    for (const statistic of props.statistics) {
      if (!STATISTICS.includes(statistic)) {
        throw new Error(`${id} unknown statistic ${statistic}`);
      }
    }
    if (props.statistics.includes('bivariateMoran') && !props.secondValues) {
      throw new Error(`${id} bivariateMoran requires secondValues`);
    }
    validatePackedView(props.results, ['float32'], `${id} results`);
    if (props.results.length < GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length) {
      throw new Error(
        `${id} results must hold ${GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length} float32 values`
      );
    }
    if (props.joinCounts) {
      validatePackedUint32View(props.joinCounts, `${id} joinCounts`);
      if (props.joinCounts.length < 3) {
        throw new Error(`${id} joinCounts must hold 3 uint32 values`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.results, props.joinCounts],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.values,
        props.secondValues,
        props.mask
      ]
    );
  }

  /** Returns the global-statistics nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, values, secondValues, mask} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      values,
      secondValues,
      mask,
      props.results,
      props.joinCounts
    ]);
    const rows = weights.offsets.length - 1;
    const capacity = weights.neighbors.length;
    const blockCount = Math.ceil(rows / BLOCK_ROWS);
    const constantsWGSL = `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${capacity}u;
${Object.entries(COLUMN)
  .map(([name, column]) => `const COLUMN_${toConstantCase(name)}: u32 = ${column}u;`)
  .join('\n')}
fn getMatrixIndex(column: u32, row: u32) -> u32 {
  return column * ROWS + row;
}
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}`;

    const matrix = createTransientView(graph, `${id}-matrix`, 'float32', COLUMN_COUNT * rows);
    const centered = createTransientView(graph, `${id}-centered`, 'float32', rows * 2);
    const totalsA = createTransientView(graph, `${id}-totals-a`, 'float32', BATCH_A.count);
    const totalsB = createTransientView(graph, `${id}-totals-b`, 'float32', BATCH_B.count);
    const totalsC = createTransientView(graph, `${id}-totals-c`, 'float32', BATCH_C.count);
    const slotRows = createTransientView(graph, `${id}-slot-rows`, 'uint32', capacity);
    const transposeKeys = createTransientView(graph, `${id}-transpose-keys`, 'uint32', capacity);
    const slotIds = createTransientView(graph, `${id}-slot-ids`, 'uint32', capacity);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', capacity);
    const sortedSlots = createTransientView(graph, `${id}-sorted-slots`, 'uint32', capacity);
    const columnCounts = createTransientView(graph, `${id}-column-counts`, 'uint32', rows);
    const columnOffsets = createTransientView(graph, `${id}-column-offsets`, 'uint32', rows + 1);
    const joinCounters = createTransientView(graph, `${id}-join-counters`, 'uint32', 3);

    const optionalInputs: MapGraphKernelBinding[] = [
      ...(secondValues
        ? [
            {
              name: 'secondValues',
              view: secondValues,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : []),
      ...(mask ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}] : [])
    ];
    const nodes: GPUCommandNode<Parameters>[] = [
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-include`,
        operation: OPERATION,
        variant: 'include',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          ...optionalInputs,
          {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: constantsWGSL,
        body: `let x = values[valuesOffset + index];
  let y = ${secondValues ? 'secondValues[secondValuesOffset + index]' : '0.0'};
  let included = ${mask ? 'mask[maskOffset + index] != 0u &&' : ''} isFiniteFloat(x) && isFiniteFloat(y);
  matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] = select(0.0, 1.0, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)] = select(0.0, x, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_Y, index)] = select(0.0, y, included);`
      }),
      ...getColumnSumNodes<Parameters>(graph, {
        id: `${id}-sum-a`,
        matrix,
        rows,
        blockCount,
        ...BATCH_A,
        totals: totalsA
      }),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-center`,
        operation: OPERATION,
        variant: 'center',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          ...(secondValues
            ? [
                {
                  name: 'secondValues',
                  view: secondValues,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
          {name: 'centered', view: centered, type: 'f32', access: 'read_write'},
          {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: constantsWGSL,
        body: `let included = matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0;
  let count = totalsA[totalsAOffset];
  let centeredX = values[valuesOffset + index] - totalsA[totalsAOffset + 1u] / count;
  let centeredY = ${secondValues ? 'secondValues[secondValuesOffset + index] - totalsA[totalsAOffset + 2u] / count' : '0.0'};
  let square = centeredX * centeredX;
  centered[centeredOffset + index * 2u] = select(getQuietNaN(index), centeredX, included);
  centered[centeredOffset + index * 2u + 1u] = select(getQuietNaN(index), centeredY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_SQUARE, index)] = select(0.0, square, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_CUBE, index)] = select(0.0, square * centeredX, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_FOURTH, index)] = select(0.0, square * square, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_SQUARE_Y, index)] = select(0.0, centeredY * centeredY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BLACK, index)] =
    select(0.0, 1.0, included && values[valuesOffset + index] != 0.0);`
      }),
      // Deterministic transpose: the source row of every slot, then a stable sort by neighbor.
      createMapGraphFillNode<Parameters>(graph, {
        id: `${id}-slot-rows-clear`,
        operation: OPERATION,
        view: slotRows,
        type: 'u32',
        value: `${rows}u`
      }),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-slot-rows`,
        operation: OPERATION,
        variant: 'slot-rows',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'centered', view: centered, type: 'f32', access: 'read'},
          {name: 'slotRows', view: slotRows, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: constantsWGSL,
        body: `if (isFiniteFloat(centered[centeredOffset + index * 2u])) {
    let begin = min(offsets[offsetsOffset + index], CAPACITY);
    let end = min(offsets[offsetsOffset + index + 1u], CAPACITY);
    for (var slot = begin; slot < end; slot++) {
      slotRows[slotRowsOffset + slot] = index;
    }
  }`
      }),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-transpose-keys`,
        operation: OPERATION,
        variant: 'transpose-keys',
        bindings: [
          {name: 'slotRows', view: slotRows, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'centered', view: centered, type: 'f32', access: 'read'},
          {name: 'transposeKeys', view: transposeKeys, type: 'u32', access: 'read_write'},
          {name: 'slotIds', view: slotIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: constantsWGSL,
        body: `let source = slotRows[slotRowsOffset + index];
  let neighbor = neighbors[neighborsOffset + index];
  var key = ROWS;
  if (source < ROWS && neighbor < ROWS && neighbor != source &&
      isFiniteFloat(centered[centeredOffset + neighbor * 2u])) {
    key = neighbor;
  }
  transposeKeys[transposeKeysOffset + index] = key;
  slotIds[slotIdsOffset + index] = index;`
      }),
      ...new GPUGroupAggregation({
        id: `${id}-column-counts`,
        keys: transposeKeys,
        output: columnCounts
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-column-scan`,
        input: columnCounts,
        output: columnOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...new GPUSort({
        id: `${id}-transpose-sort`,
        keys: transposeKeys,
        values: slotIds,
        outputKeys: sortedKeys,
        outputValues: sortedSlots,
        keyBits: getMapGraphSortKeyBits(rows)
      }).getCommandNodes(graph),
      createMapGraphFillNode<Parameters>(graph, {
        id: `${id}-join-clear`,
        operation: OPERATION,
        view: joinCounters,
        type: 'u32',
        value: '0u'
      }),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-pairs`,
        operation: OPERATION,
        variant: 'pairs',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'centered', view: centered, type: 'f32', access: 'read'},
          {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'},
          {name: 'joinCounters', view: joinCounters, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${constantsWGSL}
fn getRowEnd(row: u32) -> u32 {
  return min(offsets[offsetsOffset + row + 1u], CAPACITY);
}

// Weight w_row,column by binary search of the ascending neighbor IDs of row, or -1 when absent.
fn findWeight(row: u32, column: u32) -> f32 {
  var low = min(offsets[offsetsOffset + row], CAPACITY);
  let end = getRowEnd(row);
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (neighbors[neighborsOffset + middle] < column) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && neighbors[neighborsOffset + low] == column) {
    return weights[weightsOffset + low];
  }
  return -1.0;
}`,
        body: `let centeredX = centered[centeredOffset + index * 2u];
  var rowWeight = 0.0;
  var s1 = 0.0;
  var lag = 0.0;
  var gearySum = 0.0;
  var rawLag = 0.0;
  var lagY = 0.0;
  var outDegree = 0u;
  var binaryS1 = 0u;
  var blackBlack = 0u;
  var blackWhite = 0u;
  var whiteWhite = 0u;
  let x = values[valuesOffset + index];
  if (isFiniteFloat(centeredX)) {
    let black = x != 0.0;
    let begin = min(offsets[offsetsOffset + index], CAPACITY);
    let end = getRowEnd(index);
    for (var slot = begin; slot < end; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS || neighbor == index) {
        continue;
      }
      let neighborX = centered[centeredOffset + neighbor * 2u];
      if (!isFiniteFloat(neighborX)) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      let reverse = findWeight(neighbor, index);
      rowWeight += weight;
      s1 += select(weight * weight, 0.5 * (weight + reverse) * (weight + reverse), reverse >= 0.0);
      lag += weight * neighborX;
      let difference = centeredX - neighborX;
      gearySum += weight * difference * difference;
      rawLag += weight * values[valuesOffset + neighbor];
      lagY += weight * centered[centeredOffset + neighbor * 2u + 1u];
      if (weight > 0.0) {
        outDegree++;
        binaryS1 += select(1u, 2u, reverse > 0.0);
        let neighborBlack = values[valuesOffset + neighbor] != 0.0;
        if (black && neighborBlack) {
          blackBlack++;
        } else if (black != neighborBlack) {
          blackWhite++;
        } else {
          whiteWhite++;
        }
      }
    }
    atomicAdd(&joinCounters[joinCountersOffset], blackBlack);
    atomicAdd(&joinCounters[joinCountersOffset + 1u], blackWhite);
    atomicAdd(&joinCounters[joinCountersOffset + 2u], whiteWhite);
  }
  let included = isFiniteFloat(centeredX);
  matrix[matrixOffset + getMatrixIndex(COLUMN_ROW_WEIGHT, index)] = rowWeight;
  matrix[matrixOffset + getMatrixIndex(COLUMN_S1, index)] = s1;
  matrix[matrixOffset + getMatrixIndex(COLUMN_MORAN_CROSS, index)] = select(0.0, centeredX * lag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_GEARY_SUM, index)] = gearySum;
  matrix[matrixOffset + getMatrixIndex(COLUMN_G_CROSS, index)] = select(0.0, x * rawLag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BIVARIATE_CROSS, index)] = select(0.0, centeredX * lagY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_OUT_DEGREE, index)] = f32(outDegree);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_S1, index)] = f32(binaryS1);`
      }),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-columns`,
        operation: OPERATION,
        variant: 'columns',
        bindings: [
          {name: 'columnOffsets', view: columnOffsets, type: 'u32', access: 'read'},
          {name: 'columnCounts', view: columnCounts, type: 'u32', access: 'read'},
          {name: 'sortedSlots', view: sortedSlots, type: 'u32', access: 'read'},
          {name: 'slotRows', view: slotRows, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'centered', view: centered, type: 'f32', access: 'read'},
          {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: constantsWGSL,
        // Slots of one column are in ascending slot order (stable sort), so sums are fixed-order.
        body: `let included = isFiniteFloat(centered[centeredOffset + index * 2u]);
  let begin = columnOffsets[columnOffsetsOffset + index];
  let end = begin + columnCounts[columnCountsOffset + index];
  var columnWeight = 0.0;
  var inDegree = 0u;
  var columnLag = 0.0;
  for (var position = begin; position < end; position++) {
    let slot = sortedSlots[sortedSlotsOffset + position];
    let weight = weights[weightsOffset + slot];
    columnWeight += weight;
    inDegree += select(0u, 1u, weight > 0.0);
    columnLag += weight * centered[centeredOffset + slotRows[slotRowsOffset + slot] * 2u];
  }
  let degree = matrix[matrixOffset + getMatrixIndex(COLUMN_ROW_WEIGHT, index)] + columnWeight;
  let binaryDegree = matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_OUT_DEGREE, index)] + f32(inDegree);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE, index)] = select(0.0, degree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE, index)] = select(0.0, binaryDegree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_COLUMN_LAG, index)] = select(0.0, columnLag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_ISLAND, index)] = select(0.0, 1.0, included && binaryDegree == 0.0);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE_SQUARE, index)] = select(0.0, degree * degree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE_SQUARE, index)] =
    select(0.0, binaryDegree * binaryDegree, included);`
      }),
      ...getColumnSumNodes<Parameters>(graph, {
        id: `${id}-sum-b`,
        matrix,
        rows,
        blockCount,
        ...BATCH_B,
        totals: totalsB
      }),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-deviations`,
        operation: OPERATION,
        variant: 'deviations',
        bindings: [
          {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
          {name: 'totalsB', view: totalsB, type: 'f32', access: 'read'},
          {name: 'matrix', view: matrix, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${constantsWGSL}
const BATCH_B_FIRST: u32 = ${BATCH_B.first}u;
fn readTotalB(column: u32) -> f32 {
  return totalsB[totalsBOffset + column - BATCH_B_FIRST];
}
fn getSquaredDeviation(column: u32, row: u32, count: f32) -> f32 {
  let deviation = matrix[matrixOffset + getMatrixIndex(column, row)] - readTotalB(column) / count;
  return deviation * deviation;
}`,
        body: `let included = matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0;
  let count = totalsA[totalsAOffset];
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_DEGREE, index, count), included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_BINARY_DEGREE, index, count), included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_COLUMN_LAG_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_COLUMN_LAG, index, count), included);`
      }),
      ...getColumnSumNodes<Parameters>(graph, {
        id: `${id}-sum-c`,
        matrix,
        rows,
        blockCount,
        ...BATCH_C,
        totals: totalsC
      }),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: [
          {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
          {name: 'totalsB', view: totalsB, type: 'f32', access: 'read'},
          {name: 'totalsC', view: totalsC, type: 'f32', access: 'read'},
          {name: 'joinCounters', view: joinCounters, type: 'u32', access: 'read'},
          {name: 'results', view: props.results, type: 'f32', access: 'read_write'},
          ...(props.joinCounts
            ? [
                {
                  name: 'joinCounts',
                  view: props.joinCounts,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: 1,
        declarations: `${constantsWGSL}
${GLOBAL_STATISTICS_FINALIZE_WGSL}
const BATCH_B_FIRST: u32 = ${BATCH_B.first}u;
const BATCH_C_FIRST: u32 = ${BATCH_C.first}u;
fn readTotal(column: u32) -> f32 {
  if (column < BATCH_B_FIRST) {
    return totalsA[totalsAOffset + column];
  }
  if (column < BATCH_C_FIRST) {
    return totalsB[totalsBOffset + column - BATCH_B_FIRST];
  }
  return totalsC[totalsCOffset + column - BATCH_C_FIRST];
}
fn writeBlock(offset: u32, block: array<f32, 8>, enabled: bool) {
  for (var field = 0u; field < 8u; field++) {
    results[resultsOffset + offset + field] = select(getQuietNaN(field), block[field], enabled);
  }
}`,
        body: getFinalizeBody(props)
      })
    ];
    return nodes;
  }
}

/** Converts a camelCase column name to CAPITAL_CASE for WGSL constants. */
function toConstantCase(name: string): string {
  return name.replace(/([A-Z])/g, '_$1').toUpperCase();
}

/**
 * Sums `count` matrix columns starting at `first` into `totals` with two fixed-order workgroup
 * tree levels: one segment per (column, block of rows), then one per column over its blocks.
 */
function getColumnSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    matrix: GraphDataView<'float32'>;
    rows: number;
    blockCount: number;
    first: number;
    count: number;
    totals: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, rows, blockCount, first, count} = props;
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
      operation: OPERATION,
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
      operation: OPERATION,
      segmentCount,
      input: props.matrix,
      segmentOffsets: blockOffsets,
      output: partials
    }),
    createMapGraphSegmentSumNode<Parameters>(graph, {
      id: `${id}-columns`,
      operation: OPERATION,
      segmentCount: count,
      input: partials,
      segmentOffsets: columnOffsets,
      output: props.totals
    })
  ];
}

/** Moment formulas evaluated by the single-invocation finalize kernel. */
const GLOBAL_STATISTICS_FINALIZE_WGSL = /* wgsl */ `
fn getFinite(value: f32) -> f32 {
  return select(getQuietNaN(0u), value, isFiniteFloat(value));
}

// Division that never relies on the implementation-defined result of x / 0.
fn divide(numerator: f32, denominator: f32) -> f32 {
  return select(getQuietNaN(0u), getFinite(numerator / denominator), denominator != 0.0 && isFiniteFloat(denominator));
}

fn getZScore(statistic: f32, expected: f32, variance: f32) -> f32 {
  let z = (statistic - expected) / sqrt(variance);
  return select(getQuietNaN(0u), z, variance > 0.0 && isFiniteFloat(z));
}

fn getPValue(z: f32) -> f32 {
  return select(getQuietNaN(0u), getTwoSidedPValue(z), isFiniteFloat(z));
}

// esda.Moran: (expected, VI_norm, VI_rand).
fn getMoranMoments(n: f32, s0: f32, s1: f32, s2: f32, kurtosis: f32) -> vec3<f32> {
  let n2 = n * n;
  let s02 = s0 * s0;
  let expected = -1.0 / (n - 1.0);
  let varianceNormality = (n2 * s1 - n * s2 + 3.0 * s02) / ((n2 - 1.0) * s02) - expected * expected;
  let a = n * ((n2 - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s02);
  let b = kurtosis * ((n2 - n) * s1 - 2.0 * n * s2 + 6.0 * s02);
  let varianceRandomization =
    (a - b) / ((n - 1.0) * (n - 2.0) * (n - 3.0) * s02) - expected * expected;
  return vec3<f32>(expected, varianceNormality, varianceRandomization);
}

// esda.Geary: (VC_norm, VC_rand).
fn getGearyMoments(n: f32, s0: f32, s1: f32, s2: f32, kurtosis: f32) -> vec2<f32> {
  let n2 = n * n;
  let s02 = s0 * s0;
  let varianceNormality = ((2.0 * s1 + s2) * (n - 1.0) - 4.0 * s02) / (2.0 * (n + 1.0) * s02);
  let a = (n - 1.0) * s1 * (n2 - 3.0 * n + 3.0 - (n - 1.0) * kurtosis);
  let b = 0.25 * (n - 1.0) * s2 * (n2 + 3.0 * n - 6.0 - (n2 - n + 2.0) * kurtosis);
  let c = s02 * (n2 - 3.0 - (n - 1.0) * (n - 1.0) * kurtosis);
  let varianceRandomization = (a - b + c) / (n * (n - 2.0) * (n - 3.0) * s02);
  return vec2<f32>(varianceNormality, varianceRandomization);
}

// Randomization variance of x'Wx (zero diagonal) with x = mean + z, sum z = 0:
// mean^2 Var(d'z) + 2 mean Cov(d'z, z'Wz) + Var(z'Wz), d_i = w_i. + w_.i, Vd = sum (d - mean d)^2.
fn getQuadraticFormVariance(
  n: f32, s0: f32, s1: f32, s2: f32, degreeDeviation: f32, mean: f32,
  sumSquares: f32, sumCubes: f32, sumFourths: f32
) -> f32 {
  let kurtosis = n * sumFourths / (sumSquares * sumSquares);
  let moranVariance = getMoranMoments(n, s0, s1, s2, kurtosis).z;
  let scale = s0 * sumSquares / n;
  let linearVariance = sumSquares * degreeDeviation / (n - 1.0);
  let covariance = -sumCubes * degreeDeviation / ((n - 1.0) * (n - 2.0));
  return mean * mean * linearVariance + 2.0 * mean * covariance + moranVariance * scale * scale;
}
`;

/** Builds the finalize kernel body, writing only the requested blocks. */
function getFinalizeBody(props: GPUGlobalSpatialStatisticsProps): string {
  const enabled = (statistic: GPUGlobalSpatialStatistic) =>
    props.statistics.includes(statistic) ? 'true' : 'false';
  const layout = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT;
  return /* wgsl */ `let n = readTotal(COLUMN_INCLUDED);
  let mean = divide(readTotal(COLUMN_X), n);
  let sumSquares = readTotal(COLUMN_CENTERED_SQUARE);
  let sumCubes = readTotal(COLUMN_CENTERED_CUBE);
  let sumFourths = readTotal(COLUMN_CENTERED_FOURTH);
  let s0 = readTotal(COLUMN_ROW_WEIGHT);
  let s1 = readTotal(COLUMN_S1);
  let s2 = readTotal(COLUMN_DEGREE_SQUARE);
  let blackCount = readTotal(COLUMN_BLACK);
  let degreeDeviation = readTotal(COLUMN_DEGREE_DEVIATION);
  let kurtosis = divide(n * sumFourths, sumSquares * sumSquares);
  let valid = n >= 4.0 && s0 > 0.0 && sumSquares > 0.0;

  results[resultsOffset + ${layout.summary}u] = n;
  results[resultsOffset + ${layout.summary + 1}u] = s0;
  results[resultsOffset + ${layout.summary + 2}u] = s1;
  results[resultsOffset + ${layout.summary + 3}u] = s2;
  results[resultsOffset + ${layout.summary + 4}u] = getFinite(mean);
  results[resultsOffset + ${layout.summary + 5}u] = divide(sumSquares, n);
  results[resultsOffset + ${layout.summary + 6}u] = blackCount;
  results[resultsOffset + ${layout.summary + 7}u] = readTotal(COLUMN_ISLAND);

  // Moran's I.
  {
    let statistic = divide(n * readTotal(COLUMN_MORAN_CROSS), s0 * sumSquares);
    let moments = getMoranMoments(n, s0, s1, s2, kurtosis);
    let varianceNormality = select(getQuietNaN(1u), getFinite(moments.y), valid);
    let varianceRandomization = select(getQuietNaN(1u), getFinite(moments.z), valid);
    let zNormality = getZScore(statistic, moments.x, varianceNormality);
    let zRandomization = getZScore(statistic, moments.x, varianceRandomization);
    writeBlock(${layout.moran}u, array<f32, 8>(statistic, getFinite(moments.x), varianceNormality,
      zNormality, getPValue(zNormality), varianceRandomization, zRandomization,
      getPValue(zRandomization)), ${enabled('moran')});
  }
  // Geary's C.
  {
    let statistic = divide((n - 1.0) * readTotal(COLUMN_GEARY_SUM), 2.0 * s0 * sumSquares);
    let moments = getGearyMoments(n, s0, s1, s2, kurtosis);
    let varianceNormality = select(getQuietNaN(1u), getFinite(moments.x), valid);
    let varianceRandomization = select(getQuietNaN(1u), getFinite(moments.y), valid);
    let zNormality = getZScore(statistic, 1.0, varianceNormality);
    let zRandomization = getZScore(statistic, 1.0, varianceRandomization);
    writeBlock(${layout.geary}u, array<f32, 8>(statistic, 1.0, varianceNormality, zNormality,
      getPValue(zNormality), varianceRandomization, zRandomization, getPValue(zRandomization)),
      ${enabled('geary')});
  }
  // Getis-Ord General G.
  {
    let denominator = n * (n - 1.0) * mean * mean - sumSquares;
    let statistic = divide(readTotal(COLUMN_G_CROSS), denominator);
    let expected = divide(s0, n * (n - 1.0));
    let variance = select(getQuietNaN(1u), getFinite(getQuadraticFormVariance(n, s0, s1, s2,
      degreeDeviation, mean, sumSquares, sumCubes, sumFourths) / (denominator * denominator)), valid);
    let z = getZScore(statistic, expected, variance);
    writeBlock(${layout.getisOrdG}u, array<f32, 8>(statistic, expected, getQuietNaN(2u),
      getQuietNaN(2u), getQuietNaN(2u), variance, z, getPValue(z)), ${enabled('getisOrdG')});
  }
  // Bivariate Moran's I, randomization of y.
  {
    let sumSquaresY = readTotal(COLUMN_CENTERED_SQUARE_Y);
    let scale = divide(n, s0);
    let statistic = divide(scale * readTotal(COLUMN_BIVARIATE_CROSS), sqrt(sumSquares * sumSquaresY));
    let variance = select(getQuietNaN(1u), getFinite(scale * scale *
      readTotal(COLUMN_COLUMN_LAG_DEVIATION) / ((n - 1.0) * sumSquares)), valid);
    let z = getZScore(statistic, 0.0, variance);
    writeBlock(${layout.bivariateMoran}u, array<f32, 8>(statistic, 0.0, getQuietNaN(2u),
      getQuietNaN(2u), getQuietNaN(2u), variance, z, getPValue(z)), ${enabled('bivariateMoran')});
  }
  // Join counts over the binarized weights.
  {
    let enabled = ${enabled('joinCount')};
    let blackBlack2 = joinCounters[joinCountersOffset];
    let blackWhite2 = joinCounters[joinCountersOffset + 1u];
    let whiteWhite2 = joinCounters[joinCountersOffset + 2u];
    ${
      props.joinCounts
        ? `joinCounts[joinCountsOffset] = select(0u, blackBlack2, enabled);
    joinCounts[joinCountsOffset + 1u] = select(0u, blackWhite2, enabled);
    joinCounts[joinCountsOffset + 2u] = select(0u, whiteWhite2, enabled);`
        : ''
    }
    let b0 = readTotal(COLUMN_BINARY_OUT_DEGREE);
    let b1 = readTotal(COLUMN_BINARY_S1);
    let b2 = readTotal(COLUMN_BINARY_DEGREE_SQUARE);
    let p = divide(blackCount, n);
    let q = 1.0 - p;
    let blackSquares = n * p * q;
    let blackCubes = blackSquares * (q - p);
    let blackFourths = blackSquares * (p * p * p + q * q * q);
    let binaryValid = n >= 4.0 && b0 > 0.0;
    // A single colour makes every join count constant: zero variance.
    let oneColour = blackCount == 0.0 || blackCount == n;
    let expectedBlackBlack = getFinite(0.5 * b0 * blackCount * (blackCount - 1.0) / (n * (n - 1.0)));
    let varianceBlackBlack = select(getQuietNaN(3u), getFinite(0.25 * getQuadraticFormVariance(n,
      b0, b1, b2, readTotal(COLUMN_BINARY_DEGREE_DEVIATION), p, blackSquares, blackCubes,
      blackFourths)), binaryValid);
    let varianceBlackBlackOrZero = select(varianceBlackBlack, 0.0, binaryValid && oneColour);
    let expectedBlackWhite = getFinite(b0 * blackSquares / (n - 1.0));
    let blackKurtosis = n * blackFourths / (blackSquares * blackSquares);
    let blackWhiteScale = b0 * blackSquares / (n - 1.0);
    let varianceBlackWhite = select(getQuietNaN(3u), getFinite(getGearyMoments(n, b0, b1, b2,
      blackKurtosis).y * blackWhiteScale * blackWhiteScale), binaryValid && !oneColour);
    let varianceBlackWhiteOrZero = select(varianceBlackWhite, 0.0, binaryValid && oneColour);
    let blackBlack = 0.5 * f32(blackBlack2);
    let blackWhite = 0.5 * f32(blackWhite2);
    let zBlackBlack = getZScore(blackBlack, expectedBlackBlack, varianceBlackBlackOrZero);
    let zBlackWhite = getZScore(blackWhite, expectedBlackWhite, varianceBlackWhiteOrZero);
    let block = array<f32, 12>(blackBlack, blackWhite, 0.5 * f32(whiteWhite2), 0.5 * b0,
      expectedBlackBlack, varianceBlackBlackOrZero, zBlackBlack, getPValue(zBlackBlack),
      expectedBlackWhite, varianceBlackWhiteOrZero, zBlackWhite, getPValue(zBlackWhite));
    for (var field = 0u; field < 12u; field++) {
      results[resultsOffset + ${layout.joinCount}u + field] = select(getQuietNaN(field), block[field], enabled);
    }
  }`;
}
