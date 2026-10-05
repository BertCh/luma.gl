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
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {createSegmentSumNode, getSortKeyBits} from '../../utils/sorted-segment-sums';
import {
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH
} from './spatial-autocorrelation-parameters';

/** Rows reduced by one workgroup in the first level of the deterministic global sums. */
const BLOCK_ROWS = 4096;

/** Key of a non-finite z-score in the false-discovery-rate sort; sorts after every finite one. */
const INVALID_P_VALUE_KEY = 0x7f800001;

/** Inputs shared by both spatial-autocorrelation contributors. @internal */
export type SpatialAutocorrelationInputProps = {
  id: string;
  positions: GraphDataView<'float32x2'>;
  values: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  gridSize: readonly [number, number];
  mask?: GraphDataView<'uint32'>;
  globalStatistics?: GraphDataView<'float32'>;
};

/** Grid and moment views produced by {@link getSpatialAutocorrelationInputNodes}. @internal */
export type SpatialAutocorrelationInputs<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /** Row indices grouped by cell, ascending row order within each cell. */
  sortedRows: GraphDataView<'uint32'>;
  /** `cellCount + 1` exclusive offsets of each cell in `sortedRows`. */
  cellOffsets: GraphDataView<'uint32'>;
  /**
   * `rows + 4` floats: centered values `x - mean` (quiet NaN for excluded rows) followed by the
   * moments `[n, mean, variance, sumOfSquares]`.
   */
  statistics: GraphDataView<'float32'>;
};

/**
 * Validates the props shared by both contributors and returns the row count.
 *
 * @internal
 */
export function validateSpatialAutocorrelationInputs(
  props: SpatialAutocorrelationInputProps
): number {
  const {id} = props;
  validatePackedView(props.positions, ['float32x2'], `${id} positions`);
  validatePackedView(props.values, ['float32'], `${id} values`);
  validatePackedView(props.parameters, ['float32'], `${id} parameters`);
  const rows = props.positions.length;
  if (rows < 1) {
    throw new Error(`${id} positions must hold at least one row`);
  }
  if (rows >= 2 ** 24) {
    // Counts and ranks are compared as f32 inside the kernels.
    throw new Error(`${id} positions must hold fewer than 2^24 rows`);
  }
  if (props.values.length !== rows) {
    throw new Error(`${id} values length must equal positions length`);
  }
  if (props.parameters.length < GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH) {
    throw new Error(
      `${id} parameters must hold ${GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH} float32 values`
    );
  }
  const [columns, gridRows] = props.gridSize;
  if (
    !Number.isInteger(columns) ||
    !Number.isInteger(gridRows) ||
    columns < 1 ||
    gridRows < 1 ||
    columns * gridRows >= 0xffffffff
  ) {
    throw new Error(`${id} gridSize must be two positive integers with columns * rows < 2^32 - 1`);
  }
  if (props.mask) {
    validatePackedUint32View(props.mask, `${id} mask`);
    if (props.mask.length !== rows) {
      throw new Error(`${id} mask length must equal positions length`);
    }
  }
  if (props.globalStatistics) {
    validatePackedView(props.globalStatistics, ['float32'], `${id} globalStatistics`);
    if (props.globalStatistics.length < GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH) {
      throw new Error(
        `${id} globalStatistics must hold ${GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH} float32 values`
      );
    }
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
fn isFiniteFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

// A runtime operand keeps the quiet-NaN bit pattern out of constant evaluation.
fn getQuietNaN(seed: u32) -> f32 {
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}

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
 * WGSL shared by every kernel that reads the parameters: the lattice derivation, validity tests,
 * cell lookup, quiet NaN and the two-sided normal p-value. Kernels that include it must bind
 * `parameters` as `array<f32>`.
 *
 * Each encoding derives the active lattice from the per-frame bounds and radius so every cell is at
 * least `radius` wide (with a 2^-10 relative margin against rounding). The 3x3 cell neighborhood
 * of a point then always contains its whole distance band, so the radius can change every frame
 * without rebuilding anything but the per-frame sort.
 *
 * @internal
 */
export function getSpatialAutocorrelationSharedWGSL(gridSize: readonly [number, number]): string {
  return /* wgsl */ `
const COLUMNS: u32 = ${gridSize[0]}u;
const ROWS: u32 = ${gridSize[1]}u;
const CELL_COUNT: u32 = ${gridSize[0] * gridSize[1]}u;
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
  radiusSquared: f32
}

fn readParameter(slot: u32) -> f32 {
  return parameters[parametersOffset + slot];
}

fn readLattice() -> Lattice {
  var lattice: Lattice;
  lattice.cellWidth = 1.0;
  lattice.cellHeight = 1.0;
  lattice.columns = 1u;
  lattice.rows = 1u;
  lattice.radiusSquared = 0.0;
  let minimumX = readParameter(0u);
  let minimumY = readParameter(1u);
  let maximumX = readParameter(2u);
  let maximumY = readParameter(3u);
  let radius = readParameter(4u);
  let width = maximumX - minimumX;
  let height = maximumY - minimumY;
  let radiusSquared = radius * radius;
  let cellRadius = radius * CELL_MARGIN;
  lattice.valid =
    isFiniteFloat(minimumX) && isFiniteFloat(minimumY) && isFiniteFloat(maximumX) &&
    isFiniteFloat(maximumY) && isFiniteFloat(radius) && isFiniteFloat(width) &&
    isFiniteFloat(height) && isFiniteFloat(radiusSquared) && isFiniteFloat(cellRadius) &&
    radius > 0.0 && width >= 0.0 && height >= 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  lattice.minimumX = minimumX;
  lattice.minimumY = minimumY;
  lattice.maximumX = maximumX;
  lattice.maximumY = maximumY;
  lattice.cellWidth = max(width / f32(COLUMNS), cellRadius);
  lattice.cellHeight = max(height / f32(ROWS), cellRadius);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
  lattice.radiusSquared = radiusSquared;
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

${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
`;
}

/**
 * WGSL statements that visit, in a fixed order, every included row within the distance band of
 * `(x, y)` (the focus row itself included) and run `action` with `neighbor` (row index) in scope.
 *
 * Cells are visited row-major over the 3x3 neighborhood. The cells of one lattice row are
 * contiguous in `cellOffsets`, so each lattice row is one slot range of `sortedRows`, within which
 * rows ascend by cell and then by row index. The order is therefore independent of GPU scheduling
 * and the per-row sums are bitwise reproducible. Requires bindings `positions`, `sortedRows`,
 * `cellOffsets`, and `lattice`, `x`, `y` locals.
 *
 * @internal
 */
export function getSpatialAutocorrelationNeighborLoopWGSL(action: string): string {
  return /* wgsl */ `
  let column = getCellColumn(lattice, x);
  let row = getCellRow(lattice, y);
  let firstColumn = max(column, 1u) - 1u;
  let lastColumn = min(column + 1u, lattice.columns - 1u);
  let firstRow = max(row, 1u) - 1u;
  let lastRow = min(row + 1u, lattice.rows - 1u);
  for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
    let rowBase = cellRow * lattice.columns;
    let begin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
    let end = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
    for (var slot = begin; slot < end; slot++) {
      let neighbor = sortedRows[sortedRowsOffset + slot];
      let deltaX = positions[positionsOffset + neighbor * 2u] - x;
      let deltaY = positions[positionsOffset + neighbor * 2u + 1u] - y;
      if (deltaX * deltaX + deltaY * deltaY <= lattice.radiusSquared) {
        ${action}
      }
    }
  }`;
}

/**
 * Builds the shared front end of both contributors: per-row validity and cell keys, the cell index
 * (cell counts, exclusive offsets and a stable sort of rows by cell), and the global moments.
 *
 * Moments come from fixed-order two-level workgroup tree sums over the rows (no float atomics):
 * first the mean, then the sum of squared centered values, so the variance never subtracts two
 * large numbers. Fixed moments in the parameters replace both.
 *
 * @internal
 */
export function getSpatialAutocorrelationInputNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SpatialAutocorrelationInputProps & {operation: string}
): SpatialAutocorrelationInputs<Parameters> {
  const {id, operation, positions, values, parameters, gridSize, mask} = props;
  const rows = positions.length;
  const cellCount = gridSize[0] * gridSize[1];
  const blockCount = Math.ceil(rows / BLOCK_ROWS);
  const sharedWGSL = getSpatialAutocorrelationSharedWGSL(gridSize);
  const nodes: GPUCommandNode<Parameters>[] = [];

  const cellKeys = createTransientView(graph, `${id}-cell-keys`, 'uint32', rows);
  const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rows);
  const valueContributions = createTransientView(graph, `${id}-value-terms`, 'float32', rows);
  const cellCounts = createTransientView(graph, `${id}-cell-counts`, 'uint32', cellCount);
  const cellOffsets = createTransientView(graph, `${id}-cell-offsets`, 'uint32', cellCount + 1);
  const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rows);
  const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rows);
  const blockOffsets = createTransientView(graph, `${id}-block-offsets`, 'uint32', blockCount + 1);
  const totalOffsets = createTransientView(graph, `${id}-total-offsets`, 'uint32', 2);
  const valuePartials = createTransientView(graph, `${id}-value-partials`, 'float32', blockCount);
  const valueTotal = createTransientView(graph, `${id}-value-total`, 'float32', 1);
  const squareContributions = createTransientView(graph, `${id}-square-terms`, 'float32', rows);
  const squarePartials = createTransientView(graph, `${id}-square-partials`, 'float32', blockCount);
  const squareTotal = createTransientView(graph, `${id}-square-total`, 'float32', 1);
  const statistics = createTransientView(graph, `${id}-statistics`, 'float32', rows + 4);

  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-cell-keys`,
      operation,
      variant: 'cell-keys',
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'values', view: values, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        ...(mask
          ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'cellKeys', view: cellKeys, type: 'u32', access: 'read_write'},
        {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'},
        {name: 'valueTerms', view: valueContributions, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: sharedWGSL,
      body: `let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let value = values[valuesOffset + index];
  let included = ${mask ? 'mask[maskOffset + index] != 0u' : 'true'};
  let valid = included && isFiniteFloat(value) && isPointValid(lattice, x, y);
  var key = CELL_COUNT;
  if (valid) {
    key = getCellRow(lattice, y) * lattice.columns + getCellColumn(lattice, x);
  }
  cellKeys[cellKeysOffset + index] = key;
  rowIds[rowIdsOffset + index] = index;
  valueTerms[valueTermsOffset + index] = select(0.0, value, valid);`
    }),
    // Excluded rows carry the key CELL_COUNT, which the aggregation ignores and the sort puts last.
    ...new GPUGroupAggregation({
      id: `${id}-cell-counts`,
      keys: cellKeys,
      output: cellCounts
    }).getCommandNodes(graph),
    ...new GPUScan({
      id: `${id}-cell-scan`,
      input: cellCounts,
      output: cellOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-cell-total`,
      operation,
      variant: 'cell-total',
      bindings: [
        {name: 'counts', view: cellCounts, type: 'u32', access: 'read'},
        {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `const LAST_CELL: u32 = ${cellCount - 1}u;`,
      body: `cellOffsets[cellOffsetsOffset + LAST_CELL + 1u] =
    cellOffsets[cellOffsetsOffset + LAST_CELL] + counts[countsOffset + LAST_CELL];`
    }),
    ...new GPUSort({
      id: `${id}-cell-sort`,
      keys: cellKeys,
      values: rowIds,
      outputKeys: sortedKeys,
      outputValues: sortedRows,
      keyBits: getSortKeyBits(cellCount)
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-block-offsets`,
      operation,
      variant: 'block-offsets',
      bindings: [
        {name: 'blockOffsets', view: blockOffsets, type: 'u32', access: 'read_write'},
        {name: 'totalOffsets', view: totalOffsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: blockCount + 1,
      declarations: `const ROW_COUNT: u32 = ${rows}u;
const BLOCK_ROWS: u32 = ${BLOCK_ROWS}u;
const BLOCK_COUNT: u32 = ${blockCount}u;`,
      body: `blockOffsets[blockOffsetsOffset + index] = min(index * BLOCK_ROWS, ROW_COUNT);
  if (index == 0u) {
    totalOffsets[totalOffsetsOffset] = 0u;
    totalOffsets[totalOffsetsOffset + 1u] = BLOCK_COUNT;
  }`
    }),
    ...getTotalSumNodes<Parameters>(graph, {
      id: `${id}-value-sum`,
      operation,
      input: valueContributions,
      partials: valuePartials,
      output: valueTotal,
      blockOffsets,
      totalOffsets
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-mean`,
      operation,
      variant: 'moments-mean',
      bindings: [
        {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
        {name: 'valueTotal', view: valueTotal, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `${sharedWGSL}
const MOMENTS: u32 = ${rows}u;`,
      body: `var count = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  var mean = select(getQuietNaN(index), valueTotal[valueTotalOffset] / count, count >= 1.0);
  if (readParameter(7u) != 0.0) {
    count = readParameter(8u);
    mean = readParameter(9u);
  }
  statistics[statisticsOffset + MOMENTS] = count;
  statistics[statisticsOffset + MOMENTS + 1u] = mean;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-center`,
      operation,
      variant: 'center',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        {name: 'cellKeys', view: cellKeys, type: 'u32', access: 'read'},
        {name: 'statistics', view: statistics, type: 'f32', access: 'read_write'},
        {name: 'squareTerms', view: squareContributions, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: `const CELL_COUNT: u32 = ${cellCount}u;
const MOMENTS: u32 = ${rows}u;`,
      body: `let valid = cellKeys[cellKeysOffset + index] < CELL_COUNT;
  let centered = values[valuesOffset + index] - statistics[statisticsOffset + MOMENTS + 1u];
  // Excluded rows store quiet NaN, which every later kernel reads as "not a focus row".
  statistics[statisticsOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), centered, valid);
  squareTerms[squareTermsOffset + index] = select(0.0, centered * centered, valid);`
    }),
    ...getTotalSumNodes<Parameters>(graph, {
      id: `${id}-square-sum`,
      operation,
      input: squareContributions,
      partials: squarePartials,
      output: squareTotal,
      blockOffsets,
      totalOffsets
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments-variance`,
      operation,
      variant: 'moments-variance',
      bindings: [
        {name: 'squareTotal', view: squareTotal, type: 'f32', access: 'read'},
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
      invocationCount: 1,
      declarations: `${sharedWGSL}
const MOMENTS: u32 = ${rows}u;`,
      body: `let count = statistics[statisticsOffset + MOMENTS];
  let mean = statistics[statisticsOffset + MOMENTS + 1u];
  var sumOfSquares = squareTotal[squareTotalOffset];
  var variance = select(getQuietNaN(index), sumOfSquares / count, count >= 1.0);
  if (readParameter(7u) != 0.0) {
    variance = readParameter(10u);
    sumOfSquares = variance * count;
  }
  statistics[statisticsOffset + MOMENTS + 2u] = variance;
  statistics[statisticsOffset + MOMENTS + 3u] = sumOfSquares;
  ${
    props.globalStatistics
      ? `globalStatistics[globalStatisticsOffset] = count;
  globalStatistics[globalStatisticsOffset + 1u] = mean;
  globalStatistics[globalStatisticsOffset + 2u] = variance;
  globalStatistics[globalStatisticsOffset + 3u] = sqrt(variance);`
      : ''
  }`
    })
  );
  return {nodes, sortedRows, cellOffsets, statistics};
}

/** Two fixed-order tree-sum levels: one workgroup per block of rows, then one over the blocks. */
function getTotalSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    input: GraphDataView<'float32'>;
    partials: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
    blockOffsets: GraphDataView<'uint32'>;
    totalOffsets: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  return [
    createSegmentSumNode<Parameters>(graph, {
      id: `${props.id}-blocks`,
      operation: props.operation,
      segmentCount: props.partials.length,
      input: props.input,
      segmentOffsets: props.blockOffsets,
      output: props.partials
    }),
    createSegmentSumNode<Parameters>(graph, {
      id: `${props.id}-total`,
      operation: props.operation,
      segmentCount: 1,
      input: props.partials,
      segmentOffsets: props.totalOffsets,
      output: props.output
    })
  ];
}

/** Views produced by {@link getFalseDiscoveryRateNodes}. @internal */
export type FalseDiscoveryRateResult<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /** Per row: 1-based rank by ascending p-value, or 0 for a non-finite z-score. */
  ranks: GraphDataView<'uint32'>;
  /**
   * `[m, K_0, K_1, ...]`: the number of tested rows `m` and, per level, the largest rank `k`
   * with `p_(k) <= k * alpha / m` (0 when none). A row is significant at a level when
   * `0 < rank <= K`.
   */
  counters: GraphDataView<'uint32'>;
};

/**
 * Benjamini-Hochberg step-up false discovery rate over the two-sided p-values of `zScores`.
 *
 * Rows are sorted by descending `|z|` (ascending p) with a stable integer sort, then each sorted
 * slot tests its own BH inequality and raises the per-level threshold rank with an integer
 * `atomicMax`, so the result is exact and deterministic. Ties never straddle a threshold, so the
 * tie order does not matter. Rows with non-finite z are not tested and do not count toward `m`.
 *
 * @param levelExpressions WGSL f32 expressions of the significance levels, which may read the
 * parameters through `readParameter(slot)`.
 *
 * @internal
 */
export function getFalseDiscoveryRateNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    zScores: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    gridSize: readonly [number, number];
    levelExpressions: readonly string[];
  }
): FalseDiscoveryRateResult<Parameters> {
  const {id, operation, zScores, parameters, levelExpressions} = props;
  const rows = zScores.length;
  const levelCount = levelExpressions.length;
  const keys = createTransientView(graph, `${id}-keys`, 'uint32', rows);
  const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rows);
  const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rows);
  const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rows);
  const ranks = createTransientView(graph, `${id}-ranks`, 'uint32', rows);
  const counters = createTransientView(graph, `${id}-counters`, 'uint32', levelCount + 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-keys`,
      operation,
      variant: 'fdr-keys',
      bindings: [
        {name: 'zScores', view: zScores, type: 'f32', access: 'read'},
        {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
        {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows,
      // Non-negative finite floats order like their bits, so subtracting the bits from the +inf
      // pattern sorts larger |z| (smaller p) first.
      body: `let bits = bitcast<u32>(zScores[zScoresOffset + index]) & 0x7fffffffu;
  keys[keysOffset + index] = select(${INVALID_P_VALUE_KEY}u, 0x7f800000u - bits, bits < 0x7f800000u);
  rowIds[rowIdsOffset + index] = index;`
    }),
    ...new GPUSort({
      id: `${id}-sort`,
      keys,
      values: rowIds,
      outputKeys: sortedKeys,
      outputValues: sortedRows,
      keyBits: getSortKeyBits(INVALID_P_VALUE_KEY)
    }).getCommandNodes(graph),
    createFillNode<Parameters>(graph, {
      id: `${id}-clear`,
      operation,
      view: counters,
      type: 'u32',
      value: '0u'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-ranks`,
      operation,
      variant: 'fdr-ranks',
      bindings: [
        {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'ranks', view: ranks, type: 'u32', access: 'read_write'},
        {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: rows,
      body: `let tested = sortedKeys[sortedKeysOffset + index] != ${INVALID_P_VALUE_KEY}u;
  ranks[ranksOffset + sortedRows[sortedRowsOffset + index]] = select(0u, index + 1u, tested);
  if (tested) {
    atomicMax(&counters[countersOffset], index + 1u);
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-thresholds`,
      operation,
      variant: 'fdr-thresholds',
      bindings: [
        {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: getSpatialAutocorrelationSharedWGSL(props.gridSize),
      body: `let testedCount = atomicLoad(&counters[countersOffset]);
  let rank = index + 1u;
  if (rank <= testedCount) {
    let absoluteZ = bitcast<f32>(0x7f800000u - sortedKeys[sortedKeysOffset + index]);
    let pValue = getTwoSidedPValue(absoluteZ);
    let fraction = f32(rank) / f32(testedCount);
    ${levelExpressions
      .map(
        (expression, level) => `if (pValue <= fraction * (${expression})) {
      atomicMax(&counters[countersOffset + ${level + 1}u], rank);
    }`
      )
      .join('\n    ')}
  }`
    })
  ];
  return {nodes, ranks, counters};
}
