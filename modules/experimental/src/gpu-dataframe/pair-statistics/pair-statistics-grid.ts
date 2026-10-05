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

/**
 * Number of leading float32 parameter slots shared by every pair-statistics contributor:
 * `[minX, minY, maxX, maxY, maximumDistance]`. Contributor-specific slots follow.
 *
 * @internal
 */
export const PAIR_STATISTICS_SHARED_PARAMETER_LENGTH = 5;

/**
 * Number of float32 moments appended after the centered values in the `statistics` view of
 * {@link getPairStatisticsInputNodes}: `[n, mean, sumOfSquares, minimum, maximum,
 * maximumAbsoluteCentered]`.
 *
 * @internal
 */
export const PAIR_STATISTICS_MOMENT_LENGTH = 6;

/** Rows reduced by one workgroup in the first level of the deterministic global sums. */
const BLOCK_ROWS = 4096;

/** Inputs of {@link getPairStatisticsInputNodes}. @internal */
export type PairStatisticsInputProps = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  /** Per-frame parameters starting with the shared slots. */
  parameters: GraphDataView<'float32'>;
  /** Maximum `[columns, rows]` of the cell lattice. */
  gridSize: readonly [number, number];
  /** Optional values: rows with a non-finite value are excluded and moments are computed. */
  values?: GraphDataView<'float32'>;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
};

/** Views produced by {@link getPairStatisticsInputNodes}. @internal */
export type PairStatisticsInputs<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /** Included row indices grouped by cell (ascending row order within a cell), excluded rows last. */
  sortedRows: GraphDataView<'uint32'>;
  /** `cellCount + 1` exclusive offsets; `cellOffsets[cellCount]` is the included row count `n`. */
  cellOffsets: GraphDataView<'uint32'>;
  /** Per-row cell key, `cellCount` for excluded rows. */
  cellKeys: GraphDataView<'uint32'>;
  /**
   * Only with `values`: `rows + PAIR_STATISTICS_MOMENT_LENGTH` floats, the centered values
   * `x - mean` (quiet NaN for excluded rows) followed by `[n, mean, sumOfSquares, minimum,
   * maximum, maximumAbsoluteCentered]`.
   */
  statistics?: GraphDataView<'float32'>;
};

/**
 * Validates positions, parameters, grid size, optional values and mask, and returns the row count.
 *
 * @internal
 */
export function validatePairStatisticsInputs(
  id: string,
  props: {
    positions: GraphDataView<'float32x2'>;
    parameters: GraphDataView<'float32'>;
    parameterLength: number;
    gridSize: readonly [number, number];
    values?: GraphDataView<'float32'>;
    mask?: GraphDataView<'uint32'>;
  }
): number {
  validatePackedView(props.positions, ['float32x2'], `${id} positions`);
  validatePackedView(props.parameters, ['float32'], `${id} parameters`);
  const rows = props.positions.length;
  if (rows < 1) {
    throw new Error(`${id} positions must hold at least one row`);
  }
  if (rows >= 2 ** 24) {
    // Counts are compared and scaled as f32 inside the kernels.
    throw new Error(`${id} positions must hold fewer than 2^24 rows`);
  }
  if (props.parameters.length < props.parameterLength) {
    throw new Error(`${id} parameters must hold ${props.parameterLength} float32 values`);
  }
  const [columns, gridRows] = props.gridSize;
  if (
    !Number.isInteger(columns) ||
    !Number.isInteger(gridRows) ||
    columns < 1 ||
    gridRows < 1 ||
    columns * gridRows >= 2 ** 24
  ) {
    throw new Error(`${id} gridSize must be two positive integers with columns * rows < 2^24`);
  }
  if (props.values) {
    validatePackedView(props.values, ['float32'], `${id} values`);
    if (props.values.length !== rows) {
      throw new Error(`${id} values length must equal positions length`);
    }
  }
  if (props.mask) {
    validatePackedUint32View(props.mask, `${id} mask`);
    if (props.mask.length !== rows) {
      throw new Error(`${id} mask length must equal positions length`);
    }
  }
  return rows;
}

/**
 * Parameter-free WGSL helpers: finiteness test, runtime quiet NaN, and order-preserving `u32`
 * keys of f32 values for deterministic integer `atomicMin`/`atomicMax`.
 *
 * @internal
 */
export const PAIR_STATISTICS_FLOAT_WGSL = /* wgsl */ `
fn isFiniteFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

// A runtime operand keeps the quiet-NaN bit pattern out of constant evaluation.
fn getQuietNaN(seed: u32) -> f32 {
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}

fn getOrderedFloatKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn getOrderedFloatValue(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}
`;

/**
 * WGSL shared by every pair-statistics kernel that reads the parameters: the lattice derivation,
 * validity tests and cell lookup. Kernels that include it must bind `parameters` as `array<f32>`.
 *
 * Each encoding derives the active lattice from the per-frame bounds and maximum distance so every
 * cell is at least `maximumDistance` wide (with a 2^-10 relative margin against rounding). The 3x3
 * cell neighborhood of a point then holds every point within `maximumDistance`, so the distance
 * can change every frame without rebuilding the graph. When `maximumDistance` exceeds the extent,
 * the lattice collapses to one cell and the pair loop becomes a plain all-pairs loop.
 *
 * @internal
 */
export function getPairStatisticsSharedWGSL(gridSize: readonly [number, number]): string {
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
  maximumDistance: f32,
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
  lattice.maximumDistance = radius;
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

${PAIR_STATISTICS_FLOAT_WGSL}
`;
}

/**
 * Builds the shared front end of the pair-statistics contributors: per-row validity and cell keys, the
 * cell index (counts, exclusive offsets and a stable sort of rows by cell) and, with `values`, the
 * global value moments.
 *
 * A row is included when its mask is nonzero (if any), its value is finite (if any) and its
 * position is finite and inside the per-frame bounds. Moments come from fixed-order two-level
 * workgroup tree sums (no float atomics): first the mean, then the sum of squared centered values.
 * Minimum and maximum use integer atomics on order-preserving keys, so all of it is bitwise
 * reproducible.
 *
 * @internal
 */
export function getPairStatisticsInputNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PairStatisticsInputProps
): PairStatisticsInputs<Parameters> {
  const {id, operation, positions, values, parameters, gridSize, mask} = props;
  const rows = positions.length;
  const cellCount = gridSize[0] * gridSize[1];
  const sharedWGSL = getPairStatisticsSharedWGSL(gridSize);
  const nodes: GPUCommandNode<Parameters>[] = [];

  const cellKeys = createTransientView(graph, `${id}-cell-keys`, 'uint32', rows);
  const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', rows);
  const cellCounts = createTransientView(graph, `${id}-cell-counts`, 'uint32', cellCount);
  const cellOffsets = createTransientView(graph, `${id}-cell-offsets`, 'uint32', cellCount + 1);
  const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rows);
  const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rows);
  const valueTerms = values
    ? createTransientView(graph, `${id}-value-terms`, 'float32', rows)
    : undefined;

  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-cell-keys`,
      operation,
      variant: 'cell-keys',
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        ...(values
          ? [
              {
                name: 'values',
                view: values,
                type: 'f32' as const,
                access: 'read' as const
              }
            ]
          : []),
        ...(mask
          ? [
              {
                name: 'mask',
                view: mask,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'cellKeys', view: cellKeys, type: 'u32', access: 'read_write'},
        {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'},
        ...(valueTerms
          ? [
              {
                name: 'valueTerms',
                view: valueTerms,
                type: 'f32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: rows,
      declarations: sharedWGSL,
      body: `let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let included = ${mask ? 'mask[maskOffset + index] != 0u' : 'true'};
  ${values ? 'let value = values[valuesOffset + index];' : ''}
  let valid = included && ${values ? 'isFiniteFloat(value) &&' : ''} isPointValid(lattice, x, y);
  var key = CELL_COUNT;
  if (valid) {
    key = getCellRow(lattice, y) * lattice.columns + getCellColumn(lattice, x);
  }
  cellKeys[cellKeysOffset + index] = key;
  rowIds[rowIdsOffset + index] = index;
  ${values ? 'valueTerms[valueTermsOffset + index] = select(0.0, value, valid);' : ''}`
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
        {
          name: 'cellOffsets',
          view: cellOffsets,
          type: 'u32',
          access: 'read_write'
        }
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
    }).getCommandNodes(graph)
  );

  if (!values || !valueTerms) {
    return {nodes, sortedRows, cellOffsets, cellKeys};
  }

  const blockCount = Math.ceil(rows / BLOCK_ROWS);
  const blockOffsets = createTransientView(graph, `${id}-block-offsets`, 'uint32', blockCount + 1);
  const totalOffsets = createTransientView(graph, `${id}-total-offsets`, 'uint32', 2);
  const valuePartials = createTransientView(graph, `${id}-value-partials`, 'float32', blockCount);
  const valueTotal = createTransientView(graph, `${id}-value-total`, 'float32', 1);
  const squareTerms = createTransientView(graph, `${id}-square-terms`, 'float32', rows);
  const squarePartials = createTransientView(graph, `${id}-square-partials`, 'float32', blockCount);
  const squareTotal = createTransientView(graph, `${id}-square-total`, 'float32', 1);
  const extremes = createTransientView(graph, `${id}-extremes`, 'uint32', 2);
  const statistics = createTransientView(
    graph,
    `${id}-statistics`,
    'float32',
    rows + PAIR_STATISTICS_MOMENT_LENGTH
  );
  const momentsWGSL = `const MOMENTS: u32 = ${rows}u;
const CELL_COUNT: u32 = ${cellCount}u;`;

  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-block-offsets`,
      operation,
      variant: 'block-offsets',
      bindings: [
        {
          name: 'blockOffsets',
          view: blockOffsets,
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'totalOffsets',
          view: totalOffsets,
          type: 'u32',
          access: 'read_write'
        },
        {name: 'extremes', view: extremes, type: 'u32', access: 'read_write'}
      ],
      invocationCount: blockCount + 1,
      declarations: `const ROW_COUNT: u32 = ${rows}u;
const BLOCK_ROWS: u32 = ${BLOCK_ROWS}u;
const BLOCK_COUNT: u32 = ${blockCount}u;`,
      body: `blockOffsets[blockOffsetsOffset + index] = min(index * BLOCK_ROWS, ROW_COUNT);
  if (index == 0u) {
    totalOffsets[totalOffsetsOffset] = 0u;
    totalOffsets[totalOffsetsOffset + 1u] = BLOCK_COUNT;
    // Ordered keys: 0xffffffff is above every finite key, 0 below every finite key.
    extremes[extremesOffset] = 0xffffffffu;
    extremes[extremesOffset + 1u] = 0u;
  }`
    }),
    ...getTotalSumNodes<Parameters>(graph, {
      id: `${id}-value-sum`,
      operation,
      input: valueTerms,
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
        {
          name: 'statistics',
          view: statistics,
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: 1,
      declarations: `${momentsWGSL}
${PAIR_STATISTICS_FLOAT_WGSL}`,
      body: `let count = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  statistics[statisticsOffset + MOMENTS] = count;
  statistics[statisticsOffset + MOMENTS + 1u] =
    select(getQuietNaN(index), valueTotal[valueTotalOffset] / count, count >= 1.0);`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-center`,
      operation,
      variant: 'center',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        {name: 'cellKeys', view: cellKeys, type: 'u32', access: 'read'},
        {
          name: 'statistics',
          view: statistics,
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'squareTerms',
          view: squareTerms,
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'extremes',
          view: extremes,
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ],
      invocationCount: rows,
      declarations: `${momentsWGSL}
${PAIR_STATISTICS_FLOAT_WGSL}`,
      body: `let valid = cellKeys[cellKeysOffset + index] < CELL_COUNT;
  let value = values[valuesOffset + index];
  let centered = value - statistics[statisticsOffset + MOMENTS + 1u];
  // Excluded rows store quiet NaN, which every later kernel reads as "not included".
  statistics[statisticsOffset + index] = select(getQuietNaN(index), centered, valid);
  squareTerms[squareTermsOffset + index] = select(0.0, centered * centered, valid);
  if (valid) {
    let key = getOrderedFloatKey(value);
    atomicMin(&extremes[extremesOffset], key);
    atomicMax(&extremes[extremesOffset + 1u], key);
  }`
    }),
    ...getTotalSumNodes<Parameters>(graph, {
      id: `${id}-square-sum`,
      operation,
      input: squareTerms,
      partials: squarePartials,
      output: squareTotal,
      blockOffsets,
      totalOffsets
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments`,
      operation,
      variant: 'moments',
      bindings: [
        {name: 'squareTotal', view: squareTotal, type: 'f32', access: 'read'},
        {name: 'extremes', view: extremes, type: 'u32', access: 'read'},
        {
          name: 'statistics',
          view: statistics,
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: 1,
      declarations: `${momentsWGSL}
${PAIR_STATISTICS_FLOAT_WGSL}`,
      body: `let count = statistics[statisticsOffset + MOMENTS];
  let mean = statistics[statisticsOffset + MOMENTS + 1u];
  let hasRows = count >= 1.0;
  let minimum = select(getQuietNaN(index), getOrderedFloatValue(extremes[extremesOffset]), hasRows);
  let maximum = select(getQuietNaN(index), getOrderedFloatValue(extremes[extremesOffset + 1u]), hasRows);
  statistics[statisticsOffset + MOMENTS + 2u] = select(getQuietNaN(index), squareTotal[squareTotalOffset], hasRows);
  statistics[statisticsOffset + MOMENTS + 3u] = minimum;
  statistics[statisticsOffset + MOMENTS + 4u] = maximum;
  statistics[statisticsOffset + MOMENTS + 5u] = max(maximum - mean, mean - minimum);`
    })
  );
  return {nodes, sortedRows, cellOffsets, cellKeys, statistics};
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

/**
 * Deterministic total of a float32 row view into a one-element output: fixed-order two-level
 * workgroup tree sums (4096-row blocks, then one sum over the block partials), bitwise
 * reproducible. Creates its own graph transients under `id`.
 *
 * @internal
 */
export function getPairStatisticsTotalSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    input: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, input, output} = props;
  const rows = input.length;
  const blockCount = Math.max(1, Math.ceil(rows / BLOCK_ROWS));
  const blockOffsets = createTransientView(graph, `${id}-block-offsets`, 'uint32', blockCount + 1);
  const totalOffsets = createTransientView(graph, `${id}-total-offsets`, 'uint32', 2);
  const partials = createTransientView(graph, `${id}-partials`, 'float32', blockCount);
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-block-offsets`,
      operation,
      variant: 'block-offsets',
      bindings: [
        {
          name: 'blockOffsets',
          view: blockOffsets,
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'totalOffsets',
          view: totalOffsets,
          type: 'u32',
          access: 'read_write'
        }
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
      id: `${id}-sum`,
      operation,
      input,
      partials,
      output,
      blockOffsets,
      totalOffsets
    })
  ];
}

/**
 * Builds one node that clears `view` to zero. Re-exported so contributors in this directory need not
 * import the shared kernels for it.
 *
 * @internal
 */
export function createPairStatisticsClearNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  view: GraphDataView
): GPUCommandNode<Parameters> {
  return createFillNode<Parameters>(graph, {
    id,
    operation,
    view,
    type: 'u32',
    value: '0u'
  });
}
