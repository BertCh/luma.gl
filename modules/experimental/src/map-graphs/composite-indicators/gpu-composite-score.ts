// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {
  GPU_COMPOSITE_SCORE_COLUMN_STATISTICS_STRIDE,
  GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT,
  GPU_COMPOSITE_SCORE_PARAMETER_LENGTH,
  GPU_COMPOSITE_SCORE_POWER_ITERATIONS,
  GPU_COMPOSITE_SCORE_PRINCIPAL_COMPONENT_SUMMARY_LENGTH
} from './composite-score-parameters';

const OPERATION = 'GPUCompositeScore';
/** Rows summed in fixed order by one thread of a tile pass. */
const TILE_ROWS = 256;

/** Caller-owned outputs of {@link GPUCompositeScore}. */
export type GPUCompositeScoreOutput = {
  /** Composite score per row; NaN for excluded rows. */
  score: GraphDataView<'float32'>;
  /**
   * Optional direction-adjusted scaled indicators, row-major `row * indicatorCount + column`;
   * NaN for excluded rows. In `'principal-component'` mode they are direction-adjusted z-scores.
   */
  scaled?: GraphDataView<'float32'>;
  /**
   * Optional per-indicator `[min, max, mean, std]` (population standard deviation) over the
   * included rows, `indicatorCount * 4` values. NaN when no row is included.
   */
  columnStatistics?: GraphDataView<'float32'>;
  /**
   * Optional first-principal-component loadings, one per indicator, on direction-adjusted
   * z-scores. Needs `enablePrincipalComponent`.
   */
  loadings?: GraphDataView<'float32'>;
  /**
   * Optional `[eigenvalue, explainedVarianceRatio, residual]`, see
   * `GPU_COMPOSITE_SCORE_PRINCIPAL_COMPONENT_SUMMARY`. Needs `enablePrincipalComponent`.
   */
  principalComponentSummary?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUCompositeScore}.
 *
 * Per-frame (no rebuild or recompile): the contents of `indicators`, `mask` and `parameters`
 * (scaler, aggregation, weights, directions, epsilon). Compile-time: the row count,
 * `indicatorCount`, `enableRank`, `enablePrincipalComponent`, and which optional views are present.
 */
export type GPUCompositeScoreProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'composite-score'`. */
  id?: string;
  /** Packed float32 indicators, row-major `row * indicatorCount + column`. */
  indicators: GraphDataView<'float32'>;
  /** Number of indicators (columns), 1 to 16. Compile-time. */
  indicatorCount: number;
  /** Optional packed `uint32` row mask; zero excludes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Per-frame parameters: float32 view of at least `GPU_COMPOSITE_SCORE_PARAMETER_LENGTH` elements
   * written with `getGPUCompositeScoreParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Compiles the per-column sorts behind the `'rank'` scaler. Without it, a frame that selects the
   * rank scaler writes NaN scores. Default false.
   */
  enableRank?: boolean;
  /**
   * Compiles the correlation matrix and power iteration behind the `'principal-component'`
   * aggregation and the `loadings` output. Without it, that aggregation writes NaN scores.
   * Default false.
   */
  enablePrincipalComponent?: boolean;
  /** Caller-owned outputs. */
  output: GPUCompositeScoreOutput;
};

/**
 * Builds a composite indicator score per row from several indicator columns, as in CARTO's
 * composite-indicator procedures and ArcGIS index builders: per-column scaling (min-max, z-score
 * or percentile rank), per-indicator direction ("higher is worse"), and a weighted sum, weighted
 * geometric mean, or first-principal-component projection. Weights, directions, scaler and
 * aggregation live in the parameter view, so weight sliders recolor every row without a rebuild.
 *
 * A row is included when its mask is non-zero and every indicator is finite; excluded rows write
 * NaN and do not contribute to the column statistics.
 *
 * Scalers (direction `-1` flips them: `1 - s` for min-max and rank, `-s` for z-scores):
 * - `'min-max'`: `(x - min) / (max - min)`, or 0 for a constant column (as scikit-learn).
 * - `'z-score'`: `(x - mean) / std` with the population standard deviation, or 0 for a constant
 *   column.
 * - `'rank'`: percentile rank `r / (n - 1)` where `r` is the zero-based rank averaged over ties
 *   (0.5 when `n = 1`). Ranks come from one stable radix sort per column of order-preserving u32
 *   keys and a binary search for each tie run, so they are exact.
 *
 * Aggregations:
 * - `'weighted-sum'`: `sum(w * s) / sum(|w|)`; NaN when every weight is 0.
 * - `'weighted-geometric-mean'`: `exp(sum(w * ln(s + epsilon)) / sum(w))` over `max(w, 0)`; NaN when
 *   `s + epsilon <= 0` for a positively weighted indicator (use min-max or rank scaling) or when no
 *   weight is positive.
 * - `'principal-component'`: `sum(v * s)` where `s` are direction-adjusted z-scores and `v` is the
 *   first eigenvector of their correlation matrix (weights and the scaler are ignored). `v` comes
 *   from a fixed 64-step power iteration from the unit vector `1 / sqrt(d)`, signed so that its sum
 *   is non-negative (largest-magnitude loading positive when the sum is 0). Constant columns get
 *   zero correlations and zero loadings; the explained-variance ratio divides by the number of
 *   non-constant columns.
 *
 * Determinism: no float atomics. Column min and max use `atomicMin`/`atomicMax` on order-preserving
 * u32 keys (exact). Means and centered (co)variances are two-pass fixed-order sums: one thread per
 * (256-row tile, column) sums in row order, then one thread per column merges tiles in tile order.
 * The power iteration runs in a single thread. Results are bitwise reproducible on one adapter.
 */
export class GPUCompositeScore implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'composite-score';
  /** Validated properties. */
  readonly props: GPUCompositeScoreProps;
  /** Row count, `indicators.length / indicatorCount`. */
  readonly rowCount: number;

  constructor(props: GPUCompositeScoreProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const id = this.id;
    const {indicatorCount, output} = props;
    if (
      !Number.isInteger(indicatorCount) ||
      indicatorCount < 1 ||
      indicatorCount > GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT
    ) {
      throw new Error(
        `${id} indicatorCount must be an integer in [1, ${GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT}]`
      );
    }
    for (const [name, view] of [
      ['indicators', props.indicators],
      ['mask', props.mask],
      ['parameters', props.parameters]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.indicators, ['float32'], `${id} indicators`);
    if (props.indicators.length % indicatorCount !== 0) {
      throw new Error(`${id} indicators length must be a multiple of indicatorCount`);
    }
    this.rowCount = props.indicators.length / indicatorCount;
    if (this.rowCount < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_COMPOSITE_SCORE_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_COMPOSITE_SCORE_PARAMETER_LENGTH} float32 values`
      );
    }
    const lengths: [keyof GPUCompositeScoreOutput, number][] = [
      ['score', this.rowCount],
      ['scaled', this.rowCount * indicatorCount],
      ['columnStatistics', indicatorCount * GPU_COMPOSITE_SCORE_COLUMN_STATISTICS_STRIDE],
      ['loadings', indicatorCount],
      ['principalComponentSummary', GPU_COMPOSITE_SCORE_PRINCIPAL_COMPONENT_SUMMARY_LENGTH]
    ];
    for (const [name, length] of lengths) {
      const view = output[name];
      if (!view) {
        if (name === 'score') {
          throw new Error(`${id} output.score is required`);
        }
        continue;
      }
      validatePackedView(view, ['float32'], `${id} output.${name}`);
      if (view.length < length) {
        throw new Error(`${id} output.${name} must hold at least ${length} values`);
      }
    }
    if ((output.loadings || output.principalComponentSummary) && !props.enablePrincipalComponent) {
      throw new Error(`${id} principal-component outputs need enablePrincipalComponent`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      lengths.map(([name]) => output[name]),
      [props.indicators, props.mask, props.parameters]
    );
  }

  /**
   * Returns validate, mean, moment, optional principal-component, optional per-column rank, and
   * score nodes in order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rowCount} = this;
    const {output, indicatorCount, parameters, indicators} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      indicators,
      props.mask,
      parameters,
      output.score,
      output.scaled,
      output.columnStatistics,
      output.loadings,
      output.principalComponentSummary
    ]);
    const enableRank = Boolean(props.enableRank);
    const enablePrincipalComponent = Boolean(props.enablePrincipalComponent);
    const tileCount = Math.ceil(rowCount / TILE_ROWS);
    // Full correlation matrix for the principal component, otherwise only the variances.
    const momentCount = enablePrincipalComponent ? indicatorCount * indicatorCount : indicatorCount;
    const transient = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, Math.max(length, 1));
    const rowValid = transient('row-valid', 'uint32', rowCount);
    const minKeys = transient('min-keys', 'uint32', indicatorCount);
    const maxKeys = transient('max-keys', 'uint32', indicatorCount);
    const validCount = transient('valid-count', 'uint32', 1);
    const tileSums = transient('tile-sums', 'float32', tileCount * indicatorCount);
    const tileMoments = transient('tile-moments', 'float32', tileCount * momentCount);
    const moments = transient('moments', 'float32', momentCount);
    const statistics =
      output.columnStatistics ??
      transient(
        'column-statistics',
        'float32',
        indicatorCount * GPU_COMPOSITE_SCORE_COLUMN_STATISTICS_STRIDE
      );
    const loadings = enablePrincipalComponent
      ? (output.loadings ?? transient('loadings', 'float32', indicatorCount))
      : undefined;
    const ranks = enableRank ? transient('ranks', 'float32', rowCount * indicatorCount) : undefined;

    const read = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'f32'
    ): MapGraphKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'f32' | 'atomic<u32>'
    ): MapGraphKernelBinding => ({name, view, type, access: 'read_write'});
    const declarations = `const COLUMN_COUNT: u32 = ${indicatorCount}u;
const ROW_COUNT: u32 = ${rowCount}u;
const TILE_ROWS: u32 = ${TILE_ROWS}u;
const TILE_COUNT: u32 = ${tileCount}u;
const MOMENT_COUNT: u32 = ${momentCount}u;
const FULL_MOMENTS: bool = ${enablePrincipalComponent};
const MAXIMUM_COLUMNS: u32 = ${GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT}u;
${COMPOSITE_SCORE_WGSL}`;
    const nodes: GPUCommandNode<Parameters>[] = [];

    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-init`,
        operation: OPERATION,
        variant: 'init',
        bindings: [
          write('minKeys', minKeys, 'u32'),
          write('maxKeys', maxKeys, 'u32'),
          write('validCount', validCount, 'u32')
        ],
        invocationCount: indicatorCount,
        body: `minKeys[minKeysOffset + index] = 0xffffffffu;
  maxKeys[maxKeysOffset + index] = 0u;
  if (index == 0u) {
    validCount[validCountOffset] = 0u;
  }`
      })
    );

    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-validate`,
        operation: OPERATION,
        variant: 'validate',
        bindings: [
          read('indicators', indicators, 'f32'),
          ...(props.mask ? [read('rowMask', props.mask, 'u32')] : []),
          write('rowValid', rowValid, 'u32'),
          write('minKeys', minKeys, 'atomic<u32>'),
          write('maxKeys', maxKeys, 'atomic<u32>'),
          write('validCount', validCount, 'atomic<u32>')
        ],
        invocationCount: rowCount,
        declarations,
        body: `var isValid = true;
  ${props.mask ? 'isValid = rowMask[rowMaskOffset + index] != 0u;' : ''}
  let base = indicatorsOffset + index * COLUMN_COUNT;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    if (!isFiniteBits(indicators[base + column])) {
      isValid = false;
    }
  }
  rowValid[rowValidOffset + index] = select(0u, 1u, isValid);
  if (!isValid) {
    return;
  }
  atomicAdd(&validCount[validCountOffset], 1u);
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    let key = getOrderedKey(indicators[base + column]);
    atomicMin(&minKeys[minKeysOffset + column], key);
    atomicMax(&maxKeys[maxKeysOffset + column], key);
  }`
      })
    );

    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-tile-sums`,
        operation: OPERATION,
        variant: 'tile-sums',
        bindings: [
          read('indicators', indicators, 'f32'),
          read('rowValid', rowValid, 'u32'),
          write('tileSums', tileSums, 'f32')
        ],
        invocationCount: tileCount * indicatorCount,
        declarations,
        body: `let tile = index / COLUMN_COUNT;
  let column = index % COLUMN_COUNT;
  let firstRow = tile * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var sum = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] != 0u) {
      sum = sum + indicators[indicatorsOffset + row * COLUMN_COUNT + column];
    }
  }
  tileSums[tileSumsOffset + index] = sum;`
      })
    );

    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-means`,
        operation: OPERATION,
        variant: 'means',
        bindings: [
          read('tileSums', tileSums, 'f32'),
          read('minKeys', minKeys, 'u32'),
          read('maxKeys', maxKeys, 'u32'),
          read('validCount', validCount, 'u32'),
          write('statistics', statistics, 'f32')
        ],
        invocationCount: indicatorCount,
        declarations,
        body: `let count = validCount[validCountOffset];
  var sum = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    sum = sum + tileSums[tileSumsOffset + tile * COLUMN_COUNT + index];
  }
  let nan = getNaN();
  let hasRows = count != 0u;
  let base = statisticsOffset + index * 4u;
  statistics[base] = select(nan, decodeOrderedKey(minKeys[minKeysOffset + index]), hasRows);
  statistics[base + 1u] = select(nan, decodeOrderedKey(maxKeys[maxKeysOffset + index]), hasRows);
  statistics[base + 2u] = select(nan, sum / f32(max(count, 1u)), hasRows);`
      })
    );

    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-tile-moments`,
        operation: OPERATION,
        variant: enablePrincipalComponent ? 'tile-covariance' : 'tile-variance',
        bindings: [
          read('indicators', indicators, 'f32'),
          read('rowValid', rowValid, 'u32'),
          read('statistics', statistics, 'f32'),
          write('tileMoments', tileMoments, 'f32')
        ],
        invocationCount: tileCount * momentCount,
        declarations,
        body: `let tile = index / MOMENT_COUNT;
  let moment = index % MOMENT_COUNT;
  var left = moment;
  var right = moment;
  if (FULL_MOMENTS) {
    left = moment / COLUMN_COUNT;
    right = moment % COLUMN_COUNT;
  }
  let leftMean = statistics[statisticsOffset + left * 4u + 2u];
  let rightMean = statistics[statisticsOffset + right * 4u + 2u];
  let firstRow = tile * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var sum = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] != 0u) {
      let base = indicatorsOffset + row * COLUMN_COUNT;
      sum = sum + (indicators[base + left] - leftMean) * (indicators[base + right] - rightMean);
    }
  }
  tileMoments[tileMomentsOffset + index] = sum;`
      })
    );

    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-moments`,
        operation: OPERATION,
        variant: 'moments',
        bindings: [
          read('tileMoments', tileMoments, 'f32'),
          read('validCount', validCount, 'u32'),
          write('moments', moments, 'f32'),
          write('statistics', statistics, 'f32')
        ],
        invocationCount: momentCount,
        declarations,
        body: `let count = validCount[validCountOffset];
  var sum = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    sum = sum + tileMoments[tileMomentsOffset + tile * MOMENT_COUNT + index];
  }
  let moment = select(getNaN(), sum / f32(max(count, 1u)), count != 0u);
  moments[momentsOffset + index] = moment;
  var column = index;
  var isDiagonal = true;
  if (FULL_MOMENTS) {
    column = index / COLUMN_COUNT;
    isDiagonal = column == index % COLUMN_COUNT;
  }
  if (isDiagonal) {
    statistics[statisticsOffset + column * 4u + 3u] = select(sqrt(max(moment, 0.0)), moment, isNanBits(moment));
  }`
      })
    );

    if (enablePrincipalComponent && loadings) {
      const summary =
        output.principalComponentSummary ??
        transient(
          'principal-component-summary',
          'float32',
          GPU_COMPOSITE_SCORE_PRINCIPAL_COMPONENT_SUMMARY_LENGTH
        );
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-principal-component`,
          operation: OPERATION,
          variant: 'principal-component',
          bindings: [
            read('moments', moments, 'f32'),
            read('params', parameters, 'f32'),
            write('loadings', loadings, 'f32'),
            write('summary', summary, 'f32')
          ],
          invocationCount: 1,
          declarations: `${declarations}
const POWER_ITERATIONS: u32 = ${GPU_COMPOSITE_SCORE_POWER_ITERATIONS}u;`,
          body: PRINCIPAL_COMPONENT_BODY
        })
      );
    }

    if (enableRank && ranks) {
      const sortKeys = transient('sort-keys', 'uint32', rowCount);
      const sortRows = transient('sort-rows', 'uint32', rowCount);
      const sortedKeys = transient('sorted-keys', 'uint32', rowCount);
      const sortedRows = transient('sorted-rows', 'uint32', rowCount);
      for (let column = 0; column < indicatorCount; column++) {
        nodes.push(
          createMapGraphKernelNode<Parameters>(graph, {
            id: `${id}-rank-keys-${column}`,
            operation: OPERATION,
            variant: 'rank-keys',
            bindings: [
              read('indicators', indicators, 'f32'),
              read('rowValid', rowValid, 'u32'),
              write('sortKeys', sortKeys, 'u32'),
              write('sortRows', sortRows, 'u32')
            ],
            invocationCount: rowCount,
            declarations: `${declarations}
const RANK_COLUMN: u32 = ${column}u;`,
            body: `let isValid = rowValid[rowValidOffset + index] != 0u;
  let value = indicators[indicatorsOffset + index * COLUMN_COUNT + RANK_COLUMN];
  // Finite keys are below 0xff800000, so excluded rows sort after every included row; -0 ranks as +0.
  sortKeys[sortKeysOffset + index] = select(0xffffffffu, getOrderedKey(select(value, 0.0, value == 0.0)), isValid);
  sortRows[sortRowsOffset + index] = index;`
          })
        );
        nodes.push(
          ...new GPUSort({
            id: `${id}-rank-sort-${column}`,
            keys: sortKeys,
            values: sortRows,
            outputKeys: sortedKeys,
            outputValues: sortedRows,
            keyBits: 32
          }).getCommandNodes(graph)
        );
        nodes.push(
          createMapGraphKernelNode<Parameters>(graph, {
            id: `${id}-rank-scatter-${column}`,
            operation: OPERATION,
            variant: 'rank-scatter',
            bindings: [
              read('sortedKeys', sortedKeys, 'u32'),
              read('sortedRows', sortedRows, 'u32'),
              read('validCount', validCount, 'u32'),
              write('ranks', ranks, 'f32')
            ],
            invocationCount: rowCount,
            declarations: `${declarations}
const RANK_COLUMN: u32 = ${column}u;`,
            body: RANK_SCATTER_BODY
          })
        );
      }
    }

    const hasScaled = Boolean(output.scaled);
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-score`,
        operation: OPERATION,
        variant: 'score',
        bindings: [
          read('indicators', indicators, 'f32'),
          read('rowValid', rowValid, 'u32'),
          read('statistics', statistics, 'f32'),
          read('params', parameters, 'f32'),
          ...(ranks ? [read('ranks', ranks, 'f32')] : []),
          ...(loadings ? [read('loadings', loadings, 'f32')] : []),
          ...(output.scaled ? [write('scaledOut', output.scaled, 'f32')] : []),
          write('score', output.score, 'f32')
        ],
        invocationCount: rowCount,
        declarations: `${declarations}
const HAS_RANK: bool = ${enableRank};
const HAS_PRINCIPAL_COMPONENT: bool = ${enablePrincipalComponent};`,
        body: getScoreBody({hasRanks: enableRank, hasLoadings: Boolean(loadings), hasScaled})
      })
    );
    return nodes;
  }
}

/** Order keys, NaN helpers and finite checks shared by the composite-score kernels. */
const COMPOSITE_SCORE_WGSL = /* wgsl */ `
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn isNanBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;
}

fn isFiniteBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn getOrderedKey(x: f32) -> u32 {
  let bits = bitcast<u32>(x);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn decodeOrderedKey(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}
`;

// Correlation of direction-adjusted z-scores, then a fixed number of power-iteration steps.
const PRINCIPAL_COMPONENT_BODY = /* wgsl */ `let directionBase = paramsOffset + 4u + MAXIMUM_COLUMNS;
  var deviations: array<f32, ${GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT}>;
  var nonConstant = 0.0;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    let variance = moments[momentsOffset + column * COLUMN_COUNT + column];
    let isUsable = variance > 0.0;
    deviations[column] = select(0.0, sqrt(max(variance, 0.0)), isUsable);
    nonConstant = nonConstant + select(0.0, 1.0, isUsable);
  }
  var correlation: array<f32, ${GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT * GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT}>;
  for (var row = 0u; row < COLUMN_COUNT; row++) {
    for (var column = 0u; column < COLUMN_COUNT; column++) {
      let scale = deviations[row] * deviations[column];
      let sign = select(-1.0, 1.0, params[directionBase + row] >= 0.0) *
        select(-1.0, 1.0, params[directionBase + column] >= 0.0);
      var value = 0.0;
      if (scale > 0.0) {
        value = sign * moments[momentsOffset + row * COLUMN_COUNT + column] / scale;
      }
      correlation[row * COLUMN_COUNT + column] = value;
    }
  }
  var vector: array<f32, ${GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT}>;
  let start = 1.0 / sqrt(f32(COLUMN_COUNT));
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    vector[column] = start;
  }
  var product: array<f32, ${GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT}>;
  for (var iteration = 0u; iteration < POWER_ITERATIONS; iteration++) {
    var norm = 0.0;
    for (var row = 0u; row < COLUMN_COUNT; row++) {
      var sum = 0.0;
      for (var column = 0u; column < COLUMN_COUNT; column++) {
        sum = sum + correlation[row * COLUMN_COUNT + column] * vector[column];
      }
      product[row] = sum;
      norm = norm + sum * sum;
    }
    norm = sqrt(norm);
    if (!(norm > 0.0)) {
      break;
    }
    for (var row = 0u; row < COLUMN_COUNT; row++) {
      vector[row] = product[row] / norm;
    }
  }
  var total = 0.0;
  var largest = 0u;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    total = total + vector[column];
    if (abs(vector[column]) > abs(vector[largest])) {
      largest = column;
    }
  }
  let orientation = select(-1.0, 1.0, total > 0.0 || (total == 0.0 && vector[largest] >= 0.0));
  var eigenvalue = 0.0;
  for (var row = 0u; row < COLUMN_COUNT; row++) {
    vector[row] = orientation * vector[row];
  }
  for (var row = 0u; row < COLUMN_COUNT; row++) {
    var sum = 0.0;
    for (var column = 0u; column < COLUMN_COUNT; column++) {
      sum = sum + correlation[row * COLUMN_COUNT + column] * vector[column];
    }
    product[row] = sum;
    eigenvalue = eigenvalue + vector[row] * sum;
  }
  var residual = 0.0;
  for (var row = 0u; row < COLUMN_COUNT; row++) {
    let difference = product[row] - eigenvalue * vector[row];
    residual = residual + difference * difference;
    loadings[loadingsOffset + row] = vector[row];
  }
  let nan = getNaN();
  let hasVariance = nonConstant > 0.0;
  summary[summaryOffset] = select(nan, eigenvalue, hasVariance);
  summary[summaryOffset + 1u] = select(nan, eigenvalue / max(nonConstant, 1.0), hasVariance);
  summary[summaryOffset + 2u] = select(nan, sqrt(residual), hasVariance);`;

// Rank of each sorted position: average of the first and last position of its tie run.
const RANK_SCATTER_BODY = /* wgsl */ `let count = validCount[validCountOffset];
  if (index >= count) {
    return;
  }
  let key = sortedKeys[sortedKeysOffset + index];
  var low = 0u;
  var high = index;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let first = low;
  low = index;
  high = count;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let last = low - 1u;
  let averageRank = 0.5 * (f32(first) + f32(last));
  let percentile = select(0.5, averageRank / f32(max(count, 2u) - 1u), count > 1u);
  let row = sortedRows[sortedRowsOffset + index];
  ranks[ranksOffset + row * COLUMN_COUNT + RANK_COLUMN] = percentile;`;

function getScoreBody(props: {
  hasRanks: boolean;
  hasLoadings: boolean;
  hasScaled: boolean;
}): string {
  const {hasRanks, hasLoadings, hasScaled} = props;
  return /* wgsl */ `let nan = getNaN();
  let isValid = rowValid[rowValidOffset + index] != 0u;
  let scaler = u32(params[paramsOffset]);
  let aggregation = u32(params[paramsOffset + 1u]);
  let epsilon = params[paramsOffset + 2u];
  let weightBase = paramsOffset + 4u;
  let directionBase = paramsOffset + 4u + MAXIMUM_COLUMNS;
  let base = indicatorsOffset + index * COLUMN_COUNT;
  let isPrincipal = aggregation == 2u;
  var isSupported = isValid && aggregation <= 2u && scaler <= 2u;
  if (scaler == 2u && !isPrincipal && !HAS_RANK) {
    isSupported = false;
  }
  if (isPrincipal && !HAS_PRINCIPAL_COMPONENT) {
    isSupported = false;
  }
  var weightedSum = 0.0;
  var weightTotal = 0.0;
  var logSum = 0.0;
  var positiveWeightTotal = 0.0;
  var projection = 0.0;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    let value = indicators[base + column];
    let statisticBase = statisticsOffset + column * 4u;
    let isReversed = params[directionBase + column] < 0.0;
    var scaled = 0.0;
    if (isPrincipal || scaler == 1u) {
      let deviation = statistics[statisticBase + 3u];
      scaled = select(0.0, (value - statistics[statisticBase + 2u]) / deviation, deviation > 0.0);
      scaled = select(scaled, -scaled, isReversed);
    } else {
      if (scaler == 0u) {
        let minimum = statistics[statisticBase];
        let range = statistics[statisticBase + 1u] - minimum;
        scaled = select(0.0, (value - minimum) / range, range > 0.0);
      } else {
        ${hasRanks ? 'scaled = ranks[ranksOffset + index * COLUMN_COUNT + column];' : ''}
      }
      scaled = select(scaled, 1.0 - scaled, isReversed);
    }
    ${hasScaled ? 'scaledOut[scaledOutOffset + index * COLUMN_COUNT + column] = select(nan, scaled, isSupported);' : ''}
    let weight = params[weightBase + column];
    weightedSum = weightedSum + weight * scaled;
    weightTotal = weightTotal + abs(weight);
    let positiveWeight = max(weight, 0.0);
    if (positiveWeight > 0.0) {
      let shifted = scaled + epsilon;
      logSum = logSum + positiveWeight * select(nan, log(shifted), shifted > 0.0);
      positiveWeightTotal = positiveWeightTotal + positiveWeight;
    }
    ${hasLoadings ? 'projection = projection + loadings[loadingsOffset + column] * scaled;' : ''}
  }
  var result = nan;
  if (isSupported) {
    if (aggregation == 0u) {
      result = select(nan, weightedSum / weightTotal, weightTotal > 0.0);
    } else if (aggregation == 1u) {
      result = select(nan, exp(logSum / positiveWeightTotal), positiveWeightTotal > 0.0);
    } else {
      result = projection;
    }
  }
  score[scoreOffset + index] = result;`;
}
