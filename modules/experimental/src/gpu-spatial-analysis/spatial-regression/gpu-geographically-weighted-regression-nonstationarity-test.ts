// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getGPUPermutationParameterValues} from '../permutation-inference/permutation-parameters';
import {getPermutationParameterWGSL} from '../permutation-inference/permutation-inference-kernels';
import {PERMUTATION_RANDOM_WGSL} from '../permutation-inference/permutation-random';
import {
  createGridIndexNodes,
  getGridLookupWGSL,
  getRingNeighborDistancesWGSL
} from './gwr-grid-index';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DEFAULT_GRID_DIMENSION,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_INDEXED_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH,
  getGPUGeographicallyWeightedRegressionParameterLength
} from './geographically-weighted-regression-parameters';
import {
  GPU_GWR_NONSTATIONARITY_SUMMARY,
  GPU_GWR_NONSTATIONARITY_TABLE_STRIDE
} from './geographically-weighted-regression-nonstationarity-parameters';

const OPERATION = 'GPUGeographicallyWeightedRegressionNonstationarityTest';
/** Rows per thread of the compaction count pass. */
const SCAN_TILE_ROWS = 256;
/** Largest row count; the included count travels as an exact float32 integer. */
const MAXIMUM_ROW_COUNT = 1048576;
/** Float32 budget of the per-tile partial sums, 64 MiB. */
const MAXIMUM_PARTIAL_FLOATS = 16777216;
const MAXIMUM_TILE_COUNT = 4096;
const MINIMUM_TILE_ROWS = 8;

/** Caller-owned outputs of {@link GPUGeographicallyWeightedRegressionNonstationarityTest}. */
export type GPUGeographicallyWeightedRegressionNonstationarityTestOutput = {
  /**
   * `(predictorCount + 1) * GPU_GWR_NONSTATIONARITY_TABLE_STRIDE` float32 values, one row per
   * coefficient (intercept first, then each predictor); see `GPU_GWR_NONSTATIONARITY_TABLE`.
   */
  table: GraphDataView<'float32'>;
  /** At least `GPU_GWR_NONSTATIONARITY_SUMMARY.length` float32 values. */
  summary: GraphDataView<'float32'>;
  /**
   * Optional standard deviation of the local estimates per run and coefficient, row-major
   * `run * (predictorCount + 1) + coefficient`: run 0 is the observed fit and runs `1..P` are the
   * permutations (NaN beyond `P` or when no local fit succeeded). Length
   * `(maximumPermutations + 1) * (predictorCount + 1)`; this is the reference distribution of the
   * test, for histograms.
   */
  standardDeviations?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUGeographicallyWeightedRegressionNonstationarityTest}.
 *
 * Per-frame (no recompile): the contents of every input buffer, including the permutation
 * `parameters` (seed and count). Compile-time: the row count, `predictorCount`,
 * `maximumPermutations`, `maximumBandwidthCount`, `maximumNeighborCount`, `tileRowCount`, whether
 * `mask` and `standardDeviations` are present.
 */
export type GPUGeographicallyWeightedRegressionNonstationarityTestProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'gwr-nonstationarity-test'`. */
  id?: string;
  /** The positions given to the `GPUGeographicallyWeightedRegression` under test. */
  positions: GraphDataView<'float32x2'>;
  /** The predictors given to the regression under test, row-major, without the intercept. */
  predictors: GraphDataView<'float32'>;
  /** Number of predictor columns, 1 to 7. Compile-time. */
  predictorCount: number;
  /** The response given to the regression under test. */
  response: GraphDataView<'float32'>;
  /** The mask given to the regression under test, when it had one. */
  mask?: GraphDataView<'uint32'>;
  /**
   * The per-frame `parameters` of the regression under test (kernel, bandwidth mode, ladder),
   * written with `getGPUGeographicallyWeightedRegressionParameterValues`.
   */
  bandwidthParameters: GraphDataView<'float32'>;
  /** The regression's `output.selectedBandwidth`: the bandwidth every permuted refit reuses. */
  selectedBandwidth: GraphDataView<'float32'>;
  /** The regression's `output.coefficients`; NaN rows are excluded from the observed spread. */
  coefficients: GraphDataView<'float32'>;
  /**
   * Per-frame uint32 permutation parameters of at least `GPU_PERMUTATION_PARAMETER_LENGTH`
   * elements, written with `getGPUPermutationParameterValues` (`seed` and `permutations`).
   */
  parameters: GraphDataView<'uint32'>;
  /** Upper bound of the per-frame permutation count, 1 to 2^20. Compile-time; sizes dispatches. */
  maximumPermutations: number;
  /** Ladder capacity of the regression under test. Compile-time. Default 32. */
  maximumBandwidthCount?: number;
  /** Largest adaptive `k` of the regression under test. Compile-time. Default 128. */
  maximumNeighborCount?: number;
  /**
   * Locations refitted by one invocation (one tile of one permutation). Defaults to the smallest
   * of at least 8 that keeps the partial sums within 64 MiB. Compile-time.
   */
  tileRowCount?: number;
  /**
   * Dimensions `[columns, rows]` of the internal `GPUGridIndex` over the included rows, or `false`
   * for none. Compile-time. Defaults to a square grid of about 8 rows per cell (at most 1024 per
   * axis) from 1024 rows, otherwise none. With it, bisquare refits visit only the cells covering
   * the bandwidth and an adaptive bandwidth finds its k-th neighbour by a ring walk; Gaussian
   * refits scan every included row.
   */
  indexGridSize?: readonly [number, number] | false;
  /** Caller-owned outputs. */
  output: GPUGeographicallyWeightedRegressionNonstationarityTestOutput;
};

/**
 * Monte Carlo test of the spatial non-stationarity of GWR coefficients (Brunsdon, Fotheringham and
 * Charlton 1999; the permutation test of GWR4 and mgwr `GWRResults.spatial_variability`).
 *
 * For every coefficient the observed statistic is the standard deviation of its local estimates
 * over the fitted locations (population form, mgwr `np.std(params, axis=0)`). Permutation `p`
 * (1 to `P`) reassigns the observations `(X, y)` to the locations by a keyed pseudorandom bijection
 * of the included rows, the Philox 4x32-10 keyed Feistel permutation of the permutation tests (the
 * same seed and `p` give the same relabelling as `GPUGlobalPermutationTest`, and the TypeScript
 * mirror in `permutation-random` reproduces it exactly). Geometry is unchanged, so the kernel
 * weights are the original ones; this is equivalent to permuting the coordinates, as mgwr does.
 * Each permutation refits every location at the bandwidth the regression selected and records the
 * standard deviation of the local estimates (intercept in original coordinates, as the regression
 * reports it). The pseudo p-value is `(g + 1) / (P + 1)`, `g` the permutations whose standard
 * deviation is at least the observed one: a small value means the coefficient surface varies more
 * than chance.
 *
 * Differences from mgwr: the bandwidth is fixed at the selected one for every permutation (GWR4's
 * practice; mgwr re-selects it per permutation, which multiplies the cost by the ladder length),
 * the p-value uses the add-one pseudo p-value of the other permutation contributors instead of
 * `g / P`, and permuted fits that fail (singular local systems) are left out of that
 * permutation's standard deviation and counted in `summary`. A location whose observed fit failed
 * is left out of the observed spread.
 *
 * Numerics and cost: per-location sums are fixed-order and Jacobi-style equilibrated Cholesky
 * solves as in the regression; the spread of each tile is accumulated about the first successful
 * observed estimate and merged in tile order, so results are bitwise reproducible for a seed and a
 * `tileRowCount` (other tilings agree to float32 rounding). No `P x n` storage is kept.
 *
 * Cost: the kernel weights do not change between permutations, only the data attached to each
 * neighbour does, so with a grid index (default from 1024 rows) a bisquare refit visits only the
 * rows in the cells covering its bandwidth, `O(P n k p^2)` for `k` neighbours (the grid path sums
 * in cell order, so results differ from a scan by float rounding), and an adaptive bandwidth finds
 * its k-th neighbour by a ring walk. Gaussian kernels have unbounded support and still scan every
 * included row per refit, `O(P n^2 p^2)`; keep `n` to a few tens of thousands for them.
 */
export class GPUGeographicallyWeightedRegressionNonstationarityTest
  implements GPUCommandNodeProducer
{
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeographicallyWeightedRegressionNonstationarityTestProps;
  /** Row count, `positions.length`. */
  readonly rowCount: number;
  /** Columns of the design including the intercept, `predictorCount + 1`. */
  readonly coefficientCount: number;
  /** Ladder capacity. */
  readonly maximumBandwidthCount: number;
  /** Largest adaptive `k`. */
  readonly maximumNeighborCount: number;
  /** Locations refitted by one invocation. */
  readonly tileRowCount: number;
  /** Number of location tiles. */
  readonly tileCount: number;
  /** Grid index dimensions `[columns, rows]`, or null when no index is built. */
  readonly indexGridSize: readonly [number, number] | null;

  constructor(props: GPUGeographicallyWeightedRegressionNonstationarityTestProps) {
    const id = props.id ?? 'gwr-nonstationarity-test';
    this.id = id;
    this.props = props;
    const {predictorCount, output, maximumPermutations} = props;
    if (
      !Number.isInteger(predictorCount) ||
      predictorCount < 1 ||
      predictorCount > GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT
    ) {
      throw new Error(
        `${id} predictorCount must be an integer in [1, ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT}]`
      );
    }
    if (
      !Number.isInteger(maximumPermutations) ||
      maximumPermutations < 1 ||
      maximumPermutations > 2 ** 20
    ) {
      throw new Error(`${id} maximumPermutations must be an integer in [1, 2^20]`);
    }
    this.coefficientCount = predictorCount + 1;
    this.maximumBandwidthCount =
      props.maximumBandwidthCount ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH;
    if (
      !Number.isInteger(this.maximumBandwidthCount) ||
      this.maximumBandwidthCount < 1 ||
      this.maximumBandwidthCount > GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH
    ) {
      throw new Error(
        `${id} maximumBandwidthCount must be an integer in [1, ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH}]`
      );
    }
    this.maximumNeighborCount =
      props.maximumNeighborCount ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT;
    if (
      !Number.isInteger(this.maximumNeighborCount) ||
      this.maximumNeighborCount < 1 ||
      this.maximumNeighborCount > GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT
    ) {
      throw new Error(
        `${id} maximumNeighborCount must be an integer in [1, ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT}]`
      );
    }
    for (const [name, view] of [
      ['positions', props.positions],
      ['predictors', props.predictors],
      ['response', props.response],
      ['mask', props.mask],
      ['bandwidthParameters', props.bandwidthParameters],
      ['selectedBandwidth', props.selectedBandwidth],
      ['coefficients', props.coefficients],
      ['parameters', props.parameters],
      ...Object.entries(output)
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    this.rowCount = props.positions.length;
    if (this.rowCount < 2 || this.rowCount > MAXIMUM_ROW_COUNT) {
      throw new Error(`${id} needs between 2 and ${MAXIMUM_ROW_COUNT} rows`);
    }
    if (props.indexGridSize === false) {
      this.indexGridSize = null;
    } else if (props.indexGridSize) {
      const [columns, gridRows] = props.indexGridSize;
      if (
        !Number.isInteger(columns) ||
        !Number.isInteger(gridRows) ||
        columns < 1 ||
        gridRows < 1 ||
        columns * gridRows >= 0xffffffff
      ) {
        throw new Error(
          `${id} indexGridSize must be two positive integers with columns * rows < 2^32 - 1`
        );
      }
      this.indexGridSize = [columns, gridRows];
    } else if (this.rowCount >= GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_INDEXED_ROW_COUNT) {
      const dimension = Math.min(
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DEFAULT_GRID_DIMENSION,
        Math.max(1, Math.ceil(Math.sqrt(this.rowCount / 8)))
      );
      this.indexGridSize = [dimension, dimension];
    } else {
      this.indexGridSize = null;
    }
    validatePackedView(props.predictors, ['float32'], `${id} predictors`);
    if (props.predictors.length !== this.rowCount * predictorCount) {
      throw new Error(`${id} predictors length must equal rows * predictorCount`);
    }
    validatePackedView(props.response, ['float32'], `${id} response`);
    if (props.response.length !== this.rowCount) {
      throw new Error(`${id} response length must equal the row count`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    validatePackedView(props.bandwidthParameters, ['float32'], `${id} bandwidthParameters`);
    const parameterLength = getGPUGeographicallyWeightedRegressionParameterLength(
      this.maximumBandwidthCount
    );
    if (props.bandwidthParameters.length < parameterLength) {
      throw new Error(`${id} bandwidthParameters must hold ${parameterLength} float32 values`);
    }
    validatePackedView(props.selectedBandwidth, ['float32'], `${id} selectedBandwidth`);
    if (props.selectedBandwidth.length < 2) {
      throw new Error(`${id} selectedBandwidth must hold 2 float32 values`);
    }
    validatePackedView(props.coefficients, ['float32'], `${id} coefficients`);
    if (props.coefficients.length < this.rowCount * this.coefficientCount) {
      throw new Error(`${id} coefficients must hold rows * (predictorCount + 1) values`);
    }
    validatePackedUint32View(props.parameters, `${id} parameters`);
    if (
      props.parameters.length < getGPUPermutationParameterValues({seed: 0, permutations: 1}).length
    ) {
      throw new Error(`${id} parameters must hold GPU_PERMUTATION_PARAMETER_LENGTH uint32 values`);
    }
    const tableLength = this.coefficientCount * GPU_GWR_NONSTATIONARITY_TABLE_STRIDE;
    validatePackedView(output.table, ['float32'], `${id} output.table`);
    if (output.table.length < tableLength) {
      throw new Error(`${id} output.table must hold ${tableLength} float32 values`);
    }
    validatePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_GWR_NONSTATIONARITY_SUMMARY.length) {
      throw new Error(
        `${id} output.summary must hold ${GPU_GWR_NONSTATIONARITY_SUMMARY.length} float32 values`
      );
    }
    if (output.standardDeviations) {
      validatePackedView(output.standardDeviations, ['float32'], `${id} output.standardDeviations`);
      const length = (maximumPermutations + 1) * this.coefficientCount;
      if (output.standardDeviations.length < length) {
        throw new Error(`${id} output.standardDeviations must hold ${length} float32 values`);
      }
    }
    const partialStride = 2 * this.coefficientCount + 1;
    const partialRunCount = maximumPermutations + 1;
    if (props.tileRowCount !== undefined) {
      if (!Number.isInteger(props.tileRowCount) || props.tileRowCount < 1) {
        throw new Error(`${id} tileRowCount must be a positive integer`);
      }
      this.tileRowCount = props.tileRowCount;
    } else {
      const tileLimit = Math.max(
        1,
        Math.min(
          MAXIMUM_TILE_COUNT,
          Math.floor(MAXIMUM_PARTIAL_FLOATS / (partialRunCount * partialStride))
        )
      );
      this.tileRowCount = Math.max(MINIMUM_TILE_ROWS, Math.ceil(this.rowCount / tileLimit));
    }
    this.tileCount = Math.ceil(this.rowCount / this.tileRowCount);
    if (this.tileCount * partialRunCount * partialStride > MAXIMUM_PARTIAL_FLOATS * 4) {
      throw new Error(`${id} tileRowCount is too small for the row and permutation counts`);
    }
    if (this.tileCount * maximumPermutations >= 2 ** 31) {
      throw new Error(`${id} tileCount * maximumPermutations must stay below 2^31`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.table, output.summary, output.standardDeviations],
      [
        props.positions,
        props.predictors,
        props.response,
        props.mask,
        props.bandwidthParameters,
        props.selectedBandwidth,
        props.coefficients,
        props.parameters
      ]
    );
  }

  /**
   * Returns the nodes in order: row validation, compaction (tile counts, offsets, gather),
   * per-location bandwidths, the observed-spread anchor, observed tiles, permuted refit tiles,
   * merge, and the finish that writes `table` and `summary`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      id,
      props,
      rowCount,
      coefficientCount: P,
      maximumBandwidthCount,
      maximumNeighborCount,
      tileRowCount,
      tileCount,
      indexGridSize
    } = this;
    const {output, predictorCount, maximumPermutations, mask} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.predictors,
      props.response,
      mask,
      props.bandwidthParameters,
      props.selectedBandwidth,
      props.coefficients,
      props.parameters,
      output.table,
      output.summary,
      output.standardDeviations
    ]);
    const rowStride = predictorCount + 4;
    const partialStride = 2 * P + 1;
    const scanTileCount = Math.ceil(rowCount / SCAN_TILE_ROWS);
    const transient = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, Math.max(length, 1));
    const rowValid = transient('row-valid', 'uint32', rowCount);
    const tileCounts = transient('tile-counts', 'uint32', scanTileCount);
    const tileOffsets = transient('tile-offsets', 'uint32', scanTileCount + 1);
    // Compacted included rows: [x, y, bandwidth, predictors..., response] per row, then the count.
    const rows = transient('rows', 'float32', rowCount * rowStride + 1);
    // Grid over the compacted rows (non-finite positions are ignored), so the grid's row IDs are
    // compact row indices, in the same ascending order a scan visits.
    const compactPositions = indexGridSize
      ? createTransientView(graph, `${id}-compact-positions`, 'float32x2', Math.max(rowCount, 1))
      : null;
    const gridIndex =
      indexGridSize && compactPositions
        ? createGridIndexNodes(graph, {
            id,
            operation: OPERATION,
            positions: compactPositions,
            rowCount,
            gridSize: indexGridSize
          })
        : null;
    const shifts = transient('shifts', 'float32', P);
    const partial = transient(
      'partial',
      'float32',
      (maximumPermutations + 1) * tileCount * partialStride
    );
    const fitCounts = transient('fit-counts', 'float32', maximumPermutations + 1);
    const standardDeviations =
      output.standardDeviations ??
      transient('standard-deviations', 'float32', (maximumPermutations + 1) * P);

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
    const gridBindings: WGSLKernelBinding[] = gridIndex
      ? [
          read('cellOffsets', gridIndex.cellOffsets, 'u32'),
          read('sortedIds', gridIndex.sortedIds, 'u32')
        ]
      : [];
    const common = `const ROW_COUNT: u32 = ${rowCount}u;
const PREDICTOR_COUNT: u32 = ${predictorCount}u;
const P: u32 = ${P}u;
const ROW_STRIDE: u32 = ${rowStride}u;
const PARTIAL_STRIDE: u32 = ${partialStride}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;
const SCAN_TILE_ROWS: u32 = ${SCAN_TILE_ROWS}u;
const SCAN_TILE_COUNT: u32 = ${scanTileCount}u;
const LADDER: u32 = ${maximumBandwidthCount}u;
const MAXIMUM_NEIGHBORS: u32 = ${maximumNeighborCount}u;
const HEADER: u32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH}u;
const SENTINEL: f32 = 3.0e38;
const ADAPTIVE_FACTOR: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR};
${COMMON_WGSL}`;
    const permutationDeclarations = `${common}
${getPermutationParameterWGSL(maximumPermutations)}`;
    const gridDeclarations = indexGridSize ? getGridLookupWGSL(indexGridSize) : '';
    const rowCountSlot = rowCount * rowStride;

    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-validate`,
        operation: OPERATION,
        variant: 'validate',
        bindings: [
          read('positions', props.positions, 'f32'),
          read('predictors', props.predictors, 'f32'),
          read('response', props.response, 'f32'),
          ...(mask ? [read('rowMask', mask, 'u32')] : []),
          write('rowValid', rowValid, 'u32')
        ],
        invocationCount: rowCount,
        declarations: common,
        body: `var isValid = true;
  ${mask ? 'isValid = rowMask[rowMaskOffset + index] != 0u;' : ''}
  if (!isFiniteBits(positions[positionsOffset + 2u * index]) ||
    !isFiniteBits(positions[positionsOffset + 2u * index + 1u]) ||
    !isFiniteBits(response[responseOffset + index])) {
    isValid = false;
  }
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    if (!isFiniteBits(predictors[predictorsOffset + index * PREDICTOR_COUNT + column])) {
      isValid = false;
    }
  }
  rowValid[rowValidOffset + index] = select(0u, 1u, isValid);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tile-counts`,
        operation: OPERATION,
        variant: 'tile-counts',
        bindings: [read('rowValid', rowValid, 'u32'), write('tileCounts', tileCounts, 'u32')],
        invocationCount: scanTileCount,
        declarations: common,
        body: `let firstRow = index * SCAN_TILE_ROWS;
  let endRow = min(firstRow + SCAN_TILE_ROWS, ROW_COUNT);
  var count = 0u;
  for (var row = firstRow; row < endRow; row++) {
    count += select(0u, 1u, rowValid[rowValidOffset + row] != 0u);
  }
  tileCounts[tileCountsOffset + index] = count;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tile-offsets`,
        operation: OPERATION,
        variant: 'tile-offsets',
        bindings: [read('tileCounts', tileCounts, 'u32'), write('tileOffsets', tileOffsets, 'u32')],
        invocationCount: scanTileCount + 1,
        declarations: common,
        body: `var offset = 0u;
  for (var tile = 0u; tile < index; tile++) {
    offset += tileCounts[tileCountsOffset + tile];
  }
  tileOffsets[tileOffsetsOffset + index] = offset;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-compact`,
        operation: OPERATION,
        variant: 'compact',
        bindings: [
          read('positions', props.positions, 'f32'),
          read('predictors', props.predictors, 'f32'),
          read('response', props.response, 'f32'),
          read('rowValid', rowValid, 'u32'),
          read('tileOffsets', tileOffsets, 'u32'),
          write('rows', rows, 'f32')
        ],
        invocationCount: rowCount,
        declarations: common,
        body: `if (index == 0u) {
    rows[rowsOffset + ${rowCountSlot}u] = f32(tileOffsets[tileOffsetsOffset + SCAN_TILE_COUNT]);
  }
  if (rowValid[rowValidOffset + index] == 0u) {
    return;
  }
  let tile = index / SCAN_TILE_ROWS;
  var rank = tileOffsets[tileOffsetsOffset + tile];
  for (var row = tile * SCAN_TILE_ROWS; row < index; row++) {
    rank += select(0u, 1u, rowValid[rowValidOffset + row] != 0u);
  }
  let base = rowsOffset + rank * ROW_STRIDE;
  rows[base] = positions[positionsOffset + 2u * index];
  rows[base + 1u] = positions[positionsOffset + 2u * index + 1u];
  rows[base + 2u] = 0.0;
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    rows[base + 3u + column] = predictors[predictorsOffset + index * PREDICTOR_COUNT + column];
  }
  rows[base + 3u + PREDICTOR_COUNT] = response[responseOffset + index];`
      }),
      ...(gridIndex && compactPositions
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-compact-positions`,
              operation: OPERATION,
              variant: 'compact-positions',
              bindings: [
                read('rows', rows, 'f32'),
                write('compactPositions', compactPositions, 'f32')
              ],
              invocationCount: rowCount,
              declarations: common,
              body: `let count = u32(rows[rowsOffset + ${rowCountSlot}u]);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  let included = index < count;
  compactPositions[compactPositionsOffset + 2u * index] =
    select(invalid, rows[rowsOffset + index * ROW_STRIDE], included);
  compactPositions[compactPositionsOffset + 2u * index + 1u] =
    select(invalid, rows[rowsOffset + index * ROW_STRIDE + 1u], included);`
            }),
            ...gridIndex.nodes
          ]
        : []),
      // The selected bandwidth per included location; adaptive bandwidths come from the k-th
      // nearest included row, as in the regression.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-bandwidths`,
        operation: OPERATION,
        variant: 'bandwidths',
        bindings: [
          write('rows', rows, 'f32'),
          read('params', props.bandwidthParameters, 'f32'),
          read('selection', props.selectedBandwidth, 'f32'),
          ...gridBindings
        ],
        invocationCount: rowCount,
        declarations: `${common}
${gridDeclarations}
${getRowPositionWGSL()}
${indexGridSize ? getRingNeighborDistancesWGSL() : ''}`,
        body: `let count = u32(rows[rowsOffset + ${rowCountSlot}u]);
  if (index >= count) {
    return;
  }
  let candidate = min(u32(max(selection[selectionOffset], 0.0)), LADDER - 1u);
  let value = params[paramsOffset + HEADER + candidate];
  var bandwidth = -1.0;
  if (params[paramsOffset + 1u] < 0.5) {
    bandwidth = select(-1.0, value, value > 0.0);
  } else if (value >= 2.0 && value <= f32(MAXIMUM_NEIGHBORS)) {
    let neighborCount = u32(floor(value + 0.5));
    var list: array<f32, MAXIMUM_NEIGHBORS>;
    ${
      indexGridSize
        ? 'selectNeighborDistances(index, neighborCount, &list);'
        : `for (var slot = 0u; slot < MAXIMUM_NEIGHBORS; slot++) {
      list[slot] = SENTINEL;
    }
    let origin = getPosition(index);
    for (var row = 0u; row < count; row++) {
      let delta = getPosition(row) - origin;
      let squared = dot(delta, delta);
      if (squared < list[neighborCount - 1u]) {
        var slot = neighborCount - 1u;
        loop {
          if (slot == 0u || list[slot - 1u] <= squared) {
            break;
          }
          list[slot] = list[slot - 1u];
          slot = slot - 1u;
        }
        list[slot] = squared;
      }
    }`
    }
    let squared = list[neighborCount - 1u];
    if (squared < SENTINEL) {
      bandwidth = sqrt(squared) * ADAPTIVE_FACTOR;
    }
  }
  rows[rowsOffset + index * ROW_STRIDE + 2u] = bandwidth;`
      }),
      // Observed spread is accumulated about the first finite estimate, which is within a few
      // standard deviations of every other one, so the second moments do not cancel.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-anchor`,
        operation: OPERATION,
        variant: 'anchor',
        bindings: [read('coefficients', props.coefficients, 'f32'), write('shifts', shifts, 'f32')],
        invocationCount: 1,
        declarations: `${common}
${COEFFICIENT_ROW_WGSL}`,
        body: `for (var column = 0u; column < P; column++) {
    shifts[shiftsOffset + column] = 0.0;
  }
  for (var row = 0u; row < ROW_COUNT; row++) {
    if (isFiniteRow(row)) {
      for (var column = 0u; column < P; column++) {
        shifts[shiftsOffset + column] = coefficients[coefficientsOffset + row * P + column];
      }
      break;
    }
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-observed-tiles`,
        operation: OPERATION,
        variant: 'observed-tiles',
        bindings: [
          read('coefficients', props.coefficients, 'f32'),
          read('shifts', shifts, 'f32'),
          write('partial', partial, 'f32')
        ],
        invocationCount: tileCount,
        declarations: `${common}
${COEFFICIENT_ROW_WGSL}`,
        body: `var sums: array<f32, ${partialStride}>;
  let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  for (var row = firstRow; row < endRow; row++) {
    if (!isFiniteRow(row)) {
      continue;
    }
    sums[0] += 1.0;
    for (var column = 0u; column < P; column++) {
      let deviation = coefficients[coefficientsOffset + row * P + column] - shifts[shiftsOffset + column];
      sums[1u + column] += deviation;
      sums[1u + P + column] += deviation * deviation;
    }
  }
  let base = partialOffset + index * PARTIAL_STRIDE;
  for (var slot = 0u; slot < PARTIAL_STRIDE; slot++) {
    partial[base + slot] = sums[slot];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-permuted-tiles`,
        operation: OPERATION,
        variant: 'permuted-tiles',
        bindings: [
          read('rows', rows, 'f32'),
          read('params', props.bandwidthParameters, 'f32'),
          read('parameters', props.parameters, 'u32'),
          read('shifts', shifts, 'f32'),
          write('partial', partial, 'f32'),
          ...gridBindings
        ],
        invocationCount: tileCount * maximumPermutations,
        declarations: `${permutationDeclarations}
${gridDeclarations}
${PERMUTATION_RANDOM_WGSL}
${getCholeskyWGSL(P)}
${getPermutedFitWGSL(P, Boolean(indexGridSize))}`,
        body: `let permutation = index / TILE_COUNT + 1u;
  let tile = index % TILE_COUNT;
  var sums: array<f32, ${partialStride}>;
  if (permutation <= readPermutationCount()) {
    let count = u32(rows[rowsOffset + ${rowCountSlot}u]);
    let roundKeys = getFeistelRoundKeys(readSeedKey(), permutation);
    let halfBits = getFeistelHalfBits(max(count, 1u));
    let firstRow = tile * TILE_ROWS;
    let endRow = min(firstRow + TILE_ROWS, count);
    for (var location = firstRow; location < endRow; location++) {
      let bandwidth = rows[rowsOffset + location * ROW_STRIDE + 2u];
      if (!(bandwidth > 0.0)) {
        continue;
      }
      var estimate: array<f32, ${P}>;
      if (!fitPermuted(location, bandwidth, count, halfBits, roundKeys, &estimate)) {
        continue;
      }
      sums[0] += 1.0;
      for (var column = 0u; column < P; column++) {
        let deviation = estimate[column] - shifts[shiftsOffset + column];
        sums[1u + column] += deviation;
        sums[1u + P + column] += deviation * deviation;
      }
    }
    let base = partialOffset + (permutation * TILE_COUNT + tile) * PARTIAL_STRIDE;
    for (var slot = 0u; slot < PARTIAL_STRIDE; slot++) {
      partial[base + slot] = sums[slot];
    }
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-merge`,
        operation: OPERATION,
        variant: 'merge',
        bindings: [
          read('partial', partial, 'f32'),
          read('parameters', props.parameters, 'u32'),
          write('fitCounts', fitCounts, 'f32'),
          write('standardDeviations', standardDeviations, 'f32')
        ],
        invocationCount: maximumPermutations + 1,
        declarations: permutationDeclarations,
        body: `let nan = getNaN();
  var sums: array<f32, ${partialStride}>;
  if (index <= readPermutationCount()) {
    for (var tile = 0u; tile < TILE_COUNT; tile++) {
      let base = partialOffset + (index * TILE_COUNT + tile) * PARTIAL_STRIDE;
      for (var slot = 0u; slot < PARTIAL_STRIDE; slot++) {
        sums[slot] += partial[base + slot];
      }
    }
  } else {
    sums[0] = -1.0;
  }
  fitCounts[fitCountsOffset + index] = select(nan, sums[0], sums[0] >= 0.0);
  for (var column = 0u; column < P; column++) {
    var spread = nan;
    if (sums[0] > 0.0) {
      let mean = sums[1u + column] / sums[0];
      spread = sqrt(max(sums[1u + P + column] / sums[0] - mean * mean, 0.0));
    }
    standardDeviations[standardDeviationsOffset + index * P + column] = spread;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: OPERATION,
        variant: 'finish',
        bindings: [
          read('standardDeviations', standardDeviations, 'f32'),
          read('fitCounts', fitCounts, 'f32'),
          read('parameters', props.parameters, 'u32'),
          read('rows', rows, 'f32'),
          write('table', output.table, 'f32'),
          write('summary', output.summary, 'f32')
        ],
        invocationCount: P,
        declarations: permutationDeclarations,
        body: `let nan = getNaN();
  let permutations = readPermutationCount();
  let observed = standardDeviations[standardDeviationsOffset + index];
  var greater = 0u;
  var validCount = 0.0;
  var total = 0.0;
  for (var run = 1u; run <= permutations; run++) {
    let value = standardDeviations[standardDeviationsOffset + run * P + index];
    if (isFiniteBits(value)) {
      validCount += 1.0;
      total += value;
      if (isFiniteBits(observed) && value >= observed) {
        greater += 1u;
      }
    }
  }
  let mean = select(nan, total / validCount, validCount > 0.0);
  var squares = 0.0;
  for (var run = 1u; run <= permutations; run++) {
    let value = standardDeviations[standardDeviationsOffset + run * P + index];
    if (isFiniteBits(value)) {
      squares += (value - mean) * (value - mean);
    }
  }
  let simulatedDeviation = select(nan, sqrt(squares / validCount), validCount > 0.0);
  let base = tableOffset + index * ${GPU_GWR_NONSTATIONARITY_TABLE_STRIDE}u;
  table[base] = observed;
  table[base + 1u] = select(nan, f32(greater + 1u) / f32(permutations + 1u), isFiniteBits(observed));
  table[base + 2u] = select(nan, f32(greater), isFiniteBits(observed));
  table[base + 3u] = mean;
  table[base + 4u] = simulatedDeviation;
  table[base + 5u] = select(nan, (observed - mean) / simulatedDeviation, simulatedDeviation > 0.0);
  if (index == 0u) {
    let count = rows[rowsOffset + ${rowCountSlot}u];
    var failed = 0.0;
    for (var run = 1u; run <= permutations; run++) {
      failed += count - fitCounts[fitCountsOffset + run];
    }
    summary[summaryOffset + ${GPU_GWR_NONSTATIONARITY_SUMMARY.permutations}u] = f32(permutations);
    summary[summaryOffset + ${GPU_GWR_NONSTATIONARITY_SUMMARY.locationCount}u] = count;
    summary[summaryOffset + ${GPU_GWR_NONSTATIONARITY_SUMMARY.observedFitCount}u] = fitCounts[fitCountsOffset];
    summary[summaryOffset + ${GPU_GWR_NONSTATIONARITY_SUMMARY.failedFitCount}u] = failed;
  }`
      })
    ];
  }
}

const COMMON_WGSL = /* wgsl */ `
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn isFiniteBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
`;

/** `isFiniteRow`: every coefficient of a row is finite. Needs the `coefficients` binding. */
const COEFFICIENT_ROW_WGSL = /* wgsl */ `
fn isFiniteRow(row: u32) -> bool {
  for (var column = 0u; column < P; column++) {
    if (!isFiniteBits(coefficients[coefficientsOffset + row * P + column])) {
      return false;
    }
  }
  return true;
}
`;

/**
 * `fitPermuted`: weighted least squares at included location `location` with the observations
 * relabelled by the Feistel permutation. Kernel weights come from the (unchanged) geometry, the
 * data of included row `j` is that of row `permutation(j)`, and the design is centered on the
 * permuted focal observation exactly like the regression's own fit. Writes the estimates
 * (intercept in original coordinates, then slopes) to `estimate`; returns false when singular.
 */
function getPermutedFitWGSL(coefficientCount: number, hasGrid: boolean): string {
  const P = coefficientCount;
  return /* wgsl */ `
fn getWeight(distance: f32, bandwidth: f32) -> f32 {
  let ratio = distance / bandwidth;
  if (params[paramsOffset] > 0.5) {
    if (distance >= bandwidth) {
      return 0.0;
    }
    let t = 1.0 - ratio * ratio;
    return t * t;
  }
  return exp(-0.5 * ratio * ratio);
}

fn fitPermuted(
  location: u32,
  bandwidth: f32,
  count: u32,
  halfBits: u32,
  roundKeys: vec4<u32>,
  estimate: ptr<function, array<f32, ${P}>>
) -> bool {
  var a: array<f32, ${P * P}>;
  var b: array<f32, ${P}>;
  var z: array<f32, ${P}>;
  var scale: array<f32, ${P}>;
  let origin = vec2f(rows[rowsOffset + location * ROW_STRIDE], rows[rowsOffset + location * ROW_STRIDE + 1u]);
  let focalBase = rowsOffset + getFeistelPermutationIndex(location, count, halfBits, roundKeys) * ROW_STRIDE;
  z[0] = 1.0;
  ${
    hasGrid
      ? `// Bisquare weights vanish beyond the bandwidth, so only the covering grid cells are visited.
  let useGrid = params[paramsOffset] > 0.5 && bandwidth > 0.0 && isFiniteBits(bandwidth);
  let range = select(vec4u(0u, 0u, 0u, 0u), getGridRange(origin, bandwidth), useGrid);
  let gridRowCount = select(1u, range.w - range.z + 1u, useGrid);
  for (var gridRowStep = 0u; gridRowStep < gridRowCount; gridRowStep++) {
    var firstSlot = 0u;
    var endSlot = count;
    if (useGrid) {
      let cellRow = (range.z + gridRowStep) * INDEX_WIDTH;
      firstSlot = cellOffsets[cellOffsetsOffset + cellRow + range.x];
      endSlot = cellOffsets[cellOffsetsOffset + cellRow + range.y + 1u];
    }
    for (var slot = firstSlot; slot < endSlot; slot++) {
      var row = slot;
      if (useGrid) {
        row = sortedIds[sortedIdsOffset + slot];
      }
      let rowBase = rowsOffset + row * ROW_STRIDE;
      let delta = vec2f(rows[rowBase], rows[rowBase + 1u]) - origin;
      let weight = getWeight(sqrt(dot(delta, delta)), bandwidth);
      if (!(weight > 0.0)) {
        continue;
      }
      let dataBase = rowsOffset + getFeistelPermutationIndex(row, count, halfBits, roundKeys) * ROW_STRIDE;
      for (var column = 0u; column < PREDICTOR_COUNT; column++) {
        z[column + 1u] = rows[dataBase + 3u + column] - rows[focalBase + 3u + column];
      }
      let y = rows[dataBase + 3u + PREDICTOR_COUNT];
      for (var r = 0u; r < P; r++) {
        let weighted = weight * z[r];
        b[r] = b[r] + weighted * y;
        for (var c = 0u; c <= r; c++) {
          a[r * P + c] = a[r * P + c] + weighted * z[c];
        }
      }
    }
  }`
      : `for (var row = 0u; row < count; row++) {
    let rowBase = rowsOffset + row * ROW_STRIDE;
    let delta = vec2f(rows[rowBase], rows[rowBase + 1u]) - origin;
    let weight = getWeight(sqrt(dot(delta, delta)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let dataBase = rowsOffset + getFeistelPermutationIndex(row, count, halfBits, roundKeys) * ROW_STRIDE;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      z[column + 1u] = rows[dataBase + 3u + column] - rows[focalBase + 3u + column];
    }
    let y = rows[dataBase + 3u + PREDICTOR_COUNT];
    for (var r = 0u; r < P; r++) {
      let weighted = weight * z[r];
      b[r] = b[r] + weighted * y;
      for (var c = 0u; c <= r; c++) {
        a[r * P + c] = a[r * P + c] + weighted * z[c];
      }
    }
  }`
  }
  for (var r = 0u; r < P; r++) {
    let diagonal = a[r * P + r];
    if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) {
      return false;
    }
    scale[r] = 1.0 / sqrt(diagonal);
  }
  for (var r = 0u; r < P; r++) {
    for (var c = 0u; c <= r; c++) {
      a[r * P + c] = a[r * P + c] * scale[r] * scale[c];
    }
    b[r] = b[r] * scale[r];
  }
  if (!choleskyFactor_${P}(&a)) {
    return false;
  }
  choleskySolve_${P}(&a, &b);
  var intercept = b[0] * scale[0];
  for (var column = 1u; column < P; column++) {
    let slope = b[column] * scale[column];
    (*estimate)[column] = slope;
    intercept = intercept - slope * rows[focalBase + 2u + column];
  }
  (*estimate)[0] = intercept;
  var finite = true;
  for (var column = 0u; column < P; column++) {
    finite = finite && isFiniteBits((*estimate)[column]);
  }
  return finite;
}
`;
}

/** `getPosition`: position of a compacted row. Needs the `rows` binding. */
function getRowPositionWGSL(): string {
  return /* wgsl */ `
fn getPosition(row: u32) -> vec2f {
  return vec2f(rows[rowsOffset + row * ROW_STRIDE], rows[rowsOffset + row * ROW_STRIDE + 1u]);
}
`;
}
