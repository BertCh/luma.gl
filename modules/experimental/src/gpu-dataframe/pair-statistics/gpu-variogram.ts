// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
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
import {
  getPairHistogramNodes,
  getPairHistogramReadWGSL,
  validatePairHistogramShape
} from './pair-histogram';
import {
  getPairStatisticsInputNodes,
  PAIR_STATISTICS_FLOAT_WGSL,
  validatePairStatisticsInputs
} from './pair-statistics-grid';
import {
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPU_VARIOGRAM_STATISTICS_LENGTH
} from './variogram-parameters';

const OPERATION = 'GPUVariogram';

/** Fixed-point full scale of one pair term: terms are scaled to at most `2^24` before rounding. */
const FIXED_POINT_SCALE = 16777216;

/**
 * Properties for {@link GPUVariogram}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `values`, `mask` and
 * `parameters` (bounds, maximum distance, azimuth offset). Compile-time: the row count,
 * `gridSize`, `lagCount`, `directionCount`, and which optional outputs are present.
 */
export type GPUVariogramProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'variogram'`. */
  id?: string;
  /** Packed planar points, one row per observation. At least one and fewer than 2^24 rows. */
  positions: GraphDataView<'float32x2'>;
  /** Packed observed values, one per row. Rows with a non-finite value are excluded. */
  values: GraphDataView<'float32'>;
  /**
   * Per-frame parameters: packed float32 view of at least `GPU_VARIOGRAM_PARAMETER_LENGTH`
   * elements written with `getGPUVariogramParameterValues`. Invalid bounds or a non-positive
   * maximum distance exclude every row.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Maximum `[columns, rows]` of the neighbor-search cell lattice. Compile-time. Results never
   * depend on it, only speed: each encoding uses cells at least `maximumDistance` wide.
   */
  gridSize: readonly [number, number];
  /** Number of equal-width lag bins over `[0, maximumDistance]`. Compile-time, positive. */
  lagCount: number;
  /**
   * Number of direction sectors over `[0, pi)`. Compile-time. Defaults to 1 (omnidirectional).
   * `lagCount * directionCount` must be at most 512.
   */
  directionCount?: number;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Caller-owned Matheron semivariance per sector and lag, `directionCount * lagCount` rows
   * (sector-major). Quiet NaN for empty bins.
   */
  semivariances: GraphDataView<'float32'>;
  /** Optional caller-owned pair count per sector and lag (each unordered pair counted once). */
  pairCounts?: GraphDataView<'uint32'>;
  /** Optional caller-owned mean pair distance per sector and lag. Quiet NaN for empty bins. */
  meanDistances?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned Cressie-Hawkins robust semivariance per sector and lag. Quiet NaN for
   * empty bins.
   */
  robustSemivariances?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned `[n, mean, variance, minimum, maximum]` of the included values
   * (`GPU_VARIOGRAM_STATISTICS_LENGTH` rows; population variance). The variance is the usual
   * sill reference for a stationary field.
   */
  statistics?: GraphDataView<'float32'>;
};

/**
 * Empirical semivariogram (gstat `variogram`, scikit-gstat `Variogram`) of planar point values,
 * optionally directional, from one deterministic pass over every pair within the per-frame
 * maximum distance.
 *
 * Definition, which the GPU result matches within the fixed-point bound below:
 * - Included rows have a nonzero mask (when given), a finite value and a finite position inside
 *   the per-frame bounds.
 * - Each unordered pair `{i, j}` of included rows with `d_ij <= maximumDistance` (evaluated in
 *   f32 as `sqrt(dx * dx + dy * dy)`) falls in lag `min(floor(d / maximumDistance * lagCount),
 *   lagCount - 1)` and, with `directionCount > 1`, in sector
 *   `min(floor(a / pi * directionCount), directionCount - 1)` where `a` is
 *   `atan2(dy, dx) - azimuthOffset` reduced modulo pi into `[0, pi)`.
 * - Per bin with `N` pairs: `gamma = sum (z_i - z_j)^2 / (2 N)` (Matheron), mean distance
 *   `sum d / N`, and the Cressie-Hawkins estimator
 *   `gamma_CH = (sum |z_i - z_j|^(1/2) / N)^4 / (2 (0.457 + 0.494 / N))`.
 *
 * Determinism and precision: every pair term is accumulated as an exact 64-bit integer sum of a
 * fixed-point quantization (see the shared pair histogram), so results are bitwise reproducible
 * and independent of scheduling. Each term is scaled so its largest possible value maps to `2^24`
 * (for the semivariance `(max - min)^2 / 2` from the GPU-computed value extremes, for distances
 * `maximumDistance`, for the robust term `sqrt(max - min)`) and rounded to the nearest integer, so
 * the absolute error of a bin mean is at most `2^-25` of that full scale, for example
 * `(max - min)^2 * 2^-26` for `gamma`, plus f32 rounding of the term itself.
 *
 * Model fitting is a CPU step on the read-back bins: see `fitVariogramModel`.
 */
export class GPUVariogram implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUVariogramProps;
  /** Number of direction sectors (1 when omnidirectional). */
  readonly directionCount: number;

  constructor(props: GPUVariogramProps) {
    this.id = props.id ?? 'variogram';
    this.props = props;
    const id = this.id;
    validatePairStatisticsInputs(id, {
      ...props,
      parameterLength: GPU_VARIOGRAM_PARAMETER_LENGTH
    });
    const directionCount = props.directionCount ?? 1;
    this.directionCount = directionCount;
    if (!Number.isInteger(props.lagCount) || props.lagCount < 1) {
      throw new Error(`${id} lagCount must be a positive integer`);
    }
    if (!Number.isInteger(directionCount) || directionCount < 1) {
      throw new Error(`${id} directionCount must be a positive integer`);
    }
    const binCount = props.lagCount * directionCount;
    if (binCount > 512) {
      throw new Error(`${id} lagCount * directionCount must be at most 512`);
    }
    validatePairHistogramShape(id, binCount, 4);
    for (const [name, view] of [
      ['semivariances', props.semivariances],
      ['meanDistances', props.meanDistances],
      ['robustSemivariances', props.robustSemivariances]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length < binCount) {
          throw new Error(`${id} ${name} must hold lagCount * directionCount rows`);
        }
      }
    }
    if (props.pairCounts) {
      validatePackedUint32View(props.pairCounts, `${id} pairCounts`);
      if (props.pairCounts.length < binCount) {
        throw new Error(`${id} pairCounts must hold lagCount * directionCount rows`);
      }
    }
    if (props.statistics) {
      validatePackedView(props.statistics, ['float32'], `${id} statistics`);
      if (props.statistics.length < GPU_VARIOGRAM_STATISTICS_LENGTH) {
        throw new Error(`${id} statistics must hold ${GPU_VARIOGRAM_STATISTICS_LENGTH} rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.semivariances,
        props.pairCounts,
        props.meanDistances,
        props.robustSemivariances,
        props.statistics
      ],
      [props.positions, props.values, props.parameters, props.mask]
    );
  }

  /** Returns the variogram nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, directionCount} = this;
    const {positions, parameters, gridSize, lagCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      props.values,
      parameters,
      props.mask,
      props.semivariances,
      props.pairCounts,
      props.meanDistances,
      props.robustSemivariances,
      props.statistics
    ]);
    const rows = positions.length;
    const binCount = lagCount * directionCount;
    const robust = Boolean(props.robustSemivariances);
    const channelCount = robust ? 4 : 3;
    const inputs = getPairStatisticsInputNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      values: props.values,
      mask: props.mask
    });
    const statistics = inputs.statistics!;
    const nodes = inputs.nodes;
    // Scales derive from the GPU moments and parameters, so the pair and finishing kernels compute
    // bitwise identical values.
    const scaleWGSL = /* wgsl */ `
const MOMENTS: u32 = ${rows}u;
const LAG_COUNT: u32 = ${lagCount}u;
const DIRECTION_COUNT: u32 = ${directionCount}u;
const FIXED_POINT_SCALE: f32 = ${FIXED_POINT_SCALE}.0;
const PI: f32 = 3.14159265358979;

struct VariogramScales {
  semivariance: f32,
  robust: f32,
  distance: f32
}

fn getVariogramScales() -> VariogramScales {
  let valueRange = statistics[statisticsOffset + MOMENTS + 4u] - statistics[statisticsOffset + MOMENTS + 3u];
  var scales: VariogramScales;
  scales.semivariance = select(0.0, FIXED_POINT_SCALE / (0.5 * valueRange * valueRange), valueRange > 0.0);
  scales.robust = select(0.0, FIXED_POINT_SCALE / sqrt(valueRange), valueRange > 0.0);
  scales.distance = FIXED_POINT_SCALE / parameters[parametersOffset + 4u];
  return scales;
}`;
    const histogram = getPairHistogramNodes<Parameters>(graph, {
      id: `${id}-histogram`,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      sortedRows: inputs.sortedRows,
      cellOffsets: inputs.cellOffsets,
      slotCount: binCount,
      channelCount,
      pairOrder: 'unordered',
      extraBindings: [{name: 'statistics', view: statistics, type: 'f32', access: 'read'}],
      declarations: scaleWGSL,
      focusPrologue: `let scales = getVariogramScales();
    let focusValue = statistics[statisticsOffset + focus];
    let azimuthOffset = readParameter(5u);`,
      pairAction: `let lag = min(u32(pairDistance / lattice.maximumDistance * f32(LAG_COUNT)), LAG_COUNT - 1u);
            var sector = 0u;
            if (DIRECTION_COUNT > 1u) {
              let angle = atan2(deltaY, deltaX) - azimuthOffset;
              let reduced = angle - floor(angle / PI) * PI;
              sector = min(u32(max(reduced, 0.0) / PI * f32(DIRECTION_COUNT)), DIRECTION_COUNT - 1u);
            }
            let bin = sector * LAG_COUNT + lag;
            let difference = statistics[statisticsOffset + neighbor] - focusValue;
            accumulate(bin, 0u, 1u);
            accumulate(bin, 1u, quantizePairAmount(0.5 * difference * difference * scales.semivariance));
            accumulate(bin, 2u, quantizePairAmount(pairDistance * scales.distance));
            ${robust ? 'accumulate(bin, 3u, quantizePairAmount(sqrt(abs(difference)) * scales.robust));' : ''}`
    });
    nodes.push(...histogram.nodes);

    const finishBindings: WGSLKernelBinding[] = [
      {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
      {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'semivariances', view: props.semivariances, type: 'f32', access: 'read_write'}
    ];
    if (props.pairCounts) {
      finishBindings.push({
        name: 'pairCounts',
        view: props.pairCounts,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (props.meanDistances) {
      finishBindings.push({
        name: 'meanDistances',
        view: props.meanDistances,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.robustSemivariances) {
      finishBindings.push({
        name: 'robustSemivariances',
        view: props.robustSemivariances,
        type: 'f32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: OPERATION,
        variant: 'finish',
        bindings: finishBindings,
        invocationCount: binCount,
        declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
${scaleWGSL}
${getPairHistogramReadWGSL('accumulators', channelCount)}`,
        body: `let scales = getVariogramScales();
  let pairCount = readPairAccumulator(index, 0u);
  let hasPairs = pairCount > 0.0;
  let nan = getQuietNaN(index);
  let semivarianceSum = readPairAccumulator(index, 1u);
  // A zero value range means every difference is 0 (scale 0, sum 0).
  let semivariance = select(0.0, semivarianceSum / scales.semivariance / pairCount, scales.semivariance > 0.0);
  semivariances[semivariancesOffset + index] = select(nan, semivariance, hasPairs);
  ${props.pairCounts ? 'pairCounts[pairCountsOffset + index] = readPairCount(index, 0u);' : ''}
  ${
    props.meanDistances
      ? `meanDistances[meanDistancesOffset + index] =
    select(nan, readPairAccumulator(index, 2u) / scales.distance / pairCount, hasPairs);`
      : ''
  }
  ${
    robust
      ? `let robustMean = select(0.0, readPairAccumulator(index, 3u) / scales.robust / pairCount, scales.robust > 0.0);
  let robustSquare = robustMean * robustMean;
  let robustValue = robustSquare * robustSquare / (2.0 * (0.457 + 0.494 / pairCount));
  robustSemivariances[robustSemivariancesOffset + index] = select(nan, robustValue, hasPairs);`
      : ''
  }`
      })
    );

    if (props.statistics) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-statistics`,
          operation: OPERATION,
          variant: 'statistics',
          bindings: [
            {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
            {name: 'statisticsOut', view: props.statistics, type: 'f32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: `const MOMENTS: u32 = ${rows}u;`,
          body: `let count = statistics[statisticsOffset + MOMENTS];
  statisticsOut[statisticsOutOffset] = count;
  statisticsOut[statisticsOutOffset + 1u] = statistics[statisticsOffset + MOMENTS + 1u];
  statisticsOut[statisticsOutOffset + 2u] = statistics[statisticsOffset + MOMENTS + 2u] / count;
  statisticsOut[statisticsOutOffset + 3u] = statistics[statisticsOffset + MOMENTS + 3u];
  statisticsOut[statisticsOutOffset + 4u] = statistics[statisticsOffset + MOMENTS + 4u];`
        })
      );
    }
    return nodes;
  }
}
