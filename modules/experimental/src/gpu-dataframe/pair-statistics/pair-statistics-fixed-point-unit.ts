// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {getColumnQuantileNodes} from '../column-classification/column-quantiles-nodes';
import {PAIR_STATISTICS_FLOAT_WGSL} from './pair-statistics-grid';

/**
 * Smallest fraction of the variance that the fixed-point unit may take. It only guards the scale
 * `2^24 / u` against f32 overflow when the interquartile spread underflows; a unit far below the
 * variance costs capacity of the first channel, not accuracy, because terms above 256 units go to
 * the exact large channel.
 */
export const PAIR_STATISTICS_MINIMUM_UNIT_FRACTION = 2 ** -32;

/**
 * Fraction of the variance used as the unit when the interquartile range is zero (more than half
 * of the values tied) or not finite.
 */
export const PAIR_STATISTICS_ZERO_SPREAD_UNIT_FRACTION = 2 ** -12;

/** Standard deviations in the interquartile range of a normal distribution, `2 * 0.6744898`. */
const INTERQUARTILE_RANGE_PER_SIGMA = 1.3489795;

/** A variance below this (f32) is treated as zero: its reciprocal scale would overflow. */
const MINIMUM_VARIANCE = 1e-30;

/**
 * Builds the nodes that write the "typical term variance" `u` of a value column to a one-row
 * transient: the robust variance `(IQR / 1.349)^2` from the exact 25% and 75% quantiles of the
 * (masked, finite) values, clamped to `[variance * 2^-32, variance]` with `variance =
 * sumOfSquares / n` from the statistics pass (`variance * 2^-12` when the IQR is 0 or not finite). A kernel reads `u` and scales a pair term whose
 * typical magnitude is one variance (`(z_i - z_j)^2 / 2`, `c_i c_j`) by `2^24 / u`.
 *
 * The plain variance would not do: it is itself dominated by one outlier or sentinel
 * (`variance ~ outlier^2 / n`), so the fixed-point unit would again scale with the maximum. The
 * interquartile range ignores up to a quarter of contaminated rows. `u` is 0 when the variance is
 * not positive or finite (every term then reads as 0, as for a constant column).
 *
 * @internal
 */
export function getPairStatisticsUnitVarianceNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    /** Packed float32 observation column. */
    values: GraphDataView<'float32'>;
    /** Optional row mask. */
    mask?: GraphDataView<'uint32'>;
    /** The statistics view of `getPairStatisticsInputNodes` (centered values, then moments). */
    statistics: GraphDataView<'float32'>;
  }
): {nodes: GPUCommandNode<Parameters>[]; unitVariance: GraphDataView<'float32'>} {
  const {id, operation, values, mask, statistics} = props;
  const rows = values.length;
  const quantileParameters = createTransientView(graph, `${id}-unit-parameters`, 'float32', 6);
  const quantiles = createTransientView(graph, `${id}-unit-quantiles`, 'float32', 2);
  const validCount = createTransientView(graph, `${id}-unit-valid-count`, 'uint32', 1);
  const unitVariance = createTransientView(graph, `${id}-unit-variance`, 'float32', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    // [linear interpolation, filter lower, filter upper, 0, 0.25, 0.75]
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-unit-parameters`,
      operation,
      variant: 'unit-parameters',
      bindings: [
        {name: 'quantileParameters', view: quantileParameters, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      body: `quantileParameters[quantileParametersOffset] = 3.0;
  quantileParameters[quantileParametersOffset + 1u] = 0.0;
  quantileParameters[quantileParametersOffset + 2u] = 1.0;
  quantileParameters[quantileParametersOffset + 3u] = 0.0;
  quantileParameters[quantileParametersOffset + 4u] = 0.25;
  quantileParameters[quantileParametersOffset + 5u] = 0.75;`
    }),
    ...getColumnQuantileNodes<Parameters>(graph, {
      id: `${id}-unit-quantile`,
      operation,
      values,
      mask,
      parameters: quantileParameters,
      quantileCount: 2,
      output: {quantiles, validCount}
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-unit-variance`,
      operation,
      variant: 'unit-variance',
      bindings: [
        {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
        {name: 'quantiles', view: quantiles, type: 'f32', access: 'read'},
        {name: 'unitVariance', view: unitVariance, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
const MOMENTS: u32 = ${rows}u;`,
      body: `let variance = statistics[statisticsOffset + MOMENTS + 2u] / statistics[statisticsOffset + MOMENTS];
  let spread = (quantiles[quantilesOffset + 1u] - quantiles[quantilesOffset]) / ${INTERQUARTILE_RANGE_PER_SIGMA};
  let usable = isFiniteFloat(spread) && spread > 0.0;
  let robustVariance = select(variance * ${PAIR_STATISTICS_ZERO_SPREAD_UNIT_FRACTION}, spread * spread, usable);
  let unit = clamp(robustVariance, variance * ${PAIR_STATISTICS_MINIMUM_UNIT_FRACTION}, variance);
  unitVariance[unitVarianceOffset] =
    select(0.0, unit, isFiniteFloat(variance) && variance > ${MINIMUM_VARIANCE});`
    })
  ];
  return {nodes, unitVariance};
}

/**
 * CPU mirror of the unit variance kernel for oracles: the robust variance from the `linear`
 * interpolated quartiles of the finite `values`, clamped to `[variance * 2^-32, variance]`.
 *
 * @internal
 */
export function getPairStatisticsUnitVarianceOnCPU(values: ArrayLike<number>): number {
  const finite = Array.from(values)
    .filter(value => Number.isFinite(value))
    .sort((a, b) => a - b);
  const count = finite.length;
  if (count === 0) {
    return 0;
  }
  const mean = finite.reduce((sum, value) => sum + value, 0) / count;
  const variance = finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / count;
  if (!(variance > MINIMUM_VARIANCE)) {
    return 0;
  }
  const quantile = (p: number) => {
    const height = (count - 1) * p;
    const lower = Math.floor(height);
    const upper = Math.min(lower + 1, count - 1);
    return finite[lower] + (finite[upper] - finite[lower]) * (height - lower);
  };
  const spread = (quantile(0.75) - quantile(0.25)) / INTERQUARTILE_RANGE_PER_SIGMA;
  const robustVariance =
    spread > 0 ? spread * spread : variance * PAIR_STATISTICS_ZERO_SPREAD_UNIT_FRACTION;
  return Math.min(
    Math.max(robustVariance, variance * PAIR_STATISTICS_MINIMUM_UNIT_FRACTION),
    variance
  );
}
