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
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  GPU_CELL_DEFAULT_SUM_SCALE,
  validateSumScale
} from '../../gpu-spatial-analysis/cell-aggregation/cell-table';
import {readBinding, writeBinding} from './group-statistics-common';
import {getColumnReductionNodes} from './group-statistics-moments';
import {getColumnOrderNodes} from './group-statistics-order';
import {getGroupStructureNodes} from './group-statistics-sort';

const OPERATION = 'GPUGroupStatistics';

/** Largest number of value columns one contributor instance reduces. */
const MAXIMUM_COLUMN_COUNT = 4;

/** Statistics {@link GPUGroupStatistics} can compute per group and value column. */
export type GPUGroupStatistic =
  | 'count'
  | 'sum'
  | 'mean'
  | 'minimum'
  | 'maximum'
  | 'variance'
  | 'standardDeviation'
  | 'skewness'
  | 'kurtosis'
  | 'median'
  | 'percentiles'
  | 'mode'
  | 'uniqueCount'
  | 'zScore';

/**
 * Caller-owned output views of one value column. Provide exactly the views of the statistics that
 * the column requests (`sum` takes `sums`, `sumValues`, or both). Per-group views hold
 * `output.keys.length` rows in ascending key order; rows past `output.count` hold zero for
 * `counts`, `sums`, `sumValues` and `uniqueCounts`, and NaN for every other float statistic.
 */
export type GPUGroupStatisticsColumnOutput = {
  /** Finite values per group (`count`). */
  counts?: GraphDataView<'uint32'>;
  /** Exact fixed-point sums, signed 64-bit `(low, high)` words with the cell-table semantics. */
  sums?: GraphDataView<'uint32x2'>;
  /** `sums / sumScale` as f32 (`sum`). */
  sumValues?: GraphDataView<'float32'>;
  /** Exact fixed-point sum divided by the finite count (`mean`); NaN for empty groups. */
  means?: GraphDataView<'float32'>;
  /** Minimum finite value (`minimum`); NaN for empty groups. `-0` is reported as `+0`. */
  minimums?: GraphDataView<'float32'>;
  /** Maximum finite value (`maximum`); NaN for empty groups. `-0` is reported as `+0`. */
  maximums?: GraphDataView<'float32'>;
  /** Variance (`variance`), sample or population per {@link GPUGroupStatisticsProps.variance}. */
  variances?: GraphDataView<'float32'>;
  /** Square root of the variance (`standardDeviation`). */
  standardDeviations?: GraphDataView<'float32'>;
  /** Fisher-Pearson skewness `g1` (`skewness`); NaN when the group is constant. */
  skewness?: GraphDataView<'float32'>;
  /** Excess kurtosis `g2` (`kurtosis`); NaN when the group is constant. */
  kurtosis?: GraphDataView<'float32'>;
  /** Median, the linear-interpolated 0.5 quantile (`median`). */
  medians?: GraphDataView<'float32'>;
  /**
   * Quantiles for the `percentiles` fractions, group-major: group `g`, fraction `j` is at row
   * `g * P + j` (`percentiles`). Length is `capacity * P`.
   */
  percentiles?: GraphDataView<'float32'>;
  /** Most frequent finite value; ties resolve to the smallest value (`mode`). */
  modes?: GraphDataView<'float32'>;
  /** Number of distinct finite values (`uniqueCount`). */
  uniqueCounts?: GraphDataView<'uint32'>;
  /**
   * Per-source-row z-score (`zScore`), length equal to the row count: `(v - mean) / sd` with the
   * configured variance kind; 0 when sd is 0 (including one-row sample groups); NaN for masked,
   * invalid-key, non-finite, or capacity-dropped rows.
   */
  zScores?: GraphDataView<'float32'>;
};

/** One value column and the statistics to compute for it. */
export type GPUGroupStatisticsColumn = {
  /** Per-row values. Rows with a non-finite value are excluded from this column only. */
  values: GraphDataView<'float32'>;
  /** Statistics to compute. This set is topology: changing it needs a new graph. */
  statistics: readonly GPUGroupStatistic[];
  /** Output views, one per requested statistic. */
  output: GPUGroupStatisticsColumnOutput;
};

/**
 * Properties for {@link GPUGroupStatistics}.
 *
 * Per-frame (no recompile): the contents of `keys`, `mask`, every `values` column, and the
 * `percentiles` fractions. Topology (needs a new graph): view lengths, the capacity, each column's
 * statistic set, `variance`, `sumScale`, and the number of percentiles.
 */
export type GPUGroupStatisticsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'group-statistics'`. */
  id?: string;
  /**
   * Group key per row: `uint32`, or `uint32x2` little-endian `(low, high)` words (Arrow `Uint64`).
   * The all-ones key (`0xffffffff`, or both words) is reserved and means "no key": such rows are
   * skipped.
   */
  keys: GraphDataView<'uint32'> | GraphDataView<'uint32x2'>;
  /** Optional per-row mask; zero skips the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Dense mode. When set, `keys` must be `uint32` with group keys in `[0, keyCount)` and the
   * output tables are indexed by key: `output.keys.length` and every per-group view must hold
   * exactly `keyCount` rows, row `k` describes key `k`, and keys without rows stay in the table
   * with `counts` 0, NaN float statistics (mean, minimum, maximum, variance, standard deviation,
   * skewness, kurtosis, median, percentiles, mode) and 0 sums and unique counts. `output.keys[k]`
   * is `k`, `output.count` and `output.requiredCount` are `keyCount` and `overflow` is 0. Rows whose
   * key is outside the range (or all-ones) are skipped like masked rows. No key sort of the
   * output is needed, so dense choropleth tables line up with feature rows. Topology.
   * Defaults to the compact table of occupied keys.
   */
  keyCount?: number;
  /** Zero to four value columns. */
  columns: readonly GPUGroupStatisticsColumn[];
  /** `'sample'` (n - 1, default, as d3 and kepler) or `'population'` (n). Stdev and z-scores follow it. */
  variance?: 'sample' | 'population';
  /**
   * Per-frame quantile fractions in [0, 1] (clamped), `P` entries. Required when a column requests
   * `percentiles`. Typically a {@link GPUParameterBuffer} view; rewriting it needs no
   * rebuild.
   */
  percentiles?: GraphDataView<'float32'>;
  /** Fixed-point scale of the sums and means, a positive float32. Defaults to 65536. */
  sumScale?: number;
  /** Caller-owned group table. Capacity is `output.keys.length`. */
  output: {
    /** Distinct keys ascending, same format as `keys`. Empty tail rows hold the all-ones key. */
    keys: GraphDataView<'uint32'> | GraphDataView<'uint32x2'>;
    /** Valid rows per group (mask set and key valid, regardless of value finiteness). */
    counts: GraphDataView<'uint32'>;
    /** One-row scalar receiving `min(requiredCount, capacity)`. */
    count: GraphDataView<'uint32'>;
    /** One-row scalar receiving 1 when groups were dropped for lack of capacity. */
    overflow: GraphDataView<'uint32'>;
    /** Optional one-row scalar receiving the unclamped number of groups. */
    requiredCount?: GraphDataView<'uint32'>;
  };
};

/** Output field each statistic requires. */
const STATISTIC_OUTPUTS: Record<
  GPUGroupStatistic,
  readonly (keyof GPUGroupStatisticsColumnOutput)[]
> = {
  count: ['counts'],
  sum: ['sums', 'sumValues'],
  mean: ['means'],
  minimum: ['minimums'],
  maximum: ['maximums'],
  variance: ['variances'],
  standardDeviation: ['standardDeviations'],
  skewness: ['skewness'],
  kurtosis: ['kurtosis'],
  median: ['medians'],
  percentiles: ['percentiles'],
  mode: ['modes'],
  uniqueCount: ['uniqueCounts'],
  zScore: ['zScores']
};

const FLOAT_OUTPUTS = [
  'sumValues',
  'means',
  'minimums',
  'maximums',
  'variances',
  'standardDeviations',
  'skewness',
  'kurtosis',
  'medians',
  'percentiles',
  'modes',
  'zScores'
] as const;

/**
 * Groups rows by a 32- or 64-bit key and computes kepler-style statistics per group and column
 * entirely on the GPU, into a capacity-bounded table with ascending keys.
 *
 * Statistics: count, sum, mean, minimum, maximum, variance, standard deviation, skewness,
 * kurtosis, median, percentiles, mode, unique count, and a per-row group z-score.
 *
 * Algorithm: rows are stable-sorted by key (LSD radix passes over the 32-bit words) and group
 * ranges come from head flags and a scan. Counts, fixed-point sums (64-bit, explicit carry) and
 * extremes (order-preserving keys) use integer atomics, so they are exact and independent of
 * thread order. Moments are corrected two-pass: `d = v - mean` is reduced in a fixed order (one
 * 256-thread workgroup per group, strided partial sums plus a fixed binary tree), and the central
 * sums about the true mean come from the shifted sums, so the result is bitwise reproducible.
 * Order statistics re-sort each requested column by value, then by key, so each group's finite
 * values are ascending; quantiles are direct lookups (`lo + (hi - lo) * frac`, numpy `linear`),
 * modes and unique counts come from run heads, a scan, and integer atomics. Ties resolve to the
 * smallest value. When more groups exist than the capacity, the smallest keys are kept.
 *
 * Inputs must be single packed views. Not included: HyperLogLog-style approximate distinct counts.
 */
export class GPUGroupStatistics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGroupStatisticsProps;
  /** Variance kind used by variance, standard deviation and z-scores. */
  readonly variance: 'sample' | 'population';
  /** Fixed-point scale of the sums. */
  readonly sumScale: number;

  constructor(props: GPUGroupStatisticsProps) {
    this.id = props.id ?? 'group-statistics';
    this.props = props;
    const id = this.id;
    this.variance = props.variance ?? 'sample';
    if (this.variance !== 'sample' && this.variance !== 'population') {
      throw new Error(`${id} variance must be 'sample' or 'population'`);
    }
    this.sumScale = props.sumScale ?? GPU_CELL_DEFAULT_SUM_SCALE;
    validateSumScale(id, this.sumScale);

    const {keys, mask, columns, output} = props;
    if (columns.length > MAXIMUM_COLUMN_COUNT) {
      throw new Error(`${id} supports at most ${MAXIMUM_COLUMN_COUNT} columns`);
    }
    const singleViews: [string, GraphDataView | undefined][] = [
      ['keys', keys],
      ['mask', mask],
      ['percentiles', props.percentiles],
      ['output.keys', output.keys],
      ['output.counts', output.counts],
      ['output.count', output.count],
      ['output.overflow', output.overflow],
      ['output.requiredCount', output.requiredCount]
    ];
    for (const [name, view] of singleViews) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view`);
      }
    }
    validatePackedView(keys, ['uint32', 'uint32x2'], `${id} keys`);
    const rowCount = keys.length;
    if (rowCount < 1) {
      throw new Error(`${id} keys must hold at least one row`);
    }
    if (mask) {
      validatePackedUint32View(mask, `${id} mask`);
      if (mask.length !== rowCount) {
        throw new Error(`${id} mask must have the same length as keys`);
      }
    }
    if (props.keyCount !== undefined) {
      if (!Number.isInteger(props.keyCount) || props.keyCount < 1 || props.keyCount >= 0xffffffff) {
        throw new Error(`${id} keyCount must be an integer in [1, 2^32 - 2]`);
      }
      if (keys.format !== 'uint32') {
        throw new Error(`${id} keyCount needs uint32 keys`);
      }
      if (output.keys.length !== props.keyCount) {
        throw new Error(`${id} output.keys must hold exactly keyCount rows`);
      }
    }
    validatePackedView(output.keys, [keys.format as 'uint32' | 'uint32x2'], `${id} output.keys`);
    const capacity = output.keys.length;
    if (capacity < 1) {
      throw new Error(`${id} output.keys must hold at least one row`);
    }
    validatePackedUint32View(output.counts, `${id} output.counts`);
    if (output.counts.length !== capacity) {
      throw new Error(`${id} output.counts must have the same length as output.keys`);
    }
    for (const name of ['count', 'overflow', 'requiredCount'] as const) {
      const view = output[name];
      if (!view) {
        continue;
      }
      validatePackedUint32View(view, `${id} output.${name}`);
      if (view.length < 1) {
        throw new Error(`${id} output.${name} must contain one uint32 row`);
      }
    }

    let percentileColumnCount = 0;
    for (const [columnIndex, column] of columns.entries()) {
      const name = `columns[${columnIndex}]`;
      if ((column.values as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name}.values must be a single packed view`);
      }
      validatePackedView(column.values, ['float32'], `${id} ${name}.values`);
      if (column.values.length !== rowCount) {
        throw new Error(`${id} ${name}.values must have the same length as keys`);
      }
      const requested = new Set(column.statistics);
      for (const statistic of requested) {
        if (!(statistic in STATISTIC_OUTPUTS)) {
          throw new Error(`${id} ${name} has unknown statistic '${statistic}'`);
        }
      }
      for (const statistic of requested) {
        const fields = STATISTIC_OUTPUTS[statistic];
        if (!fields.some(field => column.output[field])) {
          throw new Error(
            `${id} ${name} statistic '${statistic}' needs output.${fields.join(' or output.')}`
          );
        }
      }
      for (const [field, view] of Object.entries(column.output)) {
        if (!view) {
          continue;
        }
        const owner = (
          Object.entries(STATISTIC_OUTPUTS) as [
            GPUGroupStatistic,
            readonly (keyof GPUGroupStatisticsColumnOutput)[]
          ][]
        ).find(([, fields]) => fields.includes(field as keyof GPUGroupStatisticsColumnOutput));
        if (!owner) {
          throw new Error(`${id} ${name} has unknown output.${field}`);
        }
        if (!requested.has(owner[0])) {
          throw new Error(`${id} ${name} output.${field} requires statistic '${owner[0]}'`);
        }
        if ((view as unknown) instanceof GraphVectorView) {
          throw new Error(`${id} ${name} output.${field} must be a single packed view`);
        }
      }
      const columnOutput = column.output;
      for (const field of ['counts', 'uniqueCounts'] as const) {
        const view = columnOutput[field];
        if (view) {
          validatePackedUint32View(view, `${id} ${name}.output.${field}`);
        }
      }
      if (columnOutput.sums) {
        validatePackedView(columnOutput.sums, ['uint32x2'], `${id} ${name}.output.sums`);
      }
      for (const field of FLOAT_OUTPUTS) {
        const view = columnOutput[field];
        if (view) {
          validatePackedView(view, ['float32'], `${id} ${name}.output.${field}`);
        }
      }
      const expectedLengths: [keyof GPUGroupStatisticsColumnOutput, number][] = [
        ['zScores', rowCount]
      ];
      for (const field of [
        'counts',
        'sums',
        'sumValues',
        'means',
        'minimums',
        'maximums',
        'variances',
        'standardDeviations',
        'skewness',
        'kurtosis',
        'medians',
        'modes',
        'uniqueCounts'
      ] as const) {
        expectedLengths.push([field, capacity]);
      }
      for (const [field, expectedLength] of expectedLengths) {
        const view = columnOutput[field];
        if (view && view.length !== expectedLength) {
          throw new Error(
            `${id} ${name}.output.${field} must have ${
              field === 'zScores' ? 'the same length as keys' : 'the same length as output.keys'
            }`
          );
        }
      }
      if (requested.has('percentiles')) {
        percentileColumnCount++;
        if (!props.percentiles) {
          throw new Error(`${id} ${name} statistic 'percentiles' needs the percentiles view`);
        }
        const percentileOutput = columnOutput.percentiles;
        if (percentileOutput && percentileOutput.length !== capacity * props.percentiles.length) {
          throw new Error(
            `${id} ${name}.output.percentiles must hold output.keys.length * percentiles.length rows`
          );
        }
      }
    }
    if (props.percentiles) {
      validatePackedView(props.percentiles, ['float32'], `${id} percentiles`);
      if (props.percentiles.length < 1) {
        throw new Error(`${id} percentiles must hold at least one fraction`);
      }
      if (percentileColumnCount === 0) {
        throw new Error(`${id} percentiles is only used by the 'percentiles' statistic`);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  private getInputViews(): (GraphDataView | undefined)[] {
    const {props} = this;
    return [
      props.keys,
      props.mask,
      props.percentiles,
      ...props.columns.map(column => column.values)
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.keys,
      output.counts,
      output.count,
      output.overflow,
      output.requiredCount,
      ...this.props.columns.flatMap(column => Object.values(column.output))
    ];
  }

  /** Returns key sort, group structure, per-column reduction, order statistic and publish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const {nodes, structure} = getGroupStructureNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      keys: props.keys,
      mask: props.mask,
      keyCount: props.keyCount,
      output: props.output
    });
    const result: GPUCommandNode<Parameters>[] = [...nodes];
    for (const [columnIndex, column] of props.columns.entries()) {
      const columnId = `${id}-c${columnIndex}`;
      const requested = new Set(column.statistics);
      const columnOutput = column.output;
      // Reductions and order statistics read raw f32 bits so NaN handling is bit-exact.
      const valueBits = createTransientView(
        graph,
        `${columnId}-value-bits`,
        'uint32',
        structure.rowCount
      );
      result.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${columnId}-value-bits`,
          operation: OPERATION,
          variant: 'value-bits',
          bindings: [
            readBinding('values', column.values, 'f32'),
            writeBinding('valueBits', valueBits)
          ],
          invocationCount: structure.rowCount,
          body: 'valueBits[valueBitsOffset + index] = bitcast<u32>(values[valuesOffset + index]);'
        })
      );
      const hasMoments = ['variance', 'standardDeviation', 'skewness', 'kurtosis'].some(statistic =>
        requested.has(statistic as GPUGroupStatistic)
      );
      const reduction = getColumnReductionNodes<Parameters>(graph, {
        id: columnId,
        operation: OPERATION,
        structure,
        valueBits,
        needs: {
          sums: requested.has('sum') || requested.has('mean'),
          minimum: requested.has('minimum'),
          maximum: requested.has('maximum'),
          moments: hasMoments,
          zScore: requested.has('zScore')
        },
        variance: this.variance,
        sumScale: this.sumScale,
        output: columnOutput
      });
      result.push(...reduction.nodes);
      if (
        requested.has('median') ||
        requested.has('percentiles') ||
        requested.has('mode') ||
        requested.has('uniqueCount')
      ) {
        result.push(
          ...getColumnOrderNodes<Parameters>(graph, {
            id: columnId,
            operation: OPERATION,
            structure,
            valueBits,
            finiteCounts: reduction.finiteCounts,
            percentiles: props.percentiles,
            output: columnOutput
          })
        );
      }
    }
    return result;
  }
}
