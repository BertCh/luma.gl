// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getColumnProfileNodes, type ColumnProfileColumn} from './column-profile-kernels';
import {
  getGPUColumnProfileParameterLength,
  GPU_COLUMN_PROFILE_MAXIMUM_COLUMN_COUNT,
  GPU_COLUMN_PROFILE_MAXIMUM_TOP_CATEGORY_COUNT,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from './column-profile-parameters';

const OPERATION = 'GPUColumnProfile';
/** Largest column: row counts are held in f32 arithmetic, exact up to 2^24. */
const MAXIMUM_ROW_COUNT = 2 ** 24;
/** Largest histogram bin count; keeps the f32 bin estimate within one bin of the answer. */
const MAXIMUM_HISTOGRAM_BIN_COUNT = 2 ** 20;
const DEFAULT_HYPERLOGLOG_PRECISION = 12;
const DEFAULT_TOP_CATEGORY_COUNT = 10;
/** Largest total number of category count slots across category columns. */
const MAXIMUM_CATEGORY_TOTAL = 2 ** 26;

/** One column of a {@link GPUColumnProfile}. */
export type GPUColumnProfileColumn =
  | {
      /** Packed float32 column. NaN rows are nulls. */
      values: GraphDataView<'float32'>;
      /** Defaults to `'numeric'`. */
      kind?: 'numeric';
    }
  | {
      /** Packed dictionary codes. `0xffffffff` is a null; codes `>= categoryCount` overflow. */
      values: GraphDataView<'uint32'>;
      kind: 'category';
      /** Number of dictionary entries, compile-time. Codes below it are counted exactly. */
      categoryCount: number;
    };

/**
 * Caller-owned outputs of {@link GPUColumnProfile}, flat and column-major. Every view is rewritten
 * on every encoding.
 */
export type GPUColumnProfileOutput = {
  /**
   * `GPU_COLUMN_PROFILE_STATISTIC_COUNT` float32 values per column at
   * `column * STATISTIC_COUNT + field`, see `GPU_COLUMN_PROFILE_STATISTIC`.
   */
  statistics: GraphDataView<'float32'>;
  /** Exact `[count, nullCount]` per column: 2 `uint32` per column. */
  counts?: GraphDataView<'uint32'>;
  /**
   * `histogramBinCount` `uint32` per column. Numeric columns: equal-width bins of the per-frame
   * domain, last bin includes `hi`, values outside the domain are not counted. Category columns:
   * the row count of code `j` in bin `j` for `j < min(histogramBinCount, categoryCount)`, else 0.
   */
  histograms?: GraphDataView<'uint32'>;
  /** `2 ** hyperLogLogPrecision` `uint32` registers per column. */
  hyperLogLogRegisters?: GraphDataView<'uint32'>;
  /**
   * `topCategoryCount` codes per column ordered by (count descending, code ascending); unused
   * slots and every numeric column hold `0xffffffff`. Requires `topCategoryCounts`.
   */
  topCategories?: GraphDataView<'uint32'>;
  /** Row counts matching `topCategories`; unused slots and numeric columns hold 0. */
  topCategoryCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUColumnProfile}.
 *
 * Per-frame (no recompile): the contents of every column, `mask`, and `parameters` (histogram
 * domains). Topology (needs a new graph): the column set and kinds, `categoryCount`s,
 * `histogramBinCount`, `hyperLogLogPrecision`, `topCategoryCount`, view lengths, whether `mask`
 * is present, and which optional outputs are present.
 */
export type GPUColumnProfileProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'column-profile'`. */
  id?: string;
  /** 1 to 16 columns of one table: all have the same length, at most 2^24 rows. */
  columns: readonly GPUColumnProfileColumn[];
  /** Optional packed `uint32` row mask applied to every column; zero skips the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Optional per-frame histogram domains, float32, `[lo, hi]` per column, written with
   * `getGPUColumnProfileParameterValues`. NaN means automatic. Omitted means automatic.
   */
  parameters?: GraphDataView<'float32'>;
  /** Equal-width histogram bins per column, 1 to 2^20. Compile-time. */
  histogramBinCount: number;
  /** HyperLogLog precision `p` (2^p registers), 4 to 16. Default 12. */
  hyperLogLogPrecision?: number;
  /** Top-category slots per column, 1 to 64. Default 10. */
  topCategoryCount?: number;
  /** Caller-owned outputs. */
  output: GPUColumnProfileOutput;
};

/**
 * Fused per-column statistics panel (the kepler and Studio dataset stats panel), recomputed on the
 * GPU every frame: count, null count, minimum, maximum, sum, mean, variance, standard deviation,
 * an equal-width histogram, a HyperLogLog distinct estimate, and top-K category counts.
 *
 * Rows are the masked-in rows of one table. Numeric nulls are NaN; category nulls are
 * `0xffffffff`. `+-Infinity` counts and enters minimum and maximum but is excluded from the moments
 * and from histograms. `-0` and `+0` are distinct for minimum and maximum (`-0 < +0`) and the same
 * value for distinct counting.
 *
 * Determinism: minimum and maximum use `atomicMin` and `atomicMax` on order-preserving u32 keys
 * (exact). Count, sum, mean and `M2` are reduced in a fixed order: each workgroup folds eight
 * rows per invocation sequentially, reduces them with a binary tree of Chan merges, then one
 * workgroup merges the tile partials in a second fixed tree. No float atomics are used, so the
 * float statistics are bitwise identical from run to run and independent of GPU thread order,
 * and are accurate to a few f32 ULP of an f64 two-pass oracle even for large offsets (the
 * pairwise update does not cancel). Histogram counts, registers, and category counts are integer
 * atomics and are exact. Histogram bins use `width = fround(range * fround(1 / binCount))`, the
 * estimate `floor(d / width)` and one correction step each way against correctly rounded
 * products, which a CPU mirror reproduces with `Math.fround`.
 *
 * HyperLogLog: the value bits (f32 with `-0` canonicalised to `+0`, NaN skipped; u32 codes as is)
 * are hashed with murmur3 `fmix32(bits ^ seed)`, `seed = (column + 1) * 0x9e3779b9`. The register
 * is the top `p` bits, the rank is `min(clz(rest), 32 - p) + 1`, merged by `atomicMax`, so the
 * registers are exact and order independent. The estimate is the standard raw estimate with the
 * linear-counting correction for small ranges, in f32 (standard error `1.04 / sqrt(2^p)`).
 *
 * Category columns: counts per code live in a `categoryCount`-long u32 array (`atomicAdd`);
 * codes `>= categoryCount` are counted in `count` and `overflowCount` but not in the top list or
 * histogram. The top list picks K times by (count descending, code ascending), so ties break on
 * the smallest code.
 *
 * Inputs must be single packed views. Cost is one pass per column per product (moments, histogram,
 * HyperLogLog, category counts), each a few ms or less for millions of rows.
 */
export class GPUColumnProfile implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUColumnProfileProps;
  /** Rows of every column. */
  readonly rowCount: number;
  /** Resolved HyperLogLog precision. */
  readonly hyperLogLogPrecision: number;
  /** Resolved top-category count. */
  readonly topCategoryCount: number;
  private readonly columns: readonly ColumnProfileColumn[];

  constructor(props: GPUColumnProfileProps) {
    this.id = props.id ?? 'column-profile';
    this.props = props;
    const id = this.id;
    const {output} = props;
    const columnCount = props.columns.length;
    if (columnCount < 1 || columnCount > GPU_COLUMN_PROFILE_MAXIMUM_COLUMN_COUNT) {
      throw new Error(
        `${id} needs between 1 and ${GPU_COLUMN_PROFILE_MAXIMUM_COLUMN_COUNT} columns`
      );
    }
    const {histogramBinCount} = props;
    if (
      !Number.isInteger(histogramBinCount) ||
      histogramBinCount < 1 ||
      histogramBinCount > MAXIMUM_HISTOGRAM_BIN_COUNT
    ) {
      throw new Error(
        `${id} histogramBinCount must be an integer in [1, ${MAXIMUM_HISTOGRAM_BIN_COUNT}]`
      );
    }
    this.hyperLogLogPrecision = props.hyperLogLogPrecision ?? DEFAULT_HYPERLOGLOG_PRECISION;
    if (
      !Number.isInteger(this.hyperLogLogPrecision) ||
      this.hyperLogLogPrecision < 4 ||
      this.hyperLogLogPrecision > 16
    ) {
      throw new Error(`${id} hyperLogLogPrecision must be an integer in [4, 16]`);
    }
    this.topCategoryCount = props.topCategoryCount ?? DEFAULT_TOP_CATEGORY_COUNT;
    if (
      !Number.isInteger(this.topCategoryCount) ||
      this.topCategoryCount < 1 ||
      this.topCategoryCount > GPU_COLUMN_PROFILE_MAXIMUM_TOP_CATEGORY_COUNT
    ) {
      throw new Error(
        `${id} topCategoryCount must be an integer in [1, ${GPU_COLUMN_PROFILE_MAXIMUM_TOP_CATEGORY_COUNT}]`
      );
    }
    for (const [name, view] of [
      ['mask', props.mask],
      ['parameters', props.parameters]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    const rowCount = props.columns[0].values.length;
    if (rowCount < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (rowCount > MAXIMUM_ROW_COUNT) {
      throw new Error(`${id} supports at most ${MAXIMUM_ROW_COUNT} rows`);
    }
    this.rowCount = rowCount;
    let categoryBase = 0;
    this.columns = props.columns.map((column, columnIndex) => {
      const name = `columns[${columnIndex}]`;
      if ((column.values as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
      if (column.values.length !== rowCount) {
        throw new Error(`${id} ${name} length must equal columns[0] length`);
      }
      if (column.kind === 'category') {
        validatePackedUint32View(column.values, `${id} ${name}.values`);
        if (
          !Number.isInteger(column.categoryCount) ||
          column.categoryCount < 1 ||
          column.categoryCount > MAXIMUM_ROW_COUNT
        ) {
          throw new Error(`${id} ${name}.categoryCount must be an integer in [1, 2^24]`);
        }
        const base = categoryBase;
        categoryBase += column.categoryCount;
        return {
          kind: 'category' as const,
          values: column.values,
          categoryCount: column.categoryCount,
          categoryBase: base
        };
      }
      if (column.kind !== undefined && column.kind !== 'numeric') {
        throw new Error(`${id} ${name}.kind must be 'numeric' or 'category'`);
      }
      validatePackedView(column.values, ['float32'], `${id} ${name}.values`);
      return {kind: 'numeric' as const, values: column.values, categoryCount: 0, categoryBase: 0};
    });
    if (categoryBase > MAXIMUM_CATEGORY_TOTAL) {
      throw new Error(
        `${id} category columns may declare at most ${MAXIMUM_CATEGORY_TOTAL} categories in total`
      );
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rowCount) {
        throw new Error(`${id} mask length must equal the column length`);
      }
    }
    if (props.parameters) {
      validatePackedView(props.parameters, ['float32'], `${id} parameters`);
      const parameterLength = getGPUColumnProfileParameterLength(columnCount);
      if (props.parameters.length < parameterLength) {
        throw new Error(`${id} parameters must hold ${parameterLength} float32 values`);
      }
    }
    const requireLength = (name: string, view: GraphDataView, length: number) => {
      if (view.length < length) {
        throw new Error(`${id} output.${name} must hold ${length} rows`);
      }
    };
    validatePackedView(output.statistics, ['float32'], `${id} output.statistics`);
    requireLength(
      'statistics',
      output.statistics,
      columnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT
    );
    for (const [name, view, length] of [
      ['counts', output.counts, columnCount * 2],
      ['histograms', output.histograms, columnCount * histogramBinCount],
      [
        'hyperLogLogRegisters',
        output.hyperLogLogRegisters,
        columnCount * 2 ** this.hyperLogLogPrecision
      ],
      ['topCategories', output.topCategories, columnCount * this.topCategoryCount],
      ['topCategoryCounts', output.topCategoryCounts, columnCount * this.topCategoryCount]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        requireLength(name, view, length);
      }
    }
    if (Boolean(output.topCategories) !== Boolean(output.topCategoryCounts)) {
      throw new Error(`${id} output.topCategories and output.topCategoryCounts go together`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.statistics,
        output.counts,
        output.histograms,
        output.hyperLogLogRegisters,
        output.topCategories,
        output.topCategoryCounts
      ],
      [...props.columns.map(column => column.values), props.mask, props.parameters]
    );
  }

  /**
   * Returns init and clear nodes, one `${id}-column-${c}-moments-tile` per column,
   * `${id}-moments-merge`, `${id}-domain` (with histograms), per-column `-histogram`,
   * `-hyperloglog`, `-category-counts`, `-top-categories`, and `-finish` nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...props.columns.map(column => column.values),
      props.mask,
      props.parameters,
      output.statistics,
      output.counts,
      output.histograms,
      output.hyperLogLogRegisters,
      output.topCategories,
      output.topCategoryCounts
    ]);
    return getColumnProfileNodes(graph, {
      id,
      operation: OPERATION,
      columns: this.columns,
      rowCount: this.rowCount,
      mask: props.mask,
      parameters: props.parameters,
      histogramBinCount: props.histogramBinCount,
      hyperLogLogPrecision: this.hyperLogLogPrecision,
      topCategoryCount: this.topCategoryCount,
      output
    });
  }
}
