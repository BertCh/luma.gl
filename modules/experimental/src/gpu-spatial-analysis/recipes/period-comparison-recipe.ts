// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUClassBreaks, GPUColorScale} from '../../gpu-dataframe/column-classification/index';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {GPUCellAggregation} from '../cell-aggregation/index';
import type {GPUCellFamily, GPUCellTable} from '../cell-aggregation/index';
import {GPUCellTableCompare} from '../cell-table-compare/index';
import type {
  GPUCellTableCompareMeasure,
  GPUCellTableCompareOutput,
  GPUCellTableCompareZScore
} from '../cell-table-compare/index';
import {assertRecipe, getOrCreateView, RecipeBuilder, type GPURecipeResult} from './recipe-utils';

const ID = 'addPeriodComparisonRecipe';

/** Rows of one period: longitude/latitude points (Quadbin) or pre-keyed 64-bit cells. */
export type GPUPeriodRows = {
  /** Longitude/latitude degrees per row. Quadbin only; exclusive with `cells`. */
  positions?: GraphDataView<'float32x2'>;
  /** One 64-bit cell key per row as two little-endian `uint32` words. Exclusive with `positions`. */
  cells?: GraphDataView<'uint32x2'>;
  /** Per-row value summed per cell (required for `measure: 'sum'`). */
  values?: GraphDataView<'float32'>;
  /** Per-row mask. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned cell table of this period; missing columns are graph-owned transients. */
  table?: Partial<GPUCellTable>;
};

/** Column of the comparison that is classified and colored. */
export type GPUPeriodComparisonVariable = 'delta' | 'ratio' | 'percentChange' | 'zScore';

/** Properties for {@link addPeriodComparisonRecipe}. */
export type GPUPeriodComparisonRecipeProps = {
  /** Prefix for every node and transient ID. Defaults to `'period-comparison'`. */
  id?: string;
  /** Grid family shared by both periods. */
  family: GPUCellFamily;
  /** Cell resolution shared by both periods. */
  resolution: number;
  /** Rows of each period's cell table. */
  tableCapacity: number;
  /** Rows of the union table (cells present in either period). Defaults to `2 * tableCapacity`. */
  unionCapacity?: number;
  /** Earlier period. */
  before: GPUPeriodRows;
  /** Later period. */
  after: GPUPeriodRows;
  /** Compared column. Defaults to `'count'`. */
  measure?: GPUCellTableCompareMeasure;
  /** Score written to `zScore`. */
  zScore?: GPUCellTableCompareZScore;
  /** Caller-owned comparison columns. `cells`, `delta`, `count` and `overflow` are always produced. */
  output?: Partial<GPUCellTableCompareOutput>;
  /** Classified column. Defaults to `'delta'`. */
  classify?: GPUPeriodComparisonVariable;
  /**
   * `getGPUClassBreaksParameterValues` view. A diverging scheme uses edges symmetric about zero
   * (`method: 'custom'`) or `'standard-deviation'`, whose classes are centred on the mean.
   */
  classBreaksParameters: GraphDataView<'float32'>;
  /** Compile-time class capacity. */
  maximumClassCount: number;
  /** Compiled class-break methods. */
  methods?: readonly (
    | 'equal-interval'
    | 'quantile'
    | 'standard-deviation'
    | 'head-tail'
    | 'box-plot'
    | 'maximum-breaks'
    | 'natural-breaks'
    | 'custom'
  )[];
  /** Packed rgba8 palette, typically a diverging ramp. */
  palette: GraphDataView<'uint32'>;
  /** `getGPUColorScaleParameterValues` view (`'threshold'` or `'quantile'` scale reads the breaks). */
  colorScaleParameters: GraphDataView<'float32'>;
  /** Compile-time palette bound. */
  maximumPaletteCount: number;
  /** Caller-owned class edges (`maximumClassCount + 1` rows). */
  breaks?: GraphDataView<'float32'>;
  /** Caller-owned class count. */
  classCount?: GraphDataView<'uint32'>;
  /** Caller-owned rgba8 color per union row (`unionCapacity` rows). */
  colors?: GraphDataView<'uint32'>;
  /** Caller-owned palette class per union row. */
  classIndices?: GraphDataView<'uint32'>;
};

/** Named outputs of {@link addPeriodComparisonRecipe}. */
export type GPUPeriodComparisonRecipeResult = GPURecipeResult & {
  unionCapacity: number;
  before: GPUCellTable;
  after: GPUCellTable;
  /** Union table: cells ascending, then the compare columns; `count` rows are occupied. */
  comparison: GPUCellTableCompareOutput;
  /** 1 for occupied union rows, 0 for the empty tail. */
  mask: GraphDataView<'uint32'>;
  breaks: GraphDataView<'float32'>;
  classCount: GraphDataView<'uint32'>;
  colors: GraphDataView<'uint32'>;
  classIndices: GraphDataView<'uint32'>;
};

/**
 * Period comparison recipe: two periods of events to a classified, colored change map.
 *
 * Chain: two `GPUCellAggregation` tables -> `GPUCellTableCompare` (outer join with delta, ratio,
 * percent change and z-score) -> `GPUClassBreaks` of the chosen column -> `GPUColorScale`. An
 * adapter kernel masks the unused tail of the capacity-bounded union table so it never reaches
 * the classification (the compare columns of the tail are undefined).
 */
export function addPeriodComparisonRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPeriodComparisonRecipeProps
): GPUPeriodComparisonRecipeResult {
  const id = props.id ?? 'period-comparison';
  const builder = new RecipeBuilder(graph);
  const {tableCapacity} = props;
  const unionCapacity = props.unionCapacity ?? 2 * tableCapacity;
  const measure = props.measure ?? 'count';
  const classify = props.classify ?? 'delta';
  assertRecipe(ID, tableCapacity >= 1, 'tableCapacity must be positive');

  const tables = (['before', 'after'] as const).map(period => {
    const rows = props[period];
    assertRecipe(
      ID,
      Boolean(rows.positions) !== Boolean(rows.cells),
      `${period} needs exactly one of positions or cells`
    );
    assertRecipe(
      ID,
      measure === 'count' || Boolean(rows.values),
      `${period} needs values for the 'sum' measure`
    );
    const provided = rows.table ?? {};
    const table: GPUCellTable = {
      cells: getOrCreateView(
        graph,
        `${id}-${period}-cells`,
        'uint32x2',
        tableCapacity,
        provided.cells
      ),
      counts: getOrCreateView(
        graph,
        `${id}-${period}-counts`,
        'uint32',
        tableCapacity,
        provided.counts
      ),
      sums: rows.values
        ? getOrCreateView(graph, `${id}-${period}-sums`, 'uint32x2', tableCapacity, provided.sums)
        : provided.sums,
      count: getOrCreateView(graph, `${id}-${period}-count`, 'uint32', 1, provided.count),
      overflow: getOrCreateView(graph, `${id}-${period}-overflow`, 'uint32', 1, provided.overflow)
    };
    builder.add(
      new GPUCellAggregation({
        id: `${id}-${period}`,
        family: props.family,
        resolution: props.resolution,
        positions: rows.positions,
        cells: rows.cells,
        values: rows.values,
        mask: rows.mask,
        output: table
      })
    );
    return table;
  });
  const [before, after] = tables;

  const given = props.output ?? {};
  const comparison: GPUCellTableCompareOutput = {
    cells: getOrCreateView(graph, `${id}-union-cells`, 'uint32x2', unionCapacity, given.cells),
    presence: given.presence,
    before: given.before,
    after: given.after,
    delta: getOrCreateView(graph, `${id}-delta`, 'float32', unionCapacity, given.delta),
    ratio: given.ratio,
    percentChange: given.percentChange,
    zScore: given.zScore,
    count: getOrCreateView(graph, `${id}-union-count`, 'uint32', 1, given.count),
    overflow: getOrCreateView(graph, `${id}-union-overflow`, 'uint32', 1, given.overflow),
    totalCount: given.totalCount
  };
  if (classify === 'ratio') {
    comparison.ratio = getOrCreateView(graph, `${id}-ratio`, 'float32', unionCapacity, given.ratio);
  } else if (classify === 'percentChange') {
    comparison.percentChange = getOrCreateView(
      graph,
      `${id}-percent-change`,
      'float32',
      unionCapacity,
      given.percentChange
    );
  } else if (classify === 'zScore') {
    comparison.zScore = getOrCreateView(
      graph,
      `${id}-z-score`,
      'float32',
      unionCapacity,
      given.zScore
    );
  }
  builder.add(
    new GPUCellTableCompare({
      id: `${id}-compare`,
      before,
      after,
      measure,
      zScore: props.zScore,
      output: comparison
    })
  );

  const mask = createTransientView(graph, `${id}-union-mask`, 'uint32', unionCapacity);
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-union-mask`,
      operation: 'GPUPeriodComparisonMask',
      bindings: [
        {name: 'unionCount', view: comparison.count, type: 'u32', access: 'read'},
        {name: 'maskOutput', view: mask, type: 'u32', access: 'read_write'}
      ],
      invocationCount: unionCapacity,
      body: 'maskOutput[maskOutputOffset + index] = select(0u, 1u, index < unionCount[unionCountOffset]);'
    })
  );

  const classified = comparison[classify]!;
  const breaks = getOrCreateView(
    graph,
    `${id}-breaks`,
    'float32',
    props.maximumClassCount + 1,
    props.breaks
  );
  const classCount = getOrCreateView(graph, `${id}-class-count`, 'uint32', 1, props.classCount);
  const colors = getOrCreateView(graph, `${id}-colors`, 'uint32', unionCapacity, props.colors);
  const classIndices = getOrCreateView(
    graph,
    `${id}-class-indices`,
    'uint32',
    unionCapacity,
    props.classIndices
  );
  builder.add(
    new GPUClassBreaks({
      id: `${id}-class-breaks`,
      values: classified,
      mask,
      parameters: props.classBreaksParameters,
      maximumClassCount: props.maximumClassCount,
      methods: props.methods,
      output: {breaks, classCount}
    })
  );
  builder.add(
    new GPUColorScale({
      id: `${id}-color-scale`,
      values: classified,
      mask,
      domain: breaks,
      domainCount: classCount,
      palette: props.palette,
      parameters: props.colorScaleParameters,
      maximumDomainCount: props.maximumClassCount + 1,
      maximumPaletteCount: props.maximumPaletteCount,
      output: {colors, classIndices}
    })
  );

  return {
    contributors: builder.contributors,
    unionCapacity,
    before,
    after,
    comparison,
    mask,
    breaks,
    classCount,
    colors,
    classIndices
  };
}
