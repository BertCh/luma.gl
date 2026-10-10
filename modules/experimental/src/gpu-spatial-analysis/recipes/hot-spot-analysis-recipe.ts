// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUClassBreaks, GPUColorScale} from '../../gpu-dataframe/column-classification/index';
import type {
  GPUClassBreaksProps,
  GPUColorScaleProps
} from '../../gpu-dataframe/column-classification/index';
import {GPUCellAggregation} from '../cell-aggregation/index';
import type {GPUCellFamily, GPUCellTable} from '../cell-aggregation/index';
import {GPUCellGeometry, GPUPointToCell} from '../cell-indexing/index';
import {GPUNeighborSearch} from '../neighbor-search/index';
import {GPULocalPermutationTest} from '../permutation-inference/index';
import type {
  GPULocalPermutationStatistic,
  GPUPermutationAlternative
} from '../permutation-inference/index';
import {GPUHotSpotAnalysis} from '../spatial-autocorrelation/index';
import {GPULatticeWeights} from '../spatial-weights/index';
import type {GPULatticeCriterion, GPUSpatialWeights} from '../spatial-weights/index';
import {
  addCellTableColumnsNode,
  assertRecipe,
  createTransientSpatialWeights,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUHotSpotAnalysisRecipe';

/** Input rows: points binned into a cell table, then linked by a radius search over cell centers. */
export type GPUHotSpotPointsSource = {
  kind: 'points';
  /** Longitude/latitude degrees per point. */
  positions: GraphDataView<'float32x2'>;
  /** Optional per-point value summed per cell; the cell count is analysed when absent. */
  values?: GraphDataView<'float32'>;
  /** Optional per-point mask. */
  mask?: GraphDataView<'uint32'>;
  /** Cell family. `'quadbin'` bins the points directly; `'h3'` goes through `GPUPointToCell`. */
  family: GPUCellFamily;
  /** Cell resolution. */
  resolution: number;
  /** Rows of the capacity-bounded cell table (and of every per-cell output). */
  tableCapacity: number;
  /** Slots of the neighbor CSR. */
  neighborCapacity: number;
  /** Search grid size passed to `GPUNeighborSearch` (`[columns, rows]`). */
  gridSize: readonly [number, number];
  /**
   * Per-frame `GPUNeighborSearch` parameters (`getGPUNeighborSearchParameterValues`): `bounds`
   * in cell-center units (degrees), `radius`, binary weights for classic Gi*.
   */
  neighborSearchParameters: GraphDataView<'float32'>;
};

/** Input rows: a dense raster of values on a regular lattice (row-major `y * width + x`). */
export type GPUHotSpotLatticeSource = {
  kind: 'lattice';
  /** One value per lattice cell, `width * height` rows. */
  values: GraphDataView<'float32'>;
  /** Optional per-cell mask. */
  mask?: GraphDataView<'uint32'>;
  /** Lattice columns. */
  width: number;
  /** Lattice rows. */
  height: number;
  /** Neighborhood shape. Defaults to `'queen'`. */
  criterion?: GPULatticeCriterion;
  /** Neighborhood radius in cells. Defaults to 1. */
  radius?: number;
  /** Slots of the neighbor CSR. */
  neighborCapacity: number;
};

/** Optional pseudo p-value confirmation of the Gi* hot spots. */
export type GPUHotSpotPermutationOptions = {
  /** `getGPUPermutationParameterValues` view. */
  parameters: GraphDataView<'uint32'>;
  /** Compile-time upper bound of the per-frame permutation count. */
  maximumPermutations: number;
  /** Local statistic. Defaults to `'localGStar'`. */
  statistic?: GPULocalPermutationStatistic;
  /** Tail of the pseudo p-value. */
  alternative?: GPUPermutationAlternative;
  /** Compile-time neighbor bound, up to 64. */
  maximumNeighbors?: number;
  /** Benjamini-Hochberg control of `significant`. */
  falseDiscoveryRate?: boolean;
};

/** Optional class breaks and colors of the Gi* z-scores. */
export type GPUHotSpotColorOptions = {
  /** `getGPUClassBreaksParameterValues` view. */
  classBreaksParameters: GraphDataView<'float32'>;
  /** Compile-time class capacity. */
  maximumClassCount: number;
  /** Compiled class-break methods. */
  methods?: GPUClassBreaksProps['methods'];
  /** `getGPUColorScaleParameterValues` view. */
  colorScaleParameters: GraphDataView<'float32'>;
  /** Packed rgba8 palette. */
  palette: GraphDataView<'uint32'>;
  /** Compile-time palette length bound. */
  maximumPaletteCount: number;
};

/** Properties for {@link addHotSpotAnalysisRecipe}. */
export type GPUHotSpotAnalysisRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    zScores?: GraphDataView<'float32'>;
    bins?: GraphDataView<'sint32'>;
    pValues?: GraphDataView<'float32'>;
    neighborCounts?: GraphDataView<'uint32'>;
    globalStatistics?: GraphDataView<'float32'>;
    weightsOverflow?: GraphDataView<'uint32'>;
    permutation?: {
      exceedances?: GraphDataView<'uint32'>;
      pseudoPValues?: GraphDataView<'float32'>;
      significant?: GraphDataView<'uint32'>;
      overflow?: GraphDataView<'uint32'>;
    };
    color?: {
      breaks?: GraphDataView<'float32'>;
      classCount?: GraphDataView<'uint32'>;
      colors?: GraphDataView<'uint32'>;
      classIndices?: GraphDataView<'uint32'>;
    };
  },
  {
    table?: Partial<GPUCellTable>;
    centers?: GraphDataView<'float32x2'>;
    values?: GraphDataView<'float32'>;
    mask?: GraphDataView<'uint32'>;
    weights?: Partial<GPUSpatialWeights>;
    pointKeys?: GraphDataView<'uint32x2'>;
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'hot-spot-recipe'`. */
  id?: string;
  /** Where the rows come from. */
  source: GPUHotSpotPointsSource | GPUHotSpotLatticeSource;
  /** `getGPUSpatialAutocorrelationParameterValues` view (significance level, optional moments). */
  parameters: GraphDataView<'float32'>;
  /** Weight of the focal row in Gi*. Defaults to 1 (classic Gi*). */
  selfWeight?: number;
  /** Benjamini-Hochberg false discovery rate correction of the Gi* bins. */
  falseDiscoveryRate?: boolean;
  /** Permutation confirmation; skipped when absent. */
  permutation?: GPUHotSpotPermutationOptions;
  /** Class breaks and colors of the z-scores; skipped when absent. */
  color?: GPUHotSpotColorOptions;
};

/** Named outputs of {@link addHotSpotAnalysisRecipe}. Views are caller-owned when passed in. */
export type GPUHotSpotAnalysisRecipeResult = GPURecipeResult & {
  /** Rows of every per-cell output. */
  rowCount: number;
  /** Cell table (points source only). */
  table?: GPUCellTable;
  /** Cell centers (points source only). */
  centers?: GraphDataView<'float32x2'>;
  /** Per-row analysis value (cell count or sum) and inclusion mask. */
  values: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  weights: GPUSpatialWeights;
  weightsOverflow: GraphDataView<'uint32'>;
  zScores: GraphDataView<'float32'>;
  bins: GraphDataView<'sint32'>;
  pValues: GraphDataView<'float32'>;
  neighborCounts: GraphDataView<'uint32'>;
  globalStatistics: GraphDataView<'float32'>;
  /** Present when `permutation` was requested. */
  permutation?: {
    exceedances: GraphDataView<'uint32'>;
    pseudoPValues: GraphDataView<'float32'>;
    significant: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
  };
  /** Present when `color` was requested. */
  color?: {
    breaks: GraphDataView<'float32'>;
    classCount: GraphDataView<'uint32'>;
    colors: GraphDataView<'uint32'>;
    classIndices: GraphDataView<'uint32'>;
  };
};

/**
 * Hot spot analysis recipe: points or a raster to Getis-Ord Gi* hot and cold spots, optionally
 * confirmed by conditional permutation and colored by class breaks.
 *
 * Chain (points source): `GPUPointToCell` (H3 only) -> `GPUCellAggregation` -> `GPUCellGeometry`
 * centers -> `GPUNeighborSearch` radius weights -> `GPUHotSpotAnalysis` ->
 * `GPULocalPermutationTest` -> `GPUClassBreaks` -> `GPUColorScale`. A lattice source replaces the
 * first three stages with `GPULatticeWeights` over the dense raster. An adapter kernel converts
 * the cell table into the value and mask columns (empty table rows are masked out).
 *
 * Rows are table rows (`tableCapacity`) or lattice cells. Everything except the output views is
 * graph-owned; pass caller-owned views for the results you read back.
 */
export function addHotSpotAnalysisRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUHotSpotAnalysisRecipeProps
): GPUHotSpotAnalysisRecipeResult {
  const id = props.id ?? 'hot-spot-recipe';
  const builder = new RecipeBuilder(graph);
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  const {source} = props;
  let rowCount: number;
  let values: GraphDataView<'float32'>;
  let mask: GraphDataView<'uint32'> | undefined;
  let table: GPUCellTable | undefined;
  let centers: GraphDataView<'float32x2'> | undefined;
  let weights: GPUSpatialWeights;
  const weightsOverflow = getOrCreateView(
    graph,
    `${id}-weights-overflow`,
    'uint32',
    1,
    outputs.weightsOverflow
  );

  if (source.kind === 'points') {
    rowCount = source.tableCapacity;
    assertRecipe(ID, rowCount >= 1, 'tableCapacity must be positive');
    const provided = scratch.table ?? {};
    table = {
      cells: getOrCreateView(graph, `${id}-cells`, 'uint32x2', rowCount, provided.cells),
      counts: getOrCreateView(graph, `${id}-counts`, 'uint32', rowCount, provided.counts),
      sumValues: source.values
        ? getOrCreateView(graph, `${id}-sum-values`, 'float32', rowCount, provided.sumValues)
        : provided.sumValues,
      count: getOrCreateView(graph, `${id}-count`, 'uint32', 1, provided.count),
      overflow: getOrCreateView(graph, `${id}-table-overflow`, 'uint32', 1, provided.overflow)
    };
    if (source.family === 'quadbin') {
      builder.add(
        new GPUCellAggregation({
          id: `${id}-aggregate`,
          family: 'quadbin',
          resolution: source.resolution,
          positions: source.positions,
          values: source.values,
          mask: source.mask,
          output: table
        })
      );
    } else {
      const keys = getOrCreateView(
        graph,
        `${id}-point-keys`,
        'uint32x2',
        source.positions.length,
        scratch.pointKeys
      );
      builder.add(
        new GPUPointToCell({
          id: `${id}-point-to-cell`,
          family: source.family,
          resolution: source.resolution,
          positions: source.positions,
          mask: source.mask,
          output: {cells: keys}
        })
      );
      builder.add(
        new GPUCellAggregation({
          id: `${id}-aggregate`,
          family: source.family,
          resolution: source.resolution,
          cells: keys,
          values: source.values,
          mask: source.mask,
          output: table
        })
      );
    }
    centers = getOrCreateView(graph, `${id}-centers`, 'float32x2', rowCount, scratch.centers);
    builder.add(
      new GPUCellGeometry({
        id: `${id}-geometry`,
        family: source.family,
        cells: table.cells,
        output: {centers}
      })
    );
    values = getOrCreateView(graph, `${id}-values`, 'float32', rowCount, scratch.values);
    mask = getOrCreateView(graph, `${id}-mask`, 'uint32', rowCount, scratch.mask);
    addCellTableColumnsNode(graph, id, table, {values, mask});
    weights = createTransientSpatialWeights(
      graph,
      `${id}-weights`,
      rowCount,
      source.neighborCapacity,
      scratch.weights
    );
    builder.add(
      new GPUNeighborSearch({
        id: `${id}-neighbors`,
        mode: 'radius',
        positions: centers,
        mask,
        parameters: source.neighborSearchParameters,
        gridSize: source.gridSize,
        weights,
        overflow: weightsOverflow
      })
    );
  } else {
    rowCount = source.width * source.height;
    values = source.values;
    mask = source.mask;
    weights = createTransientSpatialWeights(
      graph,
      `${id}-weights`,
      rowCount,
      source.neighborCapacity,
      scratch.weights
    );
    builder.add(
      new GPULatticeWeights({
        id: `${id}-lattice`,
        width: source.width,
        height: source.height,
        criterion: source.criterion ?? 'queen',
        radius: source.radius,
        mask,
        weights,
        overflow: weightsOverflow
      })
    );
  }

  const zScores = getOrCreateView(graph, `${id}-z-scores`, 'float32', rowCount, outputs.zScores);
  const bins = getOrCreateView(graph, `${id}-bins`, 'sint32', rowCount, outputs.bins);
  const pValues = getOrCreateView(graph, `${id}-p-values`, 'float32', rowCount, outputs.pValues);
  const neighborCounts = getOrCreateView(
    graph,
    `${id}-neighbor-counts`,
    'uint32',
    rowCount,
    outputs.neighborCounts
  );
  const globalStatistics = getOrCreateView(
    graph,
    `${id}-global-statistics`,
    'float32',
    4,
    outputs.globalStatistics
  );
  builder.add(
    new GPUHotSpotAnalysis({
      id: `${id}-gi-star`,
      weights,
      selfWeight: props.selfWeight,
      values,
      mask,
      parameters: props.parameters,
      falseDiscoveryRate: props.falseDiscoveryRate,
      zScores,
      bins,
      pValues,
      neighborCounts,
      globalStatistics
    })
  );

  const result: GPUHotSpotAnalysisRecipeResult = {
    contributors: builder.contributors,
    rowCount,
    table,
    centers,
    values,
    mask,
    weights,
    weightsOverflow,
    zScores,
    bins,
    pValues,
    neighborCounts,
    globalStatistics,
    outputs: {zScores, bins, pValues, neighborCounts, globalStatistics},
    intermediates: {table, centers, values, mask, weights},
    status: {
      stages: [
        ...(table
          ? [{stage: 'cell-aggregation', status: {count: table.count, overflow: table.overflow}}]
          : []),
        {stage: 'weights', status: {overflow: weightsOverflow}}
      ]
    }
  };

  const {permutation} = props;
  if (permutation) {
    const provided = outputs.permutation ?? {};
    const permutationOutputs = {
      exceedances: getOrCreateView(
        graph,
        `${id}-exceedances`,
        'uint32',
        rowCount,
        provided.exceedances
      ),
      pseudoPValues: getOrCreateView(
        graph,
        `${id}-pseudo-p-values`,
        'float32',
        rowCount,
        provided.pseudoPValues
      ),
      significant: getOrCreateView(
        graph,
        `${id}-significant`,
        'uint32',
        rowCount,
        provided.significant
      ),
      overflow: getOrCreateView(graph, `${id}-permutation-overflow`, 'uint32', 1, provided.overflow)
    };
    builder.add(
      new GPULocalPermutationTest({
        id: `${id}-permutation`,
        weights,
        values,
        mask,
        statistic: permutation.statistic ?? 'localGStar',
        alternative: permutation.alternative,
        parameters: permutation.parameters,
        maximumPermutations: permutation.maximumPermutations,
        maximumNeighbors: permutation.maximumNeighbors,
        falseDiscoveryRate: permutation.falseDiscoveryRate,
        ...permutationOutputs
      })
    );
    result.permutation = permutationOutputs;
    result.outputs['permutation'] = permutationOutputs;
    result.status.stages = [
      ...result.status.stages,
      {stage: 'permutation', status: {overflow: permutationOutputs.overflow}}
    ];
  }

  const {color} = props;
  if (color) {
    const provided = outputs.color ?? {};
    const colorOutputs = {
      breaks: getOrCreateView(
        graph,
        `${id}-breaks`,
        'float32',
        color.maximumClassCount + 1,
        provided.breaks
      ),
      classCount: getOrCreateView(graph, `${id}-class-count`, 'uint32', 1, provided.classCount),
      colors: getOrCreateView(graph, `${id}-colors`, 'uint32', rowCount, provided.colors),
      classIndices: getOrCreateView(
        graph,
        `${id}-class-indices`,
        'uint32',
        rowCount,
        provided.classIndices
      )
    };
    builder.add(
      new GPUClassBreaks({
        id: `${id}-class-breaks`,
        values: zScores,
        mask,
        parameters: color.classBreaksParameters,
        maximumClassCount: color.maximumClassCount,
        methods: color.methods,
        output: {breaks: colorOutputs.breaks, classCount: colorOutputs.classCount}
      })
    );
    const colorProps: GPUColorScaleProps = {
      id: `${id}-color-scale`,
      values: zScores,
      mask,
      domain: colorOutputs.breaks,
      domainCount: colorOutputs.classCount,
      palette: color.palette,
      parameters: color.colorScaleParameters,
      maximumDomainCount: color.maximumClassCount + 1,
      maximumPaletteCount: color.maximumPaletteCount,
      output: {colors: colorOutputs.colors, classIndices: colorOutputs.classIndices}
    };
    builder.add(new GPUColorScale(colorProps));
    result.color = colorOutputs;
    result.outputs['color'] = colorOutputs;
  }
  return result;
}
