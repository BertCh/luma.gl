// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUGroupStatistics} from '../../gpu-dataframe/group-statistics/index';
import type {GPUGroupStatisticsColumnOutput} from '../../gpu-dataframe/group-statistics/index';
import {GPUClassBreaks, GPUColorScale} from '../../gpu-dataframe/column-classification/index';
import type {GPUClassBreaksProps} from '../../gpu-dataframe/column-classification/index';
import {GPUPointInPolygonJoin} from '../spatial-join/index';
import {GPUZonalStatistics} from '../zonal-statistics/index';
import {
  assertRecipe,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUPointsInPolygonsChoroplethRecipe';

/** Statistic mapped to the choropleth. `'count'` needs no values. */
export type GPUChoroplethStatistic = 'count' | 'sum' | 'mean' | 'minimum' | 'maximum';

/** Polygon features, GeoArrow layout of `GPUPointInPolygonJoin`. */
export type GPUChoroplethPolygons = {
  /** Flattened polygon vertices. */
  polygonPositions: GraphDataView<'float32x2'>;
  /** Feature-to-polygon offsets, `featureCount + 1` entries. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Maximum bounding-box candidate pairs per encoding. */
  candidateCapacity: number;
  /** BVH leaf slots (power of two). */
  leafCapacity?: number;
  /** Count boundary points as contained. Defaults to true. */
  includeBoundary?: boolean;
};

/** Optional class breaks and colors of the per-feature statistic. */
export type GPUChoroplethColorOptions = {
  /** `getGPUClassBreaksParameterValues` view. */
  classBreaksParameters: GraphDataView<'float32'>;
  /** Compile-time class capacity. */
  maximumClassCount: number;
  /** Compiled class-break methods. */
  methods?: GPUClassBreaksProps['methods'];
  /** `getGPUColorScaleParameterValues` view (`quantile` or `threshold` scale for class colors). */
  colorScaleParameters: GraphDataView<'float32'>;
  /** Packed rgba8 palette. */
  palette: GraphDataView<'uint32'>;
  /** Compile-time palette length bound. */
  maximumPaletteCount: number;
};

/** Properties for {@link addPointsInPolygonsChoroplethRecipe}. */
export type GPUPointsInPolygonsChoroplethRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    counts?: GraphDataView<'uint32'>;
    featureValues?: GraphDataView<'float32'>;
    overflow?: GraphDataView<'uint32'>;
    color?: {
      breaks?: GraphDataView<'float32'>;
      classCount?: GraphDataView<'uint32'>;
      colors?: GraphDataView<'uint32'>;
      classIndices?: GraphDataView<'uint32'>;
    };
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'choropleth-recipe'`. */
  id?: string;
  /**
   * Aggregation back-end. `'zonal'` (default) joins and reduces inside `GPUZonalStatistics`;
   * `'group'` runs `GPUPointInPolygonJoin` then `GPUGroupStatistics` in dense mode (`keyCount` is
   * the feature count), so its tables already hold one row per feature.
   */
  backend?: 'zonal' | 'group';
  /** Planar points. */
  points: GraphDataView<'float32x2'>;
  /** Per-point values. Required unless `statistic` is `'count'`. */
  values?: GraphDataView<'float32'>;
  /** Polygon features. */
  polygons: GPUChoroplethPolygons;
  /** Statistic to map. Defaults to `'count'`. */
  statistic?: GPUChoroplethStatistic;
  /** Class breaks and colors; skipped when absent. */
  color?: GPUChoroplethColorOptions;
};

/** Named outputs of {@link addPointsInPolygonsChoroplethRecipe}. */
export type GPUPointsInPolygonsChoroplethRecipeResult = GPURecipeResult & {
  /** Number of polygon features, the row count of every per-feature output. */
  featureCount: number;
  /** Points per feature. */
  counts: GraphDataView<'uint32'>;
  /** The mapped statistic per feature: the `uint32` `counts` view for `'count'`, else float32. */
  featureValues: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** One-row overflow flag of the join. */
  overflow: GraphDataView<'uint32'>;
  /** Present when `color` was requested. */
  color?: {
    breaks: GraphDataView<'float32'>;
    classCount: GraphDataView<'uint32'>;
    colors: GraphDataView<'uint32'>;
    classIndices: GraphDataView<'uint32'>;
  };
};

/**
 * Points-in-polygons choropleth recipe: points to a per-polygon statistic to class breaks to
 * colors.
 *
 * Chain: `GPUZonalStatistics` (or `GPUPointInPolygonJoin` -> dense `GPUGroupStatistics`) ->
 * `GPUClassBreaks` -> `GPUColorScale`. Features with no points keep NaN (mean, minimum, maximum)
 * or 0 (count, sum); NaN rows are no-data in the color scale and are skipped by the class breaks.
 * Counts feed the classification as the `uint32` column they are, with no cast.
 */
export function addPointsInPolygonsChoroplethRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPointsInPolygonsChoroplethRecipeProps
): GPUPointsInPolygonsChoroplethRecipeResult {
  const id = props.id ?? 'choropleth-recipe';
  const statistic = props.statistic ?? 'count';
  const backend = props.backend ?? 'zonal';
  const providedOutputs = props.outputs ?? {};
  const {polygons, points} = props;
  const featureCount = polygons.featureOffsets.length - 1;
  assertRecipe(ID, featureCount >= 1, 'needs at least one feature');
  assertRecipe(ID, statistic === 'count' || props.values, `statistic ${statistic} needs values`);
  const builder = new RecipeBuilder(graph);
  const counts = getOrCreateView(
    graph,
    `${id}-counts`,
    'uint32',
    featureCount,
    providedOutputs.counts
  );
  assertRecipe(
    ID,
    statistic !== 'count' || !providedOutputs.featureValues,
    'count has no featureValues'
  );
  const featureValues: GraphDataView<'float32'> | GraphDataView<'uint32'> =
    statistic === 'count'
      ? counts
      : getOrCreateView(
          graph,
          `${id}-feature-values`,
          'float32',
          featureCount,
          providedOutputs.featureValues
        );
  const overflow = getOrCreateView(graph, `${id}-overflow`, 'uint32', 1, providedOutputs.overflow);
  let groupCount: GraphDataView<'uint32'> | undefined;
  let groupOverflow: GraphDataView<'uint32'> | undefined;

  if (backend === 'zonal') {
    const statisticOutput = {
      sum: 'sums',
      mean: 'means',
      minimum: 'minima',
      maximum: 'maxima'
    } as const;
    builder.add(
      new GPUZonalStatistics({
        id: `${id}-zonal`,
        features: {kind: 'polygons', ...polygons},
        points,
        values: statistic === 'count' ? undefined : props.values,
        output: {
          counts,
          overflow,
          ...(statistic === 'count' ? {} : {[statisticOutput[statistic]]: featureValues})
        }
      })
    );
  } else {
    const pointFeatureRows = getOrCreateView(
      graph,
      `${id}-point-features`,
      'uint32',
      points.length
    );
    builder.add(
      new GPUPointInPolygonJoin({
        id: `${id}-join`,
        points,
        ...polygons,
        pointFeatureIds: pointFeatureRows,
        overflow
      })
    );
    const groupKeys = getOrCreateView(graph, `${id}-group-keys`, 'uint32', featureCount);
    groupCount = getOrCreateView(graph, `${id}-group-count`, 'uint32', 1);
    groupOverflow = getOrCreateView(graph, `${id}-group-overflow`, 'uint32', 1);
    const columnOutput = {
      sum: {sumValues: featureValues},
      mean: {means: featureValues},
      minimum: {minimums: featureValues},
      maximum: {maximums: featureValues}
    } as const;
    builder.add(
      new GPUGroupStatistics({
        id: `${id}-group-statistics`,
        keys: pointFeatureRows,
        keyCount: featureCount,
        columns:
          statistic === 'count'
            ? []
            : [
                {
                  values: props.values!,
                  statistics: [statistic],
                  output: columnOutput[statistic] as GPUGroupStatisticsColumnOutput
                }
              ],
        output: {keys: groupKeys, counts, count: groupCount, overflow: groupOverflow}
      })
    );
  }

  const result: GPUPointsInPolygonsChoroplethRecipeResult = {
    contributors: builder.contributors,
    featureCount,
    counts,
    featureValues,
    overflow,
    outputs: {counts, featureValues},
    intermediates: {},
    status: {
      stages:
        backend === 'zonal'
          ? [{stage: 'zonal-statistics', status: {overflow}}]
          : [
              {stage: 'point-in-polygon', status: {overflow}},
              {stage: 'group-statistics', status: {count: groupCount!, overflow: groupOverflow!}}
            ]
    }
  };
  const {color} = props;
  if (color) {
    const providedColorOutputs = providedOutputs.color ?? {};
    const outputs = {
      breaks: getOrCreateView(
        graph,
        `${id}-breaks`,
        'float32',
        color.maximumClassCount + 1,
        providedColorOutputs.breaks
      ),
      classCount: getOrCreateView(
        graph,
        `${id}-class-count`,
        'uint32',
        1,
        providedColorOutputs.classCount
      ),
      colors: getOrCreateView(
        graph,
        `${id}-colors`,
        'uint32',
        featureCount,
        providedColorOutputs.colors
      ),
      classIndices: getOrCreateView(
        graph,
        `${id}-class-indices`,
        'uint32',
        featureCount,
        providedColorOutputs.classIndices
      )
    };
    builder.add(
      new GPUClassBreaks({
        id: `${id}-class-breaks`,
        values: featureValues,
        parameters: color.classBreaksParameters,
        maximumClassCount: color.maximumClassCount,
        methods: color.methods,
        output: {breaks: outputs.breaks, classCount: outputs.classCount}
      })
    );
    builder.add(
      new GPUColorScale({
        id: `${id}-color-scale`,
        values: featureValues,
        integerValues: 'numeric',
        domain: outputs.breaks,
        domainCount: outputs.classCount,
        palette: color.palette,
        parameters: color.colorScaleParameters,
        maximumDomainCount: color.maximumClassCount + 1,
        maximumPaletteCount: color.maximumPaletteCount,
        output: {colors: outputs.colors, classIndices: outputs.classIndices}
      })
    );
    result.color = outputs;
  }
  if (result.color) {
    result.outputs['color'] = result.color;
  }
  return result;
}
