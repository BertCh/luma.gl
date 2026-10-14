// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUGroupStatistics} from '../../gpu-dataframe/group-statistics/index';
import type {GPUGroupStatisticsColumnOutput} from '../../gpu-dataframe/group-statistics/index';
import {GPUClassBreaks, GPUColorScale} from '../../gpu-dataframe/column-classification/index';
import type {GPUClassBreaksProps} from '../../gpu-dataframe/column-classification/index';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPUPackedGeometryPort,
  GPUPointGeometryPort,
  GPUPolygonGeometryPort,
  GPURecipeStatusPort,
  GPUStatusScalar
} from '../../analysis-core/contracts/index';
import {GPUPointInPolygonJoin} from '../spatial-join/index';
import {
  getPointInPolygonContainmentContext,
  type GPUPointInPolygonBoundaryRule
} from '../spatial-join/gpu-point-in-polygon-join';
import {GPUZonalStatistics} from '../zonal-statistics/index';
import {
  assertRecipe,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUPointsInPolygonsChoroplethRecipe';
/** Planar points and polygons, or longitude/latitude degrees (topological containment only). */
const SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const)
});

/** Statistic mapped to the choropleth. `'count'` needs no values. */
export type GPUChoroplethStatistic = 'count' | 'sum' | 'mean' | 'minimum' | 'maximum';

/**
 * Polygon features in the four-level `GPUPolygonGeometryPort` layout (one packed `positions`
 * view), one output row per feature, plus the join options. `sourceIds` are not read: every
 * per-feature output is indexed by feature row.
 */
export type GPUChoroplethPolygons = GPUPackedGeometryPort<GPUPolygonGeometryPort> & {
  /** Maximum bounding-box candidate pairs per encoding. */
  candidateCapacity: number;
  /** BVH leaf slots (power of two). */
  leafCapacity?: number;
  /**
   * Ownership of points exactly on a boundary, see `GPUPointInPolygonJoinProps.boundaryRule`.
   * Defaults to `'half-open'`: a point on an edge shared by two features counts once.
   */
  boundaryRule?: GPUPointInPolygonBoundaryRule;
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
    /** Mirrors `result.status`. */
    status?: {
      overflow?: GPUStatusScalar;
      /** One-row count of uncertain point-in-polygon pairs (stage `zonal-statistics` or `point-in-polygon`). */
      uncertainCount?: GPUStatusScalar;
    };
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
  /**
   * Coordinate context of the points and polygons. Omitted means native planar. The mapped
   * statistics (count, sum, mean, minimum, maximum) are topological, so a `longitude-latitude`
   * context is accepted: containment is tested with straight polygon edges in degrees (not
   * geodesics) and polygons crossing the antimeridian must be split. The recipe maps no area
   * statistic (no density), so no spherical area is needed. The resolved context is returned as
   * `result.spatialContext`.
   */
  spatialContext?: GPUSpatialContext;
  /** Points in the coordinates of `spatialContext` (planar unless declared otherwise). */
  points: GPUPointGeometryPort;
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
  /** Resolved coordinate context of the points and polygons. */
  spatialContext: GPUSpatialContext;
  /** Number of polygon features, the row count of every per-feature output. */
  featureCount: number;
  /** Points per feature. */
  counts: GraphDataView<'uint32'>;
  /** The mapped statistic per feature: the `uint32` `counts` view for `'count'`, else float32. */
  featureValues: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Present when `color` was requested. */
  color?: {
    breaks: GraphDataView<'float32'>;
    classCount: GraphDataView<'uint32'>;
    colors: GraphDataView<'uint32'>;
    classIndices: GraphDataView<'uint32'>;
  };
  /** Status words next to the per-stage `stages`. */
  status: {
    /** One-row overflow flag of the join. */
    overflow: GPUStatusScalar;
    /**
     * One-row count of (point, feature) candidates whose containment could not be proven; those
     * points stay unassigned, so a nonzero value means the statistics may be low.
     */
    uncertainCount: GPUStatusScalar;
  } & GPURecipeStatusPort;
};

/**
 * Points-in-polygons choropleth recipe: points to a per-polygon statistic to class breaks to
 * colors.
 *
 * Chain: `GPUZonalStatistics` (or `GPUPointInPolygonJoin` -> dense `GPUGroupStatistics`) ->
 * `GPUClassBreaks` -> `GPUColorScale`. Features with no points keep NaN (mean, minimum, maximum)
 * or 0 (count, sum); NaN rows are no-data in the color scale and are skipped by the class breaks.
 * Counts feed the classification as the `uint32` column they are, with no cast.
 *
 * Spatial context: planar by default; `longitude-latitude` is accepted because counts and value
 * statistics are topological (see `spatialContext`). Density is not mapped, so no polygon area is
 * involved.
 *
 * Status: stage `zonal-statistics` (overflow, uncertain count), or `point-in-polygon` (overflow,
 * uncertain count) and `group-statistics` (count, overflow, required count) with `backend: 'group'`.
 */
export function addPointsInPolygonsChoroplethRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPointsInPolygonsChoroplethRecipeProps
): GPUPointsInPolygonsChoroplethRecipeResult {
  const id = props.id ?? 'choropleth-recipe';
  const spatialContext = resolveGPUSpatialContext(id, props.spatialContext, SUPPORT);
  const statistic = props.statistic ?? 'count';
  const backend = props.backend ?? 'zonal';
  const providedOutputs = props.outputs ?? {};
  const {polygons, points} = props;
  // Rows are features: feature source IDs never index the outputs.
  const polygonRows = {...polygons, sourceIds: undefined};
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
  const overflow = getOrCreateView(
    graph,
    `${id}-overflow`,
    'uint32',
    1,
    providedOutputs.status?.overflow
  );
  const uncertainCount = getOrCreateView(
    graph,
    `${id}-uncertain-count`,
    'uint32',
    1,
    providedOutputs.status?.uncertainCount
  );
  let groupCount: GraphDataView<'uint32'> | undefined;
  let groupRequiredCount: GraphDataView<'uint32'> | undefined;
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
        spatialContext,
        features: polygonRows,
        positions: points.positions,
        values: statistic === 'count' ? undefined : props.values,
        output: {
          counts,
          status: {overflow, uncertainCount},
          ...(statistic === 'count' ? {} : {[statisticOutput[statistic]]: featureValues})
        }
      })
    );
  } else {
    const pointFeatureRows = getOrCreateView(
      graph,
      `${id}-point-features`,
      'uint32',
      points.positions.length
    );
    builder.add(
      new GPUPointInPolygonJoin({
        id: `${id}-join`,
        spatialContext: getPointInPolygonContainmentContext(spatialContext),
        points: {...points, sourceIds: undefined},
        polygons: polygonRows,
        candidateCapacity: polygons.candidateCapacity,
        leafCapacity: polygons.leafCapacity,
        boundaryRule: polygons.boundaryRule,
        output: {pointFeatureIds: pointFeatureRows, status: {overflow, uncertainCount}}
      })
    );
    const groupKeys = getOrCreateView(graph, `${id}-group-keys`, 'uint32', featureCount);
    groupCount = getOrCreateView(graph, `${id}-group-count`, 'uint32', 1);
    groupOverflow = getOrCreateView(graph, `${id}-group-overflow`, 'uint32', 1);
    groupRequiredCount = getOrCreateView(graph, `${id}-group-required-count`, 'uint32', 1);
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
        output: {
          keys: groupKeys,
          counts,
          status: {count: groupCount, overflow: groupOverflow, requiredCount: groupRequiredCount}
        }
      })
    );
  }

  const result: GPUPointsInPolygonsChoroplethRecipeResult = {
    contributors: builder.contributors,
    spatialContext,
    featureCount,
    counts,
    featureValues,
    outputs: {counts, featureValues},
    intermediates: {},
    status: {
      overflow,
      uncertainCount,
      stages:
        backend === 'zonal'
          ? [{stage: 'zonal-statistics', status: {overflow, uncertainCount}}]
          : [
              {stage: 'point-in-polygon', status: {overflow, uncertainCount}},
              {
                stage: 'group-statistics',
                status: {
                  count: groupCount!,
                  overflow: groupOverflow!,
                  requiredCount: groupRequiredCount!
                }
              }
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
