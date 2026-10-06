// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUPolygonRasterization} from '../../gpu-raster/polygon-rasterization/index';
import {GPUArealInterpolation} from '../areal-interpolation/index';
import type {
  GPUArealCategories,
  GPUArealInterpolationDenominator
} from '../areal-interpolation/index';
import {GPUSpatialLag} from '../spatial-weights/index';
import type {GPUSpatialWeights} from '../spatial-weights/index';
import {
  assertRecipe,
  createTransientSpatialWeights,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUChangeOfSupportRecipe';

/** One zone system as polygon features (GeoArrow layout of `GPUPolygonRasterization`). */
export type GPUChangeOfSupportZones = {
  /** Flattened polygon vertices. */
  polygonPositions: GraphDataView<'float32x2'>;
  /** Feature-to-polygon offsets, `zoneCount + 1` entries. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Capacity of the rasterizer's scanline crossing list. */
  crossingCapacity: number;
  /** Caller-owned zone raster (`width * height` rows); graph-owned when absent. */
  raster?: GraphDataView<'uint32'>;
};

/** Properties for {@link addChangeOfSupportRecipe}. */
export type GPUChangeOfSupportRecipeProps = {
  /** Prefix for every node and transient ID. Defaults to `'change-of-support-recipe'`. */
  id?: string;
  /** Common raster columns. */
  width: number;
  /** Common raster rows. */
  height: number;
  /** `getGPUPolygonRasterizationExtentValues` view: origin and cell size of the raster. */
  extent: GraphDataView<'float32'>;
  /** Source zone system. */
  source: GPUChangeOfSupportZones;
  /** Target zone system. */
  target: GPUChangeOfSupportZones;
  /** Optional dasymetric raster (`width * height`): area becomes a sum of these weights. */
  cellWeights?: GraphDataView<'float32'>;
  /** Normalization of the areas: whole zones (default) or overlap only. */
  denominator?: GPUArealInterpolationDenominator;
  /** Capacity of the (target, source) overlap pair list. */
  pairCapacity: number;
  /**
   * Source values, `sourceCount * columnCount` rows row-major. With neither `extensiveValues`
   * nor `intensiveValues` given, no lag nodes are added and only the weights are produced.
   */
  sourceValues?: GraphDataView<'float32'>;
  /** Values per source zone. Defaults to 1. */
  columnCount?: number;
  /** Caller-owned extensive transfer (`targetCount * columnCount`): source mass split by area. */
  extensiveValues?: GraphDataView<'float32'>;
  /** Caller-owned intensive transfer (`targetCount * columnCount`): area-weighted source mean. */
  intensiveValues?: GraphDataView<'float32'>;
  /** Optional categorical shares of the sources per target. */
  categories?: GPUArealCategories;
  /** Caller-owned extensive weights CSR (targets by sources). */
  extensiveWeights?: Partial<GPUSpatialWeights>;
  /** Caller-owned intensive weights values (one per pair slot). */
  intensiveWeightValues?: GraphDataView<'float32'>;
  /** Caller-owned raw overlap area per pair slot. */
  areas?: GraphDataView<'float32'>;
  /** Caller-owned one-row flag: 1 when any rasterization or the pair list overflowed. */
  overflow?: GraphDataView<'uint32'>;
  /** Caller-owned one-row unclamped pair count. */
  totalPairs?: GraphDataView<'uint32'>;
};

/** Named outputs of {@link addChangeOfSupportRecipe}. */
export type GPUChangeOfSupportRecipeResult = GPURecipeResult & {
  sourceCount: number;
  targetCount: number;
  sourceZones: GraphDataView<'uint32'>;
  targetZones: GraphDataView<'uint32'>;
  /** Extensive weights `a_st / A_s`: rows are targets, neighbors are sources. */
  extensiveWeights: GPUSpatialWeights;
  /** Intensive weights `a_st / B_t` on the same pattern. */
  intensiveWeights: GPUSpatialWeights;
  areas: GraphDataView<'float32'>;
  /** One-row flag: 1 when the overlap pair list overflowed `pairCapacity`. */
  overflow: GraphDataView<'uint32'>;
  /** One-row flags of the source and target rasterizations (1: crossing list overflowed). */
  rasterOverflow: [GraphDataView<'uint32'>, GraphDataView<'uint32'>];
  totalPairs: GraphDataView<'uint32'>;
  extensiveValues?: GraphDataView<'float32'>;
  intensiveValues?: GraphDataView<'float32'>;
};

/**
 * Change of support recipe: transfer values between two polygon zone systems by area.
 *
 * Chain: `GPUPolygonRasterization` (source and target to one raster) -> `GPUArealInterpolation`
 * (overlap areas, extensive and intensive weights) -> `GPUSpatialLag` over each weights set (the
 * lag treats the target-by-source weights as a cross matrix). Rasterization overflow flags are
 * reported separately in `rasterOverflow`.
 *
 * Accuracy is bounded by the raster resolution (cell centers decide membership).
 */
export function addChangeOfSupportRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUChangeOfSupportRecipeProps
): GPUChangeOfSupportRecipeResult {
  const id = props.id ?? 'change-of-support-recipe';
  const cellCount = props.width * props.height;
  const sourceCount = props.source.featureOffsets.length - 1;
  const targetCount = props.target.featureOffsets.length - 1;
  assertRecipe(ID, sourceCount >= 1 && targetCount >= 1, 'needs source and target zones');
  const columnCount = props.columnCount ?? 1;
  const builder = new RecipeBuilder(graph);

  const rasterOverflow: [GraphDataView<'uint32'>, GraphDataView<'uint32'>] = [
    getOrCreateView(graph, `${id}-source-raster-overflow`, 'uint32', 1),
    getOrCreateView(graph, `${id}-target-raster-overflow`, 'uint32', 1)
  ];
  const zones = [props.source, props.target].map((system, index) => {
    const name = index === 0 ? 'source' : 'target';
    const raster = getOrCreateView(
      graph,
      `${id}-${name}-zones`,
      'uint32',
      cellCount,
      system.raster
    );
    builder.add(
      new GPUPolygonRasterization({
        id: `${id}-${name}-raster`,
        width: props.width,
        height: props.height,
        extent: props.extent,
        polygonPositions: system.polygonPositions,
        featureOffsets: system.featureOffsets,
        polygonOffsets: system.polygonOffsets,
        ringOffsets: system.ringOffsets,
        crossingCapacity: system.crossingCapacity,
        zones: raster,
        overflow: rasterOverflow[index]
      })
    );
    return raster;
  });

  const extensiveWeights = createTransientSpatialWeights(
    graph,
    `${id}-extensive`,
    targetCount,
    props.pairCapacity,
    props.extensiveWeights
  );
  const intensiveValues = getOrCreateView(
    graph,
    `${id}-intensive-weights`,
    'float32',
    props.pairCapacity,
    props.intensiveWeightValues
  );
  const areas = getOrCreateView(graph, `${id}-areas`, 'float32', props.pairCapacity, props.areas);
  const overflow = getOrCreateView(graph, `${id}-overflow`, 'uint32', 1, props.overflow);
  const totalPairs = getOrCreateView(graph, `${id}-total-pairs`, 'uint32', 1, props.totalPairs);
  builder.add(
    new GPUArealInterpolation({
      id: `${id}-areal`,
      sourceZones: zones[0],
      targetZones: zones[1],
      sourceCount,
      targetCount,
      cellWeights: props.cellWeights,
      denominator: props.denominator,
      mode: 'extensive',
      weights: extensiveWeights,
      alternateWeights: intensiveValues,
      areas,
      overflow,
      totalPairs,
      categories: props.categories
    })
  );
  const intensiveWeights: GPUSpatialWeights = {...extensiveWeights, weights: intensiveValues};

  const result: GPUChangeOfSupportRecipeResult = {
    contributors: builder.contributors,
    sourceCount,
    targetCount,
    sourceZones: zones[0],
    targetZones: zones[1],
    extensiveWeights,
    intensiveWeights,
    areas,
    overflow,
    totalPairs,
    rasterOverflow
  };
  if (props.sourceValues) {
    for (const kind of ['extensive', 'intensive'] as const) {
      const provided = kind === 'extensive' ? props.extensiveValues : props.intensiveValues;
      const output = getOrCreateView(
        graph,
        `${id}-${kind}-values`,
        'float32',
        targetCount * columnCount,
        provided
      );
      builder.add(
        new GPUSpatialLag({
          id: `${id}-${kind}-lag`,
          weights: kind === 'extensive' ? extensiveWeights : intensiveWeights,
          sourceCount,
          columnCount,
          values: props.sourceValues,
          output
        })
      );
      result[`${kind}Values`] = output;
    }
  }
  return result;
}
