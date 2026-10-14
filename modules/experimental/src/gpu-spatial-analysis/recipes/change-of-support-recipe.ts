// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUPolygonRasterization} from '../../gpu-raster/polygon-rasterization/index';
import {GPUArealInterpolation} from '../areal-interpolation/index';
import type {
  GPUArealCategories,
  GPUArealInterpolationDenominator,
  GPUArealInterpolationDenominators
} from '../areal-interpolation/index';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../contracts/spatial-context';
import {GPUSpatialLag} from '../spatial-weights/index';
import type {GPUSpatialWeights} from '../spatial-weights/index';
import {
  assertRecipe,
  createTransientSpatialWeights,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUChangeOfSupportRecipe';
/** Planar rasters, or longitude/latitude rasters weighted by spherical cell area. */
const SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  metrics: Object.freeze(['none', 'native', 'great-circle'] as const)
});

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
};

/** Properties for {@link addChangeOfSupportRecipe}. */
export type GPUChangeOfSupportRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    extensiveWeights?: Partial<GPUSpatialWeights>;
    intensiveWeightValues?: GraphDataView<'float32'>;
    areas?: GraphDataView<'float32'>;
    overflow?: GraphDataView<'uint32'>;
    requiredCount?: GraphDataView<'uint32'>;
    rasterOverflow?: [GraphDataView<'uint32'>, GraphDataView<'uint32'>];
    extensiveValues?: GraphDataView<'float32'>;
    intensiveValues?: GraphDataView<'float32'>;
    sourceAreas?: GraphDataView<'float32'>;
    sourceTransferredFractions?: GraphDataView<'float32'>;
    invalidCount?: GraphDataView<'uint32'>;
  },
  {
    sourceZones?: GraphDataView<'uint32'>;
    targetZones?: GraphDataView<'uint32'>;
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'change-of-support-recipe'`. */
  id?: string;
  /** Common raster columns. */
  width: number;
  /** Common raster rows. */
  height: number;
  /**
   * Coordinate space of the raster and the polygons. Omitted means native planar (cell counts are
   * areas). `longitude-latitude` (metric `'none'` or `'great-circle'`) declares a longitude/latitude
   * raster whose `extent` is in degrees: areas become spherical cell areas in square meters, so
   * the weights are ground-area shares. The polygon rasterization stage itself only tests cell
   * centers in the coordinates as given (straight edges in degrees; split polygons at the
   * antimeridian). The resolved context is returned as `result.spatialContext`.
   */
  spatialContext?: GPUSpatialContext;
  /** `getGPUPolygonRasterizationExtentValues` view: origin and cell size of the raster. */
  extent: GraphDataView<'float32'>;
  /** Source zone system. */
  source: GPUChangeOfSupportZones;
  /** Target zone system. */
  target: GPUChangeOfSupportZones;
  /** Optional dasymetric raster (`width * height`): area becomes a sum of these weights. */
  cellWeights?: GraphDataView<'float32'>;
  /**
   * Normalization of the areas, for both kinds or per kind. Defaults to extensive `'zone'` and
   * intensive `'overlap'`; see `GPUArealInterpolationProps.denominator`.
   */
  denominator?: GPUArealInterpolationDenominator | GPUArealInterpolationDenominators;
  /**
   * Optional per-source fallback raster cell for sources that rasterize to no cell (see
   * `GPUArealInterpolationProps.sourceAnchorCells`).
   */
  sourceAnchorCells?: GraphDataView<'uint32'>;
  /** Capacity of the (target, source) overlap pair list. */
  pairCapacity: number;
  /**
   * Source values, `sourceCount * columnCount` rows row-major. With neither `extensiveValues`
   * nor `intensiveValues` given, no lag nodes are added and only the weights are produced.
   */
  sourceValues?: GraphDataView<'float32'>;
  /** Values per source zone. Defaults to 1. */
  columnCount?: number;
  /** Optional categorical shares of the sources per target. */
  categories?: GPUArealCategories;
};

/** Named outputs of {@link addChangeOfSupportRecipe}. */
export type GPUChangeOfSupportRecipeResult = GPURecipeResult & {
  /** Resolved coordinate context of the raster. */
  spatialContext: GPUSpatialContext;
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
  /** Raster area of every source (cells, or square meters on a longitude/latitude raster). */
  sourceAreas: GraphDataView<'float32'>;
  /**
   * Fraction of each source's area that reached a target (the conserved fraction of its extensive
   * mass with the default denominator), NaN for sources without raster area.
   */
  sourceTransferredFractions: GraphDataView<'float32'>;
  /** One-row count of sources with no raster area: their mass is lost. */
  invalidCount: GraphDataView<'uint32'>;
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
 * Accuracy is bounded by the raster resolution (cell centers decide membership). Mass that a
 * source loses (it lies partly outside the target system, or it is smaller than a cell) is
 * reported per source in `sourceTransferredFractions` and `sourceAreas`, and sources with no
 * raster area are counted in `invalidCount` (stage `areal-interpolation`), unless
 * `sourceAnchorCells` rescues them. Default change: intensive weights now use the `'overlap'`
 * denominator, so a target only partly covered by sources is the mean of what covers it.
 *
 * Spatial context: `spatialContext` is resolved once and handed to the areal stage; see
 * `GPUChangeOfSupportRecipeProps.spatialContext`.
 */
export function addChangeOfSupportRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUChangeOfSupportRecipeProps
): GPUChangeOfSupportRecipeResult {
  const id = props.id ?? 'change-of-support-recipe';
  const spatialContext = resolveGPUSpatialContext(id, props.spatialContext, SUPPORT);
  const geographic = spatialContext.coordinateSpace === 'longitude-latitude';
  const cellCount = props.width * props.height;
  const sourceCount = props.source.featureOffsets.length - 1;
  const targetCount = props.target.featureOffsets.length - 1;
  assertRecipe(ID, sourceCount >= 1 && targetCount >= 1, 'needs source and target zones');
  const columnCount = props.columnCount ?? 1;
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  const builder = new RecipeBuilder(graph);

  const rasterOverflow: [GraphDataView<'uint32'>, GraphDataView<'uint32'>] = [
    getOrCreateView(
      graph,
      `${id}-source-raster-overflow`,
      'uint32',
      1,
      outputs.rasterOverflow?.[0]
    ),
    getOrCreateView(graph, `${id}-target-raster-overflow`, 'uint32', 1, outputs.rasterOverflow?.[1])
  ];
  const zones = [props.source, props.target].map((system, index) => {
    const name = index === 0 ? 'source' : 'target';
    const raster = getOrCreateView(
      graph,
      `${id}-${name}-zones`,
      'uint32',
      cellCount,
      index === 0 ? scratch.sourceZones : scratch.targetZones
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
    outputs.extensiveWeights
  );
  const intensiveValues = getOrCreateView(
    graph,
    `${id}-intensive-weights`,
    'float32',
    props.pairCapacity,
    outputs.intensiveWeightValues
  );
  const areas = getOrCreateView(graph, `${id}-areas`, 'float32', props.pairCapacity, outputs.areas);
  const overflow = getOrCreateView(graph, `${id}-overflow`, 'uint32', 1, outputs.overflow);
  const totalPairs = getOrCreateView(
    graph,
    `${id}-required-count`,
    'uint32',
    1,
    outputs.requiredCount
  );
  const sourceAreas = getOrCreateView(
    graph,
    `${id}-source-areas`,
    'float32',
    sourceCount,
    outputs.sourceAreas
  );
  const sourceTransferredFractions = getOrCreateView(
    graph,
    `${id}-source-transferred-fractions`,
    'float32',
    sourceCount,
    outputs.sourceTransferredFractions
  );
  const invalidCount = getOrCreateView(
    graph,
    `${id}-invalid-count`,
    'uint32',
    1,
    outputs.invalidCount
  );
  builder.add(
    new GPUArealInterpolation({
      id: `${id}-areal`,
      spatialContext,
      ...(geographic ? {extent: props.extent, width: props.width} : {}),
      sourceAnchorCells: props.sourceAnchorCells,
      sourceAreas,
      sourceTransferredFractions,
      invalidCount,
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
    spatialContext,
    sourceCount,
    targetCount,
    sourceZones: zones[0],
    targetZones: zones[1],
    extensiveWeights,
    intensiveWeights,
    areas,
    overflow,
    totalPairs,
    sourceAreas,
    sourceTransferredFractions,
    invalidCount,
    rasterOverflow,
    outputs: {
      extensiveWeights,
      intensiveWeights,
      areas,
      sourceAreas,
      sourceTransferredFractions,
      invalidCount
    },
    intermediates: {sourceZones: zones[0], targetZones: zones[1]},
    status: {
      stages: [
        {stage: 'source-rasterization', status: {overflow: rasterOverflow[0]}},
        {stage: 'target-rasterization', status: {overflow: rasterOverflow[1]}},
        {
          stage: 'areal-interpolation',
          status: {overflow, requiredCount: totalPairs, invalidCount}
        }
      ]
    }
  };
  if (props.sourceValues) {
    for (const kind of ['extensive', 'intensive'] as const) {
      const provided = kind === 'extensive' ? outputs.extensiveValues : outputs.intensiveValues;
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
      result.outputs[`${kind}Values`] = output;
    }
  }
  return result;
}
