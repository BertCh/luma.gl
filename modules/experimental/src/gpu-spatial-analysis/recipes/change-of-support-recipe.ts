// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUPolygonRasterization} from '../../gpu-raster/polygon-rasterization/index';
import type {GPURasterMetadata} from '../../gpu-raster/index';
import type {
  GPUPackedGeometryPort,
  GPURecipeStatusPort,
  GPUStatusScalar,
  GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';
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
} from '../../analysis-core/contracts/spatial-context';
import {GPUSpatialLag} from '../spatial-weights/index';
import type {GPUSpatialWeights, GPUSpatialWeightsOutput} from '../spatial-weights/index';
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

/**
 * One zone system as polygon features in the four-level `GPUPolygonGeometryPort` layout (one
 * packed `positions` view), one zone per feature. `sourceIds` are not read: zones are feature rows.
 */
export type GPUChangeOfSupportZones = GPUPackedGeometryPort<GPUPolygonGeometryPort> & {
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
    /** Mirrors `result.status`. */
    status?: {
      overflow?: GPUStatusScalar;
      requiredCount?: GPUStatusScalar;
      invalidCount?: GPUStatusScalar;
      rasterOverflow?: [GPUStatusScalar, GPUStatusScalar];
      /** One-row unclamped crossing counts of the source and target rasterizations. */
      rasterRequiredCount?: [GPUStatusScalar, GPUStatusScalar];
    };
    extensiveValues?: GraphDataView<'float32'>;
    intensiveValues?: GraphDataView<'float32'>;
    /** `targetCount * categories.categoryCount` category shares; required with `categories`. */
    categoryShares?: GraphDataView<'float32'>;
    sourceAreas?: GraphDataView<'float32'>;
    sourceTransferredFractions?: GraphDataView<'float32'>;
  },
  {
    sourceZones?: GraphDataView<'uint32'>;
    targetZones?: GraphDataView<'uint32'>;
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'change-of-support-recipe'`. */
  id?: string;
  /** Common raster grid of both rasterizations (axis-aligned). */
  grid: GPURasterMetadata;
  /**
   * Coordinate space of the raster and the polygons. Omitted means native planar (cell counts are
   * areas). `longitude-latitude` (metric `'none'` or `'great-circle'`) declares a longitude/latitude
   * raster whose `parameters` are in degrees: areas become spherical cell areas in square meters, so
   * the weights are ground-area shares. The polygon rasterization stage itself only tests cell
   * centers in the coordinates as given (straight edges in degrees; split polygons at the
   * antimeridian). The resolved context is returned as `result.spatialContext`.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * `getGPUPolygonRasterizationParameterValues(grid)` view: per-frame origin and cell size of the
   * raster (degrees on a longitude/latitude raster).
   */
  parameters: GraphDataView<'float32'>;
  /** Source zone system. */
  source: GPUChangeOfSupportZones;
  /** Target zone system. */
  target: GPUChangeOfSupportZones;
  /** Optional dasymetric raster (`grid.width * grid.height`): area becomes a sum of these weights. */
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
   * Source values, `sourceZoneCount * columnCount` rows row-major. With neither `extensiveValues`
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
  sourceZoneCount: number;
  targetZoneCount: number;
  sourceZones: GraphDataView<'uint32'>;
  targetZones: GraphDataView<'uint32'>;
  /** Extensive weights `a_st / A_s`: rows are targets, neighbors are sources. */
  extensiveWeights: GPUSpatialWeightsOutput;
  /** Intensive weights `a_st / B_t` on the same pattern. */
  intensiveWeights: GPUSpatialWeights;
  areas: GraphDataView<'float32'>;
  /** Raster area of every source (cells, or square meters on a longitude/latitude raster). */
  sourceAreas: GraphDataView<'float32'>;
  /**
   * Fraction of each source's area that reached a target (the conserved fraction of its extensive
   * mass with the default denominator), NaN for sources without raster area.
   */
  sourceTransferredFractions: GraphDataView<'float32'>;
  extensiveValues?: GraphDataView<'float32'>;
  intensiveValues?: GraphDataView<'float32'>;
  /** Present with `categories`: `targetZoneCount * categoryCount` category shares. */
  categoryShares?: GraphDataView<'float32'>;
  /** Status words next to the per-stage `stages`. */
  status: {
    /** One-row flag: 1 when the overlap pair list overflowed `pairCapacity`. */
    overflow: GPUStatusScalar;
    /** One-row unclamped distinct pair count, for sizing `pairCapacity` (stage `areal-interpolation`). */
    requiredCount: GPUStatusScalar;
    /** One-row count of sources with no raster area: their mass is lost. */
    invalidCount: GPUStatusScalar;
    /** One-row flags of the source and target rasterizations (1: crossing list overflowed). */
    rasterOverflow: [GPUStatusScalar, GPUStatusScalar];
    /**
     * One-row unclamped crossing counts of the source and target rasterizations, for sizing
     * `crossingCapacity` (reported as `requiredCount` of the rasterization stages).
     */
    rasterRequiredCount: [GPUStatusScalar, GPUStatusScalar];
  } & GPURecipeStatusPort;
};

/**
 * Change of support recipe: transfer values between two polygon zone systems by area.
 *
 * Chain: `GPUPolygonRasterization` (source and target to one raster) -> `GPUArealInterpolation`
 * (overlap areas, extensive and intensive weights) -> `GPUSpatialLag` over each weights set (the
 * lag treats the target-by-source weights as a cross matrix). Rasterization overflow flags are
 * reported separately in `status.rasterOverflow` and `status.rasterRequiredCount` (stages `source-rasterization` and
 * `target-rasterization`).
 *
 * Accuracy is bounded by the raster resolution (cell centers decide membership). Mass that a
 * source loses (it lies partly outside the target system, or it is smaller than a cell) is
 * reported per source in `sourceTransferredFractions` and `sourceAreas`, and sources with no
 * raster area are counted in `status.invalidCount` (stage `areal-interpolation`), unless
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
  const {grid} = props;
  const cellCount = grid.width * grid.height;
  const sourceCount = props.source.featureOffsets.length - 1;
  const targetCount = props.target.featureOffsets.length - 1;
  assertRecipe(ID, sourceCount >= 1 && targetCount >= 1, 'needs source and target zones');
  const columnCount = props.columnCount ?? 1;
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  const builder = new RecipeBuilder(graph);

  const rasterOverflow: [GPUStatusScalar, GPUStatusScalar] = [
    getOrCreateView(
      graph,
      `${id}-source-raster-overflow`,
      'uint32',
      1,
      outputs.status?.rasterOverflow?.[0]
    ),
    getOrCreateView(
      graph,
      `${id}-target-raster-overflow`,
      'uint32',
      1,
      outputs.status?.rasterOverflow?.[1]
    )
  ];
  const rasterRequiredCount: [GPUStatusScalar, GPUStatusScalar] = [
    getOrCreateView(
      graph,
      `${id}-source-raster-required-count`,
      'uint32',
      1,
      outputs.status?.rasterRequiredCount?.[0]
    ),
    getOrCreateView(
      graph,
      `${id}-target-raster-required-count`,
      'uint32',
      1,
      outputs.status?.rasterRequiredCount?.[1]
    )
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
        grid,
        parameters: props.parameters,
        positions: system.positions,
        featureOffsets: system.featureOffsets,
        polygonOffsets: system.polygonOffsets,
        ringOffsets: system.ringOffsets,
        crossingCapacity: system.crossingCapacity,
        output: {
          zones: raster,
          status: {overflow: rasterOverflow[index], crossingCount: rasterRequiredCount[index]}
        }
      })
    );
    return raster;
  });

  const overflow = getOrCreateView(graph, `${id}-overflow`, 'uint32', 1, outputs.status?.overflow);
  const requiredCount = getOrCreateView(
    graph,
    `${id}-required-count`,
    'uint32',
    1,
    outputs.status?.requiredCount
  );
  const extensiveWeights = createTransientSpatialWeights(
    graph,
    `${id}-extensive`,
    targetCount,
    props.pairCapacity,
    {...outputs.extensiveWeights, status: {overflow, requiredCount}}
  );
  const intensiveValues = getOrCreateView(
    graph,
    `${id}-intensive-weights`,
    'float32',
    props.pairCapacity,
    outputs.intensiveWeightValues
  );
  const areas = getOrCreateView(graph, `${id}-areas`, 'float32', props.pairCapacity, outputs.areas);
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
    outputs.status?.invalidCount
  );
  const categoryShares = props.categories
    ? getOrCreateView(
        graph,
        `${id}-category-shares`,
        'float32',
        targetCount * props.categories.categoryCount,
        outputs.categoryShares
      )
    : undefined;
  builder.add(
    new GPUArealInterpolation({
      id: `${id}-areal`,
      spatialContext,
      grid,
      ...(geographic ? {parameters: props.parameters} : {}),
      sourceAnchorCells: props.sourceAnchorCells,
      sourceZones: zones[0],
      targetZones: zones[1],
      sourceZoneCount: sourceCount,
      targetZoneCount: targetCount,
      cellWeights: props.cellWeights,
      denominator: props.denominator,
      mode: 'extensive',
      categories: props.categories,
      output: {
        weights: extensiveWeights,
        status: {overflow, requiredPairCount: requiredCount, invalidCount},
        alternateWeights: intensiveValues,
        areas,
        sourceAreas,
        sourceTransferredFractions,
        categoryShares
      }
    })
  );
  const intensiveWeights: GPUSpatialWeights = {...extensiveWeights, weights: intensiveValues};

  const result: GPUChangeOfSupportRecipeResult = {
    contributors: builder.contributors,
    spatialContext,
    sourceZoneCount: sourceCount,
    targetZoneCount: targetCount,
    sourceZones: zones[0],
    targetZones: zones[1],
    extensiveWeights,
    intensiveWeights,
    areas,
    sourceAreas,
    sourceTransferredFractions,
    outputs: {
      extensiveWeights,
      intensiveWeights,
      areas,
      sourceAreas,
      sourceTransferredFractions,
      ...(categoryShares ? {categoryShares} : {})
    },
    ...(categoryShares ? {categoryShares} : {}),
    intermediates: {sourceZones: zones[0], targetZones: zones[1]},
    status: {
      overflow,
      requiredCount,
      invalidCount,
      rasterOverflow,
      rasterRequiredCount,
      stages: [
        {
          stage: 'source-rasterization',
          status: {overflow: rasterOverflow[0], requiredCount: rasterRequiredCount[0]}
        },
        {
          stage: 'target-rasterization',
          status: {overflow: rasterOverflow[1], requiredCount: rasterRequiredCount[1]}
        },
        {
          stage: 'areal-interpolation',
          status: {overflow, requiredCount, invalidCount}
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
