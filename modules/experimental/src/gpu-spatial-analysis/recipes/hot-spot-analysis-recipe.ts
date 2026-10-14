// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPUClassBreaks, GPUColorScale} from '../../gpu-dataframe/column-classification/index';
import type {
  GPUClassBreaksProps,
  GPUColorScaleProps
} from '../../gpu-dataframe/column-classification/index';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {GET_QUIET_NAN_SEEDED_WGSL, IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {GPUCellAggregation} from '../cell-aggregation/index';
import type {GPUCellFamily, GPUCellTable} from '../cell-aggregation/index';
import {CELL_KEY_WGSL} from '../cell-aggregation/cell-keys';
import {GPUCellCover} from '../cell-cover/index';
import {GPUCellGeometry, GPUCellMeasures, GPUPointToCell} from '../cell-indexing/index';
import {getCellTopologyStride} from '../cell-topology/index';
import {
  GPU_DEFAULT_INFERENCE_ALTERNATIVE,
  GPU_EARTH_AUTHALIC_RADIUS_METERS,
  resolveGPUSpatialContext,
  type GPUInferenceMetadata,
  type GPUMultipleTestingMethod,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../contracts/index';
import {GPUNeighborSearch} from '../neighbor-search/index';
import {GPULocalPermutationTest} from '../permutation-inference/index';
import type {
  GPULocalPermutationStatistic,
  GPUPermutationAlternative
} from '../permutation-inference/index';
import {
  GPUHotSpotAnalysis,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH
} from '../spatial-autocorrelation/index';
import {GPULatticeWeights} from '../spatial-weights/index';
import type {GPULatticeCriterion, GPUSpatialWeights} from '../spatial-weights/index';
import {
  addCellDiskWeightsNodes,
  assertRecipe,
  createTransientSpatialWeights,
  getCellKeySearchWGSL,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUHotSpotAnalysisRecipe';

/** Default of {@link GPUHotSpotAnalysisRecipeProps.minimumNeighbors} (ArcGIS guidance for Gi*). */
export const GPU_HOT_SPOT_DEFAULT_MINIMUM_NEIGHBORS = 8;

/** Default of {@link GPUHotSpotAnalysisRecipeProps.multipleTesting}. */
export const GPU_HOT_SPOT_DEFAULT_MULTIPLE_TESTING: GPUMultipleTestingMethod = 'benjamini-hochberg';

/**
 * Study area of the points source: polygons in GeoArrow-style offsets, as `GPUCellCover` takes
 * them. Every cell whose center lies inside the polygons is an analysis row, empty or not.
 */
export type GPUHotSpotStudyArea = {
  /** Flattened polygon vertices, planar longitude/latitude degrees. */
  polygonPositions: GraphDataView<'float32x2'>;
  /** Feature-to-polygon offsets with `featureCount + 1` entries, first 0. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets with a terminal entry. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets with a terminal entry. */
  ringOffsets: GraphDataView<'uint32'>;
  /** Candidate cells `GPUCellCover` tests (see its `candidateCapacity`). */
  candidateCapacity: number;
  /**
   * Capacity of the raw cover, which lists a cell once per covering feature before it is
   * deduplicated. Defaults to `cellCapacity`; raise it when features overlap.
   */
  coverCapacity?: number;
};

/**
 * Input rows: points aggregated onto every cell of a study area, linked by grid topology.
 *
 * Rows are the study-area cells (cell centers inside `studyArea`, sorted by key), zero-count cells
 * included, so Gi* sees the true null (cold spots are possible and the mean is not inflated).
 * Points outside the study area are excluded and counted in `pointsOutsideStudyArea`. Neighbors
 * are the cells within `neighborRings` grid steps that are also study cells, not a metric radius.
 */
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
  /** Study area whose cells are the analysis rows. */
  studyArea: GPUHotSpotStudyArea;
  /** Rows of the study table (and of every per-cell output): an upper bound on the study cells. */
  cellCapacity: number;
  /**
   * Capacity of the occupied-cell table of the points, which also holds cells outside the study
   * area. Defaults to `cellCapacity`. Overflow drops the largest keys (see the `point-aggregation`
   * status).
   */
  pointCellCapacity?: number;
  /**
   * Grid distance of the neighborhood: all cells within this many steps (a disk, self excluded).
   * Defaults to 1 for `'quadbin'` (8 neighbors) and 2 for `'h3'` (18 neighbors), the smallest disks
   * that reach the default `minimumNeighbors` of 8. Integer in `[1, 8]`.
   */
  neighborRings?: number;
  /**
   * Slots of the neighbor CSR. Defaults to the largest possible, `cellCapacity` times the disk
   * size minus one.
   */
  neighborCapacity?: number;
  /**
   * Optional metric neighborhood replacing the grid rings: a great-circle distance band in meters
   * over the cell centers. Exclusive with `neighborRings` and `neighborCapacity`. The
   * `minimumNeighbors` guard still applies (rows with a sparse band are NaN).
   */
  distanceBand?: GPUHotSpotDistanceBand;
};

/**
 * Metric neighborhood of the points source: every study cell whose center lies within a
 * great-circle radius of the focal cell's center (`GPUNeighborSearch` in radius mode over the cell
 * centers). Cell centers are longitude/latitude degrees, so the context must be
 * `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` and the radius is in its
 * `units` (meters by default); a radius in degrees would turn the band into an ellipse that
 * shrinks east-west with latitude, so any other context throws.
 */
export type GPUHotSpotDistanceBand = {
  /** Longitude/latitude great-circle context of the cell centers. */
  spatialContext: GPUSpatialContext;
  /**
   * Per-frame `getGPUNeighborSearchParameterValues` view: `bounds` in degrees, `radius` in
   * `spatialContext.units` (meters by default), binary weights for classic Gi*.
   */
  neighborSearchParameters: GraphDataView<'float32'>;
  /** Search grid size passed to `GPUNeighborSearch` (`[columns, rows]`). */
  gridSize: readonly [number, number];
  /** Slots of the neighbor CSR. */
  neighborCapacity: number;
  /**
   * Row-standardize the band weights, passed to `GPUNeighborSearch` as its compile-time
   * `rowStandardize` so the weights carry `info.transform: 'row'` and masked neighbors are
   * re-normalized by the consumers. It overrides the per-frame `rowStandardize` field of
   * `neighborSearchParameters`. Defaults to false: classic binary Gi* (`info.transform: 'binary'`).
   */
  rowStandardize?: boolean;
};

/** Contexts {@link GPUHotSpotDistanceBand} accepts. */
const DISTANCE_BAND_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = {
  coordinateSpaces: ['longitude-latitude'],
  metrics: ['great-circle']
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
    /** One-row scalar: rows set to NaN by the `minimumNeighbors` guard. */
    invalidCount?: GraphDataView<'uint32'>;
    /** One-row scalar: points that fell outside the study area (points source). */
    pointsOutsideStudyArea?: GraphDataView<'uint32'>;
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
    /** The deduplicated study table (`cells`, `counts` = cover multiplicity, `count`, `overflow`). */
    table?: Partial<GPUCellTable>;
    /** The occupied-cell table of the points. */
    pointTable?: Partial<GPUCellTable>;
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
  /**
   * Points source only. `'density'` (default) analyses value (or count) per square kilometer of
   * cell area, which is what makes Web Mercator `'quadbin'` cells comparable across latitudes
   * (a tile's area shrinks with the cosine of latitude). `'count'` analyses the raw count or sum
   * and is allowed for `'h3'` only (cells are approximately equal area); `'quadbin'` throws.
   * Areas are spherical on the authalic sphere (`GPU_EARTH_AUTHALIC_RADIUS_METERS`), the same
   * sphere `GPUCellMeasures` uses for H3, so both families share one model.
   */
  analyze?: 'density' | 'count';
  /**
   * Minimum number of neighbors other than the row itself for a row to be tested. Rows below it
   * get NaN `zScores` and `pValues` and bin 0, are excluded from the multiple-testing family, and
   * are counted in the `minimum-neighbors` status `invalidCount` (together with any other included
   * row whose z-score is undefined). Defaults to 8 (ArcGIS guidance for Gi*). Values 0 and 1 both
   * only reject rows with no neighbor (the minimum `GPUHotSpotAnalysis` accepts). Rows below the
   * minimum still count in the global moments and in the neighbor sums of other rows.
   */
  minimumNeighbors?: number;
  /**
   * Multiple-testing correction of the Gi* bins, passed to `GPUHotSpotAnalysis`. Defaults to
   * `'benjamini-hochberg'` (the ArcGIS "Apply False Discovery Rate Correction" option). The test
   * family is every row with a defined z-score, never a viewport.
   *
   * Legend contract, stable across this setting: bins `+-1`, `+-2`, `+-3` mean 90%, 95% and 99%
   * confidence hot (`+`) or cold (`-`) spots. Without correction that is `|z| >= 1.645, 1.960,
   * 2.576`; with Benjamini-Hochberg it means the BH step-up test passes at q = 0.10, 0.05, 0.01.
   * `inference` on the result states which one applies.
   */
  multipleTesting?: GPUMultipleTestingMethod;
  /** Permutation confirmation; skipped when absent. */
  permutation?: GPUHotSpotPermutationOptions;
  /** Class breaks and colors of the z-scores; skipped when absent. */
  color?: GPUHotSpotColorOptions;
};

/** Named outputs of {@link addHotSpotAnalysisRecipe}. Views are caller-owned when passed in. */
export type GPUHotSpotAnalysisRecipeResult = GPURecipeResult & {
  /** Rows of every per-cell output. */
  rowCount: number;
  /**
   * Study table (points source only): `cells` are the sorted study-area cells, `counts` and
   * `sumValues` the points joined onto them (0 for empty cells), `count` the occupied rows.
   */
  table?: GPUCellTable;
  /** Cell centers (points source only). */
  centers?: GraphDataView<'float32x2'>;
  /** Per-row analysis value (density or count/sum) and inclusion mask. */
  values: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  weights: GPUSpatialWeights;
  weightsOverflow: GraphDataView<'uint32'>;
  /** Rows set to NaN because they had fewer than `minimumNeighbors` neighbors. */
  invalidCount: GraphDataView<'uint32'>;
  /** Points outside the study area (points source only). */
  pointsOutsideStudyArea?: GraphDataView<'uint32'>;
  zScores: GraphDataView<'float32'>;
  bins: GraphDataView<'sint32'>;
  pValues: GraphDataView<'float32'>;
  neighborCounts: GraphDataView<'uint32'>;
  globalStatistics: GraphDataView<'float32'>;
  /** How `zScores`, `pValues` and `bins` were produced, for building a legend. */
  inference: GPUInferenceMetadata;
  /** Metadata of the permutation confirmation; present with `permutation`. */
  permutationInference?: GPUInferenceMetadata;
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

/** CPU-side description of the Gi* `zScores`, `pValues` and `bins` of the recipe. */
function getHotSpotInferenceMetadata(
  method: GPUMultipleTestingMethod,
  minimumNeighbors: number,
  testedRows: string
): GPUInferenceMetadata {
  return {
    alternative: GPU_DEFAULT_INFERENCE_ALTERNATIVE,
    permutationCount: 0,
    multipleTesting: method,
    testedRowsDefinition:
      minimumNeighbors > 0
        ? `${testedRows} with at least ${minimumNeighbors} neighbors`
        : testedRows
  };
}

const AUTHALIC_RADIUS_KILOMETERS = GPU_EARTH_AUTHALIC_RADIUS_METERS / 1000;

/**
 * WGSL `getQuadbinAreaKilometers(key: vec2u(high, low)) -> f32`: the exact area of a Web Mercator
 * tile on the authalic sphere, the spherical rectangle `R^2 dLon (sin(lat1) - sin(lat2))`.
 * With the tile's Mercator ordinates `t = pi (1 - 2 y / 2^z)`, `sin(lat) = tanh(t)`, and
 * `tanh(t1) - tanh(t2) = sinh(t1 - t2) / (cosh(t1) cosh(t2))` with `t1 - t2 = 2 pi / 2^z`, which
 * avoids subtracting two nearly equal sines at high resolutions. The tile row decode mirrors
 * `geometryGetQuadbinTile` of `GPUCellGeometry`.
 */
function getQuadbinAreaWGSL(): string {
  return `const PI: f32 = 3.14159265358979;
const EARTH_RADIUS_SQUARED_KILOMETERS: f32 = ${getWGSLFloatLiteral(AUTHALIC_RADIUS_KILOMETERS * AUTHALIC_RADIUS_KILOMETERS)};
fn compactEvenBits(value: u32) -> u32 {
  var x = value & 0x55555555u;
  x = (x | (x >> 1u)) & 0x33333333u;
  x = (x | (x >> 2u)) & 0x0f0f0f0fu;
  x = (x | (x >> 4u)) & 0x00ff00ffu;
  x = (x | (x >> 8u)) & 0x0000ffffu;
  return x;
}
fn getQuadbinAreaKilometers(key: vec2u) -> f32 {
  if (!cellIsValidQuadbin(key)) {
    return 0.0;
  }
  let z = (key.x >> 20u) & 0x1fu;
  let morton = cellGetCompactKey(key, 52u - 2u * z, 2u * z);
  let y = compactEvenBits(morton.y >> 1u) | (compactEvenBits(morton.x >> 1u) << 16u);
  let tiles = f32(1u << z);
  let step = 2.0 * PI / tiles;
  let north = PI * f32(i32(1u << z) - 2 * i32(y)) / tiles;
  let south = north - step;
  return EARTH_RADIUS_SQUARED_KILOMETERS * step * sinh(step) / (cosh(north) * cosh(south));
}`;
}

/** Fixed neighbors per disk used only to size defaults. */
function getDiskNeighborCount(family: GPUCellFamily, rings: number): number {
  return getCellTopologyStride(family, {type: 'disk', k: rings}) - 1;
}

/**
 * Hot spot analysis recipe: points on a study area, or a raster, to Getis-Ord Gi* hot and cold
 * spots, optionally confirmed by conditional permutation and colored by class breaks.
 *
 * Chain (points source): `GPUCellCover` (study-area cells) -> `GPUCellAggregation` (sorted unique
 * study table) and `GPUPointToCell` (H3 only) -> `GPUCellAggregation` (sorted point table) -> a
 * join kernel (binary search of each study key in the point table; empty cells get 0) -> area
 * normalization (`GPUCellMeasures` for H3, the exact spherical rectangle for Quadbin) ->
 * `GPUCellTopology` disk -> binary-searched CSR weights -> `GPUHotSpotAnalysis` ->
 * `invalidCount` tally -> `GPULocalPermutationTest` -> `GPUClassBreaks` -> `GPUColorScale`.
 * A lattice source replaces the study and aggregation stages with `GPULatticeWeights` over the
 * dense raster.
 *
 * Statistical contract:
 * - Every study-area cell is a row, zeros included, so the global mean and deviation describe the
 *   whole area and clusters of empty cells can be cold spots.
 * - Neighbors are topological (grid disk), never a metric radius in degrees.
 * - Rows with fewer than `minimumNeighbors` (default 8) neighbors are NaN, not a plain
 *   standardized value, and are outside the multiple-testing family.
 * - `multipleTesting` defaults to Benjamini-Hochberg. Bins `+-1/2/3` always mean 90/95/99%
 *   confidence; with correction, BH at q = 0.10/0.05/0.01. `result.inference` records this.
 *
 * Rows are study cells (`cellCapacity`) or lattice cells. Everything except the output views is
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
  const minimumNeighbors = props.minimumNeighbors ?? GPU_HOT_SPOT_DEFAULT_MINIMUM_NEIGHBORS;
  assertRecipe(
    ID,
    Number.isInteger(minimumNeighbors) && minimumNeighbors >= 0,
    'minimumNeighbors must be a non-negative integer'
  );
  const multipleTesting = props.multipleTesting ?? GPU_HOT_SPOT_DEFAULT_MULTIPLE_TESTING;
  let rowCount: number;
  let values: GraphDataView<'float32'>;
  let mask: GraphDataView<'uint32'> | undefined;
  let table: GPUCellTable | undefined;
  let centers: GraphDataView<'float32x2'> | undefined;
  let weights: GPUSpatialWeights;
  let pointsOutsideStudyArea: GraphDataView<'uint32'> | undefined;
  const stages: {stage: string; status: Record<string, GraphDataView<'uint32'>>}[] = [];
  const weightsOverflow = getOrCreateView(
    graph,
    `${id}-weights-overflow`,
    'uint32',
    1,
    outputs.weightsOverflow
  );
  let testedRows: string;

  if (source.kind === 'points') {
    testedRows = 'study-area cells';
    rowCount = source.cellCapacity;
    assertRecipe(ID, Number.isInteger(rowCount) && rowCount >= 1, 'cellCapacity must be positive');
    const analyze = props.analyze ?? 'density';
    assertRecipe(
      ID,
      analyze === 'density' || analyze === 'count',
      "analyze must be 'density' or 'count'"
    );
    assertRecipe(
      ID,
      !(analyze === 'count' && source.family === 'quadbin'),
      "analyze 'count' is not allowed for quadbin: Web Mercator cells vary in area with latitude, use 'density'"
    );
    const {distanceBand} = source;
    if (distanceBand) {
      assertRecipe(
        ID,
        source.neighborRings === undefined && source.neighborCapacity === undefined,
        'distanceBand is exclusive with neighborRings and neighborCapacity'
      );
      // Cell centers are degrees: only a great-circle band in ground units is meaningful.
      resolveGPUSpatialContext(
        `${ID} distanceBand`,
        distanceBand.spatialContext,
        DISTANCE_BAND_SPATIAL_CONTEXT_SUPPORT
      );
    }
    const rings = source.neighborRings ?? (source.family === 'h3' ? 2 : 1);
    const neighborsPerDisk = getDiskNeighborCount(source.family, rings);
    assertRecipe(
      ID,
      Number.isInteger(rings) && rings >= 1,
      'neighborRings must be an integer >= 1'
    );
    assertRecipe(
      ID,
      Boolean(distanceBand) || minimumNeighbors <= neighborsPerDisk,
      `minimumNeighbors ${minimumNeighbors} exceeds the ${neighborsPerDisk} neighbors of a ${source.family} disk with neighborRings ${rings}; raise neighborRings or lower minimumNeighbors`
    );
    const neighborCapacity =
      distanceBand?.neighborCapacity ?? source.neighborCapacity ?? rowCount * neighborsPerDisk;
    const pointCapacity = source.pointCellCapacity ?? rowCount;
    const coverCapacity = source.studyArea.coverCapacity ?? rowCount;
    const studyProvided = scratch.table ?? {};
    const pointProvided = scratch.pointTable ?? {};

    // Study area: cover cells, then dedupe and sort them with a cell aggregation on the keys.
    const cover = {
      featureIds: createTransientView(graph, `${id}-cover-features`, 'uint32', coverCapacity),
      cells: createTransientView(graph, `${id}-cover-cells`, 'uint32x2', coverCapacity),
      count: createTransientView(graph, `${id}-cover-count`, 'uint32', 1),
      overflow: createTransientView(graph, `${id}-cover-overflow`, 'uint32', 1),
      candidateOverflow: createTransientView(graph, `${id}-cover-candidate-overflow`, 'uint32', 1)
    };
    builder.add(
      new GPUCellCover({
        id: `${id}-cover`,
        family: source.family,
        resolution: source.resolution,
        containment: 'center',
        polygonPositions: source.studyArea.polygonPositions,
        featureOffsets: source.studyArea.featureOffsets,
        polygonOffsets: source.studyArea.polygonOffsets,
        ringOffsets: source.studyArea.ringOffsets,
        candidateCapacity: source.studyArea.candidateCapacity,
        output: cover
      })
    );
    // Rows past the cover count are unwritten; mask them out of the aggregation.
    const coverMask = createTransientView(graph, `${id}-cover-mask`, 'uint32', coverCapacity);
    graph.add(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cover-mask`,
        operation: ID,
        variant: 'cover-mask',
        bindings: [
          {name: 'coverCount', view: cover.count, type: 'u32', access: 'read'},
          {name: 'coverMask', view: coverMask, type: 'u32', access: 'read_write'}
        ],
        invocationCount: coverCapacity,
        body: 'coverMask[coverMaskOffset + index] = select(0u, 1u, index < coverCount[coverCountOffset]);'
      })
    );
    const studyTable: GPUCellTable = {
      cells: getOrCreateView(graph, `${id}-cells`, 'uint32x2', rowCount, studyProvided.cells),
      counts: getOrCreateView(
        graph,
        `${id}-study-counts`,
        'uint32',
        rowCount,
        studyProvided.counts
      ),
      count: getOrCreateView(graph, `${id}-count`, 'uint32', 1, studyProvided.count),
      overflow: getOrCreateView(graph, `${id}-table-overflow`, 'uint32', 1, studyProvided.overflow)
    };
    builder.add(
      new GPUCellAggregation({
        id: `${id}-study-cells`,
        family: source.family,
        resolution: source.resolution,
        cells: cover.cells,
        mask: coverMask,
        output: studyTable
      })
    );

    // Points: aggregate onto the sorted point table.
    const pointTable: GPUCellTable = {
      cells: getOrCreateView(
        graph,
        `${id}-point-cells`,
        'uint32x2',
        pointCapacity,
        pointProvided.cells
      ),
      counts: getOrCreateView(
        graph,
        `${id}-point-counts`,
        'uint32',
        pointCapacity,
        pointProvided.counts
      ),
      sumValues: source.values
        ? getOrCreateView(
            graph,
            `${id}-point-sum-values`,
            'float32',
            pointCapacity,
            pointProvided.sumValues
          )
        : undefined,
      count: getOrCreateView(graph, `${id}-point-count`, 'uint32', 1, pointProvided.count),
      overflow: getOrCreateView(graph, `${id}-point-overflow`, 'uint32', 1, pointProvided.overflow)
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
          output: pointTable
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
          output: pointTable
        })
      );
    }

    // Join: every study row looks up its key in the point table; absent means 0.
    const joinedCounts = createTransientView(graph, `${id}-joined-counts`, 'uint32', rowCount);
    const joinedSums = source.values
      ? createTransientView(graph, `${id}-joined-sums`, 'float32', rowCount)
      : undefined;
    graph.add(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-join`,
        operation: ID,
        variant: 'join',
        bindings: [
          {name: 'studyCells', view: studyTable.cells, type: 'u32', access: 'read'},
          {name: 'pointCells', view: pointTable.cells, type: 'u32', access: 'read'},
          {name: 'pointCounts', view: pointTable.counts, type: 'u32', access: 'read'},
          ...(pointTable.sumValues
            ? [
                {
                  name: 'pointSums',
                  view: pointTable.sumValues,
                  type: 'f32',
                  access: 'read'
                } as const
              ]
            : []),
          {name: 'pointCount', view: pointTable.count, type: 'u32', access: 'read'},
          {name: 'joinedCounts', view: joinedCounts, type: 'u32', access: 'read_write'},
          ...(joinedSums
            ? [{name: 'joinedSums', view: joinedSums, type: 'f32', access: 'read_write'} as const]
            : [])
        ],
        invocationCount: rowCount,
        declarations: getCellKeySearchWGSL(
          'findPointCell',
          'pointCells',
          'pointCount[pointCountOffset]'
        ),
        body: `let key = vec2u(studyCells[studyCellsOffset + 2u * index + 1u], studyCells[studyCellsOffset + 2u * index]);
  let row = findPointCell(key);
  let found = row != 0xffffffffu;
  let safeRow = select(0u, row, found);
  joinedCounts[joinedCountsOffset + index] = select(0u, pointCounts[pointCountsOffset + safeRow], found);
  ${joinedSums ? 'joinedSums[joinedSumsOffset + index] = select(0.0, pointSums[pointSumsOffset + safeRow], found);' : ''}`
      })
    );

    // Points whose cell is not a study cell.
    pointsOutsideStudyArea = getOrCreateView(
      graph,
      `${id}-points-outside`,
      'uint32',
      1,
      outputs.pointsOutsideStudyArea
    );
    graph.add(
      createFillNode<Parameters>(graph, {
        id: `${id}-points-outside-clear`,
        operation: ID,
        view: pointsOutsideStudyArea,
        type: 'u32',
        value: '0u'
      })
    );
    graph.add(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-points-outside-count`,
        operation: ID,
        variant: 'outside',
        bindings: [
          {name: 'pointCells', view: pointTable.cells, type: 'u32', access: 'read'},
          {name: 'pointCounts', view: pointTable.counts, type: 'u32', access: 'read'},
          {name: 'pointCount', view: pointTable.count, type: 'u32', access: 'read'},
          {name: 'studyCells', view: studyTable.cells, type: 'u32', access: 'read'},
          {name: 'studyCount', view: studyTable.count, type: 'u32', access: 'read'},
          {name: 'outside', view: pointsOutsideStudyArea, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pointCapacity,
        declarations: getCellKeySearchWGSL(
          'findStudyCell',
          'studyCells',
          'studyCount[studyCountOffset]'
        ),
        body: `if (index >= pointCount[pointCountOffset]) {
    return;
  }
  let key = vec2u(pointCells[pointCellsOffset + 2u * index + 1u], pointCells[pointCellsOffset + 2u * index]);
  if (findStudyCell(key) == 0xffffffffu) {
    atomicAdd(&outside[outsideOffset], pointCounts[pointCountsOffset + index]);
  }`
      })
    );

    centers = getOrCreateView(graph, `${id}-centers`, 'float32x2', rowCount, scratch.centers);
    builder.add(
      new GPUCellGeometry({
        id: `${id}-geometry`,
        family: source.family,
        cells: studyTable.cells,
        output: {centers}
      })
    );

    // Analysis values: count or sum, divided by the cell area for density; mask = study rows.
    values = getOrCreateView(graph, `${id}-values`, 'float32', rowCount, scratch.values);
    mask = getOrCreateView(graph, `${id}-mask`, 'uint32', rowCount, scratch.mask);
    let areas: GraphDataView<'float32'> | undefined;
    if (analyze === 'density' && source.family === 'h3') {
      areas = createTransientView(graph, `${id}-areas`, 'float32', rowCount);
      builder.add(
        new GPUCellMeasures({
          id: `${id}-measures`,
          family: 'h3',
          cells: studyTable.cells,
          units: 'kilometers',
          output: {areas}
        })
      );
    }
    const quadbinDensity = analyze === 'density' && source.family === 'quadbin';
    const areaWGSL = quadbinDensity
      ? 'getQuadbinAreaKilometers(key)'
      : areas
        ? 'areas[areasOffset + index]'
        : '1.0';
    graph.add(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-analysis-values`,
        operation: ID,
        variant: 'analysis-values',
        bindings: [
          {name: 'joinedCounts', view: joinedCounts, type: 'u32', access: 'read'},
          ...(joinedSums
            ? [{name: 'joinedSums', view: joinedSums, type: 'f32', access: 'read'} as const]
            : []),
          ...(quadbinDensity
            ? [{name: 'studyCells', view: studyTable.cells, type: 'u32', access: 'read'} as const]
            : []),
          ...(areas ? [{name: 'areas', view: areas, type: 'f32', access: 'read'} as const] : []),
          {name: 'studyCount', view: studyTable.count, type: 'u32', access: 'read'},
          {name: 'analysisValues', view: values, type: 'f32', access: 'read_write'},
          {name: 'analysisMask', view: mask, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: `${CELL_KEY_WGSL}
${GET_QUIET_NAN_SEEDED_WGSL}
${quadbinDensity ? getQuadbinAreaWGSL() : ''}`,
        body: `${quadbinDensity ? 'let key = vec2u(studyCells[studyCellsOffset + 2u * index + 1u], studyCells[studyCellsOffset + 2u * index]);' : ''}
  let area = ${areaWGSL};
  let rowIsStudy = index < studyCount[studyCountOffset];
  let valid = rowIsStudy && area > 0.0;
  let raw = ${joinedSums ? 'joinedSums[joinedSumsOffset + index]' : 'f32(joinedCounts[joinedCountsOffset + index])'};
  analysisMask[analysisMaskOffset + index] = select(0u, 1u, valid);
  analysisValues[analysisValuesOffset + index] = select(getQuietNaN(index), ${analyze === 'density' ? 'raw / area' : 'raw'}, valid);`
      })
    );

    // Topological CSR over the study cells.
    weights = createTransientSpatialWeights(
      graph,
      `${id}-weights`,
      rowCount,
      neighborCapacity,
      scratch.weights
    );
    if (distanceBand) {
      const search = new GPUNeighborSearch({
        id: `${id}-neighbors`,
        mode: 'radius',
        positions: centers,
        mask,
        spatialContext: distanceBand.spatialContext,
        parameters: distanceBand.neighborSearchParameters,
        gridSize: distanceBand.gridSize,
        weights,
        overflow: weightsOverflow,
        rowStandardize: distanceBand.rowStandardize ?? false,
        // The band documents binary weights (`weightKind: 'binary'`).
        weightTransform: distanceBand.rowStandardize ? 'row' : 'binary'
      });
      builder.add(search);
      // Downstream consumers read the described weights, so info and status flow.
      weights = search.outputWeights;
    } else {
      addCellDiskWeightsNodes(graph, builder, `${id}-neighbors`, {
        family: source.family,
        cells: studyTable.cells,
        count: studyTable.count,
        mask,
        rings,
        weights,
        overflow: weightsOverflow
      });
      weights.info = {
        rowCount,
        transform: 'binary',
        symmetric: true,
        provenance: {source: 'GPUCellTopology', criterion: 'disk', order: rings}
      };
    }

    table = {
      cells: studyTable.cells,
      counts: joinedCounts,
      sumValues: joinedSums,
      count: studyTable.count,
      overflow: studyTable.overflow
    };
    stages.push(
      {
        stage: 'study-area',
        status: {overflow: cover.overflow, candidateOverflow: cover.candidateOverflow}
      },
      {stage: 'study-table', status: {count: studyTable.count, overflow: studyTable.overflow}},
      {stage: 'point-aggregation', status: {count: pointTable.count, overflow: pointTable.overflow}}
    );
  } else {
    testedRows = 'lattice cells';
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
    const lattice = new GPULatticeWeights({
      id: `${id}-lattice`,
      width: source.width,
      height: source.height,
      criterion: source.criterion ?? 'queen',
      radius: source.radius,
      mask,
      weights,
      overflow: weightsOverflow
    });
    builder.add(lattice);
    weights = lattice.outputWeights;
  }
  stages.push({stage: 'weights', status: {overflow: weightsOverflow}});

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
    GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH,
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
      multipleTesting,
      minimumNeighbors: Math.max(1, minimumNeighbors),
      zScores,
      bins,
      pValues,
      neighborCounts,
      globalStatistics
    })
  );

  // `GPUHotSpotAnalysis` writes NaN for rows below `minimumNeighbors`; tally included rows with
  // an undefined z-score for the status.
  const invalidCount = getOrCreateView(
    graph,
    `${id}-invalid-count`,
    'uint32',
    1,
    outputs.invalidCount
  );
  graph.add(
    createFillNode<Parameters>(graph, {
      id: `${id}-invalid-clear`,
      operation: ID,
      view: invalidCount,
      type: 'u32',
      value: '0u'
    })
  );
  const tallyBindings: WGSLKernelBinding[] = [
    {name: 'analysisValues', view: values, type: 'f32', access: 'read'},
    ...(mask ? [{name: 'analysisMask', view: mask, type: 'u32', access: 'read'} as const] : []),
    {name: 'zScores', view: zScores, type: 'f32', access: 'read'},
    {name: 'invalid', view: invalidCount, type: 'atomic<u32>', access: 'read_write'}
  ];
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-invalid-tally`,
      operation: ID,
      variant: 'invalid-tally',
      bindings: tallyBindings,
      invocationCount: rowCount,
      declarations: IS_FINITE_VALUE_WGSL,
      body: `let included = ${mask ? 'analysisMask[analysisMaskOffset + index] != 0u && ' : ''}isFiniteValue(analysisValues[analysisValuesOffset + index]);
  if (included && !isFiniteValue(zScores[zScoresOffset + index])) {
    atomicAdd(&invalid[invalidOffset], 1u);
  }`
    })
  );
  stages.push({stage: 'minimum-neighbors', status: {invalidCount}});

  const result: GPUHotSpotAnalysisRecipeResult = {
    contributors: builder.contributors,
    rowCount,
    table,
    centers,
    values,
    mask,
    weights,
    weightsOverflow,
    invalidCount,
    pointsOutsideStudyArea,
    zScores,
    bins,
    pValues,
    neighborCounts,
    globalStatistics,
    inference: getHotSpotInferenceMetadata(multipleTesting, minimumNeighbors, testedRows),
    outputs: {zScores, bins, pValues, neighborCounts, globalStatistics, invalidCount},
    intermediates: {table, centers, values, mask, weights},
    status: {stages}
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
    const permutationAlternative = permutation.alternative ?? GPU_DEFAULT_INFERENCE_ALTERNATIVE;
    builder.add(
      new GPULocalPermutationTest({
        id: `${id}-permutation`,
        weights,
        values,
        mask,
        statistic: permutation.statistic ?? 'localGStar',
        alternative: permutationAlternative,
        parameters: permutation.parameters,
        maximumPermutations: permutation.maximumPermutations,
        maximumNeighbors: permutation.maximumNeighbors,
        falseDiscoveryRate: permutation.falseDiscoveryRate,
        ...permutationOutputs
      })
    );
    result.permutation = permutationOutputs;
    result.permutationInference = {
      alternative: permutationAlternative === 'less' ? 'lesser' : permutationAlternative,
      permutationCount: permutation.maximumPermutations,
      multipleTesting: permutation.falseDiscoveryRate ? 'benjamini-hochberg' : 'none',
      testedRowsDefinition: result.inference.testedRowsDefinition
    };
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
