// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Spatial weights and autocorrelation of Manhattan taxi activity. Trip vertices and points of
 * interest are binned once into square cells; each non-empty cell is a point with a count value.
 * One pipeline of four contributors runs on the GPU from there:
 *
 * 1. `GPUNeighborSearch` writes a spatial-weights CSR (exact kNN, or every cell within a distance
 *    band) with a selectable weight function and row standardization.
 * 2. `GPUGlobalSpatialStatistics` reads the CSR: Moran's I, Geary's C and General G with z and p.
 * 3. `GPUGlobalPermutationTest` builds the Moran reference distribution (`p_sim`).
 * 4. `GPULocalPermutationTest` runs conditional permutations for local Moran (LISA) and writes
 *    pseudo p-values and a significance mask, optionally Benjamini-Hochberg corrected.
 *
 * `k` and the Benjamini-Hochberg choice are compile-time contributor options, so one search graph per
 * `k` and both local-test variants are compiled up front and the controls pick among them. All
 * other controls (distance band, weight function, row standardization, permutation count, seed,
 * significance level, value column) are buffer writes: the rebuild counter stays 0. The graphs are
 * encoded only when an input changed (the data is static) and the only readback is a ring-buffered
 * summary of a few hundred bytes.
 *
 * Around that pipeline the mode exercises the rest of the weights toolkit on the same cells:
 *
 * - `GPUSpatialWeightsAlgebra` combines the k-nearest weights (A) with a distance-band weights
 *   (B, close to queen contiguity): union, intersection, difference, symmetric difference,
 *   higher order (exact or cumulative), a subgraph by a busyness mask and block weights of 600 m
 *   blocks. The result is drawn as orange links and colors the "cardinality" map.
 * - `GPUSpatialWeightsSummary` reads S0, S1, S2, isolates, asymmetry and the cardinality
 *   histogram of A and of the result.
 * - `GPUSpatialWeightsTransform` applies the double (`D`) or variance (`V`) transform in place.
 * - `GPUNeighborhoodSummary` colors each cell by a statistic of its neighborhood (mean, sum, min,
 *   max, standard deviation, median, dominant POI category, category entropy).
 *
 * - `GPUSpatialWeightsTranspose` and the symmetrised union (`GPUSpatialWeightsAlgebra`): the
 *   transpose of the kNN weights and `W union W'` are drawn next to `W`, with the count of
 *   asymmetric pairs read from the transpose's symmetry check.
 * - `GPUSpatialLag` lags the value column over the active weights (plain or normalized).
 *
 * The algebra operation, higher order, cumulative flag and transform are compile-time contributor
 * options: every variant is compiled up front and the controls pick among them.
 *
 * Two more weights producers share the same analysis kit (transforms, summary, transpose, lag, focus
 * links), selected by "Weights source". All sources are compiled up front, so switching is free:
 *
 * - `GPULatticeWeights` writes rook or queen neighbors (radius 1 to 3, a compile-time choice with
 *   every variant precompiled) of a regular lattice over Manhattan. A per-frame mask removes
 *   cells with too little activity, which makes them islands.
 * - `GPUContiguityWeights` writes rook or queen contiguity of irregular blocks with shared jittered
 *   corners (an activity-derived tessellation: the SF ZIP codes of other modes are outside this
 *   mode's camera).
 *
 * Hover the map to draw the neighbor links of the nearest row; click to pin it.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUHistogram,
  GPUReadbackRing,
  GPUReduction,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUNeighborSearchParameterValues,
  getGPUPermutationParameterValues,
  GPU_GLOBAL_PERMUTATION_RESULT,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_NEIGHBORHOOD_SUMMARY_NO_MODE,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT,
  GPUContiguityWeights,
  GPUGlobalPermutationTest,
  GPUGlobalSpatialStatistics,
  GPULatticeWeights,
  GPULocalPermutationTest,
  GPUNeighborhoodSummary,
  GPUNeighborSearch,
  GPUSpatialWeightsAlgebra,
  GPUSpatialWeightsSummary,
  type GPUContiguityCriterion,
  type GPULatticeCriterion,
  type GPUNeighborhoodSummaryStatistic,
  type GPUNeighborSearchKernel,
  type GPUNeighborSearchWeightKind
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor
} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {addKernelPass} from './mode-kernels';
import {
  binToLattice,
  buildBlockGeometry,
  buildLatticeGeometry,
  createWeightsKit,
  FOCUS_SLOTS,
  isPointInBlock,
  WEIGHTS_TRANSFORM_OPTIONS,
  type WeightsKit,
  type WeightsKitSummary,
  type WeightsMatrixChoice,
  type WeightsTransformChoice
} from './spatial-weights-layers';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

/** Side length of the square cells the activity is binned into. */
const CELL_METERS = 120;
/** Neighbor-search lattice (compile-time; results do not depend on it). */
const GRID_SIZE: readonly [number, number] = [96, 96];
/** Compiled `k` choices. */
const K_CHOICES = [4, 6, 8, 12, 16] as const;
/** Slot capacity per row. Radius bands past this overflow (reported). */
const SLOTS_PER_ROW = 40;
const MAXIMUM_PERMUTATIONS = 999;
const HISTOGRAM_BINS = 24;
const CLASS_COUNT = 5;
const MAXIMUM_LOCAL_NEIGHBORS = 64;
const NOT_TESTED = 0xffffffff;
/** Slot capacity per row of the distance-band weights B (radius up to 300 m). */
const BAND_SLOTS_PER_ROW = 14;
/** Slot capacity per row of an algebra result (higher order and block weights grow quickly). */
const RESULT_SLOTS_PER_ROW = 64;
/** Side length of the blocks of the block weights, in cells. */
const BLOCK_CELLS = 5;
/** Edges of the cardinality histogram bins `[edge, next edge)`. */
const CARDINALITY_EDGES = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 65];
const CARDINALITY_BINS = CARDINALITY_EDGES.length - 1;
/** Slot capacity per row of the symmetrised union of the cell weights and their transpose. */
const UNION_SLOTS_PER_ROW = 48;
/** Target edge of a lattice cell, and the most cells along one axis. */
const LATTICE_CELL_METERS = 240;
const LATTICE_MAXIMUM_SIDE = 160;
/** Half side of the square window around the origin that the lattice and blocks cover. */
const FOCUS_HALF_EXTENT_METERS = 5000;
/** Queen radius 3 has 48 neighbors. */
const LATTICE_SLOTS_PER_ROW = 48;
/** Compiled lattice radii in cells. */
const LATTICE_RADII = [1, 2, 3] as const;
/** Nominal edge of a polygon block and its slot capacity per row. */
const BLOCK_METERS = 480;
const BLOCK_SLOTS_PER_ROW = 16;
const NEIGHBORHOOD_COLUMNS: readonly GPUNeighborhoodSummaryStatistic[] = [
  'mean',
  'sum',
  'min',
  'max',
  'standardDeviation',
  'median'
];

const STATISTICS_LENGTH = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length;
const PERMUTATION_RESULT_LENGTH = GPU_GLOBAL_PERMUTATION_RESULT.length;
// Summary layout (float32 words): statistics, permutation results, histogram, class counts, flags.
const SUMMARY_STATISTICS = 0;
const SUMMARY_PERMUTATION = SUMMARY_STATISTICS + STATISTICS_LENGTH;
const SUMMARY_HISTOGRAM = SUMMARY_PERMUTATION + PERMUTATION_RESULT_LENGTH;
const SUMMARY_CLASSES = SUMMARY_HISTOGRAM + HISTOGRAM_BINS;
const SUMMARY_FLAGS = SUMMARY_CLASSES + CLASS_COUNT;
const SUMMARY_TOOLKIT = SUMMARY_FLAGS + 3;
// Toolkit words: A statistics and counts, result statistics and counts, cardinality histogram,
// neighborhood extent, then result overflow, result total, band overflow, band total, neighborhood overflow.
const SUMMARY_STATISTICS_A = SUMMARY_TOOLKIT;
const SUMMARY_COUNTS_A = SUMMARY_STATISTICS_A + 3;
const SUMMARY_STATISTICS_C = SUMMARY_COUNTS_A + 5;
const SUMMARY_COUNTS_C = SUMMARY_STATISTICS_C + 3;
const SUMMARY_CARDINALITY = SUMMARY_COUNTS_C + 5;
const SUMMARY_EXTENT = SUMMARY_CARDINALITY + CARDINALITY_BINS;
const SUMMARY_TOOLKIT_FLAGS = SUMMARY_EXTENT + 2;
const SUMMARY_WORDS = SUMMARY_TOOLKIT_FLAGS + 5;

const FOCUS_COLORS: Record<WeightsMatrixChoice, SpatialAnalysisColor> = {
  weights: [255, 170, 40, 255],
  transpose: [255, 80, 200, 255],
  union: [80, 235, 140, 255]
};
const OVERLAY_COLORS: Record<WeightsMatrixChoice, SpatialAnalysisColor> = {
  weights: [255, 170, 40, 90],
  transpose: [255, 80, 200, 90],
  union: [80, 235, 140, 90]
};
const HIDDEN: SpatialAnalysisColor = [0, 0, 0, 0];
const NEUTRAL: SpatialAnalysisColor = [150, 160, 175, 150];
/** LISA classes: 1 HH, 2 LH, 3 LL, 4 HL. */
const LISA_COLORS: Record<number, SpatialAnalysisColor> = {
  1: [215, 48, 39, 255],
  2: [145, 191, 219, 245],
  3: [49, 54, 149, 255],
  4: [253, 174, 97, 245]
};

type NeighborChoice = `k${(typeof K_CHOICES)[number]}` | 'radius';
type WeightScheme = 'binary' | 'inverse' | 'inverse-squared' | 'gaussian' | 'bisquare';
type ValueColumn = 'trips' | 'pois';
type AlgebraOperation =
  | 'union'
  | 'intersection'
  | 'difference'
  | 'symmetricDifference'
  | 'higherOrder'
  | 'subgraph'
  | 'block';
type WeightsSource = 'cells' | 'lattice' | 'blocks';
type SourceMap = 'values' | 'lag' | 'neighbors';
type NeighborhoodStatistic =
  | 'mean'
  | 'sum'
  | 'min'
  | 'max'
  | 'standardDeviation'
  | 'median'
  | 'mode'
  | 'entropy';
type MapKind = 'lisa' | 'neighborhood' | 'cardinality' | 'lag';

const NEIGHBORHOOD_STATISTICS: readonly {value: NeighborhoodStatistic; label: string}[] = [
  {value: 'mean', label: 'Mean of neighbors (weighted)'},
  {value: 'sum', label: 'Sum (the spatial lag)'},
  {value: 'min', label: 'Minimum'},
  {value: 'max', label: 'Maximum'},
  {value: 'standardDeviation', label: 'Standard deviation'},
  {value: 'median', label: 'Median'},
  {value: 'mode', label: 'Dominant POI category (mode)'},
  {value: 'entropy', label: 'POI category entropy'}
];
const CATEGORY_COLORS: SpatialAnalysisColor[] = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
];

const WEIGHT_SCHEMES: Record<
  WeightScheme,
  {kind: GPUNeighborSearchWeightKind; power?: number; kernel?: GPUNeighborSearchKernel}
> = {
  binary: {kind: 'binary'},
  inverse: {kind: 'inverseDistance', power: 1},
  'inverse-squared': {kind: 'inverseDistance', power: 2},
  gaussian: {kind: 'kernel', kernel: 'gaussian'},
  bisquare: {kind: 'kernel', kernel: 'bisquare'}
};

type SearchVariant = {compiled: CompiledGPUCommandGraph<void>};

/** Bins trip vertices and points of interest into square cells; non-empty cells become rows. */
function binActivityCells(
  tripPositions: Float32Array,
  poiPositions: Float32Array,
  poiCategories: Uint32Array,
  categoryCount: number
): {
  positions: Float32Array;
  tripCounts: Float32Array;
  poiCounts: Float32Array;
  /** Most frequent POI category per cell; `categoryCount` when the cell has none. */
  dominantCategories: Uint32Array;
  /** Dense block ID of the `BLOCK_CELLS` x `BLOCK_CELLS` cell block containing each cell. */
  blockIds: Uint32Array;
  blockCount: number;
  bounds: [number, number, number, number];
} {
  const cells = new Map<
    number,
    {column: number; row: number; trips: number; pois: number; categories: Uint32Array}
  >();
  const getCell = (x: number, y: number) => {
    const column = Math.floor(x / CELL_METERS);
    const row = Math.floor(y / CELL_METERS);
    const key = (column + 32768) * 65536 + (row + 32768);
    let cell = cells.get(key);
    if (!cell) {
      cell = {column, row, trips: 0, pois: 0, categories: new Uint32Array(categoryCount + 1)};
      cells.set(key, cell);
    }
    return cell;
  };
  for (let index = 0; index < tripPositions.length; index += 2) {
    getCell(tripPositions[index], tripPositions[index + 1]).trips++;
  }
  for (let index = 0; index < poiPositions.length; index += 2) {
    const cell = getCell(poiPositions[index], poiPositions[index + 1]);
    cell.pois++;
    cell.categories[Math.min(categoryCount - 1, poiCategories[index / 2])]++;
  }
  const sorted = [...cells.entries()].sort((a, b) => a[0] - b[0]).map(entry => entry[1]);
  const positions = new Float32Array(sorted.length * 2);
  const tripCounts = new Float32Array(sorted.length);
  const poiCounts = new Float32Array(sorted.length);
  const dominantCategories = new Uint32Array(sorted.length);
  const blockIds = new Uint32Array(sorted.length);
  const blockKeys = new Map<number, number>();
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  sorted.forEach((cell, index) => {
    const x = (cell.column + 0.5) * CELL_METERS;
    const y = (cell.row + 0.5) * CELL_METERS;
    positions[index * 2] = x;
    positions[index * 2 + 1] = y;
    tripCounts[index] = cell.trips;
    poiCounts[index] = cell.pois;
    let best = categoryCount;
    let bestCount = 0;
    for (let category = 0; category < categoryCount; category++) {
      if (cell.categories[category] > bestCount) {
        bestCount = cell.categories[category];
        best = category;
      }
    }
    dominantCategories[index] = best;
    const blockKey =
      Math.floor(cell.column / BLOCK_CELLS) * 65536 + Math.floor(cell.row / BLOCK_CELLS);
    let block = blockKeys.get(blockKey);
    if (block === undefined) {
      block = blockKeys.size;
      blockKeys.set(blockKey, block);
    }
    blockIds[index] = block;
    minimumX = Math.min(minimumX, x);
    maximumX = Math.max(maximumX, x);
    minimumY = Math.min(minimumY, y);
    maximumY = Math.max(maximumY, y);
  });
  const margin = CELL_METERS;
  return {
    positions,
    tripCounts,
    poiCounts,
    dominantCategories,
    blockIds,
    blockCount: blockKeys.size,
    bounds: [minimumX - margin, minimumY - margin, maximumX + margin, maximumY + margin]
  };
}

/** WGSL body of the one-thread-per-slot link segment pass (finds the row by binary search). */
const SEGMENT_BODY = /* wgsl */ `
  let total = offsets[offsetsOffset + ROW_COUNT];
  if (index >= total) {
    segments[segmentsOffset + index * 4u] = 0.0;
    segments[segmentsOffset + index * 4u + 1u] = 0.0;
    segments[segmentsOffset + index * 4u + 2u] = 0.0;
    segments[segmentsOffset + index * 4u + 3u] = 0.0;
    fade[fadeOffset + index] = 0.0;
    return;
  }
  var low = 0u;
  var high = ROW_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  segments[segmentsOffset + index * 4u] = positions[positionsOffset + low * 2u];
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + low * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + neighbor * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + neighbor * 2u + 1u];
  fade[fadeOffset + index] = 1.0;`;

export const spatialWeightsMode: SpatialAnalysisModeDefinition = {
  id: 'spatial-weights',
  title: 'Weights',
  contributors: [
    'GPUNeighborSearch',
    'GPUGlobalSpatialStatistics',
    'GPULocalPermutationTest',
    'GPUGlobalPermutationTest',
    'GPUSpatialWeightsAlgebra',
    'GPUSpatialWeightsSummary',
    'GPUSpatialWeightsTransform',
    'GPUSpatialWeightsTranspose',
    'GPUSpatialLag',
    'GPULatticeWeights',
    'GPUContiguityWeights',
    'GPUNeighborhoodSummary'
  ],
  description:
    'Is taxi activity spatially clustered? Trip vertices and points of interest are binned into ' +
    'cells, a GPU neighbor search builds the weights matrix, and global Moran, Geary and General G ' +
    'are tested analytically and by permutation, with a local Moran cluster map. Combine the ' +
    'weights with a distance band (union, higher order, blocks), read S0/S1/S2 and cardinalities, ' +
    'and switch the map to a neighborhood statistic or to the result cardinality. Switch the ' +
    'weights source to a rook or queen lattice or to polygon contiguity, transform and lag any ' +
    "of them, compare kNN weights with their transpose, and hover the map to see one row's links.",
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const [trips, pois] = await Promise.all([
      context.data.getNewYorkTrips(),
      context.data.getNewYorkPointsOfInterest()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'spatial-weights');
    const cells = binActivityCells(
      trips.vertexPositions,
      pois.positions,
      pois.categories,
      pois.categoryNames.length
    );
    const cellCount = cells.positions.length / 2;
    const capacity = cellCount * SLOTS_PER_ROW;

    let neighborChoice: NeighborChoice = 'k8';
    let weightScheme: WeightScheme = 'inverse';
    let rowStandardize = true;
    let radiusMeters = 240;
    let capKnn = false;
    let valueColumn: ValueColumn = 'trips';
    let permutations = 499;
    let seed = 1;
    let significanceLevel = 0.05;
    let falseDiscoveryRate = false;
    let showEdges = true;
    let showNotSignificant = true;
    let linkCount = capacity;
    let resultLinkCount = 0;
    let bandRadius = 180;
    let algebraOperation: AlgebraOperation = 'union';
    let higherOrder = 2;
    let cumulative = false;
    let maskPercentile = 50;
    let weightTransform: WeightsTransformChoice = 'none';
    let source: WeightsSource = 'cells';
    let latticeCriterion: GPULatticeCriterion = 'queen';
    let latticeRadius = 1;
    let latticeThreshold = 1;
    let blockCriterion: GPUContiguityCriterion = 'queen';
    let matrixChoice: WeightsMatrixChoice = 'weights';
    let sourceMap: SourceMap = 'lag';
    let normalizeLag = false;
    let focusPinned = false;
    let dirtySource = true;
    let dirtyKit = true;
    let dirtyFocus = true;
    let kitNeedsReadback = true;
    let kitReadbackPending = false;
    let kitSummary: WeightsKitSummary | null = null;
    let transposeLinkCount = 0;
    let unionLinkCount = 0;
    let neighborhoodStatistic: NeighborhoodStatistic = 'mean';
    let mapKind: MapKind = 'lisa';
    let showResult = true;
    let neighborhoodRange: [number, number] = [0, 1];
    let cardinalityMaximum = 8;
    let dirtyAlgebra = true;
    let dirtyNeighborhood = true;
    let dirty = true;
    let needsReadback = true;
    let readbackPending = false;
    let destroyed = false;

    // Shared buffers: every search variant writes the same CSR, every analysis graph reads it.
    const positionsBuffer = resources.createBuffer('positions', cells.positions);
    const valuesBuffer = resources.createBuffer('values', cells.tripCounts);
    const offsetsBuffer = resources.createBuffer('offsets', (cellCount + 1) * 4);
    const neighborsBuffer = resources.createBuffer('neighbors', capacity * 4);
    const weightsBuffer = resources.createBuffer('weights', capacity * 4);
    const searchOverflowBuffer = resources.createBuffer('search-overflow', 4);
    const totalNeighborsBuffer = resources.createBuffer('total-neighbors', 4);
    const segmentsBuffer = resources.createBuffer('segments', capacity * 16);
    const segmentFadeBuffer = resources.createBuffer('segment-fade', capacity * 4);
    const statisticsBuffer = resources.createBuffer('statistics', STATISTICS_LENGTH * 4);
    const permutationResultBuffer = resources.createBuffer(
      'permutation-results',
      PERMUTATION_RESULT_LENGTH * 4
    );
    const referenceBuffer = resources.createBuffer('reference', MAXIMUM_PERMUTATIONS * 4);
    const histogramBuffer = resources.createBuffer('histogram', HISTOGRAM_BINS * 4);
    const exceedancesBuffer = resources.createBuffer('exceedances', cellCount * 4);
    const pseudoPValuesBuffer = resources.createBuffer('pseudo-p-values', cellCount * 4);
    const observedBuffer = resources.createBuffer('observed', cellCount * 4);
    const significantBuffer = resources.createBuffer('significant', cellCount * 4);
    const localOverflowBuffer = resources.createBuffer('local-overflow', 4);
    const classesBuffer = resources.createBuffer('classes', cellCount * 4);
    const classCountsBuffer = resources.createBuffer('class-counts', CLASS_COUNT * 4);
    const searchParameters = resources.createParameterBuffer(
      'search-parameters',
      'float32',
      GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
    );
    const permutationParameters = resources.createParameterBuffer(
      'permutation-parameters',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'spatial-weights-summary', byteLength: SUMMARY_WORDS * 4})
    );

    function importWeights(graph: GPUCommandGraph<void>) {
      return {
        offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', cellCount + 1),
        neighbors: importGraphBuffer(graph, 'neighbors', neighborsBuffer, 'uint32', capacity),
        weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', capacity)
      };
    }

    function compileSearch(mode: 'knn' | 'radius', k: number): SearchVariant {
      const graph = new GPUCommandGraph<void>(device, {id: `weights-search-${mode}-${k}`});
      const positions = importGraphBuffer(
        graph,
        'positions',
        positionsBuffer,
        'float32x2',
        cellCount
      );
      const weights = importWeights(graph);
      graph.add(
        new GPUNeighborSearch({
          id: 'neighbor-search',
          mode,
          k: mode === 'knn' ? k : undefined,
          gridSize: GRID_SIZE,
          positions,
          parameters: searchParameters.importToGraph(graph),
          weights,
          overflow: importGraphBuffer(graph, 'search-overflow', searchOverflowBuffer, 'uint32', 1),
          totalNeighbors: importGraphBuffer(
            graph,
            'total-neighbors',
            totalNeighborsBuffer,
            'uint32',
            1
          )
        })
      );
      // One thread per slot finds its row by binary search and writes the drawable segment.
      addKernelPass(graph, {
        id: 'weights-segments',
        invocationCount: capacity,
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {
            name: 'segments',
            view: importGraphBuffer(graph, 'segments', segmentsBuffer, 'float32', capacity * 4),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'fade',
            view: importGraphBuffer(graph, 'segment-fade', segmentFadeBuffer, 'float32', capacity),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const ROW_COUNT: u32 = ${cellCount}u;`,
        body: SEGMENT_BODY
      });
      return {compiled: resources.track(graph.compile())};
    }

    // Global statistics and the Moran reference distribution: one graph, no compile-time choices
    // that a control changes.
    const globalGraph = new GPUCommandGraph<void>(device, {id: 'weights-global'});
    {
      const weights = importWeights(globalGraph);
      const values = importGraphBuffer(globalGraph, 'values', valuesBuffer, 'float32', cellCount);
      globalGraph.add(
        new GPUGlobalSpatialStatistics({
          id: 'global-statistics',
          weights,
          values,
          statistics: ['moran', 'geary', 'getisOrdG'],
          results: importGraphBuffer(
            globalGraph,
            'statistics',
            statisticsBuffer,
            'float32',
            STATISTICS_LENGTH
          )
        })
      );
      globalGraph.add(
        new GPUGlobalPermutationTest({
          id: 'global-permutation',
          weights,
          values,
          statistic: 'moran',
          parameters: permutationParameters.importToGraph(globalGraph),
          maximumPermutations: MAXIMUM_PERMUTATIONS,
          results: importGraphBuffer(
            globalGraph,
            'permutation-results',
            permutationResultBuffer,
            'float32',
            PERMUTATION_RESULT_LENGTH
          ),
          referenceDistribution: importGraphBuffer(
            globalGraph,
            'reference',
            referenceBuffer,
            'float32',
            MAXIMUM_PERMUTATIONS
          ),
          histogram: importGraphBuffer(
            globalGraph,
            'histogram',
            histogramBuffer,
            'uint32',
            HISTOGRAM_BINS
          )
        })
      );
    }
    const globalCompiled = resources.track(globalGraph.compile());

    // The local test's Benjamini-Hochberg option is compile-time: compile both and pick.
    function compileLocal(fdr: boolean): SearchVariant {
      const graph = new GPUCommandGraph<void>(device, {
        id: `weights-local${fdr ? '-fdr' : ''}`
      });
      const weights = importWeights(graph);
      const values = importGraphBuffer(graph, 'values', valuesBuffer, 'float32', cellCount);
      const observed = importGraphBuffer(graph, 'observed', observedBuffer, 'float32', cellCount);
      const significant = importGraphBuffer(
        graph,
        'significant',
        significantBuffer,
        'uint32',
        cellCount
      );
      const classes = importGraphBuffer(graph, 'classes', classesBuffer, 'uint32', cellCount);
      graph.add(
        new GPULocalPermutationTest({
          id: 'local-permutation',
          weights,
          values,
          statistic: 'localMoran',
          parameters: permutationParameters.importToGraph(graph),
          maximumPermutations: MAXIMUM_PERMUTATIONS,
          maximumNeighbors: MAXIMUM_LOCAL_NEIGHBORS,
          exceedances: importGraphBuffer(
            graph,
            'exceedances',
            exceedancesBuffer,
            'uint32',
            cellCount
          ),
          pseudoPValues: importGraphBuffer(
            graph,
            'pseudo-p-values',
            pseudoPValuesBuffer,
            'float32',
            cellCount
          ),
          observed,
          significant,
          falseDiscoveryRate: fdr,
          overflow: importGraphBuffer(graph, 'local-overflow', localOverflowBuffer, 'uint32', 1)
        })
      );
      // LISA class from the sign of the local statistic and of the centered value.
      addKernelPass(graph, {
        id: 'lisa-classes',
        invocationCount: cellCount,
        bindings: [
          {name: 'observed', view: observed, type: 'f32', access: 'read'},
          {name: 'significant', view: significant, type: 'u32', access: 'read'},
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {
            name: 'statistics',
            view: importGraphBuffer(
              graph,
              'statistics',
              statisticsBuffer,
              'float32',
              STATISTICS_LENGTH
            ),
            type: 'f32',
            access: 'read'
          },
          {name: 'classes', view: classes, type: 'u32', access: 'read_write'}
        ],
        body: /* wgsl */ `
  var lisaClass = 0u;
  if (significant[significantOffset + index] != 0u) {
    let local = observed[observedOffset + index];
    let centered = values[valuesOffset + index] - statistics[statisticsOffset + ${GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY.mean}u];
    if (local > 0.0) {
      lisaClass = select(3u, 1u, centered > 0.0);
    } else if (local < 0.0) {
      lisaClass = select(2u, 4u, centered > 0.0);
    }
  }
  classes[classesOffset + index] = lisaClass;`
      });
      graph.add(
        new GPUHistogram({
          id: 'lisa-class-counts',
          input: classes,
          output: importGraphBuffer(
            graph,
            'class-counts',
            classCountsBuffer,
            'uint32',
            CLASS_COUNT
          ),
          edges: [0, 1, 2, 3, 4, 5]
        })
      );
      return {compiled: resources.track(graph.compile())};
    }

    const searchVariants = new Map<NeighborChoice, SearchVariant>();
    for (const k of K_CHOICES) {
      searchVariants.set(`k${k}`, compileSearch('knn', k));
    }
    searchVariants.set('radius', compileSearch('radius', 1));
    const localVariants = {plain: compileLocal(false), fdr: compileLocal(true)};

    const getActiveSearch = () => searchVariants.get(neighborChoice)!;
    const getActiveLocal = () => (falseDiscoveryRate ? localVariants.fdr : localVariants.plain);

    // ---- Weights toolkit: algebra, summary, transforms and neighborhood statistics -----------
    const bandCapacity = cellCount * BAND_SLOTS_PER_ROW;
    const resultCapacity = cellCount * RESULT_SLOTS_PER_ROW;
    const bandOffsetsBuffer = resources.createBuffer('band-offsets', (cellCount + 1) * 4);
    const bandNeighborsBuffer = resources.createBuffer('band-neighbors', bandCapacity * 4);
    const bandWeightsBuffer = resources.createBuffer('band-weights', bandCapacity * 4);
    const bandOverflowBuffer = resources.createBuffer('band-overflow', 4);
    const bandTotalBuffer = resources.createBuffer('band-total', 4);
    const resultOffsetsBuffer = resources.createBuffer('result-offsets', (cellCount + 1) * 4);
    const resultNeighborsBuffer = resources.createBuffer('result-neighbors', resultCapacity * 4);
    const resultWeightsBuffer = resources.createBuffer('result-weights', resultCapacity * 4);
    const resultOverflowBuffer = resources.createBuffer('result-overflow', 4);
    const resultTotalBuffer = resources.createBuffer('result-total', 4);
    const resultSegmentsBuffer = resources.createBuffer('result-segments', resultCapacity * 16);
    const resultFadeBuffer = resources.createBuffer('result-fade', resultCapacity * 4);
    const maskBuffer = resources.createBuffer('subgraph-mask', cellCount * 4);
    const groupIdsBuffer = resources.createBuffer('block-ids', cells.blockIds);
    const summaryStatisticsABuffer = resources.createBuffer('summary-statistics-a', 3 * 4);
    const summaryCountsABuffer = resources.createBuffer('summary-counts-a', 5 * 4);
    const summaryStatisticsCBuffer = resources.createBuffer('summary-statistics-c', 3 * 4);
    const summaryCountsCBuffer = resources.createBuffer('summary-counts-c', 5 * 4);
    const cardinalityBuffer = resources.createBuffer('cardinality', cellCount * 4);
    const cardinalityCountsBuffer = resources.createBuffer(
      'cardinality-counts',
      CARDINALITY_BINS * 4
    );
    const categoriesBuffer = resources.createBuffer('categories', cells.dominantCategories);
    const neighborhoodTableBuffer = resources.createBuffer(
      'neighborhood-table',
      cellCount * NEIGHBORHOOD_COLUMNS.length * 4
    );
    const neighborhoodModesBuffer = resources.createBuffer('neighborhood-modes', cellCount * 4);
    const neighborhoodEntropyBuffer = resources.createBuffer('neighborhood-entropy', cellCount * 4);
    const neighborhoodOverflowBuffer = resources.createBuffer('neighborhood-overflow', 4);
    const neighborhoodSourceBuffer = resources.createBuffer(
      'neighborhood-source',
      new Uint32Array([0])
    );
    const neighborhoodValuesBuffer = resources.createBuffer('neighborhood-values', cellCount * 4);
    const neighborhoodExtentBuffer = resources.createBuffer('neighborhood-extent', 2 * 4);
    const bandParameters = resources.createParameterBuffer(
      'band-parameters',
      'float32',
      GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
    );

    const importCsr = (
      graph: GPUCommandGraph<void>,
      name: string,
      csr: {
        offsets: typeof offsetsBuffer;
        neighbors: typeof offsetsBuffer;
        weights: typeof offsetsBuffer;
      },
      slots: number,
      rows = cellCount
    ) => ({
      offsets: importGraphBuffer(graph, `${name}-offsets`, csr.offsets, 'uint32', rows + 1),
      neighbors: importGraphBuffer(graph, `${name}-neighbors`, csr.neighbors, 'uint32', slots),
      weights: importGraphBuffer(graph, `${name}-weights`, csr.weights, 'float32', slots)
    });
    const weightsA = {offsets: offsetsBuffer, neighbors: neighborsBuffer, weights: weightsBuffer};
    const weightsB = {
      offsets: bandOffsetsBuffer,
      neighbors: bandNeighborsBuffer,
      weights: bandWeightsBuffer
    };
    const weightsC = {
      offsets: resultOffsetsBuffer,
      neighbors: resultNeighborsBuffer,
      weights: resultWeightsBuffer
    };

    // B: a distance band close to queen contiguity, written by its own radius search.
    const bandGraph = new GPUCommandGraph<void>(device, {id: 'weights-band'});
    bandGraph.add(
      new GPUNeighborSearch({
        id: 'band-search',
        mode: 'radius',
        gridSize: GRID_SIZE,
        positions: importGraphBuffer(
          bandGraph,
          'positions',
          positionsBuffer,
          'float32x2',
          cellCount
        ),
        parameters: bandParameters.importToGraph(bandGraph),
        weights: importCsr(bandGraph, 'b', weightsB, bandCapacity),
        overflow: importGraphBuffer(bandGraph, 'band-overflow', bandOverflowBuffer, 'uint32', 1),
        totalNeighbors: importGraphBuffer(bandGraph, 'band-total', bandTotalBuffer, 'uint32', 1)
      })
    );
    const bandCompiled = resources.track(bandGraph.compile());

    // Algebra: every operation, order and cumulative choice is a compiled variant.
    const compileAlgebra = (
      operation: AlgebraOperation,
      order: number,
      cumulative: boolean
    ): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {
        id: `weights-algebra-${operation}${operation === 'higherOrder' ? `-${order}-${cumulative}` : ''}`
      });
      const left = importCsr(graph, 'a', weightsA, capacity);
      const right = importCsr(graph, 'b', weightsB, bandCapacity);
      const common = {
        id: 'algebra',
        output: importCsr(graph, 'c', weightsC, resultCapacity),
        overflow: importGraphBuffer(graph, 'result-overflow', resultOverflowBuffer, 'uint32', 1),
        totalNeighbors: importGraphBuffer(graph, 'result-total', resultTotalBuffer, 'uint32', 1)
      };
      switch (operation) {
        case 'union':
        case 'intersection':
        case 'difference':
        case 'symmetricDifference':
          graph.add(new GPUSpatialWeightsAlgebra({...common, operation, left, right}));
          break;
        case 'higherOrder':
          graph.add(
            new GPUSpatialWeightsAlgebra({...common, operation, weights: left, order, cumulative})
          );
          break;
        case 'subgraph':
          graph.add(
            new GPUSpatialWeightsAlgebra({
              ...common,
              operation,
              weights: left,
              mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', cellCount)
            })
          );
          break;
        case 'block':
          graph.add(
            new GPUSpatialWeightsAlgebra({
              ...common,
              operation,
              groupIds: importGraphBuffer(graph, 'block-ids', groupIdsBuffer, 'uint32', cellCount),
              groupCount: cells.blockCount
            })
          );
          break;
      }
      return resources.track(graph.compile());
    };
    const algebraVariants = new Map<string, CompiledGPUCommandGraph<void>>();
    const getAlgebraKey = (operation: AlgebraOperation, order: number, cumulative: boolean) =>
      operation === 'higherOrder' ? `higherOrder-${order}-${cumulative}` : operation;
    for (const operation of [
      'union',
      'intersection',
      'difference',
      'symmetricDifference',
      'subgraph',
      'block'
    ] as const) {
      algebraVariants.set(getAlgebraKey(operation, 0, false), compileAlgebra(operation, 0, false));
    }
    for (const order of [2, 3]) {
      for (const cumulative of [false, true]) {
        algebraVariants.set(
          getAlgebraKey('higherOrder', order, cumulative),
          compileAlgebra('higherOrder', order, cumulative)
        );
      }
    }

    // Result: drawable links, summary statistics and the cardinality histogram of the result C.
    const resultGraph = new GPUCommandGraph<void>(device, {id: 'weights-result'});
    {
      const result = importCsr(resultGraph, 'c', weightsC, resultCapacity);
      const positions = importGraphBuffer(
        resultGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        cellCount
      );
      const cardinality = importGraphBuffer(
        resultGraph,
        'cardinality',
        cardinalityBuffer,
        'uint32',
        cellCount
      );
      resultGraph.add(
        new GPUSpatialWeightsSummary({
          id: 'result-summary',
          weights: result,
          statistics: importGraphBuffer(
            resultGraph,
            'summary-statistics-c',
            summaryStatisticsCBuffer,
            'float32',
            3
          ),
          counts: importGraphBuffer(
            resultGraph,
            'summary-counts-c',
            summaryCountsCBuffer,
            'uint32',
            5
          ),
          cardinality
        })
      );
      resultGraph.add(
        new GPUHistogram({
          id: 'cardinality-histogram',
          input: cardinality,
          output: importGraphBuffer(
            resultGraph,
            'cardinality-counts',
            cardinalityCountsBuffer,
            'uint32',
            CARDINALITY_BINS
          ),
          edges: CARDINALITY_EDGES
        })
      );
      addKernelPass(resultGraph, {
        id: 'result-segments',
        invocationCount: resultCapacity,
        bindings: [
          {name: 'offsets', view: result.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: result.neighbors, type: 'u32', access: 'read'},
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {
            name: 'segments',
            view: importGraphBuffer(
              resultGraph,
              'result-segments',
              resultSegmentsBuffer,
              'float32',
              resultCapacity * 4
            ),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'fade',
            view: importGraphBuffer(
              resultGraph,
              'result-fade',
              resultFadeBuffer,
              'float32',
              resultCapacity
            ),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const ROW_COUNT: u32 = ${cellCount}u;`,
        body: SEGMENT_BODY
      });
    }
    const resultCompiled = resources.track(resultGraph.compile());

    // Summary of A (after any transform).
    const summaryGraph = new GPUCommandGraph<void>(device, {id: 'weights-summary'});
    summaryGraph.add(
      new GPUSpatialWeightsSummary({
        id: 'weights-summary',
        weights: importCsr(summaryGraph, 'a', weightsA, capacity),
        statistics: importGraphBuffer(
          summaryGraph,
          'summary-statistics-a',
          summaryStatisticsABuffer,
          'float32',
          3
        ),
        counts: importGraphBuffer(
          summaryGraph,
          'summary-counts-a',
          summaryCountsABuffer,
          'uint32',
          5
        )
      })
    );
    const summaryCompiled = resources.track(summaryGraph.compile());

    // ---- Weights sources: the analysis kit (transforms, transpose, lag, focus links) ----------
    const projection = new LocalMetricProjection(trips.origin);
    // Focus starts at the row nearest the map center (the data origin).
    const centerX = 0;
    const centerY = 0;
    const findCentralRow = (rowPositions: Float32Array, accept: (row: number) => boolean) => {
      let best = 0;
      let bestDistance = Infinity;
      for (let row = 0; row < rowPositions.length / 2; row++) {
        if (!accept(row)) continue;
        const distance = Math.hypot(
          rowPositions[row * 2] - centerX,
          rowPositions[row * 2 + 1] - centerY
        );
        if (distance < bestDistance) {
          bestDistance = distance;
          best = row;
        }
      }
      return best;
    };

    type WeightsSourceState = {
      label: string;
      rowCount: number;
      positions: Buffer;
      values: Buffer;
      kit: WeightsKit;
      /** Current value column on the CPU, for tooltips and color ranges. */
      getValues: () => Float32Array;
      /** Row under planar meters `x, y`, or -1. */
      pickRow: (x: number, y: number) => number;
      /** Writes the weights (alternative sources only). */
      produce: (commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]) => void;
      /** Row-specific tooltip detail. */
      describeRow: (row: number) => string;
      focusRow: number;
      linkCapacity: number;
    };

    const cellIndexByKey = new Map<number, number>();
    for (let row = 0; row < cellCount; row++) {
      const column = Math.floor(cells.positions[row * 2] / CELL_METERS);
      const cellRow = Math.floor(cells.positions[row * 2 + 1] / CELL_METERS);
      cellIndexByKey.set((column + 32768) * 65536 + (cellRow + 32768), row);
    }
    const cellsSource: WeightsSourceState = {
      label: 'Cell',
      rowCount: cellCount,
      positions: positionsBuffer,
      values: valuesBuffer,
      kit: createWeightsKit({
        device,
        resources,
        id: 'cells',
        rowCount: cellCount,
        slots: capacity,
        unionSlots: cellCount * UNION_SLOTS_PER_ROW,
        positions: positionsBuffer,
        values: valuesBuffer,
        csr: weightsA,
        producer: {overflow: searchOverflowBuffer, total: totalNeighborsBuffer},
        summary: {statistics: summaryStatisticsABuffer, counts: summaryCountsABuffer},
        drawAllLinks: true
      }),
      getValues: () => (valueColumn === 'trips' ? cells.tripCounts : cells.poiCounts),
      pickRow: (x, y) =>
        cellIndexByKey.get(
          (Math.floor(x / CELL_METERS) + 32768) * 65536 + (Math.floor(y / CELL_METERS) + 32768)
        ) ?? -1,
      produce: () => {},
      describeRow: row => `${cells.tripCounts[row]} trip vertices, ${cells.poiCounts[row]} POIs`,
      focusRow: findCentralRow(cells.positions, () => true),
      linkCapacity: capacity
    };

    // The lattice and the blocks cover the core of the activity (the camera window), not the
    // sparse outskirts that stretch the data bounds.
    const focusBounds: [number, number, number, number] = [
      Math.max(cells.bounds[0], -FOCUS_HALF_EXTENT_METERS),
      Math.max(cells.bounds[1], -FOCUS_HALF_EXTENT_METERS),
      Math.min(cells.bounds[2], FOCUS_HALF_EXTENT_METERS),
      Math.min(cells.bounds[3], FOCUS_HALF_EXTENT_METERS)
    ];
    if (!(focusBounds[2] > focusBounds[0] && focusBounds[3] > focusBounds[1])) {
      focusBounds.splice(0, 4, ...cells.bounds);
    }
    const isInsideFocus = (row: number) =>
      cells.positions[row * 2] >= focusBounds[0] &&
      cells.positions[row * 2] < focusBounds[2] &&
      cells.positions[row * 2 + 1] >= focusBounds[1] &&
      cells.positions[row * 2 + 1] < focusBounds[3];
    // Lattice: a regular grid over the activity extent; a per-frame mask empties quiet cells.
    const lattice = buildLatticeGeometry(focusBounds, LATTICE_CELL_METERS, LATTICE_MAXIMUM_SIDE);
    const latticeRows = lattice.width * lattice.height;
    const latticeTrips = binToLattice(lattice, cells.positions, cells.tripCounts);
    const latticePois = binToLattice(lattice, cells.positions, cells.poiCounts);
    const latticeSlots = latticeRows * LATTICE_SLOTS_PER_ROW;
    const latticePositionsBuffer = resources.createBuffer('lattice-positions', lattice.positions);
    const latticeValuesBuffer = resources.createBuffer('lattice-values', latticeTrips);
    const latticeMaskBuffer = resources.createBuffer('lattice-mask', latticeRows * 4);
    const latticeCsr = {
      offsets: resources.createBuffer('lattice-offsets', (latticeRows + 1) * 4),
      neighbors: resources.createBuffer('lattice-neighbors', latticeSlots * 4),
      weights: resources.createBuffer('lattice-weights', latticeSlots * 4)
    };
    const latticeOverflowBuffer = resources.createBuffer('lattice-overflow', 4);
    const latticeTotalBuffer = resources.createBuffer('lattice-total', 4);
    const latticeProducers = new Map<string, CompiledGPUCommandGraph<void>>();
    for (const criterion of ['rook', 'queen'] as const) {
      for (const radius of LATTICE_RADII) {
        const graph = new GPUCommandGraph<void>(device, {id: `lattice-${criterion}-${radius}`});
        graph.add(
          new GPULatticeWeights({
            id: 'lattice',
            width: lattice.width,
            height: lattice.height,
            criterion,
            radius,
            cellSize: [lattice.cellMeters, lattice.cellMeters],
            mask: importGraphBuffer(graph, 'mask', latticeMaskBuffer, 'uint32', latticeRows),
            weights: importCsr(graph, 'l', latticeCsr, latticeSlots, latticeRows),
            overflow: importGraphBuffer(graph, 'overflow', latticeOverflowBuffer, 'uint32', 1),
            totalNeighbors: importGraphBuffer(graph, 'total', latticeTotalBuffer, 'uint32', 1)
          })
        );
        latticeProducers.set(`${criterion}-${radius}`, resources.track(graph.compile()));
      }
    }
    let latticeActive = 0;
    const writeLatticeMask = () => {
      const mask = new Uint32Array(latticeRows);
      latticeActive = 0;
      for (let row = 0; row < latticeRows; row++) {
        if (latticeTrips[row] + latticePois[row] >= latticeThreshold) {
          mask[row] = 1;
          latticeActive++;
        }
      }
      latticeMaskBuffer.write(mask);
      dirtySource = true;
      kitNeedsReadback = true;
    };
    writeLatticeMask();
    const latticeSource: WeightsSourceState = {
      label: 'Lattice cell',
      rowCount: latticeRows,
      positions: latticePositionsBuffer,
      values: latticeValuesBuffer,
      kit: createWeightsKit({
        device,
        resources,
        id: 'lattice',
        rowCount: latticeRows,
        slots: latticeSlots,
        unionSlots: latticeSlots,
        positions: latticePositionsBuffer,
        values: latticeValuesBuffer,
        csr: latticeCsr,
        producer: {overflow: latticeOverflowBuffer, total: latticeTotalBuffer},
        drawAllLinks: false
      }),
      getValues: () => (valueColumn === 'trips' ? latticeTrips : latticePois),
      pickRow: (x, y) => {
        const column = Math.floor((x - lattice.bounds[0]) / lattice.cellMeters);
        const row = Math.floor((y - lattice.bounds[1]) / lattice.cellMeters);
        return column >= 0 && column < lattice.width && row >= 0 && row < lattice.height
          ? row * lattice.width + column
          : -1;
      },
      produce: commandEncoder =>
        latticeProducers
          .get(`${latticeCriterion}-${latticeRadius}`)!
          .encode(commandEncoder, {parameters: undefined}),
      describeRow: row =>
        `(${row % lattice.width}, ${Math.floor(row / lattice.width)}), ${latticeTrips[row]} trips + ${latticePois[row]} POIs${
          latticeTrips[row] + latticePois[row] >= latticeThreshold ? '' : ' (masked out)'
        }`,
      focusRow: findCentralRow(
        lattice.positions,
        row => latticeTrips[row] + latticePois[row] >= latticeThreshold
      ),
      linkCapacity: latticeSlots
    };

    // Polygon blocks: jittered shared corners, kept where the block holds any activity.
    const activeBlocks = new Set<number>();
    const blockKey = (column: number, row: number) => column * 65536 + row;
    for (let row = 0; row < cellCount; row++) {
      if (!isInsideFocus(row)) continue;
      activeBlocks.add(
        blockKey(
          Math.floor((cells.positions[row * 2] - focusBounds[0]) / BLOCK_METERS),
          Math.floor((cells.positions[row * 2 + 1] - focusBounds[1]) / BLOCK_METERS)
        )
      );
    }
    const blocks = buildBlockGeometry(focusBounds, BLOCK_METERS, (column, row) =>
      activeBlocks.has(blockKey(column, row))
    );
    const blockCount = blocks.polygonCount;
    const blockIndexByKey = new Map<number, number>();
    for (let polygon = 0; polygon < blockCount; polygon++) {
      blockIndexByKey.set(
        blockKey(blocks.gridCells[polygon * 2], blocks.gridCells[polygon * 2 + 1]),
        polygon
      );
    }
    const blockTrips = new Float32Array(blockCount);
    const blockPois = new Float32Array(blockCount);
    for (let row = 0; row < cellCount; row++) {
      if (!isInsideFocus(row)) continue;
      const polygon = blockIndexByKey.get(
        blockKey(
          Math.floor((cells.positions[row * 2] - focusBounds[0]) / BLOCK_METERS),
          Math.floor((cells.positions[row * 2 + 1] - focusBounds[1]) / BLOCK_METERS)
        )
      )!;
      blockTrips[polygon] += cells.tripCounts[row];
      blockPois[polygon] += cells.poiCounts[row];
    }
    const blockSlots = blockCount * BLOCK_SLOTS_PER_ROW;
    const blockVerticesBuffer = resources.createBuffer('block-vertices', blocks.positions);
    const blockRingOffsetsBuffer = resources.createBuffer('block-ring-offsets', blocks.ringOffsets);
    const blockPolygonOffsetsBuffer = resources.createBuffer(
      'block-polygon-offsets',
      blocks.polygonOffsets
    );
    const blockCentroidsBuffer = resources.createBuffer('block-centroids', blocks.centroids);
    const blockOutlineBuffer = resources.createBuffer('block-outline', blocks.outline);
    const blockValuesBuffer = resources.createBuffer('block-values', blockTrips);
    const blockCsr = {
      offsets: resources.createBuffer('block-offsets', (blockCount + 1) * 4),
      neighbors: resources.createBuffer('block-neighbors', blockSlots * 4),
      weights: resources.createBuffer('block-weights', blockSlots * 4)
    };
    const blockOverflowBuffer = resources.createBuffer('block-overflow', 4);
    const blockTotalBuffer = resources.createBuffer('block-total', 4);
    const blockProducers = new Map<GPUContiguityCriterion, CompiledGPUCommandGraph<void>>();
    for (const criterion of ['rook', 'queen'] as const) {
      const graph = new GPUCommandGraph<void>(device, {id: `contiguity-${criterion}`});
      graph.add(
        new GPUContiguityWeights({
          id: 'contiguity',
          criterion,
          positions: importGraphBuffer(
            graph,
            'vertices',
            blockVerticesBuffer,
            'float32x2',
            blockCount * 4
          ),
          ringOffsets: importGraphBuffer(
            graph,
            'ring-offsets',
            blockRingOffsetsBuffer,
            'uint32',
            blockCount + 1
          ),
          polygonOffsets: importGraphBuffer(
            graph,
            'polygon-offsets',
            blockPolygonOffsetsBuffer,
            'uint32',
            blockCount + 1
          ),
          weights: importCsr(graph, 'p', blockCsr, blockSlots, blockCount),
          overflow: importGraphBuffer(graph, 'overflow', blockOverflowBuffer, 'uint32', 1),
          totalNeighbors: importGraphBuffer(graph, 'total', blockTotalBuffer, 'uint32', 1)
        })
      );
      blockProducers.set(criterion, resources.track(graph.compile()));
    }
    const blocksSource: WeightsSourceState = {
      label: 'Block',
      rowCount: blockCount,
      positions: blockCentroidsBuffer,
      values: blockValuesBuffer,
      kit: createWeightsKit({
        device,
        resources,
        id: 'blocks',
        rowCount: blockCount,
        slots: blockSlots,
        unionSlots: blockSlots,
        positions: blockCentroidsBuffer,
        values: blockValuesBuffer,
        csr: blockCsr,
        producer: {overflow: blockOverflowBuffer, total: blockTotalBuffer},
        drawAllLinks: false
      }),
      getValues: () => (valueColumn === 'trips' ? blockTrips : blockPois),
      pickRow: (x, y) => {
        const column = Math.floor((x - focusBounds[0]) / BLOCK_METERS);
        const row = Math.floor((y - focusBounds[1]) / BLOCK_METERS);
        // Corners are jittered by up to 0.275 blocks, so a point can sit in a nearby block's polygon.
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const polygon = blockIndexByKey.get(blockKey(column + dx, row + dy));
            if (polygon !== undefined && isPointInBlock(blocks, polygon, x, y)) return polygon;
          }
        }
        return -1;
      },
      produce: commandEncoder =>
        blockProducers.get(blockCriterion)!.encode(commandEncoder, {parameters: undefined}),
      describeRow: row => `${blockTrips[row]} trip vertices, ${blockPois[row]} POIs`,
      focusRow: findCentralRow(blocks.centroids, () => true),
      linkCapacity: blockSlots
    };
    const sources: Record<WeightsSource, WeightsSourceState> = {
      cells: cellsSource,
      lattice: latticeSource,
      blocks: blocksSource
    };
    for (const entry of Object.values(sources)) entry.kit.setFocusRow(entry.focusRow);
    const getActiveSource = () => sources[source];
    const getActiveMatrix = (): WeightsMatrixChoice =>
      source === 'cells' ? matrixChoice : 'weights';

    // Neighborhood statistics of the value column over A, one column chosen by a buffer write.
    const neighborhoodGraph = new GPUCommandGraph<void>(device, {id: 'weights-neighborhood'});
    {
      const table = importGraphBuffer(
        neighborhoodGraph,
        'neighborhood-table',
        neighborhoodTableBuffer,
        'float32',
        cellCount * NEIGHBORHOOD_COLUMNS.length
      );
      const modes = importGraphBuffer(
        neighborhoodGraph,
        'neighborhood-modes',
        neighborhoodModesBuffer,
        'uint32',
        cellCount
      );
      const entropy = importGraphBuffer(
        neighborhoodGraph,
        'neighborhood-entropy',
        neighborhoodEntropyBuffer,
        'float32',
        cellCount
      );
      const selected = importGraphBuffer(
        neighborhoodGraph,
        'neighborhood-values',
        neighborhoodValuesBuffer,
        'float32',
        cellCount
      );
      neighborhoodGraph.add(
        new GPUNeighborhoodSummary({
          id: 'neighborhood',
          weights: importCsr(neighborhoodGraph, 'a', weightsA, capacity),
          values: importGraphBuffer(
            neighborhoodGraph,
            'values',
            valuesBuffer,
            'float32',
            cellCount
          ),
          categories: importGraphBuffer(
            neighborhoodGraph,
            'categories',
            categoriesBuffer,
            'uint32',
            cellCount
          ),
          includeFocal: false,
          statistics: NEIGHBORHOOD_COLUMNS,
          output: table,
          maximumNeighbors: 64,
          overflow: importGraphBuffer(
            neighborhoodGraph,
            'neighborhood-overflow',
            neighborhoodOverflowBuffer,
            'uint32',
            1
          ),
          modes,
          entropy
        })
      );
      addKernelPass(neighborhoodGraph, {
        id: 'neighborhood-select',
        invocationCount: cellCount,
        bindings: [
          {name: 'table', view: table, type: 'f32', access: 'read'},
          {name: 'modes', view: modes, type: 'u32', access: 'read'},
          {name: 'entropy', view: entropy, type: 'f32', access: 'read'},
          {
            name: 'source',
            view: importGraphBuffer(
              neighborhoodGraph,
              'neighborhood-source',
              neighborhoodSourceBuffer,
              'uint32',
              1
            ),
            type: 'u32',
            access: 'read'
          },
          {name: 'selected', view: selected, type: 'f32', access: 'read_write'}
        ],
        body: /* wgsl */ `
  let choice = source[sourceOffset];
  var value = 0.0;
  if (choice < ${NEIGHBORHOOD_COLUMNS.length}u) {
    value = table[tableOffset + index * ${NEIGHBORHOOD_COLUMNS.length}u + choice];
  } else if (choice == ${NEIGHBORHOOD_COLUMNS.length}u) {
    let mode = modes[modesOffset + index];
    value = select(f32(mode), 0.0, mode == ${GPU_NEIGHBORHOOD_SUMMARY_NO_MODE}u);
  } else {
    value = entropy[entropyOffset + index];
  }
  // Rows without members hold quiet NaN: draw them as zero and keep the extent finite.
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    value = 0.0;
  }
  selected[selectedOffset + index] = value;`
      });
      neighborhoodGraph.add(
        new GPUReduction({
          id: 'neighborhood-extent',
          input: selected,
          output: importGraphBuffer(
            neighborhoodGraph,
            'neighborhood-extent',
            neighborhoodExtentBuffer,
            'float32',
            2
          ),
          operation: 'extent'
        })
      );
    }
    const neighborhoodCompiled = resources.track(neighborhoodGraph.compile());

    const getActiveAlgebra = () =>
      algebraVariants.get(getAlgebraKey(algebraOperation, higherOrder, cumulative))!;

    const writeSearchParameters = () => {
      const scheme = WEIGHT_SCHEMES[weightScheme];
      searchParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds: cells.bounds,
          radius: neighborChoice === 'radius' || capKnn ? radiusMeters : Infinity,
          weightKind: scheme.kind,
          power: scheme.power,
          distanceFloor: CELL_METERS / 2,
          kernel: scheme.kernel,
          rowStandardize
        })
      );
      dirty = true;
      needsReadback = true;
    };
    const writePermutationParameters = () => {
      permutationParameters.write(
        getGPUPermutationParameterValues({seed, permutations, significanceLevel})
      );
      dirty = true;
      needsReadback = true;
    };

    const writeBandParameters = () => {
      bandParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds: cells.bounds,
          radius: bandRadius,
          weightKind: 'binary'
        })
      );
      dirtyAlgebra = true;
      needsReadback = true;
    };
    /** Rows whose value is at or above the percentile stay in the subgraph. */
    const writeMask = () => {
      const column = valueColumn === 'trips' ? cells.tripCounts : cells.poiCounts;
      const sorted = Float32Array.from(column).sort();
      const threshold = sorted[Math.floor((maskPercentile / 100) * (sorted.length - 1))];
      const mask = new Uint32Array(cellCount);
      for (let row = 0; row < cellCount; row++) mask[row] = column[row] >= threshold ? 1 : 0;
      maskBuffer.write(mask);
      dirtyAlgebra = true;
      needsReadback = true;
    };

    const cellOnlyControls: {setDisabled: (disabled: boolean) => void}[] = [];
    const trackCellOnly = <T extends {setDisabled: (disabled: boolean) => void}>(handle: T): T => {
      cellOnlyControls.push(handle);
      return handle;
    };
    const latticeControls: {setDisabled: (disabled: boolean) => void}[] = [];
    const blockControls: {setDisabled: (disabled: boolean) => void}[] = [];
    const applySourceControls = () => {
      for (const handle of cellOnlyControls) handle.setDisabled(source !== 'cells');
      radiusControl.setDisabled(source !== 'cells' || (neighborChoice !== 'radius' && !capKnn));
      orderControl.setDisabled(source !== 'cells' || algebraOperation !== 'higherOrder');
      cumulativeControl.setDisabled(source !== 'cells' || algebraOperation !== 'higherOrder');
      maskControl.setDisabled(source !== 'cells' || algebraOperation !== 'subgraph');
      for (const handle of latticeControls) handle.setDisabled(source !== 'lattice');
      for (const handle of blockControls) handle.setDisabled(source !== 'blocks');
      sourceMapControl.setDisabled(source === 'cells');
      matrixControl.setDisabled(source !== 'cells');
    };
    const requestSourceRefresh = () => {
      // The producer, the transform and the analysis rerun; only buffers and variant picks change.
      if (source === 'cells') dirty = true;
      else dirtySource = true;
      needsReadback = true;
      kitNeedsReadback = true;
    };
    context.controls.addSelect<WeightsSource>({
      label: 'Weights source (all sources precompiled; switching is not a rebuild)',
      options: [
        {value: 'cells', label: 'Activity cells: GPUNeighborSearch (kNN or distance band)'},
        {value: 'lattice', label: 'Regular lattice: GPULatticeWeights (rook or queen)'},
        {value: 'blocks', label: 'Irregular blocks: GPUContiguityWeights (rook or queen)'}
      ],
      value: source,
      onChange: value => {
        source = value;
        requestSourceRefresh();
        dirtyKit = true;
        dirtyFocus = true;
        applySourceControls();
        context.updateLayers();
      }
    });
    latticeControls.push(
      context.controls.addSelect<GPULatticeCriterion>({
        label: 'Lattice criterion (compile-time: both precompiled)',
        options: [
          {value: 'queen', label: 'Queen (Chebyshev, 8 neighbors at radius 1)'},
          {value: 'rook', label: 'Rook (Manhattan, 4 neighbors at radius 1)'}
        ],
        value: latticeCriterion,
        onChange: value => {
          latticeCriterion = value;
          requestSourceRefresh();
        }
      }),
      context.controls.addSlider({
        label: 'Lattice radius in cells (compile-time: 1 to 3 precompiled)',
        min: 1,
        max: 3,
        step: 1,
        value: latticeRadius,
        format: value => `radius ${value}`,
        onChange: value => {
          latticeRadius = value;
          requestSourceRefresh();
        }
      }),
      context.controls.addSlider({
        label: 'Lattice activity threshold: quieter cells are masked out (buffer write)',
        min: 1,
        max: 40,
        step: 1,
        value: latticeThreshold,
        format: value => `>= ${value}`,
        onChange: value => {
          latticeThreshold = value;
          writeLatticeMask();
          requestSourceRefresh();
        }
      })
    );
    blockControls.push(
      context.controls.addSelect<GPUContiguityCriterion>({
        label: 'Block contiguity criterion (compile-time: both precompiled)',
        options: [
          {value: 'queen', label: 'Queen (share a vertex)'},
          {value: 'rook', label: 'Rook (share an edge)'}
        ],
        value: blockCriterion,
        onChange: value => {
          blockCriterion = value;
          requestSourceRefresh();
        }
      })
    );
    context.controls.addSelect<WeightsTransformChoice>({
      label: 'Transform the active weights in place (compile-time: all variants precompiled)',
      options: WEIGHTS_TRANSFORM_OPTIONS,
      value: weightTransform,
      onChange: value => {
        weightTransform = value;
        requestSourceRefresh();
      }
    });
    context.controls.addToggle({
      label: 'Normalize the spatial lag by the weight sum (compile-time: both precompiled)',
      value: normalizeLag,
      onChange: value => {
        normalizeLag = value;
        dirtyKit = true;
        kitNeedsReadback = true;
      }
    });
    const matrixControl = context.controls.addSelect<WeightsMatrixChoice>({
      label: 'Matrix drawn from the activity cells: W, its transpose, or the symmetrised union',
      options: [
        {value: 'weights', label: 'W (as built)'},
        {value: 'transpose', label: 'Transpose W′ (magenta)'},
        {value: 'union', label: 'Symmetrised union W ∪ W′ (green)'}
      ],
      value: matrixChoice,
      onChange: value => {
        matrixChoice = value;
        dirtyKit = true;
        kitNeedsReadback = true;
        context.updateLayers();
      }
    });
    const sourceMapControl = context.controls.addSelect<SourceMap>({
      label: 'Lattice and block map colors',
      options: [
        {value: 'lag', label: 'Spatial lag of the value column (GPUSpatialLag)'},
        {value: 'values', label: 'Value column'},
        {value: 'neighbors', label: 'Neighbor count (GPUSpatialWeightsSummary)'}
      ],
      value: sourceMap,
      onChange: value => {
        sourceMap = value;
        context.updateLayers();
      }
    });
    trackCellOnly(
      context.controls.addSelect<NeighborChoice>({
        label: 'Neighbors (k is compile-time: one search graph per k)',
        options: [
          ...K_CHOICES.map(k => ({value: `k${k}` as NeighborChoice, label: `k nearest, k = ${k}`})),
          {value: 'radius', label: 'Distance band (radius)'}
        ],
        value: neighborChoice,
        onChange: value => {
          neighborChoice = value;
          radiusControl.setDisabled(value !== 'radius' && !capKnn);
          writeSearchParameters();
        }
      })
    );
    const radiusControl = context.controls.addSlider({
      label: 'Distance band, or kNN distance cap (per-frame parameter)',
      min: 120,
      max: 480,
      step: 20,
      value: radiusMeters,
      format: value => `${value} m`,
      onChange: value => {
        radiusMeters = value;
        writeSearchParameters();
      }
    });
    radiusControl.setDisabled(true);
    trackCellOnly(
      context.controls.addToggle({
        label: 'Cap kNN at the distance above (per-frame)',
        value: capKnn,
        onChange: value => {
          capKnn = value;
          radiusControl.setDisabled(neighborChoice !== 'radius' && !capKnn);
          writeSearchParameters();
        }
      })
    );
    trackCellOnly(
      context.controls.addSelect<WeightScheme>({
        label: 'Weights (per-frame parameter)',
        options: [
          {value: 'binary', label: 'Binary'},
          {value: 'inverse', label: 'Inverse distance'},
          {value: 'inverse-squared', label: 'Inverse distance squared'},
          {value: 'gaussian', label: 'Gaussian kernel (adaptive bandwidth for kNN)'},
          {value: 'bisquare', label: 'Bisquare kernel'}
        ],
        value: weightScheme,
        onChange: value => {
          weightScheme = value;
          writeSearchParameters();
        }
      })
    );
    trackCellOnly(
      context.controls.addToggle({
        label: 'Row-standardize weights (per-frame)',
        value: rowStandardize,
        onChange: value => {
          rowStandardize = value;
          writeSearchParameters();
        }
      })
    );
    context.controls.addSelect<ValueColumn>({
      label: 'Value column (buffer write)',
      options: [
        {value: 'trips', label: 'Trip vertices per cell'},
        {value: 'pois', label: 'Points of interest per cell'}
      ],
      value: valueColumn,
      onChange: value => {
        valueColumn = value;
        valuesBuffer.write(value === 'trips' ? cells.tripCounts : cells.poiCounts);
        latticeValuesBuffer.write(value === 'trips' ? latticeTrips : latticePois);
        blockValuesBuffer.write(value === 'trips' ? blockTrips : blockPois);
        writeMask();
        dirty = true;
        dirtySource = true;
        dirtyKit = true;
        kitNeedsReadback = true;
        needsReadback = true;
      }
    });
    trackCellOnly(
      context.controls.addSlider({
        label: 'Permutations (per-frame parameter)',
        min: 99,
        max: MAXIMUM_PERMUTATIONS,
        step: 100,
        value: permutations,
        format: value => `${value}`,
        onChange: value => {
          permutations = value;
          writePermutationParameters();
        }
      })
    );
    trackCellOnly(
      context.controls.addSlider({
        label: 'Local significance level (per-frame parameter)',
        min: 0.001,
        max: 0.2,
        step: 0.001,
        value: significanceLevel,
        format: value => `p ≤ ${value.toFixed(3)}`,
        onChange: value => {
          significanceLevel = value;
          writePermutationParameters();
        }
      })
    );
    trackCellOnly(
      context.controls.addToggle({
        label: 'Benjamini-Hochberg FDR (compile-time option, both variants precompiled)',
        value: falseDiscoveryRate,
        onChange: value => {
          falseDiscoveryRate = value;
          dirty = true;
          needsReadback = true;
        }
      })
    );
    trackCellOnly(
      context.controls.addButton({
        label: 'New random seed',
        onClick: () => {
          seed++;
          writePermutationParameters();
        }
      })
    );
    trackCellOnly(
      context.controls.addToggle({
        label: 'Draw the weights graph',
        value: showEdges,
        onChange: value => {
          showEdges = value;
          context.updateLayers();
        }
      })
    );
    trackCellOnly(
      context.controls.addToggle({
        label: 'Show not significant cells',
        value: showNotSignificant,
        onChange: value => {
          showNotSignificant = value;
          context.updateLayers();
        }
      })
    );
    trackCellOnly(
      context.controls.addSelect<MapKind>({
        label: 'Map colors',
        options: [
          {value: 'lisa', label: 'Local Moran clusters (LISA)'},
          {value: 'neighborhood', label: 'Neighborhood statistic of A'},
          {value: 'cardinality', label: 'Neighbor count of the algebra result'},
          {value: 'lag', label: 'Spatial lag of the value column (GPUSpatialLag)'}
        ],
        value: mapKind,
        onChange: value => {
          mapKind = value;
          context.updateLayers();
        }
      })
    );
    trackCellOnly(
      context.controls.addSelect<NeighborhoodStatistic>({
        label: 'Neighborhood statistic (buffer write; all columns computed)',
        options: NEIGHBORHOOD_STATISTICS,
        value: neighborhoodStatistic,
        onChange: value => {
          neighborhoodStatistic = value;
          neighborhoodSourceBuffer.write(
            new Uint32Array([NEIGHBORHOOD_STATISTICS.findIndex(entry => entry.value === value)])
          );
          dirtyNeighborhood = true;
          needsReadback = true;
          context.updateLayers();
        }
      })
    );
    trackCellOnly(
      context.controls.addSelect<AlgebraOperation>({
        label: 'Algebra of A (k nearest) and B (distance band); compile-time, all precompiled',
        options: [
          {value: 'union', label: 'A ∪ B (union)'},
          {value: 'intersection', label: 'A ∩ B (intersection)'},
          {value: 'difference', label: 'A − B (difference)'},
          {value: 'symmetricDifference', label: 'A △ B (symmetric difference)'},
          {value: 'higherOrder', label: 'Higher order of A'},
          {value: 'subgraph', label: 'Subgraph of A (busy cells only)'},
          {value: 'block', label: `Block weights (${BLOCK_CELLS * CELL_METERS} m blocks)`}
        ],
        value: algebraOperation,
        onChange: value => {
          algebraOperation = value;
          orderControl.setDisabled(value !== 'higherOrder');
          cumulativeControl.setDisabled(value !== 'higherOrder');
          maskControl.setDisabled(value !== 'subgraph');
          dirtyAlgebra = true;
          needsReadback = true;
        }
      })
    );
    trackCellOnly(
      context.controls.addSlider({
        label: 'Band B radius (per-frame parameter)',
        min: 130,
        max: 300,
        step: 10,
        value: bandRadius,
        format: value => `${value} m`,
        onChange: value => {
          bandRadius = value;
          writeBandParameters();
        }
      })
    );
    const orderControl = context.controls.addSlider({
      label: 'Higher order (compile-time: 2 or 3)',
      min: 2,
      max: 3,
      step: 1,
      value: higherOrder,
      format: value => `order ${value}`,
      onChange: value => {
        higherOrder = value;
        dirtyAlgebra = true;
        needsReadback = true;
      }
    });
    const cumulativeControl = context.controls.addToggle({
      label: 'Cumulative: all orders up to this one (compile-time)',
      value: cumulative,
      onChange: value => {
        cumulative = value;
        dirtyAlgebra = true;
        needsReadback = true;
      }
    });
    const maskControl = context.controls.addSlider({
      label: 'Subgraph keeps cells above this percentile of the value column (buffer write)',
      min: 0,
      max: 95,
      step: 5,
      value: maskPercentile,
      format: value => `${value}%`,
      onChange: value => {
        maskPercentile = value;
        writeMask();
      }
    });
    orderControl.setDisabled(true);
    cumulativeControl.setDisabled(true);
    maskControl.setDisabled(true);
    trackCellOnly(
      context.controls.addToggle({
        label: 'Draw the algebra result (orange links, cluster map only)',
        value: showResult,
        onChange: value => {
          showResult = value;
          context.updateLayers();
        }
      })
    );
    context.controls.addLegend({
      title: 'Neighborhood statistic (viridis, over the current range)',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: 'low',
        maximumLabel: 'high'
      }
    });
    context.controls.addLegend({
      title: 'Local Moran cluster (LISA)',
      entries: [
        {color: LISA_COLORS[1], label: 'High-High'},
        {color: LISA_COLORS[3], label: 'Low-Low'},
        {color: LISA_COLORS[4], label: 'High-Low outlier'},
        {color: LISA_COLORS[2], label: 'Low-High outlier'},
        {color: NEUTRAL, label: 'Not significant'}
      ]
    });
    context.controls.addNote(
      `Cells are ${CELL_METERS} m squares holding at least one trip vertex or point of interest. ` +
        'Weights are asymmetric for kNN. The local test conditions on each cell and permutes the ' +
        'other cells among its neighbors; the global test relabels all values.'
    );
    context.controls.addReadout('Cells', formatCount(cellCount));
    const linksReadout = context.controls.addReadout('Weight links');
    const islandsReadout = context.controls.addReadout('Islands (no neighbors)');
    const overflowReadout = context.controls.addReadout('Capacity / untested rows');
    const moranReadout = context.controls.addReadout("Moran's I (E[I])");
    const moranNormalityReadout = context.controls.addReadout('Moran z, p (randomization)');
    const permutationReadout = context.controls.addReadout('Moran p_sim (z_sim)');
    const nullReadout = context.controls.addReadout('Permutation null (mean ± sd)');
    const histogramReadout = context.controls.addReadout('Null distribution');
    const gearyReadout = context.controls.addReadout("Geary's C z, p");
    const generalGReadout = context.controls.addReadout('General G z, p');
    const classReadout = context.controls.addReadout('HH / LL / HL / LH / n.s.');
    const summaryAReadout = context.controls.addReadout('A: S0 / S1 / S2');
    const summaryACountsReadout = context.controls.addReadout('A: asymmetric slots / isolates');
    const summaryCReadout = context.controls.addReadout('Result: S0 / S1 / S2');
    const summaryCCountsReadout = context.controls.addReadout(
      'Result: links / asymmetric / isolates'
    );
    const resultCardinalityReadout = context.controls.addReadout('Result cardinality min-max');
    const histogramCardinalityReadout = context.controls.addReadout('Result cardinality histogram');
    const resultOverflowReadout = context.controls.addReadout('Result / band overflow');
    const neighborhoodReadout = context.controls.addReadout('Neighborhood statistic range');
    context.controls.addNote(
      'Active source readouts (any weights source, after the transform): GPUSpatialWeightsSummary, ' +
        'GPUSpatialWeightsTranspose symmetry check and GPUSpatialLag.'
    );
    const sourceRowsReadout = context.controls.addReadout('Source rows / links');
    const sourceDegreeReadout = context.controls.addReadout('Neighbors min / mean / max');
    const sourceIslandsReadout = context.controls.addReadout(
      'Islands (no neighbors; masked lattice cells count)'
    );
    const sourceMomentsReadout = context.controls.addReadout('S0 / S1 / S2 (after transform)');
    const sourceAsymmetryReadout = context.controls.addReadout(
      'Asymmetric pairs: one-way links; weight-differing slots'
    );
    const sourceUnionReadout = context.controls.addReadout(
      'Union W \u222a W\u2032 links / overflow'
    );
    const sourceLagReadout = context.controls.addReadout('Spatial lag range');
    const sourceFocusReadout = context.controls.addReadout('Focus row: neighbors / weight sum');
    context.controls.addReadout('Data', `${trips.attribution}; ${pois.attribution}`);

    const searchTimingReadout = context.controls.addReadout('Search + segments (GPU)', '...');
    const globalTimingReadout = context.controls.addReadout(
      'Global statistics + p_sim (GPU)',
      '...'
    );
    const localTimingReadout = context.controls.addReadout('Local permutation test (GPU)', '...');
    context.controls.addButton({
      label: 'Measure the active graphs',
      onClick: () => {
        void (async () => {
          const options = {
            parameters: undefined,
            completionBuffer: searchOverflowBuffer,
            signal: context.signal
          };
          try {
            const search = getActiveSearch().compiled;
            const local = getActiveLocal().compiled;
            const timings = [
              await measureCompiledGraph(device, search, options),
              await measureCompiledGraph(device, globalCompiled, options),
              await measureCompiledGraph(device, local, options)
            ];
            if (destroyed) return;
            const format = (index: number, graph: typeof search) =>
              `${graph.stats.nodeOrder.length} nodes, ${formatCompiledGraphTiming(timings[index])}`;
            searchTimingReadout.setValue(format(0, search));
            globalTimingReadout.setValue(format(1, globalCompiled));
            localTimingReadout.setValue(format(2, local));
          } catch (error) {
            if (!destroyed) searchTimingReadout.setValue(`failed: ${String(error)}`);
          }
        })();
      }
    });

    writeSearchParameters();
    writePermutationParameters();
    writeBandParameters();
    writeMask();

    const formatP = (value: number) => (value < 0.001 ? '< 0.001' : value.toFixed(3));
    const formatNumber = (value: number, digits = 3) =>
      Number.isFinite(value) ? value.toFixed(digits) : 'n/a';

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const copy = (source: typeof valuesBuffer, wordOffset: number, words: number) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: source,
          destinationBuffer: ticket.buffer,
          destinationOffset: wordOffset * 4,
          size: words * 4
        });
      copy(statisticsBuffer, SUMMARY_STATISTICS, STATISTICS_LENGTH);
      copy(permutationResultBuffer, SUMMARY_PERMUTATION, PERMUTATION_RESULT_LENGTH);
      copy(histogramBuffer, SUMMARY_HISTOGRAM, HISTOGRAM_BINS);
      copy(classCountsBuffer, SUMMARY_CLASSES, CLASS_COUNT);
      copy(searchOverflowBuffer, SUMMARY_FLAGS, 1);
      copy(totalNeighborsBuffer, SUMMARY_FLAGS + 1, 1);
      copy(localOverflowBuffer, SUMMARY_FLAGS + 2, 1);
      copy(summaryStatisticsABuffer, SUMMARY_STATISTICS_A, 3);
      copy(summaryCountsABuffer, SUMMARY_COUNTS_A, 5);
      copy(summaryStatisticsCBuffer, SUMMARY_STATISTICS_C, 3);
      copy(summaryCountsCBuffer, SUMMARY_COUNTS_C, 5);
      copy(cardinalityCountsBuffer, SUMMARY_CARDINALITY, CARDINALITY_BINS);
      copy(neighborhoodExtentBuffer, SUMMARY_EXTENT, 2);
      copy(resultOverflowBuffer, SUMMARY_TOOLKIT_FLAGS, 1);
      copy(resultTotalBuffer, SUMMARY_TOOLKIT_FLAGS + 1, 1);
      copy(bandOverflowBuffer, SUMMARY_TOOLKIT_FLAGS + 2, 1);
      copy(bandTotalBuffer, SUMMARY_TOOLKIT_FLAGS + 3, 1);
      copy(neighborhoodOverflowBuffer, SUMMARY_TOOLKIT_FLAGS + 4, 1);
      ticket.markEncoded({byteOffset: 0, byteLength: SUMMARY_WORDS * 4});
      readbackPending = true;
      needsReadback = false;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const floats = new Float32Array(bytes.buffer, bytes.byteOffset, SUMMARY_WORDS);
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, SUMMARY_WORDS);
        const statistic = (block: number, field: number) =>
          floats[SUMMARY_STATISTICS + block + field];
        const moran = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran;
        const geary = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.geary;
        const generalG = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.getisOrdG;
        const field = GPU_GLOBAL_SPATIAL_STATISTIC_FIELD;
        const summary = GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY;

        const totalLinks = words[SUMMARY_FLAGS + 1];
        const searchOverflow = words[SUMMARY_FLAGS] !== 0;
        const localOverflow = words[SUMMARY_FLAGS + 2] !== 0;
        const storedLinks = Math.min(totalLinks, capacity);
        linksReadout.setValue(
          `${formatCount(storedLinks)} (mean degree ${(storedLinks / cellCount).toFixed(1)})`
        );
        islandsReadout.setValue(formatCount(statistic(0, summary.islandCount)));
        overflowReadout.setValue(
          `${searchOverflow ? 'OVERFLOW: reduce the band' : 'ok'} / ${
            localOverflow ? `rows over ${MAXIMUM_LOCAL_NEIGHBORS} neighbors skipped` : 'none'
          }`
        );
        if (storedLinks !== linkCount) {
          linkCount = storedLinks;
          context.updateLayers();
        }

        moranReadout.setValue(
          `${formatNumber(statistic(moran, field.statistic))} (${formatNumber(
            statistic(moran, field.expected),
            4
          )})`
        );
        moranNormalityReadout.setValue(
          `z ${formatNumber(statistic(moran, field.zRandomization), 2)}, p ${formatP(
            statistic(moran, field.pRandomization)
          )}`
        );
        gearyReadout.setValue(
          `C ${formatNumber(statistic(geary, field.statistic))}: z ${formatNumber(
            statistic(geary, field.zRandomization),
            2
          )}, p ${formatP(statistic(geary, field.pRandomization))}`
        );
        generalGReadout.setValue(
          `G ${formatNumber(statistic(generalG, field.statistic), 5)}: z ${formatNumber(
            statistic(generalG, field.zRandomization),
            2
          )}, p ${formatP(statistic(generalG, field.pRandomization))}`
        );

        const permutation = (index: number) => floats[SUMMARY_PERMUTATION + index];
        const result = GPU_GLOBAL_PERMUTATION_RESULT;
        permutationReadout.setValue(
          `${formatP(permutation(result.pseudoPValue))} (z ${formatNumber(
            permutation(result.zSimulated),
            1
          )}), P = ${permutation(result.permutations)}`
        );
        nullReadout.setValue(
          `${formatNumber(permutation(result.simulatedMean), 4)} ± ${formatNumber(
            permutation(result.simulatedStandardDeviation),
            4
          )}`
        );
        histogramReadout.setValue(
          formatHistogram(
            words.subarray(SUMMARY_HISTOGRAM, SUMMARY_HISTOGRAM + HISTOGRAM_BINS),
            permutation(result.observed),
            permutation(result.minimum),
            permutation(result.maximum)
          )
        );

        // Histogram rows 0..4 are classes 0 (none), 1 HH, 2 LH, 3 LL, 4 HL.
        const count = (index: number) => formatCount(words[SUMMARY_CLASSES + index]);
        classReadout.setValue(
          `${count(1)} / ${count(3)} / ${count(4)} / ${count(2)} / ${count(0)}`
        );

        // Weights toolkit: summaries of A and of the algebra result.
        const layout = GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT;
        const formatStatistics = (start: number) =>
          `${formatNumber(floats[start + layout.statistics.s0], 1)} / ${formatNumber(floats[start + layout.statistics.s1], 1)} / ${formatNumber(floats[start + layout.statistics.s2], 1)}`;
        const counts = (start: number, field: number) => words[start + field];
        summaryAReadout.setValue(formatStatistics(SUMMARY_STATISTICS_A));
        summaryACountsReadout.setValue(
          `${formatCount(counts(SUMMARY_COUNTS_A, layout.counts.asymmetricSlots))} / ${formatCount(counts(SUMMARY_COUNTS_A, layout.counts.isolates))}`
        );
        summaryCReadout.setValue(formatStatistics(SUMMARY_STATISTICS_C));
        const resultSlots = counts(SUMMARY_COUNTS_C, layout.counts.slots);
        summaryCCountsReadout.setValue(
          `${formatCount(resultSlots)} / ${formatCount(counts(SUMMARY_COUNTS_C, layout.counts.asymmetricSlots))} / ${formatCount(counts(SUMMARY_COUNTS_C, layout.counts.isolates))}`
        );
        const maximumCardinality = counts(SUMMARY_COUNTS_C, layout.counts.maximumCardinality);
        resultCardinalityReadout.setValue(
          `${counts(SUMMARY_COUNTS_C, layout.counts.minimumCardinality)} - ${maximumCardinality} (mean ${(resultSlots / cellCount).toFixed(1)})`
        );
        histogramCardinalityReadout.setValue(
          formatHistogram(
            words.subarray(SUMMARY_CARDINALITY, SUMMARY_CARDINALITY + CARDINALITY_BINS),
            NaN,
            0,
            1
          )
        );
        const resultTotal = words[SUMMARY_TOOLKIT_FLAGS + 1];
        resultOverflowReadout.setValue(
          `${words[SUMMARY_TOOLKIT_FLAGS] ? `OVERFLOW (needs ${formatCount(resultTotal)} of ${formatCount(resultCapacity)})` : 'ok'} / ${
            words[SUMMARY_TOOLKIT_FLAGS + 2] ? 'BAND OVERFLOW' : 'ok'
          }`
        );
        const nextResultLinks = Math.min(resultSlots, resultCapacity);
        const extentLow = floats[SUMMARY_EXTENT];
        const extentHigh = floats[SUMMARY_EXTENT + 1];
        neighborhoodReadout.setValue(
          `${formatNumber(extentLow)} to ${formatNumber(extentHigh)}${words[SUMMARY_TOOLKIT_FLAGS + 4] ? ' (median overflow: rows over 64 neighbors are NaN)' : ''}`
        );
        if (
          nextResultLinks !== resultLinkCount ||
          maximumCardinality !== cardinalityMaximum ||
          extentLow !== neighborhoodRange[0] ||
          extentHigh !== neighborhoodRange[1]
        ) {
          resultLinkCount = nextResultLinks;
          cardinalityMaximum = Math.max(1, maximumCardinality);
          neighborhoodRange = [extentLow, Math.max(extentHigh, extentLow + 1e-6)];
          context.updateLayers();
        }
      } catch {
        // The ring or device was destroyed while the read was in flight.
        needsReadback = true;
      } finally {
        readbackPending = false;
      }
    };

    const readKitSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const active = getActiveSource();
      const summaryPromise = active.kit.readSummary(commandEncoder);
      kitReadbackPending = true;
      kitNeedsReadback = false;
      try {
        const summary = await summaryPromise;
        if (destroyed) return;
        if (!summary) {
          kitNeedsReadback = true;
          return;
        }
        if (active !== getActiveSource()) {
          kitNeedsReadback = true;
          return;
        }
        kitSummary = summary;
        const slotCount = Math.min(summary.slots, active.linkCapacity);
        const mean = slotCount / active.rowCount;
        const activeNote = source === 'lattice' ? ` (${formatCount(latticeActive)} active)` : '';
        sourceRowsReadout.setValue(
          `${formatCount(active.rowCount)}${activeNote} / ${formatCount(slotCount)}${
            summary.producerOverflow ? ' OVERFLOW' : ''
          }`
        );
        sourceDegreeReadout.setValue(
          `${summary.minimumCardinality} / ${mean.toFixed(1)} / ${summary.maximumCardinality}`
        );
        sourceIslandsReadout.setValue(formatCount(summary.isolates));
        sourceMomentsReadout.setValue(
          `${formatNumber(summary.s0, 1)} / ${formatNumber(summary.s1, 1)} / ${formatNumber(summary.s2, 1)}`
        );
        // A one-way link is a slot whose reverse is missing; the union adds exactly those slots.
        const oneWayLinks = Math.max(0, summary.unionSlots - slotCount);
        sourceAsymmetryReadout.setValue(
          oneWayLinks === 0 && summary.transposeAsymmetricSlots === 0
            ? '0 (W equals its transpose)'
            : `${formatCount(oneWayLinks)} one-way links; ${formatCount(summary.transposeAsymmetricSlots)} of ${formatCount(summary.unionSlots)} union slots differ`
        );
        sourceUnionReadout.setValue(
          `${formatCount(Math.min(summary.unionSlots, source === 'cells' ? cellCount * UNION_SLOTS_PER_ROW : active.linkCapacity))} / ${
            summary.unionOverflow ? 'OVERFLOW' : 'ok'
          }`
        );
        sourceLagReadout.setValue(
          `${formatNumber(summary.lagMinimum, 1)} to ${formatNumber(summary.lagMaximum, 1)}`
        );
        const focus = summary.focus[getActiveMatrix()];
        sourceFocusReadout.setValue(
          `${active.label} ${active.focusRow}: ${focus.degree} / ${formatNumber(focus.weightSum, 3)}`
        );
        const nextTranspose = Math.min(summary.slots, capacity);
        const nextUnion = Math.min(summary.unionSlots, cellCount * UNION_SLOTS_PER_ROW);
        if (
          source === 'cells' &&
          (nextTranspose !== transposeLinkCount || nextUnion !== unionLinkCount)
        ) {
          transposeLinkCount = nextTranspose;
          unionLinkCount = nextUnion;
          context.updateLayers();
        }
      } catch {
        kitNeedsReadback = true;
      } finally {
        kitReadbackPending = false;
      }
    };

    const setFocus = (row: number): boolean => {
      const active = getActiveSource();
      if (row < 0 || row === active.focusRow) return false;
      active.focusRow = row;
      active.kit.setFocusRow(row);
      dirtyFocus = true;
      kitNeedsReadback = true;
      return true;
    };
    const getRowAt = (
      event: Parameters<NonNullable<SpatialAnalysisModeInstance['onClick']>>[0]
    ) => {
      if (!event.coordinate) return -1;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      return getActiveSource().pickRow(x, y);
    };

    const getCellLayers = (): Layer[] => {
      const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
      const layers: Layer[] = [];
      const showLinks = mapKind === 'lisa';
      if (showEdges && showLinks) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'weights-edges',
            coordinateOrigin,
            segments: segmentsBuffer,
            weights: segmentFadeBuffer,
            instanceCount: linkCount,
            widthPixels: 0.8,
            colormap: 'uniform',
            color: [120, 190, 255, 55]
          })
        );
      }
      const overlay = matrixChoice === 'weights' ? null : cellsSource.kit.allSegments[matrixChoice];
      if (showEdges && showLinks && overlay) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `weights-matrix-${matrixChoice}`,
            coordinateOrigin,
            segments: overlay.segments,
            weights: overlay.fade,
            instanceCount: matrixChoice === 'transpose' ? transposeLinkCount : unionLinkCount,
            widthPixels: 0.9,
            colormap: 'uniform',
            color: OVERLAY_COLORS[matrixChoice]
          })
        );
      }
      if (showResult && showLinks) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'weights-result-edges',
            coordinateOrigin,
            segments: resultSegmentsBuffer,
            weights: resultFadeBuffer,
            instanceCount: resultLinkCount,
            widthPixels: 0.8,
            colormap: 'uniform',
            color: [255, 150, 60, 60]
          })
        );
      }
      const palette = (visible: (index: number) => SpatialAnalysisColor) =>
        Array.from({length: 8}, (_, index) => visible(index));
      if (mapKind === 'neighborhood') {
        if (neighborhoodStatistic === 'mode') {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'weights-neighborhood-mode',
              coordinateOrigin,
              positions: positionsBuffer,
              instanceCount: cellCount,
              values: neighborhoodModesBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: CATEGORY_COLORS,
              noDataValue: GPU_NEIGHBORHOOD_SUMMARY_NO_MODE,
              noDataColor: HIDDEN,
              radiusPixels: 3.5
            })
          );
        } else {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'weights-neighborhood',
              coordinateOrigin,
              positions: positionsBuffer,
              instanceCount: cellCount,
              values: neighborhoodValuesBuffer,
              valueFormat: 'float32',
              colormap: 'viridis',
              valueRange: neighborhoodRange,
              radiusPixels: 3.5
            })
          );
        }
        return layers;
      }
      if (mapKind === 'lag') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'weights-lag',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: cellCount,
            values: cellsSource.kit.lag,
            valueFormat: 'float32',
            colormap: 'viridis',
            extent: cellsSource.kit.lagExtent,
            radiusPixels: 3.5
          })
        );
        return layers;
      }
      if (mapKind === 'cardinality') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'weights-cardinality',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: cellCount,
            values: cardinalityBuffer,
            valueFormat: 'uint32',
            colormap: 'viridis',
            valueRange: [0, cardinalityMaximum],
            radiusPixels: 3.5
          })
        );
        return layers;
      }
      const common = {
        coordinateOrigin,
        positions: positionsBuffer,
        instanceCount: cellCount,
        values: classesBuffer,
        valueFormat: 'uint32' as const,
        colormap: 'category' as const,
        noDataValue: NOT_TESTED,
        noDataColor: HIDDEN
      };
      if (showNotSignificant) {
        layers.push(
          new SpatialAnalysisPointLayer({
            ...common,
            id: 'weights-cells-insignificant',
            radiusPixels: 2.2,
            palette: palette(index => (index === 0 ? NEUTRAL : HIDDEN))
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          ...common,
          id: 'weights-cells-significant',
          radiusPixels: 4,
          palette: palette(index => LISA_COLORS[index] ?? HIDDEN)
        })
      );
      return layers;
    };

    const getAlternativeLayers = (): Layer[] => {
      const active = getActiveSource();
      const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
      const kit = active.kit;
      const layers: Layer[] = [];
      const column = active.getValues();
      let maximum = 1;
      for (const value of column) maximum = Math.max(maximum, value);
      const style =
        sourceMap === 'lag'
          ? {values: kit.lag, valueFormat: 'float32' as const, extent: kit.lagExtent}
          : sourceMap === 'neighbors'
            ? {
                values: kit.cardinality,
                valueFormat: 'uint32' as const,
                valueRange: [0, Math.max(1, kitSummary?.maximumCardinality ?? 8)] as const
              }
            : {
                values: active.values,
                valueFormat: 'float32' as const,
                valueRange: [0, maximum] as const,
                sqrtScale: true
              };
      if (source === 'lattice') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'weights-lattice',
            coordinateOrigin,
            gridSize: [lattice.width, lattice.height],
            bounds: lattice.bounds,
            colormap: 'viridis',
            discardAtOrBelow: 0,
            opacity: 0.75,
            ...style
          })
        );
      } else {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'weights-blocks-outline',
            coordinateOrigin,
            segments: blockOutlineBuffer,
            instanceCount: blockCount * 4,
            widthPixels: 1,
            colormap: 'uniform',
            color: [235, 240, 250, 140]
          }),
          new SpatialAnalysisPointLayer({
            id: 'weights-blocks',
            coordinateOrigin,
            positions: blockCentroidsBuffer,
            instanceCount: blockCount,
            colormap: 'viridis',
            radiusPixels: 7,
            ...style
          })
        );
      }
      return layers;
    };

    const getFocusLayers = (): Layer[] => {
      const active = getActiveSource();
      const matrix = getActiveMatrix();
      const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
      const kit = active.kit;
      return [
        new SpatialAnalysisSegmentLayer({
          id: 'weights-focus-links',
          coordinateOrigin,
          segments: kit.focusSegments[matrix],
          instanceCount: FOCUS_SLOTS,
          widthPixels: 2.5,
          colormap: 'uniform',
          color: FOCUS_COLORS[matrix]
        }),
        new SpatialAnalysisPointLayer({
          id: 'weights-focus-neighbors',
          coordinateOrigin,
          positions: active.positions,
          ids: kit.focusNeighbors[matrix],
          instanceCount: FOCUS_SLOTS,
          colormap: 'uniform',
          color: FOCUS_COLORS[matrix],
          radiusPixels: 4.5
        }),
        new SpatialAnalysisPointLayer({
          id: 'weights-focus-row',
          coordinateOrigin,
          positions: active.positions,
          ids: kit.focusRow,
          instanceCount: 1,
          colormap: 'uniform',
          color: [255, 255, 255, 255],
          radiusPixels: 7
        })
      ];
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [
        ...[...searchVariants.values()].map(variant => variant.compiled),
        globalCompiled,
        localVariants.plain.compiled,
        localVariants.fdr.compiled,
        bandCompiled,
        ...algebraVariants.values(),
        resultCompiled,
        summaryCompiled,
        neighborhoodCompiled,
        ...Object.values(sources).flatMap(entry => entry.kit.compiledGraphs()),
        ...latticeProducers.values(),
        ...blockProducers.values()
      ],
      encode(commandEncoder, frame) {
        // The data is static: results only change when a parameter, the value column, the seed or
        // a variant choice changed. Only the active source's pipeline runs.
        const first = frame.frameIndex < 2;
        const active = getActiveSource();
        if (source === 'cells') {
          const everything = dirty || first;
          if (everything) {
            getActiveSearch().compiled.encode(commandEncoder, {parameters: undefined});
            cellsSource.kit.encodeTransform(commandEncoder, weightTransform);
            globalCompiled.encode(commandEncoder, {parameters: undefined});
            getActiveLocal().compiled.encode(commandEncoder, {parameters: undefined});
            summaryCompiled.encode(commandEncoder, {parameters: undefined});
          }
          if (everything || dirtyAlgebra) {
            bandCompiled.encode(commandEncoder, {parameters: undefined});
            getActiveAlgebra().encode(commandEncoder, {parameters: undefined});
            resultCompiled.encode(commandEncoder, {parameters: undefined});
          }
          if (everything || dirtyNeighborhood) {
            neighborhoodCompiled.encode(commandEncoder, {parameters: undefined});
          }
          if (everything || dirtyKit) {
            active.kit.encodeAnalysis(commandEncoder, {
              normalizeLag,
              matrix: getActiveMatrix()
            });
          } else if (dirtyFocus) {
            active.kit.encodeFocus(commandEncoder);
          }
        } else if (dirtySource || first) {
          active.produce(commandEncoder);
          active.kit.encodeTransform(commandEncoder, weightTransform);
          active.kit.encodeAnalysis(commandEncoder, {normalizeLag, matrix: 'weights'});
        } else if (dirtyKit) {
          active.kit.encodeAnalysis(commandEncoder, {normalizeLag, matrix: 'weights'});
        } else if (dirtyFocus) {
          active.kit.encodeFocus(commandEncoder);
        }
        dirty = false;
        dirtySource = false;
        dirtyKit = false;
        dirtyFocus = false;
        dirtyAlgebra = false;
        dirtyNeighborhood = false;
        if (source === 'cells' && needsReadback && !readbackPending && frame.frameIndex >= 1) {
          void readSummary(commandEncoder);
        }
        if (kitNeedsReadback && !kitReadbackPending && frame.frameIndex >= 1) {
          void readKitSummary(commandEncoder);
        }
      },
      onClick(event) {
        const row = getRowAt(event);
        if (row < 0) {
          focusPinned = false;
          return false;
        }
        const active = getActiveSource();
        focusPinned = !(focusPinned && row === active.focusRow);
        setFocus(row);
        return true;
      },
      getTooltip(event) {
        const row = getRowAt(event);
        if (row < 0) return null;
        const active = getActiveSource();
        if (!focusPinned) setFocus(row);
        return `${active.label} ${row}: ${active.describeRow(row)}${
          focusPinned ? ' (focus pinned: click to release)' : ''
        }`;
      },
      getLayers() {
        const layers = source === 'cells' ? getCellLayers() : getAlternativeLayers();
        return [...layers, ...getFocusLayers()];
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};

const SPARK_CHARACTERS = '▁▂▃▄▅▆▇█';

/** Renders a permutation histogram as a sparkline and marks the observed statistic's bin. */
function formatHistogram(
  counts: Uint32Array,
  observed: number,
  minimum: number,
  maximum: number
): string {
  let largest = 1;
  for (const count of counts) largest = Math.max(largest, count);
  const characters = Array.from(counts, count =>
    count === 0 ? '·' : SPARK_CHARACTERS[Math.min(7, Math.floor((count / largest) * 7.999))]
  );
  if (!Number.isFinite(observed) || !(maximum > minimum)) return characters.join('');
  if (observed < minimum) return `◆ ${characters.join('')}`;
  if (observed > maximum) return `${characters.join('')} ◆`;
  const bin = Math.min(
    counts.length - 1,
    Math.floor(((observed - minimum) / (maximum - minimum)) * counts.length)
  );
  characters[bin] = '◆';
  return characters.join('');
}
