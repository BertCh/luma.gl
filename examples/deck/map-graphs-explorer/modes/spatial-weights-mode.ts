// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Spatial weights and autocorrelation of Manhattan taxi activity. Trip vertices and points of
 * interest are binned once into square cells; each non-empty cell is a point with a count value.
 * One pipeline of four recipes runs on the GPU from there:
 *
 * 1. `GPUNeighborSearch` writes a spatial-weights CSR (exact kNN, or every cell within a distance
 *    band) with a selectable weight function and row standardization.
 * 2. `GPUGlobalSpatialStatistics` reads the CSR: Moran's I, Geary's C and General G with z and p.
 * 3. `GPUGlobalPermutationTest` builds the Moran reference distribution (`p_sim`).
 * 4. `GPULocalPermutationTest` runs conditional permutations for local Moran (LISA) and writes
 *    pseudo p-values and a significance mask, optionally Benjamini-Hochberg corrected.
 *
 * `k` and the Benjamini-Hochberg choice are compile-time recipe options, so one search graph per
 * `k` and both local-test variants are compiled up front and the controls pick among them. All
 * other controls (distance band, weight function, row standardization, permutation count, seed,
 * significance level, value column) are buffer writes: the rebuild counter stays 0. The graphs are
 * encoded only when an input changed (the data is static) and the only readback is a ring-buffered
 * summary of about 400 bytes.
 */

import type {Layer} from '@deck.gl/core';
import {
  GPUCommandGraph,
  GPUHistogram,
  GPUReadbackRing,
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
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPUGlobalPermutationTest,
  GPUGlobalSpatialStatistics,
  GPULocalPermutationTest,
  GPUNeighborSearch,
  importGraphBuffer,
  type GPUNeighborSearchKernel,
  type GPUNeighborSearchWeightKind
} from '@luma.gl/experimental/map-graphs';
import {
  MapGraphsPointLayer,
  MapGraphsSegmentLayer,
  type MapGraphsColor
} from '../map-graphs-layers';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {addKernelPass} from './mode-kernels';
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

const STATISTICS_LENGTH = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length;
const PERMUTATION_RESULT_LENGTH = GPU_GLOBAL_PERMUTATION_RESULT.length;
// Summary layout (float32 words): statistics, permutation results, histogram, class counts, flags.
const SUMMARY_STATISTICS = 0;
const SUMMARY_PERMUTATION = SUMMARY_STATISTICS + STATISTICS_LENGTH;
const SUMMARY_HISTOGRAM = SUMMARY_PERMUTATION + PERMUTATION_RESULT_LENGTH;
const SUMMARY_CLASSES = SUMMARY_HISTOGRAM + HISTOGRAM_BINS;
const SUMMARY_FLAGS = SUMMARY_CLASSES + CLASS_COUNT;
const SUMMARY_WORDS = SUMMARY_FLAGS + 3;

const HIDDEN: MapGraphsColor = [0, 0, 0, 0];
const NEUTRAL: MapGraphsColor = [150, 160, 175, 150];
/** LISA classes: 1 HH, 2 LH, 3 LL, 4 HL. */
const LISA_COLORS: Record<number, MapGraphsColor> = {
  1: [215, 48, 39, 255],
  2: [145, 191, 219, 245],
  3: [49, 54, 149, 255],
  4: [253, 174, 97, 245]
};

type NeighborChoice = `k${(typeof K_CHOICES)[number]}` | 'radius';
type WeightScheme = 'binary' | 'inverse' | 'inverse-squared' | 'gaussian' | 'bisquare';
type ValueColumn = 'trips' | 'pois';

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
  poiPositions: Float32Array
): {
  positions: Float32Array;
  tripCounts: Float32Array;
  poiCounts: Float32Array;
  bounds: [number, number, number, number];
} {
  const cells = new Map<number, {column: number; row: number; trips: number; pois: number}>();
  const add = (positions: Float32Array, field: 'trips' | 'pois') => {
    for (let index = 0; index < positions.length; index += 2) {
      const column = Math.floor(positions[index] / CELL_METERS);
      const row = Math.floor(positions[index + 1] / CELL_METERS);
      const key = (column + 32768) * 65536 + (row + 32768);
      let cell = cells.get(key);
      if (!cell) {
        cell = {column, row, trips: 0, pois: 0};
        cells.set(key, cell);
      }
      cell[field]++;
    }
  };
  add(tripPositions, 'trips');
  add(poiPositions, 'pois');
  const sorted = [...cells.entries()].sort((a, b) => a[0] - b[0]).map(entry => entry[1]);
  const positions = new Float32Array(sorted.length * 2);
  const tripCounts = new Float32Array(sorted.length);
  const poiCounts = new Float32Array(sorted.length);
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
    bounds: [minimumX - margin, minimumY - margin, maximumX + margin, maximumY + margin]
  };
}

export const spatialWeightsMode: MapGraphsModeDefinition = {
  id: 'spatial-weights',
  title: 'Weights',
  recipes: [
    'GPUNeighborSearch',
    'GPUGlobalSpatialStatistics',
    'GPULocalPermutationTest',
    'GPUGlobalPermutationTest'
  ],
  description:
    'Is taxi activity spatially clustered? Trip vertices and points of interest are binned into ' +
    'cells, a GPU neighbor search builds the weights matrix, and global Moran, Geary and General G ' +
    'are tested analytically and by permutation, with a local Moran cluster map.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const [trips, pois] = await Promise.all([
      context.data.getNewYorkTrips(),
      context.data.getNewYorkPointsOfInterest()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new MapGraphsResources(device, 'spatial-weights');
    const cells = binActivityCells(trips.vertexPositions, pois.positions);
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
        body: /* wgsl */ `
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
  fade[fadeOffset + index] = 1.0;`
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
    });
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
    context.controls.addToggle({
      label: 'Cap kNN at the distance above (per-frame)',
      value: capKnn,
      onChange: value => {
        capKnn = value;
        radiusControl.setDisabled(neighborChoice !== 'radius' && !capKnn);
        writeSearchParameters();
      }
    });
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
    });
    context.controls.addToggle({
      label: 'Row-standardize weights (per-frame)',
      value: rowStandardize,
      onChange: value => {
        rowStandardize = value;
        writeSearchParameters();
      }
    });
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
        dirty = true;
        needsReadback = true;
      }
    });
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
    });
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
    });
    context.controls.addToggle({
      label: 'Benjamini-Hochberg FDR (compile-time option, both variants precompiled)',
      value: falseDiscoveryRate,
      onChange: value => {
        falseDiscoveryRate = value;
        dirty = true;
        needsReadback = true;
      }
    });
    context.controls.addButton({
      label: 'New random seed',
      onClick: () => {
        seed++;
        writePermutationParameters();
      }
    });
    context.controls.addToggle({
      label: 'Draw the weights graph',
      value: showEdges,
      onChange: value => {
        showEdges = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Show not significant cells',
      value: showNotSignificant,
      onChange: value => {
        showNotSignificant = value;
        context.updateLayers();
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

    const formatP = (value: number) => (value < 0.001 ? '< 0.001' : value.toFixed(3));
    const formatNumber = (value: number, digits = 3) =>
      Number.isFinite(value) ? value.toFixed(digits) : 'n/a';

    const readSummary = async (commandEncoder: Parameters<MapGraphsModeInstance['encode']>[0]) => {
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
      } catch {
        // The ring or device was destroyed while the read was in flight.
        needsReadback = true;
      } finally {
        readbackPending = false;
      }
    };

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [
        ...[...searchVariants.values()].map(variant => variant.compiled),
        globalCompiled,
        localVariants.plain.compiled,
        localVariants.fdr.compiled
      ],
      encode(commandEncoder, frame) {
        // The data is static: results only change when a parameter, the value column, the seed or
        // a variant choice changed.
        if (dirty || frame.frameIndex < 2) {
          getActiveSearch().compiled.encode(commandEncoder, {parameters: undefined});
          globalCompiled.encode(commandEncoder, {parameters: undefined});
          getActiveLocal().compiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
        }
        if (needsReadback && !readbackPending && frame.frameIndex >= 1) {
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        const layers: Layer[] = [];
        if (showEdges) {
          layers.push(
            new MapGraphsSegmentLayer({
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
        const palette = (visible: (index: number) => MapGraphsColor) =>
          Array.from({length: 8}, (_, index) => visible(index));
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
            new MapGraphsPointLayer({
              ...common,
              id: 'weights-cells-insignificant',
              radiusPixels: 2.2,
              palette: palette(index => (index === 0 ? NEUTRAL : HIDDEN))
            })
          );
        }
        layers.push(
          new MapGraphsPointLayer({
            ...common,
            id: 'weights-cells-significant',
            radiusPixels: 4,
            palette: palette(index => LISA_COLORS[index] ?? HIDDEN)
          })
        );
        return layers;
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
