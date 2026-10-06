// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Rates: small-area rate instability and its remedy, on San Francisco ZIP codes. Events are the
 * bike-parking sites that `GPUPointInPolygonJoin` counts in each ZIP code; the population at risk
 * is a deterministic synthetic field (area times a density with a few sparsely populated ZIPs),
 * because no population column ships with the data.
 *
 * 1. `GPUEmpiricalBayesRates` writes the raw rate, the empirical-Bayes smoothed rate and the
 *    Assuncao-Reis standardized rate (esda `Moran_Local_Rate`'s input).
 * 2. A kNN `GPUNeighborSearch` over the ZIP centroids builds row-standardized weights.
 * 3. `GPUGlobalSpatialStatistics` gives global Moran's I of the chosen variable and
 *    `GPULocalPermutationTest` runs conditional permutations for local Moran, with a selectable
 *    `alternative` tail (directed, two-sided, greater, lesser), into a cluster map.
 *
 * `k` and `alternative` are compile-time contributor options, so one search graph per `k` and one
 * analysis graph per tail are compiled up front and the controls pick among them. The variable,
 * the population spread, the permutation count, the seed and the significance level are buffer
 * writes. The ZIP codes are tiny, so every graph is encoded each time an input changes.
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, GPUHistogram, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUNeighborSearchParameterValues,
  getGPUPermutationParameterValues,
  GPU_EMPIRICAL_BAYES_SUMMARY,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPUEmpiricalBayesRates,
  GPUGlobalSpatialStatistics,
  GPULocalPermutationTest,
  GPUNeighborSearch,
  GPUPointInPolygonJoin,
  GPUSpatialEmpiricalBayesRates,
  type GPUPermutationAlternative
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection, type SpatialAnalysisPolygons} from '../spatial-analysis-data';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor
} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';

/** Compiled `k` choices of the centroid kNN weights. */
const K_CHOICES = [3, 4, 6] as const;
const MAXIMUM_K = 6;
const MAXIMUM_PERMUTATIONS = 999;
const MAXIMUM_LOCAL_NEIGHBORS = 16;
const CLASS_COUNT = 5;
const ALTERNATIVES: readonly GPUPermutationAlternative[] = [
  'directed',
  'two-sided',
  'greater',
  'lesser'
];

const STATISTICS_LENGTH = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length;
const HIDDEN: SpatialAnalysisColor = [0, 0, 0, 0];
const NEUTRAL: SpatialAnalysisColor = [150, 160, 175, 200];
/** LISA classes: 1 HH, 2 LH, 3 LL, 4 HL. */
const LISA_COLORS: Record<number, SpatialAnalysisColor> = {
  1: [215, 48, 39, 255],
  2: [145, 191, 219, 255],
  3: [49, 54, 149, 255],
  4: [253, 174, 97, 255]
};

type Variable = 'raw' | 'smoothed' | 'spatial' | 'standardized';
type MapKind = Variable | 'cluster';
type NeighborChoice = `k${(typeof K_CHOICES)[number]}`;

const VARIABLE_LABELS: Record<Variable, string> = {
  raw: 'Raw rate e / b',
  smoothed: 'Global empirical-Bayes rate',
  spatial: 'Spatial empirical-Bayes rate (neighborhood prior)',
  standardized: 'Standardized rate (Assuncao-Reis z)'
};
const VARIABLE_INDEX: Record<Variable, number> = {
  raw: 0,
  smoothed: 1,
  standardized: 2,
  spatial: 3
};
/** Share of ZIP codes, smallest populations first, used to read the shrinkage of noisy rates. */
const SMALL_POPULATION_SHARE = 0.2;

/** Planar centroid, area (square meters) and per-feature ring vertices of each ZIP code. */
function describeZipCodes(zips: SpatialAnalysisPolygons): {
  centroids: Float32Array;
  areas: Float32Array;
  bounds: [number, number, number, number];
} {
  const featureCount = zips.featureOffsets.length - 1;
  const centroids = new Float32Array(featureCount * 2);
  const areas = new Float32Array(featureCount);
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let feature = 0; feature < featureCount; feature++) {
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    let signedArea = 0;
    for (
      let polygon = zips.featureOffsets[feature];
      polygon < zips.featureOffsets[feature + 1];
      polygon++
    ) {
      for (
        let ring = zips.polygonOffsets[polygon];
        ring < zips.polygonOffsets[polygon + 1];
        ring++
      ) {
        const start = zips.ringOffsets[ring];
        const end = zips.ringOffsets[ring + 1];
        for (let vertex = start; vertex < end; vertex++) {
          const x = zips.polygonPositions[vertex * 2];
          const y = zips.polygonPositions[vertex * 2 + 1];
          const nextVertex = vertex + 1 < end ? vertex + 1 : start;
          const nextX = zips.polygonPositions[nextVertex * 2];
          const nextY = zips.polygonPositions[nextVertex * 2 + 1];
          signedArea += x * nextY - nextX * y;
          sumX += x;
          sumY += y;
          count++;
          minimumX = Math.min(minimumX, x);
          maximumX = Math.max(maximumX, x);
          minimumY = Math.min(minimumY, y);
          maximumY = Math.max(maximumY, y);
        }
      }
    }
    centroids[feature * 2] = sumX / Math.max(1, count);
    centroids[feature * 2 + 1] = sumY / Math.max(1, count);
    areas[feature] = Math.abs(signedArea) / 2;
  }
  const margin = 500;
  return {
    centroids,
    areas,
    bounds: [minimumX - margin, minimumY - margin, maximumX + margin, maximumY + margin]
  };
}

/** Deterministic pseudo-random number in [0, 1) for a feature row and a salt. */
function getHash(row: number, salt: number): number {
  let state = (Math.imul(row + 1, 2654435761) ^ Math.imul(salt + 7, 40503)) >>> 0;
  state = Math.imul(state ^ (state >>> 15), 2246822519) >>> 0;
  state = Math.imul(state ^ (state >>> 13), 3266489917) >>> 0;
  return ((state ^ (state >>> 16)) >>> 0) / 4294967296;
}

/**
 * Synthetic populations: area times a density between 3,000 and 18,000 per square kilometer, with
 * every fifth ZIP code sparsely populated. With `events` (known after the first GPU readback) the
 * log-population is blended toward the log event count by `coupling`, so sparse ZIP codes have few
 * events and noisy rates, which is what empirical-Bayes smoothing is for. `spread` stretches the
 * log-population around its mean.
 */
function makePopulations(
  areas: Float32Array,
  spread: number,
  events: Float32Array | null,
  coupling: number
): Float32Array {
  const populations = new Float32Array(areas.length);
  const logs = new Float64Array(areas.length);
  let logMean = 0;
  let eventMean = 0;
  for (let row = 0; row < areas.length; row++) {
    const density = 3000 + getHash(row, 1) * 15000;
    const sparse = row % 5 === 2 ? 0.08 : 1;
    logs[row] = Math.log(Math.max(1, (areas[row] / 1e6) * density * sparse));
    logMean += logs[row] / areas.length;
    if (events) eventMean += Math.log(events[row] + 1) / areas.length;
  }
  if (events) {
    logMean = 0;
    for (let row = 0; row < areas.length; row++) {
      const eventLog = Math.log(events[row] + 1) - eventMean;
      const density = 3000 + getHash(row, 1) * 15000;
      logs[row] = (1 - coupling) * logs[row] + coupling * (eventLog + Math.log(density * 15));
      logMean += logs[row] / areas.length;
    }
  }
  for (let row = 0; row < areas.length; row++) {
    populations[row] = Math.max(50, Math.round(Math.exp(logMean + (logs[row] - logMean) * spread)));
  }
  return populations;
}

type SummaryLayout = {
  raw: number;
  smoothed: number;
  standardized: number;
  spatial: number;
  events: number;
  populations: number;
  classes: number;
  pseudoP: number;
  summary: number;
  statistics: number;
  classCounts: number;
  flags: number;
  words: number;
};

export const ratesMode: SpatialAnalysisModeDefinition = {
  id: 'rates',
  title: 'Rates',
  contributors: [
    'GPUPointInPolygonJoin',
    'GPUEmpiricalBayesRates',
    'GPUSpatialEmpiricalBayesRates',
    'GPUNeighborSearch',
    'GPUGlobalSpatialStatistics',
    'GPULocalPermutationTest'
  ],
  description:
    'Bike-parking sites per synthetic resident in each San Francisco ZIP code. Switch Map colors ' +
    'between the raw rate, the global and the spatial (neighborhood prior) empirical-Bayes rate ' +
    'and read how much each shrinks small-population ZIPs; then read the local Moran cluster ' +
    'map, change the tail, the population spread or click a ZIP.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.6},

  async create(context) {
    const [parking, zips] = await Promise.all([
      context.data.getSanFranciscoBikeParking(),
      context.data.getSanFranciscoZipCodes()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'rates');
    const pointCount = parking.positions.length / 2;
    const featureCount = zips.featureOffsets.length - 1;
    const segmentCount = zips.outlineSegments.length / 4;
    const geometry = describeZipCodes(zips);
    const projection = new LocalMetricProjection(zips.origin);
    const capacity = featureCount * MAXIMUM_K;

    let neighborChoice: NeighborChoice = 'k4';
    let variable: Variable = 'standardized';
    let mapKind: MapKind = 'cluster';
    let alternative: GPUPermutationAlternative = 'two-sided';
    let populationSpread = 1;
    let populationCoupling = 0.7;
    let eventsKnown = false;
    let permutations = 499;
    let seed = 1;
    let significanceLevel = 0.05;
    let selectedRow = -1;
    let dirty = true;
    let needsReadback = true;
    const ranges: Record<Variable, [number, number]> = {
      raw: [0, 1],
      smoothed: [0, 1],
      spatial: [0, 1],
      standardized: [-2, 2]
    };
    let latest: {
      raw: Float32Array;
      smoothed: Float32Array;
      spatial: Float32Array;
      standardized: Float32Array;
      events: Float32Array;
      populations: Float32Array;
      pseudoP: Float32Array;
      classes: Uint32Array;
    } | null = null;

    // One-row position buffer of the selection marker, rewritten on every click.
    const markerBuffer = resources.createBuffer('selected-marker', new Float32Array(2));
    const populationValues = makePopulations(geometry.areas, populationSpread, null, 0);
    const centroidsBuffer = resources.createBuffer('centroids', geometry.centroids);
    const polygonPositions = resources.createBuffer('polygon-positions', zips.polygonPositions);
    const featureOffsets = resources.createBuffer('feature-offsets', zips.featureOffsets);
    const polygonOffsets = resources.createBuffer('polygon-offsets', zips.polygonOffsets);
    const ringOffsets = resources.createBuffer('ring-offsets', zips.ringOffsets);
    const parkingBuffer = resources.createBuffer('parking', parking.positions);
    const pointFeatures = resources.createBuffer('point-features', pointCount * 4);
    const featureCounts = resources.createBuffer('feature-counts', featureCount * 4);
    const joinOverflow = resources.createBuffer('join-overflow', 4);
    const eventsBuffer = resources.createBuffer('events', featureCount * 4);
    const populationsBuffer = resources.createBuffer('populations', populationValues);
    const rawBuffer = resources.createBuffer('raw-rates', featureCount * 4);
    const smoothedBuffer = resources.createBuffer('smoothed-rates', featureCount * 4);
    const spatialBuffer = resources.createBuffer('spatial-rates', featureCount * 4);
    const standardizedBuffer = resources.createBuffer('standardized-rates', featureCount * 4);
    const summaryBuffer = resources.createBuffer(
      'empirical-bayes-summary',
      GPU_EMPIRICAL_BAYES_SUMMARY.length * 4
    );
    const sourceBuffer = resources.createBuffer('variable-source', new Uint32Array([2]));
    const valuesBuffer = resources.createBuffer('values', featureCount * 4);
    const offsetsBuffer = resources.createBuffer('offsets', (featureCount + 1) * 4);
    const neighborsBuffer = resources.createBuffer('neighbors', capacity * 4);
    const weightsBuffer = resources.createBuffer('weights', capacity * 4);
    const searchOverflow = resources.createBuffer('search-overflow', 4);
    const totalNeighbors = resources.createBuffer('total-neighbors', 4);
    const statisticsBuffer = resources.createBuffer('statistics', STATISTICS_LENGTH * 4);
    const exceedancesBuffer = resources.createBuffer('exceedances', featureCount * 4);
    const pseudoPBuffer = resources.createBuffer('pseudo-p-values', featureCount * 4);
    const observedBuffer = resources.createBuffer('observed', featureCount * 4);
    const significantBuffer = resources.createBuffer('significant', featureCount * 4);
    const localOverflow = resources.createBuffer('local-overflow', 4);
    const classesBuffer = resources.createBuffer('classes', featureCount * 4);
    const classCountsBuffer = resources.createBuffer('class-counts', CLASS_COUNT * 4);
    const outlineSegments = resources.createBuffer('outline-segments', zips.outlineSegments);
    const outlineFeatureRows = resources.createBuffer(
      'outline-feature-rows',
      zips.outlineFeatureRows
    );
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

    // Join graph: events per ZIP code, converted to float32 for the rate contributor.
    const joinGraph = new GPUCommandGraph<void>(device, {id: 'rates-join'});
    const featureCountsView = importGraphBuffer(
      joinGraph,
      'feature-counts',
      featureCounts,
      'uint32',
      featureCount
    );
    joinGraph.add(
      new GPUPointInPolygonJoin({
        id: 'rates-polygon-join',
        points: importGraphBuffer(joinGraph, 'points', parkingBuffer, 'float32x2', pointCount),
        polygonPositions: importGraphBuffer(
          joinGraph,
          'polygon-positions',
          polygonPositions,
          'float32x2',
          zips.polygonPositions.length / 2
        ),
        featureOffsets: importGraphBuffer(
          joinGraph,
          'feature-offsets',
          featureOffsets,
          'uint32',
          zips.featureOffsets.length
        ),
        polygonOffsets: importGraphBuffer(
          joinGraph,
          'polygon-offsets',
          polygonOffsets,
          'uint32',
          zips.polygonOffsets.length
        ),
        ringOffsets: importGraphBuffer(
          joinGraph,
          'ring-offsets',
          ringOffsets,
          'uint32',
          zips.ringOffsets.length
        ),
        candidateCapacity: Math.max(1024, pointCount * 4),
        pointFeatureIds: importGraphBuffer(
          joinGraph,
          'point-features',
          pointFeatures,
          'uint32',
          pointCount
        ),
        featureCounts: featureCountsView,
        overflow: importGraphBuffer(joinGraph, 'join-overflow', joinOverflow, 'uint32', 1)
      })
    );
    addKernelPass(joinGraph, {
      id: 'rates-events-to-float',
      invocationCount: featureCount,
      bindings: [
        {
          name: 'counts',
          view: featureCountsView,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'events',
          view: importGraphBuffer(joinGraph, 'events', eventsBuffer, 'float32', featureCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `events[eventsOffset + index] = f32(counts[countsOffset + index]);`
    });
    const joinCompiled = resources.track(joinGraph.compile());

    // Rate graph: raw, smoothed and standardized rates, then the chosen variable is copied to
    // `values` for the spatial statistics (a buffer-write choice, no recompile).
    const rateGraph = new GPUCommandGraph<void>(device, {id: 'rates-empirical-bayes'});
    const standardized = importGraphBuffer(
      rateGraph,
      'standardized',
      standardizedBuffer,
      'float32',
      featureCount
    );
    const smoothed = importGraphBuffer(
      rateGraph,
      'smoothed',
      smoothedBuffer,
      'float32',
      featureCount
    );
    const raw = importGraphBuffer(rateGraph, 'raw', rawBuffer, 'float32', featureCount);
    rateGraph.add(
      new GPUEmpiricalBayesRates({
        id: 'rates-eb',
        events: importGraphBuffer(rateGraph, 'events', eventsBuffer, 'float32', featureCount),
        populations: importGraphBuffer(
          rateGraph,
          'populations',
          populationsBuffer,
          'float32',
          featureCount
        ),
        standardizedRates: standardized,
        smoothedRates: smoothed,
        rawRates: raw,
        summary: importGraphBuffer(
          rateGraph,
          'summary',
          summaryBuffer,
          'float32',
          GPU_EMPIRICAL_BAYES_SUMMARY.length
        )
      })
    );
    const rateCompiled = resources.track(rateGraph.compile());

    const importWeights = (graph: GPUCommandGraph<void>) => ({
      offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', featureCount + 1),
      neighbors: importGraphBuffer(graph, 'neighbors', neighborsBuffer, 'uint32', capacity),
      weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', capacity)
    });

    const compileSearch = (k: number): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {id: `rates-search-k${k}`});
      graph.add(
        new GPUNeighborSearch({
          id: 'rates-neighbors',
          mode: 'knn',
          k,
          gridSize: [32, 32],
          positions: importGraphBuffer(
            graph,
            'centroids',
            centroidsBuffer,
            'float32x2',
            featureCount
          ),
          parameters: searchParameters.importToGraph(graph),
          weights: importWeights(graph),
          overflow: importGraphBuffer(graph, 'search-overflow', searchOverflow, 'uint32', 1),
          totalNeighbors: importGraphBuffer(graph, 'total-neighbors', totalNeighbors, 'uint32', 1)
        })
      );
      return resources.track(graph.compile());
    };

    // The tail of the permutation test is compile-time: one analysis graph per alternative.
    const compileAnalysis = (tail: GPUPermutationAlternative): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {id: `rates-analysis-${tail}`});
      const weights = importWeights(graph);
      const values = importGraphBuffer(graph, 'values', valuesBuffer, 'float32', featureCount);
      const observed = importGraphBuffer(
        graph,
        'observed',
        observedBuffer,
        'float32',
        featureCount
      );
      const significant = importGraphBuffer(
        graph,
        'significant',
        significantBuffer,
        'uint32',
        featureCount
      );
      const classes = importGraphBuffer(graph, 'classes', classesBuffer, 'uint32', featureCount);
      const statistics = importGraphBuffer(
        graph,
        'statistics',
        statisticsBuffer,
        'float32',
        STATISTICS_LENGTH
      );
      graph.add(
        new GPUGlobalSpatialStatistics({
          id: 'rates-global',
          weights,
          values,
          statistics: ['moran'],
          results: statistics
        })
      );
      graph.add(
        new GPULocalPermutationTest({
          id: 'rates-local',
          weights,
          values,
          statistic: 'localMoran',
          alternative: tail,
          parameters: permutationParameters.importToGraph(graph),
          maximumPermutations: MAXIMUM_PERMUTATIONS,
          maximumNeighbors: MAXIMUM_LOCAL_NEIGHBORS,
          exceedances: importGraphBuffer(
            graph,
            'exceedances',
            exceedancesBuffer,
            'uint32',
            featureCount
          ),
          pseudoPValues: importGraphBuffer(
            graph,
            'pseudo-p',
            pseudoPBuffer,
            'float32',
            featureCount
          ),
          observed,
          significant,
          overflow: importGraphBuffer(graph, 'local-overflow', localOverflow, 'uint32', 1)
        })
      );
      // Cluster class from the sign of the local statistic and of the centered value.
      addKernelPass(graph, {
        id: 'rates-cluster-classes',
        invocationCount: featureCount,
        bindings: [
          {name: 'observed', view: observed, type: 'f32', access: 'read'},
          {name: 'significant', view: significant, type: 'u32', access: 'read'},
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'statistics', view: statistics, type: 'f32', access: 'read'},
          {name: 'classes', view: classes, type: 'u32', access: 'read_write'}
        ],
        body: /* wgsl */ `
  var clusterClass = 0u;
  if (significant[significantOffset + index] != 0u) {
    let local = observed[observedOffset + index];
    let centered = values[valuesOffset + index] - statistics[statisticsOffset + ${GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY.mean}u];
    if (local > 0.0) {
      clusterClass = select(3u, 1u, centered > 0.0);
    } else if (local < 0.0) {
      clusterClass = select(2u, 4u, centered > 0.0);
    }
  }
  classes[classesOffset + index] = clusterClass;`
      });
      graph.add(
        new GPUHistogram({
          id: 'rates-class-counts',
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
      return resources.track(graph.compile());
    };

    // Spatial graph (after the weights exist): neighborhood-prior empirical-Bayes rates, then the
    // chosen variable is copied to `values` for the spatial statistics (a buffer write).
    const spatialGraph = new GPUCommandGraph<void>(device, {id: 'rates-spatial-empirical-bayes'});
    const spatialRaw = importGraphBuffer(spatialGraph, 'raw', rawBuffer, 'float32', featureCount);
    const spatialSmoothed = importGraphBuffer(
      spatialGraph,
      'smoothed',
      smoothedBuffer,
      'float32',
      featureCount
    );
    const spatialStandardized = importGraphBuffer(
      spatialGraph,
      'standardized',
      standardizedBuffer,
      'float32',
      featureCount
    );
    const spatialRates = importGraphBuffer(
      spatialGraph,
      'spatial',
      spatialBuffer,
      'float32',
      featureCount
    );
    spatialGraph.add(
      new GPUSpatialEmpiricalBayesRates({
        id: 'rates-spatial-eb',
        events: importGraphBuffer(spatialGraph, 'events', eventsBuffer, 'float32', featureCount),
        populations: importGraphBuffer(
          spatialGraph,
          'populations',
          populationsBuffer,
          'float32',
          featureCount
        ),
        weights: importWeights(spatialGraph),
        smoothedRates: spatialRates
      })
    );
    addKernelPass(spatialGraph, {
      id: 'rates-select-variable',
      invocationCount: featureCount,
      bindings: [
        {name: 'raw', view: spatialRaw, type: 'f32', access: 'read'},
        {name: 'smoothed', view: spatialSmoothed, type: 'f32', access: 'read'},
        {name: 'standardized', view: spatialStandardized, type: 'f32', access: 'read'},
        {name: 'spatial', view: spatialRates, type: 'f32', access: 'read'},
        {
          name: 'source',
          view: importGraphBuffer(spatialGraph, 'source', sourceBuffer, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'values',
          view: importGraphBuffer(spatialGraph, 'values', valuesBuffer, 'float32', featureCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let choice = source[sourceOffset];
  var value = standardized[standardizedOffset + index];
  if (choice == 0u) {
    value = raw[rawOffset + index];
  } else if (choice == 1u) {
    value = smoothed[smoothedOffset + index];
  } else if (choice == 3u) {
    value = spatial[spatialOffset + index];
  }
  values[valuesOffset + index] = value;`
    });
    const spatialCompiled = resources.track(spatialGraph.compile());

    const searchVariants = new Map<NeighborChoice, CompiledGPUCommandGraph<void>>();
    for (const k of K_CHOICES) searchVariants.set(`k${k}`, compileSearch(k));
    const analysisVariants = new Map<GPUPermutationAlternative, CompiledGPUCommandGraph<void>>();
    for (const tail of ALTERNATIVES) analysisVariants.set(tail, compileAnalysis(tail));

    const writeSearch = () => {
      searchParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds: geometry.bounds,
          weightKind: 'binary',
          rowStandardize: true
        })
      );
      dirty = true;
      needsReadback = true;
    };
    const writePermutation = () => {
      permutationParameters.write(
        getGPUPermutationParameterValues({seed, permutations, significanceLevel})
      );
      dirty = true;
      needsReadback = true;
    };

    const writePopulations = () => {
      populationsBuffer.write(
        makePopulations(
          geometry.areas,
          populationSpread,
          eventsKnown && latest ? latest.events : null,
          populationCoupling
        )
      );
      dirty = true;
      needsReadback = true;
    };

    context.controls.addSelect<Variable>({
      label: 'Moran variable (buffer write)',
      options: (Object.keys(VARIABLE_LABELS) as Variable[]).map(value => ({
        value,
        label: VARIABLE_LABELS[value]
      })),
      value: variable,
      onChange: value => {
        variable = value;
        sourceBuffer.write(new Uint32Array([VARIABLE_INDEX[value]]));
        dirty = true;
        needsReadback = true;
      }
    });
    context.controls.addSelect<MapKind>({
      label: 'Map colors',
      options: [
        {value: 'cluster', label: 'Local Moran clusters'},
        {value: 'raw', label: 'Raw rate'},
        {value: 'smoothed', label: 'Global empirical-Bayes rate'},
        {value: 'spatial', label: 'Spatial empirical-Bayes rate'},
        {value: 'standardized', label: 'Standardized rate'}
      ],
      value: mapKind,
      onChange: value => {
        mapKind = value;
        context.updateLayers();
      }
    });
    context.controls.addSelect<GPUPermutationAlternative>({
      label: 'Permutation tail (compile-time: all four graphs precompiled)',
      options: [
        {value: 'directed', label: 'directed (legacy esda default)'},
        {value: 'two-sided', label: 'two-sided'},
        {value: 'greater', label: 'greater (clusters)'},
        {value: 'lesser', label: 'lesser (outliers)'}
      ],
      value: alternative,
      onChange: value => {
        alternative = value;
        dirty = true;
        needsReadback = true;
      }
    });
    context.controls.addSelect<NeighborChoice>({
      label: 'Neighbors (k is compile-time: one search graph per k)',
      options: K_CHOICES.map(k => ({
        value: `k${k}` as NeighborChoice,
        label: `k nearest, k = ${k}`
      })),
      value: neighborChoice,
      onChange: value => {
        neighborChoice = value;
        dirty = true;
        needsReadback = true;
      }
    });
    context.controls.addSlider({
      label: 'Population spread (buffer write; 0 = equal populations)',
      min: 0,
      max: 2,
      step: 0.1,
      value: populationSpread,
      format: value => `${value.toFixed(1)}x`,
      onChange: value => {
        populationSpread = value;
        writePopulations();
      }
    });
    context.controls.addSlider({
      label: 'Population follows events (buffer write; 0 = independent density)',
      min: 0,
      max: 1,
      step: 0.1,
      value: populationCoupling,
      format: value => `${value.toFixed(1)}`,
      onChange: value => {
        populationCoupling = value;
        writePopulations();
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
        writePermutation();
      }
    });
    context.controls.addSlider({
      label: 'Significance level (per-frame parameter)',
      min: 0.01,
      max: 0.2,
      step: 0.01,
      value: significanceLevel,
      format: value => `p ≤ ${value.toFixed(2)}`,
      onChange: value => {
        significanceLevel = value;
        writePermutation();
      }
    });
    context.controls.addButton({
      label: 'New random seed',
      onClick: () => {
        seed++;
        writePermutation();
      }
    });
    context.controls.addLegend({
      title: 'Local Moran cluster',
      entries: [
        {color: LISA_COLORS[1], label: 'High-High'},
        {color: LISA_COLORS[3], label: 'Low-Low'},
        {color: LISA_COLORS[4], label: 'High-Low outlier'},
        {color: LISA_COLORS[2], label: 'Low-High outlier'},
        {color: NEUTRAL, label: 'Not significant'}
      ]
    });
    context.controls.addLegend({
      title: 'Rate maps (viridis): low to high over the current range',
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
    context.controls.addNote(
      'Events are bike-parking sites per ZIP code (GPU point-in-polygon join); populations are ' +
        'synthetic. Raw rates of sparsely populated ZIPs swing widely; the empirical-Bayes rate ' +
        'shrinks them toward the pooled rate and the spatial empirical-Bayes rate toward the ' +
        'pooled rate of their neighbors (esda Spatial_Empirical_Bayes); the standardized rate is what local Moran should ' +
        'test (esda Moran_Local_Rate). Click a ZIP for its numbers.'
    );
    context.controls.addReadout(
      'ZIP codes / sites',
      `${formatCount(featureCount)} / ${formatCount(pointCount)}`
    );
    const pooledReadout = context.controls.addReadout('Pooled rate (per 1,000)');
    const varianceReadout = context.controls.addReadout('Std of rates: raw / global / spatial');
    const smallStdReadout = context.controls.addReadout(
      'Std, smallest 20%: raw / global / spatial'
    );
    const priorReadout = context.controls.addReadout('Prior variance a / weighted s²');
    const extremeReadout = context.controls.addReadout(
      'Smallest populations: raw → global → spatial'
    );
    const moranReadout = context.controls.addReadout("Global Moran's I (E[I]), z, p");
    const classReadout = context.controls.addReadout('HH / LL / HL / LH / n.s.');
    const flagsReadout = context.controls.addReadout('Weights links / overflow');
    const selectedReadout = context.controls.addReadout('Selected ZIP (click)', 'none');
    context.controls.addReadout(
      'Data',
      `${parking.attribution}; ${zips.attribution}; synthetic population`
    );

    writeSearch();
    writePermutation();

    const layout: SummaryLayout = (() => {
      let cursor = 0;
      const take = (words: number) => {
        const start = cursor;
        cursor += words;
        return start;
      };
      return {
        raw: take(featureCount),
        smoothed: take(featureCount),
        standardized: take(featureCount),
        spatial: take(featureCount),
        events: take(featureCount),
        populations: take(featureCount),
        classes: take(featureCount),
        pseudoP: take(featureCount),
        summary: take(GPU_EMPIRICAL_BAYES_SUMMARY.length),
        statistics: take(STATISTICS_LENGTH),
        classCounts: take(CLASS_COUNT),
        flags: take(3),
        words: 0
      };
    })();
    layout.words = layout.flags + 3;

    const formatNumber = (value: number, digits = 3) =>
      Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
    const formatP = (value: number) => (value < 0.001 ? '< 0.001' : value.toFixed(3));
    const getStandardDeviation = (values: Float32Array) => {
      let count = 0;
      let mean = 0;
      for (const value of values) {
        if (Number.isFinite(value)) {
          count++;
          mean += value;
        }
      }
      mean /= Math.max(1, count);
      let sum = 0;
      for (const value of values) if (Number.isFinite(value)) sum += (value - mean) ** 2;
      return Math.sqrt(sum / Math.max(1, count));
    };
    const getRange = (values: Float32Array, symmetric: boolean): [number, number] => {
      const finite = Array.from(values)
        .filter(Number.isFinite)
        .sort((a, b) => a - b);
      if (finite.length === 0) return [0, 1];
      const low = finite[Math.floor(0.02 * (finite.length - 1))];
      const high = finite[Math.floor(0.98 * (finite.length - 1))];
      if (symmetric) {
        const bound = Math.max(1e-6, Math.abs(low), Math.abs(high));
        return [-bound, bound];
      }
      return [Math.min(0, low), Math.max(high, low + 1e-9)];
    };
    const describeSelected = () => {
      if (!latest || selectedRow < 0) return;
      const row = selectedRow;
      selectedReadout.setValue(
        `${zips.featureNames[row] ?? zips.featureIds[row]}: ${formatCount(latest.events[row])} sites / ${formatCount(latest.populations[row])} people; ` +
          `raw ${formatNumber(latest.raw[row] * 1000, 2)}, global EB ${formatNumber(latest.smoothed[row] * 1000, 2)}, spatial EB ${formatNumber(latest.spatial[row] * 1000, 2)} per 1,000; ` +
          `z ${formatNumber(latest.standardized[row], 2)}; p ${formatP(latest.pseudoP[row])}`
      );
    };

    const reader = new SummaryReader(
      resources,
      'rates',
      [
        {buffer: rawBuffer, size: featureCount * 4},
        {buffer: smoothedBuffer, size: featureCount * 4},
        {buffer: standardizedBuffer, size: featureCount * 4},
        {buffer: spatialBuffer, size: featureCount * 4},
        {buffer: eventsBuffer, size: featureCount * 4},
        {buffer: populationsBuffer, size: featureCount * 4},
        {buffer: classesBuffer, size: featureCount * 4},
        {buffer: pseudoPBuffer, size: featureCount * 4},
        {buffer: summaryBuffer, size: GPU_EMPIRICAL_BAYES_SUMMARY.length * 4},
        {buffer: statisticsBuffer, size: STATISTICS_LENGTH * 4},
        {buffer: classCountsBuffer, size: CLASS_COUNT * 4},
        {buffer: searchOverflow, size: 4},
        {buffer: totalNeighbors, size: 4},
        {buffer: localOverflow, size: 4}
      ],
      bytes => {
        const floats = new Float32Array(bytes);
        const words = new Uint32Array(bytes);
        const slice = (start: number) => floats.slice(start, start + featureCount);
        latest = {
          raw: slice(layout.raw),
          smoothed: slice(layout.smoothed),
          spatial: slice(layout.spatial),
          standardized: slice(layout.standardized),
          events: slice(layout.events),
          populations: slice(layout.populations),
          pseudoP: slice(layout.pseudoP),
          classes: words.slice(layout.classes, layout.classes + featureCount)
        };
        if (!eventsKnown) {
          // Events are known now: couple the populations to them once, then rates are recomputed.
          eventsKnown = true;
          writePopulations();
        }
        ranges.raw = getRange(latest.raw, false);
        ranges.smoothed = ranges.raw;
        ranges.spatial = ranges.raw;
        ranges.standardized = getRange(latest.standardized, true);
        const summary = floats.subarray(layout.summary, layout.summary + 8);
        pooledReadout.setValue(
          `${formatNumber(summary[GPU_EMPIRICAL_BAYES_SUMMARY.pooledRate] * 1000, 2)} (${formatCount(summary[GPU_EMPIRICAL_BAYES_SUMMARY.eventSum])} sites / ${formatCount(summary[GPU_EMPIRICAL_BAYES_SUMMARY.populationSum])} people)`
        );
        varianceReadout.setValue(
          `${formatNumber(getStandardDeviation(latest.raw) * 1000, 2)} / ${formatNumber(getStandardDeviation(latest.smoothed) * 1000, 2)} / ${formatNumber(getStandardDeviation(latest.spatial) * 1000, 2)} per 1,000`
        );
        priorReadout.setValue(
          `${formatNumber(summary[GPU_EMPIRICAL_BAYES_SUMMARY.priorVariance] * 1e6, 3)} / ${formatNumber(summary[GPU_EMPIRICAL_BAYES_SUMMARY.weightedRateVariance] * 1e6, 3)} (x1e-6)`
        );
        const byPopulation = Array.from({length: featureCount}, (_, row) => row).sort(
          (a, b) => latest!.populations[a] - latest!.populations[b]
        );
        const smallGroup = byPopulation.slice(
          0,
          Math.max(3, Math.round(featureCount * SMALL_POPULATION_SHARE))
        );
        const smallStd = (values: Float32Array) =>
          formatNumber(
            getStandardDeviation(Float32Array.from(smallGroup, row => values[row])) * 1000,
            2
          );
        smallStdReadout.setValue(
          `${smallStd(latest.raw)} / ${smallStd(latest.smoothed)} / ${smallStd(latest.spatial)} per 1,000 (${smallGroup.length} ZIPs)`
        );
        const smallest = byPopulation.slice(0, 3);
        extremeReadout.setValue(
          smallest
            .map(
              row =>
                `${zips.featureNames[row] ?? row} (${formatCount(latest!.populations[row])}): ${formatNumber(latest!.raw[row] * 1000, 1)} → ${formatNumber(latest!.smoothed[row] * 1000, 1)} → ${formatNumber(latest!.spatial[row] * 1000, 1)}`
            )
            .join('; ')
        );
        const moran = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran;
        const field = GPU_GLOBAL_SPATIAL_STATISTIC_FIELD;
        const statistic = (index: number) => floats[layout.statistics + moran + index];
        moranReadout.setValue(
          `${formatNumber(statistic(field.statistic))} (${formatNumber(statistic(field.expected), 3)}), z ${formatNumber(statistic(field.zRandomization), 2)}, p ${formatP(statistic(field.pRandomization))}`
        );
        const count = (index: number) => formatCount(words[layout.classCounts + index]);
        classReadout.setValue(
          `${count(1)} / ${count(3)} / ${count(4)} / ${count(2)} / ${count(0)}`
        );
        const links = Math.min(words[layout.flags + 1], capacity);
        flagsReadout.setValue(
          `${formatCount(links)} / ${words[layout.flags] ? 'SEARCH OVERFLOW' : 'ok'}, ${
            words[layout.flags + 2] ? 'local overflow' : 'ok'
          }`
        );
        describeSelected();
        context.updateLayers();
      }
    );

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [
        joinCompiled,
        rateCompiled,
        spatialCompiled,
        ...searchVariants.values(),
        ...analysisVariants.values()
      ],
      encode(commandEncoder, frame) {
        if (dirty || frame.frameIndex < 3) {
          joinCompiled.encode(commandEncoder, {parameters: undefined});
          rateCompiled.encode(commandEncoder, {parameters: undefined});
          searchVariants.get(neighborChoice)!.encode(commandEncoder, {parameters: undefined});
          spatialCompiled.encode(commandEncoder, {parameters: undefined});
          analysisVariants.get(alternative)!.encode(commandEncoder, {parameters: undefined});
          dirty = false;
          reader.request(commandEncoder);
        } else {
          if (needsReadback) reader.markStale();
          needsReadback = false;
          reader.flush(commandEncoder);
        }
      },
      onClick(event) {
        if (!event.coordinate) return false;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        let best = -1;
        let bestDistance = Infinity;
        for (let row = 0; row < featureCount; row++) {
          const distance =
            (geometry.centroids[row * 2] - x) ** 2 + (geometry.centroids[row * 2 + 1] - y) ** 2;
          if (distance < bestDistance) {
            bestDistance = distance;
            best = row;
          }
        }
        selectedRow = best;
        markerBuffer.write(geometry.centroids.subarray(best * 2, best * 2 + 2));
        describeSelected();
        context.updateLayers();
        return true;
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [zips.origin[0], zips.origin[1], 0];
        const palette = (color: (index: number) => SpatialAnalysisColor) =>
          Array.from({length: 8}, (_, index) => color(index));
        const layers: Layer[] = [];
        const isCluster = mapKind === 'cluster';
        const valuesBufferForMap =
          mapKind === 'raw'
            ? rawBuffer
            : mapKind === 'smoothed'
              ? smoothedBuffer
              : mapKind === 'spatial'
                ? spatialBuffer
                : standardizedBuffer;
        const styleForValues = isCluster
          ? {
              values: classesBuffer,
              valueFormat: 'uint32' as const,
              colormap: 'category' as const,
              palette: palette(index => (index === 0 ? NEUTRAL : (LISA_COLORS[index] ?? HIDDEN)))
            }
          : {
              values: valuesBufferForMap,
              valueFormat: 'float32' as const,
              colormap: 'viridis' as const,
              valueRange: ranges[mapKind as Variable]
            };
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'rates-outline',
            coordinateOrigin,
            segments: outlineSegments,
            instanceCount: segmentCount,
            valueIndices: outlineFeatureRows,
            widthPixels: 7,
            ...styleForValues
          }),
          new SpatialAnalysisPointLayer({
            id: 'rates-discs',
            coordinateOrigin,
            positions: centroidsBuffer,
            instanceCount: featureCount,
            radiusPixels: 13,
            ...styleForValues
          })
        );
        if (selectedRow >= 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'rates-selected',
              coordinateOrigin,
              positions: markerBuffer,
              instanceCount: 1,
              radiusPixels: 20,
              colormap: 'uniform',
              color: [255, 255, 255, 90]
            })
          );
        }
        return layers;
      },
      destroy() {
        reader.stop();
        resources.destroy();
      }
    };

    return instance;
  }
};
