// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTemporalReductionParameterValues,
  GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
  GPUTemporalReduction
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUNeighborSearchParameterValues,
  getGPUSpaceTimeParameterValues,
  getGPUSpatialScanParameterValues,
  getKnoxPoissonPValue,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_SCAN_STATISTIC_PARAMETER_LENGTH,
  GPU_SCAN_STATISTIC_SUMMARY_LENGTH,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  GPUKnoxTest,
  GPUMantelTest,
  GPUNeighborSearch,
  GPUSpatialScanStatistic
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {addKernelPass} from '../../engine/mode-kernels';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor
} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  getOutlineSegments,
  lookupFeature,
  rasterizePolygons,
  type B13PolygonRaster
} from './b13-polygon-raster';
import {readTracts, type B13Tracts} from './b13-tract-data';
import {createViewImporter} from './b13-views';

/** Event sources of the scene. */
export type ClusterSource = 'fungi' | 'insects' | 'birds' | 'plants' | 'mammals' | 'rats';

/** Option state of the space-time-clusters scene. */
export type SpaceTimeClustersOptions = {
  source: ClusterSource;
  analysis: 'pairs' | 'scan';
  spatialRadius: number;
  timeThreshold: number;
  permutations: number;
  showSpatialOnly: boolean;
  baseline: 'independence' | 'population' | 'uniform';
  maximumPopulationFraction: number;
  maximumWindowZones: number;
  maximumTimeBuckets: number;
  windowShape: 'circle' | 'nearest';
  scanPermutations: number;
  significance: number;
  scanView: 'clusters' | 'zone-llr' | 'rate';
  opacity: number;
};

/** Source table: dataset, category (nature observations only) and a label. */
export const CLUSTER_SOURCES: Record<
  ClusterSource,
  {dataset: string; category: number | null; label: string}
> = {
  fungi: {dataset: 'chicago-nature', category: 3, label: 'fungi observations'},
  insects: {dataset: 'chicago-nature', category: 2, label: 'insect observations'},
  birds: {dataset: 'chicago-nature', category: 1, label: 'bird observations'},
  plants: {dataset: 'chicago-nature', category: 0, label: 'plant observations'},
  mammals: {dataset: 'chicago-nature', category: 4, label: 'mammal observations'},
  rats: {dataset: 'chicago-311-rats', category: null, label: 'rodent complaints'}
};

const MAXIMUM_KNOX_EVENTS = 24000;
const SLOTS_PER_EVENT = 64;
const MAXIMUM_PERMUTATIONS = 999;
const BUCKET_DAYS = 14;
const BUCKET_COUNT = 26;
const MAXIMUM_CLUSTERS = 7;
const NO_CELL = 0xffffffff;
const SUMMARY = GPU_SPACE_TIME_SUMMARY;
const SETTLE_MILLISECONDS = 350;
const MONTH_NAMES = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec'
];
/** Palette of the cluster membership map: slot 0 is "not in a cluster". */
export const CLUSTER_PALETTE: readonly SpatialAnalysisColor[] = [
  [140, 150, 170, 55],
  [215, 48, 39, 235],
  [244, 140, 40, 235],
  [140, 60, 190, 235],
  [30, 150, 90, 235],
  [30, 110, 200, 235],
  [200, 170, 20, 235],
  [90, 90, 90, 235]
];

const formatDay = (dayOfYear: number) => {
  const date = new Date(Date.UTC(2023, 0, 1 + dayOfYear));
  return `${MONTH_NAMES[date.getUTCMonth()]} ${date.getUTCDate()}`;
};
export const formatBucket = (bucket: number) =>
  `${formatDay(bucket * BUCKET_DAYS)} to ${formatDay((bucket + 1) * BUCKET_DAYS - 1)}`;

function formatNumber(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

function formatProbability(value: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  return value < 0.001 ? '< 0.001' : value.toFixed(3);
}

/** One-line histogram of a permutation distribution with the percentile of the observed value. */
function formatDistribution(values: ArrayLike<number>, count: number, observed: number): string {
  const bins = 32;
  let minimum = observed;
  let maximum = observed;
  for (let index = 0; index < count; index++) {
    minimum = Math.min(minimum, values[index]);
    maximum = Math.max(maximum, values[index]);
  }
  const span = Math.max(maximum - minimum, 1e-9);
  const heights = new Array<number>(bins).fill(0);
  let below = 0;
  for (let index = 0; index < count; index++) {
    heights[Math.min(bins - 1, Math.floor(((values[index] - minimum) / span) * bins))]++;
    if (values[index] < observed) below++;
  }
  const bars = '.:-=+*#@';
  const peak = Math.max(...heights, 1);
  const spark = heights
    .map(height => (height === 0 ? '·' : bars[Math.min(7, Math.round((height / peak) * 7))]))
    .join('');
  return `${spark}  observed beats ${((below / Math.max(count, 1)) * 100).toFixed(1)}% of permutations`;
}

type SourceTable = {
  positions: Float32Array;
  seconds: Float32Array;
  tract: Uint32Array;
};

type KnoxPart = {
  compiled: CompiledGPUCommandGraph<void>;
  eventCount: number;
  positions: Buffer;
  times: Buffer;
  segments: Buffer;
  fade: Buffer;
  slotCapacity: number;
  reader: SummaryReader;
};

type ScanPart = {
  compiled: CompiledGPUCommandGraph<void>;
  caseCount: number;
  cases: Buffer;
  baseline: Buffer;
  membership: Buffer;
  zoneStatistics: Buffer;
  rate: Buffer;
  totals: Uint32Array;
  reader: SummaryReader;
};

type Variant = {key: ClusterSource; knox: KnoxPart; scan: ScanPart; knoxEvents: number};

/**
 * Space-time clusters of Chicago nature observations and rodent complaints. Two compiled graphs per event
 * source: `GPUNeighborSearch` pairs feed `GPUKnoxTest` and `GPUMantelTest` (are events near in
 * space also near in time?), and `GPUTemporalReduction` counts per tract and fortnight feed
 * `GPUSpatialScanStatistic` (where and when is the cluster?). Thresholds, permutations, seeds,
 * baselines and window limits are parameter writes; only the event source is compile-time.
 */
export async function createSpaceTimeClusters(
  ctx: SceneContext<SpaceTimeClustersOptions>
): Promise<SceneInstance<SpaceTimeClustersOptions>> {
  const {device} = ctx;
  const nature = ctx.datasets.get('chicago-nature');
  const rats = ctx.datasets.get('chicago-311-rats');
  const tractsDataset = ctx.datasets.get('chicago-tracts');
  const origin = nature.defaultOrigin;
  const tracts: B13Tracts = readTracts(tractsDataset, origin);
  const raster: B13PolygonRaster = rasterizePolygons(
    tracts.source,
    tractsDataset.getProjection(origin),
    tractsDataset.manifest.bbox,
    1400
  );
  const zoneCount = tracts.count;

  const resources = new SpatialAnalysisResources(device, 'clusters');
  const cellFeatureBuffer = resources.createBuffer('cell-feature', raster.cellFeature);
  const outlineSegments = getOutlineSegments(tracts.source, tractsDataset.getProjection(origin));
  const outlineBuffer = resources.createBuffer('outlines', outlineSegments);
  const zonePositionsBuffer = resources.createBuffer('zone-positions', tracts.centroids);
  const searchParameters = resources.createParameterBuffer(
    'search-parameters',
    'float32',
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
  );
  const testParameters = resources.createParameterBuffer(
    'test-parameters',
    'uint32',
    GPU_SPACE_TIME_PARAMETER_LENGTH
  );
  const drawParameters = resources.createParameterBuffer('draw-parameters', 'float32', 2);
  const reductionParameters = resources.createParameterBuffer(
    'reduction-parameters',
    'float32',
    GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
    getGPUTemporalReductionParameterValues(0, BUCKET_DAYS * 86400)
  );
  const scanParameters = resources.createParameterBuffer(
    'scan-parameters',
    'uint32',
    GPU_SCAN_STATISTIC_PARAMETER_LENGTH
  );
  const significanceParameter = resources.createParameterBuffer('significance', 'float32', 1);

  let destroyed = false;
  let seed = 7;
  let knoxDirty = true;
  let scanDirty = true;
  let scanChangedAt = performance.now() - 1000;
  let current: Variant | null = null;
  let llrMaximum = 1;
  let rateMaximum = 1;
  let cpuMembership: Uint32Array | null = null;
  let _cpuStatistics: Float32Array | null = null;
  let scanActive = false;
  const variants = new Map<ClusterSource, Variant>();
  const tables = new Map<ClusterSource, SourceTable>();

  function getTable(key: ClusterSource): SourceTable {
    let table = tables.get(key);
    if (table) return table;
    const spec = CLUSTER_SOURCES[key];
    const dataset = spec.dataset === 'chicago-nature' ? nature : rats;
    const seconds = dataset.column<Uint32Array>('timestamp');
    const category = spec.category === null ? null : dataset.column<Uint8Array>('category');
    const tractColumn = dataset.column<Uint16Array>('tract');
    const positions = dataset.projectColumn('position', origin);
    const kept: number[] = [];
    for (let index = 0; index < seconds.length; index++) {
      if (category === null || category[index] === spec.category) kept.push(index);
    }
    table = {
      positions: new Float32Array(kept.length * 2),
      seconds: new Float32Array(kept.length),
      tract: new Uint32Array(kept.length)
    };
    kept.forEach((source, index) => {
      table!.positions[index * 2] = positions[source * 2];
      table!.positions[index * 2 + 1] = positions[source * 2 + 1];
      table!.seconds[index] = seconds[source];
      table!.tract[index] = tractColumn[source] < zoneCount ? tractColumn[source] : NO_CELL;
    });
    tables.set(key, table);
    return table;
  }

  function buildKnox(key: ClusterSource, table: SourceTable): {part: KnoxPart; events: number} {
    const total = table.seconds.length;
    const step = Math.max(1, Math.ceil(total / MAXIMUM_KNOX_EVENTS));
    const eventCount = Math.ceil(total / step);
    const positions = new Float32Array(eventCount * 2);
    const days = new Float32Array(eventCount);
    let minimumX = Infinity;
    let minimumY = Infinity;
    let maximumX = -Infinity;
    let maximumY = -Infinity;
    for (let index = 0; index < eventCount; index++) {
      const source = index * step;
      positions[index * 2] = table.positions[source * 2];
      positions[index * 2 + 1] = table.positions[source * 2 + 1];
      days[index] = table.seconds[source] / 86400;
      minimumX = Math.min(minimumX, positions[index * 2]);
      maximumX = Math.max(maximumX, positions[index * 2]);
      minimumY = Math.min(minimumY, positions[index * 2 + 1]);
      maximumY = Math.max(maximumY, positions[index * 2 + 1]);
    }
    searchBounds = [minimumX - 100, minimumY - 100, maximumX + 100, maximumY + 100];
    const slotCapacity = eventCount * SLOTS_PER_EVENT;
    const buffer = (name: string, data: Float32Array | number) =>
      resources.createBuffer(`${key}-knox-${name}`, data);
    const positionsBuffer = buffer('positions', positions);
    const timesBuffer = buffer('times', days);
    const offsets = buffer('offsets', (eventCount + 1) * 4);
    const neighbors = buffer('neighbors', slotCapacity * 4);
    const weights = buffer('weights', slotCapacity * 4);
    const distances = buffer('distances', slotCapacity * 4);
    const overflow = buffer('overflow', 4);
    const knoxStatistics = buffer('knox-statistics', (MAXIMUM_PERMUTATIONS + 1) * 4);
    const knoxSummary = buffer('knox-summary', GPU_SPACE_TIME_SUMMARY_LENGTH * 4);
    const mantelStatistics = buffer('mantel-statistics', (MAXIMUM_PERMUTATIONS + 1) * 4);
    const mantelSummary = buffer('mantel-summary', GPU_SPACE_TIME_SUMMARY_LENGTH * 4);
    const segments = buffer('segments', slotCapacity * 16);
    const fade = buffer('fade', slotCapacity * 4);
    const graph = new GPUCommandGraph<void>(device, {id: `knox-${key}`});
    const v = createViewImporter(graph, `${key}-knox`);
    const positionsView = v('positions', positionsBuffer, 'float32x2', eventCount);
    const timesView = v('times', timesBuffer, 'float32', eventCount);
    const pairs = {
      offsets: v('offsets', offsets, 'uint32', eventCount + 1),
      neighbors: v('neighbors', neighbors, 'uint32', slotCapacity),
      weights: v('weights', weights, 'float32', slotCapacity),
      distances: v('distances', distances, 'float32', slotCapacity)
    };
    const testParametersView = testParameters.importToGraph(graph);
    graph.add(
      new GPUNeighborSearch({
        id: `${key}-search`,
        mode: 'radius',
        positions: positionsView,
        parameters: searchParameters.importToGraph(graph),
        gridSize: [64, 64],
        weights: pairs,
        overflow: v('overflow', overflow, 'uint32', 1)
      })
    );
    graph.add(
      new GPUKnoxTest({
        id: `${key}-knox`,
        pairs,
        times: timesView,
        parameters: testParametersView,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        statistics: v('knox-statistics', knoxStatistics, 'uint32', MAXIMUM_PERMUTATIONS + 1),
        summary: v('knox-summary', knoxSummary, 'float32', GPU_SPACE_TIME_SUMMARY_LENGTH)
      })
    );
    graph.add(
      new GPUMantelTest({
        id: `${key}-mantel`,
        pairs,
        times: timesView,
        parameters: testParametersView,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        statistics: v('mantel-statistics', mantelStatistics, 'float32', MAXIMUM_PERMUTATIONS + 1),
        summary: v('mantel-summary', mantelSummary, 'float32', GPU_SPACE_TIME_SUMMARY_LENGTH)
      })
    );
    // One thread per pair slot finds its row by binary search and writes the drawable segment of
    // each unordered pair (entries j > i); pairs close in time are opaque.
    addKernelPass(graph, {
      id: `${key}-segments`,
      invocationCount: slotCapacity,
      declarations: `const EVENT_COUNT: u32 = ${eventCount}u;`,
      bindings: [
        {name: 'offsets', view: pairs.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: pairs.neighbors, type: 'u32', access: 'read'},
        {name: 'positions', view: positionsView, type: 'f32', access: 'read'},
        {name: 'times', view: timesView, type: 'f32', access: 'read'},
        {
          name: 'drawParameters',
          view: drawParameters.importToGraph(graph),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'segments',
          view: v('segments', segments, 'float32', slotCapacity * 4),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'fade',
          view: v('fade', fade, 'float32', slotCapacity),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  segments[segmentsOffset + index * 4u] = 0.0;
  segments[segmentsOffset + index * 4u + 1u] = 0.0;
  segments[segmentsOffset + index * 4u + 2u] = 0.0;
  segments[segmentsOffset + index * 4u + 3u] = 0.0;
  fade[fadeOffset + index] = 0.0;
  if (index >= offsets[offsetsOffset + EVENT_COUNT]) {
    return;
  }
  var low = 0u;
  var high = EVENT_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  if (neighbor <= low) {
    return;
  }
  segments[segmentsOffset + index * 4u] = positions[positionsOffset + low * 2u];
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + low * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + neighbor * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + neighbor * 2u + 1u];
  let timeGap = abs(times[timesOffset + low] - times[timesOffset + neighbor]);
  fade[fadeOffset + index] = select(drawParameters[drawParametersOffset + 1u], 1.0, timeGap <= drawParameters[drawParametersOffset]);`
    });
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      `${key}-knox`,
      [
        {buffer: knoxSummary, size: GPU_SPACE_TIME_SUMMARY_LENGTH * 4},
        {buffer: mantelSummary, size: GPU_SPACE_TIME_SUMMARY_LENGTH * 4},
        {buffer: overflow, size: 4},
        {buffer: knoxStatistics, size: (MAXIMUM_PERMUTATIONS + 1) * 4},
        {buffer: mantelStatistics, size: (MAXIMUM_PERMUTATIONS + 1) * 4}
      ],
      bytes => {
        if (destroyed || current?.key !== key) return;
        const words = GPU_SPACE_TIME_SUMMARY_LENGTH;
        const knox = new Float32Array(bytes, 0, words);
        const mantel = new Float32Array(bytes, words * 4, words);
        const overflowFlag = new Uint32Array(bytes, words * 8, 1)[0];
        const knoxPermuted = new Uint32Array(bytes, words * 8 + 4, MAXIMUM_PERMUTATIONS + 1);
        const mantelPermuted = new Float32Array(
          bytes,
          words * 8 + 4 + (MAXIMUM_PERMUTATIONS + 1) * 4,
          MAXIMUM_PERMUTATIONS + 1
        );
        const permutations = ctx.options.permutations;
        ctx.setReadout('pairs', knox[SUMMARY.pairCount]);
        ctx.setReadout('timeClose', knox[SUMMARY.timeClosePairs]);
        ctx.setReadout('knoxObserved', knox[SUMMARY.observed]);
        ctx.setReadout('knoxExpected', formatNumber(knox[SUMMARY.expected], 1));
        ctx.setReadout(
          'knoxRatio',
          `${formatNumber(knox[SUMMARY.observed] / Math.max(knox[SUMMARY.expected], 1e-9), 2)}x`
        );
        ctx.setReadout(
          'knoxPermuted',
          `${formatNumber(knox[SUMMARY.permutationMean], 1)} ± ${formatNumber(Math.sqrt(knox[SUMMARY.permutationVariance]), 1)}`
        );
        ctx.setReadout('knoxP', formatProbability(knox[SUMMARY.pseudoPGreater]));
        ctx.setReadout(
          'knoxPoisson',
          formatProbability(getKnoxPoissonPValue(knox[SUMMARY.observed], knox[SUMMARY.expected]))
        );
        ctx.setReadout(
          'knoxDistribution',
          formatDistribution(knoxPermuted.subarray(1), permutations, knox[SUMMARY.observed])
        );
        ctx.setReadout('mantelR', formatNumber(mantel[SUMMARY.observed], 4));
        ctx.setReadout(
          'mantelPermuted',
          `${formatNumber(mantel[SUMMARY.permutationMean], 4)} ± ${formatNumber(Math.sqrt(mantel[SUMMARY.permutationVariance]), 4)}`
        );
        ctx.setReadout('mantelP', formatProbability(mantel[SUMMARY.pseudoPGreater]));
        ctx.setReadout(
          'mantelDistribution',
          formatDistribution(mantelPermuted.subarray(1), permutations, mantel[SUMMARY.observed])
        );
        ctx.setReadout('pairOverflow', overflowFlag ? 'yes: lower the radius' : 'no');
      }
    );
    return {
      part: {
        compiled,
        eventCount,
        positions: positionsBuffer,
        times: timesBuffer,
        segments,
        fade,
        slotCapacity,
        reader
      },
      events: eventCount
    };
  }

  function buildScan(key: ClusterSource, table: SourceTable): ScanPart {
    const caseCount = table.seconds.length;
    // CPU copy of the same counts GPUTemporalReduction produces, for the baseline model only.
    const totals = new Uint32Array(zoneCount * BUCKET_COUNT);
    for (let index = 0; index < caseCount; index++) {
      const zone = table.tract[index];
      if (zone === NO_CELL) continue;
      const bucket = Math.floor(table.seconds[index] / (BUCKET_DAYS * 86400));
      if (bucket < BUCKET_COUNT) totals[zone * BUCKET_COUNT + bucket]++;
    }
    const cellCountBins = zoneCount * BUCKET_COUNT;
    const buffer = (name: string, data: Float32Array | Uint32Array | number) =>
      resources.createBuffer(`${key}-scan-${name}`, data);
    const secondsBuffer = buffer('seconds', table.seconds);
    const onesBuffer = buffer('ones', new Float32Array(caseCount).fill(1));
    const tractBuffer = buffer('tract', table.tract);
    const casesBuffer = buffer('cases', cellCountBins * 4);
    const baseline = buffer('baseline', cellCountBins * 4);
    const clusterIndices = buffer('cluster-indices', MAXIMUM_CLUSTERS * 4 * 4);
    const clusterStatistics = buffer('cluster-statistics', MAXIMUM_CLUSTERS * 8 * 4);
    const statistics = buffer('statistics', (MAXIMUM_PERMUTATIONS + 1) * 4);
    const summary = buffer('summary', GPU_SCAN_STATISTIC_SUMMARY_LENGTH * 4);
    const zoneStatistics = buffer('zone-statistics', (zoneCount + 1) * 4);
    const membership = buffer('membership', (zoneCount + 1) * 4);
    const perCapita = new Float32Array(zoneCount + 1).fill(NaN);
    let maximumRate = 0;
    for (let zone = 0; zone < zoneCount; zone++) {
      let sum = 0;
      for (let bucket = 0; bucket < BUCKET_COUNT; bucket++)
        sum += totals[zone * BUCKET_COUNT + bucket];
      if (tracts.population[zone] >= 300) {
        perCapita[zone] = (sum / tracts.population[zone]) * 1000;
      }
    }
    const sorted = Float32Array.from(perCapita.filter(Number.isFinite)).sort();
    maximumRate = sorted[Math.floor(sorted.length * 0.97)] ?? 1;
    rateMaximum = Math.max(maximumRate, 1);
    ctx.setLegendExtent('rate', [0, rateMaximum]);
    const rate = buffer('rate', perCapita);
    const reductionOutputs = (name: string) => buffer(name, cellCountBins * 4);
    const graph = new GPUCommandGraph<void>(device, {id: `scan-${key}`});
    const v = createViewImporter(graph, `${key}-scan`);
    graph.add(
      new GPUTemporalReduction({
        id: `${key}-reduction`,
        cellIds: v('tract', tractBuffer, 'uint32', caseCount),
        timestamps: v('seconds', secondsBuffer, 'float32', caseCount),
        values: v('ones', onesBuffer, 'float32', caseCount),
        parameters: reductionParameters.importToGraph(graph),
        cellCount: zoneCount,
        bucketCount: BUCKET_COUNT,
        output: {
          counts: v('cases', casesBuffer, 'uint32', cellCountBins),
          min: v('min', reductionOutputs('min'), 'float32', cellCountBins),
          max: v('max', reductionOutputs('max'), 'float32', cellCountBins),
          first: v('first', reductionOutputs('first'), 'float32', cellCountBins),
          last: v('last', reductionOutputs('last'), 'float32', cellCountBins),
          occupiedSlots: {
            ids: v(
              'occupied-ids',
              buffer('occupied-ids', cellCountBins * 4),
              'uint32',
              cellCountBins
            ),
            count: v('occupied-count', buffer('occupied-count', 4), 'uint32', 1),
            overflow: v('occupied-overflow', buffer('occupied-overflow', 4), 'uint32', 1)
          }
        }
      })
    );
    const zonePositions = v('zone-positions', zonePositionsBuffer, 'float32x2', zoneCount);
    const clusterIndicesView = v('cluster-indices', clusterIndices, 'uint32', MAXIMUM_CLUSTERS * 4);
    const clusterStatisticsView = v(
      'cluster-statistics',
      clusterStatistics,
      'float32',
      MAXIMUM_CLUSTERS * 8
    );
    graph.add(
      new GPUSpatialScanStatistic({
        id: `${key}-scan`,
        positions: zonePositions,
        cases: v('cases', casesBuffer, 'uint32', cellCountBins),
        baseline: v('baseline', baseline, 'float32', cellCountBins),
        timeBuckets: BUCKET_COUNT,
        maximumWindowZones: 32,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        maximumClusters: MAXIMUM_CLUSTERS,
        parameters: scanParameters.importToGraph(graph),
        clusterIndices: clusterIndicesView,
        clusterStatistics: clusterStatisticsView,
        statistics: v('statistics', statistics, 'float32', MAXIMUM_PERMUTATIONS + 1),
        summary: v('summary', summary, 'uint32', GPU_SCAN_STATISTIC_SUMMARY_LENGTH),
        zoneStatistics: v('zone-statistics', zoneStatistics, 'float32', zoneCount)
      })
    );
    // Zones inside a significant cluster window get the cluster rank (1 is the most likely).
    addKernelPass(graph, {
      id: `${key}-membership`,
      invocationCount: zoneCount + 1,
      declarations: `const ZONES: u32 = ${zoneCount}u; const CLUSTERS: u32 = ${MAXIMUM_CLUSTERS}u;`,
      bindings: [
        {name: 'indices', view: clusterIndicesView, type: 'u32', access: 'read'},
        {name: 'stats', view: clusterStatisticsView, type: 'f32', access: 'read'},
        {name: 'positions', view: zonePositions, type: 'f32', access: 'read'},
        {
          name: 'alpha',
          view: significanceParameter.importToGraph(graph),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'membership',
          view: v('membership', membership, 'uint32', zoneCount + 1),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  if (index >= ZONES) {
    membership[membershipOffset + index] = 0xffffffffu;
    return;
  }
  var member = 0u;
  let px = positions[positionsOffset + index * 2u];
  let py = positions[positionsOffset + index * 2u + 1u];
  for (var cluster = 0u; cluster < CLUSTERS; cluster++) {
    if (indices[indicesOffset + cluster * 4u + 1u] == 0u) {
      continue;
    }
    if (stats[statsOffset + cluster * 8u + 3u] > alpha[alphaOffset]) {
      continue;
    }
    let center = indices[indicesOffset + cluster * 4u];
    let dx = px - positions[positionsOffset + center * 2u];
    let dy = py - positions[positionsOffset + center * 2u + 1u];
    let radius = stats[statsOffset + cluster * 8u + 4u];
    if (dx * dx + dy * dy <= radius * radius * 1.00002 + 1.0) {
      member = cluster + 1u;
      break;
    }
  }
  membership[membershipOffset + index] = member;`
    });
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      `${key}-scan`,
      [
        {buffer: summary, size: GPU_SCAN_STATISTIC_SUMMARY_LENGTH * 4},
        {buffer: clusterIndices, size: MAXIMUM_CLUSTERS * 16},
        {buffer: clusterStatistics, size: MAXIMUM_CLUSTERS * 32},
        {buffer: membership, size: (zoneCount + 1) * 4}
      ],
      bytes => {
        if (destroyed || current?.key !== key) return;
        showScan(bytes);
      }
    );
    return {
      compiled,
      caseCount,
      cases: casesBuffer,
      baseline,
      membership,
      zoneStatistics,
      rate,
      totals,
      reader
    };
  }

  let searchBounds: [number, number, number, number] = [0, 0, 1, 1];

  function getVariant(): Variant {
    const key = ctx.options.source;
    let variant = variants.get(key);
    if (!variant) {
      const table = getTable(key);
      const {part: knox, events} = buildKnox(key, table);
      const scan = buildScan(key, table);
      variant = {key, knox, scan, knoxEvents: events};
      variants.set(key, variant);
    }
    return variant;
  }

  function writeBaseline(): void {
    if (!current) return;
    const {totals} = current.scan;
    const kind = ctx.options.baseline;
    const baseline = new Float32Array(zoneCount * BUCKET_COUNT);
    const bucketTotals = new Float64Array(BUCKET_COUNT);
    const zoneTotals = new Float64Array(zoneCount);
    let grand = 0;
    for (let zone = 0; zone < zoneCount; zone++) {
      for (let bucket = 0; bucket < BUCKET_COUNT; bucket++) {
        const count = totals[zone * BUCKET_COUNT + bucket];
        bucketTotals[bucket] += count;
        zoneTotals[zone] += count;
        grand += count;
      }
    }
    for (let zone = 0; zone < zoneCount; zone++) {
      for (let bucket = 0; bucket < BUCKET_COUNT; bucket++) {
        const index = zone * BUCKET_COUNT + bucket;
        if (kind === 'independence') {
          baseline[index] = grand > 0 ? (zoneTotals[zone] * bucketTotals[bucket]) / grand : 0;
        } else if (kind === 'population') {
          baseline[index] =
            grand > 0 ? (tracts.population[zone] || 0) * (bucketTotals[bucket] / grand) : 0;
        } else {
          baseline[index] = 1;
        }
      }
    }
    current.scan.baseline.write(baseline);
    scanDirty = true;
    scanChangedAt = performance.now();
  }

  function writeKnoxParameters(): void {
    const options = ctx.options;
    searchParameters.write(
      getGPUNeighborSearchParameterValues({
        bounds: searchBounds,
        radius: options.spatialRadius,
        weightKind: 'binary'
      })
    );
    testParameters.write(
      getGPUSpaceTimeParameterValues({
        seed,
        permutations: options.permutations,
        timeThreshold: options.timeThreshold
      })
    );
    drawParameters.write(
      Float32Array.of(options.timeThreshold, options.showSpatialOnly ? 0.14 : 0)
    );
    knoxDirty = true;
  }

  function writeScanParameters(): void {
    const options = ctx.options;
    scanParameters.write(
      getGPUSpatialScanParameterValues({
        seed,
        permutations: options.scanPermutations,
        maximumPopulationFraction: options.maximumPopulationFraction,
        maximumWindowZones: options.maximumWindowZones,
        maximumTimeBuckets: options.maximumTimeBuckets,
        windowShape: options.windowShape
      })
    );
    significanceParameter.write(Float32Array.of(options.significance));
    scanDirty = true;
    scanChangedAt = performance.now();
  }

  function showScan(bytes: ArrayBuffer): void {
    const summary = new Uint32Array(bytes, 0, GPU_SCAN_STATISTIC_SUMMARY_LENGTH);
    const indices = new Uint32Array(bytes, 16, MAXIMUM_CLUSTERS * 4);
    const stats = new Float32Array(bytes, 16 + MAXIMUM_CLUSTERS * 16, MAXIMUM_CLUSTERS * 8);
    cpuMembership = new Uint32Array(bytes, 16 + MAXIMUM_CLUSTERS * 48, zoneCount + 1);
    _cpuStatistics = stats;
    const clusters = summary[0];
    ctx.setReadout('scanCases', formatCount(summary[1]));
    ctx.setReadout(
      'scanClusters',
      `${clusters} found, ${countSignificant(stats, clusters)} significant`
    );
    llrMaximum = Math.max(1e-6, stats[0]);
    ctx.setLegendExtent('llr', [0, llrMaximum]);
    for (let rank = 0; rank < 3; rank++) {
      if (rank >= clusters) {
        ctx.setReadout(`cluster${rank + 1}`, '-');
        continue;
      }
      const zones = indices[rank * 4 + 1];
      const first = indices[rank * 4 + 2];
      const last = indices[rank * 4 + 3];
      const observed = stats[rank * 8 + 1];
      const expected = stats[rank * 8 + 2];
      const probability = stats[rank * 8 + 3];
      ctx.setReadout(
        `cluster${rank + 1}`,
        `${zones} tracts within ${((stats[rank * 8 + 4] ?? 0) / 1000).toFixed(1)} km, ${formatDay(first * BUCKET_DAYS)} to ${formatDay((last + 1) * BUCKET_DAYS - 1)}. ${formatCount(observed)} cases vs ${expected.toFixed(0)} expected (${(stats[rank * 8 + 5] ?? 0).toFixed(2)}x), LLR ${stats[rank * 8].toFixed(1)}, p ${formatProbability(probability)}`
      );
    }
    ctx.requestLayers();
  }

  function countSignificant(stats: Float32Array, clusters: number): number {
    let count = 0;
    for (let cluster = 0; cluster < clusters; cluster++) {
      if (stats[cluster * 8 + 3] <= ctx.options.significance) count++;
    }
    return count;
  }

  function adoptVariant(): void {
    current = getVariant();
    cpuMembership = null;
    _cpuStatistics = null;
    ctx.setReadout(
      'events',
      `${formatCount(current.knoxEvents)} of ${formatCount(current.scan.caseCount)} ${CLUSTER_SOURCES[ctx.options.source].label}`
    );
    writeBaseline();
    writeKnoxParameters();
    writeScanParameters();
    ctx.requestLayers();
  }

  adoptVariant();

  return {
    getCompiledGraphs: () =>
      current
        ? ([current.knox.compiled, current.scan.compiled] as CompiledGPUCommandGraph<never>[])
        : [],

    setOption(id) {
      if (id === 'source') {
        adoptVariant();
      } else if (
        id === 'spatialRadius' ||
        id === 'timeThreshold' ||
        id === 'permutations' ||
        id === 'showSpatialOnly'
      ) {
        writeKnoxParameters();
      } else if (id === 'baseline') {
        writeBaseline();
      } else if (
        id === 'maximumPopulationFraction' ||
        id === 'maximumWindowZones' ||
        id === 'maximumTimeBuckets' ||
        id === 'windowShape' ||
        id === 'scanPermutations' ||
        id === 'significance'
      ) {
        writeScanParameters();
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'reseed') {
        seed++;
        writeKnoxParameters();
        writeScanParameters();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip({coordinate}) {
      if (!current || !coordinate || ctx.options.analysis !== 'scan') return null;
      const feature = lookupFeature(raster, coordinate[0], coordinate[1]);
      if (feature < 0) return null;
      const lines = [`Tract ${tracts.geoid[feature] ?? feature}`];
      lines.push(`${formatCount(tracts.population[feature] || 0)} residents`);
      let cases = 0;
      for (let bucket = 0; bucket < BUCKET_COUNT; bucket++) {
        cases += current.scan.totals[feature * BUCKET_COUNT + bucket];
      }
      lines.push(`${formatCount(cases)} ${CLUSTER_SOURCES[ctx.options.source].label} in 2023`);
      const rank = cpuMembership?.[feature] ?? 0;
      if (rank > 0 && rank !== NO_CELL) lines.push(`In significant cluster #${rank}`);
      return lines.join('\n');
    },

    encode(commandEncoder, frame) {
      if (!current) return;
      if (knoxDirty || frame.frameIndex < 2) {
        current.knox.compiled.encode(commandEncoder, {parameters: undefined});
        knoxDirty = false;
        current.knox.reader.request(commandEncoder);
      }
      const settled = performance.now() - scanChangedAt > SETTLE_MILLISECONDS;
      if ((scanDirty && settled) || (!scanActive && frame.frameIndex > 3)) {
        scanActive = true;
        current.scan.compiled.encode(commandEncoder, {parameters: undefined});
        scanDirty = false;
        current.scan.reader.request(commandEncoder);
      }
      current.knox.reader.flush(commandEncoder);
      current.scan.reader.flush(commandEncoder);
    },

    getLayers() {
      if (!current) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const layers: Layer[] = [];
      if (options.analysis === 'pairs') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'clusters-pairs',
            coordinateOrigin,
            segments: current.knox.segments,
            weights: current.knox.fade,
            instanceCount: current.knox.slotCapacity,
            widthPixels: 1.4,
            colormap: 'uniform',
            color: dark ? [255, 170, 60, 220] : [230, 110, 10, 220]
          }),
          new SpatialAnalysisPointLayer({
            id: 'clusters-events',
            coordinateOrigin,
            positions: current.knox.positions,
            instanceCount: current.knox.eventCount,
            radiusPixels: 2.6,
            values: current.knox.times,
            valueFormat: 'float32',
            colormap: 'lajolla',
            valueRange: [0, 365],
            color: [255, 255, 255, 235]
          })
        );
        return layers;
      }
      const common = {
        coordinateOrigin,
        gridSize: [raster.width, raster.height] as const,
        bounds: raster.bounds,
        valueIndices: cellFeatureBuffer,
        opacity: options.opacity,
        color: [255, 255, 255, 255] as SpatialAnalysisColor
      };
      if (options.scanView === 'clusters') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...common,
            id: 'clusters-membership',
            values: current.scan.membership,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: CLUSTER_PALETTE
          })
        );
      } else if (options.scanView === 'zone-llr') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...common,
            id: 'clusters-llr',
            values: current.scan.zoneStatistics,
            valueFormat: 'float32',
            colormap: 'inferno',
            valueRange: [0, llrMaximum],
            sqrtScale: true,
            discardAtOrBelow: 0
          })
        );
      } else {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...common,
            id: 'clusters-rate',
            values: current.scan.rate,
            valueFormat: 'float32',
            colormap: 'ylorrd',
            valueRange: [0, rateMaximum],
            sqrtScale: true
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'clusters-outlines',
          coordinateOrigin,
          segments: outlineBuffer,
          instanceCount: outlineSegments.length / 4,
          widthPixels: 0.8,
          colormap: 'uniform',
          color: dark ? [230, 235, 245, 70] : [40, 50, 70, 80]
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const variant of variants.values()) {
        variant.knox.reader.stop();
        variant.scan.reader.stop();
      }
      resources.destroy();
    }
  };
}
