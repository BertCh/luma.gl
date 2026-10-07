// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Kulldorff scan statistic over New York taxi pickups. Pickups are binned into a 32 x 32 grid of
 * zones and 12 time buckets. The baseline of a zone-bucket is the expectation from the zone and
 * bucket totals alone (`zone total * bucket total / total`), so a cluster is a place and time
 * with more pickups than that independence model: a space-time anomaly. `GPUSpatialScanStatistic`
 * scans every circular window (the nearest zones of each center, up to a population share) and
 * every run of consecutive buckets (a cylinder), keeps the most likely cluster and non-overlapping
 * secondary clusters, and ranks them against Monte Carlo replicates (multinomial redistribution of
 * the pickups) of the largest log-likelihood ratio.
 *
 * Window size, population share, time window, window shape, replicate count and seed are buffer
 * writes: one compiled graph is re-encoded and the rebuild counter stays 0. The cluster list
 * (a few hundred bytes) comes back through one ring-buffered readback; the rings are drawn by a
 * kernel from the cluster buffers.
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {
  getGPUSpatialScanParameterValues,
  GPUSpatialScanStatistic,
  GPU_SCAN_STATISTIC_CLUSTER,
  GPU_SCAN_STATISTIC_CLUSTER_INDEX,
  GPU_SCAN_STATISTIC_INDEX_WORDS,
  GPU_SCAN_STATISTIC_PARAMETER_LENGTH,
  GPU_SCAN_STATISTIC_STATISTIC_WORDS,
  GPU_SCAN_STATISTIC_SUMMARY,
  GPU_SCAN_STATISTIC_SUMMARY_LENGTH,
  type GPUSpatialScanWindowShape
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';

const GRID_SIDE = 32;
const TIME_BUCKETS = 12;
const MAXIMUM_PERMUTATIONS = 199;
const MAXIMUM_CLUSTERS = 5;
/** Segments per drawn cluster ring. */
const RING_SEGMENTS = 64;

function formatProbability(value: number): string {
  return value < 0.001 ? '< 0.001' : value.toFixed(3);
}

export const scanStatisticMode: SpatialAnalysisModeDefinition = {
  id: 'scan-statistic',
  title: 'Scan statistic',
  contributors: ['GPUSpatialScanStatistic'],
  description:
    'Where and when are there more taxi pickups than the zone and hour totals predict? Every ' +
    'circular window and run of hours is scored with a Poisson likelihood ratio; the orange ring ' +
    'is the most likely cluster, cyan rings are non-overlapping secondary clusters, and zone ' +
    'color shows the best score of windows centered there. Move the window and time sliders ' +
    'and read LLR, Monte Carlo p and observed versus expected counts per cluster.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'scan-statistic');

    // Pickups: the first vertex of every trip, inside the central 99% of the city.
    const tripCount = trips.tripOffsets.length - 1;
    const pickups: {x: number; y: number; time: number}[] = [];
    for (let trip = 0; trip < tripCount; trip++) {
      const vertex = trips.tripOffsets[trip];
      if (trips.tripOffsets[trip + 1] <= vertex) continue;
      pickups.push({
        x: trips.vertexPositions[vertex * 2],
        y: trips.vertexPositions[vertex * 2 + 1],
        time: trips.vertexTimestamps[vertex]
      });
    }
    const getRange = (values: number[]): [number, number] => {
      const sorted = Float32Array.from(values).sort();
      return [
        sorted[Math.floor(sorted.length * 0.005)],
        sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.995))]
      ];
    };
    const [minimumX, maximumX] = getRange(pickups.map(pickup => pickup.x));
    const [minimumY, maximumY] = getRange(pickups.map(pickup => pickup.y));
    const kept = pickups.filter(
      pickup =>
        pickup.x >= minimumX && pickup.x <= maximumX && pickup.y >= minimumY && pickup.y <= maximumY
    );
    let minimumTime = Infinity;
    let maximumTime = -Infinity;
    for (const pickup of kept) {
      minimumTime = Math.min(minimumTime, pickup.time);
      maximumTime = Math.max(maximumTime, pickup.time);
    }
    const cellSize = Math.max(maximumX - minimumX, maximumY - minimumY) / GRID_SIDE;
    const bucketSeconds = Math.max(maximumTime - minimumTime, 1) / TIME_BUCKETS;

    const zoneCount = GRID_SIDE * GRID_SIDE;
    const cellCount = zoneCount * TIME_BUCKETS;
    const zonePositions = new Float32Array(zoneCount * 2);
    for (let zone = 0; zone < zoneCount; zone++) {
      zonePositions[zone * 2] = minimumX + ((zone % GRID_SIDE) + 0.5) * cellSize;
      zonePositions[zone * 2 + 1] = minimumY + (Math.floor(zone / GRID_SIDE) + 0.5) * cellSize;
    }
    const cases = new Uint32Array(cellCount);
    const zoneTotals = new Float64Array(zoneCount);
    const bucketTotals = new Float64Array(TIME_BUCKETS);
    const pickupPositions = new Float32Array(kept.length * 2);
    kept.forEach((pickup, index) => {
      const column = Math.min(GRID_SIDE - 1, Math.floor((pickup.x - minimumX) / cellSize));
      const row = Math.min(GRID_SIDE - 1, Math.floor((pickup.y - minimumY) / cellSize));
      const bucket = Math.min(
        TIME_BUCKETS - 1,
        Math.floor((pickup.time - minimumTime) / bucketSeconds)
      );
      const zone = row * GRID_SIDE + column;
      cases[zone * TIME_BUCKETS + bucket]++;
      zoneTotals[zone]++;
      bucketTotals[bucket]++;
      pickupPositions[index * 2] = pickup.x;
      pickupPositions[index * 2 + 1] = pickup.y;
    });
    const baseline = new Float32Array(cellCount);
    for (let zone = 0; zone < zoneCount; zone++) {
      for (let bucket = 0; bucket < TIME_BUCKETS; bucket++) {
        baseline[zone * TIME_BUCKETS + bucket] =
          (zoneTotals[zone] * bucketTotals[bucket]) / Math.max(kept.length, 1);
      }
    }

    let maximumWindowZones = 24;
    let populationPercent = 20;
    let maximumBuckets = 4;
    let permutations = 99;
    let seed = 1;
    let windowShape: GPUSpatialScanWindowShape = 'circle';
    let dirty = true;
    let primaryLikelihood = 1;
    let clusterSignature = '';

    const positionsBuffer = resources.createBuffer('zone-positions', zonePositions);
    const casesBuffer = resources.createBuffer('cases', cases);
    const baselineBuffer = resources.createBuffer('baseline', baseline);
    const pickupBuffer = resources.createBuffer('pickups', pickupPositions);
    const clusterIndicesBuffer = resources.createBuffer(
      'cluster-indices',
      MAXIMUM_CLUSTERS * GPU_SCAN_STATISTIC_INDEX_WORDS * 4
    );
    const clusterStatisticsBuffer = resources.createBuffer(
      'cluster-statistics',
      MAXIMUM_CLUSTERS * GPU_SCAN_STATISTIC_STATISTIC_WORDS * 4
    );
    const replicateBuffer = resources.createBuffer(
      'replicate-maxima',
      (MAXIMUM_PERMUTATIONS + 1) * 4
    );
    const summaryBuffer = resources.createBuffer('summary', GPU_SCAN_STATISTIC_SUMMARY_LENGTH * 4);
    const zoneStatisticsBuffer = resources.createBuffer('zone-statistics', zoneCount * 4);
    const primarySegmentsBuffer = resources.createBuffer('primary-segments', RING_SEGMENTS * 16);
    const primaryFadeBuffer = resources.createBuffer('primary-fade', RING_SEGMENTS * 4);
    const secondarySegmentsBuffer = resources.createBuffer(
      'secondary-segments',
      (MAXIMUM_CLUSTERS - 1) * RING_SEGMENTS * 16
    );
    const secondaryFadeBuffer = resources.createBuffer(
      'secondary-fade',
      (MAXIMUM_CLUSTERS - 1) * RING_SEGMENTS * 4
    );
    const parameters = resources.createParameterBuffer(
      'parameters',
      'uint32',
      GPU_SCAN_STATISTIC_PARAMETER_LENGTH
    );

    function writeParameters(): void {
      parameters.write(
        getGPUSpatialScanParameterValues({
          seed,
          permutations,
          maximumPopulationFraction: populationPercent / 100,
          maximumWindowZones,
          maximumTimeBuckets: maximumBuckets,
          windowShape
        })
      );
    }
    writeParameters();

    const graph = new GPUCommandGraph<void>(device, {id: 'scan-statistic'});
    const view = <Format extends GPUVectorFormat>(
      name: string,
      buffer: ReturnType<SpatialAnalysisResources['createBuffer']>,
      format: Format,
      length?: number
    ) => importGraphBuffer(graph, name, buffer, format, length);
    const positionsView = view('zone-positions', positionsBuffer, 'float32x2', zoneCount);
    const clusterIndicesView = view(
      'cluster-indices',
      clusterIndicesBuffer,
      'uint32',
      MAXIMUM_CLUSTERS * GPU_SCAN_STATISTIC_INDEX_WORDS
    );
    const clusterStatisticsView = view(
      'cluster-statistics',
      clusterStatisticsBuffer,
      'float32',
      MAXIMUM_CLUSTERS * GPU_SCAN_STATISTIC_STATISTIC_WORDS
    );
    graph.add(
      new GPUSpatialScanStatistic({
        id: 'scan',
        positions: positionsView,
        cases: view('cases', casesBuffer, 'uint32', cellCount),
        baseline: view('baseline', baselineBuffer, 'float32', cellCount),
        timeBuckets: TIME_BUCKETS,
        maximumWindowZones: 32,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        maximumClusters: MAXIMUM_CLUSTERS,
        parameters: parameters.importToGraph(graph),
        clusterIndices: clusterIndicesView,
        clusterStatistics: clusterStatisticsView,
        statistics: view('replicate-maxima', replicateBuffer, 'float32', MAXIMUM_PERMUTATIONS + 1),
        summary: view('summary', summaryBuffer, 'uint32', GPU_SCAN_STATISTIC_SUMMARY_LENGTH),
        zoneStatistics: view('zone-statistics', zoneStatisticsBuffer, 'float32', zoneCount)
      })
    );
    // One thread per ring segment: cluster 0 writes the primary ring, the others the secondary
    // buffer; unused cluster slots (no expected cases) get fade 0.
    const {center} = GPU_SCAN_STATISTIC_CLUSTER_INDEX;
    addKernelPass(graph, {
      id: 'scan-rings',
      invocationCount: MAXIMUM_CLUSTERS * RING_SEGMENTS,
      declarations: `const RING: u32 = ${RING_SEGMENTS}u;
const MARGIN: f32 = ${(cellSize * 0.6).toFixed(2)};
const TAU: f32 = 6.2831853;`,
      bindings: [
        {name: 'positions', view: positionsView, type: 'f32', access: 'read'},
        {name: 'clusterIndices', view: clusterIndicesView, type: 'u32', access: 'read'},
        {name: 'clusterStats', view: clusterStatisticsView, type: 'f32', access: 'read'},
        {
          name: 'primary',
          view: view('primary-segments', primarySegmentsBuffer, 'float32', RING_SEGMENTS * 4),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'primaryFade',
          view: view('primary-fade', primaryFadeBuffer, 'float32', RING_SEGMENTS),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'secondary',
          view: view(
            'secondary-segments',
            secondarySegmentsBuffer,
            'float32',
            (MAXIMUM_CLUSTERS - 1) * RING_SEGMENTS * 4
          ),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'secondaryFade',
          view: view(
            'secondary-fade',
            secondaryFadeBuffer,
            'float32',
            (MAXIMUM_CLUSTERS - 1) * RING_SEGMENTS
          ),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let cluster = index / RING;
  let step = index - cluster * RING;
  let statsBase = cluster * ${GPU_SCAN_STATISTIC_STATISTIC_WORDS}u;
  let occupied = clusterStats[clusterStatsOffset + statsBase + ${GPU_SCAN_STATISTIC_CLUSTER.expectedCases}u] > 0.0;
  let zone = clusterIndices[clusterIndicesOffset + cluster * ${GPU_SCAN_STATISTIC_INDEX_WORDS}u + ${center}u];
  let radius = clusterStats[clusterStatsOffset + statsBase + ${GPU_SCAN_STATISTIC_CLUSTER.radius}u] + MARGIN;
  let centerX = positions[positionsOffset + zone * 2u];
  let centerY = positions[positionsOffset + zone * 2u + 1u];
  let angle0 = TAU * f32(step) / f32(RING);
  let angle1 = TAU * f32(step + 1u) / f32(RING);
  let segment = vec4<f32>(
    centerX + radius * cos(angle0), centerY + radius * sin(angle0),
    centerX + radius * cos(angle1), centerY + radius * sin(angle1)
  );
  let fade = select(0.0, 1.0, occupied);
  if (cluster == 0u) {
    for (var component = 0u; component < 4u; component++) {
      primary[primaryOffset + step * 4u + component] = segment[component];
    }
    primaryFade[primaryFadeOffset + step] = fade;
  } else {
    let slot = (cluster - 1u) * RING + step;
    for (var component = 0u; component < 4u; component++) {
      secondary[secondaryOffset + slot * 4u + component] = segment[component];
    }
    secondaryFade[secondaryFadeOffset + slot] = fade;
  }`
    });
    const compiled = resources.track(graph.compile());

    const indexWords = MAXIMUM_CLUSTERS * GPU_SCAN_STATISTIC_INDEX_WORDS;
    const statisticWords = MAXIMUM_CLUSTERS * GPU_SCAN_STATISTIC_STATISTIC_WORDS;
    const reader = new SummaryReader(
      resources,
      'scan-statistic',
      [
        {buffer: summaryBuffer, size: GPU_SCAN_STATISTIC_SUMMARY_LENGTH * 4},
        {buffer: clusterIndicesBuffer, size: indexWords * 4},
        {buffer: clusterStatisticsBuffer, size: statisticWords * 4}
      ],
      bytes => {
        const summaryBytes = GPU_SCAN_STATISTIC_SUMMARY_LENGTH * 4;
        const summary = new Uint32Array(bytes.slice(0, summaryBytes));
        const indices = new Uint32Array(bytes.slice(summaryBytes, summaryBytes + indexWords * 4));
        const statistics = new Float32Array(bytes.slice(summaryBytes + indexWords * 4));
        showClusters(summary, indices, statistics);
      }
    );

    function describeBuckets(first: number, last: number): string {
      const start = Math.round((minimumTime + first * bucketSeconds - minimumTime) / 60);
      const end = Math.round((minimumTime + (last + 1) * bucketSeconds - minimumTime) / 60);
      return `${start}-${end} min`;
    }

    // Controls.
    context.controls.addSlider({
      label: 'Max window size (zones)',
      min: 1,
      max: 32,
      step: 1,
      value: maximumWindowZones,
      format: value => `${value} zones (~${Math.round(Math.sqrt(value / Math.PI) * cellSize)} m)`,
      onChange: value => {
        maximumWindowZones = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Max window population share',
      min: 1,
      max: 50,
      step: 1,
      value: populationPercent,
      format: value => `${value} %`,
      onChange: value => {
        populationPercent = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Max time window',
      min: 1,
      max: TIME_BUCKETS,
      step: 1,
      value: maximumBuckets,
      format: value =>
        `${value} of ${TIME_BUCKETS} buckets (${Math.round((value * bucketSeconds) / 60)} min)`,
      onChange: value => {
        maximumBuckets = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSelect({
      label: 'Window shape',
      options: [
        {value: 'circle', label: 'Circle (never splits equidistant zones)'},
        {value: 'nearest', label: 'k nearest zones'}
      ],
      value: windowShape,
      onChange: value => {
        windowShape = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Monte Carlo replicates',
      min: 19,
      max: MAXIMUM_PERMUTATIONS,
      step: 10,
      value: permutations,
      onChange: value => {
        permutations = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addButton({
      label: 'New Monte Carlo seed',
      onClick: () => {
        seed++;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addLegend({
      title: 'Zone score: best LLR of windows centered there',
      gradient: {
        colors: [
          [0, 0, 4],
          [187, 55, 84],
          [252, 255, 164]
        ],
        minimumLabel: '0',
        maximumLabel: 'most likely'
      }
    });
    context.controls.addReadout('Pickups', formatCount(kept.length));
    context.controls.addReadout(
      'Zones x buckets',
      `${GRID_SIDE}x${GRID_SIDE} x ${TIME_BUCKETS} (${Math.round(cellSize)} m, ${Math.round(bucketSeconds / 60)} min)`
    );
    const clusterCountReadout = context.controls.addReadout('Clusters found');
    const clusterReadouts = Array.from({length: MAXIMUM_CLUSTERS}, (_, rank) =>
      context.controls.addReadout(rank === 0 ? 'Most likely' : `Secondary ${rank}`)
    );
    const detailReadouts = Array.from({length: MAXIMUM_CLUSTERS}, () =>
      context.controls.addNote('')
    );
    context.controls.addReadout('Data', trips.attribution);

    function showClusters(
      summary: Uint32Array,
      indices: Uint32Array,
      statistics: Float32Array
    ): void {
      const count = summary[GPU_SCAN_STATISTIC_SUMMARY.clusterCount];
      clusterCountReadout.setValue(
        `${count} (${summary[GPU_SCAN_STATISTIC_SUMMARY.permutations]} replicates)`
      );
      let signature = '';
      for (let rank = 0; rank < MAXIMUM_CLUSTERS; rank++) {
        if (rank >= count) {
          clusterReadouts[rank].setValue('–');
          detailReadouts[rank].setValue('');
          continue;
        }
        const record = rank * GPU_SCAN_STATISTIC_INDEX_WORDS;
        const statistic = rank * GPU_SCAN_STATISTIC_STATISTIC_WORDS;
        const likelihood = statistics[statistic + GPU_SCAN_STATISTIC_CLUSTER.logLikelihoodRatio];
        const pValue = statistics[statistic + GPU_SCAN_STATISTIC_CLUSTER.pValue];
        const observed = statistics[statistic + GPU_SCAN_STATISTIC_CLUSTER.observedCases];
        const expected = statistics[statistic + GPU_SCAN_STATISTIC_CLUSTER.expectedCases];
        clusterReadouts[rank].setValue(
          `LLR ${likelihood.toFixed(1)}, p ${formatProbability(pValue)}`
        );
        detailReadouts[rank].setValue(
          `${formatCount(observed)} observed vs ${expected.toFixed(1)} expected ` +
            `(${statistics[statistic + GPU_SCAN_STATISTIC_CLUSTER.observedOverExpected].toFixed(2)}x), ` +
            `${indices[record + GPU_SCAN_STATISTIC_CLUSTER_INDEX.zoneCount]} zones, ` +
            `r ${Math.round(statistics[statistic + GPU_SCAN_STATISTIC_CLUSTER.radius])} m, ` +
            describeBuckets(
              indices[record + GPU_SCAN_STATISTIC_CLUSTER_INDEX.firstBucket],
              indices[record + GPU_SCAN_STATISTIC_CLUSTER_INDEX.lastBucket]
            )
        );
        if (rank === 0) {
          primaryLikelihood = Math.max(likelihood, 1);
        }
        signature += `${indices[record]}:${likelihood.toFixed(2)};`;
      }
      if (signature !== clusterSignature) {
        clusterSignature = signature;
        context.updateLayers();
      }
    }

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        if (dirty || frame.frameIndex < 2) {
          compiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
          reader.request(commandEncoder);
        }
        reader.flush(commandEncoder);
      },
      getLayers(): Layer[] {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        return [
          new SpatialAnalysisPointLayer({
            id: 'scan-pickups',
            coordinateOrigin,
            positions: pickupBuffer,
            instanceCount: kept.length,
            radiusPixels: 1.2,
            colormap: 'uniform',
            color: [170, 190, 220, 70]
          }),
          new SpatialAnalysisPointLayer({
            id: 'scan-zone-score',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: zoneCount,
            radiusPixels: 7,
            values: zoneStatisticsBuffer,
            valueFormat: 'float32',
            colormap: 'inferno',
            valueRange: [0, primaryLikelihood],
            discardAtOrBelow: 0,
            updateTriggers: {valueRange: primaryLikelihood}
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'scan-secondary-clusters',
            coordinateOrigin,
            segments: secondarySegmentsBuffer,
            weights: secondaryFadeBuffer,
            instanceCount: (MAXIMUM_CLUSTERS - 1) * RING_SEGMENTS,
            widthPixels: 2,
            colormap: 'uniform',
            color: [80, 220, 255, 230]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'scan-primary-cluster',
            coordinateOrigin,
            segments: primarySegmentsBuffer,
            weights: primaryFadeBuffer,
            instanceCount: RING_SEGMENTS,
            widthPixels: 3.5,
            colormap: 'uniform',
            color: [255, 150, 40, 255]
          })
        ];
      },
      destroy() {
        reader.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
