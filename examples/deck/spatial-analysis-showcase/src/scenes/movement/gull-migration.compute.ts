// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTemporalReductionParameterValues,
  GPUTemporalReduction
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPULineSimplificationParameterValues,
  getGPUTrajectoryMetricsParameterValues,
  GPULineSimplification,
  GPUTrackSimilarity,
  GPUTrajectoryMetrics,
  GPUTrajectoryResample,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {createPlaybackClock} from '../../engine/playback';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {binValues, histogramChart} from './f-chart-helpers';
import {KeptSegmentLayer, StopMarkerLayer} from './b12-layers';
import {findNearestTrack, formatDuration, loadGullTracks} from './b12-tracks';

/** Option state of the gull migration scene. */
export type GullMigrationOptions = {
  colorBy: 'date' | 'speed' | 'family' | 'similarity' | 'sex' | 'plain';
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  trackOpacity: number;
  showStops: boolean;
  stopSpeed: number;
  stopHours: number;
  showSimplified: boolean;
  simplifyMetric: 'segment' | 'time-ratio';
  toleranceLog: number;
  routeSpacing: 'arc-length' | 'time';
  similarityMetric: 'hausdorff' | 'frechet';
  familyCount: number;
  similarityRangeKm: number;
  showWeekly: boolean;
  bucketDays: number;
  dayRange: readonly [number, number];
  snapshotDay: number;
  play: boolean;
  playSpeed: number;
  loop: boolean;
};

/** Family colors (up to six) and the sex colors share one palette. */
export const GULL_FAMILY_COLORS: readonly (readonly [number, number, number, number])[] = [
  [86, 180, 233, 255],
  [240, 150, 30, 255],
  [0, 158, 115, 255],
  [204, 121, 167, 255],
  [240, 228, 66, 255],
  [150, 156, 168, 255]
];
export const GULL_SEX_COLORS: readonly (readonly [number, number, number, number])[] = [
  [230, 120, 160, 255],
  [80, 150, 240, 255],
  [150, 156, 168, 255]
];

const STOP_CAPACITY = 2048;
const SIMULATION_ROUNDS = 96;
const SIMILARITY_SAMPLES = 96;
const BUCKET_COUNT = 64;
const SECONDS_PER_DAY = 86400;
const KILOMETERS_PER_SECOND_FACTOR = 3.6;
const SETTLE_MILLISECONDS = 200;
const NO_BIRD = 0xffffffff;
const EPOCH_MILLISECONDS = Date.UTC(2015, 6, 15);

/** `15 Jul` for a day count since 15 July 2015. */
export function formatGullDate(day: number): string {
  const date = new Date(EPOCH_MILLISECONDS + day * SECONDS_PER_DAY * 1000);
  return `${date.getUTCDate()} ${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][date.getUTCMonth()]}`;
}

type SimplifyVariant = {
  importance: Buffer;
  converged: Buffer;
  roundCount: Buffer;
  keptIds: Buffer;
  keptCount: Buffer;
  keptOverflow: Buffer;
  keptTotal: Buffer;
  draw: DrawCommandBuffer;
  importanceCompiled: CompiledGPUCommandGraph<void>;
  selectionCompiled: CompiledGPUCommandGraph<void>;
  importanceEncoded: boolean;
  reader: SummaryReader;
  snapshot: {kept: number; total: number; converged: boolean; rounds: number} | null;
};

type TemporalColumns = {
  counts: Buffer;
  min: Buffer;
  max: Buffer;
  first: Buffer;
  last: Buffer;
  occupiedIds: Buffer;
  occupiedCount: Buffer;
  occupiedOverflow: Buffer;
};

/**
 * Gull migration: 31 lesser black-backed gulls tracked hourly from July to November 2015. One graph
 * measures each track and detects stopovers; two pairs of graphs compute Douglas-Peucker and TD-TR
 * (time-ratio) importance once and select kept vertices per tolerance; resampled routes feed an
 * all-pairs similarity that the CPU clusters into route families; a temporal reduction thins the
 * hourly fixes to one per bird per bucket of days.
 *
 * Analysis runs in azimuthal-equidistant meters (true distances across 40 degrees of latitude);
 * drawing uses the longitude/latitude degrees of the same rows with `LNGLAT` layers.
 */
export async function createGullMigration(
  ctx: SceneContext<GullMigrationOptions>
): Promise<SceneInstance<GullMigrationOptions>> {
  const gulls = loadGullTracks(ctx.datasets.get('gull-migration'));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = gulls;
  const resources = new SpatialAnalysisResources(device, 'gulls');
  const drawProps = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;
  const slotCount = trackCount * BUCKET_COUNT;

  // ---- Static inputs --------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', gulls.positions);
  const lngLatBuffer = resources.createBuffer('lng-lat', gulls.lngLat);
  const timestampsBuffer = resources.createBuffer('timestamps', gulls.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', gulls.offsets);
  const segmentsBuffer = resources.createBuffer('segments', gulls.segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', gulls.segmentTracks);
  const segmentEndsBuffer = resources.createBuffer('segment-ends', gulls.segmentEndVertices);
  const sexBuffer = resources.createBuffer('sex', Uint32Array.from(gulls.sex));
  const vertexTracks = new Uint32Array(vertexCount);
  const longitudes = new Float32Array(vertexCount);
  const latitudes = new Float32Array(vertexCount);
  for (let track = 0; track < trackCount; track++) {
    vertexTracks.fill(track, gulls.offsets[track], gulls.offsets[track + 1]);
  }
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    longitudes[vertex] = gulls.lngLat[vertex * 2];
    latitudes[vertex] = gulls.lngLat[vertex * 2 + 1];
  }
  const vertexTracksBuffer = resources.createBuffer('vertex-tracks', vertexTracks);
  const longitudeBuffer = resources.createBuffer('longitudes', longitudes);
  const latitudeBuffer = resources.createBuffer('latitudes', latitudes);
  const speedBuffer = resources.createBuffer('speeds', gulls.speed);
  const familyBuffer = resources.createBuffer('family', new Uint32Array(trackCount));
  const distanceBuffer = resources.createBuffer('selected-distance', trackCount * 4);
  const selectedSegments = resources.createBuffer('selected-segments', gulls.longestTrack * 16);

  // ---- Metrics graph ---------------------------------------------------------------------------
  const trackLengths = resources.createBuffer('track-lengths', trackCount * 4);
  const trackDurations = resources.createBuffer('track-durations', trackCount * 4);
  const averageSpeeds = resources.createBuffer('average-speeds', trackCount * 4);
  const maximumSpeeds = resources.createBuffer('maximum-speeds', trackCount * 4);
  const stepSpeeds = resources.createBuffer('step-speeds', vertexCount * 4);
  const trackStopCounts = resources.createBuffer('track-stop-counts', trackCount * 4);
  const stopIds = resources.createBuffer('stop-ids', STOP_CAPACITY * 4);
  const stopCount = resources.createBuffer('stop-count', 4);
  const stopOverflow = resources.createBuffer('stop-overflow', 4);
  const stopTotal = resources.createBuffer('stop-total', 4);
  const stopCentroids = resources.createBuffer('stop-centroids', STOP_CAPACITY * 8);
  const stopDurations = resources.createBuffer('stop-durations', STOP_CAPACITY * 4);
  const stopStartRows = resources.createBuffer('stop-start-rows', STOP_CAPACITY * 4);
  const stopEndRows = resources.createBuffer('stop-end-rows', STOP_CAPACITY * 4);
  const stopLngLat = resources.createBuffer('stop-lng-lat', STOP_CAPACITY * 8);
  const stopParameters = resources.createParameterBuffer(
    'stop-parameters',
    'float32',
    GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH
  );
  const stopDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'gull-stop-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const metricsGraph = new GPUCommandGraph<void>(device, {id: 'gull-metrics'});
  const stopCountView = importGraphBuffer(metricsGraph, 'stop-count', stopCount, 'uint32', 1);
  const startRowsView = importGraphBuffer(
    metricsGraph,
    'stop-start-rows',
    stopStartRows,
    'uint32',
    STOP_CAPACITY
  );
  const endRowsView = importGraphBuffer(
    metricsGraph,
    'stop-end-rows',
    stopEndRows,
    'uint32',
    STOP_CAPACITY
  );
  metricsGraph.add(
    new GPUTrajectoryMetrics({
      id: 'metrics',
      positions: importGraphBuffer(
        metricsGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        vertexCount
      ),
      timestamps: importGraphBuffer(
        metricsGraph,
        'timestamps',
        timestampsBuffer,
        'float32',
        vertexCount
      ),
      trackOffsets: importGraphBuffer(
        metricsGraph,
        'offsets',
        offsetsBuffer,
        'uint32',
        trackCount + 1
      ),
      parameters: stopParameters.importToGraph(metricsGraph),
      trackLengths: importGraphBuffer(
        metricsGraph,
        'track-lengths',
        trackLengths,
        'float32',
        trackCount
      ),
      trackDurations: importGraphBuffer(
        metricsGraph,
        'track-durations',
        trackDurations,
        'float32',
        trackCount
      ),
      averageSpeeds: importGraphBuffer(
        metricsGraph,
        'average-speeds',
        averageSpeeds,
        'float32',
        trackCount
      ),
      maximumSpeeds: importGraphBuffer(
        metricsGraph,
        'maximum-speeds',
        maximumSpeeds,
        'float32',
        trackCount
      ),
      stepSpeeds: importGraphBuffer(
        metricsGraph,
        'step-speeds',
        stepSpeeds,
        'float32',
        vertexCount
      ),
      trackStopCounts: importGraphBuffer(
        metricsGraph,
        'track-stop-counts',
        trackStopCounts,
        'uint32',
        trackCount
      ),
      stops: {
        output: {
          ids: importGraphBuffer(metricsGraph, 'stop-ids', stopIds, 'uint32', STOP_CAPACITY),
          count: stopCountView,
          overflow: importGraphBuffer(metricsGraph, 'stop-overflow', stopOverflow, 'uint32', 1),
          totalCount: importGraphBuffer(metricsGraph, 'stop-total', stopTotal, 'uint32', 1)
        },
        drawInstanceCount: metricsGraph.importGPUData(
          'stop-draw-count',
          stopDraw.getInstanceCountData(0)
        ),
        startRows: startRowsView,
        endRows: endRowsView,
        centroids: importGraphBuffer(
          metricsGraph,
          'stop-centroids',
          stopCentroids,
          'float32x2',
          STOP_CAPACITY
        ),
        durations: importGraphBuffer(
          metricsGraph,
          'stop-durations',
          stopDurations,
          'float32',
          STOP_CAPACITY
        )
      }
    })
  );
  // The stop centroids are planar analysis meters; drawing needs degrees. Average the longitude and
  // latitude of the rows each stop covers.
  addKernelPass(metricsGraph, {
    id: 'stop-lng-lat',
    invocationCount: STOP_CAPACITY,
    bindings: [
      {name: 'stopCount', view: stopCountView, type: 'u32', access: 'read'},
      {name: 'startRows', view: startRowsView, type: 'u32', access: 'read'},
      {name: 'endRows', view: endRowsView, type: 'u32', access: 'read'},
      {
        name: 'lngLat',
        view: importGraphBuffer(metricsGraph, 'lng-lat', lngLatBuffer, 'float32x2', vertexCount),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'output',
        view: importGraphBuffer(
          metricsGraph,
          'stop-lng-lat',
          stopLngLat,
          'float32x2',
          STOP_CAPACITY
        ),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var lng = nan;
  var lat = nan;
  if (index < stopCount[stopCountOffset]) {
    let first = startRows[startRowsOffset + index];
    let last = endRows[endRowsOffset + index];
    var sumLng = 0.0;
    var sumLat = 0.0;
    for (var row = first; row <= last; row++) {
      sumLng += lngLat[lngLatOffset + row * 2u];
      sumLat += lngLat[lngLatOffset + row * 2u + 1u];
    }
    let n = f32(last - first + 1u);
    lng = sumLng / n;
    lat = sumLat / n;
  }
  output[outputOffset + index * 2u] = lng;
  output[outputOffset + index * 2u + 1u] = lat;`
  });
  const metricsCompiled = resources.track(metricsGraph.compile());

  // ---- Simplification graphs (both metrics compiled up front) ----------------------------------
  const toleranceParameters = resources.createParameterBuffer(
    'tolerance',
    'float32',
    GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH
  );

  function buildSimplify(metric: 'segment' | 'time-ratio'): SimplifyVariant {
    const importance = resources.createBuffer(`${metric}-importance`, vertexCount * 4);
    const converged = resources.createBuffer(`${metric}-converged`, 4);
    const roundCount = resources.createBuffer(`${metric}-rounds`, 4);
    const keptIds = resources.createBuffer(`${metric}-kept-ids`, vertexCount * 4);
    const keptCount = resources.createBuffer(`${metric}-kept-count`, 4);
    const keptOverflow = resources.createBuffer(`${metric}-kept-overflow`, 4);
    const keptTotal = resources.createBuffer(`${metric}-kept-total`, 4);
    const draw = resources.track(
      new DrawCommandBuffer(device, {
        id: `gull-kept-draw-${metric}`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const inputs = (graph: GPUCommandGraph<void>) => ({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
      trackOffsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', trackCount + 1),
      importance: importGraphBuffer(graph, 'importance', importance, 'float32', vertexCount)
    });
    const importanceGraph = new GPUCommandGraph<void>(device, {id: `gull-importance-${metric}`});
    importanceGraph.add(
      new GPULineSimplification({
        id: `importance-${metric}`,
        ...inputs(importanceGraph),
        metric,
        timestamps: importGraphBuffer(
          importanceGraph,
          'timestamps',
          timestampsBuffer,
          'float32',
          vertexCount
        ),
        maximumRounds: SIMULATION_ROUNDS,
        status: {
          converged: importGraphBuffer(importanceGraph, 'converged', converged, 'uint32', 1),
          roundCount: importGraphBuffer(importanceGraph, 'rounds', roundCount, 'uint32', 1)
        }
      })
    );
    const selectionGraph = new GPUCommandGraph<void>(device, {id: `gull-selection-${metric}`});
    selectionGraph.add(
      new GPULineSimplification({
        id: `selection-${metric}`,
        ...inputs(selectionGraph),
        metric,
        computeImportance: false,
        parameters: toleranceParameters.importToGraph(selectionGraph),
        selection: {
          output: {
            ids: importGraphBuffer(selectionGraph, 'kept-ids', keptIds, 'uint32', vertexCount),
            count: importGraphBuffer(selectionGraph, 'kept-count', keptCount, 'uint32', 1),
            overflow: importGraphBuffer(selectionGraph, 'kept-overflow', keptOverflow, 'uint32', 1),
            totalCount: importGraphBuffer(selectionGraph, 'kept-total', keptTotal, 'uint32', 1)
          }
        }
      })
    );
    const variant: SimplifyVariant = {
      importance,
      converged,
      roundCount,
      keptIds,
      keptCount,
      keptOverflow,
      keptTotal,
      draw,
      importanceCompiled: resources.track(importanceGraph.compile()),
      selectionCompiled: resources.track(selectionGraph.compile()),
      importanceEncoded: false,
      snapshot: null,
      reader: new SummaryReader(
        resources,
        `simplify-${metric}`,
        [
          {buffer: keptCount, size: 4},
          {buffer: keptOverflow, size: 4},
          {buffer: keptTotal, size: 4},
          {buffer: converged, size: 4},
          {buffer: roundCount, size: 4}
        ],
        bytes => {
          if (destroyed) return;
          const words = new Uint32Array(bytes);
          variant.snapshot = {
            kept: words[2],
            total: vertexCount,
            converged: words[3] !== 0,
            rounds: words[4]
          };
          describeSimplification();
        }
      )
    };
    return variant;
  }

  // ---- Resample + similarity graphs ------------------------------------------------------------
  const routesBuffer = resources.createBuffer('routes', trackCount * SIMILARITY_SAMPLES * 8);
  const routeOffsets = new Uint32Array(trackCount + 1);
  for (let track = 0; track <= trackCount; track++)
    routeOffsets[track] = track * SIMILARITY_SAMPLES;
  const routeOffsetsBuffer = resources.createBuffer('route-offsets', routeOffsets);
  const pairCount = (trackCount * (trackCount - 1)) / 2;
  const pairA = new Uint32Array(pairCount);
  const pairB = new Uint32Array(pairCount);
  let pairIndex = 0;
  for (let a = 0; a < trackCount; a++) {
    for (let b = a + 1; b < trackCount; b++) {
      pairA[pairIndex] = a;
      pairB[pairIndex] = b;
      pairIndex++;
    }
  }
  const pairABuffer = resources.createBuffer('pair-a', pairA);
  const pairBBuffer = resources.createBuffer('pair-b', pairB);
  const pairHausdorff = resources.createBuffer('pair-hausdorff', pairCount * 4);
  const pairFrechet = resources.createBuffer('pair-frechet', pairCount * 4);
  const pairStatus = resources.createBuffer('pair-status', pairCount * 4);

  function buildResample(spacing: 'arc-length' | 'time'): CompiledGPUCommandGraph<void> {
    const graph = new GPUCommandGraph<void>(device, {id: `gull-routes-${spacing}`});
    graph.add(
      new GPUTrajectoryResample({
        id: `routes-${spacing}`,
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
        timestamps: importGraphBuffer(
          graph,
          'timestamps',
          timestampsBuffer,
          'float32',
          vertexCount
        ),
        trackOffsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', trackCount + 1),
        sampleCount: SIMILARITY_SAMPLES,
        spacing,
        samples: importGraphBuffer(
          graph,
          'routes',
          routesBuffer,
          'float32x2',
          trackCount * SIMILARITY_SAMPLES
        )
      })
    );
    return resources.track(graph.compile());
  }
  const resampleGraphs = {
    'arc-length': buildResample('arc-length'),
    time: buildResample('time')
  };
  const similarityGraph = new GPUCommandGraph<void>(device, {id: 'gull-similarity'});
  similarityGraph.add(
    new GPUTrackSimilarity({
      id: 'route-similarity',
      positionsA: importGraphBuffer(
        similarityGraph,
        'routes',
        routesBuffer,
        'float32x2',
        trackCount * SIMILARITY_SAMPLES
      ),
      offsetsA: importGraphBuffer(
        similarityGraph,
        'route-offsets',
        routeOffsetsBuffer,
        'uint32',
        trackCount + 1
      ),
      pairA: importGraphBuffer(similarityGraph, 'pair-a', pairABuffer, 'uint32', pairCount),
      pairB: importGraphBuffer(similarityGraph, 'pair-b', pairBBuffer, 'uint32', pairCount),
      hausdorff: importGraphBuffer(
        similarityGraph,
        'pair-hausdorff',
        pairHausdorff,
        'float32',
        pairCount
      ),
      frechet: importGraphBuffer(
        similarityGraph,
        'pair-frechet',
        pairFrechet,
        'float32',
        pairCount
      ),
      status: importGraphBuffer(similarityGraph, 'pair-status', pairStatus, 'uint32', pairCount),
      maxFrechetVertices: SIMILARITY_SAMPLES
    })
  );
  const similarityCompiled = resources.track(similarityGraph.compile());

  // ---- Temporal reduction graph ----------------------------------------------------------------
  const bucketParameters = resources.createParameterBuffer(
    'bucket',
    'float32',
    2,
    getGPUTemporalReductionParameterValues(0, 7 * SECONDS_PER_DAY)
  );
  const displayParameters = resources.createParameterBuffer('weekly-display', 'float32', 4);
  const weeklyPositions = resources.createBuffer('weekly-positions', slotCount * 8);
  const focusPositions = resources.createBuffer('focus-positions', slotCount * 8);
  const weeklySegments = resources.createBuffer(
    'weekly-segments',
    trackCount * (BUCKET_COUNT - 1) * 16
  );
  const slotBuckets = new Uint32Array(slotCount);
  for (let slot = 0; slot < slotCount; slot++) slotBuckets[slot] = slot % BUCKET_COUNT;
  const slotBucketBuffer = resources.createBuffer('slot-buckets', slotBuckets);
  const segmentBucketBuffer = resources.createBuffer(
    'segment-buckets',
    Uint32Array.from(
      {length: trackCount * (BUCKET_COUNT - 1)},
      (_, index) => index % (BUCKET_COUNT - 1)
    )
  );

  function createColumns(name: string): TemporalColumns {
    return {
      counts: resources.createBuffer(`${name}-counts`, slotCount * 4),
      min: resources.createBuffer(`${name}-min`, slotCount * 4),
      max: resources.createBuffer(`${name}-max`, slotCount * 4),
      first: resources.createBuffer(`${name}-first`, slotCount * 4),
      last: resources.createBuffer(`${name}-last`, slotCount * 4),
      occupiedIds: resources.createBuffer(`${name}-occupied-ids`, slotCount * 4),
      occupiedCount: resources.createBuffer(`${name}-occupied-count`, 4),
      occupiedOverflow: resources.createBuffer(`${name}-occupied-overflow`, 4)
    };
  }
  const latitudeColumns = createColumns('latitude');
  const longitudeColumns = createColumns('longitude');
  const speedColumns = createColumns('speed');
  const temporalGraph = new GPUCommandGraph<void>(device, {id: 'gull-temporal'});
  const temporalView = <Format extends 'float32' | 'uint32'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(temporalGraph, name, buffer, format, length);
  const cellIdsView = temporalView('vertex-tracks', vertexTracksBuffer, 'uint32', vertexCount);
  const temporalTimestamps = temporalView('timestamps', timestampsBuffer, 'float32', vertexCount);
  const bucketView = bucketParameters.importToGraph(temporalGraph);
  const addReduction = (name: string, values: Buffer, columns: TemporalColumns) => {
    const counts = temporalView(`${name}-counts`, columns.counts, 'uint32', slotCount);
    temporalGraph.add(
      new GPUTemporalReduction({
        id: `reduce-${name}`,
        cellIds: cellIdsView,
        timestamps: temporalTimestamps,
        values: temporalView(`${name}-values`, values, 'float32', vertexCount),
        parameters: bucketView,
        cellCount: trackCount,
        bucketCount: BUCKET_COUNT,
        output: {
          counts,
          min: temporalView(`${name}-min`, columns.min, 'float32', slotCount),
          max: temporalView(`${name}-max`, columns.max, 'float32', slotCount),
          first: temporalView(`${name}-first`, columns.first, 'float32', slotCount),
          last: temporalView(`${name}-last`, columns.last, 'float32', slotCount),
          occupiedSlots: {
            ids: temporalView(`${name}-occupied`, columns.occupiedIds, 'uint32', slotCount),
            count: temporalView(`${name}-occupied-count`, columns.occupiedCount, 'uint32', 1),
            overflow: temporalView(
              `${name}-occupied-overflow`,
              columns.occupiedOverflow,
              'uint32',
              1
            )
          }
        }
      })
    );
    return counts;
  };
  addReduction('latitude', latitudeBuffer, latitudeColumns);
  addReduction('longitude', longitudeBuffer, longitudeColumns);
  addReduction('speed', speedBuffer, speedColumns);
  const compiledTemporal = resources.track(temporalGraph.compile());

  // Display graph: weekly positions, the focus bucket and the weekly connecting segments.
  const weeklyGraph = new GPUCommandGraph<void>(device, {id: 'gull-weekly'});
  const weeklyView = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(weeklyGraph, name, buffer, format, length);
  const weeklyPositionView = weeklyView(
    'weekly-positions',
    weeklyPositions,
    'float32x2',
    slotCount
  );
  const weeklyDisplayView = displayParameters.importToGraph(weeklyGraph);
  addKernelPass(weeklyGraph, {
    id: 'weekly-positions',
    invocationCount: slotCount,
    declarations: `const BUCKETS: u32 = ${BUCKET_COUNT}u;`,
    bindings: [
      {
        name: 'counts',
        view: weeklyView('counts', latitudeColumns.counts, 'uint32', slotCount),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'lngLast',
        view: weeklyView('lng-last', longitudeColumns.last, 'float32', slotCount),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'latLast',
        view: weeklyView('lat-last', latitudeColumns.last, 'float32', slotCount),
        type: 'f32',
        access: 'read'
      },
      {name: 'display', view: weeklyDisplayView, type: 'f32', access: 'read'},
      {name: 'weekly', view: weeklyPositionView, type: 'f32', access: 'read_write'},
      {
        name: 'focus',
        view: weeklyView('focus-positions', focusPositions, 'float32x2', slotCount),
        type: 'f32',
        access: 'read_write'
      }
    ],
    // display = [first bucket, last bucket, focus bucket, unused]
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let bucket = index % BUCKETS;
  let has = counts[countsOffset + index] > 0u;
  let inRange = f32(bucket) >= display[displayOffset] && f32(bucket) <= display[displayOffset + 1u];
  let point = vec2<f32>(lngLast[lngLastOffset + index], latLast[latLastOffset + index]);
  let weeklyPoint = select(vec2<f32>(nan), point, has && inRange);
  let focusPoint = select(vec2<f32>(nan), point, has && f32(bucket) == display[displayOffset + 2u]);
  weekly[weeklyOffset + index * 2u] = weeklyPoint.x;
  weekly[weeklyOffset + index * 2u + 1u] = weeklyPoint.y;
  focus[focusOffset + index * 2u] = focusPoint.x;
  focus[focusOffset + index * 2u + 1u] = focusPoint.y;`
  });
  addKernelPass(weeklyGraph, {
    id: 'weekly-segments',
    invocationCount: trackCount * (BUCKET_COUNT - 1),
    declarations: `const BUCKETS: u32 = ${BUCKET_COUNT}u;`,
    bindings: [
      {name: 'weekly', view: weeklyPositionView, type: 'f32', access: 'read'},
      {
        name: 'segments',
        view: weeklyView(
          'weekly-segments',
          weeklySegments,
          'float32',
          trackCount * (BUCKET_COUNT - 1) * 4
        ),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let bird = index / (BUCKETS - 1u);
  let bucket = index % (BUCKETS - 1u);
  let slot = bird * BUCKETS + bucket;
  let a = vec2<f32>(weekly[weeklyOffset + slot * 2u], weekly[weeklyOffset + slot * 2u + 1u]);
  let b = vec2<f32>(weekly[weeklyOffset + slot * 2u + 2u], weekly[weeklyOffset + slot * 2u + 3u]);
  let valid = a.x == a.x && b.x == b.x;
  segments[segmentsOffset + index * 4u] = select(nan, a.x, valid);
  segments[segmentsOffset + index * 4u + 1u] = select(nan, a.y, valid);
  segments[segmentsOffset + index * 4u + 2u] = select(nan, b.x, valid);
  segments[segmentsOffset + index * 4u + 3u] = select(nan, b.y, valid);`
  });
  const weeklyCompiled = resources.track(weeklyGraph.compile());

  // ---- State -----------------------------------------------------------------------------------
  let destroyed = false;
  let metricsDirty = true;
  let tolerancesDirty = true;
  let similarityDirty = true;
  let temporalDirty = true;
  let weeklyDirty = true;
  let settleStale = true;
  let lastChange = performance.now();
  let selectedBird = NO_BIRD;
  let currentSpacing: 'arc-length' | 'time' = ctx.options.routeSpacing;
  const simplify = {
    segment: buildSimplify('segment'),
    'time-ratio': buildSimplify('time-ratio')
  };
  let metricsSnapshot: {
    lengths: Float32Array;
    durations: Float32Array;
    maxima: Float32Array;
    stopCounts: Uint32Array;
    stopDurations: Float32Array;
    stopTotal: number;
    stopOverflow: boolean;
    stopListed: number;
  } | null = null;
  let similaritySnapshot: {
    hausdorff: Float32Array;
    frechet: Float32Array;
    status: Uint32Array;
  } | null = null;
  let matrix: Float32Array | null = null;
  let families = new Uint32Array(trackCount);
  let temporalSnapshot: {
    counts: Uint32Array;
    latitudeLast: Float32Array;
    latitudeMin: Float32Array;
    latitudeMax: Float32Array;
    speedMax: Float32Array;
  } | null = null;
  let displayMaxBucket = Math.ceil(138 / ctx.options.bucketDays);

  const markChanged = () => {
    lastChange = performance.now();
    settleStale = true;
  };

  const getToleranceMeters = () => 10 ** ctx.options.toleranceLog * 1000;
  const getBucketOf = (day: number) =>
    Math.min(BUCKET_COUNT - 1, Math.floor(day / ctx.options.bucketDays));

  function writeStopParameters(): void {
    stopParameters.write(
      getGPUTrajectoryMetricsParameterValues({
        stopSpeedThreshold: ctx.options.stopSpeed,
        stopMinimumDuration: ctx.options.stopHours * 3600
      })
    );
    metricsDirty = true;
    markChanged();
  }

  function writeBucketParameters(): void {
    bucketParameters.write(
      getGPUTemporalReductionParameterValues(0, ctx.options.bucketDays * SECONDS_PER_DAY)
    );
    displayMaxBucket = Math.ceil(139 / ctx.options.bucketDays);
    temporalDirty = true;
    weeklyDirty = true;
    markChanged();
    writeWeeklyDisplay();
  }

  function writeWeeklyDisplay(): void {
    const [from, to] = ctx.options.dayRange;
    displayParameters.write(
      Float32Array.of(getBucketOf(from), getBucketOf(to), getBucketOf(ctx.options.snapshotDay), 0)
    );
    weeklyDirty = true;
    markChanged();
  }

  function writeSelectedOutline(): void {
    const segments = new Float32Array(gulls.longestTrack * 4).fill(Number.NaN);
    if (selectedBird !== NO_BIRD) {
      const first = gulls.offsets[selectedBird];
      const last = gulls.offsets[selectedBird + 1] - 1;
      for (let vertex = first; vertex < last; vertex++) {
        segments.set(gulls.lngLat.subarray(vertex * 2, vertex * 2 + 4), (vertex - first) * 4);
      }
    }
    selectedSegments.write(segments);
  }

  function describeBird(bird: number): string {
    const sex = ['female', 'male', 'unknown sex'][gulls.sex[bird]];
    return `Gull ${gulls.birdIds[bird] ?? bird} (${sex})`;
  }

  function getFurthestSouth(bird: number): number {
    let south = 90;
    for (let vertex = gulls.offsets[bird]; vertex < gulls.offsets[bird + 1]; vertex++) {
      south = Math.min(south, gulls.lngLat[vertex * 2 + 1]);
    }
    return south;
  }

  // ---- Summaries -------------------------------------------------------------------------------
  function describeSimplification(): void {
    const parts: string[] = [];
    for (const [name, variant] of Object.entries(simplify) as [string, SimplifyVariant][]) {
      const snapshot = variant.snapshot;
      if (!snapshot) continue;
      parts.push(
        `${name === 'segment' ? 'Douglas-Peucker' : 'TD-TR'} ${formatCount(snapshot.kept)} (${((100 * snapshot.kept) / snapshot.total).toFixed(1)}%)${snapshot.converged ? '' : ' *'}`
      );
    }
    ctx.setReadout('simplified', parts.join(' vs '));
    const active = simplify[ctx.options.simplifyMetric].snapshot;
    ctx.setReadout(
      'rounds',
      active
        ? `${active.rounds} of ${SIMULATION_ROUNDS}${active.converged ? ', converged' : ', not converged (superset of the exact result)'}`
        : 'n/a'
    );
    ctx.setReadout('tolerance', `${formatTolerance(getToleranceMeters())}`);
  }

  function formatTolerance(meters: number): string {
    return meters >= 1000
      ? `${(meters / 1000).toFixed(meters >= 10000 ? 0 : 1)} km`
      : `${meters.toFixed(0)} m`;
  }

  // Ground speed of every hourly fix (observed by the tag): static, so charted once.
  {
    const speedKmh = Float32Array.from(gulls.speed, speed => speed * KILOMETERS_PER_SECOND_FACTOR);
    ctx.setChart(
      'speedChart',
      histogramChart(binValues(speedKmh, 0, 80, 32), 0, 80, {
        xLabel: 'ground speed of an hourly fix (km/h)',
        yLabel: 'fixes',
        formatX: value => value.toFixed(0),
        formatY: value => (value >= 1000 ? `${Math.round(value / 1000)}k` : `${value}`),
        description:
          'Histogram of tag ground speed over all hourly fixes. Most fixes are near zero (resting, foraging); a second hump at 30 to 50 km/h is migration flight.'
      })
    );
  }

  function summarizeMetrics(): void {
    const snapshot = metricsSnapshot;
    if (!snapshot) return;
    let total = 0;
    let fastest = 0;
    let longestBird = 0;
    for (let bird = 0; bird < trackCount; bird++) {
      total += snapshot.lengths[bird];
      fastest = Math.max(fastest, snapshot.maxima[bird]);
      if (snapshot.lengths[bird] > snapshot.lengths[longestBird]) longestBird = bird;
    }
    ctx.setReadout(
      'distance',
      `${formatCount(total / 1000)} km total, ${formatCount(total / 1000 / trackCount)} km per bird`
    );
    ctx.setReadout(
      'longestFlight',
      `${describeBird(longestBird)}: ${formatCount(snapshot.lengths[longestBird] / 1000)} km`
    );
    ctx.setReadout(
      'fastest',
      `${(fastest * KILOMETERS_PER_SECOND_FACTOR).toFixed(0)} km/h (hourly step)`
    );
    let birdsWithStops = 0;
    for (let bird = 0; bird < trackCount; bird++)
      if (snapshot.stopCounts[bird] > 0) birdsWithStops++;
    let longest = 0;
    for (let stop = 0; stop < snapshot.stopListed; stop++) {
      longest = Math.max(longest, snapshot.stopDurations[stop]);
    }
    ctx.setReadout(
      'stops',
      `${formatCount(snapshot.stopTotal)} stopovers${snapshot.stopOverflow ? ' (list truncated)' : ''}, ${birdsWithStops} of ${trackCount} birds`
    );
    ctx.setReadout('longestStop', formatDuration(longest));
    const stayDays = Float32Array.from(
      snapshot.stopDurations.subarray(0, snapshot.stopListed),
      seconds => seconds / SECONDS_PER_DAY
    );
    ctx.setChart(
      'stopChart',
      snapshot.stopListed
        ? histogramChart(binValues(stayDays, 0, 40, 20), 0, 40, {
            xLabel: 'length of stopover (days, 40 and over in the last bin)',
            yLabel: 'stopovers',
            color: 3,
            formatX: value => value.toFixed(0),
            description: 'Histogram of stopover durations at the current speed and minimum stay.'
          })
        : null
    );
    describeSelected();
  }

  function buildMatrix(): void {
    if (!similaritySnapshot) return;
    const source =
      ctx.options.similarityMetric === 'frechet'
        ? similaritySnapshot.frechet
        : similaritySnapshot.hausdorff;
    matrix = new Float32Array(trackCount * trackCount);
    for (let pair = 0; pair < pairCount; pair++) {
      matrix[pairA[pair] * trackCount + pairB[pair]] = source[pair];
      matrix[pairB[pair] * trackCount + pairA[pair]] = source[pair];
    }
    clusterFamilies();
    writeSelectedDistances();
  }

  /** Average-linkage agglomerative clustering of the route distance matrix into `familyCount`. */
  function clusterFamilies(): void {
    if (!matrix) return;
    const clusters: number[][] = Array.from({length: trackCount}, (_, bird) => [bird]);
    const distance = (a: number[], b: number[]) => {
      let sum = 0;
      for (const first of a) for (const second of b) sum += matrix![first * trackCount + second];
      return sum / (a.length * b.length);
    };
    while (clusters.length > ctx.options.familyCount) {
      let bestA = 0;
      let bestB = 1;
      let best = Infinity;
      for (let a = 0; a < clusters.length; a++) {
        for (let b = a + 1; b < clusters.length; b++) {
          const value = distance(clusters[a], clusters[b]);
          if (value < best) {
            best = value;
            bestA = a;
            bestB = b;
          }
        }
      }
      clusters[bestA] = clusters[bestA].concat(clusters[bestB]);
      clusters.splice(bestB, 1);
    }
    // Order families from the most northerly to the most southerly wintering area.
    const winter = (bird: number) => gulls.lngLat[(gulls.offsets[bird + 1] - 1) * 2 + 1];
    const mean = (members: number[]) =>
      members.reduce((sum, bird) => sum + winter(bird), 0) / members.length;
    clusters.sort((a, b) => mean(b) - mean(a));
    families = new Uint32Array(trackCount);
    clusters.forEach((members, family) => {
      for (const bird of members) families[bird] = family;
    });
    familyBuffer.write(families);
    const winterOf = (members: number[]) => mean(members);
    ctx.setChart('familyChart', {
      kind: 'bars',
      values: clusters.map(members => members.length),
      labels: clusters.map((members, family) => `F${family + 1} ${winterOf(members).toFixed(0)}N`),
      height: 110,
      yLabel: 'birds',
      formatY: value => value.toFixed(0),
      description:
        'Number of birds in each route family, labelled with the mean latitude of the last fix.'
    });
    ctx.setReadout(
      'families',
      clusters
        .map(
          (members, family) =>
            `${family + 1}: ${members.length} birds, last fix at ${mean(members).toFixed(0)} N`
        )
        .join('; ')
    );
    ctx.requestLayers();
  }

  function writeSelectedDistances(): void {
    const distances = new Float32Array(trackCount).fill(Number.NaN);
    if (matrix && selectedBird !== NO_BIRD) {
      for (let bird = 0; bird < trackCount; bird++) {
        distances[bird] = bird === selectedBird ? 0 : matrix[selectedBird * trackCount + bird];
      }
    }
    distanceBuffer.write(distances);
  }

  function describeSelected(): void {
    if (selectedBird === NO_BIRD) {
      ctx.setReadout('selected', 'click a track');
      return;
    }
    let text = `${describeBird(selectedBird)}, furthest south ${getFurthestSouth(selectedBird).toFixed(1)} N, family ${families[selectedBird] + 1} (${families.filter(value => value === families[selectedBird]).length} birds)`;
    if (metricsSnapshot) {
      text += `, ${formatCount(metricsSnapshot.lengths[selectedBird] / 1000)} km flown, ${metricsSnapshot.stopCounts[selectedBird]} stopovers`;
    }
    ctx.setReadout('selected', text);
  }

  function summarizeTemporal(): void {
    const snapshot = temporalSnapshot;
    if (!snapshot) return;
    const bucket = getBucketOf(ctx.options.snapshotDay);
    const width = ctx.options.bucketDays;
    const latitudes: number[] = [];
    let south = 0;
    let fastest = 0;
    for (let bird = 0; bird < trackCount; bird++) {
      const slot = bird * BUCKET_COUNT + bucket;
      if (snapshot.counts[slot] === 0) continue;
      latitudes.push(snapshot.latitudeLast[slot]);
      if (snapshot.latitudeLast[slot] < 40) south++;
      fastest = Math.max(fastest, snapshot.speedMax[slot]);
    }
    latitudes.sort((a, b) => a - b);
    const median = latitudes.length
      ? latitudes.length % 2
        ? latitudes[(latitudes.length - 1) / 2]
        : (latitudes[latitudes.length / 2 - 1] + latitudes[latitudes.length / 2]) / 2
      : NaN;
    ctx.setReadout(
      'snapshot',
      `${formatGullDate(bucket * width)} to ${formatGullDate(Math.min(138, (bucket + 1) * width - 1))}`
    );
    ctx.setReadout(
      'snapshotBirds',
      latitudes.length ? `${latitudes.length} of ${trackCount}` : 'none'
    );
    ctx.setReadout(
      'snapshotLatitude',
      Number.isFinite(median)
        ? `${median.toFixed(1)} N median, ${south} birds south of 40 N`
        : 'n/a'
    );
    ctx.setReadout(
      'snapshotSpeed',
      fastest > 0 ? `${(fastest * KILOMETERS_PER_SECOND_FACTOR).toFixed(0)} km/h` : 'n/a'
    );
    // Median and spread of the last latitude of every bucket: the whole migration as one curve.
    const bucketCount = Math.ceil(139 / width);
    const days: number[] = [];
    const medians: number[] = [];
    const lows: number[] = [];
    const highs: number[] = [];
    for (let bucketIndex = 0; bucketIndex < bucketCount; bucketIndex++) {
      const values: number[] = [];
      for (let bird = 0; bird < trackCount; bird++) {
        const slot = bird * BUCKET_COUNT + bucketIndex;
        if (snapshot.counts[slot] > 0) values.push(snapshot.latitudeLast[slot]);
      }
      if (values.length < 3) continue;
      values.sort((a, b) => a - b);
      days.push(bucketIndex * width + width / 2);
      medians.push(values[Math.floor(values.length / 2)]);
      lows.push(values[Math.floor(values.length * 0.1)]);
      highs.push(values[Math.min(values.length - 1, Math.floor(values.length * 0.9))]);
    }
    ctx.setChart('latitudeChart', {
      kind: 'line',
      series: [{label: 'median latitude', x: days, y: medians}],
      band: {x: days, low: lows, high: highs, label: '10th to 90th percentile'},
      xDomain: [0, 138],
      markers: [{x: ctx.options.snapshotDay, label: 'snapshot'}],
      xLabel: 'day since 15 July',
      yLabel: 'latitude (N)',
      height: 130,
      formatX: value => `${Math.round(value)}`,
      formatY: value => `${Math.round(value)}`,
      description:
        'Median and 10th to 90th percentile of the last latitude of each bird per time bucket.'
    });
    let occupied = 0;
    for (let slot = 0; slot < slotCount; slot++) if (snapshot.counts[slot] > 0) occupied++;
    ctx.setReadout(
      'reduction',
      `${formatCount(vertexCount)} fixes into ${formatCount(occupied)} (bird, ${width}-day) slots`
    );
  }

  // ---- Readers ---------------------------------------------------------------------------------
  const metricsReader = new SummaryReader(
    resources,
    'gull-metrics',
    [
      {buffer: trackLengths, size: trackCount * 4},
      {buffer: trackDurations, size: trackCount * 4},
      {buffer: maximumSpeeds, size: trackCount * 4},
      {buffer: trackStopCounts, size: trackCount * 4},
      {buffer: stopDurations, size: STOP_CAPACITY * 4},
      {buffer: stopCount, size: 4},
      {buffer: stopOverflow, size: 4},
      {buffer: stopTotal, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const floats = new Float32Array(bytes);
      const words = new Uint32Array(bytes);
      const n = trackCount;
      const tail = n * 4 + STOP_CAPACITY;
      metricsSnapshot = {
        lengths: floats.slice(0, n),
        durations: floats.slice(n, n * 2),
        maxima: floats.slice(n * 2, n * 3),
        stopCounts: words.slice(n * 3, n * 4),
        stopDurations: floats.slice(n * 4, tail),
        stopListed: Math.min(words[tail], STOP_CAPACITY),
        stopOverflow: words[tail + 1] !== 0,
        stopTotal: words[tail + 2]
      };
      summarizeMetrics();
    }
  );

  const similarityReader = new SummaryReader(
    resources,
    'gull-similarity',
    [
      {buffer: pairHausdorff, size: pairCount * 4},
      {buffer: pairFrechet, size: pairCount * 4},
      {buffer: pairStatus, size: pairCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      similaritySnapshot = {
        hausdorff: new Float32Array(bytes, 0, pairCount).slice(),
        frechet: new Float32Array(bytes, pairCount * 4, pairCount).slice(),
        status: new Uint32Array(bytes, pairCount * 8, pairCount).slice()
      };
      buildMatrix();
      ctx.setReadout('routes', `${trackCount} routes, ${formatCount(pairCount)} pairs scored`);
    }
  );

  const temporalReader = new SummaryReader(
    resources,
    'gull-temporal',
    [
      {buffer: latitudeColumns.counts, size: slotCount * 4},
      {buffer: latitudeColumns.last, size: slotCount * 4},
      {buffer: latitudeColumns.min, size: slotCount * 4},
      {buffer: latitudeColumns.max, size: slotCount * 4},
      {buffer: speedColumns.max, size: slotCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      temporalSnapshot = {
        counts: words.slice(0, slotCount),
        latitudeLast: floats.slice(slotCount, slotCount * 2),
        latitudeMin: floats.slice(slotCount * 2, slotCount * 3),
        latitudeMax: floats.slice(slotCount * 3, slotCount * 4),
        speedMax: floats.slice(slotCount * 4, slotCount * 5)
      };
      summarizeTemporal();
    }
  );

  // Default selection: the bird that went furthest south.
  let furthest = 0;
  for (let bird = 1; bird < trackCount; bird++) {
    if (getFurthestSouth(bird) < getFurthestSouth(furthest)) furthest = bird;
  }
  selectedBird = furthest;
  writeSelectedOutline();
  writeStopParameters();
  writeBucketParameters();
  writeWeeklyDisplay();
  ctx.setReadout('birds', `${trackCount} gulls, ${formatCount(vertexCount)} hourly fixes`);
  ctx.setReadout('period', '15 Jul to 30 Nov 2015');
  describeSelected();

  const clock = createPlaybackClock(
    ctx,
    {time: 'snapshotDay', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [0, 138], rate: 1, step: 1, notify: true}
  );

  // ---- Instance --------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [
      metricsCompiled,
      simplify.segment.importanceCompiled,
      simplify.segment.selectionCompiled,
      simplify['time-ratio'].importanceCompiled,
      simplify['time-ratio'].selectionCompiled,
      resampleGraphs[currentSpacing],
      similarityCompiled,
      compiledTemporal,
      weeklyCompiled
    ],

    setOption(id, _value, state) {
      switch (id) {
        case 'stopSpeed':
        case 'stopHours':
          writeStopParameters();
          break;
        case 'toleranceLog':
          tolerancesDirty = true;
          markChanged();
          describeSimplification();
          break;
        case 'simplifyMetric':
          describeSimplification();
          ctx.requestLayers();
          break;
        case 'routeSpacing':
          currentSpacing = state.routeSpacing;
          similarityDirty = true;
          markChanged();
          break;
        case 'similarityMetric':
        case 'familyCount':
          buildMatrix();
          ctx.requestLayers();
          break;
        case 'similarityRangeKm':
          writeSelectedDistances();
          ctx.requestLayers();
          break;
        case 'bucketDays':
          writeBucketParameters();
          ctx.requestLayers();
          break;
        case 'play':
        case 'playSpeed':
        case 'loop':
          break;
        case 'dayRange':
        case 'snapshotDay':
          writeWeeklyDisplay();
          summarizeTemporal();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      clock.advance(frame);
      if (metricsDirty) {
        metricsCompiled.encode(commandEncoder, {parameters: undefined});
        metricsDirty = false;
        metricsReader.request(commandEncoder);
      } else {
        metricsReader.flush(commandEncoder);
      }
      for (const variant of Object.values(simplify)) {
        if (!variant.importanceEncoded) {
          variant.importanceCompiled.encode(commandEncoder, {parameters: undefined});
          variant.importanceEncoded = true;
          tolerancesDirty = true;
        }
      }
      if (tolerancesDirty) {
        toleranceParameters.write(
          getGPULineSimplificationParameterValues({tolerance: Math.fround(getToleranceMeters())})
        );
        for (const variant of Object.values(simplify)) {
          variant.selectionCompiled.encode(commandEncoder, {parameters: undefined});
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: variant.keptCount,
            destinationBuffer: variant.draw.buffer,
            destinationOffset: 4,
            size: 4
          });
          variant.reader.markStale();
        }
        tolerancesDirty = false;
      }
      for (const variant of Object.values(simplify)) variant.reader.flush(commandEncoder);

      if (similarityDirty) {
        const key = currentSpacing;
        resampleGraphs[key].encode(commandEncoder, {parameters: undefined});
        similarityCompiled.encode(commandEncoder, {parameters: undefined});
        similarityDirty = false;
        similarityReader.markStale();
      }
      similarityReader.flush(commandEncoder);

      if (temporalDirty) {
        compiledTemporal.encode(commandEncoder, {parameters: undefined});
        temporalDirty = false;
        weeklyDirty = true;
      }
      if (weeklyDirty) {
        weeklyCompiled.encode(commandEncoder, {parameters: undefined});
        weeklyDirty = false;
        settleStale = true;
      }
      if (settleStale && performance.now() - lastChange > SETTLE_MILLISECONDS) {
        temporalReader.markStale();
        settleStale = false;
      }
      temporalReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const colorBy = options.colorBy;
      let colorProps: Record<string, unknown>;
      switch (colorBy) {
        case 'date':
          colorProps = {
            values: timestampsBuffer,
            valueFormat: 'float32',
            valueIndices: segmentEndsBuffer,
            valueScale: 1 / SECONDS_PER_DAY,
            valueRange: [0, 138],
            colormap: options.ramp
          };
          break;
        case 'speed':
          colorProps = {
            values: stepSpeeds,
            valueFormat: 'float32',
            valueIndices: segmentEndsBuffer,
            valueScale: KILOMETERS_PER_SECOND_FACTOR,
            valueRange: [0, 60],
            colormap: options.ramp
          };
          break;
        case 'family':
          colorProps = {
            values: familyBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTracksBuffer,
            colormap: 'category',
            palette: GULL_FAMILY_COLORS
          };
          break;
        case 'similarity':
          colorProps = {
            values: distanceBuffer,
            valueFormat: 'float32',
            valueIndices: segmentTracksBuffer,
            valueScale: 0.001,
            valueRange: [0, options.similarityRangeKm],
            colormap: options.ramp,
            noDataColor: [140, 146, 160, 80]
          };
          break;
        case 'sex':
          colorProps = {
            values: sexBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTracksBuffer,
            colormap: 'category',
            palette: GULL_SEX_COLORS
          };
          break;
        default:
          colorProps = {color: dark ? [200, 210, 230, 255] : [60, 70, 95, 255]};
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'gull-tracks',
          ...drawProps,
          segments: segmentsBuffer,
          instanceCount: segmentCount,
          widthPixels: 1.6,
          opacity: options.showSimplified ? options.trackOpacity * 0.35 : options.trackOpacity,
          ...colorProps
        })
      );
      if (options.showSimplified) {
        const variant = simplify[options.simplifyMetric];
        layers.push(
          new KeptSegmentLayer({
            id: `gull-kept-${options.simplifyMetric}`,
            ...drawProps,
            positions: lngLatBuffer,
            keptIds: variant.keptIds,
            vertexLines: vertexTracksBuffer,
            keptCount: variant.keptCount,
            drawCommands: variant.draw,
            widthPixels: 2.4,
            color: options.simplifyMetric === 'segment' ? [255, 150, 40, 255] : [255, 70, 150, 255]
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'gull-selected',
          ...drawProps,
          segments: selectedSegments,
          instanceCount: gulls.longestTrack,
          widthPixels: 3.2,
          color: dark ? [255, 255, 255, 235] : [20, 24, 32, 235]
        })
      );
      if (options.showWeekly) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'gull-weekly-lines',
            ...drawProps,
            segments: weeklySegments,
            instanceCount: trackCount * (BUCKET_COUNT - 1),
            values: segmentBucketBuffer,
            valueFormat: 'uint32',
            colormap: options.ramp,
            valueRange: [0, displayMaxBucket],
            widthPixels: 2,
            opacity: 0.9
          }),
          new SpatialAnalysisPointLayer({
            id: 'gull-weekly-dots',
            ...drawProps,
            positions: weeklyPositions,
            instanceCount: slotCount,
            values: slotBucketBuffer,
            valueFormat: 'uint32',
            colormap: options.ramp,
            valueRange: [0, displayMaxBucket],
            radiusPixels: 3.2,
            opacity: 0.95
          }),
          new SpatialAnalysisPointLayer({
            id: 'gull-focus-dots',
            ...drawProps,
            positions: focusPositions,
            instanceCount: slotCount,
            radiusPixels: 7,
            color: dark ? [255, 255, 255, 255] : [20, 24, 32, 255],
            opacity: 0.95
          })
        );
      }
      if (options.showStops) {
        layers.push(
          new StopMarkerLayer({
            id: 'gull-stops',
            ...drawProps,
            centroids: stopLngLat,
            durations: stopDurations,
            drawCommands: stopDraw,
            baseRadiusPixels: 4,
            radiusPerSqrtSecond: 0.012,
            maximumRadiusPixels: 20,
            durationForFullColor: 20 * 86400,
            opacity: 0.85
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const bird = pickBird(event.coordinate);
      if (bird < 0) return null;
      const snapshot = metricsSnapshot;
      let text = `${describeBird(bird)}: furthest south ${getFurthestSouth(bird).toFixed(1)} N, family ${families[bird] + 1}`;
      if (snapshot) {
        text += `, ${formatCount(snapshot.lengths[bird] / 1000)} km flown, ${snapshot.stopCounts[bird]} stopovers`;
      }
      return text;
    },

    onClick(event) {
      const bird = pickBird(event.coordinate);
      if (bird < 0) return false;
      selectedBird = bird;
      writeSelectedOutline();
      writeSelectedDistances();
      describeSelected();
      ctx.requestLayers();
      return true;
    },

    destroy() {
      destroyed = true;
      metricsReader.stop();
      similarityReader.stop();
      temporalReader.stop();
      for (const variant of Object.values(simplify)) variant.reader.stop();
      resources.destroy();
    }
  };

  /** Nearest track within 250 km of a longitude/latitude, or -1. */
  function pickBird(coordinate: readonly [number, number] | null): number {
    if (!coordinate) return -1;
    const [x, y] = gulls.project(coordinate[0], coordinate[1]);
    const {track, distance} = findNearestTrack(gulls, x, y);
    return track >= 0 && distance < 250000 ? track : -1;
  }
}
