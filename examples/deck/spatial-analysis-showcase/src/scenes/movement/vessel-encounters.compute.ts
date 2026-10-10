// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  addClockEncounters,
  getGPUTrajectoryClockParameterValues,
  GPUTrackSimilarity,
  GPUTrajectoryMetrics,
  GPUTrajectoryResample
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {createPlaybackClock} from '../../engine/playback';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {binValues, histogramChart} from './f-chart-helpers';
import {
  findNearestTrack,
  formatDuration,
  formatUtcClock,
  KNOTS_PER_METER_SECOND,
  loadVesselTracks,
  METERS_PER_KNOT_SECOND,
  VESSEL_CATEGORIES,
  VESSEL_CATEGORY_COLORS,
  VESSEL_CATEGORY_LABELS
} from './b12-tracks';
import {
  getVesselEncounterSearchAnnotations,
  getVesselRouteLeashWitness
} from './vessel-encounters-overlays';

/** Option state of the vessel encounters scene. */
export type VesselEncountersOptions = {
  distance: number;
  minSpeedKnots: number;
  clockStepSeconds: number;
  clockStartHour: number;
  timeOfDay: number;
  connectorTime: 'bucket' | 'first';
  connectorFilter: 'all' | 'alike' | 'different';
  similarityMetric: 'hausdorff' | 'frechet';
  alikeMeters: number;
  trackColor: 'category' | 'similarity' | 'off';
  colorRangeMeters: number;
  showVessels: boolean;
  play: boolean;
  playSpeed: number;
  loop: boolean;
};

/** Time buckets of the shared clock (compile-time: the dense table is tracks x buckets). */
const BUCKET_COUNT = 720;
/** Compile-time lattice cell width in meters; the distance option may not exceed it. */
const CELL_SIZE = 400;
const HIT_CAPACITY = 1 << 21;
const PAIR_CAPACITY = 16384;
/** Samples of the arc-length routes fed to the similarity measures. */
const ROUTE_SAMPLES = 64;
const NO_TRACK = 0xffffffff;
const SETTLE_MILLISECONDS = 250;

type PairSnapshot = {
  count: number;
  overflow: boolean;
  candidateOverflow: boolean;
  ids: Uint32Array;
  partners: Uint32Array;
  firstBuckets: Uint32Array;
  minimumDistances: Float32Array;
  bucketCounts: Uint32Array;
  hausdorff: Float32Array;
  frechet: Float32Array;
};

/**
 * Vessel encounters: one graph puts every AIS track on a shared clock (`addClockEncounters`), lists
 * the pairs that come within a distance of each other (`GPUTrajectoryEncounters`) and scores how
 * alike each pair's whole routes are (`GPUTrackSimilarity` over arc-length resampled routes). A
 * second similarity instance scores the clicked vessel against every other route. Moored vessels
 * are masked out on the GPU by a kernel that reads `GPUTrajectoryMetrics` average speeds.
 */
export async function createVesselEncounters(
  ctx: SceneContext<VesselEncountersOptions>
): Promise<SceneInstance<VesselEncountersOptions>> {
  const vessels = loadVesselTracks(ctx.datasets.get('ais-vessels'));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = vessels;
  const resources = new SpatialAnalysisResources(device, 'encounters');
  const coordinateOrigin: [number, number, number] = [vessels.origin[0], vessels.origin[1], 0];
  const sampleCount = trackCount * BUCKET_COUNT;

  // Lattice bounds: the central 99.8 percent of the fixes, padded by one cell.
  const bounds = getBounds(vessels.positions);

  // ---- Static buffers --------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', vessels.positions);
  const timestampsBuffer = resources.createBuffer('timestamps', vessels.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', vessels.offsets);
  const categoryBuffer = resources.createBuffer('category', vessels.category);
  const segmentsBuffer = resources.createBuffer('segments', vessels.segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', vessels.segmentTracks);
  const routeOffsets = new Uint32Array(trackCount + 1);
  for (let track = 0; track <= trackCount; track++) routeOffsets[track] = track * ROUTE_SAMPLES;
  const routeOffsetsBuffer = resources.createBuffer('route-offsets', routeOffsets);
  const routesBuffer = resources.createBuffer('routes', trackCount * ROUTE_SAMPLES * 8);
  const selectedSegments = resources.createBuffer('selected-segments', vessels.longestTrack * 16);

  // ---- Graph 1: routes and per-track metrics (encoded once) ------------------------------------
  const averageSpeeds = resources.createBuffer('average-speeds', trackCount * 4);
  const trackLengths = resources.createBuffer('track-lengths', trackCount * 4);
  const prepareGraph = new GPUCommandGraph<void>(device, {id: 'encounters-prepare'});
  const prepareOffsets = importGraphBuffer(
    prepareGraph,
    'offsets',
    offsetsBuffer,
    'uint32',
    trackCount + 1
  );
  const preparePositions = importGraphBuffer(
    prepareGraph,
    'positions',
    positionsBuffer,
    'float32x2',
    vertexCount
  );
  const prepareTimestamps = importGraphBuffer(
    prepareGraph,
    'timestamps',
    timestampsBuffer,
    'float32',
    vertexCount
  );
  prepareGraph.add(
    new GPUTrajectoryMetrics({
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
      id: 'metrics',
      positions: preparePositions,
      timestamps: prepareTimestamps,
      trackOffsets: prepareOffsets,
      averageSpeeds: importGraphBuffer(
        prepareGraph,
        'average-speeds',
        averageSpeeds,
        'float32',
        trackCount
      ),
      trackLengths: importGraphBuffer(
        prepareGraph,
        'track-lengths',
        trackLengths,
        'float32',
        trackCount
      )
    })
  );
  prepareGraph.add(
    new GPUTrajectoryResample({
      id: 'routes',
      positions: preparePositions,
      timestamps: prepareTimestamps,
      trackOffsets: prepareOffsets,
      sampleCount: ROUTE_SAMPLES,
      spacing: 'arc-length',
      samples: importGraphBuffer(
        prepareGraph,
        'routes',
        routesBuffer,
        'float32x2',
        trackCount * ROUTE_SAMPLES
      )
    })
  );
  const prepareCompiled = resources.track(prepareGraph.compile());

  // ---- Graph 2: encounters + pair similarity (re-encoded on a clock/distance change) -----------
  const samplesBuffer = resources.createBuffer('samples', sampleCount * 8);
  const maskedPositions = resources.createBuffer('masked-positions', vertexCount * 8);
  const vertexTracks = new Uint32Array(vertexCount);
  for (let track = 0; track < trackCount; track++) {
    vertexTracks.fill(track, vessels.offsets[track], vessels.offsets[track + 1]);
  }
  const vertexTracksBuffer = resources.createBuffer('vertex-tracks', vertexTracks);
  const pairIds = resources.createBuffer('pair-ids', PAIR_CAPACITY * 4);
  const pairPartners = resources.createBuffer('pair-partners', PAIR_CAPACITY * 4);
  const pairFirstBuckets = resources.createBuffer('pair-first-buckets', PAIR_CAPACITY * 4);
  const pairMinimumDistances = resources.createBuffer('pair-minimum-distances', PAIR_CAPACITY * 4);
  const pairBucketCounts = resources.createBuffer('pair-bucket-counts', PAIR_CAPACITY * 4);
  const pairCount = resources.createBuffer('pair-count', 4);
  const pairOverflow = resources.createBuffer('pair-overflow', 4);
  const pairCandidateOverflow = resources.createBuffer('pair-candidate-overflow', 4);
  const pairHausdorff = resources.createBuffer('pair-hausdorff', PAIR_CAPACITY * 4);
  const pairFrechet = resources.createBuffer('pair-frechet', PAIR_CAPACITY * 4);
  const pairStatus = resources.createBuffer('pair-status', PAIR_CAPACITY * 4);
  const distanceParameter = resources.createParameterBuffer(
    'distance',
    'float32',
    1,
    Float32Array.of(ctx.options.distance)
  );
  const clockParameter = resources.createParameterBuffer(
    'clock',
    'float32',
    4,
    getGPUTrajectoryClockParameterValues({start: 0, step: 120})
  );
  const validityParameter = resources.createParameterBuffer(
    'validity',
    'float32',
    1,
    Float32Array.of(ctx.options.minSpeedKnots * METERS_PER_KNOT_SECOND)
  );

  const encounterGraph = new GPUCommandGraph<void>(device, {id: 'encounters-pairs'});
  const encounterOffsets = importGraphBuffer(
    encounterGraph,
    'offsets',
    offsetsBuffer,
    'uint32',
    trackCount + 1
  );
  const maskedPositionsView = importGraphBuffer(
    encounterGraph,
    'masked-positions',
    maskedPositions,
    'float32x2',
    vertexCount
  );
  // Moored vessels would swamp the pair list. Their fixes are replaced by NaN (absent) so the
  // clock resample, and with it every encounter, ignores them. The mask follows the speed slider.
  addKernelPass(encounterGraph, {
    id: 'mask-moored-fixes',
    invocationCount: vertexCount,
    bindings: [
      {
        name: 'positions',
        view: importGraphBuffer(
          encounterGraph,
          'positions',
          positionsBuffer,
          'float32x2',
          vertexCount
        ),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'vertexTracks',
        view: importGraphBuffer(
          encounterGraph,
          'vertex-tracks',
          vertexTracksBuffer,
          'uint32',
          vertexCount
        ),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'averageSpeeds',
        view: importGraphBuffer(
          encounterGraph,
          'average-speeds',
          averageSpeeds,
          'float32',
          trackCount
        ),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'validity',
        view: validityParameter.importToGraph(encounterGraph),
        type: 'f32',
        access: 'read'
      },
      {name: 'masked', view: maskedPositionsView, type: 'f32', access: 'read_write'}
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let moves = averageSpeeds[averageSpeedsOffset + vertexTracks[vertexTracksOffset + index]] >= validity[validityOffset];
  masked[maskedOffset + index * 2u] = select(nan, positions[positionsOffset + index * 2u], moves);
  masked[maskedOffset + index * 2u + 1u] = select(nan, positions[positionsOffset + index * 2u + 1u], moves);`
  });
  const pairIdsView = importGraphBuffer(
    encounterGraph,
    'pair-ids',
    pairIds,
    'uint32',
    PAIR_CAPACITY
  );
  const pairPartnersView = importGraphBuffer(
    encounterGraph,
    'pair-partners',
    pairPartners,
    'uint32',
    PAIR_CAPACITY
  );
  const pairCountView = importGraphBuffer(encounterGraph, 'pair-count', pairCount, 'uint32', 1);
  addClockEncounters(encounterGraph, {
    id: 'encounters',
    positions: maskedPositionsView,
    timestamps: importGraphBuffer(
      encounterGraph,
      'timestamps',
      timestampsBuffer,
      'float32',
      vertexCount
    ),
    trackOffsets: encounterOffsets,
    clock: clockParameter.importToGraph(encounterGraph),
    bucketCount: BUCKET_COUNT,
    samples: importGraphBuffer(encounterGraph, 'samples', samplesBuffer, 'float32x2', sampleCount),
    distance: distanceParameter.importToGraph(encounterGraph),
    cellSize: CELL_SIZE,
    bounds,
    hitCapacity: HIT_CAPACITY,
    pairs: {
      candidateOverflow: importGraphBuffer(
        encounterGraph,
        'pair-candidate-overflow',
        pairCandidateOverflow,
        'uint32',
        1
      ),
      output: {
        ids: pairIdsView,
        count: pairCountView,
        overflow: importGraphBuffer(encounterGraph, 'pair-overflow', pairOverflow, 'uint32', 1)
      },
      partners: pairPartnersView,
      firstBuckets: importGraphBuffer(
        encounterGraph,
        'pair-first-buckets',
        pairFirstBuckets,
        'uint32',
        PAIR_CAPACITY
      ),
      minimumDistances: importGraphBuffer(
        encounterGraph,
        'pair-minimum-distances',
        pairMinimumDistances,
        'float32',
        PAIR_CAPACITY
      ),
      bucketCounts: importGraphBuffer(
        encounterGraph,
        'pair-bucket-counts',
        pairBucketCounts,
        'uint32',
        PAIR_CAPACITY
      )
    }
  });
  encounterGraph.add(
    new GPUTrackSimilarity({
      id: 'pair-similarity',
      positionsA: importGraphBuffer(
        encounterGraph,
        'routes',
        routesBuffer,
        'float32x2',
        trackCount * ROUTE_SAMPLES
      ),
      offsetsA: importGraphBuffer(
        encounterGraph,
        'route-offsets',
        routeOffsetsBuffer,
        'uint32',
        trackCount + 1
      ),
      pairA: pairIdsView,
      pairB: pairPartnersView,
      activePairCount: pairCountView,
      hausdorff: importGraphBuffer(
        encounterGraph,
        'pair-hausdorff',
        pairHausdorff,
        'float32',
        PAIR_CAPACITY
      ),
      frechet: importGraphBuffer(
        encounterGraph,
        'pair-frechet',
        pairFrechet,
        'float32',
        PAIR_CAPACITY
      ),
      status: importGraphBuffer(encounterGraph, 'pair-status', pairStatus, 'uint32', PAIR_CAPACITY),
      maxFrechetVertices: ROUTE_SAMPLES
    })
  );
  const encounterCompiled = resources.track(encounterGraph.compile());

  // ---- Graph 3: connectors and vessel positions at the chosen time (cheap kernels) -------------
  const connectors = resources.createBuffer('connectors', PAIR_CAPACITY * 16);
  const vehicles = resources.createBuffer('vehicles', trackCount * 8);
  const viewParameters = resources.createParameterBuffer('view', 'float32', 8);
  const viewGraph = new GPUCommandGraph<void>(device, {id: 'encounters-view'});
  const samplesFloats = importGraphBuffer(
    viewGraph,
    'samples',
    samplesBuffer,
    'float32',
    sampleCount * 2
  );
  const viewParameterView = viewParameters.importToGraph(viewGraph);
  addKernelPass(viewGraph, {
    id: 'encounter-connectors',
    invocationCount: PAIR_CAPACITY,
    declarations: `const BUCKETS: u32 = ${BUCKET_COUNT}u;`,
    bindings: [
      {
        name: 'pairIds',
        view: importGraphBuffer(viewGraph, 'pair-ids', pairIds, 'uint32', PAIR_CAPACITY),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'pairPartners',
        view: importGraphBuffer(viewGraph, 'pair-partners', pairPartners, 'uint32', PAIR_CAPACITY),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'firstBuckets',
        view: importGraphBuffer(
          viewGraph,
          'pair-first-buckets',
          pairFirstBuckets,
          'uint32',
          PAIR_CAPACITY
        ),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'hausdorff',
        view: importGraphBuffer(
          viewGraph,
          'pair-hausdorff',
          pairHausdorff,
          'float32',
          PAIR_CAPACITY
        ),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'frechet',
        view: importGraphBuffer(viewGraph, 'pair-frechet', pairFrechet, 'float32', PAIR_CAPACITY),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'pairCount',
        view: importGraphBuffer(viewGraph, 'pair-count', pairCount, 'uint32', 1),
        type: 'u32',
        access: 'read'
      },
      {name: 'samples', view: samplesFloats, type: 'f32', access: 'read'},
      {name: 'view', view: viewParameterView, type: 'f32', access: 'read'},
      {
        name: 'segments',
        view: importGraphBuffer(viewGraph, 'connectors', connectors, 'float32', PAIR_CAPACITY * 4),
        type: 'f32',
        access: 'read_write'
      }
    ],
    // view = [useBucket, bucket, distance, filter (0 all, 1 alike, 2 different), threshold, frechet]
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var segment = vec4<f32>(nan);
  if (index < pairCount[pairCountOffset]) {
    let useBucket = view[viewOffset] > 0.5;
    let bucket = select(firstBuckets[firstBucketsOffset + index], u32(view[viewOffset + 1u]), useBucket);
    let a = (pairIds[pairIdsOffset + index] * BUCKETS + bucket) * 2u;
    let b = (pairPartners[pairPartnersOffset + index] * BUCKETS + bucket) * 2u;
    let pa = vec2<f32>(samples[samplesOffset + a], samples[samplesOffset + a + 1u]);
    let pb = vec2<f32>(samples[samplesOffset + b], samples[samplesOffset + b + 1u]);
    let similarity = select(hausdorff[hausdorffOffset + index], frechet[frechetOffset + index], view[viewOffset + 5u] > 0.5);
    let connectorMode = view[viewOffset + 3u];
    var passes = true;
    if (connectorMode > 0.5 && connectorMode < 1.5) { passes = similarity <= view[viewOffset + 4u]; }
    if (connectorMode > 1.5) { passes = !(similarity <= view[viewOffset + 4u]); }
    if (passes && (!useBucket || distance(pa, pb) <= view[viewOffset + 2u] + 1e-3)) {
      segment = vec4<f32>(pa, pb);
    }
  }
  segments[segmentsOffset + index * 4u] = segment.x;
  segments[segmentsOffset + index * 4u + 1u] = segment.y;
  segments[segmentsOffset + index * 4u + 2u] = segment.z;
  segments[segmentsOffset + index * 4u + 3u] = segment.w;`
  });
  addKernelPass(viewGraph, {
    id: 'encounter-vehicles',
    invocationCount: trackCount,
    declarations: `const BUCKETS: u32 = ${BUCKET_COUNT}u;`,
    bindings: [
      {name: 'samples', view: samplesFloats, type: 'f32', access: 'read'},
      {name: 'view', view: viewParameterView, type: 'f32', access: 'read'},
      {
        name: 'vehicles',
        view: importGraphBuffer(viewGraph, 'vehicles', vehicles, 'float32', trackCount * 2),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let bucket = u32(view[viewOffset + 1u]);
  let source = (index * BUCKETS + bucket) * 2u;
  vehicles[vehiclesOffset + index * 2u] = samples[samplesOffset + source];
  vehicles[vehiclesOffset + index * 2u + 1u] = samples[samplesOffset + source + 1u];`
  });
  const viewCompiled = resources.track(viewGraph.compile());

  // ---- Graph 4: the selected vessel against every other route ----------------------------------
  const selectedPairA = resources.createBuffer('selected-pair-a', trackCount * 4);
  const selectedPairB = resources.createBuffer('selected-pair-b', trackCount * 4);
  const selectedHausdorff = resources.createBuffer('selected-hausdorff', trackCount * 4);
  const selectedFrechet = resources.createBuffer('selected-frechet', trackCount * 4);
  const selectedStatus = resources.createBuffer('selected-status', trackCount * 4);
  const selectionParameter = resources.createParameterBuffer(
    'selection',
    'uint32',
    1,
    Uint32Array.of(0)
  );
  const selectionGraph = new GPUCommandGraph<void>(device, {id: 'encounters-selection'});
  const selectedAView = importGraphBuffer(
    selectionGraph,
    'selected-pair-a',
    selectedPairA,
    'uint32',
    trackCount
  );
  const selectedBView = importGraphBuffer(
    selectionGraph,
    'selected-pair-b',
    selectedPairB,
    'uint32',
    trackCount
  );
  addKernelPass(selectionGraph, {
    id: 'selection-pairs',
    invocationCount: trackCount,
    bindings: [
      {
        name: 'selection',
        view: selectionParameter.importToGraph(selectionGraph),
        type: 'u32',
        access: 'read'
      },
      {name: 'pairA', view: selectedAView, type: 'u32', access: 'read_write'},
      {name: 'pairB', view: selectedBView, type: 'u32', access: 'read_write'}
    ],
    body: `pairA[pairAOffset + index] = selection[selectionOffset];
  pairB[pairBOffset + index] = index;`
  });
  selectionGraph.add(
    new GPUTrackSimilarity({
      id: 'selected-similarity',
      positionsA: importGraphBuffer(
        selectionGraph,
        'routes',
        routesBuffer,
        'float32x2',
        trackCount * ROUTE_SAMPLES
      ),
      offsetsA: importGraphBuffer(
        selectionGraph,
        'route-offsets',
        routeOffsetsBuffer,
        'uint32',
        trackCount + 1
      ),
      pairA: selectedAView,
      pairB: selectedBView,
      hausdorff: importGraphBuffer(
        selectionGraph,
        'selected-hausdorff',
        selectedHausdorff,
        'float32',
        trackCount
      ),
      frechet: importGraphBuffer(
        selectionGraph,
        'selected-frechet',
        selectedFrechet,
        'float32',
        trackCount
      ),
      status: importGraphBuffer(
        selectionGraph,
        'selected-status',
        selectedStatus,
        'uint32',
        trackCount
      ),
      maxFrechetVertices: ROUTE_SAMPLES
    })
  );
  const selectionCompiled = resources.track(selectionGraph.compile());

  // ---- State -----------------------------------------------------------------------------------
  let selectedTrack = NO_TRACK;
  let prepared = false;
  let encountersDirty = true;
  let viewDirty = true;
  let selectionDirty = true;
  let pairsStale = true;
  let selectionStale = true;
  let lastChange = performance.now();
  let destroyed = false;
  let pairSnapshot: PairSnapshot | null = null;
  let selectedSnapshot: {
    hausdorff: Float32Array;
    frechet: Float32Array;
    status: Uint32Array;
  } | null = null;
  let metricsSnapshot: {average: Float32Array; length: Float32Array} | null = null;

  const getClockStart = () => ctx.options.clockStartHour * 3600;
  const getBucket = () =>
    Math.min(
      BUCKET_COUNT - 1,
      Math.max(
        0,
        Math.round((ctx.options.timeOfDay - getClockStart()) / ctx.options.clockStepSeconds)
      )
    );

  const clock = createPlaybackClock(
    ctx,
    {time: 'timeOfDay', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [0, 86280], rate: 1, step: 120, notify: true}
  );
  /** Sweeps the shared clock's own window, not the whole day. */
  const syncClockRange = () => {
    const start = ctx.options.clockStartHour * 3600;
    const end = Math.min(86280, start + (BUCKET_COUNT - 1) * ctx.options.clockStepSeconds);
    clock.setRange(start, end);
  };
  syncClockRange();
  let meetingsPerHour = new Float64Array(24);

  const markChanged = () => {
    lastChange = performance.now();
    pairsStale = true;
  };

  function writeClock(): void {
    clockParameter.write(
      getGPUTrajectoryClockParameterValues({
        start: getClockStart(),
        step: ctx.options.clockStepSeconds
      })
    );
    const window = BUCKET_COUNT * ctx.options.clockStepSeconds;
    ctx.setReadout(
      'clock',
      `${BUCKET_COUNT} steps of ${ctx.options.clockStepSeconds} s: ${formatUtcClock(getClockStart())} to ${formatUtcClock(getClockStart() + window)} UTC`
    );
  }

  function writeView(): void {
    const {connectorTime, connectorFilter, alikeMeters, similarityMetric, distance} = ctx.options;
    viewParameters.write(
      Float32Array.of(
        connectorTime === 'bucket' ? 1 : 0,
        getBucket(),
        distance,
        connectorFilter === 'all' ? 0 : connectorFilter === 'alike' ? 1 : 2,
        alikeMeters,
        similarityMetric === 'frechet' ? 1 : 0,
        0,
        0
      )
    );
    viewDirty = true;
  }

  function selectTrack(track: number): void {
    selectedTrack = track;
    selectionParameter.write(Uint32Array.of(track));
    const first = vessels.offsets[track];
    const last = vessels.offsets[track + 1] - 1;
    const segments = new Float32Array(vessels.longestTrack * 4).fill(Number.NaN);
    for (let vertex = first; vertex < last; vertex++) {
      segments.set(vessels.positions.subarray(vertex * 2, vertex * 2 + 4), (vertex - first) * 4);
    }
    selectedSegments.write(segments);
    selectionDirty = true;
    selectionStale = true;
    describeSelected();
    ctx.requestLayers();
  }

  function describeTrack(track: number): string {
    const category = VESSEL_CATEGORIES[vessels.category[track]];
    const length = vessels.length[track];
    return `${VESSEL_CATEGORY_LABELS[category].split(' (')[0]} (MMSI ${vessels.mmsi[track]}${length > 0 ? `, ${length.toFixed(0)} m` : ''})`;
  }

  function describeSelected(): void {
    if (selectedTrack === NO_TRACK) {
      ctx.setReadout('selected', 'click a vessel');
      return;
    }
    let text = describeTrack(selectedTrack);
    if (metricsSnapshot) {
      text += `: ${(metricsSnapshot.length[selectedTrack] / 1852).toFixed(1)} nm, mean ${(metricsSnapshot.average[selectedTrack] * KNOTS_PER_METER_SECOND).toFixed(1)} kn`;
    }
    ctx.setReadout('selected', text);
  }

  // Default selection: the passenger track that runs closest to both Staten Island Ferry terminals.
  selectTrack(findFerryTrack());
  writeClock();
  writeView();
  ctx.setReadout(
    'tracks',
    `${formatCount(trackCount)} tracks / ${formatCount(vessels.vesselCount)} vessels`
  );

  // ---- Readbacks -------------------------------------------------------------------------------
  const metricsReader = new SummaryReader(
    resources,
    'encounters-metrics',
    [
      {buffer: averageSpeeds, size: trackCount * 4},
      {buffer: trackLengths, size: trackCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const floats = new Float32Array(bytes);
      metricsSnapshot = {
        average: floats.slice(0, trackCount),
        length: floats.slice(trackCount, trackCount * 2)
      };
      let moving = 0;
      for (let track = 0; track < trackCount; track++) {
        if (metricsSnapshot.average[track] >= ctx.options.minSpeedKnots * METERS_PER_KNOT_SECOND)
          moving++;
      }
      ctx.setReadout('tracksUsed', `${formatCount(moving)} of ${formatCount(trackCount)}`);
      describeSelected();
    }
  );

  const pairReader = new SummaryReader(
    resources,
    'encounters-pairs',
    [
      {buffer: pairCount, size: 4},
      {buffer: pairOverflow, size: 4},
      {buffer: pairCandidateOverflow, size: 4},
      {buffer: pairIds, size: PAIR_CAPACITY * 4},
      {buffer: pairPartners, size: PAIR_CAPACITY * 4},
      {buffer: pairFirstBuckets, size: PAIR_CAPACITY * 4},
      {buffer: pairMinimumDistances, size: PAIR_CAPACITY * 4},
      {buffer: pairBucketCounts, size: PAIR_CAPACITY * 4},
      {buffer: pairHausdorff, size: PAIR_CAPACITY * 4},
      {buffer: pairFrechet, size: PAIR_CAPACITY * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const column = (index: number) => 3 + index * PAIR_CAPACITY;
      const count = Math.min(words[0], PAIR_CAPACITY);
      pairSnapshot = {
        count,
        overflow: words[1] !== 0,
        candidateOverflow: words[2] !== 0,
        ids: words.slice(column(0), column(1)),
        partners: words.slice(column(1), column(2)),
        firstBuckets: words.slice(column(2), column(3)),
        minimumDistances: floats.slice(column(3), column(4)),
        bucketCounts: words.slice(column(4), column(5)),
        hausdorff: floats.slice(column(5), column(6)),
        frechet: floats.slice(column(6), column(7))
      };
      summarizePairs();
    }
  );

  const selectionReader = new SummaryReader(
    resources,
    'encounters-selection',
    [
      {buffer: selectedStatus, size: trackCount * 4},
      {buffer: selectedHausdorff, size: trackCount * 4},
      {buffer: selectedFrechet, size: trackCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      selectedSnapshot = {
        status: words.slice(0, trackCount),
        hausdorff: floats.slice(trackCount, trackCount * 2),
        frechet: floats.slice(trackCount * 2, trackCount * 3)
      };
      summarizeSelection();
    }
  );

  function getSimilarity(snapshot: PairSnapshot, pair: number): number {
    return ctx.options.similarityMetric === 'frechet'
      ? snapshot.frechet[pair]
      : snapshot.hausdorff[pair];
  }

  function summarizePairs(): void {
    const snapshot = pairSnapshot;
    if (!snapshot) return;
    const {clockStepSeconds, alikeMeters} = ctx.options;
    ctx.setReadout(
      'pairs',
      `${formatCount(snapshot.count)}` +
        `${snapshot.overflow ? ' (output capacity overflow)' : ''}` +
        `${snapshot.candidateOverflow ? ' (candidate scratch overflow)' : ''}`
    );
    let alike = 0;
    let scored = 0;
    let closest = Infinity;
    let longest = 0;
    const typeCounts = new Map<string, number>();
    const hours = new Float64Array(24);
    for (let pair = 0; pair < snapshot.count; pair++) {
      const value = getSimilarity(snapshot, pair);
      if (Number.isFinite(value)) {
        scored++;
        if (value <= alikeMeters) alike++;
      }
      closest = Math.min(closest, snapshot.minimumDistances[pair]);
      longest = Math.max(longest, snapshot.bucketCounts[pair] * clockStepSeconds);
      const a = VESSEL_CATEGORIES[vessels.category[snapshot.ids[pair]]];
      const b = VESSEL_CATEGORIES[vessels.category[snapshot.partners[pair]]];
      const key = a <= b ? `${a} + ${b}` : `${b} + ${a}`;
      typeCounts.set(key, (typeCounts.get(key) ?? 0) + 1);
      const second =
        ctx.options.clockStartHour * 3600 + snapshot.firstBuckets[pair] * clockStepSeconds;
      hours[Math.min(23, Math.floor((second % 86400) / 3600))]++;
    }
    meetingsPerHour = hours;
    updateMeetingsChart();
    ctx.setChart(
      'approachChart',
      snapshot.count
        ? histogramChart(
            binValues(snapshot.minimumDistances, 0, ctx.options.distance, 20, snapshot.count),
            0,
            ctx.options.distance,
            {
              xLabel: 'closest approach of a pair (m)',
              yLabel: 'pairs',
              formatX: value => value.toFixed(0),
              description:
                'Histogram of the smallest distance reached by each encounter pair, up to the encounter distance.'
            }
          )
        : null
    );
    const alikeRange = ctx.options.colorRangeMeters;
    const similarities = new Float32Array(snapshot.count);
    for (let pair = 0; pair < snapshot.count; pair++)
      similarities[pair] = getSimilarity(snapshot, pair);
    ctx.setChart(
      'similarityChart',
      scored
        ? histogramChart(binValues(similarities, 0, alikeRange, 25), 0, alikeRange, {
            xLabel: `route ${ctx.options.similarityMetric === 'frechet' ? 'Frechet' : 'Hausdorff'} distance of a pair (m)`,
            yLabel: 'pairs',
            color: 2,
            markers: [{x: alikeMeters, label: 'alike'}],
            formatX: value => value.toFixed(0),
            description:
              'Histogram of whole-route distance for every encounter pair; the rule marks the alike distance. Pairs left of it are traveling together.'
          })
        : null
    );
    ctx.setReadout(
      'alike',
      scored
        ? `${formatCount(alike)} of ${formatCount(scored)} (${((100 * alike) / scored).toFixed(0)}%) within ${formatCount(alikeMeters)} m`
        : 'n/a'
    );
    ctx.setReadout('closest', Number.isFinite(closest) ? `${closest.toFixed(1)} m` : 'n/a');
    ctx.setReadout('longest', formatDuration(longest));
    const topTypes = [...typeCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3);
    ctx.setReadout(
      'typePairs',
      topTypes.length
        ? topTypes.map(([key, value]) => `${key} ${formatCount(value)}`).join('; ')
        : 'n/a'
    );
    let busiest = 0;
    for (let hour = 1; hour < 24; hour++) if (hours[hour] > hours[busiest]) busiest = hour;
    ctx.setReadout(
      'busiestHour',
      snapshot.count
        ? `${String(busiest).padStart(2, '0')}:00 UTC (${formatCount(hours[busiest])} first meetings)`
        : 'n/a'
    );
    // Show the actual 3 x 3 search neighbourhood for a pair from this run. The grid, radius and
    // route-sample leash are derived from the same trajectories as the pair readback.
    if (snapshot.count) {
      let strongest = 0;
      for (let pair = 1; pair < snapshot.count; pair++) {
        if (snapshot.bucketCounts[pair] > snapshot.bucketCounts[strongest]) strongest = pair;
      }
      const second =
        ctx.options.clockStartHour * 3600 +
        snapshot.firstBuckets[strongest] * ctx.options.clockStepSeconds;
      const first = interpolate(snapshot.ids[strongest], second);
      const partner = interpolate(snapshot.partners[strongest], second);
      if (first && partner) {
        const annotations = getVesselEncounterSearchAnnotations(
          vessels,
          first,
          partner,
          CELL_SIZE,
          ctx.options.distance,
          getVesselRouteLeashWitness(vessels, snapshot.ids[strongest], snapshot.partners[strongest])
        );
        annotations.push({
          kind: 'note',
          id: 'encounter-duration',
          coordinate: vessels.unproject((first[0] + partner[0]) / 2, (first[1] + partner[1]) / 2),
          title: `Together ${formatDuration(snapshot.bucketCounts[strongest] * ctx.options.clockStepSeconds)}`,
          text: `Closest ${snapshot.minimumDistances[strongest].toFixed(0)} m`,
          priority: 7
        });
        ctx.setAnnotations('encounter-search', annotations);
      } else {
        ctx.setAnnotations('encounter-search', null);
      }
    } else {
      ctx.setAnnotations('encounter-search', null);
    }
  }

  function updateMeetingsChart(): void {
    const marker = ctx.options.timeOfDay / 3600;
    ctx.setChart('meetingsChart', {
      kind: 'bars',
      values: meetingsPerHour,
      labels: Array.from({length: 24}, (_, hour) => (hour % 3 === 0 ? `${hour}` : '')),
      highlight: [Math.min(23, Math.floor(marker))],
      height: 120,
      yLabel: 'first meetings',
      formatY: value => value.toFixed(0),
      description:
        'Pairs that first came within the encounter distance, by UTC hour. The highlighted bar is the hour of the time shown on the map.'
    });
  }

  function summarizeSelection(): void {
    const snapshot = selectedSnapshot;
    if (!snapshot) return;
    const metric =
      ctx.options.similarityMetric === 'frechet' ? snapshot.frechet : snapshot.hausdorff;
    let near = 0;
    for (let track = 0; track < trackCount; track++) {
      if (track !== selectedTrack && metric[track] <= ctx.options.alikeMeters) near++;
    }
    const distances = metric.slice(0, trackCount);
    distances[selectedTrack] = Number.NaN;
    ctx.setChart(
      'selectedChart',
      histogramChart(
        binValues(distances, 0, ctx.options.colorRangeMeters, 25),
        0,
        ctx.options.colorRangeMeters,
        {
          xLabel: 'route distance from the selected vessel (m)',
          yLabel: 'tracks',
          color: 1,
          markers: [{x: ctx.options.alikeMeters, label: 'alike'}],
          formatX: value => value.toFixed(0),
          description:
            'Histogram of the route distance from the selected vessel to every other track.'
        }
      )
    );
    ctx.setReadout(
      'routesLikeSelected',
      `${formatCount(near)} tracks within ${formatCount(ctx.options.alikeMeters)} m`
    );
  }

  // ---- Instance --------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [prepareCompiled, encounterCompiled, viewCompiled, selectionCompiled],

    setOption(id, _value, state) {
      switch (id) {
        case 'distance':
          distanceParameter.write(Float32Array.of(state.distance));
          encountersDirty = true;
          markChanged();
          writeView();
          break;
        case 'minSpeedKnots':
          validityParameter.write(Float32Array.of(state.minSpeedKnots * METERS_PER_KNOT_SECOND));
          encountersDirty = true;
          markChanged();
          writeView();
          break;
        case 'clockStepSeconds':
        case 'clockStartHour':
          writeClock();
          syncClockRange();
          encountersDirty = true;
          markChanged();
          writeView();
          break;
        case 'play':
        case 'playSpeed':
        case 'loop':
          break;
        case 'timeOfDay':
          writeView();
          updateMeetingsChart();
          break;
        case 'connectorTime':
        case 'connectorFilter':
          writeView();
          break;
        case 'similarityMetric':
        case 'alikeMeters':
          writeView();
          summarizePairs();
          summarizeSelection();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'selectFerry') {
        selectTrack(findFerryTrack());
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      clock.advance(frame);
      if (!prepared) {
        prepareCompiled.encode(commandEncoder, {parameters: undefined});
        prepared = true;
        metricsReader.request(commandEncoder);
      } else {
        metricsReader.flush(commandEncoder);
      }
      if (encountersDirty) {
        encounterCompiled.encode(commandEncoder, {parameters: undefined});
        encountersDirty = false;
        viewDirty = true;
      }
      if (viewDirty) {
        viewCompiled.encode(commandEncoder, {parameters: undefined});
        viewDirty = false;
      }
      if (selectionDirty) {
        selectionCompiled.encode(commandEncoder, {parameters: undefined});
        selectionDirty = false;
      }
      const settled = performance.now() - lastChange > SETTLE_MILLISECONDS;
      if (pairsStale && settled) {
        pairReader.markStale();
        pairsStale = false;
      }
      if (selectionStale) {
        selectionReader.markStale();
        selectionStale = false;
      }
      pairReader.flush(commandEncoder);
      selectionReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.trackColor !== 'off') {
        const bySimilarity = options.trackColor === 'similarity';
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'encounter-tracks',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 1.4,
            values: bySimilarity
              ? options.similarityMetric === 'frechet'
                ? selectedFrechet
                : selectedHausdorff
              : categoryBuffer,
            valueFormat: bySimilarity ? 'float32' : 'uint32',
            valueIndices: segmentTracksBuffer,
            colormap: bySimilarity ? 'magma' : 'category',
            valueRange: [0, options.colorRangeMeters],
            palette: VESSEL_CATEGORY_COLORS,
            noDataColor: dark ? [140, 146, 160, 60] : [100, 108, 124, 60],
            opacity: bySimilarity ? 0.8 : 0.38
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'encounter-selected-route',
          coordinateOrigin,
          segments: selectedSegments,
          instanceCount: vessels.longestTrack,
          widthPixels: 4,
          color: dark ? [255, 255, 255, 255] : [20, 24, 32, 255]
        })
      );
      if (options.showVessels) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'encounter-vessels',
            coordinateOrigin,
            positions: vehicles,
            instanceCount: trackCount,
            values: categoryBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: VESSEL_CATEGORY_COLORS,
            radiusPixels: 3.2,
            opacity: 0.95
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'encounter-connectors',
          coordinateOrigin,
          segments: connectors,
          instanceCount: PAIR_CAPACITY,
          widthPixels: 6,
          values: options.similarityMetric === 'frechet' ? pairFrechet : pairHausdorff,
          valueFormat: 'float32',
          colormap: 'magma',
          valueRange: [0, options.colorRangeMeters],
          noDataColor: [255, 140, 60, 255],
          opacity: 0.95
        })
      );
      return layers;
    },

    getTooltip(event) {
      const snapshot = pairSnapshot;
      const viewport = ctx.getViewport();
      if (!snapshot || !viewport) return null;
      const options = ctx.options;
      const time = getClockStart() + getBucket() * options.clockStepSeconds;
      let best = -1;
      let bestDistance = 12 * 12;
      for (let pair = 0; pair < snapshot.count; pair++) {
        let first = null as [number, number] | null;
        let second = null as [number, number] | null;
        if (options.connectorTime === 'bucket') {
          first = interpolate(snapshot.ids[pair], time);
          second = interpolate(snapshot.partners[pair], time);
        } else {
          const start = getClockStart() + snapshot.firstBuckets[pair] * options.clockStepSeconds;
          first = interpolate(snapshot.ids[pair], start);
          second = interpolate(snapshot.partners[pair], start);
        }
        if (!first || !second) continue;
        if (
          options.connectorTime === 'bucket' &&
          Math.hypot(first[0] - second[0], first[1] - second[1]) > options.distance + 1e-3
        )
          continue;
        const [longitude, latitude] = vessels.unproject(
          (first[0] + second[0]) / 2,
          (first[1] + second[1]) / 2
        );
        const [x, y] = viewport.project([longitude, latitude]);
        const squared = (x - event.pixel[0]) ** 2 + (y - event.pixel[1]) ** 2;
        if (squared < bestDistance) {
          bestDistance = squared;
          best = pair;
        }
      }
      if (best < 0) return null;
      const similarity = getSimilarity(snapshot, best);
      const name = options.similarityMetric === 'frechet' ? 'Frechet' : 'Hausdorff';
      return `${describeTrack(snapshot.ids[best])} meets ${describeTrack(snapshot.partners[best])}: closest ${snapshot.minimumDistances[best].toFixed(0)} m, together for ${formatDuration(snapshot.bucketCounts[best] * options.clockStepSeconds)}; whole routes ${Number.isFinite(similarity) ? `${(similarity / 1000).toFixed(1)} km apart (${name})` : 'not scored'}`;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const [x, y] = vessels.project(event.coordinate[0], event.coordinate[1]);
      const {track, distance} = findNearestTrack(vessels, x, y);
      if (track < 0 || distance > 1500) return false;
      selectTrack(track);
      return true;
    },

    destroy() {
      destroyed = true;
      metricsReader.stop();
      pairReader.stop();
      selectionReader.stop();
      resources.destroy();
    }
  };

  /** Position of a track at a clock time (seconds), or null outside its span. */
  function interpolate(track: number, time: number): [number, number] | null {
    const first = vessels.offsets[track];
    const last = vessels.offsets[track + 1] - 1;
    if (last < first || time < vessels.timestamps[first] || time > vessels.timestamps[last])
      return null;
    let low = first;
    let high = last;
    while (low < high) {
      const middle = (low + high + 1) >> 1;
      if (vessels.timestamps[middle] <= time) low = middle;
      else high = middle - 1;
    }
    if (low === last) return [vessels.positions[low * 2], vessels.positions[low * 2 + 1]];
    const span = vessels.timestamps[low + 1] - vessels.timestamps[low];
    const t = span > 0 ? (time - vessels.timestamps[low]) / span : 0;
    return [
      vessels.positions[low * 2] +
        t * (vessels.positions[low * 2 + 2] - vessels.positions[low * 2]),
      vessels.positions[low * 2 + 1] +
        t * (vessels.positions[low * 2 + 3] - vessels.positions[low * 2 + 1])
    ];
  }

  function findFerryTrack(): number {
    const stGeorge = vessels.project(-74.0736, 40.6437);
    const whitehall = vessels.project(-74.0132, 40.7013);
    let best = 0;
    let bestScore = Infinity;
    for (let track = 0; track < trackCount; track++) {
      if (VESSEL_CATEGORIES[vessels.category[track]] !== 'passenger') continue;
      let nearGeorge = Infinity;
      let nearWhitehall = Infinity;
      for (let vertex = vessels.offsets[track]; vertex < vessels.offsets[track + 1]; vertex++) {
        nearGeorge = Math.min(
          nearGeorge,
          Math.hypot(
            vessels.positions[vertex * 2] - stGeorge[0],
            vessels.positions[vertex * 2 + 1] - stGeorge[1]
          )
        );
        nearWhitehall = Math.min(
          nearWhitehall,
          Math.hypot(
            vessels.positions[vertex * 2] - whitehall[0],
            vessels.positions[vertex * 2 + 1] - whitehall[1]
          )
        );
      }
      // Prefer long ferry days: more crossings means more encounters.
      const score =
        nearGeorge + nearWhitehall - (vessels.offsets[track + 1] - vessels.offsets[track]) * 2;
      if (score < bestScore) {
        bestScore = score;
        best = track;
      }
    }
    return best;
  }
}

/** Central range of the fixes padded by one lattice cell: `[minX, minY, maxX, maxY]`. */
function getBounds(positions: Float32Array): [number, number, number, number] {
  const xs: number[] = [];
  const ys: number[] = [];
  for (let index = 0; index < positions.length; index += 2 * 4) {
    xs.push(positions[index]);
    ys.push(positions[index + 1]);
  }
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  const low = (values: number[]) => values[Math.floor(values.length * 0.001)];
  const high = (values: number[]) => values[Math.floor(values.length * 0.999)];
  return [low(xs) - CELL_SIZE, low(ys) - CELL_SIZE, high(xs) + CELL_SIZE, high(ys) + CELL_SIZE];
}
