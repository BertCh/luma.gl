// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUTrajectoryPlayheadParameterValues,
  GPUTrajectoryPlayhead,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getClassBreaks, getClassCounts, getExtent, getHistogram} from '../../cartography/breaks';
import {getClassTableLayerProps} from '../../cartography/class-table';
import {formatCount, formatDistance, formatPercent} from '../../cartography/live-text';
import type {LngLat} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {AgeFadeTrailLayer} from './nyc-taxi-trails-layers';
import {
  BACKDROP_INK,
  CHORD_INK,
  getCompassName,
  getFareTable,
  getHeadingColors,
  HEAD_HALO,
  HEAD_INK,
  HEAD_RADIUS_STOPS,
  HEADING_BINS,
  HEADING_RAMP,
  INSPECT_ROUTE_INK,
  INSPECT_TICK_INK,
  MAXIMUM_CHORDS,
  MINIMUM_CHORD_METERS,
  ROUTE_INK,
  TRAIL_INK,
  TRAIL_WIDTH_STOPS,
  type RampMatch
} from './nyc-taxi-trails-style';

/** Option state of the nyc-taxi-trails scene. */
export type NycTaxiTrailsOptions = {
  play: boolean;
  time: number;
  playSpeed: number;
  loop: boolean;
  trailMinutes: number;
  colorBy: 'none' | 'heading' | 'fare';
  rampMatch: RampMatch;
  compareRamps: boolean;
  showBackdrop: boolean;
  showChords: boolean;
  inspect: boolean;
};

/**
 * The window of the TLC records in the dataset's clock: 08:00 to 08:30 is 60 to 1,860 s after the
 * 07:59:00 time origin. The minute of padding at either end of the archive is not played.
 */
export const TRAILS_FIRST_SECOND = 60;
export const TRAILS_LAST_SECOND = 1860;
/** The ramp-up band of the road chart: the window opens with few cabs already under way. */
const RAMP_UP_END_SECOND = 300;
const STATUS_INTERVAL_FRAMES = 10;
const COMPASS_DEGREES = 360;
const FARE_HISTOGRAM_BINS = 30;
const FARE_HISTOGRAM_MAXIMUM = 60;

/**
 * Routed taxi trails over a 30-minute Friday morning window. A `GPUTrajectoryPlayhead` graph
 * interpolates every taxi at the clock, a `GPUTimeWindowFilter` graph selects the route segments of
 * the last few minutes with fade weights and clip fractions, and both are drawn straight from GPU
 * storage with indirect counts. The clock and the trail length are parameter buffers. The CPU adds
 * what the story needs to explain the data: the heading of every segment, the circuity of every
 * route, the fare classes, and a rose of the heads' headings read back every few frames.
 */
export async function createNycTaxiTrails(
  ctx: SceneContext<NycTaxiTrailsOptions>
): Promise<SceneInstance<NycTaxiTrailsOptions>> {
  const {device} = ctx;
  const dataset = ctx.datasets.get('poopdeck-nyc-taxi-paths');
  const origin = dataset.defaultOrigin;
  const projection = dataset.getProjection(origin);
  const offsets = dataset.column<Uint32Array>('pathOffsets');
  const timestamps = Float32Array.from(dataset.column<Uint32Array>('timestamp'));
  const positions = dataset.projectColumn('vertices', origin);
  const fares = dataset.column<Float32Array>('fare_amount');
  const trackCount = offsets.length - 1;
  const vertexCount = timestamps.length;
  const timeOriginSeconds =
    (dataset.manifest.properties as {timeOriginMs: number}).timeOriginMs / 1000;
  const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
  const resources = new SpatialAnalysisResources(device, 'taxi-trails');

  // ---- Segment tables and per-route facts (CPU, once) --------------------------------------------
  const segmentCount = vertexCount - trackCount;
  const segments = new Float32Array(segmentCount * 4);
  const segmentTracks = new Uint32Array(segmentCount);
  const segmentStartTimes = new Float32Array(segmentCount);
  const segmentEndTimes = new Float32Array(segmentCount);
  /** Heading of each segment as a share of the full circle, clockwise from north (0 to <1). */
  const segmentHeadings = new Float32Array(segmentCount);
  const routeLengths = new Float32Array(trackCount);
  const chordLengths = new Float32Array(trackCount);
  let maximumTrackVertices = 0;
  let timeEnd = 0;
  {
    let row = 0;
    for (let track = 0; track < trackCount; track++) {
      const first = offsets[track];
      const last = offsets[track + 1] - 1;
      maximumTrackVertices = Math.max(maximumTrackVertices, last - first + 1);
      let routeLength = 0;
      let heading = 0;
      for (let vertex = first; vertex < last; vertex++, row++) {
        const x0 = positions[vertex * 2];
        const y0 = positions[vertex * 2 + 1];
        const x1 = positions[vertex * 2 + 2];
        const y1 = positions[vertex * 2 + 3];
        segments.set([x0, y0, x1, y1], row * 4);
        segmentTracks[row] = track;
        segmentStartTimes[row] = timestamps[vertex];
        segmentEndTimes[row] = timestamps[vertex + 1];
        const dx = x1 - x0;
        const dy = y1 - y0;
        routeLength += Math.hypot(dx, dy);
        // A zero-length step keeps the heading before it, so it never flashes a wrong colour.
        if (dx !== 0 || dy !== 0) {
          heading = (((Math.atan2(dx, dy) / (2 * Math.PI)) % 1) + 1) % 1;
        }
        segmentHeadings[row] = heading;
      }
      routeLengths[track] = routeLength;
      chordLengths[track] = Math.hypot(
        positions[last * 2] - positions[first * 2],
        positions[last * 2 + 1] - positions[first * 2 + 1]
      );
      timeEnd = Math.max(timeEnd, timestamps[last]);
    }
  }
  const circuities: number[] = [];
  for (let track = 0; track < trackCount; track++) {
    // A route over a very short chord is mostly a loop around the block, not a detour.
    if (chordLengths[track] > 200) circuities.push(routeLengths[track] / chordLengths[track]);
  }
  circuities.sort((a, b) => a - b);
  const medianCircuity = circuities.length ? circuities[circuities.length >> 1] : 1;
  const detourShare = circuities.length
    ? circuities.filter(value => value > 1.3).length / circuities.length
    : 0;

  // Fare classes: five quantile classes computed once, shared by the trails, the legend and the chart.
  const fareBreaks = getClassBreaks(fares, 5, 'quantile');
  const fareExtent = getExtent(fares);
  const fareCounts = getClassCounts(fares, fareBreaks);

  // ---- Buffers -----------------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', positions);
  const timestampsBuffer = resources.createBuffer('timestamps', timestamps);
  const offsetsBuffer = resources.createBuffer('track-offsets', offsets);
  const segmentsBuffer = resources.createBuffer('segments', segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', segmentTracks);
  const segmentStartTimesBuffer = resources.createBuffer('segment-start-times', segmentStartTimes);
  const segmentEndTimesBuffer = resources.createBuffer('segment-end-times', segmentEndTimes);
  const segmentHeadingBuffer = resources.createBuffer('segment-headings', segmentHeadings);
  const fareBuffer = resources.createBuffer('fare', fares);

  // ---- Playhead graph: every taxi at the clock -----------------------------------------------------
  const currentPositions = resources.createBuffer('current-positions', trackCount * 8);
  const headings = resources.createBuffer('headings', trackCount * 4);
  const speeds = resources.createBuffer('speeds', trackCount * 4);
  const status = resources.createBuffer('status', trackCount * 4);
  const activeIds = resources.createBuffer('active-ids', trackCount * 4);
  const activeCount = resources.createBuffer('active-count', 4);
  const activeOverflow = resources.createBuffer('active-overflow', 4);
  const playheadParameters = resources.createParameterBuffer(
    'playhead',
    'float32',
    GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
  );
  const markerDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'taxi-marker-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'taxi-playhead'});
  playheadGraph.add(
    new GPUTrajectoryPlayhead({
      id: 'playhead',
      positions: importGraphBuffer(
        playheadGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        vertexCount
      ),
      timestamps: importGraphBuffer(
        playheadGraph,
        'timestamps',
        timestampsBuffer,
        'float32',
        vertexCount
      ),
      trackOffsets: importGraphBuffer(
        playheadGraph,
        'offsets',
        offsetsBuffer,
        'uint32',
        trackCount + 1
      ),
      parameters: playheadParameters.importToGraph(playheadGraph),
      currentPositions: importGraphBuffer(
        playheadGraph,
        'current-positions',
        currentPositions,
        'float32x2',
        trackCount
      ),
      headings: importGraphBuffer(playheadGraph, 'headings', headings, 'float32', trackCount),
      speeds: importGraphBuffer(playheadGraph, 'speeds', speeds, 'float32', trackCount),
      status: importGraphBuffer(playheadGraph, 'status', status, 'uint32', trackCount),
      activeTracks: {
        ids: importGraphBuffer(playheadGraph, 'active-ids', activeIds, 'uint32', trackCount),
        count: importGraphBuffer(playheadGraph, 'active-count', activeCount, 'uint32', 1),
        overflow: importGraphBuffer(playheadGraph, 'active-overflow', activeOverflow, 'uint32', 1)
      },
      drawInstanceCount: playheadGraph.importGPUData(
        'marker-draw-count',
        markerDraw.getInstanceCountData(0)
      )
    })
  );
  const playheadCompiled = resources.track(playheadGraph.compile());

  // ---- Trail graph: route segments of the last few minutes -----------------------------------------
  const trailIds = resources.createBuffer('trail-ids', segmentCount * 4);
  const trailCount = resources.createBuffer('trail-count', 4);
  const trailOverflow = resources.createBuffer('trail-overflow', 4);
  const fadeWeights = resources.createBuffer('fade-weights', segmentCount * 4);
  const clipFractions = resources.createBuffer('clip-fractions', segmentCount * 8);
  const trackVisibleCounts = resources.createBuffer('track-visible-counts', trackCount * 4);
  const windowParameters = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const trailDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'taxi-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'taxi-trails'});
  trailGraph.add(
    new GPUTimeWindowFilter({
      id: 'trail-window',
      timestamps: importGraphBuffer(
        trailGraph,
        'segment-start-times',
        segmentStartTimesBuffer,
        'float32',
        segmentCount
      ),
      endTimestamps: importGraphBuffer(
        trailGraph,
        'segment-end-times',
        segmentEndTimesBuffer,
        'float32',
        segmentCount
      ),
      window: windowParameters.importToGraph(trailGraph),
      output: {
        ids: importGraphBuffer(trailGraph, 'trail-ids', trailIds, 'uint32', segmentCount),
        count: importGraphBuffer(trailGraph, 'trail-count', trailCount, 'uint32', 1),
        overflow: importGraphBuffer(trailGraph, 'trail-overflow', trailOverflow, 'uint32', 1)
      },
      fadeWeights: importGraphBuffer(
        trailGraph,
        'fade-weights',
        fadeWeights,
        'float32',
        segmentCount
      ),
      clipFractions: importGraphBuffer(
        trailGraph,
        'clip-fractions',
        clipFractions,
        'float32x2',
        segmentCount
      ),
      trackIds: importGraphBuffer(
        trailGraph,
        'segment-tracks',
        segmentTracksBuffer,
        'uint32',
        segmentCount
      ),
      trackVisibleCounts: importGraphBuffer(
        trailGraph,
        'track-visible-counts',
        trackVisibleCounts,
        'uint32',
        trackCount
      ),
      drawInstanceCount: trailGraph.importGPUData(
        'trail-draw-count',
        trailDraw.getInstanceCountData(0)
      )
    })
  );
  const trailCompiled = resources.track(trailGraph.compile());

  // ---- Chords and the inspected cab: small buffers rewritten from the active-id readback ---------
  const chordBuffer = resources.createBuffer('chords', MAXIMUM_CHORDS * 16);
  const chordRouteBuffer = resources.createBuffer(
    'chord-routes',
    MAXIMUM_CHORDS * maximumTrackVertices * 16
  );
  const inspectRouteBuffer = resources.createBuffer('inspect-route', maximumTrackVertices * 16);
  const inspectVertexBuffer = resources.createBuffer('inspect-vertices', maximumTrackVertices * 8);
  const inspectIdBuffer = resources.createBuffer('inspect-id', 4);
  let chordCount = 0;
  let chordRouteCount = 0;
  let chordKey = '';
  let inspectedTrack = -1;
  let inspectedVertexCount = 0;
  let pendingPick = false;
  let lastActive = new Uint32Array(0);
  let hasRead = false;

  // ---- Charts that do not change with the clock ---------------------------------------------------
  const BIN_SECONDS = 30;
  const binCount = Math.ceil(timeEnd / BIN_SECONDS) + 1;
  const onRoad = new Float64Array(binCount);
  const startTimes = dataset.column<Uint32Array>('startTime');
  const endTimes = dataset.column<Uint32Array>('endTime');
  for (let track = 0; track < trackCount; track++) {
    const firstBin = Math.floor(startTimes[track] / BIN_SECONDS);
    const lastBin = Math.min(binCount - 1, Math.floor(endTimes[track] / BIN_SECONDS));
    for (let bin = firstBin; bin <= lastBin; bin++) onRoad[bin]++;
  }

  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [TRAILS_FIRST_SECOND, TRAILS_LAST_SECOND], rate: 1, step: 5}
  );

  let destroyed = false;
  const statusReader = new SummaryReader(
    resources,
    'taxi-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: activeOverflow, size: 4},
      {buffer: trailCount, size: 4},
      {buffer: trailOverflow, size: 4},
      {buffer: activeIds, size: trackCount * 4},
      {buffer: headings, size: trackCount * 4}
    ],
    bytes => {
      if (!destroyed) processStatus(bytes);
    }
  );

  function formatClock(seconds: number): string {
    const total = Math.round(timeOriginSeconds + seconds);
    const hours = Math.floor((total % 86400) / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  function getTrackLngLat(x: number, y: number): LngLat {
    return projection.unproject(x, y);
  }

  // ---- Readback of the active list --------------------------------------------------------------
  function processStatus(bytes: ArrayBuffer): void {
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const [active, activeOver, trail, trailOver] = words;
    ctx.setReadout('active', active);
    ctx.setReadout('segments', trail);
    ctx.setReadout(
      'overflow',
      activeOver || trailOver ? 'yes: some taxis or segments are not drawn' : 'no'
    );
    const shown = Math.min(active, trackCount);
    lastActive = words.slice(4, 4 + shown);
    hasRead = true;
    const headingStart = 4 + trackCount;
    if (ctx.options.colorBy === 'heading') publishRose(floats, headingStart, shown);
    if (ctx.options.showChords && !ctx.options.play) buildChords();
    if (pendingPick || (ctx.options.inspect && inspectedTrack < 0)) pickTaxi();
  }

  /** Bins the heads' headings (radians counter-clockwise from east) into compass sectors. */
  function publishRose(floats: Float32Array, headingStart: number, shown: number): void {
    const bins = new Array<number>(HEADING_BINS).fill(0);
    const degreesPerBin = COMPASS_DEGREES / HEADING_BINS;
    for (let index = 0; index < shown; index++) {
      const radians = floats[headingStart + lastActive[index]];
      const compass = (((90 - (radians * 180) / Math.PI) % 360) + 360) % 360;
      bins[Math.floor(compass / degreesPerBin) % HEADING_BINS]++;
    }
    // The busiest three neighbouring sectors, as a share of the heads on the road.
    let bestStart = 0;
    let bestSum = -1;
    for (let start = 0; start < HEADING_BINS; start++) {
      const sum = bins[start] + bins[(start + 1) % HEADING_BINS] + bins[(start + 2) % HEADING_BINS];
      if (sum > bestSum) {
        bestSum = sum;
        bestStart = start;
      }
    }
    const centerDegrees = (bestStart + 1.5) * degreesPerBin;
    ctx.setReadout(
      'dominantHeading',
      shown > 0 ? `${formatPercent(bestSum / shown)} head ${getCompassName(centerDegrees)}` : null
    );
    ctx.setChart('headingRose', {
      kind: 'rose',
      values: bins,
      labels: ['N', 'E', 'S', 'W'],
      colors: getHeadingColors(),
      baseline: new Array<number>(HEADING_BINS).fill(shown / HEADING_BINS),
      height: 180,
      description:
        'A rose of the headings of the cabs on the road, in ten-degree sectors from north, against an even spread: the avenue grid shows as spikes.'
    });
  }

  /** The chords of up to 40 cabs now on the road, each with its full route. */
  function buildChords(): void {
    const candidates: number[] = [];
    for (const track of lastActive) {
      if (chordLengths[track] > MINIMUM_CHORD_METERS) candidates.push(track);
    }
    const stride = Math.max(1, Math.floor(candidates.length / MAXIMUM_CHORDS));
    const chosen: number[] = [];
    for (
      let index = 0;
      index < candidates.length && chosen.length < MAXIMUM_CHORDS;
      index += stride
    ) {
      chosen.push(candidates[index]);
    }
    const key = chosen.join(',');
    if (key === chordKey) return;
    chordKey = key;
    const chords = new Float32Array(chosen.length * 4);
    let routeSegments = 0;
    for (const track of chosen) routeSegments += offsets[track + 1] - offsets[track] - 1;
    const routes = new Float32Array(routeSegments * 4);
    let routeRow = 0;
    chosen.forEach((track, index) => {
      const first = offsets[track];
      const last = offsets[track + 1] - 1;
      chords.set(
        [
          positions[first * 2],
          positions[first * 2 + 1],
          positions[last * 2],
          positions[last * 2 + 1]
        ],
        index * 4
      );
      // Segment rows of a track start at `offset - track`: every earlier track owns one vertex more.
      const firstSegment = first - track;
      const length = last - first;
      routes.set(segments.subarray(firstSegment * 4, (firstSegment + length) * 4), routeRow * 4);
      routeRow += length;
    });
    chordCount = chosen.length;
    chordRouteCount = routeSegments;
    if (chordCount > 0) {
      chordBuffer.write(chords);
      chordRouteBuffer.write(routes);
    }
    ctx.requestLayers();
  }

  /**
   * Picks the longest route among the cabs on the road now (skipping `exclude`). Before the first
   * readback there is no list, so the pick waits for it.
   */
  function pickTaxi(exclude = -1): void {
    if (!hasRead) {
      pendingPick = true;
      return;
    }
    let best = -1;
    let bestLength = -1;
    const consider = (track: number) => {
      if (track !== exclude && routeLengths[track] > bestLength) {
        bestLength = routeLengths[track];
        best = track;
      }
    };
    for (const track of lastActive) consider(track);
    // Nothing on the road at this clock (the clock stays inside the window, so this is a guard).
    if (best < 0) for (let track = 0; track < trackCount; track++) consider(track);
    pendingPick = false;
    inspectedTrack = best;
    const first = offsets[best];
    const last = offsets[best + 1] - 1;
    inspectedVertexCount = last - first + 1;
    const firstSegment = first - best;
    inspectRouteBuffer.write(
      segments.subarray(firstSegment * 4, (firstSegment + last - first) * 4)
    );
    inspectVertexBuffer.write(positions.subarray(first * 2, (last + 1) * 2));
    inspectIdBuffer.write(Uint32Array.of(best));
    ctx.setReadout(
      'searchSteps',
      `${Math.ceil(Math.log2(inspectedVertexCount))} steps over ${formatCount(inspectedVertexCount)} vertices`
    );
    ctx.setReadout('inspectRoute', `${formatDistance(routeLengths[best])} of route`);
    let west = Infinity;
    let south = Infinity;
    let east = -Infinity;
    let north = -Infinity;
    for (let vertex = first; vertex <= last; vertex++) {
      const [longitude, latitude] = getTrackLngLat(
        positions[vertex * 2],
        positions[vertex * 2 + 1]
      );
      west = Math.min(west, longitude);
      east = Math.max(east, longitude);
      south = Math.min(south, latitude);
      north = Math.max(north, latitude);
    }
    // The camera moves only while paused: a flight under a running clock would hide the cab.
    if (!ctx.options.play) {
      ctx.fitBounds([west, south, east, north], {transitionMs: 1400, maxZoom: 14.5, padding: 40});
    }
    ctx.requestLayers();
  }

  function clearInspection(): void {
    inspectedTrack = -1;
    inspectedVertexCount = 0;
    pendingPick = false;
    ctx.setReadout('searchSteps', null);
    ctx.setReadout('inspectRoute', null);
  }

  // ---- Charts ------------------------------------------------------------------------------------
  function publishRoadChart(): void {
    const times = Array.from(onRoad, (_, index) => index * BIN_SECONDS);
    ctx.setChart('roadChart', {
      kind: 'line',
      series: [{label: 'cabs on the road', x: times, y: Array.from(onRoad), area: true}],
      xLabel: 'Local time',
      yLabel: 'trips on the road (sample)',
      height: 110,
      xDomain: [TRAILS_FIRST_SECOND, TRAILS_LAST_SECOND],
      formatX: value => formatClock(value).slice(0, 5),
      bands: [{from: TRAILS_FIRST_SECOND, to: RAMP_UP_END_SECOND, label: 'ramp-up'}],
      link: {option: 'time', label: value => formatClock(value).slice(0, 5)},
      description:
        'Trips on the road per half minute in the sample, with the first minutes shaded because the window opens with few cabs already under way.'
    });
  }

  function publishFare(): void {
    const match = ctx.options.compareRamps ? 'matched' : ctx.options.rampMatch;
    const table = getFareTable({breaks: fareBreaks, extent: fareExtent, match});
    ctx.setLegendData('fare', {breaks: fareBreaks, extent: fareExtent, counts: fareCounts});
    ctx.setChart('fareHistogram', {
      kind: 'histogram',
      values: getHistogram(fares, [0, FARE_HISTOGRAM_MAXIMUM], FARE_HISTOGRAM_BINS),
      xDomain: [0, FARE_HISTOGRAM_MAXIMUM],
      breaks: fareBreaks,
      classColors: table.colors,
      xLabel: 'Fare (USD)',
      yLabel: 'trips',
      height: 110,
      formatX: value => `$${value}`,
      table: false,
      description:
        'Histogram of the fares of the sampled trips with the four quintile breaks marked and the five classes in the map colours.'
    });
  }

  function getTrailStyle(options: NycTaxiTrailsOptions, match: RampMatch) {
    if (options.colorBy === 'heading') {
      return {
        values: segmentHeadingBuffer,
        valueFormat: 'float32' as const,
        colormap: HEADING_RAMP,
        valueRange: [0, 1] as [number, number]
      };
    }
    if (options.colorBy === 'fare') {
      return {
        values: fareBuffer,
        valueFormat: 'float32' as const,
        valueIndices: segmentTracksBuffer,
        colormap: 'grayscale' as const,
        ...getClassTableLayerProps(getFareTable({breaks: fareBreaks, extent: fareExtent, match}))
      };
    }
    return {color: TRAIL_INK};
  }

  ctx.setReadout('trips', trackCount);
  ctx.setReadout('vertices', vertexCount);
  ctx.setReadout('segmentsTotal', segmentCount);
  ctx.setReadout('circuity', `${medianCircuity.toFixed(2)} times the chord`);
  ctx.setReadout('detourShare', formatPercent(detourShare));
  ctx.setCost({records: segmentCount, passes: 2});
  ctx.setFurniture({
    title: {
      sample: `${formatCount(trackCount)} trips, a seeded half of the routed sample`,
      chips: ['Sample', 'Modelled routes']
    }
  });
  publishRoadChart();
  publishFare();

  return {
    getCompiledGraphs: () => [playheadCompiled, trailCompiled] as never[],

    setOption(id) {
      switch (id) {
        case 'time':
        case 'playSpeed':
        case 'loop':
          return;
        case 'play':
          // Paused: refresh the chords for the frozen moment.
          chordKey = '';
          statusReader.markStale();
          break;
        case 'showChords':
          chordKey = '';
          if (!ctx.options.showChords) chordCount = 0;
          statusReader.markStale();
          break;
        case 'inspect':
          if (ctx.options.inspect) pickTaxi();
          else clearInspection();
          break;
        case 'rampMatch':
        case 'compareRamps':
          publishFare();
          break;
        default:
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id !== 'pickTaxi') return;
      ctx.setOptions({inspect: true}, {notify: false});
      // The next press takes the longest of the other cabs.
      pickTaxi(inspectedTrack);
    },

    onGroundChange() {
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      const playhead = clock.advance(frame);
      ctx.setReadout('clock', formatClock(playhead));
      playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead}));
      playheadCompiled.encode(commandEncoder, {parameters: undefined});
      const trailSeconds = options.trailMinutes * 60;
      windowParameters.write(
        getGPUTimeWindowParameterValues({
          start: playhead - trailSeconds,
          end: playhead,
          // Fade across the whole trail: weight 1 at the head, 0 at the tail.
          startFadeDuration: trailSeconds
        })
      );
      trailCompiled.encode(commandEncoder, {parameters: undefined});
      if (frame.frameIndex % STATUS_INTERVAL_FRAMES === 0) statusReader.markStale();
      statusReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const layers: Layer[] = [];
      if (options.showBackdrop) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-backdrop',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 0.75,
            color: BACKDROP_INK
          })
        );
      }
      const comparing = options.compareRamps && options.colorBy === 'fare';
      const sides: {match: RampMatch; side: 'a' | 'b' | undefined}[] = comparing
        ? [
            {match: 'mismatched', side: 'a'},
            {match: 'matched', side: 'b'}
          ]
        : [{match: options.rampMatch, side: undefined}];
      for (const {match, side} of sides) {
        layers.push(
          new AgeFadeTrailLayer({
            id: `taxi-trails${side ? `-${side}` : ''}`,
            coordinateOrigin,
            segments: segmentsBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            widthPixels: TRAIL_WIDTH_STOPS,
            blending: 'additive',
            opacity: 0.9,
            compareSide: side,
            ...getTrailStyle(options, match)
          })
        );
      }
      if (options.showChords && chordCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-chord-routes',
            coordinateOrigin,
            segments: chordRouteBuffer,
            instanceCount: chordRouteCount,
            widthPixels: 1.5,
            color: ROUTE_INK
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-chords',
            coordinateOrigin,
            segments: chordBuffer,
            instanceCount: chordCount,
            widthPixels: 1,
            dashArray: [5, 4],
            cap: 'butt',
            color: CHORD_INK
          })
        );
      }
      if (options.inspect && inspectedTrack >= 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-inspect-route',
            coordinateOrigin,
            segments: inspectRouteBuffer,
            instanceCount: inspectedVertexCount - 1,
            widthPixels: 2.5,
            color: INSPECT_ROUTE_INK,
            outlineColor: HEAD_HALO,
            outlineWidthPixels: 1.2
          }),
          new SpatialAnalysisPointLayer({
            id: 'taxi-inspect-vertices',
            coordinateOrigin,
            positions: inspectVertexBuffer,
            instanceCount: inspectedVertexCount,
            radiusPixels: 2.2,
            color: INSPECT_TICK_INK
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'taxi-heads',
          coordinateOrigin,
          positions: currentPositions,
          ids: activeIds,
          drawCommands: markerDraw,
          radiusPixels: HEAD_RADIUS_STOPS,
          color: HEAD_INK,
          outlineColor: HEAD_HALO,
          outlineWidthPixels: 1
        })
      );
      if (options.inspect && inspectedTrack >= 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'taxi-inspect-head',
            coordinateOrigin,
            positions: currentPositions,
            ids: inspectIdBuffer,
            instanceCount: 1,
            shape: 'ring',
            radiusPixels: 10,
            outlineWidthPixels: 2,
            color: HEAD_INK
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      statusReader.stop();
      resources.destroy();
    }
  };
}
