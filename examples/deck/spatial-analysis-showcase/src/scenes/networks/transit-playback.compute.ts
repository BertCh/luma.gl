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
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_STATUS
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {VesselMarkerLayer} from '../movement/b12-layers';
import type {SceneContext, SceneInstance} from '../scene';
import {
  countVehiclesInService,
  formatTransitClock,
  loadTransitTrips,
  TRANSIT_MODE_COLORS,
  TRANSIT_MODE_LABELS,
  TRANSIT_MODES,
  TRANSIT_WINDOW_SECONDS
} from './transit-data';
import {
  getTransitSpeedClassColors,
  TRANSIT_SPEED_BREAKS_KILOMETERS_PER_HOUR
} from './randstad-network-cartography';

/** Option state of the transit playback scene. */
export type TransitPlaybackOptions = {
  play: boolean;
  time: number;
  speed: string;
  loop: boolean;
  modeFilter: string;
  markerColor: 'mode' | 'speed';
  hierarchy: 'flat' | 'weighted';
  showTrails: boolean;
  trailMinutes: number;
  tailFade: number;
  showBackdrop: boolean;
};

const NO_TRACK = 0xffffffff;
/** Frames between status readbacks. */
const STATUS_INTERVAL_FRAMES = 10;
const KILOMETERS_PER_HOUR = 3.6;
/** Seconds between redraws of the live charts. */
const CHART_INTERVAL_SECONDS = 0.25;
const SPEED_CLASS_BREAKS_METERS_PER_SECOND = TRANSIT_SPEED_BREAKS_KILOMETERS_PER_HOUR.map(
  speed => speed / KILOMETERS_PER_HOUR
);
const MODE_PAINTER_ORDER = [1, 0, 4, 2, 3] as const;
const WEIGHTED_MARKER_SIZES = [6, 4, 8, 9, 8] as const;

/**
 * Transit playback: one `GPUTrajectoryPlayhead` graph interpolates every scheduled trip at the
 * clock, one `GPUTimeWindowFilter` graph selects the trail segments of the last few minutes. Both
 * take their per-frame input as a parameter buffer, so the clock, the trail length and the fade
 * never recompile anything. The in-service curve is counted once on the CPU from trip start and
 * end times; the bars are counted from the GPU status of every trip.
 */
export async function createTransitPlayback(
  ctx: SceneContext<TransitPlaybackOptions>
): Promise<SceneInstance<TransitPlaybackOptions>> {
  const trips = loadTransitTrips(ctx.datasets.get('poopdeck-gtfs-nl'));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = trips;
  const resources = new SpatialAnalysisResources(device, 'transit');
  const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];

  // ---- Static inputs ----------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', trips.positions);
  const timestampsBuffer = resources.createBuffer('timestamps', trips.timestamps);
  const offsetsBuffer = resources.createBuffer('trip-offsets', trips.offsets);
  const modeBuffer = resources.createBuffer('mode', trips.mode);
  const segmentsBuffer = resources.createBuffer('segments', trips.segments);
  const segmentTripsBuffer = resources.createBuffer('segment-trips', trips.segmentTracks);
  const segmentStartTimesBuffer = resources.createBuffer(
    'segment-start-times',
    trips.segmentStartTimes
  );
  const segmentEndTimesBuffer = resources.createBuffer('segment-end-times', trips.segmentEndTimes);
  const segmentMaskBuffer = resources.createBuffer(
    'segment-mask',
    new Uint32Array(segmentCount).fill(1)
  );
  const dataFrameBuffer = resources.createBuffer('data-frame', 4 * 4 * 4);
  const selectedRouteBuffer = resources.createBuffer(
    'selected-route',
    Math.max(1, trips.longestTrack - 1) * 4 * 4
  );
  const selectedVerticesBuffer = resources.createBuffer(
    'selected-vertices',
    Math.max(1, trips.longestTrack) * 2 * 4
  );
  const selectedBracketBuffer = resources.createBuffer('selected-bracket', 4 * 4);

  // ---- Playhead graph ---------------------------------------------------------------------------
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
      id: 'transit-marker-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'transit-playhead'});
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

  // ---- Trail (time window) graph ----------------------------------------------------------------
  const trailIds = resources.createBuffer('trail-ids', segmentCount * 4);
  const trailCount = resources.createBuffer('trail-count', 4);
  const trailOverflow = resources.createBuffer('trail-overflow', 4);
  const fadeWeights = resources.createBuffer('fade-weights', segmentCount * 4);
  const clipFractions = resources.createBuffer('clip-fractions', segmentCount * 8);
  const tripVisibleCounts = resources.createBuffer('trip-visible-counts', trackCount * 4);
  const windowParameters = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const trailDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'transit-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'transit-trails'});
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
      additionalPredicates: [
        {
          kind: 'selection',
          mask: importGraphBuffer(
            trailGraph,
            'segment-mask',
            segmentMaskBuffer,
            'uint32',
            segmentCount
          )
        }
      ],
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
        'segment-trips',
        segmentTripsBuffer,
        'uint32',
        segmentCount
      ),
      trackVisibleCounts: importGraphBuffer(
        trailGraph,
        'trip-visible-counts',
        tripVisibleCounts,
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

  // ---- State ------------------------------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, TRANSIT_WINDOW_SECONDS], rate: 1, step: 10}
  );
  let playhead = ctx.options.time;
  let destroyed = false;
  let selectedTrip = NO_TRACK;
  let statusStale = true;
  let lastChartSeconds = -Infinity;
  let lastChartKey = '';
  let furnitureKey = '';
  let latestActiveCount: number | null = null;
  let legendGround: 'light' | 'dark' | null = null;
  let statusSnapshot: {status: Uint32Array; positions: Float32Array; speeds: Float32Array} | null =
    null;
  const modeCounts = new Float64Array(TRANSIT_MODES.length);

  function publishFurniture(): void {
    const clockMinute = Math.floor(playhead / 60);
    const active =
      latestActiveCount === null
        ? 'active count pending'
        : `${formatCount(latestActiveCount)} in service`;
    const key = `${clockMinute}|${active}`;
    if (key === furnitureKey) return;
    furnitureKey = key;
    ctx.setFurniture({
      title: {
        subtitle: `Randstad timetable · ${formatTransitClock(playhead)} CEST · ${active}`,
        chips: ['Scheduled, not observed']
      },
      scaleBar: {units: 'metric'}
    });
  }

  function publishLegendGround(): void {
    const ground = ctx.ground();
    if (ground === legendGround) return;
    legendGround = ground;
    ctx.setLegendData('ground', ground);
  }

  function writeDataFrame(): void {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      minX = Math.min(minX, trips.positions[vertex * 2]);
      minY = Math.min(minY, trips.positions[vertex * 2 + 1]);
      maxX = Math.max(maxX, trips.positions[vertex * 2]);
      maxY = Math.max(maxY, trips.positions[vertex * 2 + 1]);
    }
    dataFrameBuffer.write(
      new Float32Array([
        minX,
        minY,
        maxX,
        minY,
        maxX,
        minY,
        maxX,
        maxY,
        maxX,
        maxY,
        minX,
        maxY,
        minX,
        maxY,
        minX,
        minY
      ])
    );
  }

  // In-service curve, counted once on the CPU.
  const inService = countVehiclesInService(trips, 60);
  const minutes = Float64Array.from(inService.times, seconds => seconds / 60);
  let peakVehicles = 0;
  let peakSecond = 0;
  inService.total.forEach((value, bin) => {
    if (value > peakVehicles) {
      peakVehicles = value;
      peakSecond = inService.times[bin];
    }
  });
  ctx.setReadout('trips', `${formatCount(trackCount)} trips, ${formatCount(vertexCount)} vertices`);
  ctx.setReadout(
    'peak',
    `${formatCount(peakVehicles)} vehicles at ${formatTransitClock(peakSecond)}`
  );
  const modeIndexOfFilter = () => TRANSIT_MODES.indexOf(ctx.options.modeFilter as never);

  function updateInServiceChart(): void {
    const selected = modeIndexOfFilter();
    const focus = selected >= 0 ? selected : TRANSIT_MODES.indexOf('rail');
    ctx.setChart('inService', {
      kind: 'line',
      height: 130,
      xLabel: 'local time',
      yLabel: 'vehicles in service',
      xDomain: [0, TRANSIT_WINDOW_SECONDS / 60],
      formatX: value => formatTransitClock(value * 60),
      formatY: value => formatCount(value),
      markers: [{x: playhead / 60, label: 'now'}],
      description:
        'Scheduled vehicles in service in every minute of the window, all modes and one mode, with a line at the playhead.',
      series: [
        {label: 'all modes', x: minutes, y: inService.total, area: true, color: 0},
        {
          label: TRANSIT_MODE_LABELS[TRANSIT_MODES[focus]].toLowerCase(),
          x: minutes,
          y: inService.perMode[focus],
          color: 1
        }
      ]
    });
  }

  function updateModeChart(): void {
    const selected = modeIndexOfFilter();
    ctx.setChart('modeChart', {
      kind: 'bars',
      height: 110,
      values: modeCounts,
      labels: TRANSIT_MODES.map(mode => TRANSIT_MODE_LABELS[mode]),
      highlight: selected >= 0 ? [selected] : undefined,
      yLabel: 'vehicles',
      description: 'Vehicles in service at the playhead, by mode.'
    });
  }

  function updateSpeedChart(): void {
    if (!statusSnapshot) return;
    const values = new Float64Array(5);
    for (let trip = 0; trip < trackCount; trip++) {
      if (statusSnapshot.status[trip] === GPU_TRAJECTORY_PLAYHEAD_STATUS.active) {
        const speed = statusSnapshot.speeds[trip] * KILOMETERS_PER_HOUR;
        const index = TRANSIT_SPEED_BREAKS_KILOMETERS_PER_HOUR.findIndex(
          threshold => speed < threshold
        );
        values[index < 0 ? values.length - 1 : index]++;
      }
    }
    ctx.setChart('speedChart', {
      kind: 'bars',
      height: 110,
      values,
      labels: ['0–15', '15–30', '30–60', '60–100', '100+'],
      yLabel: 'vehicles',
      description:
        'Fixed scheduled-speed classes of every vehicle in service at the playhead. These are the same classes drawn on the map.'
    });
  }

  function writeSegmentMask(): void {
    const selected = modeIndexOfFilter();
    const mask = new Uint32Array(segmentCount);
    if (selected < 0) mask.fill(1);
    else {
      for (let segment = 0; segment < segmentCount; segment++) {
        mask[segment] = trips.mode[trips.segmentTracks[segment]] === selected ? 1 : 0;
      }
    }
    segmentMaskBuffer.write(mask);
  }

  function describeTrip(trip: number): string {
    const mode = TRANSIT_MODE_LABELS[TRANSIT_MODES[trips.mode[trip]]];
    const name = trips.routeNames[trips.routeName[trip]] || 'no line name';
    return `${mode} ${name}`;
  }

  function describeSelection(): void {
    if (selectedTrip === NO_TRACK) {
      ctx.setReadout('selected', 'click a vehicle');
      return;
    }
    const speed = statusSnapshot ? statusSnapshot.speeds[selectedTrip] * KILOMETERS_PER_HOUR : NaN;
    ctx.setReadout(
      'selected',
      `${describeTrip(selectedTrip)}, in the window ${formatTransitClock(trips.startTime[selectedTrip])}-${formatTransitClock(trips.endTime[selectedTrip])}${Number.isFinite(speed) ? `, ${speed.toFixed(0)} km/h now` : ''}`
    );
  }

  function updateSelectionEvidence(): void {
    if (selectedTrip === NO_TRACK) {
      ctx.setReadout('binarySearch', 'Waiting for an active scheduled trip');
      ctx.setChart('binarySearch', null);
      return;
    }
    const first = trips.offsets[selectedTrip];
    const last = trips.offsets[selectedTrip + 1] - 1;
    let lower = first;
    let upper = last;
    while (upper - lower > 1) {
      const middle = (lower + upper) >>> 1;
      if (trips.timestamps[middle] <= playhead) lower = middle;
      else upper = middle;
    }
    const start = trips.timestamps[lower];
    const end = trips.timestamps[upper];
    const fraction = end > start ? Math.max(0, Math.min(1, (playhead - start) / (end - start))) : 0;
    const route = new Float32Array(Math.max(1, trips.longestTrack - 1) * 4).fill(Number.NaN);
    for (let vertex = first; vertex < last; vertex++) {
      const target = (vertex - first) * 4;
      route.set(trips.positions.subarray(vertex * 2, vertex * 2 + 4), target);
    }
    selectedRouteBuffer.write(route);
    const vertices = new Float32Array(Math.max(1, trips.longestTrack) * 2).fill(Number.NaN);
    vertices.set(trips.positions.subarray(first * 2, (last + 1) * 2));
    selectedVerticesBuffer.write(vertices);
    selectedBracketBuffer.write(
      new Float32Array([
        trips.positions[lower * 2],
        trips.positions[lower * 2 + 1],
        trips.positions[upper * 2],
        trips.positions[upper * 2 + 1]
      ])
    );
    ctx.setReadout(
      'binarySearch',
      `${formatTransitClock(start)} → ${formatTransitClock(end)} · ${(fraction * 100).toFixed(0)}%`
    );
    ctx.setChart('binarySearch', {
      kind: 'diagram',
      width: 360,
      height: 86,
      description: `Selected ${describeTrip(selectedTrip)}: the binary-search bracket around the current playhead and its interpolation fraction.`,
      svg: `<line x1="24" y1="42" x2="336" y2="42" class="diagram-muted" stroke-width="3"/><circle cx="58" cy="42" r="6" class="diagram-ink"/><circle cx="302" cy="42" r="6" class="diagram-ink"/><line x1="${58 + 244 * fraction}" y1="18" x2="${58 + 244 * fraction}" y2="66" class="diagram-signal" stroke-width="3"/><text x="58" y="80" text-anchor="middle">${formatTransitClock(start)}</text><text x="302" y="80" text-anchor="middle">${formatTransitClock(end)}</text><text x="${58 + 244 * fraction}" y="14" text-anchor="middle">${(fraction * 100).toFixed(0)}%</text>`
    });
  }

  writeSegmentMask();
  writeDataFrame();
  publishLegendGround();
  publishFurniture();
  updateInServiceChart();
  updateModeChart();
  describeSelection();

  // ---- Readbacks --------------------------------------------------------------------------------
  const statusReader = new SummaryReader(
    resources,
    'transit-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: status, size: trackCount * 4},
      {buffer: currentPositions, size: trackCount * 8},
      {buffer: speeds, size: trackCount * 4},
      {buffer: trailCount, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const positionStart = 1 + trackCount;
      const speedStart = positionStart + trackCount * 2;
      const trailStart = speedStart + trackCount;
      statusSnapshot = {
        status: words.slice(1, 1 + trackCount),
        positions: floats.slice(positionStart, positionStart + trackCount * 2),
        speeds: floats.slice(speedStart, speedStart + trackCount)
      };
      modeCounts.fill(0);
      let before = 0;
      let after = 0;
      for (let trip = 0; trip < trackCount; trip++) {
        const value = statusSnapshot.status[trip];
        if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.active) modeCounts[trips.mode[trip]]++;
        else if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.beforeStart) before++;
        else if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.afterEnd) after++;
      }
      ctx.setReadout('active', words[0]);
      latestActiveCount = words[0];
      publishFurniture();
      ctx.setReadout('waiting', before);
      ctx.setReadout('finished', after);
      ctx.setReadout('trailSegments', words[trailStart]);
      for (const [index, mode] of TRANSIT_MODES.entries()) {
        ctx.setReadout(`active-${mode}`, modeCounts[index]);
      }
      const now = performance.now() / 1000;
      const chartKey = `${Math.round(playhead)}|${ctx.options.modeFilter}`;
      if (now - lastChartSeconds > CHART_INTERVAL_SECONDS && chartKey !== lastChartKey) {
        lastChartSeconds = now;
        lastChartKey = chartKey;
        updateModeChart();
        updateSpeedChart();
        updateInServiceChart();
      }
      if (
        selectedTrip === NO_TRACK ||
        statusSnapshot.status[selectedTrip] !== GPU_TRAJECTORY_PLAYHEAD_STATUS.active
      ) {
        const filter = modeIndexOfFilter();
        selectedTrip = statusSnapshot.status.findIndex(
          (value, trip) =>
            value === GPU_TRAJECTORY_PLAYHEAD_STATUS.active &&
            (filter < 0 || trips.mode[trip] === filter)
        );
        if (selectedTrip < 0) selectedTrip = NO_TRACK;
      }
      describeSelection();
      updateSelectionEvidence();
    }
  );

  function pickVehicle(pixel: readonly [number, number]): number {
    const viewport = ctx.getViewport();
    const snapshot = statusSnapshot;
    if (!viewport || !snapshot) return -1;
    const filter = modeIndexOfFilter();
    let best = -1;
    let bestDistance = 14 * 14;
    for (let trip = 0; trip < trackCount; trip++) {
      if (snapshot.status[trip] !== GPU_TRAJECTORY_PLAYHEAD_STATUS.active) continue;
      if (filter >= 0 && trips.mode[trip] !== filter) continue;
      const [longitude, latitude] = trips.unproject(
        snapshot.positions[trip * 2],
        snapshot.positions[trip * 2 + 1]
      );
      const [x, y] = viewport.project([longitude, latitude]);
      const squared = (x - pixel[0]) ** 2 + (y - pixel[1]) ** 2;
      if (squared < bestDistance) {
        bestDistance = squared;
        best = trip;
      }
    }
    return best;
  }

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [playheadCompiled, trailCompiled],

    setOption(id, _value, _state) {
      switch (id) {
        case 'modeFilter':
          writeSegmentMask();
          statusStale = true;
          updateModeChart();
          updateInServiceChart();
          ctx.requestLayers();
          break;
        case 'time':
        case 'play':
        case 'speed':
        case 'loop':
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      publishLegendGround();
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      // Paused, the clock sits on the slider, so story steps and deep links are deterministic.
      playhead = clock.advance(frame);
      ctx.setReadout('clock', `${formatTransitClock(playhead)} CEST`);
      publishFurniture();

      playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: 0}));
      playheadCompiled.encode(commandEncoder, {parameters: undefined});

      if (options.showTrails) {
        const trailSeconds = options.trailMinutes * 60;
        windowParameters.write(
          getGPUTimeWindowParameterValues({
            start: playhead - trailSeconds,
            end: playhead,
            startFadeDuration: trailSeconds * options.tailFade
          })
        );
        trailCompiled.encode(commandEncoder, {parameters: undefined});
      }

      if (statusStale || frame.frameIndex % STATUS_INTERVAL_FRAMES === 0) {
        statusReader.markStale();
        statusStale = false;
      }
      statusReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.ground() === 'dark';
      publishLegendGround();
      const layers: Layer[] = [];
      if (options.showBackdrop) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'transit-backdrop',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 0.7,
            color: dark ? [190, 200, 220, 22] : [60, 70, 90, 34]
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'transit-data-frame',
          coordinateOrigin,
          segments: dataFrameBuffer,
          instanceCount: 4,
          widthPixels: 0.9,
          dashArray: [5, 4],
          color: dark ? [225, 232, 250, 135] : [36, 46, 70, 135]
        })
      );
      if (options.showTrails) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'transit-trails',
            coordinateOrigin,
            segments: segmentsBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            values: modeBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTripsBuffer,
            colormap: 'category',
            palette: TRANSIT_MODE_COLORS,
            widthPixels: 2.4
          })
        );
      }
      const requestedMode = modeIndexOfFilter();
      const markerPalette =
        options.markerColor === 'speed' ? getTransitSpeedClassColors(dark) : TRANSIT_MODE_COLORS;
      for (const mode of MODE_PAINTER_ORDER) {
        if (requestedMode >= 0 && requestedMode !== mode) continue;
        layers.push(
          new VesselMarkerLayer({
            id: `transit-vehicles-${TRANSIT_MODES[mode]}`,
            coordinateOrigin,
            ids: activeIds,
            positions: currentPositions,
            headings,
            speeds,
            categories: modeBuffer,
            drawCommands: markerDraw,
            sizePixels: options.hierarchy === 'weighted' ? WEIGHTED_MARKER_SIZES[mode] : 6,
            colorMode: options.markerColor === 'speed' ? 'speedClasses' : 'category',
            speedClassBreaks: SPEED_CLASS_BREAKS_METERS_PER_SECOND,
            palette: markerPalette,
            categoryFilter: mode,
            selectedTrack: selectedTrip === NO_TRACK ? null : selectedTrip,
            outlineColor: dark ? [8, 10, 16, 235] : [20, 24, 32, 215]
          })
        );
      }
      if (selectedTrip !== NO_TRACK) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'transit-selected-timetable',
            coordinateOrigin,
            segments: selectedRouteBuffer,
            instanceCount: Math.max(1, trips.longestTrack - 1),
            widthPixels: 1.5,
            color: dark ? [255, 255, 255, 190] : [20, 25, 36, 190]
          }),
          new SpatialAnalysisPointLayer({
            id: 'transit-selected-timetable-vertices',
            coordinateOrigin,
            positions: selectedVerticesBuffer,
            instanceCount: Math.max(1, trips.longestTrack),
            radiusPixels: 2.5,
            color: [255, 218, 74, 235]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'transit-selected-bracket',
            coordinateOrigin,
            segments: selectedBracketBuffer,
            instanceCount: 1,
            widthPixels: 3,
            dashArray: [4, 3],
            color: [255, 218, 74, 255]
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const trip = pickVehicle(event.pixel);
      if (trip < 0 || !statusSnapshot) return null;
      const speed = statusSnapshot.speeds[trip] * KILOMETERS_PER_HOUR;
      return `${describeTrip(trip)} · ${TRANSIT_MODE_LABELS[TRANSIT_MODES[trips.mode[trip]]]} · ${formatTransitClock(playhead)} CEST · ${speed.toFixed(0)} km/h scheduled · ${formatTransitClock(trips.startTime[trip])}–${formatTransitClock(trips.endTime[trip])}`;
    },

    onClick(event) {
      const trip = pickVehicle(event.pixel);
      selectedTrip = trip < 0 || trip === selectedTrip ? NO_TRACK : trip;
      describeSelection();
      ctx.requestLayers();
      return trip >= 0;
    },

    destroy() {
      destroyed = true;
      statusReader.stop();
      resources.destroy();
    }
  };
}
