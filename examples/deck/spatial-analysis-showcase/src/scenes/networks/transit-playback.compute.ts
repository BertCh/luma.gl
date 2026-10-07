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
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {VesselMarkerLayer} from '../movement/b12-layers';
import {binValues, histogramChart} from '../movement/f-chart-helpers';
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

/** Option state of the transit playback scene. */
export type TransitPlaybackOptions = {
  play: boolean;
  time: number;
  speed: string;
  loop: boolean;
  modeFilter: string;
  markerColor: 'mode' | 'speed';
  markerSize: number;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  showTrails: boolean;
  trailMinutes: number;
  tailFade: number;
  showBackdrop: boolean;
};

const NO_TRACK = 0xffffffff;
/** Frames between status readbacks. */
const STATUS_INTERVAL_FRAMES = 10;
/** Speed in m/s at which the speed ramp ends (108 km/h). */
const SPEED_RAMP_METERS_PER_SECOND = 30;
const KILOMETERS_PER_HOUR = 3.6;
/** Seconds between redraws of the live charts. */
const CHART_INTERVAL_SECONDS = 0.25;

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
  let statusSnapshot: {status: Uint32Array; positions: Float32Array; speeds: Float32Array} | null =
    null;
  const modeCounts = new Float64Array(TRANSIT_MODES.length);

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
    const values: number[] = [];
    for (let trip = 0; trip < trackCount; trip++) {
      if (statusSnapshot.status[trip] === GPU_TRAJECTORY_PLAYHEAD_STATUS.active) {
        values.push(statusSnapshot.speeds[trip] * KILOMETERS_PER_HOUR);
      }
    }
    ctx.setChart(
      'speedChart',
      histogramChart(binValues(values, 0, 150, 15), 0, 150, {
        xLabel: 'speed of the vehicle now (km/h)',
        yLabel: 'vehicles',
        formatX: value => `${Math.round(value)}`,
        description:
          'Histogram of the scheduled speed of every vehicle in service at the playhead. Trams and buses sit below 40 km/h, trains reach 100 to 160.'
      })
    );
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

  writeSegmentMask();
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
      describeSelection();
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
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      // Paused, the clock sits on the slider, so story steps and deep links are deterministic.
      playhead = clock.advance(frame);
      ctx.setReadout('clock', `${formatTransitClock(playhead)} CEST`);

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
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showBackdrop) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'transit-backdrop',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 1,
            color: dark ? [190, 200, 220, 22] : [60, 70, 90, 34]
          })
        );
      }
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
            widthPixels: 2.2
          })
        );
      }
      const filter = modeIndexOfFilter();
      layers.push(
        new VesselMarkerLayer({
          id: 'transit-vehicles',
          coordinateOrigin,
          ids: activeIds,
          positions: currentPositions,
          headings,
          speeds,
          categories: modeBuffer,
          drawCommands: markerDraw,
          sizePixels: options.markerSize,
          colorMode: options.markerColor === 'speed' ? 'speed' : 'category',
          ramp: options.ramp,
          speedForFullColor: SPEED_RAMP_METERS_PER_SECOND,
          palette: TRANSIT_MODE_COLORS,
          categoryFilter: filter >= 0 ? filter : null,
          selectedTrack: selectedTrip === NO_TRACK ? null : selectedTrip,
          outlineColor: dark ? [8, 10, 16, 235] : [20, 24, 32, 215]
        })
      );
      return layers;
    },

    getTooltip(event) {
      const trip = pickVehicle(event.pixel);
      if (trip < 0 || !statusSnapshot) return null;
      const speed = statusSnapshot.speeds[trip] * KILOMETERS_PER_HOUR;
      return `${describeTrip(trip)}, ${speed.toFixed(0)} km/h (scheduled)`;
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
