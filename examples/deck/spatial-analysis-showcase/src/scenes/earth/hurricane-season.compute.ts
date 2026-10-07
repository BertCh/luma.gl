// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
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
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  formatDayOfYear,
  getCategoryOfWind,
  getStormLabel,
  HURRICANE_CATEGORIES,
  HURRICANE_CATEGORY_COLORS,
  loadHurricaneTracks
} from './hurricane-data';

/** Option state of the hurricane season scene. */
export type HurricaneSeasonOptions = {
  playing: boolean;
  time: number;
  speed: number;
  loop: boolean;
  dayRange: readonly [number, number];
  seasons: readonly [number, number];
  maxGapHours: number;
  showTrails: boolean;
  trailDays: number;
  tailFade: number;
  headColor: 'wind' | 'peak' | 'season';
  headSize: number;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  showBackdrop: boolean;
  backdropOpacity: number;
};

const STATUS_INTERVAL_FRAMES = 6;
const DAYS_IN_YEAR = 365;
const HOURS_PER_DAY = 24;
const FIRST_SEASON = 1980;
const LAST_SEASON = 2025;

/**
 * Hurricane season on one calendar: the fixes of every storm are re-based to 1 January of their
 * own year, so the 46 seasons play on top of each other. `GPUTrajectoryPlayhead` places every storm
 * at the playhead day, `GPUTimeWindowFilter` keeps the trail behind it, and a small kernel reads
 * the interpolated wind at each head.
 *
 * Time is in days since 1 January 00:00 UTC (float32 days are exact to a few seconds over a year).
 */
export async function createHurricaneSeason(
  ctx: SceneContext<HurricaneSeasonOptions>
): Promise<SceneInstance<HurricaneSeasonOptions>> {
  const storms = loadHurricaneTracks(ctx.datasets.get('ibtracs-north-atlantic'), {
    projection: 'azimuthal',
    timeBase: 'calendar'
  });
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = storms;
  const resources = new SpatialAnalysisResources(device, 'hurricane-season');
  const drawProps = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;

  // Calendar hours to calendar days, for the playhead and the slider.
  const days = Float32Array.from(storms.timestamps, hours => hours / HOURS_PER_DAY);
  const segmentStartDays = Float32Array.from(
    storms.segmentStartTimes,
    hours => hours / HOURS_PER_DAY
  );
  const segmentEndDays = Float32Array.from(storms.segmentEndTimes, hours => hours / HOURS_PER_DAY);

  // ---- Static inputs -------------------------------------------------------------------------
  const lngLatBuffer = resources.createBuffer('lng-lat', storms.lngLat);
  const daysBuffer = resources.createBuffer('days', days);
  const offsetsBuffer = resources.createBuffer('offsets', storms.offsets);
  const segmentsBuffer = resources.createBuffer('segments', storms.segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', storms.segmentTracks);
  const segmentEndsBuffer = resources.createBuffer('segment-ends', storms.segmentEndVertices);
  const segmentStartDaysBuffer = resources.createBuffer('segment-start-days', segmentStartDays);
  const segmentEndDaysBuffer = resources.createBuffer('segment-end-days', segmentEndDays);
  const categoryBuffer = resources.createBuffer('category', storms.category);
  const windBuffer = resources.createBuffer('wind', storms.wind);
  const maxCategoryBuffer = resources.createBuffer('max-category', storms.maxCategoryWords);
  const seasonBuffer = resources.createBuffer('season', Float32Array.from(storms.season));
  const trackMaskBuffer = resources.createBuffer('track-mask', new Uint32Array(trackCount).fill(1));
  const segmentMaskBuffer = resources.createBuffer(
    'segment-mask',
    new Uint32Array(segmentCount).fill(1)
  );

  // ---- Playhead graph and head kernel -------------------------------------------------------------
  const currentPositions = resources.createBuffer('current-positions', trackCount * 8);
  const segmentRows = resources.createBuffer('segment-rows', trackCount * 4);
  const segmentFractions = resources.createBuffer('segment-fractions', trackCount * 4);
  const status = resources.createBuffer('status', trackCount * 4);
  const activeIds = resources.createBuffer('active-ids', trackCount * 4);
  const activeCount = resources.createBuffer('active-count', 4);
  const activeOverflow = resources.createBuffer('active-overflow', 4);
  const headPositions = resources.createBuffer('head-positions', trackCount * 8);
  const headWind = resources.createBuffer('head-wind', trackCount * 4);
  const headCategory = resources.createBuffer('head-category', trackCount * 4);
  const playheadParameters = resources.createParameterBuffer(
    'playhead',
    'float32',
    GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
  );
  const headDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'season-head-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'season-playhead'});
  const currentView = importGraphBuffer(
    playheadGraph,
    'current-positions',
    currentPositions,
    'float32x2',
    trackCount
  );
  const rowsView = importGraphBuffer(
    playheadGraph,
    'segment-rows',
    segmentRows,
    'uint32',
    trackCount
  );
  const fractionsView = importGraphBuffer(
    playheadGraph,
    'segment-fractions',
    segmentFractions,
    'float32',
    trackCount
  );
  const statusView = importGraphBuffer(playheadGraph, 'status', status, 'uint32', trackCount);
  playheadGraph.add(
    new GPUTrajectoryPlayhead({
      id: 'season-playhead',
      positions: importGraphBuffer(
        playheadGraph,
        'lng-lat',
        lngLatBuffer,
        'float32x2',
        vertexCount
      ),
      timestamps: importGraphBuffer(playheadGraph, 'days', daysBuffer, 'float32', vertexCount),
      trackOffsets: importGraphBuffer(
        playheadGraph,
        'offsets',
        offsetsBuffer,
        'uint32',
        trackCount + 1
      ),
      parameters: playheadParameters.importToGraph(playheadGraph),
      currentPositions: currentView,
      segmentRows: rowsView,
      segmentFractions: fractionsView,
      status: statusView,
      activeTracks: {
        ids: importGraphBuffer(playheadGraph, 'active-ids', activeIds, 'uint32', trackCount),
        count: importGraphBuffer(playheadGraph, 'active-count', activeCount, 'uint32', 1),
        overflow: importGraphBuffer(playheadGraph, 'active-overflow', activeOverflow, 'uint32', 1)
      },
      drawInstanceCount: playheadGraph.importGPUData(
        'head-draw-count',
        headDraw.getInstanceCountData(0)
      )
    })
  );
  // Heads: hide storms outside the chosen seasons and read the wind at the head from the
  // bracketing fixes (the playhead reports the row and the fraction, not the wind).
  addKernelPass(playheadGraph, {
    id: 'season-heads',
    invocationCount: trackCount,
    declarations: `fn classOf(wind: f32) -> u32 {
  var value = 0u;
  if (wind >= 34.0) { value = 1u; }
  if (wind >= 64.0) { value = 2u; }
  if (wind >= 83.0) { value = 3u; }
  if (wind >= 96.0) { value = 4u; }
  if (wind >= 113.0) { value = 5u; }
  if (wind >= 137.0) { value = 6u; }
  return value;
}`,
    bindings: [
      {name: 'currentPositions', view: currentView, type: 'f32', access: 'read'},
      {name: 'segmentRows', view: rowsView, type: 'u32', access: 'read'},
      {name: 'fractions', view: fractionsView, type: 'f32', access: 'read'},
      {name: 'status', view: statusView, type: 'u32', access: 'read'},
      {
        name: 'trackMask',
        view: importGraphBuffer(playheadGraph, 'track-mask', trackMaskBuffer, 'uint32', trackCount),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'wind',
        view: importGraphBuffer(playheadGraph, 'wind', windBuffer, 'float32', vertexCount),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'headPositions',
        view: importGraphBuffer(
          playheadGraph,
          'head-positions',
          headPositions,
          'float32x2',
          trackCount
        ),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'headWind',
        view: importGraphBuffer(playheadGraph, 'head-wind', headWind, 'float32', trackCount),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'headCategory',
        view: importGraphBuffer(playheadGraph, 'head-category', headCategory, 'uint32', trackCount),
        type: 'u32',
        access: 'read_write'
      }
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let shown = status[statusOffset + index] == 1u && trackMask[trackMaskOffset + index] != 0u;
  var x = nan;
  var y = nan;
  var blended = -1.0;
  var kind = 0u;
  if (shown) {
    x = currentPositions[currentPositionsOffset + index * 2u];
    y = currentPositions[currentPositionsOffset + index * 2u + 1u];
    let row = segmentRows[segmentRowsOffset + index];
    let fraction = fractions[fractionsOffset + index];
    blended = wind[windOffset + row] * (1.0 - fraction) + wind[windOffset + row + 1u] * fraction;
    kind = classOf(blended);
  }
  headPositions[headPositionsOffset + index * 2u] = x;
  headPositions[headPositionsOffset + index * 2u + 1u] = y;
  headWind[headWindOffset + index] = blended;
  headCategory[headCategoryOffset + index] = kind;`
  });
  const playheadCompiled = resources.track(playheadGraph.compile());

  // ---- Trail (time window) graph ------------------------------------------------------------------
  const trailIds = resources.createBuffer('trail-ids', segmentCount * 4);
  const trailCount = resources.createBuffer('trail-count', 4);
  const trailOverflow = resources.createBuffer('trail-overflow', 4);
  const fadeWeights = resources.createBuffer('fade-weights', segmentCount * 4);
  const clipFractions = resources.createBuffer('clip-fractions', segmentCount * 8);
  const windowParameters = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const trailDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'season-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'season-trails'});
  trailGraph.add(
    new GPUTimeWindowFilter({
      id: 'trail-window',
      timestamps: importGraphBuffer(
        trailGraph,
        'segment-start-days',
        segmentStartDaysBuffer,
        'float32',
        segmentCount
      ),
      endTimestamps: importGraphBuffer(
        trailGraph,
        'segment-end-days',
        segmentEndDaysBuffer,
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
      drawInstanceCount: trailGraph.importGPUData(
        'trail-draw-count',
        trailDraw.getInstanceCountData(0)
      )
    })
  );
  const trailCompiled = resources.track(trailGraph.compile());

  // ---- State -----------------------------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'playing', speed: 'speed', loop: 'loop'},
    {range: [ctx.options.dayRange[0], ctx.options.dayRange[1]], rate: 1, step: 0.25}
  );
  let playhead = ctx.options.time;
  let destroyed = false;
  let statusStale = true;
  let chartDay = -1;
  let aliveByDay = new Float64Array(DAYS_IN_YEAR);
  let selectedSeasonCount = 0;

  function writeMasks(): void {
    const [first, last] = ctx.options.seasons;
    const trackMask = new Uint32Array(trackCount);
    let selected = 0;
    for (let track = 0; track < trackCount; track++) {
      const inside = storms.season[track] >= first && storms.season[track] <= last;
      trackMask[track] = inside ? 1 : 0;
      if (inside) selected++;
    }
    const segmentMask = new Uint32Array(segmentCount);
    for (let segment = 0; segment < segmentCount; segment++) {
      segmentMask[segment] = trackMask[storms.segmentTracks[segment]];
    }
    trackMaskBuffer.write(trackMask);
    segmentMaskBuffer.write(segmentMask);
    selectedSeasonCount = Math.max(1, last - first + 1);
    ctx.setReadout(
      'seasonsShown',
      `${first} to ${last}: ${selected} storms in ${last - first + 1} seasons, ${(selected / selectedSeasonCount).toFixed(1)} per season`
    );
    // Average storms alive on each day of the year in those seasons.
    aliveByDay = new Float64Array(DAYS_IN_YEAR);
    for (let track = 0; track < trackCount; track++) {
      if (!trackMask[track]) continue;
      const alive = new Set<number>();
      for (let vertex = storms.offsets[track]; vertex < storms.offsets[track + 1]; vertex++) {
        alive.add(Math.min(DAYS_IN_YEAR - 1, Math.floor(days[vertex])));
      }
      for (const day of alive) aliveByDay[day] += 1 / selectedSeasonCount;
    }
    chartDay = -1;
    updateSparkline();
    statusStale = true;
    ctx.requestLayers();
  }

  function updateSparkline(): void {
    const day = Math.min(DAYS_IN_YEAR - 1, Math.max(0, Math.floor(playhead)));
    if (day === chartDay) return;
    chartDay = day;
    ctx.setChart('aliveSpark', {
      kind: 'sparkline',
      values: aliveByDay,
      highlight: day,
      height: 44,
      description:
        'Average number of storms alive on each day of the year in the chosen seasons. The dot is the playhead day.'
    });
  }

  // ---- Readback ----------------------------------------------------------------------------------
  const statusReader = new SummaryReader(
    resources,
    'season-status',
    [
      {buffer: headWind, size: trackCount * 4},
      {buffer: trailCount, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const winds = new Float32Array(bytes, 0, trackCount);
      const trail = new Uint32Array(bytes, trackCount * 4, 1)[0];
      const classes = new Float64Array(HURRICANE_CATEGORIES.length);
      let alive = 0;
      let strongest = -1;
      for (let track = 0; track < trackCount; track++) {
        const wind = winds[track];
        if (!(wind >= 0)) continue;
        alive++;
        classes[getCategoryOfWind(wind)]++;
        if (strongest < 0 || wind > winds[strongest]) strongest = track;
      }
      ctx.setReadout('alive', alive);
      ctx.setReadout(
        'strongest',
        strongest >= 0
          ? `${getStormLabel(storms, strongest)}: ${winds[strongest].toFixed(0)} kt (${HURRICANE_CATEGORIES[getCategoryOfWind(winds[strongest])]})`
          : 'no storm alive'
      );
      ctx.setReadout('trailSegments', trail);
      ctx.setChart('classChart', {
        kind: 'bars',
        values: classes,
        labels: HURRICANE_CATEGORIES.map(label => label.replace('Cat ', 'C')),
        highlight: strongest >= 0 ? [getCategoryOfWind(winds[strongest])] : [],
        height: 110,
        yLabel: 'storms',
        formatY: value => value.toFixed(0),
        description:
          'Storms alive on the playhead day by Saffir-Simpson class of the wind at their head.'
      });
    }
  );

  ctx.setReadout(
    'storms',
    `${formatCount(trackCount)} storms, ${formatCount(vertexCount)} fixes, ${LAST_SEASON - FIRST_SEASON + 1} seasons overlaid`
  );
  writeMasks();

  // ---- Instance -----------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [playheadCompiled, trailCompiled],

    setOption(id) {
      switch (id) {
        case 'dayRange':
          clock.setRange(ctx.options.dayRange[0], ctx.options.dayRange[1]);
          break;
        case 'seasons':
          writeMasks();
          break;
        case 'time':
        case 'playing':
        case 'speed':
        case 'loop':
        case 'maxGapHours':
        case 'trailDays':
        case 'tailFade':
          statusStale = true;
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
      playhead = clock.advance(frame);
      const hours = Math.round((playhead - Math.floor(playhead)) * 24);
      ctx.setReadout(
        'clock',
        `${formatDayOfYear(playhead)}, ${String(hours % 24).padStart(2, '0')}:00 UTC`
      );
      updateSparkline();

      playheadParameters.write(
        getGPUTrajectoryPlayheadParameterValues({
          playhead,
          maxGap: options.maxGapHours / HOURS_PER_DAY
        })
      );
      playheadCompiled.encode(commandEncoder, {parameters: undefined});

      if (options.showTrails) {
        windowParameters.write(
          getGPUTimeWindowParameterValues({
            start: playhead - options.trailDays,
            end: playhead,
            startFadeDuration: options.trailDays * options.tailFade
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
            id: 'season-backdrop',
            ...drawProps,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            values: trackMaskBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTracksBuffer,
            colormap: 'mask',
            color: dark ? [190, 200, 220, 255] : [60, 70, 90, 255],
            noDataColor: [0, 0, 0, 0],
            widthPixels: 1,
            opacity: options.backdropOpacity
          })
        );
      }
      if (options.showTrails) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'season-trails',
            ...drawProps,
            segments: segmentsBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            values: categoryBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentEndsBuffer,
            colormap: 'category',
            palette: HURRICANE_CATEGORY_COLORS,
            widthPixels: 2.6
          })
        );
      }
      const headProps =
        options.headColor === 'wind'
          ? {
              values: headCategory,
              valueFormat: 'uint32' as const,
              colormap: 'category' as const,
              palette: HURRICANE_CATEGORY_COLORS
            }
          : options.headColor === 'peak'
            ? {
                values: maxCategoryBuffer,
                valueFormat: 'uint32' as const,
                colormap: 'category' as const,
                palette: HURRICANE_CATEGORY_COLORS
              }
            : {
                values: seasonBuffer,
                valueFormat: 'float32' as const,
                valueRange: [FIRST_SEASON, LAST_SEASON] as const,
                colormap: options.ramp
              };
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'season-head-halo',
          ...drawProps,
          positions: headPositions,
          ids: activeIds,
          drawCommands: headDraw,
          radiusPixels: options.headSize + 2.2,
          color: dark ? [10, 14, 24, 235] : [255, 255, 255, 235]
        }),
        new SpatialAnalysisPointLayer({
          id: `season-heads-${options.headColor}`,
          ...drawProps,
          positions: headPositions,
          ids: activeIds,
          drawCommands: headDraw,
          radiusPixels: options.headSize,
          opacity: 1,
          ...headProps
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      statusReader.stop();
      resources.destroy();
    }
  };
}
