// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTimeWindowParameterValues,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPULineDensityParameterValues,
  getGPUTrajectoryPlayheadParameterValues,
  GPU_LINE_DENSITY_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  GPULineDensity,
  GPUTrajectoryPlayhead
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {binValues, histogramChart} from '../movement/f-chart-helpers';
import {formatDriftDate, loadDrifters, OCEAN_REGIONS} from './ocean-drifters-data';

/** Option state of the ocean-drifters-playback scene. */
export type OceanDriftersPlaybackOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  trailDays: number;
  tailFade: number;
  maxGapDays: number;
  colorBy: 'sst' | 'age' | 'date';
  sstRange: readonly [number, number];
  ramp: 'magma' | 'inferno' | 'cividis';
  showBackdrop: boolean;
  showTrails: boolean;
  showDots: boolean;
  dotSize: number;
  background: 'none' | 'density';
  densityCell: '1' | '0.5' | '0.25';
  densityMax: number;
  backgroundOpacity: number;
  region: string;
};

/** Cell sizes of the density grids, compiled up front. */
const DENSITY_CELLS = [1, 0.5, 0.25] as const;
const STATUS_INTERVAL_FRAMES = 12;
const CHART_INTERVAL_SECONDS = 0.3;
const NO_ROW = 0xffffffff;
/** Days of age at the top of the age ramp. */
const AGE_RAMP_DAYS = 60;

type Viewer = <Format extends GPUVectorFormat>(
  name: string,
  buffer: Buffer,
  format: Format,
  length?: number
) => GraphDataView<Format>;

/** Imports each buffer once per graph and returns typed views of it. */
function createViewer(graph: GPUCommandGraph<void>): Viewer {
  const handles = new Map<Buffer, ReturnType<GPUCommandGraph<void>['importBuffer']>>();
  return (name, buffer, format, length) => {
    let handle = handles.get(buffer);
    if (!handle) {
      handle = graph.importBuffer(
        {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      handles.set(buffer, handle);
    }
    const rowBytes = format === 'float32x2' ? 8 : 4;
    return graph.createDataView(handle, {
      format,
      length: length ?? Math.floor(buffer.byteLength / rowBytes)
    });
  };
}

type DensityGrid = {
  cell: number;
  columns: number;
  rows: number;
  compiled: CompiledGPUCommandGraph<void>;
  densities: Buffer;
};

/**
 * Drifter playback. `GPUTrajectoryPlayhead` interpolates every drifter piece at the clock (a binary
 * search per piece) and reports which are active; a small kernel turns its bracketing row and
 * fraction into the sea-surface temperature at the playhead. `GPUTimeWindowFilter` selects the
 * trail segments inside `[playhead - trail, playhead]` and fades them. `GPULineDensity` turns all
 * tracks into per-cell track length at one of three compiled cell sizes. The clock, window, gap
 * limit and colors are parameter writes.
 */
export async function createOceanDriftersPlayback(
  ctx: SceneContext<OceanDriftersPlaybackOptions>
): Promise<SceneInstance<OceanDriftersPlaybackOptions>> {
  const {device} = ctx;
  const drifters = loadDrifters(ctx.datasets.get('poopdeck-drifters'));
  const pieceCount = drifters.trackCount;
  const vertexCount = drifters.vertexCount;
  const segmentCount = drifters.segmentCount;
  const resources = new SpatialAnalysisResources(device, 'drifter-playback');
  const drawProps = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;

  const lastTime = drifters.timeRange[1];
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, Math.ceil(lastTime)], rate: 1, step: 1, loop: true}
  );

  // ---- Static inputs ------------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', drifters.lngLat);
  const releaseBuffer = resources.createBuffer('release-positions', drifters.release);
  const timestampsBuffer = resources.createBuffer('timestamps', drifters.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', drifters.offsets);
  const sstBuffer = resources.createBuffer('sst', drifters.sst);
  const ages = new Float32Array(vertexCount);
  for (let piece = 0; piece < pieceCount; piece++) {
    const release = drifters.releaseDays[drifters.pieceTrack[piece]];
    for (let vertex = drifters.offsets[piece]; vertex < drifters.offsets[piece + 1]; vertex++) {
      ages[vertex] = drifters.timestamps[vertex] - release;
    }
  }
  const ageBuffer = resources.createBuffer('age', ages);
  const segmentsBuffer = resources.createBuffer('segments', drifters.segments);
  const segmentEndsBuffer = resources.createBuffer('segment-ends', drifters.segmentEndVertices);
  const segmentStartTimesBuffer = resources.createBuffer(
    'segment-start-times',
    drifters.segmentStartTimes
  );
  const segmentEndTimesBuffer = resources.createBuffer(
    'segment-end-times',
    drifters.segmentEndTimes
  );

  // ---- Playhead graph (+ SST at the playhead) --------------------------------------------------------
  const currentPositions = resources.createBuffer('current-positions', pieceCount * 8);
  const headings = resources.createBuffer('headings', pieceCount * 4);
  const speeds = resources.createBuffer('speeds', pieceCount * 4);
  const status = resources.createBuffer('status', pieceCount * 4);
  const segmentRows = resources.createBuffer('segment-rows', pieceCount * 4);
  const segmentFractions = resources.createBuffer('segment-fractions', pieceCount * 4);
  const headSst = resources.createBuffer('head-sst', pieceCount * 4);
  const activeIds = resources.createBuffer('active-ids', pieceCount * 4);
  const activeCount = resources.createBuffer('active-count', 4);
  const activeOverflow = resources.createBuffer('active-overflow', 4);
  const playheadParameters = resources.createParameterBuffer(
    'playhead',
    'float32',
    GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
  );
  const markerDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'drifter-marker-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'drifter-playhead'});
  {
    const view = createViewer(playheadGraph);
    const rows = view('segment-rows', segmentRows, 'uint32', pieceCount);
    const fractions = view('segment-fractions', segmentFractions, 'float32', pieceCount);
    playheadGraph.add(
      new GPUTrajectoryPlayhead({
        id: 'playhead',
        positions: view('positions', positionsBuffer, 'float32x2', vertexCount),
        timestamps: view('timestamps', timestampsBuffer, 'float32', vertexCount),
        trackOffsets: view('offsets', offsetsBuffer, 'uint32', pieceCount + 1),
        parameters: playheadParameters.importToGraph(playheadGraph),
        currentPositions: view('current-positions', currentPositions, 'float32x2', pieceCount),
        headings: view('headings', headings, 'float32', pieceCount),
        speeds: view('speeds', speeds, 'float32', pieceCount),
        status: view('status', status, 'uint32', pieceCount),
        segmentRows: rows,
        segmentFractions: fractions,
        activeTracks: {
          ids: view('active-ids', activeIds, 'uint32', pieceCount),
          count: view('active-count', activeCount, 'uint32', 1),
          overflow: view('active-overflow', activeOverflow, 'uint32', 1)
        },
        drawInstanceCount: playheadGraph.importGPUData(
          'marker-draw-count',
          markerDraw.getInstanceCountData(0)
        )
      })
    );
    // Temperature at the playhead: interpolate the two fixes of the bracketing segment.
    addKernelPass(playheadGraph, {
      id: 'drifter-head-sst',
      invocationCount: pieceCount,
      declarations: /* wgsl */ `fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }`,
      bindings: [
        {name: 'rows', view: rows, type: 'u32', access: 'read'},
        {name: 'fractions', view: fractions, type: 'f32', access: 'read'},
        {
          name: 'sst',
          view: view('sst', sstBuffer, 'float32', vertexCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'headSst',
          view: view('head-sst', headSst, 'float32', pieceCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let row = rows[rowsOffset + index];
  var value = getQuietNaN();
  if (row != ${NO_ROW}u && row + 1u < ${vertexCount}u) {
    value = mix(sst[sstOffset + row], sst[sstOffset + row + 1u], fractions[fractionsOffset + index]);
  }
  headSst[headSstOffset + index] = value;`
    });
  }
  const playheadCompiled = resources.track(playheadGraph.compile());

  // ---- Trail (time window) graph ----------------------------------------------------------------------
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
      id: 'drifter-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'drifter-trails'});
  {
    const view = createViewer(trailGraph);
    trailGraph.add(
      new GPUTimeWindowFilter({
        id: 'trail-window',
        timestamps: view('segment-start-times', segmentStartTimesBuffer, 'float32', segmentCount),
        endTimestamps: view('segment-end-times', segmentEndTimesBuffer, 'float32', segmentCount),
        window: windowParameters.importToGraph(trailGraph),
        output: {
          ids: view('trail-ids', trailIds, 'uint32', segmentCount),
          count: view('trail-count', trailCount, 'uint32', 1),
          overflow: view('trail-overflow', trailOverflow, 'uint32', 1)
        },
        fadeWeights: view('fade-weights', fadeWeights, 'float32', segmentCount),
        clipFractions: view('clip-fractions', clipFractions, 'float32x2', segmentCount),
        drawInstanceCount: trailGraph.importGPUData(
          'trail-draw-count',
          trailDraw.getInstanceCountData(0)
        )
      })
    );
  }
  const trailCompiled = resources.track(trailGraph.compile());

  // ---- Density grids (one compiled graph per cell size) ------------------------------------------------
  const densityGrids = new Map<string, DensityGrid>();
  for (const cell of DENSITY_CELLS) {
    const columns = Math.round(360 / cell);
    const rows = Math.round(160 / cell);
    const cells = columns * rows;
    const lengths = resources.createBuffer(`lengths-${cell}`, cells * 4);
    const densities = resources.createBuffer(`densities-${cell}`, cells * 4);
    const overflow = resources.createBuffer(`overflow-${cell}`, 4);
    const parameters = resources.createParameterBuffer(
      `density-parameters-${cell}`,
      'float32',
      GPU_LINE_DENSITY_PARAMETER_LENGTH,
      getGPULineDensityParameterValues({
        minX: -180,
        minY: -80,
        cellWidth: cell,
        cellHeight: cell
      })
    );
    const graph = new GPUCommandGraph<void>(device, {id: `drifter-density-${cell}`});
    const view = createViewer(graph);
    graph.add(
      new GPULineDensity({
        id: `density-${cell}`,
        positions: view('positions', positionsBuffer, 'float32x2', vertexCount),
        pathOffsets: view('offsets', offsetsBuffer, 'uint32', pieceCount + 1),
        columns,
        rows,
        coordinateSystem: 'spherical',
        parameters: parameters.importToGraph(graph),
        output: {
          lengths: view('lengths', lengths, 'float32', cells),
          densities: view('densities', densities, 'float32', cells),
          overflow: view('overflow', overflow, 'uint32', 1)
        }
      })
    );
    densityGrids.set(String(cell), {
      cell,
      columns,
      rows,
      compiled: resources.track(graph.compile()),
      densities
    });
  }
  const densityPending = new Set<string>(densityGrids.keys());
  const defaultGrid = densityGrids.get('0.5');
  const getDensityGrid = (key: string): DensityGrid => {
    const grid = densityGrids.get(key) ?? defaultGrid;
    if (!grid) throw new Error('No density grid was compiled');
    return grid;
  };

  // ---- Readback and charts -------------------------------------------------------------------------------
  const statusReader = new SummaryReader(
    resources,
    'playback-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: trailCount, size: 4},
      {buffer: activeOverflow, size: 4}
    ],
    bytes => {
      const words = new Uint32Array(bytes);
      ctx.setReadout('active', words[0]);
      ctx.setReadout('trailSegments', words[1]);
      ctx.setReadout('overflow', words[2] ? 'active list overflowed' : 'none');
    }
  );

  // Drifters active per day of 2017 (a piece is active between its first and last fix).
  const dayCount = Math.ceil(lastTime) + 1;
  const activePerDay = new Float64Array(dayCount);
  for (let piece = 0; piece < pieceCount; piece++) {
    const first = Math.max(0, Math.ceil(drifters.timestamps[drifters.offsets[piece]]));
    const last = Math.min(
      dayCount - 1,
      Math.floor(drifters.timestamps[drifters.offsets[piece + 1] - 1])
    );
    for (let day = first; day <= last; day++) activePerDay[day]++;
  }
  let peakActive = 0;
  for (const value of activePerDay) peakActive = Math.max(peakActive, value);
  const days = Array.from({length: dayCount}, (_, day) => day);
  let sstMin = Number.POSITIVE_INFINITY;
  let sstMax = Number.NEGATIVE_INFINITY;
  let sstSum = 0;
  let sstCount = 0;
  for (const value of drifters.sst) {
    if (!Number.isFinite(value)) continue;
    sstMin = Math.min(sstMin, value);
    sstMax = Math.max(sstMax, value);
    sstSum += value;
    sstCount++;
  }
  const sstBins = binValues(drifters.sst, -2, 34, 36);
  const releasesByWeek = new Float64Array(53);
  for (const releaseDay of drifters.releaseDays) {
    if (releaseDay >= 0 && releaseDay <= 365)
      releasesByWeek[Math.min(52, Math.floor(releaseDay / 7))]++;
  }
  let lastChartSeconds = -Infinity;
  let lastChartDay = -1;
  let statusStale = true;

  const publishCharts = (playhead: number) => {
    ctx.setChart('activeChart', {
      kind: 'line',
      height: 110,
      series: [{label: 'drifters reporting', x: days, y: activePerDay, area: true}],
      xLabel: 'day of 2017',
      yLabel: 'drifters',
      markers: [{x: playhead, label: formatDriftDate(drifters.timeOriginMs, playhead).slice(0, 6)}],
      description: 'Number of drifter records with a position on each day of 2017.'
    });
  };
  ctx.setChart(
    'sstChart',
    histogramChart(sstBins, -2, 34, {
      xLabel: 'sea-surface temperature at the fixes (deg C)',
      yLabel: 'fixes',
      height: 110
    })
  );
  ctx.setChart('releaseChart', {
    kind: 'bars',
    values: releasesByWeek,
    xLabel: 'week of 2017',
    yLabel: 'first records',
    height: 92,
    description:
      'First retained daily position of each drifter by week. Uneven bars show the archive’s release and record effort.'
  });
  ctx.setReadout(
    'tracks',
    `${drifters.releaseCount.toLocaleString('en-US')} drifters in ${pieceCount.toLocaleString('en-US')} pieces`
  );
  ctx.setReadout('vertices', vertexCount.toLocaleString('en-US'));
  ctx.setReadout(
    'sstRange',
    `${sstMin.toFixed(1)} to ${sstMax.toFixed(1)} deg C (mean ${(sstSum / sstCount).toFixed(1)})`
  );
  ctx.setReadout('peakActive', peakActive);
  ctx.setReadout('density', 'computing');
  ctx.setStatus('');

  const regionOf = (id: string) =>
    OCEAN_REGIONS.find(region => region.id === id) ?? OCEAN_REGIONS[0];

  // Density graphs are static: encode all three once, on the first frames.
  const encodeDensity = (
    commandEncoder: Parameters<SceneInstance<OceanDriftersPlaybackOptions>['encode']>[0]
  ) => {
    for (const key of [...densityPending]) {
      densityGrids.get(key)?.compiled.encode(commandEncoder, {parameters: undefined});
      densityPending.delete(key);
    }
    if (densityPending.size === 0) {
      ctx.setReadout('density', 'three grids computed (1, 0.5 and 0.25 degree cells)');
    }
  };

  return {
    getCompiledGraphs: () => [
      playheadCompiled,
      trailCompiled,
      getDensityGrid(ctx.options.densityCell).compiled
    ],

    setOption(id, _value, state) {
      if (id === 'region') {
        ctx.flyTo(regionOf(state.region).view, {transitionMs: 1400});
      }
      statusStale = true;
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      const playhead = clock.advance(frame);
      encodeDensity(commandEncoder);
      playheadParameters.write(
        getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: options.maxGapDays})
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
      if (
        frame.timeSeconds - lastChartSeconds > CHART_INTERVAL_SECONDS &&
        Math.round(playhead) !== lastChartDay
      ) {
        lastChartSeconds = frame.timeSeconds;
        lastChartDay = Math.round(playhead);
        publishCharts(playhead);
        ctx.setReadout('date', formatDriftDate(drifters.timeOriginMs, playhead));
      }
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const ageDays = AGE_RAMP_DAYS;
      const colorProps = (() => {
        switch (options.colorBy) {
          case 'age':
            return {
              values: ageBuffer,
              valueFormat: 'float32' as const,
              valueIndices: segmentEndsBuffer,
              valueRange: [0, ageDays] as const,
              colormap: options.ramp
            };
          case 'date':
            return {
              values: timestampsBuffer,
              valueFormat: 'float32' as const,
              valueIndices: segmentEndsBuffer,
              valueRange: [0, Math.ceil(lastTime)] as const,
              colormap: options.ramp
            };
          default:
            return {
              values: sstBuffer,
              valueFormat: 'float32' as const,
              valueIndices: segmentEndsBuffer,
              valueRange: options.sstRange,
              colormap: options.ramp,
              noDataColor: [140, 146, 160, 120] as const
            };
        }
      })();
      const grid = getDensityGrid(options.densityCell);
      if (options.background === 'density') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `drifter-density-${grid.cell}`,
            ...drawProps,
            gridSize: [grid.columns, grid.rows],
            bounds: [-180, -80, 180, 80],
            tessellation: 64,
            values: grid.densities,
            valueFormat: 'float32',
            colormap: 'inferno',
            // Track length per area is 1/m; times 1e6 is km of track per 1,000 km2.
            valueScale: 1e6,
            valueRange: [0, options.densityMax],
            sqrtScale: true,
            discardAtOrBelow: 0,
            opacity: options.backgroundOpacity
          })
        );
      }
      // Release rings make the sampling design visible in the static opening and effort views.
      if (!options.play && (options.time <= 60 || options.background === 'density')) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'drifter-release-rings',
            ...drawProps,
            positions: releaseBuffer,
            instanceCount: drifters.releaseCount,
            radiusPixels: 4.5,
            color: dark ? [220, 232, 255, 110] : [20, 42, 78, 110]
          }),
          new SpatialAnalysisPointLayer({
            id: 'drifter-release-centres',
            ...drawProps,
            positions: releaseBuffer,
            instanceCount: drifters.releaseCount,
            radiusPixels: 1.4,
            color: dark ? [220, 232, 255, 210] : [20, 42, 78, 210]
          })
        );
      }
      if (options.showBackdrop) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'drifter-backdrop',
            ...drawProps,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 1,
            opacity: dark ? 0.16 : 0.2,
            ...colorProps
          })
        );
      }
      if (options.showTrails) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'drifter-trails',
            ...drawProps,
            segments: segmentsBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            widthPixels: 2.2,
            ...colorProps
          })
        );
      }
      if (options.showDots) {
        const bySst = options.colorBy === 'sst';
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'drifter-dot-halo',
            ...drawProps,
            positions: currentPositions,
            ids: activeIds,
            drawCommands: markerDraw,
            color: dark ? [255, 255, 255, 110] : [15, 20, 35, 120],
            radiusPixels: options.dotSize + 1.6
          }),
          new SpatialAnalysisPointLayer({
            id: 'drifter-dots',
            ...drawProps,
            positions: currentPositions,
            ids: activeIds,
            drawCommands: markerDraw,
            radiusPixels: options.dotSize,
            ...(bySst
              ? {
                  values: headSst,
                  valueFormat: 'float32' as const,
                  valueRange: options.sstRange,
                  colormap: options.ramp,
                  noDataColor: [140, 146, 160, 255] as const
                }
              : {color: dark ? [255, 255, 255, 255] : [20, 24, 36, 255]})
          })
        );
      }
      return layers;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    destroy() {
      statusReader.stop();
      resources.destroy();
    }
  };
}
