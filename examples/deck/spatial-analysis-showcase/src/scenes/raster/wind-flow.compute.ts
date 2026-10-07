// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPULineIntegralConvolutionParameterValues,
  getGPULineIntegralConvolutionWordParameterValues,
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionWordParameterValues,
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesWordParameterValues,
  GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH,
  GPU_LINE_INTEGRAL_CONVOLUTION_WORD_PARAMETER_LENGTH,
  GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH,
  GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH,
  GPU_STREAMLINES_PARAMETER_LENGTH,
  GPU_STREAMLINES_WORD_PARAMETER_LENGTH,
  GPULineIntegralConvolution,
  GPUParticleAdvection,
  GPUStreamlines
} from '@luma.gl/experimental/gpu-raster';
import {fetchBytes, getDataFileUrl} from '../../data/loaders';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphViewer, formatFixed} from './b16-common';
import {WIND_RAMP, WindPathLayer, WindTextureLayer, WindTrailLayer} from './b16-flow-layers';

/** Option state of the wind-flow scene. */
export type WindFlowOptions = {
  level: '10m' | '250hPa';
  hour: number;
  play: boolean;
  playSpeed: number;
  speedMax: number;
  showParticles: boolean;
  particleCount: '10000' | '30000' | '60000';
  trailLength: '8' | '16' | '32';
  speedScale: number;
  dropRate: number;
  maximumAge: number;
  minimumSpeed: number;
  spawnInView: boolean;
  zoomAdaptive: boolean;
  seed: number;
  trailWidth: number;
  particleOpacity: number;
  showLic: boolean;
  licSteps: '16' | '32' | '64';
  licStep: number;
  licMinimumSpeed: number;
  animateLic: boolean;
  licPeriod: number;
  licSeed: number;
  licContrast: number;
  licOpacity: number;
  showStreamlines: boolean;
  spacing: number;
  streamMinimumPoints: number;
  streamSeed: number;
  streamMinimumSpeed: number;
  streamWidth: number;
};

const EARTH_RADIUS = 6371008.8;
const FIELD_WIDTH = 181;
const FIELD_HEIGHT = 153;
const FIELD_CELLS = FIELD_WIDTH * FIELD_HEIGHT;
/** Cell-center lon/lat of cell (0, 0) is (-105, 12); the extent origin is half a cell lower. */
const FIELD_EXTENT: readonly [number, number, number, number] = [-105.125, 11.875, 0.25, 0.25];
const LIC_WIDTH = 896;
const LIC_HEIGHT = 704;
/** Streamline occupancy grid, seed lattice and capacities (compile-time). */
const STREAM_GRID_WIDTH = 128;
const STREAM_GRID_HEIGHT = 112;
const STREAM_SEED_COLUMNS = 192;
const STREAM_SEED_ROWS = 160;
const STREAM_STEPS = 48;
const STREAM_ROUNDS = 16;
const STREAM_LINE_CAPACITY = 16384;
const STREAM_POINT_CAPACITY = 262144;
const TIME_UPDATE_SECONDS = 0.1;
const STREAM_REFRESH_SECONDS = 0.18;
/** Local-time label: forecast run 2024-09-26 12Z. */
const RUN_START_UTC = Date.UTC(2024, 8, 26, 12);
const LIC_CYCLES_PER_SECOND = 1.2;
const DEGREES_PER_METER = 180 / (Math.PI * EARTH_RADIUS);

type FieldFrames = {
  /** Interleaved `(u, v)` in m/s, south row first. */
  frames: Float32Array[];
  /** Forecast hour of each frame. */
  hours: number[];
};

type WindManifest = {
  properties: {
    wind10m: {
      files: string[];
      forecastHours: number[];
      uRange: [number, number];
      vRange: [number, number];
      maxSpeedMs: number;
      maxSpeedFrame: number;
      maxSpeedLonLat: [number, number];
    };
    jet250hPa: {
      files: string[];
      forecastHours: number[];
      uRange: [number, number];
      vRange: [number, number];
      maxSpeedMs: number;
    };
  };
};

async function decodeWindFrame(
  bytes: ArrayBuffer,
  uRange: readonly [number, number],
  vRange: readonly [number, number]
): Promise<Float32Array> {
  const bitmap = await createImageBitmap(new Blob([bytes], {type: 'image/png'}), {
    premultiplyAlpha: 'none',
    colorSpaceConversion: 'none'
  });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext('2d', {willReadFrequently: true});
  if (!context) throw new Error('2D canvas unavailable for wind decoding');
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const {width, height} = canvas;
  const pixels = context.getImageData(0, 0, width, height).data;
  const field = new Float32Array(width * height * 2);
  for (let row = 0; row < height; row++) {
    // The PNG rows run north to south; the contributors want the smallest latitude first.
    const targetRow = height - 1 - row;
    for (let column = 0; column < width; column++) {
      const source = (row * width + column) * 4;
      const target = (targetRow * width + column) * 2;
      field[target] = uRange[0] + (pixels[source] / 255) * (uRange[1] - uRange[0]);
      field[target + 1] = vRange[0] + (pixels[source + 1] / 255) * (vRange[1] - vRange[0]);
    }
  }
  return field;
}

async function loadFrames(
  files: readonly string[],
  hours: number[],
  uRange: [number, number],
  vRange: [number, number],
  signal: AbortSignal
): Promise<FieldFrames> {
  const frames = await Promise.all(
    files.map(async file =>
      decodeWindFrame(await fetchBytes(getDataFileUrl('gfs-wind', file), signal), uRange, vRange)
    )
  );
  return {frames, hours};
}

type ParticleSet = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  positions: ReturnType<SpatialAnalysisResources['createBuffer']>;
  trails: ReturnType<SpatialAnalysisResources['createBuffer']>;
  count: number;
  trailLength: number;
};

type LicSet = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  values: ReturnType<SpatialAnalysisResources['createBuffer']>;
  stepCount: number;
};

/**
 * Hurricane Helene in GFS winds. One velocity buffer (u and v converted to degrees per second so
 * the contributors integrate in longitude and latitude) feeds three drawings: advected particles
 * with data-space trails, an animated line-integral-convolution texture that follows the camera,
 * and evenly spaced streamlines whose occupancy grid also follows the camera. Time, speed scale,
 * drop rate, age, spacing, seeds and phase are parameter writes. The particle count, trail
 * length and LIC tap count are compile-time and rebuild their own graph.
 */
export async function createWindFlow(
  ctx: SceneContext<WindFlowOptions>
): Promise<SceneInstance<WindFlowOptions>> {
  const dataset = ctx.datasets.get('gfs-wind');
  const {device} = ctx;
  ctx.setStatus('Decoding 32 GFS wind frames');
  const manifest = dataset.manifest as unknown as WindManifest;
  const properties = manifest.properties;
  const [surface, jet] = await Promise.all([
    loadFrames(
      properties.wind10m.files,
      properties.wind10m.forecastHours,
      properties.wind10m.uRange,
      properties.wind10m.vRange,
      ctx.signal
    ),
    loadFrames(
      properties.jet250hPa.files,
      properties.jet250hPa.forecastHours,
      properties.jet250hPa.uRange,
      properties.jet250hPa.vRange,
      ctx.signal
    )
  ]);
  ctx.signal.throwIfAborted();
  const levels = {'10m': surface, '250hPa': jet};
  const resources = new SpatialAnalysisResources(device, 'wind');

  // Degrees per second conversion factors by row: dlon = u / (R cos(lat)), dlat = v / R.
  const longitudeFactor = new Float32Array(FIELD_HEIGHT);
  for (let row = 0; row < FIELD_HEIGHT; row++) {
    const latitude = 12 + row * 0.25;
    longitudeFactor[row] = DEGREES_PER_METER / Math.cos((latitude * Math.PI) / 180);
  }

  // --- Shared buffers ----------------------------------------------------------------------------
  const velocityBuffer = resources.createBuffer('velocity', FIELD_CELLS * 8);
  const speedBuffer = resources.createBuffer('speed', FIELD_CELLS * 4);
  const velocities = new Float32Array(FIELD_CELLS * 2);
  const speeds = new Float32Array(FIELD_CELLS);
  /** Current field in m/s (interleaved u, v), kept for tooltips and the peak readout. */
  const currentMs = new Float32Array(FIELD_CELLS * 2);

  const particleParameters = resources.createParameterBuffer(
    'particle-parameters',
    'float32',
    GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH
  );
  const particleWords = resources.createParameterBuffer(
    'particle-words',
    'uint32',
    GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH
  );
  const licParameters = resources.createParameterBuffer(
    'lic-parameters',
    'float32',
    GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH
  );
  const licWords = resources.createParameterBuffer(
    'lic-words',
    'uint32',
    GPU_LINE_INTEGRAL_CONVOLUTION_WORD_PARAMETER_LENGTH
  );
  const streamParameters = resources.createParameterBuffer(
    'stream-parameters',
    'float32',
    GPU_STREAMLINES_PARAMETER_LENGTH
  );
  const streamWords = resources.createParameterBuffer(
    'stream-words',
    'uint32',
    GPU_STREAMLINES_WORD_PARAMETER_LENGTH
  );

  // --- Streamlines (compile-time grid and seed lattice) -------------------------------------------
  const streamIds = resources.createBuffer('stream-ids', STREAM_LINE_CAPACITY * 4);
  const streamCount = resources.createBuffer('stream-count', 4);
  const streamOverflow = resources.createBuffer('stream-overflow', 4);
  const streamOffsets = resources.createBuffer('stream-offsets', (STREAM_LINE_CAPACITY + 1) * 4);
  const streamPoints = resources.createBuffer('stream-points', STREAM_POINT_CAPACITY * 8);
  const streamPointCount = resources.createBuffer('stream-point-count', 4);
  const streamUnconverged = resources.createBuffer('stream-unconverged', 4);
  const streamDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'wind-stream-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const summaryRing = resources.track(
    new GPUReadbackRing(device, {id: 'wind-stream-summary', byteLength: 16})
  );
  const streamGraph = new GPUCommandGraph<void>(device, {id: 'wind-streamlines'});
  {
    const view = createGraphViewer(streamGraph);
    streamGraph.add(
      new GPUStreamlines({
        id: 'streamlines',
        velocities: view('velocity', velocityBuffer, 'float32x2', FIELD_CELLS),
        fieldWidth: FIELD_WIDTH,
        fieldHeight: FIELD_HEIGHT,
        gridWidth: STREAM_GRID_WIDTH,
        gridHeight: STREAM_GRID_HEIGHT,
        seedColumns: STREAM_SEED_COLUMNS,
        seedRows: STREAM_SEED_ROWS,
        stepsPerDirection: STREAM_STEPS,
        roundCount: STREAM_ROUNDS,
        parameters: streamParameters.importToGraph(streamGraph),
        wordParameters: streamWords.importToGraph(streamGraph),
        output: {
          lines: {
            ids: view('ids', streamIds, 'uint32', STREAM_LINE_CAPACITY),
            count: view('count', streamCount, 'uint32', 1),
            overflow: view('overflow', streamOverflow, 'uint32', 1)
          },
          pathOffsets: view('offsets', streamOffsets, 'uint32', STREAM_LINE_CAPACITY + 1),
          points: view('points', streamPoints, 'float32x2', STREAM_POINT_CAPACITY),
          pointCount: view('point-count', streamPointCount, 'uint32', 1),
          unconverged: view('unconverged', streamUnconverged, 'uint32', 1)
        }
      })
    );
  }
  const compiledStreamlines = resources.track(streamGraph.compile());

  // --- Rebuildable graphs ------------------------------------------------------------------------
  let setCounter = 0;
  const buildParticles = (count: number, trailLength: number): ParticleSet => {
    const scope = new SpatialAnalysisResources(device, `wind-particles-${setCounter++}`);
    const positions = scope.createBuffer('positions', count * 8);
    const ages = scope.createBuffer('ages', count * 4);
    const generations = scope.createBuffer('generations', count * 4);
    const speedsOut = scope.createBuffer('speeds', count * 4);
    const trails = scope.createBuffer('trails', count * trailLength * 8);
    const graph = new GPUCommandGraph<void>(device, {id: `wind-particles-${count}-${trailLength}`});
    const view = createGraphViewer(graph);
    graph.add(
      new GPUParticleAdvection({
        id: 'particles',
        velocities: view('velocity', velocityBuffer, 'float32x2', FIELD_CELLS),
        fieldWidth: FIELD_WIDTH,
        fieldHeight: FIELD_HEIGHT,
        parameters: particleParameters.importToGraph(graph),
        wordParameters: particleWords.importToGraph(graph),
        state: {
          positions: view('positions', positions, 'float32x2', count),
          ages: view('ages', ages, 'uint32', count),
          generations: view('generations', generations, 'uint32', count)
        },
        speeds: view('speeds', speedsOut, 'float32', count),
        trails: {
          positions: view('trails', trails, 'float32x2', count * trailLength),
          length: trailLength
        }
      })
    );
    return {
      resources: scope,
      compiled: scope.track(graph.compile()),
      positions,
      trails,
      count,
      trailLength
    };
  };
  const buildLic = (stepCount: number): LicSet => {
    const scope = new SpatialAnalysisResources(device, `wind-lic-${setCounter++}`);
    const values = scope.createBuffer('values', LIC_WIDTH * LIC_HEIGHT * 4);
    const graph = new GPUCommandGraph<void>(device, {id: `wind-lic-${stepCount}`});
    const view = createGraphViewer(graph);
    graph.add(
      new GPULineIntegralConvolution({
        id: 'lic',
        velocities: view('velocity', velocityBuffer, 'float32x2', FIELD_CELLS),
        fieldWidth: FIELD_WIDTH,
        fieldHeight: FIELD_HEIGHT,
        width: LIC_WIDTH,
        height: LIC_HEIGHT,
        stepCount,
        parameters: licParameters.importToGraph(graph),
        wordParameters: licWords.importToGraph(graph),
        output: {values: view('values', values, 'float32', LIC_WIDTH * LIC_HEIGHT)}
      })
    );
    return {resources: scope, compiled: scope.track(graph.compile()), values, stepCount};
  };

  let particles = buildParticles(
    Number(ctx.options.particleCount),
    Number(ctx.options.trailLength)
  );
  let lic = buildLic(Number(ctx.options.licSteps));

  // --- State -----------------------------------------------------------------------------------
  let destroyed = false;
  let hour = ctx.options.hour;
  let lastFieldHour = Number.NaN;
  let lastFieldLevel = '';
  let lastFieldTime = -Infinity;
  let particleFrame = 0;
  let resetParticles = true;
  let licDirty = true;
  let streamDirty = true;
  let lastStreamTime = -Infinity;
  let lastCameraKey = '';
  let summaryPending = false;
  let measuring = false;
  let appliedSeed = ctx.options.seed;
  let appliedHourOption = ctx.options.hour;
  let appliedLicSettings = '';
  let appliedStreamSettings = '';
  let streamCameraKey = '';
  /** `[west, south, east, north]` the LIC raster was last computed for; the layer reads it live. */
  const licBounds: [number, number, number, number] = [-105, 12, -60, 50];

  /** Blends the two bracketing frames at the forecast hour and uploads degrees per second. */
  const updateField = (state: WindFlowOptions) => {
    const {frames, hours} = levels[state.level];
    const last = frames.length - 1;
    const clamped = Math.min(Math.max(hour, hours[0]), hours[last]);
    let upper = hours.findIndex(value => value >= clamped);
    if (upper < 0) upper = last;
    const lower = Math.max(0, upper - (hours[upper] > clamped ? 1 : 0));
    const span = hours[upper] - hours[lower];
    const weight = span > 0 ? (clamped - hours[lower]) / span : 0;
    const a = frames[lower];
    const b = frames[upper];
    for (let row = 0; row < FIELD_HEIGHT; row++) {
      const factor = longitudeFactor[row];
      for (let column = 0; column < FIELD_WIDTH; column++) {
        const cell = row * FIELD_WIDTH + column;
        const u = a[cell * 2] + (b[cell * 2] - a[cell * 2]) * weight;
        const v = a[cell * 2 + 1] + (b[cell * 2 + 1] - a[cell * 2 + 1]) * weight;
        currentMs[cell * 2] = u;
        currentMs[cell * 2 + 1] = v;
        velocities[cell * 2] = u * factor;
        velocities[cell * 2 + 1] = v * DEGREES_PER_METER;
        speeds[cell] = Math.hypot(u, v);
      }
    }
    velocityBuffer.write(velocities);
    speedBuffer.write(speeds);
    lastFieldHour = hour;
    lastFieldLevel = state.level;
    licDirty = true;
    streamDirty = true;
    publishField(state);
  };

  const publishField = (state: WindFlowOptions) => {
    const time = new Date(RUN_START_UTC + hour * 3600_000);
    const label = `${time.toISOString().slice(0, 16).replace('T', ' ')} UTC (forecast hour ${hour.toFixed(1)})`;
    ctx.setReadout('time', label);
    let peak = 0;
    let peakCell = 0;
    for (let cell = 0; cell < FIELD_CELLS; cell++) {
      if (speeds[cell] > peak) {
        peak = speeds[cell];
        peakCell = cell;
      }
    }
    const column = peakCell % FIELD_WIDTH;
    const row = Math.floor(peakCell / FIELD_WIDTH);
    const lon = -105 + column * 0.25;
    const lat = 12 + row * 0.25;
    const manifestPeak = properties.wind10m.maxSpeedMs;
    const nearManifest =
      state.level === '10m' && Math.abs(hour - properties.wind10m.maxSpeedFrame) < 0.26;
    ctx.setReadout(
      'peak',
      `${formatFixed(peak, 1)} m/s at ${lat.toFixed(2)}°N ${Math.abs(lon).toFixed(2)}°W${
        nearManifest
          ? ` · GFS peak in the file: ${formatFixed(manifestPeak, 1)} m/s at ${properties.wind10m.maxSpeedLonLat[1]}°N ${Math.abs(properties.wind10m.maxSpeedLonLat[0])}°W`
          : ''
      }`
    );
  };

  const getWindowBounds = (viewport: {getBounds: () => number[]}) => {
    const [west, south, east, north] = viewport.getBounds();
    return {west, south, east, north};
  };

  const writeLic = (
    state: WindFlowOptions,
    bounds: ReturnType<typeof getWindowBounds>,
    phase: number
  ) => {
    licParameters.write(
      getGPULineIntegralConvolutionParameterValues({
        fieldExtent: FIELD_EXTENT,
        outputExtent: [
          bounds.west,
          bounds.south,
          (bounds.east - bounds.west) / LIC_WIDTH,
          (bounds.north - bounds.south) / LIC_HEIGHT
        ],
        stepLength: state.licStep,
        // Degrees per second: one meter per second is DEGREES_PER_METER at the equator.
        minimumSpeed: state.licMinimumSpeed * DEGREES_PER_METER,
        phase,
        period: state.licPeriod
      })
    );
    licWords.write(getGPULineIntegralConvolutionWordParameterValues({seed: state.licSeed}));
    licBounds[0] = bounds.west;
    licBounds[1] = bounds.south;
    licBounds[2] = bounds.east;
    licBounds[3] = bounds.north;
  };

  const writeStreamlines = (
    state: WindFlowOptions,
    camera: {longitude: number; latitude: number; zoom: number}
  ) => {
    const degreesPerPixel = 360 / (512 * 2 ** camera.zoom);
    const cellX = state.spacing * degreesPerPixel;
    const cellY = cellX * Math.cos((camera.latitude * Math.PI) / 180);
    streamParameters.write(
      getGPUStreamlinesParameterValues({
        fieldExtent: FIELD_EXTENT,
        gridExtent: [
          camera.longitude - (STREAM_GRID_WIDTH * cellX) / 2,
          camera.latitude - (STREAM_GRID_HEIGHT * cellY) / 2,
          cellX,
          cellY
        ],
        stepLength: 0.4 * cellX,
        minimumSpeed: state.streamMinimumSpeed * DEGREES_PER_METER
      })
    );
    streamWords.write(
      getGPUStreamlinesWordParameterValues({
        seed: state.streamSeed,
        minimumPoints: state.streamMinimumPoints
      })
    );
  };

  const readStreamSummary = async (
    commandEncoder: Parameters<SceneInstance<WindFlowOptions>['encode']>[0]
  ) => {
    const ticket = summaryRing.tryAcquire();
    if (!ticket) return;
    [streamCount, streamPointCount, streamOverflow, streamUnconverged].forEach((buffer, index) => {
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: buffer,
        sourceOffset: 0,
        destinationBuffer: ticket.buffer,
        destinationOffset: index * 4,
        size: 4
      });
    });
    ticket.markEncoded({byteOffset: 0, byteLength: 16});
    summaryPending = true;
    try {
      const bytes = await ticket.read();
      if (destroyed) return;
      const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 4);
      ctx.setReadout(
        'streamlines',
        `${formatCount(words[0])} lines · ${formatCount(words[1])} points · ${
          words[2] ? 'capacity overflow' : 'no overflow'
        } · ${words[3] ? 'unconverged (undecided lines dropped)' : 'converged'}`
      );
    } catch {
      // The ring or device was destroyed while the read was in flight.
    } finally {
      summaryPending = false;
    }
  };

  const describeParticles = () => {
    ctx.setReadout(
      'particles',
      `${formatCount(particles.count)} · trail ring ${particles.trailLength} slots · ${(
        (particles.count * particles.trailLength * 8) / 1048576
      ).toFixed(1)} MB`
    );
  };
  describeParticles();
  ctx.setReadout(
    'field',
    `${FIELD_WIDTH} × ${FIELD_HEIGHT} cells, 0.25° (about 28 km), u and v as degrees per second`
  );
  ctx.setReadout(
    'licRaster',
    `${LIC_WIDTH} × ${LIC_HEIGHT} px follows the camera, ${lic.stepCount * 2} taps`
  );
  ctx.setReadout('streamlines', 'not encoded yet');
  ctx.setReadout('timing', 'press the button');
  ctx.setStatus('');

  async function measureAll(): Promise<void> {
    if (measuring || destroyed) return;
    measuring = true;
    try {
      const options = {parameters: undefined, completionBuffer: streamOverflow, signal: ctx.signal};
      const particleTiming = await measureCompiledGraph(device, particles.compiled, options);
      const licTiming = await measureCompiledGraph(device, lic.compiled, options);
      const streamTiming = await measureCompiledGraph(device, compiledStreamlines, {
        ...options,
        runs: 3
      });
      if (destroyed) return;
      ctx.setReadout(
        'timing',
        `advection ${particleTiming.milliseconds.toFixed(2)} ms · LIC ${licTiming.milliseconds.toFixed(2)} ms · streamlines ${streamTiming.milliseconds.toFixed(2)} ms`
      );
      streamDirty = true;
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
    }
  }

  // --- Wind lookup for tooltips -----------------------------------------------------------------
  const getWindAt = (longitude: number, latitude: number) => {
    const x = (longitude + 105) / 0.25;
    const y = (latitude - 12) / 0.25;
    if (x < 0 || y < 0 || x > FIELD_WIDTH - 1 || y > FIELD_HEIGHT - 1) return null;
    const column = Math.min(Math.floor(x), FIELD_WIDTH - 2);
    const row = Math.min(Math.floor(y), FIELD_HEIGHT - 2);
    const fx = x - column;
    const fy = y - row;
    const sample = (offset: number) => {
      const read = (c: number, r: number) => currentMs[(r * FIELD_WIDTH + c) * 2 + offset];
      return (
        read(column, row) * (1 - fx) * (1 - fy) +
        read(column + 1, row) * fx * (1 - fy) +
        read(column, row + 1) * (1 - fx) * fy +
        read(column + 1, row + 1) * fx * fy
      );
    };
    return {u: sample(0), v: sample(1)};
  };

  const compass = (degrees: number) =>
    [
      'N',
      'NNE',
      'NE',
      'ENE',
      'E',
      'ESE',
      'SE',
      'SSE',
      'S',
      'SSW',
      'SW',
      'WSW',
      'W',
      'WNW',
      'NW',
      'NNW'
    ][Math.round(degrees / 22.5) % 16];

  // --- Instance --------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [particles.compiled, lic.compiled, compiledStreamlines],

    setOption(id, _value, state) {
      if (id === 'hour' || id === 'level') {
        if (id === 'hour') {
          hour = state.hour;
          appliedHourOption = state.hour;
        }
        updateField(state);
        resetParticles = id === 'level' ? true : resetParticles;
      } else if (id === 'particleCount' || id === 'trailLength') {
        const next = buildParticles(Number(state.particleCount), Number(state.trailLength));
        particles.resources.destroy();
        particles = next;
        resetParticles = true;
        describeParticles();
      } else if (id === 'licSteps') {
        const next = buildLic(Number(state.licSteps));
        lic.resources.destroy();
        lic = next;
        licDirty = true;
        ctx.setReadout(
          'licRaster',
          `${LIC_WIDTH} × ${LIC_HEIGHT} px follows the camera, ${lic.stepCount * 2} taps`
        );
      } else if (id === 'seed') {
        resetParticles = true;
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'reset') resetParticles = true;
      else if (id === 'measure') void measureAll();
    },

    encode(commandEncoder, frameInfo) {
      const state = ctx.options;
      const viewport = frameInfo.viewport;
      const camera = viewport as unknown as {longitude: number; latitude: number; zoom: number};
      // Time: the hour option is the source of truth when it changes; playing advances it.
      if (state.hour !== appliedHourOption) {
        appliedHourOption = state.hour;
        hour = state.hour;
      }
      if (state.play) {
        const span = state.level === '10m' ? 23 : 21;
        hour = (hour + frameInfo.deltaSeconds * state.playSpeed) % (span + 0.001);
      }
      if (
        lastFieldLevel !== state.level ||
        (Math.abs(hour - lastFieldHour) > 0.0005 &&
          frameInfo.timeSeconds - lastFieldTime > TIME_UPDATE_SECONDS)
      ) {
        lastFieldTime = frameInfo.timeSeconds;
        updateField(state);
      }
      if (state.seed !== appliedSeed) {
        appliedSeed = state.seed;
        resetParticles = true;
      }

      // Particles.
      if (state.showParticles) {
        const zoomFactor = state.zoomAdaptive ? 2 ** (5 - camera.zoom) : 1;
        const bounds = getWindowBounds(viewport);
        const spawnBounds = state.spawnInView
          ? ([
              Math.max(bounds.west, FIELD_EXTENT[0]),
              Math.max(bounds.south, FIELD_EXTENT[1]),
              Math.min(bounds.east, FIELD_EXTENT[0] + FIELD_WIDTH * 0.25),
              Math.min(bounds.north, FIELD_EXTENT[1] + FIELD_HEIGHT * 0.25)
            ] as const)
          : undefined;
        particleParameters.write(
          getGPUParticleAdvectionParameterValues(
            {
              fieldExtent: FIELD_EXTENT,
              // Seconds of wind per frame: about 2 px for a 40 m/s wind at zoom 5, kept constant on screen.
              timeStep: 120 * zoomFactor,
              speedScale: state.speedScale,
              dropRate: state.dropRate,
              minimumSpeed: state.minimumSpeed * DEGREES_PER_METER * state.speedScale,
              spawnBounds
            },
            [FIELD_WIDTH, FIELD_HEIGHT]
          )
        );
        particleWords.write(
          getGPUParticleAdvectionWordParameterValues({
            seed: state.seed,
            frame: particleFrame,
            maximumAge: state.maximumAge,
            reset: resetParticles
          })
        );
        particles.compiled.encode(commandEncoder, {parameters: undefined});
        particleFrame++;
        resetParticles = false;
      }

      // LIC: every frame while animating (it also follows the camera), otherwise after a change.
      if (state.showLic) {
        const bounds = getWindowBounds(viewport);
        const settings = `${state.licStep}|${state.licMinimumSpeed}|${state.licPeriod}|${state.licSeed}`;
        const cameraKey = `${bounds.west.toFixed(4)},${bounds.south.toFixed(4)},${bounds.east.toFixed(4)},${bounds.north.toFixed(4)}`;
        if (
          state.animateLic ||
          licDirty ||
          settings !== appliedLicSettings ||
          cameraKey !== lastCameraKey
        ) {
          appliedLicSettings = settings;
          writeLic(
            state,
            bounds,
            state.animateLic ? -frameInfo.timeSeconds * LIC_CYCLES_PER_SECOND : 0
          );
          lic.compiled.encode(commandEncoder, {parameters: undefined});
          licDirty = false;
        }
        lastCameraKey = cameraKey;
      }

      // Streamlines: when the field, a setting or the camera changed (at most every 180 ms).
      if (state.showStreamlines) {
        const settings = `${state.spacing}|${state.streamMinimumPoints}|${state.streamSeed}|${state.streamMinimumSpeed}`;
        const key = `${camera.longitude.toFixed(3)},${camera.latitude.toFixed(3)},${camera.zoom.toFixed(2)}`;
        if (settings !== appliedStreamSettings) {
          appliedStreamSettings = settings;
          streamDirty = true;
        }
        if (
          (streamDirty || key !== streamCameraKey) &&
          frameInfo.timeSeconds - lastStreamTime > STREAM_REFRESH_SECONDS
        ) {
          streamCameraKey = key;
          lastStreamTime = frameInfo.timeSeconds;
          writeStreamlines(state, camera);
          compiledStreamlines.encode(commandEncoder, {parameters: undefined});
          // The published point count becomes the indirect instance count, with no readback.
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: streamPointCount,
            sourceOffset: 0,
            destinationBuffer: streamDraw.buffer,
            destinationOffset: 4,
            size: 4
          });
          streamDirty = false;
          if (!summaryPending) void readStreamSummary(commandEncoder);
        }
      }
    },

    getLayers() {
      const state = ctx.options;
      const layers: Layer[] = [];
      const common = {
        speedField: speedBuffer,
        field: {extent: FIELD_EXTENT, size: [FIELD_WIDTH, FIELD_HEIGHT] as const},
        ramp: WIND_RAMP,
        speedRange: [0, state.speedMax] as const
      };
      const dark = ctx.theme() === 'dark';
      if (state.showLic) {
        layers.push(
          new WindTextureLayer({
            ...common,
            id: 'wind-lic',
            values: lic.values,
            gridSize: [LIC_WIDTH, LIC_HEIGHT],
            bounds: licBounds,
            contrast: state.licContrast,
            opacity: state.licOpacity
          })
        );
      }
      if (state.showStreamlines) {
        layers.push(
          new WindPathLayer({
            ...common,
            id: 'wind-streamlines',
            points: streamPoints,
            pathOffsets: streamOffsets,
            lineCount: streamCount,
            pointCount: streamPointCount,
            pointCapacity: STREAM_POINT_CAPACITY,
            drawCommands: streamDraw,
            color: dark ? [255, 255, 255, 255] : [20, 30, 60, 255],
            widthPixels: state.streamWidth,
            opacity: 0.6
          })
        );
      }
      if (state.showParticles) {
        layers.push(
          new WindTrailLayer({
            ...common,
            id: 'wind-trails',
            trailPositions: particles.trails,
            wordParameters: particleWords.buffer,
            ringLength: particles.trailLength,
            particleCount: particles.count,
            widthPixels: state.trailWidth,
            opacity: state.particleOpacity
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const wind = getWindAt(event.coordinate[0], event.coordinate[1]);
      if (!wind) return null;
      const speed = Math.hypot(wind.u, wind.v);
      // Meteorological direction: where the wind blows from, degrees clockwise from north.
      const from = ((Math.atan2(-wind.u, -wind.v) * 180) / Math.PI + 360) % 360;
      return `${speed.toFixed(1)} m/s (${(speed * 1.944).toFixed(0)} kn) from the ${compass(from)} (${Math.round(from)}°)\nu ${formatFixed(wind.u, 1)}, v ${formatFixed(wind.v, 1)} m/s · ${ctx.options.level === '10m' ? '10 m above ground' : '250 hPa (about 10.5 km)'}`;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    destroy() {
      destroyed = true;
      particles.resources.destroy();
      lic.resources.destroy();
      resources.destroy();
    }
  };
}
