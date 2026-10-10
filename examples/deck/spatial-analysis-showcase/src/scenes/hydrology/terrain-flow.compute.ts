// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionWordParameterValues,
  GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH,
  GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH,
  GPUParticleAdvection
} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainFlowFieldParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH,
  GPUTerrainDerivatives,
  GPUTerrainFlowField
} from '@luma.gl/experimental/gpu-terrain';
import {GPUCommandGraph, GPUHistogram, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {getViewportMetricBounds, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {RampName} from '../../engine/ramps';
import type {SceneContext, SceneInstance} from '../scene';
import {FlowTrailLayer} from './b15-flow-layers';
import {createCanyonGrid, flipRows, formatInteger, getCellAt, NAN_DECLARATIONS} from './b15-common';

/** Option state of the terrain-flow scene. */
export type TerrainFlowOptions = {
  windSpeed: number;
  windBearing: number;
  exaggeration: number;
  background: 'relief' | 'speed';
  timeStep: number;
  speedScale: number;
  dropRate: number;
  maximumAge: number;
  minimumSpeed: number;
  spawnRegion: 'window' | 'view';
  seed: number;
  trailWidth: number;
  trailOpacity: number;
  ramp: Extract<RampName, 'isolum' | 'viridis' | 'magma' | 'inferno' | 'cividis'>;
  paused: boolean;
};

/** Compile-time particle count and trail ring length. */
export const PARTICLE_COUNT = 40000;
export const TRAIL_LENGTH = 24;
/** Histogram of the field speed as a share of the free-stream wind. */
const SPEED_BIN_COUNT = 20;
const SUN_AZIMUTH_DEGREES = 315;
const SUN_ALTITUDE_DEGREES = 40;

/**
 * Terrain-following wind over the Grand Canyon. `GPUTerrainFlowField` projects one uniform wind
 * onto the tangent plane of the DEM (one kernel); `GPUParticleAdvection` moves tens of thousands
 * of particles through it every frame and writes a trail ring that a layer draws straight from the
 * buffer. The wind, deflection and every particle setting are parameter writes: three graphs, none
 * recompiled. The raster is flipped to south-first because the particles live in the y-up meter
 * frame of the map layers.
 */
export async function createTerrainFlow(
  ctx: SceneContext<TerrainFlowOptions>
): Promise<SceneInstance<TerrainFlowOptions>> {
  const {device} = ctx;
  const grid = createCanyonGrid(ctx.datasets.get('grand-canyon-dem'), 1);
  const {width, height, cellCount} = grid;
  const origin: [number, number, number] = [grid.origin[0], grid.origin[1], 0];
  const resources = new SpatialAnalysisResources(device, 'terrain-flow');
  let destroyed = false;

  const elevationBuffer = resources.createBuffer(
    'elevation-south-first',
    flipRows(grid.elevation, width, height)
  );
  const hillshadeBuffer = resources.createBuffer('hillshade', cellCount * 4);
  const velocityBuffer = resources.createBuffer('velocity', cellCount * 8);
  const ratioBuffer = resources.createBuffer('speed-ratio', cellCount * 4);
  const ratioHistogramBuffer = resources.createBuffer('ratio-histogram', SPEED_BIN_COUNT * 4);
  const positionBuffer = resources.createBuffer('particle-positions', PARTICLE_COUNT * 8);
  const ageBuffer = resources.createBuffer('particle-ages', PARTICLE_COUNT * 4);
  const generationBuffer = resources.createBuffer('particle-generations', PARTICLE_COUNT * 4);
  const speedBuffer = resources.createBuffer('particle-speeds', PARTICLE_COUNT * 4);
  const trailBuffer = resources.createBuffer('particle-trails', PARTICLE_COUNT * TRAIL_LENGTH * 8);
  const derivativesSettings = resources.createParameterBuffer(
    'derivatives-settings',
    'float32',
    GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
  );
  const windSettings = resources.createParameterBuffer(
    'wind-settings',
    'float32',
    GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH
  );
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

  // The raster is south-first, so the edge of row 0 is the south edge.
  const flippedSettings = {
    cellSize: grid.cellSize,
    northEdge: grid.southEdge,
    southEdge: grid.northEdge
  };
  const elevationBand = (graph: GPUCommandGraph<void>) => ({
    id: 'elevation',
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', cellCount)
    }
  });

  // Hillshade backdrop, once.
  const setupGraph = new GPUCommandGraph<void>(device, {id: 'terrain-flow-setup'});
  setupGraph.add(
    new GPUTerrainDerivatives({
      id: 'derivatives',
      width,
      height,
      elevation: elevationBand(setupGraph),
      settings: derivativesSettings.importToGraph(setupGraph),
      hillshade: importGraphBuffer(setupGraph, 'hillshade', hillshadeBuffer, 'float32', cellCount),
      cellSizeMode: grid.cellSizeMode,
      rowDirection: 'north'
    })
  );
  const compiledSetup: CompiledGPUCommandGraph<void> = resources.track(setupGraph.compile());

  // Wind field graph: the deflected velocities, then their share of the free-stream speed.
  const windGraph = new GPUCommandGraph<void>(device, {id: 'terrain-flow-field'});
  const velocityView = importGraphBuffer(
    windGraph,
    'velocity',
    velocityBuffer,
    'float32x2',
    cellCount
  );
  const windSettingsView = windSettings.importToGraph(windGraph);
  windGraph.add(
    new GPUTerrainFlowField({
      id: 'wind-field',
      width,
      height,
      elevation: elevationBand(windGraph),
      settings: windSettingsView,
      cellSizeMode: grid.cellSizeMode,
      velocities: velocityView
    })
  );
  const ratioView = importGraphBuffer(windGraph, 'speed-ratio', ratioBuffer, 'float32', cellCount);
  addKernelPass(windGraph, {
    id: 'speed-ratio',
    bindings: [
      {name: 'velocity', view: velocityView, type: 'f32', access: 'read'},
      {name: 'settings', view: windSettingsView, type: 'f32', access: 'read'},
      {name: 'ratio', view: ratioView, type: 'f32', access: 'read_write'}
    ],
    invocationCount: cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let u = velocity[velocityOffset + 2u * index];
  let v = velocity[velocityOffset + 2u * index + 1u];
  let wind = max(length(vec2<f32>(settings[settingsOffset + 4u], settings[settingsOffset + 5u])), 0.000001);
  let share = length(vec2<f32>(u, v)) / wind;
  ratio[ratioOffset + index] = select(share, -1.0, isNaNValue(u) || isNaNValue(v));`
  });
  windGraph.add(
    new GPUHistogram({
      id: 'speed-histogram',
      input: ratioView,
      output: importGraphBuffer(
        windGraph,
        'ratio-histogram',
        ratioHistogramBuffer,
        'uint32',
        SPEED_BIN_COUNT
      ),
      domain: [0, 1.0001]
    })
  );
  const compiledWind: CompiledGPUCommandGraph<void> = resources.track(windGraph.compile());

  // Particle graph: advection every frame.
  const particleGraph = new GPUCommandGraph<void>(device, {id: 'terrain-flow-particles'});
  particleGraph.add(
    new GPUParticleAdvection({
      id: 'particles',
      velocities: importGraphBuffer(
        particleGraph,
        'velocity',
        velocityBuffer,
        'float32x2',
        cellCount
      ),
      fieldWidth: width,
      fieldHeight: height,
      parameters: particleParameters.importToGraph(particleGraph),
      wordParameters: particleWords.importToGraph(particleGraph),
      state: {
        positions: importGraphBuffer(
          particleGraph,
          'positions',
          positionBuffer,
          'float32x2',
          PARTICLE_COUNT
        ),
        ages: importGraphBuffer(particleGraph, 'ages', ageBuffer, 'uint32', PARTICLE_COUNT),
        generations: importGraphBuffer(
          particleGraph,
          'generations',
          generationBuffer,
          'uint32',
          PARTICLE_COUNT
        )
      },
      speeds: importGraphBuffer(particleGraph, 'speeds', speedBuffer, 'float32', PARTICLE_COUNT),
      trails: {
        positions: importGraphBuffer(
          particleGraph,
          'trails',
          trailBuffer,
          'float32x2',
          PARTICLE_COUNT * TRAIL_LENGTH
        ),
        length: TRAIL_LENGTH
      }
    })
  );
  const compiledParticles: CompiledGPUCommandGraph<void> = resources.track(particleGraph.compile());

  // --- State --------------------------------------------------------------------------------
  let windDirty = true;
  let resetParticles = true;
  let particleFrame = 0;
  let lastSeed = ctx.options.seed;

  const writeWind = () => {
    // Bearing is clockwise from north; the raster is south-first, so +y is north.
    const bearing = (ctx.options.windBearing * Math.PI) / 180;
    windSettings.write(
      getGPUTerrainFlowFieldParameterValues({
        ...flippedSettings,
        wind: [
          ctx.options.windSpeed * Math.sin(bearing),
          ctx.options.windSpeed * Math.cos(bearing)
        ],
        verticalExaggeration: ctx.options.exaggeration
      })
    );
    windDirty = true;
    describeStep();
  };
  const describeStep = () => {
    const metersPerFrame = ctx.options.windSpeed * ctx.options.timeStep * ctx.options.speedScale;
    ctx.setReadout('travel', `${metersPerFrame.toFixed(0)} m per frame at free-stream speed`);
    ctx.setReadout(
      'trail',
      `${formatInteger(metersPerFrame * (TRAIL_LENGTH - 1))} m (${TRAIL_LENGTH} frames)`
    );
  };
  derivativesSettings.write(
    getGPUTerrainDerivativesParameterValues({
      ...flippedSettings,
      azimuthDegrees: SUN_AZIMUTH_DEGREES,
      altitudeDegrees: SUN_ALTITUDE_DEGREES
    })
  );
  writeWind();
  ctx.setReadout('particles', PARTICLE_COUNT);

  const summary = new SummaryReader(
    resources,
    'terrain-flow',
    [{buffer: ratioHistogramBuffer, size: SPEED_BIN_COUNT * 4}],
    bytes => {
      if (destroyed) return;
      const bins = new Uint32Array(bytes);
      let total = 0;
      let sheltered = 0;
      let blocked = 0;
      for (let bin = 0; bin < SPEED_BIN_COUNT; bin++) {
        total += bins[bin];
        if (bin < SPEED_BIN_COUNT / 2) sheltered += bins[bin];
        if (bin < SPEED_BIN_COUNT / 10) blocked += bins[bin];
      }
      if (total === 0) return;
      ctx.setReadout(
        'sheltered',
        `${((sheltered / total) * 100).toFixed(1)} % of cells below half the free-stream wind`
      );
      ctx.setReadout('blocked', `${((blocked / total) * 100).toFixed(1)} % of cells below a tenth`);
    }
  );

  return {
    getCompiledGraphs: () => [compiledSetup, compiledWind, compiledParticles],

    setOption(id, _value, state) {
      switch (id) {
        case 'windSpeed':
        case 'windBearing':
        case 'exaggeration':
          writeWind();
          ctx.requestLayers();
          break;
        case 'seed':
          if (state.seed !== lastSeed) {
            lastSeed = state.seed;
            resetParticles = true;
          }
          break;
        case 'timeStep':
        case 'speedScale':
          describeStep();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'reset') resetParticles = true;
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      if (frame.frameIndex < 2) compiledSetup.encode(commandEncoder, {parameters: undefined});
      if (windDirty || frame.frameIndex < 2) {
        compiledWind.encode(commandEncoder, {parameters: undefined});
        windDirty = false;
        summary.request(commandEncoder);
      }
      summary.flush(commandEncoder);
      if (options.paused && !resetParticles) return;
      let spawnBounds: [number, number, number, number] | undefined;
      if (options.spawnRegion === 'view') {
        const view = getViewportMetricBounds(frame.viewport, grid.projection);
        spawnBounds = [
          Math.max(view[0], grid.bounds[0]),
          Math.max(view[1], grid.bounds[1]),
          Math.min(view[2], grid.bounds[2]),
          Math.min(view[3], grid.bounds[3])
        ];
        if (spawnBounds[0] >= spawnBounds[2] || spawnBounds[1] >= spawnBounds[3]) {
          spawnBounds = undefined;
        }
      }
      particleParameters.write(
        getGPUParticleAdvectionParameterValues(
          {
            fieldExtent: [
              grid.bounds[0],
              grid.bounds[1],
              grid.displayCellSize[0],
              grid.displayCellSize[1]
            ],
            timeStep: options.timeStep,
            speedScale: options.speedScale,
            dropRate: options.dropRate,
            minimumSpeed: options.minimumSpeed,
            spawnBounds
          },
          [width, height]
        )
      );
      particleWords.write(
        getGPUParticleAdvectionWordParameterValues({
          seed: options.seed,
          frame: particleFrame,
          maximumAge: options.maximumAge,
          reset: resetParticles
        })
      );
      compiledParticles.encode(commandEncoder, {parameters: undefined});
      particleFrame++;
      resetParticles = false;
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const rasterProps = {
        coordinateOrigin: origin,
        gridSize: [width, height] as const,
        bounds: grid.bounds,
        rowOrigin: 'south' as const
      };
      const layers: Layer[] = [
        new SpatialAnalysisRasterLayer({
          ...rasterProps,
          id: 'terrain-flow-hillshade',
          values: hillshadeBuffer,
          valueFormat: 'float32',
          colormap: 'grayscale',
          valueRange: [0, 1],
          color: [255, 255, 255, options.background === 'speed' ? 120 : dark ? 190 : 235]
        })
      ];
      if (options.background === 'speed') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id: 'terrain-flow-speed',
            values: ratioBuffer,
            valueFormat: 'float32',
            colormap: 'magma',
            valueRange: [0, 1],
            discardAtOrBelow: -0.5,
            color: [255, 255, 255, 215]
          })
        );
      }
      layers.push(
        new FlowTrailLayer({
          id: `terrain-flow-trails-${options.ramp}`,
          coordinateOrigin: origin,
          trailPositions: trailBuffer,
          speeds: speedBuffer,
          wordParameters: particleWords.buffer,
          ringLength: TRAIL_LENGTH,
          particleCount: PARTICLE_COUNT,
          speedRange: [0, options.windSpeed * 1.1],
          ramp: options.ramp,
          widthPixels: options.trailWidth,
          opacity: options.trailOpacity
        })
      );
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const cell = getCellAt(grid, event.coordinate[0], event.coordinate[1]);
      if (cell < 0) return null;
      return `Elevation ${formatInteger(grid.elevation[cell])} m`;
    },

    destroy() {
      destroyed = true;
      summary.stop();
      resources.destroy();
    }
  };
}
