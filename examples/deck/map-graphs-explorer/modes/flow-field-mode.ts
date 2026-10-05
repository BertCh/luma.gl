// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Earth.nullschool-style flow over San Francisco from one vector-field raster, drawn three ways
 * that all read the same velocity buffer:
 * - `GPUParticleAdvection`: 20,000 particles advected every frame (RK2, Philox respawn, drop rate,
 *   age) with a 24-slot data-space trail ring that a mode-local layer draws without readback,
 *   colored by the particle's speed.
 * - `GPULineIntegralConvolution`: an animated LIC texture (the phase advances every frame).
 * - `GPUStreamlines`: evenly spaced, deterministic streamlines as CSR polylines.
 *
 * Two fields share the raster grid and the velocity buffer: a synthetic cyclone-plus-jet wind in
 * m/s, and downhill flow, the negative gradient of the terrain (sea cells are NaN, so particles
 * die there). Switching is one `Buffer.write` of 2 MB; nothing recompiles.
 *
 * Per-frame (buffer writes only): speed scale, time step, drop rate, maximum age, minimum speed,
 * LIC phase, streamline spacing (the occupancy-grid cell size is a parameter; the grid and seed
 * lattice sizes are compile-time) and the field itself. The advection graph encodes every frame
 * (animation); LIC encodes every frame only while it animates; streamlines encode only when the
 * field, spacing or seed changed (each encode records about 60 nodes), then persist in their
 * output buffers.
 */

import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getGPULineIntegralConvolutionParameterValues, getGPULineIntegralConvolutionWordParameterValues, getGPUParticleAdvectionParameterValues, getGPUParticleAdvectionWordParameterValues, getGPUStreamlinesParameterValues, getGPUStreamlinesWordParameterValues, GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH, GPU_LINE_INTEGRAL_CONVOLUTION_WORD_PARAMETER_LENGTH, GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH, GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH, GPU_STREAMLINES_PARAMETER_LENGTH, GPU_STREAMLINES_WORD_PARAMETER_LENGTH, GPULineIntegralConvolution, GPUParticleAdvection, GPUStreamlines} from '@luma.gl/experimental/gpu-raster';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import type {MapGraphsTerrain} from '../map-graphs-data';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {
  FLOW_SPEED_RAMP,
  FlowPathLayer,
  FlowTextureLayer,
  FlowTrailLayer
} from './flow-field-layers';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

/** Particles advected and drawn. Compile-time (state and trail buffers are sized by it). */
const PARTICLE_COUNT = 20000;
/** Trail ring length in frames. Compile-time. */
const TRAIL_LENGTH = 24;
/** LIC output pixels per axis (the output extent is independent of the field raster). */
const LIC_WIDTH = 448;
const LIC_HEIGHT = 568;
/** LIC streamline steps in each direction. Compile-time. */
const LIC_STEP_COUNT = 32;
/** LIC step length in output pixels, and ripple period in steps. */
const LIC_STEP_PIXELS = 0.8;
const LIC_PERIOD = 10;
/** LIC ripple cycles per second (negative phase advance moves the texture downstream). */
const LIC_CYCLES_PER_SECOND = 1.2;
/** Streamline occupancy grid (cells) and seed lattice. Compile-time. */
const STREAMLINE_GRID_WIDTH = 100;
const STREAMLINE_GRID_HEIGHT = 124;
const STREAMLINE_SEED_COLUMNS = 160;
const STREAMLINE_SEED_ROWS = 200;
const STREAMLINE_STEPS = 40;
const STREAMLINE_ROUNDS = 16;
const STREAMLINE_LINE_CAPACITY = 8192;
const STREAMLINE_POINT_CAPACITY = 98304;
/** Occupancy-grid cell size in meters at spacing multiplier 1. */
const STREAMLINE_BASE_SPACING = 150;
const STREAMLINE_MINIMUM_POINTS = 6;
/** Random seed of particles, LIC noise and streamline seeds. */
const SEED = 20261;

type FieldKind = 'wind' | 'downhill';

type FieldDefinition = {
  /** `(u, v)` per cell, row 0 at the smallest y. */
  velocities: Float32Array;
  /** Speed mapped to the end of the color ramp, in field units (m/s). */
  speedMaximum: number;
  label: string;
};

export const flowFieldMode: MapGraphsModeDefinition = {
  id: 'flow-field',
  title: 'Flow',
  recipes: ['GPUParticleAdvection', 'GPULineIntegralConvolution', 'GPUStreamlines'],
  description:
    'Wind-map style flow: 20,000 GPU-advected particles with data-space trails, an animated ' +
    'line-integral-convolution texture and evenly spaced streamlines, all from one velocity ' +
    'raster. Switch between a synthetic cyclone-plus-jet wind and downhill flow on the SF ' +
    'terrain; every control is a buffer write.',
  initialViewState: {longitude: -122.44, latitude: 37.755, zoom: 11.6},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const {width: fieldWidth, height: fieldHeight, bounds, cellSize} = terrain;
    const fieldCellCount = fieldWidth * fieldHeight;
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new MapGraphsResources(device, 'flow');
    const fieldExtent = [bounds[0], bounds[1], cellSize[0], cellSize[1]] as const;
    const fields: Record<FieldKind, FieldDefinition> = {
      wind: {
        velocities: createWindField(terrain),
        speedMaximum: 24,
        label: 'Synthetic cyclone + jet (m/s)'
      },
      downhill: {
        velocities: createDownhillField(terrain),
        speedMaximum: 10,
        label: 'Downhill flow: 40 x slope m/s along the negative terrain gradient'
      }
    };

    // --- Buffers -------------------------------------------------------------------------------
    const velocityBuffer = resources.createBuffer('velocity', fields.wind.velocities);
    const positionBuffer = resources.createBuffer('positions', PARTICLE_COUNT * 8);
    const ageBuffer = resources.createBuffer('ages', PARTICLE_COUNT * 4);
    const generationBuffer = resources.createBuffer('generations', PARTICLE_COUNT * 4);
    const speedBuffer = resources.createBuffer('speeds', PARTICLE_COUNT * 4);
    const trailBuffer = resources.createBuffer('trails', PARTICLE_COUNT * TRAIL_LENGTH * 8);
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

    const licValueBuffer = resources.createBuffer('lic-values', LIC_WIDTH * LIC_HEIGHT * 4);
    const licSpeedBuffer = resources.createBuffer('lic-speeds', LIC_WIDTH * LIC_HEIGHT * 4);
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

    const streamlineIds = resources.createBuffer('streamline-ids', STREAMLINE_LINE_CAPACITY * 4);
    const streamlineCount = resources.createBuffer('streamline-count', 4);
    const streamlineOverflow = resources.createBuffer('streamline-overflow', 4);
    const streamlineOffsets = resources.createBuffer(
      'streamline-offsets',
      (STREAMLINE_LINE_CAPACITY + 1) * 4
    );
    const streamlinePoints = resources.createBuffer(
      'streamline-points',
      STREAMLINE_POINT_CAPACITY * 8
    );
    const streamlinePointCount = resources.createBuffer('streamline-point-count', 4);
    const streamlineUnconverged = resources.createBuffer('streamline-unconverged', 4);
    const streamlineParameters = resources.createParameterBuffer(
      'streamline-parameters',
      'float32',
      GPU_STREAMLINES_PARAMETER_LENGTH
    );
    const streamlineWords = resources.createParameterBuffer(
      'streamline-words',
      'uint32',
      GPU_STREAMLINES_WORD_PARAMETER_LENGTH
    );
    const streamlineDraw = resources.track(
      new DrawCommandBuffer(device, {
        id: 'flow-streamline-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const summaryRing = resources.track(
      new GPUReadbackRing(device, {id: 'flow-streamline-summary', byteLength: 16})
    );

    // --- Graphs --------------------------------------------------------------------------------
    const velocityView = (graph: GPUCommandGraph<void>) =>
      importGraphBuffer(graph, 'velocity', velocityBuffer, 'float32x2', fieldCellCount);

    const particleGraph = new GPUCommandGraph<void>(device, {id: 'flow-particles'});
    particleGraph.add(
      new GPUParticleAdvection({
        id: 'particles',
        velocities: velocityView(particleGraph),
        fieldWidth,
        fieldHeight,
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
    const compiledParticles: CompiledGPUCommandGraph<void> = resources.track(
      particleGraph.compile()
    );

    const licGraph = new GPUCommandGraph<void>(device, {id: 'flow-lic'});
    licGraph.add(
      new GPULineIntegralConvolution({
        id: 'lic',
        velocities: velocityView(licGraph),
        fieldWidth,
        fieldHeight,
        width: LIC_WIDTH,
        height: LIC_HEIGHT,
        stepCount: LIC_STEP_COUNT,
        parameters: licParameters.importToGraph(licGraph),
        wordParameters: licWords.importToGraph(licGraph),
        output: {
          values: importGraphBuffer(
            licGraph,
            'lic-values',
            licValueBuffer,
            'float32',
            LIC_WIDTH * LIC_HEIGHT
          ),
          speeds: importGraphBuffer(
            licGraph,
            'lic-speeds',
            licSpeedBuffer,
            'float32',
            LIC_WIDTH * LIC_HEIGHT
          )
        }
      })
    );
    const compiledLic: CompiledGPUCommandGraph<void> = resources.track(licGraph.compile());

    const streamlineGraph = new GPUCommandGraph<void>(device, {id: 'flow-streamlines'});
    streamlineGraph.add(
      new GPUStreamlines({
        id: 'streamlines',
        velocities: velocityView(streamlineGraph),
        fieldWidth,
        fieldHeight,
        gridWidth: STREAMLINE_GRID_WIDTH,
        gridHeight: STREAMLINE_GRID_HEIGHT,
        seedColumns: STREAMLINE_SEED_COLUMNS,
        seedRows: STREAMLINE_SEED_ROWS,
        stepsPerDirection: STREAMLINE_STEPS,
        roundCount: STREAMLINE_ROUNDS,
        parameters: streamlineParameters.importToGraph(streamlineGraph),
        wordParameters: streamlineWords.importToGraph(streamlineGraph),
        output: {
          lines: {
            ids: importGraphBuffer(
              streamlineGraph,
              'ids',
              streamlineIds,
              'uint32',
              STREAMLINE_LINE_CAPACITY
            ),
            count: importGraphBuffer(streamlineGraph, 'count', streamlineCount, 'uint32', 1),
            overflow: importGraphBuffer(
              streamlineGraph,
              'overflow',
              streamlineOverflow,
              'uint32',
              1
            )
          },
          pathOffsets: importGraphBuffer(
            streamlineGraph,
            'offsets',
            streamlineOffsets,
            'uint32',
            STREAMLINE_LINE_CAPACITY + 1
          ),
          points: importGraphBuffer(
            streamlineGraph,
            'points',
            streamlinePoints,
            'float32x2',
            STREAMLINE_POINT_CAPACITY
          ),
          pointCount: importGraphBuffer(
            streamlineGraph,
            'point-count',
            streamlinePointCount,
            'uint32',
            1
          ),
          unconverged: importGraphBuffer(
            streamlineGraph,
            'unconverged',
            streamlineUnconverged,
            'uint32',
            1
          )
        }
      })
    );
    const compiledStreamlines: CompiledGPUCommandGraph<void> = resources.track(
      streamlineGraph.compile()
    );

    // --- State ---------------------------------------------------------------------------------
    let fieldKind: FieldKind = 'wind';
    let showParticles = true;
    let showLic = true;
    let animateLic = true;
    let showStreamlines = false;
    let speedScale = 1;
    let timeStep = 3;
    let dropRate = 0.008;
    let maximumAge = 160;
    let trailWidth = 1.5;
    let streamlineSpacing = 2;
    let particleFrame = 0;
    let resetParticles = true;
    let licDirty = true;
    let streamlinesDirty = true;
    let summaryPending = false;
    let destroyed = false;

    const centerX = (bounds[0] + bounds[2]) / 2;
    const centerY = (bounds[1] + bounds[3]) / 2;

    function writeLicParameters(phase: number): void {
      licParameters.write(
        getGPULineIntegralConvolutionParameterValues({
          fieldExtent,
          outputExtent: [
            bounds[0],
            bounds[1],
            (bounds[2] - bounds[0]) / LIC_WIDTH,
            (bounds[3] - bounds[1]) / LIC_HEIGHT
          ],
          stepLength: LIC_STEP_PIXELS,
          minimumSpeed: 0.05,
          phase,
          period: LIC_PERIOD
        })
      );
    }

    function writeStreamlineParameters(): void {
      const cell = STREAMLINE_BASE_SPACING * streamlineSpacing;
      const gridWidthMeters = STREAMLINE_GRID_WIDTH * cell;
      const gridHeightMeters = STREAMLINE_GRID_HEIGHT * cell;
      streamlineParameters.write(
        getGPUStreamlinesParameterValues({
          fieldExtent,
          // The grid cell is the separation distance: scaling it about the map center is a
          // parameter write (the grid and seed lattice sizes stay compiled).
          gridExtent: [centerX - gridWidthMeters / 2, centerY - gridHeightMeters / 2, cell, cell],
          stepLength: 0.4 * cell,
          minimumSpeed: 0.05
        })
      );
      streamlineWords.write(
        getGPUStreamlinesWordParameterValues({seed: SEED, minimumPoints: STREAMLINE_MINIMUM_POINTS})
      );
      streamlinesDirty = true;
    }

    function setField(kind: FieldKind): void {
      fieldKind = kind;
      velocityBuffer.write(fields[kind].velocities);
      resetParticles = true;
      licDirty = true;
      streamlinesDirty = true;
    }

    writeLicParameters(0);
    writeStreamlineParameters();
    licWords.write(getGPULineIntegralConvolutionWordParameterValues({seed: SEED}));

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<FieldKind>({
      label: 'Vector field (buffer write, no recompile)',
      options: [
        {value: 'wind', label: fields.wind.label},
        {value: 'downhill', label: 'Downhill flow (negative terrain gradient)'}
      ],
      value: fieldKind,
      onChange: value => {
        setField(value);
        speedReadout.setValue(getSpeedReadout());
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Particles + trails (GPUParticleAdvection)',
      value: showParticles,
      onChange: value => {
        showParticles = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Speed scale (per-frame)',
      min: 0.25,
      max: 4,
      step: 0.25,
      value: speedScale,
      format: value => `${value}x`,
      onChange: value => {
        speedScale = value;
      }
    });
    context.controls.addSlider({
      label: 'Time step (per-frame)',
      min: 1,
      max: 30,
      step: 1,
      value: timeStep,
      format: value => `${value} s / frame`,
      onChange: value => {
        timeStep = value;
      }
    });
    context.controls.addSlider({
      label: 'Drop rate (per-frame)',
      min: 0,
      max: 0.05,
      step: 0.002,
      value: dropRate,
      format: value => value.toFixed(3),
      onChange: value => {
        dropRate = value;
      }
    });
    context.controls.addSlider({
      label: 'Maximum age',
      min: 40,
      max: 400,
      step: 20,
      value: maximumAge,
      format: value => `${value} frames`,
      onChange: value => {
        maximumAge = value;
      }
    });
    context.controls.addSlider({
      label: 'Trail width',
      min: 0.8,
      max: 3,
      step: 0.1,
      value: trailWidth,
      format: value => `${value.toFixed(1)} px`,
      onChange: value => {
        trailWidth = value;
        context.updateLayers();
      }
    });
    context.controls.addButton({
      label: 'Reset particles',
      onClick: () => {
        resetParticles = true;
      }
    });
    context.controls.addToggle({
      label: 'LIC texture (GPULineIntegralConvolution)',
      value: showLic,
      onChange: value => {
        showLic = value;
        licDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Animate LIC phase (per-frame)',
      value: animateLic,
      onChange: value => {
        animateLic = value;
      }
    });
    context.controls.addToggle({
      label: 'Streamlines (GPUStreamlines)',
      value: showStreamlines,
      onChange: value => {
        showStreamlines = value;
        streamlinesDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Streamline spacing (per-frame grid cell)',
      min: 1,
      max: 4,
      step: 0.5,
      value: streamlineSpacing,
      format: value => `${Math.round(STREAMLINE_BASE_SPACING * value)} m`,
      onChange: value => {
        streamlineSpacing = value;
        writeStreamlineParameters();
      }
    });
    context.controls.addLegend({
      title: 'Flow speed (particles and LIC)',
      gradient: {
        colors: FLOW_SPEED_RAMP,
        minimumLabel: 'slow',
        maximumLabel: 'fast'
      }
    });
    context.controls.addNote(
      'Advection encodes every frame (it is the animation). LIC encodes every frame only while ' +
        'its phase animates. Streamlines re-encode only when the field, spacing or toggle ' +
        'changes (about 60 nodes). Their grid and seed-lattice sizes are compile-time; the ' +
        'spacing works by scaling the grid cell, so larger values cover a wider area with ' +
        'fewer lines.'
    );
    const getSpeedReadout = () =>
      `0 to ${fields[fieldKind].speedMaximum} m/s (${fieldKind === 'wind' ? 'wind' : 'downhill'})`;
    const speedReadout = context.controls.addReadout('Speed ramp', getSpeedReadout());
    context.controls.addReadout(
      'Particles',
      `${formatCount(PARTICLE_COUNT)} · trail ring ${TRAIL_LENGTH} slots · ` +
        `${((PARTICLE_COUNT * TRAIL_LENGTH * 8) / 1048576).toFixed(1)} MB`
    );
    context.controls.addReadout(
      'Field raster',
      `${fieldWidth} × ${fieldHeight} cells, ${(cellSize[0]).toFixed(1)} × ${cellSize[1].toFixed(1)} m`
    );
    context.controls.addReadout(
      'LIC raster',
      `${LIC_WIDTH} × ${LIC_HEIGHT}, ${LIC_STEP_COUNT * 2} taps`
    );
    const streamlineReadout = context.controls.addReadout('Streamlines', 'not encoded yet');
    const particleTimingReadout = context.controls.addReadout('Advection graph', 'measuring...');
    const licTimingReadout = context.controls.addReadout('LIC graph', 'measuring...');
    const streamlineTimingReadout = context.controls.addReadout(
      'Streamlines graph',
      'measuring...'
    );
    context.controls.addReadout('Data', terrain.attribution);

    const measureAll = async () => {
      try {
        const options = {
          parameters: undefined,
          completionBuffer: streamlineOverflow,
          signal: context.signal
        };
        const particleTiming = await measureCompiledGraph(device, compiledParticles, options);
        particleTimingReadout.setValue(
          `${compiledParticles.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(particleTiming)}`
        );
        const licTiming = await measureCompiledGraph(device, compiledLic, options);
        licTimingReadout.setValue(
          `${compiledLic.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(licTiming)}`
        );
        const streamlineTiming = await measureCompiledGraph(device, compiledStreamlines, {
          ...options,
          runs: 3
        });
        streamlineTimingReadout.setValue(
          `${compiledStreamlines.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(streamlineTiming)}`
        );
        streamlinesDirty = true;
      } catch (error) {
        if (!destroyed) particleTimingReadout.setValue(`failed: ${(error as Error).message}`);
      }
    };
    context.controls.addButton({label: 'Measure GPU cost', onClick: () => void measureAll()});

    // --- Per-frame encode ----------------------------------------------------------------------
    const readStreamlineSummary = async (
      commandEncoder: Parameters<MapGraphsModeInstance['encode']>[0]
    ) => {
      const ticket = summaryRing.tryAcquire();
      if (!ticket) return;
      [streamlineCount, streamlinePointCount, streamlineOverflow, streamlineUnconverged].forEach(
        (buffer, index) => {
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: buffer,
            sourceOffset: 0,
            destinationBuffer: ticket.buffer,
            destinationOffset: index * 4,
            size: 4
          });
        }
      );
      ticket.markEncoded({byteOffset: 0, byteLength: 16});
      summaryPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 4);
        streamlineReadout.setValue(
          `${formatCount(words[0])} lines · ${formatCount(words[1])} points · ` +
            `${words[2] ? 'capacity overflow' : 'no overflow'} · ` +
            `${words[3] ? 'unconverged (undecided lines dropped)' : 'converged'}`
        );
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        summaryPending = false;
      }
    };

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [compiledParticles, compiledLic, compiledStreamlines],
      encode(commandEncoder, frame) {
        // Particles: always (this graph is the animation).
        particleParameters.write(
          getGPUParticleAdvectionParameterValues(
            {
              fieldExtent,
              timeStep,
              speedScale,
              dropRate,
              minimumSpeed: fieldKind === 'downhill' ? 0.1 : 0
            },
            [fieldWidth, fieldHeight]
          )
        );
        particleWords.write(
          getGPUParticleAdvectionWordParameterValues({
            seed: SEED,
            frame: particleFrame,
            maximumAge,
            reset: resetParticles
          })
        );
        if (showParticles) {
          compiledParticles.encode(commandEncoder, {parameters: undefined});
          particleFrame++;
          resetParticles = false;
        }

        // LIC: every frame while animating, otherwise only after a change.
        if (showLic && (animateLic || licDirty)) {
          writeLicParameters(animateLic ? -frame.timeSeconds * LIC_CYCLES_PER_SECOND : 0);
          compiledLic.encode(commandEncoder, {parameters: undefined});
          licDirty = false;
        }

        // Streamlines: only when something they depend on changed.
        if (showStreamlines && streamlinesDirty) {
          compiledStreamlines.encode(commandEncoder, {parameters: undefined});
          // The published point count becomes the indirect instance count, with no readback.
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: streamlinePointCount,
            sourceOffset: 0,
            destinationBuffer: streamlineDraw.buffer,
            destinationOffset: 4,
            size: 4
          });
          streamlinesDirty = false;
          if (!summaryPending) void readStreamlineSummary(commandEncoder);
        }
      },
      getLayers() {
        const layers: Layer[] = [];
        const speedRange = [0, fields[fieldKind].speedMaximum] as const;
        if (showLic) {
          layers.push(
            new FlowTextureLayer({
              id: 'flow-lic',
              coordinateOrigin: origin,
              values: licValueBuffer,
              speeds: licSpeedBuffer,
              gridSize: [LIC_WIDTH, LIC_HEIGHT],
              bounds,
              speedRange,
              opacity: showParticles ? 0.4 : 0.9
            })
          );
        }
        if (showStreamlines) {
          layers.push(
            new FlowPathLayer({
              id: 'flow-streamlines',
              coordinateOrigin: origin,
              points: streamlinePoints,
              pathOffsets: streamlineOffsets,
              lineCount: streamlineCount,
              pointCount: streamlinePointCount,
              pointCapacity: STREAMLINE_POINT_CAPACITY,
              drawCommands: streamlineDraw,
              color: [255, 255, 255, 255],
              widthPixels: 1,
              opacity: 0.55
            })
          );
        }
        if (showParticles) {
          layers.push(
            new FlowTrailLayer({
              id: 'flow-trails',
              coordinateOrigin: origin,
              trailPositions: trailBuffer,
              speeds: speedBuffer,
              wordParameters: particleWords.buffer,
              ringLength: TRAIL_LENGTH,
              particleCount: PARTICLE_COUNT,
              speedRange,
              widthPixels: trailWidth,
              opacity: 0.95
            })
          );
        }
        return layers;
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };

    setTimeout(() => {
      if (!destroyed) void measureAll();
    }, 1500);
    return instance;
  }
};

/**
 * Synthetic wind in meters per second over the terrain raster: a westerly background, a jet band
 * and two opposite-signed Rankine-style vortices. Row 0 is the south edge.
 */
function createWindField(terrain: MapGraphsTerrain): Float32Array {
  const {width, height, bounds, cellSize} = terrain;
  const centerX = (bounds[0] + bounds[2]) / 2;
  const centerY = (bounds[1] + bounds[3]) / 2;
  const vortices = [
    {x: centerX - 1500, y: centerY + 2500, circulation: 84000, core: 3000},
    {x: centerX + 4500, y: centerY - 3500, circulation: -60000, core: 2500}
  ];
  const velocities = new Float32Array(width * height * 2);
  for (let row = 0; row < height; row++) {
    const y = bounds[1] + (row + 0.5) * cellSize[1];
    for (let column = 0; column < width; column++) {
      const x = bounds[0] + (column + 0.5) * cellSize[0];
      let u = 6 + 9 * Math.exp(-(((y - centerY + 3500) / 2600) ** 2));
      let v = 1.5;
      for (const vortex of vortices) {
        const dx = x - vortex.x;
        const dy = y - vortex.y;
        const strength = vortex.circulation / (dx * dx + dy * dy + vortex.core * vortex.core);
        u += -dy * strength;
        v += dx * strength;
      }
      const index = (row * width + column) * 2;
      velocities[index] = u;
      velocities[index + 1] = v;
    }
  }
  return velocities;
}

/**
 * Downhill flow: `-40 * grad(elevation)` in m/s (a 25 percent slope flows at 10 m/s). Sea cells
 * and cells whose 3x3 neighborhood touches the sea are NaN, which the recipes treat as "no data".
 * Terrain row 0 is the north edge, so rows are flipped into the field's south-first layout.
 */
function createDownhillField(terrain: MapGraphsTerrain): Float32Array {
  const {width, height, elevation, cellSize} = terrain;
  const velocities = new Float32Array(width * height * 2);
  const isLand = (column: number, row: number) =>
    column >= 0 &&
    column < width &&
    row >= 0 &&
    row < height &&
    elevation[row * width + column] > 0.5;
  for (let fieldRow = 0; fieldRow < height; fieldRow++) {
    const terrainRow = height - 1 - fieldRow;
    for (let column = 0; column < width; column++) {
      const index = (fieldRow * width + column) * 2;
      const hasData =
        isLand(column, terrainRow) &&
        isLand(column - 1, terrainRow) &&
        isLand(column + 1, terrainRow) &&
        isLand(column, terrainRow - 1) &&
        isLand(column, terrainRow + 1);
      if (!hasData) {
        velocities[index] = Number.NaN;
        velocities[index + 1] = Number.NaN;
        continue;
      }
      const east = elevation[terrainRow * width + column + 1];
      const west = elevation[terrainRow * width + column - 1];
      const north = elevation[(terrainRow - 1) * width + column];
      const south = elevation[(terrainRow + 1) * width + column];
      velocities[index] = -40 * ((east - west) / (2 * cellSize[0]));
      velocities[index + 1] = -40 * ((north - south) / (2 * cellSize[1]));
    }
  }
  return velocities;
}
