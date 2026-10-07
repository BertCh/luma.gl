// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Drainage analysis of the San Francisco elevation raster, entirely on the GPU. One compiled graph
 * chains the round-8 hydrology contributors on a depression-filled surface:
 *
 * - `GPUTerrainFlow`: Planchon-Darboux fill, D8 directions, flat resolution (Barnes et al. 2014),
 *   and accumulation under D8, D-infinity or multiple-flow-direction routing, in m^2.
 * - `GPUTerrainHeightAboveDrainage` (HAND), `GPUTerrainWatersheds` (drainage basins, and watersheds
 *   of clicked pour points), `GPUTerrainStreamOrder` (Strahler) and `GPUTerrainHydrologicIndices`
 *   (specific catchment area, wetness, stream power).
 *
 * A second pair of graphs shows a terrain-following wind: `GPUTerrainFlowField` deflects a uniform
 * wind around the relief and `GPUParticleAdvection` moves particles through it.
 *
 * Compile-time choices (flow routing, flat resolution) rebuild the drainage graph; the stream
 * threshold, fill epsilon, flow exponent, pour points and wind are per-frame buffer writes. The
 * drainage graph is encoded only when an input changed; every output persists in its buffer.
 * Sea (elevation 0) is invalid, so water leaves the grid at the coast.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUHistogram,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainFlowFieldParameterValues,
  getGPUTerrainFlowParameterValues,
  getGPUTerrainHydrologicIndicesParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
  GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH,
  GPUTerrainDerivatives,
  GPUTerrainFlow,
  GPUTerrainFlowField,
  GPUTerrainHeightAboveDrainage,
  GPUTerrainHydrologicIndices,
  GPUTerrainStreamOrder,
  GPUTerrainWatersheds,
  type GPUTerrainFlowRouting
} from '@luma.gl/experimental/gpu-terrain';
import {
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionWordParameterValues,
  GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH,
  GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH,
  GPUParticleAdvection
} from '@luma.gl/experimental/gpu-raster';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance,
  SpatialAnalysisPointerEvent
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {FLOW_SPEED_RAMP, FlowTrailLayer} from './flow-field-layers';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

/** Compile-time iteration limits of the relaxations. */
const MAXIMUM_FILL_ITERATIONS = 256;
const MAXIMUM_FLAT_ITERATIONS = 256;
const MAXIMUM_ACCUMULATION_ITERATIONS = 128;
const SUN_AZIMUTH_DEGREES = 315;
const SUN_ALTITUDE_DEGREES = 40;
/** Fill epsilon choices in meters; index 0 leaves filled depressions as exact flats. */
const FILL_EPSILONS = [0, 0.0001, 0.001, 0.01, 0.1] as const;
/** Pour point capacity. Compile-time length; unused slots hold an out-of-grid index. */
const POUR_POINT_CAPACITY = 8;
const UNUSED_POUR_POINT = 0xffffffff;
/** Cells searched around a click for the cell of highest accumulation. */
const POUR_POINT_SNAP_RADIUS = 5;
/** Strahler order histogram bins (orders 0 to 7). */
const ORDER_BIN_COUNT = 8;
/** Wind particles and trails. */
const PARTICLE_COUNT = 20000;
const TRAIL_LENGTH = 24;
const PARTICLE_SEED = 8115;
const PARTICLE_MAXIMUM_AGE = 160;
const PARTICLE_DROP_RATE = 0.008;
const PARTICLE_TIME_STEP = 3;
/** Frames the summary is allowed to lag a change. */
const MEASURE_DELAY_MILLISECONDS = 1500;
/** Display value for NaN and infinite cells; hidden with `discardAtOrBelow`. */
const HIDDEN_VALUE = -9999;

const BASIN_PALETTE = [
  [230, 90, 90, 215],
  [240, 170, 60, 215],
  [225, 215, 80, 215],
  [110, 205, 100, 215],
  [70, 200, 190, 215],
  [80, 140, 240, 215],
  [150, 110, 235, 215],
  [225, 100, 200, 215]
] as const;
const CELL_CLASS_PALETTE = [
  [0, 0, 0, 0],
  [255, 170, 40, 230],
  [235, 60, 60, 240],
  [70, 150, 255, 235],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0]
] as const;
const ORDER_PALETTE = [
  [0, 0, 0, 0],
  [120, 190, 255, 255],
  [60, 225, 225, 255],
  [110, 235, 110, 255],
  [250, 230, 70, 255],
  [255, 150, 50, 255],
  [255, 80, 80, 255],
  [240, 90, 235, 255]
] as const;
/** Line widths in CSS pixels for Strahler orders 1, 2, 3 and 4 or more. */
const ORDER_WIDTHS = [1.2, 2, 3.2, 4.6] as const;

type Product =
  | 'accumulation'
  | 'cell-classes'
  | 'hand'
  | 'watersheds'
  | 'stream-order'
  | 'indices'
  | 'wind';
type IndexKind = 'wetness' | 'catchment' | 'power';

export const drainageMode: SpatialAnalysisModeDefinition = {
  id: 'drainage',
  title: 'Drainage',
  contributors: [
    'GPUTerrainFlow',
    'GPUTerrainHeightAboveDrainage',
    'GPUTerrainWatersheds',
    'GPUTerrainStreamOrder',
    'GPUTerrainHydrologicIndices',
    'GPUTerrainFlowField',
    'GPUParticleAdvection',
    'GPUTerrainDerivatives',
    'GPUHistogram'
  ],
  description:
    'Drainage products on the San Francisco elevation raster: flow accumulation under D8, ' +
    'D-infinity and multiple-flow-direction routing with flat resolution, height above nearest ' +
    'drainage, watersheds from clicked pour points, Strahler stream order, wetness and stream ' +
    'power indices, and a terrain-following wind. Routing and flat resolution rebuild the graph; ' +
    'everything else is a buffer write.',
  initialViewState: {longitude: -122.44, latitude: 37.735, zoom: 12},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const cellCount = width * height;
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const projection = new LocalMetricProjection(terrain.origin);
    const resources = new SpatialAnalysisResources(device, 'drainage');
    const cellAreaSquareMeters = cellSize[0] * cellSize[1];

    // Sea (elevation 0) is invalid. The wind graph reads the raster south-first, because its
    // particles run in the y-up meter frame of the layers.
    const validityValues = new Uint32Array(cellCount);
    const flippedElevation = new Float32Array(cellCount);
    const flippedValidity = new Uint32Array(cellCount);
    let landCellCount = 0;
    for (let row = 0; row < height; row++) {
      for (let column = 0; column < width; column++) {
        const index = row * width + column;
        const flippedIndex = (height - 1 - row) * width + column;
        const isLand = terrain.elevation[index] > 0.5 ? 1 : 0;
        validityValues[index] = isLand;
        flippedElevation[flippedIndex] = terrain.elevation[index];
        flippedValidity[flippedIndex] = isLand;
        landCellCount += isLand;
      }
    }

    // --- Buffers -------------------------------------------------------------------------------
    const float32Cells = (name: string) => resources.createBuffer(name, cellCount * 4);
    const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
    const validityBuffer = resources.createBuffer('validity', validityValues);
    const windElevationBuffer = resources.createBuffer('wind-elevation', flippedElevation);
    const windValidityBuffer = resources.createBuffer('wind-validity', flippedValidity);
    const hillshadeBuffer = float32Cells('hillshade');
    const filledBuffer = float32Cells('filled');
    const directionsBuffer = float32Cells('directions');
    const classesBuffer = float32Cells('classes');
    const accumulationBuffer = float32Cells('accumulation');
    const streamsBuffer = float32Cells('streams');
    const handBuffer = float32Cells('hand');
    const basinLabelsBuffer = float32Cells('basin-labels');
    const watershedLabelsBuffer = float32Cells('watershed-labels');
    const streamOrderBuffer = float32Cells('stream-order');
    const catchmentBuffer = float32Cells('catchment');
    const wetnessBuffer = float32Cells('wetness');
    const powerBuffer = float32Cells('power');
    const accumulationDisplayBuffer = float32Cells('accumulation-display');
    const wetnessDisplayBuffer = float32Cells('wetness-display');
    const catchmentDisplayBuffer = float32Cells('catchment-display');
    const powerDisplayBuffer = float32Cells('power-display');
    const basinClassBuffer = float32Cells('basin-class');
    const segmentsBuffer = resources.createBuffer('stream-segments', cellCount * 16);
    const widthBuffers = ORDER_WIDTHS.map((_, index) => float32Cells(`stream-width-${index}`));
    const flagNames = ['fill', 'flats', 'accumulation', 'hand', 'basins', 'watersheds', 'order'];
    const flagBuffers = flagNames.map(name => resources.createBuffer(`${name}-converged`, 4));
    // Iterations executed by the fill, flat resolution, accumulation and stream order loops.
    const iterationNames = ['fill', 'flats', 'accumulation', 'order'];
    const iterationBuffers = iterationNames.map(name =>
      resources.createBuffer(`${name}-iterations`, 4)
    );
    const streamCountsBuffer = resources.createBuffer('stream-counts', 8);
    const orderHistogramBuffer = resources.createBuffer('order-histogram', ORDER_BIN_COUNT * 4);
    const flowSettings = resources.createParameterBuffer(
      'flow-settings',
      'float32',
      GPU_TERRAIN_FLOW_PARAMETER_LENGTH
    );
    const indicesSettings = resources.createParameterBuffer(
      'indices-settings',
      'float32',
      GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH
    );
    const derivativesSettings = resources.createParameterBuffer(
      'derivatives-settings',
      'float32',
      GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
    );
    const pourPointValues = new Uint32Array(POUR_POINT_CAPACITY).fill(UNUSED_POUR_POINT);
    const pourPointBuffer = resources.createParameterBuffer(
      'pour-points',
      'uint32',
      POUR_POINT_CAPACITY,
      pourPointValues
    );
    const pourPointPositionsBuffer = resources.createBuffer(
      'pour-point-positions',
      POUR_POINT_CAPACITY * 8
    );
    const pourPointIndexBuffer = resources.createBuffer(
      'pour-point-indices',
      new Uint32Array(Array.from({length: POUR_POINT_CAPACITY}, (_, index) => index))
    );

    // Wind buffers.
    const velocityBuffer = resources.createBuffer('velocity', cellCount * 8);
    const positionBuffer = resources.createBuffer('particle-positions', PARTICLE_COUNT * 8);
    const ageBuffer = resources.createBuffer('particle-ages', PARTICLE_COUNT * 4);
    const generationBuffer = resources.createBuffer('particle-generations', PARTICLE_COUNT * 4);
    const speedBuffer = resources.createBuffer('particle-speeds', PARTICLE_COUNT * 4);
    const trailBuffer = resources.createBuffer(
      'particle-trails',
      PARTICLE_COUNT * TRAIL_LENGTH * 8
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

    // --- Graphs --------------------------------------------------------------------------------
    // Hillshade and wind do not depend on routing, so they compile once.
    const setupGraph = new GPUCommandGraph<void>(device, {id: 'drainage-setup'});
    setupGraph.add(
      new GPUTerrainDerivatives({
        id: 'derivatives',
        width,
        height,
        elevation: {
          id: 'elevation',
          format: 'float32' as const,
          storage: {
            kind: 'buffer' as const,
            values: importGraphBuffer(
              setupGraph,
              'elevation',
              elevationBuffer,
              'float32',
              cellCount
            )
          },
          validity: importGraphBuffer(setupGraph, 'validity', validityBuffer, 'uint32', cellCount)
        },
        settings: derivativesSettings.importToGraph(setupGraph),
        hillshade: importGraphBuffer(
          setupGraph,
          'hillshade',
          hillshadeBuffer,
          'float32',
          cellCount
        ),
        cellSizeMode: 'uniform',
        rowDirection: 'south'
      })
    );
    const compiledSetup: CompiledGPUCommandGraph<void> = resources.track(setupGraph.compile());

    const windGraph = new GPUCommandGraph<void>(device, {id: 'drainage-wind'});
    windGraph.add(
      new GPUTerrainFlowField({
        id: 'wind-field',
        width,
        height,
        elevation: {
          id: 'wind-elevation',
          format: 'float32' as const,
          storage: {
            kind: 'buffer' as const,
            values: importGraphBuffer(
              windGraph,
              'wind-elevation',
              windElevationBuffer,
              'float32',
              cellCount
            )
          },
          validity: importGraphBuffer(
            windGraph,
            'wind-validity',
            windValidityBuffer,
            'uint32',
            cellCount
          )
        },
        settings: windSettings.importToGraph(windGraph),
        cellSizeMode: 'uniform',
        velocities: importGraphBuffer(windGraph, 'velocity', velocityBuffer, 'float32x2', cellCount)
      })
    );
    const compiledWind: CompiledGPUCommandGraph<void> = resources.track(windGraph.compile());

    const particleGraph = new GPUCommandGraph<void>(device, {id: 'drainage-particles'});
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
    const compiledParticles: CompiledGPUCommandGraph<void> = resources.track(
      particleGraph.compile()
    );

    /** Builds the drainage graph for one routing and flat-resolution choice (both compile-time). */
    function compileDrainageGraph(
      flowRouting: GPUTerrainFlowRouting,
      resolveFlats: boolean
    ): CompiledGPUCommandGraph<void> {
      const graph = new GPUCommandGraph<void>(device, {id: 'drainage'});
      // A graph imports each buffer once; later uses of the same id reuse the view.
      const imported = new Map<string, GraphDataView<'float32' | 'uint32'>>();
      const view = <Format extends 'float32' | 'uint32'>(
        id: string,
        buffer: Buffer,
        format: Format,
        length = cellCount
      ): GraphDataView<Format> => {
        let existing = imported.get(id);
        if (!existing) {
          existing = importGraphBuffer(graph, id, buffer, format, length);
          imported.set(id, existing);
        }
        return existing as GraphDataView<Format>;
      };
      const elevationView = view('elevation', elevationBuffer, 'float32');
      const filled = view('filled', filledBuffer, 'float32');
      const directions = view('directions', directionsBuffer, 'uint32');
      const accumulation = view('accumulation', accumulationBuffer, 'float32');
      const streams = view('streams', streamsBuffer, 'uint32');
      const streamOrder = view('stream-order', streamOrderBuffer, 'uint32');
      const flag = (index: number) =>
        view(`${flagNames[index]}-converged`, flagBuffers[index], 'uint32', 1);
      const iterations = (index: number) =>
        view(`${iterationNames[index]}-iterations`, iterationBuffers[index], 'uint32', 1);
      // The routing surface is the filled elevation; its NaNs mark invalid cells.
      const filledBand = {
        id: 'filled',
        format: 'float32' as const,
        storage: {kind: 'buffer' as const, values: filled}
      };

      graph.add(
        new GPUTerrainFlow({
          id: 'flow',
          width,
          height,
          elevation: {
            id: 'elevation',
            format: 'float32' as const,
            storage: {kind: 'buffer' as const, values: elevationView},
            validity: view('validity', validityBuffer, 'uint32')
          },
          settings: flowSettings.importToGraph(graph),
          cellSizeMode: 'uniform',
          fillDepressions: true,
          maxFillIterations: MAXIMUM_FILL_ITERATIONS,
          resolveFlats,
          maxFlatIterations: MAXIMUM_FLAT_ITERATIONS,
          flowRouting,
          maxAccumulationIterations: MAXIMUM_ACCUMULATION_ITERATIONS,
          accumulationUnits: 'area',
          filledElevation: filled,
          flowDirections: directions,
          cellClasses: view('classes', classesBuffer, 'uint32'),
          accumulation,
          streams,
          fillConverged: flag(0),
          fillIterations: iterations(0),
          ...(resolveFlats ? {flatsConverged: flag(1), flatsIterations: iterations(1)} : {}),
          accumulationConverged: flag(2),
          accumulationIterations: iterations(2)
        })
      );
      graph.add(
        new GPUTerrainHeightAboveDrainage({
          id: 'hand',
          width,
          height,
          elevation: filledBand,
          flowDirections: directions,
          streams,
          heightAboveDrainage: view('hand', handBuffer, 'float32'),
          converged: flag(3)
        })
      );
      graph.add(
        new GPUTerrainWatersheds({
          id: 'basins',
          width,
          height,
          flowDirections: directions,
          labels: view('basin-labels', basinLabelsBuffer, 'uint32'),
          converged: flag(4)
        })
      );
      graph.add(
        new GPUTerrainWatersheds({
          id: 'watersheds',
          width,
          height,
          flowDirections: directions,
          pourPoints: pourPointBuffer.importToGraph(graph),
          labels: view('watershed-labels', watershedLabelsBuffer, 'uint32'),
          converged: flag(5)
        })
      );
      graph.add(
        new GPUTerrainStreamOrder({
          id: 'order',
          width,
          height,
          flowDirections: directions,
          streams,
          streamOrder,
          converged: flag(6),
          iterationCount: iterations(3)
        })
      );
      graph.add(
        new GPUTerrainHydrologicIndices({
          id: 'indices',
          width,
          height,
          elevation: filledBand,
          accumulation,
          settings: indicesSettings.importToGraph(graph),
          cellSizeMode: 'uniform',
          specificCatchmentArea: view('catchment', catchmentBuffer, 'float32'),
          wetnessIndex: view('wetness', wetnessBuffer, 'float32'),
          streamPowerIndex: view('power', powerBuffer, 'float32')
        })
      );
      graph.add(
        new GPUHistogram({
          id: 'stream-count',
          input: streams,
          output: view('stream-counts', streamCountsBuffer, 'uint32', 2),
          domain: [0, 2]
        })
      );
      graph.add(
        new GPUHistogram({
          id: 'order-histogram',
          input: streamOrder,
          output: view('order-histogram', orderHistogramBuffer, 'uint32', ORDER_BIN_COUNT),
          domain: [0, ORDER_BIN_COUNT]
        })
      );
      const displays = [
        ['accumulation', accumulation, accumulationDisplayBuffer, true],
        ['wetness', view('wetness', wetnessBuffer, 'float32'), wetnessDisplayBuffer, false],
        ['catchment', view('catchment', catchmentBuffer, 'float32'), catchmentDisplayBuffer, true],
        ['power', view('power', powerBuffer, 'float32'), powerDisplayBuffer, true]
      ] as const;
      for (const [name, input, buffer, useLog] of displays) {
        addDisplayPass(graph, {
          id: `${name}-display`,
          cellCount,
          input,
          output: view(`${name}-display`, buffer, 'float32'),
          useLog
        });
      }
      addBasinClassPass(graph, {
        cellCount,
        labels: view('basin-labels', basinLabelsBuffer, 'uint32'),
        output: view('basin-class', basinClassBuffer, 'uint32')
      });
      addStreamSegmentPass(graph, {
        cellCount,
        width,
        bounds,
        cellSize,
        directions,
        streamOrder,
        segments: view('stream-segments', segmentsBuffer, 'float32', cellCount * 4),
        widths: widthBuffers.map((buffer, index) =>
          view(`stream-width-${index}`, buffer, 'float32')
        )
      });
      return graph.compile();
    }

    // --- State ---------------------------------------------------------------------------------
    let flowRouting: GPUTerrainFlowRouting = 'd8';
    let resolveFlats = true;
    let compiledDrainage = resources.track(compileDrainageGraph(flowRouting, resolveFlats));
    let product: Product = 'accumulation';
    let indexKind: IndexKind = 'wetness';
    let showStreams = true;
    let thresholdExponent = 2.3;
    let epsilonIndex = 0;
    let flowExponent = 0;
    let handMaximum = 60;
    let windSpeed = 12;
    let windBearingDegrees = 100;
    let verticalExaggeration = 6;
    let drainageDirty = true;
    let windDirty = true;
    let resetParticles = true;
    let particleFrame = 0;
    let destroyed = false;
    let timingsRequested = false;
    let rebuildCount = 0;
    const pourPoints: number[] = [];

    const getThresholdCells = () => Math.round(10 ** thresholdExponent);
    const writeFlowSettings = () => {
      flowSettings.write(
        getGPUTerrainFlowParameterValues({
          cellSize,
          fillEpsilon: FILL_EPSILONS[epsilonIndex],
          streamThreshold: getThresholdCells() * cellAreaSquareMeters,
          flowExponent
        })
      );
      drainageDirty = true;
    };
    const writeWindSettings = () => {
      const bearing = (windBearingDegrees * Math.PI) / 180;
      windSettings.write(
        getGPUTerrainFlowFieldParameterValues({
          cellSize,
          // The raster is south-first here, so +y is north: a bearing is clockwise from north.
          wind: [windSpeed * Math.sin(bearing), windSpeed * Math.cos(bearing)],
          verticalExaggeration
        })
      );
      windDirty = true;
    };
    const writePourPoints = () => {
      pourPointValues.fill(UNUSED_POUR_POINT);
      const positions = new Float32Array(POUR_POINT_CAPACITY * 2);
      pourPoints.forEach((cell, index) => {
        pourPointValues[index] = cell;
        positions[2 * index] = bounds[0] + ((cell % width) + 0.5) * cellSize[0];
        positions[2 * index + 1] = bounds[3] - (Math.floor(cell / width) + 0.5) * cellSize[1];
      });
      pourPointBuffer.write(pourPointValues);
      pourPointPositionsBuffer.write(positions);
      drainageDirty = true;
    };
    derivativesSettings.write(
      getGPUTerrainDerivativesParameterValues({
        cellSize,
        azimuthDegrees: SUN_AZIMUTH_DEGREES,
        altitudeDegrees: SUN_ALTITUDE_DEGREES
      })
    );
    indicesSettings.write(getGPUTerrainHydrologicIndicesParameterValues({cellSize}));
    writeFlowSettings();
    writeWindSettings();

    // --- Readouts ------------------------------------------------------------------------------
    const summary = new SummaryReader(
      resources,
      'drainage',
      [
        ...flagBuffers.map(buffer => ({buffer, size: 4})),
        {buffer: streamCountsBuffer, size: 8},
        {buffer: orderHistogramBuffer, size: ORDER_BIN_COUNT * 4},
        ...iterationBuffers.map(buffer => ({buffer, size: 4}))
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        const iterationBase = 9 + ORDER_BIN_COUNT;
        const flag = (index: number, label: string, limit: number, iterationIndex: number) =>
          `${label} ${words[index] ? 'yes' : 'NO'} ${words[iterationBase + iterationIndex]}/${limit}`;
        convergedHandle.setValue(
          [
            flag(0, 'fill', MAXIMUM_FILL_ITERATIONS, 0),
            resolveFlats ? flag(1, 'flats', MAXIMUM_FLAT_ITERATIONS, 1) : 'flats off',
            flag(2, 'accumulation', MAXIMUM_ACCUMULATION_ITERATIONS, 2),
            words[3] && words[4] && words[5] && words[6] ? 'tracing yes' : 'tracing NO',
            `order ${words[iterationBase + 3]} rounds`
          ].join(', ')
        );
        streamHandle.setValue(
          `${formatCount(words[8])} of ${formatCount(landCellCount)} land cells`
        );
        const orders = Array.from(words.slice(9 + 1, 9 + ORDER_BIN_COUNT));
        orderHandle.setValue(
          orders.map((count, index) => `${index + 1}: ${formatCount(count)}`).join('  ')
        );
      }
    );

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<Product>({
      label: 'Product',
      options: [
        {value: 'accumulation', label: 'Log flow accumulation (GPUTerrainFlow)'},
        {value: 'cell-classes', label: 'Flats, pits and outlets (flat resolution)'},
        {value: 'hand', label: 'Height above nearest drainage'},
        {value: 'watersheds', label: 'Watersheds (click pour points)'},
        {value: 'stream-order', label: 'Strahler stream order'},
        {value: 'indices', label: 'Hydrologic indices'},
        {value: 'wind', label: 'Terrain-following wind'}
      ],
      value: product,
      onChange: value => {
        product = value;
        if (value === 'wind') resetParticles = true;
        context.updateLayers();
      }
    });
    context.controls.addSelect<GPUTerrainFlowRouting>({
      label: 'Flow routing (compile-time, rebuilds the graph)',
      options: [
        {value: 'd8', label: 'D8 (steepest receiver)'},
        {value: 'd-infinity', label: 'D-infinity (Tarboton)'},
        {value: 'mfd-freeman', label: 'MFD Freeman (tan^p)'},
        {value: 'mfd-quinn', label: 'MFD Quinn (tan * contour length)'}
      ],
      value: flowRouting,
      onChange: value => {
        flowRouting = value;
        rebuildDrainageGraph();
      }
    });
    context.controls.addToggle({
      label: 'Resolve flats (compile-time)',
      value: resolveFlats,
      onChange: value => {
        resolveFlats = value;
        rebuildDrainageGraph();
      }
    });
    context.controls.addSlider({
      label: 'Flow exponent p (per-frame, MFD only)',
      min: 0,
      max: 4,
      step: 0.1,
      value: flowExponent,
      format: value => (value === 0 ? 'published (1.1 Freeman, 1 Quinn)' : value.toFixed(1)),
      onChange: value => {
        flowExponent = value;
        writeFlowSettings();
      }
    });
    context.controls.addSlider({
      label: 'Stream threshold (per-frame)',
      min: 1,
      max: 4,
      step: 0.1,
      value: thresholdExponent,
      format: value => {
        const cells = Math.round(10 ** value);
        return `${formatCount(cells)} cells, ${(cells * cellAreaSquareMeters * 1e-6).toFixed(2)} km²`;
      },
      onChange: value => {
        thresholdExponent = value;
        writeFlowSettings();
      }
    });
    context.controls.addSlider({
      label: 'Fill epsilon (per-frame; 0 leaves flats)',
      min: 0,
      max: FILL_EPSILONS.length - 1,
      step: 1,
      value: epsilonIndex,
      format: value => (value === 0 ? '0 (flats)' : `${FILL_EPSILONS[value]} m per step`),
      onChange: value => {
        epsilonIndex = value;
        writeFlowSettings();
      }
    });
    context.controls.addToggle({
      label: 'Stream overlay',
      value: showStreams,
      onChange: value => {
        showStreams = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'HAND color range (per-frame display)',
      min: 10,
      max: 200,
      step: 10,
      value: handMaximum,
      format: value => `0 to ${value} m`,
      onChange: value => {
        handMaximum = value;
        context.updateLayers();
      }
    });
    context.controls.addSelect<IndexKind>({
      label: 'Hydrologic index',
      options: [
        {value: 'wetness', label: 'Topographic wetness index ln(a / tan b)'},
        {value: 'catchment', label: 'Specific catchment area (log10 m)'},
        {value: 'power', label: 'Stream power index (log10 m)'}
      ],
      value: indexKind,
      onChange: value => {
        indexKind = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Wind speed (per-frame)',
      min: 1,
      max: 25,
      step: 1,
      value: windSpeed,
      format: value => `${value} m/s`,
      onChange: value => {
        windSpeed = value;
        writeWindSettings();
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Wind blows toward (per-frame)',
      min: 0,
      max: 355,
      step: 5,
      value: windBearingDegrees,
      format: value => `${value}° (0 north, 90 east)`,
      onChange: value => {
        windBearingDegrees = value;
        writeWindSettings();
      }
    });
    context.controls.addSlider({
      label: 'Terrain deflection exaggeration (per-frame)',
      min: 1,
      max: 20,
      step: 1,
      value: verticalExaggeration,
      format: value => `${value}x`,
      onChange: value => {
        verticalExaggeration = value;
        writeWindSettings();
      }
    });
    context.controls.addButton({
      label: 'Clear pour points',
      onClick: () => {
        pourPoints.length = 0;
        writePourPoints();
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Log10 contributing area (m²)',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '1 km²',
        maximumLabel: '100 km²'
      }
    });
    context.controls.addLegend({
      title: 'Strahler order',
      entries: [1, 2, 3, 4, 5, 6].map(order => ({color: ORDER_PALETTE[order], label: `${order}`}))
    });
    context.controls.addLegend({
      title: 'Cell class',
      entries: [
        {color: CELL_CLASS_PALETTE[1], label: 'flat'},
        {color: CELL_CLASS_PALETTE[2], label: 'pit'},
        {color: CELL_CLASS_PALETTE[3], label: 'outlet'}
      ]
    });
    context.controls.addNote(
      'Sea is no data, so water leaves the grid at the coast. Fill epsilon 0 leaves filled ' +
        'depressions and 2 m DEM plateaus as flats; resolve flats routes flow across them. ' +
        'Click the map to add watershed pour points (snapped to the highest accumulation ' +
        'within 5 cells); with none, basins are colored by outlet. Wind: slope-parallel ' +
        'component removed, exaggeration scales the gradient.'
    );
    context.controls.addReadout('Raster', `${width} × ${height} cells`);
    context.controls.addReadout(
      'Cell size',
      `${cellSize[0].toFixed(1)} × ${cellSize[1].toFixed(1)} m`
    );
    const convergedHandle = context.controls.addReadout('Converged', 'pending');
    const streamHandle = context.controls.addReadout('Stream cells', 'pending');
    const orderHandle = context.controls.addReadout('Cells by Strahler order', 'pending');
    const pourPointHandle = context.controls.addReadout('Pour points', 'none (drainage basins)');
    const timingHandles = {
      setup: context.controls.addReadout('Setup graph (once)', 'measuring...'),
      drainage: context.controls.addReadout('Drainage graph (on change)', 'measuring...'),
      wind: context.controls.addReadout('Wind field graph', 'measuring...'),
      particles: context.controls.addReadout('Particle graph (every frame)', 'measuring...')
    };
    context.controls.addReadout('Data', terrain.attribution);

    function rebuildDrainageGraph(): void {
      const previous = compiledDrainage;
      compiledDrainage = resources.track(compileDrainageGraph(flowRouting, resolveFlats));
      resources.release(previous);
      rebuildCount++;
      drainageDirty = true;
      timingHandles.drainage.setValue(`rebuilt (${rebuildCount}x), measuring...`);
      void measureDrainage();
    }

    const measureDrainage = async () => {
      try {
        const options = {
          parameters: undefined,
          completionBuffer: flagBuffers[0],
          signal: context.signal
        };
        const graph = compiledDrainage;
        const timing = await measureCompiledGraph(device, graph, options);
        timingHandles.drainage.setValue(
          `${graph.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(timing)}`
        );
      } catch (error) {
        if (!destroyed) timingHandles.drainage.setValue(`failed: ${(error as Error).message}`);
      }
    };
    const measureAll = async () => {
      try {
        const options = {
          parameters: undefined,
          completionBuffer: flagBuffers[0],
          signal: context.signal,
          runs: 3,
          warmUpRuns: 1,
          repetitions: 2
        };
        for (const [handle, graph] of [
          [timingHandles.setup, compiledSetup],
          [timingHandles.wind, compiledWind],
          [timingHandles.particles, compiledParticles]
        ] as const) {
          const timing = await measureCompiledGraph(device, graph, options);
          handle.setValue(
            `${graph.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(timing)}`
          );
        }
        await measureDrainage();
      } catch (error) {
        if (!destroyed) timingHandles.setup.setValue(`failed: ${(error as Error).message}`);
      }
    };
    context.controls.addButton({label: 'Measure GPU cost', onClick: () => void measureAll()});

    // --- Pour points ---------------------------------------------------------------------------
    const addPourPoint = async (event: SpatialAnalysisPointerEvent) => {
      if (!event.coordinate) return;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const column = Math.floor((x - bounds[0]) / cellSize[0]);
      const row = Math.floor((bounds[3] - y) / cellSize[1]);
      if (column < 0 || column >= width || row < 0 || row >= height) return;
      // Snap to the highest accumulation nearby so a click near a stream lands on the stream.
      const bytes = await accumulationBuffer.readAsync();
      if (destroyed) return;
      const accumulation = new Float32Array(bytes.buffer, bytes.byteOffset, cellCount);
      let best = -1;
      let bestValue = -Infinity;
      for (let dRow = -POUR_POINT_SNAP_RADIUS; dRow <= POUR_POINT_SNAP_RADIUS; dRow++) {
        for (let dColumn = -POUR_POINT_SNAP_RADIUS; dColumn <= POUR_POINT_SNAP_RADIUS; dColumn++) {
          const candidateRow = row + dRow;
          const candidateColumn = column + dColumn;
          if (
            candidateRow < 0 ||
            candidateRow >= height ||
            candidateColumn < 0 ||
            candidateColumn >= width
          ) {
            continue;
          }
          const index = candidateRow * width + candidateColumn;
          const value = accumulation[index];
          if (validityValues[index] && Number.isFinite(value) && value > bestValue) {
            best = index;
            bestValue = value;
          }
        }
      }
      if (best < 0) return;
      if (pourPoints.length >= POUR_POINT_CAPACITY) pourPoints.shift();
      pourPoints.push(best);
      writePourPoints();
      context.updateLayers();
    };

    // --- Per-frame encode ----------------------------------------------------------------------
    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiledSetup, compiledDrainage, compiledWind, compiledParticles],
      encode(commandEncoder, frame) {
        if (frame.frameIndex < 2) {
          compiledSetup.encode(commandEncoder, {parameters: undefined});
        }
        if (drainageDirty || frame.frameIndex < 2) {
          compiledDrainage.encode(commandEncoder, {parameters: undefined});
          drainageDirty = false;
          summary.request(commandEncoder);
        }
        if (product === 'wind') {
          if (windDirty || frame.frameIndex < 2) {
            compiledWind.encode(commandEncoder, {parameters: undefined});
            windDirty = false;
          }
          particleParameters.write(
            getGPUParticleAdvectionParameterValues(
              {
                fieldExtent: [bounds[0], bounds[1], cellSize[0], cellSize[1]],
                timeStep: PARTICLE_TIME_STEP,
                speedScale: 1,
                dropRate: PARTICLE_DROP_RATE,
                minimumSpeed: 0.05
              },
              [width, height]
            )
          );
          particleWords.write(
            getGPUParticleAdvectionWordParameterValues({
              seed: PARTICLE_SEED,
              frame: particleFrame,
              maximumAge: PARTICLE_MAXIMUM_AGE,
              reset: resetParticles
            })
          );
          compiledParticles.encode(commandEncoder, {parameters: undefined});
          particleFrame++;
          resetParticles = false;
        }
        summary.flush(commandEncoder);
        if (!timingsRequested && frame.frameIndex > 2) {
          timingsRequested = true;
          setTimeout(() => {
            if (!destroyed) void measureAll();
          }, MEASURE_DELAY_MILLISECONDS);
        }
      },
      getLayers() {
        pourPointHandle.setValue(
          pourPoints.length === 0
            ? 'none (drainage basins)'
            : `${pourPoints.length} of ${POUR_POINT_CAPACITY} (click adds, oldest dropped)`
        );
        const rasterProps = {
          coordinateOrigin: origin,
          gridSize: [width, height] as const,
          bounds,
          rowOrigin: 'north' as const
        };
        const layers: Layer[] = [
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id: 'drainage-hillshade',
            values: hillshadeBuffer,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 1],
            color: [255, 255, 255, product === 'wind' ? 255 : 150]
          })
        ];
        const scalar = (
          id: string,
          values: Buffer,
          colormap: 'viridis' | 'inferno',
          valueRange: readonly [number, number],
          discardAtOrBelow: number
        ) =>
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id,
            values,
            valueFormat: 'float32',
            colormap,
            valueRange,
            discardAtOrBelow,
            color: [255, 255, 255, 225]
          });
        const categorical = (
          id: string,
          values: Buffer,
          palette: readonly (readonly [number, number, number, number])[]
        ) =>
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id,
            values,
            valueFormat: 'uint32',
            colormap: 'category',
            palette,
            noDataColor: [0, 0, 0, 0]
          });
        if (product === 'accumulation') {
          layers.push(
            scalar('drainage-accumulation', accumulationDisplayBuffer, 'viridis', [3.2, 7], 0)
          );
        } else if (product === 'cell-classes') {
          layers.push(categorical('drainage-classes', classesBuffer, CELL_CLASS_PALETTE));
        } else if (product === 'hand') {
          layers.push(
            scalar('drainage-hand', handBuffer, 'inferno', [0, handMaximum], HIDDEN_VALUE / 2)
          );
        } else if (product === 'watersheds') {
          layers.push(
            categorical(
              'drainage-watersheds',
              pourPoints.length > 0 ? watershedLabelsBuffer : basinClassBuffer,
              BASIN_PALETTE
            )
          );
        } else if (product === 'indices') {
          if (indexKind === 'wetness') {
            layers.push(scalar('drainage-wetness', wetnessDisplayBuffer, 'viridis', [2, 16], -50));
          } else if (indexKind === 'catchment') {
            layers.push(
              scalar('drainage-catchment', catchmentDisplayBuffer, 'viridis', [1, 4.5], -50)
            );
          } else {
            layers.push(scalar('drainage-power', powerDisplayBuffer, 'inferno', [-2, 3], -50));
          }
        }
        if (product === 'stream-order') {
          ORDER_WIDTHS.forEach((widthPixels, index) => {
            layers.push(
              new SpatialAnalysisSegmentLayer({
                id: `drainage-order-${index}`,
                coordinateOrigin: origin,
                segments: segmentsBuffer,
                instanceCount: cellCount,
                values: streamOrderBuffer,
                valueFormat: 'uint32',
                colormap: 'category',
                palette: ORDER_PALETTE,
                weights: widthBuffers[index],
                widthPixels
              })
            );
          });
        } else if (
          showStreams &&
          product !== 'wind' &&
          product !== 'accumulation' &&
          product !== 'cell-classes'
        ) {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: 'drainage-streams',
              values: streamsBuffer,
              valueFormat: 'uint32',
              colormap: 'mask',
              color: [40, 170, 255, 255],
              noDataColor: [0, 0, 0, 0]
            })
          );
        }
        if (product === 'wind') {
          layers.push(
            new FlowTrailLayer({
              id: 'drainage-trails',
              coordinateOrigin: origin,
              trailPositions: trailBuffer,
              speeds: speedBuffer,
              wordParameters: particleWords.buffer,
              ringLength: TRAIL_LENGTH,
              particleCount: PARTICLE_COUNT,
              speedRange: [0, windSpeed * 1.4],
              widthPixels: 1.5,
              opacity: 0.95
            })
          );
        }
        if (product === 'watersheds' && pourPoints.length > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'drainage-pour-halo',
              coordinateOrigin: origin,
              positions: pourPointPositionsBuffer,
              instanceCount: pourPoints.length,
              radiusPixels: 9,
              color: [255, 255, 255, 255]
            }),
            new SpatialAnalysisPointLayer({
              id: 'drainage-pour-points',
              coordinateOrigin: origin,
              positions: pourPointPositionsBuffer,
              instanceCount: pourPoints.length,
              radiusPixels: 6,
              values: pourPointIndexBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: BASIN_PALETTE.map(([r, g, b]) => [r, g, b, 255] as const)
            })
          );
        }
        return layers;
      },
      onClick(event) {
        if (product !== 'watersheds') return false;
        void addPourPoint(event);
        return true;
      },
      getTooltip: () => null,
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    // FLOW_SPEED_RAMP is the trail layer's built-in ramp; referenced so the legend can match it.
    context.controls.addLegend({
      title: 'Wind speed',
      gradient: {
        colors: FLOW_SPEED_RAMP,
        minimumLabel: '0',
        maximumLabel: '1.4 × wind'
      }
    });
    return instance;
  }
};

const NAN_DECLARATIONS = /* wgsl */ `
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn isInfiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) == 0x7f800000u; }`;

/**
 * Adds `output = value` or `log10(value)` (floored at 1e-6) with a finite sentinel for NaN and
 * infinite cells; the raster layer hides rows at or below `discardAtOrBelow`.
 */
function addDisplayPass(
  graph: GPUCommandGraph<void>,
  props: {
    id: string;
    cellCount: number;
    input: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
    useLog: boolean;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    bindings: [
      {name: 'source', view: props.input, type: 'f32', access: 'read'},
      {name: 'display', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let value = source[sourceOffset + index];
  let invalid = isNaNValue(value) || isInfiniteValue(value);
  ${props.useLog ? 'let shown = log2(max(value, 0.000001)) * 0.30102999566;' : 'let shown = value;'}
  display[displayOffset + index] = select(shown, ${HIDDEN_VALUE.toFixed(1)}, invalid);`
  });
}

/**
 * Adds `class = hash(label) % 8` for drainage basin labels so neighboring basins differ in color;
 * the no-data label stays as is.
 */
function addBasinClassPass(
  graph: GPUCommandGraph<void>,
  props: {
    cellCount: number;
    labels: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
  }
): void {
  addKernelPass(graph, {
    id: 'basin-class',
    bindings: [
      {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
      {name: 'classes', view: props.output, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    body: `let label = labels[labelsOffset + index];
  classes[classesOffset + index] =
    select(((label * 2654435761u) >> 13u) % 8u, 0xffffffffu, label == 0xffffffffu);`
  });
}

/**
 * Adds one segment per cell from its center to its D8 receiver's center (meters, y up) and four
 * alpha weights that select the segments of each Strahler width class, so one layer per class can
 * draw stream lines of order-dependent width.
 */
function addStreamSegmentPass(
  graph: GPUCommandGraph<void>,
  props: {
    cellCount: number;
    width: number;
    bounds: readonly [number, number, number, number];
    cellSize: readonly [number, number];
    directions: GraphDataView<'uint32'>;
    streamOrder: GraphDataView<'uint32'>;
    segments: GraphDataView<'float32'>;
    widths: readonly GraphDataView<'float32'>[];
  }
): void {
  const meters = (value: number) => value.toFixed(4);
  addKernelPass(graph, {
    id: 'stream-segments',
    bindings: [
      {name: 'directions', view: props.directions, type: 'u32', access: 'read'},
      {name: 'orders', view: props.streamOrder, type: 'u32', access: 'read'},
      {name: 'segments', view: props.segments, type: 'f32', access: 'read_write'},
      ...props.widths.map((view, index) => ({
        name: `width${index}`,
        view,
        type: 'f32' as const,
        access: 'read_write' as const
      }))
    ],
    invocationCount: props.cellCount,
    declarations: `
const GRID_WIDTH: u32 = ${props.width}u;
const MINIMUM_X: f32 = ${meters(props.bounds[0])};
const MAXIMUM_Y: f32 = ${meters(props.bounds[3])};
const CELL_X: f32 = ${meters(props.cellSize[0])};
const CELL_Y: f32 = ${meters(props.cellSize[1])};`,
    body: `let column = index % GRID_WIDTH;
  let row = index / GRID_WIDTH;
  let code = directions[directionsOffset + index];
  var columnStep = 0.0;
  var rowStep = 0.0;
  if (code == 1u) { columnStep = 1.0; }
  else if (code == 2u) { columnStep = 1.0; rowStep = 1.0; }
  else if (code == 4u) { rowStep = 1.0; }
  else if (code == 8u) { columnStep = -1.0; rowStep = 1.0; }
  else if (code == 16u) { columnStep = -1.0; }
  else if (code == 32u) { columnStep = -1.0; rowStep = -1.0; }
  else if (code == 64u) { rowStep = -1.0; }
  else if (code == 128u) { columnStep = 1.0; rowStep = -1.0; }
  let startX = MINIMUM_X + (f32(column) + 0.5) * CELL_X;
  let startY = MAXIMUM_Y - (f32(row) + 0.5) * CELL_Y;
  let base = segmentsOffset + 4u * index;
  segments[base] = startX;
  segments[base + 1u] = startY;
  segments[base + 2u] = startX + columnStep * CELL_X;
  segments[base + 3u] = startY - rowStep * CELL_Y;
  let order = orders[ordersOffset + index];
  let isStream = order > 0u && order != 0xffffffffu;
  width0[width0Offset + index] = select(0.0, 1.0, isStream && order == 1u);
  width1[width1Offset + index] = select(0.0, 1.0, isStream && order == 2u);
  width2[width2Offset + index] = select(0.0, 1.0, isStream && order == 3u);
  width3[width3Offset + index] = select(0.0, 1.0, isStream && order >= 4u);`
  });
}
