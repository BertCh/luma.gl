// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainFlowParameterValues,
  getGPUTerrainHydrologicIndicesParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
  GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH,
  GPUTerrainDerivatives,
  GPUTerrainFlow,
  GPUTerrainHeightAboveDrainage,
  GPUTerrainHydrologicIndices,
  GPUTerrainStreamOrder,
  GPUTerrainWatersheds,
  type GPUTerrainFlowRouting
} from '@luma.gl/experimental/gpu-terrain';
import {
  GPUCommandGraph,
  GPUHistogram,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  addFillDepthPass,
  addLogDisplayPass,
  addStreamSegmentPass,
  createCanyonGrid,
  formatInteger,
  formatKilometers,
  getCellAt,
  getCellCenter,
  getGridSettings,
  HIDDEN_VALUE,
  NAN_DECLARATIONS
} from './b15-common';

/** Option state of the watersheds scene. */
export type WatershedsOptions = {
  product: 'accumulation' | 'fill' | 'watersheds' | 'stream-order' | 'hand' | 'flood' | 'indices';
  routing: GPUTerrainFlowRouting;
  resolveFlats: boolean;
  flowExponent: number;
  streamArea: number;
  fillEpsilon: number;
  showStreams: boolean;
  basinMode: 'pour-points' | 'outlets';
  handRange: number;
  floodStage: number;
  indexKind: 'wetness' | 'catchment' | 'power';
  minimumSlope: number;
  overlayOpacity: number;
};

/** Grid reduction: 1 keeps the 15 m DEM, 2 averages 2 x 2 blocks to 31 m. */
export const WATERSHEDS_GRID_STRIDE = 1;
/** Compile-time iteration limits of the relaxations. */
const MAXIMUM_FILL_ITERATIONS = 256;
const MAXIMUM_FLAT_ITERATIONS = 256;
const MAXIMUM_ACCUMULATION_ITERATIONS = 128;
/** Elevation per fill step in meters; index 0 leaves filled depressions as exact flats. */
export const FILL_EPSILONS = [0, 0.0001, 0.001, 0.01, 0.1] as const;
const SUN_AZIMUTH_DEGREES = 315;
const SUN_ALTITUDE_DEGREES = 40;
const POUR_POINT_CAPACITY = 8;
const UNUSED_POUR_POINT = 0xffffffff;
/** Cells searched around a click for the cell of highest accumulation. */
const POUR_POINT_SNAP_RADIUS = 5;
const ORDER_BIN_COUNT = 8;

/** Pour points the story starts from: Bright Angel Creek, Garden Creek and the Colorado. */
export const DEFAULT_POUR_POINTS: readonly {
  name: string;
  lngLat: [number, number];
  /** Cells searched for the stream, and the largest drainage area (km²) it may have. */
  snapRadius: number;
  maximumAreaKm2: number;
}[] = [
  {
    name: 'Phantom Ranch (Bright Angel Creek)',
    lngLat: [-112.0953, 36.107],
    snapRadius: 16,
    maximumAreaKm2: 300
  },
  {
    name: 'Indian Garden (Garden Creek)',
    lngLat: [-112.1262, 36.0771],
    snapRadius: 24,
    maximumAreaKm2: 120
  },
  {
    name: 'Colorado River at Bright Angel Creek',
    lngLat: [-112.093, 36.106],
    snapRadius: 16,
    maximumAreaKm2: Infinity
  }
];

/** Colors of watershed labels and pour points. */
export const BASIN_PALETTE = [
  [230, 90, 90, 215],
  [240, 170, 60, 215],
  [225, 215, 80, 215],
  [110, 205, 100, 215],
  [70, 200, 190, 215],
  [80, 140, 240, 215],
  [150, 110, 235, 215],
  [225, 100, 200, 215]
] as const;
/** Strahler orders 0 (none) to 7. */
export const ORDER_PALETTE = [
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
/** Fixed color range of the log10 accumulation display, log10 of square meters. */
export const ACCUMULATION_LOG_RANGE: readonly [number, number] = [4, 8.8];
export const WETNESS_RANGE: readonly [number, number] = [2, 18];
export const CATCHMENT_LOG_RANGE: readonly [number, number] = [0.5, 3.5];
export const POWER_LOG_RANGE: readonly [number, number] = [-1, 3];

/**
 * Drainage of the central Grand Canyon on the GPU. One compiled graph chains `GPUTerrainFlow`
 * (fill, flat resolution, routing and accumulation, streams), `GPUTerrainHeightAboveDrainage`,
 * `GPUTerrainWatersheds` (outlet basins and pour-point watersheds), `GPUTerrainStreamOrder` and
 * `GPUTerrainHydrologicIndices`. Routing and flat resolution are compile-time and rebuild the graph;
 * everything else is a parameter-buffer write and the graph re-encodes only when an input changed.
 */
export async function createWatersheds(
  ctx: SceneContext<WatershedsOptions>
): Promise<SceneInstance<WatershedsOptions>> {
  const {device} = ctx;
  const grid = createCanyonGrid(ctx.datasets.get('grand-canyon-dem'), WATERSHEDS_GRID_STRIDE);
  const {width, height, cellCount} = grid;
  const origin: [number, number, number] = [grid.origin[0], grid.origin[1], 0];
  const cellAreaSquareMeters = grid.groundCellSize[0] * grid.groundCellSize[1];
  const resources = new SpatialAnalysisResources(device, 'watersheds');
  let destroyed = false;

  const float32Cells = (name: string) => resources.createBuffer(name, cellCount * 4);
  const elevationBuffer = resources.createBuffer('elevation', grid.elevation);
  const hillshadeBuffer = float32Cells('hillshade');
  const filledBuffer = float32Cells('filled');
  const directionsBuffer = float32Cells('directions');
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
  const indexDisplayBuffer = float32Cells('index-display');
  const basinClassBuffer = float32Cells('basin-class');
  const fillDepthBuffer = float32Cells('fill-depth');
  const floodDepthBuffer = float32Cells('flood-depth');
  const floodMaskBuffer = float32Cells('flood-mask');
  const segmentsBuffer = resources.createBuffer('stream-segments', cellCount * 16);
  const widthBuffers = ORDER_WIDTHS.map((_, index) => float32Cells(`stream-width-${index}`));
  const flagNames = ['fill', 'flats', 'accumulation', 'hand', 'basins', 'watersheds', 'order'];
  const flagBuffers = flagNames.map(name => resources.createBuffer(`${name}-converged`, 4));
  const iterationNames = ['fill', 'flats', 'accumulation', 'order'];
  const iterationBuffers = iterationNames.map(name =>
    resources.createBuffer(`${name}-iterations`, 4)
  );
  const streamCountsBuffer = resources.createBuffer('stream-counts', 8);
  const floodCountsBuffer = resources.createBuffer('flood-counts', 8);
  const orderHistogramBuffer = resources.createBuffer('order-histogram', ORDER_BIN_COUNT * 4);
  const watershedHistogramBuffer = resources.createBuffer(
    'watershed-histogram',
    POUR_POINT_CAPACITY * 4
  );
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
  const displayParameters = resources.createParameterBuffer('display-parameters', 'float32', 2);
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

  // --- Setup graph: hillshade backdrop, computed once -------------------------------------------
  const setupGraph = new GPUCommandGraph<void>(device, {id: 'watersheds-setup'});
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
          values: importGraphBuffer(setupGraph, 'elevation', elevationBuffer, 'float32', cellCount)
        }
      },
      settings: derivativesSettings.importToGraph(setupGraph),
      hillshade: importGraphBuffer(setupGraph, 'hillshade', hillshadeBuffer, 'float32', cellCount),
      cellSizeMode: grid.cellSizeMode,
      rowDirection: 'south'
    })
  );
  const compiledSetup: CompiledGPUCommandGraph<void> = resources.track(setupGraph.compile());

  /** Builds the drainage graph for one routing and flat-resolution choice (both compile-time). */
  function compileDrainageGraph(
    flowRouting: GPUTerrainFlowRouting,
    resolveFlats: boolean
  ): CompiledGPUCommandGraph<void> {
    const graph = new GPUCommandGraph<void>(device, {id: 'watersheds'});
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
          storage: {kind: 'buffer' as const, values: elevationView}
        },
        settings: flowSettings.importToGraph(graph),
        cellSizeMode: grid.cellSizeMode,
        fillDepressions: true,
        maxFillIterations: MAXIMUM_FILL_ITERATIONS,
        resolveFlats,
        maxFlatIterations: MAXIMUM_FLAT_ITERATIONS,
        flowRouting,
        maxAccumulationIterations: MAXIMUM_ACCUMULATION_ITERATIONS,
        accumulationUnits: 'area',
        filledElevation: filled,
        flowDirections: directions,
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
        cellSizeMode: grid.cellSizeMode,
        specificCatchmentArea: view('catchment', catchmentBuffer, 'float32'),
        wetnessIndex: view('wetness', wetnessBuffer, 'float32'),
        streamPowerIndex: view('power', powerBuffer, 'float32')
      })
    );
    // Display products ----------------------------------------------------------------------
    const displayView = displayParameters.importToGraph(graph);
    addLogDisplayPass(graph, {
      id: 'accumulation-display',
      cellCount,
      input: accumulation,
      output: view('accumulation-display', accumulationDisplayBuffer, 'float32'),
      useLog: true
    });
    addIndexDisplayPass(graph, {
      cellCount,
      wetness: view('wetness', wetnessBuffer, 'float32'),
      catchment: view('catchment', catchmentBuffer, 'float32'),
      power: view('power', powerBuffer, 'float32'),
      kind: displayView,
      output: view('index-display', indexDisplayBuffer, 'float32')
    });
    addFillDepthPass(graph, {
      id: 'fill-depth',
      cellCount,
      elevation: elevationView as GraphDataView<'float32'>,
      filled,
      output: view('fill-depth', fillDepthBuffer, 'float32')
    });
    addFloodPass(graph, {
      cellCount,
      hand: view('hand', handBuffer, 'float32'),
      stage: displayView,
      depth: view('flood-depth', floodDepthBuffer, 'float32'),
      mask: view('flood-mask', floodMaskBuffer, 'uint32')
    });
    addBasinClassPass(graph, {
      cellCount,
      labels: view('basin-labels', basinLabelsBuffer, 'uint32'),
      output: view('basin-class', basinClassBuffer, 'uint32')
    });
    addStreamSegmentPass(graph, {
      cellCount,
      grid,
      directions,
      streamOrder,
      segments: view('stream-segments', segmentsBuffer, 'float32', cellCount * 4),
      widths: widthBuffers.map((buffer, index) => view(`stream-width-${index}`, buffer, 'float32'))
    });
    // Summaries -----------------------------------------------------------------------------
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
        id: 'flood-count',
        input: view('flood-mask', floodMaskBuffer, 'uint32'),
        output: view('flood-counts', floodCountsBuffer, 'uint32', 2),
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
    graph.add(
      new GPUHistogram({
        id: 'watershed-histogram',
        input: view('watershed-labels', watershedLabelsBuffer, 'uint32'),
        output: view(
          'watershed-histogram',
          watershedHistogramBuffer,
          'uint32',
          POUR_POINT_CAPACITY
        ),
        domain: [0, POUR_POINT_CAPACITY]
      })
    );
    return graph.compile();
  }

  // --- State --------------------------------------------------------------------------------
  let builtRouting = ctx.options.routing;
  let builtResolveFlats = ctx.options.resolveFlats;
  let compiledDrainage = resources.track(compileDrainageGraph(builtRouting, builtResolveFlats));
  let drainageDirty = true;
  const pourPoints: number[] = [];
  let defaultsSeeded = false;
  let landCellCount = cellCount;

  const getStreamThreshold = () => 10 ** ctx.options.streamArea * 1e6;
  const writeFlowSettings = () => {
    flowSettings.write(
      getGPUTerrainFlowParameterValues({
        ...getGridSettings(grid),
        fillEpsilon: FILL_EPSILONS[ctx.options.fillEpsilon] ?? 0,
        streamThreshold: getStreamThreshold(),
        flowExponent: ctx.options.flowExponent
      })
    );
    drainageDirty = true;
  };
  const writeIndicesSettings = () => {
    indicesSettings.write(
      getGPUTerrainHydrologicIndicesParameterValues({
        ...getGridSettings(grid),
        minimumSlope: ctx.options.minimumSlope / 100
      })
    );
    drainageDirty = true;
  };
  const writeDisplayParameters = () => {
    const kind = {wetness: 0, catchment: 1, power: 2}[ctx.options.indexKind];
    displayParameters.write(Float32Array.of(kind, ctx.options.floodStage));
    drainageDirty = true;
  };
  const writePourPoints = () => {
    pourPointValues.fill(UNUSED_POUR_POINT);
    const positions = new Float32Array(POUR_POINT_CAPACITY * 2);
    pourPoints.forEach((cell, index) => {
      pourPointValues[index] = cell;
      const [x, y] = getCellCenter(grid, cell);
      positions[2 * index] = x;
      positions[2 * index + 1] = y;
    });
    pourPointBuffer.write(pourPointValues);
    pourPointPositionsBuffer.write(positions);
    drainageDirty = true;
    ctx.setReadout('pourPoints', pourPoints.length);
    ctx.requestLayers();
  };
  derivativesSettings.write(
    getGPUTerrainDerivativesParameterValues({
      ...getGridSettings(grid),
      azimuthDegrees: SUN_AZIMUTH_DEGREES,
      altitudeDegrees: SUN_ALTITUDE_DEGREES
    })
  );
  writeFlowSettings();
  writeIndicesSettings();
  writeDisplayParameters();

  ctx.setReadout('grid', `${width} × ${height} cells`);
  ctx.setReadout(
    'cellSize',
    `${grid.groundCellSize[0].toFixed(1)} m (${formatInteger(cellAreaSquareMeters)} m² per cell)`
  );
  let minimumElevation = Infinity;
  let maximumElevation = -Infinity;
  for (const value of grid.elevation) {
    if (value < minimumElevation) minimumElevation = value;
    if (value > maximumElevation) maximumElevation = value;
  }
  ctx.setReadout(
    'relief',
    `${formatInteger(minimumElevation)} to ${formatInteger(maximumElevation)} m (${formatInteger(maximumElevation - minimumElevation)} m)`
  );
  ctx.setReadout('pourPoints', 0);

  const summary = new SummaryReader(
    resources,
    'watersheds',
    [
      ...flagBuffers.map(buffer => ({buffer, size: 4})),
      {buffer: streamCountsBuffer, size: 8},
      {buffer: orderHistogramBuffer, size: ORDER_BIN_COUNT * 4},
      ...iterationBuffers.map(buffer => ({buffer, size: 4})),
      {buffer: watershedHistogramBuffer, size: POUR_POINT_CAPACITY * 4},
      {buffer: floodCountsBuffer, size: 8}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const iterationBase = 9 + ORDER_BIN_COUNT;
      const watershedBase = iterationBase + 4;
      const floodBase = watershedBase + POUR_POINT_CAPACITY;
      const flag = (index: number, label: string, limit: number, iterationIndex: number) =>
        `${label} ${words[index] ? 'yes' : 'NO'} ${words[iterationBase + iterationIndex]}/${limit}`;
      ctx.setReadout(
        'converged',
        [
          flag(0, 'fill', MAXIMUM_FILL_ITERATIONS, 0),
          builtResolveFlats ? flag(1, 'flats', MAXIMUM_FLAT_ITERATIONS, 1) : 'flats off',
          flag(2, 'accumulation', MAXIMUM_ACCUMULATION_ITERATIONS, 2),
          words[3] && words[4] && words[5] && words[6] ? 'tracing yes' : 'tracing NO',
          `order ${words[iterationBase + 3]} rounds`
        ].join(', ')
      );
      const streamCells = words[8];
      ctx.setReadout(
        'streams',
        `${formatInteger(streamCells)} cells, ${((streamCells / landCellCount) * 100).toFixed(1)} % of the window`
      );
      const orders = Array.from(words.slice(9 + 1, 9 + ORDER_BIN_COUNT));
      const maximumOrder = orders.reduce((last, count, index) => (count > 0 ? index + 1 : last), 0);
      ctx.setReadout('maximumOrder', maximumOrder);
      ctx.setReadout(
        'orders',
        orders
          .slice(0, Math.max(maximumOrder, 1))
          .map((count, index) => `${index + 1}: ${formatInteger(count)}`)
          .join('  ')
      );
      const watersheds = Array.from(words.slice(watershedBase, watershedBase + pourPoints.length));
      ctx.setReadout(
        'watershedAreas',
        watersheds.length === 0
          ? 'no pour points'
          : watersheds
              .map(
                (count, index) => `${index + 1}: ${formatKilometers(count * cellAreaSquareMeters)}`
              )
              .join('  ')
      );
      ctx.setReadout(
        'floodArea',
        `${formatKilometers(words[floodBase + 1] * cellAreaSquareMeters)} within ${ctx.options.floodStage} m of a stream`
      );
      if (!defaultsSeeded) {
        defaultsSeeded = true;
        void seedDefaultPourPoints();
      }
    }
  );

  /** Highest-accumulation cell within the snap radius, read from a small window of rows. */
  async function snapToStream(
    cell: number,
    radius = POUR_POINT_SNAP_RADIUS,
    maximumArea = Infinity
  ): Promise<number> {
    const column = cell % width;
    const row = Math.floor(cell / width);
    const firstRow = Math.max(0, row - radius);
    const lastRow = Math.min(height - 1, row + radius);
    const rowCount = lastRow - firstRow + 1;
    const bytes = await accumulationBuffer.readAsync(firstRow * width * 4, rowCount * width * 4);
    const window = new Float32Array(bytes.buffer, bytes.byteOffset, rowCount * width);
    let best = cell;
    let bestValue = -Infinity;
    for (let dRow = -radius; dRow <= radius; dRow++) {
      const candidateRow = row + dRow;
      if (candidateRow < firstRow || candidateRow > lastRow) continue;
      for (let dColumn = -radius; dColumn <= radius; dColumn++) {
        const candidateColumn = column + dColumn;
        if (candidateColumn < 0 || candidateColumn >= width) continue;
        const value = window[(candidateRow - firstRow) * width + candidateColumn];
        if (Number.isFinite(value) && value <= maximumArea && value > bestValue) {
          best = candidateRow * width + candidateColumn;
          bestValue = value;
        }
      }
    }
    return best;
  }

  async function addPourPoint(cell: number): Promise<void> {
    const snapped = await snapToStream(cell);
    if (destroyed) return;
    if (pourPoints.length >= POUR_POINT_CAPACITY) pourPoints.shift();
    pourPoints.push(snapped);
    writePourPoints();
  }

  async function seedDefaultPourPoints(): Promise<void> {
    try {
      for (const {lngLat, snapRadius, maximumAreaKm2} of DEFAULT_POUR_POINTS) {
        const cell = getCellAt(grid, lngLat[0], lngLat[1]);
        if (cell >= 0 && !destroyed) {
          const snapped = await snapToStream(cell, snapRadius, maximumAreaKm2 * 1e6);
          if (destroyed) return;
          pourPoints.push(snapped);
        }
      }
      writePourPoints();
    } catch {
      // The device was destroyed while the accumulation window was being read.
    }
  }

  function rebuildDrainageGraph(): void {
    const previous = compiledDrainage;
    builtRouting = ctx.options.routing;
    builtResolveFlats = ctx.options.resolveFlats;
    compiledDrainage = resources.track(compileDrainageGraph(builtRouting, builtResolveFlats));
    resources.release(previous);
    drainageDirty = true;
    ctx.requestLayers();
  }

  return {
    getCompiledGraphs: () => [compiledSetup, compiledDrainage],

    setOption(id, _value, state) {
      switch (id) {
        case 'routing':
        case 'resolveFlats':
          if (state.routing !== builtRouting || state.resolveFlats !== builtResolveFlats) {
            rebuildDrainageGraph();
          }
          break;
        case 'flowExponent':
        case 'streamArea':
        case 'fillEpsilon':
          writeFlowSettings();
          ctx.requestLayers();
          break;
        case 'minimumSlope':
          writeIndicesSettings();
          break;
        case 'indexKind':
        case 'floodStage':
          writeDisplayParameters();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'clearPourPoints') {
        pourPoints.length = 0;
        writePourPoints();
      } else if (id === 'resetPourPoints') {
        pourPoints.length = 0;
        void seedDefaultPourPoints();
      }
    },

    encode(commandEncoder, frame) {
      if (frame.frameIndex < 2) {
        compiledSetup.encode(commandEncoder, {parameters: undefined});
      }
      if (drainageDirty || frame.frameIndex < 2) {
        compiledDrainage.encode(commandEncoder, {parameters: undefined});
        drainageDirty = false;
        summary.request(commandEncoder);
      }
      summary.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const rasterProps = {
        coordinateOrigin: origin,
        gridSize: [width, height] as const,
        bounds: grid.bounds,
        rowOrigin: 'north' as const
      };
      const dark = ctx.theme() === 'dark';
      const alpha = Math.round(options.overlayOpacity * 255);
      const layers: Layer[] = [
        new SpatialAnalysisRasterLayer({
          ...rasterProps,
          id: 'watersheds-hillshade',
          values: hillshadeBuffer,
          valueFormat: 'float32',
          colormap: 'grayscale',
          valueRange: [0, 1],
          color: [255, 255, 255, dark ? 200 : 170]
        })
      ];
      const scalar = (
        id: string,
        values: Buffer,
        colormap: 'viridis' | 'inferno' | 'cividis',
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
          color: [255, 255, 255, alpha]
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
          noDataColor: [0, 0, 0, 0],
          color: [255, 255, 255, alpha]
        });
      switch (options.product) {
        case 'accumulation':
          layers.push(
            scalar(
              'watersheds-accumulation',
              accumulationDisplayBuffer,
              'viridis',
              ACCUMULATION_LOG_RANGE,
              // Hillslope cells below 0.01 km² stay transparent so the stream network reads.
              ACCUMULATION_LOG_RANGE[0]
            )
          );
          break;
        case 'fill':
          layers.push(scalar('watersheds-fill', fillDepthBuffer, 'inferno', [0, 10], 0.0005));
          break;
        case 'hand':
          layers.push(
            scalar(
              'watersheds-hand',
              handBuffer,
              'inferno',
              [0, options.handRange],
              HIDDEN_VALUE / 2
            )
          );
          break;
        case 'flood':
          layers.push(
            scalar('watersheds-flood', floodDepthBuffer, 'cividis', [0, options.floodStage], 0)
          );
          break;
        case 'watersheds':
          layers.push(
            categorical(
              'watersheds-labels',
              options.basinMode === 'pour-points' ? watershedLabelsBuffer : basinClassBuffer,
              BASIN_PALETTE
            )
          );
          break;
        case 'indices': {
          const range =
            options.indexKind === 'wetness'
              ? WETNESS_RANGE
              : options.indexKind === 'catchment'
                ? CATCHMENT_LOG_RANGE
                : POWER_LOG_RANGE;
          layers.push(
            scalar(
              'watersheds-index',
              indexDisplayBuffer,
              options.indexKind === 'power' ? 'inferno' : 'viridis',
              range,
              HIDDEN_VALUE / 2
            )
          );
          break;
        }
        case 'stream-order':
          ORDER_WIDTHS.forEach((widthPixels, index) => {
            layers.push(
              new SpatialAnalysisSegmentLayer({
                id: `watersheds-order-${index}`,
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
          break;
      }
      if (
        options.showStreams &&
        options.product !== 'stream-order' &&
        options.product !== 'accumulation'
      ) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id: 'watersheds-streams',
            values: streamsBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: dark ? [90, 190, 255, 255] : [20, 110, 230, 255],
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      if (options.product === 'watersheds' && options.basinMode === 'pour-points') {
        const markerCount = pourPoints.length;
        if (markerCount > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'watersheds-pour-halo',
              coordinateOrigin: origin,
              positions: pourPointPositionsBuffer,
              instanceCount: markerCount,
              radiusPixels: 9,
              color: [255, 255, 255, 255]
            }),
            new SpatialAnalysisPointLayer({
              id: 'watersheds-pour-points',
              coordinateOrigin: origin,
              positions: pourPointPositionsBuffer,
              instanceCount: markerCount,
              radiusPixels: 6,
              values: pourPointIndexBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: BASIN_PALETTE.map(([r, g, b]) => [r, g, b, 255] as const)
            })
          );
        }
      }
      return layers;
    },

    onClick(event) {
      if (ctx.options.product !== 'watersheds' || ctx.options.basinMode !== 'pour-points') {
        return false;
      }
      if (!event.coordinate) return false;
      const cell = getCellAt(grid, event.coordinate[0], event.coordinate[1]);
      if (cell < 0) return false;
      void addPourPoint(cell).catch(() => {});
      return true;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const cell = getCellAt(grid, event.coordinate[0], event.coordinate[1]);
      if (cell < 0) return null;
      return `Elevation ${formatInteger(grid.elevation[cell])} m\n${event.coordinate[1].toFixed(4)}°, ${event.coordinate[0].toFixed(4)}°`;
    },

    destroy() {
      destroyed = true;
      summary.stop();
      resources.destroy();
    }
  };
}

/**
 * `display = wetness | log10(specific catchment area) | log10(stream power)` selected by the first
 * value of `kind` (0, 1, 2), with {@link HIDDEN_VALUE} for NaN and infinite cells.
 */
function addIndexDisplayPass(
  graph: GPUCommandGraph<void>,
  props: {
    cellCount: number;
    wetness: GraphDataView<'float32'>;
    catchment: GraphDataView<'float32'>;
    power: GraphDataView<'float32'>;
    kind: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }
): void {
  addKernelPass(graph, {
    id: 'index-display',
    bindings: [
      {name: 'wetness', view: props.wetness, type: 'f32', access: 'read'},
      {name: 'catchment', view: props.catchment, type: 'f32', access: 'read'},
      {name: 'power', view: props.power, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.kind, type: 'f32', access: 'read'},
      {name: 'display', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let kind = u32(parameters[parametersOffset] + 0.5);
  var value = wetness[wetnessOffset + index];
  if (kind == 1u) { value = log2(max(catchment[catchmentOffset + index], 0.000001)) * 0.30102999566; }
  if (kind == 2u) { value = log2(max(power[powerOffset + index], 0.000001)) * 0.30102999566; }
  let invalid = isNaNValue(value) || isInfiniteValue(value);
  display[displayOffset + index] = select(value, ${HIDDEN_VALUE.toFixed(1)}, invalid);`
  });
}

/**
 * Flood at a stage: `depth = stage - HAND` where the cell is lower than `stage` above its drainage
 * (else 0, hidden by the layer) and `mask` is 1 there. `stage` is the second value of the view.
 */
function addFloodPass(
  graph: GPUCommandGraph<void>,
  props: {
    cellCount: number;
    hand: GraphDataView<'float32'>;
    stage: GraphDataView<'float32'>;
    depth: GraphDataView<'float32'>;
    mask: GraphDataView<'uint32'>;
  }
): void {
  addKernelPass(graph, {
    id: 'flood-depth',
    bindings: [
      {name: 'hand', view: props.hand, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.stage, type: 'f32', access: 'read'},
      {name: 'depth', view: props.depth, type: 'f32', access: 'read_write'},
      {name: 'mask', view: props.mask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let stage = parameters[parametersOffset + 1u];
  let height = hand[handOffset + index];
  let valid = !isNaNValue(height) && !isInfiniteValue(height) && height >= 0.0;
  let flooded = valid && height <= stage;
  depth[depthOffset + index] = select(0.0, max(stage - height, 0.0001), flooded);
  mask[maskOffset + index] = select(0u, 1u, flooded);`
  });
}

/** `class = hash(label) % 8` of outlet-basin labels so neighbouring basins differ in colour. */
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
