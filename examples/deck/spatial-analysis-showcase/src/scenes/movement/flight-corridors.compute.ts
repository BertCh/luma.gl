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
  GPU_TRAJECTORY_PLAYHEAD_STATUS,
  GPULineDensity,
  GPUParameterBuffer,
  GPUTrajectoryPlayhead
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  FLIGHT_DATASET_ID,
  FLIGHT_DIRECTION_COLORS,
  formatClockUtc,
  loadFlights
} from './flight-corridors-data';
import {FlightPointLayer, FlightSegmentLayer} from './flight-corridors-layers';
import {binValues, lineChart} from './f-chart-helpers';

/** Option state of the flight corridors scene. */
export type FlightCorridorsOptions = {
  playing: boolean;
  time: number;
  playbackSpeed: number;
  loop: boolean;
  maxGapMinutes: number;
  direction: 'all' | 'east' | 'west' | 'north' | 'south';
  showMarkers: boolean;
  markerColor: 'altitude' | 'direction';
  markerSize: number;
  showTrails: boolean;
  trailMinutes: number;
  tailFade: number;
  trailColor: 'altitude' | 'direction';
  showDensity: boolean;
  densityView: 'density' | 'direction-balance';
  cellDegrees: '0.2' | '0.1';
  densityRamp: RampName;
  densityOpacity: number;
  showBackdrop: boolean;
  altitudeFloor: number;
  extrude: boolean;
  exaggeration: number;
  altitudeRamp: 'viridis' | 'magma' | 'inferno' | 'cividis';
};

/** Altitude in metres at the end of the altitude ramps. */
export const ALTITUDE_RAMP_METERS = 13000;

const SECONDS_PER_DAY = 86400;
const STATUS_INTERVAL_FRAMES = 12;
const GRID_BOUNDS = [-125, 25, -65, 50] as const;
const DIRECTION_CODES = {all: -1, east: 0, west: 1, north: 2, south: 3} as const;
const EARTH_RADIUS = 6371008.8;
const HOURLY_BINS = 144;
const ALTITUDE_HISTOGRAM_BINS = 26;
/** Minimum combined east/west support: 0.05 km of track per km². */
const DIRECTION_BALANCE_MINIMUM_DENSITY = 0.05 / 1000;

type TrackSubset = {
  lngLat: Buffer;
  offsets: Buffer;
  vertexCount: number;
  trackCount: number;
  vertices: Float32Array;
  trackOffsets: Uint32Array;
};

type DensityGrid = {
  cell: number;
  columns: number;
  rows: number;
  parameters: GPUParameterBuffer<'float32'>;
  lengths: Buffer;
  densities: Buffer;
  overflow: Buffer;
  totalRecords: Buffer;
  capacity: Map<number, number>;
  reader: SummaryReader;
};

type DensityVariant = {
  compiled: CompiledGPUCommandGraph<void>;
  grid: DensityGrid;
  encoded: boolean;
};

/** One side of the paired east/west computation. Buffers deliberately never alias. */
type DirectionDensityGrid = Omit<DensityGrid, 'reader'>;

type DirectionBalanceVariant = {
  cell: number;
  columns: number;
  rows: number;
  east: DirectionDensityGrid;
  west: DirectionDensityGrid;
  balance: Buffer;
  eastCapacity: number;
  westCapacity: number;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  encoded: boolean;
};

/**
 * Flight corridors: one UTC day of US jet traffic. A playhead graph places every flight at the
 * clock (longitude, latitude and altitude), a time-window graph selects the live trail segments,
 * and a line-density graph sums track length per grid cell over the whole day, so the airways
 * appear as bright corridors. Bespoke 3-D layers draw it flat or extruded by altitude.
 */
export async function createFlightCorridors(
  ctx: SceneContext<FlightCorridorsOptions>
): Promise<SceneInstance<FlightCorridorsOptions>> {
  const flights = loadFlights(ctx.datasets.get(FLIGHT_DATASET_ID));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = flights;
  const resources = new SpatialAnalysisResources(device, 'flights');

  // ---- Static inputs ----------------------------------------------------------------------------
  const lngLatBuffer = resources.createBuffer('lng-lat', flights.lngLat);
  const altitudeBuffer = resources.createBuffer('altitude', flights.altitude);
  const timestampsBuffer = resources.createBuffer('timestamps', flights.timestamps);
  const offsetsBuffer = resources.createBuffer('offsets', flights.offsets);
  const endVerticesBuffer = resources.createBuffer('segment-ends', flights.segmentEndVertices);
  const segmentStartTimes = resources.createBuffer(
    'segment-start-times',
    flights.segmentStartTimes
  );
  const segmentEndTimes = resources.createBuffer('segment-end-times', flights.segmentEndTimes);
  const vertexDirectionBuffer = resources.createBuffer('vertex-direction', flights.vertexDirection);
  const trackDirectionBuffer = resources.createBuffer('track-direction', flights.trackDirection);
  const segmentMaskBuffer = resources.createBuffer(
    'segment-mask',
    new Uint32Array(segmentCount).fill(1)
  );

  // ---- Playhead graph ---------------------------------------------------------------------------
  const currentPositions = resources.createBuffer('current-positions', trackCount * 8);
  const currentElevations = resources.createBuffer('current-elevations', trackCount * 4);
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
      id: 'flight-marker-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'flight-playhead'});
  playheadGraph.add(
    new GPUTrajectoryPlayhead({
      id: 'playhead',
      // Interpolating longitude and latitude directly keeps the output drawable as degrees.
      positions: importGraphBuffer(
        playheadGraph,
        'lng-lat',
        lngLatBuffer,
        'float32x2',
        vertexCount
      ),
      elevations: importGraphBuffer(
        playheadGraph,
        'altitude',
        altitudeBuffer,
        'float32',
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
      currentElevations: importGraphBuffer(
        playheadGraph,
        'current-elevations',
        currentElevations,
        'float32',
        trackCount
      ),
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
  const windowParameters = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_PARAMETER_LENGTH
  );
  const trailDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'flight-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'flight-trails'});
  trailGraph.add(
    new GPUTimeWindowFilter({
      id: 'trail-window',
      timestamps: importGraphBuffer(
        trailGraph,
        'segment-start-times',
        segmentStartTimes,
        'float32',
        segmentCount
      ),
      endTimestamps: importGraphBuffer(
        trailGraph,
        'segment-end-times',
        segmentEndTimes,
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

  // ---- Line density: grid, subsets and variants -------------------------------------------------
  const allTracks: TrackSubset = {
    lngLat: lngLatBuffer,
    offsets: offsetsBuffer,
    vertexCount,
    trackCount,
    vertices: flights.lngLat,
    trackOffsets: flights.offsets
  };
  const subsets = new Map<number, TrackSubset>();
  function getSubset(direction: number): TrackSubset {
    if (direction < 0) return allTracks;
    let subset = subsets.get(direction);
    if (subset) return subset;
    let count = 0;
    let rows = 0;
    for (let track = 0; track < trackCount; track++) {
      if (flights.trackDirection[track] !== direction) continue;
      count++;
      rows += flights.offsets[track + 1] - flights.offsets[track];
    }
    const vertices = new Float32Array(rows * 2);
    const trackOffsets = new Uint32Array(count + 1);
    let row = 0;
    let index = 0;
    for (let track = 0; track < trackCount; track++) {
      if (flights.trackDirection[track] !== direction) continue;
      trackOffsets[index++] = row;
      const first = flights.offsets[track];
      const length = flights.offsets[track + 1] - first;
      vertices.set(flights.lngLat.subarray(first * 2, (first + length) * 2), row * 2);
      row += length;
    }
    trackOffsets[count] = row;
    subset = {
      lngLat: resources.createBuffer(`subset-${direction}-lng-lat`, vertices),
      offsets: resources.createBuffer(`subset-${direction}-offsets`, trackOffsets),
      vertexCount: rows,
      trackCount: count,
      vertices,
      trackOffsets
    };
    subsets.set(direction, subset);
    return subset;
  }

  /** Upper bound of segment-cell pieces: a segment crossing k grid lines gives k + 1 pieces. */
  function estimatePieces(subset: TrackSubset, cell: number): number {
    let pieces = 0;
    const {vertices, trackOffsets} = subset;
    for (let track = 0; track < subset.trackCount; track++) {
      for (let vertex = trackOffsets[track]; vertex < trackOffsets[track + 1] - 1; vertex++) {
        const dx = Math.abs(vertices[vertex * 2 + 2] - vertices[vertex * 2]);
        const dy = Math.abs(vertices[vertex * 2 + 3] - vertices[vertex * 2 + 1]);
        pieces += Math.floor(dx / cell) + Math.floor(dy / cell) + 2;
      }
    }
    return Math.max(1024, Math.ceil(pieces * 1.02));
  }

  let densityMaximum = 5;
  let destroyed = false;
  const grids = new Map<number, DensityGrid>();

  function getGrid(cell: number): DensityGrid {
    let grid = grids.get(cell);
    if (grid) return grid;
    const columns = Math.round((GRID_BOUNDS[2] - GRID_BOUNDS[0]) / cell);
    const rows = Math.round((GRID_BOUNDS[3] - GRID_BOUNDS[1]) / cell);
    const cells = columns * rows;
    const parameters = resources.createParameterBuffer(
      `density-parameters-${cell}`,
      'float32',
      GPU_LINE_DENSITY_PARAMETER_LENGTH,
      getGPULineDensityParameterValues({
        minX: GRID_BOUNDS[0],
        minY: GRID_BOUNDS[1],
        cellWidth: cell,
        cellHeight: cell
      })
    );
    const lengths = resources.createBuffer(`cell-lengths-${cell}`, cells * 4);
    const densities = resources.createBuffer(`cell-densities-${cell}`, cells * 4);
    const overflow = resources.createBuffer(`density-overflow-${cell}`, 4);
    const totalRecords = resources.createBuffer(`density-records-${cell}`, 4);
    const reader = new SummaryReader(
      resources,
      `density-${cell}`,
      [
        {buffer: densities, size: cells * 4},
        {buffer: overflow, size: 4},
        {buffer: totalRecords, size: 4}
      ],
      bytes => onDensity(cell, bytes)
    );
    grid = {
      cell,
      columns,
      rows,
      parameters,
      lengths,
      densities,
      overflow,
      totalRecords,
      capacity: new Map(),
      reader
    };
    grids.set(cell, grid);
    return grid;
  }

  const variants = new Map<string, DensityVariant>();
  function getVariant(cell: number, direction: number): DensityVariant {
    const key = `${cell}|${direction}`;
    let variant = variants.get(key);
    if (variant) return variant;
    const grid = getGrid(cell);
    const subset = getSubset(direction);
    const cells = grid.columns * grid.rows;
    const capacity = estimatePieces(subset, cell);
    grid.capacity.set(direction, capacity);
    const graph = new GPUCommandGraph<void>(device, {id: `flight-density-${key}`});
    graph.add(
      new GPULineDensity({
        id: 'line-density',
        positions: importGraphBuffer(
          graph,
          'positions',
          subset.lngLat,
          'float32x2',
          subset.vertexCount
        ),
        pathOffsets: importGraphBuffer(
          graph,
          'path-offsets',
          subset.offsets,
          'uint32',
          subset.trackCount + 1
        ),
        columns: grid.columns,
        rows: grid.rows,
        // Grid and positions are longitude/latitude degrees; lengths are great-circle metres.
        coordinateSystem: 'spherical',
        maximumRecords: capacity,
        parameters: grid.parameters.importToGraph(graph),
        output: {
          lengths: importGraphBuffer(graph, 'lengths', grid.lengths, 'float32', cells),
          densities: importGraphBuffer(graph, 'densities', grid.densities, 'float32', cells),
          overflow: importGraphBuffer(graph, 'overflow', grid.overflow, 'uint32', 1),
          totalRecords: importGraphBuffer(graph, 'total-records', grid.totalRecords, 'uint32', 1)
        }
      })
    );
    variant = {compiled: resources.track(graph.compile()), grid, encoded: false};
    variants.set(key, variant);
    return variant;
  }

  // The signed map needs both inputs at once. Do not reuse the regular density grid here: each
  // GPULineDensity writes its own lengths and densities before a third pass derives the ratio.
  const directionBalances = new Map<number, DirectionBalanceVariant>();
  function createDirectionDensityGrid(
    cell: number,
    direction: 'east' | 'west'
  ): DirectionDensityGrid {
    const columns = Math.round((GRID_BOUNDS[2] - GRID_BOUNDS[0]) / cell);
    const rows = Math.round((GRID_BOUNDS[3] - GRID_BOUNDS[1]) / cell);
    const cells = columns * rows;
    return {
      cell,
      columns,
      rows,
      parameters: resources.createParameterBuffer(
        `direction-${direction}-parameters-${cell}`,
        'float32',
        GPU_LINE_DENSITY_PARAMETER_LENGTH,
        getGPULineDensityParameterValues({
          minX: GRID_BOUNDS[0],
          minY: GRID_BOUNDS[1],
          cellWidth: cell,
          cellHeight: cell
        })
      ),
      lengths: resources.createBuffer(`direction-${direction}-lengths-${cell}`, cells * 4),
      densities: resources.createBuffer(`direction-${direction}-densities-${cell}`, cells * 4),
      overflow: resources.createBuffer(`direction-${direction}-overflow-${cell}`, 4),
      totalRecords: resources.createBuffer(`direction-${direction}-records-${cell}`, 4),
      capacity: new Map()
    };
  }

  function getDirectionBalance(cell: number): DirectionBalanceVariant {
    let variant = directionBalances.get(cell);
    if (variant) return variant;
    const east = createDirectionDensityGrid(cell, 'east');
    const west = createDirectionDensityGrid(cell, 'west');
    const cells = east.columns * east.rows;
    const eastSubset = getSubset(DIRECTION_CODES.east);
    const westSubset = getSubset(DIRECTION_CODES.west);
    const eastCapacity = estimatePieces(eastSubset, cell);
    const westCapacity = estimatePieces(westSubset, cell);
    const balance = resources.createBuffer(`direction-balance-${cell}`, cells * 4);
    const graph = new GPUCommandGraph<void>(device, {id: `flight-direction-balance-${cell}`});
    const addDensity = (
      id: string,
      subset: TrackSubset,
      grid: DirectionDensityGrid,
      capacity: number
    ) => {
      graph.add(
        new GPULineDensity({
          id,
          positions: importGraphBuffer(
            graph,
            `${id}-positions`,
            subset.lngLat,
            'float32x2',
            subset.vertexCount
          ),
          pathOffsets: importGraphBuffer(
            graph,
            `${id}-offsets`,
            subset.offsets,
            'uint32',
            subset.trackCount + 1
          ),
          columns: grid.columns,
          rows: grid.rows,
          coordinateSystem: 'spherical',
          maximumRecords: capacity,
          parameters: grid.parameters.importToGraph(graph),
          output: {
            lengths: importGraphBuffer(graph, `${id}-lengths`, grid.lengths, 'float32', cells),
            densities: importGraphBuffer(
              graph,
              `${id}-densities`,
              grid.densities,
              'float32',
              cells
            ),
            overflow: importGraphBuffer(graph, `${id}-overflow`, grid.overflow, 'uint32', 1),
            totalRecords: importGraphBuffer(graph, `${id}-records`, grid.totalRecords, 'uint32', 1)
          }
        })
      );
    };
    addDensity('east-density', eastSubset, east, eastCapacity);
    addDensity('west-density', westSubset, west, westCapacity);
    addKernelPass(graph, {
      id: 'direction-balance',
      invocationCount: cells,
      bindings: [
        {
          name: 'east',
          view: importGraphBuffer(graph, 'east-balance-input', east.densities, 'float32', cells),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'west',
          view: importGraphBuffer(graph, 'west-balance-input', west.densities, 'float32', cells),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'balance',
          view: importGraphBuffer(graph, 'direction-balance-output', balance, 'float32', cells),
          type: 'f32',
          access: 'read_write'
        }
      ],
      // A balance has no meaning below the design-sheet support threshold. NaN uses the raster
      // layer's transparent no-data path, rather than turning quiet cells into a neutral field.
      body: `let total = east[eastOffset + index] + west[westOffset + index];
  let noData = bitcast<f32>(0x7fc00000u);
  balance[balanceOffset + index] = select(noData, (east[eastOffset + index] - west[westOffset + index]) / total, total >= ${DIRECTION_BALANCE_MINIMUM_DENSITY});`
    });
    const reader = new SummaryReader(
      resources,
      `direction-balance-${cell}`,
      [
        {buffer: east.densities, size: cells * 4},
        {buffer: west.densities, size: cells * 4},
        {buffer: balance, size: cells * 4},
        {buffer: east.overflow, size: 4},
        {buffer: east.totalRecords, size: 4},
        {buffer: west.overflow, size: 4},
        {buffer: west.totalRecords, size: 4}
      ],
      bytes => onDirectionBalance(cell, bytes)
    );
    variant = {
      cell,
      columns: east.columns,
      rows: east.rows,
      east,
      west,
      balance,
      eastCapacity,
      westCapacity,
      compiled: resources.track(graph.compile()),
      reader,
      encoded: false
    };
    directionBalances.set(cell, variant);
    return variant;
  }

  /** Area in km^2 of a grid row of `cell`-degree cells. */
  function getCellArea(cell: number, row: number): number {
    const south = ((GRID_BOUNDS[1] + row * cell) * Math.PI) / 180;
    const north = ((GRID_BOUNDS[1] + (row + 1) * cell) * Math.PI) / 180;
    const width = (cell * Math.PI) / 180;
    return (EARTH_RADIUS / 1000) ** 2 * width * (Math.sin(north) - Math.sin(south));
  }

  function onDensity(cell: number, bytes: ArrayBuffer): void {
    if (destroyed) return;
    const grid = grids.get(cell);
    if (!grid || cell !== Number(ctx.options.cellDegrees)) return;
    const cells = grid.columns * grid.rows;
    const densities = new Float32Array(bytes, 0, cells);
    const words = new Uint32Array(bytes, cells * 4, 2);
    const positive: number[] = [];
    let flownKilometers = 0;
    const kilometers: number[] = [];
    for (let row = 0; row < grid.rows; row++) {
      const area = getCellArea(cell, row);
      for (let column = 0; column < grid.columns; column++) {
        const density = densities[row * grid.columns + column];
        if (density > 0) {
          positive.push(density);
          const km = density * 1000 * area;
          kilometers.push(km);
          flownKilometers += km;
        }
      }
    }
    positive.sort((a, b) => a - b);
    const p99 = positive.length
      ? positive[Math.min(positive.length - 1, Math.floor(positive.length * 0.99))]
      : 0.001;
    const nextMaximum = p99 * 1000;
    // Concentration: the share of flown distance carried by the busiest share of occupied cells.
    kilometers.sort((a, b) => b - a);
    const share = new Float64Array(51);
    const percent = new Float64Array(51);
    let running = 0;
    let cursor = 0;
    for (let step = 0; step <= 50; step++) {
      const limit = Math.round((kilometers.length * step) / 50);
      while (cursor < limit) running += kilometers[cursor++];
      percent[step] = step * 2;
      share[step] = flownKilometers > 0 ? (100 * running) / flownKilometers : 0;
    }
    const top10 = share[5];
    ctx.setReadout('flown', `${formatCount(flownKilometers)} km of track in the grid`);
    ctx.setReadout(
      'cells',
      `${formatCount(positive.length)} of ${formatCount(cells)} cells are crossed`
    );
    ctx.setReadout(
      'concentration',
      `The busiest 10% of crossed cells carry ${top10.toFixed(0)}% of the distance`
    );
    const capacity = grid.capacity.get(DIRECTION_CODES[ctx.options.direction]) ?? 0;
    ctx.setReadout(
      'pieces',
      `${formatCount(words[1])} of ${formatCount(capacity)}${words[0] ? ' (OVERFLOW, low counts)' : ''}`
    );
    ctx.setChart(
      'concentrationChart',
      lineChart(percent, share, {
        label: 'share of distance',
        xLabel: 'busiest share of crossed cells (%)',
        yLabel: 'share of flown distance (%)',
        xDomain: [0, 100],
        yDomain: [0, 100],
        formatX: value => `${Math.round(value)}`,
        formatY: value => `${Math.round(value)}`,
        guides: [{y: top10}],
        description:
          'Cumulative share of the distance flown against the busiest share of grid cells the flights cross.'
      })
    );
    ctx.setLegendExtent('density', [0, nextMaximum]);
    if (Math.abs(nextMaximum - densityMaximum) > densityMaximum * 0.05) {
      densityMaximum = nextMaximum;
      ctx.requestLayers();
    }
  }

  function onDirectionBalance(cell: number, bytes: ArrayBuffer): void {
    if (destroyed || cell !== Number(ctx.options.cellDegrees)) return;
    const variant = directionBalances.get(cell);
    if (!variant) return;
    const cells = variant.columns * variant.rows;
    const east = new Float32Array(bytes, 0, cells);
    const west = new Float32Array(bytes, cells * 4, cells);
    const balance = new Float32Array(bytes, cells * 8, cells);
    const words = new Uint32Array(bytes, cells * 12, 4);
    let occupied = 0;
    let eastDistance = 0;
    let westDistance = 0;
    let weightedBalance = 0;
    for (let index = 0; index < cells; index++) {
      const total = east[index] + west[index];
      if (total < DIRECTION_BALANCE_MINIMUM_DENSITY) continue;
      occupied++;
      eastDistance += east[index];
      westDistance += west[index];
      weightedBalance += balance[index] * total;
    }
    const totalDistance = eastDistance + westDistance;
    const share = totalDistance > 0 ? (100 * eastDistance) / totalDistance : 0;
    ctx.setReadout(
      'directionCells',
      `${formatCount(occupied)} cells have at least 0.05 km/km² of directional track`
    );
    ctx.setReadout(
      'directionBalance',
      `${share.toFixed(1)}% eastbound overall; mean occupied-cell balance ${(weightedBalance / Math.max(totalDistance, 1e-20)).toFixed(2)}`
    );
    ctx.setReadout(
      'directionPieces',
      `east ${formatCount(words[1])}/${formatCount(variant.eastCapacity)}${words[0] ? ' overflow' : ''}; west ${formatCount(words[3])}/${formatCount(variant.westCapacity)}${words[2] ? ' overflow' : ''}`
    );
  }

  // ---- State ------------------------------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'playing', speed: 'playbackSpeed', loop: 'loop'},
    {range: [0, SECONDS_PER_DAY], rate: 1, step: 60}
  );
  let playhead = ctx.options.time;
  let currentDirection = DIRECTION_CODES[ctx.options.direction];
  let currentCell = Number(ctx.options.cellDegrees);
  let densityDirty = true;
  let statusStale = true;
  let chartMarker = -1;

  // Flights airborne per ten minutes (static), with the playhead as a marker.
  const airborne = new Float64Array(HOURLY_BINS);
  for (let track = 0; track < trackCount; track++) {
    const first = Math.floor(flights.timestamps[flights.offsets[track]] / 600);
    const last = Math.min(
      HOURLY_BINS - 1,
      Math.floor(flights.timestamps[flights.offsets[track + 1] - 1] / 600)
    );
    for (let bin = Math.max(0, first); bin <= last; bin++) airborne[bin]++;
  }
  const hourCenters = Array.from({length: HOURLY_BINS}, (_, bin) => (bin + 0.5) / 6);
  function updateAirborneChart(): void {
    const marker = Math.round((playhead / 3600) * 6) / 6;
    if (marker === chartMarker) return;
    chartMarker = marker;
    ctx.setChart(
      'airborneChart',
      lineChart(hourCenters, airborne, {
        label: 'flights',
        xLabel: 'hour of day (UTC)',
        yLabel: 'flights airborne in the data',
        xDomain: [0, 24],
        markers: [{x: marker, label: 'now'}],
        formatX: value => `${Math.round(value)}`,
        formatY: value => formatCount(value),
        description:
          'Number of flights with a position in each ten-minute bin of the UTC day; a line marks the playhead.'
      })
    );
  }
  updateAirborneChart();

  ctx.setReadout('flights', `${formatCount(trackCount)} flights`);
  ctx.setReadout('vertices', `${formatCount(vertexCount)} vertices`);
  const eastAltitudes: number[] = [];
  const westAltitudes: number[] = [];
  for (let track = 0; track < trackCount; track++) {
    const altitudes =
      flights.trackDirection[track] === DIRECTION_CODES.east
        ? eastAltitudes
        : flights.trackDirection[track] === DIRECTION_CODES.west
          ? westAltitudes
          : null;
    if (!altitudes) continue;
    for (let vertex = flights.offsets[track]; vertex < flights.offsets[track + 1]; vertex++) {
      altitudes.push(flights.altitude[vertex]);
    }
  }
  const eastAltitudeHistogram = binValues(
    eastAltitudes,
    0,
    ALTITUDE_RAMP_METERS,
    ALTITUDE_HISTOGRAM_BINS
  );
  const westAltitudeHistogram = binValues(
    westAltitudes,
    0,
    ALTITUDE_RAMP_METERS,
    ALTITUDE_HISTOGRAM_BINS
  );
  const altitudeCenters = Array.from(
    {length: ALTITUDE_HISTOGRAM_BINS},
    (_, bin) => ((bin + 0.5) * ALTITUDE_RAMP_METERS) / ALTITUDE_HISTOGRAM_BINS
  );
  const toPercent = (counts: Float64Array, total: number) =>
    Float64Array.from(counts, count => (total > 0 ? (100 * count) / total : 0));
  const getPeak = (counts: Float64Array) => {
    let peak = 0;
    for (let bin = 1; bin < counts.length; bin++) if (counts[bin] > counts[peak]) peak = bin;
    return altitudeCenters[peak];
  };
  const eastPeak = getPeak(eastAltitudeHistogram);
  const westPeak = getPeak(westAltitudeHistogram);
  ctx.setReadout('eastLevelPeak', `${(eastPeak / 1000).toFixed(1)} km`);
  ctx.setReadout('westLevelPeak', `${(westPeak / 1000).toFixed(1)} km`);
  ctx.setChart('altitudeChart', {
    kind: 'line',
    height: 150,
    series: [
      {
        label: 'eastbound',
        x: altitudeCenters,
        y: toPercent(eastAltitudeHistogram, eastAltitudes.length),
        color: 0,
        area: true
      },
      {
        label: 'westbound',
        x: altitudeCenters,
        y: toPercent(westAltitudeHistogram, westAltitudes.length),
        color: 1,
        area: true
      }
    ],
    xLabel: 'stored ADS-B altitude (km)',
    yLabel: 'share of positions (%)',
    xDomain: [0, ALTITUDE_RAMP_METERS],
    markers: [
      {x: eastPeak, label: `E ${(eastPeak / 1000).toFixed(1)} km`},
      {x: westPeak, label: `W ${(westPeak / 1000).toFixed(1)} km`}
    ],
    formatX: value => `${Math.round(value / 1000)}`,
    formatY: value => value.toFixed(0),
    description:
      'Flight-level distributions of stored ADS-B positions. Eastbound and westbound samples are normalized separately, with their data-derived modal altitude bands marked.'
  });

  function writeSegmentMask(): void {
    const code = DIRECTION_CODES[ctx.options.direction];
    const mask = new Uint32Array(segmentCount);
    if (code < 0) mask.fill(1);
    else {
      for (let segment = 0; segment < segmentCount; segment++) {
        mask[segment] = flights.trackDirection[flights.segmentTracks[segment]] === code ? 1 : 0;
      }
    }
    segmentMaskBuffer.write(mask);
  }
  writeSegmentMask();
  getVariant(currentCell, currentDirection);
  getDirectionBalance(currentCell);

  // ---- Status readback --------------------------------------------------------------------------
  const statusReader = new SummaryReader(
    resources,
    'flight-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: trailCount, size: 4},
      {buffer: status, size: trackCount * 4},
      {buffer: currentElevations, size: trackCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      let before = 0;
      let after = 0;
      let gap = 0;
      let high = 0;
      for (let track = 0; track < trackCount; track++) {
        const value = words[2 + track];
        if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.beforeStart) before++;
        else if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.afterEnd) after++;
        else if (value === GPU_TRAJECTORY_PLAYHEAD_STATUS.gap) gap++;
        else if (
          value === GPU_TRAJECTORY_PLAYHEAD_STATUS.active &&
          floats[2 + trackCount + track] >= 9000
        ) {
          high++;
        }
      }
      ctx.setReadout('airborne', words[0]);
      ctx.setReadout('cruising', high);
      ctx.setReadout('landed', after);
      ctx.setReadout('notYet', before);
      ctx.setReadout('inGap', gap);
      ctx.setReadout('trailSegments', words[1]);
      updateAirborneChart();
    }
  );

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [
      playheadCompiled,
      trailCompiled,
      ctx.options.densityView === 'direction-balance'
        ? getDirectionBalance(currentCell).compiled
        : getVariant(currentCell, currentDirection).compiled
    ],

    setOption(id) {
      switch (id) {
        case 'direction':
          writeSegmentMask();
          currentDirection = DIRECTION_CODES[ctx.options.direction];
          getVariant(currentCell, currentDirection);
          densityDirty = true;
          ctx.requestLayers();
          break;
        case 'cellDegrees':
          currentCell = Number(ctx.options.cellDegrees);
          getVariant(currentCell, currentDirection);
          getDirectionBalance(currentCell);
          densityDirty = true;
          ctx.requestLayers();
          break;
        case 'densityView':
          if (ctx.options.densityView === 'direction-balance') getDirectionBalance(currentCell);
          densityDirty = true;
          ctx.requestLayers();
          break;
        case 'time':
        case 'playing':
        case 'playbackSpeed':
        case 'loop':
        case 'maxGapMinutes':
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
      ctx.setReadout('clock', formatClockUtc(playhead));

      if (options.densityView === 'direction-balance') {
        const variant = getDirectionBalance(currentCell);
        if (densityDirty || !variant.encoded) {
          variant.compiled.encode(commandEncoder, {parameters: undefined});
          variant.encoded = true;
          densityDirty = false;
          variant.reader.request(commandEncoder);
        } else {
          variant.reader.flush(commandEncoder);
        }
      } else {
        const variant = getVariant(currentCell, currentDirection);
        if (densityDirty || !variant.encoded) {
          variant.compiled.encode(commandEncoder, {parameters: undefined});
          variant.encoded = true;
          densityDirty = false;
          variant.grid.reader.request(commandEncoder);
        } else {
          variant.grid.reader.flush(commandEncoder);
        }
      }

      playheadParameters.write(
        getGPUTrajectoryPlayheadParameterValues({playhead, maxGap: options.maxGapMinutes * 60})
      );
      playheadCompiled.encode(commandEncoder, {parameters: undefined});

      if (options.showTrails) {
        const seconds = options.trailMinutes * 60;
        windowParameters.write(
          getGPUTimeWindowParameterValues({
            start: playhead - seconds,
            end: playhead,
            startFadeDuration: seconds * options.tailFade
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
      const elevationScale = options.extrude ? options.exaggeration : 0;
      const layers: Layer[] = [];
      const grid = getGrid(currentCell);
      if (options.showDensity && options.densityView === 'density') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `flight-density-${grid.cell}`,
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            gridSize: [grid.columns, grid.rows],
            bounds: GRID_BOUNDS,
            values: grid.densities,
            valueFormat: 'float32',
            valueScale: 1000,
            valueRange: [0, densityMaximum],
            colormap: options.densityRamp,
            sqrtScale: true,
            discardAtOrBelow: 0,
            opacity: options.densityOpacity,
            tessellation: 48
          })
        );
      }
      if (options.showDensity && options.densityView === 'direction-balance') {
        const balance = getDirectionBalance(currentCell);
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `flight-direction-balance-${balance.cell}`,
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            gridSize: [balance.columns, balance.rows],
            bounds: GRID_BOUNDS,
            values: balance.balance,
            valueFormat: 'float32',
            valueRange: [-1, 1],
            colormap: 'diverging',
            opacity: options.densityOpacity,
            // Low-support cells are NaN from the derive pass and take this transparent path.
            noDataColor: [0, 0, 0, 0],
            tessellation: 48
          })
        );
      }
      if (options.showBackdrop) {
        layers.push(
          new FlightSegmentLayer({
            id: 'flight-backdrop',
            lngLat: lngLatBuffer,
            elevations: altitudeBuffer,
            endVertices: endVerticesBuffer,
            instanceCount: segmentCount,
            colorMode: options.extrude ? 'altitude' : 'uniform',
            ramp: options.altitudeRamp,
            valueRange: [0, ALTITUDE_RAMP_METERS],
            color: dark ? [190, 205, 230, 26] : [40, 60, 90, 34],
            opacity: options.extrude ? 0.45 : 1,
            minAltitude: options.altitudeFloor,
            elevationScale,
            widthPixels: options.extrude ? 1 : 0.8
          })
        );
      }
      if (options.showTrails) {
        const byDirection = options.trailColor === 'direction';
        layers.push(
          new FlightSegmentLayer({
            id: 'flight-trails',
            lngLat: lngLatBuffer,
            elevations: altitudeBuffer,
            endVertices: endVerticesBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            values: byDirection ? vertexDirectionBuffer : null,
            valuesAreFloat: false,
            colorMode: byDirection ? 'category' : 'altitude',
            palette: FLIGHT_DIRECTION_COLORS,
            ramp: options.altitudeRamp,
            valueRange: [0, ALTITUDE_RAMP_METERS],
            minAltitude: options.altitudeFloor,
            elevationScale,
            widthPixels: 2.2
          })
        );
      }
      if (options.showMarkers) {
        const byDirection = options.markerColor === 'direction';
        layers.push(
          new FlightPointLayer({
            id: 'flight-markers',
            lngLat: currentPositions,
            elevations: currentElevations,
            ids: activeIds,
            drawCommands: markerDraw,
            values: byDirection ? trackDirectionBuffer : null,
            valuesAreFloat: false,
            colorMode: byDirection ? 'category' : 'altitude',
            palette: FLIGHT_DIRECTION_COLORS,
            ramp: options.altitudeRamp,
            valueRange: [0, ALTITUDE_RAMP_METERS],
            elevationScale,
            radiusPixels: options.markerSize
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      statusReader.stop();
      for (const grid of grids.values()) grid.reader.stop();
      for (const balance of directionBalances.values()) balance.reader.stop();
      resources.destroy();
    }
  };
}
