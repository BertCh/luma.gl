// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUDistanceFieldParameterValues,
  getGPUPolygonRasterizationExtentValues,
  GPUDistanceField,
  GPUPolygonRasterization,
  GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
  GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH,
  GPURasterZonalStatistics
} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUTerrainDerivativesParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPUTerrainDerivatives
} from '@luma.gl/experimental/gpu-terrain';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {
  buildRingEdges,
  createGraphImporter,
  findFeatureAt,
  formatNumber
} from '../geometry/b3-common';
import {
  describeFire,
  fromMercator,
  loadWildfires,
  toMercator,
  type WildfireData
} from '../geometry/wildfire-data';
import type {SceneContext, SceneInstance} from '../scene';
import {WildfireMercatorRasterLayer} from './wildfire-raster-layer';

/** Option state of the wildfire-terrain scene. */
export type WildfireTerrainOptions = {
  display: 'zones' | 'boundary' | 'slope' | 'southness' | 'hillshade';
  ramp: Extract<RampName, 'viridis' | 'magma' | 'inferno' | 'cividis'>;
  overlayOpacity: number;
  showHillshade: boolean;
  showOutlines: boolean;
  ringOuter: number;
  ringInner: number;
  flatSlope: number;
  steepSlope: number;
  year: 'all' | '2020' | '2021' | '2022' | '2023';
  minAcres: number;
  sunAzimuth: number;
  sunAltitude: number;
  zFactor: number;
  borderMode: 'clamp' | 'reflect' | 'constant' | 'nodata';
  distanceMode: 'exact' | 'jump-flood';
  refinement: '0' | '1' | '2';
  sumOrder: 'sorted' | 'atomic';
};

/** Cells a fire needs inside and in the ring to be compared. */
const MINIMUM_CELLS = 20;
const OCTANTS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
const INSIDE_COLOR = [232, 90, 60, 190] as const;
const RING_COLOR = [78, 168, 222, 160] as const;
const SLOPE_RANGE: [number, number] = [0, 40];

/** Per-zone statistics read back from the zonal graph. Zone `2 * fire` is inside, `2 * fire + 1` the ring. */
type ZonalTable = {
  slopeCells: Uint32Array;
  slopeCount: Uint32Array;
  slopeSum: Float32Array;
  slopeMax: Float32Array;
  southCount: Uint32Array;
  southSum: Float32Array;
  steepCount: Uint32Array;
  steepSum: Float32Array;
  elevationCount: Uint32Array;
  elevationSum: Float32Array;
  rose: Uint32Array;
};

/**
 * Wildfire terrain: polygon rasterization of the fire perimeters on a DEM grid, an exact distance
 * field for the buffer ring, Horn slope and aspect, and zonal statistics for "inside" against
 * "ring". Everything runs on the GPU; a few hundred numbers are read back for the charts.
 */
export async function createWildfireTerrain(
  ctx: SceneContext<WildfireTerrainOptions>
): Promise<SceneInstance<WildfireTerrainOptions>> {
  const {device} = ctx;
  const data: WildfireData = loadWildfires(ctx.datasets.get('poopdeck-wildfires'));
  const demDataset = ctx.datasets.get('wildfire-california-dem');
  const demRaster = demDataset.raster;
  if (!demRaster) throw new Error('wildfire-california-dem has no raster');
  const spec = demRaster.spec as unknown as {
    boundsMercator: [number, number, number, number];
    cellSizeMercatorM: number;
    noData: number;
  };
  const {width, height} = demRaster;
  const cellCount = width * height;
  const boundsMercator = spec.boundsMercator;
  const cellSize = spec.cellSizeMercatorM;
  const fireCount = data.count;
  const zoneCapacity = fireCount * 2;
  const roseCapacity = zoneCapacity * 8;
  const {layout} = data;
  const resources = new SpatialAnalysisResources(device, 'wildfire-terrain');
  let destroyed = false;
  const WORLD_METERS = 40075016.68557849;
  const northEdge = 0.5 - boundsMercator[3] / WORLD_METERS;
  const southEdge = 0.5 - boundsMercator[1] / WORLD_METERS;

  // ---- Fire geometry in raster space: x east from the DEM west edge, y south from its north edge ----
  const rasterPositions = new Float32Array(layout.vertexCount * 2);
  for (let vertex = 0; vertex < layout.vertexCount; vertex++) {
    const [x, y] = toMercator(layout.lngLat[vertex * 2], layout.lngLat[vertex * 2 + 1]);
    rasterPositions[vertex * 2] = x - boundsMercator[0];
    rasterPositions[vertex * 2 + 1] = boundsMercator[3] - y;
  }
  // Crossing records of the rasterizer: one per (edge, row-center line) pair inside the grid.
  let crossingTotal = 0;
  for (let ring = 0; ring < layout.ringCount; ring++) {
    const first = layout.ringOffsets[ring];
    const last = layout.ringOffsets[ring + 1];
    for (let vertex = first; vertex < last; vertex++) {
      const next = vertex + 1 < last ? vertex + 1 : first;
      const y0 = rasterPositions[vertex * 2 + 1] / cellSize;
      const y1 = rasterPositions[next * 2 + 1] / cellSize;
      const low = Math.min(Math.max(Math.ceil(Math.min(y0, y1) - 0.5), 0), height);
      const high = Math.min(Math.max(Math.ceil(Math.max(y0, y1) - 0.5), 0), height);
      crossingTotal += Math.max(0, high - low);
    }
  }
  // Margin: the GPU rounds row thresholds in float32, so an edge on a row center line can add a record.
  const crossingCapacity = Math.max(2, Math.ceil(crossingTotal * 1.05) + 256);
  // Fires that touch the DEM (for the outline layer and the story).
  const inDem = new Uint8Array(fireCount);
  const [demWest, demSouth, demEast, demNorth] = demDataset.manifest.bbox as [
    number,
    number,
    number,
    number
  ];
  for (let fire = 0; fire < fireCount; fire++) {
    const b = fire * 4;
    inDem[fire] =
      layout.featureBounds[b + 2] >= demWest &&
      layout.featureBounds[b] <= demEast &&
      layout.featureBounds[b + 3] >= demSouth &&
      layout.featureBounds[b + 1] <= demNorth
        ? 1
        : 0;
  }

  // ---- Outlines ---------------------------------------------------------------------------------------
  const allEdges = buildRingEdges(layout);
  let outlineCount = 0;
  for (let edge = 0; edge < allEdges.edgeCount; edge++)
    if (inDem[allEdges.featureRows[edge]]) outlineCount++;
  const outlineSegments = new Float32Array(outlineCount * 4);
  const outlineRows = new Uint32Array(outlineCount);
  const edgesPerFire = new Uint32Array(fireCount);
  {
    let out = 0;
    for (let edge = 0; edge < allEdges.edgeCount; edge++) {
      const fire = allEdges.featureRows[edge];
      edgesPerFire[fire]++;
      if (!inDem[fire]) continue;
      outlineSegments.set(allEdges.starts.subarray(edge * 2, edge * 2 + 2), out * 4);
      outlineSegments.set(allEdges.ends.subarray(edge * 2, edge * 2 + 2), out * 4 + 2);
      outlineRows[out++] = fire;
    }
  }
  const maximumFireEdges = Math.max(1, ...edgesPerFire);

  // ---- Buffers ----------------------------------------------------------------------------------------
  const demBuffer = resources.createBuffer(
    'dem',
    Float32Array.from(demRaster.values as Float32Array)
  );
  const polygonPositions = resources.createBuffer('polygon-positions', rasterPositions);
  const featureOffsets = resources.createBuffer('feature-offsets', layout.featureOffsets);
  const polygonOffsets = resources.createBuffer('polygon-offsets', layout.polygonOffsets);
  const ringOffsets = resources.createBuffer('ring-offsets', layout.ringOffsets);
  const column = (name: string, rows = cellCount) =>
    resources.createBuffer(name, Math.max(rows, 1) * 4);
  const zones = column('zones');
  const boundary = column('boundary');
  const seeds = column('seeds');
  const distances = column('distances');
  const allocation = column('allocation');
  const slope = column('slope');
  const aspect = column('aspect');
  const hillshade = column('hillshade');
  const zoneFire = column('zone-fire');
  const zoneRose = column('zone-rose');
  const southness = column('southness');
  const steep = column('steep');
  const classes = column('classes');
  const rasterOverflow = column('raster-overflow', 1);
  const rasterCrossings = column('raster-crossings', 1);
  const enabled = column('enabled', fireCount);
  const enabledFloat = column('enabled-float', fireCount);
  const selectionSegments = resources.createBuffer('selection-segments', maximumFireEdges * 16);
  const outlineBuffer = resources.createBuffer('outline-segments', outlineSegments);
  const outlineRowBuffer = resources.createBuffer('outline-rows', outlineRows);
  const extentParameters = resources.createParameterBuffer(
    'extent',
    'float32',
    GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH,
    getGPUPolygonRasterizationExtentValues(0, 0, cellSize, cellSize)
  );
  const distanceSettings = resources.createParameterBuffer(
    'distance-settings',
    'float32',
    GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
    getGPUDistanceFieldParameterValues({
      origin: [0, 0],
      cellSize: [cellSize, cellSize],
      maxDistance: 1e7
    })
  );
  const terrainSettings = resources.createParameterBuffer(
    'terrain-settings',
    'float32',
    GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
  );
  const stageParameters = resources.createParameterBuffer('stage-parameters', 'float32', 8);

  // Zonal outputs.
  const zonalOutputs = {
    slopeCells: column('z-slope-cells', zoneCapacity),
    slopeCount: column('z-slope-count', zoneCapacity),
    slopeSum: column('z-slope-sum', zoneCapacity),
    slopeMax: column('z-slope-max', zoneCapacity),
    southCount: column('z-south-count', zoneCapacity),
    southSum: column('z-south-sum', zoneCapacity),
    steepCount: column('z-steep-count', zoneCapacity),
    steepSum: column('z-steep-sum', zoneCapacity),
    elevationCount: column('z-elevation-count', zoneCapacity),
    elevationSum: column('z-elevation-sum', zoneCapacity),
    rose: column('z-rose', roseCapacity)
  };

  // ---- Parameters -------------------------------------------------------------------------------------
  const writeStageParameters = (options: WildfireTerrainOptions) => {
    stageParameters.write(
      Float32Array.of(
        options.ringInner * 1000,
        options.ringOuter * 1000,
        boundsMercator[3],
        cellSize,
        options.flatSlope,
        options.steepSlope,
        0,
        0
      )
    );
  };
  const writeTerrainSettings = (options: WildfireTerrainOptions) => {
    terrainSettings.write(
      getGPUTerrainDerivativesParameterValues({
        cellSize: [cellSize, cellSize],
        zFactor: options.zFactor,
        azimuthDegrees: options.sunAzimuth,
        altitudeDegrees: options.sunAltitude,
        northEdge,
        southEdge
      })
    );
  };
  const visibleFires = new Uint8Array(fireCount);
  const updateEnabled = (options: WildfireTerrainOptions) => {
    const words = new Uint32Array(fireCount);
    const flags = new Float32Array(fireCount);
    for (let fire = 0; fire < fireCount; fire++) {
      const on =
        (options.year === 'all' || String(data.year[fire]) === options.year) &&
        data.acres[fire] >= options.minAcres;
      visibleFires[fire] = on ? 1 : 0;
      words[fire] = on ? 1 : 0;
      flags[fire] = on ? 1 : 0;
    }
    enabled.write(words);
    enabledFloat.write(flags);
  };
  writeStageParameters(ctx.options);
  writeTerrainSettings(ctx.options);
  updateEnabled(ctx.options);

  // ---- Graphs -----------------------------------------------------------------------------------------
  /** Polygon rasterization, seed mask and exact or jump-flood distance field. Encoded once per build. */
  const buildRaster = (options: WildfireTerrainOptions): CompiledGPUCommandGraph<void> => {
    const graph = new GPUCommandGraph<void>(device, {id: 'wildfire-raster'});
    const imp = createGraphImporter(graph);
    const zonesView = imp('zones', zones, 'uint32', cellCount);
    graph.add(
      new GPUPolygonRasterization({
        id: 'rasterize',
        width,
        height,
        extent: extentParameters.importToGraph(graph),
        polygonPositions: imp(
          'polygon-positions',
          polygonPositions,
          'float32x2',
          layout.vertexCount
        ),
        featureOffsets: imp('feature-offsets', featureOffsets, 'uint32', fireCount + 1),
        polygonOffsets: imp('polygon-offsets', polygonOffsets, 'uint32', layout.partCount + 1),
        ringOffsets: imp('ring-offsets', ringOffsets, 'uint32', layout.ringCount + 1),
        crossingCapacity,
        zones: zonesView,
        boundary: imp('boundary', boundary, 'uint32', cellCount),
        overflow: imp('raster-overflow', rasterOverflow, 'uint32', 1),
        crossingCount: imp('raster-crossings', rasterCrossings, 'uint32', 1)
      })
    );
    const seedView = imp('seeds', seeds, 'uint32', cellCount);
    addKernelPass(graph, {
      id: 'seed-mask',
      invocationCount: cellCount,
      bindings: [
        {name: 'zones', view: zonesView, type: 'u32', access: 'read'},
        {name: 'seeds', view: seedView, type: 'u32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let zone = zones[zonesOffset + index];
  seeds[seedsOffset + index] = select(zone + 1u, 0u, zone == 0xffffffffu);`
    });
    graph.add(
      new GPUDistanceField({
        id: 'ring-distance',
        width,
        height,
        settings: distanceSettings.importToGraph(graph),
        seedMask: seedView,
        mode: options.distanceMode,
        jumpFloodRefinementPasses: Number(options.refinement) as 0 | 1 | 2,
        output: {
          distances: imp('distances', distances, 'float32', cellCount),
          allocation: imp('allocation', allocation, 'uint32', cellCount)
        }
      })
    );
    return resources.track(graph.compile());
  };

  /** Horn slope, aspect and hillshade. */
  const buildTerrain = (options: WildfireTerrainOptions): CompiledGPUCommandGraph<void> => {
    const graph = new GPUCommandGraph<void>(device, {id: 'wildfire-derivatives'});
    const imp = createGraphImporter(graph);
    graph.add(
      new GPUTerrainDerivatives({
        id: 'derivatives',
        width,
        height,
        elevation: {
          id: 'dem',
          format: 'float32',
          noDataValue: spec.noData,
          storage: {kind: 'buffer', values: imp('dem', demBuffer, 'float32', cellCount)}
        },
        settings: terrainSettings.importToGraph(graph),
        slope: imp('slope', slope, 'float32', cellCount),
        aspect: imp('aspect', aspect, 'float32', cellCount),
        hillshade: imp('hillshade', hillshade, 'float32', cellCount),
        cellSizeMode: 'web-mercator',
        rowDirection: 'south',
        slopeUnits: 'degrees',
        borderMode: options.borderMode
      })
    );
    return resources.track(graph.compile());
  };

  /** Inside and ring zone assignment, derived bands and five zonal statistics. */
  const buildStats = (options: WildfireTerrainOptions): CompiledGPUCommandGraph<void> => {
    const graph = new GPUCommandGraph<void>(device, {id: 'wildfire-zonal'});
    const imp = createGraphImporter(graph);
    const zonesView = imp('zones', zones, 'uint32', cellCount);
    const zoneFireView = imp('zone-fire', zoneFire, 'uint32', cellCount);
    const zoneRoseView = imp('zone-rose', zoneRose, 'uint32', cellCount);
    const slopeView = imp('slope', slope, 'float32', cellCount);
    const southView = imp('southness', southness, 'float32', cellCount);
    const steepView = imp('steep', steep, 'float32', cellCount);
    const parameterView = stageParameters.importToGraph(graph);
    addKernelPass(graph, {
      id: 'assign-zones',
      invocationCount: cellCount,
      bindings: [
        {name: 'zones', view: zonesView, type: 'u32', access: 'read'},
        {
          name: 'dist',
          view: imp('distances', distances, 'float32', cellCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'alloc',
          view: imp('allocation', allocation, 'uint32', cellCount),
          type: 'u32',
          access: 'read'
        },
        {name: 'params', view: parameterView, type: 'f32', access: 'read'},
        {
          name: 'enabled',
          view: imp('enabled', enabled, 'uint32', fireCount),
          type: 'u32',
          access: 'read'
        },
        {name: 'outZone', view: zoneFireView, type: 'u32', access: 'read_write'},
        {
          name: 'outClass',
          view: imp('classes', classes, 'uint32', cellCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      declarations: `const WIDTH: u32 = ${width}u;`,
      body: /* wgsl */ `
  // Ground scale at the latitude of the cell row: Mercator meters shrink by cos(latitude).
  let row = index / WIDTH;
  let mercatorY = params[paramsOffset + 2u] - (f32(row) + 0.5) * params[paramsOffset + 3u];
  let latitude = 2.0 * atan(exp(mercatorY / 6378137.0)) - 1.5707963;
  let groundScale = cos(latitude);
  let zone = zones[zonesOffset + index];
  var assigned = 0xffffffffu;
  var kind = 0u;
  if (zone != 0xffffffffu) {
    if (enabled[enabledOffset + zone] != 0u) {
      assigned = zone * 2u;
      kind = 1u;
    }
  } else {
    let nearest = alloc[allocOffset + index];
    if (nearest != 0xffffffffu && enabled[enabledOffset + nearest] != 0u) {
      let ground = dist[distOffset + index] * groundScale;
      if (ground >= params[paramsOffset] && ground <= params[paramsOffset + 1u]) {
        assigned = nearest * 2u + 1u;
        kind = 2u;
      }
    }
  }
  outZone[outZoneOffset + index] = assigned;
  outClass[outClassOffset + index] = kind;`
    });
    addKernelPass(graph, {
      id: 'derive-bands',
      invocationCount: cellCount,
      bindings: [
        {name: 'zoneFire', view: zoneFireView, type: 'u32', access: 'read'},
        {name: 'slope', view: slopeView, type: 'f32', access: 'read'},
        {
          name: 'aspect',
          view: imp('aspect', aspect, 'float32', cellCount),
          type: 'f32',
          access: 'read'
        },
        {name: 'params', view: parameterView, type: 'f32', access: 'read'},
        {name: 'outRose', view: zoneRoseView, type: 'u32', access: 'read_write'},
        {name: 'outSouth', view: southView, type: 'f32', access: 'read_write'},
        {name: 'outSteep', view: steepView, type: 'f32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let steepness = slope[slopeOffset + index];
  let direction = aspect[aspectOffset + index];
  var nanBits = 0x7fc00000u;
  var south = bitcast<f32>(nanBits);
  var steepFlag = bitcast<f32>(nanBits);
  var rose = 0xffffffffu;
  let valid = (bitcast<u32>(steepness) & 0x7fffffffu) < 0x7f800000u;
  if (valid) {
    steepFlag = select(0.0, 1.0, steepness >= params[paramsOffset + 5u]);
    // Aspect is the downslope direction, clockwise from north; flat cells report -1.
    if (steepness >= params[paramsOffset + 4u] && direction >= 0.0) {
      south = -cos(radians(direction));
      let assigned = zoneFire[zoneFireOffset + index];
      if (assigned != 0xffffffffu) {
        let octant = u32(floor(fract((direction + 22.5) / 360.0) * 8.0)) % 8u;
        rose = assigned * 8u + octant;
      }
    }
  }
  outRose[outRoseOffset + index] = rose;
  outSouth[outSouthOffset + index] = south;
  outSteep[outSteepOffset + index] = steepFlag;`
    });
    const counts = (name: string, buffer: Buffer, rows = zoneCapacity) =>
      imp(name, buffer, 'uint32', rows);
    const floats = (name: string, buffer: Buffer, rows = zoneCapacity) =>
      imp(name, buffer, 'float32', rows);
    const band = (id: string, values: ReturnType<typeof floats>, noDataValue?: number) => ({
      id,
      format: 'float32' as const,
      ...(noDataValue === undefined ? {} : {noDataValue}),
      storage: {kind: 'buffer' as const, values}
    });
    const common = {width, height, zoneCapacity, sumOrder: options.sumOrder};
    graph.add(
      new GPURasterZonalStatistics({
        id: 'zonal-slope',
        ...common,
        zones: zoneFireView,
        values: band('slope', slopeView),
        output: {
          cellCounts: counts('z-slope-cells', zonalOutputs.slopeCells),
          valueCounts: counts('z-slope-count', zonalOutputs.slopeCount),
          sums: floats('z-slope-sum', zonalOutputs.slopeSum),
          maximums: floats('z-slope-max', zonalOutputs.slopeMax)
        }
      })
    );
    graph.add(
      new GPURasterZonalStatistics({
        id: 'zonal-south',
        ...common,
        zones: zoneFireView,
        values: band('southness', southView),
        output: {
          valueCounts: counts('z-south-count', zonalOutputs.southCount),
          sums: floats('z-south-sum', zonalOutputs.southSum)
        }
      })
    );
    graph.add(
      new GPURasterZonalStatistics({
        id: 'zonal-steep',
        ...common,
        zones: zoneFireView,
        values: band('steep', steepView),
        output: {
          valueCounts: counts('z-steep-count', zonalOutputs.steepCount),
          sums: floats('z-steep-sum', zonalOutputs.steepSum)
        }
      })
    );
    graph.add(
      new GPURasterZonalStatistics({
        id: 'zonal-elevation',
        ...common,
        zones: zoneFireView,
        values: band('dem', imp('dem', demBuffer, 'float32', cellCount), spec.noData),
        output: {
          valueCounts: counts('z-elevation-count', zonalOutputs.elevationCount),
          sums: floats('z-elevation-sum', zonalOutputs.elevationSum)
        }
      })
    );
    graph.add(
      new GPURasterZonalStatistics({
        id: 'zonal-rose',
        width,
        height,
        zoneCapacity: roseCapacity,
        sumOrder: options.sumOrder,
        zones: zoneRoseView,
        values: band('slope-for-rose', slopeView),
        output: {
          cellCounts: counts('z-rose', zonalOutputs.rose, roseCapacity)
        }
      })
    );
    return resources.track(graph.compile());
  };

  let rasterGraph = buildRaster(ctx.options);
  let terrainGraph = buildTerrain(ctx.options);
  let statsGraph = buildStats(ctx.options);

  // ---- Readback and analysis --------------------------------------------------------------------------
  const zonalSources: {buffer: Buffer; size: number}[] = [
    {buffer: zonalOutputs.slopeCells, size: zoneCapacity * 4},
    {buffer: zonalOutputs.slopeCount, size: zoneCapacity * 4},
    {buffer: zonalOutputs.slopeSum, size: zoneCapacity * 4},
    {buffer: zonalOutputs.slopeMax, size: zoneCapacity * 4},
    {buffer: zonalOutputs.southCount, size: zoneCapacity * 4},
    {buffer: zonalOutputs.southSum, size: zoneCapacity * 4},
    {buffer: zonalOutputs.steepCount, size: zoneCapacity * 4},
    {buffer: zonalOutputs.steepSum, size: zoneCapacity * 4},
    {buffer: zonalOutputs.elevationCount, size: zoneCapacity * 4},
    {buffer: zonalOutputs.elevationSum, size: zoneCapacity * 4},
    {buffer: zonalOutputs.rose, size: roseCapacity * 4}
  ];
  let table: ZonalTable | null = null;
  let selected = -1;
  let selectedEdgeCount = 0;
  const dirty = {raster: true, terrain: true, stats: true};
  let lastChange = performance.now();
  let lastStatsEncode = -Infinity;

  const parseTable = (bytes: ArrayBuffer): ZonalTable => {
    let offset = 0;
    const take = <T extends Uint32Array | Float32Array>(
      Type: new (b: ArrayBuffer, o: number, l: number) => T,
      rows: number
    ) => {
      const out = new Type(bytes, offset, rows);
      offset += rows * 4;
      return out;
    };
    return {
      slopeCells: take(Uint32Array, zoneCapacity),
      slopeCount: take(Uint32Array, zoneCapacity),
      slopeSum: take(Float32Array, zoneCapacity),
      slopeMax: take(Float32Array, zoneCapacity),
      southCount: take(Uint32Array, zoneCapacity),
      southSum: take(Float32Array, zoneCapacity),
      steepCount: take(Uint32Array, zoneCapacity),
      steepSum: take(Float32Array, zoneCapacity),
      elevationCount: take(Uint32Array, zoneCapacity),
      elevationSum: take(Float32Array, zoneCapacity),
      rose: take(Uint32Array, roseCapacity)
    };
  };

  const median = (values: number[]) => {
    if (!values.length) return Number.NaN;
    const sorted = [...values].sort((a, b) => a - b);
    return (
      sorted[Math.floor((sorted.length - 1) / 2)] / 2 +
      sorted[Math.ceil((sorted.length - 1) / 2)] / 2
    );
  };

  const analyze = () => {
    if (!table) return;
    const t = table;
    const options = ctx.options;
    const comparable: number[] = [];
    for (let fire = 0; fire < fireCount; fire++) {
      if (
        visibleFires[fire] &&
        t.slopeCount[2 * fire] >= MINIMUM_CELLS &&
        t.slopeCount[2 * fire + 1] >= MINIMUM_CELLS
      ) {
        comparable.push(fire);
      }
    }
    const scope = selected >= 0 && comparable.includes(selected) ? [selected] : comparable;
    const sum = (zonesOfScope: number[], values: ArrayLike<number>, ring: number) =>
      zonesOfScope.reduce((total, fire) => total + values[2 * fire + ring], 0);
    const pooled = (sums: ArrayLike<number>, counts: ArrayLike<number>, ring: number) => {
      const count = sum(scope, counts, ring);
      return count > 0 ? sum(scope, sums, ring) / count : Number.NaN;
    };
    const slopeInside = pooled(t.slopeSum, t.slopeCount, 0);
    const slopeRing = pooled(t.slopeSum, t.slopeCount, 1);
    const steepInside = pooled(t.steepSum, t.steepCount, 0);
    const steepRing = pooled(t.steepSum, t.steepCount, 1);
    const southInside = pooled(t.southSum, t.southCount, 0);
    const southRing = pooled(t.southSum, t.southCount, 1);
    const elevationInside = pooled(t.elevationSum, t.elevationCount, 0);
    const elevationRing = pooled(t.elevationSum, t.elevationCount, 1);
    const rose = [new Float64Array(8), new Float64Array(8)];
    for (const fire of scope) {
      for (let ring = 0; ring < 2; ring++) {
        for (let octant = 0; octant < 8; octant++)
          rose[ring][octant] += t.rose[(2 * fire + ring) * 8 + octant];
      }
    }
    const roseTotals = rose.map(counts => counts.reduce((a, b) => a + b, 0));
    const share = rose.map((counts, ring) =>
      Array.from(counts, count => (roseTotals[ring] ? (count / roseTotals[ring]) * 100 : 0))
    );
    const southShare = share.map(values => values[3] + values[4] + values[5]);

    const differences = comparable.map(
      fire =>
        t.slopeSum[2 * fire] / t.slopeCount[2 * fire] -
        t.slopeSum[2 * fire + 1] / t.slopeCount[2 * fire + 1]
    );
    const steeperCount = differences.filter(value => value > 0).length;
    const fmt = (value: number, digits = 1) =>
      Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
    const scopeName = scope.length === 1 ? data.names[scope[0]] : `${scope.length} fires`;
    ctx.setReadout('scope', comparable.length ? scopeName : 'none (check the filters)');
    ctx.setReadout(
      'comparable',
      `${comparable.length} of ${inDem.reduce((a, b) => a + b, 0)} fires on the DEM`
    );
    ctx.setReadout('cellsInside', sum(scope, t.slopeCells, 0));
    ctx.setReadout('cellsRing', sum(scope, t.slopeCells, 1));
    ctx.setReadout(
      'slopeMean',
      `${fmt(slopeInside)} vs ${fmt(slopeRing)} deg (${slopeInside >= slopeRing ? '+' : ''}${fmt(slopeInside - slopeRing)})`
    );
    ctx.setReadout('steepShare', `${fmt(steepInside * 100)}% vs ${fmt(steepRing * 100)}%`);
    ctx.setReadout('southShare', `${fmt(southShare[0])}% vs ${fmt(southShare[1])}%`);
    ctx.setReadout('southness', `${fmt(southInside, 2)} vs ${fmt(southRing, 2)}`);
    ctx.setReadout('elevation', `${fmt(elevationInside, 0)} vs ${fmt(elevationRing, 0)} m`);
    ctx.setReadout('steeperFires', `${steeperCount} of ${comparable.length}`);
    ctx.setReadout('medianDifference', `${fmt(median(differences), 2)} deg`);

    ctx.setChart(
      'slopeChart',
      Number.isFinite(slopeInside)
        ? {
            kind: 'bars',
            values: [slopeInside, slopeRing],
            labels: ['inside the perimeter', 'ring'],
            highlight: [0],
            yLabel: 'mean slope (deg)',
            description: `Mean slope inside the perimeters and in the buffer ring, ${scopeName}`
          }
        : null
    );
    ctx.setChart(
      'steepChart',
      Number.isFinite(steepInside)
        ? {
            kind: 'bars',
            values: [steepInside * 100, steepRing * 100],
            labels: ['inside the perimeter', 'ring'],
            highlight: [0],
            yLabel: `% of cells at or above ${options.steepSlope} deg`,
            description: `Share of steep cells inside the perimeters and in the ring, ${scopeName}`
          }
        : null
    );
    if (differences.length) {
      const limit = Math.max(4, Math.ceil(Math.max(...differences.map(Math.abs))));
      const bins = new Array(16).fill(0);
      for (const value of differences) {
        bins[Math.min(15, Math.max(0, Math.floor(((value + limit) / (2 * limit)) * 16)))]++;
      }
      ctx.setChart('differenceChart', {
        kind: 'histogram',
        values: bins,
        xDomain: [-limit, limit],
        xLabel: 'mean slope inside minus ring (deg), one count per fire',
        yLabel: 'fires',
        markers: [
          {x: 0, label: 'equal'},
          {x: median(differences), label: 'median'}
        ],
        description: 'Per fire difference of the mean slope inside the perimeter and in the ring'
      });
    } else {
      ctx.setChart('differenceChart', null);
    }
    if (roseTotals[0] > 0 && roseTotals[1] > 0) {
      const closed = (values: number[]) => [...values, values[0]];
      ctx.setChart('aspectChart', {
        kind: 'line',
        series: [
          {
            label: 'inside',
            x: [0, 45, 90, 135, 180, 225, 270, 315, 360],
            y: closed(share[0]),
            color: 0
          },
          {
            label: 'ring',
            x: [0, 45, 90, 135, 180, 225, 270, 315, 360],
            y: closed(share[1]),
            color: 2
          }
        ],
        xDomain: [0, 360],
        xLabel: 'direction the slope faces',
        yLabel: '% of sloping cells',
        formatX: value => OCTANTS[Math.round(value / 45) % 8],
        guides: [{y: 12.5, label: 'uniform'}],
        description: 'Share of sloping cells facing each of eight directions, inside against ring'
      });
      const differenceShare = share[0].map((value, octant) => value - share[1][octant]);
      ctx.setChart('aspectDifferenceChart', {
        kind: 'bars',
        values: differenceShare,
        labels: OCTANTS,
        highlight: differenceShare
          .map((value, octant) => (value > 0 ? octant : -1))
          .filter(octant => octant >= 0),
        yLabel: 'inside minus ring (points)',
        description: 'Percentage point excess of each facing direction inside the perimeters'
      });
    } else {
      ctx.setChart('aspectChart', null);
      ctx.setChart('aspectDifferenceChart', null);
    }
    ctx.requestLayers();
  };

  const stageReader = new SummaryReader(resources, 'wildfire-zonal', zonalSources, bytes => {
    if (destroyed) return;
    table = parseTable(bytes);
    analyze();
    if (selected >= 0) describeSelected();
  });
  const rasterReader = new SummaryReader(
    resources,
    'wildfire-raster',
    [
      {buffer: rasterOverflow, size: 4},
      {buffer: rasterCrossings, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      ctx.setReadout(
        'rasterizer',
        `${formatCount(words[1])} of ${formatCount(crossingCapacity)} crossing records${words[0] ? ' (OVERFLOW: zones are empty)' : ''}`
      );
    }
  );

  const describeSelected = () => {
    if (selected < 0) {
      ctx.setReadout('selected', 'Click a fire');
      return;
    }
    const t = table;
    if (!t) {
      ctx.setReadout('selected', describeFire(data, selected));
      return;
    }
    const cells = (ring: number) => t.slopeCount[2 * selected + ring];
    const mean = (sums: Float32Array, counts: Uint32Array, ring: number) =>
      counts[2 * selected + ring]
        ? sums[2 * selected + ring] / counts[2 * selected + ring]
        : Number.NaN;
    ctx.setReadout(
      'selected',
      [
        describeFire(data, selected),
        `cells inside ${formatCount(cells(0))}, ring ${formatCount(cells(1))}`,
        `slope ${mean(t.slopeSum, t.slopeCount, 0).toFixed(1)} vs ${mean(t.slopeSum, t.slopeCount, 1).toFixed(1)} deg (max ${t.slopeMax[2 * selected].toFixed(0)})`,
        `elevation ${mean(t.elevationSum, t.elevationCount, 0).toFixed(0)} vs ${mean(t.elevationSum, t.elevationCount, 1).toFixed(0)} m`
      ].join('\n')
    );
  };

  const selectFire = (fire: number) => {
    selected = fire;
    selectedEdgeCount = 0;
    if (fire >= 0) {
      const segments: number[] = [];
      for (let edge = 0; edge < allEdges.edgeCount; edge++) {
        if (allEdges.featureRows[edge] === fire) {
          segments.push(
            allEdges.starts[edge * 2],
            allEdges.starts[edge * 2 + 1],
            allEdges.ends[edge * 2],
            allEdges.ends[edge * 2 + 1]
          );
        }
      }
      selectionSegments.write(Float32Array.from(segments));
      selectedEdgeCount = segments.length / 4;
    }
    describeSelected();
    analyze();
  };
  ctx.setReadout('selected', 'Click a fire');
  ctx.setReadout('rasterizer', 'pending');

  return {
    getCompiledGraphs: () => [rasterGraph, terrainGraph, statsGraph],

    setOption(id, _value, options) {
      lastChange = performance.now();
      switch (id) {
        case 'distanceMode':
        case 'refinement':
          resources.release(rasterGraph);
          rasterGraph = buildRaster(options);
          dirty.raster = true;
          dirty.stats = true;
          break;
        case 'borderMode':
          resources.release(terrainGraph);
          terrainGraph = buildTerrain(options);
          dirty.terrain = true;
          dirty.stats = true;
          break;
        case 'sumOrder':
          resources.release(statsGraph);
          statsGraph = buildStats(options);
          dirty.stats = true;
          break;
        case 'sunAzimuth':
        case 'sunAltitude':
          writeTerrainSettings(options);
          dirty.terrain = true;
          break;
        case 'zFactor':
          writeTerrainSettings(options);
          dirty.terrain = true;
          dirty.stats = true;
          break;
        case 'ringOuter':
        case 'ringInner':
        case 'flatSlope':
        case 'steepSlope':
          writeStageParameters(options);
          dirty.stats = true;
          break;
        case 'year':
        case 'minAcres':
          updateEnabled(options);
          dirty.stats = true;
          break;
        default:
          break;
      }
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [longitude, latitude] = event.coordinate;
      const [x, y] = toMercator(longitude, latitude);
      const column = Math.floor((x - boundsMercator[0]) / cellSize);
      const row = Math.floor((boundsMercator[3] - y) / cellSize);
      const elevation =
        column >= 0 && row >= 0 && column < width && row < height
          ? (demRaster.values as Float32Array)[row * width + column]
          : Number.NaN;
      const fire = findFeatureAt(layout, longitude, latitude);
      const terrain =
        Number.isFinite(elevation) && elevation > -9000 ? `${formatNumber(elevation)} m` : null;
      if (fire >= 0)
        return [data.names[fire], `${formatNumber(data.acres[fire])} acres`, terrain]
          .filter(Boolean)
          .join(' · ');
      return terrain;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const fire = findFeatureAt(layout, event.coordinate[0], event.coordinate[1]);
      selectFire(fire < 0 || fire === selected || !inDem[fire] ? -1 : fire);
      return true;
    },

    encode(commandEncoder) {
      const now = performance.now();
      if (dirty.raster) {
        rasterGraph.encode(commandEncoder, {parameters: undefined});
        rasterReader.request(commandEncoder);
        dirty.raster = false;
      }
      if (dirty.terrain) {
        terrainGraph.encode(commandEncoder, {parameters: undefined});
        dirty.terrain = false;
      }
      // The zonal graph runs five sorts over the whole grid: while a slider is dragged, at most every 150 ms.
      if (dirty.stats && now - lastStatsEncode > 150 && now - lastChange > 40) {
        statsGraph.encode(commandEncoder, {parameters: undefined});
        stageReader.request(commandEncoder);
        dirty.stats = false;
        lastStatsEncode = now;
      } else {
        stageReader.flush(commandEncoder);
        rasterReader.flush(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const common = {
        coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
        gridSize: [width, height] as [number, number],
        bounds: boundsMercator,
        rowOrigin: 'north' as const
      };
      if (options.showHillshade || options.display === 'hillshade') {
        layers.push(
          new WildfireMercatorRasterLayer({
            id: 'hillshade',
            ...common,
            values: hillshade,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 1],
            noDataColor: [0, 0, 0, 0],
            opacity: dark ? 0.75 : 0.9
          })
        );
      }
      if (options.display === 'zones') {
        layers.push(
          new WildfireMercatorRasterLayer({
            id: 'zones',
            ...common,
            values: classes,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [[0, 0, 0, 0], INSIDE_COLOR, RING_COLOR],
            opacity: options.overlayOpacity
          })
        );
      } else if (options.display === 'boundary') {
        layers.push(
          new WildfireMercatorRasterLayer({
            id: 'boundary',
            ...common,
            values: boundary,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [
              [0, 0, 0, 0],
              [255, 244, 214, 255]
            ],
            opacity: options.overlayOpacity
          })
        );
      } else if (options.display === 'slope') {
        layers.push(
          new WildfireMercatorRasterLayer({
            id: 'slope',
            ...common,
            values: slope,
            valueFormat: 'float32',
            colormap: options.ramp,
            valueRange: SLOPE_RANGE,
            noDataColor: [0, 0, 0, 0],
            opacity: options.overlayOpacity
          })
        );
      } else if (options.display === 'southness') {
        layers.push(
          new WildfireMercatorRasterLayer({
            id: 'southness',
            ...common,
            values: southness,
            valueFormat: 'float32',
            colormap: 'diverging',
            valueRange: [-1, 1],
            noDataColor: [0, 0, 0, 0],
            opacity: options.overlayOpacity
          })
        );
      }
      if (options.showOutlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'fire-outline',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: outlineBuffer,
            instanceCount: outlineCount,
            valueIndices: outlineRowBuffer,
            values: enabledFloat,
            valueFormat: 'float32',
            colormap: 'mask',
            color: dark ? [255, 235, 190, 235] : [60, 30, 10, 235],
            noDataColor: [0, 0, 0, 0],
            widthPixels: 1.2
          })
        );
      }
      if (selected >= 0 && selectedEdgeCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'fire-selection',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: selectionSegments,
            instanceCount: selectedEdgeCount,
            color: [255, 214, 90, 255],
            widthPixels: 3
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      stageReader.stop();
      rasterReader.stop();
      resources.destroy();
    }
  };
}

/** Longitude and latitude of the DEM center, for the initial view and cameras. */
export function getDemCenter(bounds: readonly [number, number, number, number]): [number, number] {
  return fromMercator((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2);
}
