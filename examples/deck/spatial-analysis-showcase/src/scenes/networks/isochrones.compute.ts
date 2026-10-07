// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPU_NETWORK_REACHABILITY_NONE,
  getGPUNetworkIsochroneParameterValues,
  GPUNetworkIsochrones,
  GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-network';
import {getGPUDistanceFieldParameterValues} from '@luma.gl/experimental/gpu-raster';
import {
  addDriveTimeCatchmentRecipe,
  addStraightLineCatchmentsRecipe,
  GPU_SPATIAL_JOIN_NO_FEATURE,
  GPUPointInPolygonJoin
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {fetchText, getDataFileUrl, parseCsv} from '../../data/loaders';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {RampName} from '../../engine/ramps';
import {getViewportMetricBounds, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {IsobandTriangleLayer, PolylineLayer} from './b9-network-layers';
import {buildRoadNetwork, writeDriveCosts} from './b9-road-network';
import {
  CATEGORY_PALETTE,
  createPackedPalette,
  formatInteger,
  getBandColor,
  getRoadColors,
  sliceSections
} from './b9-shared';
import {buildTractDemand, rasterizeTractValues} from './b9-tracts';

/** Option state of the isochrones scene. */
export type IsochronesOptions = {
  facilityType: 'fire_station' | 'hospital' | 'library' | 'cps_school';
  budgetMinutes: number;
  bandCount: number;
  trafficSlowdown: number;
  intersectionDelay: number;
  walkBuffer: number;
  rasterMode: 'min' | 'max';
  cellChoice: 'h3-8' | 'h3-9' | 'quadbin-16' | 'quadbin-17';
  straightRadius: number;
  distanceMode: 'exact' | 'jump-flood';
  sumOrder: 'atomic' | 'sorted';
  showBands: boolean;
  showServiceAreas: boolean;
  showRings: boolean;
  showStraightLine: boolean;
  showDemand: boolean;
  showStreets: boolean;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
};

const CATEGORY_INDEX: Record<IsochronesOptions['facilityType'], number> = {
  hospital: 0,
  library: 1,
  cps_school: 2,
  fire_station: 3
};
const CELL_CHOICES: Record<
  IsochronesOptions['cellChoice'],
  {family: 'quadbin' | 'h3'; resolution: number}
> = {
  'h3-8': {family: 'h3', resolution: 8},
  'h3-9': {family: 'h3', resolution: 9},
  'quadbin-16': {family: 'quadbin', resolution: 16},
  'quadbin-17': {family: 'quadbin', resolution: 17}
};

const MAXIMUM_BREAKS = 6;
const MAXIMUM_ITERATIONS = 64;
const RASTER_WIDTH = 1024;
const RASTER_HEIGHT = 640;
const RASTER_ASPECT = RASTER_WIDTH / RASTER_HEIGHT;
const MAXIMUM_BUFFER_PIXELS = 16;
const MAXIMUM_SAMPLES_PER_EDGE = 256;
const TRIANGLE_CAPACITY = 800_000;
const SEGMENT_CAPACITY = 300_000;
const RING_CAPACITY = 2048;
const RING_VERTEX_CAPACITY = 32768;
const PALETTE_SIZE = 64;
const BAND_ALPHA = 165;
const BAND_TABLE_CAPACITY = 8;
const WALK_SECONDS_PER_METER = 1 / 1.34;
const NO_BAND = 0xffffffff;
const EXTENT_THROTTLE_MILLISECONDS = 350;
const STRAIGHT_LINE_WIDTH = 384;

type BuiltGraphs = {
  catchment: CompiledGPUCommandGraph<void>;
  straight: CompiledGPUCommandGraph<void>;
  facilityCount: number;
};

/**
 * Drive-time and straight-line catchments of Chicago's public facilities. Two compiled graphs:
 *
 * - catchment: `addDriveTimeCatchmentRecipe` (snap the facilities to street edges, multi-source
 *   `GPUNetworkServiceAreas`, raster `GPUNetworkIsochrones` bands, demand snapped and classified
 *   into drive-time bands and summarised per band), then a second `GPUNetworkIsochrones` producer
 *   that outlines the reached H3 or Quadbin cells into rings, and a `GPUPointInPolygonJoin` of the
 *   tract centroids against those rings;
 * - straight: `addStraightLineCatchmentsRecipe` (`GPUDistanceField` Voronoi zones and
 *   `GPURasterZonalStatistics` of a population raster).
 *
 * The budget, bands, traffic, buffer, radius and moved facilities are buffer writes; the facility
 * set, cell family, raster mode, distance-field algorithm and sum order rebuild the graphs.
 */
export async function createIsochrones(
  ctx: SceneContext<IsochronesOptions>
): Promise<SceneInstance<IsochronesOptions>> {
  const {device} = ctx;
  const roads = ctx.datasets.get('chicago-roads');
  const network = buildRoadNetwork(roads);
  const {nodeCount, edgeCount} = network;
  const facilitiesDataset = ctx.datasets.get('chicago-facilities');
  const demand = buildTractDemand(ctx.datasets.get('chicago-tracts'), network.origin);
  const resources = new SpatialAnalysisResources(device, 'isochrones');
  const segmentCount = network.segmentEdges.length;

  let facilityNames: string[] = [];
  try {
    const text = await fetchText(getDataFileUrl('chicago-facilities', 'names.csv'), ctx.signal);
    facilityNames = parseCsv(text).rows.map(row => row[1] ?? '');
  } catch {
    // Names are only used for tooltips.
  }
  const facilityPosition = facilitiesDataset.projectColumn('position', network.origin);
  const facilityCategory = facilitiesDataset.column<Uint8Array>('category');
  const totalPopulation = demand.population.reduce((sum, value) => sum + value, 0);

  // ---- Static buffers ---------------------------------------------------------------------
  const offsetsBuffer = resources.createBuffer('offsets', network.offsets);
  const neighborsBuffer = resources.createBuffer('neighbors', network.targets);
  const weightsBuffer = resources.createBuffer('weights', edgeCount * 4);
  const nodePositionsBuffer = resources.createBuffer('node-positions', network.nodePositions);
  const nodeLngLatBuffer = resources.createBuffer('node-lnglat', network.nodeLngLat);
  const segmentsBuffer = resources.createBuffer('segments', network.segments);
  const majorSegmentsBuffer = resources.createBuffer('major-segments', network.majorSegments);
  const segmentSourcesBuffer = resources.createBuffer(
    'segment-sources',
    network.segmentSourceNodes
  );
  const demandPositionsBuffer = resources.createBuffer('demand-positions', demand.centroids);
  const demandLngLatBuffer = resources.createBuffer('demand-lnglat', demand.centroidsLngLat);
  const demandPopulationBuffer = resources.createBuffer('demand-population', demand.population);

  // Straight-line raster: the whole city at about 90 m per cell, with people spread over tracts.
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let node = 0; node < nodeCount; node++) {
    minX = Math.min(minX, network.nodePositions[node * 2]);
    maxX = Math.max(maxX, network.nodePositions[node * 2]);
    minY = Math.min(minY, network.nodePositions[node * 2 + 1]);
    maxY = Math.max(maxY, network.nodePositions[node * 2 + 1]);
  }
  const straightBounds = [minX - 400, minY - 400, maxX + 400, maxY + 400] as const;
  const straightHeight = Math.round(
    (STRAIGHT_LINE_WIDTH * (straightBounds[3] - straightBounds[1])) /
      (straightBounds[2] - straightBounds[0])
  );
  const straightCellCount = STRAIGHT_LINE_WIDTH * straightHeight;
  const populationRaster = rasterizeTractValues(
    demand,
    demand.population,
    straightBounds,
    STRAIGHT_LINE_WIDTH,
    straightHeight
  );
  const populationRasterBuffer = resources.createBuffer('population-raster', populationRaster);
  const allocationBuffer = resources.createBuffer('allocation', straightCellCount * 4);
  const distancesBuffer = resources.createBuffer('distances', straightCellCount * 4);
  const distanceSettings = resources.createParameterBuffer('distance-settings', 'float32', 8);

  // ---- Parameter buffers ------------------------------------------------------------------
  const costLimitParameter = resources.createParameterBuffer('cost-limit', 'float32', 1);
  const breaksParameter = resources.createParameterBuffer('breaks', 'float32', MAXIMUM_BREAKS);
  const isochroneParameters = resources.createParameterBuffer(
    'isochrone-parameters',
    'float32',
    GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH
  );
  const paletteBuffer = resources.createBuffer(
    'palette',
    createPackedPalette(ctx.options.ramp, PALETTE_SIZE, BAND_ALPHA)
  );

  // ---- Network outputs --------------------------------------------------------------------
  const assignmentsBuffer = resources.createBuffer('assignments', nodeCount * 4);
  const nodeFacilityBuffer = resources.createBuffer('node-facility', nodeCount * 4);
  const nodeCostsBuffer = resources.createBuffer('node-costs', nodeCount * 4);
  const demandTimesBuffer = resources.createBuffer('demand-times', demand.count * 4);
  const demandBandsBuffer = resources.createBuffer('demand-bands', demand.count * 4);
  const bandKeys = resources.createBuffer('band-keys', BAND_TABLE_CAPACITY * 4);
  const bandCounts = resources.createBuffer('band-counts', BAND_TABLE_CAPACITY * 4);
  const bandCount = resources.createBuffer('band-count', 4);
  const bandOverflow = resources.createBuffer('band-overflow', 4);
  const bandSums = resources.createBuffer('band-sums', BAND_TABLE_CAPACITY * 4);
  const bandMeans = resources.createBuffer('band-means', BAND_TABLE_CAPACITY * 4);
  const triangles = resources.createBuffer('triangles', TRIANGLE_CAPACITY * 3 * 8);
  const triangleBands = resources.createBuffer('triangle-bands', TRIANGLE_CAPACITY * 4);
  const triangleCount = resources.createBuffer('triangle-count', 4);
  const triangleOverflow = resources.createBuffer('triangle-overflow', 4);
  const bandVertexCount = resources.createBuffer('band-vertex-count', 4);
  const tableCells = resources.createBuffer('table-cells', nodeCount * 8);
  const tableCounts = resources.createBuffer('table-counts', nodeCount * 4);
  const tableCount = resources.createBuffer('table-count', 4);
  const tableOverflow = resources.createBuffer('table-overflow', 4);
  const outlineRows = resources.createBuffer('outline-rows', SEGMENT_CAPACITY * 4);
  const outlineCells = resources.createBuffer('outline-cells', SEGMENT_CAPACITY * 8);
  const outlineEdges = resources.createBuffer('outline-edges', SEGMENT_CAPACITY * 4);
  const outlineEndpoints = resources.createBuffer('outline-endpoints', SEGMENT_CAPACITY * 16);
  const outlineCount = resources.createBuffer('outline-count', 4);
  const outlineOverflow = resources.createBuffer('outline-overflow', 4);
  const outlineTotal = resources.createBuffer('outline-total', 4);
  const ringOffsets = resources.createBuffer('ring-offsets', (RING_CAPACITY + 1) * 4);
  const ringPositions = resources.createBuffer('ring-positions', RING_VERTEX_CAPACITY * 8);
  const ringIsHole = resources.createBuffer('ring-is-hole', RING_CAPACITY * 4);
  const ringShells = resources.createBuffer('ring-shells', RING_CAPACITY * 4);
  const ringCount = resources.createBuffer('ring-count', 4);
  const ringOverflow = resources.createBuffer('ring-overflow', 4);
  const ringTotal = resources.createBuffer('ring-total', 4);
  const ringOpen = resources.createBuffer('ring-open', 4);
  const ringTouching = resources.createBuffer('ring-touching', 4);
  const polygonPositions = resources.createBuffer('polygon-positions', RING_VERTEX_CAPACITY * 8);
  const polygonRingOffsets = resources.createBuffer(
    'polygon-ring-offsets',
    (RING_CAPACITY + 1) * 4
  );
  const polygonOffsets = resources.createBuffer('polygon-offsets', (RING_CAPACITY + 1) * 4);
  const polygonFeatureOffsets = resources.createBuffer('polygon-feature-offsets', 8);
  const demandInside = resources.createBuffer('demand-inside', demand.count * 4);
  const joinOverflow = resources.createBuffer('join-overflow', 4);
  const drawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'isochrones-bands-draw',
      type: 'draw',
      commands: [{vertexCount: 0, instanceCount: 1}]
    })
  );

  // ---- State ------------------------------------------------------------------------------
  let destroyed = false;
  let built: BuiltGraphs | null = null;
  let perBuild: {destroy: () => void}[] = [];
  let facilityPositionsBuffer!: Buffer;
  let facilityIdsBuffer!: Buffer;
  let seedCountParameter!: {write: (values: Uint32Array) => void; importToGraph: never};
  let statisticsBuffers: Buffer[] = [];
  let facilityRows: number[] = [];
  let facilityPositions = new Float32Array(0);
  let movedFacilities = new Set<number>();
  let catchmentDirty = 2;
  let straightDirty = 2;
  let extent: [number, number, number, number] | null = null;
  let extentKey = '';
  let extentChanged = false;
  let lastCatchmentEncode = 0;
  let builtOptions = {
    facilityType: ctx.options.facilityType,
    cellChoice: ctx.options.cellChoice,
    rasterMode: ctx.options.rasterMode,
    distanceMode: ctx.options.distanceMode,
    sumOrder: ctx.options.sumOrder
  };

  const writeCosts = () => {
    const options = ctx.options;
    const costs = new Float32Array(edgeCount);
    writeDriveCosts(
      network,
      {
        closeExpressways: false,
        expresswaySlowdown: 1,
        allRoadsSlowdown: options.trafficSlowdown,
        intersectionDelay: options.intersectionDelay
      },
      costs
    );
    weightsBuffer.write(costs);
  };

  const writeBudget = () => {
    const options = ctx.options;
    const budgetSeconds = options.budgetMinutes * 60;
    costLimitParameter.write(Float32Array.of(budgetSeconds + 180));
    const values = new Float32Array(MAXIMUM_BREAKS);
    for (let band = 0; band < MAXIMUM_BREAKS; band++) {
      values[band] = (budgetSeconds * Math.min(band + 1, options.bandCount)) / options.bandCount;
    }
    breaksParameter.write(values);
    catchmentDirty = Math.max(catchmentDirty, 1);
  };

  const writeStraightSettings = () => {
    distanceSettings.write(
      getGPUDistanceFieldParameterValues({
        bounds: straightBounds,
        gridSize: [STRAIGHT_LINE_WIDTH, straightHeight],
        maxDistance: ctx.options.straightRadius
      })
    );
    straightDirty = Math.max(straightDirty, 1);
  };

  const writeFacilities = () => {
    facilityPositionsBuffer.write(facilityPositions);
    catchmentDirty = Math.max(catchmentDirty, 1);
    straightDirty = Math.max(straightDirty, 1);
    ctx.setReadout('moved', movedFacilities.size ? `${movedFacilities.size} moved` : 'none');
  };

  const loadFacilityRows = () => {
    const category = CATEGORY_INDEX[ctx.options.facilityType];
    facilityRows = [];
    for (let row = 0; row < facilityCategory.length; row++) {
      if (facilityCategory[row] === category) facilityRows.push(row);
    }
    facilityPositions = new Float32Array(facilityRows.length * 2);
    facilityRows.forEach((row, slot) => {
      facilityPositions[slot * 2] = facilityPosition[row * 2];
      facilityPositions[slot * 2 + 1] = facilityPosition[row * 2 + 1];
    });
    movedFacilities = new Set();
  };

  // ---- Summaries --------------------------------------------------------------------------
  let statisticsReader: SummaryReader | null = null;
  const summarySizes = [
    BAND_TABLE_CAPACITY * 4,
    BAND_TABLE_CAPACITY * 4,
    4,
    4,
    BAND_TABLE_CAPACITY * 4,
    4,
    4,
    4,
    4,
    4,
    4,
    4
  ];
  const catchmentReader = new SummaryReader(
    resources,
    'isochrones-catchment',
    [
      {buffer: bandKeys, size: summarySizes[0]},
      {buffer: bandCounts, size: summarySizes[1]},
      {buffer: bandCount, size: 4},
      {buffer: bandOverflow, size: 4},
      {buffer: bandSums, size: summarySizes[4]},
      {buffer: triangleCount, size: 4},
      {buffer: triangleOverflow, size: 4},
      {buffer: tableCount, size: 4},
      {buffer: outlineCount, size: 4},
      {buffer: outlineOverflow, size: 4},
      {buffer: ringCount, size: 4},
      {buffer: ringOverflow, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const [
        keyBytes,
        countBytes,
        tableRows,
        tableOverflowBytes,
        sumBytes,
        triangleWords,
        triangleOver,
        cells,
        outline,
        outlineOver,
        rings,
        ringOver
      ] = sliceSections(bytes, summarySizes);
      const keys = new Uint32Array(keyBytes);
      const counts = new Uint32Array(countBytes);
      const sums = new Float32Array(sumBytes);
      const rowCount = new Uint32Array(tableRows)[0];
      const options = ctx.options;
      const perBand = new Float32Array(MAXIMUM_BREAKS);
      for (let row = 0; row < Math.min(rowCount, BAND_TABLE_CAPACITY); row++) {
        if (keys[row] < MAXIMUM_BREAKS) perBand[keys[row]] = sums[row];
      }
      let cumulative = 0;
      const parts: string[] = [];
      for (let band = 0; band < options.bandCount; band++) {
        cumulative += perBand[band];
        const minutes = (options.budgetMinutes * (band + 1)) / options.bandCount;
        parts.push(
          `${minutes % 1 === 0 ? minutes : minutes.toFixed(1)} min ${((100 * cumulative) / Math.max(totalPopulation, 1)).toFixed(0)}%`
        );
      }
      ctx.setReadout('servedBands', parts.join(' / '));
      ctx.setReadout(
        'servedWithin',
        `${formatInteger(cumulative)} of ${formatInteger(totalPopulation)}`
      );
      let unreachedTracts = 0;
      for (let row = 0; row < Math.min(rowCount, BAND_TABLE_CAPACITY); row++) {
        if (keys[row] === NO_BAND) unreachedTracts = counts[row];
      }
      ctx.setReadout(
        'unreachedTracts',
        `${formatInteger(unreachedTracts)} of ${formatInteger(demand.count)}`
      );
      ctx.setReadout(
        'triangles',
        `${formatInteger(new Uint32Array(triangleWords)[0])} of ${formatInteger(TRIANGLE_CAPACITY)}` +
          (new Uint32Array(triangleOver)[0] ? ' OVERFLOW' : '')
      );
      ctx.setReadout(
        'cells',
        `${formatInteger(new Uint32Array(cells)[0])} reached, ${formatInteger(new Uint32Array(outline)[0])} outline segments` +
          (new Uint32Array(outlineOver)[0] ? ' OVERFLOW' : '')
      );
      ctx.setReadout(
        'rings',
        `${formatInteger(new Uint32Array(rings)[0])}${new Uint32Array(ringOver)[0] ? ' OVERFLOW' : ''}`
      );
      void tableOverflowBytes;
    }
  );
  const demandSizes = [4, 4, 4, 4, 4, RING_CAPACITY * 4, demand.count * 4, demand.count * 4];
  const ringReader = new SummaryReader(
    resources,
    'isochrones-rings',
    [
      {buffer: ringCount, size: 4},
      {buffer: ringOverflow, size: 4},
      {buffer: ringOpen, size: 4},
      {buffer: ringTouching, size: 4},
      {buffer: joinOverflow, size: 4},
      {buffer: ringIsHole, size: RING_CAPACITY * 4},
      {buffer: demandInside, size: demand.count * 4},
      {buffer: demandBandsBuffer, size: demand.count * 4}
    ],
    bytes => {
      if (destroyed) return;
      const [ringWords, , openWords, touchingWords, , holeBytes, insideBytes, bandBytes] =
        sliceSections(bytes, demandSizes);
      const ringTotalCount = Math.min(new Uint32Array(ringWords)[0], RING_CAPACITY);
      const holes = new Uint32Array(holeBytes);
      let holeCount = 0;
      for (let ring = 0; ring < ringTotalCount; ring++) holeCount += holes[ring] ? 1 : 0;
      const inside = new Uint32Array(insideBytes);
      const bands = new Uint32Array(bandBytes);
      let residentsInside = 0;
      let residentsBands = 0;
      for (let tract = 0; tract < demand.count; tract++) {
        if (inside[tract] !== GPU_SPATIAL_JOIN_NO_FEATURE)
          residentsInside += demand.population[tract];
        if (bands[tract] !== NO_BAND) residentsBands += demand.population[tract];
      }
      ctx.setReadout(
        'ringJoin',
        `${formatInteger(residentsInside)} residents (${((100 * residentsInside) / Math.max(totalPopulation, 1)).toFixed(0)}%)`
      );
      ctx.setReadout(
        'ringJoinGap',
        `${residentsInside >= residentsBands ? '+' : ''}${formatInteger(residentsInside - residentsBands)} residents`
      );
      ctx.setReadout(
        'ringShapes',
        `${formatInteger(ringTotalCount - holeCount)} shells / ${formatInteger(holeCount)} holes`
      );
      ctx.setReadout(
        'ringHealth',
        `${formatInteger(new Uint32Array(openWords)[0])} open / ${formatInteger(new Uint32Array(touchingWords)[0])} touching`
      );
    }
  );

  // ---- Graph construction -----------------------------------------------------------------
  function buildGraphs(): void {
    if (built) {
      statisticsReader?.stop();
      resources.release(built.catchment);
      resources.release(built.straight);
      for (const resource of perBuild) resources.release(resource);
      perBuild = [];
    }
    loadFacilityRows();
    const facilityCount = facilityRows.length;
    const options = ctx.options;
    const track = <T extends {destroy: () => void}>(resource: T): T => {
      perBuild.push(resource);
      return resource;
    };
    facilityPositionsBuffer = track(
      resources.createBuffer('facility-positions', facilityPositions)
    );
    facilityIdsBuffer = track(
      resources.createBuffer(
        'facility-ids',
        Uint32Array.from({length: facilityCount}, (_, index) => index)
      )
    );
    const seedCount = track(resources.createParameterBuffer('seed-count', 'uint32', 1));
    seedCount.write(Uint32Array.of(facilityCount));
    seedCountParameter = seedCount as never;
    const statistics = ['cell-counts', 'value-counts', 'sums', 'means', 'minimums', 'maximums'].map(
      name => track(resources.createBuffer(`stat-${name}`, facilityCount * 4))
    );
    statisticsBuffers = statistics;

    const importer = (graph: GPUCommandGraph<void>) => {
      const cache = new Map<Buffer, GraphDataView>();
      return <Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x2' | 'float32x4'>(
        buffer: Buffer,
        format: Format,
        length: number
      ) => {
        let view = cache.get(buffer);
        if (!view) {
          view = importGraphBuffer(graph, buffer.id, buffer, format, length) as GraphDataView;
          cache.set(buffer, view);
        }
        return view as unknown as GraphDataView<Format>;
      };
    };

    // -- drive-time catchments
    const catchmentGraph = new GPUCommandGraph<void>(device, {id: 'isochrones-catchment'});
    {
      const view = importer(catchmentGraph);
      const offsets = view(offsetsBuffer, 'uint32', nodeCount + 1);
      const neighbors = view(neighborsBuffer, 'uint32', edgeCount);
      const weights = view(weightsBuffer, 'float32', edgeCount);
      const breaks = breaksParameter.importToGraph(catchmentGraph);
      const parameters = isochroneParameters.importToGraph(catchmentGraph);
      const nodeCostsView = view(nodeCostsBuffer, 'float32', nodeCount);
      const assignments = view(assignmentsBuffer, 'uint32', nodeCount);
      const recipe = addDriveTimeCatchmentRecipe(catchmentGraph, {
        id: 'catchment',
        network: {
          offsets,
          neighbors,
          weights,
          nodePositions: view(nodePositionsBuffer, 'float32x2', nodeCount)
        },
        facilities: view(facilityPositionsBuffer, 'float32x2', facilityCount),
        demand: view(demandPositionsBuffer, 'float32x2', demand.count),
        demandValues: view(demandPopulationBuffer, 'float32', demand.count),
        costLimit: costLimitParameter.importToGraph(catchmentGraph),
        maxIterations: MAXIMUM_ITERATIONS,
        bandBreaks: breaks,
        bandCapacity: BAND_TABLE_CAPACITY,
        assignments,
        nodeCosts: nodeCostsView,
        demandTimes: view(demandTimesBuffer, 'float32', demand.count),
        demandBands: view(demandBandsBuffer, 'uint32', demand.count),
        bands: {
          keys: view(bandKeys, 'uint32', BAND_TABLE_CAPACITY),
          counts: view(bandCounts, 'uint32', BAND_TABLE_CAPACITY),
          count: view(bandCount, 'uint32', 1),
          overflow: view(bandOverflow, 'uint32', 1),
          sumValues: view(bandSums, 'float32', BAND_TABLE_CAPACITY),
          means: view(bandMeans, 'float32', BAND_TABLE_CAPACITY)
        },
        isochrones: {
          breaks,
          parameters,
          raster: {
            width: RASTER_WIDTH,
            height: RASTER_HEIGHT,
            mode: options.rasterMode,
            maximumBufferPixels: MAXIMUM_BUFFER_PIXELS,
            maximumSamplesPerEdge: MAXIMUM_SAMPLES_PER_EDGE,
            output: {
              triangles: view(triangles, 'float32x2', TRIANGLE_CAPACITY * 3),
              triangleBands: view(triangleBands, 'uint32', TRIANGLE_CAPACITY),
              count: view(triangleCount, 'uint32', 1),
              overflow: view(triangleOverflow, 'uint32', 1),
              vertexCount: view(bandVertexCount, 'uint32', 1)
            }
          }
        }
      });
      void recipe;
      // Cell producer: outlines the cells that hold reached nodes into closed rings.
      const choice = CELL_CHOICES[options.cellChoice];
      const polygonPositionsView = view(polygonPositions, 'float32x2', RING_VERTEX_CAPACITY);
      const polygonRingOffsetsView = view(polygonRingOffsets, 'uint32', RING_CAPACITY + 1);
      const polygonOffsetsView = view(polygonOffsets, 'uint32', RING_CAPACITY + 1);
      const polygonFeatureOffsetsView = view(polygonFeatureOffsets, 'uint32', 2);
      catchmentGraph.add(
        new GPUNetworkIsochrones({
          id: 'catchment-rings',
          offsets,
          neighbors,
          weights,
          nodePositions: view(nodeLngLatBuffer, 'float32x2', nodeCount),
          costs: nodeCostsView,
          breaks,
          parameters,
          cellOutline: {
            family: choice.family,
            resolution: choice.resolution,
            table: {
              cells: view(tableCells, 'uint32x2', nodeCount),
              counts: view(tableCounts, 'uint32', nodeCount),
              count: view(tableCount, 'uint32', 1),
              overflow: view(tableOverflow, 'uint32', 1)
            },
            rings: {
              normalizeWinding: true,
              output: {
                ringOffsets: view(ringOffsets, 'uint32', RING_CAPACITY + 1),
                positions: view(ringPositions, 'float32x2', RING_VERTEX_CAPACITY),
                ringIsHole: view(ringIsHole, 'uint32', RING_CAPACITY),
                ringShells: view(ringShells, 'uint32', RING_CAPACITY),
                count: view(ringCount, 'uint32', 1),
                overflow: view(ringOverflow, 'uint32', 1),
                totalCount: view(ringTotal, 'uint32', 1),
                openSegmentCount: view(ringOpen, 'uint32', 1),
                touchingSegmentCount: view(ringTouching, 'uint32', 1),
                polygons: {
                  positions: polygonPositionsView,
                  ringOffsets: polygonRingOffsetsView,
                  polygonOffsets: polygonOffsetsView,
                  featureOffsets: polygonFeatureOffsetsView
                }
              }
            },
            output: {
              rows: view(outlineRows, 'uint32', SEGMENT_CAPACITY),
              cells: view(outlineCells, 'uint32x2', SEGMENT_CAPACITY),
              edgeIndices: view(outlineEdges, 'uint32', SEGMENT_CAPACITY),
              endpoints: view(outlineEndpoints, 'float32x4', SEGMENT_CAPACITY),
              count: view(outlineCount, 'uint32', 1),
              overflow: view(outlineOverflow, 'uint32', 1),
              totalCount: view(outlineTotal, 'uint32', 1)
            }
          }
        })
      );
      catchmentGraph.add(
        new GPUPointInPolygonJoin({
          id: 'catchment-join',
          points: view(demandLngLatBuffer, 'float32x2', demand.count),
          polygonPositions: polygonPositionsView,
          featureOffsets: polygonFeatureOffsetsView,
          polygonOffsets: polygonOffsetsView,
          ringOffsets: polygonRingOffsetsView,
          candidateCapacity: Math.max(65536, demand.count * 8),
          pointFeatureIds: view(demandInside, 'uint32', demand.count),
          overflow: view(joinOverflow, 'uint32', 1)
        })
      );
      // Display adapter: the facility (not the snapped seed row) that serves each node.
      addKernelPass(catchmentGraph, {
        id: 'node-facility',
        bindings: [
          {name: 'assignments', view: assignments, type: 'u32', access: 'read'},
          {
            name: 'facilities',
            view: view(nodeFacilityBuffer, 'uint32', nodeCount),
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: nodeCount,
        body: `let seed = assignments[assignmentsOffset + index];
  facilities[facilitiesOffset + index] = select(seed / 2u, 0xffffffffu, seed == 0xffffffffu);`
      });
    }

    // -- straight-line catchments
    const straightGraph = new GPUCommandGraph<void>(device, {id: 'isochrones-straight'});
    {
      const view = importer(straightGraph);
      addStraightLineCatchmentsRecipe(straightGraph, {
        id: 'straight',
        width: STRAIGHT_LINE_WIDTH,
        height: straightHeight,
        settings: distanceSettings.importToGraph(straightGraph),
        seedPositions: view(facilityPositionsBuffer, 'float32x2', facilityCount),
        seedCount: seedCount.importToGraph(straightGraph),
        values: view(populationRasterBuffer, 'float32', straightCellCount),
        mode: options.distanceMode,
        sumOrder: options.sumOrder,
        allocation: view(allocationBuffer, 'uint32', straightCellCount),
        distances: view(distancesBuffer, 'float32', straightCellCount),
        statistics: {
          cellCounts: view(statistics[0], 'uint32', facilityCount),
          valueCounts: view(statistics[1], 'uint32', facilityCount),
          sums: view(statistics[2], 'float32', facilityCount),
          means: view(statistics[3], 'float32', facilityCount),
          minimums: view(statistics[4], 'float32', facilityCount),
          maximums: view(statistics[5], 'float32', facilityCount)
        }
      });
    }

    built = {
      catchment: resources.track(catchmentGraph.compile()),
      straight: resources.track(straightGraph.compile()),
      facilityCount
    };
    builtOptions = {
      facilityType: options.facilityType,
      cellChoice: options.cellChoice,
      rasterMode: options.rasterMode,
      distanceMode: options.distanceMode,
      sumOrder: options.sumOrder
    };
    statisticsReader = new SummaryReader(
      resources,
      `isochrones-straight-${facilityCount}`,
      [
        {buffer: statistics[0], size: facilityCount * 4},
        {buffer: statistics[2], size: facilityCount * 4}
      ],
      bytes => {
        if (destroyed) return;
        const cellCounts = new Uint32Array(bytes, 0, facilityCount);
        const sums = new Float32Array(bytes, facilityCount * 4, facilityCount);
        let people = 0;
        let largest = 0;
        let largestSlot = 0;
        let covered = 0;
        for (let slot = 0; slot < facilityCount; slot++) {
          people += sums[slot];
          covered += cellCounts[slot];
          if (sums[slot] > largest) {
            largest = sums[slot];
            largestSlot = slot;
          }
        }
        ctx.setReadout(
          'straightServed',
          `${formatInteger(people)} residents (${((100 * people) / Math.max(totalPopulation, 1)).toFixed(0)}%)`
        );
        ctx.setReadout(
          'straightCells',
          `${formatInteger(covered)} of ${formatInteger(straightCellCount)} cells`
        );
        const name = facilityNames[facilityRows[largestSlot]] ?? `facility ${largestSlot + 1}`;
        ctx.setReadout('busiestZone', `${name}: ${formatInteger(largest)} residents`);
      }
    );
    ctx.setReadout('facilities', formatInteger(facilityCount));
    ctx.setReadout('moved', 'none');
    catchmentDirty = 2;
    straightDirty = 2;
  }

  ctx.setReadout('nodes', `${formatInteger(nodeCount)} / ${formatInteger(edgeCount)}`);
  ctx.setReadout(
    'demandPoints',
    `${formatInteger(demand.count)} tracts, ${formatInteger(totalPopulation)} residents`
  );
  writeCosts();
  writeBudget();
  writeStraightSettings();
  buildGraphs();
  writeFacilities();

  const getBandPalette = (): [number, number, number, number][] =>
    Array.from({length: MAXIMUM_BREAKS}, (_, band) =>
      getBandColor(ctx.options.ramp as RampName, band, ctx.options.bandCount, 255)
    );

  return {
    getCompiledGraphs: () => (built ? [built.catchment, built.straight] : []),

    setOption(id, _value, state) {
      switch (id) {
        case 'facilityType':
        case 'cellChoice':
        case 'rasterMode':
        case 'distanceMode':
        case 'sumOrder':
          if (
            builtOptions.facilityType !== state.facilityType ||
            builtOptions.cellChoice !== state.cellChoice ||
            builtOptions.rasterMode !== state.rasterMode ||
            builtOptions.distanceMode !== state.distanceMode ||
            builtOptions.sumOrder !== state.sumOrder
          ) {
            buildGraphs();
            writeFacilities();
            ctx.requestLayers();
          }
          break;
        case 'budgetMinutes':
        case 'bandCount':
          writeBudget();
          ctx.requestLayers();
          break;
        case 'trafficSlowdown':
        case 'intersectionDelay':
          writeCosts();
          catchmentDirty = Math.max(catchmentDirty, 1);
          break;
        case 'walkBuffer':
          catchmentDirty = Math.max(catchmentDirty, 1);
          break;
        case 'straightRadius':
          writeStraightSettings();
          break;
        case 'ramp':
          paletteBuffer.write(createPackedPalette(state.ramp, PALETTE_SIZE, BAND_ALPHA));
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'reset') {
        loadFacilityRows();
        writeFacilities();
        ctx.requestLayers();
      }
    },

    onClick(event) {
      if (!event.coordinate || !built) return false;
      const [x, y] = network.projection.project(event.coordinate[0], event.coordinate[1]);
      let nearest = 0;
      let nearestDistance = Infinity;
      for (let slot = 0; slot < built.facilityCount; slot++) {
        const distance =
          (facilityPositions[slot * 2] - x) ** 2 + (facilityPositions[slot * 2 + 1] - y) ** 2;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = slot;
        }
      }
      facilityPositions[nearest * 2] = x;
      facilityPositions[nearest * 2 + 1] = y;
      movedFacilities.add(nearest);
      writeFacilities();
      return true;
    },

    getTooltip(event) {
      if (!event.coordinate || !built) return null;
      const [x, y] = network.projection.project(event.coordinate[0], event.coordinate[1]);
      let nearest = -1;
      let nearestDistance = 250 ** 2;
      for (let slot = 0; slot < built.facilityCount; slot++) {
        const distance =
          (facilityPositions[slot * 2] - x) ** 2 + (facilityPositions[slot * 2 + 1] - y) ** 2;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = slot;
        }
      }
      if (nearest < 0) return null;
      const name = facilityNames[facilityRows[nearest]];
      return movedFacilities.has(nearest) ? `${name ?? 'Facility'} (moved)` : (name ?? null);
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (!built) return;
      const options = ctx.options;
      // The isochrone raster follows the viewport; quantize its size and origin so panning is calm.
      const viewBounds = getViewportMetricBounds(frame.viewport, network.projection);
      const width =
        Math.max(viewBounds[2] - viewBounds[0], (viewBounds[3] - viewBounds[1]) * RASTER_ASPECT) *
        1.1;
      const step = Math.log(1.05);
      const snappedWidth = Math.exp(Math.round(Math.log(width) / step) * step);
      const pixelSize = snappedWidth / RASTER_WIDTH;
      const snap = pixelSize * 16;
      const centerX = Math.round((viewBounds[0] + viewBounds[2]) / 2 / snap) * snap;
      const centerY = Math.round((viewBounds[1] + viewBounds[3]) / 2 / snap) * snap;
      const nextExtent: [number, number, number, number] = [
        centerX - snappedWidth / 2,
        centerY - snappedWidth / RASTER_ASPECT / 2,
        centerX + snappedWidth / 2,
        centerY + snappedWidth / RASTER_ASPECT / 2
      ];
      const key = nextExtent.map(value => value.toFixed(2)).join(',');
      if (key !== extentKey) {
        extentKey = key;
        extent = nextExtent;
        extentChanged = true;
      }
      // The raster follows the camera, but the whole catchment graph re-runs with it, so camera
      // motion re-encodes at most a few times a second; option changes encode at once.
      const now = performance.now();
      const extentDue = extentChanged && now - lastCatchmentEncode > EXTENT_THROTTLE_MILLISECONDS;
      if ((catchmentDirty > 0 || extentDue) && extent) {
        lastCatchmentEncode = now;
        if (extentDue) extentChanged = false;
        const effectiveBuffer = Math.min(options.walkBuffer, MAXIMUM_BUFFER_PIXELS * pixelSize);
        isochroneParameters.write(
          getGPUNetworkIsochroneParameterValues({
            breakCount: options.bandCount,
            extent,
            bufferRadius: effectiveBuffer,
            walkCostPerUnit: WALK_SECONDS_PER_METER,
            cellCostLimit: options.budgetMinutes * 60
          })
        );
        ctx.setReadout(
          'raster',
          `${RASTER_WIDTH} x ${RASTER_HEIGHT}, ${pixelSize.toFixed(1)} m per pixel` +
            (effectiveBuffer < options.walkBuffer
              ? `; buffer capped at ${effectiveBuffer.toFixed(0)} m (zoom in)`
              : '')
        );
        built.catchment.encode(commandEncoder, {parameters: undefined});
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: bandVertexCount,
          destinationBuffer: drawCommands.buffer,
          destinationOffset: 0,
          size: 4
        });
        catchmentDirty = Math.max(0, catchmentDirty - 1);
        catchmentReader.markStale();
        ringReader.markStale();
      }
      if (straightDirty > 0) {
        built.straight.encode(commandEncoder, {parameters: undefined});
        straightDirty--;
        statisticsReader?.markStale();
      }
      catchmentReader.flush(commandEncoder);
      ringReader.flush(commandEncoder);
      statisticsReader?.flush(commandEncoder);
    },

    getLayers() {
      if (!built) return [];
      const options = ctx.options;
      const colors = getRoadColors(ctx.theme());
      const coordinateOrigin: [number, number, number] = [network.origin[0], network.origin[1], 0];
      const layers: Layer[] = [];
      if (options.showStraightLine) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'isochrones-voronoi',
            coordinateOrigin,
            gridSize: [STRAIGHT_LINE_WIDTH, straightHeight],
            bounds: [straightBounds[0], straightBounds[1], straightBounds[2], straightBounds[3]],
            values: allocationBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: CATEGORY_PALETTE.map(color => [color[0], color[1], color[2], 105] as const),
            noDataColor: [0, 0, 0, 0],
            opacity: 1
          })
        );
      }
      if (options.showStreets) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'isochrones-roads',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 1,
            color: colors.minor
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'isochrones-major-roads',
            coordinateOrigin,
            segments: majorSegmentsBuffer,
            instanceCount: network.majorSegments.length / 4,
            widthPixels: 1.5,
            color: colors.major
          })
        );
      }
      if (options.showBands) {
        layers.push(
          new IsobandTriangleLayer({
            id: 'isochrones-bands',
            coordinateOrigin,
            gridSize: [1, 1],
            bounds: [0, 0, 1, 1],
            triangles,
            triangleBands,
            values: paletteBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            extent: isochroneParameters.buffer,
            drawCommands,
            drawCommandIndex: 0
          })
        );
      }
      if (options.showServiceAreas) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'isochrones-service-areas',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 2.4,
            values: nodeFacilityBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentSourcesBuffer,
            colormap: 'category',
            palette: CATEGORY_PALETTE,
            noDataValue: GPU_NETWORK_REACHABILITY_NONE,
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      if (options.showRings) {
        layers.push(
          new PolylineLayer({
            id: `isochrones-rings-halo-${options.cellChoice}`,
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: ringPositions,
            polylineOffsets: ringOffsets,
            valueIndices: ringShells,
            extent: ringCount,
            instanceCount: RING_VERTEX_CAPACITY,
            widthPixels: 6,
            color: colors.halo
          }),
          new PolylineLayer({
            id: `isochrones-rings-${options.cellChoice}`,
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: ringPositions,
            polylineOffsets: ringOffsets,
            valueIndices: ringShells,
            extent: ringCount,
            instanceCount: RING_VERTEX_CAPACITY,
            widthPixels: 2.5,
            color: [255, 80, 60, 255]
          })
        );
      }
      if (options.showDemand) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'isochrones-demand-halo',
            coordinateOrigin,
            positions: demandPositionsBuffer,
            instanceCount: demand.count,
            radiusPixels: 3.4,
            color: colors.halo
          }),
          new SpatialAnalysisPointLayer({
            id: 'isochrones-demand',
            coordinateOrigin,
            positions: demandPositionsBuffer,
            instanceCount: demand.count,
            radiusPixels: 2.2,
            values: demandBandsBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: getBandPalette().slice(0, 8),
            noDataValue: NO_BAND,
            noDataColor: ctx.theme() === 'dark' ? [150, 155, 170, 130] : [110, 115, 130, 150]
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'isochrones-facility-halo',
          coordinateOrigin,
          positions: facilityPositionsBuffer,
          instanceCount: built.facilityCount,
          radiusPixels: built.facilityCount > 200 ? 4 : 7,
          color: colors.halo
        }),
        new SpatialAnalysisPointLayer({
          id: 'isochrones-facilities',
          coordinateOrigin,
          positions: facilityPositionsBuffer,
          instanceCount: built.facilityCount,
          radiusPixels: built.facilityCount > 200 ? 2.6 : 5,
          values: facilityIdsBuffer,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: CATEGORY_PALETTE
        })
      );
      void statisticsBuffers;
      void seedCountParameter;
      return layers;
    },

    destroy() {
      destroyed = true;
      catchmentReader.stop();
      ringReader.stop();
      statisticsReader?.stop();
      resources.destroy();
    }
  };
}
