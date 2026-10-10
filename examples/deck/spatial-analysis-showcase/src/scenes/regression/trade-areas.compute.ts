// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {COORDINATE_SYSTEM} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUNeighborSearchParameterValues,
  GPUCatchmentAccessibility,
  GPUHuffTradeAreas,
  GPUNeighborSearch,
  GPU_HUFF_NO_TRADE_AREA,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  type GPUNeighborSearchKernel,
  type GPUNeighborSearchWeightKind
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {REGION_PALETTE} from './b6-colors';
import {createNamedReader} from './b6-reader';
import {PolygonFillLayer} from './b6-fill-layer';
import {SizedPointLayer} from './b6-sized-point-layer';
import {
  buildOutlineSegments,
  computeCentroids,
  computeQueenAdjacency,
  createFeatureLocator,
  createLocalProjector,
  formatInteger,
  formatNumber,
  getQuantile,
  NO_ROW,
  projectRows,
  readPolygonGeometry,
  triangulatePolygonGeometry
} from './b6-polygons';

/** Option state of the trade-areas scene. */
export type TradeAreasOptions = {
  facility: 'grocery' | 'clinic' | 'hospital' | 'library' | 'school';
  demand: 'population' | 'noVehicle' | 'seniors' | 'children' | 'uninsured';
  supply: 'uniform' | 'confidence' | 'agglomeration';
  decay: 'binary' | 'gaussian' | 'triangular' | 'bisquare' | 'inverse';
  bandwidth: number;
  power: number;
  alpha: number;
  map: 'access3sfca' | 'access2sfca' | 'huffProbability' | 'tradeArea' | 'reachable';
  facilityView: 'expectedDemand' | 'ratio';
  showSpokes: boolean;
  showFacilities: boolean;
};

type FacilityKind = TradeAreasOptions['facility'];

/** Facility kinds in category-index order. */
const FACILITY_KINDS: readonly FacilityKind[] = [
  'grocery',
  'clinic',
  'hospital',
  'library',
  'school'
];
const SLOTS_PER_DEMAND = 1200;
const SLOTS_PER_FACILITY = 96;

/** Slider ranges of the scene. */
export const TRADE_AREA_LIMITS = {bandwidthMin: 400, bandwidthMax: 3200};

/**
 * Trade areas and floor-catchment accessibility for Chicago. Demand is a count per census tract
 * (centroid); facilities are Overture grocery stores and clinics or the city's hospitals,
 * libraries and CPS schools. Two `GPUNeighborSearch` cross joins write the demand-to-facility and
 * facility-to-demand weights (decay kernel and bandwidth are parameters). `GPUCatchmentAccessibility`
 * (2SFCA and 3SFCA both compiled) gives supply per demand; `GPUHuffTradeAreas` gives each tract
 * its modal facility and each facility its expected demand. Bandwidth, decay, exponent, demand
 * measure, supply measure and facility type are buffer writes: one compiled graph, re-encoded on
 * change.
 */
export async function createTradeAreas(
  ctx: SceneContext<TradeAreasOptions>
): Promise<SceneInstance<TradeAreasOptions>> {
  const tracts = ctx.datasets.get('chicago-tracts');
  const places = ctx.datasets.get('chicago-places');
  const facilities = ctx.datasets.get('chicago-facilities');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'trade-areas');
  const origin = tracts.defaultOrigin;
  const project = createLocalProjector(origin);

  // ---- Demand: one row per tract (centroid), several measures. ----
  const geometry = readPolygonGeometry(tracts);
  const demandCount = geometry.featureCount;
  const lngLat = computeCentroids(geometry);
  const demandPositions = projectRows(lngLat, project);
  const column = (name: string) => tracts.column<Float32Array>(name);
  const population = column('population');
  const measures: Record<TradeAreasOptions['demand'], Float32Array> = {
    population: Float32Array.from(population, value => (Number.isFinite(value) ? value : 0)),
    noVehicle: Float32Array.from(column('noVehicle'), value =>
      Number.isFinite(value) ? value : 0
    ),
    seniors: Float32Array.from(column('age65'), value => (Number.isFinite(value) ? value : 0)),
    children: Float32Array.from(column('age17'), value => (Number.isFinite(value) ? value : 0)),
    uninsured: Float32Array.from(column('uninsured'), value => (Number.isFinite(value) ? value : 0))
  };
  const poverty = column('poverty150Pct');
  const geoids = (tracts.manifest as unknown as {geoid: string[]}).geoid;

  // ---- Facilities: grocery and clinics from Overture places, the rest from city facilities. ----
  const placeCategories = places.categories('category');
  const placeCategoryColumn = places.column<Uint8Array>('category');
  const placeConfidence = places.column<Uint8Array>('confidence');
  const placeMeters = places.projectColumn('position', origin);
  const facilityCategoryColumn = facilities.column<Uint8Array>('category');
  const facilityCategories = facilities.categories('category');
  const facilityMeters = facilities.projectColumn('position', origin);
  const groceryIndex = placeCategories.indexOf('grocery');
  const healthIndex = placeCategories.indexOf('health');
  const hospitalIndex = facilityCategories.indexOf('hospital');
  const libraryIndex = facilityCategories.indexOf('library');
  const schoolIndex = facilityCategories.indexOf('cps_school');
  const kindPositions: number[] = [];
  const kinds: number[] = [];
  const confidences: number[] = [];
  for (let index = 0; index < placeCategoryColumn.length; index++) {
    const category = placeCategoryColumn[index];
    const kind = category === groceryIndex ? 0 : category === healthIndex ? 1 : -1;
    if (kind < 0) continue;
    kindPositions.push(placeMeters[index * 2], placeMeters[index * 2 + 1]);
    kinds.push(kind);
    confidences.push(placeConfidence[index] / 100);
  }
  for (let index = 0; index < facilityCategoryColumn.length; index++) {
    const category = facilityCategoryColumn[index];
    const kind =
      category === hospitalIndex
        ? 2
        : category === libraryIndex
          ? 3
          : category === schoolIndex
            ? 4
            : -1;
    if (kind < 0) continue;
    kindPositions.push(facilityMeters[index * 2], facilityMeters[index * 2 + 1]);
    kinds.push(kind);
    confidences.push(1);
  }
  const facilityCount = kinds.length;
  const facilityPositions = Float32Array.from(kindPositions);

  // Agglomeration proxy: listings of the same kind within 150 m (malls, campuses, medical blocks).
  const agglomeration = new Float32Array(facilityCount);
  {
    const cell = 150;
    const grid = new Map<string, number[]>();
    const key = (x: number, y: number, kind: number) =>
      `${Math.floor(x / cell)},${Math.floor(y / cell)},${kind}`;
    for (let index = 0; index < facilityCount; index++) {
      const k = key(facilityPositions[index * 2], facilityPositions[index * 2 + 1], kinds[index]);
      const list = grid.get(k);
      if (list) list.push(index);
      else grid.set(k, [index]);
    }
    for (let index = 0; index < facilityCount; index++) {
      const x = facilityPositions[index * 2];
      const y = facilityPositions[index * 2 + 1];
      let count = 0;
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const list = grid.get(key(x + dx * cell, y + dy * cell, kinds[index]));
          if (!list) continue;
          for (const other of list) {
            if (
              Math.hypot(facilityPositions[other * 2] - x, facilityPositions[other * 2 + 1] - y) <=
              cell
            )
              count++;
          }
        }
      }
      agglomeration[index] = count;
    }
  }
  const getSupply = (measure: TradeAreasOptions['supply']) =>
    measure === 'uniform'
      ? new Float32Array(facilityCount).fill(1)
      : measure === 'confidence'
        ? Float32Array.from(confidences)
        : agglomeration;
  const kindTotals = FACILITY_KINDS.map((_, kind) => kinds.filter(value => value === kind).length);

  // ---- Geometry and layer buffers. ----
  const triangles = triangulatePolygonGeometry(geometry);
  const trianglesBuffer = resources.createBuffer('triangles', triangles.corners);
  const ownersBuffer = resources.createBuffer('owners', triangles.owners);
  const outlineSegments = buildOutlineSegments(geometry);
  const outlineBuffer = resources.createBuffer('outline', outlineSegments);
  const locate = createFeatureLocator(geometry);
  const adjacency = computeQueenAdjacency(geometry);

  // ---- Inputs, weights and outputs. ----
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (const positions of [demandPositions, facilityPositions]) {
    for (let row = 0; row < positions.length / 2; row++) {
      minimumX = Math.min(minimumX, positions[row * 2]);
      maximumX = Math.max(maximumX, positions[row * 2]);
      minimumY = Math.min(minimumY, positions[row * 2 + 1]);
      maximumY = Math.max(maximumY, positions[row * 2 + 1]);
    }
  }
  const searchBounds: [number, number, number, number] = [
    minimumX - 200,
    minimumY - 200,
    maximumX + 200,
    maximumY + 200
  ];
  const demandSlots = demandCount * SLOTS_PER_DEMAND;
  const facilitySlots = facilityCount * SLOTS_PER_FACILITY;
  const facilityPositionsBuffer = resources.createBuffer('facility-positions', facilityPositions);
  const demandPositionsBuffer = resources.createBuffer('demand-positions', demandPositions);
  const supplyBuffer = resources.createBuffer('supply', getSupply(ctx.options.supply));
  const demandBuffer = resources.createBuffer('demand', measures[ctx.options.demand]);
  const facilityMask = new Uint32Array(facilityCount);
  const facilityMaskBuffer = resources.createBuffer('facility-mask', facilityMask);
  type Csr = {
    offsets: Buffer;
    neighbors: Buffer;
    weights: Buffer;
    overflow: Buffer;
    slots: number;
    rows: number;
  };
  const createCsr = (name: string, rows: number, slots: number): Csr => ({
    rows,
    slots,
    offsets: resources.createBuffer(`${name}-offsets`, (rows + 1) * 4),
    neighbors: resources.createBuffer(`${name}-neighbors`, slots * 4),
    weights: resources.createBuffer(`${name}-weights`, slots * 4),
    overflow: resources.createBuffer(`${name}-overflow`, 4)
  });
  const demandRows = createCsr('demand-rows', demandCount, demandSlots);
  const facilityRows = createCsr('facility-rows', facilityCount, facilitySlots);
  const access2 = resources.createBuffer('access-2sfca', demandCount * 4);
  const access3 = resources.createBuffer('access-3sfca', demandCount * 4);
  const ratios2 = resources.createBuffer('ratios-2sfca', facilityCount * 4);
  const ratios3 = resources.createBuffer('ratios-3sfca', facilityCount * 4);
  const reachable = resources.createBuffer('reachable', demandCount * 4);
  const tradeArea = resources.createBuffer('trade-area', demandCount * 4);
  const tradeAreaProbability = resources.createBuffer('trade-area-probability', demandCount * 4);
  const expectedDemand = resources.createBuffer('expected-demand', facilityCount * 4);
  const spokeSegments = resources.createBuffer('spoke-segments', demandCount * 16);
  const spokeFade = resources.createBuffer('spoke-fade', demandCount * 4);
  const tradeAreaColors = resources.createBuffer('trade-area-colors', demandCount * 4);
  const searchParameters = resources.createParameterBuffer(
    'search-parameters',
    'float32',
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
  );
  const huffExponent = resources.createParameterBuffer('huff-exponent', 'float32', 1);
  const spokeParameters = resources.createParameterBuffer('spoke-parameters', 'float32', 1);

  let selectedSupplyTotal = 0;
  const writeFacilityMask = () => {
    const kind = FACILITY_KINDS.indexOf(ctx.options.facility);
    const supply = getSupply(ctx.options.supply);
    selectedSupplyTotal = 0;
    for (let index = 0; index < facilityCount; index++) {
      const selected = kinds[index] === kind;
      facilityMask[index] = selected ? 1 : 0;
      if (selected) selectedSupplyTotal += supply[index];
    }
    facilityMaskBuffer.write(facilityMask);
    supplyBuffer.write(supply);
  };
  const writeParameters = () => {
    const {decay, bandwidth, power, alpha, showSpokes} = ctx.options;
    const weightKind: GPUNeighborSearchWeightKind =
      decay === 'binary' ? 'binary' : decay === 'inverse' ? 'inverseDistance' : 'kernel';
    const kernel: GPUNeighborSearchKernel =
      decay === 'triangular' ? 'triangular' : decay === 'bisquare' ? 'bisquare' : 'gaussian';
    searchParameters.write(
      getGPUNeighborSearchParameterValues({
        bounds: searchBounds,
        radius: bandwidth,
        weightKind,
        kernel,
        power,
        distanceFloor: 100
      })
    );
    huffExponent.write(Float32Array.of(alpha));
    spokeParameters.write(Float32Array.of(showSpokes ? 1 : 0));
  };
  writeFacilityMask();
  writeParameters();

  // ---- One graph: both searches, both catchment methods, Huff and the spokes. ----
  const graph = new GPUCommandGraph<void>(device, {id: 'trade-areas'});
  const view = <Format extends GPUVectorFormat>(
    name: string,
    buffer: Buffer,
    format: Format,
    length?: number
  ) => importGraphBuffer(graph, name, buffer, format, length);
  const facilityPositionsView = view(
    'facility-positions',
    facilityPositionsBuffer,
    'float32x2',
    facilityCount
  );
  const demandPositionsView = view(
    'demand-positions',
    demandPositionsBuffer,
    'float32x2',
    demandCount
  );
  const facilityMaskView = view('facility-mask', facilityMaskBuffer, 'uint32', facilityCount);
  const supplyView = view('supply', supplyBuffer, 'float32', facilityCount);
  const demandView = view('demand', demandBuffer, 'float32', demandCount);
  const weightsOf = (name: string, csr: Csr) => ({
    offsets: view(`${name}-offsets`, csr.offsets, 'uint32', csr.rows + 1),
    neighbors: view(`${name}-neighbors`, csr.neighbors, 'uint32', csr.slots),
    weights: view(`${name}-weights`, csr.weights, 'float32', csr.slots)
  });
  const demandWeights = weightsOf('demand-rows', demandRows);
  const facilityWeights = weightsOf('facility-rows', facilityRows);
  const searchParametersView = searchParameters.importToGraph(graph);
  // Demand rows list the facilities within reach; facility rows list the demand.
  graph.add(
    new GPUNeighborSearch({
      id: 'demand-to-facility',
      mode: 'radius',
      positions: facilityPositionsView,
      queryPositions: demandPositionsView,
      mask: facilityMaskView,
      parameters: searchParametersView,
      gridSize: [64, 64],
      weights: demandWeights,
      overflow: view('demand-overflow', demandRows.overflow, 'uint32', 1)
    })
  );
  graph.add(
    new GPUNeighborSearch({
      id: 'facility-to-demand',
      mode: 'radius',
      positions: demandPositionsView,
      queryPositions: facilityPositionsView,
      queryMask: facilityMaskView,
      parameters: searchParametersView,
      gridSize: [64, 64],
      weights: facilityWeights,
      overflow: view('facility-overflow', facilityRows.overflow, 'uint32', 1)
    })
  );
  const reachableView = view('reachable', reachable, 'uint32', demandCount);
  for (const [method, accessBuffer, ratiosBuffer] of [
    ['2sfca', access2, ratios2],
    ['3sfca', access3, ratios3]
  ] as const) {
    graph.add(
      new GPUCatchmentAccessibility({
        id: `catchment-${method}`,
        method,
        supply: supplyView,
        demand: demandView,
        facilityWeights,
        demandWeights,
        accessibility: view(`access-${method}`, accessBuffer, 'float32', demandCount),
        ratios: view(`ratios-${method}`, ratiosBuffer, 'float32', facilityCount),
        reachableFacilities: method === '2sfca' ? reachableView : undefined
      })
    );
  }
  const tradeAreaView = view('trade-area', tradeArea, 'uint32', demandCount);
  const probabilityView = view(
    'trade-area-probability',
    tradeAreaProbability,
    'float32',
    demandCount
  );
  graph.add(
    new GPUHuffTradeAreas({
      id: 'huff',
      attractiveness: supplyView,
      demandWeights,
      facilityWeights,
      demand: demandView,
      parameters: huffExponent.importToGraph(graph),
      tradeArea: tradeAreaView,
      tradeAreaProbability: probabilityView,
      expectedDemand: view('expected-demand', expectedDemand, 'float32', facilityCount)
    })
  );
  // Spokes from each tract to its modal facility, faded by the Huff probability.
  addKernelPass(graph, {
    id: 'trade-area-spokes',
    invocationCount: demandCount,
    bindings: [
      {name: 'demandPositions', view: demandPositionsView, type: 'f32', access: 'read'},
      {name: 'facilityPositions', view: facilityPositionsView, type: 'f32', access: 'read'},
      {name: 'tradeArea', view: tradeAreaView, type: 'u32', access: 'read'},
      {name: 'probability', view: probabilityView, type: 'f32', access: 'read'},
      {name: 'parameters', view: spokeParameters.importToGraph(graph), type: 'f32', access: 'read'},
      {
        name: 'segments',
        view: view('spoke-segments', spokeSegments, 'float32', demandCount * 4),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'fade',
        view: view('spoke-fade', spokeFade, 'float32', demandCount),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: /* wgsl */ `
  let facility = tradeArea[tradeAreaOffset + index];
  let x = demandPositions[demandPositionsOffset + index * 2u];
  let y = demandPositions[demandPositionsOffset + index * 2u + 1u];
  segments[segmentsOffset + index * 4u] = x;
  segments[segmentsOffset + index * 4u + 1u] = y;
  if (facility == ${GPU_HUFF_NO_TRADE_AREA}u) {
    segments[segmentsOffset + index * 4u + 2u] = x;
    segments[segmentsOffset + index * 4u + 3u] = y;
    fade[fadeOffset + index] = 0.0;
    return;
  }
  segments[segmentsOffset + index * 4u + 2u] = facilityPositions[facilityPositionsOffset + facility * 2u];
  segments[segmentsOffset + index * 4u + 3u] = facilityPositions[facilityPositionsOffset + facility * 2u + 1u];
  fade[fadeOffset + index] = probability[probabilityOffset + index] * parameters[parametersOffset];`
  });
  const compiled = resources.track(graph.compile());

  // Spokes and facilities are drawn in meters around the local origin; fill and outline use degrees.
  // ---- State and readback. ----
  let dirty = true;
  let accessRange: Record<'access2sfca' | 'access3sfca', number> = {access2sfca: 1, access3sfca: 1};
  let expectedRange = 1;
  let ratioRange = 1;
  let latest: {
    access2: Float32Array;
    access3: Float32Array;
    reachable: Uint32Array;
    tradeArea: Uint32Array;
    probability: Float32Array;
  } | null = null;

  const reader = createNamedReader(
    resources,
    'trade-areas',
    [
      {name: 'overflows', buffer: demandRows.overflow, bytes: 4},
      {name: 'facilityOverflow', buffer: facilityRows.overflow, bytes: 4},
      {name: 'access2', buffer: access2, bytes: demandCount * 4},
      {name: 'access3', buffer: access3, bytes: demandCount * 4},
      {name: 'reachable', buffer: reachable, bytes: demandCount * 4},
      {name: 'tradeArea', buffer: tradeArea, bytes: demandCount * 4},
      {name: 'probability', buffer: tradeAreaProbability, bytes: demandCount * 4},
      {name: 'expected', buffer: expectedDemand, bytes: facilityCount * 4},
      {name: 'ratios', buffer: ratios2, bytes: facilityCount * 4}
    ],
    get => {
      const demandValues = measures[ctx.options.demand];
      const a2 = get('access2').f32;
      const a3 = get('access3').f32;
      const counts = get('reachable').u32;
      const modal = get('tradeArea').u32;
      const probability = get('probability').f32;
      const expected = get('expected').f32;
      const ratios = get('ratios').f32;
      latest = {access2: a2, access3: a3, reachable: counts, tradeArea: modal, probability};
      let total = 0;
      let weighted2 = 0;
      let weighted3 = 0;
      let unreached = 0;
      let unreachedDemand = 0;
      let reachedDemand = 0;
      const group = {highDemand: 0, high2: 0, high3: 0, lowDemand: 0, low2: 0, low3: 0};
      for (let row = 0; row < demandCount; row++) {
        const weight = demandValues[row];
        total += weight;
        weighted2 += weight * (a2[row] || 0);
        weighted3 += weight * (a3[row] || 0);
        if (counts[row] === 0) {
          unreached++;
          unreachedDemand += weight;
        } else reachedDemand += weight;
        if (poverty[row] >= 30) {
          group.highDemand += weight;
          group.high2 += weight * (a2[row] || 0);
          group.high3 += weight * (a3[row] || 0);
        } else if (poverty[row] < 10) {
          group.lowDemand += weight;
          group.low2 += weight * (a2[row] || 0);
          group.low3 += weight * (a3[row] || 0);
        }
      }
      let captured = 0;
      let busiest = 0;
      for (const value of expected) {
        if (Number.isFinite(value)) {
          captured += value;
          busiest = Math.max(busiest, value);
        }
      }
      const positiveExpected = Array.from(expected).filter(value => value > 0);
      expectedRange = Math.max(1e-6, getQuantile(positiveExpected, 0.98));
      const positiveRatios = Array.from(ratios).filter(
        value => value > 0 && Number.isFinite(value)
      );
      ratioRange = Math.max(1e-9, getQuantile(positiveRatios, 0.98));
      accessRange = {
        access2sfca: Math.max(
          1e-9,
          getQuantile(
            Array.from(a2).filter(value => value > 0),
            0.95
          )
        ),
        access3sfca: Math.max(
          1e-9,
          getQuantile(
            Array.from(a3).filter(value => value > 0),
            0.95
          )
        )
      };
      ctx.setLegendExtent('access', [
        0,
        accessRange[ctx.options.map === 'access2sfca' ? 'access2sfca' : 'access3sfca']
      ]);
      ctx.setLegendExtent('expected', [0, expectedRange]);
      const per1000 = (value: number) => formatNumber(1000 * value, 2);
      ctx.setReadout('demand', `${formatInteger(total)} in ${formatInteger(demandCount)} tracts`);
      ctx.setReadout(
        'facilities',
        `${formatInteger(kindTotals[FACILITY_KINDS.indexOf(ctx.options.facility)])} sites (supply total ${formatNumber(selectedSupplyTotal, 0)})`
      );
      ctx.setReadout(
        'meanAccess',
        total > 0
          ? `2SFCA ${per1000(weighted2 / total)}, 3SFCA ${per1000(weighted3 / total)} per 1,000`
          : 'n/a'
      );
      ctx.setReadout(
        'unreached',
        `${formatNumber(total > 0 ? (100 * unreachedDemand) / total : 0, 1)}% of demand (${formatInteger(unreached)} tracts) reaches no facility`
      );
      ctx.setReadout(
        'conservation',
        selectedSupplyTotal > 0
          ? `2SFCA ${formatNumber(weighted2 / selectedSupplyTotal, 3)}, 3SFCA ${formatNumber(weighted3 / selectedSupplyTotal, 3)} (1 = all supply allocated)`
          : 'n/a'
      );
      ctx.setReadout(
        'equity',
        group.highDemand > 0 && group.lowDemand > 0
          ? `3SFCA: poverty ≥ 30% ${per1000(group.high3 / group.highDemand)} vs < 10% ${per1000(group.low3 / group.lowDemand)} per 1,000`
          : 'n/a'
      );
      ctx.setReadout(
        'captured',
        `${formatInteger(captured)} of ${formatInteger(reachedDemand)} (busiest site ${formatInteger(busiest)})`
      );
      const overflow = get('overflows').u32[0] || get('facilityOverflow').u32[0];
      ctx.setReadout(
        'overflow',
        overflow ? 'overflow: reduce the bandwidth or pick a rarer facility' : 'ok'
      );
      tradeAreaColors.write(assignTradeColors(modal, adjacency));
      ctx.requestLayers();
    }
  );

  const dark = () => ctx.theme() === 'dark';

  return {
    getCompiledGraphs: () => [compiled],

    setOption(id) {
      if (id === 'map' || id === 'facilityView' || id === 'showFacilities') {
        ctx.setLegendExtent('access', [
          0,
          accessRange[ctx.options.map === 'access2sfca' ? 'access2sfca' : 'access3sfca']
        ]);
        ctx.requestLayers();
        return;
      }
      if (id === 'facility' || id === 'supply') writeFacilityMask();
      if (id === 'demand') demandBuffer.write(measures[ctx.options.demand]);
      writeParameters();
      dirty = true;
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 2) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.request(commandEncoder);
      } else {
        reader.flush(commandEncoder);
      }
    },

    getLayers() {
      const {map, facilityView, showSpokes, showFacilities} = ctx.options;
      const isDark = dark();
      const layers: Layer[] = [];
      const common = {
        id: `trade-areas-fill-${map}`,
        triangles: trianglesBuffer,
        owners: ownersBuffer,
        triangleCount: triangles.triangleCount,
        opacity: 0.86,
        noDataColor: [110, 114, 124, 150] as const,
        color: [255, 255, 255, 255] as const
      };
      if (map === 'tradeArea') {
        layers.push(
          new PolygonFillLayer({
            ...common,
            values: tradeAreaColors,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: REGION_PALETTE
          })
        );
      } else if (map === 'reachable') {
        layers.push(
          new PolygonFillLayer({
            ...common,
            values: reachable,
            valueFormat: 'uint32',
            colormap: 'cividis',
            valueRange: [0, 12]
          })
        );
      } else {
        layers.push(
          new PolygonFillLayer({
            ...common,
            values:
              map === 'access2sfca'
                ? access2
                : map === 'access3sfca'
                  ? access3
                  : tradeAreaProbability,
            valueFormat: 'float32',
            colormap: 'ylgnbu',
            valueRange: map === 'huffProbability' ? [0, 1] : [0, accessRange[map]]
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'trade-areas-outline',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          segments: outlineBuffer,
          instanceCount: outlineSegments.length / 4,
          widthPixels: 0.6,
          color: isDark ? [235, 238, 248, 55] : [30, 34, 46, 70]
        })
      );
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      if (showSpokes) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'trade-areas-spokes',
            coordinateOrigin,
            segments: spokeSegments,
            weights: spokeFade,
            instanceCount: demandCount,
            widthPixels: 1.1,
            color: isDark ? [255, 255, 255, 120] : [20, 24, 36, 120]
          })
        );
      }
      if (showFacilities) {
        layers.push(
          new SizedPointLayer({
            id: `trade-areas-facilities-${facilityView}`,
            coordinateOrigin,
            positions: facilityPositionsBuffer,
            instanceCount: facilityCount,
            radiusPixels:
              ctx.options.facility === 'clinic' || ctx.options.facility === 'grocery' ? 6 : 8,
            values: facilityView === 'ratio' ? ratios2 : expectedDemand,
            valueFormat: 'float32',
            colormap: 'inferno',
            valueRange:
              facilityView === 'ratio'
                ? [-0.35 * ratioRange, ratioRange]
                : [-0.35 * expectedRange, expectedRange],
            discardAtOrBelow: 0,
            color: [255, 255, 255, 220]
          })
        );
      }
      return layers;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate || !latest) return null;
      const row = locate(event.coordinate[0], event.coordinate[1]);
      if (row < 0) return null;
      const per1000 = (value: number) =>
        Number.isFinite(value) ? (1000 * value).toFixed(2) : 'n/a';
      return [
        `Tract ${geoids[row]}`,
        `${ctx.options.demand}: ${formatInteger(measures[ctx.options.demand][row])}, poverty ${Number.isFinite(poverty[row]) ? poverty[row].toFixed(0) : 'n/a'}%`,
        `Access per 1,000: 2SFCA ${per1000(latest.access2[row])}, 3SFCA ${per1000(latest.access3[row])}`,
        `Facilities in reach: ${latest.reachable[row]}`,
        latest.tradeArea[row] === GPU_HUFF_NO_TRADE_AREA
          ? 'No Huff trade area'
          : `Huff: site ${latest.tradeArea[row]}, probability ${(100 * latest.probability[row]).toFixed(0)}%`
      ].join('\n');
    },

    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}

/** Greedy coloring of trade areas (modal facilities) so touching areas get different colors. */
function assignTradeColors(modal: Uint32Array, adjacency: readonly number[][]): Uint32Array {
  const none = GPU_HUFF_NO_TRADE_AREA;
  const sizes = new Map<number, number>();
  for (const label of modal) if (label !== none) sizes.set(label, (sizes.get(label) ?? 0) + 1);
  const neighborLabels = new Map<number, Set<number>>();
  for (let row = 0; row < modal.length; row++) {
    if (modal[row] === none) continue;
    for (const neighbor of adjacency[row]) {
      if (modal[neighbor] !== none && modal[neighbor] !== modal[row]) {
        const set = neighborLabels.get(modal[row]) ?? new Set<number>();
        set.add(modal[neighbor]);
        neighborLabels.set(modal[row], set);
      }
    }
  }
  const order = [...sizes.keys()].sort((a, b) => sizes.get(b)! - sizes.get(a)! || a - b);
  const colors = new Map<number, number>();
  for (const label of order) {
    const used = new Array<number>(REGION_PALETTE.length).fill(0);
    for (const other of neighborLabels.get(label) ?? []) {
      const color = colors.get(other);
      if (color !== undefined) used[color]++;
    }
    let best = 0;
    for (let color = 1; color < used.length; color++) if (used[color] < used[best]) best = color;
    colors.set(label, best);
  }
  return Uint32Array.from(modal, label => (label === none ? NO_ROW : colors.get(label)!));
}
