// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Catchments and trade areas. New York points of interest are the facilities (one category at a
 * time) and taxi activity per square cell is the demand. Four cross-join `GPUNeighborSearch`
 * runs write the demand-to-facility and facility-to-demand weights twice: a catchment weighting
 * (Gaussian or binary within the bandwidth) for two-step and three-step floating catchment
 * accessibility, and an inverse-distance power weighting for Huff trade areas.
 *
 * `GPUCatchmentAccessibility` (2SFCA and 3SFCA are both compiled; the select only chooses which
 * output is drawn) gives supply per demand. `GPUHuffTradeAreas` gives each demand cell its modal
 * facility and each facility its expected demand. Bandwidth, decay power, attractiveness exponent,
 * decay shape and facility category are buffer writes: one compiled graph, re-encoded on change.
 *
 * The POI file has no size attribute, so facility supply and attractiveness are one seeded
 * pseudo-random capacity per facility (shown as such in the readouts).
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {
  getGPUNeighborSearchParameterValues,
  GPUCatchmentAccessibility,
  GPUHuffTradeAreas,
  GPUNeighborSearch,
  GPU_HUFF_NO_TRADE_AREA,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor
} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {createSeededRandom} from '../spatial-analysis-data';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';
import {SizedPointLayer} from './trade-areas-layers';

const DEMAND_CELL_METERS = 250;
/** Demand cells with fewer trip vertices are dropped. */
const MINIMUM_DEMAND_VERTICES = 4;
/** Facility slots per demand cell in each cross-join CSR; denser catchments overflow (reported). */
const SLOTS_PER_DEMAND = 160;
const CATEGORY_CHOICES = 6;
const ALL_CATEGORIES = 'all';

type DemandColor = '2sfca' | '3sfca' | 'trade-area' | 'probability';
type Decay = 'gaussian' | 'binary';

const TRADE_AREA_PALETTE: readonly SpatialAnalysisColor[] = [
  [78, 201, 255, 235],
  [255, 148, 72, 235],
  [189, 122, 255, 235],
  [87, 235, 168, 235],
  [255, 105, 168, 235],
  [245, 220, 87, 235],
  [107, 158, 255, 235],
  [255, 92, 92, 235]
];

/** Returns the value below which `fraction` of the finite positive entries fall. */
function getPositiveQuantile(values: Float32Array, fraction: number): number {
  const positives = Array.from(values).filter(value => Number.isFinite(value) && value > 0);
  if (positives.length === 0) return 1;
  positives.sort((a, b) => a - b);
  return positives[Math.min(positives.length - 1, Math.floor(positives.length * fraction))];
}

function formatDecimal(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '–';
}

export const tradeAreasMode: SpatialAnalysisModeDefinition = {
  id: 'trade-areas',
  title: 'Trade areas',
  contributors: ['GPUCatchmentAccessibility', 'GPUHuffTradeAreas', 'GPUNeighborSearch'],
  description:
    'Who is well served? Taxi activity cells (dots) take supply from nearby points of interest ' +
    'through 2SFCA or 3SFCA catchments, or choose a facility by Huff probability. Switch the ' +
    'colour, tune bandwidth, decay and attractiveness; larger facilities capture more expected demand.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const [trips, pois] = await Promise.all([
      context.data.getNewYorkTrips(),
      context.data.getNewYorkPointsOfInterest()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'trade-areas');

    // Window: the central 99% of the vertices per axis (a few trips are far outside the city).
    const vertexCount = trips.vertexTimestamps.length;
    const getRange = (axis: 0 | 1): [number, number] => {
      const values = new Float32Array(vertexCount);
      for (let index = 0; index < vertexCount; index++) {
        values[index] = trips.vertexPositions[index * 2 + axis];
      }
      values.sort();
      return [values[Math.floor(vertexCount * 0.005)], values[Math.floor(vertexCount * 0.995)]];
    };
    const [minimumX, maximumX] = getRange(0);
    const [minimumY, maximumY] = getRange(1);

    // Demand: trip vertices per square cell inside the window.
    const demandCells = new Map<number, {x: number; y: number; count: number}>();
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const x = trips.vertexPositions[vertex * 2];
      const y = trips.vertexPositions[vertex * 2 + 1];
      if (x < minimumX || x > maximumX || y < minimumY || y > maximumY) continue;
      const column = Math.floor((x - minimumX) / DEMAND_CELL_METERS);
      const row = Math.floor((y - minimumY) / DEMAND_CELL_METERS);
      const key = row * 4096 + column;
      let cell = demandCells.get(key);
      if (!cell) {
        cell = {
          x: minimumX + (column + 0.5) * DEMAND_CELL_METERS,
          y: minimumY + (row + 0.5) * DEMAND_CELL_METERS,
          count: 0
        };
        demandCells.set(key, cell);
      }
      cell.count++;
    }
    const demandRows = [...demandCells.values()].filter(
      cell => cell.count >= MINIMUM_DEMAND_VERTICES
    );
    const demandCount = demandRows.length;
    const demandPositions = new Float32Array(demandCount * 2);
    const demandValues = new Float32Array(demandCount);
    let totalDemand = 0;
    demandRows.forEach((cell, index) => {
      demandPositions[index * 2] = cell.x;
      demandPositions[index * 2 + 1] = cell.y;
      demandValues[index] = cell.count;
      totalDemand += cell.count;
    });

    // Facilities: every point of interest, with a seeded pseudo-random capacity.
    const facilityCount = pois.positions.length / 2;
    const random = createSeededRandom(2026);
    const capacities = new Float32Array(facilityCount);
    for (let index = 0; index < facilityCount; index++) {
      // Roughly log-normal, 0.3 to 4.
      const gaussian = random() + random() + random() + random() - 2;
      capacities[index] = Math.min(4, Math.max(0.3, Math.exp(gaussian * 0.9)));
    }
    const categoryTotals = new Map<number, number>();
    for (const category of pois.categories) {
      categoryTotals.set(category, (categoryTotals.get(category) ?? 0) + 1);
    }
    const topCategories = [...categoryTotals.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, CATEGORY_CHOICES);

    const slotCapacity = demandCount * SLOTS_PER_DEMAND;
    const searchBounds: [number, number, number, number] = [
      minimumX - 100,
      minimumY - 100,
      maximumX + 100,
      maximumY + 100
    ];

    let bandwidth = 600;
    let decay: Decay = 'gaussian';
    let huffExponent = 1;
    let huffDecayPower = 2;
    let demandColor: DemandColor = '3sfca';
    let category: string = String(topCategories[0][0]);
    let showSpokes = true;
    let dirty = true;
    let facilitySupply = 0;
    let accessibilityRange = 1;
    let expectedRange = 1;

    // Inputs.
    const facilityPositionsBuffer = resources.createBuffer('facility-positions', pois.positions);
    const demandPositionsBuffer = resources.createBuffer('demand-positions', demandPositions);
    const capacityBuffer = resources.createBuffer('capacity', capacities);
    const demandBuffer = resources.createBuffer('demand', demandValues);
    const facilityMask = new Uint32Array(facilityCount);
    const facilityMaskBuffer = resources.createBuffer('facility-mask', facilityMask);
    // Weights: A rows are demand cells (listing facilities), B rows are facilities (listing demand).
    type Csr = {
      offsets: ReturnType<SpatialAnalysisResources['createBuffer']>;
      neighbors: ReturnType<SpatialAnalysisResources['createBuffer']>;
      weights: ReturnType<SpatialAnalysisResources['createBuffer']>;
      overflow: ReturnType<SpatialAnalysisResources['createBuffer']>;
    };
    const createCsr = (name: string, rowCount: number): Csr => ({
      offsets: resources.createBuffer(`${name}-offsets`, (rowCount + 1) * 4),
      neighbors: resources.createBuffer(`${name}-neighbors`, slotCapacity * 4),
      weights: resources.createBuffer(`${name}-weights`, slotCapacity * 4),
      overflow: resources.createBuffer(`${name}-overflow`, 4)
    });
    const catchmentDemandRows = createCsr('catchment-demand', demandCount);
    const catchmentFacilityRows = createCsr('catchment-facility', facilityCount);
    const huffDemandRows = createCsr('huff-demand', demandCount);
    const huffFacilityRows = createCsr('huff-facility', facilityCount);
    // Outputs.
    const accessibility2 = resources.createBuffer('accessibility-2sfca', demandCount * 4);
    const accessibility3 = resources.createBuffer('accessibility-3sfca', demandCount * 4);
    const ratios2 = resources.createBuffer('ratios-2sfca', facilityCount * 4);
    const ratios3 = resources.createBuffer('ratios-3sfca', facilityCount * 4);
    const reachable = resources.createBuffer('reachable', demandCount * 4);
    const tradeArea = resources.createBuffer('trade-area', demandCount * 4);
    const tradeAreaProbability = resources.createBuffer('trade-area-probability', demandCount * 4);
    const expectedDemand = resources.createBuffer('expected-demand', facilityCount * 4);
    const spokeSegments = resources.createBuffer('spoke-segments', demandCount * 16);
    const spokeFade = resources.createBuffer('spoke-fade', demandCount * 4);

    // Parameters.
    const catchmentParameters = resources.createParameterBuffer(
      'catchment-parameters',
      'float32',
      GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
    );
    const huffParameters = resources.createParameterBuffer(
      'huff-parameters',
      'float32',
      GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
    );
    const huffExponentParameter = resources.createParameterBuffer('huff-exponent', 'float32', 1);
    const spokeParameters = resources.createParameterBuffer('spoke-parameters', 'float32', 1);

    function writeParameters(): void {
      catchmentParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds: searchBounds,
          radius: bandwidth,
          weightKind: decay === 'gaussian' ? 'kernel' : 'binary',
          kernel: 'gaussian'
        })
      );
      huffParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds: searchBounds,
          radius: bandwidth,
          weightKind: 'inverseDistance',
          power: huffDecayPower,
          distanceFloor: 100
        })
      );
      huffExponentParameter.write(Float32Array.of(huffExponent));
      spokeParameters.write(Float32Array.of(showSpokes ? 1 : 0));
    }

    function writeFacilityMask(): void {
      facilitySupply = 0;
      for (let index = 0; index < facilityCount; index++) {
        const selected = category === ALL_CATEGORIES || pois.categories[index] === Number(category);
        facilityMask[index] = selected ? 1 : 0;
        if (selected) facilitySupply += capacities[index];
      }
      facilityMaskBuffer.write(facilityMask);
    }
    writeParameters();
    writeFacilityMask();

    // One graph: four searches, both catchment methods, Huff, and the trade-area spokes.
    const graph = new GPUCommandGraph<void>(device, {id: 'trade-areas'});
    const view = <Format extends GPUVectorFormat>(
      name: string,
      buffer: ReturnType<SpatialAnalysisResources['createBuffer']>,
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
    const capacityView = view('capacity', capacityBuffer, 'float32', facilityCount);
    const demandView = view('demand', demandBuffer, 'float32', demandCount);
    const weightsOf = (name: string, csr: Csr, rowCount: number) => ({
      offsets: view(`${name}-offsets`, csr.offsets, 'uint32', rowCount + 1),
      neighbors: view(`${name}-neighbors`, csr.neighbors, 'uint32', slotCapacity),
      weights: view(`${name}-weights`, csr.weights, 'float32', slotCapacity)
    });
    const catchmentDemandWeights = weightsOf('catchment-demand', catchmentDemandRows, demandCount);
    const catchmentFacilityWeights = weightsOf(
      'catchment-facility',
      catchmentFacilityRows,
      facilityCount
    );
    const huffDemandWeights = weightsOf('huff-demand', huffDemandRows, demandCount);
    const huffFacilityWeights = weightsOf('huff-facility', huffFacilityRows, facilityCount);
    const catchmentParametersView = catchmentParameters.importToGraph(graph);
    const huffParametersView = huffParameters.importToGraph(graph);
    const addSearch = (
      name: string,
      toFacilities: boolean,
      parameters: ReturnType<typeof catchmentParameters.importToGraph>,
      weights: ReturnType<typeof weightsOf>,
      csr: Csr
    ) => {
      graph.add(
        new GPUNeighborSearch({
          id: name,
          mode: 'radius',
          // Demand rows list the facilities within reach; facility rows list the demand.
          positions: toFacilities ? facilityPositionsView : demandPositionsView,
          queryPositions: toFacilities ? demandPositionsView : facilityPositionsView,
          mask: toFacilities ? facilityMaskView : undefined,
          queryMask: toFacilities ? undefined : facilityMaskView,
          parameters,
          gridSize: [64, 64],
          weights,
          overflow: view(`${name}-overflow`, csr.overflow, 'uint32', 1)
        })
      );
    };
    addSearch(
      'catchment-demand-search',
      true,
      catchmentParametersView,
      catchmentDemandWeights,
      catchmentDemandRows
    );
    addSearch(
      'catchment-facility-search',
      false,
      catchmentParametersView,
      catchmentFacilityWeights,
      catchmentFacilityRows
    );
    addSearch('huff-demand-search', true, huffParametersView, huffDemandWeights, huffDemandRows);
    addSearch(
      'huff-facility-search',
      false,
      huffParametersView,
      huffFacilityWeights,
      huffFacilityRows
    );
    const reachableView = view('reachable', reachable, 'uint32', demandCount);
    for (const [method, accessibilityBuffer, ratiosBuffer] of [
      ['2sfca', accessibility2, ratios2],
      ['3sfca', accessibility3, ratios3]
    ] as const) {
      graph.add(
        new GPUCatchmentAccessibility({
          id: `catchment-${method}`,
          method,
          supply: capacityView,
          demand: demandView,
          facilityWeights: catchmentFacilityWeights,
          demandWeights: catchmentDemandWeights,
          accessibility: view(
            `accessibility-${method}`,
            accessibilityBuffer,
            'float32',
            demandCount
          ),
          ratios: view(`ratios-${method}`, ratiosBuffer, 'float32', facilityCount),
          // Both methods report the same count, so only the first writes it.
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
        attractiveness: capacityView,
        demandWeights: huffDemandWeights,
        facilityWeights: huffFacilityWeights,
        demand: demandView,
        parameters: huffExponentParameter.importToGraph(graph),
        tradeArea: tradeAreaView,
        tradeAreaProbability: probabilityView,
        expectedDemand: view('expected-demand', expectedDemand, 'float32', facilityCount)
      })
    );
    // Spokes from each demand cell to its modal facility, faded by the Huff probability.
    addKernelPass(graph, {
      id: 'trade-area-spokes',
      invocationCount: demandCount,
      bindings: [
        {name: 'demandPositions', view: demandPositionsView, type: 'f32', access: 'read'},
        {name: 'facilityPositions', view: facilityPositionsView, type: 'f32', access: 'read'},
        {name: 'tradeArea', view: tradeAreaView, type: 'u32', access: 'read'},
        {name: 'probability', view: probabilityView, type: 'f32', access: 'read'},
        {
          name: 'parameters',
          view: spokeParameters.importToGraph(graph),
          type: 'f32',
          access: 'read'
        },
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

    const accessibilityRanges = new Map<string, number>();
    // Summary readback: statistics sources, then both accessibilities and the expected demand.
    const reader = new SummaryReader(
      resources,
      'trade-areas',
      [
        {buffer: catchmentDemandRows.overflow, size: 4},
        {buffer: catchmentFacilityRows.overflow, size: 4},
        {buffer: huffDemandRows.overflow, size: 4},
        {buffer: huffFacilityRows.overflow, size: 4},
        {buffer: accessibility2, size: demandCount * 4},
        {buffer: accessibility3, size: demandCount * 4},
        {buffer: expectedDemand, size: facilityCount * 4},
        {buffer: reachable, size: demandCount * 4}
      ],
      bytes => {
        const overflows = new Uint32Array(bytes.slice(0, 16));
        let offset = 16;
        const take = (length: number) => {
          const slice = bytes.slice(offset, offset + length * 4);
          offset += length * 4;
          return slice;
        };
        const access2 = new Float32Array(take(demandCount));
        const access3 = new Float32Array(take(demandCount));
        const expected = new Float32Array(take(facilityCount));
        const reachableCounts = new Uint32Array(take(demandCount));
        let weighted2 = 0;
        let weighted3 = 0;
        let unreached = 0;
        let reachedDemand = 0;
        for (let index = 0; index < demandCount; index++) {
          weighted2 += demandValues[index] * (access2[index] || 0);
          weighted3 += demandValues[index] * (access3[index] || 0);
          if (reachableCounts[index] === 0) unreached++;
          else reachedDemand += demandValues[index];
        }
        let captured = 0;
        let busiest = 0;
        for (const value of expected) {
          captured += value;
          busiest = Math.max(busiest, value);
        }
        accessibilityRange = getPositiveQuantile(demandColor === '2sfca' ? access2 : access3, 0.95);
        expectedRange = getPositiveQuantile(expected, 0.98);
        accessibilityRanges.set('2sfca', getPositiveQuantile(access2, 0.95));
        accessibilityRanges.set('3sfca', getPositiveQuantile(access3, 0.95));
        overflowReadout.setValue(
          overflows.some(Boolean)
            ? `yes (${Array.from(overflows).join('/')}); lower the bandwidth`
            : 'no'
        );
        unreachedReadout.setValue(`${formatCount(unreached)} of ${formatCount(demandCount)} cells`);
        conservationReadout.setValue(
          `${formatDecimal(facilitySupply > 0 ? weighted2 / facilitySupply : NaN)} / ${formatDecimal(facilitySupply > 0 ? weighted3 / facilitySupply : NaN)}`
        );
        capturedReadout.setValue(
          `${formatCount(captured)} of ${formatCount(reachedDemand)} (busiest ${formatCount(busiest)})`
        );
        context.updateLayers();
      }
    );
    // Controls.
    context.controls.addSelect<string>({
      label: 'Facility category (mask)',
      options: [
        ...topCategories.map(([index, total]) => ({
          value: String(index),
          label: `${pois.categoryNames[index]} (${formatCount(total)})`
        })),
        {value: ALL_CATEGORIES, label: `All points of interest (${formatCount(facilityCount)})`}
      ],
      value: category,
      onChange: value => {
        category = value;
        writeFacilityMask();
        dirty = true;
      }
    });
    context.controls.addSelect<DemandColor>({
      label: 'Colour demand cells by',
      options: [
        {value: '3sfca', label: '3SFCA accessibility (supply per demand)'},
        {value: '2sfca', label: '2SFCA accessibility (supply per demand)'},
        {value: 'trade-area', label: 'Huff trade area (modal facility)'},
        {value: 'probability', label: 'Huff probability of the modal facility'}
      ],
      value: demandColor,
      onChange: value => {
        demandColor = value;
        accessibilityRange = accessibilityRanges.get(value) ?? accessibilityRange;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Catchment bandwidth (also the Huff cut-off)',
      min: 200,
      max: 1200,
      step: 50,
      value: bandwidth,
      format: value => `${value} m`,
      onChange: value => {
        bandwidth = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSelect<Decay>({
      label: 'Catchment weights',
      options: [
        {value: 'gaussian', label: 'Gaussian decay'},
        {value: 'binary', label: 'Binary (inside = 1)'}
      ],
      value: decay,
      onChange: value => {
        decay = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Huff attractiveness exponent',
      min: 0.5,
      max: 3,
      step: 0.1,
      value: huffExponent,
      format: value => value.toFixed(1),
      onChange: value => {
        huffExponent = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Huff distance decay (inverse-distance power)',
      min: 0.5,
      max: 4,
      step: 0.25,
      value: huffDecayPower,
      format: value => value.toFixed(2),
      onChange: value => {
        huffDecayPower = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addToggle({
      label: 'Show trade-area spokes',
      value: showSpokes,
      onChange: value => {
        showSpokes = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addLegend({
      title: 'Demand: accessibility or Huff probability, low to high',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: 'low',
        maximumLabel: 'high (95th percentile)'
      }
    });
    context.controls.addLegend({
      title: 'Facilities (disc area = expected Huff demand)',
      gradient: {
        colors: [
          [87, 16, 110],
          [120, 28, 109],
          [237, 105, 37],
          [252, 255, 164]
        ],
        minimumLabel: 'little',
        maximumLabel: 'much'
      }
    });
    context.controls.addReadout('Demand cells', formatCount(demandCount));
    context.controls.addReadout('Demand (trip vertices)', formatCount(totalDemand));
    context.controls.addReadout('Facilities', formatCount(facilityCount));
    const unreachedReadout = context.controls.addReadout('Cells reaching no facility');
    const conservationReadout = context.controls.addReadout(
      'Σ demand·access / supply (2SFCA / 3SFCA)'
    );
    const capturedReadout = context.controls.addReadout('Huff demand captured');
    const overflowReadout = context.controls.addReadout('Pair overflow');
    context.controls.addReadout('Facility size', 'seeded pseudo-random capacity');
    context.controls.addReadout('Data', `${trips.attribution}; ${pois.attribution}`);

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        if (dirty || frame.frameIndex < 2) {
          compiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
          reader.request(commandEncoder);
        }
        reader.flush(commandEncoder);
      },
      getLayers(): Layer[] {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        const layers: Layer[] = [
          new SpatialAnalysisPointLayer({
            id: 'trade-areas-all-facilities',
            coordinateOrigin,
            positions: facilityPositionsBuffer,
            instanceCount: facilityCount,
            radiusPixels: 1.3,
            color: [150, 160, 175, 70]
          })
        ];
        const common = {
          coordinateOrigin,
          positions: demandPositionsBuffer,
          instanceCount: demandCount,
          radiusPixels: 4.2
        };
        if (demandColor === 'trade-area') {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'trade-areas-demand-modal',
              ...common,
              values: tradeArea,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: TRADE_AREA_PALETTE,
              noDataValue: GPU_HUFF_NO_TRADE_AREA,
              noDataColor: [90, 95, 105, 160]
            })
          );
        } else {
          const accessibilityBuffer =
            demandColor === '2sfca'
              ? accessibility2
              : demandColor === '3sfca'
                ? accessibility3
                : null;
          layers.push(
            new SpatialAnalysisPointLayer({
              id: `trade-areas-demand-${demandColor}`,
              ...common,
              values: accessibilityBuffer ?? tradeAreaProbability,
              valueFormat: 'float32',
              colormap: 'viridis',
              valueRange: accessibilityBuffer ? [0, accessibilityRange] : [0, 1],
              noDataColor: [90, 95, 105, 160],
              color: [255, 255, 255, 235]
            })
          );
        }
        if (showSpokes) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'trade-areas-spokes',
              coordinateOrigin,
              segments: spokeSegments,
              weights: spokeFade,
              instanceCount: demandCount,
              widthPixels: 0.9,
              color: [255, 255, 255, 90]
            })
          );
        }
        layers.push(
          new SizedPointLayer({
            id: 'trade-areas-facilities',
            coordinateOrigin,
            positions: facilityPositionsBuffer,
            instanceCount: facilityCount,
            radiusPixels: 5,
            values: expectedDemand,
            valueFormat: 'float32',
            colormap: 'inferno',
            // A negative lower bound keeps low values off the black end of the ramp (dark basemap).
            valueRange: [-0.35 * expectedRange, expectedRange],
            discardAtOrBelow: 0,
            color: [255, 255, 255, 185]
          })
        );
        return layers;
      },
      destroy() {
        reader.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
