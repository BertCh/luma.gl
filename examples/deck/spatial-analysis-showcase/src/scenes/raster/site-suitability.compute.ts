// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPURasterArithmeticParameterValues,
  getGPURasterCellStatisticsParameterValues,
  getGPURasterConditionalParameterValues,
  getGPURasterReclassifyParameterValues,
  getGPURasterSamplingParameterValues,
  getGPUWeightedOverlayParameterLength,
  getGPUWeightedOverlayParameterValues,
  GPURasterArithmetic,
  GPURasterCellStatistics,
  GPURasterConditional,
  GPURasterReclassify,
  GPURasterSampling,
  GPURasterZonalStatistics,
  GPUWeightedOverlay,
  GPU_RASTER_SAMPLING_PARAMETER_LENGTH,
  type GPURasterBufferBand,
  type GPURasterZonalStatisticsSumOrder
} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUTerrainDerivativesParameterValues,
  GPUTerrainDerivatives
} from '@luma.gl/experimental/gpu-terrain';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import type {RampName} from '../../engine/ramps';
import {createSeededRandom} from '../../engine/projection';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {COVER_COLORS, COVER_NAMES, type SuitabilityLayer} from './b16-colors';
import {
  createGraphViewer,
  formatFixed,
  formatPercent,
  loadDixieRasters,
  readDixieReference,
  UtmRasterLayer
} from './b16-common';

/** Option state of the site-suitability scene. */
export type SiteSuitabilityOptions = {
  layer: SuitabilityLayer;
  ramp: RampName;
  opacity: number;
  minimumScore: number;
  weightSeverity: number;
  weightSlope: number;
  weightCover: number;
  weightProximity: number;
  weightGreenness: number;
  normalizeWeights: boolean;
  noDataPolicy: 'propagate' | 'ignore';
  severityRange: readonly [number, number];
  slopeRange: readonly [number, number];
  proximityRange: readonly [number, number];
  greennessRange: readonly [number, number];
  coverTree: number;
  coverShrub: number;
  coverGrass: number;
  coverCrop: number;
  coverBare: number;
  restrictBuiltWater: boolean;
  statisticsPolicy: 'ignore' | 'propagate';
  minimumValid: number;
  showSites: boolean;
  siteCount: number;
  siteThreshold: number;
  siteSize: number;
  sampleMethod: 'nearest' | 'bilinear' | 'bicubic';
  sampleNoData: 'strict' | 'renormalize';
  zonalOrder: GPURasterZonalStatisticsSumOrder;
  ignoreTrees: boolean;
};

const LAYER_COUNT = 5;
const LAYER_NAMES = ['burn severity', 'slope', 'land cover', 'proximity', 'greenness'] as const;
/** Break table of the land cover layer: WorldCover codes 10, 20, ... become classes 0, 1, ... */
const COVER_BREAKS = [15, 25, 35, 45, 55, 65, 75, 85];
const MAXIMUM_BREAKS = COVER_BREAKS.length;
/** Zones the zonal statistics keep. The 9th class (herbaceous wetland, one pixel) overflows on purpose. */
const ZONE_CAPACITY = 8;
const SITE_SPACING = 200;
const SETTLE_MILLISECONDS = 350;
/** Greenville grid column and row (`properties.greenvilleGridRowCol`). */
const GREENVILLE_CELL: readonly [number, number] = [587, 538];

type Criteria = {dnbr: Float32Array; slope: Float32Array; ndviAfter: Float32Array};

/**
 * Post-fire treatment suitability around Greenville. A one-off graph derives the criteria
 * (dNBR, slope, greenness) from the Sentinel-2 bands and the DEM; every control after that is a
 * parameter write into the scoring graphs: five rescaling nodes and `GPURasterCellStatistics`,
 * then `GPUWeightedOverlay`, `GPURasterSampling` at a lattice of candidate sites, and two
 * `GPURasterZonalStatistics` instances over the land cover zones.
 */
export async function createSiteSuitability(
  ctx: SceneContext<SiteSuitabilityOptions>
): Promise<SceneInstance<SiteSuitabilityOptions>> {
  const dataset = ctx.datasets.get('dixie-fire');
  const {device} = ctx;
  ctx.setStatus('Loading Sentinel-2 bands and the DEM');
  const rasters = await loadDixieRasters(dataset, ctx.signal);
  ctx.signal.throwIfAborted();
  const reference = readDixieReference(dataset);
  const {width, height, cellCount, frame} = rasters;
  const resources = new SpatialAnalysisResources(device, 'suit');

  // --- Static inputs prepared on the CPU ---------------------------------------------------------
  // Distance from Greenville in meters, and the land cover class index (0 = tree cover).
  const distance = new Float32Array(cellCount);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      distance[row * width + column] =
        frame.cellSize * Math.hypot(column - GREENVILLE_CELL[0], row - GREENVILLE_CELL[1]);
    }
  }
  const coverIndex = new Uint32Array(cellCount);
  for (let index = 0; index < cellCount; index++) {
    coverIndex[index] = Math.max(0, Math.round(rasters.worldCoverClasses[index] / 10) - 1);
  }
  // Candidate sites: a jittered lattice in local UTM meters (east, north from the grid center).
  const random = createSeededRandom(20210713);
  const sitesPerSide = Math.floor((2 * frame.halfWidth) / SITE_SPACING) - 1;
  const maximumSites = sitesPerSide * sitesPerSide;
  const sitePositions = new Float32Array(maximumSites * 2);
  const siteSamplingPositions = new Float32Array(maximumSites * 2);
  const siteDisplayPositions = new Float32Array(maximumSites * 2);
  for (let row = 0; row < sitesPerSide; row++) {
    for (let column = 0; column < sitesPerSide; column++) {
      const index = row * sitesPerSide + column;
      const x =
        -frame.halfWidth + SITE_SPACING * (column + 1) + (random() - 0.5) * SITE_SPACING * 0.8;
      const y =
        -frame.halfHeight + SITE_SPACING * (row + 1) + (random() - 0.5) * SITE_SPACING * 0.8;
      sitePositions[index * 2] = x;
      sitePositions[index * 2 + 1] = y;
      // The sampling contributor puts row 0 at the smallest y, our rasters put row 0 north.
      siteSamplingPositions[index * 2] = x;
      siteSamplingPositions[index * 2 + 1] = -y;
      const [meterX, meterY] = frame.localToMeters(x, y);
      siteDisplayPositions[index * 2] = meterX;
      siteDisplayPositions[index * 2 + 1] = meterY;
    }
  }

  // --- Buffers ---------------------------------------------------------------------------------
  const floats = (name: string, length = cellCount) => resources.createBuffer(name, length * 4);
  const input = {
    redAfter: resources.createBuffer('red-after', rasters.redAfter),
    nirBefore: resources.createBuffer('nir-before', rasters.nirBefore),
    swirBefore: resources.createBuffer('swir-before', rasters.swirBefore),
    nirAfter: resources.createBuffer('nir-after', rasters.nirAfter),
    swirAfter: resources.createBuffer('swir-after', rasters.swirAfter),
    elevation: resources.createBuffer('elevation', rasters.elevation),
    cover: resources.createBuffer('cover-code', rasters.worldCover),
    distance: resources.createBuffer('distance', distance)
  };
  const criteria = {
    nbrBefore: floats('nbr-before'),
    nbrAfter: floats('nbr-after'),
    dnbr: floats('dnbr'),
    ndviAfter: floats('ndvi-after'),
    slope: floats('slope'),
    burnedFlag: floats('burned-flag')
  };
  /** Raw criteria stack for the weighted overlay: dNBR, slope, cover code, distance, NDVI after. */
  const rawStack = floats('raw-stack', cellCount * LAYER_COUNT);
  /** The same criteria rescaled to 0 to 1, for cell statistics. */
  const normalizedStack = floats('normalized-stack', cellCount * LAYER_COUNT);
  const normalized = Array.from({length: LAYER_COUNT}, (_, index) => floats(`normalized-${index}`));
  const statistics = {
    mean: floats('stat-mean'),
    spread: floats('stat-spread'),
    minimum: floats('stat-minimum'),
    maximum: floats('stat-maximum'),
    range: floats('stat-range'),
    count: floats('stat-count')
  };
  const score = floats('score');
  const scoreRange = resources.createBuffer('score-range', 8);
  const siteCountBuffer = resources.createParameterBuffer('site-count', 'uint32', 1);
  const sitePositionBuffer = resources.createBuffer(
    'site-sampling-positions',
    siteSamplingPositions
  );
  const siteDisplayBuffer = resources.createBuffer('site-display-positions', siteDisplayPositions);
  const siteValues = floats('site-values', maximumSites);
  const zones = resources.createBuffer('zones', coverIndex);
  const zonal = {
    sorted: {
      score: makeZonalOutputs(resources, 'sorted-score'),
      burned: makeZonalOutputs(resources, 'sorted-burned')
    },
    atomic: {score: makeZonalOutputs(resources, 'atomic-score')}
  };

  // Per-frame parameter buffers.
  const normalizerParameters = Array.from({length: 4}, (_, index) =>
    resources.createParameterBuffer(`normalizer-${index}`, 'float32', 8)
  );
  const coverBreaks = resources.createParameterBuffer('cover-breaks', 'float32', MAXIMUM_BREAKS);
  const coverValues = resources.createParameterBuffer(
    'cover-values',
    'float32',
    MAXIMUM_BREAKS + 1
  );
  const coverReclassify = resources.createParameterBuffer('cover-reclassify', 'float32', 4);
  const statisticsParameters = resources.createParameterBuffer(
    'statistics-parameters',
    'float32',
    4
  );
  const overlayParameters = resources.createParameterBuffer(
    'overlay-parameters',
    'float32',
    getGPUWeightedOverlayParameterLength(LAYER_COUNT)
  );
  const overlayBreaks = resources.createParameterBuffer(
    'overlay-breaks',
    'float32',
    LAYER_COUNT * MAXIMUM_BREAKS
  );
  const overlayValues = resources.createParameterBuffer(
    'overlay-values',
    'float32',
    LAYER_COUNT * (MAXIMUM_BREAKS + 1)
  );
  const samplingParameters = resources.createParameterBuffer(
    'sampling-parameters',
    'float32',
    GPU_RASTER_SAMPLING_PARAMETER_LENGTH
  );
  const burnedParameters = resources.createParameterBuffer('burned-parameters', 'float32', 8);
  const terrainSettings = resources.createParameterBuffer('terrain-settings', 'float32', 8);

  // --- Graphs ----------------------------------------------------------------------------------
  const band = (id: string, values: GraphDataView<'float32'>): GPURasterBufferBand<'float32'> => ({
    id,
    format: 'float32',
    storage: {kind: 'buffer', values}
  });

  // 1. Criteria (runs once at creation, then lives in the stacks).
  const criteriaGraph = new GPUCommandGraph<void>(device, {id: 'suit-criteria'});
  {
    const view = createGraphViewer(criteriaGraph);
    const f = (name: string, buffer: Buffer) => view(name, buffer, 'float32', cellCount);
    const indexParameters = resources.createParameterBuffer('index-parameters', 'float32', 8);
    indexParameters.write(
      getGPURasterArithmeticParameterValues({
        operation: 'normalizedDifference',
        scaleA: 1e-4,
        scaleB: 1e-4
      })
    );
    const differenceParameters = resources.createParameterBuffer(
      'difference-parameters',
      'float32',
      8
    );
    differenceParameters.write(getGPURasterArithmeticParameterValues({operation: 'subtract'}));
    const indexView = indexParameters.importToGraph(criteriaGraph);
    for (const [id, a, b, output] of [
      ['nbr-before', input.nirBefore, input.swirBefore, criteria.nbrBefore],
      ['nbr-after', input.nirAfter, input.swirAfter, criteria.nbrAfter],
      ['ndvi-after', input.nirAfter, input.redAfter, criteria.ndviAfter]
    ] as const) {
      criteriaGraph.add(
        new GPURasterArithmetic({
          id,
          cellCount,
          a: f(`${id}-a`, a),
          b: f(`${id}-b`, b),
          noDataValue: 0,
          parameters: indexView,
          output: {values: f(`${id}-out`, output)}
        })
      );
    }
    criteriaGraph.add(
      new GPURasterArithmetic({
        id: 'dnbr',
        cellCount,
        a: f('dnbr-a', criteria.nbrBefore),
        b: f('dnbr-b', criteria.nbrAfter),
        parameters: differenceParameters.importToGraph(criteriaGraph),
        output: {values: f('dnbr-out', criteria.dnbr)}
      })
    );
    burnedParameters.write(
      getGPURasterConditionalParameterValues({
        comparison: '>',
        threshold: 0.27,
        constantA: 1,
        constantB: 0
      })
    );
    criteriaGraph.add(
      new GPURasterConditional({
        id: 'burned',
        cellCount,
        conditionValues: f('burned-condition', criteria.dnbr),
        parameters: burnedParameters.importToGraph(criteriaGraph),
        output: {values: f('burned-out', criteria.burnedFlag)}
      })
    );
    terrainSettings.write(
      getGPUTerrainDerivativesParameterValues({cellSize: [frame.cellSize, frame.cellSize]})
    );
    criteriaGraph.add(
      new GPUTerrainDerivatives({
        id: 'slope',
        width,
        height,
        elevation: band('elevation', f('elevation', input.elevation)),
        settings: terrainSettings.importToGraph(criteriaGraph),
        slope: f('slope-out', criteria.slope),
        cellSizeMode: 'uniform',
        rowDirection: 'south'
      })
    );
  }
  const compiledCriteria = resources.track(criteriaGraph.compile());

  // 2. Scoring inputs: rescale each criterion to 0..1, then summarise across the stack.
  const normalizeGraph = new GPUCommandGraph<void>(device, {id: 'suit-normalize'});
  {
    const view = createGraphViewer(normalizeGraph);
    const f = (name: string, buffer: Buffer) => view(name, buffer, 'float32', cellCount);
    const sources = [criteria.dnbr, criteria.slope, null, input.distance, criteria.ndviAfter];
    let normalizerIndex = 0;
    for (let layer = 0; layer < LAYER_COUNT; layer++) {
      const source = sources[layer];
      if (!source) continue;
      normalizeGraph.add(
        new GPURasterArithmetic({
          id: `normalize-${LAYER_NAMES[layer].replace(' ', '-')}`,
          cellCount,
          a: f(`source-${layer}`, source),
          parameters: normalizerParameters[normalizerIndex++].importToGraph(normalizeGraph),
          output: {values: f(`normalized-${layer}`, normalized[layer])}
        })
      );
    }
    // Land cover goes through a break table (class values are the user's scores, NaN = restricted).
    normalizeGraph.add(
      new GPURasterReclassify({
        id: 'cover-scores',
        values: f('cover', input.cover),
        breaks: coverBreaks.importToGraph(normalizeGraph),
        classValues: coverValues.importToGraph(normalizeGraph),
        parameters: coverReclassify.importToGraph(normalizeGraph),
        output: {reclassified: f('normalized-2', normalized[2])}
      })
    );
  }
  const compiledNormalize = resources.track(normalizeGraph.compile());
  const statisticsGraph = new GPUCommandGraph<void>(device, {id: 'suit-statistics'});
  {
    const view = createGraphViewer(statisticsGraph);
    const f = (name: string, buffer: Buffer) => view(name, buffer, 'float32', cellCount);
    statisticsGraph.add(
      new GPURasterCellStatistics({
        id: 'criteria-statistics',
        stack: view('normalized-stack', normalizedStack, 'float32', cellCount * LAYER_COUNT),
        layerCount: LAYER_COUNT,
        cellCount,
        parameters: statisticsParameters.importToGraph(statisticsGraph),
        output: {
          mean: f('mean', statistics.mean),
          standardDeviation: f('spread', statistics.spread),
          minimum: f('minimum', statistics.minimum),
          maximum: f('maximum', statistics.maximum),
          range: f('range', statistics.range),
          count: view('count', statistics.count, 'uint32', cellCount)
        }
      })
    );
  }
  const compiledStatistics = resources.track(statisticsGraph.compile());

  // 3. Weighted overlay and sampling at the candidate sites.
  const overlayGraph = new GPUCommandGraph<void>(device, {id: 'suit-overlay'});
  {
    const view = createGraphViewer(overlayGraph);
    const f = (name: string, buffer: Buffer, length = cellCount) =>
      view(name, buffer, 'float32', length);
    overlayGraph.add(
      new GPUWeightedOverlay({
        id: 'overlay',
        stack: f('raw-stack', rawStack, cellCount * LAYER_COUNT),
        layerCount: LAYER_COUNT,
        cellCount,
        parameters: overlayParameters.importToGraph(overlayGraph),
        remapBreaks: overlayBreaks.importToGraph(overlayGraph),
        remapValues: overlayValues.importToGraph(overlayGraph),
        maximumBreakCount: MAXIMUM_BREAKS,
        output: {score: f('score', score), scoreRange: f('score-range', scoreRange, 2)}
      })
    );
    overlayGraph.add(
      new GPURasterSampling({
        id: 'sites',
        width,
        height,
        values: f('sampled-score', score),
        positions: view('site-positions', sitePositionBuffer, 'float32x2', maximumSites),
        pointCount: siteCountBuffer.importToGraph(overlayGraph),
        parameters: samplingParameters.importToGraph(overlayGraph),
        output: {values: f('site-values', siteValues, maximumSites)}
      })
    );
  }
  const compiledOverlay = resources.track(overlayGraph.compile());

  // 4. Zonal statistics over the land cover zones: sorted sums (and a burned-share instance) and an
  // atomic-sum variant to compare timings.
  const buildZonalGraph = (
    id: string,
    sumOrder: GPURasterZonalStatisticsSumOrder,
    targets: readonly {name: string; values: Buffer; outputs: ZonalOutputs}[]
  ) => {
    const graph = new GPUCommandGraph<void>(device, {id});
    const view = createGraphViewer(graph);
    for (const target of targets) {
      const column = <F extends 'float32' | 'uint32'>(name: string, buffer: Buffer, format: F) =>
        view(`${target.name}-${name}`, buffer, format, ZONE_CAPACITY);
      graph.add(
        new GPURasterZonalStatistics({
          id: `zonal-${sumOrder}-${target.name}`,
          width,
          height,
          zones: view('zones', zones, 'uint32', cellCount),
          values: band(
            target.name,
            view(`${target.name}-values`, target.values, 'float32', cellCount)
          ),
          zoneCapacity: ZONE_CAPACITY,
          sumOrder,
          output: {
            cellCounts: column('cell-counts', target.outputs.cellCounts, 'uint32'),
            valueCounts: column('value-counts', target.outputs.valueCounts, 'uint32'),
            sums: column('sums', target.outputs.sums, 'float32'),
            means: column('means', target.outputs.means, 'float32'),
            minimums: column('minimums', target.outputs.minimums, 'float32'),
            maximums: column('maximums', target.outputs.maximums, 'float32')
          },
          overflow: view(`${target.name}-overflow`, target.outputs.overflow, 'uint32', 1)
        })
      );
    }
    return resources.track(graph.compile());
  };
  const compiledZonal = {
    sorted: buildZonalGraph('suit-zonal-sorted', 'sorted', [
      {name: 'score', values: score, outputs: zonal.sorted.score},
      {name: 'burned', values: criteria.burnedFlag, outputs: zonal.sorted.burned}
    ]),
    atomic: buildZonalGraph('suit-zonal-atomic', 'atomic', [
      {name: 'score', values: score, outputs: zonal.atomic.score}
    ])
  };

  // --- Create-time run: criteria, then the raw stack copy -----------------------------------------
  {
    const encoder = device.createCommandEncoder({id: 'suit-criteria-encoder'});
    compiledCriteria.encode(encoder, {parameters: undefined});
    const layerBuffers = [
      criteria.dnbr,
      criteria.slope,
      input.cover,
      input.distance,
      criteria.ndviAfter
    ];
    layerBuffers.forEach((buffer, layer) => {
      encoder.copyBufferToBuffer({
        sourceBuffer: buffer,
        destinationBuffer: rawStack,
        destinationOffset: layer * cellCount * 4,
        size: cellCount * 4
      });
    });
    device.submit(encoder.finish());
  }

  // --- State -----------------------------------------------------------------------------------
  let destroyed = false;
  let dirty = true;
  let lastChange = performance.now();
  let summaryStale = true;
  let scoreStale = true;
  let criteriaCpu: Criteria | null = null;
  let scoreCpu: Float32Array | null = null;
  let siteSamples: Float32Array | null = null;
  let measuring = false;

  const markDirty = () => {
    dirty = true;
    lastChange = performance.now();
  };

  const getCoverTable = (state: SiteSuitabilityOptions): number[] => {
    const restricted = state.restrictBuiltWater ? Number.NaN : 0;
    // Classes: tree, shrub, grass, crop, built-up, bare, snow, water, wetland and beyond.
    return [
      state.coverTree,
      state.coverShrub,
      state.coverGrass,
      state.coverCrop,
      state.restrictBuiltWater ? restricted : state.coverCrop * 0.5,
      state.coverBare,
      0,
      restricted,
      state.coverGrass
    ];
  };

  /** Linear rescale parameters of one criterion: `clamp(a * scale + offset, 0, 1)`, optionally inverted. */
  const getRescale = (range: readonly [number, number], invert: boolean) => {
    const low = Math.min(range[0], range[1]);
    const high = Math.max(range[0], range[1]);
    const span = Math.max(high - low, 1e-6);
    const scale = 1 / span;
    const offset = -low / span;
    return invert ? {scaleA: -scale, offsetA: 1 - offset} : {scaleA: scale, offsetA: offset};
  };

  const layerSettings = (state: SiteSuitabilityOptions) => {
    const severity = orderRange(state.severityRange);
    const slope = orderRange(state.slopeRange);
    const proximity = orderRange(state.proximityRange);
    const greenness = orderRange(state.greennessRange);
    return {severity, slope, proximity, greenness};
  };

  const writeParameters = () => {
    const state = ctx.options;
    const ranges = layerSettings(state);
    [
      getRescale(ranges.severity, false),
      getRescale(ranges.slope, false),
      getRescale(ranges.proximity, true),
      getRescale(ranges.greenness, true)
    ].forEach((rescale, index) => {
      normalizerParameters[index].write(
        getGPURasterArithmeticParameterValues({
          operation: 'add',
          ...rescale,
          constantB: 0,
          clampMin: 0,
          clampMax: 1
        })
      );
    });
    const table = getCoverTable(state);
    coverBreaks.write(Float32Array.from(COVER_BREAKS));
    coverValues.write(Float32Array.from(table));
    coverReclassify.write(getGPURasterReclassifyParameterValues({breakCount: MAXIMUM_BREAKS}));
    statisticsParameters.write(
      getGPURasterCellStatisticsParameterValues({
        noDataPolicy: state.statisticsPolicy,
        minimumValidCount: state.minimumValid
      })
    );
    const weights = [
      state.weightSeverity,
      state.weightSlope,
      state.weightCover,
      state.weightProximity,
      state.weightGreenness
    ];
    overlayParameters.write(
      getGPUWeightedOverlayParameterValues({
        normalizeWeights: state.normalizeWeights,
        noDataPolicy: state.noDataPolicy,
        layers: [
          {weight: weights[0], inputMin: ranges.severity[0], inputMax: ranges.severity[1]},
          {weight: weights[1], inputMin: ranges.slope[0], inputMax: ranges.slope[1]},
          {weight: weights[2], mode: 'table', breakCount: MAXIMUM_BREAKS, closed: 'left'},
          {
            weight: weights[3],
            inputMin: ranges.proximity[0],
            inputMax: ranges.proximity[1],
            invert: true
          },
          {
            weight: weights[4],
            inputMin: ranges.greenness[0],
            inputMax: ranges.greenness[1],
            invert: true
          }
        ]
      })
    );
    // Only layer 2 uses its table; the other rows are never read.
    const breakTable = new Float32Array(LAYER_COUNT * MAXIMUM_BREAKS);
    const valueTable = new Float32Array(LAYER_COUNT * (MAXIMUM_BREAKS + 1));
    breakTable.set(COVER_BREAKS, 2 * MAXIMUM_BREAKS);
    valueTable.set(table, 2 * (MAXIMUM_BREAKS + 1));
    overlayBreaks.write(breakTable);
    overlayValues.write(valueTable);
    samplingParameters.write(
      getGPURasterSamplingParameterValues({
        width,
        height,
        extent: [-frame.halfWidth, -frame.halfHeight, frame.halfWidth, frame.halfHeight],
        method: state.sampleMethod,
        noDataPolicy: state.sampleNoData
      })
    );
    siteCountBuffer.write(Uint32Array.of(Math.min(state.siteCount, maximumSites)));
  };

  let appliedIgnoreTrees: boolean | null = null;
  const writeZones = () => {
    if (appliedIgnoreTrees === ctx.options.ignoreTrees) return;
    appliedIgnoreTrees = ctx.options.ignoreTrees;
    if (!ctx.options.ignoreTrees) {
      zones.write(coverIndex);
      return;
    }
    const masked = new Uint32Array(coverIndex);
    for (let index = 0; index < masked.length; index++) {
      if (masked[index] === 0) masked[index] = 0xffffffff;
    }
    zones.write(masked);
  };

  // --- Readback --------------------------------------------------------------------------------
  const criteriaReader = new SummaryReader(
    resources,
    'suit-criteria',
    [
      {buffer: criteria.dnbr, size: cellCount * 4},
      {buffer: criteria.slope, size: cellCount * 4},
      {buffer: criteria.ndviAfter, size: cellCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      criteriaCpu = {
        dnbr: new Float32Array(bytes, 0, cellCount),
        slope: new Float32Array(bytes, cellCount * 4, cellCount),
        ndviAfter: new Float32Array(bytes, cellCount * 8, cellCount)
      };
    }
  );
  const scoreReader = new SummaryReader(
    resources,
    'suit-score',
    [{buffer: score, size: cellCount * 4}],
    bytes => {
      if (!destroyed) scoreCpu = new Float32Array(bytes);
    }
  );
  const zonalSources = (outputs: ZonalOutputs) => [
    {buffer: outputs.cellCounts, size: ZONE_CAPACITY * 4},
    {buffer: outputs.valueCounts, size: ZONE_CAPACITY * 4},
    {buffer: outputs.means, size: ZONE_CAPACITY * 4},
    {buffer: outputs.minimums, size: ZONE_CAPACITY * 4},
    {buffer: outputs.maximums, size: ZONE_CAPACITY * 4},
    {buffer: outputs.overflow, size: 4}
  ];
  const summaryReader = new SummaryReader(
    resources,
    'suit-summary',
    [
      {buffer: scoreRange, size: 8},
      {buffer: siteValues, size: maximumSites * 4},
      ...zonalSources(zonal.sorted.score),
      ...zonalSources(zonal.sorted.burned),
      ...zonalSources(zonal.atomic.score)
    ],
    bytes => {
      if (destroyed) return;
      let offset = 0;
      const take = <T extends Float32Array | Uint32Array>(
        Type: {new (buffer: ArrayBuffer, offset: number, length: number): T},
        length: number
      ): T => {
        const result = new Type(bytes, offset, length);
        offset += length * 4;
        return result;
      };
      const range = take(Float32Array, 2);
      siteSamples = take(Float32Array, maximumSites);
      const readZonal = () => ({
        cellCounts: take(Uint32Array, ZONE_CAPACITY),
        valueCounts: take(Uint32Array, ZONE_CAPACITY),
        means: take(Float32Array, ZONE_CAPACITY),
        minimums: take(Float32Array, ZONE_CAPACITY),
        maximums: take(Float32Array, ZONE_CAPACITY),
        overflow: take(Uint32Array, 1)[0]
      });
      const sortedZones = readZonal();
      const burnedZones = readZonal();
      const atomicZones = readZonal();
      publishSummary([range[0], range[1]], sortedZones, burnedZones, atomicZones);
    }
  );

  interface ZonalReadout {
    cellCounts: Uint32Array;
    valueCounts: Uint32Array;
    means: Float32Array;
    minimums: Float32Array;
    maximums: Float32Array;
    overflow: number;
  }

  const ZONE_READOUTS: readonly {id: string; zone: number; reference?: string}[] = [
    {id: 'zoneTree', zone: 0, reference: 'Tree cover'},
    {id: 'zoneGrass', zone: 2, reference: 'Grassland'},
    {id: 'zoneCrop', zone: 3, reference: 'Cropland'},
    {id: 'zoneBuilt', zone: 4, reference: 'Built-up'},
    {id: 'zoneBare', zone: 5, reference: 'Bare / sparse vegetation'},
    {id: 'zoneWater', zone: 7, reference: 'Permanent water bodies'}
  ];

  function publishSummary(
    range: [number, number],
    sortedZones: ZonalReadout,
    burnedZones: ZonalReadout,
    atomicZones: ZonalReadout
  ): void {
    const atomic = ctx.options.zonalOrder === 'atomic';
    const scoreZones = atomic ? atomicZones : sortedZones;
    if (atomic) {
      let largest = 0;
      for (let zone = 0; zone < ZONE_CAPACITY; zone++) {
        const difference = Math.abs(atomicZones.means[zone] - sortedZones.means[zone]);
        if (Number.isFinite(difference)) largest = Math.max(largest, difference);
      }
      ctx.setReadout(
        'zonalSums',
        `atomic vs sorted, largest difference of a zone mean: ${largest.toExponential(1)}`
      );
    } else {
      ctx.setReadout('zonalSums', 'sorted: bitwise reproducible between runs');
    }
    ctx.setLegendExtent('score', range);
    ctx.setReadout(
      'scoreRange',
      Number.isFinite(range[0])
        ? `${formatFixed(range[0], 3)} to ${formatFixed(range[1], 3)}`
        : 'no defined cells'
    );
    const defaultBurn = true;
    for (const row of ZONE_READOUTS) {
      const cells = scoreZones.cellCounts[row.zone];
      if (cells === 0) {
        ctx.setReadout(row.id, 'no cells in this window');
        continue;
      }
      const share = formatPercent(cells / cellCount, cells / cellCount < 0.01 ? 2 : 1);
      const mean = scoreZones.means[row.zone];
      const burned = burnedZones.means[row.zone];
      const burnedReference = reference.burnedByCover[row.reference ?? ''];
      const scoreText = Number.isFinite(mean)
        ? `score ${formatFixed(mean, 2)} (${formatFixed(scoreZones.minimums[row.zone], 2)} to ${formatFixed(scoreZones.maximums[row.zone], 2)})`
        : 'score restricted';
      const burnedText = defaultBurn
        ? `burned ${formatPercent(burned, 1)}${burnedReference === undefined ? '' : ` (CPU ${formatPercent(burnedReference, 1)})`}`
        : '';
      ctx.setReadout(row.id, `${share} of area · ${scoreText} · ${burnedText}`);
    }
    ctx.setReadout(
      'zonalOverflow',
      scoreZones.overflow
        ? `yes: a zone id at or above the capacity of ${ZONE_CAPACITY} (herbaceous wetland, 1 pixel)`
        : 'no'
    );
    if (siteSamples) updateSiteReadouts();
  }

  function orderRange(range: readonly [number, number]): [number, number] {
    return [Math.min(range[0], range[1]), Math.max(range[0], range[1])];
  }

  /** CPU score of one cell with the same rules as the overlay, or NaN. */
  function computeCpuScore(cell: number, state: SiteSuitabilityOptions): number {
    if (!criteriaCpu) return Number.NaN;
    const ranges = layerSettings(state);
    const values = [
      criteriaCpu.dnbr[cell],
      criteriaCpu.slope[cell],
      rasters.worldCover[cell],
      distance[cell],
      criteriaCpu.ndviAfter[cell]
    ];
    const weights = [
      state.weightSeverity,
      state.weightSlope,
      state.weightCover,
      state.weightProximity,
      state.weightGreenness
    ];
    const table = getCoverTable(state);
    let sum = 0;
    let weightSum = 0;
    for (let layer = 0; layer < LAYER_COUNT; layer++) {
      const value = values[layer];
      let t: number;
      if (layer === 2) {
        let classIndex = 0;
        for (const breakValue of COVER_BREAKS) if (breakValue <= value) classIndex++;
        t = table[classIndex];
        if (Number.isNaN(t)) return Number.NaN;
      } else {
        if (Number.isNaN(value)) {
          if (state.noDataPolicy === 'propagate') return Number.NaN;
          continue;
        }
        const [low, high] = [
          ranges.severity,
          ranges.slope,
          null,
          ranges.proximity,
          ranges.greenness
        ][layer]!;
        t = Math.min(1, Math.max(0, (value - low) / (high - low)));
        if (layer === 3 || layer === 4) t = 1 - t;
      }
      sum += weights[layer] * t;
      weightSum += Math.abs(weights[layer]);
    }
    if (state.normalizeWeights) return weightSum > 0 ? sum / weightSum : Number.NaN;
    return sum;
  }

  function updateSiteReadouts(): void {
    if (!siteSamples) return;
    const state = ctx.options;
    const count = Math.min(state.siteCount, maximumSites);
    let defined = 0;
    let passing = 0;
    let total = 0;
    for (let index = 0; index < count; index++) {
      const value = siteSamples[index];
      if (!Number.isFinite(value)) continue;
      defined++;
      total += value;
      if (value > state.siteThreshold) passing++;
    }
    ctx.setReadout(
      'sites',
      `${formatCount(passing)} of ${formatCount(defined)} sites above ${state.siteThreshold.toFixed(2)} · mean ${formatFixed(defined ? total / defined : Number.NaN, 3)} · ${formatCount(count - defined)} without data`
    );
    if (state.sampleMethod === 'nearest' && criteriaCpu) {
      let maximumDifference = 0;
      let compared = 0;
      for (let index = 0; index < count; index++) {
        const x = sitePositions[index * 2];
        const y = sitePositions[index * 2 + 1];
        const column = Math.floor((x + frame.halfWidth) / frame.cellSize);
        const row = Math.floor((frame.halfHeight - y) / frame.cellSize);
        if (column < 0 || row < 0 || column >= width || row >= height) continue;
        const expected = computeCpuScore(row * width + column, state);
        const actual = siteSamples[index];
        if (Number.isNaN(expected) && Number.isNaN(actual)) {
          compared++;
          continue;
        }
        if (Number.isNaN(expected) !== Number.isNaN(actual)) {
          maximumDifference = Math.max(maximumDifference, 1);
          compared++;
          continue;
        }
        maximumDifference = Math.max(maximumDifference, Math.abs(expected - actual));
        compared++;
      }
      ctx.setReadout(
        'parity',
        `${formatCount(compared)} sites: GPU sample vs CPU recomputation, largest difference ${maximumDifference.toExponential(1)}`
      );
    } else {
      ctx.setReadout('parity', 'shown for the nearest method (exact cell lookup)');
    }
  }

  async function measureZonal(): Promise<void> {
    if (measuring || destroyed) return;
    measuring = true;
    try {
      const options = {parameters: undefined, completionBuffer: zonal.sorted.score.overflow};
      const sorted = await measureCompiledGraph(device, compiledZonal.sorted, options);
      const atomic = await measureCompiledGraph(device, compiledZonal.atomic, options);
      if (destroyed) return;
      ctx.setReadout(
        'zonalTiming',
        `sorted (2 columns) ${sorted.milliseconds.toFixed(2)} ms · atomic (1 column) ${atomic.milliseconds.toFixed(2)} ms · ${sorted.method === 'gpu-timestamps' ? 'GPU timestamps' : 'wall clock'}`
      );
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
    }
  }

  ctx.setReadout('sites', 'sampling...');
  ctx.setReadout('zonalTiming', 'press the button');
  ctx.setStatus('');

  const getSelectedBuffers = (layer: SuitabilityLayer) => {
    const layerIndex = ['c-severity', 'c-slope', 'c-cover', 'c-proximity', 'c-greenness'].indexOf(
      layer
    );
    return layerIndex;
  };

  // --- Instance --------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [
      compiledCriteria,
      compiledNormalize,
      compiledStatistics,
      compiledOverlay,
      compiledZonal.sorted,
      compiledZonal.atomic
    ],

    setOption() {
      markDirty();
      summaryStale = true;
      scoreStale = true;
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'measure') void measureZonal();
    },

    encode(commandEncoder, frameInfo) {
      const state = ctx.options;
      if (dirty || frameInfo.frameIndex < 2) {
        writeZones();
        writeParameters();
        compiledNormalize.encode(commandEncoder, {parameters: undefined});
        normalized.forEach((buffer, layer) => {
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: buffer,
            destinationBuffer: normalizedStack,
            destinationOffset: layer * cellCount * 4,
            size: cellCount * 4
          });
        });
        compiledStatistics.encode(commandEncoder, {parameters: undefined});
        compiledOverlay.encode(commandEncoder, {parameters: undefined});
        compiledZonal.sorted.encode(commandEncoder, {parameters: undefined});
        if (state.zonalOrder === 'atomic') {
          compiledZonal.atomic.encode(commandEncoder, {parameters: undefined});
        }
        dirty = false;
        summaryStale = true;
        scoreStale = true;
      }
      if (frameInfo.frameIndex === 3 && !criteriaCpu) criteriaReader.request(commandEncoder);
      const settled = performance.now() - lastChange > SETTLE_MILLISECONDS;
      if (settled && frameInfo.frameIndex > 4) {
        if (summaryStale && !summaryReader.isPending) {
          summaryReader.request(commandEncoder);
          summaryStale = false;
        }
        if (scoreStale && !scoreReader.isPending) {
          scoreReader.request(commandEncoder);
          scoreStale = false;
        }
      }
      criteriaReader.flush(commandEncoder);
      summaryReader.flush(commandEncoder);
      scoreReader.flush(commandEncoder);
    },

    getLayers() {
      const state = ctx.options;
      const base = {
        id: `suit-${state.layer}`,
        frame,
        opacity: state.opacity,
        noDataColor: [0, 0, 0, 0] as const
      };
      const layers: Layer[] = [];
      const criterionIndex = getSelectedBuffers(state.layer);
      if (state.layer === 'score') {
        layers.push(
          new UtmRasterLayer({
            ...base,
            values: score,
            valueFormat: 'float32',
            colormap: state.ramp,
            extent: scoreRange,
            discardAtOrBelow: state.minimumScore > 0 ? state.minimumScore : undefined
          })
        );
      } else if (state.layer === 'cover') {
        layers.push(
          new UtmRasterLayer({
            ...base,
            values: zones,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: COVER_COLORS
          })
        );
      } else if (criterionIndex >= 0) {
        layers.push(
          new UtmRasterLayer({
            ...base,
            id: `suit-criterion-${criterionIndex}`,
            values: normalized[criterionIndex],
            valueFormat: 'float32',
            colormap: state.ramp,
            valueRange: [0, 1]
          })
        );
      } else {
        const source = {
          meanCriteria: statistics.mean,
          spread: statistics.spread,
          weakest: statistics.minimum,
          strongest: statistics.maximum,
          range: statistics.range,
          validLayers: statistics.count
        }[state.layer as 'meanCriteria'];
        layers.push(
          new UtmRasterLayer({
            ...base,
            values: source,
            valueFormat: state.layer === 'validLayers' ? 'uint32' : 'float32',
            colormap: state.ramp,
            valueRange:
              state.layer === 'validLayers'
                ? [0, LAYER_COUNT]
                : [0, state.layer === 'spread' ? 0.5 : 1]
          })
        );
      }
      if (state.showSites) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'suit-sites',
            coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
            positions: siteDisplayBuffer,
            instanceCount: Math.min(state.siteCount, maximumSites),
            values: siteValues,
            valueFormat: 'float32',
            colormap: 'viridis',
            valueRange: [0, 1],
            discardAtOrBelow: state.siteThreshold,
            radiusPixels: state.siteSize,
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [column, row] = frame.lngLatToCell(event.coordinate[0], event.coordinate[1]);
      const c = Math.floor(column);
      const r = Math.floor(row);
      if (c < 0 || r < 0 || c >= width || r >= height) return null;
      const cell = r * width + c;
      const state = ctx.options;
      const lines: string[] = [];
      const scoreValue = scoreCpu ? scoreCpu[cell] : Number.NaN;
      lines.push(
        Number.isFinite(scoreValue)
          ? `Suitability ${formatFixed(scoreValue, 3)}`
          : 'Suitability: restricted or no data'
      );
      if (criteriaCpu) {
        lines.push(
          `dNBR ${formatFixed(criteriaCpu.dnbr[cell], 2)} · slope ${Math.round(criteriaCpu.slope[cell])}° · NDVI after ${formatFixed(criteriaCpu.ndviAfter[cell], 2)}`
        );
      }
      const coverName = COVER_NAMES[coverIndex[cell]] ?? 'Other';
      lines.push(
        `${coverName} · ${(distance[cell] / 1000).toFixed(1)} km from Greenville · ${Math.round(rasters.elevation[cell]).toLocaleString('en-US')} m`
      );
      if (state.layer.startsWith('c-') && criteriaCpu) lines.push('Rescaled criterion layer shown');
      return lines.join('\n');
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    destroy() {
      destroyed = true;
      criteriaReader.stop();
      scoreReader.stop();
      summaryReader.stop();
      resources.destroy();
    }
  };
}

type ZonalOutputs = {
  cellCounts: Buffer;
  valueCounts: Buffer;
  sums: Buffer;
  means: Buffer;
  minimums: Buffer;
  maximums: Buffer;
  overflow: Buffer;
};

function makeZonalOutputs(resources: SpatialAnalysisResources, name: string): ZonalOutputs {
  const make = (suffix: string, length = ZONE_CAPACITY) =>
    resources.createBuffer(`${name}-${suffix}`, length * 4);
  return {
    cellCounts: make('cell-counts'),
    valueCounts: make('value-counts'),
    sums: make('sums'),
    means: make('means'),
    minimums: make('minimums'),
    maximums: make('maximums'),
    overflow: make('overflow', 1)
  };
}
