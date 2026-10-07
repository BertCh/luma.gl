// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  addRateClusterMapRecipe,
  getGPUPermutationParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPU_EMPIRICAL_BAYES_SUMMARY,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPUGlobalSpatialStatistics,
  GPUSpatialEmpiricalBayesRates,
  type GPUContiguityCriterion,
  type GPUPermutationAlternative
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getClassTableLayerProps} from '../../cartography/class-table';
import {US} from '../../cartography/gazetteer';
import {NO_DATA_COLOR} from '../../cartography/hue-registry';
import {formatCount, formatOrdinal, formatPercent, liveText} from '../../cartography/live-text';
import {
  createFeatureLocator,
  getGeometryPolygons,
  getInputPolygons
} from '../../cartography/picking';
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import {getLocalProjector, projectRingsToSegments} from '../../cartography/segments';
import type {ClassTable, LngLat, MapAnnotation} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor,
  type SpatialAnalysisStyleProps
} from '../../engine/layers';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import type {ChartColor} from '../chart-types';
import {
  getGiTable,
  getHairlineColor,
  getMoranPalette,
  getMoranQuadrantColors,
  getStateLineStyle,
  MORAN_LEGEND_TO_CATEGORY
} from '../weights/hot-spots.style';
import {createByteReader} from './b5-common';
import {getPolygonLayout} from './b5-geometry';
import {packColor} from './b5-palettes';
import {
  buildFunnelChart,
  getClassCounts,
  getClassIndexAmong,
  getEmpiricalBayesWeights,
  getLocalPooling,
  getPercentile,
  getSortedQuantile,
  getStateTallies,
  sortFinite,
  type LocalPooling
} from './rate-smoothing.stats';
import {
  EXTREME_TAIL,
  FADE_ALPHA_OUTPUT,
  FEW_BIRTHS,
  getCasingColor,
  getFunnelCurveColor,
  getGhostColor,
  getInkColor,
  getRateBreaks,
  getRateTable,
  getZTable,
  NO_FADE_ALPHA_OUTPUT,
  RATE_BASIS,
  RATE_SCALE,
  SMALL_COUNTY_RESIDENTS,
  Z_BREAKS
} from './rate-smoothing.style';

/** Option state of the rate-smoothing scene. */
export type RateSmoothingOptions = {
  map: 'raw' | 'smoothed' | 'spatial' | 'standardized' | 'cluster';
  /** Swipe compare of two rate maps on the shared classes (steps set it; `off` otherwise). */
  swipe: 'off' | 'raw-eb' | 'eb-spatial';
  fadeUnreliable: boolean;
  showExtremes: boolean;
  analyze: 'standardized' | 'smoothed';
  criterion: GPUContiguityCriterion;
  gating: 'permutation' | 'analytic';
  alternative: Exclude<GPUPermutationAlternative, 'less'>;
  falseDiscoveryRate: boolean;
  permutations: number;
  seed: number;
  significance: number;
};

/** Data shared with `legends(state, data)` through `ctx.setLegendData`. */
export type RateSmoothingLegendData = {
  rateTable: ClassTable;
  zTable: ClassTable;
  /** Counties per class of the raw, empirical-Bayes and spatial maps. */
  rateCounts: {raw: number[]; smoothed: number[]; spatial: number[]};
  zCounts: number[];
  /** Counties per quadrant code: not significant, HH, LH, LL, HL. */
  clusterCounts: number[];
  groundIsDark: boolean;
  pooledPerThousand: number;
};

const MAXIMUM_PERMUTATIONS = 999;
const MAXIMUM_LOCAL_NEIGHBORS = 32;
const QUADRANT_NAMES = [
  'Not significant',
  'High-high cluster',
  'Low-high outlier',
  'Low-low cluster',
  'High-low outlier'
] as const;

type Variant = {compiled: CompiledGPUCommandGraph<void>; key: string};

/** Everything read back from the GPU after the graph ran. */
type Latest = {
  raw: Float32Array;
  smoothed: Float32Array;
  spatial: Float32Array;
  standardized: Float32Array;
  quadrants: Uint32Array;
  pseudoP: Float32Array;
  pooledRate: number;
  priorVariance: number;
  moran: {raw: number; smoothed: number; spatial: number; standardized: number};
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weightsOverflow: boolean;
  weight: Float32Array;
  local: LocalPooling;
  sorted: {raw: Float64Array; smoothed: Float64Array; spatial: Float64Array; z: Float64Array};
};

/**
 * Empirical-Bayes smoothing of county birth rates. One graph chains the contributors:
 * `addRateClusterMapRecipe` (`GPUEmpiricalBayesRates`, `GPUContiguityWeights`, row-standardised
 * weights, `GPULocalMoran` and `GPULocalPermutationTest`), then `GPUSpatialEmpiricalBayesRates` on
 * the recipe's own weights and four `GPUGlobalSpatialStatistics` nodes (Moran's I of each rate
 * variable). The compile-time options (weights criterion, analysed rate, gating, tail, FDR) select
 * among graphs compiled on demand; every slider is a parameter-buffer write.
 *
 * The maps are drawn by `SpatialAnalysisPolygonLayer` straight from the contributor buffers: one
 * septile class table shared by the raw, smoothed and spatial rates, value-by-alpha from the
 * empirical-Bayes weight (computed on the CPU, the library does not expose it) and a swipe compare.
 */
export async function createRateSmoothing(
  ctx: SceneContext<RateSmoothingOptions>
): Promise<SceneInstance<RateSmoothingOptions>> {
  const counties = ctx.datasets.get('us-counties');
  const births = ctx.datasets.get('us-county-births');
  const statesGeojson = ctx.datasets.get('us-states').geojson;
  const geojson = counties.geojson;
  if (!geojson) throw new Error('rate-smoothing needs the county polygons');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'rate-smoothing');
  const layout = getPolygonLayout(counties);
  const n = layout.featureCount;

  const events = births.column<Float32Array>('births');
  const womenYears = births.column<Float32Array>('womenYears');
  const residents = births.column<Float32Array>('population');
  const features = geojson.features;
  const countyName = (row: number): string => {
    const properties = features[row]?.properties;
    return properties ? `${properties.name}, ${properties.state}` : `County ${row}`;
  };
  const stateCodes: string[] = features.map(feature => String(feature.properties?.state ?? ''));

  // CPU-side facts that need no readback: the raw rate per 1,000, its extremes and the medians.
  const rawPerThousand = Float32Array.from(events, (count, row) =>
    womenYears[row] > 0 ? (count / womenYears[row]) * RATE_SCALE : Number.NaN
  );
  const sortedRawPerThousand = sortFinite(rawPerThousand);
  const rawExtent: [number, number] = [
    Math.floor(sortedRawPerThousand[0]),
    Math.ceil(sortedRawPerThousand[sortedRawPerThousand.length - 1])
  ];
  const lowCutoff = getSortedQuantile(sortedRawPerThousand, EXTREME_TAIL);
  const highCutoff = getSortedQuantile(sortedRawPerThousand, 1 - EXTREME_TAIL);
  const highRows: number[] = [];
  const lowRows: number[] = [];
  for (let row = 0; row < n; row++) {
    if (rawPerThousand[row] >= highCutoff) highRows.push(row);
    else if (rawPerThousand[row] <= lowCutoff) lowRows.push(row);
  }
  const getMedianResidents = (rows: readonly number[]) =>
    getSortedQuantile(sortFinite(rows.map(row => residents[row])), 0.5);
  const medianResidents = {
    all: getSortedQuantile(sortFinite(residents), 0.5),
    high: getMedianResidents(highRows),
    low: getMedianResidents(lowRows)
  };
  const fewBirthsCount = events.reduce((total, count) => total + (count < FEW_BIRTHS ? 1 : 0), 0);
  const findCountyRow = (name: string, state: string): number =>
    features.findIndex(
      feature => feature.properties?.name === name && feature.properties?.state === state
    );

  // Geometry: planar-metre mesh for the fill, outline segments for hairlines and rings.
  const origin = counties.defaultOrigin;
  const projection = counties.getProjection(origin);
  const mesh = buildPolygonMesh(geojson, (lng, lat) => projection.project(lng, lat));
  const polygons = createPolygonMeshBuffers(resources, mesh, 'counties');
  const layerOrigin: [number, number, number] = [origin[0], origin[1], 0];
  const locator = createFeatureLocator(geojson);
  const outlineStart = new Uint32Array(n + 1);
  for (const feature of mesh.outlineFeatures) outlineStart[feature + 1]++;
  for (let row = 0; row < n; row++) outlineStart[row + 1] += outlineStart[row];
  const collectSegments = (rows: Iterable<number>): Float32Array => {
    const parts: Float32Array[] = [];
    let length = 0;
    for (const row of rows) {
      const part = mesh.outlineSegments.subarray(outlineStart[row] * 4, outlineStart[row + 1] * 4);
      parts.push(part);
      length += part.length;
    }
    const segments = new Float32Array(length);
    let offset = 0;
    for (const part of parts) {
      segments.set(part, offset);
      offset += part.length;
    }
    return segments;
  };
  const highSegments = collectSegments(highRows);
  const lowSegments = collectSegments(lowRows);
  const highBuffer = resources.createBuffer('extreme-high', highSegments);
  const lowBuffer = resources.createBuffer('extreme-low', lowSegments);
  const stateSegments = statesGeojson
    ? projectRingsToSegments(
        getInputPolygons(statesGeojson).flatMap(({polygon}) => polygon),
        getLocalProjector(origin)
      )
    : new Float32Array(0);
  const stateBuffer = resources.createBuffer('state-lines', stateSegments);
  // Selection outlines are rewritten into fixed buffers (no allocation per click).
  const selectionBuffer = resources.createBuffer('selection-self', mesh.outlineSegments.byteLength);
  const neighborhoodBuffer = resources.createBuffer(
    'selection-neighbors',
    mesh.outlineSegments.byteLength
  );
  let selectionCount = 0;
  let neighborhoodCount = 0;
  let selectedRow = -1;

  const getRings = (row: number): LngLat[][] =>
    getGeometryPolygons(features[row]?.geometry).flatMap(polygon =>
      polygon.map(ring => ring.map(point => [point[0], point[1]] as LngLat))
    );

  // GPU inputs and outputs.
  const verticesBuffer = resources.createBuffer('vertices', layout.vertices);
  const ringOffsetsBuffer = resources.createBuffer('ring-offsets', layout.ringOffsets);
  const featureRingOffsetsBuffer = resources.createBuffer(
    'feature-ring-offsets',
    layout.featureRingOffsets
  );
  const eventsBuffer = resources.createBuffer('events', events);
  const populationsBuffer = resources.createBuffer('woman-years', womenYears);
  const rawBuffer = resources.createBuffer('raw-rates', n * 4);
  const smoothedBuffer = resources.createBuffer('smoothed-rates', n * 4);
  const spatialBuffer = resources.createBuffer('spatial-rates', n * 4);
  const standardizedBuffer = resources.createBuffer('standardized-rates', n * 4);
  const summaryBuffer = resources.createBuffer(
    'eb-summary',
    GPU_EMPIRICAL_BAYES_SUMMARY.length * 4
  );
  const neighborCapacity = n * 24;
  const offsetsBuffer = resources.createBuffer('weights-offsets', (n + 1) * 4);
  const neighborsBuffer = resources.createBuffer('weights-neighbors', neighborCapacity * 4);
  const weightsBuffer = resources.createBuffer('weights-values', neighborCapacity * 4);
  const weightsOverflowBuffer = resources.createBuffer('weights-overflow', 4);
  const zScoresBuffer = resources.createBuffer('z-scores', n * 4);
  const quadrantsBuffer = resources.createBuffer('quadrants', n * 4);
  const colorsBuffer = resources.createBuffer('cluster-colors', n * 4);
  const pseudoPBuffer = resources.createBuffer('pseudo-p', n * 4);
  const significantBuffer = resources.createBuffer('significant', n * 4);
  const initialTable = getGiTable('light', 'No data');
  const initialQuadrants = getMoranQuadrantColors(initialTable);
  // The recipe also colours the quadrants; the map draws the quadrant codes through the chapter's
  // class table instead (so the colours follow the ground and the legend filters), and this
  // palette only completes the recipe.
  const paletteBuffer = resources.createBuffer(
    'quadrant-palette',
    Uint32Array.from(
      [
        initialQuadrants.notSignificant,
        initialQuadrants.highHigh,
        initialQuadrants.lowHigh,
        initialQuadrants.lowLow,
        initialQuadrants.highLow
      ],
      color => packColor(color[0], color[1], color[2], color[3])
    )
  );
  // Two floats per county: the weight on its own births under the national prior and under the
  // neighbourhood prior. They drive value-by-alpha.
  const channelsBuffer = resources.createBuffer('weight-channels', new Float32Array(n * 2).fill(1));
  const globalBuffers = {
    raw: resources.createBuffer('moran-raw', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4),
    smoothed: resources.createBuffer(
      'moran-smoothed',
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4
    ),
    spatial: resources.createBuffer(
      'moran-spatial',
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4
    ),
    standardized: resources.createBuffer(
      'moran-standardized',
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4
    )
  };
  const autocorrelationParameters = resources.createParameterBuffer(
    'autocorrelation-parameters',
    'float32',
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );
  const permutationParameters = resources.createParameterBuffer(
    'permutation-parameters',
    'uint32',
    GPU_PERMUTATION_PARAMETER_LENGTH
  );

  const compileKey = (options: RateSmoothingOptions): string =>
    [
      options.criterion,
      options.analyze,
      options.gating,
      options.gating === 'permutation' ? options.alternative : '-',
      options.gating === 'permutation' && options.falseDiscoveryRate ? 'fdr' : '-'
    ].join('|');

  const compileVariant = (options: RateSmoothingOptions): Variant => {
    const key = compileKey(options);
    const graph = new GPUCommandGraph<void>(device, {id: `rate-smoothing-${key}`});
    const float = (id: string, buffer: typeof rawBuffer, length = n) =>
      importGraphBuffer(graph, id, buffer, 'float32', length);
    const word = (id: string, buffer: typeof rawBuffer, length = n) =>
      importGraphBuffer(graph, id, buffer, 'uint32', length);
    const eventsView = float('events', eventsBuffer);
    const populations = float('woman-years', populationsBuffer);
    const raw = float('raw', rawBuffer);
    const smoothed = float('smoothed', smoothedBuffer);
    const standardized = float('standardized', standardizedBuffer);
    const recipe = addRateClusterMapRecipe(graph, {
      id: 'rates',
      events: eventsView,
      populations,
      positions: importGraphBuffer(
        graph,
        'vertices',
        verticesBuffer,
        'float32x2',
        layout.vertices.length / 2
      ),
      ringOffsets: word('ring-offsets', ringOffsetsBuffer, layout.ringOffsets.length),
      polygonOffsets: word(
        'feature-ring-offsets',
        featureRingOffsetsBuffer,
        layout.featureRingOffsets.length
      ),
      criterion: options.criterion,
      neighborCapacity,
      pairCapacity: n * 96,
      analyze: options.analyze,
      parameters: autocorrelationParameters.importToGraph(graph),
      permutation:
        options.gating === 'permutation'
          ? {
              parameters: permutationParameters.importToGraph(graph),
              maximumPermutations: MAXIMUM_PERMUTATIONS,
              alternative: options.alternative,
              maximumNeighbors: MAXIMUM_LOCAL_NEIGHBORS,
              falseDiscoveryRate: options.falseDiscoveryRate,
              pseudoPValues: float('pseudo-p', pseudoPBuffer),
              significant: word('significant', significantBuffer)
            }
          : undefined,
      palette: word('palette', paletteBuffer, GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH),
      standardizedRates: standardized,
      smoothedRates: smoothed,
      rawRates: raw,
      summary: float('summary', summaryBuffer, GPU_EMPIRICAL_BAYES_SUMMARY.length),
      weights: {
        offsets: word('weights-offsets', offsetsBuffer, n + 1),
        neighbors: word('weights-neighbors', neighborsBuffer, neighborCapacity),
        weights: float('weights-values', weightsBuffer, neighborCapacity)
      },
      weightsOverflow: word('weights-overflow', weightsOverflowBuffer, 1),
      zScores: float('z-scores', zScoresBuffer),
      quadrants: word('quadrants', quadrantsBuffer),
      colors: word('cluster-colors', colorsBuffer)
    });
    const spatial = float('spatial', spatialBuffer);
    graph.add(
      new GPUSpatialEmpiricalBayesRates({
        id: 'spatial-eb',
        events: eventsView,
        populations,
        weights: recipe.weights,
        smoothedRates: spatial
      })
    );
    for (const [name, values] of [
      ['raw', raw],
      ['smoothed', smoothed],
      ['spatial', spatial],
      ['standardized', standardized]
    ] as const) {
      graph.add(
        new GPUGlobalSpatialStatistics({
          id: `moran-${name}`,
          weights: recipe.weights,
          values,
          statistics: ['moran'],
          results: float(
            `moran-${name}`,
            globalBuffers[name],
            GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
          )
        })
      );
    }
    return {key, compiled: resources.track(graph.compile())};
  };

  const variants = new Map<string, Variant>();
  const getVariant = (options: RateSmoothingOptions): Variant => {
    const key = compileKey(options);
    let variant = variants.get(key);
    if (!variant) {
      variant = compileVariant(options);
      variants.set(key, variant);
    }
    return variant;
  };

  let active = getVariant(ctx.options);
  let dirty = true;
  let latest: Latest | null = null;
  let legendHighlight: number[] | null = null;
  let legendData: RateSmoothingLegendData | null = null;

  // -------------------------------------------------------------------------------------------
  // Tables and legend data
  // ---------------------------------------------------------------------------------------------

  let rateBreaks: number[] | null = null;
  const getTables = () => {
    if (!rateBreaks) return null;
    const ground = ctx.ground();
    return {
      rateTable: getRateTable(rateBreaks, rawExtent, ground),
      zTable: getZTable(ground)
    };
  };

  const publishLegendData = () => {
    const tables = getTables();
    if (!tables || !latest) return;
    const breaks = tables.rateTable.breaks;
    legendData = {
      rateTable: tables.rateTable,
      zTable: tables.zTable,
      rateCounts: {
        raw: getClassCounts(rawPerThousand, breaks),
        smoothed: getClassCounts(
          latest.smoothed.map(value => value * RATE_SCALE),
          breaks
        ),
        spatial: getClassCounts(
          latest.spatial.map(value => value * RATE_SCALE),
          breaks
        )
      },
      zCounts: getClassCounts(latest.standardized, Z_BREAKS),
      clusterCounts: countQuadrants(latest.quadrants),
      groundIsDark: ctx.ground() === 'dark',
      pooledPerThousand: latest.pooledRate * RATE_SCALE
    };
    ctx.setLegendData('rateSmoothing', legendData);
  };

  function countQuadrants(quadrants: ArrayLike<number>): number[] {
    const counts = [0, 0, 0, 0, 0];
    for (let row = 0; row < quadrants.length; row++) counts[Math.min(quadrants[row], 4)]++;
    return counts;
  }

  // -------------------------------------------------------------------------------------------
  // Selection
  // ---------------------------------------------------------------------------------------------

  const setSelection = (row: number) => {
    selectedRow = row;
    if (row < 0) {
      selectionCount = 0;
      neighborhoodCount = 0;
    } else {
      const own = collectSegments([row]);
      selectionBuffer.write(own);
      selectionCount = own.length / 4;
      const members: number[] = [];
      if (latest) {
        for (let slot = latest.offsets[row]; slot < latest.offsets[row + 1]; slot++) {
          members.push(latest.neighbors[slot]);
        }
      }
      const around = collectSegments(members);
      neighborhoodBuffer.write(around);
      neighborhoodCount = around.length / 4;
    }
    publishSelection();
    publishFunnel();
    ctx.requestLayers();
  };

  const publishSelection = () => {
    if (selectedRow < 0 || !latest) {
      ctx.setReadout('selected', null);
      ctx.setReadout('localPool', null);
      return;
    }
    const neighborCount = latest.offsets[selectedRow + 1] - latest.offsets[selectedRow];
    ctx.setReadout(
      'selected',
      `${countyName(selectedRow)}: ${formatCount(events[selectedRow])} births, ${(
        rawPerThousand[selectedRow]
      ).toFixed(1)} per 1,000`
    );
    ctx.setReadout(
      'localPool',
      `${(latest.local.pooledRates[selectedRow] * RATE_SCALE).toFixed(1)} per 1,000 among ${neighborCount} neighbours`
    );
  };

  // -------------------------------------------------------------------------------------------
  // Readouts, chart and notes
  // ---------------------------------------------------------------------------------------------

  const formatRateText = (value: number, digits = 0) => `${value.toFixed(digits)} per 1,000`;

  const publishStatic = () => {
    ctx.setReadout('counties', n);
    ctx.setReadout('rawRange', `${rawExtent[0]} to ${rawExtent[1]} per 1,000`);
    ctx.setReadout('extremePopulation', `${formatCount(medianResidents.high)} residents`);
    ctx.setReadout('allPopulation', `${formatCount(medianResidents.all)} residents`);
    ctx.setReadout('lowPopulation', `${formatCount(medianResidents.low)} residents`);
    ctx.setReadout('fewBirths', fewBirthsCount);
  };

  const publishReadouts = () => {
    if (!latest || !legendData) return;
    const {rateCounts, zCounts, clusterCounts} = legendData;
    publishStatic();
    ctx.setReadout('pooledRate', `${(latest.pooledRate * RATE_SCALE).toFixed(1)} per 1,000 a year`);
    ctx.setReadout('rawTopClass', rateCounts.raw[rateCounts.raw.length - 1]);
    ctx.setReadout('rawBottomClass', rateCounts.raw[0]);
    const darkestBreak = legendData.rateTable.breaks[legendData.rateTable.breaks.length - 1];
    const darkestResidents: number[] = [];
    for (let row = 0; row < n; row++) {
      if (rawPerThousand[row] >= darkestBreak) darkestResidents.push(residents[row]);
    }
    ctx.setReadout(
      'darkestPopulation',
      `${formatCount(getSortedQuantile(sortFinite(darkestResidents), 0.5))} residents`
    );
    ctx.setReadout(
      'trueSpread',
      formatRateText(Math.sqrt(Math.max(latest.priorVariance, 0)) * RATE_SCALE, 1)
    );
    ctx.setReadout('priorVariance', latest.priorVariance * RATE_SCALE * RATE_SCALE);
    const smallRaw: number[] = [];
    const smallSmoothed: number[] = [];
    for (let row = 0; row < n; row++) {
      if (residents[row] < SMALL_COUNTY_RESIDENTS) {
        smallRaw.push(rawPerThousand[row]);
        smallSmoothed.push(latest.smoothed[row] * RATE_SCALE);
      }
    }
    const sortedSmallRaw = sortFinite(smallRaw);
    const sortedSmallSmoothed = sortFinite(smallSmoothed);
    const range = (sorted: Float64Array) =>
      `${getSortedQuantile(sorted, 0.05).toFixed(0)} to ${getSortedQuantile(sorted, 0.95).toFixed(0)} per 1,000`;
    ctx.setReadout('smallRaw', range(sortedSmallRaw));
    ctx.setReadout('smallEb', range(sortedSmallSmoothed));
    ctx.setReadout(
      'smallLabel',
      `${formatCount(smallRaw.length)} counties under ${formatCount(SMALL_COUNTY_RESIDENTS)} residents`
    );
    ctx.setReadout('moranRaw', latest.moran.raw.toFixed(3));
    ctx.setReadout('moranEb', latest.moran.smoothed.toFixed(3));
    ctx.setReadout('moranSpatial', latest.moran.spatial.toFixed(3));
    ctx.setReadout('moranZ', latest.moran.standardized.toFixed(3));
    let islands = 0;
    for (let row = 0; row < n; row++)
      if (latest.offsets[row + 1] === latest.offsets[row]) islands++;
    ctx.setReadout('neighbourMean', (latest.offsets[n] / n).toFixed(1));
    ctx.setReadout(
      'neighbourDetail',
      `${islands} counties without neighbours${latest.weightsOverflow ? ' (CAPACITY OVERFLOW)' : ''}`
    );
    const above = zCounts.slice(4).reduce((total, count) => total + count, 0);
    const below = zCounts[0] + zCounts[1];
    ctx.setReadout('beyond196', above + below);
    ctx.setReadout('above196', above);
    ctx.setReadout('below196', below);
    ctx.setReadout('hhCount', clusterCounts[1]);
    ctx.setReadout('llCount', clusterCounts[3]);
    ctx.setReadout('outlierCount', clusterCounts[2] + clusterCounts[4]);
    ctx.setReadout('notSignificantShare', formatPercent(clusterCounts[0] / n));
    const {quadrants} = latest;
    const names = (quadrant: number) => {
      const tallies = getStateTallies(quadrants, stateCodes, quadrant).slice(0, 3);
      const list = tallies.map(
        tally => US.places[`state-${tally.state.toLowerCase()}`]?.name ?? tally.state
      );
      if (list.length === 0) return 'no state';
      return list.length === 1
        ? list[0]
        : `${list.slice(0, -1).join(', ')} and ${list[list.length - 1]}`;
    };
    ctx.setReadout('hhRegion', names(1));
    ctx.setReadout('llRegion', names(3));
    const weights = sortFinite(latest.weight);
    ctx.setReadout(
      'weightSpread',
      `${formatPercent(getSortedQuantile(weights, 0.05))} to ${formatPercent(getSortedQuantile(weights, 0.95))} (5th to 95th percentile)`
    );
    ctx.setReadout(
      'priorCheck',
      latest.priorVariance < 0 ? 'a < 0: weights leave 0-1 (esda does not clamp)' : 'a >= 0'
    );
    const quadrantColors = getMoranQuadrantColors(getGiTable(ctx.ground(), 'No data'));
    ctx.setChart('clusterBars', {
      kind: 'bars',
      title: 'Counties by cluster type',
      values: [clusterCounts[1], clusterCounts[3], clusterCounts[4], clusterCounts[2]],
      labels: ['High-high', 'Low-low', 'High-low', 'Low-high'],
      colors: [
        quadrantColors.highHigh,
        quadrantColors.lowLow,
        quadrantColors.highLow,
        quadrantColors.lowHigh
      ],
      horizontal: true,
      table: false,
      description:
        'Bar chart of counties by local Moran cluster type: high-high and low-low clusters, high-low and low-high outliers.'
    });
  };

  const getEffectiveMap = (): RateSmoothingOptions['map'] => {
    const {swipe, map} = ctx.options;
    return swipe === 'raw-eb' ? 'smoothed' : swipe === 'eb-spatial' ? 'spatial' : map;
  };

  const toChartColors = (table: ClassTable): ChartColor[] =>
    table.colors.map(color => [color[0], color[1], color[2], color[3] ?? 255] as ChartColor);

  const publishFunnel = () => {
    if (!latest || !legendData) return;
    const ground = ctx.ground();
    const map = getEffectiveMap();
    const isZ = map === 'standardized';
    const table = isZ ? legendData.zTable : legendData.rateTable;
    const palette: ChartColor[] = [
      ...toChartColors(table),
      getGhostColor(ground),
      getFunnelCurveColor(ground)
    ];
    const shown =
      map === 'smoothed' || map === 'cluster'
        ? latest.smoothed.map(value => value * RATE_SCALE)
        : map === 'spatial'
          ? latest.spatial.map(value => value * RATE_SCALE)
          : null;
    const colored = isZ ? latest.standardized : (shown ?? rawPerThousand);
    const colorIndex = Uint8Array.from(colored, value =>
      Math.max(0, getClassIndexAmong(value, isZ ? Z_BREAKS : table.breaks))
    );
    ctx.setChart(
      'funnel',
      buildFunnelChart({
        womenYears,
        rawRates: rawPerThousand,
        shownRates: isZ ? null : shown,
        colorIndex,
        palette,
        pooledRate: latest.pooledRate,
        priorVariance: isZ ? Math.max(latest.priorVariance, 0) : 0,
        selectedRows: selectedRow >= 0 ? [selectedRow] : [],
        yMaximum: Math.ceil(rawExtent[1] / 10) * 10,
        onSelect: row => {
          setSelection(row === selectedRow ? -1 : row);
          if (selectedRow >= 0) {
            ctx.setHighlight({kind: 'polygon', rings: getRings(selectedRow), pulse: true});
          }
        }
      })
    );
  };

  const updateNotes = () => {
    const o = ctx.options;
    const notes: MapAnnotation[] = [];
    const petroleum = findCountyRow('Petroleum', 'MT');
    const loving = findCountyRow('Loving', 'TX');
    const place = (row: number): LngLat | null => {
      const point = mesh.labelPoints[row];
      return point && Number.isFinite(point[0]) ? ([point[0], point[1]] as LngLat) : null;
    };
    if (o.map === 'raw' && o.showExtremes && o.swipe === 'off') {
      for (const row of [petroleum, loving]) {
        const coordinate = row >= 0 ? place(row) : null;
        if (!coordinate) continue;
        notes.push({
          kind: 'note',
          coordinate,
          priority: 5,
          title: liveText('{births:integer} births, {residents:integer} residents', {
            births: events[row],
            residents: residents[row]
          }),
          text: `${countyName(row)}: ${rawPerThousand[row].toFixed(0)} per 1,000`
        });
      }
    }
    if (o.swipe === 'raw-eb' && latest && petroleum >= 0) {
      const coordinate = place(petroleum);
      if (coordinate) {
        notes.push({
          kind: 'note',
          coordinate,
          priority: 5,
          title: liveText('{raw:fixed:0} raw, {smoothed:fixed:0} smoothed', {
            raw: rawPerThousand[petroleum],
            smoothed: latest.smoothed[petroleum] * RATE_SCALE
          }),
          text: `${countyName(petroleum)}, per 1,000`
        });
      }
    }
    if (o.map === 'cluster' && o.swipe === 'off' && latest) {
      for (const [quadrant, label] of [
        [1, 'high-high'],
        [3, 'low-low']
      ] as const) {
        const tally = getStateTallies(latest.quadrants, stateCodes, quadrant)[0];
        const stateName = tally ? US.places[`state-${tally.state.toLowerCase()}`] : null;
        if (!tally || !stateName) continue;
        notes.push({
          kind: 'note',
          coordinate: stateName.lngLat as LngLat,
          priority: 4,
          title: liveText('{count:integer} of {total:integer} counties', tally),
          text: `${stateName.name}: ${label}`
        });
      }
    }
    ctx.setAnnotations('rate-notes', notes.length ? notes : null);
  };

  // -------------------------------------------------------------------------------------------
  // Readback
  // ---------------------------------------------------------------------------------------------

  const publishTest = () => {
    const {gating, permutations} = ctx.options;
    ctx.setReadout(
      'testName',
      gating === 'permutation' ? `a ${permutations}-permutation test` : 'the analytic p-value'
    );
  };

  const writeParameters = () => {
    const {significance, seed, permutations} = ctx.options;
    publishTest();
    autocorrelationParameters.write(
      getGPUSpatialAutocorrelationParameterValues({significanceLevel: significance})
    );
    permutationParameters.write(
      getGPUPermutationParameterValues({seed, permutations, significanceLevel: significance})
    );
    dirty = true;
    reader.markStale();
  };

  const moranOf = (buffer: Float32Array) =>
    buffer[
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.statistic
    ];

  const reader = new SummaryReader(
    resources,
    'rate-smoothing',
    [
      {buffer: rawBuffer, size: n * 4},
      {buffer: smoothedBuffer, size: n * 4},
      {buffer: spatialBuffer, size: n * 4},
      {buffer: standardizedBuffer, size: n * 4},
      {buffer: quadrantsBuffer, size: n * 4},
      {buffer: pseudoPBuffer, size: n * 4},
      {buffer: summaryBuffer, size: GPU_EMPIRICAL_BAYES_SUMMARY.length * 4},
      {buffer: globalBuffers.raw, size: GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4},
      {buffer: globalBuffers.smoothed, size: GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4},
      {buffer: globalBuffers.spatial, size: GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4},
      {buffer: globalBuffers.standardized, size: GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length * 4},
      {buffer: offsetsBuffer, size: (n + 1) * 4},
      {buffer: neighborsBuffer, size: neighborCapacity * 4},
      {buffer: weightsOverflowBuffer, size: 4}
    ],
    bytes => {
      const read = createByteReader(bytes);
      const raw = read.floats(n).slice();
      const smoothed = read.floats(n).slice();
      const spatial = read.floats(n).slice();
      const standardized = read.floats(n).slice();
      const quadrants = read.words(n).slice();
      const pseudoP = read.floats(n).slice();
      const summary = read.floats(GPU_EMPIRICAL_BAYES_SUMMARY.length).slice();
      const globals = [0, 1, 2, 3].map(() =>
        read.floats(GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length).slice()
      );
      const offsets = read.words(n + 1).slice();
      const neighbors = read.words(neighborCapacity).slice();
      const overflow = read.words(1)[0] !== 0;
      const pooledRate = summary[GPU_EMPIRICAL_BAYES_SUMMARY.pooledRate];
      const priorVariance = summary[GPU_EMPIRICAL_BAYES_SUMMARY.priorVariance];
      const toThousand = (values: Float32Array) => sortFinite(values.map(v => v * RATE_SCALE));
      latest = {
        raw,
        smoothed,
        spatial,
        standardized,
        quadrants,
        pseudoP,
        pooledRate,
        priorVariance,
        moran: {
          raw: moranOf(globals[0]),
          smoothed: moranOf(globals[1]),
          spatial: moranOf(globals[2]),
          standardized: moranOf(globals[3])
        },
        offsets,
        neighbors,
        weightsOverflow: overflow,
        weight: getEmpiricalBayesWeights(priorVariance, pooledRate, womenYears),
        local: getLocalPooling(events, womenYears, offsets, neighbors),
        sorted: {
          raw: toThousand(raw),
          smoothed: toThousand(smoothed),
          spatial: toThousand(spatial),
          z: sortFinite(standardized)
        }
      };
      // The shared septile breaks are computed once, from the first smoothed rates, and frozen.
      if (!rateBreaks) rateBreaks = getRateBreaks(smoothed.map(value => value * RATE_SCALE));
      const channels = new Float32Array(n * 2);
      for (let row = 0; row < n; row++) {
        channels[row * 2] = latest.weight[row];
        channels[row * 2 + 1] = latest.local.weights[row];
      }
      channelsBuffer.write(channels);
      publishLegendData();
      publishReadouts();
      publishFunnel();
      updateNotes();
      if (selectedRow >= 0) setSelection(selectedRow);
      ctx.requestLayers();
    }
  );

  ctx.setFurniture({
    title: {
      sample: `${formatCount(n)} counties of the contiguous US (Alaska and Hawaii not in the data)`
    }
  });
  publishStatic();
  writeParameters();

  // -------------------------------------------------------------------------------------------
  // Layers
  // ---------------------------------------------------------------------------------------------

  const getRateStyle = (
    values: typeof rawBuffer,
    channel: 0 | 1,
    table: ClassTable,
    noData: SpatialAnalysisColor
  ): SpatialAnalysisStyleProps => ({
    values,
    valueFormat: 'float32',
    valueScale: RATE_SCALE,
    colormap: 'ylorbr',
    ...getClassTableLayerProps(table),
    highlightClasses: legendHighlight,
    instanceChannels: channelsBuffer,
    channelStride: 2,
    channels: {alpha: channel},
    alphaDomain: [0, 1],
    alphaOutput: ctx.options.fadeUnreliable ? FADE_ALPHA_OUTPUT : NO_FADE_ALPHA_OUTPUT,
    noDataColor: noData
  });

  const getFillStyles = (): {style: SpatialAnalysisStyleProps; side?: 'a' | 'b'}[] => {
    const o = ctx.options;
    const tables = getTables();
    if (!tables || !latest) return [];
    const noData = NO_DATA_COLOR[ctx.ground()];
    const rate = (values: typeof rawBuffer, channel: 0 | 1, side?: 'a' | 'b') => ({
      style: getRateStyle(values, channel, tables.rateTable, noData),
      side
    });
    if (o.swipe === 'raw-eb') return [rate(rawBuffer, 0, 'a'), rate(smoothedBuffer, 0, 'b')];
    if (o.swipe === 'eb-spatial') {
      return [rate(smoothedBuffer, 0, 'a'), rate(spatialBuffer, 1, 'b')];
    }
    if (o.map === 'raw') return [rate(rawBuffer, 0)];
    if (o.map === 'smoothed') return [rate(smoothedBuffer, 0)];
    if (o.map === 'spatial') return [rate(spatialBuffer, 1)];
    if (o.map === 'standardized') {
      return [
        {
          style: {
            values: standardizedBuffer,
            valueFormat: 'float32',
            colormap: 'rdbu',
            ...getClassTableLayerProps(tables.zTable),
            highlightClasses: legendHighlight,
            noDataColor: noData
          }
        }
      ];
    }
    const quadrantTable = getGiTable(ctx.ground(), 'No data');
    return [
      {
        style: {
          values: quadrantsBuffer,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: getMoranPalette(quadrantTable, true),
          noDataColor: noData,
          highlightClasses: legendHighlight
        }
      }
    ];
  };

  const getLayers = (): Layer[] => {
    const o = ctx.options;
    const ground = ctx.ground();
    const layers: Layer[] = [];
    for (const {style, side} of getFillStyles()) {
      layers.push(
        new SpatialAnalysisPolygonLayer({
          id: `rate-fill${side ? `-${side}` : ''}`,
          coordinateOrigin: layerOrigin,
          triangles: polygons.triangles,
          features: polygons.triangleFeatures,
          vertexCount: polygons.vertexCount,
          ...style,
          compareSide: side,
          // Opaque on the national paper sheet: the ground adds nothing under a full-coverage map.
          opacity: 1
        })
      );
    }
    // Tier 3: county hairlines, thin and in the ground colour.
    layers.push(
      new SpatialAnalysisSegmentLayer({
        id: 'rate-county-hairlines',
        coordinateOrigin: layerOrigin,
        segments: polygons.outline,
        instanceCount: polygons.outlineCount,
        widthPixels: n > 2000 ? 0.4 : 0.5,
        color: getHairlineColor(ground)
      })
    );
    // Zone boundary tier: state lines over a casing, always on in county steps.
    if (stateSegments.length) {
      const line = getStateLineStyle(ground);
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'rate-state-lines',
          coordinateOrigin: layerOrigin,
          segments: stateBuffer,
          instanceCount: stateSegments.length / 4,
          widthPixels: line.widthPixels,
          color: line.color,
          outlineColor: line.casing,
          outlineWidthPixels: (line.casingPixels - line.widthPixels) / 2
        })
      );
    }
    if (o.showExtremes) {
      // The highest 5 % of raw rates solid, the lowest 5 % dashed: ink over a ground casing.
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'rate-extreme-low',
          coordinateOrigin: layerOrigin,
          segments: lowBuffer,
          instanceCount: lowSegments.length / 4,
          widthPixels: 1.6,
          color: getInkColor(ground),
          dashArray: [4, 3]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'rate-extreme-high',
          coordinateOrigin: layerOrigin,
          segments: highBuffer,
          instanceCount: highSegments.length / 4,
          widthPixels: 1.6,
          color: getInkColor(ground),
          outlineColor: getCasingColor(ground),
          outlineWidthPixels: 0.7
        })
      );
    }
    if (selectedRow >= 0) {
      if (
        (o.map === 'spatial' || o.map === 'cluster' || o.swipe === 'eb-spatial') &&
        neighborhoodCount
      ) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'rate-neighborhood',
            coordinateOrigin: layerOrigin,
            segments: neighborhoodBuffer,
            instanceCount: neighborhoodCount,
            widthPixels: 1.2,
            color: getInkColor(ground)
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'rate-selection',
          coordinateOrigin: layerOrigin,
          segments: selectionBuffer,
          instanceCount: selectionCount,
          widthPixels: 2.5,
          color: getInkColor(ground),
          outlineColor: getCasingColor(ground),
          outlineWidthPixels: 1
        })
      );
    }
    return layers;
  };

  // -------------------------------------------------------------------------------------------
  // Tooltip
  // ---------------------------------------------------------------------------------------------

  const describeCounty = (row: number): TooltipContent | null => {
    if (!latest || !legendData) return null;
    const map = getEffectiveMap();
    const {rateTable: table, zTable} = legendData;
    const swatchOf = (value: number, source: ClassTable) => {
      const color = source.colors[Math.max(0, getClassIndexAmong(value, source.breaks))];
      return [color[0], color[1], color[2], color[3] ?? 255] as const;
    };
    const rate = (value: number): TooltipRow => ({
      label: '',
      value: value.toFixed(1),
      unit: 'per 1,000 women a year'
    });
    const rawValue = rawPerThousand[row];
    const smoothedValue = latest.smoothed[row] * RATE_SCALE;
    const spatialValue = latest.spatial[row] * RATE_SCALE;
    const z = latest.standardized[row];
    const rows: TooltipRow[] = [];
    const rateRow = (label: string, value: number, emphasis = false): TooltipRow => ({
      ...rate(value),
      label,
      swatch: swatchOf(value, table),
      emphasis
    });
    const quadrant = latest.quadrants[row];
    const quadrantColors = getMoranQuadrantColors(getGiTable(ctx.ground(), 'No data'));
    const quadrantSwatch = [
      quadrantColors.notSignificant,
      quadrantColors.highHigh,
      quadrantColors.lowHigh,
      quadrantColors.lowLow,
      quadrantColors.highLow
    ][Math.min(quadrant, 4)];
    const pValue =
      ctx.options.gating === 'permutation' ? `, p = ${latest.pseudoP[row].toFixed(3)}` : '';
    // First row: the mapped value with its class swatch; second: its percentile.
    let rankValue = rawValue;
    let sortedRank = latest.sorted.raw;
    if (map === 'raw') rows.push(rateRow('Raw rate', rawValue, true));
    else if (map === 'smoothed') {
      rows.push(rateRow('Empirical Bayes rate', smoothedValue, true));
      rankValue = smoothedValue;
      sortedRank = latest.sorted.smoothed;
    } else if (map === 'spatial') {
      rows.push(rateRow('Spatial empirical Bayes rate', spatialValue, true));
      rankValue = spatialValue;
      sortedRank = latest.sorted.spatial;
    } else if (map === 'standardized') {
      rows.push({
        label: 'Standardised rate',
        value: z.toFixed(2),
        unit: 'standard deviations',
        swatch: swatchOf(z, zTable),
        emphasis: true
      });
      rankValue = z;
      sortedRank = latest.sorted.z;
    } else {
      rows.push({
        label: 'Cluster',
        value: `${QUADRANT_NAMES[Math.min(quadrant, 4)]}${pValue}`,
        swatch: quadrantSwatch,
        emphasis: true
      });
      rankValue = smoothedValue;
      sortedRank = latest.sorted.smoothed;
    }
    rows.push({
      label: 'Rank',
      value: `${formatOrdinal(getPercentile(sortedRank, rankValue))} percentile`,
      unit: `of ${formatCount(n)} counties`
    });
    rows.push(
      {label: 'Births', value: formatCount(events[row]), unit: 'in 2021-2023'},
      {
        label: 'Women aged 15-44',
        value: formatCount(womenYears[row] / 3),
        unit: 'a year, on average'
      }
    );
    if (map !== 'raw') rows.push(rateRow('Raw rate', rawValue));
    if (map !== 'smoothed' && map !== 'cluster')
      rows.push(rateRow('Empirical Bayes', smoothedValue));
    if (map !== 'spatial') rows.push(rateRow('Spatial empirical Bayes', spatialValue));
    rows.push({
      label: 'Weight on own births',
      value: formatPercent(map === 'spatial' ? latest.local.weights[row] : latest.weight[row])
    });
    if (map !== 'standardized') {
      rows.push({label: 'Standardised rate', value: z.toFixed(2), unit: 'standard deviations'});
    }
    if (map === 'spatial' || ctx.options.swipe === 'eb-spatial') {
      rows.push({
        label: 'Neighbourhood rate',
        value: (latest.local.pooledRates[row] * RATE_SCALE).toFixed(1),
        unit: `per 1,000 (${latest.offsets[row + 1] - latest.offsets[row]} neighbours)`
      });
    }
    if (map !== 'cluster') {
      rows.push({
        label: 'Cluster',
        value: `${QUADRANT_NAMES[Math.min(quadrant, 4)]}${pValue}`,
        swatch: quadrantSwatch
      });
    }
    const labelPoint = mesh.labelPoints[row];
    return {
      title: countyName(row),
      subtitle: `County, ${RATE_BASIS}`,
      rows,
      anchor: labelPoint && Number.isFinite(labelPoint[0]) ? (labelPoint as LngLat) : undefined,
      highlight: {kind: 'polygon', rings: getRings(row)}
    };
  };

  return {
    getCompiledGraphs: () => [...variants.values()].map(variant => variant.compiled),

    setOption(id) {
      if (['criterion', 'analyze', 'gating', 'alternative', 'falseDiscoveryRate'].includes(id)) {
        active = getVariant(ctx.options);
        publishTest();
        dirty = true;
        reader.markStale();
      } else if (['permutations', 'seed', 'significance'].includes(id)) {
        writeParameters();
      } else if (id === 'map' || id === 'swipe') {
        legendHighlight = null;
        publishFunnel();
        updateNotes();
      } else if (id === 'showExtremes') {
        updateNotes();
      }
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 2) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.markStale();
      }
      if (frame.frameIndex >= 2) reader.flush(commandEncoder);
    },

    getLayers,

    onGroundChange() {
      publishLegendData();
      publishReadouts();
      publishFunnel();
      ctx.requestLayers();
    },

    onLegendFilter(id, classes) {
      if (classes === null) legendHighlight = null;
      else if (id === 'moran-classes') {
        legendHighlight = classes.map(index => MORAN_LEGEND_TO_CATEGORY[index]);
      } else legendHighlight = [...classes];
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const found = locator.find(event.coordinate);
      return found ? describeCounty(found.index) : null;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const found = locator.find(event.coordinate);
      const row = found ? found.index : -1;
      setSelection(row === selectedRow ? -1 : row);
      ctx.setHighlight(null);
      return true;
    },

    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}
