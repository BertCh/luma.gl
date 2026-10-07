// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPUBivariateClassification,
  GPUClassBreaks,
  GPUColorScale,
  GPUColumnProfile,
  GPUColumnQuantiles,
  GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH,
  GPU_CLASS_BREAKS_METHODS,
  GPU_CLASS_BREAKS_METHOD_CODES,
  GPU_COLOR_SCALE_PARAMETER_LENGTH,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT,
  getGPUBivariateClassificationParameterValues,
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues,
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesParameterValues,
  type GPUClassBreaksMethod,
  type GPUColorScaleType
} from '@luma.gl/experimental/gpu-dataframe';
import {
  GPU_CLASSIFICATION_FIT_ADAM,
  GPU_CLASSIFICATION_FIT_ADCM,
  GPU_CLASSIFICATION_FIT_GADF,
  GPU_CLASSIFICATION_FIT_GVF,
  GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH,
  GPUClassAssignment,
  GPUClassificationFit
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  createTransientView,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {getClassCounts} from '../../cartography/breaks';
import {
  getClassIndexOf,
  getClassLabel,
  getClassTableLayerProps,
  makeClassTable
} from '../../cartography/class-table';
import {hexToRgba, MAP_INK, NO_DATA_COLOR} from '../../cartography/hue-registry';
import {formatCount, formatPercent, formatOrdinal, liveText} from '../../cartography/live-text';
import type {ClassColor, ClassTable, LngLat, MapAnnotation} from '../../cartography/types';
import {SpatialAnalysisPolygonLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {getHairlineColor, getStateLineStyle} from '../weights/hot-spots.style';
import {addBreaksSelectPass, createGraphImporter} from './b5-classify';
import {createByteReader, getQuantile, getSortedFinite} from './b5-common';
import {getCountyVariable} from './b5-variables';
import {createCountyGeometry} from './choropleth-classes.geometry';
import {
  formatClassValue,
  getBinUsage,
  getBivariatePalette,
  getGoodnessOfAbsoluteDeviationFit,
  getMethodScheme,
  getNoDataLabel,
  getPackedNoDataColor,
  getRankShare,
  getSchemeColors,
  isDivergingMethod,
  LOG_VARIABLES,
  makeBaselineTable,
  METHOD_LABELS,
  packColors
} from './choropleth-classes.style';

/** Option state of the choropleth-classes scene. */
export type ChoroplethClassesOptions = {
  variable: string;
  method: GPUClassBreaksMethod;
  classCount: number;
  /** The frozen classification drawn left of the swipe divider, or `'none'`. */
  compareBaseline: 'none' | 'equal-interval' | 'quantile';
  /** Which data-driven annotations the step shows. */
  annotationSet: 'none' | 'outlier' | 'methods' | 'extremes' | 'counts';
  scale: 'threshold' | 'quantize' | 'linear' | 'sqrt' | 'pow' | 'log' | 'symlog';
  smoothBlend: boolean;
  clamp: boolean;
  exponent: number;
  logFloorExponent: number;
  lowerPercentile: number;
  upperPercentile: number;
  quantileInterpolation: 'linear' | 'lower' | 'higher' | 'nearest' | 'midpoint';
  standardDeviationInterval: number;
  headTailRatio: number;
  boxPlotHinge: number;
  naturalBreaksBinCount: string;
  bivariate: boolean;
  bivariateVariable: string;
  bivariateMethod: 'quantile' | 'equal-interval';
  valueByAlpha: boolean;
  minimumAlpha: number;
  histogramBins: string;
  hllPrecision: string;
  outlines: boolean;
};

/** Data the legends read, stored with `ctx.setLegendData('classes', ...)`. */
export type ClassesLegendData = {
  /** The class table of the live (GPU) classification: layer, legend and tooltip read it. */
  table: ClassTable;
  /** Counties per class. */
  counts: number[];
  /** Equal-width bins of the filtered values (`GPUColumnProfile`), spanning the table extent. */
  histogram: number[];
  /** The frozen classification left of the swipe divider, or `null`. */
  baseline: ClassTable | null;
  /** Counties per class of the frozen classification. */
  baselineCounts: number[];
  /**
   * Which classification the map shows: `'a'` while the toggle compare holds the frozen side, else
   * `'b'` (the live GPU classification, also during a swipe where the legend follows side b).
   */
  side: 'a' | 'b';
  /** Bivariate tertile edges and counts per palette cell, or `null` when off. */
  bivariate: {edgesX: number[]; edgesY: number[]; counts: number[]} | null;
  /** True when a percentile cut removes counties from the classification. */
  cut: boolean;
};

const MAXIMUM_CLASS_COUNT = 9;
/** Bivariate maps are 3 x 3 (tertiles). The contributors are compiled for up to 4 classes. */
const BIVARIATE_SIZE = 3;
const MAXIMUM_BIVARIATE_CLASS_COUNT = 4;
const QUANTILE_PROBABILITIES = [0.25, 0.5, 0.75] as const;
const MAXIMUM_PROFILE_BINS = 96;
const TOP_CATEGORY_COUNT = 10;
/** Profile columns: the mapped value, the second variable, rural-urban codes, log10 of the value. */
const PROFILE_COLUMN = {value: 0, second: 1, rucc: 2, logValue: 3} as const;
const PROFILE_COLUMN_COUNT = 4;
const RUCC_LABELS = [
  '?',
  'metro 1M+',
  'metro 250k-1M',
  'metro under 250k',
  'urban 20k+, next to metro',
  'urban 20k+, remote',
  'urban 5-20k, next to metro',
  'urban 5-20k, remote',
  'rural, next to metro',
  'rural, remote'
];
const READBACK_INTERVAL_FRAMES = 6;
const ENCODED_FRAMES_AFTER_CHANGE = 3;
const NO_CLASS = 0xffffffff;
/** Value-by-alpha ramps from 5,000 to 500,000 residents on a log scale: small counties fade. */
const ALPHA_POPULATION_DOMAIN: readonly [number, number] = [5000, 500000];
const ALPHA_LOG_DOMAIN: readonly [number, number] = [
  Math.log10(ALPHA_POPULATION_DOMAIN[0]),
  Math.log10(ALPHA_POPULATION_DOMAIN[1])
];

type Summary = {
  breaks: Float32Array;
  classCount: number;
  classCounts: Uint32Array;
  filterBounds: Float32Array;
  quantiles: Float32Array;
  validCount: number;
  bivariateCounts: Uint32Array;
  bivariateBreaksX: Float32Array;
  bivariateBreaksY: Float32Array;
  profileStatistics: Float32Array;
  profileHistograms: Uint32Array;
  topCategories: Uint32Array;
  topCategoryCounts: Uint32Array;
  fit: Float32Array;
  classIndices: Uint32Array;
};

/** Round-number class edges spanning the 2nd to 98th percentile. */
function getRoundEdges(sorted: Float64Array, classCount: number): number[] {
  const lo = getQuantile(sorted, 0.02);
  const hi = getQuantile(sorted, 0.98);
  const span = Math.max(hi - lo, 1e-9);
  const rough = span / classCount;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map(m => m * magnitude).find(value => value >= rough) ?? rough;
  const start = Math.floor(lo / step) * step;
  const edges = Array.from({length: classCount + 1}, (_, index) => start + index * step);
  edges[0] = Math.min(edges[0], sorted[0]);
  edges[classCount] = Math.max(edges[classCount], sorted[sorted.length - 1]);
  return edges;
}

/**
 * Choropleth classification of US counties, entirely on the GPU: `GPUColumnQuantiles` (percentile
 * filter), `GPUClassBreaks` (8 methods), `GPUColorScale` (scale type and the exact ColorBrewer
 * palette), `GPUClassAssignment` and `GPUClassificationFit` (goodness of fit),
 * `GPUBivariateClassification` (two variables and value-by-alpha) and `GPUColumnProfile` (the
 * histogram of the legend and the distribution chart). The map layer reads the packed rgba8 colours
 * of `GPUColorScale` directly, so a recolour never leaves the GPU. The swipe compare draws a
 * frozen CPU classification (equal interval or quantile, same counties) on the left side over the
 * same value buffer. Every control except the natural-breaks bin count and the profile settings is
 * a parameter or palette write.
 */
export async function createChoroplethClasses(
  ctx: SceneContext<ChoroplethClassesOptions>
): Promise<SceneInstance<ChoroplethClassesOptions>> {
  const counties = ctx.datasets.get('us-counties');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'choropleth-classes');
  const geometry = createCountyGeometry(resources, counties, ctx.datasets.get('us-states'));
  const features = geometry.features;
  const n = features.length;
  const getProperty = (row: number, key: string): string =>
    String(features[row]?.properties?.[key] ?? '');
  const nameOf = (row: number): string =>
    `${getProperty(row, 'name')}, ${getProperty(row, 'state')}`;
  const population = counties.column<Float32Array>('population');
  const landArea = counties.column<Float32Array>('landArea');
  const getLabelPoint = (row: number): LngLat => geometry.mesh.labelPoints[row] as LngLat;
  const findRow = (name: string, state: string): number =>
    features.findIndex(
      feature => feature.properties?.['name'] === name && feature.properties?.['state'] === state
    );

  const xBuffer = resources.createBuffer('x-values', n * 4);
  const yBuffer = resources.createBuffer('y-values', n * 4);
  const logXBuffer = resources.createBuffer('log-x-values', n * 4);
  const alphaBuffer = resources.createBuffer(
    'alpha-values',
    Float32Array.from(population, value => (value > 0 ? Math.log10(value) : Number.NaN))
  );
  const occupiedBuffer = resources.createBuffer('occupied', n * 4);
  const filterMask = resources.createBuffer('filter-mask', n * 4);
  const colors = resources.createBuffer('colors', n * 4);
  const classIndices = resources.createBuffer('class-indices', n * 4);
  const bivariateColors = resources.createBuffer('bivariate-colors', n * 4);
  const fitClasses = resources.createBuffer('fit-classes', n * 4);

  const quantileParameters = resources.createParameterBuffer(
    'quantile-parameters',
    'float32',
    getGPUColumnQuantilesParameterLength(QUANTILE_PROBABILITIES.length)
  );
  const breaksParameters = resources.createParameterBuffer(
    'breaks-parameters',
    'float32',
    getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT)
  );
  const scaleParameters = resources.createParameterBuffer(
    'scale-parameters',
    'float32',
    GPU_COLOR_SCALE_PARAMETER_LENGTH
  );
  const axisParameterLength = getGPUClassBreaksParameterLength(MAXIMUM_BIVARIATE_CLASS_COUNT);
  const axisXParameters = resources.createParameterBuffer(
    'axis-x-parameters',
    'float32',
    axisParameterLength
  );
  const axisYParameters = resources.createParameterBuffer(
    'axis-y-parameters',
    'float32',
    axisParameterLength
  );
  const bivariateParameters = resources.createParameterBuffer(
    'bivariate-parameters',
    'float32',
    GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH
  );
  const paletteBuffer = resources.createBuffer('palette', MAXIMUM_CLASS_COUNT * 4);
  const bivariatePaletteBuffer = resources.createBuffer(
    'bivariate-palette',
    MAXIMUM_BIVARIATE_CLASS_COUNT ** 2 * 4
  );
  const filterBounds = resources.createBuffer('filter-bounds', 8);
  const quantiles = resources.createBuffer('quantiles', QUANTILE_PROBABILITIES.length * 4);
  const validCount = resources.createBuffer('valid-count', 4);
  const breaks = resources.createBuffer('breaks', (MAXIMUM_CLASS_COUNT + 1) * 4);
  const classCountBuffer = resources.createBuffer('class-count', 4);
  const scaleClassCounts = resources.createBuffer('scale-class-counts', MAXIMUM_CLASS_COUNT * 4);
  const axisBreaksX = resources.createBuffer(
    'axis-breaks-x',
    (MAXIMUM_BIVARIATE_CLASS_COUNT + 1) * 4
  );
  const axisBreaksY = resources.createBuffer(
    'axis-breaks-y',
    (MAXIMUM_BIVARIATE_CLASS_COUNT + 1) * 4
  );
  const bivariateCounts = resources.createBuffer(
    'bivariate-counts',
    MAXIMUM_BIVARIATE_CLASS_COUNT ** 2 * 4
  );
  const profileStatistics = resources.createBuffer(
    'profile-statistics',
    PROFILE_COLUMN_COUNT * GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4
  );
  const profileHistograms = resources.createBuffer(
    'profile-histograms',
    PROFILE_COLUMN_COUNT * MAXIMUM_PROFILE_BINS * 4
  );
  const topCategories = resources.createBuffer(
    'top-categories',
    PROFILE_COLUMN_COUNT * TOP_CATEGORY_COUNT * 4
  );
  const topCategoryCounts = resources.createBuffer(
    'top-category-counts',
    PROFILE_COLUMN_COUNT * TOP_CATEGORY_COUNT * 4
  );
  const ruccCodes = resources.createBuffer('rucc-codes', n * 4);
  const fitCounts = resources.createBuffer('fit-counts', (MAXIMUM_CLASS_COUNT + 1) * 4);
  const fitMedians = resources.createBuffer('fit-medians', (MAXIMUM_CLASS_COUNT + 1) * 4);
  const fitAbsolute = resources.createBuffer('fit-absolute', (MAXIMUM_CLASS_COUNT + 1) * 4);
  const fitSquared = resources.createBuffer('fit-squared', (MAXIMUM_CLASS_COUNT + 1) * 4);
  const fitSummary = resources.createBuffer(
    'fit-summary',
    GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH * 4
  );

  const readbackLayout: {buffer: Buffer; byteLength: number}[] = [
    {buffer: breaks, byteLength: (MAXIMUM_CLASS_COUNT + 1) * 4},
    {buffer: classCountBuffer, byteLength: 4},
    {buffer: scaleClassCounts, byteLength: MAXIMUM_CLASS_COUNT * 4},
    {buffer: filterBounds, byteLength: 8},
    {buffer: quantiles, byteLength: QUANTILE_PROBABILITIES.length * 4},
    {buffer: validCount, byteLength: 4},
    {buffer: bivariateCounts, byteLength: MAXIMUM_BIVARIATE_CLASS_COUNT ** 2 * 4},
    {buffer: axisBreaksX, byteLength: (MAXIMUM_BIVARIATE_CLASS_COUNT + 1) * 4},
    {buffer: axisBreaksY, byteLength: (MAXIMUM_BIVARIATE_CLASS_COUNT + 1) * 4},
    {
      buffer: profileStatistics,
      byteLength: PROFILE_COLUMN_COUNT * GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4
    },
    {buffer: profileHistograms, byteLength: PROFILE_COLUMN_COUNT * MAXIMUM_PROFILE_BINS * 4},
    {buffer: topCategories, byteLength: PROFILE_COLUMN_COUNT * TOP_CATEGORY_COUNT * 4},
    {buffer: topCategoryCounts, byteLength: PROFILE_COLUMN_COUNT * TOP_CATEGORY_COUNT * 4},
    {buffer: fitSummary, byteLength: GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH * 4},
    {buffer: classIndices, byteLength: n * 4}
  ];
  const readbackByteLength = readbackLayout.reduce((total, part) => total + part.byteLength, 0);
  const readbackRing = resources.track(
    new GPUReadbackRing(device, {id: 'classes-summary', byteLength: readbackByteLength})
  );

  // CPU copies of the selected variables, for the round-number edges, the frozen baseline,
  // the rank in tooltips and the readouts that are not GPU outputs.
  let xColumn: Float32Array = new Float32Array(n);
  let yColumn: Float32Array = new Float32Array(n);
  let sortedX: Float64Array = new Float64Array(0);
  let loadedVariable = '';
  let loadedBivariateVariable = '';

  let writtenClassCount = 0;
  let writtenPaletteKey = '';
  /** The palette written to the GPU, one RGBA colour per class (what the legend swatches show). */
  let activeColors: ClassColor[] = [];
  let latestSummary: Summary | null = null;
  let currentTable: ClassTable | null = null;
  let dirtyFrames = ENCODED_FRAMES_AFTER_CHANGE;
  let lastReadbackFrame = -READBACK_INTERVAL_FRAMES;
  let readbackWanted = true;
  let readbackPending = false;
  let destroyed = false;
  let selectedRow = -1;
  let selectionSegments = 0;
  let hoverRow = -1;
  let legendHighlight: number[] | null = null;
  let parameterVersion = 0;
  let activeNaturalBins = Number(ctx.options.naturalBreaksBinCount);
  let activeProfileBins = Number(ctx.options.histogramBins);
  let activeHllPrecision = ctx.options.hllPrecision;
  let exactDistinct = 0;
  let furnitureKey = '';

  /** The frozen classification of the swipe compare, recomputed only when its inputs change. */
  let baselineKey = '';
  let baselineTable: ClassTable | null = null;
  let baselineCounts: number[] = [];
  /** What the compare divider shows now (`'a'` is only reported by the hold-to-compare toggle). */
  let compareShowing: 'a' | 'b' | 'both' = 'both';
  let legendData: Omit<ClassesLegendData, 'side'> | null = null;

  /** Method comparison sweep state. */
  let sweep: {queue: GPUClassBreaksMethod[]; current: GPUClassBreaksMethod} | null = null;
  const fitTable = new Map<GPUClassBreaksMethod, {gadf: number; gvf: number}>();
  let fitKey = '';

  let main: CompiledGPUCommandGraph<void> | null = null;
  let bivariateGraph: CompiledGPUCommandGraph<void> | null = null;
  let profileGraph: CompiledGPUCommandGraph<void> | null = null;
  let fitGraph: CompiledGPUCommandGraph<void> | null = null;

  // Counties that anchor the data-driven annotations, found by name in the data.
  const cookRow = findRow('Cook', 'IL');
  const maricopaRow = findRow('Maricopa', 'AZ');
  const loudounRow = findRow('Loudoun', 'VA');
  const losAngelesRow = findRow('Los Angeles', 'CA');
  /**
   * The county with the most land among those in the lowest quintile of both residents and
   * density: a large, nearly empty county of the West.
   */
  const emptyRow = (() => {
    const populationEdge = getQuantile(getSortedFinite(population), 0.2);
    const densityEdge = getQuantile(
      getSortedFinite(counties.column<Float32Array>('popDensity')),
      0.2
    );
    const density = counties.column<Float32Array>('popDensity');
    let best = -1;
    for (let row = 0; row < n; row++) {
      if (!(population[row] < populationEdge) || !(density[row] < densityEdge)) continue;
      if (best < 0 || landArea[row] > landArea[best]) best = row;
    }
    return best;
  })();

  function buildGraphs(): void {
    for (const graph of [main, bivariateGraph, profileGraph, fitGraph]) {
      if (graph) resources.release(graph);
    }
    activeNaturalBins = Number(ctx.options.naturalBreaksBinCount);
    activeProfileBins = Number(ctx.options.histogramBins);
    activeHllPrecision = ctx.options.hllPrecision;

    const graph = new GPUCommandGraph<void>(device, {id: 'classes-main'});
    const bind = createGraphImporter(graph);
    const values = bind.float(xBuffer, n);
    const filter = bind.word(filterMask, n);
    graph.add(
      new GPUColumnQuantiles({
        id: 'percentile-filter',
        values,
        mask: bind.word(occupiedBuffer, n),
        parameters: quantileParameters.importToGraph(graph),
        quantileCount: QUANTILE_PROBABILITIES.length,
        output: {
          quantiles: bind.float(quantiles, QUANTILE_PROBABILITIES.length),
          validCount: bind.word(validCount, 1),
          filterMask: filter,
          filterBounds: bind.float(filterBounds, 2)
        }
      })
    );
    const breaksParameterView = breaksParameters.importToGraph(graph);
    const primaryBreaks = createTransientView(
      graph,
      'primary-breaks',
      'float32',
      MAXIMUM_CLASS_COUNT + 1
    );
    const primaryClassCount = createTransientView(graph, 'primary-class-count', 'uint32', 1);
    const headTailBreaks = createTransientView(
      graph,
      'head-tail-breaks',
      'float32',
      MAXIMUM_CLASS_COUNT + 1
    );
    const headTailClassCount = createTransientView(graph, 'head-tail-class-count', 'uint32', 1);
    graph.add(
      new GPUClassBreaks({
        id: 'breaks',
        values,
        mask: filter,
        parameters: breaksParameterView,
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        methods: GPU_CLASS_BREAKS_METHODS.filter(value => value !== 'head-tail'),
        naturalBreaksBinCount: activeNaturalBins,
        output: {breaks: primaryBreaks, classCount: primaryClassCount}
      })
    );
    graph.add(
      new GPUClassBreaks({
        id: 'head-tail',
        values,
        mask: filter,
        parameters: breaksParameterView,
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        methods: ['head-tail'],
        output: {breaks: headTailBreaks, classCount: headTailClassCount}
      })
    );
    const breaksView = bind.float(breaks, MAXIMUM_CLASS_COUNT + 1);
    const classCountView = bind.word(classCountBuffer, 1);
    addBreaksSelectPass(graph, {
      id: 'select-breaks',
      maximumClassCount: MAXIMUM_CLASS_COUNT,
      parameters: breaksParameterView,
      alternateMethodCode: GPU_CLASS_BREAKS_METHOD_CODES['head-tail'],
      primaryBreaks,
      primaryClassCount,
      alternateBreaks: headTailBreaks,
      alternateClassCount: headTailClassCount,
      breaks: breaksView,
      classCount: classCountView
    });
    graph.add(
      new GPUColorScale({
        id: 'scale',
        values,
        mask: filter,
        domain: breaksView,
        domainCount: classCountView,
        palette: bind.word(paletteBuffer, MAXIMUM_CLASS_COUNT),
        parameters: scaleParameters.importToGraph(graph),
        maximumDomainCount: MAXIMUM_CLASS_COUNT + 1,
        maximumPaletteCount: MAXIMUM_CLASS_COUNT,
        output: {
          colors: bind.word(colors, n),
          classIndices: bind.word(classIndices, n),
          classCounts: bind.word(scaleClassCounts, MAXIMUM_CLASS_COUNT)
        }
      })
    );
    main = resources.track(graph.compile());

    const biGraph = new GPUCommandGraph<void>(device, {id: 'classes-bivariate'});
    const biBind = createGraphImporter(biGraph);
    const biX = biBind.float(xBuffer, n);
    const biY = biBind.float(yBuffer, n);
    const biFilter = biBind.word(filterMask, n);
    const biBreaksX = biBind.float(axisBreaksX, MAXIMUM_BIVARIATE_CLASS_COUNT + 1);
    const biBreaksY = biBind.float(axisBreaksY, MAXIMUM_BIVARIATE_CLASS_COUNT + 1);
    for (const [axis, axisValues, axisBreaks, parameters] of [
      ['x', biX, biBreaksX, axisXParameters],
      ['y', biY, biBreaksY, axisYParameters]
    ] as const) {
      biGraph.add(
        new GPUClassBreaks({
          id: `axis-${axis}`,
          values: axisValues,
          mask: biFilter,
          parameters: parameters.importToGraph(biGraph),
          maximumClassCount: MAXIMUM_BIVARIATE_CLASS_COUNT,
          methods: ['quantile', 'equal-interval'],
          output: {
            breaks: axisBreaks,
            classCount: createTransientView(biGraph, `axis-${axis}-class-count`, 'uint32', 1)
          }
        })
      );
    }
    biGraph.add(
      new GPUBivariateClassification({
        id: 'bivariate',
        valuesX: biX,
        valuesY: biY,
        mask: biFilter,
        breaksX: biBreaksX,
        breaksY: biBreaksY,
        palette: biBind.word(bivariatePaletteBuffer, MAXIMUM_BIVARIATE_CLASS_COUNT ** 2),
        alphaValues: biBind.float(alphaBuffer, n),
        parameters: bivariateParameters.importToGraph(biGraph),
        maximumClassCount: MAXIMUM_BIVARIATE_CLASS_COUNT,
        output: {
          colors: biBind.word(bivariateColors, n),
          classCounts: biBind.word(bivariateCounts, MAXIMUM_BIVARIATE_CLASS_COUNT ** 2)
        }
      })
    );
    bivariateGraph = resources.track(biGraph.compile());

    const profile = new GPUCommandGraph<void>(device, {id: 'classes-profile'});
    const profileBind = createGraphImporter(profile);
    profile.add(
      new GPUColumnProfile({
        id: 'profile',
        columns: [
          {values: profileBind.float(xBuffer, n)},
          {values: profileBind.float(yBuffer, n)},
          {kind: 'category', values: profileBind.word(ruccCodes, n), categoryCount: 10},
          {values: profileBind.float(logXBuffer, n)}
        ],
        mask: profileBind.word(filterMask, n),
        histogramBinCount: activeProfileBins,
        hyperLogLogPrecision: Number(ctx.options.hllPrecision),
        topCategoryCount: TOP_CATEGORY_COUNT,
        output: {
          statistics: profileBind.float(
            profileStatistics,
            PROFILE_COLUMN_COUNT * GPU_COLUMN_PROFILE_STATISTIC_COUNT
          ),
          histograms: profileBind.word(profileHistograms, PROFILE_COLUMN_COUNT * activeProfileBins),
          topCategories: profileBind.word(topCategories, PROFILE_COLUMN_COUNT * TOP_CATEGORY_COUNT),
          topCategoryCounts: profileBind.word(
            topCategoryCounts,
            PROFILE_COLUMN_COUNT * TOP_CATEGORY_COUNT
          )
        }
      })
    );
    profileGraph = resources.track(profile.compile());

    const fit = new GPUCommandGraph<void>(device, {id: 'classes-fit'});
    const fitBind = createGraphImporter(fit);
    const fitValues = fitBind.float(xBuffer, n);
    const assigned = fitBind.word(fitClasses, n);
    fit.add(
      new GPUClassAssignment({
        id: 'fit-assign',
        values: fitValues,
        breaks: fitBind.float(breaks, MAXIMUM_CLASS_COUNT + 1),
        classCount: fitBind.word(classCountBuffer, 1),
        mask: fitBind.word(filterMask, n),
        output: assigned
      })
    );
    fit.add(
      new GPUClassificationFit({
        id: 'fit',
        values: fitValues,
        classes: assigned,
        classCount: MAXIMUM_CLASS_COUNT,
        output: {
          counts: fitBind.word(fitCounts, MAXIMUM_CLASS_COUNT + 1),
          medians: fitBind.float(fitMedians, MAXIMUM_CLASS_COUNT + 1),
          absoluteDeviations: fitBind.float(fitAbsolute, MAXIMUM_CLASS_COUNT + 1),
          squaredDeviations: fitBind.float(fitSquared, MAXIMUM_CLASS_COUNT + 1),
          summary: fitBind.float(fitSummary, GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH)
        }
      })
    );
    fitGraph = resources.track(fit.compile());
    publishCost();
  }

  /** The cost line of the card: counties and GPU passes of the graphs that run every frame. */
  function publishCost(): void {
    const passes = [main, profileGraph, fitGraph].reduce(
      (total, graph) => total + (graph?.stats.nodeOrder.length ?? 0),
      0
    );
    ctx.setCost({records: n, passes});
  }

  /** Loads the chosen variables into the value buffers when they changed. */
  function loadVariables(): void {
    const {variable, bivariateVariable} = ctx.options;
    if (variable !== loadedVariable) {
      loadedVariable = variable;
      xColumn = counties.column<Float32Array>(variable);
      xBuffer.write(xColumn);
      logXBuffer.write(
        Float32Array.from(xColumn, value => (value > 0 ? Math.log10(value) : Number.NaN))
      );
      const occupied = new Uint32Array(n);
      for (let row = 0; row < n; row++) occupied[row] = Number.isFinite(xColumn[row]) ? 1 : 0;
      occupiedBuffer.write(occupied);
      sortedX = getSortedFinite(xColumn);
      exactDistinct = new Set(sortedX).size;
      latestSummary = null;
      baselineKey = '';
    }
    if (bivariateVariable !== loadedBivariateVariable) {
      loadedBivariateVariable = bivariateVariable;
      yColumn = counties.column<Float32Array>(bivariateVariable);
      yBuffer.write(yColumn);
    }
  }

  function getEffectiveMethod(): GPUClassBreaksMethod {
    return sweep ? sweep.current : ctx.options.method;
  }

  /** Identity of the palette the GPU should hold: scheme and ground (the class count is separate). */
  function getPaletteKey(): string {
    return `${getMethodScheme(ctx.options.variable, getEffectiveMethod())}|${ctx.ground()}`;
  }

  /**
   * Writes the exact published ColorBrewer table of the scheme (sequential hue of the variable, or
   * RdBu around the mean, PuOr around the median) to the palette buffer `GPUColorScale` reads.
   */
  function writePalette(classCount: number): void {
    writtenClassCount = Math.min(Math.max(classCount, 1), MAXIMUM_CLASS_COUNT);
    writtenPaletteKey = getPaletteKey();
    activeColors = getSchemeColors(
      getMethodScheme(ctx.options.variable, getEffectiveMethod()),
      writtenClassCount,
      ctx.ground()
    );
    const padded = new Uint32Array(MAXIMUM_CLASS_COUNT);
    padded.set(packColors(activeColors));
    paletteBuffer.write(padded);
  }

  function writeBivariatePalette(): void {
    const palette = new Uint32Array(MAXIMUM_BIVARIATE_CLASS_COUNT ** 2);
    const source = packColors(getBivariatePalette(ctx.ground()));
    palette.set(source.subarray(0, BIVARIATE_SIZE ** 2));
    bivariatePaletteBuffer.write(palette);
  }

  /** Recomputes the frozen left-hand classification when its inputs (not the live method) change. */
  function updateBaseline(): void {
    const options = ctx.options;
    if (options.compareBaseline === 'none' || options.bivariate || sortedX.length === 0) {
      baselineTable = null;
      baselineKey = '';
      ctx.setReadout('gadfBaseline', null);
      return;
    }
    const ground = ctx.ground();
    const key = `${options.variable}|${options.compareBaseline}|${options.classCount}|${ground}`;
    if (key === baselineKey) return;
    baselineKey = key;
    const variable = getCountyVariable(options.variable);
    baselineTable = makeBaselineTable({
      values: xColumn,
      method: options.compareBaseline,
      classCount: options.classCount,
      scheme: getMethodScheme(options.variable, 'quantile'),
      ground,
      unit: variable.unit,
      noDataLabel: getNoDataLabel(options.lowerPercentile, options.upperPercentile)
    });
    baselineCounts = getClassCounts(xColumn, baselineTable.breaks);
    const gadf = getGoodnessOfAbsoluteDeviationFit(sortedX, baselineTable.breaks);
    ctx.setReadout('gadfBaseline', gadf.toFixed(2));
  }

  /** Rewrites every per-frame parameter buffer from the options. */
  function writeParameters(): void {
    loadVariables();
    const options = ctx.options;
    const method = getEffectiveMethod();
    const requestedClassCount = options.classCount;
    const wanted = Math.min(latestSummary?.classCount || requestedClassCount, MAXIMUM_CLASS_COUNT);
    if (wanted !== writtenClassCount || getPaletteKey() !== writtenPaletteKey) writePalette(wanted);
    quantileParameters.write(
      getGPUColumnQuantilesParameterValues({
        quantiles: QUANTILE_PROBABILITIES,
        interpolation: options.quantileInterpolation,
        filterRange: [options.lowerPercentile / 100, options.upperPercentile / 100]
      })
    );
    const customEdges = getRoundEdges(sortedX, Math.min(requestedClassCount, MAXIMUM_CLASS_COUNT));
    breaksParameters.write(
      getGPUClassBreaksParameterValues(
        {
          method,
          classCount: requestedClassCount,
          standardDeviationInterval: options.standardDeviationInterval,
          headTailRatio: options.headTailRatio,
          boxPlotHinge: options.boxPlotHinge,
          customEdges
        },
        MAXIMUM_CLASS_COUNT
      )
    );
    const noDataColor = getPackedNoDataColor(ctx.ground());
    scaleParameters.write(
      getGPUColorScaleParameterValues({
        scale: options.scale as GPUColorScaleType,
        domainCount: writtenClassCount + 1,
        paletteCount: writtenClassCount,
        interpolation: options.smoothBlend && options.scale !== 'threshold' ? 'linear' : 'step',
        clamp: options.clamp,
        noDataColor,
        logFloor: 10 ** options.logFloorExponent,
        exponent: options.exponent
      })
    );
    const axisSettings = {method: options.bivariateMethod, classCount: BIVARIATE_SIZE};
    axisXParameters.write(
      getGPUClassBreaksParameterValues(axisSettings, MAXIMUM_BIVARIATE_CLASS_COUNT)
    );
    axisYParameters.write(
      getGPUClassBreaksParameterValues(axisSettings, MAXIMUM_BIVARIATE_CLASS_COUNT)
    );
    bivariateParameters.write(
      getGPUBivariateClassificationParameterValues({
        classCountX: BIVARIATE_SIZE,
        classCountY: BIVARIATE_SIZE,
        noDataColor,
        // Opacity follows log10 of the population, linear between 5,000 and 500,000 residents.
        valueByAlpha: options.valueByAlpha
          ? {domain: ALPHA_LOG_DOMAIN, minimumAlpha: options.minimumAlpha}
          : undefined
      })
    );
    writeBivariatePalette();
    updateBaseline();
    parameterVersion++;
    dirtyFrames = ENCODED_FRAMES_AFTER_CHANGE;
    readbackWanted = true;
  }

  /** Class edges as the legend shows them: the breaks, or the scale's own steps for other scales. */
  const getLegendEdges = (summary: Summary, k: number): number[] => {
    const edges: number[] = [];
    const first = summary.breaks[0];
    const last = summary.breaks[k];
    const floor = 10 ** ctx.options.logFloorExponent;
    const scale = ctx.options.scale;
    for (let index = 0; index <= k; index++) {
      const t = index / Math.max(k, 1);
      if (scale === 'threshold') edges.push(summary.breaks[index]);
      else if (scale === 'sqrt') {
        const low = Math.sqrt(Math.max(first, 0));
        const high = Math.sqrt(Math.max(last, 0));
        edges.push((low + (high - low) * t) ** 2);
      } else if (scale === 'log') {
        const low = Math.log(Math.max(first, floor));
        const high = Math.log(Math.max(last, floor));
        edges.push(Math.exp(low + (high - low) * t));
      } else if (scale === 'pow') {
        const exponent = ctx.options.exponent;
        const low = Math.sign(first) * Math.abs(first) ** exponent;
        const high = Math.sign(last) * Math.abs(last) ** exponent;
        const value = low + (high - low) * t;
        edges.push(Math.sign(value) * Math.abs(value) ** (1 / exponent));
      } else if (scale === 'symlog') {
        const transform = (value: number) => Math.sign(value) * Math.log1p(Math.abs(value));
        const inverse = (value: number) => Math.sign(value) * Math.expm1(Math.abs(value));
        const low = transform(first);
        const high = transform(last);
        edges.push(inverse(low + (high - low) * t));
      } else edges.push(first + (last - first) * t);
    }
    return edges;
  };

  const statistic = (
    summary: Summary,
    column: number,
    field: keyof typeof GPU_COLUMN_PROFILE_STATISTIC
  ): number =>
    summary.profileStatistics[
      column * GPU_COLUMN_PROFILE_STATISTIC_COUNT + GPU_COLUMN_PROFILE_STATISTIC[field]
    ];

  /** The histogram bins of one profile column. */
  const getProfileBins = (summary: Summary, column: number): number[] =>
    Array.from(
      summary.profileHistograms.subarray(
        column * activeProfileBins,
        (column + 1) * activeProfileBins
      )
    );

  /** The class table of the live classification, from the GPU breaks and the written palette. */
  function buildTable(summary: Summary, gadf: number, gvf: number): ClassTable {
    const options = ctx.options;
    const variable = getCountyVariable(options.variable);
    const classCount = Math.min(summary.classCount, MAXIMUM_CLASS_COUNT);
    const edges = getLegendEdges(summary, classCount);
    const method = getEffectiveMethod();
    const continuous =
      options.scale === 'threshold'
        ? ''
        : `, ${options.scale} colour scale (class ranges follow the scale)`;
    return makeClassTable({
      breaks: edges.slice(1, classCount),
      colors: activeColors.slice(0, classCount),
      unit: variable.unit,
      extent: [edges[0], edges[classCount]],
      format: formatClassValue,
      method: `${METHOD_LABELS[method]}, GADF ${gadf.toFixed(2)}, GVF ${gvf.toFixed(2)}${continuous}`,
      noData: {
        label: getNoDataLabel(options.lowerPercentile, options.upperPercentile),
        count: Math.max(0, n - summary.validCount)
      }
    });
  }

  /** Marker of the hovered county on the distribution chart. */
  function publishDistribution(hoverValue: number | null): void {
    const summary = latestSummary;
    const table = getDisplayedTable();
    if (!summary || !table) return;
    const options = ctx.options;
    const variable = getCountyVariable(options.variable);
    const log = LOG_VARIABLES.has(options.variable);
    const column = log ? PROFILE_COLUMN.logValue : PROFILE_COLUMN.value;
    const low = statistic(summary, column, 'minimum');
    const high = statistic(summary, column, 'maximum');
    if (!Number.isFinite(low) || !Number.isFinite(high) || !(high > low)) {
      ctx.setChart('distribution', null);
      return;
    }
    const transform = (value: number) => (log ? Math.log10(Math.max(value, 1e-6)) : value);
    ctx.setChart('distribution', {
      kind: 'histogram',
      values: getProfileBins(summary, column),
      xDomain: [low, high],
      breaks: table.breaks.map(transform),
      classColors: table.colors,
      formatX: log ? value => formatClassValue(10 ** value) : formatClassValue,
      xLabel: `${variable.label} (${variable.unit})${log ? ', log scale' : ''}`,
      yLabel: 'Counties',
      height: 120,
      ...(hoverValue !== null && Number.isFinite(hoverValue)
        ? {now: transform(hoverValue), nowLabel: formatClassValue(hoverValue)}
        : {}),
      description: `Histogram of ${variable.label.toLowerCase()} over counties with the class edges of the current method${log ? ' on a logarithmic axis' : ''}.`
    });
  }

  /** Data-driven annotations of the step: notes and county labels read from the data. */
  function publishAnnotations(): void {
    const summary = latestSummary;
    const options = ctx.options;
    if (options.annotationSet === 'none' || !summary || options.bivariate) {
      ctx.setAnnotations('classes', null);
      return;
    }
    const variable = getCountyVariable(options.variable);
    const classCount = summary.classCount;
    const valueText = (row: number) => `${formatClassValue(xColumn[row])} ${variable.unit}`;
    const classText = (row: number) => {
      const classIndex = summary.classIndices[row];
      return classIndex === NO_CLASS ? 'excluded' : `class ${classIndex + 1} of ${classCount}`;
    };
    const list: MapAnnotation[] = [];
    if (options.annotationSet === 'outlier' || options.annotationSet === 'extremes') {
      let highest = -1;
      let lowest = -1;
      for (let row = 0; row < n; row++) {
        if (!Number.isFinite(xColumn[row])) continue;
        if (highest < 0 || xColumn[row] > xColumn[highest]) highest = row;
        if (lowest < 0 || xColumn[row] < xColumn[lowest]) lowest = row;
      }
      const note = (row: number, label: string): MapAnnotation => ({
        kind: 'note',
        coordinate: getLabelPoint(row),
        title: valueText(row),
        text: `${nameOf(row)}: ${label}`
      });
      if (highest >= 0) list.push(note(highest, 'highest value'));
      if (options.annotationSet === 'extremes' && lowest >= 0 && lowest !== highest) {
        list.push(note(lowest, 'lowest value'));
      }
    } else if (options.annotationSet === 'methods') {
      for (const row of [cookRow, maricopaRow, loudounRow]) {
        if (row < 0) continue;
        list.push({
          kind: 'point',
          coordinate: getLabelPoint(row),
          text: nameOf(row),
          detail: classText(row),
          rank: 'subject',
          priority: 3
        });
      }
    } else if (options.annotationSet === 'counts') {
      if (losAngelesRow >= 0) {
        list.push({
          kind: 'note',
          coordinate: getLabelPoint(losAngelesRow),
          title: valueText(losAngelesRow),
          text: `${nameOf(losAngelesRow)}: ${classText(losAngelesRow)}`
        });
      }
      if (emptyRow >= 0) {
        list.push({
          kind: 'note',
          coordinate: getLabelPoint(emptyRow),
          title: valueText(emptyRow),
          text: liveText('{name}: {area:integer} km² of land, {classText}', {
            name: nameOf(emptyRow),
            area: landArea[emptyRow],
            classText: classText(emptyRow)
          })
        });
      }
    }
    ctx.setAnnotations('classes', list.length ? list : null);
  }

  /** Cartouche subtitle and chips from the live variable and method (the claim stays the step's). */
  function publishFurniture(summary: Summary): void {
    const options = ctx.options;
    const variable = getCountyVariable(options.variable);
    const other = getCountyVariable(options.bivariateVariable);
    const modelled = (id: string) => id.startsWith('places_');
    const chips = [
      ...(modelled(options.variable) || (options.bivariate && modelled(options.bivariateVariable))
        ? ['Modelled']
        : []),
      ...(!modelled(options.variable) || options.bivariate ? ['Estimates'] : [])
    ];
    const subtitle = options.bivariate
      ? `${variable.label} (across) by ${other.label} (up), tertiles`
      : options.variable === 'population' && options.annotationSet === 'counts'
        ? 'Counts on unequal areas: the wrong map'
        : `${variable.label}, ${variable.unit}; ${METHOD_LABELS[getEffectiveMethod()]}, ${summary.classCount} classes`;
    const sample = `${formatCount(n)} counties of the contiguous US (Alaska and Hawaii not in the data)`;
    const key = `${subtitle}|${chips.join(',')}`;
    if (key === furnitureKey) return;
    furnitureKey = key;
    ctx.setFurniture({title: {subtitle, sample, chips}});
  }

  /** The classification the map shows: the frozen side while the toggle holds side a. */
  function getDisplayedTable(): ClassTable | null {
    return compareShowing === 'a' && baselineTable && !ctx.options.bivariate
      ? baselineTable
      : currentTable;
  }

  /** Hands the legend its data, with the side the map shows. */
  function pushLegend(): void {
    if (!legendData) return;
    ctx.setLegendData('classes', {
      ...legendData,
      baseline: baselineTable,
      baselineCounts,
      side: getDisplayedTable() === baselineTable && baselineTable ? 'a' : 'b'
    } satisfies ClassesLegendData);
  }

  function publishSummary(summary: Summary, gadf: number, gvf: number): void {
    const options = ctx.options;
    const variable = getCountyVariable(options.variable);
    const classCount = Math.min(summary.classCount, MAXIMUM_CLASS_COUNT);
    const counts = Array.from(summary.classCounts.subarray(0, classCount));
    const table = buildTable(summary, gadf, gvf);
    currentTable = table;
    const histogram = getProfileBins(summary, PROFILE_COLUMN.value);
    const bivariateCountsList = Array.from(
      summary.bivariateCounts.subarray(0, BIVARIATE_SIZE ** 2)
    );
    legendData = {
      table,
      counts,
      histogram,
      baseline: baselineTable,
      baselineCounts,
      bivariate: {
        edgesX: Array.from(summary.bivariateBreaksX.subarray(0, BIVARIATE_SIZE + 1)),
        edgesY: Array.from(summary.bivariateBreaksY.subarray(0, BIVARIATE_SIZE + 1)),
        counts: bivariateCountsList
      },
      cut: options.lowerPercentile > 0 || options.upperPercentile < 100
    };
    pushLegend();

    // Readouts the story cites.
    const valid = summary.validCount;
    const edges = getLegendEdges(summary, classCount);
    const maximum = sortedX.length ? sortedX[sortedX.length - 1] : Number.NaN;
    ctx.setReadout('lowestShare', valid ? formatPercent(counts[0] / valid, 1) : null);
    ctx.setReadout('outsideLowest', `${formatCount(valid - counts[0])} counties`);
    ctx.setReadout('largest', `${formatClassValue(maximum)} ${variable.unit}`);
    ctx.setReadout('classEdges', edges.map(formatClassValue).join(' | '));
    ctx.setReadout('gadf', gadf.toFixed(2));
    ctx.setReadout('gvf', gvf.toFixed(2));
    ctx.setReadout(
      'meanSd',
      `${formatClassValue(statistic(summary, PROFILE_COLUMN.value, 'mean'))} ${variable.unit}, SD ${formatClassValue(statistic(summary, PROFILE_COLUMN.value, 'standardDeviation'))}`
    );
    const middle = (classCount - 1) / 2;
    ctx.setReadout(
      'neutralShare',
      isDivergingMethod(getEffectiveMethod()) && Number.isInteger(middle) && valid
        ? formatPercent(counts[middle] / valid, 0)
        : null
    );
    const usage = getBinUsage(
      sortedX,
      summary.filterBounds[0],
      summary.filterBounds[1],
      activeNaturalBins
    );
    ctx.setReadout('binsUsed', `${formatCount(usage.used)} of ${formatCount(activeNaturalBins)}`);
    ctx.setReadout('firstBinShare', formatPercent(usage.firstShare, 0));

    // Area and population of the darkest class: counts on unequal areas map size.
    let totalArea = 0;
    let totalPopulation = 0;
    let topArea = 0;
    let topPopulation = 0;
    for (let row = 0; row < n; row++) {
      const classIndex = summary.classIndices[row];
      if (classIndex === NO_CLASS) continue;
      totalArea += landArea[row];
      totalPopulation += population[row];
      if (classIndex === classCount - 1) {
        topArea += landArea[row];
        topPopulation += population[row];
      }
    }
    ctx.setReadout('topAreaShare', totalArea ? formatPercent(topArea / totalArea, 1) : null);
    ctx.setReadout(
      'topPopulationShare',
      totalPopulation ? formatPercent(topPopulation / totalPopulation, 1) : null
    );

    // Bivariate cells in palette order: row * 3 + column, column = first variable, row = second.
    const bivariateTotal = bivariateCountsList.reduce((total, count) => total + count, 0);
    const cell = (index: number) =>
      bivariateTotal
        ? `${formatCount(bivariateCountsList[index])} counties (${formatPercent(bivariateCountsList[index] / bivariateTotal, 0)})`
        : null;
    ctx.setReadout(
      'deprived',
      options.bivariate ? cell(BIVARIATE_SIZE * (BIVARIATE_SIZE - 1)) : null
    );
    ctx.setReadout('bothHigh', options.bivariate ? cell(BIVARIATE_SIZE ** 2 - 1) : null);
    let small = 0;
    for (let row = 0; row < n; row++) {
      if (population[row] < ALPHA_POPULATION_DOMAIN[0]) small++;
    }
    ctx.setReadout('smallCounties', `${formatCount(small)} of ${formatCount(n)} counties`);

    publishDistribution(hoverRow >= 0 ? xColumn[hoverRow] : null);
    publishAnnotations();
    publishFurniture(summary);
    ctx.requestLayers();
  }

  function applySummary(summary: Summary, version: number): void {
    // A readback of an earlier parameter state (another variable or method) is stale: the
    // changed parameters already asked for a new one.
    if (version !== parameterVersion) return;
    latestSummary = summary;
    const options = ctx.options;
    const variable = getCountyVariable(options.variable);
    ctx.setReadout('valid', `${formatCount(summary.validCount)} of ${formatCount(n)} counties`);
    ctx.setReadout(
      'filter',
      `${formatClassValue(summary.filterBounds[0])} to ${formatClassValue(summary.filterBounds[1])} ${variable.unit}`
    );
    ctx.setReadout(
      'quartiles',
      `${Array.from(summary.quantiles, formatClassValue).join(' / ')} ${variable.unit}`
    );
    ctx.setReadout(
      'classes',
      `${summary.classCount} produced of ${options.method === 'box-plot' ? 6 : options.classCount} requested`
    );
    const precision = Number(activeHllPrecision);
    const unfiltered = options.lowerPercentile === 0 && options.upperPercentile === 100;
    ctx.setReadout(
      'profile',
      `mean ${formatClassValue(statistic(summary, PROFILE_COLUMN.value, 'mean'))}, sd ${formatClassValue(statistic(summary, PROFILE_COLUMN.value, 'standardDeviation'))}, about ${formatCount(statistic(summary, PROFILE_COLUMN.value, 'distinctEstimate'))} distinct values (HyperLogLog p=${precision}, error about ${(104 / Math.sqrt(2 ** precision)).toFixed(1)}%${unfiltered ? `; exact ${formatCount(exactDistinct)}` : ''})`
    );
    const topCodes: string[] = [];
    const categoryBase = PROFILE_COLUMN.rucc * TOP_CATEGORY_COUNT;
    for (let slot = 0; slot < TOP_CATEGORY_COUNT; slot++) {
      const code = summary.topCategories[categoryBase + slot];
      const count = summary.topCategoryCounts[categoryBase + slot];
      if (code !== NO_CLASS && count > 0) {
        topCodes.push(`${code} ${RUCC_LABELS[code] ?? ''} (${formatCount(count)})`);
      }
    }
    ctx.setReadout('rucc', topCodes.slice(0, 4).join('; ') || 'none');
    const gadf = summary.fit[GPU_CLASSIFICATION_FIT_GADF];
    const gvf = summary.fit[GPU_CLASSIFICATION_FIT_GVF];
    ctx.setReadout(
      'deviations',
      `${formatClassValue(summary.fit[GPU_CLASSIFICATION_FIT_ADCM])} / ${formatClassValue(summary.fit[GPU_CLASSIFICATION_FIT_ADAM])}`
    );

    // Fit scores are only comparable on the same counties and class count.
    const key = `${options.variable}|${options.lowerPercentile}|${options.upperPercentile}|${summary.classCount}|${options.quantileInterpolation}`;
    if (key !== fitKey && !sweep) {
      fitKey = key;
      fitTable.clear();
    }
    if (version === parameterVersion) {
      fitTable.set(getEffectiveMethod(), {gadf, gvf});
      const compared = GPU_CLASS_BREAKS_METHODS.filter(method => fitTable.has(method));
      ctx.setChart(
        'methodFits',
        compared.length > 1
          ? {
              kind: 'bars',
              values: compared.map(method => fitTable.get(method)!.gadf),
              labels: compared.map(method => METHOD_LABELS[method].replace(/ \(.*\)/, '')),
              horizontal: true,
              highlight: [compared.indexOf(getEffectiveMethod())],
              xLabel: 'GADF (higher is better)'
            }
          : null
      );
    }
    // The palette must have as many colours as the method produced classes (head/tail and
    // maximum breaks may return fewer than requested): rewrite it and wait for the next summary.
    const paletteMatches = summary.classCount === writtenClassCount;
    if (summary.classCount > 0 && !paletteMatches) {
      writePalette(summary.classCount);
      writeParameters();
    }
    if (sweep && version === parameterVersion) {
      const next = sweep.queue.shift();
      if (next) {
        sweep.current = next;
        writeParameters();
      } else {
        sweep = null;
        writeParameters();
      }
    }
    if (summary.classCount > 0 && paletteMatches) publishSummary(summary, gadf, gvf);
  }

  async function readSummary(commandEncoder: CommandEncoder): Promise<void> {
    const ticket = readbackRing.tryAcquire();
    if (!ticket) return;
    const versionAtCopy = parameterVersion;
    let offset = 0;
    for (const part of readbackLayout) {
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: part.buffer,
        destinationBuffer: ticket.buffer,
        destinationOffset: offset,
        size: part.byteLength
      });
      offset += part.byteLength;
    }
    ticket.markEncoded({byteOffset: 0, byteLength: readbackByteLength});
    readbackPending = true;
    try {
      const bytes = await ticket.read();
      if (destroyed) return;
      const copy = new ArrayBuffer(bytes.byteLength);
      new Uint8Array(copy).set(bytes);
      const read = createByteReader(copy);
      const summary: Summary = {
        breaks: read.floats(MAXIMUM_CLASS_COUNT + 1),
        classCount: read.words(1)[0],
        classCounts: read.words(MAXIMUM_CLASS_COUNT),
        filterBounds: read.floats(2),
        quantiles: read.floats(QUANTILE_PROBABILITIES.length),
        validCount: read.words(1)[0],
        bivariateCounts: read.words(MAXIMUM_BIVARIATE_CLASS_COUNT ** 2),
        bivariateBreaksX: read.floats(MAXIMUM_BIVARIATE_CLASS_COUNT + 1),
        bivariateBreaksY: read.floats(MAXIMUM_BIVARIATE_CLASS_COUNT + 1),
        profileStatistics: read.floats(PROFILE_COLUMN_COUNT * GPU_COLUMN_PROFILE_STATISTIC_COUNT),
        profileHistograms: read.words(PROFILE_COLUMN_COUNT * MAXIMUM_PROFILE_BINS),
        topCategories: read.words(PROFILE_COLUMN_COUNT * TOP_CATEGORY_COUNT),
        topCategoryCounts: read.words(PROFILE_COLUMN_COUNT * TOP_CATEGORY_COUNT),
        fit: read.floats(GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH),
        classIndices: read.words(n)
      };
      applySummary(summary, versionAtCopy);
    } catch {
      // The ring or device was destroyed while the read was in flight.
    } finally {
      readbackPending = false;
    }
  }

  ruccCodes.write(
    Uint32Array.from(counties.column<Float32Array>('rucc2023'), value =>
      Number.isFinite(value) ? Math.round(value) : NO_CLASS
    )
  );
  loadVariables();
  writePalette(ctx.options.classCount);
  buildGraphs();
  writeParameters();
  ctx.setFurniture({
    title: {
      sample: `${formatCount(n)} counties of the contiguous US (Alaska and Hawaii not in the data)`
    }
  });

  /** The tooltip of one county: value with its class swatch, class, percentile, population. */
  const describeRow = (row: number): TooltipContent => {
    const options = ctx.options;
    const variable = getCountyVariable(options.variable);
    const value = xColumn[row];
    const summary = latestSummary;
    // The class of the classification the map shows: the frozen side while the toggle holds it.
    const table = getDisplayedTable();
    const onBaseline = table !== null && table === baselineTable;
    const rows: TooltipRow[] = [];
    let note: string | undefined;
    if (!Number.isFinite(value)) {
      rows.push({label: variable.label, value: 'No data', emphasis: true});
    } else {
      const classIndex = onBaseline
        ? getClassIndexOf(table, value)
        : (summary?.classIndices[row] ?? NO_CLASS);
      const classified =
        table && summary && classIndex !== NO_CLASS && classIndex < table.colors.length;
      rows.push({
        label: variable.label,
        value: formatClassValue(value),
        unit: variable.unit,
        swatch: classified ? table.colors[classIndex] : undefined,
        emphasis: true
      });
      if (classified) {
        rows.push({
          label: 'Class',
          value: `${classIndex + 1} of ${table.colors.length}`,
          unit: getClassLabel(table, classIndex)
        });
      } else if (summary) {
        note = 'Excluded by the percentile cut';
      }
      rows.push({
        label: 'Rank',
        value: `${formatOrdinal(Math.round(getRankShare(sortedX, value) * 100))} percentile`,
        unit: 'of counties'
      });
    }
    if (options.bivariate) {
      const other = getCountyVariable(options.bivariateVariable);
      rows.push({
        label: other.label,
        value: formatClassValue(yColumn[row]),
        unit: other.unit
      });
    }
    rows.push({label: 'Population', value: formatCount(population[row]), unit: 'residents'});
    return {
      title: getProperty(row, 'name'),
      subtitle: getProperty(row, 'state'),
      rows,
      note,
      anchor: getLabelPoint(row),
      highlight: {kind: 'polygon', rings: geometry.getRings(row)}
    };
  };

  /** Clears the hover marker of the distribution chart and returns no tooltip. */
  const clearHover = (): null => {
    if (hoverRow !== -1) {
      hoverRow = -1;
      publishDistribution(null);
    }
    return null;
  };

  return {
    getCompiledGraphs: () =>
      [main, bivariateGraph, profileGraph, fitGraph].filter(
        (graph): graph is CompiledGPUCommandGraph<void> => graph !== null
      ),

    setOption(id) {
      if (id === 'naturalBreaksBinCount' || id === 'histogramBins' || id === 'hllPrecision') {
        if (
          Number(ctx.options.naturalBreaksBinCount) !== activeNaturalBins ||
          Number(ctx.options.histogramBins) !== activeProfileBins ||
          ctx.options.hllPrecision !== activeHllPrecision
        ) {
          buildGraphs();
          latestSummary = null;
        }
      }
      if (id === 'variable' || id === 'bivariateVariable' || id === 'classCount') {
        latestSummary = null;
      }
      if (id === 'variable' || id === 'method' || id === 'classCount' || id === 'scale') {
        // A manual change ends a method comparison and a legend isolation.
        if (id !== 'scale') sweep = null;
        legendHighlight = null;
      }
      if (id === 'annotationSet') publishAnnotations();
      writeParameters();
      ctx.requestLayers();
    },

    onAction(id) {
      if (id !== 'compareMethods') return;
      fitTable.clear();
      const queue = GPU_CLASS_BREAKS_METHODS.filter(method => method !== 'custom');
      sweep = {queue: queue.slice(1), current: queue[0]};
      writeParameters();
    },

    encode(commandEncoder, frame) {
      if (!main || !bivariateGraph || !profileGraph || !fitGraph) return;
      if (dirtyFrames > 0 || frame.frameIndex < 3) {
        main.encode(commandEncoder, {parameters: undefined});
        if (ctx.options.bivariate) bivariateGraph.encode(commandEncoder, {parameters: undefined});
        profileGraph.encode(commandEncoder, {parameters: undefined});
        fitGraph.encode(commandEncoder, {parameters: undefined});
        dirtyFrames = Math.max(0, dirtyFrames - 1);
      }
      if (
        readbackWanted &&
        !readbackPending &&
        frame.frameIndex - lastReadbackFrame >= READBACK_INTERVAL_FRAMES &&
        dirtyFrames === 0
      ) {
        lastReadbackFrame = frame.frameIndex;
        readbackWanted = false;
        void readSummary(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const ground = ctx.ground();
      const origin: [number, number, number] = [geometry.origin[0], geometry.origin[1], 0];
      const noDataColor = NO_DATA_COLOR[ground];
      const common = {
        coordinateOrigin: origin,
        triangles: geometry.buffers.triangles,
        features: geometry.buffers.triangleFeatures,
        vertexCount: geometry.buffers.vertexCount,
        // Opaque on the paper sheet: translucency would blend the class colours with the ground.
        opacity: 1
      };
      const layers: Layer[] = [];
      const comparing = !options.bivariate && baselineTable !== null;
      if (comparing && baselineTable) {
        // Side a: the frozen classification over the same value buffer, exact class colours.
        layers.push(
          new SpatialAnalysisPolygonLayer({
            id: 'classes-fill-a',
            ...common,
            values: xBuffer,
            valueFormat: 'float32',
            colormap: 'greys',
            ...getClassTableLayerProps(baselineTable),
            noDataColor,
            compareSide: 'a'
          })
        );
      }
      const side = comparing ? ('b' as const) : undefined;
      if (options.bivariate) {
        layers.push(
          new SpatialAnalysisPolygonLayer({
            id: 'classes-fill-b',
            ...common,
            values: bivariateColors,
            valueFormat: 'uint32',
            colormap: 'rgba'
          })
        );
      } else if (legendHighlight && options.scale === 'threshold') {
        // An isolated legend class: the same exact palette over the class indices, others dimmed.
        layers.push(
          new SpatialAnalysisPolygonLayer({
            id: 'classes-fill-b',
            ...common,
            values: classIndices,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: activeColors,
            noDataValue: NO_CLASS,
            noDataColor,
            highlightClasses: legendHighlight,
            compareSide: side
          })
        );
      } else {
        // Side b: the packed rgba8 colours of GPUColorScale, drawn as they are.
        layers.push(
          new SpatialAnalysisPolygonLayer({
            id: 'classes-fill-b',
            ...common,
            values: colors,
            valueFormat: 'uint32',
            colormap: 'rgba',
            compareSide: side
          })
        );
      }
      if (options.outlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'classes-county-lines',
            coordinateOrigin: origin,
            segments: geometry.buffers.outline,
            instanceCount: geometry.buffers.outlineCount,
            // Tier 3: a thin hairline over thousands of polygons, never a dark mesh.
            widthPixels: n > 2000 ? 0.4 : 0.5,
            color: getHairlineColor(ground)
          })
        );
      }
      if (geometry.stateLines) {
        // The zone-boundary tier, always on in county steps.
        const line = getStateLineStyle(ground);
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'classes-state-lines',
            coordinateOrigin: origin,
            segments: geometry.stateLines.buffer,
            instanceCount: geometry.stateLines.segmentCount,
            widthPixels: line.widthPixels,
            color: line.color,
            outlineColor: line.casing,
            outlineWidthPixels: (line.casingPixels - line.widthPixels) / 2
          })
        );
      }
      if (selectedRow >= 0 && selectionSegments > 0) {
        // The pinned county: an achromatic ink core over a ground-colour casing.
        const ink = hexToRgba(MAP_INK[ground].ink);
        const halo = hexToRgba(MAP_INK[ground].halo);
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'classes-selection',
            coordinateOrigin: origin,
            segments: geometry.selectionBuffer,
            instanceCount: selectionSegments,
            widthPixels: 2.5,
            color: ink,
            outlineColor: halo,
            outlineWidthPixels: 1
          })
        );
      }
      return layers;
    },

    // The class tables and the packed colours are authored per ground, so a flip rewrites them.
    onGroundChange() {
      baselineKey = '';
      writeParameters();
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    onCompareChange(state) {
      compareShowing = state.showing;
      pushLegend();
      publishDistribution(hoverRow >= 0 ? xColumn[hoverRow] : null);
    },

    onLegendFilter(_id, classes) {
      legendHighlight = classes ? [...classes] : null;
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return clearHover();
      const hit = geometry.locator.find(event.coordinate);
      if (!hit) return clearHover();
      if (hit.index !== hoverRow) {
        hoverRow = hit.index;
        publishDistribution(xColumn[hit.index]);
      }
      return describeRow(hit.index);
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const hit = geometry.locator.find(event.coordinate);
      const row = hit ? hit.index : -1;
      selectedRow = row === selectedRow ? -1 : row;
      selectionSegments = geometry.setSelection(selectedRow);
      const tooltip = selectedRow >= 0 ? describeRow(selectedRow) : null;
      ctx.setReadout(
        'selected',
        tooltip
          ? `${tooltip.title}, ${tooltip.subtitle}: ${(tooltip.rows ?? []).map(item => `${item.label} ${item.value}${item.unit ? ` ${item.unit}` : ''}`).join(' | ')}`
          : null
      );
      ctx.requestLayers();
      return true;
    },

    destroy() {
      destroyed = true;
      ctx.setAnnotations('classes', null);
      resources.destroy();
    }
  };
}
