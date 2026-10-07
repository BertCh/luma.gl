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
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {addBreaksSelectPass, createGraphImporter} from './b5-classify';
import {
  createByteReader,
  createChoroplethGeometry,
  formatNumber,
  getOutlineColor,
  getQuantile,
  getSortedFinite
} from './b5-common';
import {clearLegendData, formatCompact, formatSparkline, setLegendData} from './b5-legend-bus';
import {CLASS_PALETTES, getStopPalette, packColor} from './b5-palettes';
import {getCountyVariable} from './b5-variables';

/** Option state of the choropleth-classes scene. */
export type ChoroplethClassesOptions = {
  variable: string;
  method: GPUClassBreaksMethod;
  classCount: number;
  scale: 'threshold' | 'quantize' | 'linear' | 'sqrt' | 'pow' | 'log' | 'symlog';
  smoothBlend: boolean;
  clamp: boolean;
  exponent: number;
  logFloorExponent: number;
  palette: string;
  reversePalette: boolean;
  lowerPercentile: number;
  upperPercentile: number;
  quantileInterpolation: 'linear' | 'lower' | 'higher' | 'nearest' | 'midpoint';
  standardDeviationInterval: number;
  headTailRatio: number;
  boxPlotHinge: number;
  naturalBreaksBinCount: string;
  bivariate: boolean;
  bivariateVariable: string;
  bivariateClasses: number;
  bivariateMethod: 'quantile' | 'equal-interval';
  valueByAlpha: boolean;
  minimumAlpha: number;
  noDataColor: 'transparent' | 'gray';
  histogramBins: string;
  hllPrecision: string;
  outlines: boolean;
};

/** Data the legends read. */
export type ClassesLegendData = {
  classCount: number;
  breaks: number[];
  counts: number[];
  colors: number[];
  unit: string;
  methodLabel: string;
  bivariate: {
    n: number;
    counts: number[];
    edgesX: number[];
    edgesY: number[];
    colors: number[];
  } | null;
  fits: {method: string; label: string; gadf: number; gvf: number}[];
  fitClassCount: number;
};

const MAXIMUM_CLASS_COUNT = 9;
const MAXIMUM_BIVARIATE_CLASS_COUNT = 4;
const QUANTILE_PROBABILITIES = [0.25, 0.5, 0.75] as const;
const MAXIMUM_PROFILE_BINS = 96;
const TOP_CATEGORY_COUNT = 10;
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
const NO_DATA_TRANSPARENT = packColor(0, 0, 0, 0);
const NO_DATA_GRAY = packColor(140, 140, 150, 110);
/** Value-by-alpha ramps from 5,000 to 500,000 residents: small, noisy counties fade. */
const ALPHA_POPULATION_DOMAIN: readonly [number, number] = [5000, 500000];

export const METHOD_LABELS: Record<GPUClassBreaksMethod, string> = {
  quantile: 'Quantile',
  'equal-interval': 'Equal interval',
  'standard-deviation': 'Standard deviation',
  'head-tail': 'Head/tail breaks',
  'box-plot': 'Box plot (6 classes)',
  'maximum-breaks': 'Maximum breaks',
  'natural-breaks': 'Natural breaks (Jenks)',
  custom: 'Custom round-number edges'
};

/** Stevens-style bivariate corners: low/low, high X, high Y, high/high. */
const BIVARIATE_CORNERS = {
  lowLow: [232, 232, 232],
  highX: [200, 90, 90],
  highY: [100, 172, 190],
  highHigh: [87, 66, 73]
} as const;

/** Returns the colour of one bivariate class. */
function getBivariateColor(n: number, x: number, y: number): [number, number, number] {
  const tx = n > 1 ? x / (n - 1) : 0;
  const ty = n > 1 ? y / (n - 1) : 0;
  const mix = (channel: 0 | 1 | 2) =>
    Math.round(
      BIVARIATE_CORNERS.lowLow[channel] * (1 - tx) * (1 - ty) +
        BIVARIATE_CORNERS.highX[channel] * tx * (1 - ty) +
        BIVARIATE_CORNERS.highY[channel] * (1 - tx) * ty +
        BIVARIATE_CORNERS.highHigh[channel] * tx * ty
    );
  return [mix(0), mix(1), mix(2)];
}

/** Merges histogram bins down to at most `target` so a sparkline stays short. */
function rebin(bins: ArrayLike<number>, target: number): number[] {
  const group = Math.max(1, Math.ceil(bins.length / target));
  const merged: number[] = [];
  for (let start = 0; start < bins.length; start += group) {
    let total = 0;
    for (let index = start; index < Math.min(start + group, bins.length); index++)
      total += bins[index];
    merged.push(total);
  }
  return merged;
}

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

/**
 * Choropleth classification of US counties, entirely on the GPU: `GPUColumnQuantiles` (percentile
 * filter), `GPUClassBreaks` (8 methods), `GPUColorScale` (scale type and palette), `GPUClassAssignment`
 * and `GPUClassificationFit` (goodness of fit), `GPUBivariateClassification` (two variables and
 * value-by-alpha) and `GPUColumnProfile` (field summary). Every control except the natural-breaks
 * bin count is a parameter or palette write.
 */
export async function createChoroplethClasses(
  ctx: SceneContext<ChoroplethClassesOptions>
): Promise<SceneInstance<ChoroplethClassesOptions>> {
  const counties = ctx.datasets.get('us-counties');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'choropleth-classes');
  const geometry = createChoroplethGeometry(resources, counties);
  const n = geometry.featureCount;
  const features = counties.geojson?.features ?? [];
  const nameOf = (row: number): string => {
    const properties = features[row]?.properties;
    return properties ? `${properties.name}, ${properties.state}` : `County ${row}`;
  };
  const population = counties.column<Float32Array>('population');

  const xBuffer = resources.createBuffer('x-values', n * 4);
  const yBuffer = resources.createBuffer('y-values', n * 4);
  const alphaBuffer = resources.createBuffer('alpha-values', population);
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
  const profileColumnCount = 3;
  const profileStatistics = resources.createBuffer(
    'profile-statistics',
    profileColumnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4
  );
  const profileHistograms = resources.createBuffer(
    'profile-histograms',
    profileColumnCount * MAXIMUM_PROFILE_BINS * 4
  );
  const topCategories = resources.createBuffer(
    'top-categories',
    profileColumnCount * TOP_CATEGORY_COUNT * 4
  );
  const topCategoryCounts = resources.createBuffer(
    'top-category-counts',
    profileColumnCount * TOP_CATEGORY_COUNT * 4
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
      byteLength: profileColumnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT * 4
    },
    {buffer: profileHistograms, byteLength: profileColumnCount * MAXIMUM_PROFILE_BINS * 4},
    {buffer: topCategories, byteLength: profileColumnCount * TOP_CATEGORY_COUNT * 4},
    {buffer: topCategoryCounts, byteLength: profileColumnCount * TOP_CATEGORY_COUNT * 4},
    {buffer: fitSummary, byteLength: GPU_CLASSIFICATION_FIT_SUMMARY_LENGTH * 4},
    {buffer: classIndices, byteLength: n * 4}
  ];
  const readbackByteLength = readbackLayout.reduce((total, part) => total + part.byteLength, 0);
  const readbackRing = resources.track(
    new GPUReadbackRing(device, {id: 'classes-summary', byteLength: readbackByteLength})
  );

  // CPU copies of the selected variables, for the round-number edges, value filters and tooltips.
  let xColumn: Float32Array = new Float32Array(n);
  let yColumn: Float32Array = new Float32Array(n);
  let sortedX: Float64Array = new Float64Array(0);
  let loadedVariable = '';
  let loadedBivariateVariable = '';

  let writtenClassCount = 0;
  let activePalette: Uint32Array = new Uint32Array(0);
  let latestSummary: Summary | null = null;
  let dirtyFrames = ENCODED_FRAMES_AFTER_CHANGE;
  let lastReadbackFrame = -READBACK_INTERVAL_FRAMES;
  let readbackWanted = true;
  let readbackPending = false;
  let destroyed = false;
  let selectedRow = -1;
  let parameterVersion = 0;
  let activeNaturalBins = Number(ctx.options.naturalBreaksBinCount);
  let activeProfileBins = Number(ctx.options.histogramBins);
  let activeHllPrecision = ctx.options.hllPrecision;
  let exactDistinct = 0;

  /** Method comparison sweep state. */
  let sweep: {queue: GPUClassBreaksMethod[]; current: GPUClassBreaksMethod} | null = null;
  const fitTable = new Map<GPUClassBreaksMethod, {gadf: number; gvf: number}>();
  let fitKey = '';

  let main: CompiledGPUCommandGraph<void> | null = null;
  let bivariateGraph: CompiledGPUCommandGraph<void> | null = null;
  let profileGraph: CompiledGPUCommandGraph<void> | null = null;
  let fitGraph: CompiledGPUCommandGraph<void> | null = null;

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
          {kind: 'category', values: profileBind.word(ruccCodes, n), categoryCount: 10}
        ],
        mask: profileBind.word(filterMask, n),
        histogramBinCount: activeProfileBins,
        hyperLogLogPrecision: Number(ctx.options.hllPrecision),
        topCategoryCount: TOP_CATEGORY_COUNT,
        output: {
          statistics: profileBind.float(
            profileStatistics,
            profileColumnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT
          ),
          histograms: profileBind.word(profileHistograms, profileColumnCount * activeProfileBins),
          topCategories: profileBind.word(topCategories, profileColumnCount * TOP_CATEGORY_COUNT),
          topCategoryCounts: profileBind.word(
            topCategoryCounts,
            profileColumnCount * TOP_CATEGORY_COUNT
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
  }

  /** Loads the chosen variables into the value buffers when they changed. */
  function loadVariables(): void {
    const {variable, bivariateVariable} = ctx.options;
    if (variable !== loadedVariable) {
      loadedVariable = variable;
      xColumn = counties.column<Float32Array>(variable);
      xBuffer.write(xColumn);
      const occupied = new Uint32Array(n);
      for (let row = 0; row < n; row++) occupied[row] = Number.isFinite(xColumn[row]) ? 1 : 0;
      occupiedBuffer.write(occupied);
      sortedX = getSortedFinite(xColumn);
      exactDistinct = new Set(sortedX).size;
      latestSummary = null;
    }
    if (bivariateVariable !== loadedBivariateVariable) {
      loadedBivariateVariable = bivariateVariable;
      yColumn = counties.column<Float32Array>(bivariateVariable);
      yBuffer.write(yColumn);
    }
  }

  function writePalette(classCount: number): void {
    writtenClassCount = classCount;
    const stops = (CLASS_PALETTES[ctx.options.palette] ?? CLASS_PALETTES.ylorrd).stops;
    const ordered = ctx.options.reversePalette ? [...stops].reverse() : stops;
    activePalette = getStopPalette(ordered, classCount, 235);
    const padded = new Uint32Array(MAXIMUM_CLASS_COUNT);
    padded.set(activePalette);
    paletteBuffer.write(padded);
  }

  function writeBivariatePalette(): void {
    const palette = new Uint32Array(MAXIMUM_BIVARIATE_CLASS_COUNT ** 2);
    const size = ctx.options.bivariateClasses;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const [r, g, b] = getBivariateColor(size, x, y);
        palette[y * size + x] = packColor(r, g, b, 240);
      }
    }
    bivariatePaletteBuffer.write(palette);
  }

  function getEffectiveMethod(): GPUClassBreaksMethod {
    return sweep ? sweep.current : ctx.options.method;
  }

  /** Rewrites every per-frame parameter buffer from the options. */
  function writeParameters(): void {
    loadVariables();
    const options = ctx.options;
    const method = getEffectiveMethod();
    const requestedClassCount = options.classCount;
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
    const classCount = latestSummary?.classCount || requestedClassCount;
    if (classCount !== writtenClassCount) writePalette(Math.min(classCount, MAXIMUM_CLASS_COUNT));
    scaleParameters.write(
      getGPUColorScaleParameterValues({
        scale: (options.scale === 'threshold' ? 'threshold' : options.scale) as GPUColorScaleType,
        domainCount: writtenClassCount + 1,
        paletteCount: writtenClassCount,
        interpolation: options.smoothBlend && options.scale !== 'threshold' ? 'linear' : 'step',
        clamp: options.clamp,
        noDataColor: options.noDataColor === 'gray' ? NO_DATA_GRAY : NO_DATA_TRANSPARENT,
        logFloor: 10 ** options.logFloorExponent,
        exponent: options.exponent
      })
    );
    const axisSettings = {
      method: options.bivariateMethod,
      classCount: options.bivariateClasses
    };
    axisXParameters.write(
      getGPUClassBreaksParameterValues(axisSettings, MAXIMUM_BIVARIATE_CLASS_COUNT)
    );
    axisYParameters.write(
      getGPUClassBreaksParameterValues(axisSettings, MAXIMUM_BIVARIATE_CLASS_COUNT)
    );
    bivariateParameters.write(
      getGPUBivariateClassificationParameterValues({
        classCountX: options.bivariateClasses,
        classCountY: options.bivariateClasses,
        noDataColor: options.noDataColor === 'gray' ? NO_DATA_GRAY : NO_DATA_TRANSPARENT,
        valueByAlpha: options.valueByAlpha
          ? {domain: ALPHA_POPULATION_DOMAIN, minimumAlpha: options.minimumAlpha}
          : undefined
      })
    );
    writeBivariatePalette();
    parameterVersion++;
    dirtyFrames = ENCODED_FRAMES_AFTER_CHANGE;
    readbackWanted = true;
  }

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

  function publishLegend(summary: Summary): void {
    const k = Math.min(summary.classCount, MAXIMUM_CLASS_COUNT);
    const options = ctx.options;
    const variable = getCountyVariable(options.variable);
    const edges = getLegendEdges(summary, k);
    const n2 = options.bivariateClasses;
    const bivariateColorsList: number[] = [];
    for (let y = 0; y < n2; y++) {
      for (let x = 0; x < n2; x++) {
        const [r, g, b] = getBivariateColor(n2, x, y);
        bivariateColorsList.push(packColor(r, g, b, 255));
      }
    }
    setLegendData<ClassesLegendData>('choropleth-classes', {
      classCount: k,
      breaks: edges,
      counts: Array.from(summary.classCounts.subarray(0, k)),
      colors: Array.from(activePalette.subarray(0, k)),
      unit: variable.unit,
      methodLabel: METHOD_LABELS[getEffectiveMethod()],
      bivariate: {
        n: n2,
        counts: Array.from(summary.bivariateCounts.subarray(0, n2 * n2)),
        edgesX: Array.from(summary.bivariateBreaksX.subarray(0, n2 + 1)),
        edgesY: Array.from(summary.bivariateBreaksY.subarray(0, n2 + 1)),
        colors: bivariateColorsList
      },
      fits: GPU_CLASS_BREAKS_METHODS.filter(method => fitTable.has(method)).map(method => ({
        method,
        label: METHOD_LABELS[method].replace(/ \(.*\)/, ''),
        gadf: fitTable.get(method)!.gadf,
        gvf: fitTable.get(method)!.gvf
      })),
      fitClassCount: k
    });
    ctx.setLegendExtent('classes', [0, parameterVersion]);
  }

  function applySummary(summary: Summary, version: number): void {
    const previousClassCount = latestSummary?.classCount;
    latestSummary = summary;
    const options = ctx.options;
    const variable = getCountyVariable(options.variable);
    const bivariateVariable = getCountyVariable(options.bivariateVariable);
    ctx.setReadout('valid', `${formatCount(summary.validCount)} of ${formatCount(n)} counties`);
    ctx.setReadout(
      'filter',
      `${formatCompact(summary.filterBounds[0])} to ${formatCompact(summary.filterBounds[1])} ${variable.unit}`
    );
    ctx.setReadout(
      'quartiles',
      `${Array.from(summary.quantiles, formatCompact).join(' / ')} ${variable.unit}`
    );
    ctx.setReadout(
      'classes',
      `${summary.classCount} produced of ${options.method === 'box-plot' ? 6 : options.classCount} requested`
    );
    const statistic = (column: number, field: keyof typeof GPU_COLUMN_PROFILE_STATISTIC) =>
      summary.profileStatistics[
        column * GPU_COLUMN_PROFILE_STATISTIC_COUNT + GPU_COLUMN_PROFILE_STATISTIC[field]
      ];
    const precision = Number(activeHllPrecision);
    const unfiltered = options.lowerPercentile === 0 && options.upperPercentile === 100;
    ctx.setReadout(
      'profile',
      `mean ${formatCompact(statistic(0, 'mean'))}, sd ${formatCompact(statistic(0, 'standardDeviation'))}, ~${formatNumber(statistic(0, 'distinctEstimate'))} distinct values (HyperLogLog p=${precision}, error about ${(104 / Math.sqrt(2 ** precision)).toFixed(1)}%${unfiltered ? `; exact ${formatNumber(exactDistinct)}` : ''})`
    );
    const topCodes: string[] = [];
    const categoryBase = 2 * TOP_CATEGORY_COUNT;
    for (let slot = 0; slot < TOP_CATEGORY_COUNT; slot++) {
      const code = summary.topCategories[categoryBase + slot];
      const count = summary.topCategoryCounts[categoryBase + slot];
      if (code !== 0xffffffff && count > 0) {
        topCodes.push(`${code} ${RUCC_LABELS[code] ?? ''} (${formatCount(count)})`);
      }
    }
    ctx.setReadout('rucc', topCodes.slice(0, 4).join('; ') || 'none');
    ctx.setReadout(
      'histogram',
      `${formatCompact(statistic(0, 'minimum'))} ${formatSparkline(rebin(summary.profileHistograms.subarray(0, activeProfileBins), 32))} ${formatCompact(statistic(0, 'maximum'))}`
    );
    ctx.setReadout(
      'histogramY',
      options.bivariate
        ? `${bivariateVariable.label}: ${formatCompact(statistic(1, 'minimum'))} ${formatSparkline(rebin(summary.profileHistograms.subarray(activeProfileBins, 2 * activeProfileBins), 32))} ${formatCompact(statistic(1, 'maximum'))}`
        : 'bivariate off'
    );
    const gadf = summary.fit[GPU_CLASSIFICATION_FIT_GADF];
    const gvf = summary.fit[GPU_CLASSIFICATION_FIT_GVF];
    ctx.setReadout('fit', `${gadf.toFixed(3)} / ${gvf.toFixed(3)}`);
    ctx.setReadout(
      'deviations',
      `${formatCompact(summary.fit[GPU_CLASSIFICATION_FIT_ADCM])} / ${formatCompact(summary.fit[GPU_CLASSIFICATION_FIT_ADAM])}`
    );

    // Fit scores are only comparable on the same counties and class count.
    const key = `${options.variable}|${options.lowerPercentile}|${options.upperPercentile}|${summary.classCount}|${options.quantileInterpolation}`;
    if (key !== fitKey && !sweep) {
      fitKey = key;
      fitTable.clear();
    }
    if (version === parameterVersion) {
      fitTable.set(getEffectiveMethod(), {gadf, gvf});
      ctx.setReadout(
        'methodFits',
        fitTable.size > 1
          ? GPU_CLASS_BREAKS_METHODS.filter(method => fitTable.has(method))
              .map(
                method =>
                  `${METHOD_LABELS[method].split(' (')[0]} ${fitTable.get(method)!.gadf.toFixed(2)}`
              )
              .join(' · ')
          : 'press "Compare all methods"'
      );
    }
    if (summary.classCount > 0 && summary.classCount !== previousClassCount) {
      if (summary.classCount !== writtenClassCount) {
        writePalette(summary.classCount);
        writeParameters();
      }
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
    publishLegend(summary);
    ctx.requestLayers();
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
        profileStatistics: read.floats(profileColumnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT),
        profileHistograms: read.words(profileColumnCount * MAXIMUM_PROFILE_BINS),
        topCategories: read.words(profileColumnCount * TOP_CATEGORY_COUNT),
        topCategoryCounts: read.words(profileColumnCount * TOP_CATEGORY_COUNT),
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
      Number.isFinite(value) ? Math.round(value) : 0xffffffff
    )
  );
  writePalette(ctx.options.classCount);
  loadVariables();
  buildGraphs();
  writeParameters();

  const describeRow = (row: number): string => {
    const options = ctx.options;
    const variable = getCountyVariable(options.variable);
    const lines = [
      nameOf(row),
      `${variable.label}: ${formatCompact(xColumn[row])} ${variable.unit}`
    ];
    if (latestSummary) {
      const classIndex = latestSummary.classIndices[row];
      if (classIndex !== 0xffffffff && classIndex < latestSummary.classCount) {
        lines.push(
          `Class ${classIndex + 1} of ${latestSummary.classCount} (${formatCompact(latestSummary.breaks[classIndex])} to ${formatCompact(latestSummary.breaks[classIndex + 1])})`
        );
      } else {
        lines.push('Filtered out by the percentile range');
      }
    }
    if (options.bivariate) {
      lines.push(
        `${getCountyVariable(options.bivariateVariable).label}: ${formatCompact(yColumn[row])} ${getCountyVariable(options.bivariateVariable).unit}`
      );
    }
    lines.push(`Population ${formatNumber(population[row])}`);
    return lines.join('\n');
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
      if (id === 'palette' || id === 'reversePalette')
        writePalette(writtenClassCount || ctx.options.classCount);
      if (id === 'method' || id === 'classCount' || id === 'variable') {
        // A manual change ends a method comparison.
        sweep = null;
      }
      if (id === 'outlines' || id === 'bivariate') ctx.requestLayers();
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
      const layers: Layer[] = [
        geometry.createFillLayer(`classes-${ctx.options.bivariate ? 'bivariate' : 'classes'}`, {
          values: ctx.options.bivariate ? bivariateColors : colors,
          mode: 'packed',
          selectedRow,
          fillOpacity: 0.92
        })
      ];
      if (ctx.options.outlines) {
        layers.push(
          geometry.createOutlineLayer('classes-outline', getOutlineColor(ctx.theme(), 80), 0.7)
        );
      }
      return layers;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const row = geometry.locator.locate(event.coordinate[0], event.coordinate[1]);
      return row >= 0 ? describeRow(row) : null;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const row = geometry.locator.locate(event.coordinate[0], event.coordinate[1]);
      selectedRow = row === selectedRow ? -1 : row;
      ctx.setReadout(
        'selected',
        selectedRow >= 0 ? describeRow(selectedRow).replace(/\n/g, ' | ') : null
      );
      ctx.requestLayers();
      return true;
    },

    destroy() {
      destroyed = true;
      clearLegendData('choropleth-classes');
      resources.destroy();
    }
  };
}
