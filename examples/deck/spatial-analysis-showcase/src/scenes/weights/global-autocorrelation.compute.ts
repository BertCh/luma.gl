// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  getGPUSpatialCorrelogramParameterValues,
  GPU_SPATIAL_CORRELOGRAM_NO_BAND,
  GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
  GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH,
  GPUSpatialCorrelogram,
  type GPUSpatialCorrelogramBandMode,
  type GPUSpatialCorrelogramVarianceAssumption
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUPermutationParameterValues,
  GPU_GLOBAL_JOIN_COUNT_FIELD,
  GPU_GLOBAL_PERMUTATION_RESULT,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPUGlobalPermutationTest,
  GPUGlobalSpatialStatistics,
  GPUSpatialLag,
  type GPUGlobalPermutationStatistic,
  type GPUParameterBuffer,
  type GPUPermutationAlternative
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getClassBreaks, getClassCounts, getExtent} from '../../cartography/breaks';
import {getClassIndexOf, getClassTableLayerProps} from '../../cartography/class-table';
import {CHICAGO, labelsFor} from '../../cartography/gazetteer';
import {NO_DATA_COLOR} from '../../cartography/hue-registry';
import {
  formatCount,
  formatDistance,
  formatOrdinal,
  formatPercent
} from '../../cartography/live-text';
import {getInputPolygons} from '../../cartography/picking';
import {mercatorCaveat} from '../../cartography/projection-notes';
import {getLocalProjector, projectRingsToSegments} from '../../cartography/segments';
import type {ClassTable, LngLat, MapAnnotation, MapHighlight} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor,
  type SpatialAnalysisStyleProps
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {LocalMetricProjection} from '../../engine/projection';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {formatNumber, formatPValue, getPercentile, runGuarded} from './b4-format';
import {
  getVariableInfo,
  loadGeography,
  type Geography,
  type GeographyId,
  type VariableId
} from './b4-geography';
import {createGeographyBuffers, type GeographyBuffers} from './b4-layers';
import {
  createWeightsCore,
  type WeightsConfig,
  type WeightsCore,
  type WeightsSource,
  type WeightsTransform
} from './b4-weights-core';
import {
  buildMoranScatter,
  formatPseudoPValue,
  getBivariateClasses,
  getGroundDistanceRatio,
  getRank,
  getTertileBreaks,
  NO_CLASS,
  rebinHistogram,
  shuffleValues,
  type MoranScatter
} from './global-autocorrelation.stats';
import {
  BLACK_JOIN,
  CLASS_COUNT,
  formatVariableValue,
  getBivariateColors,
  getQuadrantColors,
  getQuadrantPalette,
  getScatterPalette,
  getVariableClassTable,
  QUADRANT_LEGEND_TO_CODE,
  WHITE_JOIN
} from './global-autocorrelation.style';
import {getHairlineColor, getStateLineStyle} from './hot-spots.style';

/** Statistic the permutation test, map and readouts concentrate on. */
export type GlobalStatistic = 'moran' | 'geary' | 'getisOrdG' | 'bivariateMoran' | 'joinCount';

/** Option state of the global-autocorrelation scene. */
export type GlobalAutocorrelationOptions = {
  geography: GeographyId;
  variable: VariableId;
  secondVariable: VariableId;
  source: WeightsSource;
  k: number;
  bandFactor: number;
  transform: WeightsTransform;
  /** Compares the four neighbour rules on a second weights set (step "Does the neighbour rule matter?"). */
  showRuleSweep: boolean;
  statistic: GlobalStatistic;
  alternative: GPUPermutationAlternative;
  permutations: number;
  seed: number;
  /** Draws a seeded shuffle of x left of the swipe divider. */
  showShuffled: boolean;
  joinPercentile: number;
  bandCount: number;
  bandMode: GPUSpatialCorrelogramBandMode;
  maxDistanceFactor: number;
  varianceAssumption: GPUSpatialCorrelogramVarianceAssumption;
  display: 'value' | 'second' | 'lag' | 'quadrant' | 'bivariate' | 'binary';
  showBands: boolean;
  showOutlines: boolean;
};

const MAXIMUM_PERMUTATIONS = 999;
const HISTOGRAM_BINS = 24;
const MAXIMUM_BANDS = 32;
const MAXIMUM_SEED = 50;
const STATISTICS_LENGTH = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length;
const PERMUTATION_RESULT_LENGTH = GPU_GLOBAL_PERMUTATION_RESULT.length;
/** Polygons drawn by a chart brush: more would be an unreadable tangle of outlines. */
const MAXIMUM_BRUSHED_OUTLINES = 300;
const FOCUS_POINTS: Record<GeographyId, [number, number]> = {
  'us-counties': [-87.65, 41.84],
  'chicago-tracts': [-87.63, 41.88]
};
const GEOGRAPHY_BOUNDS: Record<GeographyId, [number, number, number, number]> = {
  'us-counties': [-124.8, 24.4, -66.9, 49.4],
  'chicago-tracts': [-87.94, 41.644, -87.524, 42.023]
};
/** The four rules of the neighbour comparison, in the order of the bars. */
const RULES: readonly WeightsSource[] = ['queen', 'rook', 'knn', 'band'];
const QUADRANT_NAMES = ['High-High', 'Low-High', 'Low-Low', 'High-Low'] as const;

const CLASSIFY_BODY = /* wgsl */ `
  let included = mask[maskOffset + index] != 0u;
  let value = x[xOffset + index];
  let lagged = lag[lagOffset + index];
  if (!included) {
    quadrant[quadrantOffset + index] = 0u;
    joinClass[joinClassOffset + index] = 0xffffffffu;
    var nanBits = 0x7fc00000u;
    binary[binaryOffset + index] = bitcast<f32>(nanBits);
    return;
  }
  // Moran scatterplot quadrant of (x - mean, lag - mean): 1 HH, 2 LH, 3 LL, 4 HL.
  let mean = summary[summaryOffset + ${GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY.mean}u];
  let high = value >= mean;
  let neighborsHigh = lagged >= mean;
  var quadrantValue = 3u;
  if (high && neighborsHigh) { quadrantValue = 1u; }
  else if (high) { quadrantValue = 4u; }
  else if (neighborsHigh) { quadrantValue = 2u; }
  quadrant[quadrantOffset + index] = quadrantValue;
  let black = value > threshold[thresholdOffset];
  joinClass[joinClassOffset + index] = select(0u, 1u, black);
  binary[binaryOffset + index] = select(0.0, 1.0, black);`;

// A county with no neighbour (an island) or no value has no quadrant and no neighbour average:
// the lag the contributor writes for it is a 0 that means nothing, so it becomes NaN (no data).
const ISLAND_BODY = /* wgsl */ `
  let isolated = rowOffsets[rowOffsetsOffset + index + 1u] == rowOffsets[rowOffsetsOffset + index];
  if (isolated || mask[maskOffset + index] == 0u) {
    quadrant[quadrantOffset + index] = 0u;
    var nanBits = 0x7fc00000u;
    lag[lagOffset + index] = bitcast<f32>(nanBits);
  }`;

/** The neighbour-rule comparison: a second weights set and a one-pass Moran graph, run rule by rule. */
type Sweep = {
  core: WeightsCore;
  graph: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  /** Moran's I per rule, in {@link RULES} order. */
  values: number[];
  /** Rule being computed. */
  index: number;
  state: 'idle' | 'running' | 'done';
  /** The rule `index` was encoded and its read is outstanding. */
  encoded: boolean;
};

/** Frozen quantile classes of one variable (computed once, shared by every map and toggle). */
type FrozenClasses = {breaks: number[]; extent: [number, number]};

type World = {
  geography: Geography;
  projection: LocalMetricProjection;
  resources: SpatialAnalysisResources;
  core: WeightsCore;
  buffers: GeographyBuffers;
  stateSegments: Float32Array | null;
  stateBuffer: Buffer | null;
  x: Buffer;
  y: Buffer;
  maskX: Buffer;
  maskY: Buffer;
  maskXY: Buffer;
  threshold: Buffer;
  lag: Buffer;
  quadrant: Buffer;
  joinClass: Buffer;
  shuffled: Buffer;
  bivariateClass: Buffer;
  /** CPU copy of the bivariate classes, for tooltips. */
  bivariateClasses: Uint32Array;
  focusRow: number;
  statsCompiled: CompiledGPUCommandGraph<void>;
  permutation: Map<string, CompiledGPUCommandGraph<void>>;
  correlogram: Map<string, CompiledGPUCommandGraph<void>>;
  permutationParameters: GPUParameterBuffer<'uint32'>;
  correlogramParameters: GPUParameterBuffer<'float32'>;
  buffersByName: Record<string, Buffer>;
  reader: SummaryReader;
  snapshot: Snapshot | null;
  classes: Map<VariableId, FrozenClasses>;
  /** Ascending finite values of x, for tooltip ranks. */
  sortedX: Float32Array;
  /** Bivariate class counts (nine cells) of x and y. */
  bivariateCounts: number[];
  sweep: Sweep | null;
};

type Snapshot = {
  global: Float32Array;
  bivariate: Float32Array;
  join: Float32Array;
  joinCounts: Uint32Array;
  permutation: Float32Array;
  histogram: Uint32Array;
  moransI: Float32Array;
  zScores: Float32Array;
  pValues: Float32Array;
  pairCounts: Uint32Array;
  peaks: Uint32Array;
  statistics: Float32Array;
  shuffled: Float32Array;
  lag: Float32Array;
  offsets: Uint32Array;
  scatter: MoranScatter;
};

const geographyCache = new Map<GeographyId, Promise<Geography>>();

/**
 * Global spatial autocorrelation of health and income. One `GPUGlobalSpatialStatistics` pass per
 * variable family computes Moran's I, Geary's C, Getis-Ord G, bivariate Moran's I and join counts
 * with analytic z and p; `GPUSpatialLag` gives the neighbour average that makes the Moran
 * scatterplot; `GPUGlobalPermutationTest` builds the null distribution; and
 * `GPUSpatialCorrelogram` repeats Moran's I over a ladder of distance bands. A second weights
 * set runs the four neighbour rules one after the other for the comparison step.
 */
export async function createGlobalAutocorrelation(
  ctx: SceneContext<GlobalAutocorrelationOptions>
): Promise<SceneInstance<GlobalAutocorrelationOptions>> {
  const {device} = ctx;
  let destroyed = false;
  let world: World | null = null;
  let switchToken = 0;
  let dirty = true;
  let weightsDirty = true;
  let stale = true;
  /** Class (or entry) indices an interactive legend isolates; `null` shows everything. */
  let legendHighlight: number[] | null = null;

  const getGeography = (id: GeographyId) => {
    let promise = geographyCache.get(id);
    if (!promise) {
      promise = loadGeography(id, ctx.datasets, ctx.signal);
      promise.catch(() => geographyCache.delete(id));
      geographyCache.set(id, promise);
    }
    return promise;
  };

  const getConfig = (): WeightsConfig => {
    const o = ctx.options;
    return {
      source: o.source,
      k: o.k,
      snapTolerance: 0,
      bandFactor: o.bandFactor,
      knnCapFactor: 0,
      weightKind: 'binary',
      kernel: 'triangular',
      power: 1,
      distanceFloor: 0,
      rowStandardize: false,
      transform: o.transform,
      transformKernel: 'bisquare',
      bandwidthFactor: 0,
      doubleSum: 'one'
    };
  };

  /** The configuration of one rule of the comparison: row standardised, with the current k and band. */
  const getRuleConfig = (rule: WeightsSource): WeightsConfig => ({
    ...getConfig(),
    source: rule,
    transform: 'row'
  });

  const getMaxDistance = (geography: Geography) =>
    ctx.options.maxDistanceFactor * geography.medianSpacing;

  // ---------------------------------------------------------------------------------------
  // The world: buffers and graphs of one geography
  // ---------------------------------------------------------------------------------------
  const buildWorld = (geography: Geography): World => {
    const resources = new SpatialAnalysisResources(device, `global-${geography.id}`);
    const rowCount = geography.count;
    const create = (name: string, data: number | Float32Array | Uint32Array) =>
      resources.createBuffer(name, data);
    const core = createWeightsCore({device, resources, id: 'global-core', geography});
    const o = ctx.options;
    const buffersByName = {
      globalResults: create('global-results', STATISTICS_LENGTH * 4),
      bivariateResults: create('bivariate-results', STATISTICS_LENGTH * 4),
      joinResults: create('join-results', STATISTICS_LENGTH * 4),
      shuffledResults: create('shuffled-results', STATISTICS_LENGTH * 4),
      joinCounts: create('join-counts', 16),
      permutationResults: create('permutation-results', PERMUTATION_RESULT_LENGTH * 4),
      histogram: create('permutation-histogram', HISTOGRAM_BINS * 4),
      moransI: create('correlogram-i', MAXIMUM_BANDS * 4),
      zScores: create('correlogram-z', MAXIMUM_BANDS * 4),
      pValues: create('correlogram-p', MAXIMUM_BANDS * 4),
      expectedI: create('correlogram-expected', MAXIMUM_BANDS * 4),
      pairCounts: create('correlogram-pairs', MAXIMUM_BANDS * 4),
      peaks: create('correlogram-peaks', 8),
      correlogramStatistics: create(
        'correlogram-statistics',
        GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH * 4
      ),
      binary: create('binary-values', rowCount * 4)
    };
    const x = create('x', Float32Array.from(geography.getVariable(o.variable)));
    const y = create('y', Float32Array.from(geography.getVariable(o.secondVariable)));
    const maskX = create('mask-x', new Uint32Array(rowCount));
    const maskY = create('mask-y', new Uint32Array(rowCount));
    const maskXY = create('mask-xy', new Uint32Array(rowCount));
    const threshold = create('join-threshold', new Float32Array(1));
    const lag = create('lag', rowCount * 4);
    const quadrant = create('quadrant', rowCount * 4);
    const joinClass = create('join-class', rowCount * 4);
    const shuffled = create('shuffled-x', rowCount * 4);
    const bivariateClass = create('bivariate-class', rowCount * 4);
    const permutationParameters = resources.createParameterBuffer(
      'permutation-parameters',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const correlogramParameters = resources.createParameterBuffer(
      'correlogram-parameters',
      'float32',
      GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH
    );
    const focusRow = Math.max(0, geography.pick(...FOCUS_POINTS[geography.id]));

    // The zone-boundary tier of the counties: state lines from the `us-states` dataset, projected
    // around the counties' origin so they sit exactly on the county edges.
    const statesGeojson =
      geography.id === 'us-counties' ? ctx.datasets.get('us-states').geojson : null;
    const stateSegments = statesGeojson
      ? projectRingsToSegments(
          getInputPolygons(statesGeojson).flatMap(({polygon}) => polygon),
          getLocalProjector(geography.origin)
        )
      : null;

    // One graph: lag, classes, every analytic statistic (plus Moran's I of the shuffled map).
    const graph = new GPUCommandGraph<void>(device, {id: `global-statistics-${geography.id}`});
    const weights = {
      offsets: importGraphBuffer(graph, 'offsets', core.csr.offsets, 'uint32', rowCount + 1),
      neighbors: importGraphBuffer(graph, 'neighbors', core.csr.neighbors, 'uint32', core.slots),
      weights: importGraphBuffer(graph, 'weights', core.csr.weights, 'float32', core.slots)
    };
    const xView = importGraphBuffer(graph, 'x', x, 'float32', rowCount);
    const yView = importGraphBuffer(graph, 'y', y, 'float32', rowCount);
    const maskXView = importGraphBuffer(graph, 'mask-x', maskX, 'uint32', rowCount);
    const maskXYView = importGraphBuffer(graph, 'mask-xy', maskXY, 'uint32', rowCount);
    const globalResults = importGraphBuffer(
      graph,
      'global-results',
      buffersByName.globalResults,
      'float32',
      STATISTICS_LENGTH
    );
    graph.add(
      new GPUGlobalSpatialStatistics({
        id: 'global-moran-geary-g',
        weights,
        values: xView,
        mask: maskXView,
        statistics: ['moran', 'geary', 'getisOrdG'],
        results: globalResults
      })
    );
    graph.add(
      new GPUGlobalSpatialStatistics({
        id: 'global-bivariate',
        weights,
        values: xView,
        secondValues: yView,
        mask: maskXYView,
        statistics: ['bivariateMoran'],
        results: importGraphBuffer(
          graph,
          'bivariate-results',
          buffersByName.bivariateResults,
          'float32',
          STATISTICS_LENGTH
        )
      })
    );
    graph.add(
      new GPUGlobalSpatialStatistics({
        id: 'global-shuffled',
        weights,
        values: importGraphBuffer(graph, 'shuffled-x', shuffled, 'float32', rowCount),
        mask: maskXView,
        statistics: ['moran'],
        results: importGraphBuffer(
          graph,
          'shuffled-results',
          buffersByName.shuffledResults,
          'float32',
          STATISTICS_LENGTH
        )
      })
    );
    const lagView = importGraphBuffer(graph, 'lag', lag, 'float32', rowCount);
    graph.add(
      new GPUSpatialLag({
        id: 'spatial-lag',
        values: xView,
        mask: maskXView,
        weights,
        normalize: true,
        output: lagView
      })
    );
    const binaryView = importGraphBuffer(
      graph,
      'binary',
      buffersByName.binary,
      'float32',
      rowCount
    );
    const quadrantView = importGraphBuffer(graph, 'quadrant', quadrant, 'uint32', rowCount);
    addKernelPass(graph, {
      id: 'global-classes',
      invocationCount: rowCount,
      bindings: [
        {name: 'x', view: xView, type: 'f32', access: 'read'},
        {name: 'lag', view: lagView, type: 'f32', access: 'read'},
        {name: 'mask', view: maskXView, type: 'u32', access: 'read'},
        {name: 'summary', view: globalResults, type: 'f32', access: 'read'},
        {
          name: 'threshold',
          view: importGraphBuffer(graph, 'threshold', threshold, 'float32', 1),
          type: 'f32',
          access: 'read'
        },
        {name: 'quadrant', view: quadrantView, type: 'u32', access: 'read_write'},
        {
          name: 'joinClass',
          view: importGraphBuffer(graph, 'join-class', joinClass, 'uint32', rowCount),
          type: 'u32',
          access: 'read_write'
        },
        {name: 'binary', view: binaryView, type: 'f32', access: 'read_write'}
      ],
      body: CLASSIFY_BODY
    });
    addKernelPass(graph, {
      id: 'global-islands',
      invocationCount: rowCount,
      bindings: [
        {name: 'rowOffsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'mask', view: maskXView, type: 'u32', access: 'read'},
        {name: 'quadrant', view: quadrantView, type: 'u32', access: 'read_write'},
        {name: 'lag', view: lagView, type: 'f32', access: 'read_write'}
      ],
      body: ISLAND_BODY
    });
    graph.add(
      new GPUGlobalSpatialStatistics({
        id: 'global-join-count',
        weights,
        values: binaryView,
        mask: maskXView,
        statistics: ['joinCount'],
        results: importGraphBuffer(
          graph,
          'join-results',
          buffersByName.joinResults,
          'float32',
          STATISTICS_LENGTH
        ),
        joinCounts: importGraphBuffer(graph, 'join-counts', buffersByName.joinCounts, 'uint32', 3)
      })
    );
    const statsCompiled = resources.track(graph.compile());

    const reader = new SummaryReader(
      resources,
      `global-${geography.id}`,
      [
        {buffer: buffersByName.globalResults, size: STATISTICS_LENGTH * 4},
        {buffer: buffersByName.bivariateResults, size: STATISTICS_LENGTH * 4},
        {buffer: buffersByName.joinResults, size: STATISTICS_LENGTH * 4},
        {buffer: buffersByName.joinCounts, size: 16},
        {buffer: buffersByName.permutationResults, size: PERMUTATION_RESULT_LENGTH * 4},
        {buffer: buffersByName.histogram, size: HISTOGRAM_BINS * 4},
        {buffer: buffersByName.moransI, size: MAXIMUM_BANDS * 4},
        {buffer: buffersByName.zScores, size: MAXIMUM_BANDS * 4},
        {buffer: buffersByName.pValues, size: MAXIMUM_BANDS * 4},
        {buffer: buffersByName.pairCounts, size: MAXIMUM_BANDS * 4},
        {buffer: buffersByName.peaks, size: 8},
        {
          buffer: buffersByName.correlogramStatistics,
          size: GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH * 4
        },
        {buffer: buffersByName.shuffledResults, size: STATISTICS_LENGTH * 4},
        {buffer: lag, size: rowCount * 4},
        {buffer: core.csr.offsets, size: (rowCount + 1) * 4}
      ],
      bytes => runGuarded('global snapshot', () => handleSnapshot(bytes))
    );

    return {
      geography,
      projection: new LocalMetricProjection(geography.origin),
      resources,
      core,
      buffers: createGeographyBuffers(resources, geography, 'global'),
      stateSegments,
      stateBuffer: stateSegments?.length ? create('state-lines', stateSegments) : null,
      x,
      y,
      maskX,
      maskY,
      maskXY,
      threshold,
      lag,
      quadrant,
      joinClass,
      shuffled,
      bivariateClass,
      bivariateClasses: new Uint32Array(rowCount).fill(NO_CLASS),
      focusRow,
      statsCompiled,
      permutation: new Map(),
      correlogram: new Map(),
      permutationParameters,
      correlogramParameters,
      buffersByName,
      reader,
      snapshot: null,
      classes: new Map(),
      sortedX: new Float32Array(0),
      bivariateCounts: new Array<number>(9).fill(0),
      sweep: null
    };
  };

  // ---------------------------------------------------------------------------------------
  // Variables, classes and the CPU-derived buffers
  // ---------------------------------------------------------------------------------------

  /** Quantile breaks of a variable, computed once and frozen for every toggle of the story. */
  const getFrozenClasses = (target: World, variable: VariableId): FrozenClasses => {
    let frozen = target.classes.get(variable);
    if (!frozen) {
      const values = target.geography.getVariable(variable);
      frozen = {
        breaks: getClassBreaks(values, CLASS_COUNT, 'quantile'),
        extent: getExtent(values)
      };
      target.classes.set(variable, frozen);
    }
    return frozen;
  };

  const getTable = (target: World, variable: VariableId): ClassTable => {
    const frozen = getFrozenClasses(target, variable);
    return getVariableClassTable(
      getVariableInfo(variable),
      target.geography.id,
      frozen.breaks,
      frozen.extent,
      ctx.ground()
    );
  };

  /** The class tables, colours and counts the legends read (rebuilt on a ground flip). */
  const publishTables = () => {
    const target = world;
    if (!target) return;
    const ground = ctx.ground();
    const o = ctx.options;
    const xValues = target.geography.getVariable(o.variable);
    const yValues = target.geography.getVariable(o.secondVariable);
    const valueTable = getTable(target, o.variable);
    const secondTable = getTable(target, o.secondVariable);
    ctx.setLegendData('valueTable', valueTable);
    ctx.setLegendData('secondTable', secondTable);
    ctx.setLegendData('valueCounts', getClassCounts(xValues, valueTable.breaks));
    ctx.setLegendData('secondCounts', getClassCounts(yValues, secondTable.breaks));
    ctx.setLegendData('bivariateColors', getBivariateColors(ground));
    ctx.setLegendData('quadrantColors', getQuadrantColors(ground));
    ctx.setLegendData('noDataColor', NO_DATA_COLOR[ground]);
  };

  /** Seeded shuffle of x, written to the buffer the shuffled map and its Moran pass read. */
  const writeShuffled = (target: World) => {
    const values = target.geography.getVariable(ctx.options.variable);
    target.shuffled.write(shuffleValues(values, ctx.options.seed));
  };

  /** Writes values, masks, the join threshold, bivariate classes and the shuffle of the current variables. */
  const writeVariables = (target: World) => {
    const o = ctx.options;
    const xValues = target.geography.getVariable(o.variable);
    const yValues = target.geography.getVariable(o.secondVariable);
    target.x.write(Float32Array.from(xValues));
    target.y.write(Float32Array.from(yValues));
    const finite = (value: number) => Number.isFinite(value);
    target.maskX.write(Uint32Array.from(xValues, value => (finite(value) ? 1 : 0)));
    target.maskY.write(Uint32Array.from(yValues, value => (finite(value) ? 1 : 0)));
    target.maskXY.write(
      Uint32Array.from(xValues, (value, index) => (finite(value) && finite(yValues[index]) ? 1 : 0))
    );
    target.threshold.write(Float32Array.of(getPercentile(xValues, o.joinPercentile)));
    // Income is inverted so that the dark corner of the bivariate key is the unfavourable one.
    const classes = getBivariateClasses(
      xValues,
      yValues,
      getTertileBreaks(xValues),
      getTertileBreaks(yValues),
      o.secondVariable === 'income'
    );
    target.bivariateClass.write(classes);
    target.bivariateClasses = classes;
    target.bivariateCounts = new Array<number>(9).fill(0);
    for (const value of classes) if (value !== NO_CLASS) target.bivariateCounts[value]++;
    target.sortedX = Float32Array.from(xValues.filter(value => finite(value))).sort();
    writeShuffled(target);
    publishTables();
  };

  const writeParameters = () => {
    const target = world;
    if (!target) return;
    const o = ctx.options;
    target.core.writeParameters(getConfig());
    target.permutationParameters.write(
      getGPUPermutationParameterValues({seed: o.seed, permutations: o.permutations})
    );
    const bounds = target.geography.bounds;
    target.correlogramParameters.write(
      getGPUSpatialCorrelogramParameterValues({
        bounds: [bounds[0] - 1000, bounds[1] - 1000, bounds[2] + 1000, bounds[3] + 1000],
        maximumDistance: getMaxDistance(target.geography),
        varianceAssumption: o.varianceAssumption
      })
    );
  };

  const getPermutationKey = () => {
    const o = ctx.options;
    return o.statistic === 'joinCount' ? null : `${o.statistic}-${o.alternative}`;
  };

  const preparePermutation = (target: World): boolean => {
    const key = getPermutationKey();
    if (!key || target.permutation.has(key)) return false;
    const o = ctx.options;
    const statistic = o.statistic as GPUGlobalPermutationStatistic;
    const rowCount = target.geography.count;
    const graph = new GPUCommandGraph<void>(device, {id: `global-permutation-${key}`});
    const bivariate = statistic === 'bivariateMoran';
    graph.add(
      new GPUGlobalPermutationTest({
        id: 'global-permutation',
        weights: {
          offsets: importGraphBuffer(
            graph,
            'offsets',
            target.core.csr.offsets,
            'uint32',
            rowCount + 1
          ),
          neighbors: importGraphBuffer(
            graph,
            'neighbors',
            target.core.csr.neighbors,
            'uint32',
            target.core.slots
          ),
          weights: importGraphBuffer(
            graph,
            'weights',
            target.core.csr.weights,
            'float32',
            target.core.slots
          )
        },
        values: importGraphBuffer(graph, 'x', target.x, 'float32', rowCount),
        secondValues: bivariate
          ? importGraphBuffer(graph, 'y', target.y, 'float32', rowCount)
          : undefined,
        mask: importGraphBuffer(
          graph,
          'mask',
          bivariate ? target.maskXY : target.maskX,
          'uint32',
          rowCount
        ),
        statistic,
        alternative: o.alternative,
        parameters: target.permutationParameters.importToGraph(graph),
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        results: importGraphBuffer(
          graph,
          'results',
          target.buffersByName.permutationResults,
          'float32',
          PERMUTATION_RESULT_LENGTH
        ),
        histogram: importGraphBuffer(
          graph,
          'histogram',
          target.buffersByName.histogram,
          'uint32',
          HISTOGRAM_BINS
        )
      })
    );
    target.permutation.set(key, target.resources.track(graph.compile()));
    return true;
  };

  const getCorrelogramKey = () => `${ctx.options.bandCount}-${ctx.options.bandMode}`;

  const prepareCorrelogram = (target: World): boolean => {
    const key = getCorrelogramKey();
    if (target.correlogram.has(key)) return false;
    const o = ctx.options;
    const rowCount = target.geography.count;
    const graph = new GPUCommandGraph<void>(device, {id: `global-correlogram-${key}`});
    const view = <Format extends 'float32' | 'uint32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length);
    graph.add(
      new GPUSpatialCorrelogram({
        id: 'correlogram',
        positions: importGraphBuffer(
          graph,
          'positions',
          target.core.positions,
          'float32x2',
          rowCount
        ),
        values: view('x', target.x, 'float32', rowCount),
        mask: view('mask', target.maskX, 'uint32', rowCount),
        parameters: target.correlogramParameters.importToGraph(graph),
        gridSize: [128, 128],
        bandCount: o.bandCount,
        bandMode: o.bandMode,
        moransI: view('moransI', target.buffersByName.moransI, 'float32', o.bandCount),
        zScores: view('zScores', target.buffersByName.zScores, 'float32', o.bandCount),
        pValues: view('pValues', target.buffersByName.pValues, 'float32', o.bandCount),
        expectedI: view('expectedI', target.buffersByName.expectedI, 'float32', o.bandCount),
        pairCounts: view('pairCounts', target.buffersByName.pairCounts, 'uint32', o.bandCount),
        peakBands: view('peaks', target.buffersByName.peaks, 'uint32', 2),
        statistics: view(
          'statistics',
          target.buffersByName.correlogramStatistics,
          'float32',
          GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH
        )
      })
    );
    target.correlogram.set(key, target.resources.track(graph.compile()));
    return true;
  };

  // ---------------------------------------------------------------------------------------
  // The neighbour-rule comparison
  // ---------------------------------------------------------------------------------------

  /** Creates the second weights set and the one-pass Moran graph the comparison runs on. */
  const ensureSweep = (target: World): Sweep => {
    if (target.sweep) return target.sweep;
    const {resources, geography} = target;
    const rowCount = geography.count;
    const core = createWeightsCore({device, resources, id: 'sweep-core', geography});
    const results = resources.createBuffer('sweep-results', STATISTICS_LENGTH * 4);
    const graph = new GPUCommandGraph<void>(device, {id: `sweep-statistics-${geography.id}`});
    graph.add(
      new GPUGlobalSpatialStatistics({
        id: 'sweep-moran',
        weights: {
          offsets: importGraphBuffer(graph, 'offsets', core.csr.offsets, 'uint32', rowCount + 1),
          neighbors: importGraphBuffer(
            graph,
            'neighbors',
            core.csr.neighbors,
            'uint32',
            core.slots
          ),
          weights: importGraphBuffer(graph, 'weights', core.csr.weights, 'float32', core.slots)
        },
        values: importGraphBuffer(graph, 'x', target.x, 'float32', rowCount),
        mask: importGraphBuffer(graph, 'mask', target.maskX, 'uint32', rowCount),
        statistics: ['moran'],
        results: importGraphBuffer(graph, 'results', results, 'float32', STATISTICS_LENGTH)
      })
    );
    const sweep: Sweep = {
      core,
      graph: resources.track(graph.compile()),
      reader: new SummaryReader(
        resources,
        `sweep-${geography.id}`,
        [{buffer: results, size: STATISTICS_LENGTH * 4}],
        bytes => runGuarded('sweep result', () => handleSweepResult(bytes))
      ),
      values: RULES.map(() => Number.NaN),
      index: 0,
      state: 'idle',
      encoded: false
    };
    target.sweep = sweep;
    return sweep;
  };

  /** Compiles every rule's weights graph (a compile-time step, never inside `encode`). */
  const prepareSweep = (target: World) => {
    const sweep = ensureSweep(target);
    for (const rule of RULES) sweep.core.prepare(getRuleConfig(rule));
  };

  /** Starts the comparison again from the first rule (the variable or a rule parameter changed). */
  const restartSweep = (target: World) => {
    if (!ctx.options.showRuleSweep) return;
    const sweep = ensureSweep(target);
    sweep.values = RULES.map(() => Number.NaN);
    sweep.index = 0;
    sweep.state = 'running';
    sweep.encoded = false;
    ctx.setChart('iByW', null);
    ctx.setReadout('iRange', null);
  };

  const getRuleLabels = (): string[] => {
    const o = ctx.options;
    return ['Queen', 'Rook', `${o.k} nearest`, `Band, ${o.bandFactor} x spacing`];
  };

  const publishSweepChart = (sweep: Sweep) => {
    const o = ctx.options;
    const finite = sweep.values.filter(value => Number.isFinite(value));
    if (finite.length < RULES.length) return;
    ctx.setChart('iByW', {
      kind: 'bars',
      horizontal: true,
      title: "Moran's I under each neighbour rule",
      values: sweep.values,
      labels: getRuleLabels(),
      highlight: [RULES.indexOf(o.source)],
      formatX: value => value.toFixed(2),
      description:
        "Horizontal bars of row-standardised Moran's I under queen, rook, k nearest and distance-band neighbours; the rule currently chosen is highlighted.",
      table: false
    });
    ctx.setReadout(
      'iRange',
      `${formatNumber(Math.min(...finite), 2)} to ${formatNumber(Math.max(...finite), 2)}`
    );
  };

  const handleSweepResult = (bytes: ArrayBuffer) => {
    const target = world;
    const sweep = target?.sweep;
    if (!target || !sweep || sweep.state !== 'running') return;
    const results = new Float32Array(bytes, 0, STATISTICS_LENGTH);
    sweep.values[sweep.index] =
      results[
        GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.statistic
      ];
    sweep.index++;
    sweep.encoded = false;
    if (sweep.index >= RULES.length) {
      sweep.state = 'done';
      publishSweepChart(sweep);
      publishCost();
    }
    // A new frame encodes the next rule.
    ctx.requestLayers();
  };

  // ---------------------------------------------------------------------------------------
  // Geometry helpers: rings of a county, ground distances
  // ---------------------------------------------------------------------------------------

  const getCentroid = (target: World, row: number): [number, number] => [
    target.geography.centroids[row * 2],
    target.geography.centroids[row * 2 + 1]
  ];

  const getCentroidLngLat = (target: World, row: number): LngLat => {
    const [x, y] = getCentroid(target, row);
    return target.projection.unproject(x, y) as LngLat;
  };

  /** The rings of a county in longitude and latitude, for the achromatic selection outline. */
  const getRowRings = (target: World, row: number): LngLat[][] => {
    const {geography, projection} = target;
    const rings: LngLat[][] = [];
    for (
      let ring = geography.featureRingOffsets[row];
      ring < geography.featureRingOffsets[row + 1];
      ring++
    ) {
      const points: LngLat[] = [];
      for (
        let vertex = geography.contiguityRingOffsets[ring];
        vertex < geography.contiguityRingOffsets[ring + 1];
        vertex++
      ) {
        points.push(
          projection.unproject(
            geography.contiguityVertices[vertex * 2],
            geography.contiguityVertices[vertex * 2 + 1]
          ) as LngLat
        );
      }
      rings.push(points);
    }
    return rings;
  };

  /** Ground metres at the focus county per planar metre of the correlogram's frame. */
  const getGroundRatio = (target: World): number => {
    const focus = getCentroidLngLat(target, target.focusRow);
    return getGroundDistanceRatio(focus[1], target.geography.origin[1]);
  };

  /** The band the correlogram marks: the first z peak, else the strongest band, or -1. */
  const getPeakBand = (snapshot: Snapshot): {band: number; isFirstPeak: boolean} => {
    if (snapshot.peaks[0] !== GPU_SPATIAL_CORRELOGRAM_NO_BAND) {
      return {band: snapshot.peaks[0], isFirstPeak: true};
    }
    if (snapshot.peaks[1] !== GPU_SPATIAL_CORRELOGRAM_NO_BAND) {
      return {band: snapshot.peaks[1], isFirstPeak: false};
    }
    return {band: -1, isFirstPeak: false};
  };

  // ---------------------------------------------------------------------------------------
  // Charts, annotations, furniture and readouts
  // ---------------------------------------------------------------------------------------

  /** The Moran scatterplot: value against neighbour average, with the slope and a linked brush. */
  const publishScatter = () => {
    const target = world;
    const snapshot = target?.snapshot;
    if (!target || !snapshot) {
      ctx.setChart('moranScatter', null);
      return;
    }
    const o = ctx.options;
    const {scatter} = snapshot;
    const L = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT;
    const F = GPU_GLOBAL_SPATIAL_STATISTIC_FIELD;
    const moran = snapshot.global[L.moran + F.statistic];
    const rowStandardised = o.transform === 'row';
    let reach = 1;
    for (let index = 0; index < scatter.x.length; index++) {
      reach = Math.max(reach, Math.abs(scatter.x[index]), Math.abs(scatter.y[index]));
    }
    const limit = Math.ceil(reach);
    ctx.setChart('moranScatter', {
      kind: 'scatter',
      title: 'Moran scatterplot',
      x: scatter.x,
      y: scatter.y,
      colorIndex: scatter.quadrant,
      palette: getScatterPalette(ctx.ground()),
      xLabel: `${getVariableInfo(o.variable).label} (standard score)`,
      yLabel: 'Neighbour average (standard score)',
      xDomain: [-limit, limit],
      yDomain: [-limit, limit],
      radius: 1.5,
      quadrants: {x: 0, y: 0, labels: QUADRANT_NAMES},
      fit: {
        slope: rowStandardised ? moran : scatter.slope,
        intercept: 0,
        label: rowStandardised
          ? `slope = I = ${formatNumber(moran, 2)}`
          : `slope = ${formatNumber(scatter.slope, 2)} (lag is row standardised)`
      },
      onBrush: indices => {
        const current = world;
        if (!current || !indices?.length) {
          ctx.setHighlight(null);
          return;
        }
        const step = Math.max(1, Math.ceil(indices.length / MAXIMUM_BRUSHED_OUTLINES));
        const highlights: MapHighlight[] = [];
        for (let index = 0; index < indices.length; index += step) {
          highlights.push({
            kind: 'polygon',
            rings: getRowRings(current, scatter.rows[indices[index]])
          });
        }
        ctx.setHighlight(highlights);
      },
      description:
        'Scatterplot of every county: its standardised value across, the standardised average of its neighbours up. The cloud rises from lower left to upper right; the fitted line has slope equal to Moran I when the weights are row standardised.',
      table: false
    });
  };

  /** The permutation histogram with the observed statistic on the same axis. */
  const publishNull = () => {
    const target = world;
    const snapshot = target?.snapshot;
    const o = ctx.options;
    if (!target || !snapshot || o.statistic === 'joinCount') {
      ctx.setChart('nullDistribution', null);
      return;
    }
    const R = GPU_GLOBAL_PERMUTATION_RESULT;
    const permutation = snapshot.permutation;
    const observed = permutation[R.observed];
    const {values, domain} = rebinHistogram(
      snapshot.histogram,
      permutation[R.minimum],
      permutation[R.maximum],
      observed
    );
    const names: Record<GlobalStatistic, string> = {
      moran: "Moran's I",
      geary: "Geary's C",
      getisOrdG: 'General G',
      bivariateMoran: "Bivariate Moran's I",
      joinCount: 'Join counts'
    };
    ctx.setChart('nullDistribution', {
      kind: 'histogram',
      title: `${names[o.statistic]} of shuffled maps`,
      values,
      xDomain: domain,
      xLabel: names[o.statistic],
      yLabel: 'Shuffles',
      formatX: value => value.toFixed(3),
      now: observed,
      nowLabel: 'observed',
      description: `Histogram of ${names[o.statistic]} over the shuffles of the county values, with the observed value marked at ${formatNumber(observed, 3)}.`,
      table: false
    });
  };

  /** The correlogram: Moran's I by distance with the z-score on the right axis. */
  const publishCorrelogram = () => {
    const target = world;
    const snapshot = target?.snapshot;
    if (!target || !snapshot) {
      ctx.setChart('correlogram', null);
      return;
    }
    const o = ctx.options;
    const bandCount = o.bandCount;
    const maximumDistance = getMaxDistance(target.geography);
    const ratio = getGroundRatio(target);
    const distances = Array.from(
      {length: bandCount},
      (_, band) => (((band + 1) * maximumDistance) / bandCount) * ratio
    );
    const kilometers = distances.map(distance => distance / 1000);
    const zScores = Array.from(snapshot.zScores.subarray(0, bandCount));
    const finiteZ = zScores.filter(value => Number.isFinite(value));
    const peak = getPeakBand(snapshot);
    ctx.setChart('correlogram', {
      kind: 'line',
      title: "Moran's I by distance",
      xLabel: 'Distance (km at the focus county)',
      yLabel: "Moran's I",
      series: [
        {label: "Moran's I", x: kilometers, y: Array.from(snapshot.moransI.subarray(0, bandCount))},
        {label: 'z-score', x: kilometers, y: zScores, axis: 'y2', dashed: true, color: 1}
      ],
      y2Domain: finiteZ.length ? [Math.min(0, ...finiteZ), Math.max(...finiteZ) * 1.05] : [0, 1],
      y2Label: 'z-score',
      markers: peak.band >= 0 ? [{x: kilometers[peak.band], label: 'z peak'}] : [],
      formatX: value => (value < 10 ? value.toFixed(1) : value.toFixed(0)),
      description:
        "Line chart of global Moran's I for neighbours within each distance, falling as the distance grows, with the z-score on a second axis and a marker where it peaks.",
      table: false
    });
  };

  /** The standing sample line and the scale bar, which carries a tick at the correlogram peak. */
  const publishFurniture = () => {
    const target = world;
    if (!target) return;
    const o = ctx.options;
    const geography = target.geography;
    const counties = geography.id === 'us-counties';
    const sample = counties
      ? `${formatCount(geography.count)} counties of the contiguous US (Alaska and Hawaii not in the data)`
      : `${formatCount(geography.count)} census tracts, ACS 2018-2022 via CDC SVI 2022`;
    const focusLatitude = getCentroidLngLat(target, target.focusRow)[1];
    const snapshot = target.snapshot;
    const peak = snapshot ? getPeakBand(snapshot) : {band: -1, isFirstPeak: false};
    const ticks =
      o.showBands && snapshot && peak.band >= 0
        ? [(((peak.band + 1) * getMaxDistance(geography)) / o.bandCount) * getGroundRatio(target)]
        : undefined;
    ctx.setFurniture({
      title: {sample},
      scaleBar: o.showBands
        ? {latitude: focusLatitude, minZoom: counties ? 3.5 : undefined, ticks}
        : counties
          ? {latitude: 37, minZoom: 3.5}
          : {units: 'metric'},
      caveat: counties ? mercatorCaveat() : ''
    });
  };

  /**
   * Annotations that follow the data: the highest and lowest county of the first step, the
   * correlogram rings and the focus county of the last, and the Chicago names of the tract map.
   */
  const updateAnnotations = () => {
    const target = world;
    const snapshot = target?.snapshot;
    const o = ctx.options;
    if (!target) return;
    const counties = target.geography.id === 'us-counties';
    ctx.setAnnotations(
      'geography',
      counties ? null : labelsFor(CHICAGO, ['lake-michigan', 'loop'], {loop: {minZoom: 9}})
    );
    // The extremes of the plain value map: two finding notes with a number and a place.
    if (counties && o.display === 'value' && !o.showShuffled && !o.showBands) {
      const info = getVariableInfo(o.variable);
      const values = target.geography.getVariable(o.variable);
      let high = -1;
      let low = -1;
      for (let row = 0; row < values.length; row++) {
        if (!Number.isFinite(values[row])) continue;
        if (high < 0 || values[row] > values[high]) high = row;
        if (low < 0 || values[row] < values[low]) low = row;
      }
      const note = (row: number, label: string, tone: 'accent' | 'ink'): MapAnnotation => ({
        kind: 'note',
        id: `extreme-${label}`,
        coordinate: getCentroidLngLat(target, row),
        title: `${formatVariableValue(info, values[row])} ${info.unit}`,
        text: `${label}: ${target.geography.getName(row)}, ${target.geography.getGroupName(row)}`,
        tone,
        priority: 5
      });
      ctx.setAnnotations(
        'extremes',
        high >= 0 && low >= 0 ? [note(high, 'Highest', 'accent'), note(low, 'Lowest', 'ink')] : null
      );
    } else {
      ctx.setAnnotations('extremes', null);
    }
    // The focus county and the two correlogram rings, in true ground metres at the focus.
    if (snapshot && o.showBands) {
      const focus = getCentroidLngLat(target, target.focusRow);
      const ratio = getGroundRatio(target);
      const longest = getMaxDistance(target.geography) * ratio;
      const peak = getPeakBand(snapshot);
      const rings: MapAnnotation[] = [
        {
          kind: 'point',
          id: 'focus',
          coordinate: focus,
          text: target.geography.getName(target.focusRow),
          marker: 'ring',
          tone: 'signal',
          rank: 'subject',
          priority: 9
        },
        {
          kind: 'ring',
          id: 'band-longest',
          coordinate: focus,
          radiusMeters: longest,
          text: formatDistance(longest),
          dashed: true,
          geodesic: true
        }
      ];
      if (peak.band >= 0) {
        const peakDistance =
          (((peak.band + 1) * getMaxDistance(target.geography)) / o.bandCount) * ratio;
        rings.push({
          kind: 'ring',
          id: 'band-peak',
          coordinate: focus,
          radiusMeters: peakDistance,
          text: `${peak.isFirstPeak ? 'z peak' : 'strongest'} ${formatDistance(peakDistance)}`,
          tone: 'accent',
          geodesic: true
        });
      }
      ctx.setAnnotations('rings', rings);
    } else {
      ctx.setAnnotations('rings', null);
    }
  };

  const publishCost = () => {
    const target = world;
    if (!target) return;
    const included =
      target.snapshot?.global[
        GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.summary + GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY.count
      ];
    const sweep = target.sweep;
    ctx.setCost({
      records: included ?? target.geography.count,
      passes: target.statsCompiled.stats.nodeOrder.length,
      note:
        sweep && sweep.state === 'done'
          ? `${sweep.core.getGraphs().length} weights graphs compiled once, reused for 4 rules`
          : undefined
    });
  };

  const updateReadouts = () => {
    const target = world;
    const snapshot = target?.snapshot;
    if (!target || !snapshot) return;
    const o = ctx.options;
    const L = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT;
    const F = GPU_GLOBAL_SPATIAL_STATISTIC_FIELD;
    const S = GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY;
    const J = GPU_GLOBAL_JOIN_COUNT_FIELD;
    const R = GPU_GLOBAL_PERMUTATION_RESULT;
    const global = snapshot.global;
    const describe = (block: number, values: Float32Array) => ({
      statistic: values[block + F.statistic],
      expected: values[block + F.expected],
      zRand: values[block + F.zRandomization],
      pRand: values[block + F.pRandomization]
    });
    ctx.setReadout('places', formatCount(global[L.summary + S.count]));
    const moran = describe(L.moran, global);
    ctx.setReadout('moran', formatNumber(moran.statistic, 2));
    ctx.setReadout('moranZ', formatNumber(moran.zRand, 1));
    ctx.setReadout('expected', formatNumber(moran.expected, 4));
    const {counts, rows} = {counts: snapshot.scatter.counts, rows: snapshot.scatter.rows.length};
    ctx.setReadout('quadrantShare', rows ? formatPercent((counts[0] + counts[2]) / rows) : null);
    const geary = describe(L.geary, global);
    ctx.setReadout(
      'geary',
      `C = ${formatNumber(geary.statistic, 3)} (E = 1), z ${formatNumber(geary.zRand, 1)}, p ${formatPValue(geary.pRand)}`
    );
    const getis = describe(L.getisOrdG, global);
    ctx.setReadout(
      'getis',
      `G = ${formatNumber(getis.statistic, 5)} (E = ${formatNumber(getis.expected, 5)}), z ${formatNumber(getis.zRand, 1)}, p ${formatPValue(getis.pRand)}`
    );
    const bivariate = describe(L.bivariateMoran, snapshot.bivariate);
    ctx.setReadout('bivariate', formatNumber(bivariate.statistic, 2));
    ctx.setReadout('darkCorner', formatCount(target.bivariateCounts[8]));
    const join = snapshot.join;
    ctx.setReadout(
      'joins',
      `BB ${formatCount(snapshot.joinCounts[0] / 2)} (E ${formatNumber(join[L.joinCount + J.expectedBlackBlack], 0)}), BW ${formatCount(snapshot.joinCounts[1] / 2)} (E ${formatNumber(join[L.joinCount + J.expectedBlackWhite], 0)}), WW ${formatCount(snapshot.joinCounts[2] / 2)}`
    );
    ctx.setReadout(
      'joinTest',
      `BB z ${formatNumber(join[L.joinCount + J.zBlackBlack], 1)} p ${formatPValue(join[L.joinCount + J.pBlackBlack])}; BW z ${formatNumber(join[L.joinCount + J.zBlackWhite], 1)} p ${formatPValue(join[L.joinCount + J.pBlackWhite])}`
    );
    // Permutation test of the selected statistic.
    const permutation = snapshot.permutation;
    if (o.statistic === 'joinCount') {
      for (const id of ['pSim', 'pFloor', 'permutations', 'exceedances', 'zSim']) {
        ctx.setReadout(id, 'no permutation test for join counts');
      }
    } else {
      const shuffles = permutation[R.permutations];
      ctx.setReadout('pSim', formatPseudoPValue(permutation[R.pseudoPValue]));
      ctx.setReadout('pFloor', formatPseudoPValue(1 / (shuffles + 1)));
      ctx.setReadout('permutations', formatCount(shuffles));
      ctx.setReadout('exceedances', formatCount(permutation[R.exceedances]));
      ctx.setReadout('zSim', formatNumber(permutation[R.zSimulated], 1));
    }
    ctx.setReadout('shuffledI', formatNumber(snapshot.shuffled[L.moran + F.statistic], 2));
    // Correlogram.
    const maximumDistance = getMaxDistance(target.geography);
    const ratio = getGroundRatio(target);
    const peak = getPeakBand(snapshot);
    ctx.setReadout(
      'peak',
      peak.band >= 0
        ? formatDistance((((peak.band + 1) * maximumDistance) / o.bandCount) * ratio)
        : 'no defined band'
    );
    ctx.setReadout('maxDistance', formatDistance(maximumDistance * ratio));
    ctx.setReadout(
      'bandSpan',
      `${formatDistance(maximumDistance * ratio)} in ${o.bandCount} ${o.bandMode} bands${peak.band >= 0 && !peak.isFirstPeak ? '; no z peak, strongest band shown' : ''}`
    );
    ctx.setReadout(
      'distanceNote',
      `planar metres of the map frame, true at ${target.geography.origin[1].toFixed(1)} N; ground distance at the focus county = planar x ${ratio.toFixed(3)}`
    );
    publishScatter();
    publishNull();
    publishCorrelogram();
    publishFurniture();
    updateAnnotations();
    publishCost();
  };

  const handleSnapshot = (bytes: ArrayBuffer) => {
    const target = world;
    if (!target) return;
    const rowCount = target.geography.count;
    let offset = 0;
    const floats = (count: number) => {
      const view = new Float32Array(bytes, offset, count);
      offset += count * 4;
      return view;
    };
    const words = (count: number) => {
      const view = new Uint32Array(bytes, offset, count);
      offset += count * 4;
      return view;
    };
    const global = floats(STATISTICS_LENGTH);
    const bivariate = floats(STATISTICS_LENGTH);
    const join = floats(STATISTICS_LENGTH);
    const joinCounts = words(4);
    const permutation = floats(PERMUTATION_RESULT_LENGTH);
    const histogram = words(HISTOGRAM_BINS);
    const moransI = floats(MAXIMUM_BANDS);
    const zScores = floats(MAXIMUM_BANDS);
    const pValues = floats(MAXIMUM_BANDS);
    const pairCounts = words(MAXIMUM_BANDS);
    const peaks = words(2);
    const statistics = floats(GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH);
    const shuffled = floats(STATISTICS_LENGTH);
    const lag = floats(rowCount);
    const offsets = words(rowCount + 1);
    const S = GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY;
    const L = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT;
    const o = ctx.options;
    const scatter = buildMoranScatter(
      target.geography.getVariable(o.variable),
      lag,
      offsets,
      global[L.summary + S.mean],
      global[L.summary + S.variance]
    );
    target.snapshot = {
      global,
      bivariate,
      join,
      joinCounts,
      permutation,
      histogram,
      moransI,
      zScores,
      pValues,
      pairCounts,
      peaks,
      statistics,
      shuffled,
      lag,
      offsets,
      scatter
    };
    // Counts per quadrant for the legend (HH, LH, LL, HL in chart order) and the islands.
    const islands = global[L.summary + S.islandCount];
    ctx.setLegendData('quadrantCounts', [...scatter.counts, islands]);
    ctx.setLegendData(
      'lagCounts',
      getClassCounts(
        Array.from(scatter.rows, row => lag[row]),
        getFrozenClasses(target, o.variable).breaks
      )
    );
    updateReadouts();
    ctx.requestLayers();
  };

  const switchWorld = async (id: GeographyId) => {
    const token = ++switchToken;
    ctx.setStatus('Loading...');
    const geography = await getGeography(id);
    if (token !== switchToken || destroyed) return;
    const previous = world;
    world = null;
    ctx.requestLayers();
    if (previous) {
      previous.reader.stop();
      previous.sweep?.reader.stop();
      setTimeout(() => previous.resources.destroy(), 200);
    }
    const next = buildWorld(geography);
    world = next;
    legendHighlight = null;
    writeVariables(next);
    prepareAll();
    writeParameters();
    if (ctx.options.showRuleSweep) restartSweep(next);
    dirty = true;
    weightsDirty = true;
    stale = true;
    publishFurniture();
    updateAnnotations();
    ctx.setStatus('');
    ctx.fitBounds(GEOGRAPHY_BOUNDS[id], {transitionMs: 1200});
    ctx.requestLayers();
  };

  const prepareAll = () => {
    const target = world;
    if (!target) return;
    target.core.prepare(getConfig());
    preparePermutation(target);
    prepareCorrelogram(target);
    if (ctx.options.showRuleSweep) prepareSweep(target);
  };

  const initial = await getGeography(ctx.options.geography);
  if (ctx.signal.aborted) throw new Error('aborted');
  world = buildWorld(initial);
  writeVariables(world);
  prepareAll();
  writeParameters();
  if (ctx.options.showRuleSweep) restartSweep(world);
  publishFurniture();
  updateAnnotations();
  ctx.setStatus('');

  const pick = (event: {coordinate: readonly [number, number] | null}) =>
    world && event.coordinate ? world.geography.pick(event.coordinate[0], event.coordinate[1]) : -1;

  // ---------------------------------------------------------------------------------------
  // Layers and tooltip
  // ---------------------------------------------------------------------------------------

  /** Class styling of a value buffer with a frozen class table (the layer, legend and tooltip share it). */
  const getClassedStyle = (values: Buffer, table: ClassTable): SpatialAnalysisStyleProps => ({
    values,
    valueFormat: 'float32',
    colormap: 'greys',
    ...getClassTableLayerProps(table),
    noDataColor: NO_DATA_COLOR[ctx.ground()],
    highlightClasses: legendHighlight
  });

  const getCategoryStyle = (
    values: Buffer,
    palette: readonly SpatialAnalysisColor[],
    noDataColor: SpatialAnalysisColor
  ): SpatialAnalysisStyleProps => ({
    values,
    valueFormat: 'uint32',
    colormap: 'category',
    palette,
    noDataValue: NO_CLASS,
    noDataColor,
    highlightClasses: legendHighlight
  });

  const getTooltipRows = (target: World, row: number): TooltipRow[] => {
    const o = ctx.options;
    const geography = target.geography;
    const ground = ctx.ground();
    const first = getVariableInfo(o.variable);
    const second = getVariableInfo(o.secondVariable);
    const xValue = geography.getVariable(o.variable)[row];
    const yValue = geography.getVariable(o.secondVariable)[row];
    const xTable = getTable(target, o.variable);
    const yTable = getTable(target, o.secondVariable);
    const snapshot = target.snapshot;
    const lagValue = snapshot?.lag[row];
    const island = snapshot ? snapshot.offsets[row + 1] === snapshot.offsets[row] : false;
    const swatchOf = (table: ClassTable, value: number) => {
      const index = getClassIndexOf(table, value);
      return index >= 0 ? table.colors[index] : undefined;
    };
    // Quadrant of the county, in chart order, from the same rule as the GPU kernel.
    let quadrant = -1;
    if (snapshot && !island && Number.isFinite(xValue)) {
      const mean = snapshot.global[GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY.mean];
      const high = xValue >= mean;
      const neighborsHigh = (lagValue ?? 0) >= mean;
      quadrant = high ? (neighborsHigh ? 0 : 3) : neighborsHigh ? 1 : 2;
    }
    const quadrantColors = getQuadrantColors(ground);
    const quadrantSwatch =
      quadrant < 0
        ? undefined
        : [
            quadrantColors.highHigh,
            quadrantColors.lowHigh,
            quadrantColors.lowLow,
            quadrantColors.highLow
          ][quadrant];
    const xRow: TooltipRow = {
      label: first.label,
      value: formatVariableValue(first, xValue),
      unit: first.unit,
      swatch: swatchOf(xTable, xValue),
      emphasis: o.display === 'value'
    };
    const lagRow: TooltipRow | null =
      snapshot && !island && Number.isFinite(lagValue)
        ? {
            label: 'Neighbour average',
            value: formatVariableValue(first, lagValue as number),
            unit: first.unit,
            swatch: swatchOf(xTable, lagValue as number),
            emphasis: o.display === 'lag'
          }
        : null;
    const quadrantRow: TooltipRow = {
      label: 'Moran quadrant',
      value: quadrant >= 0 ? QUADRANT_NAMES[quadrant] : island ? 'No neighbours' : 'No data',
      swatch: quadrantSwatch,
      emphasis: o.display === 'quadrant'
    };
    const secondRow: TooltipRow = {
      label: second.label,
      value: formatVariableValue(second, yValue),
      unit: second.unit,
      swatch: swatchOf(yTable, yValue),
      emphasis: o.display === 'second'
    };
    const rank: TooltipRow = {
      label: 'Rank',
      value: Number.isFinite(xValue)
        ? `${formatOrdinal(getRank(target.sortedX, xValue) * 100)} percentile`
        : '-',
      unit: `of ${geography.unitPlural}`
    };
    if (o.display === 'lag' && lagRow) return [lagRow, rank, xRow, quadrantRow];
    if (o.display === 'quadrant') return [quadrantRow, rank, xRow, ...(lagRow ? [lagRow] : [])];
    if (o.display === 'second') return [secondRow, rank, xRow];
    if (o.display === 'bivariate') {
      const cell = target.bivariateClasses[row];
      const levels = ['Low', 'Medium', 'High'];
      const invert = o.secondVariable === 'income';
      const secondLevel =
        cell === NO_CLASS ? '' : levels[invert ? 2 - Math.floor(cell / 3) : Math.floor(cell / 3)];
      return [
        {
          label: `${first.label} with ${second.label.toLowerCase()}`,
          value:
            cell === NO_CLASS
              ? 'no data'
              : `${levels[cell % 3].toLowerCase()} x, ${secondLevel.toLowerCase()} y`,
          swatch: cell === NO_CLASS ? undefined : getBivariateColors(ground)[cell],
          emphasis: true
        },
        rank,
        xRow,
        secondRow
      ];
    }
    if (o.display === 'binary') {
      const black =
        Number.isFinite(xValue) &&
        xValue > getPercentile(geography.getVariable(o.variable), o.joinPercentile);
      return [
        {
          label: 'Join-count colour',
          value: black ? 'Black' : 'White',
          swatch: black ? BLACK_JOIN : WHITE_JOIN,
          emphasis: true
        },
        rank,
        xRow
      ];
    }
    return [{...xRow, emphasis: true}, rank, ...(lagRow ? [lagRow] : []), quadrantRow];
  };

  // ---------------------------------------------------------------------------------------
  // The instance
  // ---------------------------------------------------------------------------------------
  return {
    getCompiledGraphs() {
      const target = world;
      if (!target) return [];
      return [
        ...target.core.getGraphs(),
        target.statsCompiled,
        ...target.permutation.values(),
        ...target.correlogram.values(),
        ...(target.sweep ? [...target.sweep.core.getGraphs(), target.sweep.graph] : [])
      ] as CompiledGPUCommandGraph<never>[];
    },

    setOption(id, value) {
      if (id === 'geography') {
        void switchWorld(value as GeographyId);
        return;
      }
      const target = world;
      if (!target) return;
      // A legend filter belongs to the legend it was made in.
      if (['display', 'variable', 'secondVariable', 'showShuffled'].includes(id)) {
        legendHighlight = null;
      }
      if (id === 'variable' || id === 'secondVariable' || id === 'joinPercentile') {
        writeVariables(target);
        if (id !== 'secondVariable' && id !== 'joinPercentile') restartSweep(target);
      }
      if (id === 'seed') writeShuffled(target);
      if (['source', 'k', 'bandFactor', 'transform'].includes(id)) {
        weightsDirty = true;
      }
      if (id === 'showRuleSweep' || id === 'k' || id === 'bandFactor') {
        if (ctx.options.showRuleSweep) {
          prepareSweep(target);
          restartSweep(target);
        }
      }
      if (['display', 'showBands', 'showShuffled', 'showOutlines', 'variable'].includes(id)) {
        publishFurniture();
        updateAnnotations();
      }
      // Pure display options never touch the GPU graphs.
      if (['display', 'showBands', 'showShuffled', 'showOutlines', 'showRuleSweep'].includes(id)) {
        if (id === 'showRuleSweep' || id === 'showBands') publishFurniture();
        if (target.snapshot) publishScatter();
        if (id === 'showRuleSweep') {
          const sweep = target.sweep;
          if (sweep?.state === 'done') publishSweepChart(sweep);
          publishCost();
        }
        ctx.requestLayers();
        return;
      }
      if (id === 'source' && target.sweep?.state === 'done') publishSweepChart(target.sweep);
      prepareAll();
      writeParameters();
      dirty = true;
      stale = true;
      ctx.requestLayers();
    },

    onAction(id) {
      if (id !== 'shuffle') return;
      const next = (ctx.options.seed % MAXIMUM_SEED) + 1;
      ctx.setOptions({seed: next}, {notify: true});
    },

    onGroundChange() {
      publishTables();
      publishScatter();
      ctx.requestLayers();
    },

    onLegendFilter(id, classes) {
      legendHighlight =
        classes === null
          ? null
          : id === 'quadrant-classes'
            ? classes.map(index => QUADRANT_LEGEND_TO_CODE[index] ?? 0)
            : [...classes];
      ctx.requestLayers();
    },

    encode(commandEncoder: CommandEncoder, frame) {
      const target = world;
      if (!target) return;
      if (dirty || frame.frameIndex < 2) {
        if (weightsDirty || frame.frameIndex < 2) {
          target.core.encode(commandEncoder, getConfig());
          weightsDirty = false;
        }
        target.statsCompiled.encode(commandEncoder, {parameters: undefined});
        const permutationKey = getPermutationKey();
        if (permutationKey) {
          target.permutation.get(permutationKey)?.encode(commandEncoder, {parameters: undefined});
        }
        target.correlogram
          .get(getCorrelogramKey())
          ?.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        stale = true;
      }
      if (stale && frame.frameIndex >= 1 && !target.reader.isPending) {
        stale = false;
        target.reader.request(commandEncoder);
      }
      // The neighbour-rule comparison: one rule per frame on the second weights set.
      const sweep = target.sweep;
      if (sweep && sweep.state === 'running' && frame.frameIndex >= 1) {
        if (!sweep.encoded && !sweep.reader.isPending) {
          const config = getRuleConfig(RULES[sweep.index]);
          sweep.core.writeParameters(config);
          sweep.core.encode(commandEncoder, config);
          sweep.graph.encode(commandEncoder, {parameters: undefined});
          sweep.encoded = true;
          sweep.reader.request(commandEncoder);
        } else if (sweep.encoded) {
          // The ring was busy when the rule was encoded: copy its result now.
          sweep.reader.flush(commandEncoder);
        }
      }
    },

    getLayers() {
      const target = world;
      if (!target) return [];
      const o = ctx.options;
      const ground = ctx.ground();
      const {geography, buffers} = target;
      const coordinateOrigin: [number, number, number] = [
        geography.origin[0],
        geography.origin[1],
        0
      ];
      const noDataColor = NO_DATA_COLOR[ground];
      const valueTable = getTable(target, o.variable);
      const secondTable = getTable(target, o.secondVariable);
      const bivariatePalette = getBivariateColors(ground);
      const fill = (id: string, style: SpatialAnalysisStyleProps, side?: 'a' | 'b') =>
        new SpatialAnalysisPolygonLayer({
          id,
          coordinateOrigin,
          triangles: buffers.triangles,
          features: buffers.features,
          vertexCount: buffers.triangleVertexCount,
          ...style,
          compareSide: side,
          // The national paper sheet takes opaque fills; the quadrant colours carry their own alpha.
          opacity: 1
        });
      const layers: Layer[] = [];
      if (o.showShuffled && o.display === 'value') {
        // Swipe: the same classes on shuffled values (left) and the real values (right).
        layers.push(
          fill('global-fill-shuffled', getClassedStyle(target.shuffled, valueTable), 'a')
        );
        layers.push(fill('global-fill', getClassedStyle(target.x, valueTable), 'b'));
      } else if (o.display === 'quadrant') {
        layers.push(
          fill(
            'global-fill',
            getCategoryStyle(target.quadrant, getQuadrantPalette(ground), noDataColor)
          )
        );
      } else if (o.display === 'bivariate') {
        layers.push(
          fill(
            'global-fill',
            getCategoryStyle(target.bivariateClass, bivariatePalette, noDataColor)
          )
        );
      } else if (o.display === 'binary') {
        layers.push(
          fill(
            'global-fill',
            getCategoryStyle(target.joinClass, [WHITE_JOIN, BLACK_JOIN], noDataColor)
          )
        );
      } else if (o.display === 'second') {
        layers.push(fill('global-fill', getClassedStyle(target.y, secondTable)));
      } else if (o.display === 'lag') {
        layers.push(fill('global-fill', getClassedStyle(target.lag, valueTable)));
      } else {
        layers.push(fill('global-fill', getClassedStyle(target.x, valueTable)));
      }
      if (o.showOutlines) {
        // Tier 3: a thin hairline in the ground colour, never a dark mesh over thousands of polygons.
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'global-outline',
            coordinateOrigin,
            segments: buffers.outline,
            instanceCount: buffers.outlineSegmentCount,
            widthPixels: geography.count > 2000 ? 0.4 : 0.5,
            color: getHairlineColor(ground)
          })
        );
        if (target.stateBuffer && target.stateSegments) {
          // The zone-boundary tier: state lines over a casing.
          const line = getStateLineStyle(ground);
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'global-state-lines',
              coordinateOrigin,
              segments: target.stateBuffer,
              instanceCount: target.stateSegments.length / 4,
              widthPixels: line.widthPixels,
              color: line.color,
              outlineColor: line.casing,
              outlineWidthPixels: (line.casingPixels - line.widthPixels) / 2
            })
          );
        }
      }
      return layers;
    },

    getTooltip(event) {
      const target = world;
      const row = pick(event);
      if (!target || row < 0) return null;
      const geography = target.geography;
      const content: TooltipContent = {
        title: geography.getName(row),
        subtitle: geography.getGroupName(row),
        rows: getTooltipRows(target, row),
        anchor: getCentroidLngLat(target, row),
        highlight: {kind: 'polygon', rings: getRowRings(target, row)}
      };
      return content;
    },

    onClick(event) {
      const target = world;
      const row = pick(event);
      if (!target || row < 0) return false;
      target.focusRow = row;
      updateAnnotations();
      publishFurniture();
      if (target.snapshot) publishCorrelogram();
      ctx.requestLayers();
      return true;
    },

    destroy() {
      destroyed = true;
      switchToken++;
      const target = world;
      world = null;
      if (target) {
        target.reader.stop();
        target.sweep?.reader.stop();
        target.resources.destroy();
      }
    }
  };
}
