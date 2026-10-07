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
import {GPUCommandGraph, GPUReduction, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor
} from '../../engine/layers';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  BLACK_JOIN_COLOR,
  getOutlineColor,
  MORAN_COLORS,
  NO_DATA,
  WHITE_JOIN_COLOR
} from './b4-colors';
import {
  appendCircleSegments,
  formatNumber,
  formatPValue,
  formatSparkline,
  getPercentile,
  runGuarded
} from './b4-format';
import {
  getVariableInfo,
  loadGeography,
  type Geography,
  type GeographyId,
  type VariableId
} from './b4-geography';
import {B4PolygonFillLayer, createGeographyBuffers, type GeographyBuffers} from './b4-layers';
import {
  createWeightsCore,
  type WeightsConfig,
  type WeightsCore,
  type WeightsSource,
  type WeightsTransform
} from './b4-weights-core';

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
  statistic: GlobalStatistic;
  alternative: GPUPermutationAlternative;
  permutations: number;
  seed: number;
  joinPercentile: number;
  bandCount: number;
  bandMode: GPUSpatialCorrelogramBandMode;
  maxDistanceFactor: number;
  varianceAssumption: GPUSpatialCorrelogramVarianceAssumption;
  display: 'value' | 'second' | 'lag' | 'quadrant' | 'binary';
  ramp: RampName;
  showBands: boolean;
  showOutlines: boolean;
};

const MAXIMUM_PERMUTATIONS = 999;
const HISTOGRAM_BINS = 24;
const MAXIMUM_BANDS = 32;
const RING_SEGMENTS = 96;
const STATISTICS_LENGTH = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length;
const PERMUTATION_RESULT_LENGTH = GPU_GLOBAL_PERMUTATION_RESULT.length;
const NO_DATA_CLASS = 0xffffffff;
const HIDDEN: SpatialAnalysisColor = [0, 0, 0, 0];
const FOCUS_POINTS: Record<GeographyId, [number, number]> = {
  'us-counties': [-87.65, 41.84],
  'chicago-tracts': [-87.63, 41.88]
};

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

type World = {
  geography: Geography;
  resources: SpatialAnalysisResources;
  core: WeightsCore;
  buffers: GeographyBuffers;
  x: Buffer;
  y: Buffer;
  maskX: Buffer;
  maskY: Buffer;
  maskXY: Buffer;
  threshold: Buffer;
  lag: Buffer;
  quadrant: Buffer;
  joinClass: Buffer;
  extents: {x: Buffer; y: Buffer; lag: Buffer};
  focusIds: Buffer;
  rings: Buffer;
  ringCount: number;
  ringPeak: boolean;
  focusRow: number;
  statsCompiled: CompiledGPUCommandGraph<void>;
  permutation: Map<string, CompiledGPUCommandGraph<void>>;
  correlogram: Map<string, CompiledGPUCommandGraph<void>>;
  permutationParameters: GPUParameterBuffer<'uint32'>;
  correlogramParameters: GPUParameterBuffer<'float32'>;
  buffersByName: Record<string, Buffer>;
  reader: SummaryReader;
  snapshot: Snapshot | null;
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
  extentX: [number, number];
  extentY: [number, number];
  extentLag: [number, number];
};

const geographyCache = new Map<GeographyId, Promise<Geography>>();

/**
 * Global spatial autocorrelation of health and income. One `GPUGlobalSpatialStatistics` pass per
 * variable family computes Moran's I, Geary's C, Getis-Ord G, bivariate Moran's I and join counts
 * with analytic z and p; `GPUGlobalPermutationTest` builds the null distribution; and
 * `GPUSpatialCorrelogram` repeats Moran's I over a ladder of distance bands.
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

  const getGeography = (id: GeographyId) => {
    let promise = geographyCache.get(id);
    if (!promise) {
      promise = loadGeography(id, ctx.datasets, ctx.signal);
      promise.catch(() => geographyCache.delete(id));
      geographyCache.set(id, promise);
    }
    return promise;
  };

  const getConfig = (geography: Geography): WeightsConfig => {
    const o = ctx.options;
    void geography;
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

  const getMaxDistance = (geography: Geography) =>
    ctx.options.maxDistanceFactor * geography.medianSpacing;

  const buildWorld = (geography: Geography): World => {
    const resources = new SpatialAnalysisResources(device, `global-${geography.id}`);
    const rowCount = geography.count;
    const create = (name: string, data: number | Float32Array | Uint32Array) =>
      resources.createBuffer(name, data);
    const core = createWeightsCore({device, resources, id: 'global-core', geography});
    const o = ctx.options;
    const xValues = geography.getVariable(o.variable);
    const yValues = geography.getVariable(o.secondVariable);
    const buffersByName = {
      globalResults: create('global-results', STATISTICS_LENGTH * 4),
      bivariateResults: create('bivariate-results', STATISTICS_LENGTH * 4),
      joinResults: create('join-results', STATISTICS_LENGTH * 4),
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
    const x = create('x', Float32Array.from(xValues));
    const y = create('y', Float32Array.from(yValues));
    const maskX = create('mask-x', new Uint32Array(rowCount));
    const maskY = create('mask-y', new Uint32Array(rowCount));
    const maskXY = create('mask-xy', new Uint32Array(rowCount));
    const threshold = create('join-threshold', new Float32Array(1));
    const lag = create('lag', rowCount * 4);
    const quadrant = create('quadrant', rowCount * 4);
    const joinClass = create('join-class', rowCount * 4);
    const extents = {
      x: create('extent-x', 8),
      y: create('extent-y', 8),
      lag: create('extent-lag', 8)
    };
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

    // One graph: lag, classes, every analytic statistic and the extents.
    const graph = new GPUCommandGraph<void>(device, {id: `global-statistics-${geography.id}`});
    const weights = {
      offsets: importGraphBuffer(graph, 'offsets', core.csr.offsets, 'uint32', rowCount + 1),
      neighbors: importGraphBuffer(graph, 'neighbors', core.csr.neighbors, 'uint32', core.slots),
      weights: importGraphBuffer(graph, 'weights', core.csr.weights, 'float32', core.slots)
    };
    const xView = importGraphBuffer(graph, 'x', x, 'float32', rowCount);
    const yView = importGraphBuffer(graph, 'y', y, 'float32', rowCount);
    const maskXView = importGraphBuffer(graph, 'mask-x', maskX, 'uint32', rowCount);
    const maskYView = importGraphBuffer(graph, 'mask-y', maskY, 'uint32', rowCount);
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
        {
          name: 'quadrant',
          view: importGraphBuffer(graph, 'quadrant', quadrant, 'uint32', rowCount),
          type: 'u32',
          access: 'read_write'
        },
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
    for (const [name, view, mask, target] of [
      ['x', xView, maskXView, extents.x],
      ['y', yView, maskYView, extents.y],
      ['lag', lagView, maskXView, extents.lag]
    ] as const) {
      graph.add(
        new GPUReduction({
          id: `extent-${name}`,
          input: view,
          mask,
          output: importGraphBuffer(graph, `extent-${name}`, target, 'float32', 2),
          operation: 'extent'
        })
      );
    }
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
        {buffer: extents.x, size: 8},
        {buffer: extents.y, size: 8},
        {buffer: extents.lag, size: 8}
      ],
      bytes => runGuarded('global snapshot', () => handleSnapshot(bytes))
    );

    const created: World = {
      geography,
      resources,
      core,
      buffers: createGeographyBuffers(resources, geography, 'global'),
      x,
      y,
      maskX,
      maskY,
      maskXY,
      threshold,
      lag,
      quadrant,
      joinClass,
      extents,
      focusIds: create('focus-id', new Uint32Array([focusRow])),
      rings: create('rings', (MAXIMUM_BANDS + 1) * RING_SEGMENTS * 16),
      ringCount: 0,
      ringPeak: false,
      focusRow,
      statsCompiled,
      permutation: new Map(),
      correlogram: new Map(),
      permutationParameters,
      correlogramParameters,
      buffersByName,
      reader,
      snapshot: null
    };
    return created;
  };

  /** Writes values, masks and the join-count threshold of the current variables. */
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
  };

  const writeParameters = () => {
    const target = world;
    if (!target) return;
    const o = ctx.options;
    target.core.writeParameters(getConfig(target.geography));
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

  const prepareAll = () => {
    const target = world;
    if (!target) return;
    target.core.prepare(getConfig(target.geography));
    preparePermutation(target);
    prepareCorrelogram(target);
  };

  const getCentroid = (target: World, row: number): [number, number] => [
    target.geography.centroids[row * 2],
    target.geography.centroids[row * 2 + 1]
  ];

  const writeRings = (target: World) => {
    const snapshot = target.snapshot;
    const o = ctx.options;
    const maximumDistance = getMaxDistance(target.geography);
    const [centerX, centerY] = getCentroid(target, target.focusRow);
    const segments = new Float32Array((MAXIMUM_BANDS + 1) * RING_SEGMENTS * 4);
    let cursor = 0;
    // The first-peak ring comes first so it can be drawn separately.
    let peakBand = -1;
    if (snapshot && snapshot.peaks[0] !== GPU_SPATIAL_CORRELOGRAM_NO_BAND) {
      peakBand = snapshot.peaks[0];
    } else if (snapshot && snapshot.peaks[1] !== GPU_SPATIAL_CORRELOGRAM_NO_BAND) {
      peakBand = snapshot.peaks[1];
    }
    if (peakBand >= 0) {
      cursor = appendCircleSegments(
        segments,
        cursor,
        centerX,
        centerY,
        (maximumDistance * (peakBand + 1)) / o.bandCount,
        RING_SEGMENTS
      );
    }
    for (let band = 0; band < o.bandCount; band++) {
      cursor = appendCircleSegments(
        segments,
        cursor,
        centerX,
        centerY,
        (maximumDistance * (band + 1)) / o.bandCount,
        RING_SEGMENTS
      );
    }
    target.rings.write(segments);
    target.ringCount = o.bandCount;
    target.ringPeak = peakBand >= 0;
  };

  const updateReadouts = () => {
    const target = world;
    const snapshot = target?.snapshot;
    if (!target || !snapshot) return;
    const o = ctx.options;
    const rowCount = target.geography.count;
    const L = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT;
    const F = GPU_GLOBAL_SPATIAL_STATISTIC_FIELD;
    const S = GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY;
    const J = GPU_GLOBAL_JOIN_COUNT_FIELD;
    const variable = getVariableInfo(o.variable);
    const second = getVariableInfo(o.secondVariable);
    const global = snapshot.global;
    const included = global[L.summary + S.count];
    ctx.setReadout(
      'places',
      `${formatCount(included)} of ${formatCount(rowCount)} with data (${formatCount(global[L.summary + S.islandCount])} islands)`
    );
    ctx.setReadout(
      'summary',
      `mean ${formatNumber(global[L.summary + S.mean], variable.digits + 1)} ${variable.unit}, S0 ${formatCount(global[L.summary + S.s0])}`
    );
    const describe = (block: number, values: Float32Array) => ({
      statistic: values[block + F.statistic],
      expected: values[block + F.expected],
      zRand: values[block + F.zRandomization],
      pRand: values[block + F.pRandomization],
      zNorm: values[block + F.zNormality],
      pNorm: values[block + F.pNormality]
    });
    const moran = describe(L.moran, global);
    ctx.setReadout(
      'moran',
      `I = ${formatNumber(moran.statistic)} (E = ${formatNumber(moran.expected, 4)})`
    );
    ctx.setReadout(
      'moranTest',
      `z ${formatNumber(moran.zRand, 1)} rand / ${formatNumber(moran.zNorm, 1)} norm, p ${formatPValue(moran.pRand)}`
    );
    const geary = describe(L.geary, global);
    ctx.setReadout(
      'geary',
      `C = ${formatNumber(geary.statistic)} (E = 1), z ${formatNumber(geary.zRand, 1)}, p ${formatPValue(geary.pRand)}`
    );
    const getis = describe(L.getisOrdG, global);
    ctx.setReadout(
      'getis',
      `G = ${formatNumber(getis.statistic, 5)} (E = ${formatNumber(getis.expected, 5)}), z ${formatNumber(getis.zRand, 1)}, p ${formatPValue(getis.pRand)}`
    );
    const bivariate = describe(L.bivariateMoran, snapshot.bivariate);
    ctx.setReadout(
      'bivariate',
      `${variable.label.split(' ')[0]} x ${second.label.split(' ')[0]}: I = ${formatNumber(bivariate.statistic)}, z ${formatNumber(bivariate.zRand, 1)}, p ${formatPValue(bivariate.pRand)}`
    );
    const join = snapshot.join;
    const joinOffset = L.joinCount;
    ctx.setReadout(
      'joins',
      `BB ${formatCount(snapshot.joinCounts[0] / 2)} (E ${formatNumber(join[joinOffset + J.expectedBlackBlack], 0)}), BW ${formatCount(snapshot.joinCounts[1] / 2)} (E ${formatNumber(join[joinOffset + J.expectedBlackWhite], 0)}), WW ${formatCount(snapshot.joinCounts[2] / 2)}`
    );
    ctx.setReadout(
      'joinTest',
      `BB z ${formatNumber(join[joinOffset + J.zBlackBlack], 1)} p ${formatPValue(join[joinOffset + J.pBlackBlack])}; BW z ${formatNumber(join[joinOffset + J.zBlackWhite], 1)} p ${formatPValue(join[joinOffset + J.pBlackWhite])}`
    );
    // Permutation test of the selected statistic.
    const R = GPU_GLOBAL_PERMUTATION_RESULT;
    const permutation = snapshot.permutation;
    if (o.statistic === 'joinCount') {
      ctx.setReadout('permutation', 'no permutation test for join counts');
      ctx.setReadout('nullDistribution', 'n/a');
    } else {
      ctx.setReadout(
        'permutation',
        `observed ${formatNumber(permutation[R.observed], 4)}, p_sim ${formatPValue(permutation[R.pseudoPValue])}, z_sim ${formatNumber(permutation[R.zSimulated], 1)}`
      );
      ctx.setReadout(
        'nullDistribution',
        `${formatSparkline(snapshot.histogram)}  ${formatNumber(permutation[R.minimum], 3)} to ${formatNumber(permutation[R.maximum], 3)} (mean ${formatNumber(permutation[R.simulatedMean], 4)}, sd ${formatNumber(permutation[R.simulatedStandardDeviation], 4)})`
      );
    }
    // Correlogram.
    const bandCount = o.bandCount;
    const maximumDistance = getMaxDistance(target.geography);
    const profile = snapshot.moransI.subarray(0, bandCount);
    ctx.setReadout('correlogram', `${formatSparkline(profile, 0)}  I by distance band`);
    ctx.setReadout(
      'correlogramZ',
      `${formatSparkline(snapshot.zScores.subarray(0, bandCount), 0)}  z by band`
    );
    const peak = snapshot.peaks[0];
    const maximum = snapshot.peaks[1];
    ctx.setReadout(
      'peak',
      peak !== GPU_SPATIAL_CORRELOGRAM_NO_BAND
        ? `${(((peak + 1) * maximumDistance) / bandCount / 1000).toFixed(0)} km (band ${peak + 1} of ${bandCount})`
        : maximum !== GPU_SPATIAL_CORRELOGRAM_NO_BAND
          ? `no peak; strongest at ${(((maximum + 1) * maximumDistance) / bandCount / 1000).toFixed(0)} km`
          : 'no defined band'
    );
    ctx.setReadout(
      'bandSpan',
      `${(maximumDistance / 1000).toFixed(0)} km in ${bandCount} ${o.bandMode} bands`
    );
  };

  const handleSnapshot = (bytes: ArrayBuffer) => {
    const target = world;
    if (!target) return;
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
    const snapshot: Snapshot = {
      global: floats(STATISTICS_LENGTH),
      bivariate: floats(STATISTICS_LENGTH),
      join: floats(STATISTICS_LENGTH),
      joinCounts: words(4),
      permutation: floats(PERMUTATION_RESULT_LENGTH),
      histogram: words(HISTOGRAM_BINS),
      moransI: floats(MAXIMUM_BANDS),
      zScores: floats(MAXIMUM_BANDS),
      pValues: floats(MAXIMUM_BANDS),
      pairCounts: words(MAXIMUM_BANDS),
      peaks: words(2),
      statistics: floats(GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH),
      extentX: [0, 1],
      extentY: [0, 1],
      extentLag: [0, 1]
    };
    snapshot.extentX = [floats(2)[0], new Float32Array(bytes, offset - 4, 1)[0]];
    snapshot.extentY = [floats(2)[0], new Float32Array(bytes, offset - 4, 1)[0]];
    snapshot.extentLag = [floats(2)[0], new Float32Array(bytes, offset - 4, 1)[0]];
    target.snapshot = snapshot;
    const o = ctx.options;
    const extent =
      o.display === 'second'
        ? snapshot.extentY
        : o.display === 'lag'
          ? snapshot.extentLag
          : snapshot.extentX;
    if (Number.isFinite(extent[0]) && Number.isFinite(extent[1])) {
      ctx.setLegendExtent('display', extent);
    }
    writeRings(target);
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
      setTimeout(() => previous.resources.destroy(), 200);
    }
    world = buildWorld(geography);
    writeVariables(world);
    prepareAll();
    writeParameters();
    dirty = true;
    weightsDirty = true;
    stale = true;
    ctx.setStatus('');
    ctx.requestLayers();
  };

  const initial = await getGeography(ctx.options.geography);
  if (ctx.signal.aborted) throw new Error('aborted');
  world = buildWorld(initial);
  writeVariables(world);
  prepareAll();
  writeParameters();
  ctx.setStatus('');

  const pick = (event: {coordinate: readonly [number, number] | null}) =>
    world && event.coordinate ? world.geography.pick(event.coordinate[0], event.coordinate[1]) : -1;

  return {
    getCompiledGraphs() {
      const target = world;
      if (!target) return [];
      return [
        ...target.core.getGraphs(),
        target.statsCompiled,
        ...target.permutation.values(),
        ...target.correlogram.values()
      ] as CompiledGPUCommandGraph<never>[];
    },

    setOption(id, value) {
      const target = world;
      if (id === 'geography') {
        void switchWorld(value as GeographyId);
        return;
      }
      if (!target) return;
      if (id === 'variable' || id === 'secondVariable' || id === 'joinPercentile') {
        writeVariables(target);
      }
      if (
        ['source', 'k', 'bandFactor', 'transform'].includes(id) // weights
      ) {
        weightsDirty = true;
      }
      if (id === 'maxDistanceFactor' || id === 'bandCount' || id === 'bandMode') {
        target.ringPeak = false;
      }
      prepareAll();
      writeParameters();
      dirty = true;
      stale = true;
      ctx.requestLayers();
    },

    encode(commandEncoder: CommandEncoder, frame) {
      const target = world;
      if (!target) return;
      if (dirty || frame.frameIndex < 2) {
        const o = ctx.options;
        if (weightsDirty || frame.frameIndex < 2) {
          target.core.encode(commandEncoder, getConfig(target.geography));
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
        void o;
        dirty = false;
        stale = true;
      }
      if (stale && frame.frameIndex >= 1 && !target.reader.isPending) {
        stale = false;
        target.reader.request(commandEncoder);
      }
    },

    getLayers() {
      const target = world;
      if (!target) return [];
      const o = ctx.options;
      const theme = ctx.theme();
      const {geography, buffers} = target;
      const coordinateOrigin: [number, number, number] = [
        geography.origin[0],
        geography.origin[1],
        0
      ];
      const quadrantPalette: SpatialAnalysisColor[] = [
        HIDDEN,
        MORAN_COLORS[1],
        MORAN_COLORS[2],
        MORAN_COLORS[3],
        MORAN_COLORS[4]
      ];
      const style =
        o.display === 'quadrant'
          ? {
              values: target.quadrant,
              valueFormat: 'uint32' as const,
              colormap: 'category' as const,
              palette: quadrantPalette
            }
          : o.display === 'binary'
            ? {
                values: target.joinClass,
                valueFormat: 'uint32' as const,
                colormap: 'category' as const,
                palette: [WHITE_JOIN_COLOR, BLACK_JOIN_COLOR],
                noDataValue: NO_DATA_CLASS
              }
            : {
                values:
                  o.display === 'second' ? target.y : o.display === 'lag' ? target.lag : target.x,
                valueFormat: 'float32' as const,
                colormap: o.ramp,
                extent:
                  o.display === 'second'
                    ? target.extents.y
                    : o.display === 'lag'
                      ? target.extents.lag
                      : target.extents.x
              };
      const layers: Layer[] = [
        new B4PolygonFillLayer({
          id: 'global-fill',
          coordinateOrigin,
          triangles: buffers.triangles,
          features: buffers.features,
          triangleVertexCount: buffers.triangleVertexCount,
          ...style,
          noDataColor: o.display === 'binary' || o.display === 'quadrant' ? HIDDEN : NO_DATA,
          opacity: 0.92
        })
      ];
      if (o.showOutlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'global-outline',
            coordinateOrigin,
            segments: buffers.outline,
            instanceCount: buffers.outlineSegmentCount,
            widthPixels: 0.8,
            color: getOutlineColor(theme)
          })
        );
      }
      if (o.showBands && target.ringCount > 0) {
        const ringColor: SpatialAnalysisColor =
          theme === 'dark' ? [255, 255, 255, 120] : [20, 30, 60, 110];
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'global-rings',
            coordinateOrigin,
            segments: target.rings,
            instanceCount: (target.ringCount + (target.ringPeak ? 1 : 0)) * RING_SEGMENTS,
            widthPixels: 1,
            color: ringColor
          })
        );
        if (target.ringPeak) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'global-peak-ring',
              coordinateOrigin,
              segments: target.rings,
              instanceCount: RING_SEGMENTS,
              widthPixels: 3.4,
              color: [230, 90, 20, 255]
            })
          );
        }
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'global-focus',
          coordinateOrigin,
          positions: target.core.positions,
          ids: target.focusIds,
          instanceCount: 1,
          radiusPixels: 6,
          color: theme === 'dark' ? [255, 214, 120, 255] : [20, 20, 40, 255]
        })
      );
      return layers;
    },

    getTooltip(event) {
      const target = world;
      const row = pick(event);
      if (!target || row < 0) return null;
      const o = ctx.options;
      const x = target.geography.getVariable(o.variable)[row];
      const y = target.geography.getVariable(o.secondVariable)[row];
      const first = getVariableInfo(o.variable);
      const second = getVariableInfo(o.secondVariable);
      const format = (value: number, info: ReturnType<typeof getVariableInfo>) =>
        Number.isFinite(value) ? `${value.toFixed(info.digits)} ${info.unit}` : 'no data';
      return [
        `${target.geography.getName(row)} (${target.geography.getGroupName(row)})`,
        `${first.label}: ${format(x, first)}`,
        `${second.label}: ${format(y, second)}`
      ].join('\n');
    },

    onClick(event) {
      const target = world;
      const row = pick(event);
      if (!target || row < 0) return false;
      target.focusRow = row;
      target.focusIds.write(new Uint32Array([row]));
      writeRings(target);
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
        target.resources.destroy();
      }
    }
  };
}
