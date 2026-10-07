// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUPointPatternIndicesParameterValues,
  getGPURipleyDistanceParameterValues,
  getGPURipleyParameterValues,
  GPU_CLARK_EVANS_LENGTH,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  GPU_QUADRAT_STATISTICS_LENGTH,
  GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH,
  GPU_RIPLEY_PARAMETER_LENGTH,
  GPUPointPatternIndices,
  GPURipley,
  GPURipleyDistanceFunctions,
  type GPURipleyDistanceEdgeCorrection,
  type GPURipleyEdgeCorrection
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {createSeededRandom} from '../../engine/projection';
import {
  formatCount,
  getViewportMetricBounds,
  SpatialAnalysisResources
} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {formatCategory} from './b1-nature-data';

/** Option state of the point-patterns scene. */
export type PointPatternOptions = {
  subject: 'observations' | 'places';
  groupCategory: string;
  placeCategory: string;
  maximumDistance: number;
  window: 'city' | 'view';
  ripleyCorrection: GPURipleyEdgeCorrection;
  distanceCorrection: GPURipleyDistanceEdgeCorrection;
  jitter: number;
  radiusCount: string;
  quadratGrid: string;
  referenceGrid: string;
  colorPoints: 'selection' | 'nearest';
  showQuadrats: boolean;
  showWindow: boolean;
  showOthers: boolean;
};

const SEARCH_GRID: readonly [number, number] = [96, 96];
const SPARK_LEVELS = '▁▂▃▄▅▆▇█';
const SETTLE_MILLISECONDS = 350;

/** Human labels for place categories. */
export const PLACE_LABELS: Record<string, string> = {
  restaurant_cafe: 'Restaurants and cafes',
  bar_nightlife: 'Bars and nightlife',
  grocery: 'Grocery and convenience',
  health: 'Health care',
  school_education: 'Schools and education',
  park_recreation: 'Parks and recreation',
  transit: 'Transit stops',
  retail: 'Retail',
  finance_business: 'Finance and business',
  personal_services: 'Personal services',
  arts_culture: 'Arts and culture',
  worship_community: 'Worship and community',
  lodging: 'Lodging',
  other: 'Other'
};

type SubjectData = {
  id: 'observations' | 'places';
  label: string;
  count: number;
  base: Float32Array;
  category: Uint8Array;
  categoryNames: readonly string[];
  positionsBuffer: ReturnType<SpatialAnalysisResources['createBuffer']>;
  maskBuffer: ReturnType<SpatialAnalysisResources['createBuffer']>;
  jitterApplied: number;
};

type PatternSet = {
  key: string;
  subject: SubjectData;
  radiusCount: number;
  quadratGrid: readonly [number, number];
  compiled: CompiledGPUCommandGraph<void>;
  nearestDistances: ReturnType<SpatialAnalysisResources['createBuffer']>;
  quadratCounts: ReturnType<SpatialAnalysisResources['createBuffer']>;
  reader: SummaryReader;
  summaryWords: number;
};

/**
 * Point-pattern analysis of one Chicago observation group or place type: `GPURipley` (K, L, L minus r),
 * `GPURipleyDistanceFunctions` (G, F, J) and `GPUPointPatternIndices` (nearest-neighbour distances,
 * Clark-Evans, quadrat counts) run over the same points, mask and window. The category, the window,
 * the radius, the edge corrections and the jitter are buffer writes; the number of radii, the quadrat
 * grid and the F reference lattice are compile-time and select cached graphs.
 */
export async function createPointPatterns(
  ctx: SceneContext<PointPatternOptions>
): Promise<SceneInstance<PointPatternOptions>> {
  const {device} = ctx;
  const observations = ctx.datasets.get('chicago-nature');
  const places = ctx.datasets.get('chicago-places');
  const origin = observations.defaultOrigin;
  const projection = observations.getProjection(origin);
  const resources = new SpatialAnalysisResources(device, 'point-patterns');

  const makeSubject = (
    id: 'observations' | 'places',
    label: string,
    dataset: typeof observations
  ): SubjectData => {
    const base = dataset.projectColumn('position', origin);
    const count = base.length / 2;
    return {
      id,
      label,
      count,
      base,
      category: dataset.column<Uint8Array>('category'),
      categoryNames: dataset.categories('category'),
      positionsBuffer: resources.createBuffer(`${id}-positions`, base.slice()),
      maskBuffer: resources.createBuffer(`${id}-mask`, new Uint32Array(count).fill(1)),
      jitterApplied: 0
    };
  };
  const subjects = {
    observations: makeSubject('observations', 'Nature observations', observations),
    places: makeSubject('places', 'Places (Overture)', places)
  };

  // The study window: the observation bounding box in local meters (one window for both subjects).
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const subject of [subjects.observations, subjects.places]) {
    for (let index = 0; index < subject.count; index++) {
      const x = subject.base[index * 2];
      const y = subject.base[index * 2 + 1];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  const cityBounds: [number, number, number, number] = [minX - 5, minY - 5, maxX + 5, maxY + 5];

  const ripleyParameters = resources.createParameterBuffer(
    'ripley',
    'float32',
    GPU_RIPLEY_PARAMETER_LENGTH
  );
  const distanceParameters = resources.createParameterBuffer(
    'distance',
    'float32',
    GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH
  );
  const indicesParameters = resources.createParameterBuffer(
    'indices',
    'float32',
    GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH
  );
  const windowBuffer = resources.createParameterBuffer('window', 'float32', 4);
  const windowSegments = resources.createBuffer('window-segments', 4 * 16);

  let destroyed = false;
  let measuring = false;
  let dirty = true;
  let lastChangeTime = performance.now();
  let bounds: [number, number, number, number] = [...cityBounds];
  let lastViewBounds: string | null = null;
  let displayedKey = '';
  let active: PatternSet | null = null;
  let previousCurve: {label: string; text: string} | null = null;
  let currentCurve: {label: string; text: string} | null = null;
  let nearestMax = 300;
  let quadratMax = 1;
  const sets = new Map<string, PatternSet>();

  const subjectOf = (options: PointPatternOptions) => subjects[options.subject];
  const categoryOf = (options: PointPatternOptions): string =>
    options.subject === 'observations' ? options.groupCategory : options.placeCategory;
  const categoryLabel = (options: PointPatternOptions): string => {
    const category = categoryOf(options);
    if (category === 'all')
      return options.subject === 'observations' ? 'All observations' : 'All places';
    return options.subject === 'observations'
      ? formatCategory(category)
      : (PLACE_LABELS[category] ?? category);
  };
  const getKey = (options: PointPatternOptions) =>
    `${options.subject}|${options.radiusCount}|${options.quadratGrid}|${options.referenceGrid}`;

  function parseGrid(value: string): [number, number] {
    const side = Number(value);
    return [side, side];
  }

  function buildSet(options: PointPatternOptions): PatternSet {
    const key = getKey(options);
    const subject = subjectOf(options);
    const radiusCount = Number(options.radiusCount);
    const quadratGrid = parseGrid(options.quadratGrid);
    const referenceGrid = parseGrid(options.referenceGrid);
    const id = key.replace(/\|/g, '-');
    const quadratCells = quadratGrid[0] * quadratGrid[1];
    const buffer = (name: string, floats: number) =>
      resources.createBuffer(`${id}-${name}`, floats * 4);
    const outputs = {
      k: buffer('k', radiusCount),
      l: buffer('l', radiusCount),
      lMinusR: buffer('l-minus-r', radiusCount),
      radii: buffer('radii', radiusCount),
      g: buffer('g', radiusCount),
      f: buffer('f', radiusCount),
      j: buffer('j', radiusCount),
      nearest: buffer('nearest', subject.count),
      clarkEvans: buffer('clark-evans', GPU_CLARK_EVANS_LENGTH),
      quadratCounts: buffer('quadrat-counts', quadratCells),
      quadratStatistics: buffer('quadrat-statistics', GPU_QUADRAT_STATISTICS_LENGTH)
    };
    const graph = new GPUCommandGraph<void>(device, {id: `point-patterns-${id}`});
    const positions = importGraphBuffer(
      graph,
      'positions',
      subject.positionsBuffer,
      'float32x2',
      subject.count
    );
    const mask = importGraphBuffer(graph, 'mask', subject.maskBuffer, 'uint32', subject.count);
    const view = <F extends 'float32' | 'uint32'>(
      name: keyof typeof outputs,
      format: F,
      length: number
    ) => importGraphBuffer(graph, name, outputs[name], format, length);
    graph.add(
      new GPURipley({
        id: 'ripley',
        positions,
        mask,
        parameters: ripleyParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        radiusCount,
        k: view('k', 'float32', radiusCount),
        l: view('l', 'float32', radiusCount),
        lMinusR: view('lMinusR', 'float32', radiusCount),
        radii: view('radii', 'float32', radiusCount)
      })
    );
    graph.add(
      new GPURipleyDistanceFunctions({
        id: 'ripley-distance',
        positions,
        mask,
        parameters: distanceParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        referenceGrid,
        radiusCount,
        g: view('g', 'float32', radiusCount),
        f: view('f', 'float32', radiusCount),
        j: view('j', 'float32', radiusCount)
      })
    );
    graph.add(
      new GPUPointPatternIndices({
        id: 'indices',
        positions,
        mask,
        parameters: indicesParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        quadratGrid,
        nearestNeighborDistances: view('nearest', 'float32', subject.count),
        clarkEvans: view('clarkEvans', 'float32', GPU_CLARK_EVANS_LENGTH),
        quadratCounts: view('quadratCounts', 'uint32', quadratCells),
        quadratStatistics: view('quadratStatistics', 'float32', GPU_QUADRAT_STATISTICS_LENGTH)
      })
    );
    const compiled = resources.track(graph.compile());
    // Summary: L-r, G, F, J, radii, Clark-Evans, quadrat statistics, then quadrat counts.
    const summaryWords =
      radiusCount * 5 + GPU_CLARK_EVANS_LENGTH + GPU_QUADRAT_STATISTICS_LENGTH + quadratCells;
    const set: PatternSet = {
      key,
      subject,
      radiusCount,
      quadratGrid,
      compiled,
      nearestDistances: outputs.nearest,
      quadratCounts: outputs.quadratCounts,
      summaryWords,
      reader: new SummaryReader(
        resources,
        `summary-${id}`,
        [
          {buffer: outputs.lMinusR, size: radiusCount * 4},
          {buffer: outputs.g, size: radiusCount * 4},
          {buffer: outputs.f, size: radiusCount * 4},
          {buffer: outputs.j, size: radiusCount * 4},
          {buffer: outputs.radii, size: radiusCount * 4},
          {buffer: outputs.clarkEvans, size: GPU_CLARK_EVANS_LENGTH * 4},
          {buffer: outputs.quadratStatistics, size: GPU_QUADRAT_STATISTICS_LENGTH * 4},
          {buffer: outputs.quadratCounts, size: quadratCells * 4}
        ],
        bytes => handleSummary(set, bytes)
      )
    };
    return set;
  }

  function sparkline(values: ArrayLike<number>, low?: number, high?: number): string {
    let minimum = Infinity;
    let maximum = -Infinity;
    for (let index = 0; index < values.length; index++) {
      const value = values[index];
      if (!Number.isFinite(value)) continue;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    if (low !== undefined) minimum = low;
    if (high !== undefined) maximum = high;
    if (!(maximum >= minimum)) return 'no data';
    const span = maximum - minimum || 1;
    let text = '';
    for (let index = 0; index < values.length; index++) {
      const value = values[index];
      if (!Number.isFinite(value)) {
        text += '·';
        continue;
      }
      const level = Math.min(7, Math.max(0, Math.round(((value - minimum) / span) * 7)));
      text += SPARK_LEVELS[level];
    }
    return text;
  }

  const formatMeters = (value: number) =>
    !Number.isFinite(value)
      ? '-'
      : Math.abs(value) >= 1000
        ? `${(value / 1000).toFixed(2)} km`
        : `${value.toFixed(0)} m`;

  /** Upper tail of a chi-square statistic (Wilson-Hilferty). */
  function chiSquareUpperTail(statistic: number, degrees: number): number {
    if (!(degrees > 0) || !Number.isFinite(statistic)) return Number.NaN;
    const scale = 2 / (9 * degrees);
    const z = ((statistic / degrees) ** (1 / 3) - (1 - scale)) / Math.sqrt(scale);
    const t = 1 / (1 + 0.3275911 * Math.abs(z / Math.SQRT2));
    const polynomial =
      t *
      (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    const erfc = polynomial * Math.exp(-((z / Math.SQRT2) ** 2));
    return 0.5 * (z / Math.SQRT2 >= 0 ? erfc : 2 - erfc);
  }

  function handleSummary(set: PatternSet, bytes: ArrayBuffer): void {
    if (destroyed || set !== active) return;
    const floats = new Float32Array(bytes);
    const words = new Uint32Array(bytes);
    const count = set.radiusCount;
    const lMinusR = floats.subarray(0, count);
    const g = floats.subarray(count, count * 2);
    const f = floats.subarray(count * 2, count * 3);
    const j = floats.subarray(count * 3, count * 4);
    const radii = floats.subarray(count * 4, count * 5);
    const clarkOffset = count * 5;
    const quadratOffset = clarkOffset + GPU_CLARK_EVANS_LENGTH;
    const countsOffset = quadratOffset + GPU_QUADRAT_STATISTICS_LENGTH;
    const clark = floats.subarray(clarkOffset, quadratOffset);
    const quadrat = floats.subarray(quadratOffset, countsOffset);
    const options = ctx.options;
    const area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);

    ctx.setReadout('included', formatCount(clark[0]));
    ctx.setReadout(
      'area',
      `${(area / 1e6).toFixed(0)} km², ${(clark[0] / (area / 1e6)).toFixed(1)} per km²`
    );
    // L(r) - r
    let peakIndex = -1;
    let peakValue = -Infinity;
    for (let index = 0; index < count; index++) {
      if (Number.isFinite(lMinusR[index]) && lMinusR[index] > peakValue) {
        peakValue = lMinusR[index];
        peakIndex = index;
      }
    }
    const lSpark = sparkline(lMinusR);
    ctx.setReadout('lCurve', lSpark);
    ctx.setReadout(
      'lPeak',
      peakIndex >= 0
        ? `${peakValue >= 0 ? '+' : ''}${formatMeters(peakValue)} at r = ${formatMeters(radii[peakIndex])}`
        : 'undefined'
    );
    ctx.setReadout(
      'lReading',
      peakIndex < 0
        ? '-'
        : peakValue > 0
          ? 'clustered (L(r) > r)'
          : 'regular or edge-biased (L(r) < r)'
    );
    currentCurve = {label: categoryLabel(options), text: lSpark};
    ctx.setReadout(
      'previousCurve',
      previousCurve
        ? `${previousCurve.text}  ${previousCurve.label}`
        : 'change the selection to compare'
    );
    ctx.setReadout('gCurve', sparkline(g, 0, 1));
    ctx.setReadout('fCurve', sparkline(f, 0, 1));
    const jClamped = Array.from(j, value =>
      Number.isFinite(value) ? Math.min(Math.max(value, 0), 2) : Number.NaN
    );
    ctx.setReadout('jCurve', sparkline(jClamped, 0, 2));
    // Clark-Evans: [n, observed, expected, ratio, standardError, z]
    const ratio = clark[3];
    ctx.setReadout(
      'clarkEvans',
      Number.isFinite(ratio)
        ? `R = ${ratio.toFixed(3)} (${ratio < 1 ? 'clustered' : 'dispersed'}), z = ${clark[5].toFixed(1)}`
        : 'undefined'
    );
    ctx.setReadout(
      'nearest',
      `observed ${formatMeters(clark[1])}, expected under CSR ${formatMeters(clark[2])}`
    );
    // Quadrat: [count, mean, variance, VMR, chiSquare, df]
    const pValue = chiSquareUpperTail(quadrat[4], quadrat[5]);
    ctx.setReadout(
      'quadrat',
      Number.isFinite(quadrat[3])
        ? `VMR ${quadrat[3].toFixed(1)}, chi-square ${quadrat[4].toFixed(0)} (${quadrat[5].toFixed(0)} df), p ${pValue < 0.001 ? '< 0.001' : pValue.toFixed(3)}`
        : 'undefined'
    );
    // Display scales.
    let nextQuadratMax = 1;
    for (let index = 0; index < set.quadratGrid[0] * set.quadratGrid[1]; index++) {
      nextQuadratMax = Math.max(nextQuadratMax, words[countsOffset + index]);
    }
    const nextNearestMax = Number.isFinite(clark[2]) ? Math.max(10, clark[2] * 2) : 300;
    const quadratChanged = nextQuadratMax !== quadratMax;
    const nearestChanged = Math.abs(nextNearestMax - nearestMax) > nearestMax * 0.1;
    quadratMax = nextQuadratMax;
    if (nearestChanged) nearestMax = nextNearestMax;
    ctx.setLegendExtent('quadrat', [0, quadratMax]);
    ctx.setLegendExtent('nearest', [0, nearestMax]);
    if (quadratChanged || nearestChanged) ctx.requestLayers();
  }

  function writeMask(): void {
    const options = ctx.options;
    const subject = subjectOf(options);
    const category = categoryOf(options);
    const wanted = category === 'all' ? -1 : subject.categoryNames.indexOf(category);
    const mask = new Uint32Array(subject.count);
    for (let index = 0; index < subject.count; index++) {
      mask[index] = wanted < 0 || subject.category[index] === wanted ? 1 : 0;
    }
    subject.maskBuffer.write(mask);
    dirty = true;
    lastChangeTime = performance.now();
  }

  function writeJitter(subject: SubjectData, radius: number): void {
    if (subject.jitterApplied === radius) return;
    subject.jitterApplied = radius;
    if (radius === 0) {
      subject.positionsBuffer.write(subject.base);
    } else {
      const random = createSeededRandom(subject.id === 'observations' ? 11 : 23);
      const jittered = new Float32Array(subject.base.length);
      for (let index = 0; index < subject.count; index++) {
        const angle = random() * Math.PI * 2;
        const distance = Math.sqrt(random()) * radius;
        jittered[index * 2] = subject.base[index * 2] + Math.cos(angle) * distance;
        jittered[index * 2 + 1] = subject.base[index * 2 + 1] + Math.sin(angle) * distance;
      }
      subject.positionsBuffer.write(jittered);
    }
    dirty = true;
    lastChangeTime = performance.now();
  }

  function writeWindow(): void {
    const options = ctx.options;
    const maximumDistance = options.maximumDistance;
    const safe = bounds;
    const ripleyOptions = {bounds: safe, maximumDistance};
    ripleyParameters.write(
      getGPURipleyParameterValues({...ripleyOptions, edgeCorrection: options.ripleyCorrection})
    );
    distanceParameters.write(
      getGPURipleyDistanceParameterValues({
        ...ripleyOptions,
        edgeCorrection: options.distanceCorrection
      })
    );
    indicesParameters.write(
      getGPUPointPatternIndicesParameterValues({
        bounds: safe,
        maximumDistance: Math.min(maximumDistance, 400)
      })
    );
    windowBuffer.write(Float32Array.from(safe));
    const [x0, y0, x1, y1] = safe;
    windowSegments.write(
      Float32Array.of(x0, y0, x1, y0, x1, y0, x1, y1, x1, y1, x0, y1, x0, y1, x0, y0)
    );
    dirty = true;
    lastChangeTime = performance.now();
  }

  function selectSet(): void {
    const key = getKey(ctx.options);
    if (key === displayedKey && active) return;
    let next = sets.get(key);
    if (!next) {
      next = buildSet(ctx.options);
      sets.set(key, next);
    }
    active = next;
    displayedKey = key;
    dirty = true;
    lastChangeTime = performance.now();
  }

  /** Remembers the curve being replaced so the next selection can be compared with it. */
  function rememberCurve(): void {
    if (currentCurve) previousCurve = currentCurve;
  }

  writeMask();
  writeJitter(
    subjects.observations,
    ctx.options.subject === 'observations' ? ctx.options.jitter : 0
  );
  writeJitter(subjects.places, ctx.options.subject === 'places' ? ctx.options.jitter : 0);
  writeWindow();
  selectSet();

  async function measurePattern(): Promise<void> {
    const set = active;
    if (!set || measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('timing', 'measuring…');
    try {
      const timing = await measureCompiledGraph(device, set.compiled, {
        parameters: undefined,
        completionBuffer: set.quadratCounts,
        signal: ctx.signal
      });
      if (!destroyed) {
        ctx.setReadout(
          'timing',
          `${formatCompiledGraphTiming(timing).split(' · ')[0]} for ${formatCount(set.subject.count)} points, r up to ${formatMeters(ctx.options.maximumDistance)}`
        );
      }
    } catch {
      // Interrupted.
    } finally {
      measuring = false;
    }
  }

  return {
    getCompiledGraphs: () =>
      (active ? [active.compiled] : []) as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id, _value, state) {
      if (id === 'subject' || id === 'groupCategory' || id === 'placeCategory') {
        rememberCurve();
        selectSet();
        writeJitter(subjectOf(state), state.jitter);
        writeMask();
        ctx.requestLayers();
      } else if (id === 'radiusCount' || id === 'quadratGrid' || id === 'referenceGrid') {
        selectSet();
        ctx.requestLayers();
      } else if (id === 'jitter') {
        writeJitter(subjectOf(state), state.jitter);
      } else if (id === 'window') {
        if (state.window === 'city') bounds = [...cityBounds];
        lastViewBounds = null;
        writeWindow();
        ctx.requestLayers();
      } else if (['maximumDistance', 'ripleyCorrection', 'distanceCorrection'].includes(id)) {
        writeWindow();
      } else {
        ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'measure') void measurePattern();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const set = active;
      if (!set) return;
      if (ctx.options.window === 'view') {
        const viewBounds = getViewportMetricBounds(frame.viewport, projection);
        const signature = viewBounds.map(value => Math.round(value / 5)).join(',');
        if (signature !== lastViewBounds) {
          lastViewBounds = signature;
          bounds = [...viewBounds];
          writeWindow();
        }
      }
      const settled = performance.now() - lastChangeTime > SETTLE_MILLISECONDS;
      if (dirty && settled) {
        dirty = false;
        set.compiled.encode(commandEncoder, {parameters: undefined});
        set.reader.request(commandEncoder);
        return;
      }
      set.reader.flush(commandEncoder);
    },

    getLayers() {
      const set = active;
      if (!set) return [];
      const options = ctx.options;
      const subject = set.subject;
      const dark = ctx.theme() === 'dark';
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const layers: Layer[] = [];
      if (options.showQuadrats) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `pattern-quadrats-${set.key}`,
            coordinateOrigin: [origin[0], origin[1], 0],
            gridSize: set.quadratGrid,
            bounds: windowBuffer.buffer,
            values: set.quadratCounts,
            valueFormat: 'uint32',
            colormap: 'magma',
            valueRange: [0, quadratMax],
            sqrtScale: true,
            discardAtOrBelow: 0,
            color: [255, 255, 255, 150]
          })
        );
      }
      const radiusPixels = subject.count > 100000 ? 1.4 : 2;
      if (options.colorPoints === 'nearest') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `pattern-nearest-${subject.id}`,
            coordinateOrigin,
            positions: subject.positionsBuffer,
            instanceCount: subject.count,
            values: set.nearestDistances,
            valueFormat: 'float32',
            colormap: 'viridis',
            valueRange: [0, nearestMax],
            sqrtScale: true,
            noDataColor: [0, 0, 0, 0],
            radiusPixels: radiusPixels + 0.6
          })
        );
      } else {
        if (options.showOthers) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: `pattern-others-${subject.id}`,
              coordinateOrigin,
              positions: subject.positionsBuffer,
              instanceCount: subject.count,
              radiusPixels: 1,
              color: dark ? [140, 150, 170, 40] : [90, 100, 120, 40]
            })
          );
        }
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `pattern-selected-${subject.id}`,
            coordinateOrigin,
            positions: subject.positionsBuffer,
            instanceCount: subject.count,
            values: subject.maskBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: subject.id === 'observations' ? [96, 200, 110, 210] : [64, 196, 255, 210],
            noDataColor: [0, 0, 0, 0],
            radiusPixels: radiusPixels + 0.4
          })
        );
      }
      if (options.showWindow) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'pattern-window',
            coordinateOrigin,
            segments: windowSegments,
            instanceCount: 4,
            widthPixels: 2,
            color: dark ? [255, 255, 255, 190] : [20, 20, 40, 190]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const set of sets.values()) set.reader.stop();
      resources.destroy();
    }
  };
}
