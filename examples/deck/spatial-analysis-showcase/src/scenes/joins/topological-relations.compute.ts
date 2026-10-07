// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  formatGPUSpatialRelate,
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin,
  packGPUSpatialRelatePattern,
  type GPUSpatialJoinGeometry,
  type GPUSpatialPredicate
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer, type SpatialAnalysisColormap} from '../../engine/layers';
import type {RampName} from '../../engine/ramps';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createLineSet,
  createPolygonSet,
  findLineNear,
  findPolygonAt,
  formatInteger,
  type LineSet,
  type PolygonSet
} from './b2-geometry';
import {
  createZoneRaster,
  importLines,
  importPolygons,
  uploadLines,
  uploadPolygons,
  type LineBuffers,
  type PolygonBuffers,
  type ZoneRaster
} from './b2-graph';
import {ZoneFillLayer} from './b2-layers';

/** Option state of the topological-relations scene. */
export type TopologicalOptions = {
  combo: 'tracts-areas' | 'roads-tracts' | 'tracts-self';
  predicate:
    | 'intersects'
    | 'contains'
    | 'within'
    | 'covers'
    | 'coveredBy'
    | 'touches'
    | 'crosses'
    | 'overlaps'
    | 'equals'
    | 'containsProperly'
    | 'dwithin'
    | 'relate';
  pattern: string;
  distance: number;
  engine: 'auto' | 'fast' | 'relate';
  matrix: boolean;
  show: 'status' | 'count' | 'anti';
  ramp: RampName;
  opacity: number;
  showRight: boolean;
};

/** DE-9IM pattern presets of the `relate` predicate (any-of lists). */
export const RELATE_PATTERNS: Record<string, {label: string; patterns: readonly string[]}> = {
  rook: {label: 'Shared edge (rook contiguity)', patterns: ['F***1****']},
  queen: {
    label: 'Any shared boundary point (queen contiguity)',
    patterns: ['FT*******', 'F**T*****', 'F***T****']
  },
  corner: {label: 'Meet at a single point only', patterns: ['F***0****']},
  interiors: {label: 'Interiors intersect', patterns: ['T********']},
  straddle: {label: 'Straddle: overlap and extend outside both', patterns: ['T*T***T**']},
  inside: {label: 'Completely inside (within)', patterns: ['T*F**F***']},
  crossing: {label: 'Interior crossing: through and out', patterns: ['T*T******']}
};

const PATTERN_SLOTS = 4;
const MAXIMUM_DISTANCE = 500;
const PREDICATES_WITH_FAST_ENGINE = ['intersects', 'contains', 'within', 'dwithin'];
const SETTLE_FRAMES = 3;
const STATUS_PALETTE = [
  [120, 135, 160, 120],
  [255, 150, 40, 230],
  [40, 200, 255, 255],
  [90, 220, 120, 235]
] as const;

type Variant = {
  key: string;
  combo: TopologicalOptions['combo'];
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  prepared: GPUSpatialJoinPrepared;
  reader: SummaryReader;
  pairCapacity: number;
  hasMatrix: boolean;
  engineText: string;
  leftCount: number;
  rightCount: number;
  withWeights: boolean;
  pairCount: Buffer;
  frames: number;
  predicate: GPUSpatialPredicate;
};

/**
 * Exact topological relations between three pairs of Chicago layers. Each variant compiles one
 * graph with a shared `GPUSpatialJoinPrepared` right-hand index feeding an inner and an anti
 * `GPUSpatialPredicateJoin` (and, for the self join, the weights output). The predicate, engine
 * and matrix output are compile-time; the DE-9IM pattern and the `dwithin` distance are per-frame
 * buffers, so the pattern presets and the distance slider never rebuild.
 */
export async function createTopologicalRelations(
  ctx: SceneContext<TopologicalOptions>
): Promise<SceneInstance<TopologicalOptions>> {
  const {device} = ctx;
  const tractsData = ctx.datasets.get('chicago-tracts');
  const areasData = ctx.datasets.get('chicago-community-areas');
  const roadsData = ctx.datasets.get('chicago-roads');
  const origin = tractsData.defaultOrigin;
  const projection = tractsData.getProjection(origin);

  const tractSet = createPolygonSet(tractsData, origin);
  const areaSet = createPolygonSet(areasData, origin);
  const tractCommunityArea = tractsData.column<Uint8Array>('communityArea');
  const geoids = (tractsData.geojson?.features ?? []).map(feature =>
    String((feature.properties as Record<string, unknown>)?.GEOID ?? '')
  );
  const areaNames = (areasData.geojson?.features ?? []).map(feature =>
    String((feature.properties as Record<string, unknown>)?.name ?? '')
  );

  // Roads: one direction of every road up to tertiary, as polylines.
  const edgeVertices = roadsData.projectColumn('edgeVertices', origin);
  const edgePathOffsets = roadsData.column<Uint32Array>('edgePathOffsets');
  const edgeClass = roadsData.column<Uint8Array>('edgeClass');
  const edgeReverse = roadsData.column<Uint32Array>('edgeReverse');
  const classNames = roadsData.categories('edgeClass');
  const polylines: Float32Array[] = [];
  const roadEdge: number[] = [];
  for (let edge = 0; edge < edgeClass.length; edge++) {
    if (edgeClass[edge] > 4) continue;
    const reverse = edgeReverse[edge];
    if (reverse < edgeClass.length && reverse < edge) continue;
    polylines.push(edgeVertices.slice(edgePathOffsets[edge] * 2, edgePathOffsets[edge + 1] * 2));
    roadEdge.push(edge);
  }
  const roadSet: LineSet = createLineSet(polylines);

  ctx.setStatus('Rasterizing the tract fill');
  const resources = new SpatialAnalysisResources(device, 'topological-relations');
  const tractBuffers = uploadPolygons(resources, 'tracts', tractSet);
  const areaBuffers = uploadPolygons(resources, 'areas', areaSet);
  const roadBuffers: LineBuffers = uploadLines(resources, 'roads', roadSet);
  const tractRaster: ZoneRaster = await createZoneRaster(
    resources,
    'tract-fill',
    tractBuffers,
    tractSet.bounds,
    1200
  );
  ctx.signal.throwIfAborted();

  const patternParameters = resources.createParameterBuffer(
    'pattern',
    'uint32',
    PATTERN_SLOTS * 2,
    packGPUSpatialRelatePattern(RELATE_PATTERNS.rook.patterns, PATTERN_SLOTS)
  );
  const distanceParameters = resources.createParameterBuffer(
    'distance',
    'float32',
    1,
    Float32Array.of(ctx.options.distance)
  );
  const maximumLeft = Math.max(tractSet.featureCount, roadSet.featureCount);
  const statusBuffer = resources.createBuffer('status', new Uint32Array(maximumLeft));
  const countBuffer = resources.createBuffer('counts', new Float32Array(maximumLeft));
  const rightStatusBuffer = resources.createBuffer(
    'right-status',
    new Uint32Array(Math.max(tractSet.featureCount, areaSet.featureCount))
  );

  // --- state -----------------------------------------------------------------------------------
  let destroyed = false;
  let variant: Variant;
  let dirty = true;
  let settle = 0;
  let measuring = false;
  let selected = -1;
  let leftCounts = new Float32Array(maximumLeft);
  let antiFlags = new Uint8Array(maximumLeft);
  let pairs: {left: number; right: number; matrix: number}[] = [];
  let maximumCount = 1;

  const getKey = (state: TopologicalOptions): string =>
    `${state.combo}|${state.predicate}|${PREDICATES_WITH_FAST_ENGINE.includes(state.predicate) ? state.engine : '-'}|${state.matrix && state.predicate !== 'dwithin'}`;

  const getSides = (combo: TopologicalOptions['combo']) =>
    combo === 'tracts-areas'
      ? {left: tractBuffers, right: areaBuffers, leftIsLines: false}
      : combo === 'roads-tracts'
        ? {left: roadBuffers, right: tractBuffers, leftIsLines: true}
        : {left: tractBuffers, right: tractBuffers, leftIsLines: false};

  const getLeftCount = (combo: TopologicalOptions['combo']) =>
    combo === 'roads-tracts' ? roadSet.featureCount : tractSet.featureCount;
  const getRightCount = (combo: TopologicalOptions['combo']) =>
    combo === 'tracts-areas' ? areaSet.featureCount : tractSet.featureCount;

  const getPatterns = (state: TopologicalOptions): readonly string[] =>
    (RELATE_PATTERNS[state.pattern] ?? RELATE_PATTERNS.rook).patterns;

  /** Builds one compiled configuration. `engineOverride` is used by the timing comparison only. */
  function buildVariant(
    state: TopologicalOptions,
    engineOverride?: 'fast' | 'relate',
    idSuffix = ''
  ): Variant {
    const key = `${getKey(state)}${idSuffix}`;
    const own = new SpatialAnalysisResources(device, `tr-${key}`);
    const graph = new GPUCommandGraph<void>(device, {id: `tr-${key}`});
    const sides = getSides(state.combo);
    const leftViews: GPUSpatialJoinGeometry = sides.leftIsLines
      ? importLines(graph, 'left', sides.left as LineBuffers)
      : importPolygons(graph, 'left', sides.left as PolygonBuffers);
    const rightViews: GPUSpatialJoinGeometry =
      state.combo === 'tracts-self'
        ? leftViews
        : importPolygons(graph, 'right', sides.right as PolygonBuffers);
    const leftCount = getLeftCount(state.combo);
    const rightCount = getRightCount(state.combo);
    const predicate = state.predicate;
    const hasMatrix = state.matrix && predicate !== 'dwithin';
    const self = state.combo === 'tracts-self';
    const pairCapacity = Math.max(1024, leftCount * (sides.leftIsLines ? 6 : 24));
    const candidateCapacity = pairCapacity * 2;
    const engine = engineOverride ?? state.engine;

    const prepared = new GPUSpatialJoinPrepared({id: 'right-index', geometry: rightViews});
    graph.add(prepared);

    const leftIds = own.createBuffer('left-ids', pairCapacity * 4);
    const rightIds = own.createBuffer('right-ids', pairCapacity * 4);
    const pairCount = own.createBuffer('pair-count', 4);
    const pairOverflow = own.createBuffer('pair-overflow', 4);
    const pairTotal = own.createBuffer('pair-total', 4);
    const relateBuffer = own.createBuffer('relate', pairCapacity * 4);
    const uncertain = own.createBuffer('uncertain', 4);
    const candidates = own.createBuffer('candidates', 4);
    const antiIds = own.createBuffer('anti-ids', leftCount * 4);
    const antiCount = own.createBuffer('anti-count', 4);
    const antiOverflow = own.createBuffer('anti-overflow', 4);
    const csrOffsets = own.createBuffer('csr-offsets', (leftCount + 1) * 4);
    const csrNeighbors = own.createBuffer('csr-neighbors', pairCapacity * 4);
    const csrWeights = own.createBuffer('csr-weights', pairCapacity * 4);

    const common = {
      left: leftViews,
      right: rightViews,
      predicate,
      ...(predicate === 'relate' ? {pattern: patternParameters.importToGraph(graph)} : {}),
      ...(predicate === 'dwithin' ? {distance: distanceParameters.importToGraph(graph)} : {}),
      ...(PREDICATES_WITH_FAST_ENGINE.includes(predicate) ? {engine} : {}),
      ...(self ? {excludeSameRow: true} : {}),
      candidateCapacity,
      prepared
    };
    const innerJoin = new GPUSpatialPredicateJoin({
      ...common,
      id: 'inner',
      pairs: {
        leftIds: importGraphBuffer(graph, 'left-ids', leftIds, 'uint32', pairCapacity),
        rightIds: importGraphBuffer(graph, 'right-ids', rightIds, 'uint32', pairCapacity),
        count: importGraphBuffer(graph, 'pair-count', pairCount, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'pair-overflow', pairOverflow, 'uint32', 1),
        totalCount: importGraphBuffer(graph, 'pair-total', pairTotal, 'uint32', 1)
      },
      ...(hasMatrix
        ? {relate: importGraphBuffer(graph, 'relate', relateBuffer, 'uint32', pairCapacity)}
        : {}),
      ...(self
        ? {
            weights: {
              offsets: importGraphBuffer(graph, 'csr-offsets', csrOffsets, 'uint32', leftCount + 1),
              neighbors: importGraphBuffer(
                graph,
                'csr-neighbors',
                csrNeighbors,
                'uint32',
                pairCapacity
              ),
              weights: importGraphBuffer(graph, 'csr-weights', csrWeights, 'float32', pairCapacity)
            }
          }
        : {}),
      uncertainCount: importGraphBuffer(graph, 'uncertain', uncertain, 'uint32', 1),
      candidateCount: importGraphBuffer(graph, 'candidates', candidates, 'uint32', 1)
    });
    graph.add(innerJoin);
    const antiJoin = new GPUSpatialPredicateJoin({
      ...common,
      id: 'anti',
      how: 'anti',
      unmatched: {
        ids: importGraphBuffer(graph, 'anti-ids', antiIds, 'uint32', leftCount),
        count: importGraphBuffer(graph, 'anti-count', antiCount, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'anti-overflow', antiOverflow, 'uint32', 1)
      }
    });
    graph.add(antiJoin);

    const sources = [
      {buffer: pairCount, size: 4},
      {buffer: pairOverflow, size: 4},
      {buffer: pairTotal, size: 4},
      {buffer: uncertain, size: 4},
      {buffer: candidates, size: 4},
      {buffer: antiCount, size: 4},
      {buffer: antiOverflow, size: 4},
      {buffer: leftIds, size: pairCapacity * 4},
      {buffer: rightIds, size: pairCapacity * 4},
      {buffer: relateBuffer, size: pairCapacity * 4},
      {buffer: antiIds, size: leftCount * 4},
      {buffer: csrOffsets, size: (leftCount + 1) * 4}
    ];
    const result: Variant = {
      key,
      combo: state.combo,
      resources: own,
      compiled: undefined as never,
      prepared,
      reader: undefined as never,
      pairCapacity,
      hasMatrix,
      engineText: `inner ${describeEngine(innerJoin)}${hasMatrix ? ' (matrix output)' : ''}, anti ${describeEngine(antiJoin)}`,
      leftCount,
      rightCount,
      withWeights: self,
      pairCount,
      frames: 0,
      predicate
    };
    result.reader = new SummaryReader(own, `summary-${key}`, sources, bytes => {
      if (destroyed || variant !== result) return;
      handleSummary(result, bytes);
    });
    result.compiled = own.track(graph.compile());
    own.track({destroy: () => prepared.destroy()});
    return result;
  }

  function describeEngine(join: GPUSpatialPredicateJoin): string {
    if (join.usesRelateEngine) return 'relate engine';
    return join.usesWorkgroupDistance ? 'workgroup distance' : 'fast kernel';
  }

  const getRightName = (combo: TopologicalOptions['combo'], row: number): string =>
    combo === 'tracts-areas' ? (areaNames[row] ?? `Area ${row + 1}`) : `Tract ${geoids[row]}`;
  const getLeftName = (combo: TopologicalOptions['combo'], row: number): string =>
    combo === 'roads-tracts'
      ? `${classNames[edgeClass[roadEdge[row]]] ?? 'road'} edge ${roadEdge[row]}`
      : `Tract ${geoids[row]}${areaNames[tractCommunityArea[row] - 1] ? ` (${areaNames[tractCommunityArea[row] - 1]})` : ''}`;

  /** Recomputes the status and count buffers from the CPU copy of the join result. */
  function refreshStatus(): void {
    const state = ctx.options;
    const leftCount = variant.leftCount;
    const status = new Uint32Array(leftCount);
    const rightStatus = new Uint32Array(variant.rightCount);
    const related = new Set<number>();
    for (const pair of pairs) {
      if (pair.left === selected) {
        rightStatus[pair.right] = 1;
        related.add(pair.right);
      }
    }
    for (let row = 0; row < leftCount; row++) {
      const hasResult = state.show === 'anti' ? antiFlags[row] === 1 : leftCounts[row] > 0;
      status[row] =
        row === selected
          ? 2
          : variant.combo === 'tracts-self' && related.has(row)
            ? 3
            : hasResult
              ? 1
              : 0;
    }
    statusBuffer.write(status);
    countBuffer.write(leftCounts.subarray(0, leftCount));
    rightStatusBuffer.write(rightStatus);
    ctx.requestLayers();
  }

  function handleSummary(source: Variant, bytes: ArrayBuffer): void {
    const state = ctx.options;
    const words = new Uint32Array(bytes);
    const capacity = source.pairCapacity;
    const leftCount = source.leftCount;
    const rightCount = source.rightCount;
    const count = Math.min(words[0], capacity);
    const lefts = words.subarray(7, 7 + capacity);
    const rights = words.subarray(7 + capacity, 7 + capacity * 2);
    const matrices = words.subarray(7 + capacity * 2, 7 + capacity * 3);
    const antiIds = words.subarray(7 + capacity * 3, 7 + capacity * 3 + leftCount);
    const csrOffsets = words.subarray(
      7 + capacity * 3 + leftCount,
      7 + capacity * 3 + leftCount * 2 + 1
    );

    leftCounts = new Float32Array(maximumLeft);
    antiFlags = new Uint8Array(maximumLeft);
    const rightCounts = new Uint32Array(rightCount);
    pairs = [];
    const histogram = new Map<string, number>();
    for (let slot = 0; slot < count; slot++) {
      if (lefts[slot] >= leftCount || rights[slot] >= rightCount) continue;
      leftCounts[lefts[slot]]++;
      rightCounts[rights[slot]]++;
      pairs.push({left: lefts[slot], right: rights[slot], matrix: matrices[slot]});
      if (source.hasMatrix) {
        const text = formatGPUSpatialRelate(matrices[slot] & 0x3ffff);
        histogram.set(text, (histogram.get(text) ?? 0) + 1);
      }
    }
    const antiCount = Math.min(words[5], leftCount);
    for (let slot = 0; slot < antiCount; slot++) {
      if (antiIds[slot] < leftCount) antiFlags[antiIds[slot]] = 1;
    }
    let matchedLeft = 0;
    let complement = true;
    maximumCount = 1;
    for (let row = 0; row < leftCount; row++) {
      if (leftCounts[row] > 0) matchedLeft++;
      if (leftCounts[row] > 0 === (antiFlags[row] === 1)) complement = false;
      maximumCount = Math.max(maximumCount, leftCounts[row]);
    }
    const matchedRight = rightCounts.reduce((total, value) => total + (value > 0 ? 1 : 0), 0);
    const describe = describePredicate(state);
    ctx.setReadout(
      'left',
      `${formatInteger(leftCount)} ${source.combo === 'roads-tracts' ? 'road edges' : 'tracts'}`
    );
    ctx.setReadout(
      'right',
      `${formatInteger(rightCount)} ${source.combo === 'tracts-areas' ? 'community areas' : 'tracts'}`
    );
    ctx.setReadout(
      'pairs',
      `${formatInteger(words[2])} pairs: ${formatInteger(matchedLeft)} of ${formatInteger(leftCount)} left, ${formatInteger(matchedRight)} of ${formatInteger(rightCount)} right`
    );
    ctx.setReadout(
      'anti',
      `${formatInteger(words[5])} of ${formatInteger(leftCount)}; ${complement ? 'exactly the complement of the pairs' : 'DIFFERS from the complement'}`
    );
    ctx.setReadout(
      'candidates',
      `${formatInteger(words[4])} bounding-box candidates, ${formatInteger(words[2])} pass ${describe} (${words[4] ? ((100 * words[2]) / words[4]).toFixed(0) : 0}%)`
    );
    ctx.setReadout(
      'flags',
      `${words[1] || words[6] ? 'OVERFLOW' : 'no overflow'}; ${formatInteger(words[3])} uncertain pairs`
    );
    ctx.setReadout('engine', source.engineText);
    ctx.setReadout(
      'matrices',
      source.predicate === 'dwithin'
        ? 'none: dwithin has no matrix'
        : source.hasMatrix
          ? [...histogram.entries()]
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3)
              .map(([text, total]) => `${text} x${formatInteger(total)}`)
              .join(', ') || 'none'
          : 'enable "Output DE-9IM matrices"'
    );
    if (source.withWeights) {
      const nonZero = csrOffsets[leftCount];
      let islands = 0;
      for (let row = 0; row < leftCount; row++) if (leftCounts[row] === 0) islands++;
      const mean = nonZero / leftCount;
      ctx.setReadout(
        'contiguity',
        `${formatInteger(nonZero)} links in the weights matrix, ${mean.toFixed(2)} neighbours per tract on average, ${islands} island${islands === 1 ? '' : 's'}`
      );
      ctx.setReadout(
        'parity',
        'libpysal on the same tracts: queen 6.6 and rook 4.7 neighbours on average, 1 island each'
      );
    } else {
      ctx.setReadout('contiguity', 'self join only');
      ctx.setReadout('parity', 'self join only');
    }
    const builds = source.prepared.encodedBuildCount;
    ctx.setReadout(
      'indexBuilds',
      `${builds} build${builds === 1 ? '' : 's'} in ${formatInteger(source.frames)} join runs (right index reused)`
    );
    refreshSelection();
    refreshStatus();
  }

  function describePredicate(state: TopologicalOptions): string {
    if (state.predicate === 'relate') return `pattern ${getPatterns(state).join(' | ')}`;
    if (state.predicate === 'dwithin') return `dwithin ${state.distance} m`;
    return state.predicate;
  }

  function refreshSelection(): void {
    if (selected < 0) {
      ctx.setReadout('selection', 'Click a feature on the map');
      return;
    }
    const combo = variant.combo;
    const matches = pairs.filter(pair => pair.left === selected);
    const lines = matches.slice(0, 4).map(pair => {
      const text = variant.hasMatrix ? ` ${formatGPUSpatialRelate(pair.matrix & 0x3ffff)}` : '';
      return `${getRightName(combo, pair.right)}${text}`;
    });
    ctx.setReadout(
      'selection',
      `${getLeftName(combo, selected)}: ${matches.length} match${matches.length === 1 ? '' : 'es'}${lines.length ? ` (${lines.join('; ')}${matches.length > 4 ? '; ...' : ''})` : ''}`
    );
  }

  function replaceVariant(state: TopologicalOptions): void {
    const previous = variant;
    selected = -1;
    pairs = [];
    leftCounts = new Float32Array(maximumLeft);
    antiFlags = new Uint8Array(maximumLeft);
    variant = buildVariant(state);
    dirty = true;
    if (previous) {
      previous.reader.stop();
      requestAnimationFrame(() => requestAnimationFrame(() => previous.resources.destroy()));
    }
  }

  function writePatterns(state: TopologicalOptions): void {
    patternParameters.write(packGPUSpatialRelatePattern(getPatterns(state), PATTERN_SLOTS));
  }

  writePatterns(ctx.options);
  variant = buildVariant(ctx.options);

  /** Times the fast and the relate engine on the current predicate, outside the frame. */
  async function measureEngines(): Promise<void> {
    if (measuring || destroyed) return;
    const state = ctx.options;
    if (!PREDICATES_WITH_FAST_ENGINE.includes(state.predicate)) {
      ctx.setReadout('timing', 'engine choice applies to intersects, contains, within and dwithin');
      return;
    }
    measuring = true;
    ctx.setReadout('timing', 'measuring...');
    const results: string[] = [];
    try {
      for (const engine of ['fast', 'relate'] as const) {
        const built = buildVariant({...state, matrix: false}, engine, `-time-${engine}`);
        try {
          const timing = await measureCompiledGraph(device, built.compiled, {
            parameters: undefined,
            completionBuffer: built.pairCount,
            signal: ctx.signal
          });
          results.push(`${engine} ${formatCompiledGraphTiming(timing).split(' · ')[0]}`);
        } finally {
          built.compiled.destroy();
          built.reader.stop();
          built.resources.destroy();
        }
      }
      if (!destroyed) ctx.setReadout('timing', results.join(' | '));
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
      dirty = true;
    }
  }

  return {
    getCompiledGraphs: () => [variant.compiled as CompiledGPUCommandGraph<never>],

    setOption(id, _value, state) {
      switch (id) {
        case 'combo':
        case 'predicate':
        case 'engine':
        case 'matrix':
          if (getKey(state) !== variant.key) replaceVariant(state);
          break;
        case 'pattern':
          writePatterns(state);
          dirty = true;
          break;
        case 'distance':
          dirty = true;
          break;
        case 'show':
          refreshStatus();
          break;
        default:
          break;
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'measure') void measureEngines();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const row =
        variant.combo === 'roads-tracts'
          ? findLineNear(roadSet, x, y, 40)
          : findPolygonAt(tractSet, x, y);
      selected = row === selected ? -1 : row;
      refreshSelection();
      refreshStatus();
      return true;
    },

    getTooltip(event) {
      if (!event.coordinate || variant.combo === 'roads-tracts') return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const row = findPolygonAt(tractSet, x, y);
      if (row < 0) return null;
      return `${getLeftName(variant.combo, row)}\n${formatInteger(leftCounts[row])} match${leftCounts[row] === 1 ? '' : 'es'}`;
    },

    encode(commandEncoder) {
      const state = ctx.options;
      distanceParameters.write(Float32Array.of(Math.min(state.distance, MAXIMUM_DISTANCE)));
      if (dirty) {
        variant.compiled.encode(commandEncoder, {parameters: undefined});
        variant.frames++;
        dirty = false;
        settle = SETTLE_FRAMES;
      } else if (settle > 0) {
        settle--;
        if (settle === 0) variant.reader.request(commandEncoder);
      }
      variant.reader.flush(commandEncoder);
    },

    getLayers() {
      const state = ctx.options;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const sides = getSides(variant.combo);
      if (!sides.leftIsLines) {
        layers.push(
          new ZoneFillLayer({
            id: 'left-fill',
            coordinateOrigin,
            gridSize: [tractRaster.width, tractRaster.height],
            bounds: tractRaster.bounds,
            rowOrigin: 'south',
            valueIndices: tractRaster.zones,
            values: state.show === 'count' ? countBuffer : statusBuffer,
            valueFormat: state.show === 'count' ? 'float32' : 'uint32',
            colormap: state.show === 'count' ? (state.ramp as SpatialAnalysisColormap) : 'category',
            valueRange: [0, maximumCount],
            palette: STATUS_PALETTE,
            opacity: state.opacity
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'left-outline',
            coordinateOrigin,
            segments: tractBuffers.outline,
            instanceCount: tractSet.outline.length / 4,
            color: dark ? [235, 240, 250, 90] : [20, 30, 50, 100],
            widthPixels: 0.7
          })
        );
      } else {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'tract-context',
            coordinateOrigin,
            segments: tractBuffers.outline,
            instanceCount: tractSet.outline.length / 4,
            color: dark ? [235, 240, 250, 70] : [20, 30, 50, 80],
            widthPixels: 0.8
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'road-lines',
            coordinateOrigin,
            segments: roadBuffers.segments,
            instanceCount: roadSet.segments.length / 4,
            valueIndices: roadBuffers.segmentRows,
            values: state.show === 'count' ? countBuffer : statusBuffer,
            valueFormat: state.show === 'count' ? 'float32' : 'uint32',
            colormap: state.show === 'count' ? (state.ramp as SpatialAnalysisColormap) : 'category',
            valueRange: [0, maximumCount],
            palette: STATUS_PALETTE,
            widthPixels: 1.8
          })
        );
      }
      if (state.showRight && variant.combo !== 'tracts-self' && variant.combo !== 'roads-tracts') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'right-outline',
            coordinateOrigin,
            segments: areaBuffers.outline,
            instanceCount: areaSet.outline.length / 4,
            color: dark ? [255, 220, 120, 220] : [150, 80, 0, 220],
            widthPixels: 2.2
          })
        );
      }
      // Right features related to the selected left feature.
      if (selected >= 0) {
        const rightBuffers = variant.combo === 'tracts-areas' ? areaBuffers : tractBuffers;
        const rightSet: PolygonSet = variant.combo === 'tracts-areas' ? areaSet : tractSet;
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'right-selected',
            coordinateOrigin,
            segments: rightBuffers.outline,
            instanceCount: rightSet.outline.length / 4,
            valueIndices: rightBuffers.outlineRows,
            values: rightStatusBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [60, 230, 255, 255],
            noDataColor: [0, 0, 0, 0],
            widthPixels: 3.5
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      variant.reader.stop();
      variant.resources.destroy();
      resources.destroy();
    }
  };
}
