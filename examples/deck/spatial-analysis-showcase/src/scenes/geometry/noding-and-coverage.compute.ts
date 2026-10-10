// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPULineSimplificationParameterValues,
  GPUCoverageSimplification,
  GPULineMerge,
  GPULineSimplification,
  GPULineSplit,
  GPUSegmentIntersection,
  GPUSegmentRingAssembly,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkNoding} from '@luma.gl/experimental/gpu-network';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {loadRailPaths, loadStreetPaths, type PathSet} from './b3-city-data';
import {
  buildRingEdges,
  copyCountToDrawRecord,
  createGraphImporter,
  createStaticPaths,
  formatDistance,
  loadPolygonLayout,
  triangulatePolygons,
  type GeometryView,
  type PolygonLayout
} from './b3-common';
import {
  FeatureTriangleLayer,
  KeptSegmentLayer,
  PairSegmentLayer,
  PathOutputLayer
} from './b3-layers';
import {buildDirectedEdges, buildNodingLines} from './b3-noding-data';
import type {NodingAndCoverageOptions} from './b3-noding-options';

export type {NodingAndCoverageOptions};

type Options = NodingAndCoverageOptions;
type View = GeometryView<Options>;

const LOCAL = COORDINATE_SYSTEM.METER_OFFSETS;
const LNGLAT = COORDINATE_SYSTEM.LNGLAT;

/** Crossing kinds of GPUSegmentIntersection, index 1 to 5. */
const KIND_NAMES = [
  '',
  'proper crossing',
  'touch',
  'collinear touch',
  'overlap',
  'uncertain'
] as const;
const KIND_COLORS: readonly (readonly [number, number, number, number])[] = [
  [255, 70, 70, 255],
  [255, 200, 40, 255],
  [255, 90, 220, 255],
  [70, 205, 150, 255],
  [255, 255, 255, 255]
];
/** Node degree palette: degree 1 (dead end) to 6 or more. */
const DEGREE_COLORS: readonly (readonly [number, number, number, number])[] = [
  [226, 96, 80, 255],
  [150, 160, 180, 255],
  [255, 200, 40, 255],
  [70, 205, 150, 255],
  [78, 168, 222, 255],
  [190, 120, 255, 255]
];

type Environment = {
  ctx: SceneContext<Options>;
  origin: readonly [number, number];
  coordinateOrigin: [number, number, number];
  getStreets: () => PathSet;
  rail: PathSet;
  buildNoding: (scope: 'rail' | 'rail-bus') => ReturnType<typeof buildNodingLines>;
  getCounties: () => {layout: PolygonLayout; groups: {state: Uint32Array; rucc: Uint32Array}};
};

/**
 * Noding, intersection and coverage tools on real layers: street and rail crossings, CTA routes
 * noded into a network, US counties dissolved into regions, and counties generalised without gaps.
 * Each tool is created the first time it is shown.
 */
export async function createNodingAndCoverage(
  ctx: SceneContext<Options>
): Promise<SceneInstance<Options>> {
  const roads = ctx.datasets.get('chicago-roads');
  const transit = ctx.datasets.get('cta-transit');
  const counties = ctx.datasets.get('us-counties');
  const origin = roads.defaultOrigin;
  const projection = roads.getProjection(origin);
  let streets: PathSet | null = null;
  let countyData: ReturnType<Environment['getCounties']> | null = null;
  const env: Environment = {
    ctx,
    origin,
    coordinateOrigin: [origin[0], origin[1], 0],
    getStreets: () => (streets ??= loadStreetPaths(roads, projection)),
    rail: loadRailPaths(transit, projection),
    buildNoding: scope => buildNodingLines(transit, projection, scope),
    getCounties: () => {
      if (!countyData) {
        const layout = loadPolygonLayout(counties);
        const state = new Uint32Array(layout.featureCount);
        const stateColumn = counties.column<Uint8Array>('stateFips');
        const dense = new Map<number, number>();
        for (let feature = 0; feature < layout.featureCount; feature++) {
          const key = stateColumn[feature];
          if (!dense.has(key)) dense.set(key, dense.size);
          state[feature] = dense.get(key)!;
        }
        const rucc = new Uint32Array(layout.featureCount);
        if (counties.hasColumn('rucc2023')) {
          const column = counties.column<Uint8Array>('rucc2023');
          for (let feature = 0; feature < layout.featureCount; feature++)
            rucc[feature] = column[feature];
        }
        countyData = {layout, groups: {state, rucc}};
      }
      return countyData;
    }
  };

  const factories: Record<Options['view'], () => View> = {
    crossings: () => createCrossingsView(env),
    noding: () => createNodingView(env),
    dissolve: () => createDissolveView(env),
    generalise: () => createGeneraliseView(env)
  };
  const views = new Map<Options['view'], View>();
  let destroyed = false;
  const getView = () => {
    const name = ctx.options.view;
    let view = views.get(name);
    if (!view) {
      view = factories[name]();
      views.set(name, view);
    }
    return view;
  };
  getView();

  return {
    getCompiledGraphs: () => getView().getCompiledGraphs(),
    setOption(id, _value, options) {
      if (id === 'view') getView();
      else getView().setOption(id, options);
      ctx.requestLayers();
    },
    onThemeChange: () => ctx.requestLayers(),
    encode(commandEncoder, frame) {
      if (!destroyed) getView().encode(commandEncoder, frame);
    },
    getLayers: () => getView().getLayers(),
    destroy() {
      destroyed = true;
      for (const view of views.values()) view.destroy();
      views.clear();
    }
  };
}

// ---------------------------------------------------------------------------------------------
// View: crossings (GPUSegmentIntersection)
// ---------------------------------------------------------------------------------------------

const PAIR_CAPACITY = 1 << 19;

function createCrossingsView(env: Environment): View {
  const {ctx} = env;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'crossings');
  const streets = env.getStreets();
  const rail = env.rail;
  const streetPositions = resources.createBuffer('street-positions', streets.local);
  const streetOffsets = resources.createBuffer('street-offsets', streets.offsets);
  const railPositions = resources.createBuffer('rail-positions', rail.local);
  const railOffsets = resources.createBuffer('rail-offsets', rail.offsets);
  const staticStreets = createStaticPaths(
    resources,
    'crossing-streets',
    streets.local,
    streets.offsets
  );
  const staticRail = createStaticPaths(resources, 'crossing-rail', rail.local, rail.offsets);
  const kindFilter = resources.createParameterBuffer('kind-filter', 'float32', 1);

  type Build = {
    key: string;
    compiled: CompiledGPUCommandGraph<void>;
    kinds: Buffer;
    points: Buffer;
    shown: Buffer;
    count: Buffer;
    overflow: Buffer;
    total: Buffer;
    uncertain: Buffer;
    drawCommands: DrawCommandBuffer;
    reader: SummaryReader;
    segments: number;
  };
  const builds = new Map<string, Build>();
  let active: Build | null = null;
  let dirty = true;

  const build = (options: Options): Build => {
    const twoSided = options.crossMode === 'streets-rail';
    const sameFeature = !twoSided && options.sameFeatureOnly;
    const key = `${options.crossMode}|${sameFeature}|${options.crossSpatialSort}`;
    const cached = builds.get(key);
    if (cached) return cached;
    const pairLeft = resources.createBuffer(`pair-left-${key}`, PAIR_CAPACITY * 4);
    const pairRight = resources.createBuffer(`pair-right-${key}`, PAIR_CAPACITY * 4);
    const count = resources.createBuffer(`count-${key}`, 4);
    const overflow = resources.createBuffer(`overflow-${key}`, 4);
    const total = resources.createBuffer(`total-${key}`, 4);
    const kinds = resources.createBuffer(`kinds-${key}`, PAIR_CAPACITY * 4);
    const points = resources.createBuffer(`points-${key}`, PAIR_CAPACITY * 8);
    const shown = resources.createBuffer(`shown-${key}`, PAIR_CAPACITY * 4);
    const uncertain = resources.createBuffer(`uncertain-${key}`, 4);
    const graph = new GPUCommandGraph<void>(device, {id: `crossings-${key}`});
    const imp = createGraphImporter(graph);
    const kindsView = imp('kinds', kinds, 'uint32', PAIR_CAPACITY);
    graph.add(
      new GPUSegmentIntersection({
        id: 'crossings',
        left: {
          kind: 'lines',
          positions: imp('street-positions', streetPositions, 'float32x2', streets.vertexCount),
          lineOffsets: imp('street-offsets', streetOffsets, 'uint32', streets.pathCount + 1)
        },
        ...(twoSided
          ? {
              right: {
                kind: 'lines' as const,
                positions: imp('rail-positions', railPositions, 'float32x2', rail.vertexCount),
                lineOffsets: imp('rail-offsets', railOffsets, 'uint32', rail.pathCount + 1)
              }
            }
          : {}),
        sameFeatureOnly: sameFeature,
        spatialSort: options.crossSpatialSort,
        pairs: {
          leftIds: imp('pair-left', pairLeft, 'uint32', PAIR_CAPACITY),
          rightIds: imp('pair-right', pairRight, 'uint32', PAIR_CAPACITY),
          count: imp('count', count, 'uint32', 1),
          overflow: imp('overflow', overflow, 'uint32', 1),
          requiredCount: imp('total', total, 'uint32', 1)
        },
        kinds: kindsView,
        points: imp('points', points, 'float32x2', PAIR_CAPACITY),
        uncertainCount: imp('uncertain', uncertain, 'uint32', 1)
      })
    );
    // Marker color index (kind - 1), or 0xffffffff when the kind is filtered out.
    addKernelPass(graph, {
      id: 'kind-filter',
      invocationCount: PAIR_CAPACITY,
      bindings: [
        {name: 'kinds', view: kindsView, type: 'u32', access: 'read'},
        {
          name: 'kindFilter',
          view: kindFilter.importToGraph(graph),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'shown',
          view: imp('shown', shown, 'uint32', PAIR_CAPACITY),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let kind = kinds[kindsOffset + index];
  var value = 0xffffffffu;
  if (kind >= 1u && kind <= 5u && ((u32(kindFilter[kindFilterOffset]) >> (kind - 1u)) & 1u) != 0u) {
    value = kind - 1u;
  }
  shown[shownOffset + index] = value;`
    });
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: `crossing-draw-${key}`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const segments = streets.vertexCount - streets.pathCount;
    const reader = new SummaryReader(
      resources,
      `crossings-${key}`,
      [
        {buffer: count, size: 4},
        {buffer: overflow, size: 4},
        {buffer: total, size: 4},
        {buffer: uncertain, size: 4},
        {buffer: kinds, size: PAIR_CAPACITY * 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        const used = Math.min(words[0], PAIR_CAPACITY);
        const histogram = [0, 0, 0, 0, 0, 0];
        for (const kind of words.subarray(4, 4 + used))
          if (kind < histogram.length) histogram[kind]++;
        ctx.setReadout(
          'crossPairs',
          `${formatCount(words[2])} intersecting pairs among ${formatCount(segments)} street segments${twoSided ? ' and the L routes' : ''}`
        );
        ctx.setReadout(
          'crossKinds',
          [1, 2, 3, 4, 5]
            .map(kind => `${KIND_NAMES[kind]} ${formatCount(histogram[kind])}`)
            .join(', ')
        );
        ctx.setReadout(
          'crossFlags',
          `${words[1] ? 'OVERFLOW' : 'no overflow'}, ${formatCount(words[3])} uncertain`
        );
      }
    );
    const entry: Build = {
      key,
      compiled: resources.track(graph.compile()),
      kinds,
      points,
      shown,
      count,
      overflow,
      total,
      uncertain,
      drawCommands,
      reader,
      segments
    };
    builds.set(key, entry);
    return entry;
  };

  const writeFilter = (options: Options) => {
    const mask = {proper: 1, touches: 2 | 4, overlaps: 8, all: 31}[options.crossKinds];
    kindFilter.write(Float32Array.of(mask));
    dirty = true;
  };
  const select = (options: Options) => {
    active = build(options);
    writeFilter(options);
    dirty = true;
  };
  select(ctx.options);

  return {
    getCompiledGraphs: () =>
      [...builds.values()].map(entry => entry.compiled as CompiledGPUCommandGraph<never>),
    setOption(id, options) {
      if (id === 'crossKinds') writeFilter(options);
      else select(options);
    },
    encode(commandEncoder) {
      if (!active) return;
      if (dirty) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, active.count, active.drawCommands);
        active.reader.request(commandEncoder);
        dirty = false;
      } else {
        active.reader.flush(commandEncoder);
      }
    },
    getLayers() {
      if (!active) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [
        new PathOutputLayer({
          id: 'crossing-streets',
          coordinateSystem: LOCAL,
          coordinateOrigin: env.coordinateOrigin,
          positions: staticStreets.positions,
          pathOffsets: staticStreets.offsets,
          pathOffsetCount: staticStreets.offsetCount,
          vertexCount: staticStreets.vertexCount,
          drawCommands: staticStreets.drawCommands,
          color: dark ? [150, 160, 180, 90] : [90, 100, 120, 100],
          widthPixels: 0.8
        })
      ];
      if (options.crossMode === 'streets-rail') {
        layers.push(
          new PathOutputLayer({
            id: 'crossing-rail',
            coordinateSystem: LOCAL,
            coordinateOrigin: env.coordinateOrigin,
            positions: staticRail.positions,
            pathOffsets: staticRail.offsets,
            pathOffsetCount: staticRail.offsetCount,
            vertexCount: staticRail.vertexCount,
            drawCommands: staticRail.drawCommands,
            color: dark ? [120, 200, 255, 220] : [20, 100, 170, 220],
            widthPixels: 2
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'crossing-points',
          coordinateSystem: LOCAL,
          coordinateOrigin: env.coordinateOrigin,
          positions: active.points,
          drawCommands: active.drawCommands,
          values: active.shown,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: KIND_COLORS,
          noDataValue: 0xffffffff,
          noDataColor: [0, 0, 0, 0],
          radiusPixels: options.crossKinds === 'proper' ? 4.5 : 2.6
        })
      );
      return layers;
    },
    destroy() {
      for (const entry of builds.values()) entry.reader.stop();
      resources.destroy();
    }
  };
}

// ---------------------------------------------------------------------------------------------
// View: noding (GPULineSplit, GPULineMerge, GPUNetworkNoding)
// ---------------------------------------------------------------------------------------------

function createNodingView(env: Environment): View {
  const {ctx} = env;
  const {device} = ctx;
  let resources = new SpatialAnalysisResources(device, 'noding');
  let current: {
    key: string;
    compiledA: CompiledGPUCommandGraph<void>;
    compiledB: CompiledGPUCommandGraph<void>;
  } | null = null;
  let layersFor: (() => Layer[]) | null = null;
  let encodeFor: ((commandEncoder: Parameters<View['encode']>[0]) => void) | null = null;
  let toleranceWrite = null as ((value: number) => void) | null;
  let readers: SummaryReader[] = [];
  let dirty = true;
  let toleranceDirty = true;

  const rebuild = (options: Options) => {
    const key = `${options.nodingInput}|${options.nodingCapacity}|${options.nodingSpatialSort}`;
    if (current?.key === key) return;
    // Retire the old resources a moment later: Deck may still hold the last frame's layers.
    const previous = resources;
    const previousReaders = readers;
    setTimeout(() => {
      for (const reader of previousReaders) reader.stop();
      previous.destroy();
    }, 500);
    resources = new SpatialAnalysisResources(device, `noding-${key}`);
    readers = [];
    const lines = env.buildNoding(options.nodingInput);
    const lineCount = lines.pathCount;
    const vertexCount = lines.vertexCount;
    const segmentCount = vertexCount - lineCount;
    const intersectionCapacity = Number(options.nodingCapacity);
    const pieceCapacity = segmentCount * 3 + 4096;
    const vertexCapacity = vertexCount + 2 * pieceCapacity;
    const nodeCapacity = vertexCount + pieceCapacity;
    const linePositions = resources.createBuffer('line-positions', lines.local);
    const lineOffsets = resources.createBuffer('line-offsets', lines.offsets);
    const staticLines = createStaticPaths(resources, 'noding-lines', lines.local, lines.offsets);

    // Graph A: split, then merge the pieces back into chains.
    const splitLineIds = resources.createBuffer('split-line-ids', pieceCapacity * 4);
    const splitOffsets = resources.createBuffer('split-offsets', (pieceCapacity + 1) * 4);
    const splitPositions = resources.createBuffer('split-positions', vertexCapacity * 8);
    const splitCount = resources.createBuffer('split-count', 4);
    const splitVertices = resources.createBuffer('split-vertices', 4);
    const splitOverflow = resources.createBuffer('split-overflow', 4);
    const splitTotal = resources.createBuffer('split-total', 4);
    const splitUncertain = resources.createBuffer('split-uncertain', 4);
    const chainOffsets = resources.createBuffer('chain-offsets', (pieceCapacity + 1) * 4);
    const chainPositions = resources.createBuffer('chain-positions', vertexCapacity * 8);
    const chainCount = resources.createBuffer('chain-count', 4);
    const chainVertices = resources.createBuffer('chain-vertices', 4);
    const splitDraw = resources.track(
      new DrawCommandBuffer(device, {
        id: 'split-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const chainDraw = resources.track(
      new DrawCommandBuffer(device, {
        id: 'chain-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );

    const graphA = new GPUCommandGraph<void>(device, {id: `noding-split-${key}`});
    {
      const imp = createGraphImporter(graphA);
      const offsetsView = imp('split-offsets', splitOffsets, 'uint32', pieceCapacity + 1);
      const positionsView = imp('split-positions', splitPositions, 'float32x2', vertexCapacity);
      graphA.add(
        new GPULineSplit({
          id: 'split',
          lines: {
            kind: 'lines',
            positions: imp('line-positions', linePositions, 'float32x2', vertexCount),
            lineOffsets: imp('line-offsets', lineOffsets, 'uint32', lineCount + 1)
          },
          intersectionCapacity,
          spatialSort: options.nodingSpatialSort,
          uncertainCount: imp('split-uncertain', splitUncertain, 'uint32', 1),
          pieces: {
            geometry: {kind: 'lines', positions: positionsView, lineOffsets: offsetsView},
            sourceIds: imp('split-line-ids', splitLineIds, 'uint32', pieceCapacity),
            status: {
              count: imp('split-count', splitCount, 'uint32', 1),
              overflow: imp('split-overflow', splitOverflow, 'uint32', 1),
              requiredCount: imp('split-total', splitTotal, 'uint32', 1)
            },
            vertexCount: imp('split-vertices', splitVertices, 'uint32', 1)
          }
        })
      );
      const chainOffsetsView = imp('chain-offsets', chainOffsets, 'uint32', pieceCapacity + 1);
      const chainCountView = imp('chain-count', chainCount, 'uint32', 1);
      graphA.add(
        new GPULineMerge({
          id: 'merge',
          positions: positionsView,
          lineOffsets: offsetsView,
          output: {
            chainOffsets: chainOffsetsView,
            positions: imp('chain-positions', chainPositions, 'float32x2', vertexCapacity),
            count: chainCountView
          }
        })
      );
      // Vertex count of the merged chains: the offset after the last chain.
      addKernelPass(graphA, {
        id: 'chain-vertex-count',
        invocationCount: 1,
        bindings: [
          {name: 'offsets', view: chainOffsetsView, type: 'u32', access: 'read'},
          {name: 'count', view: chainCountView, type: 'u32', access: 'read'},
          {
            name: 'vertices',
            view: imp('chain-vertices', chainVertices, 'uint32', 1),
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: /* wgsl */ `vertices[verticesOffset] = offsets[offsetsOffset + min(count[countOffset], ${pieceCapacity}u)];`
      });
    }
    const compiledA = resources.track(graphA.compile());

    // Graph B: network noding with a per-frame snap tolerance.
    const tolerance = resources.createParameterBuffer('tolerance', 'float32', 1);
    const piecesLineIds = resources.createBuffer('net-line-ids', pieceCapacity * 4);
    const piecesOffsets = resources.createBuffer('net-offsets', (pieceCapacity + 1) * 4);
    const piecesPositions = resources.createBuffer('net-positions', vertexCapacity * 8);
    const piecesCount = resources.createBuffer('net-count', 4);
    const piecesVertices = resources.createBuffer('net-vertices', 4);
    const piecesTotal = resources.createBuffer('net-total', 4);
    const nodePositions = resources.createBuffer('node-positions', nodeCapacity * 8);
    const nodeCount = resources.createBuffer('node-count', 4);
    const nodeTotal = resources.createBuffer('node-total', 4);
    const fromNodes = resources.createBuffer('from-nodes', pieceCapacity * 4);
    const toNodes = resources.createBuffer('to-nodes', pieceCapacity * 4);
    const edgeLengths = resources.createBuffer('edge-lengths', pieceCapacity * 4);
    const csrOffsets = resources.createBuffer('csr-offsets', (nodeCapacity + 1) * 4);
    const csrNeighbors = resources.createBuffer('csr-neighbors', pieceCapacity * 8);
    const csrWeights = resources.createBuffer('csr-weights', pieceCapacity * 8);
    const netOverflow = resources.createBuffer('net-overflow', 4);
    const netUncertain = resources.createBuffer('net-uncertain', 4);
    const degrees = resources.createBuffer('node-degrees', nodeCapacity * 4);
    const netDraw = resources.track(
      new DrawCommandBuffer(device, {
        id: 'net-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const nodeDraw = resources.track(
      new DrawCommandBuffer(device, {
        id: 'node-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );

    const graphB = new GPUCommandGraph<void>(device, {id: `noding-network-${key}`});
    {
      const imp = createGraphImporter(graphB);
      const nodeCountView = imp('node-count', nodeCount, 'uint32', 1);
      const csrOffsetsView = imp('csr-offsets', csrOffsets, 'uint32', nodeCapacity + 1);
      graphB.add(
        new GPUNetworkNoding({
          id: 'network',
          lines: {
            kind: 'lines',
            positions: imp('line-positions', linePositions, 'float32x2', vertexCount),
            lineOffsets: imp('line-offsets', lineOffsets, 'uint32', lineCount + 1)
          },
          intersectionCapacity,
          spatialSort: options.nodingSpatialSort,
          uncertainCount: imp('net-uncertain', netUncertain, 'uint32', 1),
          tolerance: tolerance.importToGraph(graphB),
          pieces: {
            geometry: {
              kind: 'lines',
              positions: imp('net-positions', piecesPositions, 'float32x2', vertexCapacity),
              lineOffsets: imp('net-offsets', piecesOffsets, 'uint32', pieceCapacity + 1)
            },
            sourceIds: imp('net-line-ids', piecesLineIds, 'uint32', pieceCapacity),
            status: {
              count: imp('net-count', piecesCount, 'uint32', 1),
              overflow: imp('net-piece-overflow', splitOverflow, 'uint32', 1),
              requiredCount: imp('net-total', piecesTotal, 'uint32', 1)
            },
            vertexCount: imp('net-vertices', piecesVertices, 'uint32', 1)
          },
          nodes: {
            positions: imp('node-positions', nodePositions, 'float32x2', nodeCapacity),
            count: nodeCountView,
            requiredCount: imp('node-total', nodeTotal, 'uint32', 1)
          },
          edges: {
            fromNodes: imp('from-nodes', fromNodes, 'uint32', pieceCapacity),
            toNodes: imp('to-nodes', toNodes, 'uint32', pieceCapacity),
            lengths: imp('edge-lengths', edgeLengths, 'float32', pieceCapacity)
          },
          csr: {
            offsets: csrOffsetsView,
            neighbors: imp('csr-neighbors', csrNeighbors, 'uint32', pieceCapacity * 2),
            weights: imp('csr-weights', csrWeights, 'float32', pieceCapacity * 2)
          },
          overflow: imp('net-overflow', netOverflow, 'uint32', 1)
        })
      );
      addKernelPass(graphB, {
        id: 'node-degree',
        invocationCount: nodeCapacity,
        bindings: [
          {name: 'offsets', view: csrOffsetsView, type: 'u32', access: 'read'},
          {name: 'count', view: nodeCountView, type: 'u32', access: 'read'},
          {
            name: 'degrees',
            view: imp('node-degrees', degrees, 'uint32', nodeCapacity),
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: /* wgsl */ `
  var value = 0xffffffffu;
  if (index < count[countOffset]) {
    value = min(offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index], 6u) - 1u;
  }
  degrees[degreesOffset + index] = value;`
      });
    }
    const compiledB = resources.track(graphB.compile());

    current = {key, compiledA, compiledB};
    toleranceDirty = true;
    dirty = true;
    toleranceWrite = value => tolerance.write(Float32Array.of(value));

    const readerA = new SummaryReader(
      resources,
      'noding-a',
      [
        {buffer: splitCount, size: 4},
        {buffer: splitOverflow, size: 4},
        {buffer: splitTotal, size: 4},
        {buffer: splitUncertain, size: 4},
        {buffer: chainCount, size: 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        ctx.setReadout(
          'nodingPieces',
          `${formatCount(words[0])} pieces from ${formatCount(lineCount)} lines${words[1] ? ' (OVERFLOW: raise the intersection capacity)' : ''}`
        );
        ctx.setReadout('nodingMerged', `${formatCount(words[4])} chains`);
        ctx.setReadout('nodingUncertain', formatCount(words[3]));
      }
    );
    const degreeReaderBuffer = degrees;
    const readerB = new SummaryReader(
      resources,
      'noding-b',
      [
        {buffer: piecesCount, size: 4},
        {buffer: nodeCount, size: 4},
        {buffer: netOverflow, size: 4},
        {buffer: degreeReaderBuffer, size: nodeCapacity * 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        const histogram = new Array(6).fill(0);
        for (const degree of words.subarray(3, 3 + Math.min(words[1], nodeCapacity)))
          if (degree < 6) histogram[degree]++;
        ctx.setReadout(
          'nodingEdges',
          `${formatCount(words[0])} edges between ${formatCount(words[1])} nodes${words[2] ? ' (OVERFLOW)' : ''}`
        );
        ctx.setReadout(
          'nodingDegrees',
          `dead ends ${formatCount(histogram[0])}, degree 2 ${formatCount(histogram[1])}, 3 ${formatCount(histogram[2])}, 4 ${formatCount(histogram[3])}, 5+ ${formatCount(histogram[4] + histogram[5])}`
        );
      }
    );
    readers = [readerA, readerB];
    ctx.setReadout(
      'nodingInput',
      `${formatCount(lineCount)} polylines (${lines.routeCount} routes${lines.duplicatesRemoved ? `, ${formatCount(lines.duplicatesRemoved)} exact duplicates removed` : ''}), ${formatCount(vertexCount)} vertices`
    );

    encodeFor = commandEncoder => {
      const options = ctx.options;
      if (dirty) {
        compiledA.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, splitVertices, splitDraw);
        copyCountToDrawRecord(commandEncoder, chainVertices, chainDraw);
        readerA.request(commandEncoder);
        dirty = false;
      } else {
        readerA.flush(commandEncoder);
      }
      if (options.nodingOutput === 'network' || toleranceDirty) {
        if (toleranceDirty || options.nodingOutput === 'network') {
          compiledB.encode(commandEncoder, {parameters: undefined});
          copyCountToDrawRecord(commandEncoder, piecesVertices, netDraw);
          copyCountToDrawRecord(commandEncoder, nodeCount, nodeDraw);
          if (toleranceDirty) readerB.request(commandEncoder);
          toleranceDirty = false;
        }
      }
      readerB.flush(commandEncoder);
    };

    layersFor = () => {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      layers.push(
        new PathOutputLayer({
          id: 'noding-input',
          coordinateSystem: LOCAL,
          coordinateOrigin: env.coordinateOrigin,
          positions: staticLines.positions,
          pathOffsets: staticLines.offsets,
          pathOffsetCount: staticLines.offsetCount,
          vertexCount: staticLines.vertexCount,
          drawCommands: staticLines.drawCommands,
          color: dark ? [150, 160, 180, 70] : [90, 100, 120, 70],
          widthPixels: 6
        })
      );
      if (options.nodingOutput === 'pieces') {
        layers.push(
          new PathOutputLayer({
            id: 'noding-pieces',
            coordinateSystem: LOCAL,
            coordinateOrigin: env.coordinateOrigin,
            positions: splitPositions,
            pathOffsets: splitOffsets,
            pathOffsetCount: pieceCapacity + 1,
            vertexCount: splitVertices,
            drawCommands: splitDraw,
            colorSource: 'path-index',
            color: [255, 255, 255, 255],
            widthPixels: 2.6
          })
        );
      } else if (options.nodingOutput === 'merged') {
        layers.push(
          new PathOutputLayer({
            id: 'noding-chains',
            coordinateSystem: LOCAL,
            coordinateOrigin: env.coordinateOrigin,
            positions: chainPositions,
            pathOffsets: chainOffsets,
            pathOffsetCount: pieceCapacity + 1,
            vertexCount: chainVertices,
            drawCommands: chainDraw,
            colorSource: 'path-index',
            color: [255, 255, 255, 255],
            widthPixels: 3
          })
        );
      } else {
        layers.push(
          new PathOutputLayer({
            id: 'noding-edges',
            coordinateSystem: LOCAL,
            coordinateOrigin: env.coordinateOrigin,
            positions: piecesPositions,
            pathOffsets: piecesOffsets,
            pathOffsetCount: pieceCapacity + 1,
            vertexCount: piecesVertices,
            drawCommands: netDraw,
            colorSource: 'path-index',
            color: [255, 255, 255, 200],
            widthPixels: 2.2
          }),
          new SpatialAnalysisPointLayer({
            id: 'noding-nodes',
            coordinateSystem: LOCAL,
            coordinateOrigin: env.coordinateOrigin,
            positions: nodePositions,
            drawCommands: nodeDraw,
            values: degrees,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: DEGREE_COLORS,
            noDataValue: 0xffffffff,
            noDataColor: [0, 0, 0, 0],
            radiusPixels: 4.5
          })
        );
      }
      return layers;
    };
  };
  rebuild(ctx.options);
  toleranceWrite?.(ctx.options.nodingTolerance);

  return {
    getCompiledGraphs: () =>
      current ? ([current.compiledA, current.compiledB] as CompiledGPUCommandGraph<never>[]) : [],
    setOption(id, options) {
      if (id === 'nodingInput' || id === 'nodingCapacity' || id === 'nodingSpatialSort') {
        rebuild(options);
        toleranceWrite?.(options.nodingTolerance);
      } else if (id === 'nodingTolerance') {
        toleranceWrite?.(options.nodingTolerance);
        toleranceDirty = true;
      }
    },
    encode(commandEncoder) {
      encodeFor?.(commandEncoder);
    },
    getLayers: () => (layersFor ? layersFor() : []),
    destroy() {
      for (const reader of readers) reader.stop();
      resources.destroy();
    }
  };
}

// ---------------------------------------------------------------------------------------------
// View: dissolve (GPUSegmentRingAssembly)
// ---------------------------------------------------------------------------------------------

const RING_CAPACITY = 8192;

function createDissolveView(env: Environment): View {
  const {ctx} = env;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'dissolve');
  const {layout, groups} = env.getCounties();
  const edges = buildDirectedEdges(layout);
  const segmentCount = edges.count;
  const vertexCapacity = segmentCount + 2 * RING_CAPACITY;
  const endpoints = resources.createBuffer('endpoints', edges.endpoints);
  const segmentGroups = resources.createBuffer('segment-groups', segmentCount * 4);
  const countyGroups = resources.createBuffer('county-groups', layout.featureCount * 4);
  const fill = triangulatePolygons(layout);
  const fillCorners = resources.createBuffer('fill-corners', fill.corners);
  const fillRows = resources.createBuffer('fill-rows', fill.featureRows);
  const outline = buildRingEdges(layout);
  const outlineStarts = resources.createBuffer('outline-starts', outline.starts);
  const outlineEnds = resources.createBuffer('outline-ends', outline.ends);

  type Build = {
    key: string;
    compiled: CompiledGPUCommandGraph<void>;
    ringOffsets: Buffer;
    positions: Buffer;
    count: Buffer;
    vertices: Buffer;
    groupValues: Buffer;
    holeValues: Buffer;
    drawCommands: DrawCommandBuffer;
    reader: SummaryReader;
  };
  const builds = new Map<string, Build>();
  let active: Build | null = null;
  let dirty = true;

  const writeGroups = (options: Options) => {
    const source =
      options.dissolveBy === 'state'
        ? groups.state
        : options.dissolveBy === 'rucc'
          ? groups.rucc
          : new Uint32Array(layout.featureCount);
    const perSegment = new Uint32Array(segmentCount);
    for (let segment = 0; segment < segmentCount; segment++)
      perSegment[segment] = source[edges.featureRows[segment]];
    segmentGroups.write(perSegment);
    countyGroups.write(Float32Array.from(source));
    dirty = true;
  };

  const build = (options: Options): Build => {
    const key = [
      options.cancelOpposing,
      options.splitTouching,
      options.vertexTolerance,
      options.interiorSide,
      options.normalizeWinding
    ].join('|');
    const cached = builds.get(key);
    if (cached) return cached;
    const ringOffsets = resources.createBuffer(`ring-offsets-${key}`, (RING_CAPACITY + 1) * 4);
    const positions = resources.createBuffer(`positions-${key}`, vertexCapacity * 8);
    const count = resources.createBuffer(`count-${key}`, 4);
    const overflow = resources.createBuffer(`overflow-${key}`, 4);
    const total = resources.createBuffer(`total-${key}`, 4);
    const open = resources.createBuffer(`open-${key}`, 4);
    const touching = resources.createBuffer(`touching-${key}`, 4);
    const ringGroups = resources.createBuffer(`ring-groups-${key}`, RING_CAPACITY * 4);
    const ringHoles = resources.createBuffer(`ring-holes-${key}`, RING_CAPACITY * 4);
    const ringAreas = resources.createBuffer(`ring-areas-${key}`, RING_CAPACITY * 4);
    const flags = resources.createBuffer(`flags-${key}`, segmentCount * 4);
    const vertices = resources.createBuffer(`vertices-${key}`, 4);
    const groupValues = resources.createBuffer(`group-values-${key}`, RING_CAPACITY * 4);
    const holeValues = resources.createBuffer(`hole-values-${key}`, RING_CAPACITY * 4);
    const graph = new GPUCommandGraph<void>(device, {id: `dissolve-${key}`});
    const imp = createGraphImporter(graph);
    const offsetsView = imp('ring-offsets', ringOffsets, 'uint32', RING_CAPACITY + 1);
    const countView = imp('count', count, 'uint32', 1);
    const groupsView = imp('ring-groups', ringGroups, 'uint32', RING_CAPACITY);
    const holesView = imp('ring-holes', ringHoles, 'uint32', RING_CAPACITY);
    graph.add(
      new GPUSegmentRingAssembly({
        id: 'dissolve',
        endpoints: imp('endpoints', endpoints, 'float32x4', segmentCount),
        groups: imp('segment-groups', segmentGroups, 'uint32', segmentCount),
        vertexTolerance: Number(options.vertexTolerance),
        interiorSide: options.interiorSide,
        normalizeWinding: options.normalizeWinding,
        cancelOpposingSegments: options.cancelOpposing,
        splitTouchingRings: options.splitTouching,
        output: {
          ringOffsets: offsetsView,
          positions: imp('positions', positions, 'float32x2', vertexCapacity),
          ringAreas: imp('ring-areas', ringAreas, 'float32', RING_CAPACITY),
          ringIsHole: holesView,
          ringGroups: groupsView,
          segmentFlags: imp('flags', flags, 'uint32', segmentCount),
          count: countView,
          overflow: imp('overflow', overflow, 'uint32', 1),
          requiredCount: imp('total', total, 'uint32', 1),
          openSegmentCount: imp('open', open, 'uint32', 1),
          touchingSegmentCount: imp('touching', touching, 'uint32', 1)
        }
      })
    );
    addKernelPass(graph, {
      id: 'ring-values',
      invocationCount: RING_CAPACITY,
      bindings: [
        {name: 'groups', view: groupsView, type: 'u32', access: 'read'},
        {name: 'holes', view: holesView, type: 'u32', access: 'read'},
        {
          name: 'groupValues',
          view: imp('group-values', groupValues, 'float32', RING_CAPACITY),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'holeValues',
          view: imp('hole-values', holeValues, 'float32', RING_CAPACITY),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  groupValues[groupValuesOffset + index] = f32(groups[groupsOffset + index]);
  holeValues[holeValuesOffset + index] = f32(holes[holesOffset + index]);`
    });
    addKernelPass(graph, {
      id: 'ring-vertex-count',
      invocationCount: 1,
      bindings: [
        {name: 'offsets', view: offsetsView, type: 'u32', access: 'read'},
        {name: 'count', view: countView, type: 'u32', access: 'read'},
        {
          name: 'vertices',
          view: imp('vertices', vertices, 'uint32', 1),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `vertices[verticesOffset] = offsets[offsetsOffset + min(count[countOffset], ${RING_CAPACITY}u)];`
    });
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: `dissolve-draw-${key}`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const reader = new SummaryReader(
      resources,
      `dissolve-${key}`,
      [
        {buffer: count, size: 4},
        {buffer: overflow, size: 4},
        {buffer: total, size: 4},
        {buffer: open, size: 4},
        {buffer: touching, size: 4},
        {buffer: vertices, size: 4},
        {buffer: ringHoles, size: RING_CAPACITY * 4},
        {buffer: flags, size: segmentCount * 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        const rings = Math.min(words[0], RING_CAPACITY);
        let holes = 0;
        for (const hole of words.subarray(6, 6 + rings)) holes += hole;
        const flagWords = words.subarray(6 + RING_CAPACITY);
        let cancelled = 0;
        let dangling = 0;
        let conflicts = 0;
        for (const flag of flagWords) {
          if (flag & 8) cancelled++;
          if (flag & 2) dangling++;
          if (flag & 4) conflicts++;
        }
        ctx.setReadout(
          'dissolveRings',
          `${formatCount(words[2])} rings (${formatCount(rings - holes)} shells, ${formatCount(holes)} holes), ${formatCount(words[5])} vertices${words[1] ? ', OVERFLOW' : ''}`
        );
        ctx.setReadout(
          'dissolveSegments',
          `${formatCount(segmentCount)} in, ${formatCount(cancelled)} cancelled, ${formatCount(words[3])} on no ring, ${formatCount(words[4])} at touching vertices`
        );
        ctx.setReadout(
          'dissolveFlags',
          `${formatCount(dangling)} dangling, ${formatCount(conflicts)} conflicts`
        );
      }
    );
    const entry: Build = {
      key,
      compiled: resources.track(graph.compile()),
      ringOffsets,
      positions,
      count,
      vertices,
      groupValues,
      holeValues,
      drawCommands,
      reader
    };
    builds.set(key, entry);
    return entry;
  };
  active = build(ctx.options);
  writeGroups(ctx.options);

  return {
    getCompiledGraphs: () =>
      [...builds.values()].map(entry => entry.compiled as CompiledGPUCommandGraph<never>),
    setOption(id, options) {
      if (id === 'dissolveBy') writeGroups(options);
      if (
        id === 'cancelOpposing' ||
        id === 'splitTouching' ||
        id === 'vertexTolerance' ||
        id === 'interiorSide' ||
        id === 'normalizeWinding'
      ) {
        active = build(options);
        dirty = true;
      }
    },
    encode(commandEncoder) {
      if (!active) return;
      if (dirty) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, active.vertices, active.drawCommands);
        active.reader.request(commandEncoder);
        dirty = false;
      } else {
        active.reader.flush(commandEncoder);
      }
    },
    getLayers() {
      if (!active) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showFill) {
        layers.push(
          new FeatureTriangleLayer({
            id: 'dissolve-fill',
            coordinateSystem: LNGLAT,
            corners: fillCorners,
            featureRows: fillRows,
            instanceCount: fill.triangleCount,
            values: countyGroups,
            valueMapping: 'category',
            color: [255, 255, 255, 255],
            opacity: 0.55
          })
        );
      }
      layers.push(
        new PairSegmentLayer({
          id: 'dissolve-county-edges',
          coordinateSystem: LNGLAT,
          starts: outlineStarts,
          ends: outlineEnds,
          instanceCount: outline.edgeCount,
          color: dark ? [255, 255, 255, 60] : [20, 30, 50, 60],
          widthPixels: 0.6
        }),
        new PathOutputLayer({
          id: 'dissolve-rings',
          coordinateSystem: LNGLAT,
          positions: active.positions,
          pathOffsets: active.ringOffsets,
          pathOffsetCount: RING_CAPACITY + 1,
          vertexCount: active.vertices,
          drawCommands: active.drawCommands,
          values: options.dissolveColor === 'hole' ? active.holeValues : null,
          colorSource: options.dissolveColor === 'hole' ? 'path-value' : 'uniform',
          valueMapping: 'flag',
          color:
            options.dissolveColor === 'hole'
              ? [226, 96, 80, 255]
              : dark
                ? [255, 255, 255, 255]
                : [10, 20, 40, 255],
          secondaryColor: dark ? [255, 255, 255, 255] : [10, 20, 40, 255],
          widthPixels: 2.2
        })
      );
      return layers;
    },
    destroy() {
      for (const entry of builds.values()) entry.reader.stop();
      resources.destroy();
    }
  };
}

// ---------------------------------------------------------------------------------------------
// View: generalise (GPUCoverageSimplification against independent simplification)
// ---------------------------------------------------------------------------------------------

function createGeneraliseView(env: Environment): View {
  const {ctx} = env;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'generalise');
  const {layout} = env.getCounties();
  const {vertexCount, ringCount, partCount} = layout;
  const positions = resources.createBuffer('positions', layout.lngLat);
  const ringOffsets = resources.createBuffer('ring-offsets', layout.ringOffsets);
  const polygonOffsets = resources.createBuffer('polygon-offsets', layout.polygonOffsets);
  const vertexRings = new Uint32Array(vertexCount);
  for (let ring = 0; ring < ringCount; ring++) {
    for (let vertex = layout.ringOffsets[ring]; vertex < layout.ringOffsets[ring + 1]; vertex++)
      vertexRings[vertex] = ring;
  }
  const vertexRingBuffer = resources.createBuffer('vertex-rings', vertexRings);
  const parameters = resources.createParameterBuffer(
    'tolerance',
    'float32',
    GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH
  );
  const staticRings = createStaticPaths(
    resources,
    'original-rings',
    layout.lngLat,
    layout.ringOffsets
  );
  const fill = triangulatePolygons(layout);
  const fillCorners = resources.createBuffer('fill-corners', fill.corners);
  const fillRows = resources.createBuffer('fill-rows', fill.featureRows);
  const fillValues = resources.createBuffer(
    'fill-values',
    Float32Array.from({length: layout.featureCount}, (_, index) => index % 7)
  );

  // Independent per-ring simplification: importance once, then a per-frame selection.
  const importance = resources.createBuffer('importance', vertexCount * 4);
  const importanceConverged = resources.createBuffer('importance-converged', 4);
  const importanceRounds = resources.createBuffer('importance-rounds', 4);
  const keptIds = resources.createBuffer('kept-ids', vertexCount * 4);
  const keptCount = resources.createBuffer('kept-count', 4);
  const keptOverflow = resources.createBuffer('kept-overflow', 4);
  const keptDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'kept-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const independentGraphs = new Map<
    string,
    {
      importance: CompiledGPUCommandGraph<void>;
      selection: CompiledGPUCommandGraph<void>;
      encoded: boolean;
    }
  >();
  const buildIndependent = (options: Options) => {
    const key = options.coverageRounds;
    let entry = independentGraphs.get(key);
    if (entry) return entry;
    const maximumRounds = Number(options.coverageRounds);
    const views = (graph: GPUCommandGraph<void>) => {
      const imp = createGraphImporter(graph);
      return {
        imp,
        base: {
          positions: imp('positions', positions, 'float32x2', vertexCount),
          trackOffsets: imp('ring-offsets', ringOffsets, 'uint32', ringCount + 1),
          importance: imp('importance', importance, 'float32', vertexCount)
        }
      };
    };
    const importanceGraph = new GPUCommandGraph<void>(device, {
      id: `independent-importance-${key}`
    });
    {
      const {imp, base} = views(importanceGraph);
      importanceGraph.add(
        new GPULineSimplification({
          id: 'independent-importance',
          ...base,
          maximumRounds,
          status: {
            converged: imp('converged', importanceConverged, 'uint32', 1),
            roundCount: imp('rounds', importanceRounds, 'uint32', 1)
          }
        })
      );
    }
    const selectionGraph = new GPUCommandGraph<void>(device, {id: `independent-selection-${key}`});
    {
      const {imp, base} = views(selectionGraph);
      selectionGraph.add(
        new GPULineSimplification({
          id: 'independent-selection',
          ...base,
          computeImportance: false,
          parameters: parameters.importToGraph(selectionGraph),
          selection: {
            output: {
              ids: imp('kept-ids', keptIds, 'uint32', vertexCount),
              count: imp('kept-count', keptCount, 'uint32', 1),
              overflow: imp('kept-overflow', keptOverflow, 'uint32', 1)
            }
          }
        })
      );
    }
    entry = {
      importance: resources.track(importanceGraph.compile()),
      selection: resources.track(selectionGraph.compile()),
      encoded: false
    };
    independentGraphs.set(key, entry);
    return entry;
  };

  // Coverage simplification variants.
  type Coverage = {
    key: string;
    compiled: CompiledGPUCommandGraph<void>;
    outPositions: Buffer;
    outRings: Buffer;
    vertices: Buffer;
    topology: Buffer;
    converged: Buffer;
    draw: DrawCommandBuffer;
    reader: SummaryReader;
  };
  const coverages = new Map<string, Coverage>();
  const buildCoverage = (options: Options): Coverage => {
    const key = `${options.topologyRounds}|${options.coverageSnap}|${options.coverageRounds}`;
    const cached = coverages.get(key);
    if (cached) return cached;
    const outPositions = resources.createBuffer(`out-positions-${key}`, vertexCount * 8);
    const outRings = resources.createBuffer(`out-rings-${key}`, (ringCount + 1) * 4);
    const keepMask = resources.createBuffer(`keep-mask-${key}`, vertexCount * 4);
    const overflow = resources.createBuffer(`overflow-${key}`, 4);
    const converged = resources.createBuffer(`converged-${key}`, 4);
    const topology = resources.createBuffer(`topology-${key}`, 16);
    const vertices = resources.createBuffer(`vertices-${key}`, 4);
    const graph = new GPUCommandGraph<void>(device, {id: `coverage-${key}`});
    const imp = createGraphImporter(graph);
    const outRingsView = imp('out-rings', outRings, 'uint32', ringCount + 1);
    graph.add(
      new GPUCoverageSimplification({
        id: 'coverage',
        positions: imp('positions', positions, 'float32x2', vertexCount),
        ringOffsets: imp('ring-offsets', ringOffsets, 'uint32', ringCount + 1),
        polygonOffsets: imp('polygon-offsets', polygonOffsets, 'uint32', partCount + 1),
        snapTolerance: Number(options.coverageSnap),
        parameters: parameters.importToGraph(graph),
        maximumRounds: Number(options.coverageRounds),
        topologyRounds: Number(options.topologyRounds),
        converged: imp('converged', converged, 'uint32', 1),
        output: {
          positions: imp('out-positions', outPositions, 'float32x2', vertexCount),
          ringOffsets: outRingsView,
          keepMask: imp('keep-mask', keepMask, 'uint32', vertexCount),
          overflow: imp('overflow', overflow, 'uint32', 1),
          topologyStats: imp('topology', topology, 'uint32', 4)
        }
      })
    );
    addKernelPass(graph, {
      id: 'coverage-vertex-count',
      invocationCount: 1,
      bindings: [
        {name: 'offsets', view: outRingsView, type: 'u32', access: 'read'},
        {
          name: 'vertices',
          view: imp('vertices', vertices, 'uint32', 1),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `vertices[verticesOffset] = offsets[offsetsOffset + ${ringCount}u];`
    });
    const draw = resources.track(
      new DrawCommandBuffer(device, {
        id: `coverage-draw-${key}`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const reader = new SummaryReader(
      resources,
      `coverage-${key}`,
      [
        {buffer: vertices, size: 4},
        {buffer: converged, size: 4},
        {buffer: overflow, size: 4},
        {buffer: topology, size: 16},
        {buffer: keptCount, size: 4},
        {buffer: keptOverflow, size: 4},
        {buffer: importanceConverged, size: 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        ctx.setReadout(
          'coverageKept',
          `${formatCount(words[0])} of ${formatCount(vertexCount)} vertices (${((100 * words[0]) / vertexCount).toFixed(1)}%)${words[2] ? ', OVERFLOW' : ''}`
        );
        ctx.setReadout('coverageConverged', words[1] ? 'yes' : 'no: stopped at the round cap');
        ctx.setReadout(
          'coverageTopology',
          `${formatCount(words[3])} crossings found, ${formatCount(words[4])} remain, ${formatCount(words[5])} vertices restored${words[6] ? ', pair capacity overflow' : ''}`
        );
        ctx.setReadout(
          'independentKept',
          `${formatCount(words[7])} of ${formatCount(vertexCount)} vertices (${((100 * words[7]) / vertexCount).toFixed(1)}%)`
        );
      }
    );
    const entry: Coverage = {
      key,
      compiled: resources.track(graph.compile()),
      outPositions,
      outRings,
      vertices,
      topology,
      converged,
      draw,
      reader
    };
    coverages.set(key, entry);
    return entry;
  };

  let active: Coverage | null = null;
  let independent: ReturnType<typeof buildIndependent> | null = null;
  let appliedTolerance = Number.NaN;
  let needsRead = true;
  const parameterValues = new Float32Array(GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH);

  const select = (options: Options) => {
    active = buildCoverage(options);
    independent = buildIndependent(options);
    appliedTolerance = Number.NaN;
    needsRead = true;
  };
  select(ctx.options);
  ctx.setReadout(
    'coverageInput',
    `${formatCount(layout.featureCount)} counties, ${formatCount(ringCount)} rings, ${formatCount(vertexCount)} vertices`
  );

  return {
    getCompiledGraphs() {
      const graphs: CompiledGPUCommandGraph<never>[] = [];
      for (const entry of coverages.values())
        graphs.push(entry.compiled as CompiledGPUCommandGraph<never>);
      for (const entry of independentGraphs.values())
        graphs.push(
          entry.importance as CompiledGPUCommandGraph<never>,
          entry.selection as CompiledGPUCommandGraph<never>
        );
      return graphs;
    },
    setOption(id, options) {
      if (id === 'topologyRounds' || id === 'coverageSnap' || id === 'coverageRounds')
        select(options);
    },
    encode(commandEncoder) {
      if (!active || !independent) return;
      const tolerance = Math.fround(10 ** ctx.options.coverageTolerance);
      if (!independent.encoded) {
        independent.importance.encode(commandEncoder, {parameters: undefined});
        independent.encoded = true;
        appliedTolerance = Number.NaN;
      }
      if (tolerance !== appliedTolerance) {
        appliedTolerance = tolerance;
        parameters.write(getGPULineSimplificationParameterValues({tolerance}, parameterValues));
        active.compiled.encode(commandEncoder, {parameters: undefined});
        independent.selection.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, active.vertices, active.draw);
        copyCountToDrawRecord(commandEncoder, keptCount, keptDraw);
        ctx.setReadout(
          'coverageTolerance',
          `${tolerance < 0.01 ? tolerance.toExponential(1) : tolerance.toFixed(3)}° (about ${formatDistance(tolerance * 95000)})`
        );
        needsRead = true;
      }
      if (needsRead) {
        active.reader.request(commandEncoder);
        needsRead = false;
      } else {
        active.reader.flush(commandEncoder);
      }
    },
    getLayers() {
      if (!active) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [
        new FeatureTriangleLayer({
          id: 'generalise-fill',
          coordinateSystem: LNGLAT,
          corners: fillCorners,
          featureRows: fillRows,
          instanceCount: fill.triangleCount,
          values: fillValues,
          valueMapping: 'category',
          color: [255, 255, 255, 255],
          opacity: dark ? 0.18 : 0.3
        })
      ];
      if (options.coverageOutline === 'original') {
        layers.push(
          new PathOutputLayer({
            id: 'generalise-original',
            coordinateSystem: LNGLAT,
            positions: staticRings.positions,
            pathOffsets: staticRings.offsets,
            pathOffsetCount: staticRings.offsetCount,
            vertexCount: staticRings.vertexCount,
            drawCommands: staticRings.drawCommands,
            color: dark ? [255, 255, 255, 200] : [20, 30, 50, 200],
            widthPixels: 1
          })
        );
      } else {
        layers.push(
          new PathOutputLayer({
            id: 'generalise-original-faint',
            coordinateSystem: LNGLAT,
            positions: staticRings.positions,
            pathOffsets: staticRings.offsets,
            pathOffsetCount: staticRings.offsetCount,
            vertexCount: staticRings.vertexCount,
            drawCommands: staticRings.drawCommands,
            color: dark ? [255, 255, 255, 40] : [20, 30, 50, 40],
            widthPixels: 0.6
          })
        );
      }
      if (
        independent &&
        (options.coverageOutline === 'independent' || options.coverageOutline === 'both')
      ) {
        layers.push(
          new KeptSegmentLayer({
            id: 'generalise-independent',
            coordinateSystem: LNGLAT,
            positions,
            keptIds,
            vertexLines: vertexRingBuffer,
            keptCount,
            drawCommands: keptDraw,
            color: [226, 96, 80, 255],
            widthPixels: 1.6
          })
        );
      }
      if (options.coverageOutline === 'coverage' || options.coverageOutline === 'both') {
        layers.push(
          new PathOutputLayer({
            id: 'generalise-coverage',
            coordinateSystem: LNGLAT,
            positions: active.outPositions,
            pathOffsets: active.outRings,
            pathOffsetCount: ringCount + 1,
            vertexCount: active.vertices,
            drawCommands: active.draw,
            closed: true,
            color: [30, 130, 190, 255],
            widthPixels: 1.6
          })
        );
      }
      return layers;
    },
    destroy() {
      for (const entry of coverages.values()) entry.reader.stop();
      resources.destroy();
    }
  };
}
