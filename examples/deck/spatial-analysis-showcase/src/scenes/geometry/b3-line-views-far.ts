// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUGreatCircleArcsParameterValues,
  getGPULineSimplificationParameterValues,
  getGPULineSmoothParameterValues,
  GPUGeodesicDestination,
  GPUGeodesicPairs,
  GPUGreatCircleArcs,
  GPULineSimplification,
  GPULineSmooth,
  GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH,
  GPU_LINE_SMOOTH_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {LoadedDataset} from '../../data/catalog';
import type {SceneContext} from '../scene';
import {
  copyCountToDrawRecord,
  createGraphImporter,
  createPathOutputBuffers,
  createStaticPaths,
  formatDistance,
  getMetersPerPixel,
  importPathOutput,
  type GeometryView
} from './b3-common';
import {KeptSegmentLayer, PathOutputLayer} from './b3-layers';
import type {GreatCirclesOptions, LineOperationsOptions} from './b3-line-options';

type Options = LineOperationsOptions;
type WorldOptions = GreatCirclesOptions;

const LOCAL = COORDINATE_SYSTEM.METER_OFFSETS;
const LNGLAT = COORDINATE_SYSTEM.LNGLAT;

// ---------------------------------------------------------------------------------------------
// Tracks: GPULineSimplification and GPULineSmooth on AIS ship tracks
// ---------------------------------------------------------------------------------------------

export type TrackEnvironment = {
  ctx: SceneContext<Options>;
  vessels: LoadedDataset;
};

const SMOOTH_LEVEL_LIMIT = 5;

/**
 * 897 AIS ship tracks of New York Harbor on 12 June 2024. Douglas-Peucker importance is computed
 * once per metric, then the per-frame tolerance only re-selects the kept vertices; Chaikin smoothing
 * is compiled per iteration count.
 */
export function createTracksView(env: TrackEnvironment): GeometryView<Options> {
  const {ctx, vessels} = env;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'tracks');
  const origin = vessels.defaultOrigin;
  const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
  const local = vessels.projectColumn('vertices', origin);
  const offsets = vessels.column<Uint32Array>('pathOffsets');
  const timestamps = vessels.column<Uint32Array>('timestamp');
  const vertexCount = local.length / 2;
  const trackCount = offsets.length - 1;

  const positions = resources.createBuffer('positions', local);
  const trackOffsets = resources.createBuffer('track-offsets', offsets);
  const times = new Float32Array(vertexCount);
  for (let row = 0; row < vertexCount; row++) times[row] = timestamps[row];
  const timeBuffer = resources.createBuffer('timestamps', times);
  const vertexLines = new Uint32Array(vertexCount);
  for (let track = 0; track < trackCount; track++) {
    for (let row = offsets[track]; row < offsets[track + 1]; row++) vertexLines[row] = track;
  }
  const vertexLineBuffer = resources.createBuffer('vertex-lines', vertexLines);
  const staticTracks = createStaticPaths(resources, 'tracks-static', local, offsets);
  const parameters = resources.createParameterBuffer(
    'tolerance',
    'float32',
    GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH
  );

  type Importance = {
    key: string;
    importance: Buffer;
    converged: Buffer;
    rounds: Buffer;
    compiled: CompiledGPUCommandGraph<void>;
    encoded: boolean;
    selection: {
      compiled: CompiledGPUCommandGraph<void>;
      ids: Buffer;
      count: Buffer;
      overflow: Buffer;
      total: Buffer;
      drawCommands: DrawCommandBuffer;
    };
  };
  const importances = new Map<string, Importance>();
  let activeImportance: Importance | null = null;
  let appliedTolerance = Number.NaN;
  let readbackNeeded = true;
  let reader: SummaryReader | null = null;
  let readerOwner: Importance | null = null;
  let currentTolerance = 0;
  const parameterValues = new Float32Array(GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH);

  const buildImportance = (options: Options): Importance => {
    const key = `${options.simplifyMetric}|${options.simplifyRounds}`;
    const cached = importances.get(key);
    if (cached) return cached;
    const maximumRounds = Number(options.simplifyRounds);
    const importance = resources.createBuffer(`importance-${key}`, vertexCount * 4);
    const converged = resources.createBuffer(`converged-${key}`, 4);
    const rounds = resources.createBuffer(`rounds-${key}`, 4);
    const ids = resources.createBuffer(`ids-${key}`, vertexCount * 4);
    const count = resources.createBuffer(`count-${key}`, 4);
    const overflow = resources.createBuffer(`overflow-${key}`, 4);
    const total = resources.createBuffer(`total-${key}`, 4);

    const views = (graph: GPUCommandGraph<void>) => {
      const imp = createGraphImporter(graph);
      return {
        imp,
        base: {
          positions: imp('positions', positions, 'float32x2', vertexCount),
          trackOffsets: imp('track-offsets', trackOffsets, 'uint32', trackCount + 1),
          timestamps: imp('timestamps', timeBuffer, 'float32', vertexCount),
          metric: options.simplifyMetric,
          importance: imp('importance', importance, 'float32', vertexCount)
        }
      };
    };
    const importanceGraph = new GPUCommandGraph<void>(device, {id: `simplify-importance-${key}`});
    {
      const {imp, base} = views(importanceGraph);
      importanceGraph.add(
        new GPULineSimplification({
          id: 'importance',
          ...base,
          maximumRounds,
          status: {
            converged: imp('converged', converged, 'uint32', 1),
            roundCount: imp('rounds', rounds, 'uint32', 1)
          }
        })
      );
    }
    const selectionGraph = new GPUCommandGraph<void>(device, {id: `simplify-selection-${key}`});
    {
      const {imp, base} = views(selectionGraph);
      selectionGraph.add(
        new GPULineSimplification({
          id: 'selection',
          ...base,
          computeImportance: false,
          parameters: parameters.importToGraph(selectionGraph),
          selection: {
            output: {
              ids: imp('ids', ids, 'uint32', vertexCount),
              count: imp('count', count, 'uint32', 1),
              overflow: imp('overflow', overflow, 'uint32', 1),
              totalCount: imp('total', total, 'uint32', 1)
            }
          }
        })
      );
    }
    const entry: Importance = {
      key,
      importance,
      converged,
      rounds,
      compiled: resources.track(importanceGraph.compile()),
      encoded: false,
      selection: {
        compiled: resources.track(selectionGraph.compile()),
        ids,
        count,
        overflow,
        total,
        drawCommands: resources.track(
          new DrawCommandBuffer(device, {
            id: `simplify-draw-${key}`,
            type: 'draw',
            commands: [{vertexCount: 6, instanceCount: 0}]
          })
        )
      }
    };
    importances.set(key, entry);
    return entry;
  };

  const startReader = (entry: Importance, maximumRounds: number) => {
    reader?.stop();
    readerOwner = entry;
    reader = new SummaryReader(
      resources,
      `simplify-${entry.key}`,
      [
        {buffer: entry.selection.count, size: 4},
        {buffer: entry.selection.overflow, size: 4},
        {buffer: entry.selection.total, size: 4},
        {buffer: entry.converged, size: 4},
        {buffer: entry.rounds, size: 4}
      ],
      bytes => {
        if (readerOwner !== entry) return;
        const words = new Uint32Array(bytes);
        ctx.setReadout(
          'tracksKept',
          `${formatCount(words[0])} of ${formatCount(vertexCount)}${words[1] ? ' (overflow)' : ''}`
        );
        ctx.setReadout('tracksRatio', `${((100 * words[2]) / vertexCount).toFixed(1)}% kept`);
        ctx.setReadout(
          'tracksConverged',
          words[3]
            ? 'yes: equals classic Douglas-Peucker'
            : `no: stopped at the ${maximumRounds}-round cap (a superset)`
        );
        ctx.setReadout('tracksRounds', `${words[4]} of ${maximumRounds}`);
      }
    );
  };

  // Smoothing levels, compiled per iteration count.
  type Smooth = {
    iterations: number;
    compiled: CompiledGPUCommandGraph<void>;
    output: ReturnType<typeof createPathOutputBuffers>;
    reader: SummaryReader;
  };
  const smoothLevels = new Map<number, Smooth>();
  const smoothParameters = resources.createParameterBuffer(
    'smooth-parameters',
    'float32',
    GPU_LINE_SMOOTH_PARAMETER_LENGTH
  );
  let smoothDirty = true;
  const buildSmooth = (iterations: number): Smooth => {
    const existing = smoothLevels.get(iterations);
    if (existing) return existing;
    const capacity = vertexCount * 2 ** iterations + 16;
    const output = createPathOutputBuffers(resources, `smooth-${iterations}`, capacity, trackCount);
    const graph = new GPUCommandGraph<void>(device, {id: `smooth-${iterations}`});
    const imp = createGraphImporter(graph);
    graph.add(
      new GPULineSmooth({
        id: `smooth-${iterations}`,
        positions: imp('positions', positions, 'float32x2', vertexCount),
        pathOffsets: imp('offsets', trackOffsets, 'uint32', trackCount + 1),
        iterations,
        closed: false,
        parameters: smoothParameters.importToGraph(graph),
        output: importPathOutput(graph, output)
      })
    );
    const reader = new SummaryReader(
      resources,
      `smooth-${iterations}`,
      [
        {buffer: output.count, size: 4},
        {buffer: output.overflow, size: 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        ctx.setReadout(
          'tracksSmoothVertices',
          `${formatCount(words[0])} (${(words[0] / vertexCount).toFixed(1)}× the input)${words[1] ? ', overflow' : ''}`
        );
      }
    );
    const level: Smooth = {iterations, compiled: resources.track(graph.compile()), output, reader};
    smoothLevels.set(iterations, level);
    return level;
  };

  const select = (options: Options) => {
    if (options.trackTool === 'simplify') {
      activeImportance = buildImportance(options);
      startReaderIfNeeded(options);
    } else {
      buildSmooth(Math.min(options.smoothIterations, SMOOTH_LEVEL_LIMIT));
      smoothParameters.write(getGPULineSmoothParameterValues({ratio: options.smoothRatio}));
      smoothDirty = true;
    }
    ctx.setReadout(
      'tracksInputs',
      `${formatCount(trackCount)} tracks, ${formatCount(vertexCount)} fixes`
    );
  };
  const startReaderIfNeeded = (options: Options) => {
    if (activeImportance && readerOwner !== activeImportance) {
      startReader(activeImportance, Number(options.simplifyRounds));
      appliedTolerance = Number.NaN;
      readbackNeeded = true;
    }
  };
  select(ctx.options);

  return {
    getCompiledGraphs() {
      const graphs: CompiledGPUCommandGraph<never>[] = [];
      for (const entry of importances.values()) {
        graphs.push(
          entry.compiled as CompiledGPUCommandGraph<never>,
          entry.selection.compiled as CompiledGPUCommandGraph<never>
        );
      }
      for (const level of smoothLevels.values())
        graphs.push(level.compiled as CompiledGPUCommandGraph<never>);
      return graphs;
    },
    setOption(id, options) {
      if (
        id === 'trackTool' ||
        id === 'simplifyMetric' ||
        id === 'simplifyRounds' ||
        id === 'smoothIterations'
      ) {
        select(options);
      }
      if (id === 'smoothRatio') {
        smoothParameters.write(getGPULineSmoothParameterValues({ratio: options.smoothRatio}));
        smoothDirty = true;
      }
    },
    encode(commandEncoder, frame) {
      const options = ctx.options;
      if (options.trackTool === 'smooth') {
        const level = smoothLevels.get(Math.min(options.smoothIterations, SMOOTH_LEVEL_LIMIT));
        if (level && smoothDirty) {
          level.compiled.encode(commandEncoder, {parameters: undefined});
          copyCountToDrawRecord(commandEncoder, level.output.count, level.output.drawCommands);
          level.reader.request(commandEncoder);
          smoothDirty = false;
        } else {
          level?.reader.flush(commandEncoder);
        }
        return;
      }
      const entry = activeImportance;
      if (!entry) return;
      currentTolerance = options.simplifyAuto
        ? options.simplifyPixels *
          getMetersPerPixel(
            frame.viewport.zoom,
            frame.viewport.unproject([frame.viewport.width / 2, frame.viewport.height / 2])[1]
          )
        : 10 ** options.simplifyTolerance;
      const tolerance = Math.fround(currentTolerance);
      if (!entry.encoded) {
        entry.compiled.encode(commandEncoder, {parameters: undefined});
        entry.encoded = true;
        appliedTolerance = Number.NaN;
      }
      if (tolerance !== appliedTolerance) {
        appliedTolerance = tolerance;
        parameters.write(getGPULineSimplificationParameterValues({tolerance}, parameterValues));
        entry.selection.compiled.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, entry.selection.count, entry.selection.drawCommands);
        ctx.setReadout('tracksTolerance', formatDistance(tolerance));
        readbackNeeded = true;
      }
      if (readbackNeeded) {
        reader?.request(commandEncoder);
        readbackNeeded = false;
      } else {
        reader?.flush(commandEncoder);
      }
    },
    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showOriginal) {
        layers.push(
          new PathOutputLayer({
            id: 'tracks-original',
            coordinateSystem: LOCAL,
            coordinateOrigin,
            positions: staticTracks.positions,
            pathOffsets: staticTracks.offsets,
            pathOffsetCount: staticTracks.offsetCount,
            vertexCount: staticTracks.vertexCount,
            drawCommands: staticTracks.drawCommands,
            color: dark ? [120, 170, 255, 80] : [60, 90, 150, 70],
            widthPixels: 1
          })
        );
      }
      if (options.trackTool === 'smooth') {
        const level = smoothLevels.get(Math.min(options.smoothIterations, SMOOTH_LEVEL_LIMIT));
        if (level) {
          layers.push(
            new PathOutputLayer({
              id: 'tracks-smooth',
              coordinateSystem: LOCAL,
              coordinateOrigin,
              positions: level.output.positions,
              pathOffsets: level.output.offsets,
              pathOffsetCount: level.output.offsetCount,
              vertexCount: level.output.count,
              drawCommands: level.output.drawCommands,
              color: [240, 150, 60, 255],
              widthPixels: 1.8
            })
          );
        }
      } else if (activeImportance) {
        layers.push(
          new KeptSegmentLayer({
            id: 'tracks-kept',
            coordinateSystem: LOCAL,
            coordinateOrigin,
            positions,
            keptIds: activeImportance.selection.ids,
            vertexLines: vertexLineBuffer,
            keptCount: activeImportance.selection.count,
            drawCommands: activeImportance.selection.drawCommands,
            color: [240, 150, 60, 255],
            widthPixels: 1.8
          })
        );
      }
      return layers;
    },
    destroy() {
      reader?.stop();
      for (const level of smoothLevels.values()) level.reader.stop();
      resources.destroy();
    }
  };
}

// ---------------------------------------------------------------------------------------------
// World: GPUGreatCircleArcs, GPUGeodesicPairs, GPUGeodesicDestination on the OpenFlights network
// ---------------------------------------------------------------------------------------------

export type WorldEnvironment = {
  ctx: SceneContext<WorldOptions>;
  flows: LoadedDataset;
  /** IATA code to airport row, from the dataset's airports table. */
  airportRows: Map<string, number>;
};

/** Airports offered as hubs: IATA code and the label shown in the panel. */
export const HUBS = [
  ['ORD', "Chicago O'Hare"],
  ['AMS', 'Amsterdam Schiphol'],
  ['DXB', 'Dubai'],
  ['SIN', 'Singapore Changi'],
  ['SYD', 'Sydney'],
  ['GRU', 'Sao Paulo Guarulhos'],
  ['JNB', 'Johannesburg'],
  ['HND', 'Tokyo Haneda']
] as const;

const ARC_MAXIMUM_SEGMENTS = 64;
const RING_VERTEX_COUNT = 361;
const WORLD_COPIES = [-360, 0, 360] as const;

/**
 * All 18,930 airport pairs of the OpenFlights route network as great-circle arcs, the distance and
 * bearing from a chosen hub to every airport (`GPUGeodesicPairs`, sphere or WGS84), and a geodesic
 * range ring around the hub (`GPUGeodesicDestination`).
 */
export function createWorldView(env: WorldEnvironment): GeometryView<WorldOptions> {
  const {ctx, flows, airportRows} = env;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'world');
  const airports = flows.column<Float32Array>('locations');
  const airportCount = airports.length / 2;
  const origins = flows.column<Uint32Array>('origin');
  const destinations = flows.column<Uint32Array>('destination');
  const pairCount = origins.length;
  const sourcePositions = new Float32Array(pairCount * 2);
  const targetPositions = new Float32Array(pairCount * 2);
  for (let pair = 0; pair < pairCount; pair++) {
    sourcePositions.set(airports.subarray(origins[pair] * 2, origins[pair] * 2 + 2), pair * 2);
    targetPositions.set(
      airports.subarray(destinations[pair] * 2, destinations[pair] * 2 + 2),
      pair * 2
    );
  }
  const distanceKm = flows.column<Float32Array>('distanceKm');
  const airlineCount = Float32Array.from(flows.column<Uint16Array>('airlineCount'));
  const routeCount = Float32Array.from(flows.column<Uint16Array>('count'));

  const sources = resources.createBuffer('arc-sources', sourcePositions);
  const targets = resources.createBuffer('arc-targets', targetPositions);
  const arcValues = {
    distance: resources.createBuffer('arc-distance', distanceKm),
    airlines: resources.createBuffer('arc-airlines', airlineCount),
    routes: resources.createBuffer('arc-routes', routeCount)
  };
  const arcVertexCapacity = pairCount * (ARC_MAXIMUM_SEGMENTS + 1);
  const arcs = createPathOutputBuffers(resources, 'arcs', arcVertexCapacity, pairCount);
  const arcParameters = resources.createParameterBuffer(
    'arc-parameters',
    'float32',
    GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH
  );
  const arcGraph = new GPUCommandGraph<void>(device, {id: 'arcs'});
  {
    const imp = createGraphImporter(arcGraph);
    arcGraph.add(
      new GPUGreatCircleArcs({
        id: 'arcs',
        sources: imp('sources', sources, 'float32x2', pairCount),
        targets: imp('targets', targets, 'float32x2', pairCount),
        maximumSegments: ARC_MAXIMUM_SEGMENTS,
        parameters: arcParameters.importToGraph(arcGraph),
        output: importPathOutput(arcGraph, arcs)
      })
    );
  }
  const compiledArcs = resources.track(arcGraph.compile());

  // Hub to every airport: distance, bearing and the great-circle midpoint.
  const hubOrigins = resources.createBuffer('hub-origins', new Float32Array(airportCount * 2));
  const airportBuffer = resources.createBuffer('airports', airports);
  const airportDistances = resources.createBuffer('airport-distances', airportCount * 4);
  const airportBearings = resources.createBuffer('airport-bearings', airportCount * 4);
  const airportConverged = resources.createBuffer('airport-converged', airportCount * 4);
  const pairGraphs = new Map<WorldOptions['geodesicModel'], CompiledGPUCommandGraph<void>>();
  const buildPairs = (model: WorldOptions['geodesicModel']) => {
    let compiled = pairGraphs.get(model);
    if (!compiled) {
      const graph = new GPUCommandGraph<void>(device, {id: `geodesic-pairs-${model}`});
      const imp = createGraphImporter(graph);
      graph.add(
        new GPUGeodesicPairs({
          id: `pairs-${model}`,
          origins: imp('origins', hubOrigins, 'float32x2', airportCount),
          targets: imp('airports', airportBuffer, 'float32x2', airportCount),
          model,
          output: {
            distances: imp('distances', airportDistances, 'float32', airportCount),
            initialBearings: imp('bearings', airportBearings, 'float32', airportCount),
            converged: imp('converged', airportConverged, 'uint32', airportCount)
          }
        })
      );
      compiled = resources.track(graph.compile());
      pairGraphs.set(model, compiled);
    }
    return compiled;
  };

  // The range ring: 361 bearings at one distance from the hub.
  const ringOrigins = resources.createBuffer(
    'ring-origins',
    new Float32Array(RING_VERTEX_COUNT * 2)
  );
  const ringBearings = resources.createBuffer(
    'ring-bearings',
    Float32Array.from({length: RING_VERTEX_COUNT}, (_, index) => index)
  );
  const ringDistances = resources.createBuffer('ring-distances', RING_VERTEX_COUNT * 4);
  const ringDestinations = resources.createBuffer('ring-destinations', RING_VERTEX_COUNT * 8);
  const ringPaths = createStaticPaths(
    resources,
    'ring-paths',
    new Float32Array(RING_VERTEX_COUNT * 2),
    Uint32Array.of(0, RING_VERTEX_COUNT)
  );
  const ringGraphs = new Map<WorldOptions['geodesicModel'], CompiledGPUCommandGraph<void>>();
  const buildRing = (model: WorldOptions['geodesicModel']) => {
    let compiled = ringGraphs.get(model);
    if (!compiled) {
      const graph = new GPUCommandGraph<void>(device, {id: `geodesic-ring-${model}`});
      const imp = createGraphImporter(graph);
      graph.add(
        new GPUGeodesicDestination({
          id: `ring-${model}`,
          origins: imp('origins', ringOrigins, 'float32x2', RING_VERTEX_COUNT),
          bearings: imp('bearings', ringBearings, 'float32', RING_VERTEX_COUNT),
          distances: imp('distances', ringDistances, 'float32', RING_VERTEX_COUNT),
          model,
          output: {
            destinations: imp('destinations', ringDestinations, 'float32x2', RING_VERTEX_COUNT)
          }
        })
      );
      compiled = resources.track(graph.compile());
      ringGraphs.set(model, compiled);
    }
    return compiled;
  };

  // Airport color values: distance in km, or bearing mapped to [0, 360).
  const airportValues = resources.createBuffer('airport-values', airportCount * 4);
  const airportBearingValues = resources.createBuffer('airport-bearing-values', airportCount * 4);
  const valueGraph = new GPUCommandGraph<void>(device, {id: 'airport-values'});
  {
    const imp = createGraphImporter(valueGraph);
    addKernelPass(valueGraph, {
      id: 'airport-values',
      invocationCount: airportCount,
      bindings: [
        {
          name: 'distances',
          view: imp('distances', airportDistances, 'float32', airportCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'bearings',
          view: imp('bearings', airportBearings, 'float32', airportCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'distanceKm',
          view: imp('airport-values', airportValues, 'float32', airportCount),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'bearing360',
          view: imp('airport-bearing-values', airportBearingValues, 'float32', airportCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  distanceKm[distanceKmOffset + index] = distances[distancesOffset + index] / 1000.0;
  bearing360[bearing360Offset + index] = (bearings[bearingsOffset + index] + 360.0) % 360.0;`
    });
  }
  const compiledValues = resources.track(valueGraph.compile());
  const valueReader = new SummaryReader(
    resources,
    'airport-distances',
    [{buffer: airportValues, size: airportCount * 4}],
    bytes => {
      const values = new Float32Array(bytes);
      const ring = ctx.options.ringDistance;
      let farthest = 0;
      let inside = 0;
      let total = 0;
      for (const value of values) {
        if (!Number.isFinite(value)) continue;
        farthest = Math.max(farthest, value);
        total += value;
        if (value <= ring) inside++;
      }
      ctx.setReadout('worldFarthest', `${formatCount(farthest)} km`);
      ctx.setReadout(
        'worldInside',
        `${formatCount(inside)} of ${formatCount(airportCount)} airports within ${formatCount(ring)} km`
      );
      ctx.setReadout('worldMean', `${formatCount(total / airportCount)} km`);
    }
  );

  let hubRow = 0;
  let dirtyArcs = true;
  let dirtyHub = true;
  let dirtyRing = true;
  const writeHub = (options: WorldOptions) => {
    hubRow = airportRows.get(options.worldHub) ?? 0;
    const longitude = airports[hubRow * 2];
    const latitude = airports[hubRow * 2 + 1];
    hubOrigins.write(
      Float32Array.from({length: airportCount * 2}, (_, index) =>
        index % 2 === 0 ? longitude : latitude
      )
    );
    ringOrigins.write(
      Float32Array.from({length: RING_VERTEX_COUNT * 2}, (_, index) =>
        index % 2 === 0 ? longitude : latitude
      )
    );
    dirtyHub = true;
    dirtyRing = true;
  };
  const writeRing = (options: WorldOptions) => {
    ringDistances.write(new Float32Array(RING_VERTEX_COUNT).fill(options.ringDistance * 1000));
    dirtyRing = true;
    dirtyHub = true;
  };
  const writeArcs = (options: WorldOptions) => {
    arcParameters.write(
      getGPUGreatCircleArcsParameterValues({
        maximumSegmentLength: options.arcMaximumLength * 1000,
        minimumSegments: options.arcMinimumSegments
      })
    );
    dirtyArcs = true;
  };
  writeHub(ctx.options);
  writeRing(ctx.options);
  writeArcs(ctx.options);
  buildPairs(ctx.options.geodesicModel);
  buildRing(ctx.options.geodesicModel);
  ctx.setReadout(
    'worldInputs',
    `${formatCount(airportCount)} airports, ${formatCount(pairCount)} routes`
  );

  const arcReader = new SummaryReader(
    resources,
    'arcs',
    [
      {buffer: arcs.count, size: 4},
      {buffer: arcs.overflow, size: 4}
    ],
    bytes => {
      const words = new Uint32Array(bytes);
      ctx.setReadout(
        'worldArcVertices',
        `${formatCount(words[0])} vertices (${(words[0] / pairCount).toFixed(1)} per arc)${words[1] ? ', overflow' : ''}`
      );
    }
  );

  return {
    getCompiledGraphs() {
      return [
        compiledArcs,
        compiledValues,
        ...pairGraphs.values(),
        ...ringGraphs.values()
      ] as CompiledGPUCommandGraph<never>[];
    },
    setOption(id, options) {
      if (id === 'worldHub') writeHub(options);
      if (id === 'ringDistance') writeRing(options);
      if (id === 'arcMinimumSegments' || id === 'arcMaximumLength') writeArcs(options);
      if (id === 'geodesicModel') {
        buildPairs(options.geodesicModel);
        buildRing(options.geodesicModel);
        dirtyHub = true;
        dirtyRing = true;
      }
    },
    encode(commandEncoder) {
      const options = ctx.options;
      if (dirtyArcs) {
        compiledArcs.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, arcs.count, arcs.drawCommands);
        arcReader.request(commandEncoder);
        dirtyArcs = false;
      } else {
        arcReader.flush(commandEncoder);
      }
      if (dirtyHub) {
        pairGraphs.get(options.geodesicModel)?.encode(commandEncoder, {parameters: undefined});
        compiledValues.encode(commandEncoder, {parameters: undefined});
        valueReader.request(commandEncoder);
        dirtyHub = false;
      } else {
        valueReader.flush(commandEncoder);
      }
      if (dirtyRing) {
        ringGraphs.get(options.geodesicModel)?.encode(commandEncoder, {parameters: undefined});
        dirtyRing = false;
      }
    },
    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showArcs) {
        const range: [number, number] =
          options.arcColor === 'distance'
            ? [0, 12000]
            : options.arcColor === 'airlines'
              ? [1, 8]
              : [1, 12];
        for (const offset of WORLD_COPIES) {
          layers.push(
            new PathOutputLayer({
              id: `world-arcs-${offset}`,
              coordinateSystem: LNGLAT,
              positionOffset: [offset, 0],
              positions: arcs.positions,
              pathOffsets: arcs.offsets,
              pathOffsetCount: arcs.offsetCount,
              vertexCount: arcs.count,
              drawCommands: arcs.drawCommands,
              values: arcValues[options.arcColor],
              colorSource: 'path-value',
              colormap: options.arcColor === 'distance' ? 'viridis' : 'magma',
              valueRange: range,
              color: [255, 255, 255, 255],
              widthPixels: 0.7,
              opacity: dark ? 0.35 : 0.5
            })
          );
        }
      }
      for (const offset of WORLD_COPIES) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `world-airports-${offset}`,
            coordinateSystem: LNGLAT,
            positions: airportBuffer,
            instanceCount: airportCount,
            radiusPixels: 2.6,
            values: options.airportColor === 'distance' ? airportValues : airportBearingValues,
            valueFormat: 'float32',
            colormap: options.airportColor === 'distance' ? 'inferno' : 'viridis',
            valueRange: options.airportColor === 'distance' ? [0, 15000] : [0, 360],
            positionOffset: [offset, 0]
          })
        );
      }
      if (options.showRing) {
        for (const offset of WORLD_COPIES) {
          layers.push(
            new PathOutputLayer({
              id: `world-ring-${offset}`,
              coordinateSystem: LNGLAT,
              positionOffset: [offset, 0],
              positions: ringDestinations,
              pathOffsets: ringPaths.offsets,
              pathOffsetCount: ringPaths.offsetCount,
              vertexCount: ringPaths.vertexCount,
              drawCommands: ringPaths.drawCommands,
              color: dark ? [255, 214, 120, 255] : [200, 90, 10, 255],
              widthPixels: 2.5
            })
          );
        }
      }
      return layers;
    },
    destroy() {
      arcReader.stop();
      valueReader.stop();
      resources.destroy();
    }
  };
}
