// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUNetworkKFunctionParameterValues,
  GPUNetworkKFunction,
  GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-network';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createSeededRandom} from '../../engine/projection';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  buildRoadGraph,
  formatLength,
  NO_EDGE,
  ROAD_CLASS_NAMES,
  SegmentIndex,
  sparkBar
} from './b10-road-graph';

/** Which crashes become events. */
export type CrashEventSet = 'all' | 'injury' | 'severe' | 'fatal' | 'rush' | 'night';

/** Option state of the crash K-function scene. */
export type CrashKOptions = {
  events: CrashEventSet;
  eventCount: number;
  sampleSeed: number;
  maxDistance: number;
  maxSnapDistance: number;
  simulations: number;
  envelopeSeed: number;
  bandCount: '12' | '24' | '48' | '96';
  rowsPerBlock: '32' | '64' | '128';
  spatialSort: boolean;
  roadStyle: 'crashes' | 'plain';
  ramp: 'inferno' | 'magma' | 'viridis' | 'cividis';
  showEvents: boolean;
};

/** Event rows of the compile-time K graph; unused rows sit far outside the network. */
export const EVENT_CAPACITY = 256;
/** Simulated patterns of the envelope (compile-time; `simulations` activates 0 to this many). */
export const SIMULATION_CAPACITY = 19;
const FAR_AWAY = 1e7;
const LOCAL_ITERATIONS = 16;
const MAXIMUM_ITERATIONS = 20;
const SECONDS_PER_HOUR = 3600;

type KBuild = {
  compiled: CompiledGPUCommandGraph<void>;
  bandCount: number;
  kValues: Buffer;
  envelope: Buffer;
  pairCounts: Buffer;
  converged: Buffer;
  reader: SummaryReader;
};

/**
 * Network-constrained K function of Chicago crashes on the street graph. One compiled graph
 * (`GPUNetworkKFunction`) snaps the sampled events onto the roads, runs bounded multi-source
 * shortest-path searches for the observed pattern and for every simulated pattern, and counts
 * event pairs per distance band. The maximum distance, snap distance, active simulations, seed
 * and the event positions are buffer writes; band count, rows per block and the snapping sort are
 * compile-time and rebuild the graph.
 */
export async function createCrashKFunction(
  ctx: SceneContext<CrashKOptions>
): Promise<SceneInstance<CrashKOptions>> {
  const roads = ctx.datasets.get('chicago-roads');
  const crashes = ctx.datasets.get('chicago-crashes');
  const {device} = ctx;
  const origin = roads.defaultOrigin;
  const graph = buildRoadGraph(roads, origin);
  const crashPositions = crashes.projectColumn('position', origin);
  const crashTimes = crashes.column<Uint32Array>('timestamp');
  const crashSeverity = crashes.column<Uint8Array>('severity');
  const crashInjuries = crashes.column<Uint8Array>('injuries');
  const crashFatalities = crashes.column<Uint8Array>('fatalities');
  const crashEdges = crashes.column<Uint32Array>('edgeIndex');
  const crashCount = crashTimes.length;
  const {nodeCount, slotCount} = graph;
  const resources = new SpatialAnalysisResources(device, 'crash-k');
  const segmentIndex = new SegmentIndex(graph.segments, graph.bounds);
  const projection = roads.getProjection(origin);

  // ---- Event sets: eligible crash rows per preset, counted per street for the map ----
  const matchesSet = (row: number, set: CrashEventSet): boolean => {
    if (crashEdges[row] === NO_EDGE) return false;
    const hour = (crashTimes[row] / SECONDS_PER_HOUR) % 24;
    const weekday = Math.floor(crashTimes[row] / 86400) % 7; // 2023-01-01 is a Sunday
    switch (set) {
      case 'all':
        return true;
      case 'injury':
        return crashInjuries[row] > 0;
      case 'severe':
        return crashSeverity[row] >= 3;
      case 'fatal':
        return crashFatalities[row] > 0;
      case 'rush':
        return weekday >= 1 && weekday <= 5 && hour >= 15 && hour < 19;
      case 'night':
        return hour < 5;
    }
  };
  const eligibleCache = new Map<CrashEventSet, Uint32Array>();
  const getEligible = (set: CrashEventSet): Uint32Array => {
    let rows = eligibleCache.get(set);
    if (!rows) {
      const list: number[] = [];
      for (let row = 0; row < crashCount; row++) if (matchesSet(row, set)) list.push(row);
      rows = Uint32Array.from(list);
      eligibleCache.set(set, rows);
    }
    return rows;
  };

  // ---- Buffers ----
  const eventBuffer = resources.createBuffer('events', EVENT_CAPACITY * 8);
  const nodeBuffer = resources.createBuffer('node-positions', graph.nodePositions);
  const offsetsBuffer = resources.createBuffer('offsets', graph.offsets);
  const neighborsBuffer = resources.createBuffer('neighbors', graph.neighbors);
  const weightsBuffer = resources.createBuffer('weights', graph.weights);
  const segmentsBuffer = resources.createBuffer('segments', graph.segments);
  const segmentSlotsBuffer = resources.createBuffer('segment-slots', graph.segmentSlots);
  const slotCrashes = resources.createBuffer('slot-crashes', slotCount * 4);
  const maxDistanceParameter = resources.createParameterBuffer('max-distance', 'float32', 1);
  const snapParameter = resources.createParameterBuffer('max-snap-distance', 'float32', 1);
  const lengthParameter = resources.createParameterBuffer(
    'network-length',
    'float32',
    1,
    Float32Array.of(graph.networkLength)
  );
  const kParameters = resources.createParameterBuffer(
    'k-parameters',
    'uint32',
    GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH
  );
  const snappedCount = resources.createBuffer('snapped-count', 4);
  const snapOverflow = resources.createBuffer('snap-overflow', 4);

  let destroyed = false;
  let kDirty = true;
  let build: KBuild | null = null;
  let eventsUsed = 0;
  let eligibleCount = 0;
  let measuring = false;

  ctx.setReadout(
    'network',
    `${formatCount(nodeCount)} intersections, ${formatCount(graph.edgeCount)} directed edges`
  );
  ctx.setReadout('length', formatLength(graph.networkLength));

  /** Writes the sampled events and the per-street crash counts of the selected event set. */
  function writeEvents(): void {
    const {events, eventCount, sampleSeed} = ctx.options;
    const rows = getEligible(events);
    eligibleCount = rows.length;
    const order = Array.from(rows);
    const random = createSeededRandom(sampleSeed * 7919 + 13);
    for (let index = order.length - 1; index > 0; index--) {
      const swap = Math.floor(random() * (index + 1));
      [order[index], order[swap]] = [order[swap], order[index]];
    }
    const positions = new Float32Array(EVENT_CAPACITY * 2).fill(FAR_AWAY);
    eventsUsed = Math.min(eventCount, EVENT_CAPACITY, order.length);
    for (let slot = 0; slot < eventsUsed; slot++) {
      positions[slot * 2] = crashPositions[order[slot] * 2];
      positions[slot * 2 + 1] = crashPositions[order[slot] * 2 + 1];
    }
    eventBuffer.write(positions);

    const counts = new Float32Array(slotCount);
    let maximum = 1;
    for (const row of rows) {
      const edge = crashEdges[row];
      const reverse = graph.edgeReverse[edge];
      const drawn = reverse === NO_EDGE || edge < reverse ? edge : reverse;
      const slot = graph.slotOfEdge[drawn];
      counts[slot]++;
      maximum = Math.max(maximum, counts[slot]);
    }
    slotCrashes.write(counts);
    crashCounts = counts;
    crashMaximum = maximum;
    ctx.setLegendExtent('crashes', [0, maximum]);
    ctx.setReadout('eligible', eligibleCount);
    ctx.setReadout('eventsUsed', eventsUsed);
    kDirty = true;
  }
  let crashCounts: Float32Array = new Float32Array(slotCount);
  let crashMaximum = 1;

  function writeParameters(): void {
    const {maxDistance, maxSnapDistance, simulations, envelopeSeed} = ctx.options;
    maxDistanceParameter.write(Float32Array.of(maxDistance));
    snapParameter.write(Float32Array.of(maxSnapDistance));
    kParameters.write(
      getGPUNetworkKFunctionParameterValues({seed: envelopeSeed, activeSimulations: simulations})
    );
    kDirty = true;
  }

  function buildKGraph(): void {
    if (build) {
      build.reader.stop();
      resources.release(build.compiled);
      for (const buffer of [build.kValues, build.envelope, build.pairCounts, build.converged]) {
        resources.release(buffer);
      }
      build = null;
    }
    const bandCount = Number(ctx.options.bandCount);
    const rowsPerBlock = Number(ctx.options.rowsPerBlock);
    const patternCount = 1 + SIMULATION_CAPACITY;
    const kValues = resources.createBuffer(`k-values-${bandCount}`, patternCount * bandCount * 4);
    const envelope = resources.createBuffer(`envelope-${bandCount}`, 3 * bandCount * 4);
    const pairCounts = resources.createBuffer(
      `pair-counts-${bandCount}`,
      patternCount * bandCount * 4
    );
    const converged = resources.createBuffer(`converged-${bandCount}`, 4);
    const kGraph = new GPUCommandGraph<void>(device, {id: 'crash-network-k'});
    kGraph.add(
      new GPUNetworkKFunction({
        id: 'k',
        points: importGraphBuffer(kGraph, 'events', eventBuffer, 'float32x2', EVENT_CAPACITY),
        nodePositions: importGraphBuffer(
          kGraph,
          'node-positions',
          nodeBuffer,
          'float32x2',
          nodeCount
        ),
        offsets: importGraphBuffer(kGraph, 'offsets', offsetsBuffer, 'uint32', nodeCount + 1),
        neighbors: importGraphBuffer(kGraph, 'neighbors', neighborsBuffer, 'uint32', slotCount),
        weights: importGraphBuffer(kGraph, 'weights', weightsBuffer, 'float32', slotCount),
        maxSnapDistance: snapParameter.importToGraph(kGraph),
        maxDistance: maxDistanceParameter.importToGraph(kGraph),
        networkLength: lengthParameter.importToGraph(kGraph),
        parameters: kParameters.importToGraph(kGraph),
        bandCount,
        simulationCount: SIMULATION_CAPACITY,
        rowsPerBlock,
        maxIterations: MAXIMUM_ITERATIONS,
        localIterations: LOCAL_ITERATIONS,
        spatialSort: ctx.options.spatialSort,
        candidateCapacity: EVENT_CAPACITY * 512,
        kValues: importGraphBuffer(
          kGraph,
          'k-values',
          kValues,
          'float32',
          patternCount * bandCount
        ),
        pairCounts: importGraphBuffer(
          kGraph,
          'pair-counts',
          pairCounts,
          'uint32',
          patternCount * bandCount
        ),
        envelope: importGraphBuffer(kGraph, 'envelope', envelope, 'float32', 3 * bandCount),
        snappedEventCount: importGraphBuffer(kGraph, 'snapped-count', snappedCount, 'uint32', 1),
        overflow: importGraphBuffer(kGraph, 'snap-overflow', snapOverflow, 'uint32', 1),
        converged: importGraphBuffer(kGraph, 'converged', converged, 'uint32', 1)
      })
    );
    const compiled: CompiledGPUCommandGraph<void> = resources.track(kGraph.compile());
    const observedBytes = bandCount * 4;
    const reader = new SummaryReader(
      resources,
      `k-${bandCount}`,
      [
        {buffer: kValues, size: observedBytes},
        {buffer: envelope, size: 3 * observedBytes},
        {buffer: pairCounts, size: observedBytes},
        {buffer: snappedCount, size: 4},
        {buffer: converged, size: 4},
        {buffer: snapOverflow, size: 4}
      ],
      bytes => {
        if (destroyed || build?.reader !== reader) return;
        handleSummary(bytes, bandCount);
      }
    );
    build = {compiled, bandCount, kValues, envelope, pairCounts, converged, reader};
    ctx.setReadout('rows', (1 + ctx.options.simulations) * EVENT_CAPACITY);
    ctx.setReadout('blocks', Math.ceil((patternCount * EVENT_CAPACITY) / rowsPerBlock));
    kDirty = true;
  }

  function handleSummary(bytes: ArrayBuffer, bandCount: number): void {
    const floats = new Float32Array(bytes);
    const words = new Uint32Array(bytes);
    const observed = floats.slice(0, bandCount);
    const lower = floats.slice(bandCount, 2 * bandCount);
    const mean = floats.slice(2 * bandCount, 3 * bandCount);
    const upper = floats.slice(3 * bandCount, 4 * bandCount);
    const pairs = words[4 * bandCount + bandCount - 1];
    const flags = 5 * bandCount;
    const snapped = words[flags];
    const converged = words[flags + 1];
    const overflow = words[flags + 2];
    const hasEnvelope = ctx.options.simulations > 0;
    const last = bandCount - 1;
    let above = 0;
    let below = 0;
    let strip = '';
    let maxObserved = 0;
    for (let band = 1; band < bandCount; band++)
      maxObserved = Math.max(maxObserved, observed[band]);
    let profile = '';
    for (let band = 1; band < bandCount; band++) {
      profile += sparkBar(maxObserved > 0 ? observed[band] / maxObserved : 0);
      if (!hasEnvelope) {
        strip += '·';
      } else if (observed[band] > upper[band]) {
        above++;
        strip += '▲';
      } else if (observed[band] < lower[band]) {
        below++;
        strip += '▼';
      } else {
        strip += '·';
      }
    }
    ctx.setReadout(
      'events',
      `${formatCount(snapped)} of ${eventsUsed}${overflow ? ' (snap overflow)' : ''}`
    );
    ctx.setReadout('profile', profile);
    ctx.setReadout('strip', hasEnvelope ? strip : 'no envelope');
    ctx.setReadout(
      'bands',
      hasEnvelope ? `${above} above, ${below} below of ${bandCount - 1}` : 'no simulations'
    );
    ctx.setReadout('kAtMaximum', observed[last]);
    ctx.setReadout(
      'envelopeRange',
      hasEnvelope ? `${lower[last].toFixed(0)} to ${upper[last].toFixed(0)}` : null
    );
    ctx.setReadout(
      'ratio',
      hasEnvelope && mean[last] > 0 ? `${(observed[last] / mean[last]).toFixed(2)}x` : null
    );
    ctx.setReadout(
      'verdict',
      !hasEnvelope
        ? 'enable simulations'
        : above >= (bandCount - 1) * 0.8
          ? 'clustered along the streets'
          : below >= (bandCount - 1) * 0.8
            ? 'dispersed (more even than random)'
            : above + below === 0
              ? 'consistent with random'
              : 'mixed: clustered at some distances'
    );
    ctx.setReadout('pairs', pairs);
    ctx.setReadout('converged', converged ? 'yes' : 'no: raise iterations or lower distance');
  }

  async function measure(): Promise<void> {
    if (measuring || !build) return;
    measuring = true;
    try {
      const timing = await measureCompiledGraph(device, build.compiled, {
        parameters: undefined,
        completionBuffer: build.converged,
        runs: 5,
        repetitions: 2
      });
      if (!destroyed) {
        ctx.setReadout(
          'time',
          `${timing.milliseconds.toFixed(1)} ms (${timing.method === 'gpu-timestamps' ? 'GPU' : 'wall clock'})`
        );
      }
    } catch {
      // Device destroyed or measurement aborted.
    } finally {
      measuring = false;
    }
  }

  writeEvents();
  writeParameters();
  buildKGraph();

  const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];

  return {
    getCompiledGraphs: () => (build ? [build.compiled as CompiledGPUCommandGraph<never>] : []),

    setOption(id, _value, state) {
      switch (id) {
        case 'events':
        case 'eventCount':
        case 'sampleSeed':
          writeEvents();
          ctx.requestLayers();
          break;
        case 'maxDistance':
        case 'maxSnapDistance':
        case 'simulations':
        case 'envelopeSeed':
          writeParameters();
          ctx.setReadout('rows', (1 + state.simulations) * EVENT_CAPACITY);
          break;
        case 'bandCount':
        case 'rowsPerBlock':
        case 'spatialSort':
          buildKGraph();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'measure') void measure();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const segment = segmentIndex.nearest(x, y, 35);
      if (segment < 0) return null;
      const edge = graph.segmentEdge[segment];
      const slot = graph.segmentSlots[segment];
      const crashesHere = crashCounts[slot];
      return `${ROAD_CLASS_NAMES[graph.edgeClass[edge]]} block, ${Math.round(graph.edgeLength[edge])} m, ${graph.edgeSpeed[edge]} km/h\n${crashesHere} selected crash${crashesHere === 1 ? '' : 'es'} in 2023`;
    },

    encode(commandEncoder) {
      if (!build) return;
      if (kDirty) {
        build.compiled.encode(commandEncoder, {parameters: undefined});
        kDirty = false;
        build.reader.request(commandEncoder);
      } else {
        build.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      const {roadStyle, ramp, showEvents} = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [
        new SpatialAnalysisSegmentLayer({
          id: 'crash-k-roads',
          coordinateOrigin,
          segments: segmentsBuffer,
          instanceCount: graph.segmentCount,
          widthPixels: 1.1,
          color: dark
            ? [150, 160, 190, roadStyle === 'plain' ? 130 : 70]
            : [120, 130, 155, roadStyle === 'plain' ? 130 : 80]
        })
      ];
      if (roadStyle === 'crashes') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'crash-k-block-counts',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: graph.segmentCount,
            values: slotCrashes,
            valueFormat: 'float32',
            valueIndices: segmentSlotsBuffer,
            colormap: ramp,
            valueRange: [0, crashMaximum],
            sqrtScale: true,
            discardAtOrBelow: 0,
            widthPixels: 2.2
          })
        );
      }
      if (showEvents) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'crash-k-events',
            coordinateOrigin,
            positions: eventBuffer,
            instanceCount: EVENT_CAPACITY,
            radiusPixels: 3.6,
            color: dark ? [90, 230, 255, 235] : [0, 120, 200, 235]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      build?.reader.stop();
      resources.destroy();
    }
  };
}
