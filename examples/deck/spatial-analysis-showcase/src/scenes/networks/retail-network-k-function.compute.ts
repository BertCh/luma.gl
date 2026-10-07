// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {getClassTableLayerProps, makeClassTable} from '../../cartography/class-table';
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
import {buildRoadGraph, formatLength, ROAD_CLASS_NAMES, SegmentIndex} from './b10-road-graph';

/** Broad Overture retail categories used as network events. */
export type RetailCategory = 'retail' | 'restaurant_cafe' | 'grocery' | 'combined';

/** Option state of the retail network K-function scene. */
export type RetailNetworkKOptions = {
  category: RetailCategory;
  eventCount: number;
  sampleSeed: number;
  maxDistance: number;
  maxSnapDistance: number;
  simulations: number;
  envelopeSeed: number;
  bandCount: '12' | '24' | '48' | '96';
  rowsPerBlock: '32' | '64' | '128';
  spatialSort: boolean;
  roadStyle: 'perKm' | 'plain';
  showEvents: boolean;
};

/** Event rows of the compile-time K graph; unused rows sit far outside the network. */
export const EVENT_CAPACITY = 256;
/** Simulated patterns of the envelope (compile-time; `simulations` activates 0 to this many). */
export const SIMULATION_CAPACITY = 19;
const FAR_AWAY = 1e7;
const LOCAL_ITERATIONS = 16;
const MAXIMUM_ITERATIONS = 20;
const LOAD_TABLE = makeClassTable({
  breaks: [1, 3, 8, 20, 50],
  colors: [
    [105, 112, 125, 255],
    [255, 255, 178, 255],
    [254, 204, 92, 255],
    [253, 141, 60, 255],
    [227, 26, 28, 255],
    [128, 0, 38, 255]
  ]
});
const SEARCH_DISTANCE_TABLE = makeClassTable({
  breaks: [0.25, 0.5, 0.75],
  colors: [
    [215, 240, 255, 255],
    [145, 205, 245, 255],
    [70, 155, 220, 255],
    [20, 95, 175, 255]
  ]
});
const CATEGORY_LABELS: Record<RetailCategory, string> = {
  retail: 'Retail',
  restaurant_cafe: 'Restaurants and cafes',
  grocery: 'Groceries',
  combined: 'Shops, cafes and groceries'
};

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
 * Network-constrained K function of Chicago retail places on the street graph. One compiled graph
 * (`GPUNetworkKFunction`) snaps the sampled events onto the roads, runs bounded multi-source
 * shortest-path searches for the observed pattern and for every simulated pattern, and counts
 * event pairs per distance band. The maximum distance, snap distance, active simulations, seed
 * and the event positions are buffer writes; band count, rows per block and the snapping sort are
 * compile-time and rebuild the graph.
 */
export async function createRetailNetworkKFunction(
  ctx: SceneContext<RetailNetworkKOptions>
): Promise<SceneInstance<RetailNetworkKOptions>> {
  const roads = ctx.datasets.get('chicago-roads');
  const places = ctx.datasets.get('chicago-places');
  const {device} = ctx;
  const origin = roads.defaultOrigin;
  const graph = buildRoadGraph(roads, origin);
  const placePositions = places.projectColumn('position', origin);
  const placeCategories = places.column<Uint8Array>('category');
  const placeCount = placeCategories.length;
  const {nodeCount, slotCount} = graph;
  const resources = new SpatialAnalysisResources(device, 'retail-network-k');
  const segmentIndex = new SegmentIndex(graph.segments, graph.bounds);
  const projection = roads.getProjection(origin);

  // Categories follow chicago-places/manifest.json order: restaurant, grocery, retail.
  const matchesCategory = (row: number, category: RetailCategory): boolean => {
    const value = placeCategories[row];
    return category === 'combined'
      ? value === 0 || value === 2 || value === 7
      : value === 7
        ? category === 'retail'
        : value === 0
          ? category === 'restaurant_cafe'
          : value === 2 && category === 'grocery';
  };
  const eligibleCache = new Map<RetailCategory, Uint32Array>();
  const getEligible = (category: RetailCategory): Uint32Array => {
    let rows = eligibleCache.get(category);
    if (!rows) {
      const list: number[] = [];
      for (let row = 0; row < placeCount; row++) if (matchesCategory(row, category)) list.push(row);
      rows = Uint32Array.from(list);
      eligibleCache.set(category, rows);
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
  const slotPlaces = resources.createBuffer('slot-places', slotCount * 4);
  // `createBuffer(name, number)` receives bytes, not float elements. The reachable subset can
  // include every drawable segment, so reserve the whole source geometry.
  const searchBallSegments = resources.createBuffer(
    'network-search-ball',
    graph.segments.byteLength
  );
  const searchBallDistances = resources.createBuffer(
    'network-search-ball-distances',
    graph.segmentCount * 4
  );
  const radiusRingSegments = resources.createBuffer('straight-radius-ring', 96 * 16);
  const nullPatternPositions = resources.createBuffer('street-length-null-points', 48 * 8);
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
  let searchBallCount = 0;
  let selectedEventX = FAR_AWAY;
  let selectedEventY = FAR_AWAY;

  // The display-only null samples physical drawable segments once. The CSR has reverse slots;
  // sampling slots uniformly would therefore overweight short / multiply represented streets.
  const streetLengthCumulative = new Float64Array(graph.segmentCount);
  let streetLengthTotal = 0;
  for (let segment = 0; segment < graph.segmentCount; segment++) {
    const x0 = graph.segments[segment * 4];
    const y0 = graph.segments[segment * 4 + 1];
    const x1 = graph.segments[segment * 4 + 2];
    const y1 = graph.segments[segment * 4 + 3];
    streetLengthTotal += Math.hypot(x1 - x0, y1 - y0);
    streetLengthCumulative[segment] = streetLengthTotal;
  }

  ctx.setReadout(
    'network',
    `${formatCount(nodeCount)} intersections, ${formatCount(graph.edgeCount)} directed edges`
  );
  ctx.setReadout('length', formatLength(graph.networkLength));

  function publishFurniture(): void {
    const category = CATEGORY_LABELS[ctx.options.category];
    ctx.setFurniture({
      title: {
        title: 'Retail places on the street network',
        sample: `${category} · ${formatCount(eligibleCount)} mapped places · sample ${eventsUsed}`
      },
      scaleBar: {units: 'metric', ticks: [ctx.options.maxDistance]},
      credit: 'Overture Maps Foundation (mixed-permissive) · OpenStreetMap contributors (ODbL)',
      caveat: 'Street-length null; coverage, footfall and zoning are not controlled.'
    });
  }

  function writeSearchBall(selectedX: number, selectedY: number): void {
    const seedSegment = segmentIndex.nearest(selectedX, selectedY, ctx.options.maxSnapDistance);
    const distances = new Float64Array(nodeCount).fill(Infinity);
    if (seedSegment >= 0) {
      const edge = graph.segmentEdge[seedSegment];
      const x0 = graph.segments[seedSegment * 4];
      const y0 = graph.segments[seedSegment * 4 + 1];
      const x1 = graph.segments[seedSegment * 4 + 2];
      const y1 = graph.segments[seedSegment * 4 + 3];
      const dx = x1 - x0;
      const dy = y1 - y0;
      const fraction = Math.max(
        0,
        Math.min(
          1,
          ((selectedX - x0) * dx + (selectedY - y0) * dy) / Math.max(dx * dx + dy * dy, 1e-9)
        )
      );
      const length = graph.edgeLength[edge];
      const seeds: Array<[number, number]> = [
        [graph.edgeSource[edge], length * fraction],
        [graph.edgeTarget[edge], length * (1 - fraction)]
      ];
      const queue = [...seeds];
      for (const [cost, node] of seeds) distances[node] = Math.min(distances[node], cost);
      while (queue.length) {
        queue.sort((a, b) => a[0] - b[0]);
        const [cost, node] = queue.shift()!;
        if (cost !== distances[node] || cost > ctx.options.maxDistance) continue;
        for (let slot = graph.offsets[node]; slot < graph.offsets[node + 1]; slot++) {
          const next = graph.neighbors[slot];
          const nextCost = cost + graph.weights[slot];
          if (nextCost < distances[next] && nextCost <= ctx.options.maxDistance) {
            distances[next] = nextCost;
            queue.push([nextCost, next]);
          }
        }
      }
    }
    const reach: number[] = [];
    const reachDistances: number[] = [];
    for (let segment = 0; segment < graph.segmentCount; segment++) {
      const edge = graph.segmentEdge[segment];
      const distance = Math.min(
        distances[graph.edgeSource[edge]],
        distances[graph.edgeTarget[edge]]
      );
      if (distance <= ctx.options.maxDistance) {
        reach.push(...graph.segments.subarray(segment * 4, segment * 4 + 4));
        reachDistances.push(distance / Math.max(ctx.options.maxDistance, 1));
      }
    }
    searchBallCount = reachDistances.length;
    searchBallSegments.write(Float32Array.from(reach));
    searchBallDistances.write(Float32Array.from(reachDistances));
  }

  function writeNullPattern(): void {
    const nullRandom = createSeededRandom(ctx.options.envelopeSeed * 97 + 7);
    const randomPoints = new Float32Array(48 * 2);
    for (let index = 0; index < 48; index++) {
      const target = nullRandom() * streetLengthTotal;
      let low = 0;
      let high = streetLengthCumulative.length - 1;
      while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (streetLengthCumulative[middle] > target) high = middle;
        else low = middle + 1;
      }
      const segment = low;
      const start = segment === 0 ? 0 : streetLengthCumulative[segment - 1];
      const fraction = (target - start) / Math.max(streetLengthCumulative[segment] - start, 1e-9);
      const base = segment * 4;
      randomPoints[index * 2] =
        graph.segments[base] * (1 - fraction) + graph.segments[base + 2] * fraction;
      randomPoints[index * 2 + 1] =
        graph.segments[base + 1] * (1 - fraction) + graph.segments[base + 3] * fraction;
    }
    nullPatternPositions.write(randomPoints);
  }

  function writeRadiusRing(): void {
    const ring: number[] = [];
    for (let part = 0; part < 96; part++) {
      const a = (part / 96) * Math.PI * 2;
      const b = ((part + 1) / 96) * Math.PI * 2;
      ring.push(
        selectedEventX + Math.cos(a) * ctx.options.maxDistance,
        selectedEventY + Math.sin(a) * ctx.options.maxDistance,
        selectedEventX + Math.cos(b) * ctx.options.maxDistance,
        selectedEventY + Math.sin(b) * ctx.options.maxDistance
      );
    }
    radiusRingSegments.write(Float32Array.from(ring));
  }

  /** Writes sampled places and per-kilometre street loads for the selected category. */
  function writeEvents(): void {
    const {category, eventCount, sampleSeed} = ctx.options;
    const rows = getEligible(category);
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
      positions[slot * 2] = placePositions[order[slot] * 2];
      positions[slot * 2 + 1] = placePositions[order[slot] * 2 + 1];
    }
    eventBuffer.write(positions);

    selectedEventX = positions[0];
    selectedEventY = positions[1];
    // CPU provenance display only; the contributor performs the K searches on the GPU.
    writeSearchBall(selectedEventX, selectedEventY);
    writeRadiusRing();
    writeNullPattern();

    const counts = new Float32Array(slotCount);
    for (const row of rows) {
      const segment = segmentIndex.nearest(
        placePositions[row * 2],
        placePositions[row * 2 + 1],
        ctx.options.maxSnapDistance
      );
      if (segment < 0) continue;
      const slot = graph.segmentSlots[segment];
      counts[slot] += 1000 / Math.max(50, graph.edgeLength[graph.segmentEdge[segment]]);
    }
    slotPlaces.write(counts);
    placeCounts = counts;
    ctx.setReadout('eligible', eligibleCount);
    ctx.setReadout('eventsUsed', eventsUsed);
    ctx.setReadout(
      'sampleFraction',
      `${eventsUsed} of ${eligibleCount} eligible (${eligibleCount ? ((100 * eventsUsed) / eligibleCount).toFixed(1) : '0'}%)`
    );
    publishFurniture();
    kDirty = true;
  }
  let placeCounts: Float32Array = new Float32Array(slotCount);

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
    const kGraph = new GPUCommandGraph<void>(device, {id: 'retail-network-k'});
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
    let firstOutside = -1;
    for (let band = 1; band < bandCount; band++) {
      if (hasEnvelope && observed[band] > upper[band]) {
        above++;
        if (firstOutside < 0) firstOutside = band;
      } else if (hasEnvelope && observed[band] < lower[band]) {
        below++;
        if (firstOutside < 0) firstOutside = band;
      }
    }
    ctx.setReadout(
      'events',
      `${formatCount(snapped)} of ${eventsUsed}${overflow ? ' (snap overflow)' : ''}`
    );
    const distances = Array.from(
      {length: bandCount},
      (_, band) => ((band + 1) * ctx.options.maxDistance) / bandCount
    );
    const ratioSafe = hasEnvelope && mean.every(value => value > 0 && Number.isFinite(value));
    const transform = (values: Float32Array) =>
      ratioSafe ? Array.from(values, (value, index) => value / mean[index]) : values;
    const observedSeries = transform(observed);
    const lowerSeries = transform(lower);
    const upperSeries = transform(upper);
    ctx.setChart('kCurve', {
      kind: 'line',
      xLabel: 'network distance (m)',
      yLabel: ratioSafe ? 'K / simulated mean' : 'K(d)',
      series: [
        {label: 'observed', x: distances, y: observedSeries, color: 0, width: 2.4},
        {
          label: 'simulated mean',
          x: distances,
          y: ratioSafe ? distances.map(() => 1) : Array.from(mean),
          color: 1,
          dashed: true
        },
        {
          label: 'observed outside envelope',
          x: distances.filter(
            (_, band) =>
              hasEnvelope && (observed[band] > upper[band] || observed[band] < lower[band])
          ),
          y: observedSeries.filter(
            (_, band) =>
              hasEnvelope && (observed[band] > upper[band] || observed[band] < lower[band])
          ),
          color: 0,
          points: true,
          width: 0
        }
      ],
      band: {
        x: distances,
        low: lowerSeries,
        high: upperSeries,
        label: 'pointwise min–max envelope'
      },
      guides: ratioSafe ? [{y: 1, label: 'street-length null'}] : undefined,
      markers: hasEnvelope
        ? distances.flatMap((distance, band) =>
            observed[band] > upper[band] || observed[band] < lower[band]
              ? [{x: distance, label: band === firstOutside ? 'first outside' : 'outside'}]
              : []
          )
        : [{x: ctx.options.maxDistance, label: 'maximum d'}],
      link: {option: 'maxDistance', label: value => `${value} m`},
      description:
        'Observed network K against the active simulated street-length null. The grey band is pointwise, not a global test.'
    });
    ctx.setReadout(
      'firstOutside',
      firstOutside < 0
        ? 'none in active pointwise envelope'
        : `${Math.round(distances[firstOutside])} m (pointwise, not a global test)`
    );
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
        : `${above} above and ${below} below the pointwise envelope; not a global test`
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
        case 'category':
        case 'eventCount':
        case 'sampleSeed':
          writeEvents();
          ctx.requestLayers();
          break;
        case 'maxDistance':
        case 'simulations':
          writeParameters();
          if (id === 'maxDistance') {
            writeSearchBall(selectedEventX, selectedEventY);
            writeRadiusRing();
            publishFurniture();
          }
          ctx.setReadout('rows', (1 + state.simulations) * EVENT_CAPACITY);
          break;
        case 'maxSnapDistance':
          writeParameters();
          writeEvents(); // Rebuild selected reachability and first-order loads together.
          break;
        case 'envelopeSeed':
          writeParameters();
          writeNullPattern();
          publishFurniture();
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
      const placesHere = placeCounts[slot];
      return `${ROAD_CLASS_NAMES[graph.edgeClass[edge]]} block, ${Math.round(graph.edgeLength[edge])} m, ${graph.edgeSpeed[edge]} km/h\n${placesHere.toFixed(1)} mapped places per km`;
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
      const {roadStyle, showEvents} = ctx.options;
      const dark = ctx.ground() === 'dark';
      const layers: Layer[] = [
        new SpatialAnalysisSegmentLayer({
          id: 'retail-network-k-roads',
          coordinateOrigin,
          segments: segmentsBuffer,
          instanceCount: graph.segmentCount,
          widthPixels: 0.55,
          color: dark
            ? [150, 160, 190, roadStyle === 'plain' ? 130 : 70]
            : [120, 130, 155, roadStyle === 'plain' ? 130 : 80]
        })
      ];
      if (roadStyle === 'perKm') {
        for (const [classIndex, widthPixels] of [0.8, 1.2, 1.8, 2.6, 3.1, 3.4].entries())
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `retail-network-k-place-load-${classIndex}`,
              coordinateOrigin,
              segments: segmentsBuffer,
              instanceCount: graph.segmentCount,
              values: slotPlaces,
              valueFormat: 'float32',
              valueIndices: segmentSlotsBuffer,
              ...getClassTableLayerProps(LOAD_TABLE),
              highlightClasses: [classIndex],
              dimOpacity: 0,
              widthPixels
            })
          );
      }
      if (showEvents) {
        layers.push(
          ...[0.8, 1.15, 1.5, 1.9].map(
            (widthPixels, classIndex) =>
              new SpatialAnalysisSegmentLayer({
                id: `retail-network-search-ball-${classIndex}`,
                coordinateOrigin,
                segments: searchBallSegments,
                instanceCount: searchBallCount,
                widthPixels,
                values: searchBallDistances,
                valueFormat: 'float32',
                ...getClassTableLayerProps(SEARCH_DISTANCE_TABLE),
                highlightClasses: [classIndex],
                dimOpacity: 0,
                color: dark ? [95, 185, 255, 180] : [20, 110, 175, 170]
              })
          ),
          new SpatialAnalysisSegmentLayer({
            id: 'retail-straight-radius-ring',
            coordinateOrigin,
            segments: radiusRingSegments,
            instanceCount: 96,
            widthPixels: 1,
            dashArray: [4, 3],
            color: dark ? [215, 225, 240, 180] : [55, 70, 95, 150]
          }),
          new SpatialAnalysisPointLayer({
            id: 'retail-street-length-null',
            coordinateOrigin,
            positions: nullPatternPositions,
            instanceCount: 48,
            radiusPixels: 3,
            shape: 'ring',
            color: dark ? [210, 215, 225, 170] : [80, 90, 105, 170]
          }),
          new SpatialAnalysisPointLayer({
            id: 'retail-network-k-events',
            coordinateOrigin,
            positions: eventBuffer,
            instanceCount: eventsUsed,
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
