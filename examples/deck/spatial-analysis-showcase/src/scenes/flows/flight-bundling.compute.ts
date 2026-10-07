// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  createGPUEdgeBundlingParameterValues,
  GPU_EDGE_BUNDLING_WORK_BOX_PADDING,
  GPUEdgeBundling,
  type GPUEdgeBundlingParameterValues
} from '@luma.gl/experimental/gpu-network';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {formatCount, liveText} from '../../cartography/live-text';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {MapAnnotation, SceneContext, SceneInstance, TooltipContent} from '../scene';
import {
  expandAntimeridianEdges,
  getApproximateKilometres,
  readUsNetwork,
  readWorldNetwork,
  type ExpandedEdges,
  type FlightNetwork
} from './b11-flight-data';
import {loadAirportTable} from './b11-geography';
import {ADDITIVE_ROUTE_PARAMETERS, NORMAL_ROUTE_PARAMETERS} from './airline-network-layers';
import {BundleRibbonLayer} from './bixi-bundles-layers';
import {
  countInBins,
  getLivePairs,
  getRadiusAtIteration,
  getWorkBoxSideKilometres,
  resolveRoute,
  type ResolvedRoute
} from './flight-bundling-routes';
import {
  CORRIDOR_ANCHORS,
  DELAY_BREAKS,
  DISTANCE_BREAKS,
  getClassOf,
  getProbeColor,
  getRoutePalette,
  HUB_FILL,
  HUB_STROKE,
  PROBE_ROUTES,
  SLOT,
  UNDER_FLOOR_COLOR
} from './flight-bundling-style';

/** Option state of the flight-bundling scene. */
export type FlightBundlingOptions = {
  network: 'world' | 'us';
  region:
    | 'all'
    | 'Europe'
    | 'Asia'
    | 'North America'
    | 'South America'
    | 'Africa'
    | 'Oceania'
    | 'lower48';
  distanceRange: readonly [number, number];
  /** Flight floor of the delay colours (US network): pairs under it are grey. */
  minTraffic: number;
  iterations: number;
  kernelRadius: number;
  decay: number;
  stiffness: number;
  stepScale: number;
  pointsPerEdge: '8' | '16' | '24' | '32';
  densityResolution: '128' | '256' | '512';
  colorBy: 'plain' | 'distance' | 'delay';
  brightness: number;
  straight: 'off' | 'ghost' | 'full';
  probes: boolean;
  corridorLabels: boolean;
  showAirports: boolean;
};

/** What the legends read back from the scene (`ctx.setLegendData('classStats', ...)`). */
export type FlightClassStats = {
  /** Live pairs per class of the active measure. */
  counts: number[];
  /** Live pairs under the flight floor (delay colours only). */
  underFloor: number;
};

/** Compile-time iteration capacity; the slider picks how many run. */
export const MAXIMUM_ITERATIONS = 32;
const HUB_COUNT = 24;
const RETIRE_FRAMES = 4;
const SETTLE_MILLISECONDS = 350;
type ProbeState = {
  route: ResolvedRoute;
  probe: (typeof PROBE_ROUTES)[number];
  /** The straight chord of the route as `x0, y0, x1, y1` rows, one per expanded edge. */
  segments: Buffer;
};

type CorridorState = {name: string; id: string; expandedEdges: readonly number[]};

type NetworkBuffers = {
  network: FlightNetwork;
  expanded: ExpandedEdges;
  vertexCount: number;
  edgeCount: number;
  positions: Buffer;
  sources: Buffer;
  targets: Buffer;
  mask: Buffer;
  /** Float liveness mirror consumed by the engine segment renderer. */
  maskWeights: Buffer;
  /** One float per expanded edge: the palette slot, `NaN` when the edge is filtered out. */
  slots: Buffer;
  /** The straight edges as two-point paths. */
  straight: Buffer;
  airports: Buffer;
  hubs: Buffer;
  hubRows: Uint32Array;
  /** Rank of every airport by connections, 0 = best connected. */
  degreeRank: Uint32Array;
  /** Probe routes found in this network (world only). */
  probes: ProbeState[];
  /** Corridor anchor routes found in this network (world only). */
  corridors: CorridorState[];
  totalFlights: number;
};

type BundlingGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  buffers: NetworkBuffers;
  pointsPerEdge: number;
  paths: Buffer;
  reader: SummaryReader;
};

/**
 * Kernel-density edge bundling of airline routes. Every airport pair is one edge; pairs that cross
 * the antimeridian are split in two so each half leaves the map on its own side. One compiled
 * `GPUEdgeBundling` graph turns the straight edges into bundled polylines: each iteration splats
 * control-point density, advects points along the density gradient, resamples and smooths. The
 * iteration count, kernel radius, decay, stiffness and step scale are one parameter buffer; the
 * edge mask hides edges without recompiling. Only `pointsPerEdge` and `densityResolution` rebuild.
 *
 * The scene colours every route by a palette slot written from the CPU (distance class, delay
 * class, probe) into one float per edge, and mirrors the GPU work box on the CPU so the kernel
 * radius can be quoted in kilometres.
 */
export async function createFlightBundling(
  ctx: SceneContext<FlightBundlingOptions>
): Promise<SceneInstance<FlightBundlingOptions>> {
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'flight-bundling');
  const [worldAirports, usAirports] = await Promise.all([
    loadAirportTable('openflights', ctx.signal),
    loadAirportTable('us-airline-flows', ctx.signal)
  ]);
  const networks = new Map<FlightNetwork['id'], NetworkBuffers>();
  const parameterBuffer = resources.createParameterBuffer('parameters', 'uint32', 5);

  let graph: BundlingGraph | null = null;
  let serial = 0;
  let destroyed = false;
  let encodeFrames = 3;
  let lastChange = performance.now();
  let statsStale = true;
  let liveEdges = 0;
  /** Per expanded edge: 1 when the edge passes the filters (mirrors the GPU `edgeMask`). */
  let liveMask: Uint32Array | null = null;
  /** Per original pair: 1 when the pair passes the filters. */
  let livePairs: Uint8Array | null = null;
  /** Work box side in km of the live edges, and of the same filters over the whole network. */
  let boxKilometres = 0;
  let worldBoxKilometres = 0;
  /** Mid point of each corridor anchor's bundled path, from the last readback. */
  let corridorPoints: {id: string; name: string; at: readonly [number, number]}[] = [];
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];

  function prepare(id: FlightNetwork['id']): NetworkBuffers {
    const existing = networks.get(id);
    if (existing) return existing;
    const network =
      id === 'world'
        ? readWorldNetwork(ctx.datasets.get('openflights'), worldAirports)
        : readUsNetwork(ctx.datasets.get('us-airline-flows'), usAirports);
    const expanded = expandAntimeridianEdges(network);
    const edgeCount = expanded.source.length;
    const segments = new Float32Array(edgeCount * 4);
    for (let edge = 0; edge < edgeCount; edge++) {
      segments.set(
        expanded.positions.subarray(expanded.source[edge] * 2, expanded.source[edge] * 2 + 2),
        edge * 4
      );
      segments.set(
        expanded.positions.subarray(expanded.target[edge] * 2, expanded.target[edge] * 2 + 2),
        edge * 4 + 2
      );
    }
    const order = Array.from({length: network.nodeCount}, (_, airport) => airport).sort(
      (a, b) => network.degree[b] - network.degree[a]
    );
    const degreeRank = new Uint32Array(network.nodeCount);
    order.forEach((airport, rank) => {
      degreeRank[airport] = rank;
    });
    const hubRows = Uint32Array.from(order.slice(0, HUB_COUNT));
    const probes: ProbeState[] = [];
    const corridors: CorridorState[] = [];
    if (id === 'world') {
      for (const probe of PROBE_ROUTES) {
        const route = resolveRoute(network, expanded, probe.from, probe.to);
        if (!route) continue;
        const chord = new Float32Array(route.expandedEdges.length * 4);
        route.expandedEdges.forEach((row, index) => {
          chord.set(segments.subarray(row * 4, row * 4 + 4), index * 4);
        });
        probes.push({
          route,
          probe,
          segments: resources.createBuffer(`${id}-probe-${probe.id}`, chord)
        });
      }
      for (const anchor of CORRIDOR_ANCHORS) {
        const route = resolveRoute(network, expanded, anchor.from, anchor.to);
        if (route)
          corridors.push({id: anchor.id, name: anchor.name, expandedEdges: route.expandedEdges});
      }
    }
    const prepared: NetworkBuffers = {
      network,
      expanded,
      vertexCount: expanded.positions.length / 2,
      edgeCount,
      positions: resources.createBuffer(`${id}-positions`, expanded.positions),
      sources: resources.createBuffer(`${id}-sources`, expanded.source),
      targets: resources.createBuffer(`${id}-targets`, expanded.target),
      mask: resources.createBuffer(`${id}-mask`, new Uint32Array(edgeCount).fill(1)),
      maskWeights: resources.createBuffer(
        `${id}-mask-weights`,
        new Float32Array(edgeCount).fill(1)
      ),
      slots: resources.createBuffer(`${id}-slots`, new Float32Array(edgeCount)),
      straight: resources.createBuffer(`${id}-straight`, segments),
      airports: resources.createBuffer(`${id}-airports`, network.lonLat),
      hubs: resources.createBuffer(`${id}-hubs`, hubRows),
      hubRows,
      degreeRank,
      probes,
      corridors,
      totalFlights: id === 'us' ? network.traffic.reduce((sum, value) => sum + value, 0) : 0
    };
    networks.set(id, prepared);
    return prepared;
  }

  function getParameters(): GPUEdgeBundlingParameterValues {
    const {iterations, kernelRadius, decay, stiffness, stepScale} = ctx.options;
    return {
      activeIterations: iterations,
      kernelRadius,
      lambda: decay,
      smoothing: stiffness,
      stepScale
    };
  }

  function writeParameters(redrawChart = true): void {
    parameterBuffer.write(createGPUEdgeBundlingParameterValues(getParameters(), 'uint32'));
    updateRadiusReadouts(redrawChart);
    markChanged();
  }

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 2);
    statsStale = true;
    lastChange = performance.now();
  }

  /** The attribute actually colouring the routes (a world network has no delay). */
  function getColorAttribute(): FlightBundlingOptions['colorBy'] {
    const {colorBy} = ctx.options;
    if (colorBy === 'delay' && !graph?.buffers.network.delay) return 'distance';
    return colorBy;
  }

  /** Writes the edge mask from the region and distance filters; then everything that follows from it. */
  function writeMask(): void {
    if (!graph) return;
    const {buffers} = graph;
    const {network, expanded} = buffers;
    const {region, distanceRange} = ctx.options;
    const passes = getLivePairs(network, region, distanceRange);
    const worldPasses = region === 'all' ? passes : getLivePairs(network, 'all', distanceRange);
    const mask = new Uint32Array(buffers.edgeCount);
    liveEdges = 0;
    for (let edge = 0; edge < buffers.edgeCount; edge++) {
      const live = passes[expanded.original[edge]];
      mask[edge] = live;
      liveEdges += live;
    }
    buffers.mask.write(mask);
    buffers.maskWeights.write(Float32Array.from(mask));
    liveMask = mask;
    livePairs = passes;
    boxKilometres = getWorkBoxSideKilometres(expanded, passes, GPU_EDGE_BUNDLING_WORK_BOX_PADDING);
    worldBoxKilometres = getWorkBoxSideKilometres(
      expanded,
      worldPasses,
      GPU_EDGE_BUNDLING_WORK_BOX_PADDING
    );
    const liveCount = passes.reduce((sum, value) => sum + value, 0);
    ctx.setReadout('routes', liveCount);
    ctx.setReadout('edges', `${formatCount(liveCount)} of ${formatCount(network.edgeCount)}`);
    ctx.setReadout('controlPoints', liveEdges * graph.pointsPerEdge);
    ctx.setCost({
      records: liveEdges * graph.pointsPerEdge,
      passes: 4 + 3 * ctx.options.iterations,
      note: 'control points; splat, advect and resample per iteration'
    });
    updateRadiusReadouts();
    refreshDisplay();
    updateCorridorLabels();
    markChanged();
  }

  /** The radius in km at the work box of the live edges, the whole-network box, and the schedule. */
  function updateRadiusReadouts(redrawChart = true): void {
    const {kernelRadius, decay, iterations} = ctx.options;
    const radiusKilometres = kernelRadius * boxKilometres;
    const formatKilometres = (kilometres: number) =>
      kilometres > 0 ? `${formatCount(Math.round(kilometres))} km` : null;
    ctx.setReadout('radiusKm', formatKilometres(radiusKilometres));
    ctx.setReadout('radiusKmWorld', formatKilometres(kernelRadius * worldBoxKilometres));
    ctx.setReadout(
      'radiusNowKm',
      formatKilometres(getRadiusAtIteration(radiusKilometres, decay, iterations))
    );
    ctx.setReadout('boxKm', formatKilometres(boxKilometres));
    // The iteration slider moves a marker on the chart by itself; only the schedule redraws it.
    if (!redrawChart) return;
    const iterationsAxis = Array.from({length: MAXIMUM_ITERATIONS + 1}, (_, index) => index);
    ctx.setChart(
      'decayChart',
      radiusKilometres > 0
        ? {
            kind: 'line',
            xLabel: 'iteration',
            yLabel: 'kernel radius (km)',
            xDomain: [0, MAXIMUM_ITERATIONS],
            series: [
              {
                label: 'radius',
                x: iterationsAxis,
                y: iterationsAxis.map(index => getRadiusAtIteration(radiusKilometres, decay, index))
              }
            ],
            formatY: value => formatCount(Math.round(value)),
            link: {option: 'iterations', label: value => `pass ${value}`},
            description:
              'Kernel radius in kilometres at each bundling iteration: it starts at the chosen fraction of the work box and shrinks by the decay factor every pass, so corridors form coarse first and tighten after. The marker is the active iteration.'
          }
        : null
    );
    updateFurniture();
  }

  /** The cartouche sample line, and a scale-bar tick at the kernel radius when a region is bundled. */
  function updateFurniture(): void {
    if (!graph) return;
    const {network} = graph.buffers;
    const sample =
      network.id === 'us'
        ? `${formatCount(graph.buffers.totalFlights)} scheduled flights, ${formatCount(network.edgeCount)} airport pairs`
        : `${formatCount(network.edgeCount)} route pairs, ${formatCount(network.nodeCount)} airports`;
    const radiusMetres = ctx.options.kernelRadius * boxKilometres * 1000;
    ctx.setFurniture(
      ctx.options.region !== 'all' && network.id === 'world' && radiusMetres > 0
        ? {title: {sample}, scaleBar: {units: 'metric', minZoom: 3, ticks: [radiusMetres]}}
        : {title: {sample}}
    );
  }

  /**
   * Writes the palette slot of every edge (distance class, delay class, probe), counts the live
   * pairs per class for the legend, and recomputes the US delay readouts.
   */
  function refreshDisplay(): void {
    if (!graph || !liveMask || !livePairs) return;
    const {buffers} = graph;
    const {network, expanded} = buffers;
    const {minTraffic, probes} = ctx.options;
    const attribute = getColorAttribute();
    const probeSlots = new Map<number, number>();
    if (probes) {
      buffers.probes.forEach((state, index) => {
        for (const row of state.route.expandedEdges) probeSlots.set(row, SLOT.firstProbe + index);
      });
    }
    const getPairSlot = (pair: number): number => {
      // When probes reuse slots 5-7, plain routes move to the otherwise-unused first class slot.
      if (attribute === 'plain') return probes ? SLOT.firstClass : SLOT.plain;
      if (attribute === 'delay') {
        if (network.traffic[pair] < minTraffic) return SLOT.underFloor;
        return getClassOf(network.delay![pair], DELAY_BREAKS);
      }
      return getClassOf(network.distanceKm[pair], DISTANCE_BREAKS);
    };
    const slots = new Float32Array(buffers.edgeCount);
    for (let row = 0; row < buffers.edgeCount; row++) {
      slots[row] = liveMask[row]
        ? (probeSlots.get(row) ?? getPairSlot(expanded.original[row]))
        : Number.NaN;
    }
    buffers.slots.write(slots);

    const counts = [0, 0, 0, 0, 0];
    let underFloor = 0;
    let shown = 0;
    let shownFlights = 0;
    let topClassFlights = 0;
    let worstPair = -1;
    for (let pair = 0; pair < network.edgeCount; pair++) {
      if (!livePairs[pair] || attribute === 'plain') continue;
      const slot = getPairSlot(pair);
      if (slot === SLOT.underFloor) {
        underFloor++;
        continue;
      }
      counts[slot]++;
      if (attribute === 'delay') {
        shown++;
        shownFlights += network.traffic[pair];
        if (slot === 4) topClassFlights += network.traffic[pair];
        if (worstPair < 0 || network.delay![pair] > network.delay![worstPair]) worstPair = pair;
      }
    }
    const stats: FlightClassStats = {counts, underFloor};
    ctx.setLegendData('classStats', stats);

    const annotations: MapAnnotation[] = [];
    if (attribute === 'delay' && worstPair >= 0) {
      const a = network.airports[network.source[worstPair]];
      const b = network.airports[network.target[worstPair]];
      const names = `${a.iata} to ${b.iata}`;
      const delay = Math.round(network.delay![worstPair]);
      ctx.setReadout('pairsShown', `${formatCount(shown)} of ${formatCount(shown + underFloor)}`);
      ctx.setReadout(
        'worstCorridor',
        `${names}: ${delay} min over ${formatCount(network.traffic[worstPair])} flights`
      );
      ctx.setReadout('topClassShare', shownFlights > 0 ? topClassFlights / shownFlights : null);
      annotations.push({
        kind: 'note',
        id: 'worst-corridor',
        coordinate: [
          (network.lonLat[network.source[worstPair] * 2] +
            network.lonLat[network.target[worstPair] * 2]) /
            2,
          (network.lonLat[network.source[worstPair] * 2 + 1] +
            network.lonLat[network.target[worstPair] * 2 + 1]) /
            2
        ],
        title: liveText('{delay:integer} min mean delay', {delay}),
        text: `${names}, the worst pair above the floor (${formatCount(network.traffic[worstPair])} flights)`
      });
    } else {
      ctx.setReadout('pairsShown', null);
      ctx.setReadout('worstCorridor', null);
      ctx.setReadout('topClassShare', null);
    }
    ctx.setAnnotations('worst-corridor', annotations.length ? annotations : null);
    ctx.requestLayers();
  }

  /** Names the corridors on the middle of the bundled path of a representative route. */
  function updateCorridorLabels(): void {
    if (!graph || !liveMask || !ctx.options.corridorLabels || !corridorPoints.length) {
      ctx.setAnnotations('corridors', null);
      return;
    }
    ctx.setAnnotations(
      'corridors',
      corridorPoints.map(point => ({
        kind: 'area' as const,
        id: `corridor-${point.id}`,
        coordinate: point.at,
        text: point.name,
        size: 'small' as const,
        priority: 2
      }))
    );
  }

  function buildGraph(): BundlingGraph {
    const buffers = prepare(ctx.options.network);
    const pointsPerEdge = Number(ctx.options.pointsPerEdge);
    const densityResolution = Number(ctx.options.densityResolution);
    const id = ++serial;
    const graphResources = new SpatialAnalysisResources(device, `flight-bundling-${id}`);
    const paths = graphResources.createBuffer('paths', buffers.edgeCount * pointsPerEdge * 8);
    const commandGraph = new GPUCommandGraph<void>(device, {id: `flight-bundling-${id}`});
    commandGraph.add(
      new GPUEdgeBundling({
        id: 'bundling',
        positions: importGraphBuffer(
          commandGraph,
          'positions',
          buffers.positions,
          'float32x2',
          buffers.vertexCount
        ),
        sourceVertices: importGraphBuffer(
          commandGraph,
          'sources',
          buffers.sources,
          'uint32',
          buffers.edgeCount
        ),
        targetVertices: importGraphBuffer(
          commandGraph,
          'targets',
          buffers.targets,
          'uint32',
          buffers.edgeCount
        ),
        edgeMask: importGraphBuffer(
          commandGraph,
          'mask',
          buffers.mask,
          'uint32',
          buffers.edgeCount
        ),
        geographic: true,
        pointsPerEdge,
        iterations: MAXIMUM_ITERATIONS,
        densityResolution,
        parameters: parameterBuffer.importToGraph(commandGraph),
        paths: importGraphBuffer(
          commandGraph,
          'paths',
          paths,
          'float32x2',
          buffers.edgeCount * pointsPerEdge
        )
      })
    );
    const compiled = graphResources.track(commandGraph.compile());
    const built: BundlingGraph = {
      resources: graphResources,
      compiled,
      buffers,
      pointsPerEdge,
      paths,
      reader: undefined as unknown as SummaryReader
    };
    built.reader = new SummaryReader(
      graphResources,
      `flight-bundling-${id}`,
      [{buffer: paths, size: buffers.edgeCount * pointsPerEdge * 8}],
      bytes => {
        if (!destroyed && graph === built) processPaths(built, new Float32Array(bytes));
      }
    );
    return built;
  }

  function rebuild(): void {
    if (graph) retired.push({resources: graph.resources, frames: 0});
    graph = buildGraph();
    corridorPoints = [];
    parameterBuffer.write(createGPUEdgeBundlingParameterValues(getParameters(), 'uint32'));
    writeMask();
    ctx.setReadout('airports', graph.buffers.network.nodeCount);
    ctx.setReadout('pointsPerEdge', graph.pointsPerEdge);
    markChanged();
  }

  /** Path stretch of the bundled polylines against the straight edges, and the corridor label points. */
  function processPaths(current: BundlingGraph, paths: Float32Array): void {
    const {buffers, pointsPerEdge} = current;
    const stretches: number[] = [];
    let straightKilometres = 0;
    let bundledKilometres = 0;
    for (let edge = 0; edge < buffers.edgeCount; edge++) {
      if (!getLive(edge)) continue;
      const first = edge * pointsPerEdge * 2;
      const last = first + (pointsPerEdge - 1) * 2;
      const direct = getApproximateKilometres(
        paths[first],
        paths[first + 1],
        paths[last],
        paths[last + 1]
      );
      straightKilometres += direct;
      let along = 0;
      for (let point = 0; point < pointsPerEdge - 1; point++) {
        const a = first + point * 2;
        along += getApproximateKilometres(paths[a], paths[a + 1], paths[a + 2], paths[a + 3]);
      }
      bundledKilometres += along;
      if (direct > 50) stretches.push(along / direct);
    }
    const meanStretch = straightKilometres > 0 ? bundledKilometres / straightKilometres : null;
    ctx.setChart(
      'stretchChart',
      stretches.length
        ? {
            kind: 'histogram',
            values: countInBins(stretches, 1, 1.6, 24),
            xDomain: [1, 1.6],
            xLabel: 'path length / straight length (the last bin holds everything longer)',
            yLabel: 'routes',
            formatX: value => `${value.toFixed(2)}x`,
            formatY: value => (value >= 1000 ? `${(value / 1000).toFixed(1)}k` : `${value}`),
            markers:
              meanStretch !== null && meanStretch <= 1.6
                ? [{x: meanStretch, label: 'mean'}]
                : undefined,
            description:
              'Histogram of how much longer each bundled route is than its straight line. Most routes stay close to 1; the tail is routes pulled into a distant bundle.'
          }
        : null
    );
    ctx.setReadout('stretch', meanStretch === null ? null : `${meanStretch.toFixed(2)}x`);

    corridorPoints = [];
    for (const corridor of buffers.corridors) {
      for (const row of corridor.expandedEdges) {
        if (!getLive(row)) continue;
        const middle = Math.floor((pointsPerEdge - 1) / 2);
        const a = (row * pointsPerEdge + middle) * 2;
        const b = a + 2;
        const longitude = (paths[a] + paths[b]) / 2;
        if (Math.abs(longitude) > 180) continue;
        corridorPoints.push({
          id: corridor.id,
          name: corridor.name,
          at: [longitude, (paths[a + 1] + paths[b + 1]) / 2]
        });
        break;
      }
    }
    updateCorridorLabels();
  }

  function getLive(edge: number): boolean {
    return liveMask ? liveMask[edge] === 1 : true;
  }

  function getAirportTooltip(pixel: readonly [number, number]): TooltipContent | null {
    const viewport = ctx.getViewport();
    if (!viewport || !graph) return null;
    const {network, degreeRank} = graph.buffers;
    let best = -1;
    let bestDistance = 12;
    for (let airport = 0; airport < network.nodeCount; airport++) {
      const [x, y] = viewport.project([
        network.lonLat[airport * 2],
        network.lonLat[airport * 2 + 1]
      ]);
      const distance = Math.hypot(x - pixel[0], y - pixel[1]);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = airport;
      }
    }
    if (best < 0) return null;
    const record = network.airports[best];
    return {
      title: `${record.iata}: ${record.name}`,
      subtitle: `${record.city}, ${record.country}`,
      rows: [
        {
          label: network.id === 'world' ? 'Airports served' : 'Airports served in July',
          value: formatCount(network.degree[best]),
          emphasis: true
        },
        {
          label: 'Rank by connections',
          value: `${formatCount(degreeRank[best] + 1)} of ${formatCount(network.nodeCount)}`
        }
      ]
    };
  }

  rebuild();

  return {
    getCompiledGraphs: () => (graph ? [graph.compiled as CompiledGPUCommandGraph<never>] : []),

    setOption(id) {
      switch (id) {
        case 'network':
        case 'pointsPerEdge':
        case 'densityResolution':
          rebuild();
          ctx.requestLayers();
          break;
        case 'region':
        case 'distanceRange':
          writeMask();
          break;
        case 'iterations':
          writeParameters(false);
          ctx.setCost({
            records: liveEdges * (graph?.pointsPerEdge ?? 16),
            passes: 4 + 3 * ctx.options.iterations,
            note: 'control points; splat, advect and resample per iteration'
          });
          // Zero iterations draws immutable two-point paths; positive iterations draw the
          // contributor output. Rebuild the layers so this change swaps those source buffers.
          ctx.requestLayers();
          break;
        case 'kernelRadius':
        case 'decay':
        case 'stiffness':
        case 'stepScale':
          writeParameters();
          break;
        case 'colorBy':
        case 'minTraffic':
        case 'probes':
          refreshDisplay();
          break;
        case 'corridorLabels':
          updateCorridorLabels();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onGroundChange() {
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      return getAirportTooltip(event.pixel);
    },

    encode(commandEncoder) {
      if (!graph) return;
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].resources.destroy();
          retired.splice(index, 1);
        }
      }
      // Paths persist in their buffer, so the graph only re-encodes after a change.
      if (encodeFrames > 0) {
        graph.compiled.encode(commandEncoder, {parameters: undefined});
        encodeFrames--;
      }
      if (
        statsStale &&
        encodeFrames === 0 &&
        performance.now() - lastChange > SETTLE_MILLISECONDS
      ) {
        if (!graph.reader.isPending) {
          statsStale = false;
          graph.reader.request(commandEncoder);
        }
      } else {
        graph.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      if (!graph) return [];
      const options = ctx.options;
      const {buffers, pointsPerEdge} = graph;
      const lngLat = COORDINATE_SYSTEM.LNGLAT;
      const attribute = getColorAttribute();
      // Distance and the single colour are light on the night ground: additive, order free. The
      // delay classes stack with alpha, one pass per class, so the worst pairs are on top.
      const additive = attribute !== 'delay';
      const probes = options.probes ? buffers.probes : [];
      const palette = getRoutePalette(attribute, options.brightness, probes.length > 0);
      const layers: Layer[] = [];
      // The opening step is the only place where straight routes are implicit. Once bundling is
      // active, draw only the contributor output unless the explicit straight-route comparison is
      // enabled below. Even a faint copy of every source edge reconstructs the hairball and makes
      // the bundled paths read as straight lines.
      if (options.iterations === 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `flight-source-routes-${buffers.network.id}`,
            coordinateSystem: lngLat,
            segments: buffers.straight,
            weights: buffers.maskWeights,
            instanceCount: buffers.edgeCount,
            widthPixels: 1,
            color: [104, 194, 255, 190],
            parameters: ADDITIVE_ROUTE_PARAMETERS
          })
        );
      }
      // GPUEdgeBundling's output buffer is populated by its iteration passes. At zero active
      // iterations there is deliberately no bundling work, so draw the immutable two-point source
      // paths instead of reading an unwritten output buffer. This is also the honest visual for the
      // opening hairball step: the original route pairs before the display transformation begins.
      const displayPaths = options.iterations === 0 ? buffers.straight : graph.paths;
      const displayPointsPerPath = options.iterations === 0 ? 2 : pointsPerEdge;

      const addRoutes = (
        name: string,
        paths: Buffer,
        pointsPerPath: number,
        compareSide?: 'a' | 'b'
      ) => {
        layers.push(
          new BundleRibbonLayer({
            id: `flight-${name}-${buffers.network.id}-${pointsPerPath}`,
            paths,
            pointsPerPath,
            pathCount: buffers.edgeCount,
            values: buffers.maskWeights,
            classes: buffers.slots,
            edgeMask: buffers.mask,
            palette,
            maximumValue: 1,
            widthMinPixels: 1,
            widthByValue: false,
            heaviestLast: false,
            compareSide,
            parameters: additive ? ADDITIVE_ROUTE_PARAMETERS : NORMAL_ROUTE_PARAMETERS
          })
        );
      };

      if (options.straight === 'ghost') {
        layers.push(
          new BundleRibbonLayer({
            id: `flight-ghost-${buffers.network.id}`,
            paths: buffers.straight,
            pointsPerPath: 2,
            pathCount: buffers.edgeCount,
            values: buffers.maskWeights,
            classes: buffers.slots,
            edgeMask: buffers.mask,
            palette,
            flatColor: UNDER_FLOOR_COLOR,
            maximumValue: 1,
            widthMinPixels: 0.6,
            widthByValue: false,
            heaviestLast: false,
            opacity: 0.2,
            parameters: NORMAL_ROUTE_PARAMETERS
          })
        );
      }
      if (options.straight === 'full') addRoutes('straight', buffers.straight, 2, 'a');
      addRoutes(
        'bundles',
        displayPaths,
        displayPointsPerPath,
        options.straight === 'full' ? 'b' : undefined
      );

      if (probes.length) {
        for (const state of probes) {
          const color = getProbeColor(state.probe);
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `flight-chord-${state.probe.id}`,
              coordinateSystem: lngLat,
              segments: state.segments,
              instanceCount: state.route.expandedEdges.length,
              widthPixels: 1.2,
              dashArray: [5, 4],
              cap: 'butt',
              color: [color[0], color[1], color[2], 235]
            })
          );
        }
      }

      if (options.showAirports) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `flight-airports-${buffers.network.id}`,
            coordinateSystem: lngLat,
            positions: buffers.airports,
            instanceCount: buffers.network.nodeCount,
            shape: 'circle',
            radiusPixels: 1.2,
            opacityStops: [
              [2.8, 0],
              [3.2, 1]
            ],
            color: [207, 216, 227, 140]
          }),
          new SpatialAnalysisPointLayer({
            id: `flight-hubs-${buffers.network.id}`,
            coordinateSystem: lngLat,
            positions: buffers.airports,
            ids: buffers.hubs,
            instanceCount: buffers.hubRows.length,
            shape: 'circle',
            radiusPixels: 3.4,
            color: HUB_FILL,
            outlineColor: HUB_STROKE,
            outlineWidthPixels: 1.4
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      graph?.reader.stop();
      graph?.resources.destroy();
      for (const entry of retired) entry.resources.destroy();
      resources.destroy();
    }
  };
}
