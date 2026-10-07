// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  createGPUEdgeBundlingParameterValues,
  GPUEdgeBundling,
  type GPUEdgeBundlingParameterValues
} from '@luma.gl/experimental/gpu-network';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {binValues, histogramChart} from '../movement/f-chart-helpers';
import {BundledPathLayer} from './b11-flow-layers';
import {
  expandAntimeridianEdges,
  getApproximateKilometres,
  readUsNetwork,
  readWorldNetwork,
  type ExpandedEdges,
  type FlightNetwork
} from './b11-flight-data';
import {CONTINENT_NAMES, loadAirportTable} from './b11-geography';

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
  minTraffic: number;
  iterations: number;
  play: boolean;
  playSpeed: number;
  loop: boolean;
  kernelRadius: number;
  decay: number;
  stiffness: number;
  stepScale: number;
  pointsPerEdge: '8' | '16' | '24' | '32';
  densityResolution: '128' | '256' | '512';
  colorBy: 'plain' | 'distance' | 'traffic' | 'airlines' | 'delay' | 'cancel';
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  opacity: number;
  showStraight: boolean;
  showAirports: boolean;
};

/** Compile-time iteration capacity; the slider picks how many run. */
export const MAXIMUM_ITERATIONS = 32;
const HUB_COUNT = 24;
const COVERAGE_CELL_DEGREES = 0.5;
const RETIRE_FRAMES = 4;
const SETTLE_MILLISECONDS = 350;

type NetworkBuffers = {
  network: FlightNetwork;
  expanded: ExpandedEdges;
  vertexCount: number;
  edgeCount: number;
  positions: Buffer;
  sources: Buffer;
  targets: Buffer;
  values: Buffer;
  mask: Buffer;
  maskWeights: Buffer;
  straight: Buffer;
  airports: Buffer;
  hubs: Buffer;
  /** Colour range of every attribute: 5th and 95th percentile. */
  ranges: Record<string, [number, number]>;
  hubRows: Uint32Array;
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
    const hubRows = Uint32Array.from(order.slice(0, HUB_COUNT));
    const prepared: NetworkBuffers = {
      network,
      expanded,
      vertexCount: expanded.positions.length / 2,
      edgeCount,
      positions: resources.createBuffer(`${id}-positions`, expanded.positions),
      sources: resources.createBuffer(`${id}-sources`, expanded.source),
      targets: resources.createBuffer(`${id}-targets`, expanded.target),
      values: resources.createBuffer(`${id}-values`, edgeCount * 4),
      mask: resources.createBuffer(`${id}-mask`, new Uint32Array(edgeCount).fill(1)),
      maskWeights: resources.createBuffer(
        `${id}-mask-weights`,
        new Float32Array(edgeCount).fill(1)
      ),
      straight: resources.createBuffer(`${id}-straight`, segments),
      airports: resources.createBuffer(`${id}-airports`, network.lonLat),
      hubs: resources.createBuffer(`${id}-hubs`, hubRows),
      ranges: {
        distance: [0, percentile(network.distanceKm, 0.95)],
        traffic: [1, percentile(network.traffic, 0.95)],
        airlines: [1, network.airlines ? percentile(network.airlines, 0.95) : 1],
        delay: [0, 45],
        cancel: [0, 0.08]
      },
      hubRows
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

  function writeParameters(): void {
    parameterBuffer.write(createGPUEdgeBundlingParameterValues(getParameters(), 'uint32'));
    markChanged();
  }

  // Plays the bundling: the iteration count sweeps from straight lines to the full run.
  const clock = createPlaybackClock(
    ctx,
    {time: 'iterations', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [0, MAXIMUM_ITERATIONS], rate: 1, step: 1, notify: true}
  );

  function markChanged(): void {
    encodeFrames = Math.max(encodeFrames, 2);
    statsStale = true;
    lastChange = performance.now();
  }

  /** Whether an original edge passes the region, distance and traffic filters. */
  function writeMask(): void {
    if (!graph) return;
    const {buffers} = graph;
    const {network, expanded} = buffers;
    const {region, distanceRange, minTraffic} = ctx.options;
    const passes = new Uint8Array(network.edgeCount);
    for (let edge = 0; edge < network.edgeCount; edge++) {
      let inside = true;
      if (region !== 'all') {
        for (const airport of [network.source[edge], network.target[edge]]) {
          if (region === 'lower48') {
            const longitude = network.lonLat[airport * 2];
            const latitude = network.lonLat[airport * 2 + 1];
            inside &&= longitude > -125 && longitude < -66 && latitude > 24 && latitude < 50;
          } else {
            inside &&= CONTINENT_NAMES[network.continent[airport]] === region;
          }
        }
      }
      const distance = network.distanceKm[edge];
      if (distance < distanceRange[0] || distance > distanceRange[1]) inside = false;
      if (network.traffic[edge] < minTraffic) inside = false;
      passes[edge] = inside ? 1 : 0;
    }
    const mask = new Uint32Array(buffers.edgeCount);
    const weights = new Float32Array(buffers.edgeCount);
    liveEdges = 0;
    for (let edge = 0; edge < buffers.edgeCount; edge++) {
      const live = passes[expanded.original[edge]];
      mask[edge] = live;
      weights[edge] = live;
      liveEdges += live;
    }
    buffers.mask.write(mask);
    buffers.maskWeights.write(weights);
    liveMask = mask;
    ctx.setReadout(
      'edges',
      `${formatCount(passes.reduce((sum, value) => sum + value, 0))} of ${formatCount(network.edgeCount)}`
    );
    ctx.setReadout('controlPoints', liveEdges * (graph?.pointsPerEdge ?? 16));
    const liveDistances = new Float32Array(network.edgeCount).fill(Number.NaN);
    for (let edge = 0; edge < network.edgeCount; edge++) {
      if (passes[edge]) liveDistances[edge] = network.distanceKm[edge];
    }
    const maximumKm = ctx.options.network === 'us' ? 5000 : 15000;
    ctx.setChart(
      'distanceChart',
      histogramChart(binValues(liveDistances, 0, maximumKm, 25), 0, maximumKm, {
        xLabel: 'route length (km)',
        yLabel: 'routes',
        formatX: value => `${Math.round(value)}`,
        formatY: value => (value >= 1000 ? `${(value / 1000).toFixed(1)}k` : `${value}`),
        description:
          'Histogram of the length of the routes that pass the filters. Many short regional routes and a long tail of intercontinental ones.'
      })
    );
    markChanged();
  }

  /** The attribute actually colouring the edges (falls back when the network lacks it). */
  function getColorAttribute(): Exclude<FlightBundlingOptions['colorBy'], 'plain'> | null {
    const {colorBy} = ctx.options;
    if (colorBy === 'plain') return null;
    const {network} = graph!.buffers;
    if (colorBy === 'airlines' && !network.airlines) return 'distance';
    if ((colorBy === 'delay' || colorBy === 'cancel') && !network.delay) return 'distance';
    return colorBy;
  }

  function writeValues(): void {
    if (!graph) return;
    const {buffers} = graph;
    const attribute = getColorAttribute();
    const {network, expanded} = buffers;
    if (attribute) {
      const source =
        attribute === 'distance'
          ? network.distanceKm
          : attribute === 'traffic'
            ? network.traffic
            : attribute === 'airlines'
              ? network.airlines!
              : attribute === 'delay'
                ? network.delay!
                : network.cancelRate!;
      const values = new Float32Array(buffers.edgeCount);
      for (let edge = 0; edge < buffers.edgeCount; edge++)
        values[edge] = source[expanded.original[edge]];
      buffers.values.write(values);
      ctx.setLegendExtent('edge-value', buffers.ranges[attribute]);
    }
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
    parameterBuffer.write(createGPUEdgeBundlingParameterValues(getParameters(), 'uint32'));
    writeMask();
    writeValues();
    ctx.setReadout('airports', graph.buffers.network.nodeCount);
    ctx.setReadout('pointsPerEdge', graph.pointsPerEdge);
    markChanged();
  }

  /** Path stretch and map coverage of the bundled polylines against the straight edges. */
  function processPaths(current: BundlingGraph, paths: Float32Array): void {
    const {buffers, pointsPerEdge} = current;
    const stretches: number[] = [];
    let straightKilometres = 0;
    let bundledKilometres = 0;
    const straightCells = new Set<number>();
    const bundledCells = new Set<number>();
    const mark = (cells: Set<number>, ax: number, ay: number, bx: number, by: number) => {
      const steps = Math.max(
        1,
        Math.ceil(Math.max(Math.abs(bx - ax), Math.abs(by - ay)) / (COVERAGE_CELL_DEGREES * 0.5))
      );
      for (let step = 0; step <= steps; step++) {
        const t = step / steps;
        const x = Math.floor((ax + (bx - ax) * t) / COVERAGE_CELL_DEGREES);
        const y = Math.floor((ay + (by - ay) * t) / COVERAGE_CELL_DEGREES);
        cells.add((x + 2000) * 4000 + (y + 2000));
      }
    };
    for (let edge = 0; edge < buffers.edgeCount; edge++) {
      if (!getLive(edge)) continue;
      const first = edge * pointsPerEdge * 2;
      const last = first + (pointsPerEdge - 1) * 2;
      straightKilometres += getApproximateKilometres(
        paths[first],
        paths[first + 1],
        paths[last],
        paths[last + 1]
      );
      mark(straightCells, paths[first], paths[first + 1], paths[last], paths[last + 1]);
      const bundledBefore = bundledKilometres;
      for (let point = 0; point < pointsPerEdge - 1; point++) {
        const a = first + point * 2;
        bundledKilometres += getApproximateKilometres(
          paths[a],
          paths[a + 1],
          paths[a + 2],
          paths[a + 3]
        );
        mark(bundledCells, paths[a], paths[a + 1], paths[a + 2], paths[a + 3]);
      }
      const direct = getApproximateKilometres(
        paths[first],
        paths[first + 1],
        paths[last],
        paths[last + 1]
      );
      if (direct > 50) stretches.push((bundledKilometres - bundledBefore) / direct);
    }
    ctx.setChart(
      'stretchChart',
      stretches.length
        ? histogramChart(binValues(stretches, 1, 1.6, 24), 1, 1.6, {
            xLabel: 'path length / straight length (60% and over in the last bin)',
            yLabel: 'routes',
            color: 3,
            formatX: value => value.toFixed(2),
            formatY: value => (value >= 1000 ? `${(value / 1000).toFixed(1)}k` : `${value}`),
            description:
              'Histogram of how much longer each bundled route is than its straight line. Most routes stay within a few percent; the tail is routes pulled into a distant bundle.'
          })
        : null
    );
    ctx.setReadout(
      'stretch',
      straightKilometres > 0 ? bundledKilometres / straightKilometres : null
    );
    ctx.setReadout('coverageStraight', straightCells.size);
    ctx.setReadout('coverageBundled', bundledCells.size);
    ctx.setReadout(
      'inkSaved',
      straightCells.size > 0 ? 1 - bundledCells.size / straightCells.size : null
    );
  }

  function getLive(edge: number): boolean {
    return liveMask ? liveMask[edge] === 1 : true;
  }

  function findAirportNear(pixel: readonly [number, number]): string | null {
    const viewport = ctx.getViewport();
    if (!viewport || !graph) return null;
    const {network} = graph.buffers;
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
    const unit = network.id === 'world' ? 'airports served' : 'airports served in July 2023';
    return `${record.iata}: ${record.name}\n${record.city}, ${record.country}\n${network.degree[best]} ${unit}`;
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
        case 'minTraffic':
          writeMask();
          break;
        case 'play':
        case 'playSpeed':
        case 'loop':
          break;
        case 'iterations':
        case 'kernelRadius':
        case 'decay':
        case 'stiffness':
        case 'stepScale':
          writeParameters();
          break;
        case 'colorBy':
          writeValues();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      return findAirportNear(event.pixel);
    },

    encode(commandEncoder, frame) {
      if (!graph) return;
      clock.advance(frame);
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
      const dark = ctx.theme() === 'dark';
      const lngLat = COORDINATE_SYSTEM.LNGLAT;
      const attribute = getColorAttribute();
      const range = attribute ? buffers.ranges[attribute] : [0, 1];
      const layers: Layer[] = [];
      if (options.showStraight) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `flight-straight-${buffers.network.id}`,
            coordinateSystem: lngLat,
            segments: buffers.straight,
            weights: buffers.maskWeights,
            instanceCount: buffers.edgeCount,
            widthPixels: 0.8,
            color: dark ? [180, 195, 230, 40] : [60, 70, 100, 45]
          })
        );
      }
      layers.push(
        new BundledPathLayer({
          id: `flight-bundles-${buffers.network.id}-${pointsPerEdge}`,
          paths: graph.paths,
          pointsPerPath: pointsPerEdge,
          pathCount: buffers.edgeCount,
          values: attribute ? buffers.values : null,
          valueRange: range as [number, number],
          ramp: options.ramp,
          sqrtScale: attribute === 'traffic',
          edgeMask: buffers.mask,
          startColor: dark ? [96, 214, 255, 255] : [20, 110, 190, 255],
          endColor: dark ? [96, 214, 255, 255] : [20, 110, 190, 255],
          opacity: options.opacity
        })
      );
      if (options.showAirports) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `flight-airports-${buffers.network.id}`,
            coordinateSystem: lngLat,
            positions: buffers.airports,
            instanceCount: buffers.network.nodeCount,
            radiusPixels: 1.4,
            color: dark ? [235, 240, 255, 150] : [30, 40, 70, 150]
          }),
          new SpatialAnalysisPointLayer({
            id: `flight-hubs-${buffers.network.id}`,
            coordinateSystem: lngLat,
            positions: buffers.airports,
            ids: buffers.hubs,
            instanceCount: buffers.hubRows.length,
            radiusPixels: 3.6,
            color: [255, 184, 64, 235]
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

function percentile(values: ArrayLike<number>, fraction: number): number {
  const sorted = Array.from(values).sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))] ?? 1;
}
