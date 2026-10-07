// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGraph,
  GPUGraphLabelPropagation,
  GPUGraphModularity,
  GPUGraphModularityOptimization,
  GPUGraphTopology
} from '@luma.gl/gpgpu/gpu-graph';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {BixiGraphVectors, buildUndirectedEdges, normalizedMutualInformation} from './bixi-graph';
import {findStationNearPixel, formatCompact, readBixiFlows} from './bixi-data';

/** Option state of the bixi-communities scene. */
export type BixiCommunitiesOptions = {
  partition: 'optimized' | 'propagation' | 'boroughs';
  weighting: 'rides' | 'sqrt' | 'equal';
  minRides: number;
  neighbors: number;
  propagationRounds: number;
  optimizeRounds: number;
  resolution: number;
  minimumGain: '0' | '0.0001' | '0.001';
  edges: number;
  edgeWidth: number;
  edgeOpacity: number;
  showBetween: boolean;
  stationSize: number;
};

const NO_LABEL = 0xffffffff;
const RETIRE_FRAMES = 4;
/** Palette slots: communities by size, then "all other", then "between communities". */
const TOP_COMMUNITIES = 6;
const OTHER_SLOT = 6;
const BETWEEN_SLOT = 7;
export const COMMUNITY_PALETTE = [
  [86, 180, 233, 255],
  [240, 160, 20, 255],
  [214, 110, 170, 255],
  [30, 175, 125, 255],
  [225, 205, 50, 255],
  [225, 95, 40, 255],
  [150, 150, 190, 255],
  [150, 155, 170, 70]
] as const;

type Analysis = {
  resources: SpatialAnalysisResources;
  vectors: BixiGraphVectors;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  sources: Buffer;
  targets: Buffer;
  weights: Buffer;
};

type Summary = {
  propagation: Uint32Array;
  optimized: Uint32Array;
  propagationConverged: boolean;
  optimizedConverged: boolean;
  valid: boolean;
  overflow: boolean;
  propagationScore: number;
  optimizedScore: number;
  boroughScore: number;
};

/**
 * Communities of riding on the BIXI station graph. Stations are vertices, the rides between two
 * stations (both directions summed) are weighted undirected edges. One compiled graph rebuilds the
 * CSR, runs `GPUGraphLabelPropagation`, refines its partition with
 * `GPUGraphModularityOptimization`, and scores the propagation, the optimized partition and the
 * borough partition with `GPUGraphModularity` at the same resolution. Edge weights and a minimum
 * ride filter are buffer writes; rounds, resolution and minimum gain are compile-time constants.
 */
export async function createBixiCommunities(
  ctx: SceneContext<BixiCommunitiesOptions>
): Promise<SceneInstance<BixiCommunitiesOptions>> {
  const {device} = ctx;
  const flows = readBixiFlows(ctx.datasets.get('bixi-flows'));
  const stationCount = flows.stationCount;
  const edges = buildUndirectedEdges(flows);
  const resources = new SpatialAnalysisResources(device, 'bixi-communities');
  const coordinateSystem = COORDINATE_SYSTEM.LNGLAT;

  const segments = resources.createBuffer('segments', edges.segments);
  const edgeAlphaValues = new Float32Array(edges.count);
  for (let edge = 0; edge < edges.count; edge++) {
    edgeAlphaValues[edge] = 0.2 + 0.8 * Math.sqrt(edges.rides[edge] / edges.rides[0]);
  }
  const edgeAlpha = resources.createBuffer('edge-alpha', edgeAlphaValues);
  const stations = resources.createBuffer('stations', flows.lngLat);
  const stationColors = resources.createBuffer('station-colors', stationCount * 4);
  const edgeColors = resources.createBuffer('edge-colors', edges.count * 4);
  const edgeOrder = resources.createBuffer('edge-order', edges.count * 4);
  const selectedStation = resources.createBuffer('selected-station', Uint32Array.of(0));

  /** Borough index per station, as the partition the communities are compared with. */
  const boroughLabels = Uint32Array.from(flows.borough);
  /** CSR over the undirected edges, for per-station shares and the click readout. */
  const offsets = new Uint32Array(stationCount + 1);
  for (let edge = 0; edge < edges.count; edge++) {
    offsets[edges.a[edge] + 1]++;
    offsets[edges.b[edge] + 1]++;
  }
  for (let station = 0; station < stationCount; station++) offsets[station + 1] += offsets[station];
  const neighbors = new Uint32Array(offsets[stationCount]);
  const neighborRides = new Float32Array(offsets[stationCount]);
  {
    const cursor = offsets.slice(0, stationCount);
    for (let edge = 0; edge < edges.count; edge++) {
      const a = edges.a[edge];
      const b = edges.b[edge];
      neighbors[cursor[a]] = b;
      neighborRides[cursor[a]++] = edges.rides[edge];
      neighbors[cursor[b]] = a;
      neighborRides[cursor[b]++] = edges.rides[edge];
    }
  }

  /** Rank of each edge among the edges of its two stations (0 = the station's strongest link). */
  const rankAtA = new Uint16Array(edges.count);
  const rankAtB = new Uint16Array(edges.count);
  {
    const seen = new Uint16Array(stationCount);
    // Edges are sorted by rides, so the running count of a station is the rank of its next edge.
    for (let edge = 0; edge < edges.count; edge++) {
      rankAtA[edge] = seen[edges.a[edge]]++;
      rankAtB[edge] = seen[edges.b[edge]]++;
    }
  }

  let analysis: Analysis | null = null;
  let serial = 0;
  let destroyed = false;
  let dirty = true;
  let summary: Summary | null = null;
  let labels: Uint32Array = new Uint32Array(stationCount);
  let rankOfLabel = new Map<number, number>();
  let sizeOfLabel = new Map<number, number>();
  let selected = -1;
  let shownEdges = 0;
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];

  /** An edge stays when it is strong enough and among the strongest links of either end. */
  function getActiveEdges(): {mask: Uint8Array; count: number} {
    const {minRides, neighbors: keep} = ctx.options;
    const mask = new Uint8Array(edges.count);
    let count = 0;
    for (let edge = 0; edge < edges.count; edge++) {
      if (edges.rides[edge] >= minRides && Math.min(rankAtA[edge], rankAtB[edge]) < keep) {
        mask[edge] = 1;
        count++;
      }
    }
    return {mask, count};
  }

  function writeEdgeInputs(current: Analysis): void {
    const {weighting} = ctx.options;
    const {mask} = getActiveEdges();
    const weights = new Float32Array(edges.count);
    const sources = new Uint32Array(edges.count);
    for (let edge = 0; edge < edges.count; edge++) {
      weights[edge] =
        weighting === 'rides'
          ? edges.rides[edge]
          : weighting === 'sqrt'
            ? Math.sqrt(edges.rides[edge])
            : 1;
      // An out-of-domain source endpoint makes the graph contributors ignore the edge.
      sources[edge] = mask[edge] ? edges.a[edge] : NO_LABEL;
    }
    current.weights.write(weights);
    current.sources.write(sources);
  }

  function buildAnalysis(): Analysis {
    const id = ++serial;
    const graphResources = new SpatialAnalysisResources(device, `bixi-communities-${id}`);
    const vectors = graphResources.track(new BixiGraphVectors(device, `bixi-communities-${id}`));
    const sourceVector = vectors.edgeColumn('sources', 'uint32', edges.a);
    const targetVector = vectors.edgeColumn('targets', 'uint32', edges.b);
    const weightVector = vectors.edgeColumn('weights', 'float32', edges.rides);
    const graph = new GPUGraph({
      vertexCount: stationCount,
      sourceVertices: sourceVector,
      targetVertices: targetVector,
      edgeWeights: weightVector,
      directed: false
    });
    const forward = vectors.adjacency('forward', stationCount, edges.count * 2);
    const topology = new GPUGraphTopology({
      id: 'bixi-topology',
      graph,
      forward,
      invalidEdgeCount: vectors.scalar('invalid-edges', 'uint32', 1)
    });
    const propagation = vectors.scalar('propagation', 'uint32', stationCount);
    const propagationConverged = vectors.scalar('propagation-converged', 'uint32', 1);
    const optimized = vectors.scalar('optimized', 'uint32', stationCount);
    const optimizedScore = vectors.scalar('optimized-score', 'float32', 1);
    const optimizedConverged = vectors.scalar('optimized-converged', 'uint32', 1);
    const optimizedValid = vectors.scalar('optimized-valid', 'uint32', 1);
    const propagationScore = vectors.scalar('propagation-score', 'float32', 1);
    const boroughScore = vectors.scalar('borough-score', 'float32', 1);
    const boroughs = vectors.scalar('boroughs', 'uint32', stationCount, boroughLabels);
    const {propagationRounds, optimizeRounds, resolution, minimumGain} = ctx.options;
    const commandGraph = new GPUCommandGraph<void>(device, {id: `bixi-communities-${id}`});
    topology.addToGraph(commandGraph);
    new GPUGraphLabelPropagation({
      id: 'bixi-propagation',
      topology,
      output: propagation,
      iterations: propagationRounds,
      converged: propagationConverged
    }).addToGraph(commandGraph);
    new GPUGraphModularityOptimization({
      id: 'bixi-optimization',
      topology,
      output: optimized,
      modularity: optimizedScore,
      initialCommunities: propagation,
      resolution,
      iterations: optimizeRounds,
      minimumGain: Number(minimumGain),
      converged: optimizedConverged,
      valid: optimizedValid
    }).addToGraph(commandGraph);
    new GPUGraphModularity({
      id: 'bixi-score-propagation',
      graph,
      communities: propagation,
      output: propagationScore,
      resolution
    }).addToGraph(commandGraph);
    new GPUGraphModularity({
      id: 'bixi-score-boroughs',
      graph,
      communities: boroughs,
      output: boroughScore,
      resolution
    }).addToGraph(commandGraph);
    const compiled = graphResources.track(commandGraph.compile());
    const column = (vector: typeof propagation) => ({
      buffer: vectors.getBuffer(vector),
      size: stationCount * 4
    });
    const scalar = (vector: Parameters<typeof vectors.getBuffer>[0]) => ({
      buffer: vectors.getBuffer(vector),
      size: 4
    });
    const reader = new SummaryReader(
      graphResources,
      `bixi-communities-${id}`,
      [
        column(propagation),
        column(optimized),
        scalar(propagationConverged),
        scalar(optimizedConverged),
        scalar(optimizedValid),
        scalar(forward.overflow),
        scalar(propagationScore),
        scalar(optimizedScore),
        scalar(boroughScore)
      ],
      bytes => {
        if (destroyed || analysis?.reader !== reader) return;
        const words = new Uint32Array(bytes);
        const floats = new Float32Array(bytes);
        const base = stationCount * 2;
        processSummary({
          propagation: words.slice(0, stationCount),
          optimized: words.slice(stationCount, base),
          propagationConverged: words[base] === 1,
          optimizedConverged: words[base + 1] === 1,
          valid: words[base + 2] === 1,
          overflow: words[base + 3] !== 0,
          propagationScore: floats[base + 4],
          optimizedScore: floats[base + 5],
          boroughScore: floats[base + 6]
        });
      }
    );
    return {
      resources: graphResources,
      vectors,
      compiled,
      reader,
      sources: vectors.getBuffer(sourceVector),
      targets: vectors.getBuffer(targetVector),
      weights: vectors.getBuffer(weightVector)
    };
  }

  function rebuild(): void {
    if (analysis) retired.push({resources: analysis.resources, frames: 0});
    analysis = buildAnalysis();
    writeEdgeInputs(analysis);
    dirty = true;
  }

  function getPartition(current: Summary): Uint32Array {
    switch (ctx.options.partition) {
      case 'propagation':
        return current.propagation;
      case 'boroughs':
        return boroughLabels;
      default:
        return current.optimized;
    }
  }

  function getLabelName(label: number): string {
    // Name a community after its two most common boroughs.
    const counts = new Map<number, number>();
    for (let station = 0; station < stationCount; station++) {
      if (labels[station] === label) {
        counts.set(flows.borough[station], (counts.get(flows.borough[station]) ?? 0) + 1);
      }
    }
    const ordered = Array.from(counts.entries()).sort((x, y) => y[1] - x[1]);
    const first = flows.boroughNames[ordered[0][0]].replace(/^Le /, '');
    return ordered.length > 1 && ordered[1][1] >= 0.2 * (sizeOfLabel.get(label) ?? 1)
      ? `${first} + ${flows.boroughNames[ordered[1][0]].replace(/^Le /, '')}`
      : first;
  }

  /** Recomputes everything that depends on the shown partition (colors, order, readouts, charts). */
  function refresh(): void {
    if (!summary) return;
    labels = getPartition(summary);
    sizeOfLabel = new Map();
    for (let station = 0; station < stationCount; station++) {
      sizeOfLabel.set(labels[station], (sizeOfLabel.get(labels[station]) ?? 0) + 1);
    }
    const ranked = Array.from(sizeOfLabel.entries()).sort((x, y) => y[1] - x[1] || x[0] - y[0]);
    rankOfLabel = new Map(ranked.map(([label], rank) => [label, rank]));
    const slotOf = (label: number) => Math.min(rankOfLabel.get(label)!, OTHER_SLOT);
    const colors = new Uint32Array(stationCount);
    for (let station = 0; station < stationCount; station++)
      colors[station] = slotOf(labels[station]);
    stationColors.write(colors);

    const {mask, count: active} = getActiveEdges();
    const withinColors = new Uint32Array(edges.count);
    const between: number[] = [];
    const within: number[] = [];
    let withinRides = 0;
    let totalRides = 0;
    let boroughWithinRides = 0;
    for (let edge = 0; edge < edges.count; edge++) {
      if (!mask[edge]) continue;
      const a = edges.a[edge];
      const b = edges.b[edge];
      totalRides += edges.rides[edge];
      if (boroughLabels[a] === boroughLabels[b]) boroughWithinRides += edges.rides[edge];
      if (labels[a] === labels[b]) {
        withinColors[edge] = slotOf(labels[a]);
        withinRides += edges.rides[edge];
        within.push(edge);
      } else {
        withinColors[edge] = BETWEEN_SLOT;
        between.push(edge);
      }
    }
    edgeColors.write(withinColors);
    // Faint between-community edges first, then within-community edges, lightest first.
    const drawn = Math.min(ctx.options.edges, active);
    const order: number[] = [];
    if (ctx.options.showBetween) {
      for (let index = Math.min(between.length, drawn) - 1; index >= 0; index--) {
        order.push(between[index]);
      }
    }
    for (let index = Math.min(within.length, drawn) - 1; index >= 0; index--) {
      order.push(within[index]);
    }
    shownEdges = order.length;
    edgeOrder.write(Uint32Array.from(order));
    ctx.requestLayers();

    // Legend: the largest communities, named after their boroughs.
    const entries = ranked.slice(0, TOP_COMMUNITIES).map(([label, size], rank) => ({
      color: COMMUNITY_PALETTE[rank],
      label: `${getLabelName(label)} (${size})`
    }));
    if (ranked.length > TOP_COMMUNITIES) {
      const rest = ranked.slice(TOP_COMMUNITIES).reduce((sum, [, size]) => sum + size, 0);
      entries.push({
        color: COMMUNITY_PALETTE[OTHER_SLOT],
        label: `${ranked.length - TOP_COMMUNITIES} smaller groups (${rest})`
      });
    }
    ctx.setLegendData('communities', entries);

    const modularity = (score: number) => (Number.isFinite(score) ? score : null);
    ctx.setReadout('stations', stationCount);
    ctx.setReadout(
      'edgesKept',
      `${active.toLocaleString('en-US')} of ${edges.count.toLocaleString('en-US')}`
    );
    ctx.setReadout('ridesKept', totalRides / edges.totalRides);
    ctx.setReadout('communityCount', ranked.length);
    ctx.setReadout(
      'largest',
      `${ranked[0][1]} stations (${Math.round((100 * ranked[0][1]) / stationCount)}%)`
    );
    ctx.setReadout('modularityPropagation', modularity(summary.propagationScore));
    ctx.setReadout('modularityOptimized', modularity(summary.optimizedScore));
    ctx.setReadout('modularityBoroughs', modularity(summary.boroughScore));
    ctx.setReadout(
      'convergence',
      `propagation ${summary.propagationConverged ? 'converged' : 'not converged'}, refinement ${summary.optimizedConverged ? 'converged' : 'round budget used'}`
    );
    ctx.setReadout(
      'validity',
      summary.overflow
        ? 'adjacency overflow: labels invalid'
        : summary.valid
          ? 'valid'
          : 'invalid (no edges?)'
    );
    ctx.setReadout('withinShare', totalRides > 0 ? withinRides / totalRides : null);
    ctx.setReadout('withinBoroughShare', totalRides > 0 ? boroughWithinRides / totalRides : null);
    ctx.setReadout('agreement', normalizedMutualInformation(labels, boroughLabels));
    ctx.setReadout('drawn', shownEdges);

    // Community sizes.
    const top = ranked.slice(0, 12);
    ctx.setChart('sizesChart', {
      kind: 'bars',
      values: top.map(([, size]) => size),
      labels: top.map((_, rank) => `${rank + 1}`),
      highlight: selected >= 0 ? [Math.min(11, rankOfLabel.get(labels[selected]) ?? 0)] : [0],
      height: 110,
      yLabel: 'stations',
      xLabel: 'community, largest first',
      description: 'Stations in each of the twelve largest communities of the shown partition.'
    });
    // Quality of the three partitions at the current resolution.
    ctx.setChart('qualityChart', {
      kind: 'bars',
      values: [summary.boroughScore, summary.propagationScore, summary.optimizedScore].map(value =>
        Number.isFinite(value) ? value : 0
      ),
      labels: ['boroughs', 'propagation', 'optimized'],
      highlight: [
        ctx.options.partition === 'boroughs' ? 0 : ctx.options.partition === 'propagation' ? 1 : 2
      ],
      height: 110,
      yLabel: 'modularity',
      formatY: value => value.toFixed(2),
      description:
        'Weighted modularity of the borough partition, label propagation and the refined partition. Higher keeps more rides inside groups than chance predicts.'
    });
    describeSelection();
  }

  function describeSelection(): void {
    if (selected < 0 || !summary) {
      ctx.setReadout('selected', 'click a station');
      return;
    }
    const label = labels[selected];
    let inside = 0;
    let total = 0;
    for (let slot = offsets[selected]; slot < offsets[selected + 1]; slot++) {
      total += neighborRides[slot];
      if (labels[neighbors[slot]] === label) inside += neighborRides[slot];
    }
    ctx.setReadout(
      'selected',
      `${flows.names[selected]}\n${flows.boroughNames[flows.borough[selected]]}\ncommunity ${(rankOfLabel.get(label) ?? 0) + 1} (${sizeOfLabel.get(label)} stations)\n${total > 0 ? Math.round((100 * inside) / total) : 0}% of its rides stay inside`
    );
  }

  function processSummary(next: Summary): void {
    summary = next;
    refresh();
  }

  rebuild();

  return {
    getCompiledGraphs: () =>
      analysis ? [analysis.compiled as CompiledGPUCommandGraph<never>] : [],

    setOption(id) {
      switch (id) {
        case 'propagationRounds':
        case 'optimizeRounds':
        case 'resolution':
        case 'minimumGain':
          rebuild();
          break;
        case 'weighting':
        case 'minRides':
        case 'neighbors':
          if (analysis) writeEdgeInputs(analysis);
          dirty = true;
          break;
        case 'partition':
        case 'edges':
        case 'showBetween':
          refresh();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      const station = findStationNearPixel(
        ctx.getViewport(),
        flows.lngLat,
        stationCount,
        event.pixel
      );
      if (station < 0) return null;
      const community = summary
        ? `community ${(rankOfLabel.get(labels[station]) ?? 0) + 1}`
        : 'computing';
      return `${flows.names[station]}\n${flows.boroughNames[flows.borough[station]]}, ${community}\n${formatCompact(flows.departures[station])} departures, ${formatCompact(flows.arrivals[station])} arrivals`;
    },

    onClick(event) {
      const station = findStationNearPixel(
        ctx.getViewport(),
        flows.lngLat,
        stationCount,
        event.pixel,
        14
      );
      selected = station === selected ? -1 : station;
      if (selected >= 0) selectedStation.write(Uint32Array.of(selected));
      refresh();
      return station >= 0;
    },

    encode(commandEncoder) {
      if (!analysis) return;
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].resources.destroy();
          retired.splice(index, 1);
        }
      }
      if (dirty && !analysis.reader.isPending) {
        analysis.compiled.encode(commandEncoder, {parameters: undefined});
        analysis.reader.request(commandEncoder);
        dirty = false;
      } else {
        analysis.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      if (!summary) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'bixi-community-edges',
          coordinateSystem,
          segments,
          ids: edgeOrder,
          instanceCount: shownEdges,
          values: edgeColors,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: COMMUNITY_PALETTE,
          weights: edgeAlpha,
          widthPixels: options.edgeWidth,
          opacity: options.edgeOpacity
        }),
        new SpatialAnalysisPointLayer({
          id: 'bixi-community-stations-halo',
          coordinateSystem,
          positions: stations,
          instanceCount: stationCount,
          radiusPixels: options.stationSize + 1.6,
          color: dark ? [10, 12, 20, 235] : [250, 250, 252, 235]
        }),
        new SpatialAnalysisPointLayer({
          id: 'bixi-community-stations',
          coordinateSystem,
          positions: stations,
          instanceCount: stationCount,
          values: stationColors,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: COMMUNITY_PALETTE,
          radiusPixels: options.stationSize
        })
      );
      if (selected >= 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'bixi-community-selected-ring',
            coordinateSystem,
            positions: stations,
            ids: selectedStation,
            instanceCount: 1,
            radiusPixels: options.stationSize + 5,
            color: dark ? [255, 255, 255, 255] : [20, 24, 40, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'bixi-community-selected',
            coordinateSystem,
            positions: stations,
            ids: selectedStation,
            instanceCount: 1,
            values: stationColors,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: COMMUNITY_PALETTE,
            radiusPixels: options.stationSize + 2.5
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      analysis?.reader.stop();
      analysis?.resources.destroy();
      for (const entry of retired) entry.resources.destroy();
      resources.destroy();
    }
  };
}
