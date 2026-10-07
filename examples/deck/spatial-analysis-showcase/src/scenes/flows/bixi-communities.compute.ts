// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGraph,
  GPUGraphLabelPropagation,
  GPUGraphModularity,
  GPUGraphModularityOptimization,
  GPUGraphTopology
} from '@luma.gl/gpgpu/gpu-graph';
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import {formatCount, formatPercent} from '../../cartography/live-text';
import {haversineMeters} from '../../cartography/anchors';
import type {LngLat} from '../../cartography/types';
import {createPolygonOutlineBuffers} from '../../engine/polygon-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipContent} from '../scene';
import {findStationNearPixel, MONTREAL_ORIGIN, readBixiFlows, type BixiFlows} from './bixi-data';
import {
  assignGroupHues,
  getConvexHull,
  getMajorityBoroughs,
  MAX_HUES,
  OTHER_SLOT,
  replayLabelPropagation,
  trimOutliers,
  type HueMemory,
  type PropagationReplay,
  type StationGroups
} from './bixi-communities-groups';
import {
  BETWEEN_LINK_ALPHA,
  BOROUGH_CASING,
  BOROUGH_INK,
  getCommunityPalette,
  HULL_FILL_ALPHA,
  HULL_STROKE_ALPHA,
  LINK_WIDTH_STOPS,
  SEAM_INK,
  STATION_HALO,
  STATION_RADIUS_STOPS,
  type CommunityLegendData
} from './bixi-communities-style';
import {BixiGraphVectors, buildUndirectedEdges, normalizedMutualInformation} from './bixi-graph';
import {BETWEEN_GROUPS_INK} from './flows-style';

/** Option state of the bixi-communities scene. */
export type BixiCommunitiesOptions = {
  partition: 'optimized' | 'propagation' | 'boroughs';
  /** Voting round shown for the propagation partition (a CPU replay of the GPU vote). */
  replayRound: number;
  showHulls: boolean;
  showBoroughs: boolean;
  showSeams: boolean;
  showBetween: boolean;
  /** How many of the busiest links are drawn. */
  edges: number;
  neighbors: number;
  minRides: number;
  weighting: 'rides' | 'sqrt' | 'equal';
  dayType: 'all' | 'weekday' | 'weekend';
  propagationRounds: number;
  optimizeRounds: number;
  resolution: number;
  minimumGain: '0' | '0.0001' | '0.001';
  /** Chart the modularity of the refined partition against the resolution (eight analyses). */
  showSweep: boolean;
};

/** Last round of the replay scrubber (its slider maximum). */
export const REPLAY_LAST_ROUND = 32;

/** Resolutions the Q-against-gamma chart is computed at. */
export const SWEEP_RESOLUTIONS = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3] as const;

const NO_LABEL = 0xffffffff;
const RETIRE_FRAMES = 4;
const REBUILD_DELAY_MS = 250;
/** Names of the boroughs as the legend and the tooltip say them. */
const SHORT_BOROUGH_NAMES: Record<string, string> = {
  'Côte-des-Neiges - Notre-Dame-de-Grâce': 'Côte-des-Neiges',
  'Mercier - Hochelaga-Maisonneuve': 'Hochelaga-Maisonneuve',
  'Rosemont - La Petite-Patrie': 'Rosemont',
  'Villeray—Saint-Michel—Parc-Extension': 'Villeray',
  'Rivière-des-Prairies - Pointe-aux-Trembles': 'Rivière-des-Prairies',
  'Ahuntsic-Cartierville': 'Ahuntsic',
  'Le Plateau-Mont-Royal': 'Plateau-Mont-Royal',
  'Le Sud-Ouest': 'Sud-Ouest'
};

type DayType = BixiCommunitiesOptions['dayType'];

type Analysis = {
  resources: SpatialAnalysisResources;
  vectors: BixiGraphVectors;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  sources: Buffer;
  weights: Buffer;
  resolution: number;
  sweep: boolean;
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

type SweepResult = {resolution: number; refined: number; boroughs: number; groups: number};

/**
 * Communities of riding on the BIXI station graph. Stations are vertices, the rides between two
 * stations (both directions summed) are weighted undirected edges. One compiled graph rebuilds the
 * CSR, runs `GPUGraphLabelPropagation`, refines its partition with
 * `GPUGraphModularityOptimization`, and scores the propagation, the refined partition and the
 * borough partition with `GPUGraphModularity` at the same resolution. Edge weights and a minimum
 * ride filter are buffer writes; rounds, resolution and minimum gain are compile-time constants.
 *
 * Around the GPU result the scene does the cartography on the CPU: identity-stable hues (anchored
 * to the refined partition, west to east on first load), convex hulls, borough seams and a replay
 * of the propagation vote round by round that is checked against the GPU labels.
 */
export async function createBixiCommunities(
  ctx: SceneContext<BixiCommunitiesOptions>
): Promise<SceneInstance<BixiCommunitiesOptions>> {
  const {device} = ctx;
  const flows = readBixiFlows(ctx.datasets.get('bixi-flows'));
  const boroughDataset = ctx.datasets.get('montreal-boroughs');
  const stationCount = flows.stationCount;
  const edges = buildUndirectedEdges(flows);
  const resources = new SpatialAnalysisResources(device, 'bixi-communities');
  const coordinateOrigin: [number, number, number] = [MONTREAL_ORIGIN[0], MONTREAL_ORIGIN[1], 0];
  const centers = flows.centers;

  // ---- Edge data in planar metres, per day type ------------------------------------------------
  const edgeMeters = new Float32Array(edges.count * 4);
  for (let edge = 0; edge < edges.count; edge++) {
    edgeMeters[edge * 4] = centers[edges.a[edge] * 2];
    edgeMeters[edge * 4 + 1] = centers[edges.a[edge] * 2 + 1];
    edgeMeters[edge * 4 + 2] = centers[edges.b[edge] * 2];
    edgeMeters[edge * 4 + 3] = centers[edges.b[edge] * 2 + 1];
  }
  const ridesByDayType = buildRidesByDayType(flows, edges);
  const sortedEdges: Record<DayType, Uint32Array> = {
    all: getRidesOrder(ridesByDayType.all),
    weekday: getRidesOrder(ridesByDayType.weekday),
    weekend: getRidesOrder(ridesByDayType.weekend)
  };
  const ranks: Record<DayType, {a: Uint16Array; b: Uint16Array}> = {
    all: getEdgeRanks(edges.a, edges.b, sortedEdges.all, stationCount),
    weekday: getEdgeRanks(edges.a, edges.b, sortedEdges.weekday, stationCount),
    weekend: getEdgeRanks(edges.a, edges.b, sortedEdges.weekend, stationCount)
  };
  const maxRides: Record<DayType, number> = {
    all: ridesByDayType.all[sortedEdges.all[0]] || 1,
    weekday: ridesByDayType.weekday[sortedEdges.weekday[0]] || 1,
    weekend: ridesByDayType.weekend[sortedEdges.weekend[0]] || 1
  };
  /** CSR over all edges with edge ids, for per-station shares. */
  const adjacency = buildAdjacency(edges.a, edges.b, edges.count, stationCount);
  const boroughLabels = Uint32Array.from(flows.borough);
  /** Rides on the pairs of the month table (pairs with at least three rides). */
  const pairRidesTotal = ridesByDayType.all.reduce((sum, value) => sum + value, 0);
  /** Rides that go from one station to another, from the exact per-station totals. */
  const stationToStationRides = flows.departures.reduce((sum, value) => sum + value, 0);
  /** Straight-line length of every edge in metres. */
  const edgeLengths = new Float32Array(edges.count);
  for (let edge = 0; edge < edges.count; edge++) {
    const a = edges.a[edge];
    const b = edges.b[edge];
    edgeLengths[edge] = haversineMeters(
      [flows.lngLat[a * 2], flows.lngLat[a * 2 + 1]],
      [flows.lngLat[b * 2], flows.lngLat[b * 2 + 1]]
    );
  }

  const boroughMesh = boroughDataset.geojson
    ? buildPolygonMesh(boroughDataset.geojson, (longitude, latitude) =>
        flows.project(longitude, latitude)
      )
    : null;
  const boroughOutline = boroughMesh
    ? createPolygonOutlineBuffers(resources, boroughMesh, 'boroughs')
    : null;

  // ---- GPU buffers the layers read ---------------------------------------------------------------
  const segments = resources.createBuffer('segments', edgeMeters);
  const stations = resources.createBuffer('stations', centers);
  const stationSlots = resources.createBuffer('station-slots', stationCount * 4);
  const boroughSlots = resources.createBuffer('borough-slots', stationCount * 4);
  const edgeSlots = resources.createBuffer('edge-slots', edges.count * 4);
  const withinAlpha = resources.createBuffer('within-alpha', edges.count * 4);
  const betweenAlpha = resources.createBuffer('between-alpha', edges.count * 4);
  const withinOrder = resources.createBuffer('within-order', edges.count * 4);
  const betweenOrder = resources.createBuffer('between-order', edges.count * 4);
  const seamIds = resources.createBuffer('seam-ids', stationCount * 4);
  const identity = resources.createBuffer(
    'identity',
    Uint32Array.from({length: 8}, (_, i) => i)
  );
  const hullCapacity = MAX_HUES * stationCount;
  const hullTriangles = resources.createBuffer('hull-triangles', hullCapacity * 3 * 8);
  const hullFeatures = resources.createBuffer('hull-features', hullCapacity * 3 * 4);
  const hullOutline = resources.createBuffer('hull-outline', hullCapacity * 16);
  const hullOutlineFeatures = resources.createBuffer('hull-outline-features', hullCapacity * 4);

  // Per-refresh working arrays (reused: a refresh runs on every scrub of the replay).
  const edgeSlotValues = new Uint32Array(edges.count);
  const withinAlphaValues = new Float32Array(edges.count);
  const betweenAlphaValues = new Float32Array(edges.count);
  const activeMask = new Uint8Array(edges.count);

  // ---- State -------------------------------------------------------------------------------------
  let analysis: Analysis | null = null;
  let serial = 0;
  let destroyed = false;
  let dirty = true;
  let summary: Summary | null = null;
  let replay: PropagationReplay | null = null;
  let memory: HueMemory | null = null;
  let activeCount = 0;
  let medianLink: string | null = null;
  let selected = -1;
  let comparing = false;
  let rebuildTimer: ReturnType<typeof setTimeout> | null = null;
  let legendFilter: number[] | null = null;
  let lastSubtitle = '';
  const view = {
    shownEdges: 0,
    betweenEdges: 0,
    seamCount: 0,
    hullTriangleVertices: 0,
    hullSegments: 0,
    labels: new Uint32Array(stationCount) as Uint32Array<ArrayBufferLike>,
    groups: null as StationGroups | null,
    boroughGroups: null as StationGroups | null,
    legendSlots: [] as number[]
  };
  const sweep = {
    key: '',
    results: [] as SweepResult[],
    next: 0,
    active: null as Analysis | null,
    encoded: false
  };
  const retired: {resources: SpatialAnalysisResources; frames: number}[] = [];
  const formatQ = (score: number): string => (Number.isFinite(score) ? score.toFixed(2) : '–');

  function getRides(): Float32Array {
    return ridesByDayType[ctx.options.dayType];
  }

  /** An edge stays when it is strong enough and among the strongest links of either end. */
  function computeActiveEdges(): void {
    const {minRides, neighbors, dayType} = ctx.options;
    const rides = ridesByDayType[dayType];
    const rank = ranks[dayType];
    activeCount = 0;
    for (let edge = 0; edge < edges.count; edge++) {
      const keep =
        rides[edge] > 0 &&
        rides[edge] >= minRides &&
        Math.min(rank.a[edge], rank.b[edge]) < neighbors;
      activeMask[edge] = keep ? 1 : 0;
      if (keep) activeCount++;
    }
    updateMedianLink();
  }

  function writeEdgeInputs(target: Analysis): void {
    const {weighting, dayType} = ctx.options;
    const rides = ridesByDayType[dayType];
    const weights = new Float32Array(edges.count);
    const sources = new Uint32Array(edges.count);
    for (let edge = 0; edge < edges.count; edge++) {
      weights[edge] =
        weighting === 'rides' ? rides[edge] : weighting === 'sqrt' ? Math.sqrt(rides[edge]) : 1;
      // An out-of-domain source endpoint makes the graph contributors ignore the edge.
      sources[edge] = activeMask[edge] ? edges.a[edge] : NO_LABEL;
    }
    target.weights.write(weights);
    target.sources.write(sources);
  }

  // ---- The analysis graph ------------------------------------------------------------------------
  function buildAnalysis(resolution: number, sweepOnly: boolean): Analysis {
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
    const {propagationRounds, optimizeRounds, minimumGain} = ctx.options;
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
    if (!sweepOnly) {
      new GPUGraphModularity({
        id: 'bixi-score-propagation',
        graph,
        communities: propagation,
        output: propagationScore,
        resolution
      }).addToGraph(commandGraph);
    }
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
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        const floats = new Float32Array(bytes);
        const base = stationCount * 2;
        const parsed: Summary = {
          propagation: words.slice(0, stationCount),
          optimized: words.slice(stationCount, base),
          propagationConverged: words[base] === 1,
          optimizedConverged: words[base + 1] === 1,
          valid: words[base + 2] === 1,
          overflow: words[base + 3] !== 0,
          propagationScore: floats[base + 4],
          optimizedScore: floats[base + 5],
          boroughScore: floats[base + 6]
        };
        if (sweepOnly) {
          if (sweep.active === built) finishSweepStep(parsed, resolution);
        } else if (analysis === built) {
          processSummary(parsed);
        }
      }
    );
    const built: Analysis = {
      resources: graphResources,
      vectors,
      compiled,
      reader,
      sources: vectors.getBuffer(sourceVector),
      weights: vectors.getBuffer(weightVector),
      resolution,
      sweep: sweepOnly
    };
    return built;
  }

  function rebuild(): void {
    if (analysis) retired.push({resources: analysis.resources, frames: 0});
    computeActiveEdges();
    analysis = buildAnalysis(ctx.options.resolution, false);
    writeEdgeInputs(analysis);
    computeReplay();
    dirty = true;
    ctx.setCost({records: edges.count, passes: analysis.compiled.stats.nodeOrder.length});
  }

  /** Compile-time options change in bursts while a slider moves: wait for the pointer to rest. */
  function scheduleRebuild(): void {
    if (rebuildTimer) clearTimeout(rebuildTimer);
    rebuildTimer = setTimeout(() => {
      rebuildTimer = null;
      if (!destroyed) rebuild();
    }, REBUILD_DELAY_MS);
  }

  function updateEdgeInputs(): void {
    computeActiveEdges();
    if (analysis) writeEdgeInputs(analysis);
    computeReplay();
    dirty = true;
    resetSweep();
  }

  // ---- CPU replay of the label-propagation vote ---------------------------------------------------
  function computeReplay(): void {
    replay = replayLabelPropagation(
      stationCount,
      {a: edges.a, b: edges.b, mask: activeMask, count: edges.count},
      ctx.options.propagationRounds
    );
  }

  function updateReplayCheck(): void {
    if (!replay || !summary || dirty) return;
    const finalRound = replay.labels[replay.labels.length - 1];
    let different = 0;
    for (let station = 0; station < stationCount; station++) {
      if (finalRound[station] !== summary.propagation[station]) different++;
    }
    ctx.setReadout(
      'replayMatch',
      different === 0
        ? `yes, all ${formatCount(stationCount)} labels after round ${ctx.options.propagationRounds}`
        : `no: ${formatCount(different)} labels differ`
    );
  }

  // ---- Q against gamma ------------------------------------------------------------------------------
  function getSweepKey(): string {
    const o = ctx.options;
    return [
      o.neighbors,
      o.minRides,
      o.weighting,
      o.dayType,
      o.propagationRounds,
      o.optimizeRounds,
      o.minimumGain
    ].join('|');
  }

  function resetSweep(): void {
    sweep.key = '';
    sweep.results = [];
    sweep.next = 0;
    if (sweep.active) retired.push({resources: sweep.active.resources, frames: 0});
    sweep.active = null;
    sweep.encoded = false;
    ctx.setChart('sweepChart', null);
  }

  function finishSweepStep(parsed: Summary, resolution: number): void {
    const groups = new Set(parsed.optimized).size;
    sweep.results.push({
      resolution,
      refined: parsed.optimizedScore,
      boroughs: parsed.boroughScore,
      groups
    });
    if (sweep.active) retired.push({resources: sweep.active.resources, frames: 0});
    sweep.active = null;
    sweep.encoded = false;
    sweep.next++;
    publishSweepChart();
  }

  function advanceSweep(commandEncoder: CommandEncoder): void {
    const key = getSweepKey();
    if (key !== sweep.key) {
      resetSweep();
      sweep.key = key;
    }
    if (sweep.next >= SWEEP_RESOLUTIONS.length) return;
    if (!sweep.active) {
      // One analysis per frame: compile now, run on the next frame.
      sweep.active = buildAnalysis(SWEEP_RESOLUTIONS[sweep.next], true);
      writeEdgeInputs(sweep.active);
      sweep.encoded = false;
      return;
    }
    if (!sweep.encoded) {
      sweep.active.compiled.encode(commandEncoder, {parameters: undefined});
      sweep.active.reader.request(commandEncoder);
      sweep.encoded = true;
      return;
    }
    sweep.active.reader.flush(commandEncoder);
  }

  function publishSweepChart(): void {
    if (sweep.results.length === 0) return;
    const x = sweep.results.map(result => result.resolution);
    ctx.setChart('sweepChart', {
      kind: 'line',
      series: [
        {label: 'riding groups', x, y: sweep.results.map(r => r.refined), color: 0, points: true},
        {
          label: 'boroughs',
          x,
          y: sweep.results.map(r => r.boroughs),
          color: 3,
          dashed: true,
          points: true
        }
      ],
      xDomain: [SWEEP_RESOLUTIONS[0], SWEEP_RESOLUTIONS[SWEEP_RESOLUTIONS.length - 1]],
      xLabel: 'resolution gamma',
      yLabel: 'modularity Q',
      height: 140,
      formatY: value => value.toFixed(2),
      formatX: value => String(value),
      guides: [{y: 0, label: 'chance'}],
      link: {option: 'resolution', label: value => `gamma ${value}`},
      description: `Modularity of the refined partition and of the boroughs at ${sweep.results.length} resolutions, ${sweep.results.length < SWEEP_RESOLUTIONS.length ? 'still being computed' : 'each analysis compiled and run once'}. The groups at gamma ${sweep.results[sweep.results.length - 1].resolution}: ${sweep.results[sweep.results.length - 1].groups}.`
    });
    ctx.setReadout(
      'sweepStatus',
      sweep.results.length < SWEEP_RESOLUTIONS.length
        ? `${sweep.results.length} of ${SWEEP_RESOLUTIONS.length} computed`
        : `${SWEEP_RESOLUTIONS.length} analyses, each compiled and run once`
    );
  }

  // ---- Names ---------------------------------------------------------------------------------------
  const shortBorough = (index: number): string => {
    const name = flows.boroughNames[index];
    return SHORT_BOROUGH_NAMES[name] ?? name.replace(/^Le /, '');
  };

  /** A group is named after its two most common boroughs. */
  function getGroupName(labels: ArrayLike<number>, label: number, size: number): string {
    const counts = new Map<number, number>();
    for (let station = 0; station < stationCount; station++) {
      if (labels[station] === label) {
        counts.set(flows.borough[station], (counts.get(flows.borough[station]) ?? 0) + 1);
      }
    }
    const ordered = Array.from(counts.entries()).sort((x, y) => y[1] - x[1] || x[0] - y[0]);
    const first = shortBorough(ordered[0][0]);
    return ordered.length > 1 && ordered[1][1] >= 0.2 * size
      ? `${first} + ${shortBorough(ordered[1][0])}`
      : first;
  }

  // ---- Refresh: everything that depends on the shown partition ---------------------------------------
  function getEffectiveRound(): number {
    return Math.min(ctx.options.replayRound, ctx.options.propagationRounds);
  }

  /** Labels of the partition the map shows, and whether they are a CPU replay of an earlier round. */
  function getShownLabels(current: Summary): {labels: Uint32Array; replayRound: number | null} {
    const {partition, propagationRounds} = ctx.options;
    if (partition === 'boroughs') return {labels: boroughLabels, replayRound: null};
    if (partition === 'propagation') {
      const round = getEffectiveRound();
      if (round < propagationRounds && replay) {
        return {labels: replay.labels[round], replayRound: round};
      }
      return {labels: current.propagation, replayRound: null};
    }
    return {labels: current.optimized, replayRound: null};
  }

  function refresh(): void {
    if (!summary) return;
    const current = summary;
    const {partition} = ctx.options;
    const {labels, replayRound} = getShownLabels(current);
    view.labels = labels;
    const groups = assignGroupHues(labels, flows.lngLat, memory);
    const boroughGroups = assignGroupHues(boroughLabels, flows.lngLat, memory);
    view.groups = groups;
    view.boroughGroups = boroughGroups;
    stationSlots.write(groups.slots);
    boroughSlots.write(boroughGroups.slots);

    // Edges: within a group in the group's hue, between groups faint and first.
    const rides = getRides();
    const maximum = maxRides[ctx.options.dayType];
    const order = sortedEdges[ctx.options.dayType];
    const drawLimit = ctx.options.edges;
    const within: number[] = [];
    const between: number[] = [];
    let keptRides = 0;
    let withinRidesTotal = 0;
    let boroughWithinRides = 0;
    for (let index = 0; index < order.length; index++) {
      const edge = order[index];
      if (!activeMask[edge]) continue;
      const a = edges.a[edge];
      const b = edges.b[edge];
      keptRides += rides[edge];
      if (boroughLabels[a] === boroughLabels[b]) boroughWithinRides += rides[edge];
      const share = Math.sqrt(rides[edge] / maximum);
      if (labels[a] === labels[b]) {
        withinRidesTotal += rides[edge];
        edgeSlotValues[edge] = groups.slots[a];
        withinAlphaValues[edge] = 0.15 + 0.7 * share;
        if (within.length < drawLimit) within.push(edge);
      } else {
        betweenAlphaValues[edge] = 0.5 + 0.5 * share;
        if (between.length < drawLimit) between.push(edge);
      }
    }
    edgeSlots.write(edgeSlotValues);
    withinAlpha.write(withinAlphaValues);
    betweenAlpha.write(betweenAlphaValues);
    // Heaviest drawn last: the lists are heaviest first.
    // An empty list is not written: the layer's instance count is zero and draws nothing.
    if (within.length > 0) withinOrder.write(Uint32Array.from(within.reverse()));
    if (between.length > 0) betweenOrder.write(Uint32Array.from(between.reverse()));
    view.shownEdges = within.length;
    view.betweenEdges = between.length;

    // Seams: a station whose group is mostly in another borough.
    const seams: number[] = [];
    if (partition !== 'boroughs') {
      const majority = getMajorityBoroughs(labels, flows.borough);
      for (let station = 0; station < stationCount; station++) {
        if (majority.get(labels[station]) !== flows.borough[station]) seams.push(station);
      }
    }
    if (seams.length > 0) seamIds.write(Uint32Array.from(seams));
    view.seamCount = seams.length;

    writeHulls(groups, labels);
    publishLegend(groups, labels, seams.length);
    publishReadouts(current, groups, labels, {
      keptRides,
      withinRidesTotal,
      boroughWithinRides,
      replayRound
    });
    publishCharts(current);
    updateFurniture(replayRound);
    describeSelection();
    ctx.requestLayers();
  }

  /** Convex hull wash and outline per hued group, outliers trimmed first. */
  function writeHulls(groups: StationGroups, labels: ArrayLike<number>): void {
    const triangles: number[] = [];
    const features: number[] = [];
    const outline: number[] = [];
    const outlineFeatures: number[] = [];
    for (const group of groups.hued) {
      const points: [number, number][] = [];
      for (let station = 0; station < stationCount; station++) {
        if (labels[station] === group.label) {
          points.push([centers[station * 2], centers[station * 2 + 1]]);
        }
      }
      const hull = getConvexHull(trimOutliers(points));
      if (hull.length < 3) continue;
      for (let index = 1; index < hull.length - 1; index++) {
        for (const vertex of [hull[0], hull[index], hull[index + 1]]) {
          triangles.push(vertex[0], vertex[1]);
          features.push(group.slot);
        }
      }
      for (let index = 0; index < hull.length; index++) {
        const next = hull[(index + 1) % hull.length];
        outline.push(hull[index][0], hull[index][1], next[0], next[1]);
        outlineFeatures.push(group.slot);
      }
    }
    if (triangles.length > 0) {
      hullTriangles.write(new Float32Array(triangles));
      hullFeatures.write(Uint32Array.from(features));
      hullOutline.write(new Float32Array(outline));
      hullOutlineFeatures.write(Uint32Array.from(outlineFeatures));
    }
    view.hullTriangleVertices = features.length;
    view.hullSegments = outlineFeatures.length;
  }

  function publishLegend(
    groups: StationGroups,
    labels: ArrayLike<number>,
    seamCount: number
  ): void {
    const entries = groups.hued.map(group => ({
      slot: group.slot,
      label:
        ctx.options.partition === 'boroughs'
          ? shortBorough(group.label)
          : getGroupName(labels, group.label, group.size),
      stations: group.size
    }));
    // Colour by slot order so the legend does not reshuffle when sizes change.
    entries.sort((a, b) => a.slot - b.slot);
    view.legendSlots = entries.map(entry => entry.slot);
    const greyGroups = groups.groupCount - groups.hued.length;
    const data: CommunityLegendData = {
      entries,
      greyGroups,
      greyStations: groups.greyStations,
      ground: ctx.ground(),
      seamCount
    };
    ctx.setLegendData('communities', data);
  }

  function publishReadouts(
    current: Summary,
    groups: StationGroups,
    labels: ArrayLike<number>,
    stats: {
      keptRides: number;
      withinRidesTotal: number;
      boroughWithinRides: number;
      replayRound: number | null;
    }
  ): void {
    const modularity = (score: number) => (Number.isFinite(score) ? score : null);
    ctx.setReadout('stations', stationCount);
    ctx.setReadout('edgesKept', `${formatCount(activeCount)} of ${formatCount(edges.count)}`);
    ctx.setReadout('ridesKept', pairRidesTotal > 0 ? stats.keptRides / pairRidesTotal : null);
    ctx.setReadout('communityCount', groups.groupCount);
    const largest = groups.hued[0];
    ctx.setReadout(
      'largestCommunity',
      largest
        ? `${ctx.options.partition === 'boroughs' ? shortBorough(largest.label) : getGroupName(labels, largest.label, largest.size)}: ${formatCount(largest.size)} stations`
        : null
    );
    ctx.setReadout('modularityPropagation', modularity(current.propagationScore));
    ctx.setReadout('modularityRefined', modularity(current.optimizedScore));
    ctx.setReadout('modularityBoroughs', modularity(current.boroughScore));
    ctx.setReadout(
      'qualityPair',
      `boroughs ${formatQ(current.boroughScore)}, riding groups ${formatQ(current.optimizedScore)}`
    );
    ctx.setReadout(
      'propagationStatus',
      current.propagationConverged
        ? `converged within ${ctx.options.propagationRounds} rounds`
        : `no fixed point in ${ctx.options.propagationRounds} rounds`
    );
    ctx.setReadout(
      'refinementStatus',
      current.optimizedConverged
        ? 'converged'
        : `round budget used (${formatCount(ctx.options.optimizeRounds)} single moves at most)`
    );
    ctx.setReadout(
      'validity',
      current.overflow
        ? 'adjacency overflow: labels invalid'
        : current.valid
          ? 'valid'
          : 'invalid (no edges?)'
    );
    ctx.setReadout(
      'withinShare',
      stats.keptRides > 0 ? stats.withinRidesTotal / stats.keptRides : null
    );
    ctx.setReadout(
      'withinBoroughShare',
      stats.keptRides > 0 ? stats.boroughWithinRides / stats.keptRides : null
    );
    ctx.setReadout('seamStations', ctx.options.partition === 'boroughs' ? null : view.seamCount);
    ctx.setReadout('agreement', normalizedMutualInformation(labels, boroughLabels));
    ctx.setReadout('medianLink', medianLink);
  }

  /** Ride-weighted median length of the kept links; recomputed when the graph inputs change. */
  function updateMedianLink(): void {
    const rides = getRides();
    const order: number[] = [];
    let total = 0;
    for (let edge = 0; edge < edges.count; edge++) {
      if (!activeMask[edge]) continue;
      order.push(edge);
      total += rides[edge];
    }
    medianLink = null;
    if (total === 0) return;
    order.sort((x, y) => edgeLengths[x] - edgeLengths[y]);
    let running = 0;
    for (const edge of order) {
      running += rides[edge];
      if (running >= total / 2) {
        medianLink = `${formatCount(edgeLengths[edge])} m`;
        return;
      }
    }
  }

  function publishCharts(current: Summary): void {
    const {partition, propagationRounds} = ctx.options;
    const scores = [current.boroughScore, current.propagationScore, current.optimizedScore].map(
      value => (Number.isFinite(value) ? value : 0)
    );
    ctx.setChart('qualityChart', {
      kind: 'bars',
      values: scores,
      labels: ['boroughs', 'propagation', 'refined'],
      highlight: [partition === 'boroughs' ? 0 : partition === 'propagation' ? 1 : 2],
      height: 140,
      yLabel: 'modularity Q',
      yDomain: [Math.min(0, ...scores) - 0.05, Math.max(0.8, ...scores) + 0.05],
      formatY: value => value.toFixed(2),
      guides: [
        {y: 0, label: 'chance'},
        {y: 0.3, label: 'typical 0.3'},
        {y: 0.7, label: 'to 0.7'}
      ],
      description:
        'Weighted modularity of the borough partition, of label propagation and of the refined partition at the current resolution. Zero is what a random network with the same degrees would give; values from 0.3 to 0.7 are typical of networks with real community structure.'
    });
    if (replay) {
      const round = getEffectiveRound();
      ctx.setChart('changesChart', {
        kind: 'bars',
        values: replay.changed,
        labels: replay.changed.map((_, index) => String(index + 1)),
        highlight: round > 0 ? [round - 1] : [],
        height: 120,
        xLabel: 'voting round',
        yLabel: 'stations that changed label',
        formatY: value => formatCount(value),
        onBarClick: index =>
          ctx.setOptions({replayRound: Math.min(REPLAY_LAST_ROUND, index + 1)}, {notify: true}),
        description: `How many stations changed their label in each of the ${propagationRounds} synchronous voting rounds. After the early rounds the votes settle; a few stations may keep flipping.`
      });
    }
  }

  function updateFurniture(replayRound: number | null): void {
    const {partition, resolution, propagationRounds} = ctx.options;
    const rides = formatCount(stationToStationRides);
    let subtitle: string;
    if (partition === 'boroughs') {
      subtitle = 'Stations by borough of the agglomeration, BIXI station pairs, August 2024';
    } else if (partition === 'propagation') {
      subtitle =
        replayRound === null
          ? `Label propagation after ${propagationRounds} rounds (GPU), BIXI station pairs, August 2024`
          : `Label propagation after round ${replayRound} of ${propagationRounds} (CPU replay), BIXI station pairs, August 2024`;
    } else {
      subtitle = `Riding groups by modularity optimisation, resolution ${resolution}, BIXI station pairs, August 2024`;
    }
    const key = `${subtitle}|${rides}`;
    if (key === lastSubtitle) return;
    lastSubtitle = key;
    ctx.setFurniture({
      title: {
        subtitle,
        sample: `${formatCount(stationCount)} stations, ${rides} rides between stations`
      },
      scaleBar: {units: 'metric'}
    });
  }

  function processSummary(next: Summary): void {
    summary = next;
    // The refined partition carries the hues: the first one runs west to east, later ones inherit.
    memory = assignGroupHues(next.optimized, flows.lngLat, memory).memory;
    updateReplayCheck();
    refresh();
  }

  // ---- Selection and tooltip --------------------------------------------------------------------------
  function getInsideShare(station: number): number | null {
    let inside = 0;
    let total = 0;
    const rides = getRides();
    for (let slot = adjacency.offsets[station]; slot < adjacency.offsets[station + 1]; slot++) {
      const edge = adjacency.edgeIds[slot];
      if (!activeMask[edge]) continue;
      total += rides[edge];
      if (view.labels[adjacency.neighbors[slot]] === view.labels[station]) inside += rides[edge];
    }
    return total > 0 ? inside / total : null;
  }

  function describeSelection(): void {
    if (selected < 0 || !view.groups) {
      ctx.setReadout('selected', 'click a station');
      return;
    }
    const share = getInsideShare(selected);
    const size = view.groups.sizes.get(view.labels[selected]) ?? 1;
    ctx.setReadout(
      'selected',
      `${flows.names[selected]}\n${flows.boroughNames[flows.borough[selected]]}\ngroup of ${formatCount(size)} stations\n${share === null ? 'no kept links' : `${formatPercent(share)} of its rides stay inside`}`
    );
  }

  function getTooltip(station: number): TooltipContent | null {
    const groups = view.groups;
    if (!groups) return null;
    const palette = getCommunityPalette(ctx.ground());
    const slot = groups.slots[station];
    const label = view.labels[station];
    const size = groups.sizes.get(label) ?? 1;
    const hued = groups.hued.some(group => group.label === label);
    const name =
      ctx.options.partition === 'boroughs'
        ? shortBorough(label)
        : hued
          ? getGroupName(view.labels, label, size)
          : 'a small group';
    const share = getInsideShare(station);
    return {
      title: flows.names[station],
      subtitle: flows.boroughNames[flows.borough[station]],
      rows: [
        {
          label: ctx.options.partition === 'boroughs' ? 'Borough' : 'Riding group',
          value: `${name} (${formatCount(size)} stations)`,
          swatch: palette[slot],
          emphasis: true
        },
        {
          label: 'Rides that stay inside its group',
          value: share === null ? '–' : formatPercent(share)
        },
        {
          label: 'Departures',
          value: formatCount(flows.departures[station]),
          unit: 'rides in August'
        },
        {label: 'Arrivals', value: formatCount(flows.arrivals[station]), unit: 'rides in August'}
      ],
      highlight: {
        kind: 'point',
        coordinate: [flows.lngLat[station * 2], flows.lngLat[station * 2 + 1]]
      }
    };
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
          if (id === 'propagationRounds') {
            dirty = true;
            computeActiveEdges();
            computeReplay();
          }
          // The scan depends on every compile option but the resolution it varies itself.
          if (id !== 'resolution') resetSweep();
          scheduleRebuild();
          break;
        case 'weighting':
        case 'minRides':
        case 'neighbors':
        case 'dayType':
          updateEdgeInputs();
          refresh();
          break;
        case 'partition':
        case 'replayRound':
        case 'edges':
        case 'showBetween':
        case 'showSeams':
        case 'showBoroughs':
        case 'showHulls':
          refresh();
          break;
        case 'showSweep':
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      refresh();
    },

    onGroundChange() {
      refresh();
    },

    onLegendFilter(_id, classes) {
      legendFilter =
        classes === null ? null : classes.map(index => view.legendSlots[index] ?? OTHER_SLOT);
      ctx.requestLayers();
    },

    getTooltip(event) {
      const station = findStationNearPixel(
        ctx.getViewport(),
        flows.lngLat,
        stationCount,
        event.pixel
      );
      return station >= 0 ? getTooltip(station) : null;
    },

    onClick(event) {
      const station = findStationNearPixel(
        ctx.getViewport(),
        flows.lngLat,
        stationCount,
        event.pixel,
        14
      );
      if (station < 0) return false;
      selected = station === selected ? -1 : station;
      const coordinate: LngLat = [flows.lngLat[station * 2], flows.lngLat[station * 2 + 1]];
      ctx.setHighlight(selected >= 0 ? {kind: 'point', coordinate, radiusPixels: 9} : null);
      describeSelection();
      return true;
    },

    encode(commandEncoder) {
      if (!analysis) return;
      for (let index = retired.length - 1; index >= 0; index--) {
        if (++retired[index].frames > RETIRE_FRAMES) {
          retired[index].resources.destroy();
          retired.splice(index, 1);
        }
      }
      const nowComparing = ctx.getCompare() !== null;
      if (nowComparing !== comparing) {
        comparing = nowComparing;
        ctx.requestLayers();
      }
      if (dirty && !analysis.reader.isPending) {
        analysis.compiled.encode(commandEncoder, {parameters: undefined});
        analysis.reader.request(commandEncoder);
        dirty = false;
      } else {
        analysis.reader.flush(commandEncoder);
      }
      if (ctx.options.showSweep && !dirty && summary) advanceSweep(commandEncoder);
    },

    getLayers() {
      const groups = view.groups;
      if (!groups) return [];
      const options = ctx.options;
      const ground = ctx.ground();
      const hullPalette = getCommunityPalette(ground, HULL_FILL_ALPHA);
      const hullStrokePalette = getCommunityPalette(ground, HULL_STROKE_ALPHA);
      const solidPalette = getCommunityPalette(ground, 235);
      const linkPalette = getCommunityPalette(ground, 255);
      const layers: Layer[] = [];
      // In a compare step the communities are side b and the boroughs side a.
      const communitySide = comparing ? ({compareSide: 'b'} as const) : {};
      const boroughSide = comparing ? ({compareSide: 'a'} as const) : {};

      if (options.showHulls && view.hullTriangleVertices > 0) {
        layers.push(
          new SpatialAnalysisPolygonLayer({
            id: 'bixi-community-hull-fill',
            coordinateOrigin,
            triangles: hullTriangles,
            features: hullFeatures,
            vertexCount: view.hullTriangleVertices,
            values: identity,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: hullPalette,
            highlightClasses: legendFilter,
            ...communitySide
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'bixi-community-hull-outline',
            coordinateOrigin,
            segments: hullOutline,
            instanceCount: view.hullSegments,
            valueIndices: hullOutlineFeatures,
            values: identity,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: hullStrokePalette,
            widthPixels: 0.75,
            cap: 'butt',
            highlightClasses: legendFilter,
            ...communitySide
          })
        );
      }
      if (options.showBoroughs && boroughOutline && boroughOutline.outlineCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'bixi-community-boroughs',
            coordinateOrigin,
            segments: boroughOutline.outline,
            instanceCount: boroughOutline.outlineCount,
            widthPixels: 0.9,
            cap: 'butt',
            color: BOROUGH_INK[ground],
            outlineColor: BOROUGH_CASING[ground],
            outlineWidthPixels: 0.8
          })
        );
      }
      if (options.showBetween && view.betweenEdges > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'bixi-community-links-between',
            coordinateOrigin,
            segments,
            ids: betweenOrder,
            instanceCount: view.betweenEdges,
            weights: betweenAlpha,
            widthPixels: LINK_WIDTH_STOPS,
            color: [
              BETWEEN_GROUPS_INK[ground][0],
              BETWEEN_GROUPS_INK[ground][1],
              BETWEEN_GROUPS_INK[ground][2],
              BETWEEN_LINK_ALPHA + 18
            ],
            ...communitySide
          })
        );
      }
      if (view.shownEdges > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'bixi-community-links-within',
            coordinateOrigin,
            segments,
            ids: withinOrder,
            instanceCount: view.shownEdges,
            values: edgeSlots,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: linkPalette,
            weights: withinAlpha,
            widthPixels: LINK_WIDTH_STOPS,
            highlightClasses: legendFilter,
            ...communitySide
          })
        );
      }
      const stationProps = {
        coordinateOrigin,
        positions: stations,
        instanceCount: stationCount,
        valueFormat: 'uint32' as const,
        colormap: 'category' as const,
        palette: solidPalette,
        radiusPixels: STATION_RADIUS_STOPS,
        outlineColor: STATION_HALO[ground],
        outlineWidthPixels: 1,
        highlightClasses: legendFilter
      };
      if (comparing) {
        layers.push(
          new SpatialAnalysisPointLayer({
            ...stationProps,
            id: 'bixi-community-stations-boroughs',
            values: boroughSlots,
            ...boroughSide
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          ...stationProps,
          id: 'bixi-community-stations',
          values: stationSlots,
          ...communitySide
        })
      );
      if (options.showSeams && view.seamCount > 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'bixi-community-seams',
            coordinateOrigin,
            positions: stations,
            ids: seamIds,
            instanceCount: view.seamCount,
            shape: 'ring',
            radiusPixels: [
              [10, 6],
              [13, 8]
            ],
            outlineWidthPixels: 2,
            color: SEAM_INK[ground]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      if (rebuildTimer) clearTimeout(rebuildTimer);
      ctx.setHighlight(null);
      analysis?.reader.stop();
      analysis?.resources.destroy();
      sweep.active?.reader.stop();
      sweep.active?.resources.destroy();
      for (const entry of retired) entry.resources.destroy();
      resources.destroy();
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Edge helpers
// ---------------------------------------------------------------------------------------------

/**
 * Rides per undirected edge for all days, weekdays and weekend days. The day-type columns come
 * from the hourly slice rows (pairs with at least ten rides in August), so a pair that is only in
 * the month table has no weekday or weekend rides and drops out of those graphs.
 */
function buildRidesByDayType(
  flows: BixiFlows,
  edges: ReturnType<typeof buildUndirectedEdges>
): Record<DayType, Float32Array> {
  const index = new Map<number, number>();
  const {stationCount, slices} = flows;
  for (let edge = 0; edge < edges.count; edge++) {
    index.set(edges.a[edge] * stationCount + edges.b[edge], edge);
  }
  const weekday = new Float32Array(edges.count);
  const weekend = new Float32Array(edges.count);
  for (let row = 0; row < slices.rowCount; row++) {
    const from = slices.origin[row];
    const to = slices.destination[row];
    if (from >= stationCount || to >= stationCount || from === to) continue;
    const edge = index.get(from < to ? from * stationCount + to : to * stationCount + from);
    if (edge === undefined) continue;
    (slices.dayType[row] === 0 ? weekday : weekend)[edge] += slices.count[row];
  }
  return {all: edges.rides, weekday, weekend};
}

/** Edge indices, most rides first (ties by index). */
function getRidesOrder(rides: Float32Array): Uint32Array {
  return Uint32Array.from({length: rides.length}, (_, edge) => edge).sort(
    (x, y) => rides[y] - rides[x] || x - y
  );
}

/** Rank of each edge among the edges of its two stations, 0 being a station's strongest link. */
function getEdgeRanks(
  a: Uint32Array,
  b: Uint32Array,
  order: Uint32Array,
  stationCount: number
): {a: Uint16Array; b: Uint16Array} {
  const rankA = new Uint16Array(a.length);
  const rankB = new Uint16Array(a.length);
  const seen = new Uint16Array(stationCount);
  // Edges are visited strongest first, so a station's running count is the rank of its next edge.
  for (const edge of order) {
    rankA[edge] = Math.min(65535, seen[a[edge]]++);
    rankB[edge] = Math.min(65535, seen[b[edge]]++);
  }
  return {a: rankA, b: rankB};
}

/** CSR over undirected edges with the edge id of every slot. */
function buildAdjacency(a: Uint32Array, b: Uint32Array, count: number, stationCount: number) {
  const offsets = new Uint32Array(stationCount + 1);
  for (let edge = 0; edge < count; edge++) {
    offsets[a[edge] + 1]++;
    offsets[b[edge] + 1]++;
  }
  for (let station = 0; station < stationCount; station++) offsets[station + 1] += offsets[station];
  const neighbors = new Uint32Array(offsets[stationCount]);
  const edgeIds = new Uint32Array(offsets[stationCount]);
  const cursor = offsets.slice(0, stationCount);
  for (let edge = 0; edge < count; edge++) {
    neighbors[cursor[a[edge]]] = b[edge];
    edgeIds[cursor[a[edge]]++] = edge;
    neighbors[cursor[b[edge]]] = a[edge];
    edgeIds[cursor[b[edge]]++] = edge;
  }
  return {offsets, neighbors, edgeIds};
}
