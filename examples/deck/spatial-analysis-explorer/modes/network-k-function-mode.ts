// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Network-constrained K function of New York points of interest, with a simulation envelope of
 * random points on the same roads, and a second analysis on the same network: reachability over
 * the turn-restriction line graph.
 *
 * Graph 1 (`GPUNetworkKFunction`) snaps up to 192 sampled POIs onto the roads, runs multi-source
 * shortest-path searches limited to the maximum distance for the observed pattern and 19 random
 * patterns, and counts event pairs per distance band. Graph 2 builds the line graph
 * (`GPUNetworkLineGraph`: one node per directed street segment, one arc per allowed turn, turn
 * costs from the angle) and runs `GPUNetworkReachability` over it next to the plain node graph.
 * Distances, snap distance, active simulations, seed, turn costs and the routing budget are all
 * buffer writes; both graphs compile once.
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUNetworkReachability,
  GPU_NETWORK_REACHABILITY_NONE
} from '@luma.gl/experimental/gpu-network';
import {
  GPUNetworkKFunction,
  getGPUNetworkKFunctionParameterValues,
  GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-network';
import {
  GPUNetworkLineGraph,
  getGPUNetworkLineGraphParameterValues,
  GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-network';
import {importGraphBuffer} from '../graph-buffers';
import {createSeededRandom, LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {addKernelPass} from './mode-kernels';
import {MiniChart} from './point-pattern-chart';
import {SummaryReader} from './summary-reader';
import {findNearestNode, sortEdgesBySource} from './road-network-utils';

/** Events sampled from the points of interest (compile-time row count). */
const EVENT_COUNT = 192;
/** Simulated patterns of the envelope (compile-time). */
const SIMULATION_COUNT = 19;
/** Snapping candidate capacity (BVH join instead of scanning every edge for every event). */
const SNAP_CANDIDATE_CAPACITY = EVENT_COUNT * 512;
/** Distance thresholds of the K function. */
const BAND_COUNT = 24;
/**
 * Shortest-path searches per block; bounds scratch memory (128 rows of the 97k-edge network is
 * about the 128 MB lane-expansion budget of `recommendLaneCount`).
 */
const ROWS_PER_BLOCK = 128;
/** Relaxation rounds per search block; at 16 hops per round this covers 1.5 km of short edges. */
const K_MAXIMUM_ITERATIONS = 24;
/** Rounds of the routing searches (plain and line graph). */
const ROUTE_MAXIMUM_ITERATIONS = 48;
const LOCAL_ITERATIONS = 16;
const FAR_AWAY = 1e7;
const NO_EDGE = 0xffffffff;
/** Times Square. */
const DEFAULT_ORIGIN: readonly [number, number] = [-73.9855, 40.758];

type RouteColor = 'turns' | 'plain' | 'extra';

export const networkKFunctionMode: SpatialAnalysisModeDefinition = {
  id: 'network-k-function',
  title: 'Network K',
  contributors: [
    'GPUNetworkKFunction',
    'GPUNetworkSnapping',
    'GPUNetworkCostMatrix',
    'GPUNetworkReachability',
    'GPUNetworkLineGraph'
  ],
  description:
    'Network-constrained Ripley K of New York points of interest: distances run along the ' +
    'streets, not through buildings. The chart shows observed K(d) against the envelope of ' +
    'random points on the same roads (above the grey band means clustered along the network). ' +
    'Move the maximum distance, the snap distance or the seed. Switch on routing, click the map ' +
    'and compare plain reachability with reachability over the turn line graph.',
  initialViewState: {longitude: -73.985, latitude: 40.745, zoom: 12.6},

  async create(context) {
    const [roads, pois] = await Promise.all([
      context.data.getNewYorkRoads(),
      context.data.getNewYorkPointsOfInterest()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(roads.origin);
    const resources = new SpatialAnalysisResources(device, 'network-k');
    const nodeCount = roads.nodePositions.length / 2;
    const segmentCount = roads.segmentNodes.length;

    // ---- Road CSR: both directions of every street segment, lengths in meters ----
    const edges = sortEdgesBySource(roads);
    const edgeCount = edges.sources.length;
    const offsets = new Uint32Array(nodeCount + 1);
    for (let edge = 0; edge < edgeCount; edge++) offsets[edges.sources[edge] + 1]++;
    for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
    let totalLength = 0;
    for (let edge = 0; edge < edgeCount; edge++) totalLength += edges.lengths[edge];
    const networkLength = totalLength / 2;
    // Reverse edge row per edge, so the arrival cost of a node reads the line node of the edge
    // that enters it from the reverse of each outgoing edge.
    const edgeKeys = new Map<number, number>();
    for (let edge = 0; edge < edgeCount; edge++) {
      edgeKeys.set(edges.sources[edge] * nodeCount + edges.targets[edge], edge);
    }
    const reverseEdges = new Uint32Array(edgeCount);
    let arcCapacity = 0;
    let maximumDegree = 1;
    for (let edge = 0; edge < edgeCount; edge++) {
      reverseEdges[edge] =
        edgeKeys.get(edges.targets[edge] * nodeCount + edges.sources[edge]) ?? NO_EDGE;
      const head = edges.targets[edge];
      arcCapacity += offsets[head + 1] - offsets[head];
    }
    for (let node = 0; node < nodeCount; node++) {
      maximumDegree = Math.max(maximumDegree, offsets[node + 1] - offsets[node]);
    }
    arcCapacity = Math.max(arcCapacity, 1);

    // ---- Events: a deterministic sample of one POI category ----
    const categoryCounts = new Map<number, number>();
    for (const category of pois.categories) {
      categoryCounts.set(category, (categoryCounts.get(category) ?? 0) + 1);
    }
    const topCategories = [...categoryCounts.entries()]
      .sort((left, right) => right[1] - left[1])
      .slice(0, 5)
      .map(([category]) => category);
    const sampleEvents = (category: number | null): Float32Array => {
      const candidates: number[] = [];
      for (let poi = 0; poi < pois.categories.length; poi++) {
        if (category === null || pois.categories[poi] === category) candidates.push(poi);
      }
      const random = createSeededRandom(3);
      for (let index = candidates.length - 1; index > 0; index--) {
        const swap = Math.floor(random() * (index + 1));
        [candidates[index], candidates[swap]] = [candidates[swap], candidates[index]];
      }
      const positions = new Float32Array(EVENT_COUNT * 2).fill(FAR_AWAY);
      candidates.slice(0, EVENT_COUNT).forEach((poi, slot) => {
        positions[slot * 2] = pois.positions[poi * 2];
        positions[slot * 2 + 1] = pois.positions[poi * 2 + 1];
      });
      return positions;
    };

    // ---- Buffers ----
    const eventPositions = resources.createBuffer('events', sampleEvents(null));
    const nodePositions = resources.createBuffer('node-positions', roads.nodePositions);
    const csrOffsets = resources.createBuffer('offsets', offsets);
    const csrNeighbors = resources.createBuffer('neighbors', edges.targets);
    const csrWeights = resources.createBuffer('weights', edges.lengths);
    const reverseEdgeBuffer = resources.createBuffer('reverse-edges', reverseEdges);
    const segmentsBuffer = resources.createBuffer('segments', roads.segments);
    const segmentNodes = resources.createBuffer('segment-nodes', roads.segmentNodes);
    const originPosition = resources.createBuffer('origin-position', 8);

    let maxDistance = 800;
    let maxSnapDistance = 120;
    let activeSimulations = SIMULATION_COUNT;
    let seed = 1;
    const maxDistanceParameter = resources.createParameterBuffer(
      'max-distance',
      'float32',
      1,
      Float32Array.of(maxDistance)
    );
    const snapParameter = resources.createParameterBuffer(
      'max-snap-distance',
      'float32',
      1,
      Float32Array.of(maxSnapDistance)
    );
    const lengthParameter = resources.createParameterBuffer(
      'network-length',
      'float32',
      1,
      Float32Array.of(networkLength)
    );
    const kParameters = resources.createParameterBuffer(
      'k-parameters',
      'uint32',
      GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH
    );
    const patternCount = 1 + SIMULATION_COUNT;
    const kValues = resources.createBuffer('k-values', patternCount * BAND_COUNT * 4);
    const envelope = resources.createBuffer('envelope', 3 * BAND_COUNT * 4);
    const snappedCount = resources.createBuffer('snapped-count', 4);
    const kConverged = resources.createBuffer('k-converged', 4);
    const snapOverflow = resources.createBuffer('snap-overflow', 4);

    // ---- Graph 1: network K function ----
    const kGraph = new GPUCommandGraph<void>(device, {id: 'network-k'});
    kGraph.add(
      new GPUNetworkKFunction({
        id: 'k',
        points: importGraphBuffer(kGraph, 'events', eventPositions, 'float32x2', EVENT_COUNT),
        nodePositions: importGraphBuffer(
          kGraph,
          'node-positions',
          nodePositions,
          'float32x2',
          nodeCount
        ),
        offsets: importGraphBuffer(kGraph, 'offsets', csrOffsets, 'uint32', nodeCount + 1),
        neighbors: importGraphBuffer(kGraph, 'neighbors', csrNeighbors, 'uint32', edgeCount),
        weights: importGraphBuffer(kGraph, 'weights', csrWeights, 'float32', edgeCount),
        maxSnapDistance: snapParameter.importToGraph(kGraph),
        maxDistance: maxDistanceParameter.importToGraph(kGraph),
        networkLength: lengthParameter.importToGraph(kGraph),
        parameters: kParameters.importToGraph(kGraph),
        bandCount: BAND_COUNT,
        simulationCount: SIMULATION_COUNT,
        rowsPerBlock: ROWS_PER_BLOCK,
        maxIterations: K_MAXIMUM_ITERATIONS,
        localIterations: LOCAL_ITERATIONS,
        kValues: importGraphBuffer(
          kGraph,
          'k-values',
          kValues,
          'float32',
          patternCount * BAND_COUNT
        ),
        envelope: importGraphBuffer(kGraph, 'envelope', envelope, 'float32', 3 * BAND_COUNT),
        snappedEventCount: importGraphBuffer(kGraph, 'snapped-count', snappedCount, 'uint32', 1),
        candidateCapacity: SNAP_CANDIDATE_CAPACITY,
        overflow: importGraphBuffer(kGraph, 'snap-overflow', snapOverflow, 'uint32', 1),
        converged: importGraphBuffer(kGraph, 'k-converged', kConverged, 'uint32', 1)
      })
    );
    const kCompiled: CompiledGPUCommandGraph<void> = resources.track(kGraph.compile());

    // ---- Graph 2: reachability over the node graph and over the turn line graph ----
    let routeEnabled = false;
    let routeColor: RouteColor = 'turns';
    let routeBudget = 900;
    let angleCost = 40;
    let leftTurnCost = 30;
    let allowUTurns = false;
    let originNode = findNearestNode(roads.nodePositions, projection.project(...DEFAULT_ORIGIN));
    const turnParameters = resources.createParameterBuffer(
      'turn-parameters',
      'float32',
      GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH
    );
    const originParameter = resources.createParameterBuffer('origin-node', 'uint32', 1);
    const lineSources = resources.createParameterBuffer('line-sources', 'uint32', maximumDegree);
    const lineSourceCosts = resources.createParameterBuffer(
      'line-source-costs',
      'float32',
      maximumDegree
    );
    const budgetParameter = resources.createParameterBuffer('route-budget', 'float32', 1);
    const lineOffsets = resources.createBuffer('line-offsets', (edgeCount + 1) * 4);
    const lineNeighbors = resources.createBuffer('line-neighbors', arcCapacity * 4);
    const lineWeights = resources.createBuffer('line-weights', arcCapacity * 4);
    const arcCount = resources.createBuffer('arc-count', 4);
    const lineOverflow = resources.createBuffer('line-overflow', 4);
    const plainCosts = resources.createBuffer('plain-costs', nodeCount * 4);
    const lineCosts = resources.createBuffer('line-costs', edgeCount * 4);
    const turnNodeCosts = resources.createBuffer('turn-node-costs', nodeCount * 4);
    const extraCosts = resources.createBuffer('extra-costs', nodeCount * 4);
    const plainConverged = resources.createBuffer('plain-converged', 4);
    const lineConverged = resources.createBuffer('line-converged', 4);

    const routeGraph = new GPUCommandGraph<void>(device, {id: 'network-routing'});
    const routeView = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      id: string,
      buffer: typeof lineOffsets,
      format: Format,
      length: number
    ) => importGraphBuffer(routeGraph, id, buffer, format, length);
    const routeOffsets = routeView('offsets', csrOffsets, 'uint32', nodeCount + 1);
    const routeNeighbors = routeView('neighbors', csrNeighbors, 'uint32', edgeCount);
    const routeWeights = routeView('weights', csrWeights, 'float32', edgeCount);
    const routeLineOffsets = routeView('line-offsets', lineOffsets, 'uint32', edgeCount + 1);
    const routeLineNeighbors = routeView('line-neighbors', lineNeighbors, 'uint32', arcCapacity);
    const routeLineWeights = routeView('line-weights', lineWeights, 'float32', arcCapacity);
    const routeOrigin = originParameter.importToGraph(routeGraph);
    const routeBudgetView = budgetParameter.importToGraph(routeGraph);
    const routePlainCosts = routeView('plain-costs', plainCosts, 'float32', nodeCount);
    const routeLineCosts = routeView('line-costs', lineCosts, 'float32', edgeCount);
    const routeTurnNodeCosts = routeView('turn-node-costs', turnNodeCosts, 'float32', nodeCount);
    routeGraph.add(
      new GPUNetworkLineGraph({
        id: 'turns',
        offsets: routeOffsets,
        neighbors: routeNeighbors,
        weights: routeWeights,
        nodePositions: routeView('node-positions', nodePositions, 'float32x2', nodeCount),
        parameters: turnParameters.importToGraph(routeGraph),
        lineOffsets: routeLineOffsets,
        lineNeighbors: routeLineNeighbors,
        lineWeights: routeLineWeights,
        arcCount: routeView('arc-count', arcCount, 'uint32', 1),
        overflow: routeView('line-overflow', lineOverflow, 'uint32', 1)
      })
    );
    routeGraph.add(
      new GPUNetworkReachability({
        id: 'plain-route',
        offsets: routeOffsets,
        neighbors: routeNeighbors,
        weights: routeWeights,
        sources: routeOrigin,
        costLimit: routeBudgetView,
        maxIterations: ROUTE_MAXIMUM_ITERATIONS,
        localIterations: LOCAL_ITERATIONS,
        costs: routePlainCosts,
        converged: routeView('plain-converged', plainConverged, 'uint32', 1)
      })
    );
    routeGraph.add(
      new GPUNetworkReachability({
        id: 'line-route',
        offsets: routeLineOffsets,
        neighbors: routeLineNeighbors,
        weights: routeLineWeights,
        sources: lineSources.importToGraph(routeGraph),
        sourceCosts: lineSourceCosts.importToGraph(routeGraph),
        costLimit: routeBudgetView,
        maxIterations: ROUTE_MAXIMUM_ITERATIONS,
        localIterations: LOCAL_ITERATIONS,
        costs: routeLineCosts,
        converged: routeView('line-converged', lineConverged, 'uint32', 1)
      })
    );
    // Arrival cost of a node over the line graph: the cheapest line node of an edge entering it,
    // found through the reverse of each outgoing edge. The origin itself costs 0.
    addKernelPass(routeGraph, {
      id: 'turn-node-costs',
      bindings: [
        {name: 'offsets', view: routeOffsets, type: 'u32', access: 'read'},
        {
          name: 'reverseEdges',
          view: routeView('reverse-edges', reverseEdgeBuffer, 'uint32', edgeCount),
          type: 'u32',
          access: 'read'
        },
        {name: 'lineCosts', view: routeLineCosts, type: 'f32', access: 'read'},
        {name: 'origin', view: routeOrigin, type: 'u32', access: 'read'},
        {name: 'nodeCosts', view: routeTurnNodeCosts, type: 'f32', access: 'read_write'}
      ],
      invocationCount: nodeCount,
      declarations: `const EDGE_COUNT: u32 = ${edgeCount}u;
const NO_EDGE: u32 = ${NO_EDGE}u;`,
      body: `var bestBits = 0x7f800000u;
  let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + index]; edge < end; edge++) {
    let entering = reverseEdges[reverseEdgesOffset + edge];
    if (entering != NO_EDGE) {
      bestBits = min(bestBits, bitcast<u32>(lineCosts[lineCostsOffset + entering]));
    }
  }
  if (index == origin[originOffset]) {
    bestBits = 0u;
  }
  nodeCosts[nodeCostsOffset + index] = bitcast<f32>(bestBits);`
    });
    addKernelPass(routeGraph, {
      id: 'turn-extra-costs',
      bindings: [
        {name: 'turnCosts', view: routeTurnNodeCosts, type: 'f32', access: 'read'},
        {name: 'plainCosts', view: routePlainCosts, type: 'f32', access: 'read'},
        {
          name: 'extraCosts',
          view: routeView('extra-costs', extraCosts, 'float32', nodeCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: nodeCount,
      body: `let turnBits = bitcast<u32>(turnCosts[turnCostsOffset + index]);
  let plainBits = bitcast<u32>(plainCosts[plainCostsOffset + index]);
  var resultBits = 0x7f800000u;
  if (turnBits < 0x7f800000u && plainBits < 0x7f800000u) {
    resultBits = bitcast<u32>(max(turnCosts[turnCostsOffset + index] - plainCosts[plainCostsOffset + index], 0.0));
  }
  extraCosts[extraCostsOffset + index] = bitcast<f32>(resultBits);`
    });
    const routeCompiled: CompiledGPUCommandGraph<void> = resources.track(routeGraph.compile());

    // ---- Parameter writers ----
    let kDirty = true;
    let routeDirty = false;
    const writeKParameters = () => {
      kParameters.write(getGPUNetworkKFunctionParameterValues({seed, activeSimulations}));
      maxDistanceParameter.write(Float32Array.of(maxDistance));
      snapParameter.write(Float32Array.of(maxSnapDistance));
    };
    const writeTurnParameters = () => {
      turnParameters.write(
        getGPUNetworkLineGraphParameterValues({
          angleCost,
          leftTurnCost,
          uTurnCost: allowUTurns ? 120 : -1
        })
      );
      routeDirty = true;
    };
    const writeOrigin = (node: number) => {
      originNode = node;
      originParameter.write(Uint32Array.of(node));
      originPosition.write(roads.nodePositions.subarray(node * 2, node * 2 + 2));
      const first = offsets[node];
      const degree = offsets[node + 1] - first;
      const sources = new Uint32Array(maximumDegree).fill(GPU_NETWORK_REACHABILITY_NONE);
      const costs = new Float32Array(maximumDegree).fill(-1);
      for (let slot = 0; slot < degree; slot++) {
        sources[slot] = first + slot;
        costs[slot] = edges.lengths[first + slot];
      }
      lineSources.write(sources);
      lineSourceCosts.write(costs);
      routeDirty = true;
    };
    const writeBudget = () => {
      budgetParameter.write(Float32Array.of(routeBudget));
      routeDirty = true;
    };

    // ---- Controls ----
    const categoryOptions = [
      {value: 'all', label: 'All points of interest (sample)'},
      ...topCategories.map(category => ({
        value: String(category),
        label: pois.categoryNames[category]
      }))
    ];
    context.controls.addSelect<string>({
      label: 'Events (rewrites the event positions)',
      options: categoryOptions,
      value: 'all',
      onChange: value => {
        eventPositions.write(sampleEvents(value === 'all' ? null : Number(value)));
        kDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Maximum distance (per-frame cost limit)',
      min: 100,
      max: 1500,
      step: 50,
      value: maxDistance,
      format: value => `${value} m`,
      onChange: value => {
        maxDistance = value;
        kDirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Maximum snap distance',
      min: 20,
      max: 400,
      step: 10,
      value: maxSnapDistance,
      format: value => `${value} m`,
      onChange: value => {
        maxSnapDistance = value;
        kDirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Simulated patterns in the envelope',
      min: 0,
      max: SIMULATION_COUNT,
      step: 1,
      value: activeSimulations,
      onChange: value => {
        activeSimulations = value;
        kDirty = true;
      }
    });
    context.controls.addButton({
      label: 'Draw new random patterns',
      onClick: () => {
        seed += 1;
        kDirty = true;
      }
    });
    context.controls.addToggle({
      label: 'Route over the line graph (click the map for an origin)',
      value: routeEnabled,
      onChange: value => {
        routeEnabled = value;
        routeDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addSelect<RouteColor>({
      label: 'Roads colored by',
      options: [
        {value: 'turns', label: 'Travel cost with turn costs'},
        {value: 'plain', label: 'Plain travel cost (no turns)'},
        {value: 'extra', label: 'Extra cost caused by turns'}
      ],
      value: routeColor,
      onChange: value => {
        routeColor = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Turn penalty (meters per radian)',
      min: 0,
      max: 200,
      step: 5,
      value: angleCost,
      format: value => `${value} m/rad`,
      onChange: value => {
        angleCost = value;
        writeTurnParameters();
      }
    });
    context.controls.addSlider({
      label: 'Left turn extra',
      min: 0,
      max: 200,
      step: 5,
      value: leftTurnCost,
      format: value => `${value} m`,
      onChange: value => {
        leftTurnCost = value;
        writeTurnParameters();
      }
    });
    context.controls.addToggle({
      label: 'Allow U-turns (120 m) instead of banning them',
      value: allowUTurns,
      onChange: value => {
        allowUTurns = value;
        writeTurnParameters();
      }
    });
    context.controls.addSlider({
      label: 'Routing budget',
      min: 200,
      max: 1800,
      step: 100,
      value: routeBudget,
      format: value => `${value} m`,
      onChange: value => {
        routeBudget = value;
        writeBudget();
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Routing cost (m)',
      gradient: {
        colors: [
          [68, 1, 84],
          [59, 82, 139],
          [33, 145, 140],
          [94, 201, 98],
          [253, 231, 37]
        ],
        minimumLabel: '0',
        maximumLabel: 'budget'
      }
    });
    context.controls.addNote(
      'The K chart plots K(d) minus the mean of the random patterns; the grey lines are the ' +
        'lowest and highest simulated K. Observed above the upper line: events cluster along the ' +
        'streets more than random points would.'
    );
    context.controls.addReadout(
      'Network nodes / directed edges',
      `${formatCount(nodeCount)} / ${formatCount(edgeCount)}`
    );
    context.controls.addReadout('Network length', `${(networkLength / 1000).toFixed(1)} km`);
    const eventsReadout = context.controls.addReadout('Events snapped onto the network');
    const verdictReadout = context.controls.addReadout('Observed K at the maximum distance');
    const bandsReadout = context.controls.addReadout('Bands above / below the envelope');
    const convergedReadout = context.controls.addReadout('Searches converged');
    const lineReadout = context.controls.addReadout('Line graph arcs');
    context.controls.addReadout('Data', roads.attribution);

    const chart = new MiniChart('Network K(d) minus mean simulated K', 'm');
    chart.insertBefore(document.querySelector('[data-mode-readouts]'));

    // ---- Summary readback: K values, envelope, counts and flags in one small copy ----
    const kWords = patternCount * BAND_COUNT;
    const wordCount = kWords + 3 * BAND_COUNT + 7;
    const summary = new SummaryReader(
      resources,
      'network-k',
      [
        {buffer: kValues, size: kWords * 4},
        {buffer: envelope, size: 3 * BAND_COUNT * 4},
        {buffer: snappedCount, size: 4},
        {buffer: kConverged, size: 4},
        {buffer: plainConverged, size: 4},
        {buffer: lineConverged, size: 4},
        {buffer: arcCount, size: 4},
        {buffer: lineOverflow, size: 4},
        {buffer: snapOverflow, size: 4}
      ],
      bytes => {
        if (bytes.byteLength < wordCount * 4) return;
        const floats = new Float32Array(bytes);
        const words = new Uint32Array(bytes);
        const envelopeStart = kWords;
        const flagsStart = kWords + 3 * BAND_COUNT;
        const xs = Array.from(
          {length: BAND_COUNT},
          (_, band) => (maxDistance * band) / (BAND_COUNT - 1)
        );
        const observed = Array.from(floats.subarray(0, BAND_COUNT));
        const lower = Array.from(floats.subarray(envelopeStart, envelopeStart + BAND_COUNT));
        const mean = Array.from(
          floats.subarray(envelopeStart + BAND_COUNT, envelopeStart + 2 * BAND_COUNT)
        );
        const upper = Array.from(
          floats.subarray(envelopeStart + 2 * BAND_COUNT, envelopeStart + 3 * BAND_COUNT)
        );
        const hasEnvelope = activeSimulations > 0;
        let above = 0;
        let below = 0;
        for (let band = 1; band < BAND_COUNT; band++) {
          if (hasEnvelope && observed[band] > upper[band]) above++;
          if (hasEnvelope && observed[band] < lower[band]) below++;
        }
        const last = BAND_COUNT - 1;
        const reference = hasEnvelope ? mean : observed.map(() => 0);
        chart.update({
          series: [
            ...(hasEnvelope
              ? [
                  {
                    kind: 'line' as const,
                    x: xs,
                    y: lower.map((v, i) => v - reference[i]),
                    color: '#7f90ad'
                  },
                  {
                    kind: 'line' as const,
                    x: xs,
                    y: upper.map((v, i) => v - reference[i]),
                    color: '#7f90ad'
                  }
                ]
              : []),
            {
              kind: 'line' as const,
              x: xs,
              y: observed.map((v, i) => v - reference[i]),
              color: '#4ec9ff'
            }
          ],
          referenceLines: [{y: 0, color: '#4c5d7d', dashed: true}],
          xRange: [0, maxDistance],
          caption: hasEnvelope ? `${activeSimulations} simulations` : 'no simulations'
        });
        const snapped = words[flagsStart];
        eventsReadout.setValue(
          `${formatCount(snapped)} of ${EVENT_COUNT}${words[flagsStart + 6] ? ' (snap candidate overflow)' : ''}`
        );
        verdictReadout.setValue(
          hasEnvelope
            ? `${observed[last].toFixed(1)} (envelope ${lower[last].toFixed(1)} to ${upper[last].toFixed(1)})`
            : observed[last].toFixed(1)
        );
        bandsReadout.setValue(
          hasEnvelope ? `${above} above / ${below} below of ${BAND_COUNT - 1}` : 'no envelope'
        );
        convergedReadout.setValue(
          `K ${words[flagsStart + 1] ? 'yes' : 'no'}` +
            (routeEnabled
              ? `, node graph ${words[flagsStart + 2] ? 'yes' : 'no'}, line graph ${words[flagsStart + 3] ? 'yes' : 'no'}`
              : '')
        );
        lineReadout.setValue(
          routeEnabled
            ? `${formatCount(words[flagsStart + 4])} for ${formatCount(edgeCount)} nodes${words[flagsStart + 5] ? ' (capacity overflow)' : ''}`
            : 'routing off'
        );
      }
    );

    writeKParameters();
    writeTurnParameters();
    writeBudget();
    writeOrigin(originNode);
    routeDirty = false;

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [kCompiled, routeCompiled],
      encode(commandEncoder) {
        let encoded = false;
        if (kDirty) {
          writeKParameters();
          kCompiled.encode(commandEncoder, {parameters: undefined});
          kDirty = false;
          encoded = true;
        }
        if (routeEnabled && routeDirty) {
          routeCompiled.encode(commandEncoder, {parameters: undefined});
          routeDirty = false;
          encoded = true;
        }
        if (encoded) summary.request(commandEncoder);
        else summary.flush(commandEncoder);
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [roads.origin[0], roads.origin[1], 0];
        const costBuffer =
          routeColor === 'turns' ? turnNodeCosts : routeColor === 'plain' ? plainCosts : extraCosts;
        const layers: Layer[] = [
          new SpatialAnalysisSegmentLayer({
            id: 'network-k-roads',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: routeEnabled ? 2.5 : 1.5,
            ...(routeEnabled
              ? {
                  values: costBuffer,
                  valueFormat: 'float32' as const,
                  valueIndices: segmentNodes,
                  colormap: routeColor === 'extra' ? ('inferno' as const) : ('viridis' as const),
                  valueRange: [0, routeColor === 'extra' ? routeBudget * 0.3 : routeBudget] as [
                    number,
                    number
                  ],
                  noDataColor: [90, 92, 105, 90] as const
                }
              : {colormap: 'uniform' as const, color: [110, 125, 160, 170] as const})
          }),
          new SpatialAnalysisPointLayer({
            id: 'network-k-events',
            coordinateOrigin,
            positions: eventPositions,
            instanceCount: EVENT_COUNT,
            radiusPixels: 4,
            color: [255, 214, 90, 255]
          })
        ];
        if (routeEnabled) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'network-k-origin',
              coordinateOrigin,
              positions: originPosition,
              instanceCount: 1,
              radiusPixels: 8,
              color: [255, 255, 255, 255]
            })
          );
        }
        return layers;
      },
      onClick(event) {
        if (!routeEnabled || !event.coordinate) return false;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        writeOrigin(findNearestNode(roads.nodePositions, [x, y]));
        return true;
      },
      destroy() {
        summary.stop();
        chart.destroy();
        resources.destroy();
      }
    };
    return instance;
  }
};
