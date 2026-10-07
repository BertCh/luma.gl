// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUNetworkLineGraphParameterValues,
  GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH,
  GPU_NETWORK_REACHABILITY_NONE,
  GPUNetworkLineGraph,
  GPUNetworkNeighborhood,
  GPUNetworkPathExtraction,
  GPUNetworkReachability
} from '@luma.gl/experimental/gpu-network';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {buildRoadNetwork, getTurnAngle, NO_EDGE, writeDriveCosts} from './b9-road-network';
import {
  CHICAGO_PLACES,
  formatInteger,
  formatMinutes,
  getRoadColors,
  sliceSections
} from './b9-shared';

/** Option state of the routing scene. */
export type RoutingOptions = {
  originPlace: string;
  destinationPlace: string;
  destinationCount: number;
  costLimitMinutes: number;
  closeExpressways: boolean;
  expresswaySlowdown: number;
  intersectionDelay: number;
  base: 'time' | 'hops' | 'none';
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  showRoute: boolean;
  showTurnRoute: boolean;
  angleCost: number;
  leftTurnCost: number;
  rightTurnCost: number;
  uTurns: 'banned' | 'allowed';
  straightAngle: number;
  banLeftTurns: 'none' | 'arterials';
  hops: number;
  localIterations: string;
};

/** Places a route can start from or end at. */
export const ROUTING_PLACES: Record<
  string,
  {label: string; coordinate: readonly [number, number]}
> = {
  willis: {label: 'Willis Tower (Loop)', coordinate: CHICAGO_PLACES.willisTower},
  ohare: {label: "O'Hare Airport", coordinate: CHICAGO_PLACES.ohare},
  midway: {label: 'Midway Airport', coordinate: CHICAGO_PLACES.midway},
  wrigley: {label: 'Wrigley Field', coordinate: CHICAGO_PLACES.wrigleyField},
  soldier: {label: 'Soldier Field', coordinate: CHICAGO_PLACES.soldierField},
  unitedCenter: {label: 'United Center', coordinate: CHICAGO_PLACES.unitedCenter},
  msi: {label: 'Museum of Science and Industry', coordinate: CHICAGO_PLACES.scienceIndustry},
  southShore: {label: 'South Shore', coordinate: CHICAGO_PLACES.southShore},
  austin: {label: 'Austin (West Side)', coordinate: CHICAGO_PLACES.austin}
};

/** Extra destinations after the first one (extra routes from the same tree). */
const EXTRA_DESTINATIONS = ['midway', 'wrigley', 'soldier', 'msi', 'unitedCenter'] as const;

const MAXIMUM_DESTINATIONS = 6;
const MAXIMUM_PATH_LENGTH = 2048;
const MAXIMUM_ROUNDS = 80;
const MAXIMUM_HOPS = 24;
const HOOD_NODE_CAPACITY = 8192;
const HOOD_EDGE_CAPACITY = 16384;
const BAN_CAPACITY = 8192;
const MAXIMUM_SOURCES = 16;
const ROUTE_COLOR = [255, 96, 64, 255] as const;
const TURN_ROUTE_COLOR = [40, 190, 255, 255] as const;
const DESTINATION_PALETTE: readonly (readonly [number, number, number, number])[] = [
  [255, 70, 200, 255],
  [255, 148, 72, 255],
  [245, 220, 87, 255],
  [87, 235, 168, 255],
  [189, 122, 255, 255],
  [107, 158, 255, 255]
];

type TreeGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  localIterations: number;
};

/**
 * Routing on the Chicago drive network. Three compiled graphs share one CSR whose weights are
 * rewritten from the options:
 *
 * - tree: `GPUNetworkReachability` (cost and predecessor tree from the origin) then
 *   `GPUNetworkPathExtraction` (routes to up to six destinations) and two tiny kernels that
 *   scatter the route edges into a per-edge flag buffer that the road layer reads;
 * - hood: `GPUNetworkNeighborhood` (k-hop ego network around the origin);
 * - turns: `GPUNetworkLineGraph` (edge-based graph with turn costs and bans), then reachability
 *   and path extraction again on that line graph.
 *
 * Moving the origin, destination, costs, turn costs or hop count writes buffers and re-encodes the
 * affected graph; only the solver `localIterations` option rebuilds graphs.
 */
export async function createRouting(
  ctx: SceneContext<RoutingOptions>
): Promise<SceneInstance<RoutingOptions>> {
  const {device} = ctx;
  const network = buildRoadNetwork(ctx.datasets.get('chicago-roads'));
  const {nodeCount, edgeCount} = network;
  const resources = new SpatialAnalysisResources(device, 'routing');
  const segmentCount = network.segmentEdges.length;

  // ---- Static buffers ---------------------------------------------------------------------
  const offsetsBuffer = resources.createBuffer('offsets', network.offsets);
  const neighborsBuffer = resources.createBuffer('neighbors', network.targets);
  const weightsBuffer = resources.createBuffer('weights', edgeCount * 4);
  const nodePositionsBuffer = resources.createBuffer('node-positions', network.nodePositions);
  const segmentsBuffer = resources.createBuffer('segments', network.segments);
  const majorSegmentsBuffer = resources.createBuffer('major-segments', network.majorSegments);
  const segmentEdgesBuffer = resources.createBuffer('segment-edges', network.segmentEdges);
  const segmentTargetsBuffer = resources.createBuffer(
    'segment-targets',
    network.segmentTargetNodes
  );

  // Reverse CSR: edges entering each node, for the arrival edge of a destination.
  const inOffsets = new Uint32Array(nodeCount + 1);
  for (let edge = 0; edge < edgeCount; edge++) inOffsets[network.targets[edge] + 1]++;
  for (let node = 0; node < nodeCount; node++) inOffsets[node + 1] += inOffsets[node];
  const inEdges = new Uint32Array(edgeCount);
  {
    const cursor = inOffsets.slice(0, nodeCount);
    for (let edge = 0; edge < edgeCount; edge++) inEdges[cursor[network.targets[edge]]++] = edge;
  }
  const inOffsetsBuffer = resources.createBuffer('in-offsets', inOffsets);
  const inEdgesBuffer = resources.createBuffer('in-edges', inEdges);

  // Line graph capacity: one arc per possible turn.
  let arcCapacity = 0;
  let maximumDegree = 1;
  for (let edge = 0; edge < edgeCount; edge++) {
    const head = network.targets[edge];
    arcCapacity += network.offsets[head + 1] - network.offsets[head];
  }
  for (let node = 0; node < nodeCount; node++) {
    maximumDegree = Math.max(maximumDegree, network.offsets[node + 1] - network.offsets[node]);
  }
  maximumDegree = Math.min(maximumDegree, MAXIMUM_SOURCES);
  arcCapacity = Math.max(arcCapacity, 1);

  // Left turns between two primary roads: the "no left turn" candidates.
  const bannedPairs: number[] = [];
  for (let node = 0; node < nodeCount && bannedPairs.length < BAN_CAPACITY * 2; node++) {
    for (let incoming = network.offsets[node]; incoming < network.offsets[node + 1]; incoming++) {
      // Edges leaving `node` are followed by turns at their target; iterate those instead.
      const head = network.targets[incoming];
      if (network.roadClass[incoming] < 2 || network.roadClass[incoming] > 3) continue;
      for (let outgoing = network.offsets[head]; outgoing < network.offsets[head + 1]; outgoing++) {
        if (
          network.roadClass[outgoing] < 2 ||
          network.roadClass[outgoing] > 3 ||
          network.targets[outgoing] === node
        )
          continue;
        const angle = getTurnAngle(network, incoming, outgoing);
        if (angle > 0.5 && angle < 2.9 && bannedPairs.length < BAN_CAPACITY * 2) {
          bannedPairs.push(incoming, outgoing);
        }
      }
    }
  }
  const bannedTurnTotal = bannedPairs.length / 2;
  const bannedTurnsBuffer = resources.createBuffer('banned-turns', BAN_CAPACITY * 2 * 4);
  bannedTurnsBuffer.write(Uint32Array.from(bannedPairs));

  // ---- Parameter buffers ------------------------------------------------------------------
  const originParameter = resources.createParameterBuffer('origin', 'uint32', 1);
  const destinationParameter = resources.createParameterBuffer(
    'destinations',
    'uint32',
    MAXIMUM_DESTINATIONS
  );
  const destinationCountParameter = resources.createParameterBuffer(
    'destination-count',
    'uint32',
    1
  );
  const costLimitParameter = resources.createParameterBuffer('cost-limit', 'float32', 1);
  const hopsParameter = resources.createParameterBuffer('hops', 'uint32', 1);
  const turnParameters = resources.createParameterBuffer(
    'turn-parameters',
    'float32',
    GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH
  );
  const lineSources = resources.createParameterBuffer('line-sources', 'uint32', maximumDegree);
  const lineSourceCosts = resources.createParameterBuffer(
    'line-source-costs',
    'float32',
    maximumDegree
  );
  const markerBuffer = resources.createBuffer('markers', (1 + MAXIMUM_DESTINATIONS) * 8);
  const markerIds = resources.createBuffer(
    'marker-ids',
    Uint32Array.from({length: 1 + MAXIMUM_DESTINATIONS}, (_, index) => index)
  );

  // ---- Output buffers ---------------------------------------------------------------------
  const treeCosts = resources.createBuffer('tree-costs', nodeCount * 4);
  const treePredecessors = resources.createBuffer('tree-predecessors', nodeCount * 4);
  const treeConverged = resources.createBuffer('tree-converged', 4);
  const treeIterations = resources.createBuffer('tree-iterations', 4);
  const treeUnresolved = resources.createBuffer('tree-unresolved', 4);
  const pathCapacity = MAXIMUM_DESTINATIONS * MAXIMUM_PATH_LENGTH;
  const pathNodeIds = resources.createBuffer('path-node-ids', pathCapacity * 4);
  const pathNodeCount = resources.createBuffer('path-node-count', 4);
  const pathNodeOverflow = resources.createBuffer('path-node-overflow', 4);
  const pathEdgeIds = resources.createBuffer('path-edge-ids', pathCapacity * 4);
  const pathEdgeCount = resources.createBuffer('path-edge-count', 4);
  const pathEdgeOverflow = resources.createBuffer('path-edge-overflow', 4);
  const pathOffsets = resources.createBuffer('path-offsets', (MAXIMUM_DESTINATIONS + 1) * 4);
  const pathCosts = resources.createBuffer('path-costs', MAXIMUM_DESTINATIONS * 4);
  const pathFound = resources.createBuffer('path-found', MAXIMUM_DESTINATIONS * 4);
  const routeFlags = resources.createBuffer('route-flags', edgeCount * 4);

  const hopDistances = resources.createBuffer('hop-distances', nodeCount * 4);
  const hoodEdgeMask = resources.createBuffer('hood-edge-mask', edgeCount * 4);
  const hoodNodeIds = resources.createBuffer('hood-node-ids', HOOD_NODE_CAPACITY * 4);
  const hoodNodeCount = resources.createBuffer('hood-node-count', 4);
  const hoodNodeOverflow = resources.createBuffer('hood-node-overflow', 4);
  const hoodEdgeIds = resources.createBuffer('hood-edge-ids', HOOD_EDGE_CAPACITY * 4);
  const hoodEdgeCount = resources.createBuffer('hood-edge-count', 4);
  const hoodEdgeOverflow = resources.createBuffer('hood-edge-overflow', 4);

  const lineOffsetsBuffer = resources.createBuffer('line-offsets', (edgeCount + 1) * 4);
  const lineNeighborsBuffer = resources.createBuffer('line-neighbors', arcCapacity * 4);
  const lineWeightsBuffer = resources.createBuffer('line-weights', arcCapacity * 4);
  const lineArcCount = resources.createBuffer('line-arc-count', 4);
  const lineOverflow = resources.createBuffer('line-overflow', 4);
  const lineCosts = resources.createBuffer('line-costs', edgeCount * 4);
  const linePredecessors = resources.createBuffer('line-predecessors', edgeCount * 4);
  const lineConverged = resources.createBuffer('line-converged', 4);
  const lineIterations = resources.createBuffer('line-iterations', 4);
  const lineTargets = resources.createBuffer('line-targets', MAXIMUM_DESTINATIONS * 4);
  const linePathNodeIds = resources.createBuffer('line-path-node-ids', pathCapacity * 4);
  const linePathNodeCount = resources.createBuffer('line-path-node-count', 4);
  const linePathNodeOverflow = resources.createBuffer('line-path-node-overflow', 4);
  const linePathOffsets = resources.createBuffer(
    'line-path-offsets',
    (MAXIMUM_DESTINATIONS + 1) * 4
  );
  const linePathCosts = resources.createBuffer('line-path-costs', MAXIMUM_DESTINATIONS * 4);
  const linePathFound = resources.createBuffer('line-path-found', MAXIMUM_DESTINATIONS * 4);
  const turnFlags = resources.createBuffer('turn-flags', edgeCount * 4);

  // ---- State -------------------------------------------------------------------------------
  const resolvePlace = (place: string): number => {
    const coordinate = (ROUTING_PLACES[place] ?? ROUTING_PLACES.willis).coordinate;
    const [x, y] = network.projection.project(coordinate[0], coordinate[1]);
    return network.findNearestNode(x, y);
  };
  let originNode = resolvePlace(ctx.options.originPlace);
  const destinationNodes: number[] = [];
  const writeDestinationPlaces = () => {
    destinationNodes.length = 0;
    destinationNodes.push(resolvePlace(ctx.options.destinationPlace));
    for (const place of EXTRA_DESTINATIONS) destinationNodes.push(resolvePlace(place));
    // Do not let an extra destination duplicate the first.
    destinationNodes.length = MAXIMUM_DESTINATIONS;
  };
  writeDestinationPlaces();

  const costs = new Float32Array(edgeCount);
  let destroyed = false;
  let graphs: {
    tree: TreeGraph;
    hood: CompiledGPUCommandGraph<void>;
    turns: CompiledGPUCommandGraph<void>;
  } | null = null;
  let treeDirty = 2;
  let hoodDirty = 2;
  let turnDirty = 2;
  let cpuCosts: Float32Array | null = null;
  let cpuHops: Uint32Array | null = null;
  const summary = {
    treeFound: new Uint32Array(MAXIMUM_DESTINATIONS),
    treeCost: new Float32Array(MAXIMUM_DESTINATIONS),
    turnFound: new Uint32Array(MAXIMUM_DESTINATIONS),
    turnCost: new Float32Array(MAXIMUM_DESTINATIONS),
    treeEdges: new Uint32Array(0),
    turnEdges: new Uint32Array(0),
    treeOffsets: new Uint32Array(MAXIMUM_DESTINATIONS + 1),
    turnOffsets: new Uint32Array(MAXIMUM_DESTINATIONS + 1)
  };

  const writeCosts = () => {
    const options = ctx.options;
    writeDriveCosts(
      network,
      {
        closeExpressways: options.closeExpressways,
        expresswaySlowdown: options.expresswaySlowdown,
        intersectionDelay: options.intersectionDelay
      },
      costs
    );
    weightsBuffer.write(costs);
    ctx.setReadout('expressways', options.closeExpressways ? 'closed' : 'open');
  };

  const writeOriginAndMarkers = () => {
    originParameter.write(Uint32Array.of(originNode));
    const markers = new Float32Array((1 + MAXIMUM_DESTINATIONS) * 2);
    markers.set(network.nodePositions.subarray(originNode * 2, originNode * 2 + 2), 0);
    destinationNodes.forEach((node, index) => {
      markers.set(network.nodePositions.subarray(node * 2, node * 2 + 2), (1 + index) * 2);
    });
    markerBuffer.write(markers);
    destinationParameter.write(Uint32Array.from(destinationNodes));
    // Line graph sources: every out-edge of the origin at its own cost.
    const sources = new Uint32Array(maximumDegree).fill(GPU_NETWORK_REACHABILITY_NONE);
    const sourceCosts = new Float32Array(maximumDegree).fill(-1);
    let slot = 0;
    for (
      let edge = network.offsets[originNode];
      edge < network.offsets[originNode + 1] && slot < maximumDegree;
      edge++
    ) {
      if (costs[edge] >= 0) {
        sources[slot] = edge;
        sourceCosts[slot] = costs[edge];
        slot++;
      }
    }
    lineSources.write(sources);
    lineSourceCosts.write(sourceCosts);
  };

  const writeTurnParameters = () => {
    const options = ctx.options;
    turnParameters.write(
      getGPUNetworkLineGraphParameterValues({
        angleCost: options.angleCost,
        leftTurnCost: options.leftTurnCost,
        rightTurnCost: options.rightTurnCost,
        uTurnCost: options.uTurns === 'banned' ? -1 : 90,
        straightAngle: options.straightAngle,
        bannedTurnCount:
          options.banLeftTurns === 'arterials' ? Math.min(bannedTurnTotal, BAN_CAPACITY) : 0
      })
    );
  };

  const writeScalarParameters = () => {
    const options = ctx.options;
    costLimitParameter.write(Float32Array.of(options.costLimitMinutes * 60));
    hopsParameter.write(Uint32Array.of(options.hops));
    destinationCountParameter.write(Uint32Array.of(options.destinationCount));
  };

  // ---- Graphs ------------------------------------------------------------------------------
  function buildGraphs(localIterations: number): void {
    if (graphs) {
      resources.release(graphs.tree.compiled);
      resources.release(graphs.hood);
      resources.release(graphs.turns);
    }

    const importer = (graph: GPUCommandGraph<void>) => {
      const cache = new Map<Buffer, GraphDataView>();
      return <Format extends 'uint32' | 'float32' | 'float32x2'>(
        buffer: Buffer,
        format: Format,
        length: number
      ) => {
        let view = cache.get(buffer);
        if (!view) {
          view = importGraphBuffer(graph, buffer.id, buffer, format, length) as GraphDataView;
          cache.set(buffer, view);
        }
        return view as unknown as GraphDataView<Format>;
      };
    };

    // -- tree graph
    const treeGraph = new GPUCommandGraph<void>(device, {id: 'routing-tree'});
    {
      const view = importer(treeGraph);
      const offsets = view(offsetsBuffer, 'uint32', nodeCount + 1);
      const neighbors = view(neighborsBuffer, 'uint32', edgeCount);
      const weights = view(weightsBuffer, 'float32', edgeCount);
      const costsView = view(treeCosts, 'float32', nodeCount);
      const predecessors = view(treePredecessors, 'uint32', nodeCount);
      treeGraph.add(
        new GPUNetworkReachability({
          id: 'tree',
          offsets,
          neighbors,
          weights,
          sources: originParameter.importToGraph(treeGraph),
          costLimit: costLimitParameter.importToGraph(treeGraph),
          maxIterations: MAXIMUM_ROUNDS,
          localIterations,
          costs: costsView,
          predecessors,
          converged: view(treeConverged, 'uint32', 1),
          iterationCount: view(treeIterations, 'uint32', 1),
          unresolvedCount: view(treeUnresolved, 'uint32', 1)
        })
      );
      const edgeIds = view(pathEdgeIds, 'uint32', pathCapacity);
      const edgeCountView = view(pathEdgeCount, 'uint32', 1);
      treeGraph.add(
        new GPUNetworkPathExtraction({
          id: 'route',
          predecessors,
          costs: costsView,
          targets: destinationParameter.importToGraph(treeGraph),
          targetCount: destinationCountParameter.importToGraph(treeGraph),
          maxPathLength: MAXIMUM_PATH_LENGTH,
          output: {
            ids: view(pathNodeIds, 'uint32', pathCapacity),
            count: view(pathNodeCount, 'uint32', 1),
            overflow: view(pathNodeOverflow, 'uint32', 1)
          },
          pathOffsets: view(pathOffsets, 'uint32', MAXIMUM_DESTINATIONS + 1),
          pathCosts: view(pathCosts, 'float32', MAXIMUM_DESTINATIONS),
          pathFound: view(pathFound, 'uint32', MAXIMUM_DESTINATIONS),
          edges: {
            offsets,
            neighbors,
            weights,
            output: {
              ids: edgeIds,
              count: edgeCountView,
              overflow: view(pathEdgeOverflow, 'uint32', 1)
            }
          }
        })
      );
      addFlagKernels(
        treeGraph,
        'tree-flags',
        view(routeFlags, 'uint32', edgeCount),
        edgeIds,
        edgeCountView
      );
    }

    // -- neighborhood graph
    const hoodGraph = new GPUCommandGraph<void>(device, {id: 'routing-hood'});
    {
      const view = importer(hoodGraph);
      hoodGraph.add(
        new GPUNetworkNeighborhood({
          id: 'hood',
          offsets: view(offsetsBuffer, 'uint32', nodeCount + 1),
          neighbors: view(neighborsBuffer, 'uint32', edgeCount),
          seeds: originParameter.importToGraph(hoodGraph),
          hops: hopsParameter.importToGraph(hoodGraph),
          maxHops: MAXIMUM_HOPS,
          hopDistances: view(hopDistances, 'uint32', nodeCount),
          edgeMask: view(hoodEdgeMask, 'uint32', edgeCount),
          nodes: {
            ids: view(hoodNodeIds, 'uint32', HOOD_NODE_CAPACITY),
            count: view(hoodNodeCount, 'uint32', 1),
            overflow: view(hoodNodeOverflow, 'uint32', 1)
          },
          edges: {
            ids: view(hoodEdgeIds, 'uint32', HOOD_EDGE_CAPACITY),
            count: view(hoodEdgeCount, 'uint32', 1),
            overflow: view(hoodEdgeOverflow, 'uint32', 1)
          }
        })
      );
    }

    // -- turn-aware graph
    const turnGraph = new GPUCommandGraph<void>(device, {id: 'routing-turns'});
    {
      const view = importer(turnGraph);
      const offsets = view(offsetsBuffer, 'uint32', nodeCount + 1);
      const lineOffsets = view(lineOffsetsBuffer, 'uint32', edgeCount + 1);
      const lineNeighbors = view(lineNeighborsBuffer, 'uint32', arcCapacity);
      const lineWeights = view(lineWeightsBuffer, 'float32', arcCapacity);
      turnGraph.add(
        new GPUNetworkLineGraph({
          id: 'turns',
          offsets,
          neighbors: view(neighborsBuffer, 'uint32', edgeCount),
          weights: view(weightsBuffer, 'float32', edgeCount),
          nodePositions: view(nodePositionsBuffer, 'float32x2', nodeCount),
          parameters: turnParameters.importToGraph(turnGraph),
          bannedTurns: view(bannedTurnsBuffer, 'uint32', BAN_CAPACITY * 2),
          lineOffsets,
          lineNeighbors,
          lineWeights,
          arcCount: view(lineArcCount, 'uint32', 1),
          overflow: view(lineOverflow, 'uint32', 1)
        })
      );
      const lineCostsView = view(lineCosts, 'float32', edgeCount);
      const linePredecessorsView = view(linePredecessors, 'uint32', edgeCount);
      turnGraph.add(
        new GPUNetworkReachability({
          id: 'line-tree',
          offsets: lineOffsets,
          neighbors: lineNeighbors,
          weights: lineWeights,
          sources: lineSources.importToGraph(turnGraph),
          sourceCosts: lineSourceCosts.importToGraph(turnGraph),
          costLimit: costLimitParameter.importToGraph(turnGraph),
          maxIterations: MAXIMUM_ROUNDS,
          localIterations,
          costs: lineCostsView,
          predecessors: linePredecessorsView,
          converged: view(lineConverged, 'uint32', 1),
          iterationCount: view(lineIterations, 'uint32', 1)
        })
      );
      // The arrival edge of each destination: the cheapest edge entering its node.
      const destinationsView = destinationParameter.importToGraph(turnGraph);
      const targetsView = view(lineTargets, 'uint32', MAXIMUM_DESTINATIONS);
      addKernelPass(turnGraph, {
        id: 'arrival-edges',
        bindings: [
          {name: 'destinations', view: destinationsView, type: 'u32', access: 'read'},
          {name: 'lineCosts', view: lineCostsView, type: 'f32', access: 'read'},
          {
            name: 'inOffsets',
            view: view(inOffsetsBuffer, 'uint32', nodeCount + 1),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'inEdges',
            view: view(inEdgesBuffer, 'uint32', edgeCount),
            type: 'u32',
            access: 'read'
          },
          {name: 'targets', view: targetsView, type: 'u32', access: 'read_write'}
        ],
        invocationCount: MAXIMUM_DESTINATIONS,
        declarations: `const NODE_COUNT: u32 = ${nodeCount}u;`,
        body: `let node = destinations[destinationsOffset + index];
  var bestBits = 0x7f800000u;
  var bestEdge = 0xffffffffu;
  if (node < NODE_COUNT) {
    for (var slot = inOffsets[inOffsetsOffset + node]; slot < inOffsets[inOffsetsOffset + node + 1u]; slot++) {
      let edge = inEdges[inEdgesOffset + slot];
      let bits = bitcast<u32>(lineCosts[lineCostsOffset + edge]);
      if (bits < bestBits) {
        bestBits = bits;
        bestEdge = edge;
      }
    }
  }
  targets[targetsOffset + index] = bestEdge;`
      });
      const lineIds = view(linePathNodeIds, 'uint32', pathCapacity);
      const lineCount = view(linePathNodeCount, 'uint32', 1);
      turnGraph.add(
        new GPUNetworkPathExtraction({
          id: 'line-route',
          predecessors: linePredecessorsView,
          costs: lineCostsView,
          targets: targetsView,
          targetCount: destinationCountParameter.importToGraph(turnGraph),
          maxPathLength: MAXIMUM_PATH_LENGTH,
          output: {
            ids: lineIds,
            count: lineCount,
            overflow: view(linePathNodeOverflow, 'uint32', 1)
          },
          pathOffsets: view(linePathOffsets, 'uint32', MAXIMUM_DESTINATIONS + 1),
          pathCosts: view(linePathCosts, 'float32', MAXIMUM_DESTINATIONS),
          pathFound: view(linePathFound, 'uint32', MAXIMUM_DESTINATIONS)
        })
      );
      // Line nodes are edge rows, so the extracted nodes are the route edges.
      addFlagKernels(
        turnGraph,
        'turn-flags',
        view(turnFlags, 'uint32', edgeCount),
        lineIds,
        lineCount
      );
    }

    graphs = {
      tree: {compiled: resources.track(treeGraph.compile()), localIterations},
      hood: resources.track(hoodGraph.compile()),
      turns: resources.track(turnGraph.compile())
    };
    treeDirty = 2;
    hoodDirty = 2;
    turnDirty = 2;
    ctx.setReadout('solver', `${MAXIMUM_ROUNDS} rounds x ${localIterations} hops`);
  }

  /** Clears an edge flag buffer and then sets the flag of every extracted route edge. */
  function addFlagKernels(
    graph: GPUCommandGraph<void>,
    id: string,
    flags: GraphDataView<'uint32'>,
    ids: GraphDataView<'uint32'>,
    count: GraphDataView<'uint32'>
  ): void {
    addKernelPass(graph, {
      id: `${id}-clear`,
      bindings: [{name: 'flags', view: flags, type: 'u32', access: 'read_write'}],
      invocationCount: edgeCount,
      body: 'flags[flagsOffset + index] = 0u;'
    });
    addKernelPass(graph, {
      id: `${id}-set`,
      bindings: [
        {name: 'ids', view: ids, type: 'u32', access: 'read'},
        {name: 'count', view: count, type: 'u32', access: 'read'},
        {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
      ],
      invocationCount: pathCapacity,
      declarations: `const EDGE_COUNT: u32 = ${edgeCount}u;`,
      body: `if (index < count[countOffset]) {
    let edge = ids[idsOffset + index];
    if (edge < EDGE_COUNT) {
      flags[flagsOffset + edge] = 1u;
    }
  }`
    });
  }

  // ---- Readbacks ---------------------------------------------------------------------------
  const treeSizes = [
    MAXIMUM_DESTINATIONS * 4, // found
    MAXIMUM_DESTINATIONS * 4, // costs
    (MAXIMUM_DESTINATIONS + 1) * 4, // offsets
    4, // node count
    4, // node overflow
    4, // edge count
    4, // edge overflow
    4, // converged
    4, // iterations
    4, // unresolved
    MAXIMUM_PATH_LENGTH * 4, // first route's edges
    nodeCount * 4 // costs
  ];
  const treeReader = new SummaryReader(
    resources,
    'routing-tree',
    [
      {buffer: pathFound, size: treeSizes[0]},
      {buffer: pathCosts, size: treeSizes[1]},
      {buffer: pathOffsets, size: treeSizes[2]},
      {buffer: pathNodeCount, size: 4},
      {buffer: pathNodeOverflow, size: 4},
      {buffer: pathEdgeCount, size: 4},
      {buffer: pathEdgeOverflow, size: 4},
      {buffer: treeConverged, size: 4},
      {buffer: treeIterations, size: 4},
      {buffer: treeUnresolved, size: 4},
      {buffer: pathEdgeIds, size: treeSizes[10]},
      {buffer: treeCosts, size: treeSizes[11]}
    ],
    bytes => {
      if (destroyed) return;
      const [
        found,
        pathCostBytes,
        offsets,
        nodeTotal,
        nodeOver,
        edgeTotal,
        edgeOver,
        converged,
        iterations,
        unresolved,
        edges,
        costBytes
      ] = sliceSections(bytes, treeSizes);
      summary.treeFound = new Uint32Array(found);
      summary.treeCost = new Float32Array(pathCostBytes);
      summary.treeOffsets = new Uint32Array(offsets);
      summary.treeEdges = new Uint32Array(edges);
      cpuCosts = new Float32Array(costBytes);
      let reached = 0;
      for (const cost of cpuCosts) if (Number.isFinite(cost)) reached++;
      const solver = new Uint32Array(converged)[0];
      ctx.setReadout('reached', `${formatInteger(reached)} of ${formatInteger(nodeCount)}`);
      ctx.setReadout(
        'solver',
        `${solver ? 'converged' : 'NOT converged'} in ${new Uint32Array(iterations)[0]} of ${MAXIMUM_ROUNDS} rounds` +
          (new Uint32Array(unresolved)[0] ? `, ${new Uint32Array(unresolved)[0]} unresolved` : '')
      );
      const nodes = new Uint32Array(nodeTotal)[0];
      const edgesTotal = new Uint32Array(edgeTotal)[0];
      ctx.setReadout(
        'routeSize',
        `${formatInteger(nodes)} nodes / ${formatInteger(edgesTotal)} edges` +
          (new Uint32Array(nodeOver)[0] || new Uint32Array(edgeOver)[0] ? ' (overflow)' : '')
      );
      updateRouteReadouts();
    }
  );

  const turnSizes = [
    MAXIMUM_DESTINATIONS * 4,
    MAXIMUM_DESTINATIONS * 4,
    (MAXIMUM_DESTINATIONS + 1) * 4,
    MAXIMUM_PATH_LENGTH * 4,
    4,
    4,
    4,
    4
  ];
  const turnReader = new SummaryReader(
    resources,
    'routing-turns',
    [
      {buffer: linePathFound, size: turnSizes[0]},
      {buffer: linePathCosts, size: turnSizes[1]},
      {buffer: linePathOffsets, size: turnSizes[2]},
      {buffer: linePathNodeIds, size: turnSizes[3]},
      {buffer: lineArcCount, size: 4},
      {buffer: lineOverflow, size: 4},
      {buffer: lineConverged, size: 4},
      {buffer: lineIterations, size: 4}
    ],
    bytes => {
      if (destroyed || !ctx.options.showTurnRoute) return;
      const [found, pathCostBytes, offsets, edges, arcs, overflow, converged] = sliceSections(
        bytes,
        turnSizes
      );
      summary.turnFound = new Uint32Array(found);
      summary.turnCost = new Float32Array(pathCostBytes);
      summary.turnOffsets = new Uint32Array(offsets);
      summary.turnEdges = new Uint32Array(edges);
      ctx.setReadout(
        'arcs',
        `${formatInteger(new Uint32Array(arcs)[0])} turns of ${formatInteger(arcCapacity)}` +
          (new Uint32Array(overflow)[0] ? ' (overflow)' : '') +
          (new Uint32Array(converged)[0] ? '' : ' (not converged)')
      );
      updateRouteReadouts();
    }
  );

  const hoodSizes = [4, 4, 4, 4, nodeCount * 4];
  const hoodReader = new SummaryReader(
    resources,
    'routing-hood',
    [
      {buffer: hoodNodeCount, size: 4},
      {buffer: hoodNodeOverflow, size: 4},
      {buffer: hoodEdgeCount, size: 4},
      {buffer: hoodEdgeOverflow, size: 4},
      {buffer: hopDistances, size: hoodSizes[4]}
    ],
    bytes => {
      if (destroyed) return;
      const [nodes, nodeOver, edges, edgeOver, hops] = sliceSections(bytes, hoodSizes);
      cpuHops = new Uint32Array(hops);
      ctx.setReadout(
        'hood',
        `${formatInteger(new Uint32Array(nodes)[0])} intersections / ${formatInteger(new Uint32Array(edges)[0])} street segments` +
          (new Uint32Array(nodeOver)[0] || new Uint32Array(edgeOver)[0]
            ? ' (capacity reached)'
            : '')
      );
    }
  );

  /** Counts left, right and U-turns along one extracted edge sequence. */
  const countTurns = (edges: Uint32Array, first: number, last: number) => {
    let left = 0;
    let right = 0;
    let uTurns = 0;
    const straight = ctx.options.straightAngle;
    for (let slot = first + 1; slot < last && slot < edges.length; slot++) {
      const from = edges[slot - 1];
      const to = edges[slot];
      if (from >= edgeCount || to >= edgeCount || from === NO_EDGE || to === NO_EDGE) continue;
      const angle = getTurnAngle(network, from, to);
      if (Math.abs(angle) >= 2.9) uTurns++;
      else if (angle > straight) left++;
      else if (angle < -straight) right++;
    }
    return {left, right, uTurns};
  };

  function updateRouteReadouts(): void {
    const destinations = Math.max(1, ctx.options.destinationCount);
    const lines: string[] = [];
    for (let index = 0; index < destinations; index++) {
      lines.push(summary.treeFound[index] ? formatMinutes(summary.treeCost[index]) : 'not found');
    }
    ctx.setReadout('routeTimes', lines.join(' / '));
    const found = summary.treeFound[0] === 1;
    ctx.setReadout(
      'routeTime',
      found ? formatMinutes(summary.treeCost[0]) : 'not found (beyond the cost limit?)'
    );
    const treeFirst = summary.treeOffsets[0];
    const treeLast = Math.min(summary.treeOffsets[1], treeFirst + MAXIMUM_PATH_LENGTH);
    if (found) {
      const plain = countTurns(summary.treeEdges, treeFirst, treeLast);
      let meters = 0;
      for (let slot = treeFirst; slot < treeLast; slot++) {
        const edge = summary.treeEdges[slot];
        if (edge < edgeCount) meters += network.length[edge];
      }
      ctx.setReadout('routeLength', `${(meters / 1000).toFixed(1)} km`);
      ctx.setReadout('plainTurns', `${plain.left} left, ${plain.right} right, ${plain.uTurns} U`);
    } else {
      ctx.setReadout('routeLength', '-');
      ctx.setReadout('plainTurns', '-');
    }
    if (ctx.options.showTurnRoute) {
      const turnFound = summary.turnFound[0] === 1;
      ctx.setReadout('turnTime', turnFound ? formatMinutes(summary.turnCost[0]) : 'not found');
      if (turnFound) {
        const turnFirst = summary.turnOffsets[0];
        const turnLast = Math.min(summary.turnOffsets[1], turnFirst + MAXIMUM_PATH_LENGTH);
        const turns = countTurns(summary.turnEdges, turnFirst, turnLast);
        ctx.setReadout('turnTurns', `${turns.left} left, ${turns.right} right, ${turns.uTurns} U`);
        // Travel time and length of the turn-aware route without its turn penalties.
        let seconds = 0;
        let meters = 0;
        for (let slot = turnFirst; slot < turnLast; slot++) {
          const edge = summary.turnEdges[slot];
          if (edge < edgeCount) {
            seconds += costs[edge];
            meters += network.length[edge];
          }
        }
        if (found) {
          const extra = seconds - summary.treeCost[0];
          ctx.setReadout(
            'turnCost',
            `${extra >= 0 ? '+' : ''}${(extra / 60).toFixed(1)} min, ${(meters / 1000).toFixed(1)} km`
          );
        }
      } else {
        ctx.setReadout('turnTurns', '-');
        ctx.setReadout('turnCost', '-');
      }
    } else {
      for (const id of ['turnTime', 'turnTurns', 'turnCost']) ctx.setReadout(id, 'off');
    }
  }

  // ---- Initial state ------------------------------------------------------------------------
  ctx.setReadout('nodes', formatInteger(nodeCount));
  ctx.setReadout('edges', formatInteger(edgeCount));
  ctx.setReadout(
    'bans',
    `${formatInteger(bannedTurnTotal)} left turns between primary or secondary roads`
  );
  buildGraphs(Number(ctx.options.localIterations));
  writeCosts();
  writeScalarParameters();
  writeOriginAndMarkers();
  writeTurnParameters();

  const markAllDirty = () => {
    treeDirty = Math.max(treeDirty, 1);
    hoodDirty = Math.max(hoodDirty, 1);
    turnDirty = Math.max(turnDirty, 1);
  };

  const getRamp = () => ctx.options.ramp;

  return {
    getCompiledGraphs: () => (graphs ? [graphs.tree.compiled, graphs.hood, graphs.turns] : []),

    setOption(id, _value, state) {
      switch (id) {
        case 'originPlace':
          originNode = resolvePlace(state.originPlace);
          writeOriginAndMarkers();
          markAllDirty();
          ctx.requestLayers();
          break;
        case 'destinationPlace':
          writeDestinationPlaces();
          writeOriginAndMarkers();
          markAllDirty();
          ctx.requestLayers();
          break;
        case 'destinationCount':
        case 'costLimitMinutes':
        case 'hops':
          writeScalarParameters();
          markAllDirty();
          ctx.requestLayers();
          break;
        case 'closeExpressways':
        case 'expresswaySlowdown':
        case 'intersectionDelay':
          writeCosts();
          writeOriginAndMarkers();
          markAllDirty();
          break;
        case 'angleCost':
        case 'leftTurnCost':
        case 'rightTurnCost':
        case 'uTurns':
        case 'straightAngle':
        case 'banLeftTurns':
          writeTurnParameters();
          turnDirty = Math.max(turnDirty, 1);
          break;
        case 'showTurnRoute':
          turnDirty = Math.max(turnDirty, 1);
          updateRouteReadouts();
          ctx.requestLayers();
          break;
        case 'localIterations':
          buildGraphs(Number(state.localIterations));
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const [x, y] = network.projection.project(event.coordinate[0], event.coordinate[1]);
      const node = network.findNearestNode(x, y);
      // The first click moves the destination; shift is unavailable on touch, so alternate with
      // the origin: a click within 400 m of the origin picks the origin up instead.
      const originDistance = Math.hypot(
        network.nodePositions[originNode * 2] - x,
        network.nodePositions[originNode * 2 + 1] - y
      );
      if (originDistance < 400) {
        originNode = node;
      } else {
        destinationNodes[0] = node;
      }
      writeOriginAndMarkers();
      markAllDirty();
      return true;
    },

    getTooltip(event) {
      if (!event.coordinate || !cpuCosts) return null;
      const [x, y] = network.projection.project(event.coordinate[0], event.coordinate[1]);
      const node = network.findNearestNode(x, y);
      if (
        Math.hypot(network.nodePositions[node * 2] - x, network.nodePositions[node * 2 + 1] - y) >
        300
      ) {
        return null;
      }
      const parts = [`Drive time from origin: ${formatMinutes(cpuCosts[node])}`];
      if (cpuHops && ctx.options.base === 'hops') {
        parts.push(
          cpuHops[node] === GPU_NETWORK_REACHABILITY_NONE
            ? 'Outside the ego network'
            : `${cpuHops[node]} intersections from origin`
        );
      }
      return parts.join('\n');
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      if (!graphs) return;
      const options = ctx.options;
      if (treeDirty > 0) {
        graphs.tree.compiled.encode(commandEncoder, {parameters: undefined});
        treeDirty--;
        treeReader.markStale();
      }
      if (hoodDirty > 0) {
        graphs.hood.encode(commandEncoder, {parameters: undefined});
        hoodDirty--;
        hoodReader.markStale();
      }
      if (turnDirty > 0 && options.showTurnRoute) {
        graphs.turns.encode(commandEncoder, {parameters: undefined});
        turnDirty--;
        turnReader.markStale();
      }
      treeReader.flush(commandEncoder);
      hoodReader.flush(commandEncoder);
      if (options.showTurnRoute) turnReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const colors = getRoadColors(ctx.theme());
      const coordinateOrigin: [number, number, number] = [network.origin[0], network.origin[1], 0];
      const layers: Layer[] = [
        new SpatialAnalysisSegmentLayer({
          id: 'routing-roads',
          coordinateOrigin,
          segments: segmentsBuffer,
          instanceCount: segmentCount,
          widthPixels: 1,
          color: colors.minor
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'routing-major-roads',
          coordinateOrigin,
          segments: majorSegmentsBuffer,
          instanceCount: network.majorSegments.length / 4,
          widthPixels: 1.6,
          color: colors.major
        })
      ];
      if (options.base === 'time') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'routing-cost-tint',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 2.4,
            values: treeCosts,
            valueFormat: 'float32',
            valueIndices: segmentTargetsBuffer,
            colormap: getRamp(),
            valueRange: [0, options.costLimitMinutes * 60],
            color: [255, 255, 255, 235],
            noDataColor: [0, 0, 0, 0]
          })
        );
      } else if (options.base === 'hops') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'routing-hop-tint',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 2.8,
            values: hopDistances,
            valueFormat: 'uint32',
            valueIndices: segmentTargetsBuffer,
            colormap: getRamp(),
            valueRange: [0, Math.max(1, options.hops)],
            color: [255, 255, 255, 240],
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      const routeLayer = (id: string, flags: Buffer, color: readonly number[], width: number) =>
        new SpatialAnalysisSegmentLayer({
          id,
          coordinateOrigin,
          segments: segmentsBuffer,
          instanceCount: segmentCount,
          widthPixels: width,
          values: flags,
          valueFormat: 'uint32',
          valueIndices: segmentEdgesBuffer,
          colormap: 'mask',
          color: color as [number, number, number, number],
          noDataColor: [0, 0, 0, 0]
        });
      if (options.showRoute) {
        layers.push(
          routeLayer('routing-route-halo', routeFlags, colors.halo, 8),
          routeLayer('routing-route', routeFlags, ROUTE_COLOR, 4.5)
        );
      }
      if (options.showTurnRoute) {
        layers.push(routeLayer('routing-turn-route', turnFlags, TURN_ROUTE_COLOR, 2.4));
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'routing-destination-halo',
          coordinateOrigin,
          positions: markerBuffer,
          instanceCount: 1 + Math.max(1, options.destinationCount),
          radiusPixels: 10,
          color: colors.halo
        }),
        new SpatialAnalysisPointLayer({
          id: 'routing-destinations',
          coordinateOrigin,
          positions: markerBuffer,
          instanceCount: 1 + Math.max(1, options.destinationCount),
          radiusPixels: 7,
          values: markerIds,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: [[255, 255, 255, 255], ...DESTINATION_PALETTE]
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      treeReader.stop();
      hoodReader.stop();
      turnReader.stop();
      resources.destroy();
    }
  };
}
