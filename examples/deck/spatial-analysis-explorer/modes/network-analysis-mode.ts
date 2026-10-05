// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Network analysis over the New York road network: shortest-route extraction, six-facility service
 * areas, k-hop neighborhoods and node analytics (PageRank, degree, core number, communities).
 *
 * One small graph converts the sorted COO edge list to CSR once. Every analysis graph imports the
 * resulting caller-owned CSR buffers, is compiled once in `create()`, and is re-encoded only when
 * its inputs change (a click or a slider). Results persist in storage buffers and deck.gl layers
 * draw them directly, gathering through compact edge IDs whose counts are copied on the GPU into
 * indirect draw records. Only small summaries are read back through a `GPUReadbackRing`.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUCOOToCSR,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_NETWORK_REACHABILITY_NONE,
  GPUNetworkAnalyticsColumns,
  GPUNetworkNeighborhood,
  GPUNetworkPathExtraction,
  GPUNetworkReachability,
  GPUNetworkServiceAreas
} from '@luma.gl/experimental/gpu-network';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection, type SpatialAnalysisRoadNetwork} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';

type View = 'route' | 'service-areas' | 'neighborhood' | 'analytics';
type Metric = 'pageRank' | 'degree' | 'coreNumber' | 'community';

const VIEWS: readonly View[] = ['route', 'service-areas', 'neighborhood', 'analytics'];
const WALK_SPEED = 1.4;
/** Route solver round bound (each round covers up to 16 hops); the GPU stops once costs converge. */
const ROUTE_MAXIMUM_ITERATIONS = 64;
const ROUTE_MAXIMUM_PATH = 4096;
const SERVICE_MAXIMUM_ITERATIONS = 384;
const FACILITY_COUNT = 6;
const MAXIMUM_HOPS = 48;
const NODE_CAPACITY = 32768;
const EDGE_CAPACITY = 65536;
const MAX_ROUTE_COST_SECONDS = 3600;
const BASE_COLOR = [90, 92, 105, 110] as const;
const ROUTE_COLOR = [255, 80, 60, 255] as const;
const NO_DATA_COLOR = [90, 92, 105, 60] as const;
const FACILITY_COLORS = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255]
] as const;
const COMMUNITY_COLORS = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;
const INFERNO_STOPS = [
  [0, 0, 4, 255],
  [87, 16, 110, 255],
  [188, 55, 84, 255],
  [249, 142, 9, 255],
  [252, 255, 164, 255]
] as const;
const TIMES_SQUARE: readonly [number, number] = [-73.9855, 40.758];
const UNION_SQUARE: readonly [number, number] = [-73.9903, 40.7359];
const DEFAULT_FACILITIES: readonly (readonly [number, number])[] = [
  [-74.0059, 40.7127], // City Hall
  UNION_SQUARE,
  TIMES_SQUARE,
  [-73.9819, 40.7681], // Columbus Circle
  [-73.9556, 40.7794], // East 86th Street
  [-73.9442, 40.7983] // East Harlem, 116th Street
];
const METRIC_THRESHOLDS: Record<Metric, number> = {
  pageRank: 0.4,
  degree: 0.6,
  coreNumber: 1,
  community: 0
};

export const networkAnalysisMode: SpatialAnalysisModeDefinition = {
  id: 'network-analysis',
  title: 'Network analysis',
  contributors: [
    'GPUNetworkReachability',
    'GPUNetworkPathExtraction',
    'GPUNetworkServiceAreas',
    'GPUNetworkNeighborhood',
    'GPUNetworkAnalyticsColumns'
  ],
  description:
    'Routes, service areas, k-hop neighborhoods and PageRank/core/community columns on the New ' +
    'York street graph. One CSR is built once; each analysis re-encodes only when you click or ' +
    'move a slider.',
  initialViewState: {longitude: -73.975, latitude: 40.755, zoom: 12.4},

  async create(context) {
    const roads = await context.data.getNewYorkRoads();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(roads.origin);
    const resources = new SpatialAnalysisResources(device, 'network');
    const nodeCount = roads.nodePositions.length / 2;
    const segmentCount = roads.segmentNodes.length;
    const edges = sortEdgesBySource(roads);
    const edgeCount = edges.sources.length;
    const nodePosition = (node: number): [number, number] => [
      roads.nodePositions[node * 2],
      roads.nodePositions[node * 2 + 1]
    ];
    const lonLatToNode = ([longitude, latitude]: readonly [number, number]) =>
      findNearestNode(roads.nodePositions, projection.project(longitude, latitude));

    // Walking seconds. The floor avoids zero-weight edges: GPUNetworkReachability predecessors use
    // a strict `costs[u] < costs[v]` test, so nodes reached only across zero-weight edges would
    // get NONE predecessors and path extraction would report truncated routes.
    const weights = new Float32Array(edgeCount);
    // CSR edge e is sorted edge e (GPUCOOToCSR copies row-sorted COO in order).
    const edgeSegmentData = new Float32Array(edgeCount * 4);
    for (let edge = 0; edge < edgeCount; edge++) {
      weights[edge] = Math.max(edges.lengths[edge], 0.5) / WALK_SPEED;
      edgeSegmentData.set(nodePosition(edges.sources[edge]), edge * 4);
      edgeSegmentData.set(nodePosition(edges.targets[edge]), edge * 4 + 2);
    }

    // State.
    let view: View = readInitialView();
    let metric: Metric = 'pageRank';
    let costLimitMinutes = 15;
    let hops = 12;
    let originNode = lonLatToNode(TIMES_SQUARE);
    let destinationNode = lonLatToNode(UNION_SQUARE);
    let routeClickIsDestination = false;
    let seedNode = originNode;
    const facilityNodes = DEFAULT_FACILITIES.map(lonLatToNode);
    /** Remaining encodes per graph. The first two frames always encode, like reachability. */
    const pending: Record<View | 'csr', number> = {
      csr: 2,
      route: 2,
      'service-areas': 2,
      neighborhood: 2,
      analytics: 2
    };
    let destroyed = false;
    let readbackPending = false;
    const summaryStale: Record<View, boolean> = {
      route: false,
      'service-areas': false,
      neighborhood: false,
      analytics: false
    };

    // Shared buffers.
    const cooRows = resources.createBuffer('coo-rows', edges.sources);
    const cooColumns = resources.createBuffer('coo-columns', edges.targets);
    const cooWeights = resources.createBuffer('coo-weights', weights);
    const csrOffsets = resources.createBuffer('csr-offsets', (nodeCount + 1) * 4);
    const csrNeighbors = resources.createBuffer('csr-neighbors', edgeCount * 4);
    const csrWeights = resources.createBuffer('csr-weights', edgeCount * 4);
    const nodePositions = resources.createBuffer('node-positions', roads.nodePositions);
    const segmentsBuffer = resources.createBuffer('segments', roads.segments);
    const segmentNodes = resources.createBuffer('segment-nodes', roads.segmentNodes);
    const edgeSegments = resources.createBuffer('edge-segments', edgeSegmentData);
    const edgeSourceNodes = resources.createBuffer('edge-source-nodes', edges.sources);
    const originPosition = resources.createBuffer('origin-position', 8);
    const destinationPosition = resources.createBuffer('destination-position', 8);
    const seedPosition = resources.createBuffer('seed-position', 8);
    const facilityPositions = resources.createBuffer('facility-positions', FACILITY_COUNT * 8);
    const facilityRows = resources.createBuffer(
      'facility-rows',
      Uint32Array.from({length: FACILITY_COUNT}, (_, row) => row)
    );

    // Route buffers.
    const routeCosts = resources.createBuffer('route-costs', nodeCount * 4);
    const routePredecessors = resources.createBuffer('route-predecessors', nodeCount * 4);
    const routeConverged = resources.createBuffer('route-converged', 4);
    const routeIterations = resources.createBuffer('route-iterations', 4);
    const routeNodeIds = resources.createBuffer('route-node-ids', ROUTE_MAXIMUM_PATH * 4);
    const routeNodeCount = resources.createBuffer('route-node-count', 4);
    const routeNodeOverflow = resources.createBuffer('route-node-overflow', 4);
    const routeEdgeIds = resources.createBuffer('route-edge-ids', ROUTE_MAXIMUM_PATH * 4);
    const routeEdgeCount = resources.createBuffer('route-edge-count', 4);
    const routeEdgeOverflow = resources.createBuffer('route-edge-overflow', 4);
    const routePathCost = resources.createBuffer('route-path-cost', 4);
    const routePathFound = resources.createBuffer('route-path-found', 4);
    const routeDraw = resources.track(
      new DrawCommandBuffer(device, {
        id: 'network-route-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const origin = resources.createParameterBuffer('origin', 'uint32', 1);
    const destination = resources.createParameterBuffer('destination', 'uint32', 1);

    // Service area buffers.
    const serviceAssignments = resources.createBuffer('service-assignments', nodeCount * 4);
    const serviceCosts = resources.createBuffer('service-costs', nodeCount * 4);
    const serviceNodeCounts = resources.createBuffer('service-node-counts', FACILITY_COUNT * 4);
    const serviceCostSums = resources.createBuffer('service-cost-sums', FACILITY_COUNT * 4);
    const serviceConverged = resources.createBuffer('service-converged', 4);
    const facilities = resources.createParameterBuffer('facilities', 'uint32', FACILITY_COUNT);
    const serviceCostLimit = resources.createParameterBuffer('service-cost-limit', 'float32', 1);

    // Neighborhood buffers.
    const hopDistances = resources.createBuffer('hop-distances', nodeCount * 4);
    const hoodNodeIds = resources.createBuffer('hood-node-ids', NODE_CAPACITY * 4);
    const hoodNodeCount = resources.createBuffer('hood-node-count', 4);
    const hoodNodeOverflow = resources.createBuffer('hood-node-overflow', 4);
    const hoodEdgeIds = resources.createBuffer('hood-edge-ids', EDGE_CAPACITY * 4);
    const hoodEdgeCount = resources.createBuffer('hood-edge-count', 4);
    const hoodEdgeOverflow = resources.createBuffer('hood-edge-overflow', 4);
    const hoodNodeDraw = resources.track(
      new DrawCommandBuffer(device, {
        id: 'network-hood-node-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const hoodEdgeDraw = resources.track(
      new DrawCommandBuffer(device, {
        id: 'network-hood-edge-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const seeds = resources.createParameterBuffer('seeds', 'uint32', 1);
    const hopsParameter = resources.createParameterBuffer('hops', 'uint32', 1);

    // Analytics buffers.
    const degreeRaw = resources.createBuffer('degree', nodeCount * 4);
    const degreeNormalized = resources.createBuffer('degree-normalized', nodeCount * 4);
    const pageRankRaw = resources.createBuffer('page-rank', nodeCount * 4);
    const pageRankNormalized = resources.createBuffer('page-rank-normalized', nodeCount * 4);
    const pageRankResidual = resources.createBuffer('page-rank-residual', 4);
    const coreRaw = resources.createBuffer('core', nodeCount * 4);
    const coreNormalized = resources.createBuffer('core-normalized', nodeCount * 4);
    const coreDegeneracy = resources.createBuffer('core-degeneracy', 4);
    const coreConverged = resources.createBuffer('core-converged', 4);
    const communityLabels = resources.createBuffer('communities', nodeCount * 4);
    const communityConverged = resources.createBuffer('community-converged', 4);

    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'network-summary', byteLength: 64})
    );

    // Graphs. The CSR graph owns the only transform of COO data; every other graph imports the
    // caller-owned CSR buffers, which also satisfies the analytics contributor's imported-CSR rule.
    const csrGraph = new GPUCommandGraph<void>(device, {id: 'network-csr'});
    csrGraph.add(
      new GPUCOOToCSR({
        id: 'network-csr',
        rows: nodeCount,
        rowIndices: importGraphBuffer(csrGraph, 'coo-rows', cooRows, 'uint32', edgeCount),
        columnIndices: importGraphBuffer(csrGraph, 'coo-columns', cooColumns, 'uint32', edgeCount),
        values: importGraphBuffer(csrGraph, 'coo-weights', cooWeights, 'float32', edgeCount),
        rowOffsets: importGraphBuffer(csrGraph, 'csr-offsets', csrOffsets, 'uint32', nodeCount + 1),
        outputColumnIndices: importGraphBuffer(
          csrGraph,
          'csr-neighbors',
          csrNeighbors,
          'uint32',
          edgeCount
        ),
        outputValues: importGraphBuffer(csrGraph, 'csr-weights', csrWeights, 'float32', edgeCount)
      })
    );
    const importCSR = (graph: GPUCommandGraph<void>) => ({
      offsets: importGraphBuffer(graph, 'csr-offsets', csrOffsets, 'uint32', nodeCount + 1),
      neighbors: importGraphBuffer(graph, 'csr-neighbors', csrNeighbors, 'uint32', edgeCount),
      weights: importGraphBuffer(graph, 'csr-weights', csrWeights, 'float32', edgeCount)
    });
    const importOutput = <Format extends 'uint32' | 'float32'>(
      graph: GPUCommandGraph<void>,
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length);
    const compact = (
      graph: GPUCommandGraph<void>,
      name: string,
      ids: Buffer,
      capacity: number,
      count: Buffer,
      overflow: Buffer
    ) => ({
      ids: importOutput(graph, `${name}-ids`, ids, 'uint32', capacity),
      count: importOutput(graph, `${name}-count`, count, 'uint32', 1),
      overflow: importOutput(graph, `${name}-overflow`, overflow, 'uint32', 1)
    });

    const routeGraph = new GPUCommandGraph<void>(device, {id: 'network-route'});
    {
      const csr = importCSR(routeGraph);
      const costsView = importOutput(routeGraph, 'route-costs', routeCosts, 'float32', nodeCount);
      const predecessorsView = importOutput(
        routeGraph,
        'route-predecessors',
        routePredecessors,
        'uint32',
        nodeCount
      );
      routeGraph.add(
        new GPUNetworkReachability({
          id: 'route-reachability',
          ...csr,
          sources: origin.importToGraph(routeGraph),
          maxIterations: ROUTE_MAXIMUM_ITERATIONS,
          costs: costsView,
          predecessors: predecessorsView,
          converged: importOutput(routeGraph, 'route-converged', routeConverged, 'uint32', 1),
          iterationCount: importOutput(routeGraph, 'route-iterations', routeIterations, 'uint32', 1)
        })
      );
      routeGraph.add(
        new GPUNetworkPathExtraction({
          id: 'route-path',
          predecessors: predecessorsView,
          costs: costsView,
          targets: destination.importToGraph(routeGraph),
          maxPathLength: ROUTE_MAXIMUM_PATH,
          output: compact(
            routeGraph,
            'route-nodes',
            routeNodeIds,
            ROUTE_MAXIMUM_PATH,
            routeNodeCount,
            routeNodeOverflow
          ),
          pathCosts: importOutput(routeGraph, 'route-path-cost', routePathCost, 'float32', 1),
          pathFound: importOutput(routeGraph, 'route-path-found', routePathFound, 'uint32', 1),
          edges: {
            ...csr,
            output: compact(
              routeGraph,
              'route-edges',
              routeEdgeIds,
              ROUTE_MAXIMUM_PATH,
              routeEdgeCount,
              routeEdgeOverflow
            )
          }
        })
      );
    }

    const serviceGraph = new GPUCommandGraph<void>(device, {id: 'network-service'});
    serviceGraph.add(
      new GPUNetworkServiceAreas({
        id: 'service-areas',
        ...importCSR(serviceGraph),
        facilities: facilities.importToGraph(serviceGraph),
        costLimit: serviceCostLimit.importToGraph(serviceGraph),
        maxIterations: SERVICE_MAXIMUM_ITERATIONS,
        assignments: importOutput(
          serviceGraph,
          'service-assignments',
          serviceAssignments,
          'uint32',
          nodeCount
        ),
        costs: importOutput(serviceGraph, 'service-costs', serviceCosts, 'float32', nodeCount),
        facilityNodeCounts: importOutput(
          serviceGraph,
          'service-node-counts',
          serviceNodeCounts,
          'uint32',
          FACILITY_COUNT
        ),
        facilityCostSums: importOutput(
          serviceGraph,
          'service-cost-sums',
          serviceCostSums,
          'float32',
          FACILITY_COUNT
        ),
        converged: importOutput(serviceGraph, 'service-converged', serviceConverged, 'uint32', 1)
      })
    );

    const hoodGraph = new GPUCommandGraph<void>(device, {id: 'network-hood'});
    {
      const csr = importCSR(hoodGraph);
      hoodGraph.add(
        new GPUNetworkNeighborhood({
          id: 'neighborhood',
          offsets: csr.offsets,
          neighbors: csr.neighbors,
          seeds: seeds.importToGraph(hoodGraph),
          hops: hopsParameter.importToGraph(hoodGraph),
          maxHops: MAXIMUM_HOPS,
          hopDistances: importOutput(hoodGraph, 'hop-distances', hopDistances, 'uint32', nodeCount),
          nodes: compact(
            hoodGraph,
            'hood-nodes',
            hoodNodeIds,
            NODE_CAPACITY,
            hoodNodeCount,
            hoodNodeOverflow
          ),
          edges: compact(
            hoodGraph,
            'hood-edges',
            hoodEdgeIds,
            EDGE_CAPACITY,
            hoodEdgeCount,
            hoodEdgeOverflow
          )
        })
      );
    }

    const analyticsGraph = new GPUCommandGraph<void>(device, {id: 'network-analytics'});
    const analyticsCSR = importCSR(analyticsGraph);
    const analytics = new GPUNetworkAnalyticsColumns({
      id: 'network-analytics',
      offsets: analyticsCSR.offsets,
      neighbors: analyticsCSR.neighbors,
      degree: {
        output: importOutput(analyticsGraph, 'degree', degreeRaw, 'uint32', nodeCount),
        normalized: importOutput(
          analyticsGraph,
          'degree-normalized',
          degreeNormalized,
          'float32',
          nodeCount
        )
      },
      pageRank: {
        output: importOutput(analyticsGraph, 'page-rank', pageRankRaw, 'float32', nodeCount),
        normalized: importOutput(
          analyticsGraph,
          'page-rank-normalized',
          pageRankNormalized,
          'float32',
          nodeCount
        ),
        iterations: 100,
        residual: importOutput(analyticsGraph, 'page-rank-residual', pageRankResidual, 'float32', 1)
      },
      coreNumber: {
        output: importOutput(analyticsGraph, 'core', coreRaw, 'uint32', nodeCount),
        normalized: importOutput(
          analyticsGraph,
          'core-normalized',
          coreNormalized,
          'float32',
          nodeCount
        ),
        iterations: 256,
        degeneracy: importOutput(analyticsGraph, 'core-degeneracy', coreDegeneracy, 'uint32', 1),
        converged: importOutput(analyticsGraph, 'core-converged', coreConverged, 'uint32', 1)
      },
      communities: {
        output: importOutput(analyticsGraph, 'communities', communityLabels, 'uint32', nodeCount),
        converged: importOutput(
          analyticsGraph,
          'community-converged',
          communityConverged,
          'uint32',
          1
        )
      }
    });
    analyticsGraph.add(analytics);

    // The analytics contributor owns a 16-byte buffer: track it before the compiled graphs so the
    // reverse-order teardown destroys it after them.
    resources.track(analytics);
    const compiledCSR: CompiledGPUCommandGraph<void> = resources.track(csrGraph.compile());
    const compiledRoute: CompiledGPUCommandGraph<void> = resources.track(routeGraph.compile());
    const compiledService: CompiledGPUCommandGraph<void> = resources.track(serviceGraph.compile());
    const compiledHood: CompiledGPUCommandGraph<void> = resources.track(hoodGraph.compile());
    const compiledAnalytics: CompiledGPUCommandGraph<void> = resources.track(
      analyticsGraph.compile()
    );
    const compiledGraphs = [
      compiledCSR,
      compiledRoute,
      compiledService,
      compiledHood,
      compiledAnalytics
    ] as const;
    const compiledByView: Record<View, CompiledGPUCommandGraph<void>> = {
      route: compiledRoute,
      'service-areas': compiledService,
      neighborhood: compiledHood,
      analytics: compiledAnalytics
    };

    // Controls. The panel cannot hide controls, so every view's controls exist and the inactive
    // ones are disabled.
    const controls = context.controls;
    const viewSelect = controls.addSelect<View>({
      label: 'View',
      options: [
        {value: 'route', label: 'Route (shortest path)'},
        {value: 'service-areas', label: 'Service areas (6 facilities)'},
        {value: 'neighborhood', label: 'Neighborhood (k hops)'},
        {value: 'analytics', label: 'Analytics (node columns)'}
      ],
      value: view,
      onChange: value => setView(value)
    });
    const costLimitSlider = controls.addSlider({
      label: 'Service areas: cost limit',
      min: 2,
      max: 30,
      step: 1,
      value: costLimitMinutes,
      format: value => `${value} min`,
      onChange: value => {
        costLimitMinutes = value;
        writeCostLimit();
        markDirty('service-areas');
      }
    });
    const hopsSlider = controls.addSlider({
      label: 'Neighborhood: k (hops)',
      min: 0,
      max: MAXIMUM_HOPS,
      step: 1,
      value: hops,
      format: value => `${value}`,
      onChange: value => {
        hops = value;
        hopsParameter.write(Uint32Array.of(hops));
        markDirty('neighborhood');
        context.updateLayers();
      }
    });
    const metricSelect = controls.addSelect<Metric>({
      label: 'Analytics: metric',
      options: [
        {value: 'pageRank', label: 'PageRank'},
        {value: 'degree', label: 'Degree'},
        {value: 'coreNumber', label: 'Core number'},
        {value: 'community', label: 'Community (label propagation)'}
      ],
      value: metric,
      onChange: value => {
        metric = value;
        context.updateLayers();
      }
    });
    const noteReadout = controls.addNote('');
    controls.addLegend({
      title: 'Service areas (facility rows)',
      entries: FACILITY_COLORS.map((color, row) => ({color, label: `${row + 1}`}))
    });
    controls.addLegend({
      title: 'Neighborhood hops / analytics value',
      gradient: {colors: INFERNO_STOPS, minimumLabel: 'low', maximumLabel: 'high'}
    });
    const routeReadout = controls.addReadout('Route');
    const pathReadout = controls.addReadout('Path size');
    const serviceReadout = controls.addReadout('Facility nodes');
    const hoodReadout = controls.addReadout('Ego network');
    const analyticsReadout = controls.addReadout('Analytics');
    const solverReadout = controls.addReadout('Solver');
    controls.addReadout('Nodes', formatCount(nodeCount));
    controls.addReadout('Directed edges', formatCount(edgeCount));
    controls.addReadout('Data', roads.attribution);

    const NOTES: Record<View, string> = {
      route:
        'Click to set the origin (white), click again for the destination (magenta). The route ' +
        'is extracted on the GPU from predecessors; the street tint is walking time.',
      'service-areas':
        'Click to move the nearest facility. Every node is assigned to its closest facility by ' +
        'walking time within the cost limit.',
      neighborhood: 'Click to set the seed. Edges are colored by hop distance from it.',
      analytics: 'Node analytics computed once on the GPU from the shared CSR.'
    };

    const writeCostLimit = () => serviceCostLimit.write(Float32Array.of(costLimitMinutes * 60));
    const writePositionBuffer = (buffer: Buffer, node: number, slot = 0) =>
      buffer.write(roads.nodePositions.subarray(node * 2, node * 2 + 2), slot * 8);
    const writeFacilities = () => {
      facilities.write(Uint32Array.from(facilityNodes));
      facilityNodes.forEach((node, row) => writePositionBuffer(facilityPositions, node, row));
    };
    const markDirty = (target: View) => {
      pending[target] = Math.max(pending[target], 1);
    };
    const updateControlState = () => {
      costLimitSlider.setDisabled(view !== 'service-areas');
      hopsSlider.setDisabled(view !== 'neighborhood');
      metricSelect.setDisabled(view !== 'analytics');
      noteReadout.setValue(NOTES[view]);
    };
    const setView = (next: View) => {
      view = next;
      updateControlState();
      context.updateLayers();
    };

    origin.write(Uint32Array.of(originNode));
    destination.write(Uint32Array.of(destinationNode));
    seeds.write(Uint32Array.of(seedNode));
    hopsParameter.write(Uint32Array.of(hops));
    writePositionBuffer(originPosition, originNode);
    writePositionBuffer(destinationPosition, destinationNode);
    writePositionBuffer(seedPosition, seedNode);
    writeCostLimit();
    writeFacilities();
    viewSelect.setValue(view);
    updateControlState();

    const copyWords = (
      commandEncoder: CommandEncoder,
      destinationBuffer: Buffer,
      sources: readonly (readonly [Buffer, number])[]
    ) => {
      let offset = 0;
      for (const [sourceBuffer, words] of sources) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          destinationBuffer,
          destinationOffset: offset,
          size: words * 4
        });
        offset += words * 4;
      }
      return offset;
    };
    const copyCount = (commandEncoder: CommandEncoder, count: Buffer, draw: DrawCommandBuffer) =>
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: count,
        destinationBuffer: draw.buffer,
        destinationOffset: 4,
        size: 4
      });

    const summarySources = (target: View): readonly (readonly [Buffer, number])[] => {
      switch (target) {
        case 'route':
          return [
            [routePathFound, 1],
            [routePathCost, 1],
            [routeNodeCount, 1],
            [routeNodeOverflow, 1],
            [routeEdgeCount, 1],
            [routeEdgeOverflow, 1],
            [routeConverged, 1],
            [routeIterations, 1]
          ];
        case 'service-areas':
          return [
            [serviceConverged, 1],
            [serviceNodeCounts, FACILITY_COUNT],
            [serviceCostSums, FACILITY_COUNT]
          ];
        case 'neighborhood':
          return [
            [hoodNodeCount, 1],
            [hoodNodeOverflow, 1],
            [hoodEdgeCount, 1],
            [hoodEdgeOverflow, 1]
          ];
        default:
          return [
            [pageRankResidual, 1],
            [coreDegeneracy, 1],
            [coreConverged, 1],
            [communityConverged, 1]
          ];
      }
    };

    const showSummary = (target: View, words: Uint32Array, floats: Float32Array) => {
      switch (target) {
        case 'route': {
          const found = words[0] === 1;
          routeReadout.setValue(
            found ? `found, ${(floats[1] / 60).toFixed(1)} min walk` : 'no route found'
          );
          pathReadout.setValue(
            `${formatCount(words[2])} nodes / ${formatCount(words[4])} edges` +
              `${words[3] || words[5] ? ' (overflow)' : ''}`
          );
          solverReadout.setValue(
            `${words[6] ? 'converged' : 'NOT converged'} after ${words[7]} of ${ROUTE_MAXIMUM_ITERATIONS}`
          );
          break;
        }
        case 'service-areas': {
          serviceReadout.setValue(
            Array.from(words.subarray(1, 1 + FACILITY_COUNT), formatCount).join(' / ')
          );
          const means = Array.from({length: FACILITY_COUNT}, (_, row) => {
            const count = words[1 + row];
            return count ? (floats[1 + FACILITY_COUNT + row] / count / 60).toFixed(1) : '-';
          });
          solverReadout.setValue(
            `${words[0] ? 'converged' : 'NOT converged'}; mean min ${means.join(' / ')}`
          );
          break;
        }
        case 'neighborhood':
          hoodReadout.setValue(
            `${formatCount(words[0])} nodes / ${formatCount(words[2])} edges` +
              `${words[1] || words[3] ? ' (overflow)' : ''}`
          );
          break;
        default:
          analyticsReadout.setValue(
            `PageRank residual ${floats[0].toExponential(1)}, degeneracy ${words[1]}`
          );
          solverReadout.setValue(
            `core ${words[2] ? 'converged' : 'NOT converged'}, communities ` +
              `${words[3] ? 'converged' : 'NOT converged (bounded heuristic)'}`
          );
      }
    };

    const readSummary = async (commandEncoder: CommandEncoder, target: View) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      summaryStale[target] = false;
      const byteLength = copyWords(commandEncoder, ticket.buffer, summarySources(target));
      ticket.markEncoded({byteOffset: 0, byteLength});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const copy = bytes.slice();
        showSummary(
          target,
          new Uint32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4),
          new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4)
        );
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => compiledGraphs,
      encode(commandEncoder) {
        // The CSR comes first in the encoder; it never changes after the first frames.
        if (pending.csr > 0) {
          compiledCSR.encode(commandEncoder, {parameters: undefined});
          pending.csr--;
        }
        if (pending[view] > 0) {
          compiledByView[view].encode(commandEncoder, {parameters: undefined});
          pending[view]--;
          summaryStale[view] = true;
          if (view === 'route') copyCount(commandEncoder, routeEdgeCount, routeDraw);
          if (view === 'neighborhood') {
            copyCount(commandEncoder, hoodNodeCount, hoodNodeDraw);
            copyCount(commandEncoder, hoodEdgeCount, hoodEdgeDraw);
          }
        }
        if (summaryStale[view] && !readbackPending) {
          void readSummary(commandEncoder, view);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [roads.origin[0], roads.origin[1], 0];
        const base = new SpatialAnalysisSegmentLayer({
          id: 'network-base',
          coordinateOrigin,
          segments: segmentsBuffer,
          instanceCount: segmentCount,
          widthPixels: 1.5,
          color: BASE_COLOR
        });
        const layers: Layer[] = [];
        const roadLayer = (
          id: string,
          style: Partial<ConstructorParameters<typeof SpatialAnalysisSegmentLayer>[0]>
        ) =>
          new SpatialAnalysisSegmentLayer({
            id,
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            valueIndices: segmentNodes,
            ...style
          });
        const point = (id: string, positions: Buffer, color: readonly number[], radius: number) =>
          new SpatialAnalysisPointLayer({
            id,
            coordinateOrigin,
            positions,
            instanceCount: 1,
            radiusPixels: radius,
            color: color as [number, number, number, number]
          });
        if (view === 'route') {
          layers.push(
            roadLayer('network-route-cost', {
              widthPixels: 2,
              values: routeCosts,
              valueFormat: 'float32',
              colormap: 'viridis',
              valueRange: [0, MAX_ROUTE_COST_SECONDS],
              color: [255, 255, 255, 90],
              noDataColor: NO_DATA_COLOR
            }),
            new SpatialAnalysisSegmentLayer({
              id: 'network-route-glow',
              coordinateOrigin,
              segments: edgeSegments,
              ids: routeEdgeIds,
              drawCommands: routeDraw,
              widthPixels: 9,
              color: [255, 255, 255, 70]
            }),
            new SpatialAnalysisSegmentLayer({
              id: 'network-route',
              coordinateOrigin,
              segments: edgeSegments,
              ids: routeEdgeIds,
              drawCommands: routeDraw,
              widthPixels: 5,
              color: ROUTE_COLOR
            }),
            point('network-origin', originPosition, [255, 255, 255, 255], 8),
            point('network-destination', destinationPosition, [255, 60, 220, 255], 8)
          );
        } else if (view === 'service-areas') {
          layers.push(
            base,
            roadLayer('network-service-roads', {
              widthPixels: 2.5,
              values: serviceAssignments,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: FACILITY_COLORS,
              noDataValue: GPU_NETWORK_REACHABILITY_NONE,
              noDataColor: [0, 0, 0, 0]
            }),
            new SpatialAnalysisPointLayer({
              id: 'network-facility-outline',
              coordinateOrigin,
              positions: facilityPositions,
              instanceCount: FACILITY_COUNT,
              radiusPixels: 11,
              color: [255, 255, 255, 255]
            }),
            new SpatialAnalysisPointLayer({
              id: 'network-facilities',
              coordinateOrigin,
              positions: facilityPositions,
              instanceCount: FACILITY_COUNT,
              radiusPixels: 8,
              values: facilityRows,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: FACILITY_COLORS
            })
          );
        } else if (view === 'neighborhood') {
          const range: [number, number] = [0, Math.max(1, hops)];
          layers.push(
            base,
            new SpatialAnalysisSegmentLayer({
              id: 'network-hood-edges',
              coordinateOrigin,
              segments: edgeSegments,
              ids: hoodEdgeIds,
              drawCommands: hoodEdgeDraw,
              widthPixels: 3,
              values: hopDistances,
              valueFormat: 'uint32',
              valueIndices: edgeSourceNodes,
              colormap: 'inferno',
              valueRange: range,
              noDataColor: NO_DATA_COLOR
            }),
            new SpatialAnalysisPointLayer({
              id: 'network-hood-nodes',
              coordinateOrigin,
              positions: nodePositions,
              ids: hoodNodeIds,
              drawCommands: hoodNodeDraw,
              radiusPixels: 2.5,
              values: hopDistances,
              valueFormat: 'uint32',
              colormap: 'inferno',
              valueRange: range,
              noDataColor: NO_DATA_COLOR
            }),
            point('network-seed', seedPosition, [255, 255, 255, 255], 8)
          );
        } else {
          const columns: Record<Metric, Buffer> = {
            pageRank: pageRankNormalized,
            degree: degreeNormalized,
            coreNumber: coreNormalized,
            community: communityLabels
          };
          if (metric === 'community') {
            layers.push(
              roadLayer('network-analytics-community', {
                widthPixels: 2.5,
                values: columns.community,
                valueFormat: 'uint32',
                colormap: 'category',
                palette: COMMUNITY_COLORS
              })
            );
          } else {
            const style = {
              values: columns[metric],
              valueFormat: 'float32' as const,
              colormap: 'inferno' as const,
              valueRange: (metric === 'pageRank' ? [0.1, 0.6] : [0, 1]) as [number, number],

              noDataColor: NO_DATA_COLOR
            };
            layers.push(roadLayer('network-analytics', {...style, widthPixels: 1.5}));
            // Road-network core numbers are almost all equal, so only the other metrics get a
            // wider "width by value" emphasis layer.
            if (metric !== 'coreNumber') {
              layers.push(
                roadLayer('network-analytics-emphasis', {
                  ...style,
                  widthPixels: 4,
                  discardAtOrBelow: METRIC_THRESHOLDS[metric]
                })
              );
            }
          }
        }
        return layers;
      },
      onClick(event) {
        if (!event.coordinate) return false;
        const node = findNearestNode(
          roads.nodePositions,
          projection.project(event.coordinate[0], event.coordinate[1])
        );
        if (view === 'route') {
          if (routeClickIsDestination) {
            destinationNode = node;
            destination.write(Uint32Array.of(node));
            writePositionBuffer(destinationPosition, node);
          } else {
            originNode = node;
            origin.write(Uint32Array.of(node));
            writePositionBuffer(originPosition, node);
          }
          routeClickIsDestination = !routeClickIsDestination;
          markDirty('route');
        } else if (view === 'service-areas') {
          const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
          let nearest = 0;
          let nearestDistance = Infinity;
          facilityNodes.forEach((facilityNode, row) => {
            const [fx, fy] = nodePosition(facilityNode);
            const distance = (fx - x) ** 2 + (fy - y) ** 2;
            if (distance < nearestDistance) {
              nearestDistance = distance;
              nearest = row;
            }
          });
          facilityNodes[nearest] = node;
          writeFacilities();
          markDirty('service-areas');
        } else if (view === 'neighborhood') {
          seedNode = node;
          seeds.write(Uint32Array.of(node));
          writePositionBuffer(seedPosition, node);
          markDirty('neighborhood');
        } else {
          return false;
        }
        return true;
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};

/** Initial sub-view from `?network=<view>`, defaulting to the route. */
function readInitialView(): View {
  const requested = new URLSearchParams(globalThis.location?.search ?? '').get('network');
  return VIEWS.find(candidate => candidate === requested) ?? 'route';
}

/** Linear scan for the node closest to `[x, y]` meters. */
function findNearestNode(nodePositions: Float32Array, [x, y]: readonly [number, number]): number {
  let nearest = 0;
  let nearestDistance = Infinity;
  for (let node = 0; node < nodePositions.length / 2; node++) {
    const distance = (nodePositions[node * 2] - x) ** 2 + (nodePositions[node * 2 + 1] - y) ** 2;
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = node;
    }
  }
  return nearest;
}

/** Directed edges sorted by source node, the row order `GPUCOOToCSR` requires. */
type SortedEdges = {sources: Uint32Array; targets: Uint32Array; lengths: Float32Array};

/** Counting-sorts the directed edge list by source node. */
function sortEdgesBySource(roads: SpatialAnalysisRoadNetwork): SortedEdges {
  const {edgeSources, edgeTargets, edgeLengths, nodePositions} = roads;
  const edgeCount = edgeSources.length;
  const nodeCount = nodePositions.length / 2;
  const starts = new Uint32Array(nodeCount + 1);
  for (let edge = 0; edge < edgeCount; edge++) starts[edgeSources[edge] + 1]++;
  for (let node = 0; node < nodeCount; node++) starts[node + 1] += starts[node];
  const sorted: SortedEdges = {
    sources: new Uint32Array(edgeCount),
    targets: new Uint32Array(edgeCount),
    lengths: new Float32Array(edgeCount)
  };
  for (let edge = 0; edge < edgeCount; edge++) {
    const slot = starts[edgeSources[edge]]++;
    sorted.sources[slot] = edgeSources[edge];
    sorted.targets[slot] = edgeTargets[edge];
    sorted.lengths[slot] = edgeLengths[edge];
  }
  return sorted;
}
