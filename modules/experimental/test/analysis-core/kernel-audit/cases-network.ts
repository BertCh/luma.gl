// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUAdjacencyMatrix,
  GPUAdjacencyMatrixOrder,
  GPUEdgeBundling,
  GPUFlowAggregation,
  GPUMapMatching,
  addDriveTimeCatchmentRecipe,
  GPUNetworkAccessibility,
  GPUNetworkAnalyticsColumns,
  GPUNetworkCoarsening,
  GPUNetworkCostWeights,
  GPUNetworkLineGraph,
  GPUNetworkNoding,
  GPUNetworkStatistics,
  GPUNetworkSubgraphFilter,
  getGPUNetworkStatisticsLength,
  getGPUNetworkSubgraphFilterParameterLength,
  GPU_NETWORK_STATISTICS_PARAMETER_LENGTH,
  GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH,
  GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY,
  GPU_NETWORK_COARSENING_SUMMARY_LENGTH,
  GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH,
  GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH,
  GPUNetworkIsochrones,
  GPUNetworkKFunction,
  GPU_NETWORK_K_FUNCTION_MAXIMUM_BAND_COUNT,
  GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH,
  GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH,
  type GPUNetworkIsochronesProps,
  GPUNetworkNeighborhood,
  GPUNetworkPathExtraction,
  GPUNetworkServiceAreas,
  GPU_NETWORK_PATH_MAXIMUM_LENGTH,
  GPUNetworkCentrality,
  GPUNetworkCostMatrix,
  GPUNetworkCostTable,
  GPUNetworkReachability,
  GPUNetworkSnapping,
  GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH,
  GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
  GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS,
  GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS,
  getGPUMapMatchingMaximumEdgeCount,
  getGPUMapMatchingMaximumEntryCapacity,
  GPU_MAP_MATCHING_PARAMETER_LENGTH,
  getGPUNetworkCoarseningMaximumWeightedEdgeCapacity,
  getGPUNetworkLineGraphMaximumEdgeCount,
  getGPUNetworkLineGraphMaximumGeodesicEdgeCount,
  getGPUNetworkNodingMaximumIntersectionCapacity,
  getGPUNetworkReachabilityMaximumNodeCount,
  getGPUNetworkSnappingMaximumCandidateEdgeCount,
  type GPUMapMatchingProps,
  type GPUFlowAggregationProps,
  GPU_EDGE_BUNDLING_MAXIMUM_ITERATIONS,
  GPU_EDGE_BUNDLING_MAXIMUM_POINTS_PER_EDGE
} from '../../../src/gpu-network';
import type {TextureFormat} from '@luma.gl/core';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const NODES = 256;
const EDGES = 1024;
const LON_LAT: GPUSpatialContext = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'};

/** A single-mip texture view of a transient texture. */
function textureView<Format extends TextureFormat>(
  {graph, texture}: KernelAuditContext,
  format: Format,
  width: number,
  height: number
) {
  return graph.createTextureView(texture(format, width, height), {mipLevelCount: 1});
}

/** A weighted CSR topology with node positions. */
function topology(
  {view}: KernelAuditContext,
  nodes: number,
  edges: number,
  options: {weights?: boolean; positions?: boolean} = {}
) {
  const {weights = true, positions = true} = options;
  return {
    offsets: view('uint32', nodes + 1, 'offsets'),
    neighbors: view('uint32', edges, 'neighbors'),
    ...(weights ? {weights: view('float32', edges, 'weights')} : {}),
    ...(positions ? {nodePositions: view('float32x2', nodes, 'positions')} : {})
  };
}

/** Largest node count the adjacency matrix accepts at `resolution` (nodeCount * resolution < 2^32). */
const ADJACENCY_MAXIMUM_NODES = Math.floor(0xffffffff / 4096);

const ADJACENCY_MATRIX_CASES: KernelAuditCase[] = [
  ...[false, true].flatMap(directed =>
    [false, true].map(
      (mirrorSlots): KernelAuditCase => ({
        exportName: 'GPUAdjacencyMatrix',
        name: `every output, directed ${directed}, mirror ${mirrorSlots}`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUAdjacencyMatrix({
              topology: topology(context, NODES, EDGES, {positions: false}),
              directed,
              mirrorSlots,
              nodeMask: view('uint32', NODES),
              edgeMask: view('uint32', EDGES),
              order: view('uint32', NODES),
              parameters: view('uint32', 4),
              resolution: 64,
              weightScale: 256,
              output: {
                counts: view('uint32', 64 * 64),
                weightSums: view('uint32', 64 * 64),
                maximumCount: view('uint32', 1),
                maximumWeightSum: view('uint32', 1),
                texture: textureView(context, 'r32float', 64, 64),
                textureStatistic: 'weightSum'
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUAdjacencyMatrix',
    name: 'count texture, no weights',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUAdjacencyMatrix({
          topology: topology(context, NODES, EDGES, {weights: false, positions: false}),
          resolution: 32,
          output: {
            counts: view('uint32', 32 * 32),
            maximumCount: view('uint32', 1),
            texture: textureView(context, 'r32float', 32, 32),
            textureStatistic: 'count'
          }
        })
      );
    }
  },
  {
    exportName: 'GPUAdjacencyMatrix',
    name: 'maximum size, resolution 4096',
    build: context => {
      const {graph, view, maximumRows} = context;
      const edges = maximumRows(4);
      graph.add(
        new GPUAdjacencyMatrix({
          topology: topology(context, ADJACENCY_MAXIMUM_NODES, edges, {positions: false}),
          directed: true,
          mirrorSlots: true,
          nodeMask: view('uint32', ADJACENCY_MAXIMUM_NODES),
          edgeMask: view('uint32', edges),
          order: view('uint32', ADJACENCY_MAXIMUM_NODES),
          parameters: view('uint32', 4),
          resolution: 4096,
          output: {
            counts: view('uint32', 4096 * 4096),
            weightSums: view('uint32', 4096 * 4096),
            maximumCount: view('uint32', 1),
            maximumWeightSum: view('uint32', 1)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUAdjacencyMatrix',
    // The fake core device caps textures at 2048 (the core limit is 8192), so the texture output
    // of the maximum-size matrix is built at resolution 2048.
    name: 'maximum size, resolution 2048 with texture',
    build: context => {
      const {graph, view, maximumRows} = context;
      const edges = maximumRows(4);
      graph.add(
        new GPUAdjacencyMatrix({
          topology: topology(context, ADJACENCY_MAXIMUM_NODES, edges, {positions: false}),
          resolution: 2048,
          output: {
            counts: view('uint32', 2048 * 2048),
            weightSums: view('uint32', 2048 * 2048),
            texture: textureView(context, 'r32float', 2048, 2048),
            textureStatistic: 'weightSum'
          }
        })
      );
    }
  }
];

const ADJACENCY_MATRIX_ORDER_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUAdjacencyMatrixOrder',
    name: 'groups and tie keys',
    build: ({graph, view}) =>
      graph.add(
        new GPUAdjacencyMatrixOrder({
          groups: view('uint32', 300),
          tieKeys: view('uint32', 300),
          output: {order: view('uint32', 300)}
        })
      )
  },
  {
    exportName: 'GPUAdjacencyMatrixOrder',
    name: 'groups only',
    build: ({graph, view}) =>
      graph.add(
        new GPUAdjacencyMatrixOrder({
          groups: view('uint32', 300),
          output: {order: view('uint32', 300)}
        })
      )
  },
  {
    exportName: 'GPUAdjacencyMatrixOrder',
    name: 'nodes at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const nodes = maximumRows(4);
      graph.add(
        new GPUAdjacencyMatrixOrder({
          groups: view('uint32', nodes),
          tieKeys: view('uint32', nodes),
          output: {order: view('uint32', nodes)}
        })
      );
    }
  }
];

const EDGE_BUNDLING_CASES: KernelAuditCase[] = [
  ...(['uint32', 'float32'] as const).map(
    (parameterFormat): KernelAuditCase => ({
      exportName: 'GPUEdgeBundling',
      name: `every option, ${parameterFormat} parameters, longitude-latitude`,
      build: ({graph, view}) =>
        graph.add(
          new GPUEdgeBundling({
            nodePositions: view('float32x2', NODES),
            sourceNodes: view('uint32', EDGES),
            targetNodes: view('uint32', EDGES),
            edgeMask: view('uint32', EDGES),
            spatialContext: LON_LAT,
            pointsPerEdge: GPU_EDGE_BUNDLING_MAXIMUM_POINTS_PER_EDGE,
            maximumIterations: GPU_EDGE_BUNDLING_MAXIMUM_ITERATIONS,
            densityResolution: 512,
            parameters: view(parameterFormat, 5),
            output: {
              paths: view('float32x2', EDGES * GPU_EDGE_BUNDLING_MAXIMUM_POINTS_PER_EDGE),
              lineOffsets: view('uint32', EDGES + 1),
              drawRecord: view('uint32', 4)
            }
          })
        )
    })
  ),
  {
    exportName: 'GPUEdgeBundling',
    name: 'defaults, planar',
    build: ({graph, view}) =>
      graph.add(
        new GPUEdgeBundling({
          nodePositions: view('float32x2', NODES),
          sourceNodes: view('uint32', EDGES),
          targetNodes: view('uint32', EDGES),
          output: {paths: view('float32x2', EDGES * 16)}
        })
      )
  },
  {
    exportName: 'GPUEdgeBundling',
    name: 'maximum edges at maximum points per edge',
    build: ({graph, view, maximumRows}) => {
      const points = GPU_EDGE_BUNDLING_MAXIMUM_POINTS_PER_EDGE;
      // edgeCount * points * 2^8 must stay below 2^32, one edge fewer than the binding allows.
      const edges = Math.min(
        Math.floor(maximumRows(8) / points),
        Math.floor(0xffffffff / 256 / points)
      );
      graph.add(
        new GPUEdgeBundling({
          nodePositions: view('float32x2', maximumRows(8)),
          sourceNodes: view('uint32', edges),
          targetNodes: view('uint32', edges),
          edgeMask: view('uint32', edges),
          spatialContext: LON_LAT,
          pointsPerEdge: points,
          maximumIterations: GPU_EDGE_BUNDLING_MAXIMUM_ITERATIONS,
          densityResolution: 4096,
          parameters: view('uint32', 5),
          output: {
            paths: view('float32x2', edges * points),
            lineOffsets: view('uint32', edges + 1),
            drawRecord: view('uint32', 4)
          }
        })
      );
    }
  }
];

const FLOW_ROWS = 600;

/** Every optional output of a flow aggregation with `topCount` ranked flows over `zones` zones. */
function flowOutput(
  {view}: KernelAuditContext,
  topCount: number,
  zones: number,
  withZoneWeights: boolean
): GPUFlowAggregationProps['output'] {
  return {
    ids: view('uint32', topCount),
    status: {
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1),
      pairOverflow: view('uint32', 1)
    },
    flowOriginZoneIds: view('uint32', topCount),
    flowDestinationZoneIds: view('uint32', topCount),
    flowCounts: view('uint32', topCount),
    flowWeights: view('float32', topCount),
    zoneOutCounts: view('uint32', zones),
    zoneInCounts: view('uint32', zones),
    ...(withZoneWeights
      ? {zoneOutWeights: view('float32', zones), zoneInWeights: view('float32', zones)}
      : {}),
    drawInstanceCount: view('uint32', 1),
    zoneOutCountExtent: view('float32', 2),
    zoneInCountExtent: view('float32', 2)
  };
}

const FLOW_AGGREGATION_CASES: KernelAuditCase[] = [
  ...(['sorted', 'atomic'] as const).map(
    (sumOrder): KernelAuditCase => ({
      exportName: 'GPUFlowAggregation',
      name: `grid zones, ${sumOrder} sums, float time window with low parts, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUFlowAggregation({
            zones: {
              kind: 'grid',
              bounds: [0, 0, 100, 100],
              gridSize: [32, 32],
              activeGridSize: view('uint32', 2)
            },
            origins: view('float32x2', FLOW_ROWS),
            destinations: view('float32x2', FLOW_ROWS),
            sumOrder,
            weights: view('float32', FLOW_ROWS),
            mask: view('uint32', FLOW_ROWS),
            timeWindow: {
              timestamps: view('float32', FLOW_ROWS),
              timestampsLow: view('float32', FLOW_ROWS),
              window: view('float32', 8)
            },
            excludeSelfFlows: true,
            pairCapacity: 1024,
            maximumProbeCount: 64,
            output: flowOutput(context, 128, 1024, true)
          })
        );
      }
    })
  ),
  ...(['sorted', 'atomic'] as const).flatMap(sumOrder =>
    (['float32x4', 'float32'] as const).map(
      (boundsFormat): KernelAuditCase => ({
        exportName: 'GPUFlowAggregation',
        name: `hexagon zones, ${sumOrder} sums, ${boundsFormat} bounds view, view radius, word time window`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUFlowAggregation({
              zones: {
                kind: 'hexagon',
                bounds: boundsFormat === 'float32x4' ? view('float32x4', 1) : view('float32', 4),
                gridSize: [64, 48],
                activeGridSize: view('uint32', 2),
                radius: view('float32', 1)
              },
              origins: view('float32x2', FLOW_ROWS),
              destinations: view('float32x2', FLOW_ROWS),
              sumOrder,
              weights: view('float32', FLOW_ROWS),
              mask: view('uint32', FLOW_ROWS),
              timeWindow: {
                timestamps: view('uint32x2', FLOW_ROWS),
                window: view('uint32', 8)
              },
              excludeSelfFlows: true,
              pairCapacity: 4096,
              output: flowOutput(context, 256, 64 * 48, true)
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUFlowAggregation',
    name: 'hexagon zones, literal radius and bounds, no weights',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUFlowAggregation({
          zones: {kind: 'hexagon', bounds: [-180, -90, 180, 90], gridSize: [200, 100], radius: 1.5},
          origins: view('float32x2', FLOW_ROWS),
          destinations: view('float32x2', FLOW_ROWS),
          pairCapacity: 2048,
          output: flowOutput(context, 64, 20000, false)
        })
      );
    }
  },
  ...(['sorted', 'atomic'] as const).map(
    (sumOrder): KernelAuditCase => ({
      exportName: 'GPUFlowAggregation',
      name: `ids zones at the maximum zone count, ${sumOrder} sums, every option`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUFlowAggregation({
            zones: {kind: 'ids', zoneCount: 65535},
            originZoneIds: view('uint32', FLOW_ROWS),
            destinationZoneIds: view('uint32', FLOW_ROWS),
            sumOrder,
            weights: view('float32', FLOW_ROWS),
            mask: view('uint32', FLOW_ROWS),
            timeWindow: {timestamps: view('float32', FLOW_ROWS), window: view('float32', 8)},
            excludeSelfFlows: true,
            pairCapacity: 1024,
            output: flowOutput(context, 100, 65535, true)
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUFlowAggregation',
    // Review B-18: the sorted pair sum launches 256 lanes per table slot, so with weights the
    // constructor accepts at most 2^23 slots (2^24 * 256 = 2^32 is one lane too many, although its
    // message says "at most 2^24"). With the default maximumProbeCount (= pairCapacity) and rows at
    // the binding limit, rows * maximumProbeCount overflows uint32 and the constructor refuses it.
    name: 'maximum pairCapacity 2^23 with weights and sorted sums, rows at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(8);
      const capacity = 2 ** 23;
      graph.add(
        new GPUFlowAggregation({
          zones: {kind: 'ids', zoneCount: 65535},
          originZoneIds: view('uint32', rows),
          destinationZoneIds: view('uint32', rows),
          weights: view('float32', rows),
          mask: view('uint32', rows),
          timeWindow: {timestamps: view('float32', rows), window: view('float32', 8)},
          excludeSelfFlows: true,
          pairCapacity: capacity,
          output: flowOutput(context, maximumRows(4), 65535, true)
        })
      );
    }
  },
  {
    exportName: 'GPUFlowAggregation',
    // rows * maximumProbeCount = 2^24 * 255 is the largest product below 2^32.
    name: 'maximum pairCapacity 2^23 with weights and sorted sums, rows x probe limit at the maximum',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(8);
      graph.add(
        new GPUFlowAggregation({
          zones: {kind: 'ids', zoneCount: 65535},
          originZoneIds: view('uint32', rows),
          destinationZoneIds: view('uint32', rows),
          weights: view('float32', rows),
          mask: view('uint32', rows),
          timeWindow: {timestamps: view('float32', rows), window: view('float32', 8)},
          excludeSelfFlows: true,
          pairCapacity: 2 ** 23,
          maximumProbeCount: Math.floor(0xffffffff / rows),
          output: flowOutput(context, maximumRows(4), 65535, true)
        })
      );
    }
  },
  {
    exportName: 'GPUFlowAggregation',
    name: 'maximum pairCapacity 2^23 with weights and sorted sums, probe limit 64',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(8);
      graph.add(
        new GPUFlowAggregation({
          zones: {kind: 'ids', zoneCount: 65535},
          originZoneIds: view('uint32', rows),
          destinationZoneIds: view('uint32', rows),
          weights: view('float32', rows),
          mask: view('uint32', rows),
          timeWindow: {timestamps: view('float32', rows), window: view('float32', 8)},
          excludeSelfFlows: true,
          pairCapacity: 2 ** 23,
          maximumProbeCount: 64,
          output: flowOutput(context, maximumRows(4), 65535, true)
        })
      );
    }
  },
  {
    exportName: 'GPUFlowAggregation',
    name: 'maximum pairCapacity at the binding limit with atomic sums and a probe limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(8);
      graph.add(
        new GPUFlowAggregation({
          zones: {kind: 'ids', zoneCount: 65535},
          originZoneIds: view('uint32', rows),
          destinationZoneIds: view('uint32', rows),
          weights: view('float32', rows),
          sumOrder: 'atomic',
          pairCapacity: 2 ** 25,
          maximumProbeCount: 64,
          output: flowOutput(context, maximumRows(4), 65535, true)
        })
      );
    }
  },
  {
    exportName: 'GPUFlowAggregation',
    name: 'maximum pairCapacity at the binding limit without weights',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(8);
      graph.add(
        new GPUFlowAggregation({
          zones: {kind: 'ids', zoneCount: 65535},
          originZoneIds: view('uint32', rows),
          destinationZoneIds: view('uint32', rows),
          pairCapacity: 2 ** 25,
          maximumProbeCount: 64,
          output: flowOutput(context, maximumRows(4), 65535, false)
        })
      );
    }
  }
];

/** Every optional output of a map matching. */
function mapMatchingOutput(
  {view}: KernelAuditContext,
  points: number,
  tracks: number
): GPUMapMatchingProps['output'] {
  return {
    matchedEdges: view('uint32', points),
    matchedFractions: view('float32', points),
    matchedOffsets: view('float32', points),
    snapDistances: view('float32', points),
    snappedPositions: view('float32x2', points),
    breaks: view('uint32', points),
    trackLogLikelihoods: view('float32', tracks),
    matchedCount: view('uint32', 1),
    breakCount: view('uint32', 1),
    status: {overflow: view('uint32', 1), routeOverflow: view('uint32', 1)}
  };
}

const MAP_MATCHING_CASES: KernelAuditCase[] = [
  ...[
    {name: 'planar', spatialContext: undefined, bounds: {minimum: [0, 0], maximum: [5000, 5000]}},
    {
      name: 'great-circle',
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
      bounds: {minimum: [-180, -85], maximum: [181, 85]}
    }
  ].map(
    ({name, spatialContext, bounds}): KernelAuditCase => ({
      exportName: 'GPUMapMatching',
      name: `${name}, 8 candidates, route budget 128, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUMapMatching({
            spatialContext: spatialContext as GPUSpatialContext | undefined,
            positions: view('float32x2', 1000),
            trackOffsets: view('uint32', 21),
            topology: topology(context, NODES, EDGES, {weights: false}) as never,
            parameters: view('float32', GPU_MAP_MATCHING_PARAMETER_LENGTH),
            candidateCount: 8,
            cellSize: 50,
            bounds: bounds as never,
            entryCapacity: 20000,
            routeNodeBudget: 128,
            output: mapMatchingOutput(context, 1000, 20)
          })
        );
      }
    })
  ),
  ...[1, 4].map(
    (candidateCount): KernelAuditCase => ({
      exportName: 'GPUMapMatching',
      name: `${candidateCount} candidates, default budget, minimal output`,
      build: ({graph, view}) =>
        graph.add(
          new GPUMapMatching({
            positions: view('float32x2', 1000),
            trackOffsets: view('uint32', 21),
            topology: {
              offsets: view('uint32', NODES + 1),
              neighbors: view('uint32', EDGES),
              nodePositions: view('float32x2', NODES)
            },
            parameters: view('float32', GPU_MAP_MATCHING_PARAMETER_LENGTH),
            candidateCount,
            cellSize: 100,
            bounds: {minimum: [0, 0], maximum: [1000, 1000]},
            output: {matchedEdges: view('uint32', 1000)}
          })
        )
    })
  ),
  ...[1, 8].map(
    (candidateCount): KernelAuditCase => ({
      exportName: 'GPUMapMatching',
      name: `${candidateCount} candidates, route table, route budget 128, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUMapMatching({
            positions: view('float32x2', 1000),
            trackOffsets: view('uint32', 21),
            topology: topology(context, NODES, EDGES, {weights: false}) as never,
            parameters: view('float32', GPU_MAP_MATCHING_PARAMETER_LENGTH),
            candidateCount,
            cellSize: 50,
            bounds: {minimum: [0, 0], maximum: [1000, 1000]},
            routeNodeBudget: 128,
            routeTable: {
              offsets: view('uint32', NODES + 1),
              targets: view('uint32', 4096),
              costs: view('float32', 4096),
              costLimit: view('float32', 1),
              status: {overflow: view('uint32', 1)}
            },
            output: mapMatchingOutput(context, 1000, 20)
          })
        );
      }
    })
  ),
  {
    // Largest per-point scratch is the route table, 8 * 9 floats = 288 bytes per point.
    exportName: 'GPUMapMatching',
    name: 'points at the binding limit, 8 candidates, grid at 2^24 cells',
    build: context => {
      const {graph, view, maximumRows} = context;
      const points = maximumRows(288);
      const edges = Math.floor((maximumRows(4) - 256) / 16);
      graph.add(
        new GPUMapMatching({
          positions: view('float32x2', points),
          trackOffsets: view('uint32', 4097),
          topology: {
            offsets: view('uint32', maximumRows(8) + 1),
            neighbors: view('uint32', edges),
            nodePositions: view('float32x2', maximumRows(8))
          },
          parameters: view('float32', GPU_MAP_MATCHING_PARAMETER_LENGTH),
          candidateCount: 8,
          cellSize: 1,
          bounds: {minimum: [0, 0], maximum: [4096, 4096]},
          routeNodeBudget: 128,
          output: mapMatchingOutput(context, points, 4096)
        })
      );
    }
  },
  {
    // The default entryCapacity (16 * edges + 256 words) is capped at the maximum entry capacity.
    exportName: 'GPUMapMatching',
    name: 'edges at the maximum, default entryCapacity',
    build: context => {
      const {graph, view, maximumRows} = context;
      const edges = getGPUMapMatchingMaximumEdgeCount(context.device);
      graph.add(
        new GPUMapMatching({
          positions: view('float32x2', 1000),
          trackOffsets: view('uint32', 21),
          topology: {
            offsets: view('uint32', maximumRows(8) + 1),
            neighbors: view('uint32', edges),
            nodePositions: view('float32x2', maximumRows(8))
          },
          parameters: view('float32', GPU_MAP_MATCHING_PARAMETER_LENGTH),
          cellSize: 100,
          bounds: {minimum: [0, 0], maximum: [100000, 100000]},
          output: mapMatchingOutput(context, 1000, 20)
        })
      );
    }
  }
];

/** One map matching over `edges` edges with an explicit `entryCapacity`. */
function mapMatchingEdgesCase(
  name: string,
  getEdges: (device: KernelAuditContext['device']) => number,
  getEntryCapacity: (device: KernelAuditContext['device']) => number | undefined
): KernelAuditCase {
  return {
    exportName: 'GPUMapMatching',
    name,
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUMapMatching({
          positions: view('float32x2', 1000),
          trackOffsets: view('uint32', 21),
          topology: {
            offsets: view('uint32', 1025),
            neighbors: view('uint32', getEdges(context.device)),
            nodePositions: view('float32x2', 1024)
          },
          parameters: view('float32', GPU_MAP_MATCHING_PARAMETER_LENGTH),
          cellSize: 100,
          bounds: {minimum: [0, 0], maximum: [100000, 100000]},
          entryCapacity: getEntryCapacity(context.device),
          output: mapMatchingOutput(context, 1000, 20)
        })
      );
    }
  };
}

MAP_MATCHING_CASES.push(
  mapMatchingEdgesCase(
    'entryCapacity at the maximum',
    () => 1024,
    device => getGPUMapMatchingMaximumEntryCapacity(device)
  ),
  mapMatchingEdgesCase(
    'entryCapacity above the maximum',
    () => 1024,
    device => getGPUMapMatchingMaximumEntryCapacity(device) + 1
  ),
  mapMatchingEdgesCase(
    'edges above the maximum',
    device => getGPUMapMatchingMaximumEdgeCount(device) + 1,
    () => undefined
  )
);

const SNAPPING_POINTS = 500;

/** Every output of a snapping contributor with `points` points and optionally the BVH overflow flag. */
function snappingOutput({view}: KernelAuditContext, points: number, withOverflow: boolean) {
  return {
    snappedEdges: view('uint32', points),
    snapFractions: view('float32', points),
    snapDistances: view('float32', points),
    sourceCosts: view('float32', points),
    targetCosts: view('float32', points),
    snappedPositions: view('float32x2', points),
    startNodes: view('uint32', 2 * points),
    startCosts: view('float32', 2 * points),
    ...(withOverflow ? {status: {overflow: view('uint32', 1)}} : {})
  };
}

const NETWORK_SNAPPING_CASES: KernelAuditCase[] = [
  ...(['both', 'forward', 'reverse'] as const).flatMap(startDirection =>
    [undefined, LON_LAT].map(
      (spatialContext): KernelAuditCase => ({
        exportName: 'GPUNetworkSnapping',
        name: `exact scan, ${startDirection}, ${spatialContext ? 'great-circle' : 'planar'}, every output`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUNetworkSnapping({
              spatialContext,
              positions: view('float32x2', SNAPPING_POINTS),
              topology: topology(context, NODES, EDGES) as never,
              maximumSnapDistance: view('float32', 1),
              startDirection,
              output: snappingOutput(context, SNAPPING_POINTS, false)
            })
          );
        }
      })
    )
  ),
  ...[false, true].flatMap(spatialSort =>
    [undefined, LON_LAT].map(
      (spatialContext): KernelAuditCase => ({
        exportName: 'GPUNetworkSnapping',
        name: `candidate search, spatial sort ${spatialSort}, ${spatialContext ? 'great-circle' : 'planar'}, every output`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUNetworkSnapping({
              spatialContext,
              positions: view('float32x2', SNAPPING_POINTS),
              topology: topology(context, NODES, EDGES) as never,
              maximumSnapDistance: view('float32', 1),
              candidateCapacity: 4096,
              spatialSort,
              output: snappingOutput(context, SNAPPING_POINTS, true)
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUNetworkSnapping',
    name: 'exact scan, points and edges at the binding limit, no weights',
    build: context => {
      const {graph, view, maximumRows} = context;
      const points = maximumRows(8);
      graph.add(
        new GPUNetworkSnapping({
          positions: view('float32x2', points),
          topology: topology(context, maximumRows(8), maximumRows(4), {weights: false}) as never,
          output: snappingOutput(context, points, false)
        })
      );
    }
  },
  ...[0, 1].map(
    (overMaximum): KernelAuditCase => ({
      exportName: 'GPUNetworkSnapping',
      // The segment-start table takes 8 bytes per edge, so one binding holds the segments of at
      // most getGPUNetworkSnappingMaximumCandidateEdgeCount edges.
      name: overMaximum
        ? 'candidate search, edges above the maximum'
        : 'candidate search, points and candidate capacity at the binding limit, edges at the maximum',
      build: context => {
        const {graph, view, maximumRows} = context;
        const points = maximumRows(8);
        graph.add(
          new GPUNetworkSnapping({
            positions: view('float32x2', points),
            topology: topology(
              context,
              maximumRows(8),
              getGPUNetworkSnappingMaximumCandidateEdgeCount(context.device) + overMaximum
            ) as never,
            maximumSnapDistance: view('float32', 1),
            candidateCapacity: maximumRows(8),
            spatialSort: true,
            startDirection: 'forward',
            output: snappingOutput(context, points, true)
          })
        );
      }
    })
  )
];

const COST_MATRIX_ROWS = 40;

const NETWORK_COST_MATRIX_CASES: KernelAuditCase[] = [
  ...[1, 7, COST_MATRIX_ROWS].map(
    (laneCount): KernelAuditCase => ({
      exportName: 'GPUNetworkCostMatrix',
      name: `lane count ${laneCount}, target nodes, every option`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkCostMatrix({
            topology: topology(context, NODES, EDGES, {positions: false}) as never,
            startNodes: view('uint32', 2 * COST_MATRIX_ROWS),
            startCosts: view('float32', 2 * COST_MATRIX_ROWS),
            startRows: view('uint32', 2 * COST_MATRIX_ROWS),
            startsPerRow: 2,
            targetNodes: view('uint32', 17),
            costLimit: view('float32', 1),
            laneCount,
            maximumIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
            localIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS,
            output: {
              costs: view('float32', COST_MATRIX_ROWS * 17),
              status: {converged: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  ...[1, 7, COST_MATRIX_ROWS].map(
    (laneCount): KernelAuditCase => ({
      exportName: 'GPUNetworkCostMatrix',
      name: `lane count ${laneCount}, every option`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkCostMatrix({
            topology: topology(context, NODES, EDGES, {positions: false}) as never,
            startNodes: view('uint32', 2 * COST_MATRIX_ROWS),
            startCosts: view('float32', 2 * COST_MATRIX_ROWS),
            startRows: view('uint32', 2 * COST_MATRIX_ROWS),
            startsPerRow: 2,
            costLimit: view('float32', 1),
            laneCount,
            maximumIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
            localIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS,
            output: {
              costs: view('float32', COST_MATRIX_ROWS * NODES),
              status: {converged: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkCostMatrix',
    name: 'defaults, start rows derived from startsPerRow',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkCostMatrix({
          topology: topology(context, NODES, EDGES, {positions: false}) as never,
          startNodes: view('uint32', COST_MATRIX_ROWS),
          output: {costs: view('float32', COST_MATRIX_ROWS * NODES)}
        })
      );
    }
  },
  {
    exportName: 'GPUNetworkCostMatrix',
    // The frontier queues take 8 bytes per lane-expanded node, so one batch holds at most
    // maximumRows(8) expanded nodes; the matrix itself fits one binding at maximumRows(4) entries.
    name: 'matrix at the binding limit, lanes at the frontier limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const nodes = 1 << 20;
      const rows = Math.floor(maximumRows(4) / nodes);
      const laneCount = Math.floor(maximumRows(8) / nodes);
      graph.add(
        new GPUNetworkCostMatrix({
          topology: topology(context, nodes, maximumRows(4), {positions: false}) as never,
          // The expanded seeds hold startNodes.length words per batch.
          startNodes: view('uint32', Math.floor(maximumRows(4) / Math.ceil(rows / laneCount))),
          startCosts: view('float32', Math.floor(maximumRows(4) / Math.ceil(rows / laneCount))),
          startRows: view('uint32', Math.floor(maximumRows(4) / Math.ceil(rows / laneCount))),
          costLimit: view('float32', 1),
          laneCount,
          maximumIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
          localIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS,
          output: {
            costs: view('float32', rows * nodes),
            status: {converged: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUNetworkCostMatrix',
    name: 'one row of the largest network',
    build: context => {
      const {graph, view, maximumRows} = context;
      const nodes = maximumRows(8);
      graph.add(
        new GPUNetworkCostMatrix({
          topology: topology(context, nodes, maximumRows(4), {positions: false}) as never,
          startNodes: view('uint32', 1),
          output: {costs: view('float32', nodes)}
        })
      );
    }
  }
];

const ACCESSIBILITY_ROWS = 32;
const ACCESSIBILITY_NODES = 300;

const NETWORK_ACCESSIBILITY_CASES: KernelAuditCase[] = [
  ...(['opportunity-rows', 'origin-rows'] as const).map(
    (orientation): KernelAuditCase => ({
      exportName: 'GPUNetworkAccessibility',
      name: `${orientation}, cumulative and gravity`,
      build: ({graph, view}) => {
        const isOpportunityRows = orientation === 'opportunity-rows';
        const origins = isOpportunityRows ? ACCESSIBILITY_NODES : ACCESSIBILITY_ROWS;
        graph.add(
          new GPUNetworkAccessibility({
            costs: view('float32', ACCESSIBILITY_ROWS * ACCESSIBILITY_NODES),
            orientation,
            opportunityWeights: view(
              'float32',
              isOpportunityRows ? ACCESSIBILITY_ROWS : ACCESSIBILITY_NODES
            ),
            parameters: view('float32', GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH),
            cumulative: view('float32', origins),
            gravity: view('float32', origins)
          })
        );
      }
    })
  ),
  ...(['2sfca', '3sfca'] as const).flatMap(method =>
    [false, true].map(
      (withRatios): KernelAuditCase => ({
        exportName: 'GPUNetworkAccessibility',
        name: `${method} catchment, ratios ${withRatios}, cumulative and gravity`,
        build: ({graph, view}) =>
          graph.add(
            new GPUNetworkAccessibility({
              costs: view('float32', ACCESSIBILITY_ROWS * ACCESSIBILITY_NODES),
              opportunityWeights: view('float32', ACCESSIBILITY_ROWS),
              parameters: view('float32', GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH),
              cumulative: view('float32', ACCESSIBILITY_NODES),
              gravity: view('float32', ACCESSIBILITY_NODES),
              catchment: {
                method,
                demand: view('float32', ACCESSIBILITY_NODES),
                output: view('float32', ACCESSIBILITY_NODES),
                ...(withRatios ? {ratios: view('float32', ACCESSIBILITY_ROWS)} : {})
              }
            })
          )
      })
    )
  ),
  {
    exportName: 'GPUNetworkAccessibility',
    name: 'matrix at the binding limit with catchment',
    build: ({graph, view, maximumRows}) => {
      const rows = 4096;
      const nodes = Math.floor(maximumRows(4) / rows);
      graph.add(
        new GPUNetworkAccessibility({
          costs: view('float32', rows * nodes),
          opportunityWeights: view('float32', rows),
          parameters: view('float32', GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH),
          cumulative: view('float32', nodes),
          gravity: view('float32', nodes),
          catchment: {
            method: '3sfca',
            demand: view('float32', nodes),
            output: view('float32', nodes),
            ratios: view('float32', rows)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUNetworkAccessibility',
    name: 'origin rows at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = 4096;
      const nodes = Math.floor(maximumRows(4) / rows);
      graph.add(
        new GPUNetworkAccessibility({
          costs: view('float32', rows * nodes),
          orientation: 'origin-rows',
          opportunityWeights: view('float32', nodes),
          parameters: view('float32', GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH),
          cumulative: view('float32', rows),
          gravity: view('float32', rows)
        })
      );
    }
  }
];

const REACHABILITY_NODES = 1000;

/** Every optional output of one single-lane reachability over `nodes` nodes and `breaks` bands. */
function reachabilityOutput({view}: KernelAuditContext, nodes: number, breaks: number) {
  return {
    costs: view('float32', nodes),
    predecessors: view('uint32', nodes),
    bands: view('uint32', nodes),
    bandCounts: view('uint32', breaks),
    unresolvedCount: view('uint32', 1),
    status: {
      converged: view('uint32', 1),
      iterationCount: view('uint32', 1),
      relaxationCount: view('uint32', 2),
      reachedEdgeCount: view('uint32', 2)
    }
  };
}

const NETWORK_REACHABILITY_CASES: KernelAuditCase[] = [
  ...[1, 32, GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS].map(
    (localIterations): KernelAuditCase => ({
      exportName: 'GPUNetworkReachability',
      name: `local iterations ${localIterations}, maximum rounds, every output, view start count`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkReachability({
            topology: topology(context, REACHABILITY_NODES, EDGES * 4, {positions: false}) as never,
            startNodes: view('uint32', 256),
            startCosts: view('float32', 256),
            startCount: view('uint32', 1),
            maximumIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
            localIterations,
            maximumTieIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS,
            activeIterations: view('uint32', 1),
            costLimit: view('float32', 1),
            costBreaks: view('float32', 16),
            output: reachabilityOutput(context, REACHABILITY_NODES, 16)
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkReachability',
    name: 'literal start count, defaults, costs only',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkReachability({
          topology: topology(context, REACHABILITY_NODES, EDGES, {positions: false}) as never,
          startNodes: view('uint32', 8),
          startCount: 8,
          output: {costs: view('float32', REACHABILITY_NODES)}
        })
      );
    }
  },
  ...[2, 64].map(
    (laneCount): KernelAuditCase => ({
      exportName: 'GPUNetworkReachability',
      name: `${laneCount} lanes, bands and status`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkReachability({
            topology: topology(context, REACHABILITY_NODES, EDGES, {positions: false}) as never,
            startNodes: view('uint32', 256),
            startCosts: view('float32', 256),
            laneCount,
            maximumIterations: 128,
            localIterations: 16,
            costLimit: view('float32', 1),
            costBreaks: view('float32', 8),
            output: {
              costs: view('float32', laneCount * REACHABILITY_NODES),
              bands: view('uint32', laneCount * REACHABILITY_NODES),
              bandCounts: view('uint32', 8),
              status: {
                converged: view('uint32', 1),
                iterationCount: view('uint32', 1),
                relaxationCount: view('uint32', 2),
                reachedEdgeCount: view('uint32', 2)
              }
            }
          })
        );
      }
    })
  ),
  ...[0, 1].map(
    (overMaximum): KernelAuditCase => ({
      exportName: 'GPUNetworkReachability',
      // The frontier queues take 8 bytes per node; nodes beyond the maximum overflow one binding.
      name: overMaximum
        ? 'nodes above the maximum'
        : 'nodes at the maximum, edges and starts at the binding limit, every output',
      build: context => {
        const {graph, view, maximumRows} = context;
        const nodes = getGPUNetworkReachabilityMaximumNodeCount(context.device) + overMaximum;
        graph.add(
          new GPUNetworkReachability({
            topology: topology(context, nodes, maximumRows(4), {positions: false}) as never,
            startNodes: view('uint32', maximumRows(4)),
            startCosts: view('float32', maximumRows(4)),
            startCount: view('uint32', 1),
            maximumIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
            localIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS,
            maximumTieIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS,
            activeIterations: view('uint32', 1),
            costLimit: view('float32', 1),
            costBreaks: view('float32', 1024),
            output: reachabilityOutput(context, nodes, 1024)
          })
        );
      }
    })
  )
];

const PATH_NODES = 1000;
const PATH_TARGETS = 64;

/** Every optional input and output of one path extraction. */
function pathExtractionProps(
  context: KernelAuditContext,
  nodes: number,
  targets: number,
  pathCapacity: number,
  edgeCount: number,
  maximumPathLength: number
): ConstructorParameters<typeof GPUNetworkPathExtraction>[0] {
  const {view} = context;
  return {
    predecessors: view('uint32', nodes),
    costs: view('float32', nodes),
    targets: view('uint32', targets),
    targetCount: view('uint32', 1),
    predecessorsComplete: view('uint32', 1),
    nodeIds: view('uint32', nodes),
    maximumPathLength,
    output: {
      ids: view('uint32', pathCapacity),
      status: {
        count: view('uint32', 1),
        overflow: view('uint32', 1),
        requiredCount: view('uint32', 1)
      },
      lineOffsets: view('uint32', targets + 1),
      pathCosts: view('float32', targets),
      pathFound: view('uint32', targets)
    },
    edges: {
      topology: topology(context, nodes, edgeCount, {positions: false}) as never,
      edgeIds: view('uint32', edgeCount),
      output: {
        ids: view('uint32', pathCapacity),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          requiredCount: view('uint32', 1)
        },
        lineOffsets: view('uint32', targets + 1)
      }
    }
  };
}

const NETWORK_PATH_EXTRACTION_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUNetworkPathExtraction',
    name: 'every option, default path length',
    build: context =>
      context.graph.add(
        new GPUNetworkPathExtraction(
          pathExtractionProps(context, PATH_NODES, PATH_TARGETS, 4096, EDGES, 1024)
        )
      )
  },
  {
    exportName: 'GPUNetworkPathExtraction',
    name: 'every option, maximum path length',
    build: context =>
      context.graph.add(
        new GPUNetworkPathExtraction(
          pathExtractionProps(
            context,
            PATH_NODES,
            PATH_TARGETS,
            4096,
            EDGES,
            GPU_NETWORK_PATH_MAXIMUM_LENGTH
          )
        )
      )
  },
  {
    exportName: 'GPUNetworkPathExtraction',
    name: 'literal target count, nodes only, minimal path length',
    build: ({graph, view}) =>
      graph.add(
        new GPUNetworkPathExtraction({
          predecessors: view('uint32', PATH_NODES),
          costs: view('float32', PATH_NODES),
          targets: view('uint32', PATH_TARGETS),
          targetCount: 10,
          maximumPathLength: 1,
          output: {
            ids: view('uint32', 128),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        })
      )
  },
  {
    exportName: 'GPUNetworkPathExtraction',
    name: 'nodes, targets, path capacity and edges at the binding limit',
    build: context => {
      const {maximumRows} = context;
      context.graph.add(
        new GPUNetworkPathExtraction(
          pathExtractionProps(
            context,
            maximumRows(4) - 1,
            maximumRows(4) - 1,
            maximumRows(4),
            maximumRows(4),
            GPU_NETWORK_PATH_MAXIMUM_LENGTH
          )
        )
      );
    }
  }
];

const SERVICE_AREA_NODES = 1000;

const NETWORK_SERVICE_AREAS_CASES: KernelAuditCase[] = [
  ...[
    {name: 'view facility count', view: true},
    {name: 'literal facility count', view: false}
  ].map(
    ({name, view: isView}): KernelAuditCase => ({
      exportName: 'GPUNetworkServiceAreas',
      name: `${name}, maximum rounds, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkServiceAreas({
            topology: topology(context, SERVICE_AREA_NODES, EDGES * 4, {positions: false}) as never,
            facilities: view('uint32', 64),
            facilityCosts: view('float32', 64),
            facilityCount: isView ? view('uint32', 1) : 40,
            costLimit: view('float32', 1),
            maximumIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
            maximumLabelIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
            localIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS,
            activeIterations: view('uint32', 1),
            maximumTieIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS,
            output: {
              assignments: view('uint32', SERVICE_AREA_NODES),
              costs: view('float32', SERVICE_AREA_NODES),
              predecessors: view('uint32', SERVICE_AREA_NODES),
              facilityNodeCounts: view('uint32', 64),
              facilityCostSums: view('float32', 64),
              status: {converged: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkServiceAreas',
    name: 'defaults, assignments only, local iterations 1',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkServiceAreas({
          topology: topology(context, SERVICE_AREA_NODES, EDGES, {positions: false}) as never,
          facilities: view('uint32', 8),
          localIterations: 1,
          output: {assignments: view('uint32', SERVICE_AREA_NODES)}
        })
      );
    }
  },
  {
    exportName: 'GPUNetworkServiceAreas',
    // The reachability frontier queues take 8 bytes per node, so the nodes fit one binding up to
    // maximumRows(8).
    name: 'nodes at the frontier limit, edges and facilities at the binding limit, every output',
    build: context => {
      const {graph, view, maximumRows} = context;
      const nodes = maximumRows(8);
      graph.add(
        new GPUNetworkServiceAreas({
          topology: topology(context, nodes, maximumRows(4), {positions: false}) as never,
          facilities: view('uint32', maximumRows(4)),
          facilityCosts: view('float32', maximumRows(4)),
          facilityCount: view('uint32', 1),
          costLimit: view('float32', 1),
          maximumIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
          maximumLabelIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
          localIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS,
          activeIterations: view('uint32', 1),
          maximumTieIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_TIE_ITERATIONS,
          output: {
            assignments: view('uint32', nodes),
            costs: view('float32', nodes),
            predecessors: view('uint32', nodes),
            facilityNodeCounts: view('uint32', maximumRows(4)),
            facilityCostSums: view('float32', maximumRows(4)),
            status: {converged: view('uint32', 1)}
          }
        })
      );
    }
  }
];

const NEIGHBORHOOD_NODES = 1000;

/** Every optional output of one neighborhood. */
function neighborhoodOutput(
  {view}: KernelAuditContext,
  nodes: number,
  edges: number,
  capacity: number
) {
  const compact = () => ({
    ids: view('uint32', capacity),
    status: {
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1)
    }
  });
  return {
    hopDistances: view('uint32', nodes),
    nodeMask: view('uint32', nodes),
    edgeMask: view('uint32', edges),
    nodes: compact(),
    edges: compact()
  };
}

const NETWORK_NEIGHBORHOOD_CASES: KernelAuditCase[] = [
  ...[8, GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS].map(
    (maximumHops): KernelAuditCase => ({
      exportName: 'GPUNetworkNeighborhood',
      name: `maximum hops ${maximumHops}, every option`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkNeighborhood({
            topology: topology(context, NEIGHBORHOOD_NODES, EDGES * 4, {
              weights: false,
              positions: false
            }) as never,
            startNodes: view('uint32', 64),
            startCount: view('uint32', 1),
            hops: view('uint32', 1),
            maximumHops,
            nodeIds: view('uint32', NEIGHBORHOOD_NODES),
            edgeIds: view('uint32', EDGES * 4),
            output: neighborhoodOutput(context, NEIGHBORHOOD_NODES, EDGES * 4, 2048)
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkNeighborhood',
    name: 'literal start count, hop distances only',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkNeighborhood({
          topology: topology(context, NEIGHBORHOOD_NODES, EDGES, {
            weights: false,
            positions: false
          }) as never,
          startNodes: view('uint32', 64),
          startCount: 5,
          hops: view('uint32', 1),
          output: {hopDistances: view('uint32', NEIGHBORHOOD_NODES)}
        })
      );
    }
  },
  {
    exportName: 'GPUNetworkNeighborhood',
    // The reachability frontier queues take 8 bytes per node, so the nodes fit one binding up to
    // maximumRows(8).
    name: 'nodes at the frontier limit, edges and compact outputs at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      graph.add(
        new GPUNetworkNeighborhood({
          topology: topology(context, maximumRows(8), maximumRows(4), {
            weights: false,
            positions: false
          }) as never,
          startNodes: view('uint32', maximumRows(4)),
          startCount: view('uint32', 1),
          hops: view('uint32', 1),
          maximumHops: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
          nodeIds: view('uint32', maximumRows(8)),
          edgeIds: view('uint32', maximumRows(4)),
          output: neighborhoodOutput(context, maximumRows(8), maximumRows(4), maximumRows(4))
        })
      );
    }
  }
];

const ANALYTICS_NODES = 500;

/** A column with its normalized and extent companions. */
function analyticsColumn<Format extends 'uint32' | 'float32'>(
  {view}: KernelAuditContext,
  format: Format,
  nodes: number
) {
  return {
    output: view(format, nodes),
    normalized: view('float32', nodes),
    extent: view(format, 2)
  };
}

const NETWORK_ANALYTICS_COLUMNS_CASES: KernelAuditCase[] = [
  ...[false, true].map(
    (directed): KernelAuditCase => ({
      exportName: 'GPUNetworkAnalyticsColumns',
      name: `every metric with normalization, ${directed ? 'directed' : 'undirected'}`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkAnalyticsColumns({
            topology: topology(context, ANALYTICS_NODES, EDGES, {
              weights: false,
              positions: false
            }) as never,
            ...(directed
              ? {
                  reverseTopology: {
                    offsets: view('uint32', ANALYTICS_NODES + 1),
                    neighbors: view('uint32', EDGES)
                  },
                  inDegree: analyticsColumn(context, 'uint32', ANALYTICS_NODES)
                }
              : {}),
            degree: analyticsColumn(context, 'uint32', ANALYTICS_NODES),
            pageRank: {
              ...analyticsColumn(context, 'float32', ANALYTICS_NODES),
              damping: 0.9,
              iterationCount: directed ? 40 : 1024,
              residual: view('float32', 1)
            },
            coreNumber: {
              ...analyticsColumn(context, 'uint32', ANALYTICS_NODES),
              maximumIterations: directed ? 32 : 1024,
              degeneracy: view('uint32', 1),
              status: {converged: view('uint32', 1)}
            },
            components: {
              output: view('uint32', ANALYTICS_NODES),
              maximumIterations: directed ? 32 : 1024,
              status: {converged: view('uint32', 1)}
            },
            communities: {
              output: view('uint32', ANALYTICS_NODES),
              maximumIterations: directed ? 32 : 1024,
              status: {converged: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkAnalyticsColumns',
    name: 'defaults, raw columns only',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkAnalyticsColumns({
          topology: topology(context, ANALYTICS_NODES, EDGES, {
            weights: false,
            positions: false
          }) as never,
          degree: {output: view('uint32', ANALYTICS_NODES)},
          pageRank: {
            output: view('float32', ANALYTICS_NODES),
            normalized: view('float32', ANALYTICS_NODES)
          },
          coreNumber: {output: view('uint32', ANALYTICS_NODES)},
          components: {output: view('uint32', ANALYTICS_NODES)},
          communities: {output: view('uint32', ANALYTICS_NODES)}
        })
      );
    }
  },
  {
    exportName: 'GPUNetworkAnalyticsColumns',
    name: 'nodes and edges at the binding limit, every metric',
    build: context => {
      const {graph, view, maximumRows} = context;
      const nodes = maximumRows(4) - 1;
      graph.add(
        new GPUNetworkAnalyticsColumns({
          topology: topology(context, nodes, maximumRows(4), {
            weights: false,
            positions: false
          }) as never,
          reverseTopology: {
            offsets: view('uint32', nodes + 1),
            neighbors: view('uint32', maximumRows(4))
          },
          degree: analyticsColumn(context, 'uint32', nodes),
          inDegree: analyticsColumn(context, 'uint32', nodes),
          pageRank: {
            ...analyticsColumn(context, 'float32', nodes),
            residual: view('float32', 1)
          },
          coreNumber: {
            ...analyticsColumn(context, 'uint32', nodes),
            degeneracy: view('uint32', 1),
            status: {converged: view('uint32', 1)}
          },
          components: {output: view('uint32', nodes), status: {converged: view('uint32', 1)}},
          communities: {output: view('uint32', nodes), status: {converged: view('uint32', 1)}}
        })
      );
    }
  }
];

const ISOCHRONE_NODES = 400;
const ISOCHRONE_BREAKS = 16;

/** Ring assembly output with every optional column. */
function ringOutput(
  {view}: KernelAuditContext,
  capacity: number,
  vertices: number,
  withGroups: boolean
) {
  return {
    ringOffsets: view('uint32', capacity + 1),
    positions: view('float32x2', vertices),
    ringAreas: view('float32', capacity),
    ringIsHole: view('uint32', capacity),
    ringShells: view('uint32', capacity),
    ...(withGroups ? {ringGroups: view('uint32', capacity)} : {}),
    segmentRings: view('uint32', vertices),
    segmentVertices: view('uint32', vertices),
    segmentFlags: view('uint32', vertices),
    polygons: {
      kind: 'polygons' as const,
      positions: view('float32x2', vertices),
      ringOffsets: view('uint32', capacity + 1),
      polygonOffsets: view('uint32', capacity + 1),
      featureOffsets: view('uint32', capacity + 1),
      ...(withGroups ? {sourceIds: view('uint32', capacity)} : {})
    },
    status: {
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1),
      openSegmentCount: view('uint32', 1),
      touchingSegmentCount: view('uint32', 1)
    }
  };
}

/** Isoband output with every optional column, plus the raster value and facility columns. */
function isochroneRasterOutput(
  {view}: KernelAuditContext,
  pixels: number,
  triangleCapacity: number,
  edgeCapacity: number,
  withFacilities: boolean
) {
  return {
    bandClasses: view('uint32', pixels),
    triangles: view('float32x2', triangleCapacity * 3),
    triangleBands: view('uint32', triangleCapacity),
    edges: view('float32x4', edgeCapacity),
    edgeBands: view('uint32', edgeCapacity),
    status: {
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1),
      vertexCount: view('uint32', 1),
      edgeCount: view('uint32', 1),
      edgeOverflow: view('uint32', 1),
      edgeRequiredCount: view('uint32', 1)
    },
    values: view('float32', pixels),
    ...(withFacilities
      ? {
          pixelFacilities: view('uint32', pixels),
          triangleFacilities: view('uint32', triangleCapacity)
        }
      : {})
  };
}

/** Cell outline output with every optional column and ring assembly. */
function isochroneCellOutline(
  context: KernelAuditContext,
  family: 'h3' | 'quadbin',
  capacity: number,
  tableCapacity: number,
  byFacility: boolean
): NonNullable<GPUNetworkIsochronesProps['cellOutline']> {
  const {view} = context;
  return {
    family,
    resolution: family === 'h3' ? 15 : 26,
    tableCapacity,
    byFacility,
    ...(byFacility ? {cellFacilities: view('uint32', tableCapacity)} : {}),
    output: {
      rows: view('uint32', capacity),
      cells: view('uint32x2', capacity),
      edgeIndices: view('uint32', capacity),
      endpoints: view('float32x4', capacity),
      ...(byFacility ? {groups: view('uint32', capacity)} : {}),
      status: {
        count: view('uint32', 1),
        overflow: view('uint32', 1),
        requiredCount: view('uint32', 1)
      }
    },
    rings: {
      vertexTolerance: 1e-6,
      normalizeWinding: true,
      output: ringOutput(context, capacity, capacity, byFacility)
    }
  };
}

const NETWORK_ISOCHRONES_CASES: KernelAuditCase[] = [
  ...(['min', 'max'] as const).map(
    (mode): KernelAuditCase => ({
      exportName: 'GPUNetworkIsochrones',
      name: `raster ${mode}, searching from facilities, rings, maximum buffer and samples`,
      build: context => {
        const {graph, view} = context;
        const pixels = 128 * 128;
        graph.add(
          new GPUNetworkIsochrones({
            topology: topology(context, ISOCHRONE_NODES, EDGES) as never,
            costs: view('float32', ISOCHRONE_NODES),
            startNodes: view('uint32', 255),
            startCosts: view('float32', 255),
            startCount: view('uint32', 1),
            assignments: view('uint32', ISOCHRONE_NODES),
            maximumLabelIterations: 64,
            costLimit: view('float32', 1),
            maximumIterations: 128,
            localIterations: 16,
            output: {status: {converged: view('uint32', 1)}},
            costBreaks: view('float32', ISOCHRONE_BREAKS),
            parameters: view('float32', GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH),
            raster: {
              width: 128,
              height: 128,
              mode,
              maximumBufferPixels: 16,
              maximumSamplesPerEdge: 1024,
              unreachedCost: 1e20,
              output: isochroneRasterOutput(context, pixels, 4096, 4096, mode === 'min'),
              rings: {
                edgeCapacity: 4096,
                vertexTolerance: 1e-3,
                output: {
                  ...ringOutput(context, 1024, 4096, false),
                  status: {
                    ...ringOutput(context, 1024, 4096, false).status,
                    edgeCount: view('uint32', 1),
                    edgeOverflow: view('uint32', 1)
                  }
                }
              }
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkIsochrones',
    name: 'raster from caller costs, band classes only',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkIsochrones({
          topology: topology(context, ISOCHRONE_NODES, EDGES) as never,
          costs: view('float32', ISOCHRONE_NODES),
          costBreaks: view('float32', ISOCHRONE_BREAKS),
          parameters: view('float32', GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH),
          raster: {width: 64, height: 64, output: {bandClasses: view('uint32', 64 * 64)}}
        })
      );
    }
  },
  ...(['h3', 'quadbin'] as const).flatMap(family =>
    [false, true].map(
      (byFacility): KernelAuditCase => ({
        exportName: 'GPUNetworkIsochrones',
        name: `${family} cell outline, by facility ${byFacility}, rings, searching from facilities`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUNetworkIsochrones({
              spatialContext: LON_LAT,
              topology: topology(context, ISOCHRONE_NODES, EDGES) as never,
              costs: view('float32', ISOCHRONE_NODES),
              startNodes: view('uint32', 32),
              startCosts: view('float32', 32),
              startCount: 20,
              assignments: view('uint32', ISOCHRONE_NODES),
              costLimit: view('float32', 1),
              output: {status: {converged: view('uint32', 1)}},
              costBreaks: view('float32', ISOCHRONE_BREAKS),
              parameters: view('float32', GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH),
              cellOutline: isochroneCellOutline(context, family, 4096, 1024, byFacility)
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUNetworkIsochrones',
    name: 'raster and cell outline together, default table',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkIsochrones({
          topology: topology(context, ISOCHRONE_NODES, EDGES) as never,
          costs: view('float32', ISOCHRONE_NODES),
          costBreaks: view('float32', ISOCHRONE_BREAKS),
          parameters: view('float32', GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH),
          raster: {
            width: 64,
            height: 64,
            output: isochroneRasterOutput(context, 64 * 64, 512, 512, false)
          },
          cellOutline: {
            family: 'h3',
            resolution: 8,
            output: {
              rows: view('uint32', 512),
              cells: view('uint32x2', 512),
              edgeIndices: view('uint32', 512),
              endpoints: view('float32x4', 512),
              status: {count: view('uint32', 1), overflow: view('uint32', 1)}
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPUNetworkIsochrones',
    name: 'raster at the binding limit, nodes and edges at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const pixels = maximumRows(4);
      const width = Math.floor(Math.sqrt(pixels));
      const nodes = maximumRows(8);
      graph.add(
        new GPUNetworkIsochrones({
          topology: topology(context, nodes, maximumRows(4)) as never,
          costs: view('float32', nodes),
          startNodes: view('uint32', 255),
          assignments: view('uint32', nodes),
          costBreaks: view('float32', 255),
          parameters: view('float32', GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH),
          raster: {
            width,
            height: width,
            maximumBufferPixels: 16,
            maximumSamplesPerEdge: 1024,
            output: isochroneRasterOutput(
              context,
              width * width,
              Math.floor(maximumRows(8) / 3),
              maximumRows(16),
              true
            )
          }
        })
      );
    }
  }
];

const K_FUNCTION_EVENTS = 200;
const K_FUNCTION_NODES = 300;

/** Every optional output of one K function. */
function kFunctionOutput(
  {view}: KernelAuditContext,
  bands: number,
  simulations: number,
  withOverflow: boolean
) {
  return {
    kValues: view('float32', (1 + simulations) * bands),
    pairCounts: view('uint32', (1 + simulations) * bands),
    envelope: view('float32', 3 * bands),
    snappedEventCount: view('uint32', 1),
    status: {
      converged: view('uint32', 1),
      ...(withOverflow ? {overflow: view('uint32', 1)} : {})
    }
  };
}

const NETWORK_K_FUNCTION_CASES: KernelAuditCase[] = [
  ...[undefined, LON_LAT].flatMap(spatialContext =>
    [false, true].map(
      (spatialSort): KernelAuditCase => ({
        exportName: 'GPUNetworkKFunction',
        name: `candidate search, spatial sort ${spatialSort}, ${spatialContext ? 'great-circle' : 'planar'}, simulations, every output`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUNetworkKFunction({
              spatialContext,
              positions: view('float32x2', K_FUNCTION_EVENTS),
              topology: topology(context, K_FUNCTION_NODES, EDGES) as never,
              maximumSnapDistance: view('float32', 1),
              candidateCapacity: 4096,
              spatialSort,
              maximumDistance: view('float32', 1),
              networkLength: view('float32', 1),
              parameters: view('uint32', GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH),
              bandCount: 64,
              simulationCount: 99,
              rowsPerBlock: 600,
              laneCount: 200,
              maximumIterations: 128,
              localIterations: 64,
              output: kFunctionOutput(context, 64, 99, true)
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUNetworkKFunction',
    name: 'exact snapping, no simulations, defaults',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkKFunction({
          positions: view('float32x2', K_FUNCTION_EVENTS),
          topology: topology(context, K_FUNCTION_NODES, EDGES) as never,
          maximumDistance: view('float32', 1),
          networkLength: view('float32', 1),
          parameters: view('uint32', GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH),
          output: {kValues: view('float32', 32)}
        })
      );
    }
  },
  {
    // 1 + 32767 patterns of 1024 bands fill one binding of K values; row blocks are as wide as
    // one frontier binding allows.
    exportName: 'GPUNetworkKFunction',
    name: 'maximum bands and simulations, widest row blocks',
    build: context => {
      const {graph, view, maximumRows} = context;
      const bands = GPU_NETWORK_K_FUNCTION_MAXIMUM_BAND_COUNT;
      const simulations = maximumRows(4 * bands) - 1;
      const nodes = 1000;
      graph.add(
        new GPUNetworkKFunction({
          positions: view('float32x2', 64),
          topology: topology(context, nodes, EDGES) as never,
          maximumSnapDistance: view('float32', 1),
          candidateCapacity: 4096,
          spatialSort: true,
          maximumDistance: view('float32', 1),
          networkLength: view('float32', 1),
          parameters: view('uint32', GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH),
          bandCount: bands,
          simulationCount: simulations,
          rowsPerBlock: Math.floor(maximumRows(8) / nodes),
          maximumIterations: 64,
          output: kFunctionOutput(context, bands, simulations, true)
        })
      );
    }
  },
  {
    // The pair-count kernel runs rowsPerBlock * events invocations, which must stay below 2^32.
    exportName: 'GPUNetworkKFunction',
    name: 'row blocks at the 2^32 invocation limit',
    build: context => {
      const {graph, view} = context;
      const nodes = 16;
      const events = 1 << 16;
      graph.add(
        new GPUNetworkKFunction({
          positions: view('float32x2', events),
          topology: topology(context, nodes, 64) as never,
          maximumDistance: view('float32', 1),
          networkLength: view('float32', 1),
          parameters: view('uint32', GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH),
          rowsPerBlock: Math.floor(0xffffffff / events),
          maximumIterations: 8,
          output: kFunctionOutput(context, 32, 0, false)
        })
      );
    }
  },
  {
    // Events at the binding limit with rowsPerBlock * events above 2^32 - 1.
    exportName: 'GPUNetworkKFunction',
    name: 'events at the binding limit, row blocks past the invocation limit',
    build: context => kFunctionEventsCase(context, 256)
  },
  {
    exportName: 'GPUNetworkKFunction',
    name: 'events at the binding limit, row blocks at the invocation limit, seed table above the maximum',
    build: context => kFunctionEventsCase(context, 255)
  },
  {
    exportName: 'GPUNetworkKFunction',
    name: 'half the events, row blocks at the invocation limit',
    build: context => kFunctionEventsCase(context, 255, 2)
  }
];

/** One K function over the largest event table with `rowsPerBlock` rows per block. */
function kFunctionEventsCase(
  context: KernelAuditContext,
  rowsPerBlock: number,
  eventDivisor: number = 1
): void {
  const {graph, view, maximumRows} = context;
  graph.add(
    new GPUNetworkKFunction({
      positions: view('float32x2', Math.floor(maximumRows(8) / eventDivisor)),
      topology: topology(context, 16, 64) as never,
      maximumDistance: view('float32', 1),
      networkLength: view('float32', 1),
      parameters: view('uint32', GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH),
      rowsPerBlock,
      maximumIterations: 8,
      output: kFunctionOutput(context, 32, 0, false)
    })
  );
}

const COARSENING_NODES = 500;

/** Every output of one coarsening. */
function coarseningOutput(
  {view}: KernelAuditContext,
  groups: number,
  edgeCapacity: number,
  withValues: boolean,
  withPositions: boolean
) {
  return {
    groupNodeCount: view('uint32', groups),
    groupIntraEdgeCount: view('uint32', groups),
    groupIntraWeight: view('float32', groups),
    ...(withPositions
      ? {groupCentroid: view('float32x2', groups), groupBounds: view('float32x4', groups)}
      : {}),
    ...(withValues ? {groupValueSum: view('float32', groups)} : {}),
    edges: {
      ids: view('uint32', edgeCapacity),
      status: {
        count: view('uint32', 1),
        overflow: view('uint32', 1),
        requiredCount: view('uint32', 1)
      }
    },
    edgeTargetGroups: view('uint32', edgeCapacity),
    edgeCounts: view('uint32', edgeCapacity),
    edgeWeights: view('float32', edgeCapacity),
    summary: view('uint32', GPU_NETWORK_COARSENING_SUMMARY_LENGTH)
  };
}

const NETWORK_COARSENING_CASES: KernelAuditCase[] = [
  ...[false, true].map(
    (directed): KernelAuditCase => ({
      exportName: 'GPUNetworkCoarsening',
      name: `${directed ? 'directed' : 'undirected'}, every option and output, smallest fixed-point scale`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkCoarsening({
            topology: topology(context, COARSENING_NODES, EDGES) as never,
            directed,
            nodeMask: view('uint32', COARSENING_NODES),
            edgeMask: view('uint32', EDGES),
            labels: view('uint32', COARSENING_NODES),
            nodeValues: view('float32', COARSENING_NODES),
            groupCapacity: 1000,
            fixedPointScale: directed ? 1 : 2 ** 24,
            output: coarseningOutput(context, 1000, 2048, true, true)
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkCoarsening',
    name: 'defaults, counts and edges only',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkCoarsening({
          topology: topology(context, COARSENING_NODES, EDGES, {
            weights: false,
            positions: false
          }) as never,
          labels: view('uint32', COARSENING_NODES),
          groupCapacity: 16,
          output: {
            groupNodeCount: view('uint32', 16),
            edges: {
              ids: view('uint32', 64),
              status: {count: view('uint32', 1), overflow: view('uint32', 1)}
            },
            edgeTargetGroups: view('uint32', 64),
            edgeCounts: view('uint32', 64)
          }
        })
      );
    }
  },
  ...[0, 1].map(
    (overMaximum): KernelAuditCase => ({
      exportName: 'GPUNetworkCoarsening',
      // The 64-bit fixed-point weight accumulators take 8 bytes per superedge slot, so the
      // weighted superedge capacity fits one binding up to the maximum.
      name: overMaximum
        ? 'weighted superedges above the maximum'
        : 'maximum group capacity, nodes and edge slots at the limit, weighted superedges at the maximum',
      build: context => {
        const {graph, view, maximumRows} = context;
        const nodes = maximumRows(8);
        const edges = maximumRows(4);
        const groups = GPU_NETWORK_COARSENING_MAXIMUM_GROUP_CAPACITY;
        graph.add(
          new GPUNetworkCoarsening({
            topology: topology(context, nodes, edges) as never,
            nodeMask: view('uint32', nodes),
            edgeMask: view('uint32', edges),
            labels: view('uint32', nodes),
            nodeValues: view('float32', nodes),
            groupCapacity: groups,
            fixedPointScale: 2 ** 24,
            output: coarseningOutput(
              context,
              groups,
              getGPUNetworkCoarseningMaximumWeightedEdgeCapacity(context.device) + overMaximum,
              true,
              true
            )
          })
        );
      }
    })
  )
];

const COST_WEIGHTS_ROWS = 20;
const COST_WEIGHTS_COLUMNS = 300;

/** Every optional output of one cost-weights contributor. */
function costWeightsOutput({view}: KernelAuditContext, offsetRows: number, capacity: number) {
  return {
    offsets: view('uint32', offsetRows + 1),
    neighbors: view('uint32', capacity),
    weights: view('float32', capacity),
    distances: view('float32', capacity),
    status: {overflow: view('uint32', 1), requiredCount: view('uint32', 1)}
  };
}

const NETWORK_COST_WEIGHTS_CASES: KernelAuditCase[] = [
  ...(['rows', 'columns'] as const).flatMap(orientation =>
    [false, true].map(
      (withColumns): KernelAuditCase => ({
        exportName: 'GPUNetworkCostWeights',
        name: `${orientation}, ${withColumns ? 'column map' : 'every column'}, every output`,
        build: context => {
          const {graph, view} = context;
          const locations = withColumns ? 250 : COST_WEIGHTS_COLUMNS;
          graph.add(
            new GPUNetworkCostWeights({
              costs: view('float32', COST_WEIGHTS_ROWS * COST_WEIGHTS_COLUMNS),
              rowCount: COST_WEIGHTS_ROWS,
              ...(withColumns ? {columns: view('uint32', locations)} : {}),
              orientation,
              parameters: view('float32', GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH),
              output: costWeightsOutput(
                context,
                orientation === 'rows' ? COST_WEIGHTS_ROWS : locations,
                4096
              )
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUNetworkCostWeights',
    name: 'one row, outputs without diagnostics',
    build: ({graph, view}) =>
      graph.add(
        new GPUNetworkCostWeights({
          costs: view('float32', 1000),
          parameters: view('float32', GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH),
          output: {
            offsets: view('uint32', 2),
            neighbors: view('uint32', 1000),
            weights: view('float32', 1000)
          }
        })
      )
  },
  ...(['rows', 'columns'] as const).map(
    (orientation): KernelAuditCase => ({
      exportName: 'GPUNetworkCostWeights',
      name: `${orientation}, matrix and slot capacity at the binding limit`,
      build: context => {
        const {graph, view, maximumRows} = context;
        const rows = 2048;
        const columns = Math.floor(maximumRows(4) / rows);
        graph.add(
          new GPUNetworkCostWeights({
            costs: view('float32', rows * columns),
            rowCount: rows,
            columns: view('uint32', columns),
            orientation,
            parameters: view('float32', GPU_NETWORK_COST_WEIGHTS_PARAMETER_LENGTH),
            output: costWeightsOutput(
              context,
              orientation === 'rows' ? rows : columns,
              maximumRows(4)
            )
          })
        );
      }
    })
  )
];

const LINE_GRAPH_NODES = 300;

/** Every output of one line graph over `edges` base edges and `arcCapacity` arcs. */
function lineGraphOutput({view}: KernelAuditContext, edges: number, arcCapacity: number) {
  return {
    topology: {
      offsets: view('uint32', edges + 1),
      neighbors: view('uint32', arcCapacity),
      weights: view('float32', arcCapacity)
    },
    arcCount: view('uint32', 1),
    status: {overflow: view('uint32', 1)}
  };
}

const NETWORK_LINE_GRAPH_CASES: KernelAuditCase[] = [
  ...[undefined, LON_LAT].map(
    (spatialContext): KernelAuditCase => ({
      exportName: 'GPUNetworkLineGraph',
      name: `${spatialContext ? 'great-circle' : 'planar'}, banned turns, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkLineGraph({
            spatialContext,
            topology: topology(context, LINE_GRAPH_NODES, EDGES) as never,
            parameters: view('float32', GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH),
            bannedTurns: view('uint32', 2 * 512),
            output: lineGraphOutput(context, EDGES, 8192)
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkLineGraph',
    name: 'no bans, no arc count',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkLineGraph({
          topology: topology(context, LINE_GRAPH_NODES, EDGES) as never,
          parameters: view('float32', GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH),
          output: {
            topology: {
              offsets: view('uint32', EDGES + 1),
              neighbors: view('uint32', 8192),
              weights: view('float32', 8192)
            },
            status: {overflow: view('uint32', 1)}
          }
        })
      );
    }
  },
  ...[
    {spatialContext: undefined, getMaximum: getGPUNetworkLineGraphMaximumEdgeCount},
    {spatialContext: LON_LAT, getMaximum: getGPUNetworkLineGraphMaximumGeodesicEdgeCount}
  ].flatMap(({spatialContext, getMaximum}) =>
    [0, 1].map(
      (overMaximum): KernelAuditCase => ({
        exportName: 'GPUNetworkLineGraph',
        // The edge direction vectors take 8 bytes per base edge on planar coordinates and 16 on
        // great-circle coordinates (departure and arrival), so the base edges fit one binding up to
        // the maximum of their context.
        name: `${spatialContext ? 'great-circle' : 'planar'}, base edges ${
          overMaximum ? 'above' : 'at'
        } the maximum, arcs and banned turns at the binding limit`,
        build: context => {
          const {graph, view, maximumRows} = context;
          const edges = getMaximum(context.device) + overMaximum;
          graph.add(
            new GPUNetworkLineGraph({
              spatialContext,
              topology: topology(context, maximumRows(8), edges) as never,
              parameters: view('float32', GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH),
              bannedTurns: view('uint32', maximumRows(4)),
              output: lineGraphOutput(context, edges, maximumRows(4))
            })
          );
        }
      })
    )
  )
];

/** Every output of one noding with `pieces` piece capacity. */
function nodingOutput({view}: KernelAuditContext, pieces: number) {
  return {
    pieces: {
      geometry: {
        kind: 'lines' as const,
        positions: view('float32x2', 2 * pieces),
        lineOffsets: view('uint32', pieces + 1)
      },
      sourceIds: view('uint32', pieces),
      status: {
        count: view('uint32', 1),
        overflow: view('uint32', 1),
        requiredCount: view('uint32', 1),
        vertexCount: view('uint32', 1),
        requiredVertexCount: view('uint32', 1)
      },
      vertexSourceEdges: view('uint32', 2 * pieces)
    },
    nodes: {
      positions: view('float32x2', 2 * pieces),
      status: {count: view('uint32', 1), requiredCount: view('uint32', 1)}
    },
    edges: {
      fromNodes: view('uint32', pieces),
      toNodes: view('uint32', pieces),
      lengths: view('float32', pieces)
    },
    topology: {
      offsets: view('uint32', 2 * pieces + 1),
      neighbors: view('uint32', 2 * pieces),
      weights: view('float32', 2 * pieces),
      edgeIds: view('uint32', 2 * pieces)
    },
    status: {uncertainCount: view('uint32', 1), overflow: view('uint32', 1)}
  };
}

/** One noding of 100 lines with an `intersectionCapacity`. */
function nodingCapacityCase(context: KernelAuditContext, intersectionCapacity: number): void {
  const {graph, view} = context;
  graph.add(
    new GPUNetworkNoding({
      lines: {
        kind: 'lines',
        positions: view('float32x2', 1000),
        lineOffsets: view('uint32', 101)
      },
      intersectionCapacity,
      tolerance: view('float32', 1),
      output: nodingOutput(context, 256)
    })
  );
}

const NETWORK_NODING_CASES: KernelAuditCase[] = [
  ...[false, true].map(
    (spatialSort): KernelAuditCase => ({
      exportName: 'GPUNetworkNoding',
      name: `spatial sort ${spatialSort}, leaf capacity, every output`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkNoding({
            lines: {
              kind: 'lines',
              positions: view('float32x2', 1000),
              lineOffsets: view('uint32', 101)
            },
            intersectionCapacity: 4096,
            tolerance: view('float32', 1),
            spatialSort,
            leafCapacity: 1024,
            output: nodingOutput(context, 2048)
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkNoding',
    name: 'defaults, required outputs only',
    build: context => {
      const {graph, view} = context;
      const pieces = 256;
      graph.add(
        new GPUNetworkNoding({
          lines: {
            kind: 'lines',
            positions: view('float32x2', 1000),
            lineOffsets: view('uint32', 101)
          },
          intersectionCapacity: 1024,
          tolerance: view('float32', 1),
          output: {
            pieces: {
              geometry: {
                kind: 'lines',
                positions: view('float32x2', 2 * pieces),
                lineOffsets: view('uint32', pieces + 1)
              },
              sourceIds: view('uint32', pieces),
              status: {count: view('uint32', 1), overflow: view('uint32', 1)}
            },
            nodes: {
              positions: view('float32x2', 2 * pieces),
              status: {count: view('uint32', 1)}
            },
            edges: {
              fromNodes: view('uint32', pieces),
              toNodes: view('uint32', pieces),
              lengths: view('float32', pieces)
            },
            topology: {
              offsets: view('uint32', 2 * pieces + 1),
              neighbors: view('uint32', 2 * pieces),
              weights: view('float32', 2 * pieces)
            }
          }
        })
      );
    }
  },
  {
    // Each intersection pair reserves 64 bytes of split table, so the capacity is at most
    // getGPUNetworkNodingMaximumIntersectionCapacity.
    exportName: 'GPUNetworkNoding',
    name: 'intersection capacity at the maximum',
    // 100 lines + 4 x (2^20 - 25) pairs is exactly the 2^22-row piece table.
    build: context =>
      nodingCapacityCase(
        context,
        getGPUNetworkNodingMaximumIntersectionCapacity(context.device) - 25
      )
  },
  {
    exportName: 'GPUNetworkNoding',
    name: 'intersection capacity above the maximum',
    build: context =>
      nodingCapacityCase(
        context,
        getGPUNetworkNodingMaximumIntersectionCapacity(context.device) + 1
      )
  },
  {
    exportName: 'GPUNetworkNoding',
    name: 'pieces, nodes and intersections at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      // The piece table takes 32 bytes per piece.
      const pieces = maximumRows(32) - 1;
      graph.add(
        new GPUNetworkNoding({
          lines: {
            kind: 'lines',
            // The segment table takes 32 bytes per vertex and the piece table 32 bytes per
            // line plus 4 per intersection pair.
            positions: view('float32x2', maximumRows(32)),
            lineOffsets: view('uint32', 2 ** 20 + 1)
          },
          intersectionCapacity: 2 ** 19,
          tolerance: view('float32', 1),
          spatialSort: true,
          leafCapacity: 2 ** 22,
          output: nodingOutput(context, pieces)
        })
      );
    }
  }
];

const STATISTICS_NODES = 500;

const NETWORK_STATISTICS_CASES: KernelAuditCase[] = [
  ...(['linear', 'log2'] as const).flatMap(degreeBinning =>
    [32, 4096].map(
      (degreeBinCount): KernelAuditCase => ({
        exportName: 'GPUNetworkStatistics',
        name: `${degreeBinning} bins, ${degreeBinCount} bin count, every option`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUNetworkStatistics({
              topology: topology(context, STATISTICS_NODES, EDGES, {
                weights: false,
                positions: false
              }) as never,
              directed: degreeBinCount === 32,
              nodeMask: view('uint32', STATISTICS_NODES),
              edgeMask: view('uint32', EDGES),
              communities: view('uint32', STATISTICS_NODES),
              parameters: view('uint32', GPU_NETWORK_STATISTICS_PARAMETER_LENGTH),
              degreeBinCount,
              degreeBinning,
              maximumComponentIterations: 1024,
              output: view('uint32', getGPUNetworkStatisticsLength(degreeBinCount))
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUNetworkStatistics',
    name: 'defaults, no optional views',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkStatistics({
          topology: topology(context, STATISTICS_NODES, EDGES, {
            weights: false,
            positions: false
          }) as never,
          output: view('uint32', getGPUNetworkStatisticsLength())
        })
      );
    }
  },
  {
    exportName: 'GPUNetworkStatistics',
    name: 'nodes and edges at the binding limit, every option',
    build: context => {
      const {graph, view, maximumRows} = context;
      const nodes = maximumRows(4) - 1;
      graph.add(
        new GPUNetworkStatistics({
          topology: topology(context, nodes, maximumRows(4), {
            weights: false,
            positions: false
          }) as never,
          nodeMask: view('uint32', nodes),
          edgeMask: view('uint32', maximumRows(4)),
          communities: view('uint32', nodes),
          parameters: view('uint32', GPU_NETWORK_STATISTICS_PARAMETER_LENGTH),
          degreeBinCount: 4096,
          degreeBinning: 'log2',
          maximumComponentIterations: 1024,
          output: view('uint32', getGPUNetworkStatisticsLength(4096))
        })
      );
    }
  }
];

const SUBGRAPH_NODES = 400;

/** Every optional output of one subgraph filter. */
function subgraphFilterOutput(
  {view}: KernelAuditContext,
  nodes: number,
  slots: number,
  compactCapacity: number,
  slotCapacity: number
) {
  const compact = (capacity: number) => ({
    ids: view('uint32', capacity),
    status: {
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1)
    }
  });
  return {
    nodeMask: view('uint32', nodes),
    edgeMask: view('uint32', slots),
    counts: view('uint32', GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH),
    liveNodes: compact(compactCapacity),
    liveEdgeSlots: compact(compactCapacity),
    inducedTopology: {
      offsets: view('uint32', nodes + 1),
      neighbors: view('uint32', slotCapacity),
      sourceSlots: view('uint32', slotCapacity),
      status: {overflow: view('uint32', 1)}
    }
  };
}

const NETWORK_SUBGRAPH_FILTER_CASES: KernelAuditCase[] = [
  ...[false, true].flatMap(directed =>
    [false, true].map(
      (pairUndirectedSlots): KernelAuditCase => ({
        exportName: 'GPUNetworkSubgraphFilter',
        name: `${directed ? 'directed' : 'undirected'}, pair slots ${pairUndirectedSlots}, 64 columns, float time, drop isolated`,
        build: context => {
          const {graph, view} = context;
          const nodeColumns = Array.from({length: 32}, () => view('float32', SUBGRAPH_NODES));
          const edgeColumns = Array.from({length: 31}, () => view('float32', EDGES));
          graph.add(
            new GPUNetworkSubgraphFilter({
              topology: topology(context, SUBGRAPH_NODES, EDGES, {
                weights: false,
                positions: false
              }) as never,
              directed,
              pairUndirectedSlots,
              nodeMask: view('uint32', SUBGRAPH_NODES),
              edgeMask: view('uint32', EDGES),
              nodeColumns,
              edgeColumns,
              edgeTimes: view('float32', EDGES),
              parameters: view(
                'float32',
                getGPUNetworkSubgraphFilterParameterLength({
                  nodeColumnCount: 32,
                  edgeColumnCount: 31,
                  hasEdgeTimes: true
                })
              ),
              dropIsolated: true,
              output: subgraphFilterOutput(context, SUBGRAPH_NODES, EDGES, 1024, 2048)
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUNetworkSubgraphFilter',
    name: 'exact word time window with a few columns',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkSubgraphFilter({
          topology: topology(context, SUBGRAPH_NODES, EDGES, {
            weights: false,
            positions: false
          }) as never,
          nodeColumns: [view('float32', SUBGRAPH_NODES)],
          edgeColumns: [view('float32', EDGES), view('float32', EDGES)],
          edgeTimeWords: view('uint32x2', EDGES),
          parameters: view(
            'float32',
            getGPUNetworkSubgraphFilterParameterLength({
              nodeColumnCount: 1,
              edgeColumnCount: 2,
              hasEdgeTimes: false
            })
          ),
          uint32Parameters: view('uint32', 8),
          output: subgraphFilterOutput(context, SUBGRAPH_NODES, EDGES, 1024, 2048)
        })
      );
    }
  },
  {
    exportName: 'GPUNetworkSubgraphFilter',
    name: 'masks only, no optional outputs',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkSubgraphFilter({
          topology: topology(context, SUBGRAPH_NODES, EDGES, {
            weights: false,
            positions: false
          }) as never,
          output: {
            nodeMask: view('uint32', SUBGRAPH_NODES),
            edgeMask: view('uint32', EDGES)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUNetworkSubgraphFilter',
    name: 'nodes, slots and compact outputs at the binding limit, 64 columns, every option',
    build: context => {
      const {graph, view, maximumRows} = context;
      const nodes = maximumRows(4) - 1;
      const slots = maximumRows(4);
      const nodeColumns = Array.from({length: 32}, () => view('float32', nodes));
      const edgeColumns = Array.from({length: 31}, () => view('float32', slots));
      graph.add(
        new GPUNetworkSubgraphFilter({
          topology: topology(context, nodes, slots, {weights: false, positions: false}) as never,
          nodeMask: view('uint32', nodes),
          edgeMask: view('uint32', slots),
          nodeColumns,
          edgeColumns,
          edgeTimes: view('float32', slots),
          parameters: view(
            'float32',
            getGPUNetworkSubgraphFilterParameterLength({
              nodeColumnCount: 32,
              edgeColumnCount: 31,
              hasEdgeTimes: true
            })
          ),
          dropIsolated: true,
          output: subgraphFilterOutput(context, nodes, slots, slots, slots)
        })
      );
    }
  }
];

const RECIPE_NODES = 400;
const RECIPE_FACILITIES = 32;
const RECIPE_DEMAND = 600;

/** Every caller-owned output and scratch override of the drive-time recipe. */
function driveTimeOverrides(
  {view}: KernelAuditContext,
  options: {
    nodes: number;
    facilities: number;
    demand: number;
    bandCapacity: number;
    edges: number;
    candidates: boolean;
    joinDemand: boolean;
  }
) {
  const {nodes, facilities, demand, bandCapacity, edges, candidates, joinDemand} = options;
  return {
    outputs: {
      assignments: view('uint32', nodes),
      nodeCosts: view('float32', nodes),
      status: {serviceAreaConverged: view('uint32', 1)},
      demandTimes: view('float32', demand),
      demandBands: view('uint32', demand),
      bands: {
        keys: view('uint32', bandCapacity),
        counts: view('uint32', bandCapacity),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          requiredCount: view('uint32', 1)
        },
        sumValues: view('float32', bandCapacity),
        means: view('float32', bandCapacity)
      },
      ...(joinDemand
        ? {
            isochroneDemand: {
              pointFeatureIds: view('uint32', demand),
              status: {overflow: view('uint32', 1), uncertainCount: view('uint32', 1)}
            }
          }
        : {}),
      ...(candidates
        ? {
            snapping: {
              status: {
                facilitiesOverflow: view('uint32', 1),
                demandOverflow: view('uint32', 1)
              }
            }
          }
        : {})
    },
    scratch: {
      startNodes: view('uint32', 2 * facilities),
      startCosts: view('float32', 2 * facilities),
      facilityEdges: view('uint32', facilities),
      demandEdges: view('uint32', demand),
      demandSourceCosts: view('float32', demand),
      demandTargetCosts: view('float32', demand),
      bandSumWords: view('uint32x2', bandCapacity)
    },
    edges
  };
}

const DRIVE_TIME_CATCHMENT_RECIPE_CASES: KernelAuditCase[] = [
  ...[undefined, LON_LAT].map(
    (spatialContext): KernelAuditCase => ({
      exportName: 'addDriveTimeCatchmentRecipe',
      name: `exact snapping, ${spatialContext ? 'great-circle' : 'planar'}, demand values, every override`,
      build: context => {
        const {graph, view} = context;
        const {outputs, scratch} = driveTimeOverrides(context, {
          nodes: RECIPE_NODES,
          facilities: RECIPE_FACILITIES,
          demand: RECIPE_DEMAND,
          bandCapacity: 64,
          edges: EDGES,
          candidates: false,
          joinDemand: false
        });
        addDriveTimeCatchmentRecipe(graph, {
          spatialContext,
          topology: topology(context, RECIPE_NODES, EDGES) as never,
          facilities: view('float32x2', RECIPE_FACILITIES),
          demand: view('float32x2', RECIPE_DEMAND),
          demandValues: view('float32', RECIPE_DEMAND),
          maximumSnapDistance: view('float32', 1),
          costLimit: view('float32', 1),
          maximumIterations: 1024,
          costBreaks: view('float32', 16),
          bandCapacity: 64,
          outputs,
          scratch
        } as never);
      }
    })
  ),
  ...(['raster', 'h3', 'quadbin'] as const).map(
    (path): KernelAuditCase => ({
      exportName: 'addDriveTimeCatchmentRecipe',
      name: `candidate snapping, ${path} isochrones${path === 'raster' ? '' : ' with demand join'}, every override`,
      build: context => {
        const {graph, view} = context;
        const isRaster = path === 'raster';
        const {outputs, scratch} = driveTimeOverrides(context, {
          nodes: RECIPE_NODES,
          facilities: RECIPE_FACILITIES,
          demand: RECIPE_DEMAND,
          bandCapacity: 16,
          edges: EDGES,
          candidates: true,
          joinDemand: !isRaster
        });
        addDriveTimeCatchmentRecipe(graph, {
          spatialContext: isRaster ? undefined : LON_LAT,
          topology: topology(context, RECIPE_NODES, EDGES) as never,
          facilities: view('float32x2', RECIPE_FACILITIES),
          demand: view('float32x2', RECIPE_DEMAND),
          demandValues: view('float32', RECIPE_DEMAND),
          maximumSnapDistance: view('float32', 1),
          candidateCapacity: 4096,
          costLimit: view('float32', 1),
          maximumIterations: 64,
          costBreaks: view('float32', 16),
          bandCapacity: 16,
          isochrones: {
            costBreaks: view('float32', 16),
            parameters: view('float32', GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH),
            ...(isRaster
              ? {
                  raster: {
                    width: 128,
                    height: 128,
                    output: isochroneRasterOutput(context, 128 * 128, 4096, 4096, true),
                    rings: {edgeCapacity: 4096, output: ringOutput(context, 1024, 4096, false)}
                  }
                }
              : {cellOutline: isochroneCellOutline(context, path, 4096, 1024, false)}),
            ...(isRaster ? {} : {joinDemand: {candidateCapacity: 4096}})
          },
          outputs,
          scratch
        } as never);
      }
    })
  ),
  {
    exportName: 'addDriveTimeCatchmentRecipe',
    name: 'defaults, no demand values',
    build: context => {
      const {graph, view} = context;
      addDriveTimeCatchmentRecipe(graph, {
        topology: topology(context, RECIPE_NODES, EDGES) as never,
        facilities: view('float32x2', RECIPE_FACILITIES),
        demand: view('float32x2', RECIPE_DEMAND),
        costBreaks: view('float32', 4)
      } as never);
    }
  },
  {
    exportName: 'addDriveTimeCatchmentRecipe',
    name: 'nodes, edges, demand and bands at the binding limit, every override',
    build: context => {
      const {graph, view, maximumRows} = context;
      const nodes = maximumRows(8);
      // The snapping BVH takes 24 bytes per node and 2 * edges - 1 nodes.
      const edges = maximumRows(16);
      const demand = maximumRows(8);
      const facilities = 255;
      const bands = 1024;
      const {outputs, scratch} = driveTimeOverrides(context, {
        nodes,
        facilities,
        demand,
        bandCapacity: bands,
        edges,
        candidates: true,
        joinDemand: false
      });
      addDriveTimeCatchmentRecipe(graph, {
        topology: topology(context, nodes, edges) as never,
        facilities: view('float32x2', facilities),
        demand: view('float32x2', demand),
        demandValues: view('float32', demand),
        maximumSnapDistance: view('float32', 1),
        candidateCapacity: maximumRows(16),
        costLimit: view('float32', 1),
        maximumIterations: 1024,
        costBreaks: view('float32', bands),
        bandCapacity: bands,
        outputs,
        scratch
      } as never);
    }
  }
];

const NETWORK_COST_TABLE_CASES: KernelAuditCase[] = [
  ...[1, 16, GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS].map(
    (localIterations): KernelAuditCase => ({
      exportName: 'GPUNetworkCostTable',
      name: `local iterations ${localIterations}, every status`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkCostTable({
            topology: topology(context, NODES, EDGES, {positions: false}) as never,
            costLimit: view('float32', 1),
            localIterations,
            output: {
              offsets: view('uint32', NODES + 1),
              targets: view('uint32', 4096),
              costs: view('float32', 4096),
              status: {
                overflow: view('uint32', 1),
                requiredCount: view('uint32', 1),
                converged: view('uint32', 1)
              }
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkCostTable',
    name: 'one lane, defaults, no status',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkCostTable({
          topology: topology(context, NODES, EDGES, {positions: false}) as never,
          costLimit: view('float32', 1),
          laneCount: 1,
          output: {
            offsets: view('uint32', NODES + 1),
            targets: view('uint32', 4096),
            costs: view('float32', 4096)
          }
        })
      );
    }
  }
];

const NETWORK_CENTRALITY_CASES: KernelAuditCase[] = [
  ...[1, GPU_NETWORK_REACHABILITY_MAXIMUM_LOCAL_ITERATIONS].map(
    (localIterations): KernelAuditCase => ({
      exportName: 'GPUNetworkCentrality',
      name: `local iterations ${localIterations}, maximum rounds, every output, cost limit`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUNetworkCentrality({
            topology: topology(context, REACHABILITY_NODES, EDGES, {positions: false}) as never,
            reverseTopology: topology(context, REACHABILITY_NODES, EDGES, {
              positions: false
            }) as never,
            pivotNodes: view('uint32', 256),
            costLimit: view('float32', 1),
            laneCount: 4,
            maximumIterations: GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS,
            localIterations,
            betweennessScale: 4,
            output: {
              betweenness: view('float32', REACHABILITY_NODES),
              closeness: view('float32', 256),
              harmonicCloseness: view('float32', 256),
              reachedCounts: view('uint32', 256),
              status: {converged: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNetworkCentrality',
    name: 'default lanes, closeness only',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUNetworkCentrality({
          topology: topology(context, REACHABILITY_NODES, EDGES, {positions: false}) as never,
          reverseTopology: topology(context, REACHABILITY_NODES, EDGES, {
            positions: false
          }) as never,
          pivotNodes: view('uint32', 64),
          output: {closeness: view('float32', 64)}
        })
      );
    }
  }
];

/** Kernel-audit cases for network. */
export const NETWORK_CASES: KernelAuditCase[] = [
  ...ADJACENCY_MATRIX_CASES,
  ...ADJACENCY_MATRIX_ORDER_CASES,
  ...EDGE_BUNDLING_CASES,
  ...FLOW_AGGREGATION_CASES,
  ...MAP_MATCHING_CASES,
  ...NETWORK_SNAPPING_CASES,
  ...NETWORK_COST_MATRIX_CASES,
  ...NETWORK_COST_TABLE_CASES,
  ...NETWORK_CENTRALITY_CASES,
  ...NETWORK_ACCESSIBILITY_CASES,
  ...NETWORK_REACHABILITY_CASES,
  ...NETWORK_PATH_EXTRACTION_CASES,
  ...NETWORK_SERVICE_AREAS_CASES,
  ...NETWORK_NEIGHBORHOOD_CASES,
  ...NETWORK_ANALYTICS_COLUMNS_CASES,
  ...NETWORK_ISOCHRONES_CASES,
  ...NETWORK_K_FUNCTION_CASES,
  ...NETWORK_COARSENING_CASES,
  ...NETWORK_COST_WEIGHTS_CASES,
  ...NETWORK_LINE_GRAPH_CASES,
  ...NETWORK_NODING_CASES,
  ...NETWORK_STATISTICS_CASES,
  ...NETWORK_SUBGRAPH_FILTER_CASES,
  ...DRIVE_TIME_CATCHMENT_RECIPE_CASES
];
