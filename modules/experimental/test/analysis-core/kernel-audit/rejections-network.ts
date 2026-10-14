// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the network cases. */
export const NETWORK_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUFlowAggregation',
    name: 'maximum pairCapacity 2^23 with weights and sorted sums, rows at the binding limit',
    kind: 'dispatch',
    limit: 4294967295,
    rejects: /^Error: flow-aggregation rows x maximumProbeCount must be at most 4294967295$/
  },
  {
    exportName: 'GPUMapMatching',
    name: 'entryCapacity above the maximum',
    kind: 'binding',
    limit: 33554432,
    rejects: /^Error: map-matching entryCapacity must be an integer in \[1, 33554432\]$/
  },
  {
    exportName: 'GPUMapMatching',
    name: 'edges above the maximum',
    kind: 'binding',
    limit: 11184810,
    rejects: /^Error: map-matching topology edges must be at most 11184810$/
  },
  {
    exportName: 'GPUNetworkSnapping',
    name: 'candidate search, edges above the maximum',
    kind: 'binding',
    limit: 8388608,
    rejects:
      /^Error: network-snapping topology edges with candidateCapacity must be at most 8388608$/
  },
  {
    exportName: 'GPUNetworkReachability',
    name: 'nodes above the maximum',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: network-reachability output.costs rows must be at most 16777216$/
  },
  {
    exportName: 'GPUNetworkKFunction',
    name: 'events at the binding limit, row blocks past the invocation limit',
    kind: 'dispatch',
    limit: 4294967295,
    rejects: /^Error: network-k-function rowsPerBlock x events must be at most 4294967295$/
  },
  {
    exportName: 'GPUNetworkCoarsening',
    name: 'weighted superedges above the maximum',
    kind: 'binding',
    limit: 16777216,
    rejects:
      /^Error: network-coarsening output.edges.ids rows with output.edgeWeights must be at most 16777216$/
  },
  {
    exportName: 'GPUNetworkLineGraph',
    name: 'planar, base edges above the maximum, arcs and banned turns at the binding limit',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: network-line-graph topology edges must be at most 16777216$/
  },
  {
    exportName: 'GPUNetworkLineGraph',
    name: 'great-circle, base edges above the maximum, arcs and banned turns at the binding limit',
    kind: 'binding',
    limit: 8388608,
    rejects:
      /^Error: network-line-graph topology edges with a great-circle spatialContext must be at most 8388608$/
  },
  {
    exportName: 'GPUNetworkNoding',
    name: 'intersection capacity above the maximum',
    kind: 'binding',
    limit: 1048576,
    rejects: /^Error: network-noding intersectionCapacity must be an integer in \[1, 1048576\]$/
  },
  {
    exportName: 'GPUNetworkKFunction',
    name: 'events at the binding limit, row blocks at the invocation limit, seed table above the maximum',
    kind: 'binding',
    limit: 33554432,
    rejects:
      /^Error: network-k-function expanded seed words \(rows, rowsPerBlock, laneCount\) must be at most 33554432$/
  }
];
