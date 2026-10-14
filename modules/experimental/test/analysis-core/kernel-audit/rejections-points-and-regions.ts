// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the points-and-regions cases. */
export const POINTS_AND_REGIONS_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUKernelDensity',
    name: 'index grid above the maximum gaussian',
    kind: 'binding',
    limit: 33554431,
    rejects: /^Error: kernel-density indexGridSize cells must be at most 33554431$/
  },
  {
    exportName: 'GPUKernelDensity',
    name: 'index grid above the maximum epanechnikov',
    kind: 'binding',
    limit: 33554431,
    rejects: /^Error: kernel-density indexGridSize cells must be at most 33554431$/
  },
  {
    exportName: 'GPUArealInterpolation',
    name: 'cells above the maximum, pairs at the maximum',
    kind: 'binding',
    limit: 33554431,
    rejects: /^Error: areal-interpolation sourceZones rows must be at most 33554431$/
  },
  {
    exportName: 'GPUArealInterpolation',
    name: 'cells at the maximum, pairs above the maximum',
    kind: 'binding',
    limit: 16777215,
    rejects:
      /^Error: areal-interpolation pair slots \(output\.weights\.neighbors rows, cells\) must be at most 16777215$/
  },
  {
    exportName: 'GPUDotDensity',
    name: 'capacity above the maximum',
    kind: 'binding',
    limit: 8388608,
    rejects: /^Error: dot-density output\.positions rows must be at most 8388608$/
  },
  {
    exportName: 'GPULocationAllocation',
    name: 'facilities above the maximum set-covering',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: location-allocation facilityCount must be an integer in \[1, 16777216\]$/
  },
  {
    exportName: 'GPULocationAllocation',
    name: 'sparse entries above the maximum',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: location-allocation sparseCosts entries must be at most 16777216$/
  },
  {
    exportName: 'GPUSimilarLocations',
    name: 'columns above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: similar-locations columnCount must be an integer in \[1, 16777215\]$/
  },
  {
    exportName: 'GPUSpatialClustering',
    name: 'rows and clusters, index grid above the maximum atomic',
    kind: 'binding',
    limit: 33554431,
    rejects: /^Error: spatial-clustering gridSize cells must be at most 33554431$/
  },
  {
    exportName: 'GPUSpatialClustering',
    name: 'rows and clusters, index grid at the maximum sorted, clusters above the sorted maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects:
      /^Error: spatial-clustering clusters\.ids rows with sorted centroid sums must be at most 16777215$/
  },
  {
    exportName: 'GPUSpatialClustering',
    name: 'rows and clusters, index grid above the maximum sorted, clusters above the sorted maximum',
    kind: 'dispatch',
    limit: 16777215,
    // The device-independent cluster check runs in the constructor, before the grid check at declaration.
    rejects:
      /^Error: spatial-clustering clusters\.ids rows with sorted centroid sums must be at most 16777215$/
  },
  {
    exportName: 'GPUSpatialClustering',
    name: 'rows and clusters, index grid above the maximum sorted below the segment-sum dispatch limit',
    kind: 'dispatch',
    limit: 33554431,
    rejects: /^Error: spatial-clustering gridSize cells must be at most 33554431$/
  },
  {
    exportName: 'GPUInverseDistanceWeighting',
    name: 'index grid above the maximum',
    kind: 'binding',
    limit: 33554431,
    rejects: /^Error: inverse-distance-weighting indexGridSize cells must be at most 33554431$/
  },
  {
    exportName: 'GPUKriging',
    name: 'index grid above the maximum',
    kind: 'binding',
    limit: 33554431,
    rejects: /^Error: kriging indexGridSize cells must be at most 33554431$/
  },
  {
    exportName: 'GPUSkaterRegions',
    name: 'slots 16 columns above the maximum 1024 regions',
    kind: 'binding',
    limit: 33554432,
    rejects:
      /^Error: skater-regions weights\.neighbors rows x \(2 \+ columnCount\) must be at most 33554432$/
  },
  {
    exportName: 'GPURegionPartitionEvaluation',
    name: 'rows above the maximum 1 columns',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: region-partition-evaluation labels rows must be at most 16777216$/
  },
  {
    exportName: 'GPUAZPRegions',
    name: 'slots above the maximum',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: azp-regions weights\.neighbors rows must be at most 16777216$/
  },
  {
    exportName: 'GPUWardRegions',
    name: 'slots above the maximum',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: ward-regions weights\.neighbors rows must be at most 16777216$/
  },
  {
    exportName: 'GPUMaxPRegions',
    name: 'slots above the maximum',
    kind: 'binding',
    limit: 8388608,
    rejects: /^Error: max-p-regions weights\.neighbors rows must be at most 8388608$/
  }
];
