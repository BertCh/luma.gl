// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the dataframe cases. */
export const DATAFRAME_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUColumnProfile',
    name: 'category total one above the maximum, 16 columns',
    kind: 'binding',
    limit: 33554432,
    rejects: /^Error: column-profile category columns categoryCount total must be at most 33554432$/
  },
  {
    exportName: 'GPUInequality',
    name: 'zone count one above the maximum, per-zone views only',
    kind: 'binding',
    limit: 3050402,
    rejects: /^Error: inequality zoneCount must be an integer in \[1, 3050402\]$/
  },
  {
    exportName: 'GPUInequality',
    name: 'zones x 1025 knots one zone above the maximum',
    kind: 'binding',
    limit: 33554432,
    rejects: /^Error: inequality zoneCount x lorenzKnotCount must be at most 33554432$/
  },
  {
    exportName: 'GPUGroupStatistics',
    name: 'uint32x2 keys, groups one above the maximum',
    kind: 'binding',
    limit: 8388608,
    rejects: /^Error: group-statistics output.keys rows must be at most 8388608$/
  },
  ...[
    ['GPUVariogram', 'variogram', 'maximum rows and lattice, 409 bins with robust channel'],
    [
      'GPUSpatialCorrelogram',
      'spatial-correlogram',
      'cumulative bands, maximum rows, lattice and bands'
    ],
    [
      'GPUSpatialCorrelogram',
      'spatial-correlogram',
      'annulus bands, maximum rows, lattice and bands'
    ],
    ['GPURipley', 'ripley', 'maximum rows and lattice, 256 radii, every output'],
    [
      'GPURipleyEnvelope',
      'ripley-envelope',
      'maximum rows and lattice, 199 simulations, 256 radii'
    ],
    ['GPUPointPatternIndices', 'point-pattern-indices', 'maximum rows and lattice, every output'],
    [
      'GPURipleyDistanceFunctions',
      'ripley-distance-functions',
      'maximum rows, lattice and reference grid (2^24)'
    ]
  ].map(
    ([exportName, id, name]): KernelAuditRejection => ({
      exportName,
      name: `${name} (points above the maximum)`,
      kind: 'binding',
      limit: 11184810,
      rejects: new RegExp(`^Error: ${id} positions length must be an integer in \\[1, 11184810\\]$`)
    })
  )
];
