// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the cells-and-joins cases. */
export const CELLS_AND_JOINS_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUCellCover',
    name: 'quadbin candidateCapacity above the maximum',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: cell-cover candidateCapacity must be an integer in \[1, 16777216\]$/
  },
  {
    exportName: 'GPUCellCover',
    name: 'quadbin edge slab entries above the maximum',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: cell-cover positions rows \+ edgeSlabEntryCapacity must be at most 16777216$/
  },
  {
    exportName: 'GPUNeighborSearch',
    name: 'radius gridSize above the maximum',
    kind: 'binding',
    limit: 33554431,
    rejects: /^Error: neighbor-search gridSize cells must be at most 33554431$/
  },
  {
    exportName: 'GPUNearestFeatureJoin',
    name: 'neighbors node table above the maximum',
    kind: 'binding',
    limit: 33554432,
    rejects:
      /^Error: nearest-feature-join node table words \(leafCapacity, queries\) must be at most 33554432$/
  },
  {
    exportName: 'GPUSpatialJoinCandidates',
    name: 'polygons pair capacity above the maximum',
    kind: 'binding',
    limit: 16777216,
    rejects:
      /^Error: spatial-join-candidates output.leftIds.length must be an integer in \[1, 16777216\]$/
  },
  {
    exportName: 'GPUTrackSimilarity',
    name: 'pairs above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: track-similarity pairA.length must be an integer in \[1, 16777215\]$/
  },
  {
    exportName: 'GPUTrajectoryEncounters',
    name: 'lattice above the maximum',
    kind: 'binding',
    limit: 33554431,
    rejects: /^Error: trajectory-encounters columns x rows x bucketCount must be at most 33554431$/
  },
  {
    exportName: 'GPUTrajectoryPlayhead',
    name: 'tracks above the maximum',
    kind: 'binding',
    limit: 6710886,
    rejects: /^Error: trajectory-playhead tracks must be at most 6710886$/
  }
];
