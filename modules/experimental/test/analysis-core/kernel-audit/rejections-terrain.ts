// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the terrain cases. */
export const TERRAIN_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUTerrainSpikeRepair',
    name: 'grid above the maximum cell count',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: terrain-spike-repair grid cells must be at most 4194304$/
  },
  {
    exportName: 'GPUTerrainPeakSnap',
    name: 'candidates above the maximum',
    kind: 'binding',
    limit: 6710886,
    rejects: /^Error: terrain-peak-snap candidates rows must be at most 6710886$/
  },
  {
    exportName: 'GPUTerrainTopographicPosition',
    name: 'grid above the maximum cell count',
    kind: 'binding',
    limit: 11184810,
    rejects: /^Error: terrain-topographic-position grid cells must be at most 11184810$/
  },
  {
    exportName: 'GPUTerrainTopographicPosition',
    name: 'grid x scales above the maximum plane count',
    kind: 'binding',
    limit: 33554432,
    rejects: /^Error: terrain-topographic-position grid cells x scales must be at most 33554432$/
  },
  {
    exportName: 'GPUTerrainWeissLandforms',
    name: 'global, grid above the maximum cell count',
    kind: 'binding',
    limit: 11184810,
    rejects: /^Error: terrain-weiss-landforms grid cells must be at most 11184810$/
  },
  {
    exportName: 'GPUTerrainWeissLandforms',
    name: 'local, grid above the maximum cell count',
    kind: 'binding',
    limit: 11184810,
    rejects: /^Error: terrain-weiss-landforms grid cells must be at most 11184810$/
  }
];
