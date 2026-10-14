// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the raster-operations cases. */
export const RASTER_OPERATIONS_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPURasterGlobalStatisticsMerge',
    name: 'size above the maximum',
    kind: 'dispatch',
    limit: 16776960,
    rejects: /^Error: gpu-raster-global-statistics-merge width x height must be at most 16776960$/
  },
  {
    exportName: 'GPURasterGlobalHistogramMerge',
    name: 'size above the maximum',
    kind: 'dispatch',
    limit: 16776960,
    rejects: /^Error: gpu-raster-global-histogram-merge width x height must be at most 16776960$/
  },
  {
    exportName: 'GPURasterRegionMeasurements',
    name: 'maximum size, capacity above the maximum',
    kind: 'dispatch',
    limit: 16776960,
    rejects: /^Error: gpu-raster-region-measurements output capacity must be at most 16776960$/
  },
  {
    exportName: 'GPURasterCrossTileComponents',
    name: 'global capacity above the maximum, two 4096x4096 tiles, connectivity 4',
    kind: 'dispatch',
    limit: 11184810,
    rejects:
      /^Error: gpu-raster-cross-tile-components global measurement capacity must be at most 11184810$/
  },
  {
    exportName: 'GPURasterCrossTileComponents',
    name: 'global capacity above the maximum, two 4096x4096 tiles, connectivity 8',
    kind: 'dispatch',
    limit: 11184810,
    rejects:
      /^Error: gpu-raster-cross-tile-components global measurement capacity must be at most 11184810$/
  },
  {
    exportName: 'GPURasterCrossTileComponents',
    name: 'total tile region capacity above the maximum, two 4096x4096 tiles, connectivity 4',
    kind: 'dispatch',
    limit: 11184810,
    rejects:
      /^Error: gpu-raster-cross-tile-components total tile region capacity must be at most 11184810$/
  },
  {
    exportName: 'GPURasterCrossTileComponents',
    name: 'total tile region capacity above the maximum, two 4096x4096 tiles, connectivity 8',
    kind: 'dispatch',
    limit: 11184810,
    rejects:
      /^Error: gpu-raster-cross-tile-components total tile region capacity must be at most 11184810$/
  },
  {
    exportName: 'GPURasterStatistics',
    name: 'size above the maximum',
    kind: 'dispatch',
    limit: 16776960,
    rejects: /^Error: gpu-raster-statistics width x height must be at most 16776960$/
  }
];
