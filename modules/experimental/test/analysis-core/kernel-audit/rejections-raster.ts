// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the raster cases. */
export const RASTER_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUIsobandRings',
    name: 'edge capacity above the maximum',
    kind: 'binding',
    limit: 8388608,
    rejects: /^Error: isoband-rings edgeCapacity must be an integer in \[1, 8388608\]$/
  },
  {
    exportName: 'GPUIsobandRings',
    name: 'ring capacity above the maximum',
    kind: 'binding',
    limit: 8388608,
    rejects:
      /^Error: isoband-rings ring capacity \(output\.ringOffsets\.length - 1\) must be at most 8388608$/
  },
  {
    exportName: 'GPURasterJoin',
    name: 'zones above the maximum with sums, every output',
    kind: 'binding',
    limit: 16777215,
    rejects: /^Error: raster-join zoneCount must be an integer in \[1, 16777215\]$/
  },
  {
    exportName: 'GPURasterJoin',
    name: 'zones above the maximum, counts only',
    kind: 'binding',
    limit: 33554432,
    rejects: /^Error: raster-join zoneCount must be an integer in \[1, 33554432\]$/
  },
  {
    exportName: 'GPURasterProfile',
    name: 'paths above the maximum',
    kind: 'binding',
    limit: 8388608,
    rejects:
      /^Error: raster-profile line count \(lineOffsets\.length - 1\) must be at most 8388608$/
  },
  {
    exportName: 'GPURasterZonalStatistics',
    name: 'zones above the maximum, atomic sums, uniform cells',
    kind: 'binding',
    limit: 33554432,
    rejects: /^Error: raster-zonal-statistics zoneCapacity must be an integer in \[1, 33554432\]$/
  },
  {
    exportName: 'GPURasterZonalStatistics',
    name: 'zones above the maximum without moments, atomic sums, uniform cells',
    kind: 'binding',
    limit: 33554432,
    rejects: /^Error: raster-zonal-statistics zoneCapacity must be an integer in \[1, 33554432\]$/
  },
  {
    exportName: 'GPURasterZonalStatistics',
    name: 'zones above the maximum, atomic sums, geographic cells',
    kind: 'binding',
    limit: 33554432,
    rejects: /^Error: raster-zonal-statistics zoneCapacity must be an integer in \[1, 33554432\]$/
  },
  {
    exportName: 'GPURasterZonalStatistics',
    name: 'zones above the maximum without moments, atomic sums, geographic cells',
    kind: 'binding',
    limit: 33554432,
    rejects: /^Error: raster-zonal-statistics zoneCapacity must be an integer in \[1, 33554432\]$/
  },
  {
    exportName: 'GPURasterZonalStatistics',
    name: 'zones above the maximum, sorted sums, uniform cells',
    kind: 'binding',
    limit: 16777215,
    rejects: /^Error: raster-zonal-statistics zoneCapacity must be an integer in \[1, 16777215\]$/
  },
  {
    exportName: 'GPURasterZonalStatistics',
    name: 'zones above the maximum without moments, sorted sums, uniform cells',
    kind: 'binding',
    limit: 16777215,
    rejects: /^Error: raster-zonal-statistics zoneCapacity must be an integer in \[1, 16777215\]$/
  },
  {
    exportName: 'GPURasterZonalStatistics',
    name: 'zones above the maximum, sorted sums, geographic cells',
    kind: 'binding',
    limit: 16777215,
    rejects: /^Error: raster-zonal-statistics zoneCapacity must be an integer in \[1, 16777215\]$/
  },
  {
    exportName: 'GPURasterZonalStatistics',
    name: 'zones above the maximum without moments, sorted sums, geographic cells',
    kind: 'binding',
    limit: 16777215,
    rejects: /^Error: raster-zonal-statistics zoneCapacity must be an integer in \[1, 16777215\]$/
  }
];
