// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the line, buffer, clearance and triangulation geometry cases. */
export const GEOMETRY_LINES_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPULineDensity',
    name: 'grid above the maximum',
    kind: 'dispatch',
    limit: 16777215,
    rejects: /^Error: line-density columns x rows must be at most 16777215$/
  },
  {
    exportName: 'GPULineLengthPerPolygon',
    name: 'positions above the maximum',
    kind: 'binding',
    limit: 8388608,
    rejects: /^Error: line-length-per-polygon positions rows must be at most 8388608$/
  },
  {
    exportName: 'GPULinearReferencing',
    name: 'planar queries above the maximum',
    kind: 'binding',
    limit: 3728270,
    rejects: /^Error: linear-referencing queryPositions rows must be at most 3728270$/
  },
  {
    exportName: 'GPULineLocate',
    name: 'events above the maximum',
    kind: 'binding',
    limit: 5592405,
    rejects: /^Error: line-locate eventLineIndices rows must be at most 5592405$/
  },
  {
    exportName: 'GPUBufferSurface',
    name: 'polygons above the maximum',
    kind: 'binding',
    limit: 786432,
    rejects: /^Error: buffer-surface boundary rows must be at most 786432$/
  },
  {
    exportName: 'GPUPolygonTriangulation',
    name: 'vertices above the maximum',
    kind: 'binding',
    limit: 4793490,
    rejects: /^Error: polygon-triangulation scratch nodes must be at most 4793490$/
  },
  {
    exportName: 'GPUMinimumClearance',
    name: 'polygons above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: minimum-clearance geometry.positions rows must be at most 4194304$/
  },
  {
    exportName: 'GPULineLengthPerPolygon',
    name: 'every output, maximum crossings 1025',
    kind: 'binding',
    limit: 1024,
    rejects:
      /^Error: line-length-per-polygon maximumCrossingCount must be an integer in \[1, 1024\]$/
  },
  {
    exportName: 'GPULineLengthPerPolygon',
    name: 'every output, maximum crossings 4294967295',
    kind: 'binding',
    limit: 1024,
    rejects:
      /^Error: line-length-per-polygon maximumCrossingCount must be an integer in \[1, 1024\]$/
  },
  {
    exportName: 'GPUOffsetCurve',
    name: 'one vertex with quadSegments above the maximum',
    kind: 'binding',
    limit: 256,
    rejects: /^Error: offset-curve quadSegments must be an integer in \[1, 256\]$/
  },
  {
    exportName: 'GPURandomPointsInPolygon',
    name: 'triangulation scratch above the maximum',
    kind: 'binding',
    limit: 4793490,
    rejects: /^Error: random-points-in-polygon scratch nodes must be at most 4793490$/
  },
  {
    exportName: 'GPULineLengthPerPolygon',
    name: 'pairs above the maximum',
    kind: 'binding',
    limit: 16777216,
    rejects: /^Error: line-length-per-polygon pairCapacity must be an integer in \[1, 16777216\]$/
  },
  {
    exportName: 'GPULinearReferencing',
    name: 'great-circle positions above the maximum',
    kind: 'binding',
    limit: 2097152,
    rejects: /^Error: linear-referencing positions rows must be at most 2097152$/
  },
  {
    exportName: 'GPUBufferSurface',
    name: 'lines above the maximum',
    kind: 'binding',
    limit: 786432,
    rejects: /^Error: buffer-surface boundary rows must be at most 786432$/
  },
  {
    exportName: 'GPUPolygonTriangulation',
    name: 'vertices above the maximum without z-order hash',
    kind: 'binding',
    limit: 8388608,
    rejects: /^Error: polygon-triangulation scratch nodes must be at most 8388608$/
  },
  {
    exportName: 'GPULinearReferencing',
    name: 'planar positions above the maximum',
    kind: 'binding',
    limit: 8388608,
    rejects: /^Error: linear-referencing positions rows must be at most 8388608$/
  }
];
