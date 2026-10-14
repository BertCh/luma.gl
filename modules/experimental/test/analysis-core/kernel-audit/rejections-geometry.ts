// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the geometry cases, one step above each documented maximum. */
export const GEOMETRY_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPUSegmentIntersection',
    name: 'vertices above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: segment-intersection left\.positions rows must be at most 4194304$/
  },
  {
    exportName: 'GPUGeometryPredicates',
    name: 'polygons above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: geometry-predicates geometry\.positions rows must be at most 4194304$/
  },
  {
    exportName: 'GPULineNoding',
    name: 'vertices above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: line-noding lines\.positions rows must be at most 4194304$/
  },
  {
    exportName: 'GPULineNoding',
    name: 'piece bound above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: line-noding lines \+ 4 x intersectionCapacity must be at most 4194304$/
  },
  {
    exportName: 'GPUPolygonize',
    name: 'vertices above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: polygonize lines\.positions rows must be at most 4194304$/
  },
  {
    exportName: 'GPUMakeValid',
    name: 'vertices above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: make-valid lines\.positions rows must be at most 4194304$/
  },
  {
    exportName: 'GPUGeometryValidity',
    name: 'polygons above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: geometry-validity polygons\.positions rows must be at most 4194304$/
  },
  {
    exportName: 'GPULineClipByPolygon',
    name: 'line vertices above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects:
      /^Error: line-clip-by-polygon lines\.positions rows \+ 2 x intersectionCapacity must be at most 4194304$/
  },
  {
    exportName: 'GPULineClipByPolygon',
    name: 'polygon vertices above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: line-clip-by-polygon polygons\.positions rows must be at most 4194304$/
  },
  {
    exportName: 'GPULineClipByPolygon',
    name: 'sub-piece bound above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects:
      /^Error: line-clip-by-polygon lines\.positions rows \+ 2 x intersectionCapacity must be at most 4194304$/
  },
  {
    exportName: 'GPUSharedPaths',
    name: 'vertices above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: shared-paths left\.positions rows must be at most 4194304$/
  },
  {
    exportName: 'GPUSharedPaths',
    name: 'intersections above the maximum',
    kind: 'binding',
    limit: 2796202,
    rejects: /^Error: shared-paths intersectionCapacity must be an integer in \[1, 2796202\]$/
  },
  {
    exportName: 'GPULineSplit',
    name: 'vertices above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: line-split lines\.positions rows must be at most 4194304$/
  },
  {
    exportName: 'GPULineSplit',
    name: 'piece bound above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: line-split lines \+ 4 x intersectionCapacity must be at most 4194304$/
  },
  {
    exportName: 'GPUPolygonOverlay',
    name: 'vertices above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects:
      /^Error: polygon-overlay left \+ right positions rows \+ rings must be at most 4194304$/
  },
  {
    exportName: 'GPUPolygonOverlay',
    name: 'hot pixels above the maximum',
    kind: 'binding',
    limit: 1048576,
    rejects:
      /^Error: polygon-overlay left \+ right positions rows \+ rings \+ capacity\.intersections must be at most 1048576$/
  },
  {
    exportName: 'GPUCoverageSimplification',
    name: 'vertices above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: coverage-simplification positions rows must be at most 4194304$/
  },
  {
    exportName: 'GPUCoverageValidity',
    name: 'vertices above the maximum',
    kind: 'binding',
    limit: 4194304,
    rejects: /^Error: coverage-validity positions rows must be at most 4194304$/
  }
];
