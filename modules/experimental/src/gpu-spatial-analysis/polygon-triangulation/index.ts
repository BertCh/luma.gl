// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK,
  GPUPolygonTriangulation,
  getGPUPolygonTriangulationMaximumNodeCount,
  getPolygonTriangulationIndexCount,
  getPolygonTriangulationNodeCount,
  validatePolygonTriangulationScratch
} from './gpu-polygon-triangulation';
export {
  GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES,
  GPU_POLYGON_TRIANGULATION_MONOTONE_BYTES_PER_VERTEX,
  getGPUPolygonTriangulationMonotoneMaximumVertexCount,
  validatePolygonTriangulationMonotoneCapacity
} from './polygon-triangulation-monotone';
export type {
  GPUPolygonTriangulationOutput,
  GPUPolygonTriangulationProps
} from './gpu-polygon-triangulation';
