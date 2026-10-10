// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Public type contracts with focused migration evidence in the operator catalogue. */
export const GPU_SPATIAL_ANALYSIS_CONTRACT_NAMES = [
  'GPUCompactPairPort',
  'GPUGeneratedGeometryPort',
  'GPUBoundedResultStatusPort',
  'GPUDelaunayTessellationOutput',
  'GPUVoronoiDiagramOutput',
  'GPUPolygonOverlayOutput',
  'GPUPolygonGeometryPort',
  'GPURecipeResult',
  'GPUSpatialParameterSchema',
  'GPUInferenceResultPort',
  'GPUPermutationMetadata',
  'GPUCircularStatisticsContract',
  'GPUSolverStatusPort',
  'GPUOptimizationStatusPort',
  'GPUMaxPRegionsOutput',
  'GPULocationAllocationOutput',
  'GPUNodedSegmentPort',
  'GPUPolygonizeOutput'
] as const;

/** Name of a public spatial-analysis structural contract. */
export type GPUSpatialAnalysisContractName = (typeof GPU_SPATIAL_ANALYSIS_CONTRACT_NAMES)[number];
