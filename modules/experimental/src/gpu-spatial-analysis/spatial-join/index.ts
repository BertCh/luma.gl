// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPU_SPATIAL_JOIN_NO_DISTANCE,
  GPU_SPATIAL_JOIN_NO_FEATURE
} from './spatial-join-types';
export type {GPUNearestFeatureSegments, GPUNearestFeatureSource} from './spatial-join-types';
export {GPUPointInPolygonJoin} from './gpu-point-in-polygon-join';
export type {
  GPUPointInPolygonBoundaryRule,
  GPUPointInPolygonJoinOutput,
  GPUPointInPolygonJoinProps
} from './gpu-point-in-polygon-join';
export {
  GPUNearestFeatureJoin,
  getGPUNearestFeatureJoinMaximumLeafCapacity,
  getGPUNearestFeatureJoinMaximumNodeTableWordCount
} from './gpu-nearest-feature-join';
export type {
  GPUNearestFeatureJoinOutput,
  GPUNearestFeatureJoinProps
} from './gpu-nearest-feature-join';
export {GPUBufferSelection} from './gpu-buffer-selection';
export type {
  GPUBufferSelectionOutput,
  GPUBufferSelectionProps
} from './gpu-buffer-selection';
export {GPUSpatialPredicateJoin} from './gpu-spatial-predicate-join';
export type {
  GPUSpatialJoinHow,
  GPUSpatialPredicate,
  GPUSpatialPredicateJoinObservation,
  GPUSpatialPredicateJoinOutput,
  GPUSpatialPredicateJoinProps
} from './gpu-spatial-predicate-join';
export {GPUSpatialJoinPrepared} from './gpu-spatial-join-prepared';
export type {
  GPUSpatialJoinPreparedProps,
  GPUSpatialJoinPreparedStorage
} from './gpu-spatial-join-prepared';
export {
  GPUSpatialJoinCandidates,
  getGPUSpatialJoinCandidatesMaximumPairCapacity
} from './gpu-spatial-join-candidates';
export type {GPUSpatialJoinCandidatesProps} from './gpu-spatial-join-candidates';
export {
  formatGPUSpatialRelate,
  GPU_SPATIAL_RELATE_CELLS,
  GPU_SPATIAL_RELATE_PATTERN_WORDS,
  GPU_SPATIAL_RELATE_UNCERTAIN_BIT,
  packGPUSpatialRelate,
  packGPUSpatialRelatePattern
} from './spatial-relate-types';
export type {GPUSpatialRelatePattern} from './spatial-relate-types';
export type {GPUSpatialJoinGeometry, GPUSpatialJoinOnAttribute} from './spatial-join-types';
export {GPU_NEAREST_NO_SEGMENT} from './nearest-types';
export type {
  GPUNearestFeatureGeometry,
  GPUNearestQueryGeometry,
  GPUNearestTieMode
} from './nearest-types';
export {GPUNearestFeatureWeights} from './gpu-nearest-feature-weights';
export type {GPUNearestFeatureWeightsProps} from './gpu-nearest-feature-weights';
export type {SpatialSortCurve} from './spatial-join-passes';
export {getGPUSpatialJoinCapacityPlan} from './spatial-join-capacity-plan';
export type {
  GPUSpatialJoinCapacityPlan,
  GPUSpatialJoinCapacityPlanningOptions
} from './spatial-join-capacity-plan';
