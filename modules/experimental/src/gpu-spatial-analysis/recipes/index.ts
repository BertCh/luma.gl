// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  addHotSpotAnalysisRecipe,
  GPU_HOT_SPOT_DEFAULT_MINIMUM_NEIGHBORS,
  GPU_HOT_SPOT_DEFAULT_MULTIPLE_TESTING
} from './hot-spot-analysis-recipe';
export type {
  GPUHotSpotAnalysisRecipeProps,
  GPUHotSpotAnalysisRecipeResult,
  GPUHotSpotAnalysisRecipeSettings,
  GPUHotSpotColorOptions,
  GPUHotSpotLatticeSource,
  GPUHotSpotPermutationOptions,
  GPUHotSpotPointsSource,
  GPUHotSpotStudyArea,
  GPUHotSpotDistanceBand
} from './hot-spot-analysis-recipe';
export type {GPURecipeOverrides, GPURecipeResult} from './recipe-utils';

export {
  addRateClusterMapRecipe,
  GPU_RATE_CLUSTER_MAP_PALETTE_LENGTH,
  GPU_RATE_CLUSTER_MAP_DEFAULT_MULTIPLE_TESTING
} from './rate-cluster-map-recipe';
export type {
  GPURateClusterMapRecipeProps,
  GPURateClusterMapRecipeResult,
  GPURateClusterMapRecipeSettings,
  GPURateClusterMapRate
} from './rate-cluster-map-recipe';
export {
  addSpatialRegressionRecipe,
  getSpatialRegressionSpecification,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_OLS,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_LAG,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_ERROR,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_LAG_AND_ERROR,
  GPU_SPATIAL_REGRESSION_SPECIFICATION_UNDETERMINED
} from './spatial-regression-recipe';
export type {
  GPUSpatialRegressionRecipeProps,
  GPUSpatialRegressionRecipeResult,
  GPUSpatialRegressionRecipeSettings,
  GPUSpatialRegressionLocalFitOptions
} from './spatial-regression-recipe';
export {
  addSpaceTimeHotSpotsRecipe,
  getSpaceTimeDaysFromCivil,
  getSpaceTimeSliceIndex,
  getSpaceTimeStepIndex
} from './space-time-hot-spots-recipe';
export type {
  GPUSpaceTimeHotSpotsRecipeProps,
  GPUSpaceTimeHotSpotsRecipeResult,
  GPUSpaceTimeHotSpotsRecipeSettings,
  GPUSpaceTimeSlices,
  GPUSpaceTimeCellIds,
  GPUSpaceTimeLattice,
  GPUSpaceTimeStep,
  GPUSpaceTimeOrigin
} from './space-time-hot-spots-recipe';
export {addPeriodComparisonRecipe} from './period-comparison-recipe';
export type {
  GPUPeriodComparisonRecipeProps,
  GPUPeriodComparisonRecipeResult,
  GPUPeriodRows,
  GPUPeriodComparisonVariable
} from './period-comparison-recipe';
export {addPointsInPolygonsChoroplethRecipe} from './points-in-polygons-choropleth-recipe';
export type {
  GPUChoroplethColorOptions,
  GPUChoroplethPolygons,
  GPUChoroplethStatistic,
  GPUPointsInPolygonsChoroplethRecipeProps,
  GPUPointsInPolygonsChoroplethRecipeResult
} from './points-in-polygons-choropleth-recipe';
export {addClusterAndOutlineRecipe} from './cluster-and-outline-recipe';
export type {
  GPUClusterAndOutlineRecipeProps,
  GPUClusterAndOutlineRecipeResult
} from './cluster-and-outline-recipe';
export {addChangeOfSupportRecipe} from './change-of-support-recipe';
export type {
  GPUChangeOfSupportRecipeProps,
  GPUChangeOfSupportRecipeResult,
  GPUChangeOfSupportZones
} from './change-of-support-recipe';
export {addStraightLineCatchmentsRecipe} from './straight-line-catchments-recipe';
export type {
  GPUStraightLineCatchmentsRecipeProps,
  GPUStraightLineCatchmentsRecipeResult
} from './straight-line-catchments-recipe';
export {addFleetDwellRecipe, addFleetDwellZoneEventsRecipe} from './fleet-dwell-recipe';
export type {
  GPUFleetDwellPolygonZones,
  GPUFleetDwellRecipeProps,
  GPUFleetDwellRecipeResult,
  GPUFleetDwellTrajectories,
  GPUFleetDwellZoneEventsRecipeProps,
  GPUFleetDwellZoneEventsRecipeResult,
  GPUFleetDwellZoneTable,
  GPUFleetDwellZoneTableViews
} from './fleet-dwell-recipe';
