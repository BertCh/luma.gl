// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {GPU_SPATIAL_ANALYSIS_CONTRACT_NAMES} from './contract-names';
export type {GPUSpatialAnalysisContractName} from './contract-names';

export type {
  GPUApproximationStatusPort,
  GPUBoundedResultStatusPort,
  GPUCandidateStatusPort,
  GPUCardinalityPort,
  GPUInvalidStatusPort,
  GPUIterationStatusPort,
  GPUOverflowStatusPort,
  GPURecipeStageStatusPort,
  GPURecipeStatusPort,
  GPUStatusScalar,
  GPUUncertaintyStatusPort
} from './status';
export {validateGPURecipeStatusPort, validateGPUStatusPort} from './status';

export type {
  GPUCellTablePort,
  GPUClassifiedValuesPort,
  GPUCompactPairPort,
  GPUCoordinateRows,
  GPUFeatureGeometryPort,
  GPUGeneratedGeometryPort,
  GPUIdentifierRows,
  GPULineGeometryPort,
  GPUPointGeometryPort,
  GPUPolygonGeometryPort,
  GPUSampledSurfacePort,
  GPUSpatialWeightsPort,
  GPUTrajectoryGeometryPort,
  GPUTrajectoryTimestampFormat
} from './ports';
export {validateGPUCompactPairPort, validateGPUFeatureGeometryPort} from './ports';

export type {
  GPUSpatialContext,
  GPUSpatialContextCompatibility,
  GPUSpatialContextSupport,
  GPUSpatialCoordinateSpace,
  GPUSpatialEllipsoid,
  GPUSpatialMetric,
  GPUSpatialUnit,
  GPUSpatialFrame,
  GPUSpatialFrameKind,
  GPUAreaModel,
  GPUAreaUnit,
  GPULengthModel,
  GPUMeasureDescription
} from './spatial-context';
export {
  assertCompatibleGPUSpatialContexts,
  getGPUSpatialContextCompatibility,
  GPU_PLANAR_SPATIAL_CONTEXT,
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  validateGPUSpatialContext,
  getGPUAreaUnit,
  getGPUSphereRadius,
  GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT
} from './spatial-context';

export {
  GPU_EARTH_AUTHALIC_RADIUS_METERS,
  GPU_EARTH_MEAN_RADIUS_METERS,
  GPU_WGS84_FLATTENING,
  GPU_WGS84_SEMI_MAJOR_AXIS_METERS
} from './earth';

export type {GPULocalFrame, GPULocalFrameKind, GPULocalFrameProps} from './local-frame';

export {createGPULocalFrame} from './local-frame';

export type {
  GPUSpatialParameterArray,
  GPUSpatialParameterField,
  GPUSpatialParameterFormat,
  GPUSpatialParameterSchema,
  GPUSpatialParameterValues
} from './parameter-schema';
export {
  defineGPUSpatialParameterSchema,
  packGPUSpatialParameterValues,
  validateGPUSpatialParameterSchema
} from './parameter-schema';

export type {
  GPUCellCoverCapacityPlan,
  GPUCellCoverCapacityPlanningOptions,
  GPUCapacityObservation,
  GPUCapacityPlan,
  GPUCapacityPlanningOptions,
  GPUCapacityRecovery,
  GPUCapacityRecoveryPolicy,
  GPUCapacityStageObservation,
  GPUContributorCapacityPlan,
  GPUIncompleteCapacityStage,
  GPULineTopologyCapacityPlan,
  GPULineTopologyCapacityPlanningOptions,
  GPUSpatialJoinCapacityPlan,
  GPUSpatialJoinCapacityPlanningOptions,
  GPUTrajectoryEncounterCapacityPlan,
  GPUTrajectoryEncounterCapacityPlanningOptions
} from './capacity-planning';

export type {
  GPUPartitionDescriptor,
  GPUPartitionMemoryPlan,
  GPUPartitionRange,
  GPUPartitionedGeometryPort,
  GPUPreparedIndexDescriptor,
  GPUSeamOwnership,
  GPUTilePartition,
  GPUTilePartitionDescriptor,
  GPUSpatialQueryCostMeasurements,
  GPUSpatialQueryCostPlan,
  GPUSpatialQueryStrategy
} from './partitioning';
export {
  assertCompatibleGPUPreparedIndexes,
  getGPUPartitionMemoryPlan,
  getGPUSpatialQueryCostPlan,
  getGPUTileSeamOwner,
  isGPUSeamResultOwner,
  validateGPUTilePartitionDescriptor,
  validateGPUPartitionDescriptor
} from './partitioning';
export {
  getGPUCapacityPlan,
  getGPUCapacityRecovery,
  getGPUCellCoverContributorCapacityPlan,
  getGPUCellCapacityPlan,
  getGPUEventCapacityPlan,
  getGPUGeneratedGeometryCapacityPlan,
  getGPULineTopologyCapacityPlan,
  getGPUNeighborhoodCapacityPlan,
  getGPUPairCapacityPlan,
  getGPUSpatialJoinCapacityPlan,
  getGPUTrajectoryEncounterCapacityPlan
} from './capacity-planning';

export type {
  GPUCircularStatisticsContract,
  GPUInferenceAlternative,
  GPUInferenceMetadata,
  GPUInferenceResultPort,
  GPUMultipleTestingMethod,
  GPUOptimizationStatusPort,
  GPUPermutationMetadata,
  GPUPermutationStatisticAdapter,
  GPUSolverStatusPort
} from './inference';
export {GPU_DEFAULT_INFERENCE_ALTERNATIVE} from './inference';

export {
  createGPURegressionModelResult,
  getGPURegressionPredictorNames,
  GPU_REGRESSION_PRECISION_CONDITION_LIMIT
} from './model-result';

export type {
  GPURegressionModelResult,
  GPURegressionModelResultInput,
  GPURegressionModelKind,
  GPURegressionCoefficient,
  GPURegressionCoefficientRole,
  GPURegressionStatisticKind,
  GPUSigmaSquaredConvention,
  GPURegressionConditioning,
  GPURegressionTest,
  GPURegressionModelStatus
} from './model-result';
