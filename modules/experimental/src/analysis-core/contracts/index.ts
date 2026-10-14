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
  GPUCompactOutput,
  GPUCompactPairPort,
  GPUFeatureGeometryPort,
  GPUFloat32Positions,
  GPUGeneratedGeometryPort,
  GPULineGeometryFields,
  GPULineGeometryPort,
  GPUPackedGeometryPort,
  GPUPointGeometryPort,
  GPUPolygonGeometryFields,
  GPUPolygonGeometryPort,
  GPUSampledSurfacePort,
  GPUSpatialWeightsPort,
  GPUTrajectoryGeometryFields,
  GPUTrajectoryGeometryPort,
  GPUTrajectoryTimestampFormat,
  GPUUint32Rows
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
  GPUMeasureDescription,
  GPUSpatialMeasure
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
  getGPUAuthalicSphereRadius,
  getGPUEllipsoid,
  getGPULengthUnits,
  GPU_WGS84_ELLIPSOID,
  getGPUGreatCircleRadius,
  GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT
} from './spatial-context';

export {
  GPU_EARTH_AUTHALIC_RADIUS_METERS,
  GPU_EARTH_MEAN_RADIUS_METERS,
  GPU_SINUSOIDAL_KILOMETERS_PER_DEGREE,
  GPU_WGS84_FLATTENING,
  GPU_WGS84_INVERSE_FLATTENING,
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

// The other planners in capacity-planning.ts are internal: no contributor grow loop uses them.
export type {GPUCapacityPlan, GPUContributorCapacityPlan} from './capacity-planning';

export type {
  GPUPartitionDescriptor,
  GPUPartitionMemoryPlan,
  GPUPartitionRange,
  GPUSeamOwnership,
  GPUTilePartition,
  GPUTilePartitionDescriptor
} from './partitioning';
export {
  getGPUPartitionMemoryPlan,
  validateGPUTilePartitionDescriptor,
  validateGPUPartitionDescriptor
} from './partitioning';

export type {
  GPUCircularStatisticsContract,
  GPUInferenceAlternative,
  GPUInferenceMetadata,
  GPUInferenceResultPort,
  GPUMultipleTestingMethod,
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
