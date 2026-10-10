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
  GPUSpatialCoordinateSpace,
  GPUSpatialEllipsoid,
  GPUSpatialMetric,
  GPUSpatialUnit
} from './spatial-context';
export {
  assertCompatibleGPUSpatialContexts,
  getGPUSpatialContextCompatibility,
  validateGPUSpatialContext
} from './spatial-context';

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
  GPUInferenceResultPort,
  GPUMultipleTestingMethod,
  GPUOptimizationStatusPort,
  GPUPermutationMetadata,
  GPUPermutationStatisticAdapter,
  GPUSolverStatusPort
} from './inference';
