// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPU_TABLE_INDEX_COLUMN_NAME,
  isGPUTableIndexColumnName,
  type GPUField,
  type GPUSchema,
  type GPUTypeMap
} from './table/gpu-schema';
export {
  GPURecordBatch,
  type GPUDataMap,
  type GPURecordBatchFromDataProps,
  type GPURecordBatchProps,
  type GPURecordBatchSourceInfo
} from './table/gpu-record-batch';
export {
  GPUTable,
  type GPUColumn,
  type GPUColumnMap,
  type GPUTableDetachBatchesOptions,
  type GPUTableFromBatchesProps,
  type GPUTableFromColumnsProps,
  type GPUTableFromSchemaProps,
  type GPUTableFromVectorsProps,
  type GPUTablePackBatchesOptions,
  type GPUTableProps
} from './table/gpu-table';
export {GPUTableGeometry, type GPUTableGeometryProps} from './engine/gpu-table-geometry';
export {
  GPUTableModel,
  type GPUTableModelCount,
  type GPUTableModelDrawBatchesOptions,
  type GPUTableModelProps
} from './engine/gpu-table-model';
export {GPURenderable} from './engine/gpu-renderable';
export {
  TableTransform,
  type TableTransformBatchOptions,
  type TableTransformOutputCopyMap,
  type TableTransformProps
} from './engine/gpu-table-transform';
export {
  GPUTableComputation,
  type GPUTableComputationBatch,
  type GPUTableComputationProps
} from './engine/gpu-table-computation';
export {
  getGPUInputAttributeNames,
  validateGPUInputVectors,
  type GPUInputColumns,
  type GPUInputDeclaration,
  type GPUInputKind,
  type GPUInputSchema,
  type GPUInputVectors
} from './engine/gpu-input-schema';
export {
  GPUTableShaderBindings,
  getGPUTableRowMultiplierFieldName,
  type GPUTableShaderBindingBatch,
  type GPUTableShaderBindingsProps
} from './engine/gpu-table-shader-bindings';
export {
  makeGPUSceneFromCPUScene,
  makeGPUScenePartitionsFromGPUTable,
  type GPUSceneCPUAdapterContext,
  type GPUSceneCPUAdapterProps,
  type GPUSceneTableAdapterProps,
  type GPUSceneTableAdapterResult,
  type GPUSceneTableAdapterStats,
  type GPUSceneTableColumnNames,
  type GPUSceneTablePartition
} from './engine/gpu-scene-adapters';
export {
  GPUTableBufferPlanner,
  type GPUTableBufferGroup,
  type GPUTableBufferGroupKind,
  type GPUTableBufferMapping,
  type GPUTableBufferPlan,
  type GPUTableBufferPlannerMode,
  type GPUTableBufferPlannerModelInfo,
  type GPUTableBufferPlannerProps,
  type GPUTableColumnDescriptor,
  type GPUTableColumnPriority,
  type GPUTablePlannedColumn
} from './utils/gpu-table-buffer-planner';
export {
  getGeneratedBufferBatchByteLimit,
  planGeneratedBufferBatches,
  type GeneratedBufferBatch,
  type GeneratedBufferBatchPlannerProps
} from './utils/generated-buffer-batches';

export type {GPUCompactOutput, GPUUint32Rows} from '../utils/gpu-contributor-types';
export {GPUParameterBuffer} from '../utils/gpu-contributor-utils';
export type {GPUParameterBufferProps, GPUParameterFormat} from '../utils/gpu-contributor-utils';

export {
  GPU_RESIDENCY_ARENA_DEAD_SLOT,
  GPUResidencyArena,
  GPUResidentRowSelection,
  ResidencyArenaAllocator,
  ResidencyArenaFullError
} from './residency-arena/index';
export type {
  GPUResidencyArenaColumnFormat,
  GPUResidencyArenaColumnSpec,
  GPUResidencyArenaGraphViews,
  GPUResidencyArenaProps,
  GPUResidencyArenaResolvedRow,
  GPUResidencyArenaRowRange,
  GPUResidencyArenaTile,
  GPUResidencyArenaTileData,
  GPUResidentRowSelectionProps,
  ResidencyArenaAllocatorProps
} from './residency-arena/index';

export {
  getGPUTileLODFrustumPlanes,
  getGPUTileLODQuadtreeTile,
  getGPUTileLODViewParameterValues,
  GPU_TILE_LOD_BUDGET_LENGTH,
  GPU_TILE_LOD_INVALID_NODE,
  GPU_TILE_LOD_PRIORITY_BUCKET_COUNT,
  GPU_TILE_LOD_STATISTICS_LENGTH,
  GPU_TILE_LOD_UNLIMITED,
  GPU_TILE_LOD_VIEW_LENGTH,
  GPU_TILE_LOD_VIEW_OFFSETS,
  GPUTileLODSelection,
  makeGPUTileLODQuadtree
} from './tile-lod-selection/index';
export type {
  GPUTileLODFoveation,
  GPUTileLODHierarchy,
  GPUTileLODIndirectDispatch,
  GPUTileLODIndirectDraw,
  GPUTileLODQuadtree,
  GPUTileLODQuadtreeProps,
  GPUTileLODQuadtreeTile,
  GPUTileLODRequestOutput,
  GPUTileLODSelectionProps,
  GPUTileLODViewProps
} from './tile-lod-selection/index';
