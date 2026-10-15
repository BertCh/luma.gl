// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuDF.

// Every analysis subdirectory index is re-exported whole, so the entry cannot drift from its parts
// and a name exported by two subdirectories is a compile error. Visualization packs stay internal.

export * from './calendar-buckets/index';
export * from './column-classification/index';
export * from './column-profile/index';
export * from './composite-indicators/index';
export * from './group-statistics/index';
export * from './key-join/index';
export * from './pair-statistics/index';
export * from './temporal-reduction/index';
export * from './time-window-filter/index';

// Top-level files and shared contracts.
export {GPUDataFrame} from './gpu-data-frame';
export type {
  GPUDataFrameColumn,
  GPUDataFrameDictionaries,
  GPUDataFrameDictionary,
  GPUDataFrameOwnership,
  GPUDataFrameProps,
  GPUDataFrameSourceInfo,
  GPUDataFrameValidity
} from './gpu-data-frame';
export {GPUDataFrameQuery} from './gpu-data-frame-query';
export type {
  GPUDataFrameDerivedColumn,
  GPUDataFrameDerivedColumnFormat,
  GPUDataFrameDerivedColumnFormatForExpression,
  GPUDataFrameDerivedColumnOptions
} from './gpu-data-frame-query';
export {GPUDataFrameGroupByQuery, GPUDataFrameGroupedAggregationQuery} from './gpu-group-by-query';
export type {
  GPUDataFrameAggregationDefinition,
  GPUDataFrameAggregationDefinitions,
  GPUDataFrameAggregationOperation,
  GPUDataFrameAggregationValue,
  GPUDataFrameColumnNamesOfFormat,
  GPUDataFrameGroupByOptions,
  GPUDataFrameGroupedAggregationResult
} from './gpu-group-by-query';
export {GPUDataFrameAggregationQuery} from './gpu-global-aggregation-query';
export type {
  GPUDataFrameAnalyticScalarFormat,
  GPUDataFrameGlobalAggregationDefinitions,
  GPUDataFrameGlobalAggregationResult,
  GPUDataFrameGlobalAggregationValue,
  GPUDataFrameScalarColumnNames
} from './gpu-global-aggregation-query';
export {GPUDataFrameHistogramQuery} from './gpu-histogram-query';
export type {GPUDataFrameHistogramOptions} from './gpu-histogram-query';
export {GPUDataFrameGlobalSortQuery, GPUDataFrameSortQuery} from './gpu-sort-query';
export type {GPUDataFrameSortOptions} from './gpu-sort-query';
export {GPUDataFrameJoinQuery, GPUDataFrameLookupQuery} from './gpu-join-query';
export type {
  GPUDataFrameJoinOptions,
  GPUDataFrameJoinType,
  GPUDataFrameLookupOptions
} from './gpu-join-query';
export {and, column, literal, GPUExpression, not, or, parameter} from './gpu-expression';
export type {
  GPUExpressionBinaryOperator,
  GPUExpressionNode,
  GPUExpressionUnaryOperator,
  GPUExpressionValue
} from './gpu-expression';
export {CompiledGPUDataFrameQuery} from './gpu-query-compiler';
export type {GPUDataFrameQueryParameters} from './gpu-query-compiler';
export {CompiledGPUDataFrameGroupedAggregation} from './gpu-group-aggregation-compiler';
export {CompiledGPUDataFrameAggregation} from './gpu-global-aggregation-compiler';
export {CompiledGPUDataFrameHistogram} from './gpu-histogram-compiler';
export {CompiledGPUDataFrameSort} from './gpu-sort-compiler';
export {CompiledGPUDataFrameGlobalSort} from './gpu-global-sort-compiler';
export {CompiledGPUDataFrameJoin, CompiledGPUDataFrameLookup} from './gpu-join-compiler';
export type {GPUCompactOutput, GPUUint32Rows} from '../utils/gpu-contributor-types';
export {GPUParameterBuffer} from '../utils/gpu-contributor-utils';
export type {GPUParameterBufferProps, GPUParameterFormat} from '../utils/gpu-contributor-utils';

export type {
  GPUDataFrameQueryDescription,
  GPUDataFrameExpressionStatement
} from './gpu-query-describe';
