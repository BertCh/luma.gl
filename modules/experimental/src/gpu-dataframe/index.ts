// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuDF.

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

// Analysis contributors
export type {GPUCompactOutput, GPUUint32Rows} from '../utils/gpu-contributor-types';
export {GPUParameterBuffer} from '../utils/gpu-contributor-utils';
export type {GPUParameterBufferProps, GPUParameterFormat} from '../utils/gpu-contributor-utils';
export {
  GPUColumnProfile,
  getGPUColumnProfileParameterLength,
  getGPUColumnProfileParameterValues,
  GPU_COLUMN_PROFILE_MAXIMUM_COLUMN_COUNT,
  GPU_COLUMN_PROFILE_MAXIMUM_TOP_CATEGORY_COUNT,
  GPU_COLUMN_PROFILE_NULL_CATEGORY,
  GPU_COLUMN_PROFILE_PARAMETERS_PER_COLUMN,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from './column-profile/index';
export type {
  GPUColumnProfileColumn,
  GPUColumnProfileDomain,
  GPUColumnProfileOutput,
  GPUColumnProfileProps
} from './column-profile/index';
export {
  GPUColumnQuantiles,
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesParameterValues,
  GPU_COLUMN_QUANTILE_INTERPOLATION_CODES,
  GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT,
  GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH,
  GPUClassBreaks,
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT,
  GPU_CLASS_BREAKS_METHOD_CODES,
  GPU_CLASS_BREAKS_METHODS,
  GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH,
  GPUColorScale,
  getGPUColorScaleParameterValues,
  GPU_COLOR_SCALE_CODES,
  GPU_COLOR_SCALE_NO_CLASS,
  GPU_COLOR_SCALE_PARAMETER_LENGTH,
  packGPUColor,
  GPUBivariateClassification,
  getGPUBivariateClassificationParameterValues,
  GPU_BIVARIATE_CLASSIFICATION_NO_CLASS,
  GPU_BIVARIATE_CLASSIFICATION_PARAMETER_LENGTH
} from './column-classification/index';
export type {
  GPUColumnQuantilesOutput,
  GPUColumnQuantilesProps,
  GPUColumnQuantileInterpolation,
  GPUColumnQuantilesParameterInput,
  GPUClassBreaksOutput,
  GPUClassBreaksProps,
  GPUClassBreaksMethod,
  GPUClassBreaksParameters,
  GPUColorScaleOutput,
  GPUColorScaleProps,
  GPUColorScaleInterpolation,
  GPUColorScaleParameterOptions,
  GPUColorScaleType,
  GPUBivariateClassificationOutput,
  GPUBivariateClassificationProps,
  GPUBivariateClassificationParameterOptions,
  GPUBivariateValueByAlpha
} from './column-classification/index';
export {GPUGroupStatistics} from './group-statistics/index';
export type {
  GPUGroupStatistic,
  GPUGroupStatisticsColumn,
  GPUGroupStatisticsColumnOutput,
  GPUGroupStatisticsProps
} from './group-statistics/index';
export {GPUKeyJoin} from './key-join/index';
export type {
  GPUKeyJoinAggregate,
  GPUKeyJoinAggregateOperation,
  GPUKeyJoinGather,
  GPUKeyJoinKind,
  GPUKeyJoinOutput,
  GPUKeyJoinProps
} from './key-join/index';
export {GPUCompositeScore} from './composite-indicators/index';
export type {GPUCompositeScoreOutput, GPUCompositeScoreProps} from './composite-indicators/index';
export {
  getGPUCompositeScoreParameterValues,
  GPU_COMPOSITE_SCORE_AGGREGATION,
  GPU_COMPOSITE_SCORE_COLUMN_STATISTICS_STRIDE,
  GPU_COMPOSITE_SCORE_MAXIMUM_INDICATOR_COUNT,
  GPU_COMPOSITE_SCORE_PARAMETER_LENGTH,
  GPU_COMPOSITE_SCORE_POWER_ITERATIONS,
  GPU_COMPOSITE_SCORE_PRINCIPAL_COMPONENT_SUMMARY,
  GPU_COMPOSITE_SCORE_PRINCIPAL_COMPONENT_SUMMARY_LENGTH,
  GPU_COMPOSITE_SCORE_SCALER
} from './composite-indicators/index';
export type {
  GPUCompositeScoreAggregation,
  GPUCompositeScoreScaler,
  GPUCompositeScoreSettings
} from './composite-indicators/index';
export {GPUInequality} from './composite-indicators/index';
export type {GPUInequalityOutput, GPUInequalityProps} from './composite-indicators/index';
export {
  getGPUInequalityParameterValues,
  GPU_INEQUALITY_DEFAULT_PALMA_BOTTOM_SHARE,
  GPU_INEQUALITY_DEFAULT_PALMA_TOP_SHARE,
  GPU_INEQUALITY_GLOBAL_SUMMARY,
  GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH,
  GPU_INEQUALITY_MAXIMUM_EPSILON,
  GPU_INEQUALITY_PARAMETER_LENGTH
} from './composite-indicators/index';
export type {GPUInequalitySettings} from './composite-indicators/index';
export {
  GPUCalendarBuckets,
  getGPUCalendarBucketsParameterValues,
  GPU_CALENDAR_BUCKETS_INVALID_UINT32,
  GPU_CALENDAR_BUCKETS_INVALID_YEAR,
  GPU_CALENDAR_BUCKETS_MATRIX_LENGTH,
  GPU_CALENDAR_BUCKETS_MAXIMUM_DAYS,
  GPU_CALENDAR_BUCKETS_MAXIMUM_OFFSET_MINUTES,
  GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH
} from './calendar-buckets/index';
export type {GPUCalendarBucketsOutput, GPUCalendarBucketsProps} from './calendar-buckets/index';
export {
  getGPUTimeWindowParameterValues,
  getGPUTimeWindowWordParameterValues,
  getInt64TimeWords,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH,
  GPUTimeWindowFilter,
  joinTimeWords,
  splitTimestamps,
  splitTimeWords
} from './time-window-filter/index';
export type {
  GPUInt64TimeWordRows,
  GPUSplitTimestamps,
  GPUTimeWindow,
  GPUTimeWindowFilterProps,
  GPUTimeWords,
  GPUTimeWordWindow
} from './time-window-filter/index';
export {
  getGPUTemporalReductionParameterValues,
  getGPUTemporalReductionWordParameterValues,
  GPU_TEMPORAL_REDUCTION_NO_CELL,
  GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
  GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH,
  GPUTemporalReduction
} from './temporal-reduction/index';
export type {
  GPUTemporalReductionOutput,
  GPUTemporalReductionProps
} from './temporal-reduction/index';
export {
  GPUVariogram,
  GPUSpatialCorrelogram,
  GPURipley,
  GPUPointPatternIndices,
  getGPUVariogramParameterValues,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPU_VARIOGRAM_STATISTICS_LENGTH,
  evaluateVariogramModel,
  fitVariogramModel,
  getVariogramModelShape,
  getGPUSpatialCorrelogramParameterValues,
  GPU_SPATIAL_CORRELOGRAM_NO_BAND,
  GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
  GPU_SPATIAL_CORRELOGRAM_STATISTICS_LENGTH,
  GPU_RIPLEY_WEIGHT_CAP,
  getGPURipleyParameterValues,
  GPU_RIPLEY_EDGE_CORRECTION,
  GPU_RIPLEY_PARAMETER_LENGTH,
  GPU_NO_NEAREST_NEIGHBOR,
  getGPUPointPatternIndicesParameterValues,
  GPU_CLARK_EVANS_LENGTH,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  GPU_QUADRAT_MAXIMUM_COUNT,
  GPU_QUADRAT_STATISTICS_LENGTH
} from './pair-statistics/index';
export type {
  GPUVariogramProps,
  GPUVariogramParameters,
  VariogramModel,
  VariogramModelInput,
  VariogramModelOptions,
  VariogramModelType,
  VariogramModelWeighting,
  GPUSpatialCorrelogramBandMode,
  GPUSpatialCorrelogramProps,
  GPUSpatialCorrelogramParameters,
  GPUSpatialCorrelogramVarianceAssumption,
  GPURipleyProps,
  GPURipleyEdgeCorrection,
  GPURipleyParameters,
  GPUPointPatternIndicesProps,
  GPUPointPatternIndicesParameters
} from './pair-statistics/index';
