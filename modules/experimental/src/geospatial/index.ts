// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuSpatial.

export type {
  GPUDoubleSinglePositions,
  GPUFloat32Positions,
  GPUFloat64Positions,
  GPUGeospatialPositions,
  GPUPreciseScalarRows,
  GPUScalarRows
} from './types';

export {GPUSinusoidalProjection} from './gpu-sinusoidal-projection';
export type {GPUSinusoidalProjectionProps} from './gpu-sinusoidal-projection';
export {GPUHaversineDistance} from './gpu-haversine-distance';
export type {GPUHaversineDistanceProps} from './gpu-haversine-distance';
export {GPUPairwisePointDistance} from './gpu-pairwise-point-distance';
export type {GPUPairwisePointDistanceProps} from './gpu-pairwise-point-distance';
export {GPUPairwisePointSegmentDistance} from './gpu-pairwise-point-segment-distance';
export type {GPUPairwisePointSegmentDistanceProps} from './gpu-pairwise-point-segment-distance';
export {
  GPUPairwisePointInPolygon,
  GPU_POINT_IN_POLYGON_CLASSIFICATION
} from './gpu-pairwise-point-in-polygon';
export type {
  GPUPairwisePointInPolygonProps,
  GPUPointInPolygonClassification
} from './gpu-pairwise-point-in-polygon';
export {GPUPairwisePointLinestringNearest} from './gpu-pairwise-point-linestring-nearest';
export type {
  GPUFloat32PairwisePointLinestringNearestProps,
  GPUFloat64PairwisePointLinestringNearestProps,
  GPUPairwisePointLinestringNearestProps
} from './gpu-pairwise-point-linestring-nearest';
export {GPUGridIndex} from '@luma.gl/gpgpu/gpu-core';
export type {
  GPUGridIndexBounds,
  GPUGridIndexPositions,
  GPUGridIndexProps,
  GPUGridIndexSize,
  GPUGridIndexSourceIds
} from '@luma.gl/gpgpu/gpu-core';
export type {GPUSpatialQueryOutput} from './gpu-spatial-query-types';
export {GPUPointSpatialQuery} from './gpu-point-spatial-query';
export type {
  GPUGridIndexView,
  GPUPointSpatialQueryKind,
  GPUPointSpatialQueryPolygon,
  GPUPointSpatialQueryProps
} from './gpu-point-spatial-query';

export type {GPUCompactOutput, GPUUint32Rows} from '../utils/gpu-contributor-types';
export {GPUParameterBuffer} from '../utils/gpu-contributor-utils';
export type {GPUParameterBufferProps, GPUParameterFormat} from '../utils/gpu-contributor-utils';

export {
  getCellTableFirstRow,
  GPU_CELL_DEFAULT_SUM_SCALE,
  GPU_CELL_EMPTY_KEY_WORD,
  GPU_CELL_MAXIMUM_RESOLUTION,
  GPUCellAggregation,
  GPUCellLevelSelection,
  GPUCellPyramid,
  GPUCellRollup
} from './cell-aggregation/index';
export type {
  GPUCellAggregationProps,
  GPUCellFamily,
  GPUCellLevelSelectionOutput,
  GPUCellLevelSelectionProps,
  GPUCellPyramidLevel,
  GPUCellPyramidProps,
  GPUCellRollupProps,
  GPUCellTable,
  GPUCellWordOrder
} from './cell-aggregation/index';

export {
  GPUCellCover
} from './cell-cover/index';
export type {
  GPUCellCoverContainment,
  GPUCellCoverOutput,
  GPUCellCoverProps
} from './cell-cover/index';

export {
  GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT,
  GPU_CELL_GEOMETRY_VERTEX_COUNTS,
  GPU_CELL_INDEX_FAMILIES,
  GPU_CELL_INDEX_RESOLUTION_RANGES,
  GPUCellGeometry,
  GPUPointToCell,
  isGPUCellIndexFamily,
  validateCellIndexResolution
} from './cell-indexing/index';
export type {
  GPUCellGeometryFamily,
  GPUCellGeometryOutput,
  GPUCellGeometryProps,
  GPUCellIndexFamily,
  GPUCellIndexResolutionRange,
  GPUPointToCellOutput,
  GPUPointToCellProps
} from './cell-indexing/index';

export {
  GPU_CELL_COMPARE_PRESENT_AFTER,
  GPU_CELL_COMPARE_PRESENT_BEFORE,
  GPUCellTableCompare
} from './cell-table-compare/index';
export type {
  GPUCellTableCompareMeasure,
  GPUCellTableCompareOutput,
  GPUCellTableCompareProps,
  GPUCellTableCompareZScore
} from './cell-table-compare/index';

export {
  getCellTopologyStride,
  GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE,
  GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS,
  GPU_CELL_UNCOMPACT_DEFAULT_MAXIMUM_DEPTH,
  GPUCellCompaction,
  GPUCellTopology
} from './cell-topology/index';
export type {
  GPUCellCompactionOutput,
  GPUCellCompactionProps,
  GPUCellCompactionWordOrder,
  GPUCellCompactOperation,
  GPUCellTopologyOperation,
  GPUCellTopologyOutput,
  GPUCellTopologyProps,
  GPUCellUncompactOperation
} from './cell-topology/index';

export {
  getGPUDotDensityParameterValues,
  GPU_DOT_DENSITY_PARAMETER_LENGTH,
  GPUDotDensity,
  GPURandomPointsInPolygon
} from './dot-density/index';
export type {
  GPUDotDensityMask,
  GPUDotDensityOutput,
  GPUDotDensityPolygons,
  GPUDotDensityProps,
  GPUDotDensitySettings,
  GPURandomPointsInPolygonProps
} from './dot-density/index';

export {
  getGPUEmergingHotSpotParameterValues,
  GPU_EMERGING_HOT_SPOT_CATEGORIES,
  GPU_EMERGING_HOT_SPOT_CRITICAL_Z_SCORES,
  GPU_EMERGING_HOT_SPOT_MAXIMUM_RADIUS,
  GPU_EMERGING_HOT_SPOT_MAXIMUM_SLICE_COUNT,
  GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH,
  GPU_EMERGING_HOT_SPOT_STATISTICS_LENGTH,
  GPUEmergingHotSpots
} from './emerging-hot-spots/index';
export type {
  GPUEmergingHotSpotParameters,
  GPUEmergingHotSpotsProps
} from './emerging-hot-spots/index';

export {
  getGPUGeographicDistributionParameterValues,
  GPU_GEOGRAPHIC_DISTRIBUTION_DEFAULT_MEDIAN_TOLERANCE,
  GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  GPUGeographicDistribution
} from './geographic-distribution/index';
export type {
  GPUGeographicDistributionEllipseConvention,
  GPUGeographicDistributionOutput,
  GPUGeographicDistributionParameters,
  GPUGeographicDistributionProps
} from './geographic-distribution/index';

export {
  GPU_GEODESIC_DEFAULT_ITERATIONS,
  GPU_GEODESIC_MEAN_EARTH_RADIUS,
  GPU_GEODESIC_WGS84_FLATTENING,
  GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS,
  GPUGeodesicDestination,
  GPUGeodesicPairs,
  GPUGeometryMeasures
} from './geometry-measures/index';
export type {
  GPUGeodesicDestinationOutput,
  GPUGeodesicDestinationProps,
  GPUGeodesicModel,
  GPUGeodesicPairsOutput,
  GPUGeodesicPairsProps,
  GPUGeometryCoordinateSystem,
  GPUGeometryHoleRule,
  GPUGeometryMeasuresGroupOutput,
  GPUGeometryMeasuresOutput,
  GPUGeometryMeasuresProps
} from './geometry-measures/index';

export {
  GPU_GLOBAL_JOIN_COUNT_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY,
  GPUGlobalSpatialStatistics
} from './global-spatial-statistics/index';
export type {
  GPUGlobalSpatialStatistic,
  GPUGlobalSpatialStatisticsProps
} from './global-spatial-statistics/index';

export {
  getGPUGreatCircleArcsParameterValues,
  getGPULineChunkParameterValues,
  getGPULineSegmentizeParameterValues,
  getGPULineSmoothParameterValues,
  GPU_GREAT_CIRCLE_ARCS_DEFAULT_MAXIMUM_SEGMENTS,
  GPU_LINE_CHUNK_PARAMETER_LENGTH,
  GPU_LINE_NO_SOURCE,
  GPU_LINE_SEGMENTIZE_DEFAULT_MAXIMUM_PIECES,
  GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH,
  GPU_LINE_SMOOTH_MAXIMUM_ITERATIONS,
  GPU_LINE_SMOOTH_PARAMETER_LENGTH,
  GPUGreatCircleArcs,
  GPULineChunk,
  GPULineSegmentize,
  GPULineSmooth
} from './line-segmentize/index';
export type {
  GPUGreatCircleArcsParameters,
  GPUGreatCircleArcsProps,
  GPULineChunkParameters,
  GPULineChunkProps,
  GPULineCoordinateSystem,
  GPULinePathOutput,
  GPULineSegmentizeParameters,
  GPULineSegmentizeProps,
  GPULineSmoothParameters,
  GPULineSmoothProps
} from './line-segmentize/index';

export {
  getGPULineSimplificationParameterValues,
  GPU_LINE_SIMPLIFICATION_DEFAULT_MAXIMUM_ROUNDS,
  GPU_LINE_SIMPLIFICATION_MAXIMUM_ROUNDS,
  GPU_LINE_SIMPLIFICATION_PARAMETER_LENGTH,
  GPULineSimplification
} from './line-simplification/index';
export type {
  GPULineSimplificationMetric,
  GPULineSimplificationParameters,
  GPULineSimplificationProps,
  GPULineSimplificationSelection,
  GPULineSimplificationStatus
} from './line-simplification/index';

export {
  getGPULineLocateParameterValues,
  GPU_LINE_LOCATE_PARAMETER_LENGTH,
  GPU_LINE_LOCATE_STATUS,
  GPULinearReferencing,
  GPULineLocate
} from './linear-referencing/index';
export type {
  GPULinearReferencingOutput,
  GPULinearReferencingProps,
  GPULineLocateOutput,
  GPULineLocateParameters,
  GPULineLocateProps
} from './linear-referencing/index';

export {
  getGPUNeighborSearchParameterValues,
  GPU_NEIGHBOR_SEARCH_KERNEL,
  GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_NEIGHBOR_SEARCH_WEIGHT_KIND,
  GPUNeighborSearch,
  validateGPUSpatialWeights
} from './neighbor-search/index';
export type {
  GPUNeighborSearchKernel,
  GPUNeighborSearchParameters,
  GPUNeighborSearchProps,
  GPUNeighborSearchWeightKind,
  GPUSpatialWeights
} from './neighbor-search/index';

export {
  getGPUPermutationParameterValues,
  GPU_GLOBAL_PERMUTATION_RESULT,
  GPU_LOCAL_PERMUTATION_MAXIMUM_NEIGHBORS,
  GPU_LOCAL_PERMUTATION_NOT_TESTED,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPUGlobalPermutationTest,
  GPULocalPermutationTest
} from './permutation-inference/index';
export type {
  GPUGlobalPermutationStatistic,
  GPUGlobalPermutationTestProps,
  GPULocalPermutationStatistic,
  GPULocalPermutationTestProps,
  GPUPermutationParameters
} from './permutation-inference/index';

export {
  createGPUPointDensityGaussianKernel,
  getGPUPointDensityHexagonCell,
  getGPUPointDensityHexagonCenter,
  getGPUPointDensityHexagonGridSize,
  GPU_POINT_DENSITY_HEXAGON_WGSL,
  GPUPointDensity
} from './point-density/index';
export type {
  GPUPointDensityBinning,
  GPUPointDensityBounds,
  GPUPointDensityOutput,
  GPUPointDensityProps,
  GPUPointDensitySmoothing,
  GPUPointDensityStatistic
} from './point-density/index';

export {
  decodeGPURegionStatistics,
  getGPURegionStatisticsSummaryLength,
  GPU_REGION_SCREEN_TRANSFORM_LENGTH,
  GPU_REGION_STATISTICS_FLAGS,
  GPU_REGION_STATISTICS_HEADER_LENGTH,
  GPU_REGION_STATISTICS_SUMMARY_LAYOUT,
  GPUPickRegionMask,
  GPURegionMask,
  GPURegionStatistics,
  GPURegionStatisticsReadback
} from './region-statistics/index';
export type {
  GPUPickRegionMaskProps,
  GPURegionHistogramProps,
  GPURegionMaskProps,
  GPURegionMaskSelection,
  GPURegionPickSelection,
  GPURegionPolygon,
  GPURegionRadius,
  GPURegionRectangle,
  GPURegionSelection,
  GPURegionShape,
  GPURegionStatisticsGridIndex,
  GPURegionStatisticsProps,
  GPURegionStatisticsReadbackProps,
  GPURegionStatisticsResult
} from './region-statistics/index';

export {
  getGPUSpatialAutocorrelationParameterValues,
  GPU_HOT_SPOT_CRITICAL_Z_SCORES,
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS,
  GPU_LOCAL_MORAN_QUADRANT,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH,
  GPUHotSpotAnalysis,
  GPULocalMoran
} from './spatial-autocorrelation/index';
export type {
  GPUHotSpotAnalysisProps,
  GPULocalMoranProps,
  GPUSpatialAutocorrelationFixedMoments,
  GPUSpatialAutocorrelationParameters
} from './spatial-autocorrelation/index';

export {
  getGPUSpatialClusteringParameterValues,
  GPU_SPATIAL_CLUSTERING_NOISE,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  GPUSpatialClustering
} from './spatial-clustering/index';
export type {
  GPUSpatialClusteringParameters,
  GPUSpatialClusteringProps
} from './spatial-clustering/index';

export {
  getGPUFocalStatisticsParameterValues,
  getGPUInverseDistanceWeightingParameterValues,
  GPU_FOCAL_STATISTICS_PARAMETER_LENGTH,
  GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH,
  GPUFocalStatistics,
  GPUInverseDistanceWeighting
} from './spatial-interpolation/index';
export type {
  GPUFocalStatisticsOutput,
  GPUFocalStatisticsProps,
  GPUFocalStatisticsSettings,
  GPUFocalStatisticsShape,
  GPUInverseDistanceWeightingOutput,
  GPUInverseDistanceWeightingProps,
  GPUInverseDistanceWeightingSettings
} from './spatial-interpolation/index';

export {
  GPU_SPATIAL_JOIN_NO_DISTANCE,
  GPU_SPATIAL_JOIN_NO_FEATURE,
  GPUBufferSelection,
  GPUNearestFeatureJoin,
  GPUPointInPolygonJoin
} from './spatial-join/index';
export type {
  GPUBufferSelectionProps,
  GPUNearestFeatureJoinProps,
  GPUNearestFeaturePoints,
  GPUNearestFeatureSegments,
  GPUNearestFeatureSource,
  GPUPointInPolygonJoinProps
} from './spatial-join/index';

export {
  getChiSquareSurvival,
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues,
  getGPUOrdinaryLeastSquaresParameterValues,
  getLogGammaOfHalfInteger,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_BANDWIDTH_MODE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_KERNEL,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE,
  GPU_ORDINARY_LEAST_SQUARES_STATUS_OK,
  GPU_ORDINARY_LEAST_SQUARES_STATUS_SINGULAR,
  GPU_ORDINARY_LEAST_SQUARES_STATUS_TOO_FEW_ROWS,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ADJUSTED_R_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_AIC,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BIC,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_KURTOSIS,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LOG_LIKELIHOOD,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RIDGE_LAMBDA,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ROW_COUNT,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SKEWNESS,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_TOTAL_SUM_OF_SQUARES,
  GPUGeographicallyWeightedRegression,
  GPUOrdinaryLeastSquares,
  SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT
} from './spatial-regression/index';
export type {
  GPUGeographicallyWeightedRegressionBandwidthMode,
  GPUGeographicallyWeightedRegressionKernel,
  GPUGeographicallyWeightedRegressionOutput,
  GPUGeographicallyWeightedRegressionProps,
  GPUGeographicallyWeightedRegressionSettings,
  GPUOrdinaryLeastSquaresOutput,
  GPUOrdinaryLeastSquaresProps
} from './spatial-regression/index';

export {
  getGPUTrajectoryMetricsParameterValues,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
  GPUTrajectoryMetrics
} from './trajectory-analysis/index';
export type {
  GPUTrajectoryMetricsProps,
  GPUTrajectoryStopOutput,
  GPUTrajectoryStopParameters
} from './trajectory-analysis/index';

export {
  getGPUTrajectoryPlayheadParameterValues,
  getGPUTrajectoryPlayheadWordParameterValues,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_STATUS,
  GPUTrajectoryPlayhead,
  GPUTrajectoryResample
} from './trajectory-interpolation/index';
export type {
  GPUTrajectoryPlayheadProps,
  GPUTrajectoryPlayheadStatus,
  GPUTrajectoryPlayheadTime,
  GPUTrajectoryResampleProps,
  GPUTrajectoryResampleSpacing
} from './trajectory-interpolation/index';

export {
  GPUZonalStatistics
} from './zonal-statistics/index';
export type {
  GPUZonalStatisticsExtentStatistic,
  GPUZonalStatisticsFeatureRows,
  GPUZonalStatisticsFeatures,
  GPUZonalStatisticsOutput,
  GPUZonalStatisticsPolygons,
  GPUZonalStatisticsProps,
  GPUZonalStatisticsSumOrder
} from './zonal-statistics/index';
