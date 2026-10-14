// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Every subdirectory index is re-exported whole, so the entry cannot drift from its parts and a
// name exported by two subdirectories is a compile error.

export * from './change-detection/index';
export * from './cost-distance/index';
export * from './distance-field/index';
export * from './flow-texture/index';
export * from './isolines/index';
export * from './particle-advection/index';
export * from './polygon-rasterization/index';
export * from './raster-algebra/index';
export * from './raster-grid/index';
export * from './raster-patches/index';
export * from './raster-pyramid/index';
export * from './raster-sampling/index';
export * from './raster-stretch/index';
export * from './raster-zonal-statistics/index';

// Top-level files and shared contracts.
export type {
  GPURasterBand,
  GPURasterBufferBand,
  GPURasterCoordinateReferenceSystem,
  GPURasterMetadata,
  GPURasterScalarFormat,
  GPURasterTextureBand,
  GPURasterTextureFormat,
  GPURasterTile
} from './types';
export {GPURaster} from './gpu-raster';
export type {GPURasterProps} from './gpu-raster';
export {GPURasterTileReader} from './gpu-raster-tile-source';
export type {
  GPURasterDecodedBand,
  GPURasterDecodedTile,
  GPURasterPixelBounds,
  GPURasterTileBandMetadata,
  GPURasterTileCoordinateSpace,
  GPURasterTileLevel,
  GPURasterTileRequest,
  GPURasterTileSource,
  GPURasterTileSourceMetadata
} from './gpu-raster-tile-source';
export {
  GPURasterTileCache,
  GPURasterTileGraphLease,
  GPURasterTileLease
} from './gpu-raster-tile-cache';
export type {
  GPURasterResidentBand,
  GPURasterResidentTile,
  GPURasterTileCacheBudgets,
  GPURasterTileCacheProps,
  GPURasterTileCacheStats,
  GPURasterTileGraphEntry,
  GPURasterTileGraphRequest,
  GPURasterTileReleaseFence
} from './gpu-raster-tile-cache';
export {
  GPURasterTileCoreExtract,
  GPURasterTileHaloAssembler,
  GPURasterTileHaloFill,
  GPURasterTileHaloLease
} from './gpu-raster-tile-halo';
export type {
  GPURasterHaloStage,
  GPURasterTileCoreExtractProps,
  GPURasterTileHaloFillProps,
  GPURasterTileHaloPlan,
  GPURasterTileHaloRequest,
  GPURasterTileHaloSource
} from './gpu-raster-tile-halo';
export {
  GPURasterCategoricalOverview,
  GPURasterOverview,
  makeRasterOverviewMetadata
} from './gpu-raster-overview';
export type {
  GPURasterCategoricalOverviewFormat,
  GPURasterCategoricalOverviewProps,
  GPURasterOverviewCategoricalPolicy,
  GPURasterOverviewMetadataOptions,
  GPURasterOverviewProps,
  GPURasterOverviewScale
} from './gpu-raster-overview';
export {
  getGPURasterGlobalMergeMaximumPixelCount,
  GPURasterGlobalHistogramMerge,
  GPURasterGlobalInitialize,
  GPURasterGlobalPercentile,
  GPURasterGlobalStatisticsMerge
} from './gpu-raster-global-statistics';
export type {
  GPURasterGlobalAccumulator,
  GPURasterGlobalHistogramMergeProps,
  GPURasterGlobalInitializeProps,
  GPURasterGlobalPercentileProps,
  GPURasterGlobalStatisticsMergeProps
} from './gpu-raster-global-statistics';
export {GPURasterConnectedComponents} from './gpu-raster-connected-components';
export type {
  GPURasterConnectedComponentsProps,
  GPURasterConnectivity
} from './gpu-raster-connected-components';
export {GPURasterDenseComponents} from './gpu-raster-dense-components';
export type {GPURasterDenseComponentsProps} from './gpu-raster-dense-components';
export {
  getRasterRegionWorldCentroid,
  getGPURasterRegionMeasurementsMaximumCapacity,
  GPURasterRegionMeasurements
} from './gpu-raster-region-measurements';
export type {
  GPURasterRegionMeasurementOutputs,
  GPURasterRegionMeasurementsProps
} from './gpu-raster-region-measurements';
export {
  getGPURasterCrossTileComponentsMaximumCapacity,
  GPURasterCrossTileComponents
} from './gpu-raster-cross-tile-components';
export type {
  GPURasterCrossTile,
  GPURasterCrossTileComponentsProps
} from './gpu-raster-cross-tile-components';
export {GPURasterTextureToBuffer} from './gpu-raster-texture-to-buffer';
export type {GPURasterTextureToBufferProps} from './gpu-raster-texture-to-buffer';
export {GPURasterBufferToTexture} from './gpu-raster-buffer-to-texture';
export type {GPURasterBufferToTextureProps} from './gpu-raster-buffer-to-texture';
export {GPURasterBandMath} from './gpu-raster-band-math';
export type {GPURasterBandMathOperation, GPURasterBandMathProps} from './gpu-raster-band-math';
export {GPURasterNeighborhood} from './gpu-raster-neighborhood';
export type {
  GPURasterBorderMode,
  GPURasterNeighborhoodProps,
  GPURasterNeighborhoodRadius,
  GPURasterNoDataPolicy
} from './gpu-raster-neighborhood';
export {
  GPURasterBoxBlur,
  GPURasterConvolution,
  GPURasterGaussianBlur
} from './gpu-raster-convolution';
export type {
  GPURasterConvolutionProps,
  GPURasterGaussianBlurProps,
  GPURasterSmoothingProps
} from './gpu-raster-convolution';
export {
  GPURasterGradient,
  GPURasterGradientMagnitude,
  GPURasterLaplacian,
  GPURasterScharr,
  GPURasterSobel
} from './gpu-raster-edges';
export type {
  GPURasterEdgeProps,
  GPURasterGradientDirection,
  GPURasterGradientMagnitudeProps,
  GPURasterGradientOperator,
  GPURasterGradientProps,
  GPURasterLaplacianConnectivity,
  GPURasterLaplacianProps,
  GPURasterScharrProps,
  GPURasterSobelProps
} from './gpu-raster-edges';
export {
  GPURasterClosing,
  GPURasterDilation,
  GPURasterErosion,
  GPURasterMorphology,
  GPURasterOpening
} from './gpu-raster-morphology';
export type {
  GPURasterBinaryMorphologyProps,
  GPURasterClosingProps,
  GPURasterDilationProps,
  GPURasterErosionProps,
  GPURasterGrayscaleMorphologyProps,
  GPURasterMorphologyBaseProps,
  GPURasterMorphologyMode,
  GPURasterMorphologyNoDataPolicy,
  GPURasterMorphologyOperation,
  GPURasterMorphologyProps,
  GPURasterOpeningProps,
  GPURasterStructuringElement
} from './gpu-raster-morphology';
export {GPURasterContrast} from './gpu-raster-contrast';
export type {
  GPURasterContrastDomain,
  GPURasterContrastMode,
  GPURasterContrastProps
} from './gpu-raster-contrast';
export {GPURasterContourClassifier, GPURasterContours} from './gpu-raster-contours';
export type {
  GPURasterContourClassifierProps,
  GPURasterContourLevel,
  GPURasterContoursProps
} from './gpu-raster-contours';
export {GPURasterNDVI} from './gpu-raster-ndvi';
export type {GPURasterNDVIProps} from './gpu-raster-ndvi';
export {
  GPURasterStatistics,
  getGPURasterStatisticsMaximumPixelCount
} from './gpu-raster-statistics';
export type {GPURasterStatisticsProps} from './gpu-raster-statistics';
export {GPURasterHistogram} from './gpu-raster-histogram';
export type {GPURasterHistogramDomain, GPURasterHistogramProps} from './gpu-raster-histogram';
export {GPURasterOtsuThreshold, GPURasterThreshold} from './gpu-raster-threshold';
export type {
  GPURasterOtsuDomain,
  GPURasterOtsuThresholdProps,
  GPURasterThresholdOperation,
  GPURasterThresholdProps,
  GPURasterThresholdValue
} from './gpu-raster-threshold';
export {getRasterDeviceLimits, planRasterDispatchStripes} from './raster-device-limits';
export type {
  RasterDeviceLimits,
  RasterDeviceLimitsOptions,
  RasterDispatchStripe,
  RasterDispatchStripeOptions
} from './raster-device-limits';
export type {GPUCompactOutput, GPUUint32Rows} from '../utils/gpu-contributor-types';
export {GPUParameterBuffer} from '../utils/gpu-contributor-utils';
export type {GPUParameterBufferProps, GPUParameterFormat} from '../utils/gpu-contributor-utils';
export type {GPURasterCellSizeMode, GPURasterRowDirection} from '../analysis-core/raster-grid';
