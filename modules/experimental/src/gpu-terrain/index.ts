// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type {GPUCompactOutput, GPUUint32Rows} from '../utils/gpu-contributor-types';
export {GPUParameterBuffer} from '../utils/gpu-contributor-utils';
export type {GPUParameterBufferProps, GPUParameterFormat} from '../utils/gpu-contributor-utils';

export {
  getGPUGeomorphonsParameterValues,
  GPU_GEOMORPHON_FORMS,
  GPU_GEOMORPHONS_PARAMETER_LENGTH,
  GPUGeomorphons
} from './geomorphons/index';
export type {
  GPUGeomorphonComparison,
  GPUGeomorphonForm,
  GPUGeomorphonsProps,
  GPUGeomorphonsSettings
} from './geomorphons/index';

export {
  getGPUTerrainFlowParameterValues,
  getGPUTerrainHydrologicIndicesParameterValues,
  GPU_TERRAIN_DRAINAGE_NONE,
  GPU_TERRAIN_FLOW_CELL_CLASS,
  GPU_TERRAIN_FLOW_NONE,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
  GPU_TERRAIN_HYDROLOGIC_INDICES_DEFAULT_MINIMUM_SLOPE,
  GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH,
  GPU_TERRAIN_WATERSHED_NONE,
  GPUTerrainFlow,
  GPUTerrainHeightAboveDrainage,
  GPUTerrainHydrologicIndices,
  GPUTerrainStreamOrder,
  GPUTerrainWatersheds
} from './hydrology/index';
export type {
  GPUTerrainFlowAccumulationUnits,
  GPUTerrainFlowProps,
  GPUTerrainFlowRouting,
  GPUTerrainFlowSettings,
  GPUTerrainHeightAboveDrainageProps,
  GPUTerrainHydrologicIndicesProps,
  GPUTerrainHydrologicIndicesSettings,
  GPUTerrainStreamOrderProps,
  GPUTerrainWatershedsProps
} from './hydrology/index';

export {
  getGPUPointHorizonDistanceLattice,
  getGPUPointHorizonParameterValues,
  getGPUPointHorizonSegments,
  getGPUPointHorizonVisibilityParameterValues,
  GPU_POINT_HORIZON_EARTH_RADIUS,
  GPU_POINT_HORIZON_PARAMETER_LENGTH,
  GPU_POINT_HORIZON_RAYS_PER_DISPATCH,
  GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH,
  GPUPointHorizonProfile,
  GPUPointHorizonVisibility
} from './point-horizon/index';
export type {
  GPUPointHorizonDistanceLattice,
  GPUPointHorizonDistanceLatticeOptions,
  GPUPointHorizonHeightReference,
  GPUPointHorizonLatticeOctave,
  GPUPointHorizonMercatorSettings,
  GPUPointHorizonModelOptions,
  GPUPointHorizonPlanarSettings,
  GPUPointHorizonProfileProps,
  GPUPointHorizonProjection,
  GPUPointHorizonRowDirection,
  GPUPointHorizonSegmentOptions,
  GPUPointHorizonSegments,
  GPUPointHorizonSettings,
  GPUPointHorizonTraversal,
  GPUPointHorizonVisibilityProps,
  GPUPointHorizonVisibilitySettings,
  GPUPointHorizonVisibilityTolerances
} from './point-horizon/index';

export {
  getGPULocalDominanceParameterValues,
  getGPULocalDominanceShifts,
  getGPUMultiScaleReliefParameterValues,
  getGPUMultiScaleReliefRadii,
  getGPUReliefBlendParameterValues,
  getGPUSimpleLocalReliefParameterValues,
  GPU_LOCAL_DOMINANCE_MAX_SHIFT_COUNT,
  GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH,
  GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH,
  GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH,
  GPU_RELIEF_BLEND_MAX_LAYER_COUNT,
  GPU_RELIEF_BLEND_MODE_CODES,
  GPU_RELIEF_BLEND_PARAMETER_LENGTH,
  GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL,
  GPU_RELIEF_BLEND_VAT_FLAT,
  GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH,
  GPULocalDominance,
  GPUMultiScaleRelief,
  GPUReliefBlend,
  GPUSimpleLocalRelief
} from './relief-visualization/index';
export type {
  GPULocalDominanceGeometry,
  GPULocalDominanceProps,
  GPULocalDominanceSettings,
  GPULocalDominanceShifts,
  GPUMultiScaleReliefProps,
  GPUMultiScaleReliefRadii,
  GPUMultiScaleReliefScales,
  GPUMultiScaleReliefSettings,
  GPUReliefBlendLayerSettings,
  GPUReliefBlendMode,
  GPUReliefBlendProps,
  GPUSimpleLocalReliefProps,
  GPUSimpleLocalReliefSettings
} from './relief-visualization/index';

export {
  getGPUTerrainCurvatureCoefficient,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainSightLineParameterValues,
  getGPUTerrainViewshedParameterValues,
  getGPUTerrainVisibilityToleranceParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH,
  GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH,
  GPU_TERRAIN_VISIBILITY,
  GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH,
  GPUTerrainContours,
  GPUTerrainCumulativeViewshed,
  GPUTerrainDerivatives,
  GPUTerrainLineOfSight,
  GPUTerrainViewshed
} from './terrain-analysis/index';
export type {
  GPUTerrainCellSizeMode,
  GPUTerrainContourLevel,
  GPUTerrainContoursProps,
  GPUTerrainCumulativeViewshedProps,
  GPUTerrainDerivativesProps,
  GPUTerrainDerivativesSettings,
  GPUTerrainLineOfSightProps,
  GPUTerrainSightLineSettings,
  GPUTerrainSightLineTraversal,
  GPUTerrainSlopeUnits,
  GPUTerrainViewshedProps,
  GPUTerrainViewshedSettings,
  GPUTerrainVisibilityToleranceSettings
} from './terrain-analysis/index';

export {
  getGPUTerrainCurvatureParameterValues,
  GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH,
  GPUTerrainCurvature
} from './terrain-curvature/index';
export type {
  GPUTerrainCurvatureKind,
  GPUTerrainCurvatureMethod,
  GPUTerrainCurvatureProps,
  GPUTerrainCurvatureSettings
} from './terrain-curvature/index';

export {
  GPU_TERRAIN_RGB_DEFAULT_VALID_RANGE,
  GPU_TERRAIN_RGB_SEA_CLAMP_FLOOR,
  GPU_TERRAIN_SPIKE_REPAIR_DEFAULT_COMPONENT_ITERATIONS,
  GPU_TERRAIN_SPIKE_REPAIR_STATISTICS,
  GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH,
  GPUTerrainRGBDecode,
  GPUTerrainSpikeRepair
} from './terrain-decode/index';
export type {
  GPUTerrainRGBDecodeInput,
  GPUTerrainRGBDecodeProps,
  GPUTerrainRGBEncoding,
  GPUTerrainSpikeRepairProps
} from './terrain-decode/index';

export {
  getGPUProfilePeaksParameterValues,
  getGPUTerrainPeakSnapParameterValues,
  getGPUTerrainSummitsParameterValues,
  GPU_PROFILE_PEAKS_DEFAULT_NMS_ROUNDS,
  GPU_PROFILE_PEAKS_MAXIMUM_NMS_ROUNDS,
  GPU_PROFILE_PEAKS_MAXIMUM_RADIUS,
  GPU_PROFILE_PEAKS_PARAMETER_LENGTH,
  GPU_TERRAIN_CRITICAL_POINT,
  GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT,
  GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH,
  GPU_TERRAIN_PEAK_SNAP_STATUS,
  GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH,
  GPUProfilePeaks,
  GPUTerrainCriticalPoints,
  GPUTerrainPeakSnap,
  GPUTerrainSummits
} from './terrain-features/index';
export type {
  GPUProfilePeaksProps,
  GPUProfilePeaksSettings,
  GPUTerrainCriticalPointsProps,
  GPUTerrainPeakSnapProps,
  GPUTerrainPeakSnapSettings,
  GPUTerrainSummitsProps,
  GPUTerrainSummitsSettings
} from './terrain-features/index';

export {
  getGPUTerrainFlowFieldParameterValues,
  GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH,
  GPUTerrainFlowField
} from './terrain-flow-field/index';
export type {
  GPUTerrainFlowFieldProps,
  GPUTerrainFlowFieldSettings
} from './terrain-flow-field/index';

export {
  decodeGPUTerrainHorizonUnorm16,
  encodeGPUTerrainHorizonUnorm16,
  getGPUReliefShadingParameterValues,
  getGPUSolarDirectNormalIrradiance,
  getGPUSolarIrradianceParameterValues,
  getGPUSolarIrradianceSunTable,
  getGPUSolarPositionParameterValues,
  getGPUSolarShadowMaskParameterValues,
  getGPUTerrainCastShadowParameterValues,
  getGPUTerrainHorizonDirection,
  getGPUTerrainHorizonParameterValues,
  getGPUTerrainHorizonStepDistances,
  getGPUTextureShadingCascadeSigmas,
  getGPUTextureShadingKernel,
  getGPUTextureShadingParameterValues,
  getSolarPosition,
  getSolarRefractionDegrees,
  getSolarTimeParameter,
  GPU_RELIEF_SHADING_CONTRAST_PIVOT,
  GPU_RELIEF_SHADING_MAX_LIGHT_COUNT,
  GPU_RELIEF_SHADING_MAX_STOP_COUNT,
  GPU_RELIEF_SHADING_MDOW_LIGHTS,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH,
  GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES,
  GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
  GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE,
  GPU_SOLAR_POSITION_PARAMETER_LENGTH,
  GPU_SOLAR_SHADOW_MASK_PARAMETER_LENGTH,
  GPU_SOLAR_SUNRISE_ALTITUDE_DEGREES,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES,
  GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL,
  GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT,
  GPU_TERRAIN_HORIZON_MAX_DIRECTION_COUNT,
  GPU_TERRAIN_HORIZON_MIN_DIRECTION_COUNT,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES,
  GPU_TEXTURE_SHADING_MAX_LEVEL_COUNT,
  GPU_TEXTURE_SHADING_PARAMETER_LENGTH,
  GPUReliefShading,
  GPUSolarIrradiance,
  GPUSolarPosition,
  GPUSolarShadowMask,
  GPUTerrainCastShadow,
  GPUTerrainHorizon,
  GPUTextureShading,
  unpackGPUTerrainHorizonUnorm16
} from './terrain-illumination/index';
export type {
  GPUReliefShadingLight,
  GPUReliefShadingProps,
  GPUReliefShadingSettings,
  GPUReliefShadingStop,
  GPUReliefShadingWeighting,
  GPUSolarIrradianceProps,
  GPUSolarIrradianceSettings,
  GPUSolarIrradianceSunTableOptions,
  GPUSolarPositionProps,
  GPUSolarPositionSettings,
  GPUSolarShadowMaskProps,
  GPUSolarShadowMaskSettings,
  GPUTerrainCastShadowProps,
  GPUTerrainCastShadowSettings,
  GPUTerrainHorizonAlgorithm,
  GPUTerrainHorizonFormat,
  GPUTerrainHorizonProps,
  GPUTerrainHorizonSettings,
  GPUTerrainIlluminationCellSizeMode,
  GPUTextureShadingProps,
  GPUTextureShadingSettings,
  SolarPosition,
  SolarPositionOptions,
  SolarTimeParameter
} from './terrain-illumination/index';

export {
  getGPUTerrainVectorRuggednessParameterValues,
  getGPUTerrainWeissLandformsParameterValues,
  GPU_TERRAIN_TOPOGRAPHIC_POSITION_DEFAULT_QUANTUM,
  GPU_TERRAIN_TOPOGRAPHIC_POSITION_MAXIMUM_SCALES,
  GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH,
  GPU_TERRAIN_WEISS_LANDFORMS,
  GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH,
  GPUTerrainRuggedness,
  GPUTerrainTopographicPosition,
  GPUTerrainVectorRuggedness,
  GPUTerrainWeissLandforms
} from './topographic-position/index';
export type {
  GPUTerrainRuggednessAlgorithm,
  GPUTerrainRuggednessEdgeMode,
  GPUTerrainRuggednessProps,
  GPUTerrainTopographicPositionProps,
  GPUTerrainTopographicPositionScale,
  GPUTerrainVectorRuggednessProps,
  GPUTerrainVectorRuggednessSettings,
  GPUTerrainWeissLandform,
  GPUTerrainWeissLandformsProps,
  GPUTerrainWeissLandformsSettings,
  GPUTerrainWeissStandardization
} from './topographic-position/index';
