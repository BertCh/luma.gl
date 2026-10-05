// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Analysis contributors
export type {GPUCompactOutput, GPUUint32Rows} from '../utils/gpu-contributor-types';
export {GPUParameterBuffer} from '../utils/gpu-contributor-utils';
export type {GPUParameterBufferProps, GPUParameterFormat} from '../utils/gpu-contributor-utils';
export {
  getGPUTerrainCurvatureCoefficient,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainViewshedParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH,
  GPU_TERRAIN_VISIBILITY,
  GPUTerrainContours,
  GPUTerrainDerivatives,
  GPUTerrainViewshed
} from './terrain-analysis/index';
export type {
  GPUTerrainCellSizeMode,
  GPUTerrainContourLevel,
  GPUTerrainContoursProps,
  GPUTerrainDerivativesProps,
  GPUTerrainDerivativesSettings,
  GPUTerrainSlopeUnits,
  GPUTerrainViewshedProps,
  GPUTerrainViewshedSettings
} from './terrain-analysis/index';
export {
  getGPUReliefShadingParameterValues,
  getGPUSolarPositionParameterValues,
  getGPUSolarShadowMaskParameterValues,
  getGPUTerrainHorizonDirection,
  getGPUTerrainHorizonParameterValues,
  getGPUTerrainHorizonStepDistances,
  getGPUTextureShadingCascadeSigmas,
  getGPUTextureShadingKernel,
  getGPUTextureShadingParameterValues,
  getSolarPosition,
  getSolarRefractionDegrees,
  getSolarTimeParameter,
  GPU_RELIEF_SHADING_MAX_LIGHT_COUNT,
  GPU_RELIEF_SHADING_MAX_STOP_COUNT,
  GPU_RELIEF_SHADING_MDOW_LIGHTS,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH,
  GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES,
  GPU_SOLAR_POSITION_PARAMETER_LENGTH,
  GPU_SOLAR_SHADOW_MASK_PARAMETER_LENGTH,
  GPU_SOLAR_SUNRISE_ALTITUDE_DEGREES,
  GPU_TERRAIN_HORIZON_MAX_DIRECTION_COUNT,
  GPU_TERRAIN_HORIZON_MIN_DIRECTION_COUNT,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPU_TEXTURE_SHADING_MAX_LEVEL_COUNT,
  GPU_TEXTURE_SHADING_PARAMETER_LENGTH,
  GPUReliefShading,
  GPUSolarPosition,
  GPUSolarShadowMask,
  GPUTerrainHorizon,
  GPUTextureShading
} from './terrain-illumination/index';
export type {
  GPUReliefShadingLight,
  GPUReliefShadingProps,
  GPUReliefShadingSettings,
  GPUReliefShadingStop,
  GPUReliefShadingWeighting,
  GPUSolarPositionProps,
  GPUSolarPositionSettings,
  GPUSolarShadowMaskProps,
  GPUSolarShadowMaskSettings,
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
  GPU_TERRAIN_FLOW_CELL_CLASS,
  GPU_TERRAIN_FLOW_NONE,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues
} from './hydrology/index';
export type {
  GPUTerrainFlowAccumulationUnits,
  GPUTerrainFlowProps,
  GPUTerrainFlowSettings
} from './hydrology/index';
export {
  GPU_GEOMORPHON_FORMS,
  GPU_GEOMORPHONS_PARAMETER_LENGTH,
  GPUGeomorphons,
  getGPUGeomorphonsParameterValues
} from './geomorphons/gpu-geomorphons';
export type {
  GPUGeomorphonComparison,
  GPUGeomorphonForm,
  GPUGeomorphonsProps,
  GPUGeomorphonsSettings
} from './geomorphons/gpu-geomorphons';
export {
  getGPUSimpleLocalReliefParameterValues,
  GPU_SIMPLE_LOCAL_RELIEF_PARAMETER_LENGTH,
  GPUSimpleLocalRelief
} from './relief-visualization/gpu-simple-local-relief';
export type {
  GPUSimpleLocalReliefProps,
  GPUSimpleLocalReliefSettings
} from './relief-visualization/gpu-simple-local-relief';
export {
  getGPUMultiScaleReliefParameterValues,
  getGPUMultiScaleReliefRadii,
  GPU_MULTI_SCALE_RELIEF_PARAMETER_LENGTH,
  GPUMultiScaleRelief
} from './relief-visualization/gpu-multi-scale-relief';
export type {
  GPUMultiScaleReliefProps,
  GPUMultiScaleReliefRadii,
  GPUMultiScaleReliefScales,
  GPUMultiScaleReliefSettings
} from './relief-visualization/gpu-multi-scale-relief';
export {
  getGPULocalDominanceParameterValues,
  getGPULocalDominanceShifts,
  GPU_LOCAL_DOMINANCE_MAX_SHIFT_COUNT,
  GPU_LOCAL_DOMINANCE_PARAMETER_LENGTH,
  GPULocalDominance
} from './relief-visualization/gpu-local-dominance';
export type {
  GPULocalDominanceGeometry,
  GPULocalDominanceProps,
  GPULocalDominanceSettings,
  GPULocalDominanceShifts
} from './relief-visualization/gpu-local-dominance';
export {
  getGPUReliefBlendParameterValues,
  GPU_RELIEF_BLEND_LAYER_PARAMETER_LENGTH,
  GPU_RELIEF_BLEND_MAX_LAYER_COUNT,
  GPU_RELIEF_BLEND_MODE_CODES,
  GPU_RELIEF_BLEND_PARAMETER_LENGTH,
  GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL,
  GPU_RELIEF_BLEND_VAT_FLAT,
  GPUReliefBlend
} from './relief-visualization/gpu-relief-blend';
export type {
  GPUReliefBlendLayerSettings,
  GPUReliefBlendMode,
  GPUReliefBlendProps
} from './relief-visualization/gpu-relief-blend';
export {
  GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH,
  GPUTerrainCurvature,
  getGPUTerrainCurvatureParameterValues
} from './terrain-curvature/gpu-terrain-curvature';
export type {
  GPUTerrainCurvatureKind,
  GPUTerrainCurvatureMethod,
  GPUTerrainCurvatureProps,
  GPUTerrainCurvatureSettings
} from './terrain-curvature/gpu-terrain-curvature';
export {
  GPU_TERRAIN_RGB_DEFAULT_VALID_RANGE,
  GPU_TERRAIN_RGB_SEA_CLAMP_FLOOR,
  GPUTerrainRGBDecode
} from './terrain-decode/gpu-terrain-rgb-decode';
export type {
  GPUTerrainRGBDecodeInput,
  GPUTerrainRGBDecodeProps,
  GPUTerrainRGBEncoding
} from './terrain-decode/gpu-terrain-rgb-decode';
export {
  GPU_TERRAIN_SPIKE_REPAIR_DEFAULT_COMPONENT_ITERATIONS,
  GPU_TERRAIN_SPIKE_REPAIR_STATISTICS,
  GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH,
  GPUTerrainSpikeRepair
} from './terrain-decode/gpu-terrain-spike-repair';
export type {GPUTerrainSpikeRepairProps} from './terrain-decode/gpu-terrain-spike-repair';
export {
  GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH,
  GPUTerrainSummits,
  getGPUTerrainSummitsParameterValues
} from './terrain-features/gpu-terrain-summits';
export type {
  GPUTerrainSummitsProps,
  GPUTerrainSummitsSettings
} from './terrain-features/gpu-terrain-summits';
export {
  GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH,
  GPU_TERRAIN_PEAK_SNAP_STATUS,
  GPUTerrainPeakSnap,
  getGPUTerrainPeakSnapParameterValues
} from './terrain-features/gpu-terrain-peak-snap';
export type {
  GPUTerrainPeakSnapProps,
  GPUTerrainPeakSnapSettings
} from './terrain-features/gpu-terrain-peak-snap';
export {
  GPU_TERRAIN_CRITICAL_POINT,
  GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT,
  GPUTerrainCriticalPoints
} from './terrain-features/gpu-terrain-critical-points';
export type {GPUTerrainCriticalPointsProps} from './terrain-features/gpu-terrain-critical-points';
export {
  GPU_PROFILE_PEAKS_DEFAULT_NMS_ROUNDS,
  GPU_PROFILE_PEAKS_MAXIMUM_NMS_ROUNDS,
  GPU_PROFILE_PEAKS_MAXIMUM_RADIUS,
  GPU_PROFILE_PEAKS_PARAMETER_LENGTH,
  GPUProfilePeaks,
  getGPUProfilePeaksParameterValues
} from './terrain-features/gpu-profile-peaks';
export type {
  GPUProfilePeaksProps,
  GPUProfilePeaksSettings
} from './terrain-features/gpu-profile-peaks';
export {
  GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH,
  GPUTerrainFlowField,
  getGPUTerrainFlowFieldParameterValues
} from './terrain-flow-field/gpu-terrain-flow-field';
export type {
  GPUTerrainFlowFieldProps,
  GPUTerrainFlowFieldSettings
} from './terrain-flow-field/gpu-terrain-flow-field';
export {GPUTerrainRuggedness} from './topographic-position/gpu-terrain-ruggedness';
export type {
  GPUTerrainRuggednessAlgorithm,
  GPUTerrainRuggednessEdgeMode,
  GPUTerrainRuggednessProps
} from './topographic-position/gpu-terrain-ruggedness';
export {
  GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH,
  GPUTerrainVectorRuggedness,
  getGPUTerrainVectorRuggednessParameterValues
} from './topographic-position/gpu-terrain-vector-ruggedness';
export type {
  GPUTerrainVectorRuggednessProps,
  GPUTerrainVectorRuggednessSettings
} from './topographic-position/gpu-terrain-vector-ruggedness';
export {
  GPU_TERRAIN_TOPOGRAPHIC_POSITION_DEFAULT_QUANTUM,
  GPU_TERRAIN_TOPOGRAPHIC_POSITION_MAXIMUM_SCALES,
  GPUTerrainTopographicPosition
} from './topographic-position/gpu-terrain-topographic-position';
export type {
  GPUTerrainTopographicPositionProps,
  GPUTerrainTopographicPositionScale
} from './topographic-position/gpu-terrain-topographic-position';
export {
  GPU_TERRAIN_WEISS_LANDFORMS,
  GPU_TERRAIN_WEISS_LANDFORMS_PARAMETER_LENGTH,
  GPUTerrainWeissLandforms,
  getGPUTerrainWeissLandformsParameterValues
} from './topographic-position/gpu-terrain-weiss-landforms';
export type {
  GPUTerrainWeissLandform,
  GPUTerrainWeissLandformsProps,
  GPUTerrainWeissLandformsSettings,
  GPUTerrainWeissStandardization
} from './topographic-position/gpu-terrain-weiss-landforms';
export {
  GPU_POINT_HORIZON_RAYS_PER_DISPATCH,
  GPUPointHorizonProfile
} from './point-horizon/gpu-point-horizon-profile';
export type {GPUPointHorizonProfileProps} from './point-horizon/gpu-point-horizon-profile';
export {
  getGPUPointHorizonDistanceLattice,
  getGPUPointHorizonSegments
} from './point-horizon/point-horizon-march';
export type {
  GPUPointHorizonDistanceLattice,
  GPUPointHorizonDistanceLatticeOptions,
  GPUPointHorizonLatticeOctave,
  GPUPointHorizonModelOptions,
  GPUPointHorizonSegmentOptions,
  GPUPointHorizonSegments
} from './point-horizon/point-horizon-march';
export {
  GPU_POINT_HORIZON_EARTH_RADIUS,
  GPU_POINT_HORIZON_PARAMETER_LENGTH,
  GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH,
  getGPUPointHorizonParameterValues,
  getGPUPointHorizonVisibilityParameterValues
} from './point-horizon/point-horizon-parameters';
export type {
  GPUPointHorizonHeightReference,
  GPUPointHorizonMercatorSettings,
  GPUPointHorizonPlanarSettings,
  GPUPointHorizonProjection,
  GPUPointHorizonRowDirection,
  GPUPointHorizonSettings,
  GPUPointHorizonTraversal,
  GPUPointHorizonVisibilitySettings,
  GPUPointHorizonVisibilityTolerances
} from './point-horizon/point-horizon-parameters';
export {
  GPU_TERRAIN_WATERSHED_NONE,
  GPUTerrainHeightAboveDrainage,
  GPUTerrainStreamOrder,
  GPUTerrainWatersheds
} from './hydrology/index';
export type {
  GPUTerrainHeightAboveDrainageProps,
  GPUTerrainStreamOrderProps,
  GPUTerrainWatershedsProps
} from './hydrology/index';
export {GPUTerrainCumulativeViewshed, GPUTerrainLineOfSight} from './terrain-analysis/index';
export type {
  GPUTerrainCumulativeViewshedProps,
  GPUTerrainLineOfSightProps,
  GPUTerrainSightLineTraversal
} from './terrain-analysis/index';
export {
  GPU_TERRAIN_HYDROLOGIC_INDICES_DEFAULT_MINIMUM_SLOPE,
  GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH,
  GPUTerrainHydrologicIndices,
  getGPUTerrainHydrologicIndicesParameterValues
} from './hydrology/index';
export type {
  GPUTerrainHydrologicIndicesProps,
  GPUTerrainHydrologicIndicesSettings
} from './hydrology/index';
export {
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPUTerrainCastShadow,
  getGPUTerrainCastShadowParameterValues
} from './terrain-illumination/index';
export type {
  GPUTerrainCastShadowProps,
  GPUTerrainCastShadowSettings
} from './terrain-illumination/index';
