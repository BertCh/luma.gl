// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  getSolarPosition,
  getSolarRefractionDegrees,
  getSolarTimeParameter
} from './solar-position';
export type {
  SolarPosition,
  SolarPositionOptions,
  SolarTimeParameter
} from './solar-position';
export {
  getGPUSolarPositionParameterValues,
  GPU_SOLAR_POSITION_PARAMETER_LENGTH,
  GPU_SOLAR_SUNRISE_ALTITUDE_DEGREES,
  GPUSolarPosition
} from './gpu-solar-position';
export type {
  GPUSolarPositionProps,
  GPUSolarPositionSettings
} from './gpu-solar-position';
export {
  decodeGPUTerrainHorizonUnorm16,
  encodeGPUTerrainHorizonUnorm16,
  getGPUTerrainHorizonDirection,
  getGPUTerrainHorizonParameterValues,
  getGPUTerrainHorizonStepDistances,
  GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES,
  GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL,
  GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT,
  GPU_TERRAIN_HORIZON_MAX_DIRECTION_COUNT,
  GPU_TERRAIN_HORIZON_MIN_DIRECTION_COUNT,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES,
  GPUTerrainHorizon,
  unpackGPUTerrainHorizonUnorm16
} from './gpu-terrain-horizon';
export type {
  GPUTerrainHorizonAlgorithm,
  GPUTerrainHorizonFormat,
  GPUTerrainHorizonProps,
  GPUTerrainHorizonSettings
} from './gpu-terrain-horizon';
export {
  getGPUTerrainCastShadowParameterValues,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPUTerrainCastShadow
} from './gpu-terrain-cast-shadow';
export type {
  GPUTerrainCastShadowProps,
  GPUTerrainCastShadowSettings
} from './gpu-terrain-cast-shadow';
export {
  getGPUSolarDirectNormalIrradiance,
  getGPUSolarIrradianceParameterValues,
  getGPUSolarIrradianceSunTable,
  GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
  GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE,
  GPUSolarIrradiance
} from './gpu-solar-irradiance';
export type {
  GPUSolarIrradianceProps,
  GPUSolarIrradianceSettings,
  GPUSolarIrradianceSunTableOptions
} from './gpu-solar-irradiance';
export {
  getGPUSolarShadowMaskParameterValues,
  GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES,
  GPU_SOLAR_SHADOW_MASK_PARAMETER_LENGTH,
  GPUSolarShadowMask
} from './gpu-solar-shadow-mask';
export type {
  GPUSolarShadowMaskProps,
  GPUSolarShadowMaskSettings
} from './gpu-solar-shadow-mask';
export {
  getGPUReliefShadingParameterValues,
  GPU_RELIEF_SHADING_CONTRAST_PIVOT,
  GPU_RELIEF_SHADING_MAX_LIGHT_COUNT,
  GPU_RELIEF_SHADING_MAX_STOP_COUNT,
  GPU_RELIEF_SHADING_MDOW_LIGHTS,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH,
  GPUReliefShading
} from './gpu-relief-shading';
export type {
  GPUReliefShadingLight,
  GPUReliefShadingProps,
  GPUReliefShadingSettings,
  GPUReliefShadingStop,
  GPUReliefShadingWeighting
} from './gpu-relief-shading';
export {
  getGPUTextureShadingCascadeSigmas,
  getGPUTextureShadingKernel,
  getGPUTextureShadingParameterValues,
  GPU_TEXTURE_SHADING_MAX_LEVEL_COUNT,
  GPU_TEXTURE_SHADING_PARAMETER_LENGTH,
  GPUTextureShading
} from './gpu-texture-shading';
export type {
  GPUTextureShadingProps,
  GPUTextureShadingSettings
} from './gpu-texture-shading';
