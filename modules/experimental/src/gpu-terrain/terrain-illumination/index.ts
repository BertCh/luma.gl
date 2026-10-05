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
  getGPUTerrainHorizonDirection,
  getGPUTerrainHorizonParameterValues,
  getGPUTerrainHorizonStepDistances,
  GPU_TERRAIN_HORIZON_MAX_DIRECTION_COUNT,
  GPU_TERRAIN_HORIZON_MIN_DIRECTION_COUNT,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPUTerrainHorizon
} from './gpu-terrain-horizon';
export type {
  GPUTerrainHorizonProps,
  GPUTerrainHorizonSettings
} from './gpu-terrain-horizon';
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
export type {GPUTerrainIlluminationCellSizeMode} from './terrain-illumination-utils';
export {
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPUTerrainCastShadow,
  getGPUTerrainCastShadowParameterValues
} from './gpu-terrain-cast-shadow';
export type {
  GPUTerrainCastShadowProps,
  GPUTerrainCastShadowSettings
} from './gpu-terrain-cast-shadow';
