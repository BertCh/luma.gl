// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUSplatData,
  makeGPUSplatData,
  type SplatDataUpdate,
  type GPUSplatTypeMap,
  type GPUSplatVectors,
  type SplatSource
} from './splat-data';
export {
  isGLTFSplatPrimitive,
  loadGPUSplatDataFromGLTF,
  makeGPUSplatDataFromGLTF,
  makeSplatSourceFromGLTF,
  type GLTFSplatAttribute,
  type GLTFSplatAttributeValues,
  type GLTFSplatCompressionDecoder,
  type GLTFSplatPrimitive,
  type LoadGPUSplatDataFromGLTFOptions,
  type MakeGPUSplatDataFromGLTFOptions
} from './splat-gltf';
export {
  evaluateSplatSphericalHarmonics,
  getSplatSphericalHarmonicCoefficientCount,
  getSplatSphericalHarmonicsDegree,
  type SplatSphericalHarmonicsDegree
} from './splat-spherical-harmonics';
export {
  acceptsSplatSemantic,
  type SplatSemanticFilter,
  type SplatSemanticSelection
} from './splat-filter';
export {
  DEFAULT_SPLAT_SCREEN_FILTER_VARIANCE,
  getSplatClampCompensation,
  getSplatDilationCompensation,
  getSplatLogisticCdf,
  getSplatPixelIntegral,
  getSplatScreenFilterVariance,
  getSplatSupportRadius,
  type SplatAntialiasingMode,
  type SplatFragmentKernel
} from './splat-antialiasing';
export {
  getSplatDepthKeyBits,
  getSplatInvalidDepthKey,
  getSplatMaximumDepthKey,
  packSplatFloat16Bits,
  SPLAT_MAXIMUM_DEPTH_KEY_BITS,
  type SplatDepthKeyMode
} from './splat-depth-key';
export {
  getSplatClipCoverage,
  isSplatClipRegionActive,
  MAXIMUM_SPLAT_CLIP_PLANES,
  packSplatClipUniforms,
  SPLAT_CLIP_UNIFORM_BYTE_LENGTH,
  type SplatClipCombineMode,
  type SplatClipPlane,
  type SplatClipRegion
} from './splat-clipping';
export {
  getSplatLevelFadeOpacity,
  planSplatBudget,
  type SplatBudgetNode,
  type SplatBudgetPlan,
  type SplatBudgetPlanProps,
  type SplatBudgetRefinement,
  type SplatBudgetView
} from './splat-budget';
export {
  SplatPicker,
  resolveSplatPickInfo,
  SPLAT_COLOR_PICKING_FS_GLSL,
  SPLAT_PICKING_ATTRIBUTE_WGSL_SHADER,
  SPLAT_PICKING_FS_GLSL,
  SPLAT_PICKING_STORAGE_WGSL_SHADER,
  type SplatPickingInfo,
  type SplatPickingProps
} from './splat-picking';
export {
  SplatResidencyManager,
  type SplatResidencyBounds,
  type SplatResidencyBudget,
  type SplatResidencyCallbacks,
  type SplatResidencyChunk,
  type SplatResidencyChunkOptions,
  type SplatResidencyEvictionReason,
  type SplatResidencyManagerProps,
  type SplatResidencyStats
} from './splat-residency';
export {
  SplatHierarchyManager,
  getSplatHierarchyCoverage,
  getSplatHierarchyFoveatedPriority,
  getSplatHierarchyRefinementError,
  getSplatHierarchyScreenSpaceError,
  isSplatHierarchyNodeVisible,
  type SplatHierarchyFoveation,
  type SplatHierarchyFrontierEntry,
  type SplatHierarchyLoadContext,
  type SplatHierarchyManagerProps,
  type SplatHierarchyNode,
  type SplatHierarchyPageLoader,
  type SplatHierarchyRefinement,
  type SplatHierarchyStats,
  type SplatHierarchyView
} from './splat-hierarchy';
export {
  SplatRADHierarchyManager,
  getSplatRADPageBounds,
  type SplatRADHierarchyFrontierEntry,
  type SplatRADHierarchyManagerProps,
  type SplatRADHierarchyPage,
  type SplatRADHierarchyRequest,
  type SplatRADHierarchyStats
} from './splat-rad-hierarchy';
export {
  getCovarianceEllipseAxes,
  getQuaternionScaledAxes,
  projectSplatCovarianceToScreen,
  projectWorldPositionToScreen,
  transformSplatPosition,
  type ProjectedSplatCovariance,
  type SplatCovarianceProjectionProps
} from './splat-covariance';
export {
  getSortedSplatIndicesByDepth,
  packSplatDepthKey,
  sortSplatReferences,
  SPLAT_DEPTH_KEY_BITS,
  SPLAT_TILE_SIZE_PIXELS,
  type SplatSortMode,
  type SplatSortReference
} from './splat-sort';
export {
  SplatRenderer,
  SPLAT_STORAGE_GPU_INPUT_SCHEMA,
  type SplatDrawRun,
  type SplatMeshRenderable,
  type SplatMixedRenderOptions,
  type SplatRendererProps,
  type SplatRendererStats
} from './splat-renderer';
export {
  getGPUSplatStage,
  getGPUSplatStageTimings,
  type GPUSplatStage,
  type GPUSplatStageTiming,
  type GPUSplatStageTimings
} from './splat-stage-timings';
export {
  GPUSplatGraphRenderer,
  type GPUSplatAlphaMode,
  type GPUSplatGraphRendererProps,
  type GPUSplatRenderPath,
  type SplatBatchRenderParams
} from './gpu-splat-graph-renderer';
export {
  GPUPagedSplatRenderer,
  type GPUPagedSplatPage,
  type GPUPagedSplatRendererProps,
  type GPUPagedSplatRendererStats
} from './gpu-paged-splat-renderer';
export {
  GPUSplatGraphMixedRenderer,
  GPUSplatGraphPicker,
  resolveGPUSplatGraphPickInfo,
  GPU_SPLAT_GRAPH_COMPATIBLE_PICKING_SHADER,
  GPU_SPLAT_GRAPH_PICKING_SHADER,
  type GPUSplatGraphMixedRendererProps
} from './gpu-splat-graph-interaction';
export {
  GPU_SPLAT_COMPATIBLE_RENDER_SHADER,
  GPU_SPLAT_COMPATIBLE_RENDER_SHADER_LAYOUT,
  GPU_SPLAT_FEATURE_FLAGS,
  GPU_SPLAT_FEATURE_SHADER,
  GPU_SPLAT_FEATURE_SHADER_LAYOUT,
  GPU_SPLAT_FRAGMENT_SHARED_SHADER_WGSL,
  GPU_SPLAT_GATHER_SHADER,
  GPU_SPLAT_GATHER_SHADER_LAYOUT,
  GPU_SPLAT_GRAPH_SHARED_WGSL,
  GPU_SPLAT_GRAPH_UNIFORM_BYTE_LENGTH,
  GPU_SPLAT_QUAD_EXPANSION_SHADER_WGSL,
  GPU_SPLAT_MAXIMUM_CLIP_PLANES,
  GPU_SPLAT_PROJECTED_RECORD_BYTE_LENGTH,
  GPU_SPLAT_PROJECTION_SHADER,
  GPU_SPLAT_PROJECTION_SHADER_LAYOUT,
  GPU_SPLAT_RENDER_SHADER,
  GPU_SPLAT_RENDER_SHADER_LAYOUT
} from './gpu-splat-graph-shaders';
export {
  SPLAT_ATTRIBUTE_SHADER_LAYOUT,
  SPLAT_ATTRIBUTE_WGSL_SHADER,
  SPLAT_FS_GLSL,
  SPLAT_STORAGE_SHADER_LAYOUT,
  SPLAT_STORAGE_WGSL_SHADER,
  SPLAT_VS_GLSL,
  splatUniforms,
  type SplatUniforms
} from './splat-shaders';
