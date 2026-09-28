// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ShaderLayout} from '@luma.gl/core';
import {SPLAT_ANTIALIASING_WGSL} from './splat-antialiasing';
import {SPLAT_DEPTH_KEY_WGSL} from './splat-depth-key';
import {MAXIMUM_SPLAT_CLIP_PLANES, SPLAT_CLIPPING_WGSL} from './splat-clipping';

/**
 * Byte stride of one camera-dependent Gaussian projection owned by its renderer.
 *
 * The clip-space center stays single precision because it positions the quad; the screen-space
 * axes and the HDR color are half precision, which is indistinguishable at display precision and
 * takes the record from 48 to 32 bytes. At ten million resident Gaussians that difference is
 * 160 MB of VRAM.
 */
export const GPU_SPLAT_PROJECTED_RECORD_BYTE_LENGTH = 32;

/** Padded byte size of the camera, styling, and preserved-batch projection uniforms. */
export const GPU_SPLAT_GRAPH_UNIFORM_BYTE_LENGTH = 160;

/** Padded byte size of view-dependent harmonics and source-semantic selection controls. */
export const GPU_SPLAT_GRAPH_FEATURE_UNIFORM_BYTE_LENGTH = 48;

/** Bit flags packed into `GraphSplatUniforms.featureFlags`. */
export const GPU_SPLAT_FEATURE_FLAGS = {
  /** Source colors are unclamped linear Float32 radiance rather than normalized Uint8. */
  floatColor: 1 << 0,
  /** Compensate opacity for the screen-space dilation filter (Mip-Splatting). */
  compensateDilation: 1 << 1,
  /** Compensate opacity when the projected size clamp rescales the covariance. */
  compensateClamp: 1 << 2,
  /** Derive each Gaussian's support radius from its own opacity instead of a fixed 3 sigma. */
  dynamicSupportRadius: 1 << 3,
  /** Integrate the Gaussian over the pixel footprint instead of sampling its center. */
  analyticFragmentKernel: 1 << 4,
  /** Resolve coverage by dithered discard and opaque blending instead of alpha blending. */
  stochasticAlpha: 1 << 5,
  /** Evaluate the bound clip region during projection. */
  clipping: 1 << 6,
  /**
   * Spark-compatible radiance-field opacity for paged hierarchies.
   *
   * Radiance-field parent pages encode opacity above one and widen their support instead of
   * raising their peak, so a coarse page covers the area its children will refine. Adapted from
   * Spark (MIT, Copyright (c) 2025 WORLD LABS TECHNOLOGIES, INC.).
   */
  radialOpacity: 1 << 7,
  /** Decode source radiance from sRGB to linear after spherical-harmonic evaluation. */
  srgbDecode: 1 << 8
} as const;

/** Projection uses exactly eight storage bindings, the guaranteed WebGPU minimum. */
export const GPU_SPLAT_PROJECTION_SHADER_LAYOUT = {
  attributes: [],
  bindings: [
    {name: 'positions', type: 'read-only-storage', group: 0, location: 0},
    {name: 'scales', type: 'read-only-storage', group: 0, location: 1},
    {name: 'rotations', type: 'read-only-storage', group: 0, location: 2},
    {name: 'colors', type: 'read-only-storage', group: 0, location: 3},
    {name: 'opacities', type: 'read-only-storage', group: 0, location: 4},
    {name: 'projectedRecords', type: 'storage', group: 0, location: 5},
    {name: 'depthKeys', type: 'storage', group: 0, location: 6},
    {name: 'drawCommands', type: 'storage', group: 0, location: 7},
    {name: 'graphUniforms', type: 'uniform', group: 0, location: 8},
    {name: 'clipUniforms', type: 'uniform', group: 0, location: 9}
  ]
} satisfies ShaderLayout;

/** Optional directional-radiance and semantic filtering stay below the eight-buffer minimum. */
export const GPU_SPLAT_FEATURE_SHADER_LAYOUT = {
  attributes: [],
  bindings: [
    {name: 'positions', type: 'read-only-storage', group: 0, location: 0},
    {name: 'sphericalHarmonics', type: 'read-only-storage', group: 0, location: 1},
    {name: 'semanticIds', type: 'read-only-storage', group: 0, location: 2},
    {name: 'semanticSelections', type: 'read-only-storage', group: 0, location: 3},
    {name: 'projectedRecords', type: 'storage', group: 0, location: 4},
    {name: 'depthKeys', type: 'storage', group: 0, location: 5},
    {name: 'drawCommands', type: 'storage', group: 0, location: 6},
    {name: 'graphUniforms', type: 'uniform', group: 0, location: 7},
    {name: 'featureUniforms', type: 'uniform', group: 0, location: 8}
  ]
} satisfies ShaderLayout;

/** Globally sorted projected records render in one draw without binding source batches. */
export const GPU_SPLAT_RENDER_SHADER_LAYOUT = {
  attributes: [],
  bindings: [
    {name: 'graphUniforms', type: 'uniform', group: 0, location: 0},
    {name: 'projectedRecords', type: 'read-only-storage', group: 0, location: 1},
    {name: 'sortedIds', type: 'read-only-storage', group: 0, location: 2}
  ]
} satisfies ShaderLayout;

/**
 * Compatibility render layout: one uniform buffer and no storage bindings in the vertex stage.
 *
 * WebGPU compatibility mode reports `maxStorageBuffersInVertexStage: 0`, which forbids the
 * universal web splat pattern of indexing a sorted-id buffer and a projected-record buffer from
 * the vertex shader. Gathering the sorted records into a vertex stream ahead of the draw removes
 * the indirection entirely, at the cost of one extra compute pass and one extra record buffer.
 */
export const GPU_SPLAT_COMPATIBLE_RENDER_SHADER_LAYOUT = {
  attributes: [
    {name: 'instanceClipCenter', location: 0, type: 'vec4<f32>', stepMode: 'instance'},
    {name: 'instancePackedRecord', location: 1, type: 'vec4<u32>', stepMode: 'instance'}
  ],
  bindings: [{name: 'graphUniforms', type: 'uniform', group: 0, location: 0}]
} satisfies ShaderLayout;

/** Gathers globally sorted projected records into a sequential compatibility vertex stream. */
export const GPU_SPLAT_GATHER_SHADER_LAYOUT = {
  attributes: [],
  bindings: [
    {name: 'projectedRecords', type: 'read-only-storage', group: 0, location: 0},
    {name: 'sortedIds', type: 'read-only-storage', group: 0, location: 1},
    {name: 'sortedRecords', type: 'storage', group: 0, location: 2}
  ]
} satisfies ShaderLayout;

const GPU_SPLAT_GRAPH_UNIFORM_STRUCT = /* wgsl */ `\
struct GraphSplatUniforms {
  modelViewProjectionMatrix: mat4x4<f32>,
  viewportSize: vec2<f32>,
  depthRange: vec2<f32>,
  radiusScale: f32,
  alphaScale: f32,
  alphaCutoff: f32,
  screenSizeCutoffPixels: f32,
  gaussianSupportRadius: f32,
  screenFilterVariance: f32,
  levelFilterVariance: f32,
  levelFadeOpacity: f32,
  maxScreenSpaceSplatSize: f32,
  exposure: f32,
  toneMapping: u32,
  featureFlags: u32,
  depthKeyMode: u32,
  maximumDepthKey: u32,
  batchOffset: u32,
  rowCount: u32,
  frameIndex: u32,
  pickingAlphaThreshold: f32,
  // Paged sources only: whether activeRows indirects into the source batch.
  hasActiveRows: u32,
  // Paged sources only: first dense source row when activeRows is absent.
  sourceRowOffset: u32,
};

const SPLAT_FLAG_FLOAT_COLOR: u32 = ${GPU_SPLAT_FEATURE_FLAGS.floatColor}u;
const SPLAT_FLAG_COMPENSATE_DILATION: u32 = ${GPU_SPLAT_FEATURE_FLAGS.compensateDilation}u;
const SPLAT_FLAG_COMPENSATE_CLAMP: u32 = ${GPU_SPLAT_FEATURE_FLAGS.compensateClamp}u;
const SPLAT_FLAG_DYNAMIC_SUPPORT_RADIUS: u32 = ${GPU_SPLAT_FEATURE_FLAGS.dynamicSupportRadius}u;
const SPLAT_FLAG_ANALYTIC_KERNEL: u32 = ${GPU_SPLAT_FEATURE_FLAGS.analyticFragmentKernel}u;
const SPLAT_FLAG_STOCHASTIC_ALPHA: u32 = ${GPU_SPLAT_FEATURE_FLAGS.stochasticAlpha}u;
const SPLAT_FLAG_CLIPPING: u32 = ${GPU_SPLAT_FEATURE_FLAGS.clipping}u;
const SPLAT_FLAG_RADIAL_OPACITY: u32 = ${GPU_SPLAT_FEATURE_FLAGS.radialOpacity}u;
const SPLAT_FLAG_SRGB_DECODE: u32 = ${GPU_SPLAT_FEATURE_FLAGS.srgbDecode}u;

fn hasSplatFlag(flags: u32, flag: u32) -> bool {
  return (flags & flag) != 0u;
}
`;

/**
 * Support radius resolution shared by projection, both render paths, and picking.
 *
 * The projection pass sizes the quad with this and the vertex stage recomputes it from the stored
 * opacity, so the two cannot disagree about where the Gaussian ends.
 */
const GPU_SPLAT_SUPPORT_RADIUS_WGSL = /* wgsl */ `\
fn getSplatResolvedSupportRadius(flags: u32, alpha: f32, gaussianSupportRadius: f32) -> f32 {
  if (hasSplatFlag(flags, SPLAT_FLAG_RADIAL_OPACITY)) {
    return gaussianSupportRadius + 0.7 * max(alpha - 1.0, 0.0);
  }
  if (hasSplatFlag(flags, SPLAT_FLAG_DYNAMIC_SUPPORT_RADIUS)) {
    return getSplatSupportRadius(alpha, gaussianSupportRadius);
  }
  return gaussianSupportRadius;
}
`;

/**
 * Packed projected record and its accessors.
 *
 * Every stage that touches a projected record - projection, spherical harmonics, the two render
 * paths and picking - uses these accessors, so the packing cannot drift between them.
 */
const GPU_SPLAT_PROJECTED_RECORD_STRUCT = /* wgsl */ `\
struct ProjectedSplat {
  clipCenter: vec4<f32>,
  packedAxis0: u32,
  packedAxis1: u32,
  packedColorRG: u32,
  packedColorBA: u32,
};

fn packProjectedAxes(axis0: vec2<f32>, axis1: vec2<f32>) -> vec2<u32> {
  return vec2<u32>(pack2x16float(axis0), pack2x16float(axis1));
}

fn packProjectedColor(color: vec4<f32>) -> vec2<u32> {
  return vec2<u32>(pack2x16float(color.rg), pack2x16float(color.ba));
}

fn unpackProjectedColor(packedColorRG: u32, packedColorBA: u32) -> vec4<f32> {
  let redGreen = unpack2x16float(packedColorRG);
  let blueAlpha = unpack2x16float(packedColorBA);
  return vec4<f32>(redGreen.x, redGreen.y, blueAlpha.x, blueAlpha.y);
}

/**
 * Rounds an opacity through the half-precision storage format.
 *
 * The projection pass sizes a quad from the opacity-derived support radius and the vertex stage
 * recomputes that radius from the stored opacity. Quantizing before the radius is derived is what
 * makes the two agree exactly rather than to within a rounding step.
 */
fn quantizeProjectedAlpha(alpha: f32) -> f32 {
  return unpack2x16float(pack2x16float(vec2<f32>(alpha, 0.0))).x;
}
`;

const GPU_SPLAT_GRAPH_SHARED = /* wgsl */ `\
${GPU_SPLAT_GRAPH_UNIFORM_STRUCT}
${GPU_SPLAT_PROJECTED_RECORD_STRUCT}
${SPLAT_ANTIALIASING_WGSL}
${GPU_SPLAT_SUPPORT_RADIUS_WGSL}
`;

/**
 * Shared vertex-stage expansion of one projected record into its support quad.
 *
 * Both render paths and the picking path emit identical geometry and identical Gaussian
 * coordinates; they differ only in where the record comes from.
 */
const GPU_SPLAT_QUAD_EXPANSION_WGSL = /* wgsl */ `\
struct ExpandedSplatQuad {
  position: vec4<f32>,
  gaussianCoordinate: vec2<f32>,
  pixelHalfWidth: vec2<f32>,
  color: vec4<f32>,
};

fn getSplatQuadCorner(vertexIndex: u32) -> vec2<f32> {
  let corners = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0)
  );
  return corners[vertexIndex];
}

fn expandSplatQuad(
  uniforms: GraphSplatUniforms,
  vertexIndex: u32,
  clipCenter: vec4<f32>,
  packedAxis0: u32,
  packedAxis1: u32,
  packedColorRG: u32,
  packedColorBA: u32
) -> ExpandedSplatQuad {
  let corner = getSplatQuadCorner(vertexIndex);
  let axis0 = unpack2x16float(packedAxis0);
  let axis1 = unpack2x16float(packedAxis1);
  let color = unpackProjectedColor(packedColorRG, packedColorBA);
  let screenOffset = corner.x * axis0 + corner.y * axis1;
  let clipOffset = vec2<f32>(
    screenOffset.x * 2.0 / max(uniforms.viewportSize.x, 1.0),
    -screenOffset.y * 2.0 / max(uniforms.viewportSize.y, 1.0)
  ) * clipCenter.w;

  // The axes already carry the support radius, so recovering it recovers the sigma-space scale
  // the fragment stage needs. Every variant of the radius is a function of the stored opacity.
  let supportRadius = getSplatResolvedSupportRadius(
    uniforms.featureFlags,
    color.a,
    uniforms.gaussianSupportRadius
  );
  let safeRadius = max(supportRadius, 1e-6);
  let sigmaPixels = vec2<f32>(length(axis0), length(axis1)) / safeRadius;

  var output: ExpandedSplatQuad;
  output.position = vec4<f32>(clipCenter.xy + clipOffset, clipCenter.z, clipCenter.w);
  output.gaussianCoordinate = corner * supportRadius;
  // Half a physical pixel expressed in standard deviations along each projected axis.
  output.pixelHalfWidth = vec2<f32>(0.5, 0.5) / max(sigmaPixels, vec2<f32>(1e-6));
  output.color = color;
  return output;
}
`;

/**
 * Shared fragment-stage coverage and presentation.
 *
 * `getSplatFragmentCoverage` is the only place the Gaussian is evaluated, so switching between the
 * center sample and the pixel integral cannot change anything else about how a splat is shaded.
 */
const GPU_SPLAT_FRAGMENT_SHARED_WGSL = /* wgsl */ `\
fn getSplatFragmentCoverage(
  uniforms: GraphSplatUniforms,
  gaussianCoordinate: vec2<f32>,
  pixelHalfWidth: vec2<f32>
) -> f32 {
  if (hasSplatFlag(uniforms.featureFlags, SPLAT_FLAG_ANALYTIC_KERNEL)) {
    return getSplatPixelIntegral(gaussianCoordinate.x, pixelHalfWidth.x) *
      getSplatPixelIntegral(gaussianCoordinate.y, pixelHalfWidth.y);
  }
  return exp(-0.5 * dot(gaussianCoordinate, gaussianCoordinate));
}

/**
 * Resolves final fragment opacity from per-splat opacity and coverage.
 *
 * Radiance-field pages use a circular support and a nonlinear opacity profile so an opacity above
 * one reads as a solid parent rather than a brighter Gaussian.
 */
fn getSplatResolvedAlpha(
  uniforms: GraphSplatUniforms,
  splatAlpha: f32,
  coverage: f32,
  gaussianCoordinate: vec2<f32>
) -> f32 {
  if (!hasSplatFlag(uniforms.featureFlags, SPLAT_FLAG_RADIAL_OPACITY)) {
    return splatAlpha * coverage;
  }
  let radiusSquared = dot(gaussianCoordinate, gaussianCoordinate);
  let supportRadius = getSplatResolvedSupportRadius(
    uniforms.featureFlags,
    splatAlpha,
    uniforms.gaussianSupportRadius
  );
  if (radiusSquared > supportRadius * supportRadius) {
    return 0.0;
  }
  if (splatAlpha <= 1.0) {
    return splatAlpha * coverage;
  }
  let opaqueExponent = exp((splatAlpha * splatAlpha - 1.0) / 2.718281828459045);
  return 1.0 - pow(max(1.0 - coverage, 0.0), opaqueExponent);
}

fn getSplatPresentedColor(uniforms: GraphSplatUniforms, color: vec3<f32>) -> vec3<f32> {
  let linearColor = max(color * uniforms.exposure, vec3<f32>(0.0));
  return select(
    linearColor,
    linearColor / (vec3<f32>(1.0) + linearColor),
    uniforms.toneMapping == 1u
  );
}

/**
 * Per-pixel dither threshold for stochastic coverage.
 *
 * Jimenez's interleaved gradient noise, reseeded per frame so the residual noise averages out
 * under temporal accumulation. It is not true blue noise; it is the cheapest spatial distribution
 * that does not band.
 */
fn getSplatDitherThreshold(position: vec2<f32>, frameIndex: u32) -> f32 {
  let jitter = f32(frameIndex % 64u) * 5.588238;
  let jittered = position + vec2<f32>(jitter, jitter);
  return fract(52.9829189 * fract(0.06711056 * jittered.x + 0.00583715 * jittered.y));
}
`;

/** Projects one preserved source batch, culls invisible rows, and publishes global sort keys. */
export const GPU_SPLAT_PROJECTION_SHADER = /* wgsl */ `\
${GPU_SPLAT_GRAPH_SHARED}
${SPLAT_DEPTH_KEY_WGSL}
${SPLAT_CLIPPING_WGSL}

const MINIMUM_PROJECTABLE_W: f32 = 0.000001;
const MAXIMUM_FINITE_FLOAT: f32 = 3.402823466e38;

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> rotations: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> colors: array<u32>;
@group(0) @binding(4) var<storage, read> opacities: array<f32>;
@group(0) @binding(5) var<storage, read_write> projectedRecords: array<ProjectedSplat>;
@group(0) @binding(6) var<storage, read_write> depthKeys: array<u32>;
@group(0) @binding(7) var<storage, read_write> drawCommands: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> graphUniforms: GraphSplatUniforms;
@group(0) @binding(9) var<uniform> clipUniforms: GraphSplatClipUniforms;

fn isFiniteSplatValue(value: f32) -> bool {
  return abs(value) <= MAXIMUM_FINITE_FLOAT;
}

fn isFiniteSplatPosition(position: vec3<f32>) -> bool {
  return isFiniteSplatValue(position.x) &&
    isFiniteSplatValue(position.y) &&
    isFiniteSplatValue(position.z);
}

fn clearProjectedSplat(rowIndex: u32) {
  projectedRecords[rowIndex] = ProjectedSplat(
    vec4<f32>(0.0, 0.0, 2.0, 1.0),
    0u,
    0u,
    0u,
    0u
  );
  depthKeys[rowIndex] = graphUniforms.maximumDepthKey + 1u;
}

fn getProjectedScreenPosition(position: vec3<f32>) -> vec2<f32> {
  let clipPosition = graphUniforms.modelViewProjectionMatrix * vec4<f32>(position, 1.0);
  let inverseClipW = select(
    0.0,
    1.0 / clipPosition.w,
    abs(clipPosition.w) > MINIMUM_PROJECTABLE_W
  );
  return vec2<f32>(
    (clipPosition.x * inverseClipW * 0.5 + 0.5) * graphUniforms.viewportSize.x,
    (0.5 - clipPosition.y * inverseClipW * 0.5) * graphUniforms.viewportSize.y
  );
}

fn getProjectedRotation(quaternion: vec4<f32>) -> mat3x3<f32> {
  let quaternionLength = length(quaternion);
  let normalized = select(
    vec4<f32>(1.0, 0.0, 0.0, 0.0),
    quaternion / max(quaternionLength, MINIMUM_PROJECTABLE_W),
    quaternionLength > MINIMUM_PROJECTABLE_W
  );
  let quaternionW = normalized.x;
  let quaternionX = normalized.y;
  let quaternionY = normalized.z;
  let quaternionZ = normalized.w;
  return mat3x3<f32>(
    vec3<f32>(
      1.0 - 2.0 * (quaternionY * quaternionY + quaternionZ * quaternionZ),
      2.0 * (quaternionX * quaternionY + quaternionW * quaternionZ),
      2.0 * (quaternionX * quaternionZ - quaternionW * quaternionY)
    ),
    vec3<f32>(
      2.0 * (quaternionX * quaternionY - quaternionW * quaternionZ),
      1.0 - 2.0 * (quaternionX * quaternionX + quaternionZ * quaternionZ),
      2.0 * (quaternionY * quaternionZ + quaternionW * quaternionX)
    ),
    vec3<f32>(
      2.0 * (quaternionX * quaternionZ + quaternionW * quaternionY),
      2.0 * (quaternionY * quaternionZ - quaternionW * quaternionX),
      1.0 - 2.0 * (quaternionX * quaternionX + quaternionY * quaternionY)
    )
  );
}

fn getSourceColor(rowIndex: u32) -> vec4<f32> {
  if (hasSplatFlag(graphUniforms.featureFlags, SPLAT_FLAG_FLOAT_COLOR)) {
    let colorIndex = rowIndex * 4u;
    return vec4<f32>(
      bitcast<f32>(colors[colorIndex]),
      bitcast<f32>(colors[colorIndex + 1u]),
      bitcast<f32>(colors[colorIndex + 2u]),
      bitcast<f32>(colors[colorIndex + 3u])
    );
  }

  let packedColor = colors[rowIndex];
  return vec4<f32>(
    f32(packedColor & 255u),
    f32((packedColor >> 8u) & 255u),
    f32((packedColor >> 16u) & 255u),
    f32((packedColor >> 24u) & 255u)
  ) / 255.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
  let batchRowIndex = globalInvocationId.x;
  if (batchRowIndex >= graphUniforms.rowCount) {
    return;
  }

  let projectedRowIndex = graphUniforms.batchOffset + batchRowIndex;
  let componentIndex = batchRowIndex * 3u;
  let position = vec3<f32>(
    positions[componentIndex],
    positions[componentIndex + 1u],
    positions[componentIndex + 2u]
  );
  if (!isFiniteSplatPosition(position)) {
    clearProjectedSplat(projectedRowIndex);
    return;
  }

  let clipCenter = graphUniforms.modelViewProjectionMatrix * vec4<f32>(position, 1.0);
  if (
    !isFiniteSplatValue(clipCenter.x) ||
    !isFiniteSplatValue(clipCenter.y) ||
    !isFiniteSplatValue(clipCenter.z) ||
    !isFiniteSplatValue(clipCenter.w) ||
    clipCenter.w <= MINIMUM_PROJECTABLE_W ||
    clipCenter.z < -clipCenter.w ||
    clipCenter.z > clipCenter.w
  ) {
    clearProjectedSplat(projectedRowIndex);
    return;
  }

  let color = getSourceColor(batchRowIndex);
  let sourceAlpha = color.a * opacities[batchRowIndex];
  var alpha = sourceAlpha * graphUniforms.alphaScale * graphUniforms.levelFadeOpacity;
  if (hasSplatFlag(graphUniforms.featureFlags, SPLAT_FLAG_RADIAL_OPACITY) && sourceAlpha > 1.0) {
    alpha = min(sourceAlpha * 4.0 - 3.0, 5.0) * graphUniforms.alphaScale *
      graphUniforms.levelFadeOpacity;
  }
  if (!isFiniteSplatValue(alpha) || alpha < graphUniforms.alphaCutoff) {
    clearProjectedSplat(projectedRowIndex);
    return;
  }

  let scale = vec3<f32>(
    scales[componentIndex],
    scales[componentIndex + 1u],
    scales[componentIndex + 2u]
  );
  if (!isFiniteSplatPosition(scale)) {
    clearProjectedSplat(projectedRowIndex);
    return;
  }

  let rotationMatrix = getProjectedRotation(rotations[batchRowIndex]);
  let worldAxis0 = rotationMatrix[0] * scale.x;
  let worldAxis1 = rotationMatrix[1] * scale.y;
  let worldAxis2 = rotationMatrix[2] * scale.z;

  if (hasSplatFlag(graphUniforms.featureFlags, SPLAT_FLAG_CLIPPING)) {
    alpha = alpha * getSplatClipCoverage(
      clipUniforms,
      position,
      worldAxis0,
      worldAxis1,
      worldAxis2
    );
    if (alpha < graphUniforms.alphaCutoff) {
      clearProjectedSplat(projectedRowIndex);
      return;
    }
  }

  let center = getProjectedScreenPosition(position);
  let delta0 = getProjectedScreenPosition(position + worldAxis0) - center;
  let delta1 = getProjectedScreenPosition(position + worldAxis1) - center;
  let delta2 = getProjectedScreenPosition(position + worldAxis2) - center;
  let rowX = vec3<f32>(delta0.x, delta1.x, delta2.x);
  let rowY = vec3<f32>(delta0.y, delta1.y, delta2.y);

  // Pre-dilation covariance. Both filters are isotropic, so the off-diagonal term is shared
  // between the original and the dilated covariance and only the diagonal changes.
  let baseCovariance00 = dot(rowX, rowX);
  let covariance01 = dot(rowX, rowY);
  let baseCovariance11 = dot(rowY, rowY);
  let addedVariance = graphUniforms.screenFilterVariance + graphUniforms.levelFilterVariance;
  let covariance00 = baseCovariance00 + addedVariance;
  let covariance11 = baseCovariance11 + addedVariance;
  if (hasSplatFlag(graphUniforms.featureFlags, SPLAT_FLAG_COMPENSATE_DILATION)) {
    alpha = alpha * getSplatDilationCompensation(
      baseCovariance00,
      covariance01,
      baseCovariance11,
      addedVariance
    );
  }

  let halfTrace = (covariance00 + covariance11) * 0.5;
  let halfDifference = (covariance00 - covariance11) * 0.5;
  let discriminant = sqrt(max(halfDifference * halfDifference + covariance01 * covariance01, 0.0));
  let firstEigenvalue = max(halfTrace + discriminant, 0.0);
  let secondEigenvalue = max(halfTrace - discriminant, 0.0);
  var firstDirection = vec2<f32>(covariance01, firstEigenvalue - covariance00);
  if (length(firstDirection) <= MINIMUM_PROJECTABLE_W) {
    firstDirection = vec2<f32>(firstEigenvalue - covariance11, covariance01);
  }
  if (length(firstDirection) <= MINIMUM_PROJECTABLE_W) {
    firstDirection = vec2<f32>(1.0, 0.0);
  }
  firstDirection = normalize(firstDirection);
  let secondDirection = vec2<f32>(-firstDirection.y, firstDirection.x);
  let firstAxisLength = max(sqrt(firstEigenvalue), 0.001);
  let secondAxisLength = max(sqrt(secondEigenvalue), 0.001);
  let maximumAxisLength = max(firstAxisLength, secondAxisLength);
  if (
    !isFiniteSplatValue(maximumAxisLength) ||
    maximumAxisLength * graphUniforms.radiusScale < graphUniforms.screenSizeCutoffPixels
  ) {
    clearProjectedSplat(projectedRowIndex);
    return;
  }

  let isRadialOpacity = hasSplatFlag(graphUniforms.featureFlags, SPLAT_FLAG_RADIAL_OPACITY);
  // Radiance-field pages clamp the final support axes instead of the covariance, so the parent's
  // shape is preserved and only its on-screen reach is bounded.
  let clampScale = select(
    min(max(graphUniforms.maxScreenSpaceSplatSize, 0.001) / maximumAxisLength, 1.0),
    1.0,
    isRadialOpacity
  );
  if (hasSplatFlag(graphUniforms.featureFlags, SPLAT_FLAG_COMPENSATE_CLAMP)) {
    alpha = min(alpha * getSplatClampCompensation(clampScale), 1.0);
  }

  // Quantizing before the radius is derived keeps the quad the projection sizes and the
  // sigma-space coordinate the vertex stage recomputes exactly consistent.
  let storedAlpha = quantizeProjectedAlpha(alpha);
  if (storedAlpha < graphUniforms.alphaCutoff) {
    clearProjectedSplat(projectedRowIndex);
    return;
  }
  let supportRadius = getSplatResolvedSupportRadius(
    graphUniforms.featureFlags,
    storedAlpha,
    graphUniforms.gaussianSupportRadius
  );
  if (supportRadius <= 0.0) {
    clearProjectedSplat(projectedRowIndex);
    return;
  }

  let supportScale = supportRadius * graphUniforms.radiusScale * clampScale;
  let maximumSupportAxisLength = max(graphUniforms.maxScreenSpaceSplatSize, 0.001);
  var firstSupportAxisLength = firstAxisLength * supportScale;
  var secondSupportAxisLength = secondAxisLength * supportScale;
  if (isRadialOpacity) {
    firstSupportAxisLength = min(firstSupportAxisLength, maximumSupportAxisLength);
    secondSupportAxisLength = min(secondSupportAxisLength, maximumSupportAxisLength);
  }
  let axis0 = firstDirection * firstSupportAxisLength;
  let axis1 = secondDirection * secondSupportAxisLength;
  let screenExtent = abs(axis0) + abs(axis1);
  if (
    center.x + screenExtent.x < 0.0 ||
    center.y + screenExtent.y < 0.0 ||
    center.x - screenExtent.x > graphUniforms.viewportSize.x ||
    center.y - screenExtent.y > graphUniforms.viewportSize.y
  ) {
    clearProjectedSplat(projectedRowIndex);
    return;
  }

  depthKeys[projectedRowIndex] = packSplatDepthKey(
    graphUniforms.depthKeyMode,
    graphUniforms.maximumDepthKey,
    clipCenter.w,
    clipCenter.z / clipCenter.w * 0.5 + 0.5,
    graphUniforms.depthRange.x,
    graphUniforms.depthRange.y
  );
  let packedAxes = packProjectedAxes(axis0, axis1);
  let packedColor = packProjectedColor(vec4<f32>(color.rgb, storedAlpha));
  projectedRecords[projectedRowIndex] = ProjectedSplat(
    clipCenter,
    packedAxes.x,
    packedAxes.y,
    packedColor.x,
    packedColor.y
  );
  atomicAdd(&drawCommands[1u], 1u);
}
`;

/** Applies higher-order SH and semantic visibility directly to GPU-projected source records. */
export const GPU_SPLAT_FEATURE_SHADER = /* wgsl */ `\
${GPU_SPLAT_GRAPH_SHARED}

struct GraphSplatFeatureUniforms {
  cameraPosition: vec3<f32>,
  sphericalHarmonicsDegree: u32,
  sphericalHarmonicsStride: u32,
  hasSemanticIds: u32,
  includeCount: u32,
  excludeCount: u32,
  hasIncludeSelection: u32,
  includeUnlabeled: u32,
  semanticFilterActive: u32,
  padding: u32,
};

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> sphericalHarmonics: array<f32>;
@group(0) @binding(2) var<storage, read> semanticIds: array<u32>;
@group(0) @binding(3) var<storage, read> semanticSelections: array<u32>;
@group(0) @binding(4) var<storage, read_write> projectedRecords: array<ProjectedSplat>;
@group(0) @binding(5) var<storage, read_write> depthKeys: array<u32>;
@group(0) @binding(6) var<storage, read_write> drawCommands: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> graphUniforms: GraphSplatUniforms;
@group(0) @binding(8) var<uniform> featureUniforms: GraphSplatFeatureUniforms;

fn hasSelectedSemantic(semanticId: u32, offset: u32, count: u32) -> bool {
  for (var selectionIndex = 0u; selectionIndex < count; selectionIndex++) {
    if (semanticSelections[offset + selectionIndex] == semanticId) {
      return true;
    }
  }
  return false;
}

fn acceptsProjectedSemantic(batchRowIndex: u32) -> bool {
  if (featureUniforms.semanticFilterActive == 0u) {
    return true;
  }
  if (featureUniforms.hasSemanticIds == 0u) {
    return featureUniforms.includeUnlabeled != 0u;
  }

  let semanticId = semanticIds[batchRowIndex];
  if (
    featureUniforms.hasIncludeSelection != 0u &&
    !hasSelectedSemantic(semanticId, 0u, featureUniforms.includeCount)
  ) {
    return false;
  }
  return !hasSelectedSemantic(
    semanticId,
    featureUniforms.includeCount,
    featureUniforms.excludeCount
  );
}

fn getGraphSphericalHarmonicBasis(basisIndex: u32, direction: vec3<f32>) -> f32 {
  let directionXX = direction.x * direction.x;
  let directionYY = direction.y * direction.y;
  let directionZZ = direction.z * direction.z;
  switch basisIndex {
    case 0u: { return -0.4886025119029199 * direction.y; }
    case 1u: { return 0.4886025119029199 * direction.z; }
    case 2u: { return -0.4886025119029199 * direction.x; }
    case 3u: { return 1.0925484305920792 * direction.x * direction.y; }
    case 4u: { return -1.0925484305920792 * direction.y * direction.z; }
    case 5u: { return 0.31539156525252005 * (2.0 * directionZZ - directionXX - directionYY); }
    case 6u: { return -1.0925484305920792 * direction.x * direction.z; }
    case 7u: { return 0.5462742152960396 * (directionXX - directionYY); }
    case 8u: { return -0.5900435899266435 * direction.y * (3.0 * directionXX - directionYY); }
    case 9u: { return 2.890611442640554 * direction.x * direction.y * direction.z; }
    case 10u: { return -0.4570457994644658 * direction.y * (4.0 * directionZZ - directionXX - directionYY); }
    case 11u: { return 0.3731763325901154 * direction.z * (2.0 * directionZZ - 3.0 * directionXX - 3.0 * directionYY); }
    case 12u: { return -0.4570457994644658 * direction.x * (4.0 * directionZZ - directionXX - directionYY); }
    case 13u: { return 1.445305721320277 * direction.z * (directionXX - directionYY); }
    case 14u: { return -0.5900435899266435 * direction.x * (directionXX - 3.0 * directionYY); }
    default: { return 0.0; }
  }
}

fn evaluateGraphSphericalHarmonics(color: vec3<f32>, batchRowIndex: u32) -> vec3<f32> {
  let degree = featureUniforms.sphericalHarmonicsDegree;
  if (degree == 0u) {
    return color;
  }

  let positionOffset = batchRowIndex * 3u;
  let position = vec3<f32>(
    positions[positionOffset],
    positions[positionOffset + 1u],
    positions[positionOffset + 2u]
  );
  let direction = position - featureUniforms.cameraPosition;
  let directionLength = length(direction);
  if (directionLength <= 0.000001) {
    return color;
  }

  let normalizedDirection = direction / directionLength;
  let basisCount = (degree + 1u) * (degree + 1u) - 1u;
  var evaluatedColor = color;
  for (var basisIndex = 0u; basisIndex < basisCount; basisIndex++) {
    let coefficientOffset =
      batchRowIndex * featureUniforms.sphericalHarmonicsStride + basisIndex * 3u;
    let coefficients = vec3<f32>(
      sphericalHarmonics[coefficientOffset],
      sphericalHarmonics[coefficientOffset + 1u],
      sphericalHarmonics[coefficientOffset + 2u]
    );
    evaluatedColor += coefficients *
      getGraphSphericalHarmonicBasis(basisIndex, normalizedDirection);
  }
  return evaluatedColor;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
  let batchRowIndex = globalInvocationId.x;
  if (batchRowIndex >= graphUniforms.rowCount) {
    return;
  }

  let projectedRowIndex = graphUniforms.batchOffset + batchRowIndex;
  let invalidDepthKey = graphUniforms.maximumDepthKey + 1u;
  if (depthKeys[projectedRowIndex] == invalidDepthKey) {
    return;
  }
  if (!acceptsProjectedSemantic(batchRowIndex)) {
    projectedRecords[projectedRowIndex].packedColorRG = 0u;
    projectedRecords[projectedRowIndex].packedColorBA = 0u;
    depthKeys[projectedRowIndex] = invalidDepthKey;
    atomicSub(&drawCommands[1u], 1u);
    return;
  }

  if (featureUniforms.sphericalHarmonicsDegree != 0u) {
    let projectedColor = unpackProjectedColor(
      projectedRecords[projectedRowIndex].packedColorRG,
      projectedRecords[projectedRowIndex].packedColorBA
    );
    // Only the chromatic bands change here; the stored opacity must survive untouched because the
    // projected quad was already sized from it.
    let packedColor = packProjectedColor(vec4<f32>(
      evaluateGraphSphericalHarmonics(projectedColor.rgb, batchRowIndex),
      projectedColor.a
    ));
    projectedRecords[projectedRowIndex].packedColorRG = packedColor.x;
    projectedRecords[projectedRowIndex].packedColorBA = packedColor.y;
  }
  if (hasSplatFlag(graphUniforms.featureFlags, SPLAT_FLAG_SRGB_DECODE)) {
    let projectedColor = unpackProjectedColor(
      projectedRecords[projectedRowIndex].packedColorRG,
      projectedRecords[projectedRowIndex].packedColorBA
    );
    let packedColor = packProjectedColor(vec4<f32>(
      pow(max(projectedColor.rgb, vec3<f32>(0.0)), vec3<f32>(2.2)),
      projectedColor.a
    ));
    projectedRecords[projectedRowIndex].packedColorRG = packedColor.x;
    projectedRecords[projectedRowIndex].packedColorBA = packedColor.y;
  }
}
`;

/** Gathers the globally sorted records into a sequential compatibility vertex stream. */
export const GPU_SPLAT_GATHER_SHADER = /* wgsl */ `\
${GPU_SPLAT_PROJECTED_RECORD_STRUCT}

@group(0) @binding(0) var<storage, read> projectedRecords: array<ProjectedSplat>;
@group(0) @binding(1) var<storage, read> sortedIds: array<u32>;
@group(0) @binding(2) var<storage, read_write> sortedRecords: array<ProjectedSplat>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) globalInvocationId: vec3<u32>) {
  let sortedIndex = globalInvocationId.x;
  if (sortedIndex >= arrayLength(&sortedRecords)) {
    return;
  }
  sortedRecords[sortedIndex] = projectedRecords[sortedIds[sortedIndex]];
}
`;

/** Renders globally sorted, preprojected HDR Gaussians without revisiting source batches. */
export const GPU_SPLAT_RENDER_SHADER = /* wgsl */ `\
${GPU_SPLAT_GRAPH_SHARED}
${GPU_SPLAT_QUAD_EXPANSION_WGSL}
${GPU_SPLAT_FRAGMENT_SHARED_WGSL}

@group(0) @binding(0) var<uniform> graphUniforms: GraphSplatUniforms;
@group(0) @binding(1) var<storage, read> projectedRecords: array<ProjectedSplat>;
@group(0) @binding(2) var<storage, read> sortedIds: array<u32>;

struct GraphSplatFragmentInputs {
  @builtin(position) position: vec4<f32>,
  @location(0) gaussianCoordinate: vec2<f32>,
  @location(1) pixelHalfWidth: vec2<f32>,
  @location(2) color: vec4<f32>,
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> GraphSplatFragmentInputs {
  let projected = projectedRecords[sortedIds[instanceIndex]];
  let quad = expandSplatQuad(
    graphUniforms,
    vertexIndex,
    projected.clipCenter,
    projected.packedAxis0,
    projected.packedAxis1,
    projected.packedColorRG,
    projected.packedColorBA
  );

  var output: GraphSplatFragmentInputs;
  output.position = quad.position;
  output.gaussianCoordinate = quad.gaussianCoordinate;
  output.pixelHalfWidth = quad.pixelHalfWidth;
  output.color = quad.color;
  return output;
}

@fragment
fn fragmentMain(input: GraphSplatFragmentInputs) -> @location(0) vec4<f32> {
  let coverage = getSplatFragmentCoverage(
    graphUniforms,
    input.gaussianCoordinate,
    input.pixelHalfWidth
  );
  let alpha = getSplatResolvedAlpha(
    graphUniforms,
    input.color.a,
    coverage,
    input.gaussianCoordinate
  );
  if (alpha < graphUniforms.alphaCutoff) {
    discard;
  }

  let mappedColor = getSplatPresentedColor(graphUniforms, input.color.rgb);
  if (hasSplatFlag(graphUniforms.featureFlags, SPLAT_FLAG_STOCHASTIC_ALPHA)) {
    // Dithered coverage with opaque blending and depth writes: no ordering is consumed, so the
    // splats behave like opaque geometry inside whatever render pass they were recorded into.
    if (alpha < getSplatDitherThreshold(input.position.xy, graphUniforms.frameIndex)) {
      discard;
    }
    return vec4<f32>(mappedColor, 1.0);
  }
  return vec4<f32>(mappedColor, alpha);
}
`;

/**
 * Compatibility render shader: one uniform binding, no storage buffers in the vertex stage.
 *
 * The sorted records arrive as an instanced vertex stream produced by
 * {@link GPU_SPLAT_GATHER_SHADER}, so the vertex stage performs no indirection at all.
 */
export const GPU_SPLAT_COMPATIBLE_RENDER_SHADER = /* wgsl */ `\
${GPU_SPLAT_GRAPH_SHARED}
${GPU_SPLAT_QUAD_EXPANSION_WGSL}
${GPU_SPLAT_FRAGMENT_SHARED_WGSL}

@group(0) @binding(0) var<uniform> graphUniforms: GraphSplatUniforms;

struct GraphSplatFragmentInputs {
  @builtin(position) position: vec4<f32>,
  @location(0) gaussianCoordinate: vec2<f32>,
  @location(1) pixelHalfWidth: vec2<f32>,
  @location(2) color: vec4<f32>,
};

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @location(0) instanceClipCenter: vec4<f32>,
  @location(1) instancePackedRecord: vec4<u32>
) -> GraphSplatFragmentInputs {
  let quad = expandSplatQuad(
    graphUniforms,
    vertexIndex,
    instanceClipCenter,
    instancePackedRecord.x,
    instancePackedRecord.y,
    instancePackedRecord.z,
    instancePackedRecord.w
  );

  var output: GraphSplatFragmentInputs;
  output.position = quad.position;
  output.gaussianCoordinate = quad.gaussianCoordinate;
  output.pixelHalfWidth = quad.pixelHalfWidth;
  output.color = quad.color;
  return output;
}

@fragment
fn fragmentMain(input: GraphSplatFragmentInputs) -> @location(0) vec4<f32> {
  let coverage = getSplatFragmentCoverage(
    graphUniforms,
    input.gaussianCoordinate,
    input.pixelHalfWidth
  );
  let alpha = getSplatResolvedAlpha(
    graphUniforms,
    input.color.a,
    coverage,
    input.gaussianCoordinate
  );
  if (alpha < graphUniforms.alphaCutoff) {
    discard;
  }

  let mappedColor = getSplatPresentedColor(graphUniforms, input.color.rgb);
  if (hasSplatFlag(graphUniforms.featureFlags, SPLAT_FLAG_STOCHASTIC_ALPHA)) {
    if (alpha < getSplatDitherThreshold(input.position.xy, graphUniforms.frameIndex)) {
      discard;
    }
    return vec4<f32>(mappedColor, 1.0);
  }
  return vec4<f32>(mappedColor, alpha);
}
`;

/** Shared WGSL prelude reused by graph-native picking and mixed-scene composition. @internal */
export const GPU_SPLAT_GRAPH_SHARED_WGSL = GPU_SPLAT_GRAPH_SHARED;

/** Shared quad expansion reused by graph-native picking. @internal */
export const GPU_SPLAT_QUAD_EXPANSION_SHADER_WGSL = GPU_SPLAT_QUAD_EXPANSION_WGSL;

/** Shared fragment coverage and presentation reused by graph-native picking. @internal */
export const GPU_SPLAT_FRAGMENT_SHARED_SHADER_WGSL = GPU_SPLAT_FRAGMENT_SHARED_WGSL;

/** Clip planes one projection pass can evaluate, re-exported for renderer validation. */
export const GPU_SPLAT_MAXIMUM_CLIP_PLANES = MAXIMUM_SPLAT_CLIP_PLANES;
