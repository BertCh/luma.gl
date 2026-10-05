// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {GPURasterBufferToTexture} from '../../gpu-raster/index';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  captureGraphCommandNodes,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings,
  validateTerrainTexture
} from '../terrain-analysis/terrain-analysis-utils';
import {
  GPU_TERRAIN_HORIZON_MAX_DIRECTION_COUNT,
  GPU_TERRAIN_HORIZON_MIN_DIRECTION_COUNT
} from './gpu-terrain-horizon';
import {
  getTerrainHorizonReadWGSL,
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainHorizonFormat,
  validateTerrainHorizonView,
  validateTerrainIlluminationView,
  type GPUTerrainHorizonFormat
} from './terrain-illumination-utils';

/** Number of float32 values read from `GPUSolarShadowMaskProps.settings`. */
export const GPU_SOLAR_SHADOW_MASK_PARAMETER_LENGTH = 8;

/** Mean angular radius of the solar disk in degrees, the default penumbra half-width. */
export const GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES = 0.2666;

/** CPU-side description packed by {@link getGPUSolarShadowMaskParameterValues}. */
export type GPUSolarShadowMaskSettings = {
  /** Direction of the sun, degrees clockwise from north. See `getSolarPosition`. */
  azimuthDegrees: number;
  /** Elevation of the sun center above the astronomical horizon in degrees. */
  altitudeDegrees: number;
  /**
   * Angular radius of the solar disk in degrees: 0 gives hard shadows. Defaults to
   * {@link GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES}.
   */
  angularRadiusDegrees?: number;
  /** Direct light multiplier for `illumination`. Defaults to 1. */
  sunIntensity?: number;
  /** Ambient light multiplier for `illumination`, scaled by the sky-view factor. Defaults to 0. */
  ambientIntensity?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUSolarShadowMask}:
 * `[azimuthDegrees, altitudeDegrees, angularRadiusDegrees, sunIntensity, ambientIntensity, 0, 0, 0]`.
 *
 * @throws If the angular radius is negative or not finite, or `target` is too short.
 */
export function getGPUSolarShadowMaskParameterValues(
  settings: GPUSolarShadowMaskSettings,
  target: Float32Array = new Float32Array(GPU_SOLAR_SHADOW_MASK_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SOLAR_SHADOW_MASK_PARAMETER_LENGTH) {
    throw new Error('Solar shadow mask settings target must hold 8 values');
  }
  const angularRadius = settings.angularRadiusDegrees ?? GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES;
  if (!Number.isFinite(angularRadius) || angularRadius < 0) {
    throw new Error('Solar shadow mask angular radius must be finite and non-negative');
  }
  target.set([
    settings.azimuthDegrees,
    settings.altitudeDegrees,
    angularRadius,
    settings.sunIntensity ?? 1,
    settings.ambientIntensity ?? 0,
    0,
    0,
    0
  ]);
  return target;
}

/**
 * Properties for {@link GPUSolarShadowMask}.
 *
 * Topology: grid size, `directionCount`, and which inputs and outputs exist. Per-frame: `settings`
 * (sun position, disk radius, intensities) and input contents.
 */
export type GPUSolarShadowMaskProps = {
  /** Prefix for node and transient IDs. Defaults to `'solar-shadow-mask'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Number of horizon sectors, an integer in `[4, 64]`, matching the `GPUTerrainHorizon`. */
  directionCount: number;
  /**
   * Storage format of `horizon`, matching the `GPUTerrainHorizon` that wrote it. Defaults to
   * `'float32'`; `'unorm16'` reads packed 16-bit codes (0.0027 degree steps), so visibility
   * differs from the float32 path by at most one code step of horizon.
   */
  horizonFormat?: GPUTerrainHorizonFormat;
  /**
   * Pixel-major horizon angles in degrees from `GPUTerrainHorizon`: `width * height *
   * directionCount` float32 values, or for `'unorm16'` a `uint32` view of
   * `ceil(width * height * directionCount / 2)` words.
   */
  horizon: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Per-frame settings with at least 8 float32 values, see {@link getGPUSolarShadowMaskParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Optional slope in degrees, as written by `GPUTerrainDerivatives`. Required for `illumination`. */
  slope?: GraphDataView<'float32'>;
  /** Optional aspect in degrees clockwise from north, -1 when flat. Required with `slope`. */
  aspect?: GraphDataView<'float32'>;
  /** Optional sky-view factor scaling ambient light in `illumination`. */
  skyViewFactor?: GraphDataView<'float32'>;
  /** Optional fraction of the solar disk above the local horizon, in `[0, 1]`. */
  sunVisibility?: GraphDataView<'float32'>;
  /** Optional `ambient * svf + sun * sunVisibility * max(cos(incidence), 0)`. Requires slope and aspect. */
  illumination?: GraphDataView<'float32'>;
  /** Optional storage texture (`r32float` or `rgba32float`, channel 0) receiving `sunVisibility`. */
  sunVisibilityTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
  /** Optional storage texture (`r32float` or `rgba32float`, channel 0) receiving `illumination`. */
  illuminationTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
};

/**
 * Turns a precomputed horizon map and a per-frame sun into soft or hard terrain shadows.
 *
 * The horizon angle at the sun azimuth is interpolated linearly between the two bracketing sectors
 * (with wraparound). `sunVisibility` is the area fraction of the solar disk above that horizon:
 * with `u = clamp((altitude - horizon) / radius, -1, 1)` it is
 * `1 - (acos(u) - u * sqrt(1 - u^2)) / PI`, a hard step when the radius is 0, and 0 when the whole
 * disk is below the astronomical horizon. `illumination` adds Lambertian incidence from slope and
 * aspect and an ambient term scaled by the optional sky-view factor, matching a "sun + ambient"
 * light rig in data space.
 *
 * Each frame reads two horizon values per pixel, so animating time of day is one settings write
 * and never recompiles. Pixels with a NaN horizon, slope, aspect, or sky-view factor receive NaN.
 */
export class GPUSolarShadowMask implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSolarShadowMaskProps;
  /** Storage format of the horizon input. */
  readonly horizonFormat: GPUTerrainHorizonFormat;

  constructor(props: GPUSolarShadowMaskProps) {
    this.id = props.id ?? 'solar-shadow-mask';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    const {directionCount} = props;
    if (
      !Number.isSafeInteger(directionCount) ||
      directionCount < GPU_TERRAIN_HORIZON_MIN_DIRECTION_COUNT ||
      directionCount > GPU_TERRAIN_HORIZON_MAX_DIRECTION_COUNT
    ) {
      throw new Error(`${id} directionCount must be an integer in [4, 64]`);
    }
    if (
      !props.sunVisibility &&
      !props.illumination &&
      !props.sunVisibilityTexture &&
      !props.illuminationTexture
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    if (Boolean(props.slope) !== Boolean(props.aspect)) {
      throw new Error(`${id} slope and aspect must be provided together`);
    }
    if ((props.illumination || props.illuminationTexture) && !props.slope) {
      throw new Error(`${id} illumination requires slope and aspect`);
    }
    validateTerrainHorizonFormat(id, props.horizonFormat ?? 'float32');
    this.horizonFormat = props.horizonFormat ?? 'float32';
    validateTerrainHorizonView(
      id,
      'horizon',
      props.horizon,
      this.horizonFormat,
      pixelCount,
      directionCount
    );
    for (const [name, view] of [
      ['slope', props.slope],
      ['aspect', props.aspect],
      ['skyViewFactor', props.skyViewFactor],
      ['sunVisibility', props.sunVisibility],
      ['illumination', props.illumination]
    ] as const) {
      validateTerrainIlluminationView(id, name, view, 'float32', pixelCount);
    }
    validateTerrainSettings(id, props.settings, GPU_SOLAR_SHADOW_MASK_PARAMETER_LENGTH);
    for (const [name, texture] of [
      ['sunVisibilityTexture', props.sunVisibilityTexture],
      ['illuminationTexture', props.illuminationTexture]
    ] as const) {
      validateTerrainTexture(
        id,
        name,
        texture,
        ['r32float', 'rgba32float'],
        props.width,
        props.height
      );
    }
    validateTerrainBuffersDistinct(
      id,
      [props.sunVisibility, props.illumination],
      [props.horizon, props.settings, props.slope, props.aspect, props.skyViewFactor]
    );
  }

  /** Returns the shadow kernel and optional texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height, directionCount} = props;
    const pixelCount = width * height;
    validateGraphViewsBelongToGraph(id, graph, [
      props.horizon,
      props.settings,
      props.slope,
      props.aspect,
      props.skyViewFactor,
      props.sunVisibility,
      props.illumination
    ]);
    for (const texture of [props.sunVisibilityTexture, props.illuminationTexture]) {
      if (texture && texture.texture.graph !== graph) {
        throw new Error(`${id} views must belong to the target graph`);
      }
    }
    const visibilityTarget =
      props.sunVisibility ??
      (props.sunVisibilityTexture
        ? createTransientView(graph, `${id}-sun-visibility`, 'float32', pixelCount)
        : undefined);
    const illuminationTarget =
      props.illumination ??
      (props.illuminationTexture
        ? createTransientView(graph, `${id}-illumination`, 'float32', pixelCount)
        : undefined);
    const bindings: WGSLKernelBinding[] = [
      {
        name: 'horizon',
        view: props.horizon,
        type: this.horizonFormat === 'unorm16' ? 'u32' : 'f32',
        access: 'read'
      },
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
    ];
    if (props.slope && props.aspect) {
      bindings.push(
        {name: 'slopeValues', view: props.slope, type: 'f32', access: 'read'},
        {
          name: 'aspectValues',
          view: props.aspect,
          type: 'f32',
          access: 'read'
        }
      );
    }
    if (props.skyViewFactor) {
      bindings.push({
        name: 'skyView',
        view: props.skyViewFactor,
        type: 'f32',
        access: 'read'
      });
    }
    if (visibilityTarget) {
      bindings.push({
        name: 'visibilityValues',
        view: visibilityTarget,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (illuminationTarget) {
      bindings.push({
        name: 'illuminationValues',
        view: illuminationTarget,
        type: 'f32',
        access: 'read_write'
      });
    }
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-shadow`,
        operation: 'GPUSolarShadowMask',
        bindings,
        invocationCount: pixelCount,
        declarations: `const DIRECTION_COUNT: u32 = ${directionCount}u;
const SECTOR_DEGREES: f32 = ${getWGSLFloatLiteral(360 / directionCount)};
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${getTerrainHorizonReadWGSL(this.horizonFormat, 'horizon')}`,
        body: `let azimuthDegrees = settings[settingsOffset];
  let altitudeDegrees = settings[settingsOffset + 1u];
  let radiusDegrees = settings[settingsOffset + 2u];
  let wrappedAzimuth = azimuthDegrees - 360.0 * floor(azimuthDegrees / 360.0);
  let sectorPosition = wrappedAzimuth / SECTOR_DEGREES;
  let lowerSector = min(u32(floor(sectorPosition)), DIRECTION_COUNT - 1u);
  let upperSector = (lowerSector + 1u) % DIRECTION_COUNT;
  let blend = clamp(sectorPosition - f32(lowerSector), 0.0, 1.0);
  let horizonAngle = mix(
    readHorizonAngle(index * DIRECTION_COUNT + lowerSector),
    readHorizonAngle(index * DIRECTION_COUNT + upperSector),
    blend
  );
  var visibility = 0.0;
  if (altitudeDegrees + radiusDegrees > 0.0) {
    if (radiusDegrees > 0.0) {
      let u = clamp((altitudeDegrees - horizonAngle) / radiusDegrees, -1.0, 1.0);
      visibility = clamp(1.0 - (acos(u) - u * sqrt(max(1.0 - u * u, 0.0))) / PI, 0.0, 1.0);
    } else {
      visibility = select(0.0, 1.0, altitudeDegrees > horizonAngle);
    }
  }
  var isValid = isFiniteValue(horizonAngle);
  let invalidValue = getNaN(index);
  ${visibilityTarget ? 'visibilityValues[visibilityValuesOffset + index] = select(invalidValue, visibility, isValid);' : ''}
  ${
    illuminationTarget
      ? `let slope = slopeValues[slopeValuesOffset + index] * DEGREES_TO_RADIANS;
  let aspect = aspectValues[aspectValuesOffset + index];
  let azimuth = azimuthDegrees * DEGREES_TO_RADIANS;
  let altitude = altitudeDegrees * DEGREES_TO_RADIANS;
  var cosIncidence = sin(altitude);
  if (aspect >= 0.0) {
    cosIncidence = cos(slope) * sin(altitude) +
      sin(slope) * cos(altitude) * cos(azimuth - aspect * DEGREES_TO_RADIANS);
  }
  let ambient = settings[settingsOffset + 4u] * ${props.skyViewFactor ? 'skyView[skyViewOffset + index]' : '1.0'};
  let illumination = ambient + settings[settingsOffset + 3u] * visibility * max(cosIncidence, 0.0);
  isValid = isValid && isFiniteValue(illumination) && isFiniteValue(slope) && isFiniteValue(aspect);
  illuminationValues[illuminationValuesOffset + index] = select(invalidValue, illumination, isValid);`
      : ''
  }`
      })
    ];
    for (const [name, target, texture] of [
      ['sun-visibility', visibilityTarget, props.sunVisibilityTexture],
      ['illumination', illuminationTarget, props.illuminationTexture]
    ] as const) {
      if (texture && target) {
        nodes.push(
          ...captureGraphCommandNodes(graph, () =>
            new GPURasterBufferToTexture({
              id: `${id}-${name}-texture`,
              input: {
                id: `${id}-${name}-band`,
                format: 'float32',
                storage: {kind: 'buffer', values: target}
              },
              output: texture,
              channel: 0
            }).addToGraph(graph)
          )
        );
      }
    }
    return nodes;
  }
}
