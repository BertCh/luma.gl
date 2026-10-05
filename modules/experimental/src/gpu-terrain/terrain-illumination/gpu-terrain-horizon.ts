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
import {GPURasterBufferToTexture, type GPURasterBand} from '../../gpu-raster/index';
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
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings,
  validateTerrainTexture
} from '../terrain-analysis/terrain-analysis-utils';
import {
  getTerrainHorizonStorageLength,
  TERRAIN_HORIZON_UNORM16_SCALE,
  TERRAIN_HORIZON_UNORM16_STEP_DEGREES,
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainHorizonFormat,
  validateTerrainHorizonView,
  validateTerrainIlluminationBindingSize,
  validateTerrainIlluminationView,
  type GPUTerrainHorizonFormat
} from './terrain-illumination-utils';
import {
  getTerrainGroundCellSizeWGSL,
  type GPUTerrainCellSizeMode,
  type GPUTerrainRowDirection,
  validateTerrainCellSizeMode,
  validateTerrainRowDirection
} from '../terrain-grid-utils';
import {getTerrainHorizonSweepNode, TERRAIN_SWEEP_MAX_EXTENT} from './terrain-horizon-sweep';

export type {GPUTerrainHorizonFormat} from './terrain-illumination-utils';

/**
 * How {@link GPUTerrainHorizon} finds horizons:
 * - `'march'`: bounded ray march with bilinear samples on the baked step schedule, O(steps) per
 *   pixel and sector.
 * - `'sweep'`: exact discrete horizons on digital lines through pixel centres with upper-hull
 *   pointers (Stewart 1998 style), amortised O(1) per pixel and sector. Samples are pixel centres
 *   on a digital line whose lateral offset from the ideal ray is at most half a pixel, so values
 *   differ from the march by sampling, not by precision. `stepGrowth` must be 1.
 */
export type GPUTerrainHorizonAlgorithm = 'march' | 'sweep';

/** Number of float32 values read from `GPUTerrainHorizonProps.settings`. */
export const GPU_TERRAIN_HORIZON_PARAMETER_LENGTH = 8;

/**
 * Number of float32 values read from `GPUTerrainHorizonProps.settings` when
 * `anisotropicSkyViewFactor` is bound: the 8 base values plus
 * `[anisotropyAzimuthDegrees, anisotropyLevel, anisotropyMinimumWeight, 0]`.
 */
export const GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH = 12;

/**
 * Default compass azimuth of maximum anisotropic sky-view weight, degrees clockwise from north.
 * RVT's default `a_main_direction = 315` runs counterclockwise from north and corresponds to 45
 * here (`ours = (360 - rvt) mod 360`); this default is the north-west light at 315.
 */
export const GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES = 315;

/** Default anisotropic sky-view exponent (RVT "low" preset: 4). */
export const GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL = 4;

/** Default anisotropic sky-view minimum weight (RVT "low" preset: 0.4). */
export const GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT = 0.4;

/** Degrees per code step of the `'unorm16'` horizon format: `180 / 65534`. */
export const GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES = TERRAIN_HORIZON_UNORM16_STEP_DEGREES;

/**
 * Encodes a horizon angle as a 16-bit code: 0 means invalid (NaN or non-finite); otherwise
 * `1 + floor(clamp(angle, -90, 90) * 65534 / 180 + 32767.5)`, so codes 1 to 65535 cover
 * `[-90, 90]` degrees in steps of {@link GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES}.
 *
 * The GPU computes the same expression in float32 with a float32 scale, so GPU and CPU codes
 * may differ by one at rounding ties.
 */
export function encodeGPUTerrainHorizonUnorm16(angleDegrees: number): number {
  if (!Number.isFinite(angleDegrees)) {
    return 0;
  }
  const clamped = Math.min(Math.max(angleDegrees, -90), 90);
  return 1 + Math.floor(clamped * TERRAIN_HORIZON_UNORM16_SCALE + 32767 + 0.5);
}

/** Decodes a 16-bit horizon code to degrees; code 0 decodes to NaN. */
export function decodeGPUTerrainHorizonUnorm16(code: number): number {
  return code === 0 ? NaN : (code - 1) * TERRAIN_HORIZON_UNORM16_STEP_DEGREES - 90;
}

/**
 * Unpacks `elementCount` angles in degrees from packed `'unorm16'` words (element `e` lives in
 * word `e >> 1`, low half for even `e`). Invalid elements are NaN.
 */
export function unpackGPUTerrainHorizonUnorm16(
  words: ArrayLike<number>,
  elementCount: number
): Float64Array {
  const angles = new Float64Array(elementCount);
  for (let element = 0; element < elementCount; element++) {
    const word = words[element >> 1];
    angles[element] = decodeGPUTerrainHorizonUnorm16((word >>> ((element & 1) * 16)) & 0xffff);
  }
  return angles;
}

/** Smallest supported `GPUTerrainHorizonProps.directionCount`. */
export const GPU_TERRAIN_HORIZON_MIN_DIRECTION_COUNT = 4;

/** Largest supported `GPUTerrainHorizonProps.directionCount`. */
export const GPU_TERRAIN_HORIZON_MAX_DIRECTION_COUNT = 64;

/** CPU-side description packed by {@link getGPUTerrainHorizonParameterValues}. */
export type GPUTerrainHorizonSettings = {
  /** `[x, y]` cell size: meters (uniform), equatorial Web Mercator meters, or degrees (geographic). */
  cellSize: readonly [number, number];
  /** Elevation multiplier. Defaults to 1. */
  zFactor?: number;
  /**
   * Earth curvature and refraction drop in 1/meters: a sample at ground distance `d` is lowered by
   * `c * d^2`. See `getGPUTerrainCurvatureCoefficient`. Defaults to 0.
   */
  curvatureCoefficient?: number;
  /** Top edge of row 0: normalized Web Mercator y in `[0, 1]` or latitude degrees. Defaults to 0. */
  northEdge?: number;
  /** Bottom edge of the last row, same units as `northEdge`. Defaults to 0. */
  southEdge?: number;
  /** Ground search radius in meters; `<= 0` searches the full pixel radius. Defaults to 0. */
  maximumDistance?: number;
  /**
   * Compass azimuth of maximum weight for `anisotropicSkyViewFactor`, degrees clockwise from
   * north. Defaults to {@link GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES}. RVT's
   * `a_main_direction = A` equals `(360 - A) mod 360` here.
   */
  anisotropyAzimuthDegrees?: number;
  /**
   * Weight exponent of `anisotropicSkyViewFactor`, a finite number `>= 0` (RVT: low 4, high 8).
   * Defaults to {@link GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL}.
   */
  anisotropyLevel?: number;
  /**
   * Weight floor of `anisotropicSkyViewFactor` in `[0, 1]` (RVT: low 0.4, high 0.1). Defaults to
   * {@link GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT}.
   */
  anisotropyMinimumWeight?: number;
};

/**
 * Packs settings into the float layout read by {@link GPUTerrainHorizon}:
 * `[cellSizeX, cellSizeY, zFactor, curvatureCoefficient, northEdge, southEdge, maximumDistance, 0]`,
 * followed by `[anisotropyAzimuthDegrees, anisotropyLevel, anisotropyMinimumWeight, 0]` (12 values,
 * see {@link GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH}) when any anisotropy field is
 * given or `target` holds at least 12 values. Otherwise exactly 8 values are written and returned.
 *
 * @throws If a cell size is not finite and positive, an anisotropy field is out of range, or
 *   `target` is too short.
 */
export function getGPUTerrainHorizonParameterValues(
  settings: GPUTerrainHorizonSettings,
  target?: Float32Array
): Float32Array {
  const hasAnisotropy =
    settings.anisotropyAzimuthDegrees !== undefined ||
    settings.anisotropyLevel !== undefined ||
    settings.anisotropyMinimumWeight !== undefined;
  const wantsAnisotropy = hasAnisotropy || (target !== undefined && target.length >= 12);
  const output =
    target ??
    new Float32Array(
      wantsAnisotropy
        ? GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH
        : GPU_TERRAIN_HORIZON_PARAMETER_LENGTH
    );
  if (output.length < GPU_TERRAIN_HORIZON_PARAMETER_LENGTH) {
    throw new Error('Terrain horizon settings target must hold 8 values');
  }
  if (hasAnisotropy && output.length < GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH) {
    throw new Error('Terrain horizon settings target must hold 12 values with anisotropy');
  }
  if (!settings.cellSize.every(size => Number.isFinite(size) && size > 0)) {
    throw new Error('Terrain horizon cell size must be finite and positive');
  }
  const anisotropyAzimuth =
    settings.anisotropyAzimuthDegrees ?? GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES;
  const anisotropyLevel = settings.anisotropyLevel ?? GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL;
  const anisotropyMinimumWeight =
    settings.anisotropyMinimumWeight ?? GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT;
  if (wantsAnisotropy) {
    if (!Number.isFinite(anisotropyAzimuth)) {
      throw new Error('Terrain horizon anisotropy azimuth must be finite');
    }
    if (!Number.isFinite(anisotropyLevel) || anisotropyLevel < 0) {
      throw new Error('Terrain horizon anisotropy level must be finite and non-negative');
    }
    if (
      !Number.isFinite(anisotropyMinimumWeight) ||
      anisotropyMinimumWeight < 0 ||
      anisotropyMinimumWeight > 1
    ) {
      throw new Error('Terrain horizon anisotropy minimum weight must be in [0, 1]');
    }
  }
  output.set([
    settings.cellSize[0],
    settings.cellSize[1],
    settings.zFactor ?? 1,
    settings.curvatureCoefficient ?? 0,
    settings.northEdge ?? 0,
    settings.southEdge ?? 0,
    settings.maximumDistance ?? 0,
    0
  ]);
  if (wantsAnisotropy) {
    output.set([anisotropyAzimuth, anisotropyLevel, anisotropyMinimumWeight, 0], 8);
  }
  return output;
}

/**
 * Returns the ray-march sample distances in pixels used by {@link GPUTerrainHorizon}.
 *
 * `d[0] = 1` and `d[k + 1] = max(d[k] + 1, d[k] * stepGrowth)` while not above `maximumRadius`;
 * `maximumRadius` itself is appended when the sequence stops short of it. `stepGrowth = 1` gives
 * one-pixel steps; larger values give geometric steps that reach far horizons with few samples.
 * Values are rounded to float32 so CPU oracles match the baked WGSL constants.
 *
 * @throws If `maximumRadius` is not a positive integer or `stepGrowth` is not finite and `>= 1`.
 */
export function getGPUTerrainHorizonStepDistances(
  maximumRadius: number,
  stepGrowth: number = 1
): Float32Array {
  if (!Number.isSafeInteger(maximumRadius) || maximumRadius < 1) {
    throw new Error('Terrain horizon maximumRadius must be a positive integer');
  }
  if (!Number.isFinite(stepGrowth) || stepGrowth < 1) {
    throw new Error('Terrain horizon stepGrowth must be finite and >= 1');
  }
  const distances: number[] = [];
  for (let distance = 1; distance <= maximumRadius; ) {
    distances.push(distance);
    distance = Math.max(distance + 1, distance * stepGrowth);
  }
  if (distances[distances.length - 1] < maximumRadius) {
    distances.push(maximumRadius);
  }
  return Float32Array.from(distances);
}

/**
 * Returns the unit march direction in pixel space for one horizon sector.
 *
 * Sector `d` of `directionCount` has azimuth `d * 360 / directionCount` degrees clockwise from
 * north. Columns increase east; rows increase south for `rowDirection: 'south'`. Components are
 * rounded to float32, and round-off below 1e-9 snaps to 0.
 */
export function getGPUTerrainHorizonDirection(
  sector: number,
  directionCount: number,
  rowDirection: GPUTerrainRowDirection = 'south'
): [number, number] {
  const azimuth = (2 * Math.PI * sector) / directionCount;
  const northRowSign = rowDirection === 'south' ? -1 : 1;
  // Snap round-off (sin(PI) = 1.2e-16) to exact zero so axis rays stay on their pixel column/row.
  const snap = (value: number) => (Math.abs(value) < 1e-9 ? 0 : Math.fround(value));
  return [snap(Math.sin(azimuth)), snap(northRowSign * Math.cos(azimuth))];
}

/**
 * Properties for {@link GPUTerrainHorizon}.
 *
 * Topology: grid size, elevation format and calibration, `directionCount`, `maximumRadius`,
 * `stepGrowth`, `cellSizeMode`, `rowDirection`, `horizonFormat`, and which outputs exist.
 * Per-frame: `settings` (cell size, z factor, curvature, latitude band, maximum distance,
 * anisotropy) and elevation contents.
 *
 * Cell size model: `settings.cellSize` is projected meters (`'uniform'`), equatorial Web Mercator
 * meters or degrees with a latitude-dependent spacing per row (`'web-mercator'`, `'geographic'`).
 */
export type GPUTerrainHorizonProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-horizon'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /**
   * Per-frame settings with at least 8 float32 values (12 when `anisotropicSkyViewFactor` is
   * bound), see {@link getGPUTerrainHorizonParameterValues}.
   */
  settings: GraphDataView<'float32'>;
  /** Number of azimuth sectors, an integer in `[4, 64]`. Defaults to 16. */
  directionCount?: number;
  /** Search radius in pixels, a positive integer. Also the required tile halo. */
  maximumRadius: number;
  /** Geometric step growth, see {@link getGPUTerrainHorizonStepDistances}. Defaults to 1. */
  stepGrowth?: number;
  /**
   * Horizon search, see {@link GPUTerrainHorizonAlgorithm}. Defaults to `'march'`. `'sweep'` needs
   * `stepGrowth` 1 and grid extents up to 32767. The sweep is not universally faster: it has a
   * large fixed cost per pixel and sector, so it wins at large radii and loses at small ones.
   * Measured with 16 sectors on an Apple-silicon laptop: 1024^2 at the full radius 1023 sweep 41 ms
   * against march about 1300 ms, at radius 256 about 100 ms against 360 ms; 512^2 at radius 128
   * sweep 60 ms (125 ms with every output) against march about 20 ms. At 512^2 with every output
   * the sweep still loses at radius 256 (218 against 58 ms) and wins at the full radius 511 (36
   * against 109 ms). Prefer `'march'` for small radii and `'sweep'` near the full tile radius.
   */
  algorithm?: GPUTerrainHorizonAlgorithm;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPUTerrainCellSizeMode;
  /** Direction in which the row index increases. Defaults to `'south'` (north-up rasters). */
  rowDirection?: GPUTerrainRowDirection;
  /**
   * Storage of the `horizon` output. `'float32'` (default) stores degrees; `'unorm16'` packs two
   * 16-bit codes per `uint32` word (see {@link encodeGPUTerrainHorizonUnorm16}), a half-size map
   * with 0.0027 degree steps. Sky-view factor and openness sums always use the unquantised angle.
   */
  horizonFormat?: GPUTerrainHorizonFormat;
  /**
   * Optional horizon angles in degrees, pixel-major: element `pixel * directionCount + sector`.
   * `float32` format: `width * height * directionCount` values. `unorm16` format: a `uint32` view of
   * `ceil(width * height * directionCount / 2)` words.
   */
  horizon?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Optional sky-view factor per pixel in `[0, 1]`. */
  skyViewFactor?: GraphDataView<'float32'>;
  /** Optional positive openness per pixel in degrees. */
  positiveOpenness?: GraphDataView<'float32'>;
  /**
   * Optional negative openness per pixel in degrees: `90 - mean(h_nadir)`, where `h_nadir` is the
   * horizon of the inverted terrain (z factor negated; curvature still drops samples).
   */
  negativeOpenness?: GraphDataView<'float32'>;
  /**
   * Optional anisotropic sky-view factor per pixel in `[0, 1]` (Kokalj and Somrak): with sector
   * azimuth `t` and `w = (1 - wMin) * |cos((t - tMain) / 2)|^level + wMin`, it is
   * `1 - sum(w * sin(max(h, 0))) / sum(w)`. Requires 12 settings values.
   */
  anisotropicSkyViewFactor?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where the center elevation is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
  /** Optional storage texture (`r32float` or `rgba32float`, channel 0) receiving the sky-view factor. */
  skyViewFactorTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
};

/** Props of {@link getTerrainHorizonSectorOutput}. @internal */
export type TerrainHorizonSectorOutputProps = {
  /** Sector index in `[0, directionCount)`. */
  sector: number;
  /** Number of sectors. */
  directionCount: number;
  /** Storage format of `horizon`. Defaults to `'float32'`. */
  horizonFormat?: GPUTerrainHorizonFormat;
  /** Horizon map written in `'zenith'` mode (`read_write`, binding name `horizon`). */
  horizon?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Per-pixel sum of `sin(max(h, 0))` (binding `sineSum`), `'zenith'` mode. */
  sineSum?: GraphDataView<'float32'>;
  /** Per-pixel sum of `h` (binding `angleSum`), `'zenith'` mode. */
  angleSum?: GraphDataView<'float32'>;
  /**
   * Per-pixel weighted sum of `sin(max(h, 0))` (binding `anisotropicSum`), `'zenith'` mode. The
   * caller's kernel must bind `settings` (`array<f32>`, 12 values, slots 8 to 10 hold azimuth,
   * level and minimum weight) and declare {@link TERRAIN_ILLUMINATION_WGSL_CONSTANTS}.
   */
  anisotropicSum?: GraphDataView<'float32'>;
  /** Per-pixel sum of the nadir angle `h_nadir` (binding `nadirSum`), `'nadir'` mode only. */
  nadirSum?: GraphDataView<'float32'>;
  /**
   * `'zenith'` stores the horizon and accumulates the sine, angle and anisotropic sums. `'nadir'`
   * only accumulates `nadirSum`, so a caller running zenith and nadir as separate nodes emits one
   * helper per node.
   */
  mode: 'zenith' | 'nadir';
  /** Storage bindings the caller adds besides the outputs. Defaults to 3 (values, validity, settings). */
  reservedBindingCount?: number;
  /** Device storage-buffer limit per stage. Defaults to 8. */
  maximumBindingCount?: number;
};

/**
 * Generates the per-sector output of horizon kernels: horizon store (float32 or unorm16), the
 * sine, angle, anisotropic and nadir accumulations (sector 0 assigns, later sectors add). One
 * implementation serves the ray march and the sweep.
 *
 * `wgsl` consumes `pixel: u32` and `horizonAngle: f32` (degrees, NaN for an invalid center) and
 * is wrapped in its own block, so it may be emitted twice into one kernel (zenith and nadir) as
 * long as the nadir call sees the nadir angle as `horizonAngle` in an inner scope. `declarations`
 * are module-scope WGSL (empty unless `anisotropicSum` is bound or `horizonFormat` is
 * `'unorm16'`), `bindings` the `read_write` storage bindings to append after the caller's own.
 *
 * @throws If a mode-specific output is bound in the wrong mode, nothing is bound, or the kernel
 *   would exceed `maximumBindingCount` storage bindings.
 * @internal
 */
export function getTerrainHorizonSectorOutput(props: TerrainHorizonSectorOutputProps): {
  bindings: WGSLKernelBinding[];
  declarations: string;
  wgsl: string;
} {
  const {sector, directionCount, mode} = props;
  const format = props.horizonFormat ?? 'float32';
  if (mode === 'nadir') {
    if (props.horizon || props.sineSum || props.angleSum || props.anisotropicSum) {
      throw new Error('Terrain horizon nadir output only accumulates nadirSum');
    }
  } else if (props.nadirSum) {
    throw new Error('Terrain horizon zenith output cannot write nadirSum');
  }
  const bindings: WGSLKernelBinding[] = [];
  const statements: string[] = [];
  let declarations = '';
  const accumulate = sector === 0 ? '=' : '+=';
  if (props.horizon) {
    if (format === 'unorm16') {
      bindings.push({name: 'horizon', view: props.horizon, type: 'u32', access: 'read_write'});
      declarations += `fn encodeHorizonUnorm16(angle: f32) -> u32 {
  if (!isFiniteValue(angle)) { return 0u; }
  return 1u + u32(floor(clamp(angle, -90.0, 90.0) * ${getWGSLFloatLiteral(TERRAIN_HORIZON_UNORM16_SCALE)} + 32767.0 + 0.5));
}
`;
      // One sector per dispatch and DIRECTION_COUNT >= 4, so distinct pixels never share a word.
      statements.push(`let element = pixel * ${directionCount}u + ${sector}u;
  let wordIndex = horizonOffset + (element >> 1u);
  let shift = (element & 1u) * 16u;
  horizon[wordIndex] = (horizon[wordIndex] & ~(0xffffu << shift)) |
    (encodeHorizonUnorm16(horizonAngle) << shift);`);
    } else {
      bindings.push({name: 'horizon', view: props.horizon, type: 'f32', access: 'read_write'});
      statements.push(
        `horizon[horizonOffset + pixel * ${directionCount}u + ${sector}u] = horizonAngle;`
      );
    }
  }
  if (props.sineSum) {
    bindings.push({name: 'sineSum', view: props.sineSum, type: 'f32', access: 'read_write'});
    statements.push(
      `sineSum[sineSumOffset + pixel] ${accumulate} sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS);`
    );
  }
  if (props.angleSum) {
    bindings.push({name: 'angleSum', view: props.angleSum, type: 'f32', access: 'read_write'});
    statements.push(`angleSum[angleSumOffset + pixel] ${accumulate} horizonAngle;`);
  }
  if (props.anisotropicSum) {
    bindings.push({
      name: 'anisotropicSum',
      view: props.anisotropicSum,
      type: 'f32',
      access: 'read_write'
    });
    declarations += getTerrainHorizonAnisotropicWeightWGSL(directionCount);
    statements.push(
      `anisotropicSum[anisotropicSumOffset + pixel] ${accumulate} getAnisotropicWeight(${sector}u) *
    sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS);`
    );
  }
  if (props.nadirSum) {
    bindings.push({name: 'nadirSum', view: props.nadirSum, type: 'f32', access: 'read_write'});
    statements.push(`nadirSum[nadirSumOffset + pixel] ${accumulate} horizonAngle;`);
  }
  if (bindings.length === 0) {
    throw new Error('Terrain horizon sector output requires at least one output');
  }
  const total = bindings.length + (props.reservedBindingCount ?? 3);
  const limit = props.maximumBindingCount ?? 8;
  if (total > limit) {
    throw new Error(
      `Terrain horizon sector output needs ${total} storage bindings, more than the limit of ${limit}`
    );
  }
  return {bindings, declarations, wgsl: `{\n  ${statements.join('\n  ')}\n  }`};
}

/**
 * WGSL `fn getAnisotropicWeight(sector: u32) -> f32` for `directionCount` sectors, reading
 * azimuth, level and minimum weight from `settings[8..10]`. `pow(0, level)` is guarded.
 *
 * @internal
 */
function getTerrainHorizonAnisotropicWeightWGSL(directionCount: number): string {
  return `fn getAnisotropicWeight(sector: u32) -> f32 {
  let mainAzimuth = settings[settingsOffset + 8u];
  let level = settings[settingsOffset + 9u];
  let minimumWeight = settings[settingsOffset + 10u];
  let halfAngle = (f32(sector) * ${getWGSLFloatLiteral(360 / directionCount)} - mainAzimuth) * 0.5 * DEGREES_TO_RADIANS;
  let cosine = abs(cos(halfAngle));
  var power = 1.0;
  if (cosine <= 0.0) {
    power = select(0.0, 1.0, level == 0.0);
  } else {
    power = pow(cosine, level);
  }
  return (1.0 - minimumWeight) * power + minimumWeight;
}
`;
}

/**
 * Computes horizon-angle maps for `directionCount` azimuth sectors by bounded ray marching, plus
 * sky-view factor, openness and anisotropic sky-view factor from the same rays.
 *
 * For every valid pixel and sector, the ray walks the {@link getGPUTerrainHorizonStepDistances}
 * schedule, samples elevation bilinearly (a sample is skipped when any of its four corners is
 * invalid; the ray stops when it leaves the grid or `maximumDistance`), and keeps the largest
 * elevation angle `atan((zFactor * (z - z0) - c * d^2) / d)` in degrees. A ray with no valid sample
 * reports 0 (flat horizon). Ground distance uses the center row's cell size.
 *
 * - Sky-view factor (Zaksek et al. 2011): `1 - mean(sin(max(h, 0)))`.
 * - Positive openness (Yokoyama et al. 2002): `mean(90 - h)` degrees.
 * - Negative openness: `90 - mean(h_nadir)` with `h_nadir` the horizon of the inverted terrain
 *   (z factor negated, curvature still drops), tracked in the same march loop.
 * - Anisotropic sky-view factor (Kokalj and Somrak 2019, RVT `asvf`): sector weights
 *   `(1 - wMin) * |cos((t - tMain) / 2)|^level + wMin`, where `tMain` is the compass azimuth of
 *   maximum weight (RVT's `a_main_direction = A` is `(360 - A) mod 360` here).
 *
 * With `algorithm: 'sweep'` the per-sector march is replaced by an exact upper-hull sweep over
 * digital lines through pixel centres (one invocation per line, amortised O(1) per pixel and
 * sector, and a second nadir sweep when `negativeOpenness` is bound); every output keeps its
 * meaning. Measured on 1024^2 with 16 sectors: 98.6 ms against 411 ms at radius 256 and 40.6 ms
 * against 1474 ms at radius 1023 on an Apple-silicon laptop.
 *
 * One node per sector keeps each dispatch short on large tiles; sums accumulate in fixed sector
 * order, so results are deterministic. Invalid centers receive NaN and validity 0. Pixels closer
 * than `maximumRadius` to the tile edge see a truncated horizon: pass a tile with a
 * `maximumRadius` halo for seamless results (`GPURasterHaloStage` contract).
 */
export class GPUTerrainHorizon implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainHorizonProps;
  /** Number of azimuth sectors. */
  readonly directionCount: number;
  /** Receptive field in pixels (`GPURasterHaloStage` contract). */
  readonly requiredHalo: number;
  /** Baked ray-march distances in pixels. */
  readonly stepDistances: Float32Array;
  /** Storage format of the optional `horizon` output. */
  readonly horizonFormat: GPUTerrainHorizonFormat;
  /** Horizon search algorithm. */
  readonly algorithm: GPUTerrainHorizonAlgorithm;

  constructor(props: GPUTerrainHorizonProps) {
    this.id = props.id ?? 'terrain-horizon';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    const directionCount = props.directionCount ?? 16;
    if (
      !Number.isSafeInteger(directionCount) ||
      directionCount < GPU_TERRAIN_HORIZON_MIN_DIRECTION_COUNT ||
      directionCount > GPU_TERRAIN_HORIZON_MAX_DIRECTION_COUNT
    ) {
      throw new Error(`${id} directionCount must be an integer in [4, 64]`);
    }
    this.directionCount = directionCount;
    try {
      this.stepDistances = getGPUTerrainHorizonStepDistances(
        props.maximumRadius,
        props.stepGrowth ?? 1
      );
    } catch (error) {
      throw new Error(`${id}: ${(error as Error).message}`);
    }
    this.requiredHalo = props.maximumRadius;
    if (
      !props.horizon &&
      !props.skyViewFactor &&
      !props.positiveOpenness &&
      !props.negativeOpenness &&
      !props.anisotropicSkyViewFactor &&
      !props.validity &&
      !props.skyViewFactorTexture
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainHorizonFormat(id, props.horizonFormat ?? 'float32');
    this.horizonFormat = props.horizonFormat ?? 'float32';
    this.algorithm = props.algorithm ?? 'march';
    if (this.algorithm !== 'march' && this.algorithm !== 'sweep') {
      throw new Error(`${id} algorithm must be march or sweep`);
    }
    if (this.algorithm === 'sweep') {
      if ((props.stepGrowth ?? 1) !== 1) {
        throw new Error(`${id} stepGrowth applies to the march algorithm only`);
      }
      if (props.width > TERRAIN_SWEEP_MAX_EXTENT || props.height > TERRAIN_SWEEP_MAX_EXTENT) {
        throw new Error(`${id} sweep supports extents up to ${TERRAIN_SWEEP_MAX_EXTENT}`);
      }
    }
    validateTerrainHorizonView(
      id,
      'horizon',
      props.horizon,
      this.horizonFormat,
      pixelCount,
      directionCount
    );
    validateTerrainIlluminationView(
      id,
      'skyViewFactor',
      props.skyViewFactor,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'positiveOpenness',
      props.positiveOpenness,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'negativeOpenness',
      props.negativeOpenness,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'anisotropicSkyViewFactor',
      props.anisotropicSkyViewFactor,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(id, 'validity', props.validity, 'uint32', pixelCount);
    validateTerrainSettings(
      id,
      props.settings,
      props.anisotropicSkyViewFactor
        ? GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH
        : GPU_TERRAIN_HORIZON_PARAMETER_LENGTH
    );
    validateTerrainTexture(
      id,
      'skyViewFactorTexture',
      props.skyViewFactorTexture,
      ['r32float', 'rgba32float'],
      props.width,
      props.height
    );
    validateTerrainCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validateTerrainRowDirection(id, props.rowDirection ?? 'south');
    validateTerrainBuffersDistinct(
      id,
      [
        props.horizon,
        props.skyViewFactor,
        props.positiveOpenness,
        props.negativeOpenness,
        props.anisotropicSkyViewFactor,
        props.validity
      ],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
  }

  /** Returns elevation canonicalization, one march node per sector, finalize, and texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, directionCount} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, [props.skyViewFactorTexture]);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.horizon,
      props.skyViewFactor,
      props.positiveOpenness,
      props.negativeOpenness,
      props.anisotropicSkyViewFactor,
      props.validity
    ]);
    const pixelCount = width * height;
    if (props.horizon) {
      validateTerrainIlluminationBindingSize(
        id,
        'horizon',
        getTerrainHorizonStorageLength(this.horizonFormat, pixelCount, directionCount) * 4,
        graph.device.limits.maxStorageBufferBindingSize
      );
    }
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const elevationValues = source.band.storage.values as GraphDataView<'float32'>;
    const elevationValidity = source.band.validity as GraphDataView<'uint32'>;
    const skyViewTarget =
      props.skyViewFactor ??
      (props.skyViewFactorTexture
        ? createTransientView(graph, `${id}-sky-view-factor`, 'float32', pixelCount)
        : undefined);
    const needsSums = Boolean(skyViewTarget || props.positiveOpenness);
    const sineSum = needsSums
      ? createTransientView(graph, `${id}-sine-sum`, 'float32', pixelCount)
      : undefined;
    const angleSum = needsSums
      ? createTransientView(graph, `${id}-angle-sum`, 'float32', pixelCount)
      : undefined;
    const anisotropicSum = props.anisotropicSkyViewFactor
      ? createTransientView(graph, `${id}-anisotropic-sum`, 'float32', pixelCount)
      : undefined;
    const nadirSum = props.negativeOpenness
      ? createTransientView(graph, `${id}-nadir-sum`, 'float32', pixelCount)
      : undefined;
    const rowDirection = props.rowDirection ?? 'south';
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const distances = Array.from(this.stepDistances, getWGSLFloatLiteral).join(', ');
    const hull =
      this.algorithm === 'sweep'
        ? createTransientView(graph, `${id}-sweep-hull`, 'uint32', pixelCount)
        : undefined;
    for (let sector = 0; sector < directionCount; sector++) {
      const direction = getGPUTerrainHorizonDirection(sector, directionCount, rowDirection);
      if (hull) {
        nodes.push(
          ...getSweepSectorNodes(graph, {
            id: `${id}-horizon-${sector}`,
            width,
            height,
            sector,
            directionCount,
            direction,
            maximumRadius: props.maximumRadius,
            cellSizeMode,
            elevationValues,
            elevationValidity,
            settings: props.settings,
            hull,
            horizonFormat: this.horizonFormat,
            horizon: props.horizon,
            sineSum,
            angleSum,
            anisotropicSum,
            nadirSum
          })
        );
        continue;
      }
      nodes.push(
        getHorizonNode(graph, {
          id: `${id}-horizon-${sector}`,
          width,
          height,
          sector,
          directionCount,
          direction,
          distances,
          stepCount: this.stepDistances.length,
          cellSizeMode,
          elevationValues,
          elevationValidity,
          settings: props.settings,
          horizonFormat: this.horizonFormat,
          horizon: props.horizon,
          sineSum,
          angleSum,
          anisotropicSum,
          nadirSum
        })
      );
    }
    if (skyViewTarget || props.positiveOpenness || props.validity) {
      const bindings: WGSLKernelBinding[] = [
        {
          name: 'elevationValidity',
          view: elevationValidity,
          type: 'u32',
          access: 'read'
        }
      ];
      if (sineSum && angleSum) {
        bindings.push(
          {name: 'sineSum', view: sineSum, type: 'f32', access: 'read'},
          {name: 'angleSum', view: angleSum, type: 'f32', access: 'read'}
        );
      }
      if (skyViewTarget) {
        bindings.push({
          name: 'skyView',
          view: skyViewTarget,
          type: 'f32',
          access: 'read_write'
        });
      }
      if (props.positiveOpenness) {
        bindings.push({
          name: 'openness',
          view: props.positiveOpenness,
          type: 'f32',
          access: 'read_write'
        });
      }
      if (props.validity) {
        bindings.push({
          name: 'validityValues',
          view: props.validity,
          type: 'u32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-sky-view`,
          operation: 'GPUTerrainHorizon',
          variant: 'sky-view',
          bindings,
          invocationCount: pixelCount,
          declarations: `const DIRECTION_COUNT: f32 = ${directionCount}.0;
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}`,
          body: `let isValid = elevationValidity[elevationValidityOffset + index] != 0u${
            sineSum ? ' &&\n    isFiniteValue(angleSum[angleSumOffset + index])' : ''
          };
  let invalidValue = getNaN(index);
  ${
    skyViewTarget
      ? `skyView[skyViewOffset + index] = select(invalidValue,
    1.0 - sineSum[sineSumOffset + index] / DIRECTION_COUNT, isValid);`
      : ''
  }
  ${
    props.positiveOpenness
      ? `openness[opennessOffset + index] = select(invalidValue,
    90.0 - angleSum[angleSumOffset + index] / DIRECTION_COUNT, isValid);`
      : ''
  }
  ${props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
        })
      );
    }
    if (
      (anisotropicSum && props.anisotropicSkyViewFactor) ||
      (nadirSum && props.negativeOpenness)
    ) {
      // Kept apart from the sky-view kernel so no kernel binds more than 8 storage buffers.
      const bindings: WGSLKernelBinding[] = [
        {name: 'elevationValidity', view: elevationValidity, type: 'u32', access: 'read'}
      ];
      let validExpression = 'elevationValidity[elevationValidityOffset + index] != 0u';
      let statements = '';
      if (anisotropicSum && props.anisotropicSkyViewFactor) {
        bindings.push(
          {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
          {name: 'anisotropicSum', view: anisotropicSum, type: 'f32', access: 'read'},
          {
            name: 'anisotropicSkyView',
            view: props.anisotropicSkyViewFactor,
            type: 'f32',
            access: 'read_write'
          }
        );
        validExpression += ' &&\n    isFiniteValue(anisotropicSum[anisotropicSumOffset + index])';
        statements += `var weightSum = 0.0;
  for (var sector = 0u; sector < DIRECTION_COUNT_U32; sector++) {
    weightSum += getAnisotropicWeight(sector);
  }
  anisotropicSkyView[anisotropicSkyViewOffset + index] = select(invalidValue,
    1.0 - anisotropicSum[anisotropicSumOffset + index] / weightSum, isValid);
  `;
      }
      if (nadirSum && props.negativeOpenness) {
        bindings.push(
          {name: 'nadirSum', view: nadirSum, type: 'f32', access: 'read'},
          {
            name: 'negativeOpennessValues',
            view: props.negativeOpenness,
            type: 'f32',
            access: 'read_write'
          }
        );
        validExpression += ' &&\n    isFiniteValue(nadirSum[nadirSumOffset + index])';
        statements += `negativeOpennessValues[negativeOpennessValuesOffset + index] = select(invalidValue,
    90.0 - nadirSum[nadirSumOffset + index] / DIRECTION_COUNT, isValid);
  `;
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-extended`,
          operation: 'GPUTerrainHorizon',
          variant: 'extended',
          bindings,
          invocationCount: pixelCount,
          declarations: `const DIRECTION_COUNT: f32 = ${directionCount}.0;
const DIRECTION_COUNT_U32: u32 = ${directionCount}u;
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${anisotropicSum ? getTerrainHorizonAnisotropicWeightWGSL(directionCount) : ''}`,
          body: `let isValid = ${validExpression};
  let invalidValue = getNaN(index);
  ${statements}`
        })
      );
    }
    if (props.skyViewFactorTexture && skyViewTarget) {
      const texture = props.skyViewFactorTexture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-sky-view-texture`,
            input: {
              id: `${id}-sky-view-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: skyViewTarget}
            },
            output: texture,
            channel: 0
          }).addToGraph(graph)
        )
      );
    }
    return nodes;
  }
}

/** Builds the sweep kernels of one sector: zenith outputs, then the nadir pass when needed. */
function getSweepSectorNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    sector: number;
    directionCount: number;
    direction: [number, number];
    maximumRadius: number;
    cellSizeMode: GPUTerrainCellSizeMode;
    elevationValues: GraphDataView<'float32'>;
    elevationValidity: GraphDataView<'uint32'>;
    settings: GraphDataView<'float32'>;
    hull: GraphDataView<'uint32'>;
    horizonFormat: GPUTerrainHorizonFormat;
    horizon?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
    sineSum?: GraphDataView<'float32'>;
    angleSum?: GraphDataView<'float32'>;
    anisotropicSum?: GraphDataView<'float32'>;
    nadirSum?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const common = {
    width: props.width,
    height: props.height,
    direction: props.direction,
    maximumRadius: props.maximumRadius,
    cellSizeMode: props.cellSizeMode,
    elevationValues: props.elevationValues,
    elevationValidity: props.elevationValidity,
    settings: props.settings,
    hull: props.hull
  };
  const nodes: GPUCommandNode<Parameters>[] = [];
  // The sweep binds elevation values, validity, settings, and hull pointers itself.
  if (props.horizon || props.sineSum || props.angleSum || props.anisotropicSum) {
    const output = getTerrainHorizonSectorOutput({
      sector: props.sector,
      directionCount: props.directionCount,
      horizonFormat: props.horizonFormat,
      horizon: props.horizon,
      sineSum: props.sineSum,
      angleSum: props.angleSum,
      anisotropicSum: props.anisotropicSum,
      mode: 'zenith',
      reservedBindingCount: 4
    });
    nodes.push(
      getTerrainHorizonSweepNode(graph, {
        ...common,
        id: props.id,
        zFactorSign: 1,
        outputBindings: output.bindings,
        outputDeclarations: output.declarations,
        outputWGSL: output.wgsl
      })
    );
  }
  if (props.nadirSum) {
    const output = getTerrainHorizonSectorOutput({
      sector: props.sector,
      directionCount: props.directionCount,
      nadirSum: props.nadirSum,
      mode: 'nadir',
      reservedBindingCount: 4
    });
    nodes.push(
      getTerrainHorizonSweepNode(graph, {
        ...common,
        id: `${props.id}-nadir`,
        zFactorSign: -1,
        outputBindings: output.bindings,
        outputDeclarations: output.declarations,
        outputWGSL: output.wgsl
      })
    );
  }
  return nodes;
}

/** Builds the ray-march kernel for one horizon sector. */
function getHorizonNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    sector: number;
    directionCount: number;
    direction: [number, number];
    distances: string;
    stepCount: number;
    cellSizeMode: GPUTerrainCellSizeMode;
    elevationValues: GraphDataView<'float32'>;
    elevationValidity: GraphDataView<'uint32'>;
    settings: GraphDataView<'float32'>;
    horizonFormat: GPUTerrainHorizonFormat;
    horizon?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
    sineSum?: GraphDataView<'float32'>;
    angleSum?: GraphDataView<'float32'>;
    anisotropicSum?: GraphDataView<'float32'>;
    nadirSum?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {
      name: 'elevationValues',
      view: props.elevationValues,
      type: 'f32',
      access: 'read'
    },
    {
      name: 'elevationValidity',
      view: props.elevationValidity,
      type: 'u32',
      access: 'read'
    },
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
  ];
  const zenithOutput =
    props.horizon || props.sineSum || props.angleSum || props.anisotropicSum
      ? getTerrainHorizonSectorOutput({
          sector: props.sector,
          directionCount: props.directionCount,
          horizonFormat: props.horizonFormat,
          horizon: props.horizon,
          sineSum: props.sineSum,
          angleSum: props.angleSum,
          anisotropicSum: props.anisotropicSum,
          mode: 'zenith',
          reservedBindingCount: 3 + (props.nadirSum ? 1 : 0)
        })
      : undefined;
  const nadirOutput = props.nadirSum
    ? getTerrainHorizonSectorOutput({
        sector: props.sector,
        directionCount: props.directionCount,
        nadirSum: props.nadirSum,
        mode: 'nadir',
        reservedBindingCount: 3 + (zenithOutput?.bindings.length ?? 0)
      })
    : undefined;
  bindings.push(...(zenithOutput?.bindings ?? []), ...(nadirOutput?.bindings ?? []));
  const nadir = Boolean(nadirOutput);
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTerrainHorizon',
    variant: `march-${props.cellSizeMode}${nadir ? '-nadir' : ''}`,
    bindings,
    invocationCount: props.width * props.height,
    declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const SECTOR: u32 = ${props.sector}u;
const DIRECTION_COUNT: u32 = ${props.directionCount}u;
const DIRECTION: vec2<f32> = vec2<f32>(${getWGSLFloatLiteral(props.direction[0])}, ${getWGSLFloatLiteral(props.direction[1])});
const STEP_COUNT: u32 = ${props.stepCount}u;
var<private> STEP_DISTANCES: array<f32, ${props.stepCount}> = array<f32, ${props.stepCount}>(${props.distances});
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${zenithOutput?.declarations ?? ''}
${getTerrainGroundCellSizeWGSL(props.cellSizeMode, {
  cellSizeIndex: 0,
  northEdgeIndex: 4
})}
fn isValidPixel(column: u32, row: u32) -> bool {
  return elevationValidity[elevationValidityOffset + row * WIDTH + column] != 0u;
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear elevation relative to \`origin\` at an in-grid pixel-center position; .y is 0 when any
// corner is invalid. Corners are made origin-relative before interpolating, so the interpolation
// rounds at the scale of the local relief instead of the absolute elevation (at 4000 m the absolute
// form loses about 2.4e-4 m, 1.4e-3 degrees over one 10 m step).
fn sampleElevation(position: vec2<f32>, origin: f32) -> vec2<f32> {
  let base = vec2<u32>(floor(position));
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  let fraction = position - floor(position);
  if (!isValidPixel(base.x, base.y) || !isValidPixel(next.x, base.y) ||
      !isValidPixel(base.x, next.y) || !isValidPixel(next.x, next.y)) {
    return vec2<f32>(0.0, 0.0);
  }
  let top = mix(getElevation(base.x, base.y) - origin, getElevation(next.x, base.y) - origin, fraction.x);
  let bottom = mix(getElevation(base.x, next.y) - origin, getElevation(next.x, next.y) - origin, fraction.x);
  return vec2<f32>(mix(top, bottom, fraction.y), 1.0);
}`,
    body: `let pixel = index;
  let column = index % WIDTH;
  let row = index / WIDTH;
  let zFactor = settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  let maximumDistance = settings[settingsOffset + 6u];
  let groundCell = getGroundCellSize(row);
  let groundStep = length(DIRECTION * groundCell);
  var horizonAngle = getNaN(index);
  var nadirAngle = getNaN(index);
  if (isValidPixel(column, row) && groundStep > 0.0 && isFiniteValue(groundStep)) {
    let center = vec2<f32>(f32(column), f32(row));
    let centerElevation = getElevation(column, row);
    let limit = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
    var hasSample = false;
    var maximumTangent = 0.0;
    var maximumNadirTangent = 0.0;
    for (var stepIndex = 0u; stepIndex < STEP_COUNT; stepIndex++) {
      let stepDistance = STEP_DISTANCES[stepIndex];
      let position = center + DIRECTION * stepDistance;
      if (any(position < vec2<f32>(0.0)) || any(position > limit)) { break; }
      let groundDistance = stepDistance * groundStep;
      if (maximumDistance > 0.0 && groundDistance > maximumDistance) { break; }
      let elevationSample = sampleElevation(position, centerElevation);
      if (elevationSample.y == 0.0) { continue; }
      let rise = zFactor * elevationSample.x -
        curvature * groundDistance * groundDistance;
      let tangent = rise / groundDistance;
      if (!hasSample || tangent > maximumTangent) { maximumTangent = tangent; }
      ${
        nadir
          ? `// Inverted terrain: the elevation difference flips sign, the curvature drop does not.
      let nadirRise = -zFactor * elevationSample.x -
        curvature * groundDistance * groundDistance;
      let nadirTangent = nadirRise / groundDistance;
      if (!hasSample || nadirTangent > maximumNadirTangent) { maximumNadirTangent = nadirTangent; }`
          : ''
      }
      hasSample = true;
    }
    horizonAngle = select(0.0, atan(maximumTangent) * RADIANS_TO_DEGREES, hasSample);
    ${nadir ? 'nadirAngle = select(0.0, atan(maximumNadirTangent) * RADIANS_TO_DEGREES, hasSample);' : ''}
  }
  ${zenithOutput?.wgsl ?? ''}
  ${nadirOutput ? `{\n  let horizonAngle = nadirAngle;\n  ${nadirOutput.wgsl}\n  }` : ''}`
  });
}
