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
import {
  GPURasterBufferToTexture,
  type GPURasterBand,
  type GPURasterMetadata
} from '../../gpu-raster/index';
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
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateParameterTarget,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters,
  validateRasterOutputTexture
} from '../../gpu-raster/raster-band-utils';
import {
  getTerrainHorizonStorageLength,
  TERRAIN_HORIZON_UNORM16_SCALE,
  TERRAIN_HORIZON_UNORM16_STEP_DEGREES,
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainHorizonFormat,
  validateTerrainHorizonView,
  validateTerrainIlluminationBindingSize,
  validateTerrainIlluminationCellSize,
  validateTerrainIlluminationView,
  type GPUTerrainHorizonFormat
} from './terrain-illumination-utils';
import {
  getRasterGroundCellSizeWGSL,
  RASTER_PIXEL_DIRECTION_WGSL,
  type GPURasterCellSizeMode,
  type GPURasterRowDirection,
  validateRasterCellSizeMode,
  validateRasterRowDirection,
  writeRasterGridParameterValues
} from '../../analysis-core/raster-grid';
import {
  createTerrainHorizonSweepNode,
  createTerrainSweepSectorTable,
  TERRAIN_SWEEP_GRID_SLOTS,
  TERRAIN_SWEEP_MAXIMUM_EXTENT,
  TERRAIN_SWEEP_SECTOR_WGSL
} from './terrain-horizon-sweep';

export type {GPUTerrainHorizonFormat} from './terrain-illumination-utils';

/**
 * How {@link GPUTerrainHorizon} finds horizons:
 * - `'march'`: bounded ray march with bilinear samples on the baked step schedule, O(steps) per
 *   pixel and sector. Each ray is a straight pixel line along the sector's ground azimuth at the
 *   pixel's own row, and sample distances are ground meters at that row's cell size. The cell
 *   size is not re-evaluated along the ray: at 60 N and 30 km on a 45 degree sector the samples
 *   lie at a true bearing of 44.75 degrees and 0.14% closer than assumed; re-scaling per row would
 *   only reach 44.83 degrees (a rhumb line, which is not the geodesic either).
 * - `'sweep'`: exact discrete horizons on digital lines through pixel centres with upper-hull
 *   pointers (Stewart 1998 style), amortised O(1) per pixel and sector. Samples are pixel centres
 *   on a digital line whose lateral offset from the ideal ray is at most half a pixel, so values
 *   differ from the march by sampling, not by precision. `stepGrowth` must be 1. One digital line
 *   family has one pixel slope, so on anisotropic cells it follows each sector's ground azimuth
 *   at the reference row (`height / 2`); other rows of a `'geographic'` tile see the azimuth
 *   rotated by the change of cos(latitude) across the tile, worst at 45 degree sectors: at 60 N,
 *   0.11 degrees at the edge rows of a 0.25 degree tall tile, 0.44 degrees for 1 degree, 2.3
 *   degrees for 5 degrees (about `tan(lat) * halfSpan / 2` radians). The march converts the azimuth
 *   at each pixel's own row and has no such term.
 */
export type GPUTerrainHorizonAlgorithm = 'march' | 'sweep';

/** Number of float32 values read from `GPUTerrainHorizonProps.parameters`. */
export const GPU_TERRAIN_HORIZON_PARAMETER_LENGTH = 8;

/**
 * Number of float32 values read from `GPUTerrainHorizonProps.parameters` when
 * `anisotropicSkyViewFactor` is bound: the 8 base values plus
 * `[anisotropyAzimuthDegrees, anisotropyLevel, anisotropyMinimumWeight, 0]`.
 */
export const GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH = 12;

/**
 * Default direction of maximum anisotropic sky-view weight in RVT's convention: RVT's default
 * `asvf_dir = 315`. RVT direction angles run counterclockwise from north (90 is west, 270 east),
 * so 315 is compass azimuth 45 (north-east). See
 * {@link GPUTerrainHorizonSettings.anisotropyAzimuthDegrees}.
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
export const GPU_TERRAIN_HORIZON_MINIMUM_DIRECTION_COUNT = 4;

/** Largest supported `GPUTerrainHorizonProps.directionCount`. */
export const GPU_TERRAIN_HORIZON_MAXIMUM_DIRECTION_COUNT = 64;

/** CPU-side description packed by {@link getGPUTerrainHorizonParameterValues}. */
export type GPUTerrainHorizonSettings = {
  /** Elevation multiplier. Defaults to 1. */
  zFactor?: number;
  /**
   * Earth curvature and refraction drop in 1/meters: a sample at ground distance `d` is lowered by
   * `c * d^2`. See `getGPUTerrainCurvatureCoefficient`. Defaults to 0.
   */
  curvatureCoefficient?: number;
  /** Ground search radius in meters; `<= 0` searches the full pixel radius. Defaults to 0. */
  maximumDistance?: number;
  /**
   * Direction of maximum weight for `anisotropicSkyViewFactor`, in RVT's convention: the same
   * number as RVT's `asvf_dir` / `a_main_direction` means the same direction. RVT direction angles
   * run counterclockwise from north (0 north, 90 west, 180 south, 270 east), so `A` is compass
   * azimuth `(360 - A) mod 360`. Defaults to
   * {@link GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES} (315, RVT's default, which is
   * compass north-east). This is the only counterclockwise angle in gpu-terrain: sector azimuths,
   * sun azimuths and aspect are compass azimuths, clockwise from north.
   */
  anisotropyAzimuthDegrees?: number;
  /**
   * Weight exponent of `anisotropicSkyViewFactor`: a non-negative integer, RVT's `a_poly_level`.
   * RVT's presets are low (level 4, minimum weight 0.4) and high (level 8, minimum weight 0.1).
   * The power is signed as in RVT, `cos((a - A) / 2)^level`, so an odd level gives a weight below
   * the minimum weight on the far side of the main direction, exactly like RVT. Defaults to
   * {@link GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL}.
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
 * `[cellSizeX, cellSizeY, zFactor, curvatureCoefficient, northEdge, southEdge, maximumDistance, 0]`
 * (cell size and row-edge world y taken from `grid`), followed by `[anisotropyAzimuthDegrees, anisotropyLevel, anisotropyMinimumWeight, 0]` (12 values,
 * see {@link GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH}) when any anisotropy field is
 * given or `target` holds at least 12 values. Otherwise exactly 8 values are written and returned.
 *
 * @throws If the grid cell size is not finite and positive, an anisotropy field is out of range, or
 *   `target` is too short.
 */
export function getGPUTerrainHorizonParameterValues(
  settings: GPUTerrainHorizonSettings,
  grid: GPURasterMetadata,
  target?: Float32Array
): Float32Array {
  const hasAnisotropy =
    settings.anisotropyAzimuthDegrees !== undefined ||
    settings.anisotropyLevel !== undefined ||
    settings.anisotropyMinimumWeight !== undefined;
  const wantsAnisotropy = hasAnisotropy || (target !== undefined && target.length >= 12);
  const values =
    target ??
    new Float32Array(
      wantsAnisotropy
        ? GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH
        : GPU_TERRAIN_HORIZON_PARAMETER_LENGTH
    );
  validateTerrainIlluminationCellSize('getGPUTerrainHorizonParameterValues', grid);
  validateParameterTarget(
    'getGPUTerrainHorizonParameterValues',
    values,
    hasAnisotropy
      ? GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH
      : GPU_TERRAIN_HORIZON_PARAMETER_LENGTH
  );
  const anisotropyAzimuth =
    settings.anisotropyAzimuthDegrees ?? GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES;
  const anisotropyLevel = settings.anisotropyLevel ?? GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_LEVEL;
  const anisotropyMinimumWeight =
    settings.anisotropyMinimumWeight ?? GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_MINIMUM_WEIGHT;
  if (wantsAnisotropy) {
    if (!Number.isFinite(anisotropyAzimuth)) {
      throw new Error('Terrain horizon anisotropy azimuth must be finite');
    }
    if (!Number.isSafeInteger(anisotropyLevel) || anisotropyLevel < 0) {
      throw new Error('Terrain horizon anisotropy level must be a non-negative integer');
    }
    if (
      !Number.isFinite(anisotropyMinimumWeight) ||
      anisotropyMinimumWeight < 0 ||
      anisotropyMinimumWeight > 1
    ) {
      throw new Error('Terrain horizon anisotropy minimum weight must be in [0, 1]');
    }
  }
  values.set([
    0,
    0,
    settings.zFactor ?? 1,
    settings.curvatureCoefficient ?? 0,
    0,
    0,
    settings.maximumDistance ?? 0,
    0
  ]);
  writeRasterGridParameterValues(values, grid, TERRAIN_SWEEP_GRID_SLOTS);
  if (wantsAnisotropy) {
    values.set([anisotropyAzimuth, anisotropyLevel, anisotropyMinimumWeight, 0], 8);
  }
  return values;
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
 * Returns the unit ground-space direction `(east, row-ward)` of one horizon sector, which is also
 * the pixel-space march direction on square ground cells.
 *
 * Sector `d` of `directionCount` has azimuth `d * 360 / directionCount` degrees clockwise from
 * north. Columns increase east; rows increase south for `rowDirection: 'south'`. Components are
 * rounded to float32, and round-off below 1e-9 snaps to 0. The march kernels turn it into a pixel
 * step per row with the row's ground cell size (`getRasterPixelDirection`), so sector azimuths are
 * ground azimuths on anisotropic (for example geographic) grids too.
 */
export function getGPUTerrainHorizonDirection(
  sector: number,
  directionCount: number,
  rowDirection: GPURasterRowDirection = 'south'
): [number, number] {
  const azimuth = (2 * Math.PI * sector) / directionCount;
  const northRowSign = rowDirection === 'south' ? -1 : 1;
  // Snap round-off (sin(PI) = 1.2e-16) to exact zero so axis rays stay on their pixel column/row.
  const snap = (value: number) => (Math.abs(value) < 1e-9 ? 0 : Math.fround(value));
  return [snap(Math.sin(azimuth)), snap(northRowSign * Math.cos(azimuth))];
}

/**
 * Caller-owned outputs of {@link GPUTerrainHorizon}; at least one is required.
 */
export type GPUTerrainHorizonOutput = {
  /**
   * Horizon angles in degrees, pixel-major: element `pixel * directionCount + sector`.
   * `float32` format: `width * height * directionCount` values. `unorm16` format: a `uint32` view of
   * `ceil(width * height * directionCount / 2)` words.
   */
  horizon?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Sky-view factor per pixel in `[0, 1]`. */
  skyViewFactor?: GraphDataView<'float32'>;
  /** Positive openness per pixel in degrees. */
  positiveOpenness?: GraphDataView<'float32'>;
  /**
   * Negative openness per pixel in degrees: `90 - mean(h_nadir)`, where `h_nadir` is the
   * horizon of the inverted terrain (z factor negated; curvature still drops samples).
   */
  negativeOpenness?: GraphDataView<'float32'>;
  /**
   * Anisotropic sky-view factor per pixel (Kokalj and Somrak 2019, RVT `asvf`): with the
   * sector's RVT direction angle `a = (360 - t) mod 360` for compass azimuth `t`, main direction
   * `A` (RVT convention) and `w = (1 - wMin) * cos((a - A) / 2)^level + wMin` (signed power, as in
   * RVT), it is `1 - sum(w * sin(max(h, 0))) / sum(w)`. Requires 12 parameter values.
   */
  anisotropicSkyViewFactor?: GraphDataView<'float32'>;
  /** Per-pixel 1 where the center elevation is valid, else 0. */
  outputValidity?: GraphDataView<'uint32'>;
  /** Storage texture (`r32float` or `rgba32float`, channel 0) receiving the sky-view factor. */
  skyViewFactorTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
};

/**
 * Properties for {@link GPUTerrainHorizon}.
 *
 * Topology: grid size, elevation format and calibration, `directionCount`, `maximumRadius`,
 * `stepGrowth`, `cellSizeMode`, `rowDirection`, `horizonFormat`, and which outputs exist.
 * Per-frame: `parameters` (grid cell size and latitude band, z factor, curvature, maximum
 * distance, anisotropy) and elevation contents.
 *
 * Cell size model: the grid's cell size is projected meters (`'uniform'`), equatorial Web Mercator
 * meters or degrees with a latitude-dependent spacing per row (`'web-mercator'`, `'geographic'`).
 */
export type GPUTerrainHorizonProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-horizon'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /**
   * Per-frame parameters with at least 8 float32 values (12 when `output.anisotropicSkyViewFactor`
   * is bound), see {@link getGPUTerrainHorizonParameterValues}.
   */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUTerrainHorizonOutput;
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
  /**
   * Upper bound on `pixels * steps * sectors` one `'march'` dispatch may cost; the sectors are
   * split over as many dispatches as needed (at least one sector each) so a dispatch stays short
   * on large tiles. Results do not depend on it. Defaults to 2^30.
   */
  maximumStepsPerDispatch?: number;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Direction in which the row index increases. Defaults to `'south'` (north-up rasters). */
  rowDirection?: GPURasterRowDirection;
  /**
   * Storage of `output.horizon`. `'float32'` (default) stores degrees; `'unorm16'` packs two
   * 16-bit codes per `uint32` word (see {@link encodeGPUTerrainHorizonUnorm16}), a half-size map
   * with 0.0027 degree steps. Sky-view factor and openness sums always use the unquantised angle.
   */
  horizonFormat?: GPUTerrainHorizonFormat;
};

/** Props of {@link getTerrainHorizonSectorOutput}. @internal */
export type TerrainHorizonSectorOutputProps = {
  /** Sector index in `[0, directionCount)`; ignored when {@link sectorWGSL} is set. */
  sector: number;
  /**
   * WGSL `u32` expression of the sector, read at run time (library rule 12: the sector index then
   * never reaches the WGSL text). Sector 0 still assigns the sums and later sectors add.
   */
  sectorWGSL?: string;
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
   * caller's kernel must bind `settings` (`array<f32>`, 12 values, slots 8 to 10 hold the RVT main direction,
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
  const sectorExpression = props.sectorWGSL ? `(${props.sectorWGSL})` : `${sector}u`;
  // Sector 0 assigns and later sectors add: `select` keeps exactly the bits of either form.
  const accumulate = (target: string, term: string): string =>
    props.sectorWGSL
      ? `${target} = select(${target} + ${term}, ${term}, ${sectorExpression} == 0u);`
      : `${target} ${sector === 0 ? '=' : '+='} ${term};`;
  if (props.horizon) {
    if (format === 'unorm16') {
      bindings.push({name: 'horizon', view: props.horizon, type: 'u32', access: 'read_write'});
      declarations += `fn encodeHorizonUnorm16(angle: f32) -> u32 {
  if (!isFiniteValue(angle)) { return 0u; }
  return 1u + u32(floor(clamp(angle, -90.0, 90.0) * ${getWGSLFloatLiteral(TERRAIN_HORIZON_UNORM16_SCALE)} + 32767.0 + 0.5));
}
`;
      // One sector per dispatch and DIRECTION_COUNT >= 4, so distinct pixels never share a word.
      statements.push(`let element = pixel * ${directionCount}u + ${sectorExpression};
  let wordIndex = horizonOffset + (element >> 1u);
  let shift = (element & 1u) * 16u;
  horizon[wordIndex] = (horizon[wordIndex] & ~(0xffffu << shift)) |
    (encodeHorizonUnorm16(horizonAngle) << shift);`);
    } else {
      bindings.push({name: 'horizon', view: props.horizon, type: 'f32', access: 'read_write'});
      statements.push(
        `horizon[horizonOffset + pixel * ${directionCount}u + ${sectorExpression}] = horizonAngle;`
      );
    }
  }
  if (props.sineSum) {
    bindings.push({name: 'sineSum', view: props.sineSum, type: 'f32', access: 'read_write'});
    statements.push(
      accumulate(
        'sineSum[sineSumOffset + pixel]',
        'sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS)'
      )
    );
  }
  if (props.angleSum) {
    bindings.push({name: 'angleSum', view: props.angleSum, type: 'f32', access: 'read_write'});
    statements.push(accumulate('angleSum[angleSumOffset + pixel]', 'horizonAngle'));
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
      accumulate(
        'anisotropicSum[anisotropicSumOffset + pixel]',
        `getAnisotropicWeight(${sectorExpression}) *
    sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS)`
      )
    );
  }
  if (props.nadirSum) {
    bindings.push({name: 'nadirSum', view: props.nadirSum, type: 'f32', access: 'read_write'});
    statements.push(accumulate('nadirSum[nadirSumOffset + pixel]', 'horizonAngle'));
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
 * WGSL `fn getAnisotropicWeight(sector: u32) -> f32` for `directionCount` sectors, reading the
 * RVT main direction, level and minimum weight from `settings[8..10]`.
 *
 * RVT parity (`rvt.vis.sky_view_factor_compute`): RVT direction `i` has angle `i * 360 / n`
 * counterclockwise from north, so compass sector `d` is RVT direction `(n - d) mod n`, and the
 * weight is `(1 - wMin) * cos((a - A) / 2)^level + wMin` with the signed power of an integer level
 * (`0^0 = 1`, as numpy).
 *
 * @internal
 */
function getTerrainHorizonAnisotropicWeightWGSL(directionCount: number): string {
  return `fn getAnisotropicWeight(sector: u32) -> f32 {
  let mainDirection = settings[settingsOffset + 8u];
  let level = settings[settingsOffset + 9u];
  let minimumWeight = settings[settingsOffset + 10u];
  // Compass sector d is RVT direction (n - d) mod n; RVT angles run counterclockwise from north.
  let rvtDirection = f32((${directionCount}u - sector) % ${directionCount}u) * ${getWGSLFloatLiteral(360 / directionCount)};
  let cosine = cos((rvtDirection - mainDirection) * 0.5 * DEGREES_TO_RADIANS);
  var power = 1.0;
  if (level > 0.0) {
    // pow is undefined for a zero or negative base; odd integer levels keep the sign like RVT.
    power = select(0.0, pow(abs(cosine), level), abs(cosine) > 0.0);
    if (cosine < 0.0 && fract(level * 0.5) != 0.0) { power = -power; }
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
 * - Anisotropic sky-view factor (Kokalj and Somrak 2019, RVT `asvf`), with RVT's weights and
 *   direction convention: sector weights `(1 - wMin) * cos((a - A) / 2)^level + wMin`, where `a` is
 *   the sector's RVT direction angle (counterclockwise from north, `(360 - t) mod 360` for compass
 *   azimuth `t`), `A` is `anisotropyAzimuthDegrees` exactly as RVT's `asvf_dir`, and the power is
 *   signed with an integer level.
 *
 * With `algorithm: 'sweep'` the per-sector march is replaced by an exact upper-hull sweep over
 * digital lines through pixel centres (one invocation per line, amortised O(1) per pixel and
 * sector, and a second nadir sweep when `negativeOpenness` is bound); every output keeps its
 * meaning. Measured on 1024^2 with 16 sectors: 98.6 ms against 411 ms at radius 256 and 40.6 ms
 * against 1474 ms at radius 1023 on an Apple-silicon laptop.
 *
 * The march runs all sectors of a pixel in one invocation, in chunks of sectors sized so one dispatch
 * stays under about 2^30 sample steps; sums accumulate in fixed sector order, so results are
 * deterministic and identical to one dispatch per sector. Invalid centers receive NaN and validity 0. Pixels closer
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
  /** Storage format of the optional `output.horizon`. */
  readonly horizonFormat: GPUTerrainHorizonFormat;
  /** Horizon search algorithm. */
  readonly algorithm: GPUTerrainHorizonAlgorithm;

  constructor(props: GPUTerrainHorizonProps) {
    this.id = props.id ?? 'terrain-horizon';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const pixelCount = validateRasterGrid(id, props.grid);
    const directionCount = props.directionCount ?? 16;
    if (
      !Number.isSafeInteger(directionCount) ||
      directionCount < GPU_TERRAIN_HORIZON_MINIMUM_DIRECTION_COUNT ||
      directionCount > GPU_TERRAIN_HORIZON_MAXIMUM_DIRECTION_COUNT
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
      !output.horizon &&
      !output.skyViewFactor &&
      !output.positiveOpenness &&
      !output.negativeOpenness &&
      !output.anisotropicSkyViewFactor &&
      !output.outputValidity &&
      !output.skyViewFactorTexture
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
      if (
        props.grid.width > TERRAIN_SWEEP_MAXIMUM_EXTENT ||
        props.grid.height > TERRAIN_SWEEP_MAXIMUM_EXTENT
      ) {
        throw new Error(`${id} sweep supports extents up to ${TERRAIN_SWEEP_MAXIMUM_EXTENT}`);
      }
    }
    if (
      props.maximumStepsPerDispatch !== undefined &&
      !(Number.isFinite(props.maximumStepsPerDispatch) && props.maximumStepsPerDispatch > 0)
    ) {
      throw new Error(`${id} maximumStepsPerDispatch must be positive`);
    }
    validateTerrainHorizonView(
      id,
      'output.horizon',
      output.horizon,
      this.horizonFormat,
      pixelCount,
      directionCount
    );
    validateTerrainIlluminationView(
      id,
      'output.skyViewFactor',
      output.skyViewFactor,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'output.positiveOpenness',
      output.positiveOpenness,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'output.negativeOpenness',
      output.negativeOpenness,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'output.anisotropicSkyViewFactor',
      output.anisotropicSkyViewFactor,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(
      id,
      'output.outputValidity',
      output.outputValidity,
      'uint32',
      pixelCount
    );
    validateRasterParameters(
      id,
      props.parameters,
      output.anisotropicSkyViewFactor
        ? GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH
        : GPU_TERRAIN_HORIZON_PARAMETER_LENGTH
    );
    validateRasterOutputTexture(
      id,
      'output.skyViewFactorTexture',
      output.skyViewFactorTexture,
      ['r32float', 'rgba32float'],
      props.grid
    );
    validateRasterCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    validateRasterRowDirection(id, props.rowDirection ?? 'south');
    validateRasterBuffersDistinct(
      id,
      [
        output.horizon,
        output.skyViewFactor,
        output.positiveOpenness,
        output.negativeOpenness,
        output.anisotropicSkyViewFactor,
        output.outputValidity
      ],
      [...getRasterBandViews(props.elevation), props.parameters]
    );
  }

  /** Returns elevation canonicalization, the fused march (or one sweep node per sector), finalize, and texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, directionCount} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, [output.skyViewFactorTexture]);
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      output.horizon,
      output.skyViewFactor,
      output.positiveOpenness,
      output.negativeOpenness,
      output.anisotropicSkyViewFactor,
      output.outputValidity
    ]);
    const pixelCount = width * height;
    if (output.horizon) {
      validateTerrainIlluminationBindingSize(
        id,
        'horizon',
        getTerrainHorizonStorageLength(this.horizonFormat, pixelCount, directionCount) * 4,
        graph.device.limits.maxStorageBufferBindingSize
      );
    }
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const elevationValues = source.band.storage.values as GraphDataView<'float32'>;
    const elevationValidity = source.band.validity as GraphDataView<'uint32'>;
    const skyViewTarget =
      output.skyViewFactor ??
      (output.skyViewFactorTexture
        ? createTransientView(graph, `${id}-sky-view-factor`, 'float32', pixelCount)
        : undefined);
    const needsSums = Boolean(skyViewTarget || output.positiveOpenness);
    const sineSum = needsSums
      ? createTransientView(graph, `${id}-sine-sum`, 'float32', pixelCount)
      : undefined;
    const angleSum = needsSums
      ? createTransientView(graph, `${id}-angle-sum`, 'float32', pixelCount)
      : undefined;
    const anisotropicSum = output.anisotropicSkyViewFactor
      ? createTransientView(graph, `${id}-anisotropic-sum`, 'float32', pixelCount)
      : undefined;
    const nadirSum = output.negativeOpenness
      ? createTransientView(graph, `${id}-nadir-sum`, 'float32', pixelCount)
      : undefined;
    const rowDirection = props.rowDirection ?? 'south';
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const distances = Array.from(this.stepDistances, getWGSLFloatLiteral).join(', ');
    const hull =
      this.algorithm === 'sweep'
        ? createTransientView(graph, `${id}-sweep-hull`, 'uint32', pixelCount)
        : undefined;
    // The march fuses sectors into few kernels: sums stay in registers and are stored once per
    // chunk, the centre is loaded once, and every sector of a pixel is written by one thread.
    // An odd sector count with packed unorm16 codes shares words between neighbouring pixels, which
    // only the one-sector-per-dispatch path can update safely.
    const fusedMarch = !hull && !(this.horizonFormat === 'unorm16' && directionCount % 2 === 1);
    if (fusedMarch) {
      nodes.push(
        ...getFusedMarchNodes(graph, {
          id,
          width,
          height,
          directionCount,
          rowDirection,
          distances,
          stepCount: this.stepDistances.length,
          maximumStepsPerDispatch: props.maximumStepsPerDispatch ?? FUSED_MARCH_STEP_BUDGET,
          cellSizeMode,
          elevationValues,
          settings: props.parameters,
          horizonFormat: this.horizonFormat,
          horizon: output.horizon,
          sineSum,
          angleSum,
          anisotropicSum,
          nadirSum
        })
      );
    }
    const sectorRanges =
      !fusedMarch && !hull
        ? createHorizonSectorRanges<Parameters>(graph, {id, directionCount, sectorsPerRange: 1})
        : undefined;
    if (sectorRanges) {
      nodes.push(sectorRanges.writeNode);
    }
    const sweepSectors = hull
      ? createTerrainSweepSectorTable<Parameters>(graph, {
          id,
          width,
          height,
          directions: Array.from({length: directionCount}, (_, sector) =>
            getGPUTerrainHorizonDirection(sector, directionCount, rowDirection)
          )
        })
      : undefined;
    if (sweepSectors) {
      nodes.push(sweepSectors.writeNode);
    }
    for (let sector = 0; !fusedMarch && sector < directionCount; sector++) {
      if (hull) {
        nodes.push(
          ...getSweepSectorNodes(graph, {
            id: `${id}-horizon-${sector}`,
            width,
            height,
            sectorRecord: sweepSectors!.getRecord(sector),
            directionCount,
            maximumRadius: props.maximumRadius,
            cellSizeMode,
            elevationValues,
            settings: props.parameters,
            hull,
            horizonFormat: this.horizonFormat,
            horizon: output.horizon,
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
          sectorRange: sectorRanges!.getRange(sector),
          directionCount,
          rowDirection,
          distances,
          stepCount: this.stepDistances.length,
          cellSizeMode,
          elevationValues,
          settings: props.parameters,
          horizonFormat: this.horizonFormat,
          horizon: output.horizon,
          sineSum,
          angleSum,
          anisotropicSum,
          nadirSum
        })
      );
    }
    if (skyViewTarget || output.positiveOpenness || output.outputValidity) {
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
      if (output.positiveOpenness) {
        bindings.push({
          name: 'openness',
          view: output.positiveOpenness,
          type: 'f32',
          access: 'read_write'
        });
      }
      if (output.outputValidity) {
        bindings.push({
          name: 'validityValues',
          view: output.outputValidity,
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
    output.positiveOpenness
      ? `openness[opennessOffset + index] = select(invalidValue,
    90.0 - angleSum[angleSumOffset + index] / DIRECTION_COUNT, isValid);`
      : ''
  }
  ${output.outputValidity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
        })
      );
    }
    if (
      (anisotropicSum && output.anisotropicSkyViewFactor) ||
      (nadirSum && output.negativeOpenness)
    ) {
      // Kept apart from the sky-view kernel so no kernel binds more than 8 storage buffers.
      const bindings: WGSLKernelBinding[] = [
        {name: 'elevationValidity', view: elevationValidity, type: 'u32', access: 'read'}
      ];
      let validExpression = 'elevationValidity[elevationValidityOffset + index] != 0u';
      let statements = '';
      if (anisotropicSum && output.anisotropicSkyViewFactor) {
        bindings.push(
          {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'anisotropicSum', view: anisotropicSum, type: 'f32', access: 'read'},
          {
            name: 'anisotropicSkyView',
            view: output.anisotropicSkyViewFactor,
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
      if (nadirSum && output.negativeOpenness) {
        bindings.push(
          {name: 'nadirSum', view: nadirSum, type: 'f32', access: 'read'},
          {
            name: 'negativeOpennessValues',
            view: output.negativeOpenness,
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
    if (output.skyViewFactorTexture && skyViewTarget) {
      const texture = output.skyViewFactorTexture;
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
    /** Sector record from the contributor's {@link createTerrainSweepSectorTable}. */
    sectorRecord: GraphDataView<'uint32'>;
    directionCount: number;
    maximumRadius: number;
    cellSizeMode: GPURasterCellSizeMode;
    elevationValues: GraphDataView<'float32'>;
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
    geometry: {kind: 'table' as const, record: props.sectorRecord},
    maximumRadius: props.maximumRadius,
    cellSizeMode: props.cellSizeMode,
    elevationValues: props.elevationValues,
    settings: props.settings,
    hull: props.hull
  };
  const nodes: GPUCommandNode<Parameters>[] = [];
  // The sweep binds elevation values, settings, hull pointers, and the sector record itself.
  if (props.horizon || props.sineSum || props.angleSum || props.anisotropicSum) {
    const output = getTerrainHorizonSectorOutput({
      sector: 0,
      sectorWGSL: TERRAIN_SWEEP_SECTOR_WGSL,
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
      createTerrainHorizonSweepNode(graph, {
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
      sector: 0,
      sectorWGSL: TERRAIN_SWEEP_SECTOR_WGSL,
      directionCount: props.directionCount,
      nadirSum: props.nadirSum,
      mode: 'nadir',
      reservedBindingCount: 4
    });
    nodes.push(
      createTerrainHorizonSweepNode(graph, {
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

/** Elements between sector range words: 256 bytes, so every range binds at element offset 0. */
const SECTOR_RANGE_STRIDE = 64;

/**
 * Creates the graph-owned sector ranges of the march kernels and the one-thread node that writes
 * them: range `r` is `[r * sectorsPerRange, min(directionCount, (r + 1) * sectorsPerRange))` at
 * element `r * 64`. Every march kernel reads its sectors from its range word, so all of them share
 * one WGSL source (library rule 12).
 */
function createHorizonSectorRanges<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; directionCount: number; sectorsPerRange: number}
): {
  rangeCount: number;
  writeNode: GPUCommandNode<Parameters>;
  getRange: (range: number) => GraphDataView<'uint32'>;
} {
  const {directionCount, sectorsPerRange} = props;
  const rangeCount = Math.ceil(directionCount / sectorsPerRange);
  const table = createTransientView(
    graph,
    `${props.id}-sector-range-words`,
    'uint32',
    rangeCount * SECTOR_RANGE_STRIDE
  );
  const writeNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-sector-ranges`,
    operation: 'GPUTerrainHorizon',
    variant: 'sector-ranges',
    bindings: [{name: 'ranges', view: table, type: 'u32', access: 'read_write'}],
    invocationCount: 1,
    body: `for (var range = 0u; range < ${rangeCount}u; range++) {
    let first = range * ${sectorsPerRange}u;
    ranges[rangesOffset + range * ${SECTOR_RANGE_STRIDE}u] = first;
    ranges[rangesOffset + range * ${SECTOR_RANGE_STRIDE}u + 1u] = min(${directionCount}u, first + ${sectorsPerRange}u);
  }`
  });
  const getRange = (range: number): GraphDataView<'uint32'> =>
    graph.createDataView(table.buffer, {
      format: 'uint32',
      length: 2,
      byteOffset: table.byteOffset + range * SECTOR_RANGE_STRIDE * 4
    });
  return {rangeCount, writeNode, getRange};
}

/** Pixel access and bilinear sampling shared by the per-sector and the fused march kernels. */
const HORIZON_MARCH_SAMPLE_WGSL = `// Canonical elevation is NaN exactly where a pixel is invalid, so the march kernels need no
// validity binding (its slot holds the sector range word).
fn isValidPixel(column: u32, row: u32) -> bool {
  return isFiniteValue(elevationValues[elevationValuesOffset + row * WIDTH + column]);
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
  // Canonical elevation is NaN exactly where a pixel is invalid: one bit test per corner replaces
  // four validity loads.
  let corner00 = getElevation(base.x, base.y);
  let corner10 = getElevation(next.x, base.y);
  let corner01 = getElevation(base.x, next.y);
  let corner11 = getElevation(next.x, next.y);
  if (!isFiniteValue(corner00) || !isFiniteValue(corner10) ||
      !isFiniteValue(corner01) || !isFiniteValue(corner11)) {
    return vec2<f32>(0.0, 0.0);
  }
  let top = mix(corner00 - origin, corner10 - origin, fraction.x);
  let bottom = mix(corner01 - origin, corner11 - origin, fraction.x);
  return vec2<f32>(mix(top, bottom, fraction.y), 1.0);
}`;

/** Sample steps (pixels x steps x sectors) one fused march dispatch may cost, to bound its duration. */
const FUSED_MARCH_STEP_BUDGET = 2 ** 30;

/**
 * Builds the fused ray-march kernels: each invocation marches every sector of its pixel in order,
 * `sectorsPerChunk` sectors per dispatch. Per sector the arithmetic is that of
 * {@link getHorizonNode}; the sums add in the same sector order, kept in registers and stored once
 * per chunk (read back only when a later chunk continues them), so results are bit-identical to
 * one dispatch per sector with far less memory traffic and no per-sector dispatch.
 */
function getFusedMarchNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    directionCount: number;
    rowDirection: GPURasterRowDirection;
    distances: string;
    stepCount: number;
    maximumStepsPerDispatch: number;
    cellSizeMode: GPURasterCellSizeMode;
    elevationValues: GraphDataView<'float32'>;
    settings: GraphDataView<'float32'>;
    horizonFormat: GPUTerrainHorizonFormat;
    horizon?: GraphDataView<'float32'> | GraphDataView<'uint32'>;
    sineSum?: GraphDataView<'float32'>;
    angleSum?: GraphDataView<'float32'>;
    anisotropicSum?: GraphDataView<'float32'>;
    nadirSum?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {directionCount} = props;
  const hasZenith = Boolean(
    props.horizon || props.sineSum || props.angleSum || props.anisotropicSum
  );
  const zenithOutput = hasZenith
    ? getTerrainHorizonSectorOutput({
        sector: 0,
        sectorWGSL: 'sector',
        directionCount,
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
        sector: 0,
        sectorWGSL: 'sector',
        directionCount,
        nadirSum: props.nadirSum,
        mode: 'nadir',
        reservedBindingCount: 3 + (zenithOutput?.bindings.length ?? 0)
      })
    : undefined;
  const bindings: WGSLKernelBinding[] = [
    {name: 'elevationValues', view: props.elevationValues, type: 'f32', access: 'read'},
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
    ...(zenithOutput?.bindings ?? []),
    ...(nadirOutput?.bindings ?? [])
  ];
  const nadir = Boolean(nadirOutput);
  const pixelCount = props.width * props.height;
  const sectorsPerChunk = Math.max(
    1,
    Math.min(
      directionCount,
      Math.floor(props.maximumStepsPerDispatch / (pixelCount * props.stepCount))
    )
  );
  const directions = Array.from({length: directionCount}, (_, sector) => {
    const [x, y] = getGPUTerrainHorizonDirection(sector, directionCount, props.rowDirection);
    return `vec2<f32>(${getWGSLFloatLiteral(x)}, ${getWGSLFloatLiteral(y)})`;
  }).join(', ');
  const format = props.horizonFormat;
  const statements: string[] = [];
  if (props.horizon) {
    statements.push(
      format === 'unorm16'
        ? `// Both sectors of a word belong to this pixel (even sector count), so no other thread shares it.
      let element = pixel * DIRECTION_COUNT + sector;
      let wordIndex = horizonOffset + (element >> 1u);
      let shift = (element & 1u) * 16u;
      horizon[wordIndex] = (horizon[wordIndex] & ~(0xffffu << shift)) |
        (encodeHorizonUnorm16(horizonAngle) << shift);`
        : 'horizon[horizonOffset + pixel * DIRECTION_COUNT + sector] = horizonAngle;'
    );
  }
  if (props.sineSum || props.anisotropicSum) {
    statements.push(
      'let sineTerm = sin(max(horizonAngle, 0.0) * DEGREES_TO_RADIANS);',
      ...(props.sineSum
        ? ['sineAccumulator = select(sineAccumulator + sineTerm, sineTerm, sector == 0u);']
        : []),
      ...(props.anisotropicSum
        ? [
            'let anisotropicTerm = getAnisotropicWeight(sector) * sineTerm;',
            'anisotropicAccumulator = select(anisotropicAccumulator + anisotropicTerm, anisotropicTerm, sector == 0u);'
          ]
        : [])
    );
  }
  if (props.angleSum) {
    statements.push(
      'angleAccumulator = select(angleAccumulator + horizonAngle, horizonAngle, sector == 0u);'
    );
  }
  const loads = [
    props.sineSum
      ? 'var sineAccumulator = select(0.0, sineSum[sineSumOffset + pixel], firstSector > 0u);'
      : '',
    props.angleSum
      ? 'var angleAccumulator = select(0.0, angleSum[angleSumOffset + pixel], firstSector > 0u);'
      : '',
    props.anisotropicSum
      ? 'var anisotropicAccumulator = select(0.0, anisotropicSum[anisotropicSumOffset + pixel], firstSector > 0u);'
      : '',
    props.nadirSum
      ? 'var nadirAccumulator = select(0.0, nadirSum[nadirSumOffset + pixel], firstSector > 0u);'
      : ''
  ].join('\n  ');
  const stores = [
    props.sineSum ? 'sineSum[sineSumOffset + pixel] = sineAccumulator;' : '',
    props.angleSum ? 'angleSum[angleSumOffset + pixel] = angleAccumulator;' : '',
    props.anisotropicSum
      ? 'anisotropicSum[anisotropicSumOffset + pixel] = anisotropicAccumulator;'
      : '',
    props.nadirSum ? 'nadirSum[nadirSumOffset + pixel] = nadirAccumulator;' : ''
  ].join('\n  ');
  const ranges = createHorizonSectorRanges<Parameters>(graph, {
    id: props.id,
    directionCount,
    sectorsPerRange: sectorsPerChunk
  });
  const nodes: GPUCommandNode<Parameters>[] = [ranges.writeNode];
  for (let chunk = 0; chunk < ranges.rangeCount; chunk++) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-horizon-march-${chunk}`,
        operation: 'GPUTerrainHorizon',
        variant: `march-fused-${props.cellSizeMode}${nadir ? '-nadir' : ''}`,
        bindings: [
          ...bindings,
          {name: 'sectorRange', view: ranges.getRange(chunk), type: 'u32', access: 'read'}
        ],
        invocationCount: pixelCount,
        declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const DIRECTION_COUNT: u32 = ${directionCount}u;
const DIRECTIONS: array<vec2<f32>, ${directionCount}> = array<vec2<f32>, ${directionCount}>(${directions});
const STEP_COUNT: u32 = ${props.stepCount}u;
const STEP_DISTANCES: array<f32, ${props.stepCount}> = array<f32, ${props.stepCount}>(${props.distances});
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${zenithOutput?.declarations ?? ''}
${getRasterGroundCellSizeWGSL(props.cellSizeMode, {
  cellSizeIndex: 0,
  northEdgeIndex: 4
})}
${HORIZON_MARCH_SAMPLE_WGSL}
${RASTER_PIXEL_DIRECTION_WGSL}`,
        body: `let pixel = index;
  let firstSector = sectorRange[sectorRangeOffset];
  let endSector = sectorRange[sectorRangeOffset + 1u];
  let column = index % WIDTH;
  let row = index / WIDTH;
  let zFactor = settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  let maximumDistance = settings[settingsOffset + 6u];
  let groundCell = getGroundCellSize(row);
  let centerValid = isValidPixel(column, row);
  let centerElevation = select(0.0, getElevation(column, row), centerValid);
  let limit = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let center = vec2<f32>(f32(column), f32(row));
  ${loads}
  for (var sector = firstSector; sector < endSector; sector++) {
    // DIRECTIONS are ground azimuths; the pixel step follows them on anisotropic cells. The ray
    // keeps this row's cell size (see GPUTerrainHorizonAlgorithm for the along-ray error).
    let direction = getPixelDirection(DIRECTIONS[sector], groundCell);
    let groundStep = length(direction * groundCell);
    var horizonAngle = getNaN(index);
    var nadirAngle = getNaN(index);
    if (centerValid && groundStep > 0.0 && isFiniteValue(groundStep)) {
      var hasSample = false;
      var maximumTangent = 0.0;
      var maximumNadirTangent = 0.0;
      for (var stepIndex = 0u; stepIndex < STEP_COUNT; stepIndex++) {
        let stepDistance = STEP_DISTANCES[stepIndex];
        let position = center + direction * stepDistance;
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
    ${statements.join('\n    ')}
    ${nadir ? 'nadirAccumulator = select(nadirAccumulator + nadirAngle, nadirAngle, sector == 0u);' : ''}
  }
  ${stores}`
      })
    );
  }
  return nodes;
}

/**
 * Builds the ray-march kernel for one horizon sector. The sector is read from `sectorRange`, so
 * every sector's kernel has the same WGSL (library rule 12).
 */
function getHorizonNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    /** Range word `[sector, sector + 1]`, see {@link createHorizonSectorRanges}. */
    sectorRange: GraphDataView<'uint32'>;
    directionCount: number;
    rowDirection: GPURasterRowDirection;
    distances: string;
    stepCount: number;
    cellSizeMode: GPURasterCellSizeMode;
    elevationValues: GraphDataView<'float32'>;
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
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
    {name: 'sectorRange', view: props.sectorRange, type: 'u32', access: 'read'}
  ];
  const zenithOutput =
    props.horizon || props.sineSum || props.angleSum || props.anisotropicSum
      ? getTerrainHorizonSectorOutput({
          sector: 0,
          sectorWGSL: 'sector',
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
        sector: 0,
        sectorWGSL: 'sector',
        directionCount: props.directionCount,
        nadirSum: props.nadirSum,
        mode: 'nadir',
        reservedBindingCount: 3 + (zenithOutput?.bindings.length ?? 0)
      })
    : undefined;
  bindings.push(...(zenithOutput?.bindings ?? []), ...(nadirOutput?.bindings ?? []));
  const nadir = Boolean(nadirOutput);
  const directions = Array.from({length: props.directionCount}, (_, sector) => {
    const [x, y] = getGPUTerrainHorizonDirection(sector, props.directionCount, props.rowDirection);
    return `vec2<f32>(${getWGSLFloatLiteral(x)}, ${getWGSLFloatLiteral(y)})`;
  }).join(', ');
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTerrainHorizon',
    variant: `march-${props.cellSizeMode}${nadir ? '-nadir' : ''}`,
    bindings,
    invocationCount: props.width * props.height,
    declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const DIRECTION_COUNT: u32 = ${props.directionCount}u;
const DIRECTIONS: array<vec2<f32>, ${props.directionCount}> = array<vec2<f32>, ${props.directionCount}>(${directions});
const STEP_COUNT: u32 = ${props.stepCount}u;
const STEP_DISTANCES: array<f32, ${props.stepCount}> = array<f32, ${props.stepCount}>(${props.distances});
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${zenithOutput?.declarations ?? ''}
${getRasterGroundCellSizeWGSL(props.cellSizeMode, {
  cellSizeIndex: 0,
  northEdgeIndex: 4
})}
${HORIZON_MARCH_SAMPLE_WGSL}
${RASTER_PIXEL_DIRECTION_WGSL}`,
    body: `let pixel = index;
  let sector = sectorRange[sectorRangeOffset];
  let column = index % WIDTH;
  let row = index / WIDTH;
  let zFactor = settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  let maximumDistance = settings[settingsOffset + 6u];
  let groundCell = getGroundCellSize(row);
  let direction = getPixelDirection(DIRECTIONS[sector], groundCell);
  let groundStep = length(direction * groundCell);
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
      let position = center + direction * stepDistance;
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
