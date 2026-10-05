// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createMapGraphKernelNode,
  getWGSLFloatLiteral,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import {
  getTerrainIlluminationGroundCellWGSL,
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  type GPUTerrainIlluminationCellSizeMode
} from './terrain-illumination-utils';

/** Fixed-point scale of the digital line slope: `slopeFixed = round(minor / major * 65536)`. @internal */
export const TERRAIN_SWEEP_SLOPE_SCALE = 65536;

/** Largest grid extent supported by the 32-bit fixed-point line geometry. @internal */
export const TERRAIN_SWEEP_MAX_EXTENT = 32767;

/** Workgroup size of the sweep kernels (one invocation per digital line). @internal */
const TERRAIN_SWEEP_WORKGROUP_SIZE = 64;

/**
 * Exact integer description of the digital line family for one direction.
 *
 * The lines `L_m = {(a, m + F(a))}` with `F(a) = (slopeFixed * a + 32768) >> 16` partition all
 * pixels of the grid. `a` is the column for x-major directions (`|dx| >= |dy|`, minor index = row)
 * and the row for y-major directions (minor index = column). The same integers are used on CPU
 * oracles and in WGSL, so both agree on which pixels lie on which line.
 *
 * @internal
 */
export type TerrainSweepLineGeometry = {
  /** True when columns are the major axis. */
  xMajor: boolean;
  /** Fixed-point (16 fractional bits) minor offset per major step, an integer in `[-65536, 65536]`. */
  slopeFixed: number;
  /** `+1` when the ray travels toward increasing major index, else `-1`. */
  travelSign: 1 | -1;
  /** Number of lines that contain at least one pixel. */
  lineCount: number;
  /** Minor offset `m` of the first line (lines run `m = firstLine .. firstLine + lineCount - 1`). */
  firstLine: number;
  /** `Math.fround(hypot(1, slopeFixed / 65536))`: pixel length of one major step. */
  stepPixelLength: number;
};

/**
 * Returns `F(a) = (slopeFixed * a + 32768) >> 16` (arithmetic shift, i.e. floor division), the
 * minor offset of major index `a`. Exact in float64 for grid extents up to 32767.
 *
 * @internal
 */
export function getTerrainSweepMinorOffset(slopeFixed: number, major: number): number {
  return Math.floor((slopeFixed * major + 32768) / TERRAIN_SWEEP_SLOPE_SCALE);
}

/**
 * Computes the digital line family of a pixel-space direction on a `width x height` grid.
 *
 * Direction components are used as given (callers pass float32 sector directions or float64 sun
 * directions). The slope is rounded once, in float64, to the 16-bit fixed point `slopeFixed`; the
 * geometry that follows is exact integer arithmetic.
 *
 * @throws If the direction is zero or not finite, or an extent exceeds 32767.
 * @internal
 */
export function getTerrainSweepLineGeometry(
  direction: readonly [number, number],
  width: number,
  height: number
): TerrainSweepLineGeometry {
  const [dx, dy] = direction;
  if (!Number.isFinite(dx) || !Number.isFinite(dy) || (dx === 0 && dy === 0)) {
    throw new Error('Terrain sweep direction must be finite and non-zero');
  }
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width < 1 ||
    height < 1 ||
    width > TERRAIN_SWEEP_MAX_EXTENT ||
    height > TERRAIN_SWEEP_MAX_EXTENT
  ) {
    throw new Error(`Terrain sweep extents must be integers in [1, ${TERRAIN_SWEEP_MAX_EXTENT}]`);
  }
  const xMajor = Math.abs(dx) >= Math.abs(dy);
  const major = xMajor ? dx : dy;
  const minor = xMajor ? dy : dx;
  // `+ 0` turns -0 into 0.
  const slopeFixed = Math.round((minor / major) * TERRAIN_SWEEP_SLOPE_SCALE) + 0;
  const majorExtent = xMajor ? width : height;
  const minorExtent = xMajor ? height : width;
  const lastOffset = getTerrainSweepMinorOffset(slopeFixed, majorExtent - 1);
  return {
    xMajor,
    slopeFixed,
    travelSign: major > 0 ? 1 : -1,
    // F(0) is always 0.
    firstLine: -Math.max(0, lastOffset) + 0,
    lineCount: minorExtent + Math.abs(lastOffset),
    stepPixelLength: Math.fround(Math.hypot(1, slopeFixed / TERRAIN_SWEEP_SLOPE_SCALE))
  };
}

/** Where the sweep reads its line geometry. @internal */
export type TerrainHorizonSweepGeometrySource =
  | {
      /** Geometry baked into the shader from a pixel-space direction (one node per sector). */
      kind: 'baked';
      /** Pixel-space unit direction. */
      direction: readonly [number, number];
    }
  | {
      /**
       * Geometry read from `settings[index .. index + 3] = [xMajor (0/1), slopeFixed, travelSign,
       * stepPixelLength]`; the line count is derived in the shader and `width + height`
       * invocations are dispatched, surplus ones exit.
       */
      kind: 'settings';
      /** Settings index of `xMajor`. */
      index: number;
    };

/**
 * Properties for {@link getTerrainHorizonSweepNode}.
 *
 * The cell-size model is that of `GPUTerrainHorizon`: projected metres (`uniform`) or
 * latitude-dependent spacing (`web-mercator`, `geographic`) read from `settings`.
 *
 * @internal
 */
export type TerrainHorizonSweepNodeProps = {
  /** Graph-wide node ID. */
  id: string;
  /** Grid width in pixels, at most 32767. */
  width: number;
  /** Grid height in pixels, at most 32767. */
  height: number;
  /** Pixel-space unit direction of the sector, from `getGPUTerrainHorizonDirection`. */
  direction: readonly [number, number];
  /** Radius window in pixels (the recipe's `maximumRadius`). */
  maximumRadius: number;
  /** Cell size interpretation. */
  cellSizeMode: GPUTerrainIlluminationCellSizeMode;
  /** 1 for horizons, -1 for the nadir pass (inverted DEM; curvature still drops). */
  zFactorSign: 1 | -1;
  /** Canonical elevation values, one per pixel. */
  elevationValues: GraphDataView<'float32'>;
  /** Canonical elevation validity, one per pixel. */
  elevationValidity: GraphDataView<'uint32'>;
  /** GPUTerrainHorizon settings: `[cellX, cellY, zFactor, curvature, northEdge, southEdge, maximumDistance, _]`. */
  settings: GraphDataView<'float32'>;
  /** Pixel-count u32 scratch, reused by every sector (sectors run in order). */
  hull: GraphDataView<'uint32'>;
  /** Up to 4 extra bindings written by `outputWGSL`. */
  outputBindings: readonly MapGraphKernelBinding[];
  /** Extra module-scope WGSL (helpers and constants) for `outputWGSL`. */
  outputDeclarations?: string;
  /**
   * WGSL statements run once per pixel with `pixel: u32` and `horizonAngle: f32` (degrees, NaN
   * when the centre is invalid) in scope. They run inside a block, so `let` names are local.
   */
  outputWGSL: string;
};

/** {@link TerrainHorizonSweepNodeProps} with a selectable geometry source. @internal */
export type TerrainHorizonSweepKernelProps = Omit<TerrainHorizonSweepNodeProps, 'direction'> & {
  /** Baked direction or per-frame settings geometry. */
  geometry: TerrainHorizonSweepGeometrySource;
};

/**
 * Builds the exact upper-hull sweep kernel for one direction: the horizon angle of every pixel
 * along one digital line family, in amortised O(1) per pixel.
 *
 * ## Algorithm (Stewart-style hull pointers)
 *
 * Each invocation owns one digital line and walks its pixels from the far ("ahead") end back
 * toward the start. For pixel `p` with ahead samples `r` at distance index `k = |a_r - a_p|`, the
 * horizon is the sample maximising `rise(r) / k` with
 * `rise(r) = z * (h_r - h_p) - c * (k * Lref)^2`. Writing `G(x) = z * h_x - b * a_x^2` with
 * `b = c * Lref^2`, `rise / k = (G(r) - G(p)) / k + 2 * b * a_p * t`, so the best sample is the
 * tangent from `p` to the upper convex hull of the points ahead and does not depend on the
 * viewpoint's offset; this holds for any sign of `z` and any `c`. The hull of the points ahead is
 * the chain `p + 1 -> hull[p + 1] -> ...` stored in a per-pixel `hull` pointer: each pixel
 * points at its tangent vertex, found by following the chain from the next valid pixel while the
 * slope keeps growing. The unbounded walk is amortised O(1) per pixel.
 *
 * Decisions use cross-multiplication with the integer `k` (no division), so ties are exact and
 * the nearest sample wins. Radius (`k * stepPixelLength <= maximumRadius`) and distance
 * (`k * Lp <= maximumDistance`) windows are monotone in `k`; when the unbounded tangent lies
 * outside the window an exact windowed query re-walks hull chains restarted after each in-window
 * chain end (worst case O(window), typically a few steps).
 *
 * `Lref` is constant for the whole sweep: the ground step of the pixel row in `uniform` mode
 * (identical everywhere) and the ground step at row `height / 2` otherwise. The argmax is exact for
 * `c = 0` in every mode and for any `c` in uniform mode; with `c != 0` and latitude-dependent
 * cells it uses the middle row's spacing. The reported angle always uses the pixel's own row
 * spacing `Lp`.
 *
 * Elevation, validity, and hull pointers of invalid pixels follow the march semantics: invalid
 * samples are skipped, a ray without a valid in-window sample reports 0, an invalid centre reports
 * NaN.
 *
 * @internal
 */
export function getTerrainHorizonSweepNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TerrainHorizonSweepNodeProps
): GPUCommandNode<Parameters> {
  const {direction, ...rest} = props;
  return createTerrainHorizonSweepNode(graph, {
    ...rest,
    geometry: {kind: 'baked', direction}
  });
}

/**
 * Like {@link getTerrainHorizonSweepNode} with a selectable geometry source; used by the cast
 * shadow, which reads the per-frame sun direction from settings.
 *
 * @internal
 */
export function createTerrainHorizonSweepNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TerrainHorizonSweepKernelProps
): GPUCommandNode<Parameters> {
  const {width, height, geometry} = props;
  if (props.outputBindings.length > 4) {
    throw new Error(`${props.id} sweep supports at most 4 output bindings`);
  }
  if (!Number.isFinite(props.maximumRadius) || props.maximumRadius <= 0) {
    throw new Error(`${props.id} sweep maximumRadius must be positive`);
  }
  const bakedGeometry =
    geometry.kind === 'baked'
      ? getTerrainSweepLineGeometry(geometry.direction, width, height)
      : undefined;
  if (width > TERRAIN_SWEEP_MAX_EXTENT || height > TERRAIN_SWEEP_MAX_EXTENT) {
    throw new Error(`${props.id} sweep extents must not exceed ${TERRAIN_SWEEP_MAX_EXTENT}`);
  }
  const geometryWGSL = bakedGeometry
    ? `let xMajor = ${bakedGeometry.xMajor ? 'true' : 'false'};
  let slopeFixed: i32 = ${bakedGeometry.slopeFixed};
  let travelSign: i32 = ${bakedGeometry.travelSign};
  let stepPixelLength: f32 = ${getWGSLFloatLiteral(bakedGeometry.stepPixelLength)};`
    : `let xMajor = settings[settingsOffset + ${(geometry as {index: number}).index}u] != 0.0;
  let slopeFixed = i32(settings[settingsOffset + ${(geometry as {index: number}).index + 1}u]);
  let travelSign = select(-1, 1, settings[settingsOffset + ${(geometry as {index: number}).index + 2}u] > 0.0);
  let stepPixelLength = settings[settingsOffset + ${(geometry as {index: number}).index + 3}u];`;
  const bindings: MapGraphKernelBinding[] = [
    {name: 'elevationValues', view: props.elevationValues, type: 'f32', access: 'read'},
    {name: 'elevationValidity', view: props.elevationValidity, type: 'u32', access: 'read'},
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
    {name: 'hull', view: props.hull, type: 'u32', access: 'read_write'},
    ...props.outputBindings
  ];
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUTerrainHorizon',
    variant: `sweep-${props.cellSizeMode}`,
    bindings,
    workgroupSize: TERRAIN_SWEEP_WORKGROUP_SIZE,
    invocationCount: bakedGeometry ? bakedGeometry.lineCount : width + height,
    declarations: getTerrainHorizonSweepDeclarations(props),
    body: `${geometryWGSL}
  let majorExtent = select(HEIGHT, WIDTH, xMajor);
  let minorExtent = select(WIDTH, HEIGHT, xMajor);
  let lastOffset = getFixedOffset(slopeFixed, i32(majorExtent) - 1);
  let lineCount = i32(minorExtent) + abs(lastOffset);
  if (i32(index) >= lineCount) {
    return;
  }
  sweepXMajor = xMajor;
  sweepSlopeFixed = slopeFixed;
  sweepTravel = travelSign;
  sweepMajorExtent = i32(majorExtent);
  sweepMinorExtent = i32(minorExtent);
  sweepLine = i32(index) - max(0, lastOffset);
  sweepZ = f32(${props.zFactorSign}) * settings[settingsOffset + 2u];
  let curvature = settings[settingsOffset + 3u];
  sweepMaximumDistance = settings[settingsOffset + 6u];
  sweepStepPixelLength = stepPixelLength;
  let referenceStep = getStepLength(select(HEIGHT / 2u, 0u, ${props.cellSizeMode === 'uniform'}), xMajor, slopeFixed);
  sweepCurvature = curvature * referenceStep * referenceStep;
  var previous = NONE;
  for (var stepIndex = 0; stepIndex < sweepMajorExtent; stepIndex++) {
    let major = select(stepIndex, sweepMajorExtent - 1 - stepIndex, travelSign > 0);
    let minor = sweepLine + getFixedOffset(slopeFixed, major);
    if (minor < 0 || minor >= sweepMinorExtent) {
      continue;
    }
    let pixel = select(u32(major) * WIDTH + u32(minor), u32(minor) * WIDTH + u32(major), xMajor);
    let next = previous;
    previous = pixel;
    var firstAhead = NONE;
    if (next != NONE) {
      firstAhead = select(hull[hullOffset + next], next, isValidPixel(next));
    }
    var horizonAngle = getNaN(pixel);
    if (!isValidPixel(pixel)) {
      hull[hullOffset + pixel] = firstAhead;
    } else {
      sweepElevation = elevationValues[elevationValuesOffset + pixel];
      sweepMajor = major;
      var tangent = firstAhead;
      if (tangent != NONE) {
        var chain = hull[hullOffset + tangent];
        while (chain != NONE && isGrowing(chain, tangent)) {
          tangent = chain;
          chain = hull[hullOffset + chain];
        }
      }
      hull[hullOffset + pixel] = tangent;
      let pixelStep = getStepLength(pixel / WIDTH, xMajor, slopeFixed);
      if (pixelStep > 0.0 && isFiniteValue(pixelStep)) {
        sweepPixelStep = pixelStep;
        var best = NONE;
        if (firstAhead != NONE && isInsideWindow(firstAhead)) {
          if (isInsideWindow(tangent)) {
            best = tangent;
          } else {
            best = queryWindow(firstAhead);
          }
        }
        if (best == NONE) {
          horizonAngle = 0.0;
        } else {
          let groundDistance = getDistanceIndex(best) * pixelStep;
          let rise = sweepZ * (elevationValues[elevationValuesOffset + best] - sweepElevation) -
            curvature * groundDistance * groundDistance;
          horizonAngle = atan(rise / groundDistance) * RADIANS_TO_DEGREES;
        }
      }
    }
    {
      ${props.outputWGSL}
    }
  }`
  });
}

/** Module-scope WGSL of the sweep kernel. */
function getTerrainHorizonSweepDeclarations(props: TerrainHorizonSweepKernelProps): string {
  return `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const NONE: u32 = 0xffffffffu;
const MAXIMUM_RADIUS: f32 = ${getWGSLFloatLiteral(props.maximumRadius)};
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${getTerrainIlluminationGroundCellWGSL(props.cellSizeMode, {
  settingsName: 'settings',
  cellSizeIndex: 0,
  northEdgeIndex: 4
})}
var<private> sweepXMajor: bool;
var<private> sweepSlopeFixed: i32;
var<private> sweepTravel: i32;
var<private> sweepMajorExtent: i32;
var<private> sweepMinorExtent: i32;
var<private> sweepLine: i32;
var<private> sweepZ: f32;
var<private> sweepCurvature: f32;
var<private> sweepMaximumDistance: f32;
var<private> sweepStepPixelLength: f32;
var<private> sweepPixelStep: f32;
var<private> sweepElevation: f32;
var<private> sweepMajor: i32;
fn getFixedOffset(slopeFixed: i32, major: i32) -> i32 {
  return (slopeFixed * major + 32768) >> 16u;
}
// Ground length of one major step on the pixel row, using the row's cell size.
fn getStepLength(row: u32, xMajor: bool, slopeFixed: i32) -> f32 {
  let slope = f32(slopeFixed) / 65536.0;
  let cell = getGroundCellSize(row);
  return length(select(vec2<f32>(slope, 1.0), vec2<f32>(1.0, slope), xMajor) * cell);
}
fn isValidPixel(pixel: u32) -> bool {
  return elevationValidity[elevationValidityOffset + pixel] != 0u;
}
fn getMajorIndex(pixel: u32) -> i32 {
  return i32(select(pixel / WIDTH, pixel % WIDTH, sweepXMajor));
}
fn getDistanceIndex(pixel: u32) -> f32 {
  return f32(abs(getMajorIndex(pixel) - sweepMajor));
}
// Hull-metric rise for the sample at distance index kf (curvature uses the reference step).
fn getRise(pixel: u32, kf: f32) -> f32 {
  return sweepZ * (elevationValues[elevationValuesOffset + pixel] - sweepElevation) -
    sweepCurvature * (kf * kf);
}
// True when sample r has a strictly larger tangent than sample c (cross-multiplied, exact ties).
fn isGrowing(r: u32, c: u32) -> bool {
  let kr = getDistanceIndex(r);
  let kc = getDistanceIndex(c);
  return getRise(r, kr) * kc > getRise(c, kc) * kr;
}
fn isInsideWindow(pixel: u32) -> bool {
  let kf = getDistanceIndex(pixel);
  return kf * sweepStepPixelLength <= MAXIMUM_RADIUS &&
    (sweepMaximumDistance <= 0.0 || kf * sweepPixelStep <= sweepMaximumDistance);
}
// First valid pixel strictly after c on the line, NONE at the end.
fn getValidAfter(c: u32) -> u32 {
  let major = getMajorIndex(c) + sweepTravel;
  if (major < 0 || major >= sweepMajorExtent) { return NONE; }
  let minor = sweepLine + getFixedOffset(sweepSlopeFixed, major);
  if (minor < 0 || minor >= sweepMinorExtent) { return NONE; }
  let pixel = select(u32(major) * WIDTH + u32(minor), u32(minor) * WIDTH + u32(major), sweepXMajor);
  return select(hull[hullOffset + pixel], pixel, isValidPixel(pixel));
}
// Exact maximum over the in-window samples at or after start (start is valid and inside).
fn queryWindow(start: u32) -> u32 {
  var best = NONE;
  var t = start;
  loop {
    if (t == NONE || !isInsideWindow(t)) { break; }
    var c = t;
    var r = hull[hullOffset + c];
    while (r != NONE && isInsideWindow(r) && isGrowing(r, c)) {
      c = r;
      r = hull[hullOffset + r];
    }
    if (best == NONE) {
      best = c;
    } else if (isGrowing(c, best)) {
      best = c;
    }
    if (r == NONE || isInsideWindow(r)) { break; }
    t = getValidAfter(c);
  }
  return best;
}
${props.outputDeclarations ?? ''}`;
}
