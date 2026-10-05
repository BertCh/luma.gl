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
import type {MapGraphKernelBinding} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {captureGraphCommandNodes, validateGraphViewsBelongToGraph} from '../map-graph-utils';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings,
  validateTerrainTexture
} from '../terrain-analysis/terrain-analysis-utils';
import {GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES} from './gpu-solar-shadow-mask';
import {
  createTerrainHorizonSweepNode,
  getTerrainSweepLineGeometry,
  TERRAIN_SWEEP_MAX_EXTENT
} from './terrain-horizon-sweep';
import {
  validateTerrainIlluminationCellSizeMode,
  validateTerrainIlluminationRowDirection,
  validateTerrainIlluminationView,
  type GPUTerrainIlluminationCellSizeMode
} from './terrain-illumination-utils';

/** Number of float32 values read from `GPUTerrainCastShadowProps.settings`. */
export const GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH = 16;

/**
 * CPU-side description packed by {@link getGPUTerrainCastShadowParameterValues}.
 *
 * Cell-size model: `cellSize` is interpreted by the recipe's `cellSizeMode` as projected metres
 * (`uniform`), equatorial Web Mercator metres or degrees with latitude-dependent spacing
 * (`web-mercator`, `geographic`, using `northEdge` and `southEdge`).
 */
export type GPUTerrainCastShadowSettings = {
  /** `[x, y]` cell size: meters (uniform), equatorial Web Mercator meters, or degrees (geographic). */
  cellSize: readonly [number, number];
  /** Direction of the sun, degrees clockwise from north. See `getSolarPosition`. */
  azimuthDegrees: number;
  /** Elevation of the sun center above the astronomical horizon in degrees. */
  altitudeDegrees: number;
  /**
   * Angular radius of the solar disk in degrees: 0 gives hard shadows. Defaults to
   * {@link GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES}.
   */
  angularRadiusDegrees?: number;
  /** Elevation multiplier. Defaults to 1. */
  zFactor?: number;
  /** Earth curvature and refraction drop in 1/meters, see `GPUTerrainHorizonSettings`. Defaults to 0. */
  curvatureCoefficient?: number;
  /** Top edge of row 0: normalized Web Mercator y in `[0, 1]` or latitude degrees. Defaults to 0. */
  northEdge?: number;
  /** Bottom edge of the last row, same units as `northEdge`. Defaults to 0. */
  southEdge?: number;
  /** Ground search radius in meters; `<= 0` is unlimited. Defaults to 0. */
  maximumDistance?: number;
};

/**
 * Packs settings into the 16-float layout read by {@link GPUTerrainCastShadow}:
 * `[cellX, cellY, zFactor, curvature, northEdge, southEdge, maximumDistance, azimuthDegrees,
 * altitudeDegrees, angularRadiusDegrees, xMajor (0/1), slopeFixed, travelSign, stepPixelLength, 0, 0]`.
 *
 * The digital line family of the sun direction is computed here on the CPU in float64 (direction
 * `(sin(azimuth), +-cos(azimuth))` exactly as in `getGPUTerrainHorizonDirection`, but not rounded to
 * float32), so changing the sun never recompiles the graph. `rowDirection` must match the recipe.
 *
 * @throws If a cell size is not finite and positive, the angular radius is negative or not finite,
 * the azimuth is not finite, or `target` is too short.
 */
export function getGPUTerrainCastShadowParameterValues(
  settings: GPUTerrainCastShadowSettings,
  rowDirection: 'south' | 'north' = 'south',
  target: Float32Array = new Float32Array(GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH) {
    throw new Error('Terrain cast shadow settings target must hold 16 values');
  }
  if (!settings.cellSize.every(size => Number.isFinite(size) && size > 0)) {
    throw new Error('Terrain cast shadow cell size must be finite and positive');
  }
  const angularRadius = settings.angularRadiusDegrees ?? GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES;
  if (!Number.isFinite(angularRadius) || angularRadius < 0) {
    throw new Error('Terrain cast shadow angular radius must be finite and non-negative');
  }
  if (!Number.isFinite(settings.azimuthDegrees) || !Number.isFinite(settings.altitudeDegrees)) {
    throw new Error('Terrain cast shadow azimuth and altitude must be finite');
  }
  const azimuth = (settings.azimuthDegrees * Math.PI) / 180;
  const northRowSign = rowDirection === 'south' ? -1 : 1;
  const snap = (value: number) => (Math.abs(value) < 1e-9 ? 0 : value);
  // The line geometry does not depend on the grid extent, only the line count does.
  const geometry = getTerrainSweepLineGeometry(
    [snap(Math.sin(azimuth)), snap(northRowSign * Math.cos(azimuth))],
    1,
    1
  );
  target.set([
    settings.cellSize[0],
    settings.cellSize[1],
    settings.zFactor ?? 1,
    settings.curvatureCoefficient ?? 0,
    settings.northEdge ?? 0,
    settings.southEdge ?? 0,
    settings.maximumDistance ?? 0,
    settings.azimuthDegrees,
    settings.altitudeDegrees,
    angularRadius,
    geometry.xMajor ? 1 : 0,
    geometry.slopeFixed,
    geometry.travelSign,
    geometry.stepPixelLength,
    0,
    0
  ]);
  return target;
}

/**
 * Properties for {@link GPUTerrainCastShadow}.
 *
 * Topology: grid size, elevation format and calibration, `maximumRadius`, `cellSizeMode`,
 * `rowDirection`, and which outputs exist. Per-frame: `settings` (sun, disk radius, cell size, z
 * factor, curvature, maximum distance) and elevation contents. Cell-size model: see
 * {@link GPUTerrainCastShadowSettings}.
 */
export type GPUTerrainCastShadowProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-cast-shadow'`. */
  id?: string;
  /** Grid width in pixels, at most 32767. */
  width: number;
  /** Grid height in pixels, at most 32767. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame settings with 16 float32 values, see {@link getGPUTerrainCastShadowParameterValues}. */
  settings: GraphDataView<'float32'>;
  /**
   * Search radius in pixels along the sun line (counted as `k * hypot(1, slope)`), a positive
   * integer. Also the required tile halo. Defaults to the tile diagonal, `ceil(hypot(width,
   * height))`, which covers the whole tile for every sun azimuth.
   */
  maximumRadius?: number;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPUTerrainIlluminationCellSizeMode;
  /** Direction in which the row index increases. Defaults to `'south'` (north-up rasters). */
  rowDirection?: 'south' | 'north';
  /** Optional fraction of the solar disk above the local horizon, in `[0, 1]`. */
  sunVisibility?: GraphDataView<'float32'>;
  /** Optional horizon angle in degrees along the sun azimuth. */
  horizonAngle?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where the center elevation is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
  /** Optional storage texture (`r32float` or `rgba32float`, channel 0) receiving `sunVisibility`. */
  sunVisibilityTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
};

/**
 * Single-sun terrain cast shadow without a horizon map.
 *
 * One exact hull sweep (see `getTerrainHorizonSweepNode`) along the digital line family of the
 * per-frame sun azimuth yields the horizon angle toward the sun for every pixel, in amortised O(1)
 * per pixel and exact against brute force (up to the line discretisation: the ray follows a
 * Bresenham-style digital line rather than a bilinear ray). `sunVisibility` is the area fraction
 * of the solar disk above that horizon, the same formula as `GPUSolarShadowMask`: with
 * `u = clamp((altitude - horizon) / radius, -1, 1)` it is `1 - (acos(u) - u * sqrt(1 - u^2)) / PI`,
 * a hard step when the radius is 0, and 0 when the whole disk is below the astronomical horizon.
 *
 * Sun position, disk radius, cell size, z factor, curvature, and maximum distance are per-frame
 * settings; the line family is packed by the CPU settings packer, so animating time of day never
 * recompiles. The line count depends on the azimuth and is unknown at compile time, so
 * `width + height` invocations are dispatched and surplus ones exit.
 *
 * Softness comes from the physical solar disk radius. mt-image's propagated-occluder softness
 * (`smooth(-w, w, H + bias - O)`, a lateral-diffusion approximation) was not adopted: the hull
 * sweep is equally O(1) amortised and exact against brute force. Pixels with an invalid centre
 * elevation receive NaN. Pixels closer than `maximumRadius` to the tile edge see a truncated
 * horizon: pass a tile with a `maximumRadius` halo for seamless results.
 */
export class GPUTerrainCastShadow implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'terrain-cast-shadow';
  /** Validated properties. */
  readonly props: GPUTerrainCastShadowProps;
  /** Search radius in pixels. */
  readonly maximumRadius: number;
  /** Receptive field in pixels (`GPURasterHaloStage` contract). */
  readonly requiredHalo: number;

  constructor(props: GPUTerrainCastShadowProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    if (props.width > TERRAIN_SWEEP_MAX_EXTENT || props.height > TERRAIN_SWEEP_MAX_EXTENT) {
      throw new Error(`${id} width and height must not exceed ${TERRAIN_SWEEP_MAX_EXTENT}`);
    }
    const maximumRadius = props.maximumRadius ?? Math.ceil(Math.hypot(props.width, props.height));
    if (!Number.isSafeInteger(maximumRadius) || maximumRadius < 1) {
      throw new Error(`${id} maximumRadius must be a positive integer`);
    }
    this.maximumRadius = maximumRadius;
    this.requiredHalo = maximumRadius;
    if (
      !props.sunVisibility &&
      !props.horizonAngle &&
      !props.validity &&
      !props.sunVisibilityTexture
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainIlluminationView(
      id,
      'sunVisibility',
      props.sunVisibility,
      'float32',
      pixelCount
    );
    validateTerrainIlluminationView(id, 'horizonAngle', props.horizonAngle, 'float32', pixelCount);
    validateTerrainIlluminationView(id, 'validity', props.validity, 'uint32', pixelCount);
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH);
    validateTerrainTexture(
      id,
      'sunVisibilityTexture',
      props.sunVisibilityTexture,
      ['r32float', 'rgba32float'],
      props.width,
      props.height
    );
    validateTerrainIlluminationCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validateTerrainIlluminationRowDirection(id, props.rowDirection ?? 'south');
    validateTerrainBuffersDistinct(
      id,
      [props.sunVisibility, props.horizonAngle, props.validity],
      [...getTerrainBandViews(props.elevation), props.settings]
    );
  }

  /** Returns elevation canonicalization, the sweep node, and the optional texture node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, [props.sunVisibilityTexture]);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.sunVisibility,
      props.horizonAngle,
      props.validity
    ]);
    const pixelCount = width * height;
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const visibilityTarget =
      props.sunVisibility ??
      (props.sunVisibilityTexture
        ? createTransientView(graph, `${id}-sun-visibility`, 'float32', pixelCount)
        : undefined);
    const outputBindings: MapGraphKernelBinding[] = [];
    if (visibilityTarget) {
      outputBindings.push({
        name: 'visibilityValues',
        view: visibilityTarget,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.horizonAngle) {
      outputBindings.push({
        name: 'horizonValues',
        view: props.horizonAngle,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.validity) {
      outputBindings.push({
        name: 'validityValues',
        view: props.validity,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createTerrainHorizonSweepNode<Parameters>(graph, {
        id: `${id}-sweep`,
        width,
        height,
        geometry: {kind: 'settings', index: 10},
        maximumRadius: this.maximumRadius,
        cellSizeMode: props.cellSizeMode ?? 'uniform',
        zFactorSign: 1,
        elevationValues: source.band.storage.values as GraphDataView<'float32'>,
        elevationValidity: source.band.validity as GraphDataView<'uint32'>,
        settings: props.settings,
        hull: createTransientView(graph, `${id}-hull`, 'uint32', pixelCount),
        outputBindings,
        outputWGSL: `let isValid = isFiniteValue(horizonAngle);
      let invalidValue = getNaN(pixel);
      let altitudeDegrees = settings[settingsOffset + 8u];
      let radiusDegrees = settings[settingsOffset + 9u];
      var visibility = 0.0;
      if (altitudeDegrees + radiusDegrees > 0.0) {
        if (radiusDegrees > 0.0) {
          let u = clamp((altitudeDegrees - horizonAngle) / radiusDegrees, -1.0, 1.0);
          visibility = clamp(1.0 - (acos(u) - u * sqrt(max(1.0 - u * u, 0.0))) / PI, 0.0, 1.0);
        } else {
          visibility = select(0.0, 1.0, altitudeDegrees > horizonAngle);
        }
      }
      ${visibilityTarget ? 'visibilityValues[visibilityValuesOffset + pixel] = select(invalidValue, visibility, isValid);' : ''}
      ${props.horizonAngle ? 'horizonValues[horizonValuesOffset + pixel] = horizonAngle;' : ''}
      ${props.validity ? 'validityValues[validityValuesOffset + pixel] = select(0u, 1u, isValid);' : ''}`
      })
    );
    if (props.sunVisibilityTexture && visibilityTarget) {
      const texture = props.sunVisibilityTexture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-sun-visibility-texture`,
            input: {
              id: `${id}-sun-visibility-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: visibilityTarget}
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
