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
  GPURasterGradient,
  type GPURasterBand,
  type GPURasterBorderMode
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
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings,
  validateTerrainTexture
} from '../terrain-analysis/terrain-analysis-utils';
import {
  getTerrainIlluminationGroundCellWGSL,
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainIlluminationCellSizeMode,
  validateTerrainIlluminationRowDirection,
  validateTerrainIlluminationView,
  type GPUTerrainIlluminationCellSizeMode
} from './terrain-illumination-utils';

/** Maximum number of lights in {@link GPUReliefShadingSettings.lights}. */
export const GPU_RELIEF_SHADING_MAX_LIGHT_COUNT = 8;

/** Maximum number of stops in {@link GPUReliefShadingSettings.elevationStops}. */
export const GPU_RELIEF_SHADING_MAX_STOP_COUNT = 8;

/** Number of float32 values read from `GPUReliefShadingProps.settings`. */
export const GPU_RELIEF_SHADING_PARAMETER_LENGTH = 80;

/** Pivot grey of the elevation-dependent contrast term: `shade = mix(pivot, shade, k)`. */
export const GPU_RELIEF_SHADING_CONTRAST_PIVOT = 0.72;

/** Default maximum light azimuth swing of `'imhof-swing'` in degrees (mt-image `IMHOF_SWING_MAX`). */
const DEFAULT_IMHOF_SWING_DEGREES = 65;

/** Settings slot of the first light (`azimuth, altitude, weight` triples). @internal */
const LIGHT_OFFSET = 20;
/** Settings slot of the first elevation stop (`elevation, r, g, b` quadruples). @internal */
const STOP_OFFSET = 44;
/** Settings slot of `curvatureStrength`, followed by the three contrast slots. @internal */
const CURVATURE_STRENGTH_SLOT = 76;

/** One directional light of {@link GPUReliefShadingSettings}. */
export type GPUReliefShadingLight = {
  /** Direction the light comes from, degrees clockwise from north. */
  azimuthDegrees: number;
  /** Light elevation above the horizon in degrees. Defaults to 45. */
  altitudeDegrees?: number;
  /** Weight in `'fixed'` weighting. Defaults to 1. */
  weight?: number;
};

/** One elevation tint stop of {@link GPUReliefShadingSettings}. */
export type GPUReliefShadingStop = {
  /** Elevation in calibrated elevation units (before `zFactor`). */
  elevation: number;
  /** Linear `[r, g, b]` in `[0, 1]`. */
  color: readonly [number, number, number];
};

/**
 * How light weights combine:
 * - `'fixed'`: `sum(w_i * shade_i) / sum(w_i)` with the lights' own weights.
 * - `'aspect'`: USGS multidirectional oblique weighting (Mark 1992, GDAL `-multidirectional`):
 *   per-pixel `w_i = sin^2(aspect - azimuth_i)`, equal weights on flat pixels.
 * - `'imhof-swing'`: the lights' fixed weights, but each light's azimuth is swung per pixel,
 *   `azimuth' = azimuth + swing * sin(aspect - azimuth) * smoothstep(0.05, 0.4, sin(slope))`, where
 *   `aspect` is the downslope aspect and `swing` is `imhofSwingDegrees`. Slopes facing the light or
 *   straight away keep the light azimuth, side-facing slopes rotate the light toward them, which
 *   lights the faces a fixed 315 degree light leaves black (Imhof, Swiss relief shading). This ports
 *   only the azimuth swing of mt-image `imhofSwungLight`, not its mix with a multidirectional
 *   shade nor its `1 / sin(altitude)` normalisation. With `imhofSwingDegrees` 0 it equals `'fixed'`.
 */
export type GPUReliefShadingWeighting = 'fixed' | 'aspect' | 'imhof-swing';

/** CPU-side description packed by {@link getGPUReliefShadingParameterValues}. */
export type GPUReliefShadingSettings = {
  /** `[x, y]` cell size: meters (uniform), equatorial Web Mercator meters, or degrees (geographic). */
  cellSize: readonly [number, number];
  /** Elevation multiplier applied to derivatives. Defaults to 1. */
  zFactor?: number;
  /** Top edge of row 0: normalized Web Mercator y in `[0, 1]` or latitude degrees. Defaults to 0. */
  northEdge?: number;
  /** Bottom edge of the last row, same units as `northEdge`. Defaults to 0. */
  southEdge?: number;
  /**
   * Lights, or `'mdow'` for the USGS set: azimuths 225, 270, 315, 360 at 30 degrees altitude with
   * `'aspect'` weighting. Defaults to `'mdow'`.
   */
  lights?: 'mdow' | readonly GPUReliefShadingLight[];
  /** Light weighting. Defaults to `'aspect'` for `'mdow'` and `'fixed'` for custom lights. */
  lightWeighting?: GPUReliefShadingWeighting;
  /**
   * Largest azimuth swing of `'imhof-swing'` weighting in degrees (settings slot 19). Defaults to 65
   * for `'imhof-swing'` and 0 otherwise; ignored by the other weightings. 0 disables the swing.
   */
  imhofSwingDegrees?: number;
  /**
   * Multiplier of the optional `curvature` raster added to the shade (settings slot 76):
   * `shade += curvatureStrength * curvature`. Defaults to 1; ignored when no curvature is bound.
   */
  curvatureStrength?: number;
  /**
   * Elevation (calibrated units, before `zFactor`) where the elevation-dependent contrast starts
   * fading in (settings slot 77). Defaults to 0.
   */
  contrastLowElevation?: number;
  /**
   * Elevation where the contrast reaches full strength (slot 78). When it is not above
   * `contrastLowElevation` the contrast steps at the low elevation. Defaults to 0.
   */
  contrastHighElevation?: number;
  /**
   * Strength in `[0, 1]` of the elevation-dependent contrast (slot 79): with
   * `k = mix(1 - strength, 1, smoothstep(low, high, z))` the shade becomes
   * `mix(GPU_RELIEF_SHADING_CONTRAST_PIVOT, shade, k)`, so low ground is flattened toward grey
   * and high ground keeps full relief. Defaults to 0, which is off and leaves output unchanged.
   */
  contrastStrength?: number;
  /** Blend of the hillshade into `relief`: 0 ignores it, 1 uses it fully. Defaults to 1. */
  hillshadeStrength?: number;
  /** Blend of the sky-view factor into `relief`. Defaults to 1 (only when the input is bound). */
  skyViewStrength?: number;
  /** Multiplier of the texture shade added to `relief`. Defaults to 0.5 (only when bound). */
  textureShadeStrength?: number;
  /** Final `relief` multiplier before clamping to `[0, 1]`. Defaults to 1. */
  exposure?: number;
  /** Imhof aspect tint strength in `[0, 1]`. Defaults to 0.3. */
  tintStrength?: number;
  /** Tint toward which lit slopes shift. Defaults to `[1, 0.95, 0.85]`. */
  warmColor?: readonly [number, number, number];
  /** Tint toward which shaded slopes shift. Defaults to `[0.8, 0.85, 1]`. */
  coolColor?: readonly [number, number, number];
  /** Ascending elevation tint stops; empty gives a 0.85 grey base. Defaults to `[]`. */
  elevationStops?: readonly GPUReliefShadingStop[];
};

/** Lights of the USGS multidirectional oblique-weighted (MDOW) hillshade. */
export const GPU_RELIEF_SHADING_MDOW_LIGHTS: readonly GPUReliefShadingLight[] = [
  {azimuthDegrees: 225, altitudeDegrees: 30},
  {azimuthDegrees: 270, altitudeDegrees: 30},
  {azimuthDegrees: 315, altitudeDegrees: 30},
  {azimuthDegrees: 360, altitudeDegrees: 30}
];

/**
 * Packs settings into the 80-float layout read by {@link GPUReliefShading}:
 *
 * | slots | values |
 * | --- | --- |
 * | 0-4 | cellSizeX, cellSizeY, zFactor, northEdge, southEdge |
 * | 5-6 | lightCount, weighting (0 fixed, 1 aspect, 2 imhof-swing) |
 * | 7-11 | hillshadeStrength, skyViewStrength, textureShadeStrength, exposure, tintStrength |
 * | 12-14, 15 | warm rgb, stopCount |
 * | 16-18 | cool rgb |
 * | 19 | imhofSwingDegrees |
 * | 20-43 | 8 lights x (azimuth, altitude, weight) |
 * | 44-75 | 8 stops x (elevation, r, g, b) |
 * | 76 | curvatureStrength |
 * | 77-79 | contrastLowElevation, contrastHighElevation, contrastStrength |
 *
 * @throws If there are more than 8 lights or stops, stops are not ascending, or `target` is too short.
 */
export function getGPUReliefShadingParameterValues(
  settings: GPUReliefShadingSettings,
  target: Float32Array = new Float32Array(GPU_RELIEF_SHADING_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_RELIEF_SHADING_PARAMETER_LENGTH) {
    throw new Error('Relief shading settings target must hold 80 values');
  }
  const lightSet = settings.lights ?? 'mdow';
  const lights = lightSet === 'mdow' ? GPU_RELIEF_SHADING_MDOW_LIGHTS : lightSet;
  if (lights.length < 1 || lights.length > GPU_RELIEF_SHADING_MAX_LIGHT_COUNT) {
    throw new Error('Relief shading needs 1 to 8 lights');
  }
  const stops = settings.elevationStops ?? [];
  if (stops.length > GPU_RELIEF_SHADING_MAX_STOP_COUNT) {
    throw new Error('Relief shading supports at most 8 elevation stops');
  }
  for (let index = 1; index < stops.length; index++) {
    if (!(stops[index].elevation > stops[index - 1].elevation)) {
      throw new Error('Relief shading elevation stops must be strictly ascending');
    }
  }
  const weighting = settings.lightWeighting ?? (lightSet === 'mdow' ? 'aspect' : 'fixed');
  const warm = settings.warmColor ?? [1, 0.95, 0.85];
  const cool = settings.coolColor ?? [0.8, 0.85, 1];
  target.fill(0, 0, GPU_RELIEF_SHADING_PARAMETER_LENGTH);
  target.set([
    settings.cellSize[0],
    settings.cellSize[1],
    settings.zFactor ?? 1,
    settings.northEdge ?? 0,
    settings.southEdge ?? 0,
    lights.length,
    weighting === 'imhof-swing' ? 2 : weighting === 'aspect' ? 1 : 0,
    settings.hillshadeStrength ?? 1,
    settings.skyViewStrength ?? 1,
    settings.textureShadeStrength ?? 0.5,
    settings.exposure ?? 1,
    settings.tintStrength ?? 0.3,
    warm[0],
    warm[1],
    warm[2],
    stops.length,
    cool[0],
    cool[1],
    cool[2],
    settings.imhofSwingDegrees ?? (weighting === 'imhof-swing' ? DEFAULT_IMHOF_SWING_DEGREES : 0)
  ]);
  for (const [index, light] of lights.entries()) {
    target.set(
      [light.azimuthDegrees, light.altitudeDegrees ?? 45, light.weight ?? 1],
      LIGHT_OFFSET + index * 3
    );
  }
  for (const [index, stop] of stops.entries()) {
    target.set([stop.elevation, ...stop.color], STOP_OFFSET + index * 4);
  }
  target.set(
    [
      settings.curvatureStrength ?? 1,
      settings.contrastLowElevation ?? 0,
      settings.contrastHighElevation ?? 0,
      settings.contrastStrength ?? 0
    ],
    CURVATURE_STRENGTH_SLOT
  );
  return target;
}

/**
 * Properties for {@link GPUReliefShading}.
 *
 * Topology: grid size, elevation format and calibration, which inputs and outputs exist,
 * `cellSizeMode`, `rowDirection`, and `borderMode`. Per-frame: `settings` (lights, weighting,
 * blend strengths, tints, elevation ramp) and input contents.
 */
export type GPUReliefShadingProps = {
  /** Prefix for node and transient IDs. Defaults to `'relief-shading'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture; scale, offset, nodata, and validity are honored. */
  elevation: GPURasterBand;
  /** Per-frame settings with at least 80 float32 values, see {@link getGPUReliefShadingParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Optional sky-view factor per pixel, for example from `GPUTerrainHorizon`. */
  skyViewFactor?: GraphDataView<'float32'>;
  /** Optional texture shade per pixel, for example from `GPUTextureShading`. */
  textureShade?: GraphDataView<'float32'>;
  /**
   * Optional curvature raster from any producer, positive on convex ridges and negative in
   * hollows. Added to the shade as `curvatureStrength * curvature` (settings slot 76); NaN makes
   * the pixel invalid. With a curvature raster, a small pre-kernel folds it and `textureShade`
   * into one transient `detail` buffer only when the compose kernel would otherwise exceed 8
   * storage bindings (every optional input and output bound).
   */
  curvature?: GraphDataView<'float32'>;
  /** Cell size interpretation. Defaults to `'uniform'`. */
  cellSizeMode?: GPUTerrainIlluminationCellSizeMode;
  /** Direction in which the row index increases. Defaults to `'south'` (north-up rasters). */
  rowDirection?: 'south' | 'north';
  /** Sobel border treatment forwarded to `GPURasterGradient`. Defaults to `'clamp'`. */
  borderMode?: GPURasterBorderMode;
  /** Optional multidirectional hillshade per pixel in `[0, 1]`. */
  hillshade?: GraphDataView<'float32'>;
  /** Optional blended relief luminance per pixel in `[0, 1]`. */
  relief?: GraphDataView<'float32'>;
  /** Optional packed RGBA8 color per pixel (`pack4x8unorm`, red in the low byte, alpha 255). */
  color?: GraphDataView<'uint32'>;
  /** Optional per-pixel 1 where every output is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
  /**
   * Topology switch that adds the `'imhof-swing'` hillshade node (default false). Without it the
   * `'imhof-swing'` weighting behaves like `'aspect'`. The node runs every frame but exits at once
   * unless the per-frame weighting is `'imhof-swing'`, and it keeps the other weightings bit-identical.
   */
  imhofSwing?: boolean;
  /** Optional storage texture (`r32float` or `rgba32float`, channel 0) receiving the hillshade. */
  hillshadeTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
  /** Optional storage texture (`r32float` or `rgba32float`, channel 0) receiving the relief. */
  reliefTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
};

/**
 * Computes a multidirectional hillshade and a Swiss/Imhof-style cartographic relief blend.
 *
 * Normals come from Horn's 3x3 method (two Sobel `GPURasterGradient` passes, as in
 * `GPUTerrainDerivatives`). Each light contributes `max(dot(n, l), 0)`; weights are fixed or the
 * USGS aspect-dependent `sin^2(aspect - azimuth)` (see {@link GPUReliefShadingWeighting}). The
 * relief blend is
 *
 * ```
 * shade = mix(1, hillshade, hillshadeStrength) * mix(1, svf, skyViewStrength)
 *         + textureShadeStrength * textureShade
 * relief = clamp(shade * exposure, 0, 1)
 * t = clamp(2 * hillshade - 1, -1, 1)
 * tint = t >= 0 ? mix(1, warmColor, t * tintStrength) : mix(1, coolColor, -t * tintStrength)
 * color = clamp(elevationRamp(z) * relief * tint, 0, 1)
 * ```
 *
 * where unbound optional inputs drop their terms. A bound `curvature` raster adds
 * `curvatureStrength * curvature` to `shade`, and a non-zero `contrastStrength` then replaces
 * `shade` by `mix(0.72, shade, k)` with `k = mix(1 - strength, 1, smoothstep(low, high, z))`
 * (elevation-dependent contrast), both before `exposure`. `'imhof-swing'` weighting swings each
 * light's azimuth toward the aspect (see {@link GPUReliefShadingWeighting}). Every style value is per-frame, so restyling
 * never recompiles. Invalid pixels (an invalid 3x3 neighborhood or a NaN optional input) receive
 * NaN, color 0, and validity 0. For seamless tiles pass a one-pixel halo.
 */
export class GPUReliefShading implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUReliefShadingProps;
  /** Receptive field in pixels (`GPURasterHaloStage` contract). */
  readonly requiredHalo = 1;

  constructor(props: GPUReliefShadingProps) {
    this.id = props.id ?? 'relief-shading';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    if (
      !props.hillshade &&
      !props.relief &&
      !props.color &&
      !props.validity &&
      !props.hillshadeTexture &&
      !props.reliefTexture
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view] of [
      ['skyViewFactor', props.skyViewFactor],
      ['textureShade', props.textureShade],
      ['curvature', props.curvature],
      ['hillshade', props.hillshade],
      ['relief', props.relief]
    ] as const) {
      validateTerrainIlluminationView(id, name, view, 'float32', pixelCount);
    }
    validateTerrainIlluminationView(id, 'color', props.color, 'uint32', pixelCount);
    validateTerrainIlluminationView(id, 'validity', props.validity, 'uint32', pixelCount);
    validateTerrainSettings(id, props.settings, GPU_RELIEF_SHADING_PARAMETER_LENGTH);
    for (const [name, texture] of [
      ['hillshadeTexture', props.hillshadeTexture],
      ['reliefTexture', props.reliefTexture]
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
    validateTerrainIlluminationCellSizeMode(id, props.cellSizeMode ?? 'uniform');
    validateTerrainIlluminationRowDirection(id, props.rowDirection ?? 'south');
    validateTerrainBuffersDistinct(
      id,
      [props.hillshade, props.relief, props.color, props.validity],
      [
        ...getTerrainBandViews(props.elevation),
        props.settings,
        props.skyViewFactor,
        props.textureShade,
        props.curvature
      ]
    );
  }

  /** Returns elevation canonicalization, two Sobel passes, hillshade, compose, and texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, [
      props.hillshadeTexture,
      props.reliefTexture
    ]);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.skyViewFactor,
      props.textureShade,
      props.curvature,
      props.hillshade,
      props.relief,
      props.color,
      props.validity
    ]);
    const pixelCount = width * height;
    const borderMode = props.borderMode ?? 'clamp';
    const needsCompose = Boolean(
      props.relief || props.color || props.validity || props.reliefTexture
    );
    // The compose kernel reads calibrated elevation for the tint ramp.
    const source = getTerrainElevationNodes(
      graph,
      id,
      props.elevation,
      width,
      height,
      needsCompose
    );
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const gradients = (['x', 'y'] as const).map(direction => {
      const values = createTransientView(
        graph,
        `${id}-gradient-${direction}-values`,
        'float32',
        pixelCount
      );
      const validity = createTransientView(
        graph,
        `${id}-gradient-${direction}-validity`,
        'uint32',
        pixelCount
      );
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterGradient({
            id: `${id}-gradient-${direction}`,
            width,
            height,
            input: source.band,
            output: values,
            outputValidity: validity,
            operator: 'sobel',
            direction,
            borderMode,
            scale: 1
          }).addToGraph(graph)
        )
      );
      return {values, validity};
    });
    const hillshadeTarget =
      props.hillshade ?? createTransientView(graph, `${id}-hillshade`, 'float32', pixelCount);
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-hillshade`,
        operation: 'GPUReliefShading',
        variant: `hillshade-${cellSizeMode}`,
        bindings: [
          {
            name: 'gradientX',
            view: gradients[0].values,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'gradientY',
            view: gradients[1].values,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'gradientValidity',
            view: gradients[0].validity,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'settings',
            view: props.settings,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'hillshadeValues',
            view: hillshadeTarget,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: pixelCount,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const ROW_NORTH_SIGN: f32 = ${(props.rowDirection ?? 'south') === 'south' ? '-1.0' : '1.0'};
const LIGHT_OFFSET: u32 = ${LIGHT_OFFSET}u;
const MAX_LIGHT_COUNT: u32 = ${GPU_RELIEF_SHADING_MAX_LIGHT_COUNT}u;
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${getTerrainIlluminationGroundCellWGSL(cellSizeMode, {
  settingsName: 'settings',
  cellSizeIndex: 0,
  northEdgeIndex: 3
})}`,
        body: `let row = index / WIDTH;
  let rawX = gradientX[gradientXOffset + index];
  let rawY = gradientY[gradientYOffset + index];
  let groundCell = getGroundCellSize(row);
  let zFactor = settings[settingsOffset + 2u];
  // Sobel responds 8 to a unit-per-pixel ramp, so Horn's derivative is raw / (8 * cell).
  let eastGradient = zFactor * rawX / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor * rawY / (8.0 * groundCell.y);
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  let isFlat = eastGradient == 0.0 && northGradient == 0.0;
  // Downslope aspect clockwise from north.
  let aspect = atan2(-eastGradient, -northGradient);
  let lightCount = min(u32(max(settings[settingsOffset + 5u], 0.0)), MAX_LIGHT_COUNT);
  let aspectWeighting = settings[settingsOffset + 6u] != 0.0;
  var weightedSum = 0.0;
  var weightSum = 0.0;
  for (var light = 0u; light < lightCount; light++) {
    let slot = settingsOffset + LIGHT_OFFSET + light * 3u;
    let azimuth = settings[slot] * DEGREES_TO_RADIANS;
    let altitude = settings[slot + 1u] * DEGREES_TO_RADIANS;
    var weight = settings[slot + 2u];
    if (aspectWeighting) {
      let angle = sin(aspect - azimuth);
      weight = select(angle * angle, 1.0, isFlat);
    }
    let direction = vec3<f32>(sin(azimuth) * cos(altitude), cos(azimuth) * cos(altitude), sin(altitude));
    weightedSum += weight * max(dot(normal, direction), 0.0);
    weightSum += weight;
  }
  let shade = select(0.0, weightedSum / weightSum, weightSum > 0.0);
  let isValid = gradientValidity[gradientValidityOffset + index] != 0u &&
    isFiniteValue(rawX) && isFiniteValue(rawY) && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(shade);
  hillshadeValues[hillshadeValuesOffset + index] = select(getNaN(index), shade, isValid);`
      })
    );
    // A separate node (only with `imhofSwing`) keeps the fixed and aspect hillshade kernel above
    // bit-identical: it overwrites the hillshade only while the per-frame weighting is 'imhof-swing'.
    if (props.imhofSwing) {
      nodes.push(
        getSwingNode(graph, {
          id: `${id}-hillshade-swing`,
          width,
          height,
          cellSizeMode,
          rowDirection: props.rowDirection ?? 'south',
          gradientX: gradients[0].values,
          gradientY: gradients[1].values,
          gradientValidity: gradients[0].validity,
          settings: props.settings,
          hillshade: hillshadeTarget
        })
      );
    }
    const reliefTarget =
      props.relief ??
      (props.reliefTexture
        ? createTransientView(graph, `${id}-relief`, 'float32', pixelCount)
        : undefined);
    let detailTexture = props.textureShade;
    // Compose binds elevation, hillshade, settings plus the optional inputs and outputs; a ninth
    // binding would exceed the 8 storage buffers, so curvature is folded into `detail` then.
    const composeBindingCount =
      3 +
      [
        props.skyViewFactor,
        props.textureShade,
        props.curvature,
        props.relief ?? props.reliefTexture,
        props.color,
        props.validity
      ].filter(Boolean).length;
    const foldCurvature = needsCompose && Boolean(props.curvature) && composeBindingCount > 8;
    if (foldCurvature) {
      detailTexture = createTransientView(graph, `${id}-detail`, 'float32', pixelCount);
      nodes.push(
        getDetailNode(graph, {
          id: `${id}-detail`,
          pixelCount,
          settings: props.settings,
          textureShade: props.textureShade,
          curvature: props.curvature as GraphDataView<'float32'>,
          detail: detailTexture
        })
      );
    }
    if (needsCompose) {
      nodes.push(
        getComposeNode(graph, {
          id: `${id}-compose`,
          pixelCount,
          elevation: source.band.storage.values as GraphDataView<'float32'>,
          hillshade: hillshadeTarget,
          settings: props.settings,
          skyViewFactor: props.skyViewFactor,
          textureShade: detailTexture,
          textureShadeFolded: foldCurvature,
          curvature: foldCurvature ? undefined : props.curvature,
          relief: reliefTarget,
          color: props.color,
          validity: props.validity
        })
      );
    }
    for (const [name, target, texture] of [
      ['hillshade', hillshadeTarget, props.hillshadeTexture],
      ['relief', reliefTarget, props.reliefTexture]
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

/**
 * Builds the `'imhof-swing'` hillshade kernel. It recomputes Horn's normal exactly as the main
 * hillshade kernel and, only when settings slot 6 is 2, overwrites the hillshade with the
 * fixed-weight sum of lights whose azimuths are swung by
 * `swing * sin(aspect - azimuth) * smoothstep(0.05, 0.4, sin(slope))`.
 */
function getSwingNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    cellSizeMode: GPUTerrainIlluminationCellSizeMode;
    rowDirection: 'south' | 'north';
    gradientX: GraphDataView<'float32'>;
    gradientY: GraphDataView<'float32'>;
    gradientValidity: GraphDataView<'uint32'>;
    settings: GraphDataView<'float32'>;
    hillshade: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUReliefShading',
    variant: `hillshade-swing-${props.cellSizeMode}`,
    bindings: [
      {name: 'gradientX', view: props.gradientX, type: 'f32', access: 'read'},
      {name: 'gradientY', view: props.gradientY, type: 'f32', access: 'read'},
      {name: 'gradientValidity', view: props.gradientValidity, type: 'u32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'hillshadeValues', view: props.hillshade, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.width * props.height,
    declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const ROW_NORTH_SIGN: f32 = ${props.rowDirection === 'south' ? '-1.0' : '1.0'};
const LIGHT_OFFSET: u32 = ${LIGHT_OFFSET}u;
const MAX_LIGHT_COUNT: u32 = ${GPU_RELIEF_SHADING_MAX_LIGHT_COUNT}u;
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${getTerrainIlluminationGroundCellWGSL(props.cellSizeMode, {
  settingsName: 'settings',
  cellSizeIndex: 0,
  northEdgeIndex: 3
})}`,
    body: `if (settings[settingsOffset + 6u] != 2.0) {
    return;
  }
  let row = index / WIDTH;
  let rawX = gradientX[gradientXOffset + index];
  let rawY = gradientY[gradientYOffset + index];
  let groundCell = getGroundCellSize(row);
  let zFactor = settings[settingsOffset + 2u];
  let eastGradient = zFactor * rawX / (8.0 * groundCell.x);
  let northGradient = ROW_NORTH_SIGN * zFactor * rawY / (8.0 * groundCell.y);
  let normal = normalize(vec3<f32>(-eastGradient, -northGradient, 1.0));
  let aspect = atan2(-eastGradient, -northGradient);
  let lightCount = min(u32(max(settings[settingsOffset + 5u], 0.0)), MAX_LIGHT_COUNT);
  let swingRadians = settings[settingsOffset + 19u] * DEGREES_TO_RADIANS;
  let swingFactor = smoothstep(0.05, 0.4, length(normal.xy));
  var weightedSum = 0.0;
  var weightSum = 0.0;
  for (var light = 0u; light < lightCount; light++) {
    let slot = settingsOffset + LIGHT_OFFSET + light * 3u;
    let baseAzimuth = settings[slot] * DEGREES_TO_RADIANS;
    let azimuth = baseAzimuth + swingRadians * sin(aspect - baseAzimuth) * swingFactor;
    let altitude = settings[slot + 1u] * DEGREES_TO_RADIANS;
    let weight = settings[slot + 2u];
    let direction = vec3<f32>(sin(azimuth) * cos(altitude), cos(azimuth) * cos(altitude), sin(altitude));
    weightedSum += weight * max(dot(normal, direction), 0.0);
    weightSum += weight;
  }
  let shade = select(0.0, weightedSum / weightSum, weightSum > 0.0);
  let isValid = gradientValidity[gradientValidityOffset + index] != 0u &&
    isFiniteValue(rawX) && isFiniteValue(rawY) && groundCell.x > 0.0 && groundCell.y > 0.0 &&
    isFiniteValue(shade);
  hillshadeValues[hillshadeValuesOffset + index] = select(getNaN(index), shade, isValid);`
  });
}

/**
 * Folds the texture shade and curvature terms into one buffer: `textureShadeStrength * texture +
 * curvatureStrength * curvature`, NaN when either input is NaN.
 */
function getDetailNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    pixelCount: number;
    settings: GraphDataView<'float32'>;
    textureShade?: GraphDataView<'float32'>;
    curvature: GraphDataView<'float32'>;
    detail: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
    {name: 'curvatureValues', view: props.curvature, type: 'f32', access: 'read'}
  ];
  if (props.textureShade) {
    bindings.push({name: 'textureShade', view: props.textureShade, type: 'f32', access: 'read'});
  }
  bindings.push({name: 'detailValues', view: props.detail, type: 'f32', access: 'read_write'});
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUReliefShading',
    variant: 'detail',
    bindings,
    invocationCount: props.pixelCount,
    declarations: TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
    body: `var detail = settings[settingsOffset + ${CURVATURE_STRENGTH_SLOT}u] * curvatureValues[curvatureValuesOffset + index];
  ${
    props.textureShade
      ? 'detail += settings[settingsOffset + 9u] * textureShade[textureShadeOffset + index];'
      : ''
  }
  detailValues[detailValuesOffset + index] = select(getNaN(index), detail, isFiniteValue(detail));`
  });
}

/** Builds the relief, color, and validity compose kernel. */
function getComposeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    pixelCount: number;
    elevation: GraphDataView<'float32'>;
    hillshade: GraphDataView<'float32'>;
    settings: GraphDataView<'float32'>;
    skyViewFactor?: GraphDataView<'float32'>;
    textureShade?: GraphDataView<'float32'>;
    /** True when `textureShade` is the folded detail buffer, already scaled by its strengths. */
    textureShadeFolded?: boolean;
    /** Curvature bound directly (when it is not folded into `textureShade`). */
    curvature?: GraphDataView<'float32'>;
    relief?: GraphDataView<'float32'>;
    color?: GraphDataView<'uint32'>;
    validity?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {
      name: 'elevationValues',
      view: props.elevation,
      type: 'f32',
      access: 'read'
    },
    {
      name: 'hillshadeValues',
      view: props.hillshade,
      type: 'f32',
      access: 'read'
    },
    {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
  ];
  if (props.skyViewFactor) {
    bindings.push({
      name: 'skyView',
      view: props.skyViewFactor,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.textureShade) {
    bindings.push({
      name: 'textureShade',
      view: props.textureShade,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.curvature) {
    bindings.push({
      name: 'curvatureValues',
      view: props.curvature,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.relief) {
    bindings.push({
      name: 'reliefValues',
      view: props.relief,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.color) {
    bindings.push({
      name: 'colorValues',
      view: props.color,
      type: 'u32',
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
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: 'GPUReliefShading',
    variant: 'compose',
    bindings,
    invocationCount: props.pixelCount,
    declarations: `const STOP_OFFSET: u32 = ${STOP_OFFSET}u;
const CONTRAST_PIVOT: f32 = ${getWGSLFloatLiteral(GPU_RELIEF_SHADING_CONTRAST_PIVOT)};
const MAX_STOP_COUNT: u32 = ${GPU_RELIEF_SHADING_MAX_STOP_COUNT}u;
${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
fn getSettingsColor(slot: u32) -> vec3<f32> {
  return vec3<f32>(settings[settingsOffset + slot], settings[settingsOffset + slot + 1u],
    settings[settingsOffset + slot + 2u]);
}
fn getElevationColor(elevation: f32) -> vec3<f32> {
  let stopCount = min(u32(max(settings[settingsOffset + 15u], 0.0)), MAX_STOP_COUNT);
  if (stopCount == 0u) { return vec3<f32>(0.85); }
  var color = getSettingsColor(STOP_OFFSET + 1u);
  if (elevation <= settings[settingsOffset + STOP_OFFSET]) { return color; }
  for (var stop = 1u; stop < stopCount; stop++) {
    let slot = STOP_OFFSET + stop * 4u;
    let lowerElevation = settings[settingsOffset + slot - 4u];
    let upperElevation = settings[settingsOffset + slot];
    color = getSettingsColor(slot + 1u);
    if (elevation < upperElevation) {
      let blend = (elevation - lowerElevation) / (upperElevation - lowerElevation);
      return mix(getSettingsColor(slot - 3u), color, blend);
    }
  }
  return color;
}`,
    body: `let hillshade = hillshadeValues[hillshadeValuesOffset + index];
  var shade = mix(1.0, hillshade, settings[settingsOffset + 7u]);
  var isValid = isFiniteValue(hillshade);
  ${
    props.skyViewFactor
      ? `let skyViewFactor = skyView[skyViewOffset + index];
  isValid = isValid && isFiniteValue(skyViewFactor);
  shade *= mix(1.0, skyViewFactor, settings[settingsOffset + 8u]);`
      : ''
  }
  ${
    props.textureShade
      ? `let textureValue = textureShade[textureShadeOffset + index];
  isValid = isValid && isFiniteValue(textureValue);
  shade += ${props.textureShadeFolded ? '' : 'settings[settingsOffset + 9u] * '}textureValue;`
      : ''
  }
  ${
    props.curvature
      ? `let curvatureValue = curvatureValues[curvatureValuesOffset + index];
  isValid = isValid && isFiniteValue(curvatureValue);
  shade += settings[settingsOffset + ${CURVATURE_STRENGTH_SLOT}u] * curvatureValue;`
      : ''
  }
  let contrastStrength = settings[settingsOffset + ${CURVATURE_STRENGTH_SLOT + 3}u];
  if (contrastStrength != 0.0) {
    let low = settings[settingsOffset + ${CURVATURE_STRENGTH_SLOT + 1}u];
    let high = settings[settingsOffset + ${CURVATURE_STRENGTH_SLOT + 2}u];
    let elevation = elevationValues[elevationValuesOffset + index];
    let ramp = select(select(0.0, 1.0, elevation >= low), smoothstep(low, high, elevation), high > low);
    shade = mix(CONTRAST_PIVOT, shade, mix(1.0 - contrastStrength, 1.0, ramp));
    isValid = isValid && isFiniteValue(shade);
  }
  let relief = clamp(shade * settings[settingsOffset + 10u], 0.0, 1.0);
  ${props.relief ? 'reliefValues[reliefValuesOffset + index] = select(getNaN(index), relief, isValid);' : ''}
  ${
    props.color
      ? `let tintStrength = settings[settingsOffset + 11u];
  let lightness = clamp(2.0 * hillshade - 1.0, -1.0, 1.0);
  var tint = mix(vec3<f32>(1.0), getSettingsColor(12u), lightness * tintStrength);
  if (lightness < 0.0) {
    tint = mix(vec3<f32>(1.0), getSettingsColor(16u), -lightness * tintStrength);
  }
  let base = getElevationColor(elevationValues[elevationValuesOffset + index]);
  let color = clamp(base * relief * tint, vec3<f32>(0.0), vec3<f32>(1.0));
  colorValues[colorValuesOffset + index] = select(0u, pack4x8unorm(vec4<f32>(color, 1.0)), isValid);`
      : ''
  }
  ${props.validity ? 'validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);' : ''}`
  });
}
