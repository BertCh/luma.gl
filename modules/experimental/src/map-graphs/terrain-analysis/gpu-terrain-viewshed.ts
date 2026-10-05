// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {GPURasterBufferToTexture, type GPURasterBand} from '../../gpu-raster/index';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {captureGraphCommandNodes, validateGraphViewsBelongToGraph} from '../map-graph-utils';
import {
  createRasterExtremaPyramidNodes,
  getGPURasterExtremaPyramidLayout,
  type GPURasterExtremaPyramidLayout
} from '../raster-pyramid/index';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  TERRAIN_WGSL_HELPERS,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings,
  validateTerrainTexture
} from './terrain-analysis-utils';
import {getTerrainSightLineWGSL, type GPUTerrainSightLineTraversal} from './terrain-sight-line';

/**
 * Visibility codes written by the terrain sight-line recipes.
 *
 * `marginal` means the target is within the tolerance band of the sight line: neither clearly
 * hidden nor clearly visible.
 */
export const GPU_TERRAIN_VISIBILITY = {
  hidden: 0,
  visible: 1,
  outOfRange: 2,
  noData: 3,
  marginal: 4
} as const;

/** Number of float32 values read from `GPUTerrainViewshedProps.settings`. */
export const GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH = 8;

/** CPU-side description packed by {@link getGPUTerrainViewshedParameterValues}. */
export type GPUTerrainViewshedSettings = {
  /** Observer `[column, row]` in pixel-center index space; fractional values allowed. */
  observer: readonly [number, number];
  /** Eye height above the terrain in elevation units. Defaults to 1.7. */
  observerHeight?: number;
  /** Height added to every target cell. Defaults to 0. */
  targetHeight?: number;
  /** Ground radius in meters; `<= 0` means unlimited. Defaults to 0. */
  maxDistance?: number;
  /** `[x, y]` ground meters per pixel. Both must be finite and positive. */
  cellSize: readonly [number, number];
  /** Drop coefficient `c` in 1/meters: terrain at distance `d` is lowered by `c * d^2`. Defaults to 0. */
  curvatureCoefficient?: number;
};

/**
 * Packs viewshed settings into
 * `[observerColumn, observerRow, observerHeight, targetHeight, maxDistance, cellSizeX, cellSizeY, curvatureCoefficient]`.
 *
 * @throws If a cell size is not finite and positive, or `target` is too short.
 */
export function getGPUTerrainViewshedParameterValues(
  settings: GPUTerrainViewshedSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH) {
    throw new Error('Terrain viewshed settings target must hold 8 values');
  }
  if (!settings.cellSize.every(size => Number.isFinite(size) && size > 0)) {
    throw new Error('Terrain viewshed cell size must be finite and positive');
  }
  target.set([
    settings.observer[0],
    settings.observer[1],
    settings.observerHeight ?? 1.7,
    settings.targetHeight ?? 0,
    settings.maxDistance ?? 0,
    settings.cellSize[0],
    settings.cellSize[1],
    settings.curvatureCoefficient ?? 0
  ]);
  return target;
}

/** Number of float32 values read from `GPUTerrainViewshedProps.tolerance`. */
export const GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH = 4;

/**
 * CPU-side description packed by {@link getGPUTerrainVisibilityToleranceParameterValues}.
 *
 * All distances are projected metres on the same planar cell-size model as the settings.
 */
export type GPUTerrainVisibilityToleranceSettings = {
  /**
   * Constant half-width of the tolerance band in metres of sight-line height at the target.
   * mt-image uses 2. Defaults to 0.
   */
  toleranceMeters?: number;
  /**
   * Additional band half-width per kilometre of target distance, in metres. mt-image uses 1.
   * Defaults to 0.
   */
  tolerancePerKilometer?: number;
  /**
   * Samples closer than this many metres to the target are not tested (`viewPeaks` ignores the
   * last `max(150 m, 2 %)` of the ray). Defaults to 0.
   */
  targetIgnoreDistance?: number;
  /** Fraction of the target distance at the target end that is not tested. Defaults to 0. */
  targetIgnoreFraction?: number;
};

/**
 * Packs tolerance settings into
 * `[toleranceMeters, tolerancePerKilometer, targetIgnoreDistance, targetIgnoreFraction]`.
 *
 * The band half-width in slope units is `(toleranceMeters + tolerancePerKilometer * D / 1000) / D`
 * for target distance `D`. A target is `hidden` if the highest sample slope exceeds the target slope
 * plus the band, `visible` if it is at most the target slope minus the band, else `marginal`.
 *
 * @throws If a value is negative or not finite, or `target` is too short.
 */
export function getGPUTerrainVisibilityToleranceParameterValues(
  settings: GPUTerrainVisibilityToleranceSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH) {
    throw new Error('Terrain visibility tolerance target must hold 4 values');
  }
  const values = [
    settings.toleranceMeters ?? 0,
    settings.tolerancePerKilometer ?? 0,
    settings.targetIgnoreDistance ?? 0,
    settings.targetIgnoreFraction ?? 0
  ];
  if (!values.every(value => Number.isFinite(value) && value >= 0)) {
    throw new Error('Terrain visibility tolerance values must be finite and non-negative');
  }
  target.set(values);
  return target;
}

/**
 * Returns the combined earth curvature and refraction drop coefficient
 * `(1 - refractionCoefficient) / (2 * earthRadius)` for `curvatureCoefficient`.
 *
 * Convention: the drop at ground distance `d` is `c * d^2` with `c = (1 - k) / (2 R)`, `k` the
 * refraction coefficient and `R` the earth radius in metres. The default `k = 0.13` is the
 * mt-image / geodetic default. GDAL `gdal_viewshed -cc 0.85714` is `cc = 1 - k` with `k = 1 / 7`,
 * which is `getGPUTerrainCurvatureCoefficient(1 / 7)`.
 */
export function getGPUTerrainCurvatureCoefficient(
  refractionCoefficient: number = 0.13,
  earthRadius: number = 6371008.8
): number {
  return (1 - refractionCoefficient) / (2 * earthRadius);
}

/**
 * Properties for {@link GPUTerrainViewshed}.
 *
 * Topology: grid size, elevation format and calibration, traversal, and which outputs exist.
 * Per-frame: `settings` (observer, heights, radius, cell size, curvature), `tolerance` and
 * elevation contents.
 *
 * Cell-size model: the raster is in projected metres (planar); `settings.cellSize` gives ground
 * metres per pixel and distances are Euclidean in that plane. Web Mercator rasters must be
 * resampled or given a locally scaled cell size. Earth curvature follows the convention of
 * {@link getGPUTerrainCurvatureCoefficient} (mt-image `k = 0.13`; GDAL `-cc 0.85714` is
 * `k = 1 / 7`).
 */
export type GPUTerrainViewshedProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-viewshed'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture. */
  elevation: GPURasterBand;
  /** Per-frame settings with at least 8 float32 values, see {@link getGPUTerrainViewshedParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Optional `GPU_TERRAIN_VISIBILITY` code per pixel. */
  visibility?: GraphDataView<'uint32'>;
  /** Optional storage texture (`r32uint` or `rgba32uint`, channel 0) receiving the codes. */
  visibilityTexture?: GraphTextureView<'r32uint' | 'rgba32uint'>;
  /**
   * Compile-time traversal. `'march'` (default) evaluates every sample; `'pyramid'` additionally
   * builds a min-max pyramid of the elevation and skips samples that provably cannot change the
   * result. Both give bit-identical codes.
   */
  traversal?: GPUTerrainSightLineTraversal;
  /**
   * Optional per-frame tolerance band and target-ignore stretch with at least
   * {@link GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH} float32 values, see
   * {@link getGPUTerrainVisibilityToleranceParameterValues}. When absent all four are 0 and the
   * classic rule applies (no `marginal` codes).
   */
  tolerance?: GraphDataView<'float32'>;
};

/**
 * Computes a line-of-sight viewshed from one observer over an elevation tile.
 *
 * Every target cell marches the straight line to the observer at one-pixel steps with bilinear
 * elevation samples and is hidden if any sample rises above the observer-to-target sight line.
 * Precedence: no data (observer outside the grid or invalid, or target invalid), out of range,
 * then visibility. The cost is O(width * height * max(width, height)) samples for an unlimited
 * radius: crop tiles of 1024^2 and larger around the observer or set `maxDistance` to avoid GPU
 * watchdog timeouts, or use `traversal: 'pyramid'`.
 *
 * With `tolerance` (mt-image `integrity/viewshed.ts` uses 2 m plus 1 m per km), targets within the
 * band of the sight line get the code `marginal` and the last stretch before the target can be
 * ignored. With neither `traversal: 'pyramid'` nor `tolerance` the generated kernel is the
 * original one.
 */
export class GPUTerrainViewshed implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'terrain-viewshed';
  /** Validated properties. */
  readonly props: GPUTerrainViewshedProps;
  /** Min-max pyramid layout when `traversal` is `'pyramid'`. */
  readonly pyramidLayout?: GPURasterExtremaPyramidLayout;

  constructor(props: GPUTerrainViewshedProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    if (!props.visibility && !props.visibilityTexture) {
      throw new Error(`${id} requires at least one output`);
    }
    if (props.visibility) {
      validatePackedUint32View(props.visibility, `${id} visibility`);
      if (props.visibility.length !== pixelCount) {
        throw new Error(`${id} visibility must contain one value per pixel`);
      }
    }
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH);
    validateTerrainTexture(
      id,
      'visibilityTexture',
      props.visibilityTexture,
      ['r32uint', 'rgba32uint'],
      props.width,
      props.height
    );
    if (
      props.traversal !== undefined &&
      props.traversal !== 'march' &&
      props.traversal !== 'pyramid'
    ) {
      throw new Error(`${id} traversal must be 'march' or 'pyramid'`);
    }
    if (props.tolerance) {
      validateTerrainSettings(
        id,
        props.tolerance,
        GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH
      );
    }
    if (props.traversal === 'pyramid') {
      this.pyramidLayout = getGPURasterExtremaPyramidLayout(props.width, props.height, {
        firstBlockSize: 4,
        footprint: 'bilinear'
      });
    }
    validateTerrainBuffersDistinct(
      id,
      [props.visibility],
      [
        ...getTerrainBandViews(props.elevation),
        props.settings,
        ...(props.tolerance ? [props.tolerance] : [])
      ]
    );
  }

  /** Returns canonical elevation, viewshed, and optional texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, [props.visibilityTexture]);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.visibility,
      ...(props.tolerance ? [props.tolerance] : [])
    ]);
    const pixelCount = width * height;
    // Canonicalization always runs so the viewshed kernel has one float32 plus validity layout.
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const target =
      props.visibility ?? createTransientView(graph, `${id}-visibility`, 'uint32', pixelCount);
    const validity = source.band.validity as GraphDataView<'uint32'>;
    if (props.traversal === 'pyramid' || props.tolerance) {
      nodes.push(
        ...this._getSightLineNodes(
          graph,
          source.band.storage.values as GraphDataView<'float32'>,
          validity,
          target
        )
      );
    } else {
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-viewshed`,
          operation: 'GPUTerrainViewshed',
          variant: 'line-of-sight',
          bindings: [
            {
              name: 'elevationValues',
              view: source.band.storage.values as GraphDataView,
              type: 'f32',
              access: 'read'
            },
            {name: 'elevationValidity', view: validity, type: 'u32', access: 'read'},
            {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
            {name: 'visibilityValues', view: target, type: 'u32', access: 'read_write'}
          ],
          invocationCount: pixelCount,
          declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const HIDDEN: u32 = ${GPU_TERRAIN_VISIBILITY.hidden}u;
const VISIBLE: u32 = ${GPU_TERRAIN_VISIBILITY.visible}u;
const OUT_OF_RANGE: u32 = ${GPU_TERRAIN_VISIBILITY.outOfRange}u;
const NO_DATA: u32 = ${GPU_TERRAIN_VISIBILITY.noData}u;
${TERRAIN_WGSL_HELPERS}
fn isValidPixel(column: u32, row: u32) -> bool {
  return elevationValidity[elevationValidityOffset + row * WIDTH + column] != 0u;
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear elevation at a clamped pixel-center position; .y is 0 when any corner is invalid.
fn sampleElevation(position: vec2<f32>) -> vec2<f32> {
  let maximum = vec2<f32>(f32(WIDTH - 1u), f32(HEIGHT - 1u));
  let clamped = clamp(position, vec2<f32>(0.0), maximum);
  let base = vec2<u32>(floor(clamped));
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  let fraction = clamped - floor(clamped);
  if (!isValidPixel(base.x, base.y) || !isValidPixel(next.x, base.y) ||
      !isValidPixel(base.x, next.y) || !isValidPixel(next.x, next.y)) {
    return vec2<f32>(0.0, 0.0);
  }
  let top = mix(getElevation(base.x, base.y), getElevation(next.x, base.y), fraction.x);
  let bottom = mix(getElevation(base.x, next.y), getElevation(next.x, next.y), fraction.x);
  return vec2<f32>(mix(top, bottom, fraction.y), 1.0);
}`,
          body: `let column = index % WIDTH;
  let row = index / WIDTH;
  let observer = vec2<f32>(settings[settingsOffset], settings[settingsOffset + 1u]);
  let observerHeight = settings[settingsOffset + 2u];
  let targetHeight = settings[settingsOffset + 3u];
  let maxDistance = settings[settingsOffset + 4u];
  let cellSize = vec2<f32>(settings[settingsOffset + 5u], settings[settingsOffset + 6u]);
  let curvature = settings[settingsOffset + 7u];
  let observerInside = isFiniteValue(observer.x) && isFiniteValue(observer.y) &&
    observer.x >= 0.0 && observer.y >= 0.0 &&
    observer.x <= f32(WIDTH - 1u) && observer.y <= f32(HEIGHT - 1u);
  let observerSample = sampleElevation(select(vec2<f32>(0.0), observer, observerInside));
  if (!observerInside || observerSample.y == 0.0 || !isValidPixel(column, row)) {
    visibilityValues[visibilityValuesOffset + index] = NO_DATA;
    return;
  }
  let observerElevation = observerSample.x + observerHeight;
  let delta = vec2<f32>(f32(column), f32(row)) - observer;
  let targetDistance = length(delta * cellSize);
  if (maxDistance > 0.0 && targetDistance > maxDistance) {
    visibilityValues[visibilityValuesOffset + index] = OUT_OF_RANGE;
    return;
  }
  if (targetDistance == 0.0) {
    visibilityValues[visibilityValuesOffset + index] = VISIBLE;
    return;
  }
  let targetElevation = getElevation(column, row) + targetHeight -
    curvature * targetDistance * targetDistance;
  let targetSlope = (targetElevation - observerElevation) / targetDistance;
  let stepCount = u32(ceil(max(abs(delta.x), abs(delta.y))));
  var visible = true;
  for (var stepIndex = 1u; stepIndex < stepCount; stepIndex++) {
    let fraction = f32(stepIndex) / f32(stepCount);
    let elevationSample = sampleElevation(observer + delta * fraction);
    // Invalid samples never block.
    if (elevationSample.y == 0.0) { continue; }
    let sampleDistance = targetDistance * fraction;
    let sampleSlope =
      (elevationSample.x - curvature * sampleDistance * sampleDistance - observerElevation) / sampleDistance;
    // A grazing (equal) sight line stays visible.
    if (sampleSlope > targetSlope) {
      visible = false;
      break;
    }
  }
  visibilityValues[visibilityValuesOffset + index] = select(HIDDEN, VISIBLE, visible);`
        })
      );
    }
    if (props.visibilityTexture) {
      const texture = props.visibilityTexture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-visibility-texture`,
            input: {
              id: `${id}-visibility-band`,
              format: 'uint32',
              storage: {kind: 'buffer', values: target}
            },
            output: texture,
            channel: 0
          }).addToGraph(graph)
        )
      );
    }
    return nodes;
  }

  /** Tolerance-aware kernel, with the pyramid nodes when `traversal` is `'pyramid'`. */
  private _getSightLineNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    values: GraphDataView<'float32'>,
    validity: GraphDataView<'uint32'>,
    target: GraphDataView<'uint32'>
  ): GPUCommandNode<Parameters>[] {
    const {id, props, pyramidLayout} = this;
    const {width, height} = props;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const bindings: MapGraphKernelBinding[] = [
      {name: 'elevationValues', view: values, type: 'f32', access: 'read'},
      {name: 'elevationValidity', view: validity, type: 'u32', access: 'read'}
    ];
    if (pyramidLayout) {
      const combined = createTransientView(
        graph,
        `${id}-pyramid`,
        'float32',
        2 * pyramidLayout.length
      );
      nodes.push(
        ...createRasterExtremaPyramidNodes(graph, {
          id: `${id}-pyramid`,
          layout: pyramidLayout,
          values,
          validity,
          combined
        })
      );
      bindings.push({name: 'pyramid', view: combined, type: 'f32', access: 'read'});
    }
    bindings.push({name: 'settings', view: props.settings, type: 'f32', access: 'read'});
    if (props.tolerance) {
      bindings.push({name: 'toleranceBand', view: props.tolerance, type: 'f32', access: 'read'});
    }
    bindings.push({name: 'visibilityValues', view: target, type: 'u32', access: 'read_write'});
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-viewshed`,
        operation: 'GPUTerrainViewshed',
        variant: props.traversal === 'pyramid' ? 'pyramid' : 'tolerance',
        bindings,
        invocationCount: width * height,
        declarations: getTerrainSightLineWGSL({
          width,
          height,
          traversal: props.traversal ?? 'march',
          layout: pyramidLayout,
          clearance: false
        }),
        body: `let column = index % WIDTH;
  let row = index / WIDTH;
  let observer = vec2<f32>(settings[settingsOffset], settings[settingsOffset + 1u]);
  let observerHeight = settings[settingsOffset + 2u];
  let targetHeight = settings[settingsOffset + 3u];
  let maxDistance = settings[settingsOffset + 4u];
  let cellSize = vec2<f32>(settings[settingsOffset + 5u], settings[settingsOffset + 6u]);
  let curvature = settings[settingsOffset + 7u];
  ${
    props.tolerance
      ? `let toleranceValues = vec4<f32>(
    toleranceBand[toleranceBandOffset], toleranceBand[toleranceBandOffset + 1u],
    toleranceBand[toleranceBandOffset + 2u], toleranceBand[toleranceBandOffset + 3u]);`
      : 'let toleranceValues = vec4<f32>(0.0);'
  }
  let observerInside = isFiniteValue(observer.x) && isFiniteValue(observer.y) &&
    observer.x >= 0.0 && observer.y >= 0.0 &&
    observer.x <= f32(WIDTH - 1u) && observer.y <= f32(HEIGHT - 1u);
  let observerSample = sampleElevation(select(vec2<f32>(0.0), observer, observerInside));
  if (!observerInside || observerSample.y == 0.0 || !isValidPixel(column, row)) {
    visibilityValues[visibilityValuesOffset + index] = NO_DATA;
    return;
  }
  let result = traceSightLine(
    observer, vec2<f32>(f32(column), f32(row)), observerSample.x + observerHeight,
    getElevation(column, row) + targetHeight, cellSize, curvature, maxDistance, toleranceValues);
  visibilityValues[visibilityValuesOffset + index] = result.code;`
      })
    );
    return nodes;
  }
}
