// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {type GPURasterBand} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {
  createRasterExtremaPyramidNodes,
  getGPURasterExtremaPyramidLayout,
  type GPURasterExtremaPyramidLayout
} from '../../gpu-raster/raster-pyramid/index';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings
} from './terrain-analysis-utils';
import {getTerrainSightLineWGSL, type GPUTerrainSightLineTraversal} from './terrain-sight-line';

/** Number of float32 values read from the `settings` of the sight-line contributors. */
export const GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH = 12;

/**
 * CPU-side description packed by {@link getGPUTerrainSightLineParameterValues}.
 *
 * Cell-size model: projected metres on a planar grid (Euclidean distance of the pixel offsets
 * scaled by `cellSize`). Curvature convention: drop at distance `d` is `c * d^2` with
 * `c = getGPUTerrainCurvatureCoefficient(k)`; mt-image uses `k = 0.13`, GDAL `-cc 0.85714` is
 * `k = 1 / 7`.
 */
export type GPUTerrainSightLineSettings = {
  /** Eye height above the terrain in elevation units. Defaults to 1.7. */
  observerHeight?: number;
  /** Height added to the target elevation. Defaults to 0. */
  targetHeight?: number;
  /** Ground radius in metres; `<= 0` means unlimited. Defaults to 0. */
  maxDistance?: number;
  /** `[x, y]` ground metres per pixel. Both must be finite and positive. */
  cellSize: readonly [number, number];
  /** Drop coefficient `c` in 1/metres. Defaults to 0. */
  curvatureCoefficient?: number;
  /** Constant tolerance band half-width in metres. Defaults to 0. */
  toleranceMeters?: number;
  /** Additional band half-width per kilometre of distance, in metres. Defaults to 0. */
  tolerancePerKilometer?: number;
  /** Metres at the target end that are not tested. Defaults to 0. */
  targetIgnoreDistance?: number;
  /** Fraction of the distance at the target end that is not tested. Defaults to 0. */
  targetIgnoreFraction?: number;
};

/**
 * Packs sight-line settings into the 12 float32 values
 * `[observerHeight, targetHeight, maxDistance, cellSizeX, cellSizeY, curvatureCoefficient,
 * toleranceMeters, tolerancePerKilometer, targetIgnoreDistance, targetIgnoreFraction, 0, 0]`.
 *
 * @throws If a cell size is not finite and positive, a tolerance value is negative or not finite,
 *   or `target` is too short.
 */
export function getGPUTerrainSightLineParameterValues(
  settings: GPUTerrainSightLineSettings,
  target: Float32Array = new Float32Array(GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH) {
    throw new Error('Terrain sight line settings target must hold 12 values');
  }
  if (!settings.cellSize.every(size => Number.isFinite(size) && size > 0)) {
    throw new Error('Terrain sight line cell size must be finite and positive');
  }
  const tolerance = [
    settings.toleranceMeters ?? 0,
    settings.tolerancePerKilometer ?? 0,
    settings.targetIgnoreDistance ?? 0,
    settings.targetIgnoreFraction ?? 0
  ];
  if (!tolerance.every(value => Number.isFinite(value) && value >= 0)) {
    throw new Error('Terrain sight line tolerance values must be finite and non-negative');
  }
  target.set([
    settings.observerHeight ?? 1.7,
    settings.targetHeight ?? 0,
    settings.maxDistance ?? 0,
    settings.cellSize[0],
    settings.cellSize[1],
    settings.curvatureCoefficient ?? 0,
    ...tolerance,
    0,
    0
  ]);
  return target;
}

/** Creates the optional pyramid nodes and binding shared by the sight-line contributors. @internal */
export function getSightLinePyramid<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  traversal: GPUTerrainSightLineTraversal | undefined,
  layout: GPURasterExtremaPyramidLayout | undefined,
  values: GraphDataView<'float32'>,
  validity: GraphDataView<'uint32'>
): {nodes: GPUCommandNode<Parameters>[]; binding?: WGSLKernelBinding} {
  if (traversal !== 'pyramid' || !layout) {
    return {nodes: []};
  }
  const combined = createTransientView(graph, `${id}-pyramid`, 'float32', 2 * layout.length);
  return {
    nodes: createRasterExtremaPyramidNodes(graph, {
      id: `${id}-pyramid`,
      layout,
      values,
      validity,
      combined
    }),
    binding: {name: 'pyramid', view: combined, type: 'f32', access: 'read'}
  };
}

/** Validates the compile-time traversal and returns the pyramid layout it needs. @internal */
export function getSightLinePyramidLayout(
  id: string,
  traversal: GPUTerrainSightLineTraversal | undefined,
  width: number,
  height: number
): GPURasterExtremaPyramidLayout | undefined {
  if (traversal !== undefined && traversal !== 'march' && traversal !== 'pyramid') {
    throw new Error(`${id} traversal must be 'march' or 'pyramid'`);
  }
  return traversal === 'pyramid'
    ? getGPURasterExtremaPyramidLayout(width, height, {firstBlockSize: 4, footprint: 'bilinear'})
    : undefined;
}

/**
 * Properties for {@link GPUTerrainLineOfSight}.
 *
 * Cell-size model: projected metres on a planar grid, see {@link GPUTerrainSightLineSettings}
 * (curvature conventions: mt-image `k = 0.13`, GDAL `-cc 0.85714` is `k = 1 / 7`).
 */
export type GPUTerrainLineOfSightProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-line-of-sight'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture. */
  elevation: GPURasterBand;
  /**
   * One row `[observerColumn, observerRow, targetColumn, targetRow]` per pair, in pixel-center
   * index space; fractional values allowed.
   */
  pairs: GraphDataView<'float32x4'>;
  /** Optional rows `[observerHeight, targetHeight]` per pair, overriding the settings. */
  pairHeights?: GraphDataView<'float32x2'>;
  /** Per-frame settings, see {@link getGPUTerrainSightLineParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** `'march'` (default) or `'pyramid'`, which gives bit-identical results. */
  traversal?: GPUTerrainSightLineTraversal;
  /** `GPU_TERRAIN_VISIBILITY` code per pair. */
  visibility: GraphDataView<'uint32'>;
  /**
   * Optional metres per pair by which the target could sink and stay visible (negative when
   * hidden): `(targetSlope - maxSampleSlope) * distance`. `3.4028234663852886e38` when no sample was
   * tested; NaN for `noData` and `outOfRange` pairs. Disables the early exit of hidden pairs.
   */
  clearance?: GraphDataView<'float32'>;
};

/**
 * Tests explicit observer-target pairs against an elevation tile.
 *
 * Observer and target must lie inside the grid with valid bilinear samples, otherwise the pair is
 * `noData`; the target elevation is the bilinear sample at the target plus the target height.
 * Then `distance > maxDistance > 0` is `outOfRange`, a zero distance is `visible`, and otherwise
 * the tolerance-aware sight-line model of {@link GPUTerrainViewshed} applies (codes
 * `hidden`, `visible`, `marginal`). One invocation handles one pair.
 *
 * Cell-size model: projected metres (planar), see {@link GPUTerrainSightLineSettings}.
 */
export class GPUTerrainLineOfSight implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainLineOfSightProps;
  /** Min-max pyramid layout when `traversal` is `'pyramid'`. */
  readonly pyramidLayout?: GPURasterExtremaPyramidLayout;

  constructor(props: GPUTerrainLineOfSightProps) {
    this.id = props.id ?? 'terrain-line-of-sight';
    this.props = props;
    const {id} = this;
    validateTerrainGrid(id, props.width, props.height);
    validatePackedView(props.pairs, ['float32x4'], `${id} pairs`);
    const pairCount = props.pairs.length;
    if (props.pairHeights) {
      validatePackedView(props.pairHeights, ['float32x2'], `${id} pairHeights`);
      if (props.pairHeights.length !== pairCount) {
        throw new Error(`${id} pairHeights must contain one row per pair`);
      }
    }
    validatePackedView(props.visibility, ['uint32'], `${id} visibility`);
    if (props.visibility.length !== pairCount) {
      throw new Error(`${id} visibility must contain one value per pair`);
    }
    if (props.clearance) {
      validatePackedView(props.clearance, ['float32'], `${id} clearance`);
      if (props.clearance.length !== pairCount) {
        throw new Error(`${id} clearance must contain one value per pair`);
      }
    }
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH);
    this.pyramidLayout = getSightLinePyramidLayout(id, props.traversal, props.width, props.height);
    validateTerrainBuffersDistinct(
      id,
      [props.visibility, props.clearance],
      [
        ...getTerrainBandViews(props.elevation),
        props.settings,
        props.pairs,
        ...(props.pairHeights ? [props.pairHeights] : [])
      ]
    );
  }

  /** Returns canonical elevation, optional pyramid, and the line-of-sight nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, pyramidLayout} = this;
    const {width, height} = props;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.settings,
      props.pairs,
      props.pairHeights,
      props.visibility,
      props.clearance
    ]);
    const source = getTerrainElevationNodes(graph, id, props.elevation, width, height, true);
    const values = source.band.storage.values as GraphDataView<'float32'>;
    const validity = source.band.validity as GraphDataView<'uint32'>;
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const pyramid = getSightLinePyramid(
      graph,
      id,
      props.traversal,
      pyramidLayout,
      values,
      validity
    );
    nodes.push(...pyramid.nodes);
    const bindings: WGSLKernelBinding[] = [
      {name: 'elevationValues', view: values, type: 'f32', access: 'read'},
      {name: 'elevationValidity', view: validity, type: 'u32', access: 'read'},
      ...(pyramid.binding ? [pyramid.binding] : []),
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'pairs', view: props.pairs, type: 'f32', access: 'read'},
      ...(props.pairHeights
        ? [
            {
              name: 'pairHeights',
              view: props.pairHeights,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : []),
      {name: 'visibilityValues', view: props.visibility, type: 'u32', access: 'read_write'},
      ...(props.clearance
        ? [
            {
              name: 'clearanceValues',
              view: props.clearance,
              type: 'f32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-line-of-sight`,
        operation: 'GPUTerrainLineOfSight',
        variant: props.traversal === 'pyramid' ? 'pyramid' : 'march',
        bindings,
        invocationCount: props.pairs.length,
        declarations: `${getTerrainSightLineWGSL({
          width,
          height,
          traversal: props.traversal ?? 'march',
          layout: pyramidLayout,
          clearance: Boolean(props.clearance)
        })}
fn isInsideGrid(position: vec2<f32>) -> bool {
  return isFiniteValue(position.x) && isFiniteValue(position.y) &&
    position.x >= 0.0 && position.y >= 0.0 &&
    position.x <= f32(WIDTH - 1u) && position.y <= f32(HEIGHT - 1u);
}`,
        body: `let pairBase = pairsOffset + 4u * index;
  let observer = vec2<f32>(pairs[pairBase], pairs[pairBase + 1u]);
  let targetPosition = vec2<f32>(pairs[pairBase + 2u], pairs[pairBase + 3u]);
  ${
    props.pairHeights
      ? `let observerHeight = pairHeights[pairHeightsOffset + 2u * index];
  let targetHeight = pairHeights[pairHeightsOffset + 2u * index + 1u];`
      : `let observerHeight = settings[settingsOffset];
  let targetHeight = settings[settingsOffset + 1u];`
  }
  let maxDistance = settings[settingsOffset + 2u];
  let cellSize = vec2<f32>(settings[settingsOffset + 3u], settings[settingsOffset + 4u]);
  let curvature = settings[settingsOffset + 5u];
  let tolerance = vec4<f32>(
    settings[settingsOffset + 6u], settings[settingsOffset + 7u],
    settings[settingsOffset + 8u], settings[settingsOffset + 9u]);
  let observerInside = isInsideGrid(observer);
  let targetInside = isInsideGrid(targetPosition);
  let observerSample = sampleElevation(select(vec2<f32>(0.0), observer, observerInside));
  let targetSample = sampleElevation(select(vec2<f32>(0.0), targetPosition, targetInside));
  var code = NO_DATA;
  // A runtime-dependent bit pattern: WGSL rejects constant-evaluated NaN.
  var clearance = bitcast<f32>(0x7fc00000u | (index & 0u));
  if (observerInside && targetInside && observerSample.y != 0.0 && targetSample.y != 0.0) {
    let result = traceSightLine(
      observer, targetPosition, observerSample.x + observerHeight, targetSample.x + targetHeight,
      cellSize, curvature, maxDistance, tolerance);
    code = result.code;
    if (result.code != OUT_OF_RANGE) {
      clearance = result.clearance;
    }
  }
  visibilityValues[visibilityValuesOffset + index] = code;${
    props.clearance
      ? `
  clearanceValues[clearanceValuesOffset + index] = clearance;`
      : ''
  }`
      })
    );
    return nodes;
  }
}
