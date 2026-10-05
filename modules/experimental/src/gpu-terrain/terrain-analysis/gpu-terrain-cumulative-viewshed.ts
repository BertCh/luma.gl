// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {type GPURasterBand} from '../../gpu-raster/index';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import type {
  GPURasterExtremaPyramidLayout,
  GPURasterExtremaPyramidOutput
} from '../../gpu-raster/raster-pyramid/index';
import {
  getSightLinePyramid,
  getSightLinePyramidLayout,
  GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH
} from './gpu-terrain-line-of-sight';
import {
  getTerrainBandViews,
  getTerrainElevationNodes,
  validateTerrainBandBelongsToGraph,
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings
} from './terrain-analysis-utils';
import {getTerrainSightLineWGSL, type GPUTerrainSightLineTraversal} from './terrain-sight-line';

/**
 * Properties for {@link GPUTerrainCumulativeViewshed}.
 *
 * Cell-size model: projected metres on a planar grid, see `GPUTerrainSightLineSettings`
 * (curvature conventions: mt-image `k = 0.13`, GDAL `-cc 0.85714` is `k = 1 / 7`).
 */
export type GPUTerrainCumulativeViewshedProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-cumulative-viewshed'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Elevation band, buffer or texture. */
  elevation: GPURasterBand;
  /** Observers as `[column, row]` in pixel-center index space; fractional values allowed. */
  observers: GraphDataView<'float32x2'>;
  /** Optional eye height per observer overriding `settings` (one value per observer). */
  observerHeights?: GraphDataView<'float32'>;
  /**
   * Per-frame settings in the 12-float sight-line layout, see
   * `getGPUTerrainSightLineParameterValues`. Observer and target heights apply unless
   * `observerHeights` is given.
   */
  settings: GraphDataView<'float32'>;
  /** `'march'` (default) or `'pyramid'`, which gives bit-identical counts. */
  traversal?: GPUTerrainSightLineTraversal;
  /**
   * Optional prebuilt min-max pyramid to read instead of building one inside this contributor.
   * Requires `traversal: 'pyramid'`. Pass `GPURasterExtremaPyramid.output` of a pyramid created with
   * `combined` over the same elevation, `footprint: 'bilinear'` and the same `width x height`
   * (otherwise the constructor throws), added to the same graph before this contributor or run in
   * an earlier graph. One pyramid then serves every consumer and is rebuilt only when the
   * elevation changes. Results are bit-identical to a self-built pyramid and to `'march'`.
   */
  pyramid?: GPURasterExtremaPyramidOutput;
  /**
   * Observers handled by one dispatch. Defaults to `max(1, floor(2^22 / pixelCount))`; lower it
   * to keep dispatches short on large grids.
   */
  observersPerDispatch?: number;
  /** Per pixel, the number of observers from which the pixel is `visible`. */
  visibleCount: GraphDataView<'uint32'>;
  /** Optional per pixel count of observers from which the pixel is `marginal`. */
  marginalCount?: GraphDataView<'uint32'>;
};

/**
 * Counts, for every pixel, how many observers see it (the analogue of GDAL `gdal_viewshed`
 * cumulative mode).
 *
 * Observers are the dispatch dimension: batches of `observersPerDispatch` observers run one
 * kernel each, one invocation per (observer, pixel), and accumulate with `atomicAdd`. Observers
 * outside the grid or on invalid cells contribute nothing, and neither do invalid targets or
 * targets beyond `maxDistance`. The target elevation is the pixel value plus the target height
 * (not bilinear), exactly as in {@link GPUTerrainViewshed}, so the counts equal the sum of the
 * single-observer viewsheds.
 *
 * Cell-size model: projected metres (planar). Cost is
 * O(observers * width * height * max(width, height)) samples for `'march'`.
 */
export class GPUTerrainCumulativeViewshed implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTerrainCumulativeViewshedProps;
  /** Min-max pyramid layout when `traversal` is `'pyramid'`. */
  readonly pyramidLayout?: GPURasterExtremaPyramidLayout;
  /** Observers per dispatch after applying the default. */
  readonly observersPerDispatch: number;

  constructor(props: GPUTerrainCumulativeViewshedProps) {
    this.id = props.id ?? 'terrain-cumulative-viewshed';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    validatePackedView(props.observers, ['float32x2'], `${id} observers`);
    if (props.observerHeights) {
      validatePackedView(props.observerHeights, ['float32'], `${id} observerHeights`);
      if (props.observerHeights.length !== props.observers.length) {
        throw new Error(`${id} observerHeights must contain one value per observer`);
      }
    }
    validatePackedView(props.visibleCount, ['uint32'], `${id} visibleCount`);
    if (props.visibleCount.length !== pixelCount) {
      throw new Error(`${id} visibleCount must contain one value per pixel`);
    }
    if (props.marginalCount) {
      validatePackedView(props.marginalCount, ['uint32'], `${id} marginalCount`);
      if (props.marginalCount.length !== pixelCount) {
        throw new Error(`${id} marginalCount must contain one value per pixel`);
      }
    }
    validateTerrainSettings(id, props.settings, GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH);
    const perDispatch = props.observersPerDispatch ?? Math.max(1, Math.floor(2 ** 22 / pixelCount));
    if (!Number.isSafeInteger(perDispatch) || perDispatch < 1) {
      throw new Error(`${id} observersPerDispatch must be a positive integer`);
    }
    this.observersPerDispatch = perDispatch;
    this.pyramidLayout = getSightLinePyramidLayout(
      id,
      props.traversal,
      props.width,
      props.height,
      props.pyramid
    );
    validateTerrainBuffersDistinct(
      id,
      [props.visibleCount, props.marginalCount],
      [
        ...getTerrainBandViews(props.elevation),
        props.settings,
        props.observers,
        ...(props.observerHeights ? [props.observerHeights] : [])
      ]
    );
  }

  /** Returns canonical elevation, optional pyramid, zero fills, and one kernel per batch. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, pyramidLayout, observersPerDispatch} = this;
    const {width, height} = props;
    const pixelCount = width * height;
    validateTerrainBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.pyramid?.combined,
      props.settings,
      props.observers,
      props.observerHeights,
      props.visibleCount,
      props.marginalCount
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
      validity,
      props.pyramid
    );
    nodes.push(...pyramid.nodes);
    for (const [name, view] of [
      ['visible', props.visibleCount],
      ['marginal', props.marginalCount]
    ] as const) {
      if (view) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-${name}-count-zero`,
            operation: 'GPUTerrainCumulativeViewshed',
            view,
            type: 'u32',
            value: '0u'
          })
        );
      }
    }
    const bindings: WGSLKernelBinding[] = [
      {name: 'elevationValues', view: values, type: 'f32', access: 'read'},
      {name: 'elevationValidity', view: validity, type: 'u32', access: 'read'},
      ...(pyramid.binding ? [pyramid.binding] : []),
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'observers', view: props.observers, type: 'f32', access: 'read'},
      ...(props.observerHeights
        ? [
            {
              name: 'observerHeights',
              view: props.observerHeights,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : []),
      {name: 'visibleCount', view: props.visibleCount, type: 'atomic<u32>', access: 'read_write'},
      ...(props.marginalCount
        ? [
            {
              name: 'marginalCount',
              view: props.marginalCount,
              type: 'atomic<u32>' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ];
    const sightLine = getTerrainSightLineWGSL({
      width,
      height,
      traversal: props.traversal ?? 'march',
      layout: pyramidLayout,
      clearance: false
    });
    const observerCount = props.observers.length;
    for (let first = 0, batch = 0; first < observerCount; first += observersPerDispatch, batch++) {
      const count = Math.min(observersPerDispatch, observerCount - first);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-${batch}`,
          operation: 'GPUTerrainCumulativeViewshed',
          variant: props.traversal === 'pyramid' ? 'pyramid' : 'march',
          bindings,
          invocationCount: count * pixelCount,
          declarations: `${sightLine}
const FIRST_OBSERVER: u32 = ${first}u;
const PIXEL_COUNT: u32 = ${pixelCount}u;
fn isInsideGrid(position: vec2<f32>) -> bool {
  return isFiniteValue(position.x) && isFiniteValue(position.y) &&
    position.x >= 0.0 && position.y >= 0.0 &&
    position.x <= f32(WIDTH - 1u) && position.y <= f32(HEIGHT - 1u);
}`,
          body: `let observerIndex = FIRST_OBSERVER + index / PIXEL_COUNT;
  let pixel = index % PIXEL_COUNT;
  let column = pixel % WIDTH;
  let row = pixel / WIDTH;
  let observer = vec2<f32>(
    observers[observersOffset + 2u * observerIndex],
    observers[observersOffset + 2u * observerIndex + 1u]);
  ${
    props.observerHeights
      ? 'let observerHeight = observerHeights[observerHeightsOffset + observerIndex];'
      : 'let observerHeight = settings[settingsOffset];'
  }
  let targetHeight = settings[settingsOffset + 1u];
  let maxDistance = settings[settingsOffset + 2u];
  let cellSize = vec2<f32>(settings[settingsOffset + 3u], settings[settingsOffset + 4u]);
  let curvature = settings[settingsOffset + 5u];
  let tolerance = vec4<f32>(
    settings[settingsOffset + 6u], settings[settingsOffset + 7u],
    settings[settingsOffset + 8u], settings[settingsOffset + 9u]);
  if (!isInsideGrid(observer) || !isValidPixel(column, row)) {
    return;
  }
  let observerSample = sampleElevation(observer);
  if (observerSample.y == 0.0) {
    return;
  }
  let result = traceSightLine(
    observer, vec2<f32>(f32(column), f32(row)), observerSample.x + observerHeight,
    getElevation(column, row) + targetHeight, cellSize, curvature, maxDistance, tolerance);
  if (result.code == VISIBLE) {
    atomicAdd(&visibleCount[visibleCountOffset + pixel], 1u);
  }${
    props.marginalCount
      ? ` else if (result.code == MARGINAL) {
    atomicAdd(&marginalCount[marginalCountOffset + pixel], 1u);
  }`
      : ''
  }`
        })
      );
    }
    return nodes;
  }
}
