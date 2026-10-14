// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {type GPURasterBand, type GPURasterMetadata} from '../../gpu-raster/index';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import type {
  GPURasterExtremaPyramidLayout,
  GPURasterExtremaPyramidOutput
} from '../../gpu-raster/raster-pyramid/index';
import {
  getSightLinePyramid,
  getSightLinePyramidLayout,
  GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH
} from './gpu-terrain-line-of-sight';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterGrid,
  validateRasterParameters
} from '../../gpu-raster/raster-band-utils';
import {getTerrainSightLineWGSL, type GPUTerrainSightLineTraversal} from './terrain-sight-line';

/**
 * Default sight-line samples per {@link GPUTerrainCumulativeViewshed} dispatch (2^28), which keeps
 * one dispatch well inside GPU watchdog limits.
 */
export const GPU_TERRAIN_CUMULATIVE_VIEWSHED_SAMPLES_PER_DISPATCH = 2 ** 28;

/** Elements between batch words: 256 bytes, so every batch binds at element offset 0. */
const BATCH_STRIDE = 64;

/** Caller-owned outputs of {@link GPUTerrainCumulativeViewshed}, one count per pixel. */
export type GPUTerrainCumulativeViewshedOutput = {
  /** Per pixel, the number of observers from which the pixel is `visible`. */
  visibleCount: GraphDataView<'uint32'>;
  /** Optional per pixel count of observers from which the pixel is `marginal`. */
  marginalCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUTerrainCumulativeViewshed}.
 *
 * Cell-size model: projected metres on a planar grid, see `GPUTerrainLineOfSightSettings`
 * (curvature conventions: mt-image `k = 0.13`, GDAL `-cc 0.85714` is `k = 1 / 7`).
 */
export type GPUTerrainCumulativeViewshedProps = {
  /** Prefix for node and transient IDs. Defaults to `'terrain-cumulative-viewshed'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Elevation band, buffer or texture. */
  elevation: GPURasterBand;
  /** Observers as `[column, row]` in pixel-center index space; fractional values allowed. */
  observers: GraphDataView<'float32x2'>;
  /** Optional eye height per observer overriding `parameters` (one value per observer). */
  observerHeights?: GraphDataView<'float32'>;
  /**
   * Per-frame parameters in the 12-float line-of-sight layout: this contributor uses the packer
   * of {@link GPUTerrainLineOfSight}, `getGPUTerrainLineOfSightParameterValues`, because the
   * observer height, target height, distance, cell size, curvature and tolerance semantics are
   * shared. Observer and target heights apply unless `observerHeights` is given.
   */
  parameters: GraphDataView<'float32'>;
  /** `'march'` (default) or `'pyramid'`, which gives bit-identical counts. */
  traversal?: GPUTerrainSightLineTraversal;
  /**
   * Optional prebuilt min-max pyramid to read instead of building one inside this contributor.
   * Requires `traversal: 'pyramid'`. Pass `GPURasterExtremaPyramid.output` of a pyramid created with
   * `combined` over the same elevation, `footprint: 'bilinear'` and the same grid size
   * (otherwise the constructor throws), added to the same graph before this contributor or run in
   * an earlier graph. One pyramid then serves every consumer and is rebuilt only when the
   * elevation changes. Results are bit-identical to a self-built pyramid and to `'march'`.
   */
  pyramid?: GPURasterExtremaPyramidOutput;
  /**
   * Observers handled by one dispatch. By default each dispatch gets about
   * {@link GPU_TERRAIN_CUMULATIVE_VIEWSHED_SAMPLES_PER_DISPATCH} sight-line samples, estimated as
   * `max(width, height)` per (observer, pixel) pair, so one dispatch may cover several observers
   * on a small grid or only part of one observer's pixels on a large one. Lower it to keep
   * dispatches shorter.
   */
  observersPerDispatch?: number;
  /** Caller-owned outputs. */
  output: GPUTerrainCumulativeViewshedOutput;
};

/**
 * Counts, for every pixel, how many observers see it (the analogue of GDAL `gdal_viewshed`
 * cumulative mode).
 *
 * The (observer, pixel) pairs are the dispatch dimension: consecutive batches of
 * {@link invocationsPerDispatch} pairs run one kernel each, one invocation per pair, and
 * accumulate with `atomicAdd`. Every batch kernel has the same WGSL: a batch reads its first
 * observer and pixel from a GPU-written batch word, so the whole loop compiles one pipeline.
 * Observers
 * outside the grid or on invalid cells contribute nothing, and neither do invalid targets or
 * targets beyond `maximumDistance`. The target elevation is the pixel value plus the target height
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
  /**
   * Observers per dispatch after applying the default: `invocationsPerDispatch / pixelCount`,
   * fractional when one observer's pixels span several dispatches.
   */
  readonly observersPerDispatch: number;
  /** (observer, pixel) pairs per dispatch; every batch dispatches this many invocations. */
  readonly invocationsPerDispatch: number;

  constructor(props: GPUTerrainCumulativeViewshedProps) {
    this.id = props.id ?? 'terrain-cumulative-viewshed';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const pixelCount = validateRasterGrid(id, props.grid);
    validatePackedView(props.observers, ['float32x2'], `${id} observers`);
    if (props.observerHeights) {
      validatePackedView(props.observerHeights, ['float32'], `${id} observerHeights`);
      if (props.observerHeights.length !== props.observers.length) {
        throw new Error(`${id} observerHeights must contain one value per observer`);
      }
    }
    validatePackedView(output.visibleCount, ['uint32'], `${id} output.visibleCount`);
    if (output.visibleCount.length !== pixelCount) {
      throw new Error(`${id} output.visibleCount must contain one value per pixel`);
    }
    if (output.marginalCount) {
      validatePackedView(output.marginalCount, ['uint32'], `${id} output.marginalCount`);
      if (output.marginalCount.length !== pixelCount) {
        throw new Error(`${id} output.marginalCount must contain one value per pixel`);
      }
    }
    validateRasterParameters(id, props.parameters, GPU_TERRAIN_LINE_OF_SIGHT_PARAMETER_LENGTH);
    if (
      props.observersPerDispatch !== undefined &&
      (!Number.isSafeInteger(props.observersPerDispatch) || props.observersPerDispatch < 1)
    ) {
      throw new Error(`${id} observersPerDispatch must be a positive integer`);
    }
    const pairCount = Math.max(props.observers.length * pixelCount, 1);
    const samplesPerPair = Math.max(props.grid.width, props.grid.height);
    const requested =
      props.observersPerDispatch !== undefined
        ? props.observersPerDispatch * pixelCount
        : Math.max(
            1,
            Math.floor(GPU_TERRAIN_CUMULATIVE_VIEWSHED_SAMPLES_PER_DISPATCH / samplesPerPair)
          );
    this.invocationsPerDispatch = Math.min(requested, pairCount);
    // A batch addresses `pixelStart + index`, which must stay below 2^32.
    if (this.invocationsPerDispatch + pixelCount > 0xffffffff) {
      throw new Error(`${id} observersPerDispatch is too large for this grid`);
    }
    this.observersPerDispatch = this.invocationsPerDispatch / pixelCount;
    this.pyramidLayout = getSightLinePyramidLayout(id, props.traversal, props.grid, props.pyramid);
    validateRasterBuffersDistinct(
      id,
      [output.visibleCount, output.marginalCount],
      [
        ...getRasterBandViews(props.elevation),
        props.parameters,
        props.observers,
        ...(props.observerHeights ? [props.observerHeights] : [])
      ]
    );
  }

  /**
   * Returns canonical elevation, optional pyramid, zero fills, the observer records, the batch
   * table, and one kernel per batch.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, pyramidLayout} = this;
    const {output} = props;
    const {width, height} = props.grid;
    const pixelCount = width * height;
    validateRasterBandBelongsToGraph(id, graph, props.elevation, []);
    validateGraphViewsBelongToGraph(id, graph, [
      props.pyramid?.combined,
      props.parameters,
      props.observers,
      props.observerHeights,
      output.visibleCount,
      output.marginalCount
    ]);
    const source = getRasterBufferBandNodes(graph, id, props.elevation, props.grid, true);
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
      ['visible', output.visibleCount],
      ['marginal', output.marginalCount]
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
    const observerCount = props.observers.length;
    const operation = 'GPUTerrainCumulativeViewshed';
    // Observer records [column, row, eye height]: folding the optional heights into one internal
    // view keeps the batch kernel within 8 storage bindings next to its batch word.
    const records = createTransientView(
      graph,
      `${id}-observer-record-data`,
      'float32',
      3 * observerCount
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-observer-records`,
        operation,
        variant: 'observer-records',
        bindings: [
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
          {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'records', view: records, type: 'f32', access: 'read_write'}
        ],
        invocationCount: observerCount,
        body: `records[recordsOffset + 3u * index] = observers[observersOffset + 2u * index];
  records[recordsOffset + 3u * index + 1u] = observers[observersOffset + 2u * index + 1u];
  records[recordsOffset + 3u * index + 2u] = ${
    props.observerHeights
      ? 'observerHeights[observerHeightsOffset + index]'
      : 'settings[settingsOffset]'
  };`
      })
    );
    const batchSize = this.invocationsPerDispatch;
    const batchCount = Math.ceil((observerCount * pixelCount) / batchSize);
    // Batch b starts at pair b * batchSize = (first observer, first pixel). One thread walks the
    // batches with a carry, so no product overflows u32 however many pairs there are.
    const batchTable = createTransientView(
      graph,
      `${id}-batch-words`,
      'uint32',
      Math.max(batchCount, 1) * BATCH_STRIDE
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-batch-table`,
        operation,
        variant: 'batch-table',
        bindings: [{name: 'batchTable', view: batchTable, type: 'u32', access: 'read_write'}],
        invocationCount: 1,
        body: `var firstObserver = 0u;
  var firstPixel = 0u;
  for (var batch = 0u; batch < ${batchCount}u; batch++) {
    batchTable[batchTableOffset + batch * ${BATCH_STRIDE}u] = firstObserver;
    batchTable[batchTableOffset + batch * ${BATCH_STRIDE}u + 1u] = firstPixel;
    firstObserver = firstObserver + ${Math.floor(batchSize / pixelCount)}u;
    firstPixel = firstPixel + ${batchSize % pixelCount}u;
    if (firstPixel >= ${pixelCount}u) {
      firstPixel = firstPixel - ${pixelCount}u;
      firstObserver = firstObserver + 1u;
    }
  }`
      })
    );
    const bindings: WGSLKernelBinding[] = [
      {name: 'elevationValues', view: values, type: 'f32', access: 'read'},
      {name: 'elevationValidity', view: validity, type: 'u32', access: 'read'},
      ...(pyramid.binding ? [pyramid.binding] : []),
      {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'records', view: records, type: 'f32', access: 'read'},
      {name: 'visibleCount', view: output.visibleCount, type: 'atomic<u32>', access: 'read_write'},
      ...(output.marginalCount
        ? [
            {
              name: 'marginalCount',
              view: output.marginalCount,
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
    for (let batch = 0; batch < batchCount; batch++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-batch-${batch}`,
          operation,
          variant: props.traversal === 'pyramid' ? 'pyramid' : 'march',
          bindings: [
            ...bindings,
            {
              name: 'batchStart',
              view: graph.createDataView(batchTable.buffer, {
                format: 'uint32',
                length: 2,
                byteOffset: batchTable.byteOffset + batch * BATCH_STRIDE * 4
              }),
              type: 'u32',
              access: 'read'
            }
          ],
          invocationCount: batchSize,
          declarations: `${sightLine}
const OBSERVER_COUNT: u32 = ${observerCount}u;
const PIXEL_COUNT: u32 = ${pixelCount}u;
fn isInsideGrid(position: vec2<f32>) -> bool {
  return isFiniteValue(position.x) && isFiniteValue(position.y) &&
    position.x >= 0.0 && position.y >= 0.0 &&
    position.x <= f32(WIDTH - 1u) && position.y <= f32(HEIGHT - 1u);
}`,
          body: `let pair = batchStart[batchStartOffset + 1u] + index;
  let observerIndex = batchStart[batchStartOffset] + pair / PIXEL_COUNT;
  if (observerIndex >= OBSERVER_COUNT) {
    return;
  }
  let pixel = pair % PIXEL_COUNT;
  let column = pixel % WIDTH;
  let row = pixel / WIDTH;
  let observer = vec2<f32>(
    records[recordsOffset + 3u * observerIndex],
    records[recordsOffset + 3u * observerIndex + 1u]);
  let observerHeight = records[recordsOffset + 3u * observerIndex + 2u];
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
    output.marginalCount
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
