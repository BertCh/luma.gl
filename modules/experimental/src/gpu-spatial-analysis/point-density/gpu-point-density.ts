// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUConvolution,
  GPUGridAggregation,
  GPUGridBinning,
  GPUGroupAggregation,
  GPUHistogram,
  GPUReduction,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUConvolutionStrategy,
  type GraphDataView,
  type GraphTextureView,
  type GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';
import {GPURasterBufferToTexture} from '../../gpu-raster/index';
import type {GPUFloat32Positions} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  captureGraphCommandNodes,
  getGraphViewChunks,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUSampledSurfacePort} from '../contracts/index';
import {
  createPointDensityClearOverflowNode,
  createPointDensityClearSumsNode,
  createPointDensityFinalizeNode,
  createPointDensityGridKeysNode,
  createPointDensityHexagonKeysNode,
  createPointDensityMaskedPositionsNode,
  createPointDensityWorkgroupSumNode,
  POINT_DENSITY_WORKGROUP_SUM_SIZE,
  type PointDensityResolvedBounds
} from './point-density-kernels';

/** Cell shape. `'grid'` is row-major rectangular cells; `'hexagon'` is pointy-top odd-r hexagons. */
export type GPUPointDensityBinning = 'grid' | 'hexagon';

/** Statistic written to `output.values`, the field used for smoothing, extent, histogram, and texture. */
export type GPUPointDensityStatistic = 'count' | 'sum' | 'mean';

/**
 * How weighted per-cell sums are accumulated. `'auto'` (default) uses workgroup pre-aggregation
 * only for sufficiently large, dense inputs over a small cell grid. `'workgroup'` pre-aggregates
 * rows that share a cell inside each workgroup, so hotspot cells stay fast (1M points in one cell:
 * tens of milliseconds). `'atomic'` adds every row with a global compare-exchange float atomic: it
 * skips the key pass but serializes on cells that many points share (seconds at 1M points per cell).
 * Both are order-dependent in the last float bits.
 */
export type GPUPointDensitySumAccumulation = 'auto' | 'workgroup' | 'atomic';

const AUTO_WORKGROUP_SUM_MINIMUM_ROW_COUNT = POINT_DENSITY_WORKGROUP_SUM_SIZE * 4;
const AUTO_WORKGROUP_SUM_MINIMUM_ROWS_PER_CELL = 32;

/** Selects the sum implementation from compile-time topology without inspecting GPU-resident keys. */
function resolvePointDensitySumAccumulation(
  requested: GPUPointDensitySumAccumulation | undefined,
  rowCount: number,
  cellCount: number
): Exclude<GPUPointDensitySumAccumulation, 'auto'> {
  if (requested === 'atomic' || requested === 'workgroup') {
    return requested;
  }
  return rowCount >= AUTO_WORKGROUP_SUM_MINIMUM_ROW_COUNT &&
    cellCount <= POINT_DENSITY_WORKGROUP_SUM_SIZE &&
    rowCount / cellCount >= AUTO_WORKGROUP_SUM_MINIMUM_ROWS_PER_CELL
    ? 'workgroup'
    : 'atomic';
}

/**
 * Inclusive `[minX, minY, maxX, maxY]` domain.
 *
 * A literal is compile-time topology. A GPU view is per-frame: one `float32x4` row or four packed
 * `float32` rows, the shape of a `GPUParameterBuffer` with `format: 'float32', length: 4`.
 */
export type GPUPointDensityBounds =
  | readonly [number, number, number, number]
  | GraphDataView<'float32x4'>
  | GraphDataView<'float32'>;

/** Optional square-grid smoothing applied to the field with `GPUConvolution` (zero boundary). */
export type GPUPointDensitySmoothing = {
  /**
   * Dense row-major kernel weights, at least `kernelWidth * kernelHeight` rows. Per-frame contents.
   * Required unless `separableKernel` is set; use it only for non-separable kernels.
   */
  kernel?: GraphDataView<'float32'>;
  /**
   * Two 1D kernels whose outer product is the 2D kernel (`weight(x, y) = horizontal[x] * vertical[y]`,
   * for example a Gaussian). Runs two 1D passes, O(cells * (kernelWidth + kernelHeight)) instead of
   * O(cells * kernelWidth * kernelHeight). Matches the dense path within float32 rounding.
   * Takes precedence over `kernel`. Per-frame contents.
   */
  separableKernel?: {
    /** At least `kernelWidth` float32 rows. */
    horizontal: GraphDataView<'float32'>;
    /** At least `kernelHeight` float32 rows. */
    vertical: GraphDataView<'float32'>;
  };
  /** Positive odd kernel width. Compile-time. */
  kernelWidth: number;
  /** Positive odd kernel height. Compile-time. */
  kernelHeight: number;
  /**
   * Forwarded to `GPUConvolution` for the dense kernel. Defaults to `'auto'`. Ignored with
   * `separableKernel`, whose 1D passes always run the direct strategy.
   */
  strategy?: GPUConvolutionStrategy;
};

function getSmoothingKernelViews(
  smoothing: GPUPointDensitySmoothing | undefined
): GraphDataView<'float32'>[] {
  if (!smoothing) {
    return [];
  }
  if (smoothing.separableKernel) {
    return [smoothing.separableKernel.horizontal, smoothing.separableKernel.vertical];
  }
  return smoothing.kernel ? [smoothing.kernel] : [];
}

/** Caller-owned results. Every view must belong to the target graph and use its own buffer. */
export type GPUPointDensityOutput = {
  /** `cellCount` float32 rows: the selected statistic, smoothed when `smoothing` is set. */
  values: GraphDataView<'float32'>;
  /** Optional `cellCount` uint32 rows: unweighted number of accepted points per cell. */
  counts?: GraphDataView<'uint32'>;
  /** Optional `cellCount` float32 rows: sum of finite weights per cell. Requires `weights`. */
  sums?: GraphDataView<'float32'>;
  /** Optional `cellCount` float32 rows: `sums / counts`, `0` for empty cells. Requires `weights`. */
  means?: GraphDataView<'float32'>;
  /** Optional two float32 rows `[min, max]` of `values`, `[0, 0]` when nothing qualifies. */
  extent?: GraphDataView<'float32'>;
  /** Optional uint32 bin counts of `values` over the extent. Its length is the bin count. */
  histogram?: GraphDataView<'uint32'>;
  /** Optional `r32float` storage texture with `width = columns` and `height = rows`. */
  texture?: GraphTextureView<'r32float'>;
  /** Optional flag: 1 when an in-bounds point fell outside the hexagon lattice. Always 0 for grids. */
  overflow?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUPointDensity}.
 *
 * Compile-time: `gridSize`, `binning`, `statistic`, smoothing kernel size, which outputs exist,
 * literal bounds, and a literal radius. Per-frame: positions, weights, GPU bounds, GPU radius, and
 * smoothing kernel weights.
 */
export type GPUPointDensityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'point-density'`. */
  id?: string;
  /** Packed source positions (single view or chunked vector). */
  positions: GPUFloat32Positions;
  /** Optional per-point float32 weights with the same logical length as `positions`. */
  weights?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /**
   * Optional per-row uint32 mask with the same logical length as `positions` (single packed view).
   * Rows whose mask is `0` are excluded from the count, sum, mean, smoothing input, extent, and
   * histogram; any nonzero value includes the row, matching the other contributor mask inputs. The
   * contents are per-frame: rewriting them never recompiles the graph. Omitted means every row is
   * included, and the node list is unchanged.
   */
  mask?: GraphDataView<'uint32'>;
  /** Domain covered by the cells. */
  bounds: GPUPointDensityBounds;
  /** `[columns, rows]` positive integers. Compile-time. */
  gridSize: readonly [number, number];
  /** Cell shape. Defaults to `'grid'`. */
  binning?: GPUPointDensityBinning;
  /**
   * Hexagon center-to-vertex radius in position units. Required for `'hexagon'`. A number is
   * compile-time; a one-row float32 view is per-frame.
   */
  hexagonRadius?: number | GraphDataView<'float32'>;
  /**
   * Weighted sum accumulation. Defaults to `'auto'`. Auto selects workgroup aggregation only for
   * at least 1,024 rows, at most 256 cells, and at least 32 input rows per cell on average. Ignored
   * without `weights`.
   */
  sumAccumulation?: GPUPointDensitySumAccumulation;
  /** Field statistic. Defaults to `'count'`. `'sum'` and `'mean'` require `weights`. */
  statistic?: GPUPointDensityStatistic;
  /** Optional smoothing, grid binning only. */
  smoothing?: GPUPointDensitySmoothing;
  /** Caller-owned results. */
  output: GPUPointDensityOutput;
};

/**
 * Bins points into a square or hexagonal grid and writes per-cell count, sum, mean, a heatmap
 * field, its extent and histogram, and an optional `r32float` texture, all on the GPU.
 *
 * Square grids reuse `GPUGridBinning` and `GPUGridAggregation`; hexagons use one keys kernel and
 * `GPUGroupAggregation`. `GPUConvolution`, `GPUReduction`, `GPUHistogram`, and
 * `GPURasterBufferToTexture` are reused unchanged.
 */
export class GPUPointDensity implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPointDensityProps;
  /** Resolved cell shape. */
  readonly binning: GPUPointDensityBinning;
  /** Resolved field statistic. */
  readonly statistic: GPUPointDensityStatistic;
  /** Resolved weighted-sum implementation. */
  readonly sumAccumulation: Exclude<GPUPointDensitySumAccumulation, 'auto'>;
  /** `gridSize[0] * gridSize[1]`. */
  readonly cellCount: number;
  /** Canonical sampled-surface port for downstream raster and classification contributors. */
  readonly surface: GPUSampledSurfacePort<'float32'>;

  constructor(props: GPUPointDensityProps) {
    this.id = props.id ?? 'point-density';
    this.props = props;
    this.binning = props.binning ?? 'grid';
    this.statistic = props.statistic ?? 'count';
    const id = this.id;
    const [columns, rows] = props.gridSize;
    this.cellCount = columns * rows;
    this.sumAccumulation = resolvePointDensitySumAccumulation(
      props.sumAccumulation,
      props.positions.length,
      this.cellCount
    );
    this.surface = {values: props.output.values, columns, rows};
    const {output, weights, smoothing, mask} = props;

    if (!['grid', 'hexagon'].includes(this.binning)) {
      throw new Error(`${id} binning must be grid or hexagon`);
    }
    if (!['count', 'sum', 'mean'].includes(this.statistic)) {
      throw new Error(`${id} statistic must be count, sum, or mean`);
    }
    if (!['auto', 'workgroup', 'atomic'].includes(props.sumAccumulation ?? 'auto')) {
      throw new Error(`${id} sumAccumulation must be auto, workgroup, or atomic`);
    }
    if (
      !Number.isSafeInteger(columns) ||
      !Number.isSafeInteger(rows) ||
      columns < 1 ||
      rows < 1 ||
      this.cellCount >= 0xffffffff
    ) {
      throw new Error(`${id} gridSize must be two positive integers`);
    }
    for (const chunk of getGraphViewChunks(props.positions)) {
      validatePackedView(chunk, ['float32x2'], `${id} positions`);
    }
    if (weights) {
      for (const chunk of getGraphViewChunks(weights)) {
        validatePackedView(chunk, ['float32'], `${id} weights`);
      }
      if (weights.length !== props.positions.length) {
        throw new Error(`${id} weights length must equal positions length`);
      }
    }
    for (const [needsWeights, name] of [
      [this.statistic !== 'count', `statistic ${this.statistic}`],
      [Boolean(output.sums), 'output.sums'],
      [Boolean(output.means), 'output.means']
    ] as const) {
      if (needsWeights && !weights) {
        throw new Error(`${id} ${name} requires weights`);
      }
    }
    if (mask) {
      validatePackedUint32View(mask, `${id} mask`);
      if (mask.length !== props.positions.length) {
        throw new Error(`${id} mask length must equal positions length`);
      }
    }
    validatePointDensityBounds(id, props.bounds);
    if (this.binning === 'hexagon') {
      const radius = props.hexagonRadius;
      if (radius === undefined) {
        throw new Error(`${id} hexagon binning requires hexagonRadius`);
      }
      if (typeof radius === 'number') {
        if (!Number.isFinite(radius) || radius <= 0) {
          throw new Error(`${id} hexagonRadius must be positive and finite`);
        }
      } else {
        validatePackedView(radius, ['float32'], `${id} hexagonRadius`);
        if (radius.length < 1) {
          throw new Error(`${id} hexagonRadius must contain one float32 row`);
        }
      }
      if (smoothing) {
        throw new Error(`${id} smoothing requires grid binning`);
      }
    } else if (props.hexagonRadius !== undefined) {
      throw new Error(`${id} hexagonRadius requires hexagon binning`);
    }
    if (smoothing) {
      for (const size of [smoothing.kernelWidth, smoothing.kernelHeight]) {
        if (!Number.isSafeInteger(size) || size < 1 || size % 2 === 0) {
          throw new Error(`${id} smoothing kernel sizes must be positive odd integers`);
        }
      }
      const {separableKernel} = smoothing;
      if (separableKernel) {
        validatePackedView(
          separableKernel.horizontal,
          ['float32'],
          `${id} smoothing.separableKernel.horizontal`
        );
        validatePackedView(
          separableKernel.vertical,
          ['float32'],
          `${id} smoothing.separableKernel.vertical`
        );
        if (
          separableKernel.horizontal.length < smoothing.kernelWidth ||
          separableKernel.vertical.length < smoothing.kernelHeight
        ) {
          throw new Error(
            `${id} smoothing.separableKernel is shorter than kernelWidth or kernelHeight`
          );
        }
      } else {
        if (!smoothing.kernel) {
          throw new Error(`${id} smoothing requires kernel or separableKernel`);
        }
        validatePackedView(smoothing.kernel, ['float32'], `${id} smoothing.kernel`);
        if (smoothing.kernel.length < smoothing.kernelWidth * smoothing.kernelHeight) {
          throw new Error(`${id} smoothing.kernel is shorter than kernelWidth * kernelHeight`);
        }
      }
    }

    const cellOutputs = [
      ['values', output.values, 'float32'],
      ['counts', output.counts, 'uint32'],
      ['sums', output.sums, 'float32'],
      ['means', output.means, 'float32']
    ] as const;
    for (const [name, view, format] of cellOutputs) {
      if (!view) {
        continue;
      }
      validatePackedView(view, [format], `${id} output.${name}`);
      if (view.length !== this.cellCount) {
        throw new Error(`${id} output.${name} must have one row per cell`);
      }
    }
    if (output.extent) {
      validatePackedView(output.extent, ['float32'], `${id} output.extent`);
      if (output.extent.length !== 2) {
        throw new Error(`${id} output.extent must have two rows`);
      }
    }
    for (const [name, view] of [
      ['histogram', output.histogram],
      ['overflow', output.overflow]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.${name} must have at least one row`);
        }
      }
    }
    if (
      output.texture &&
      (output.texture.format !== 'r32float' ||
        output.texture.width !== columns ||
        output.texture.height !== rows)
    ) {
      throw new Error(`${id} output.texture must be r32float with gridSize extent`);
    }

    const writableBuffers = [
      output.values,
      output.counts,
      output.sums,
      output.means,
      output.extent,
      output.histogram,
      output.overflow
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    const inputBuffers = [
      ...getGraphViewChunks(props.positions),
      ...(weights ? getGraphViewChunks(weights) : []),
      ...(mask ? [mask] : []),
      ...(Array.isArray(props.bounds) ? [] : [props.bounds as GraphDataView]),
      ...(typeof props.hexagonRadius === 'object' ? [props.hexagonRadius] : []),
      ...getSmoothingKernelViews(smoothing)
    ].map(view => view.buffer);
    if (
      new Set(writableBuffers).size !== writableBuffers.length ||
      writableBuffers.some(buffer => inputBuffers.includes(buffer))
    ) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /** Returns binning, aggregation, finalize, and optional post-processing nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, statistic, sumAccumulation, cellCount} = this;
    const {positions, weights, smoothing, output, gridSize, mask} = props;
    const boundsView = Array.isArray(props.bounds) ? undefined : (props.bounds as GraphDataView);
    const radiusView = typeof props.hexagonRadius === 'object' ? props.hexagonRadius : undefined;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      weights,
      mask,
      boundsView,
      radiusView,
      ...getSmoothingKernelViews(smoothing),
      output.values,
      output.counts,
      output.sums,
      output.means,
      output.extent,
      output.histogram,
      output.overflow
    ]);
    if (output.texture && output.texture.texture.graph !== graph) {
      throw new Error(`${id} views must belong to the target graph`);
    }

    const bounds: PointDensityResolvedBounds =
      boundsView && boundsView.format === 'float32'
        ? graph.createDataView(boundsView.buffer, {
            format: 'float32x4',
            length: 1,
            byteOffset: boundsView.byteOffset
          })
        : (props.bounds as PointDensityResolvedBounds);
    const counts =
      output.counts ?? createTransientView(graph, `${id}-counts-scratch`, 'uint32', cellCount);
    const needsSums =
      Boolean(weights) && (statistic !== 'count' || Boolean(output.sums || output.means));
    const sums = needsSums
      ? (output.sums ?? createTransientView(graph, `${id}-sums-scratch`, 'float32', cellCount))
      : undefined;
    const field = smoothing
      ? createTransientView(graph, `${id}-unsmoothed-values`, 'float32', cellCount)
      : output.values;
    const extent =
      output.extent ??
      (output.histogram
        ? createTransientView(graph, `${id}-extent-scratch`, 'float32', 2)
        : undefined);

    const nodes: GPUCommandNode<Parameters>[] = [];
    if (output.overflow) {
      nodes.push(
        createPointDensityClearOverflowNode(graph, `${id}-clear-overflow`, output.overflow)
      );
    }
    if (this.binning === 'hexagon') {
      const keys = createTransientView(graph, `${id}-hexagon-keys`, 'uint32', positions.length);
      let keyStart = 0;
      for (const [chunkIndex, chunk] of getGraphViewChunks(positions).entries()) {
        if (chunk.length > 0) {
          nodes.push(
            createPointDensityHexagonKeysNode(graph, {
              id: `${id}-hexagon-keys-${chunkIndex}`,
              positions: chunk,
              keys,
              keyStart,
              gridSize,
              bounds,
              radius: props.hexagonRadius as number | GraphDataView<'float32'>,
              mask,
              maskStart: keyStart,
              overflow: output.overflow
            })
          );
        }
        keyStart += chunk.length;
      }
      nodes.push(
        ...new GPUGroupAggregation({id: `${id}-counts`, keys, output: counts}).getCommandNodes(
          graph
        )
      );
      if (sums && weights) {
        if (sumAccumulation === 'atomic') {
          nodes.push(
            ...new GPUGroupAggregation({
              id: `${id}-sums`,
              keys,
              values: weights,
              output: sums,
              operation: 'sum'
            }).getCommandNodes(graph)
          );
        } else {
          nodes.push(...createWorkgroupSumNodes(graph, `${id}-sums`, weights, keys, sums));
        }
      }
    } else {
      // Grid kernels have no mask input: masked rows become NaN positions, which they ignore.
      let gridPositions: GPUFloat32Positions = positions;
      if (mask) {
        const maskedPositions = createTransientView(
          graph,
          `${id}-masked-positions`,
          'float32x2',
          positions.length
        );
        let rowStart = 0;
        for (const [chunkIndex, chunk] of getGraphViewChunks(positions).entries()) {
          if (chunk.length > 0) {
            nodes.push(
              createPointDensityMaskedPositionsNode(graph, {
                id: `${id}-mask-positions-${chunkIndex}`,
                positions: chunk,
                mask,
                maskedPositions,
                rowStart
              })
            );
          }
          rowStart += chunk.length;
        }
        gridPositions = maskedPositions;
      }
      nodes.push(
        ...new GPUGridBinning({
          id: `${id}-counts`,
          positions: gridPositions,
          output: counts,
          gridSize,
          bounds
        }).getCommandNodes(graph)
      );
      if (sums && weights) {
        if (sumAccumulation === 'atomic') {
          nodes.push(
            ...new GPUGridAggregation({
              id: `${id}-sums`,
              positions: gridPositions,
              weights,
              output: sums,
              operation: 'sum',
              gridSize,
              bounds
            }).getCommandNodes(graph)
          );
        } else {
          const keys = createTransientView(graph, `${id}-grid-keys`, 'uint32', positions.length);
          let keyStart = 0;
          for (const [chunkIndex, chunk] of getGraphViewChunks(gridPositions).entries()) {
            if (chunk.length > 0) {
              nodes.push(
                createPointDensityGridKeysNode(graph, {
                  id: `${id}-sums-grid-keys-${chunkIndex}`,
                  positions: chunk,
                  keys,
                  keyStart,
                  gridSize,
                  bounds
                })
              );
            }
            keyStart += chunk.length;
          }
          nodes.push(...createWorkgroupSumNodes(graph, `${id}-sums`, weights, keys, sums));
        }
      }
    }
    nodes.push(
      createPointDensityFinalizeNode(graph, {
        id: `${id}-finalize`,
        statistic,
        counts,
        sums,
        values: field,
        means: output.means
      })
    );
    if (smoothing?.separableKernel) {
      const rowPass = createTransientView(graph, `${id}-smooth-rows`, 'float32', cellCount);
      const sharedProps = {width: gridSize[0], height: gridSize[1], boundary: 'zero'} as const;
      nodes.push(
        ...new GPUConvolution({
          ...sharedProps,
          id: `${id}-smooth-horizontal`,
          input: field,
          kernel: smoothing.separableKernel.horizontal,
          output: rowPass,
          kernelWidth: smoothing.kernelWidth,
          kernelHeight: 1,
          strategy: 'direct'
        }).getCommandNodes(graph),
        ...new GPUConvolution({
          ...sharedProps,
          id: `${id}-smooth-vertical`,
          input: rowPass,
          kernel: smoothing.separableKernel.vertical,
          output: output.values,
          kernelWidth: 1,
          kernelHeight: smoothing.kernelHeight,
          strategy: 'direct'
        }).getCommandNodes(graph)
      );
    } else if (smoothing) {
      nodes.push(
        ...new GPUConvolution({
          id: `${id}-smooth`,
          input: field,
          kernel: smoothing.kernel!,
          output: output.values,
          width: gridSize[0],
          height: gridSize[1],
          kernelWidth: smoothing.kernelWidth,
          kernelHeight: smoothing.kernelHeight,
          boundary: 'zero',
          strategy: smoothing.strategy ?? 'auto'
        }).getCommandNodes(graph)
      );
    }
    // Without smoothing, statistics cover occupied cells only: any nonzero count selects a row.
    const statisticsMask = smoothing ? undefined : counts;
    if (extent) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-extent`,
          input: output.values,
          mask: statisticsMask,
          output: extent,
          operation: 'extent'
        }).getCommandNodes(graph)
      );
    }
    if (output.histogram && extent) {
      nodes.push(
        ...new GPUHistogram({
          id: `${id}-histogram`,
          input: output.values,
          mask: statisticsMask,
          output: output.histogram,
          domain: extent
        }).getCommandNodes(graph)
      );
    }
    if (output.texture) {
      const texture = output.texture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-texture`,
            input: {
              id: `${id}-texture-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: output.values}
            },
            output: texture
          }).addToGraph(graph)
        )
      );
    }
    return nodes;
  }
}

/** Validates literal or GPU-resident density bounds. */
function validatePointDensityBounds(id: string, bounds: GPUPointDensityBounds): void {
  if (Array.isArray(bounds)) {
    const [minX, minY, maxX, maxY] = bounds as readonly number[];
    if (
      bounds.length !== 4 ||
      !bounds.every(value => Number.isFinite(Math.fround(value))) ||
      minX > maxX ||
      minY > maxY
    ) {
      throw new Error(`${id} bounds must be finite float32 [minX, minY, maxX, maxY]`);
    }
    return;
  }
  const view = bounds as GraphDataView;
  validatePackedView(view, ['float32x4', 'float32'], `${id} bounds`);
  if (
    (view.format === 'float32x4' && view.length !== 1) ||
    (view.format === 'float32' && view.length !== 4)
  ) {
    throw new Error(`${id} bounds view must be one float32x4 row or four float32 rows`);
  }
}

/** Clears `sums`, then adds each weight chunk through workgroup-local pre-aggregation. */
function createWorkgroupSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  weights: GraphDataView<'float32'> | GraphVectorView<'float32'>,
  keys: GraphDataView<'uint32'>,
  sums: GraphDataView<'float32'>
): GPUCommandNode<Parameters>[] {
  const nodes = [createPointDensityClearSumsNode(graph, `${id}-clear`, sums)];
  let keyStart = 0;
  for (const [chunkIndex, chunk] of getGraphViewChunks(weights).entries()) {
    if (chunk.length > 0) {
      nodes.push(
        createPointDensityWorkgroupSumNode(graph, {
          id: `${id}-workgroup-${chunkIndex}`,
          weights: chunk,
          keys,
          keyStart,
          sums
        })
      );
    }
    keyStart += chunk.length;
  }
  return nodes;
}
