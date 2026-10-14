// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGridIndex,
  GPUReduction,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {resolveGPUSpatialContext, type GPUSpatialContext} from '../contracts/index';
import {getRectangleScanWGSL} from '../spatial-interpolation/grid-neighbor-scan';
import {GRID_COORDINATE_WGSL} from '../spatial-interpolation/gpu-inverse-distance-weighting';
import {
  GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION,
  type GPUKernelDensityKernel,
  type GPUKernelDensityWindow
} from './kernel-density-math';
import {getKernelValueWGSL, getWindowMassWGSL} from './kernel-density-wgsl';

const OPERATION = 'GPUKernelDensity';

/** Number of float32 elements in a {@link GPUKernelDensity} parameter view. */
export const GPU_KERNEL_DENSITY_PARAMETER_LENGTH = 9;

/** Edge treatment at the boundary of the study window. See {@link GPUKernelDensityProps.edgeCorrection}. */
export type GPUKernelDensityEdgeCorrection = 'none' | 'uniform' | 'diggle';

/** Output scaling. See {@link GPUKernelDensityProps.normalization}. */
export type GPUKernelDensityNormalization = 'intensity' | 'probability';

/** Per-frame settings of {@link GPUKernelDensity}. */
export type GPUKernelDensitySettings = {
  /** Output raster extent `[minX, minY, maxX, maxY]` in position units. Cell centers are evaluated. */
  extent: readonly [number, number, number, number];
  /** Kernel bandwidth `h` in position units (ground units). Positive and finite. */
  bandwidth: number;
  /**
   * Study window `[minX, minY, maxX, maxY]` of the edge correction. Defaults to `extent`. Ignored
   * without edge correction.
   */
  window?: GPUKernelDensityWindow;
};

/**
 * Packs per-frame {@link GPUKernelDensity} parameters.
 *
 * Layout (float32): `[minX, minY, maxX, maxY, bandwidth, windowMinX, windowMinY, windowMaxX,
 * windowMaxY]`.
 *
 * @param settings Extent, bandwidth and optional window.
 * @param target Optional destination of at least {@link GPU_KERNEL_DENSITY_PARAMETER_LENGTH} elements.
 * @throws If the target is too short, the bandwidth is not positive and finite, or a rectangle is
 * not finite and ordered.
 */
export function getGPUKernelDensityParameterValues(
  settings: GPUKernelDensitySettings,
  target: Float32Array = new Float32Array(GPU_KERNEL_DENSITY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_KERNEL_DENSITY_PARAMETER_LENGTH) {
    throw new Error(
      `Kernel density parameter target must hold ${GPU_KERNEL_DENSITY_PARAMETER_LENGTH} elements`
    );
  }
  if (!Number.isFinite(settings.bandwidth) || settings.bandwidth <= 0) {
    throw new Error('Kernel density bandwidth must be positive and finite');
  }
  const window = settings.window ?? settings.extent;
  for (const rectangle of [settings.extent, window]) {
    if (
      rectangle.length !== 4 ||
      !rectangle.every(Number.isFinite) ||
      rectangle[0] >= rectangle[2] ||
      rectangle[1] >= rectangle[3]
    ) {
      throw new Error('Kernel density extent and window must be finite [minX, minY, maxX, maxY]');
    }
  }
  target.set(settings.extent, 0);
  target[4] = settings.bandwidth;
  target.set(window, 5);
  return target;
}

/** Caller-owned outputs of {@link GPUKernelDensity}. */
export type GPUKernelDensityOutput = {
  /**
   * Density per output cell, at least `width * height` float32 rows, row `y * width + x` with row
   * 0 at `minY`. Units: `'intensity'` is the sum of weights per unit area, position units to the
   * power -2 (points per square meter for a meter frame); `'probability'` is the same divided by
   * the sum of included weights, a probability density that integrates to about one over the
   * window. NaN where the cell center is outside the edge-correction window, the extent or
   * bandwidth is invalid, or (probability) no weight is included.
   */
  values: GraphDataView<'float32'>;
  /**
   * Optional one-row float32 sum of the included weights (the point count without `weights`),
   * before edge correction: the divisor of `'probability'`.
   */
  weightSum?: GraphDataView<'float32'>;
  /**
   * Optional one-row uint32 count of input rows that were excluded: a non-finite position or
   * weight, a negative weight, a position outside `indexBounds`, or (with edge correction) outside
   * the window. Zero-weight rows are included. Uses the status vocabulary `invalidCount`.
   */
  invalidCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUKernelDensity}.
 *
 * Per-frame (no recompile): the contents of `parameters` (extent, bandwidth, window) and of every
 * input buffer. Topology (needs a new graph): `width`, `height`, point count, `indexGridSize`,
 * `indexBounds`, `kernel`, `edgeCorrection`, `normalization`, and which optional inputs and outputs
 * are present.
 */
export type GPUKernelDensityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'kernel-density'`. */
  id?: string;
  /**
   * Coordinate and metric contract of positions, extent and bandwidth. Planar/native only
   * (omitted means planar/native): the bandwidth is a ground distance, so positions must be in a
   * planar frame with the same unit on both axes. Longitude/latitude throws at construction.
   */
  spatialContext?: GPUSpatialContext;
  /** Packed point positions in the same units as the extent and bandwidth. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Optional per-point weights, for example populations or counts. Non-finite and negative
   * weights exclude the point. Without weights every point has weight one.
   */
  weights?: GraphDataView<'float32'>;
  /** Per-frame float32 view of at least 9 values written with {@link getGPUKernelDensityParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Output raster width in cells. */
  width: number;
  /** Output raster height in cells. */
  height: number;
  /**
   * Dimensions `[columns, rows]` of the internal `GPUGridIndex` over the points. Aim for a few
   * points per index cell, and index cells not much larger than the bandwidth.
   */
  indexGridSize: readonly [number, number];
  /**
   * Inclusive point domain `[minX, minY, maxX, maxY]` of the internal grid index. It must cover the
   * extent and every point you want counted: points outside it are ignored (counted in
   * `invalidCount`), so a point just outside the domain stops contributing to cells within one
   * support radius of it.
   */
  indexBounds: readonly [number, number, number, number];
  /** Kernel, `K_h(d) = K(d / h) / h^2`, normalised to integrate to one. Defaults to `'gaussian'`. */
  kernel?: GPUKernelDensityKernel;
  /**
   * Treatment of the study window boundary, where an uncorrected estimate loses the kernel mass
   * that falls outside the data region and underestimates the intensity.
   * - `'uniform'` (default): divide each cell by `e(u)`, the kernel mass inside the window
   *   (Diggle 1985; spatstat `density.ppp(edge = TRUE, diggle = FALSE)`). Unbiased in expectation
   *   for a locally uniform intensity, deterministic, and cheap. It is the default because the
   *   uncorrected estimate is systematically low within one bandwidth of the window edge.
   * - `'diggle'`: weight each point by `1 / e(x_i)` (Diggle 1985; Jones 1993, `diggle = TRUE`):
   *   preserves the point total better but is noisier near the edge.
   * - `'none'`: the plain kernel sum; the window is ignored.
   *
   * With `'uniform'` or `'diggle'` the window is `parameters.window` (the extent by default),
   * points outside it are excluded, and cells whose center is outside it are NaN.
   */
  edgeCorrection?: GPUKernelDensityEdgeCorrection;
  /**
   * `'intensity'` (default): `sum_i w_i K_h(u - x_i)`, weights per unit area. `'probability'`:
   * that divided by the sum of included weights.
   */
  normalization?: GPUKernelDensityNormalization;
  /** Caller-owned outputs. */
  output: GPUKernelDensityOutput;
};

/**
 * Planar kernel density estimate on a raster, with the bandwidth in ground units.
 *
 * For each output cell center `u` it computes `lambda(u) = sum_i w_i K_h(|u - x_i|)` over the
 * points within the kernel support, where `K_h(d) = K(d / h) / h^2` is a radially symmetric kernel
 * normalised to integrate to one over the plane: `'gaussian'` `exp(-t / 2) / (2 pi h^2)` truncated
 * at 5 bandwidths (lost mass `exp(-12.5)`, 3.7e-6), `'quartic'` (biweight) `3 / pi (1 - t)^2 / h^2`
 * and `'epanechnikov'` `2 / pi (1 - t) / h^2`, with `t = d^2 / h^2 < 1` (Silverman 1986, Density
 * Estimation, Table 3.1). The output is therefore an intensity in weight per unit area (not a count
 * per cell), unchanged by the raster resolution, and `normalization: 'probability'` divides by the
 * included weight to give a probability density. Compare {@link GPUPointDensity}, which bins points
 * and smooths the bin counts with a kernel measured in cells.
 *
 * Edge correction (`edgeCorrection`) divides by, or reweights with, the kernel mass `e(u)` inside
 * the rectangular study window. The mass is closed form: a product of normal-CDF differences for the
 * gaussian, and an exact inclusion-exclusion of four disk-clipped quadrant integrals for the
 * compact kernels (no quadrature). Windows are rectangles; an irregular study area still needs a
 * mask or a polygon window, which is not implemented.
 *
 * Implementation: every encoding rebuilds a `GPUGridIndex` over the points, ranks the IDs inside
 * each index cell ascending, computes per-point weights (exclusion, and `w_i / e(x_i)` for
 * `'diggle'`), and gathers one output cell per invocation by visiting the index rows overlapping the
 * support square in a fixed order (index row, index column, ascending point row). No float atomics,
 * so results are deterministic for a device. Distances are formed relative to the extent origin to
 * limit f32 cancellation far from the origin, but the input positions themselves are f32.
 *
 * Bandwidth: choose it in ground units for the question being asked. Rules of thumb are available
 * on read-back positions as {@link getScottBandwidth} and {@link getSilvermanBandwidth}.
 *
 * Output cell `(x, y)` is evaluated at its center `extentMin + (x + 0.5, y + 0.5) * cellSize`, with
 * row 0 at `minY`. The estimate is a cell-center sample, not a cell average.
 */
export class GPUKernelDensity implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUKernelDensityProps;
  /** Resolved coordinate and metric contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved kernel. */
  readonly kernel: GPUKernelDensityKernel;
  /** Resolved edge correction. */
  readonly edgeCorrection: GPUKernelDensityEdgeCorrection;
  /** Resolved output scaling. */
  readonly normalization: GPUKernelDensityNormalization;
  /** `width * height`. */
  readonly cellCount: number;

  constructor(props: GPUKernelDensityProps) {
    this.id = props.id ?? 'kernel-density';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    this.kernel = props.kernel ?? 'gaussian';
    this.edgeCorrection = props.edgeCorrection ?? 'uniform';
    this.normalization = props.normalization ?? 'intensity';
    if (!['gaussian', 'quartic', 'epanechnikov'].includes(this.kernel)) {
      throw new Error(`${id} kernel must be gaussian, quartic, or epanechnikov`);
    }
    if (!['none', 'uniform', 'diggle'].includes(this.edgeCorrection)) {
      throw new Error(`${id} edgeCorrection must be none, uniform, or diggle`);
    }
    if (!['intensity', 'probability'].includes(this.normalization)) {
      throw new Error(`${id} normalization must be intensity or probability`);
    }
    for (const [name, size] of [
      ['width', props.width],
      ['height', props.height],
      ['indexGridSize[0]', props.indexGridSize[0]],
      ['indexGridSize[1]', props.indexGridSize[1]]
    ] as const) {
      if (!Number.isSafeInteger(size) || size < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    this.cellCount = props.width * props.height;
    if (
      this.cellCount > 0xffffffff ||
      props.indexGridSize[0] * props.indexGridSize[1] >= 0xffffffff
    ) {
      throw new Error(`${id} raster and index cell counts must fit in uint32`);
    }
    const [minX, minY, maxX, maxY] = props.indexBounds;
    if (
      props.indexBounds.length !== 4 ||
      !props.indexBounds.every(Number.isFinite) ||
      minX > maxX ||
      minY > maxY
    ) {
      throw new Error(`${id} indexBounds must be finite and ordered`);
    }
    for (const [name, view] of [
      ['positions', props.positions],
      ['weights', props.weights],
      ['parameters', props.parameters],
      ['output.values', props.output.values],
      ['output.weightSum', props.output.weightSum],
      ['output.invalidCount', props.output.invalidCount]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one point`);
    }
    if (props.weights) {
      validatePackedView(props.weights, ['float32'], `${id} weights`);
      if (props.weights.length !== props.positions.length) {
        throw new Error(`${id} weights length must equal positions length`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_KERNEL_DENSITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_KERNEL_DENSITY_PARAMETER_LENGTH} float32 values`
      );
    }
    validatePackedView(props.output.values, ['float32'], `${id} output.values`);
    if (props.output.values.length < this.cellCount) {
      throw new Error(`${id} output.values must hold width * height rows`);
    }
    if (props.output.weightSum) {
      validatePackedView(props.output.weightSum, ['float32'], `${id} output.weightSum`);
      if (props.output.weightSum.length !== 1) {
        throw new Error(`${id} output.weightSum must contain exactly one float32 row`);
      }
    }
    if (props.output.invalidCount) {
      validatePackedUint32View(props.output.invalidCount, `${id} output.invalidCount`);
      if (props.output.invalidCount.length !== 1) {
        throw new Error(`${id} output.invalidCount must contain exactly one uint32 row`);
      }
    }
    const outputBuffers = [props.output.values, props.output.weightSum, props.output.invalidCount]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.values, props.output.weightSum, props.output.invalidCount],
      [props.positions, props.weights, props.parameters]
    );
  }

  /**
   * Returns the grid index build, the in-cell sort, the per-point weights, the optional weight sum
   * and invalid count reductions, and the per-cell gather, in order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, cellCount, kernel, edgeCorrection, normalization} = this;
    const {output, width, height, indexGridSize, indexBounds} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.weights,
      props.parameters,
      output.values,
      output.weightSum,
      output.invalidCount
    ]);
    const pointCount = props.positions.length;
    const indexCellCount = indexGridSize[0] * indexGridSize[1];
    const cellOffsets = createTransientView(
      graph,
      `${id}-cell-offsets`,
      'uint32',
      indexCellCount + 1
    );
    const objectIds = createTransientView(graph, `${id}-object-ids`, 'uint32', pointCount);
    const sortedIds = createTransientView(graph, `${id}-sorted-ids`, 'uint32', pointCount);
    const indexCount = createTransientView(graph, `${id}-index-count`, 'uint32', 1);
    const indexOverflow = createTransientView(graph, `${id}-index-overflow`, 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUGridIndex({
        id: `${id}-index`,
        positions: props.positions,
        gridSize: indexGridSize,
        bounds: indexBounds,
        cellOffsets,
        objectIds,
        count: indexCount,
        overflow: indexOverflow
      }).getCommandNodes(graph)
    ];

    // GPUGridIndex scatters IDs with atomics, so in-cell order varies between runs. Rank every ID
    // within its cell to restore ascending row order before any floating-point sum depends on it.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sort-cells`,
        operation: OPERATION,
        variant: 'sort-cells',
        bindings: [
          {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
          {name: 'objectIds', view: objectIds, type: 'u32', access: 'read'},
          {name: 'indexCount', view: indexCount, type: 'u32', access: 'read'},
          {name: 'sortedIds', view: sortedIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pointCount,
        declarations: `const INDEX_CELL_COUNT: u32 = ${indexCellCount}u;
const POINT_COUNT: u32 = ${pointCount}u;`,
        body: `if (index >= min(indexCount[indexCountOffset], POINT_COUNT)) {
    return;
  }
  let point = objectIds[objectIdsOffset + index];
  // Largest cell whose first slot is at or before this slot.
  var low = 0u;
  var high = INDEX_CELL_COUNT - 1u;
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    if (cellOffsets[cellOffsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let start = cellOffsets[cellOffsetsOffset + low];
  let end = cellOffsets[cellOffsetsOffset + low + 1u];
  var rank = 0u;
  for (var slot = start; slot < end; slot++) {
    rank += select(0u, 1u, objectIds[objectIdsOffset + slot] < point);
  }
  sortedIds[sortedIdsOffset + start + rank] = point;`
      })
    );

    // Per-point weights: 0 for excluded rows; w / e(x) under Diggle. The raw (uncorrected) weights
    // feed the weight sum, so they are separate only when Diggle changes them.
    const needsRawWeights =
      edgeCorrection === 'diggle' && (normalization === 'probability' || Boolean(output.weightSum));
    const needsWeightSum = normalization === 'probability' || Boolean(output.weightSum);
    const pointWeights = createTransientView(graph, `${id}-point-weights`, 'float32', pointCount);
    const rawWeights = needsRawWeights
      ? createTransientView(graph, `${id}-raw-weights`, 'float32', pointCount)
      : pointWeights;
    const excluded = output.invalidCount
      ? createTransientView(graph, `${id}-excluded`, 'uint32', pointCount)
      : undefined;
    const domainConstants = `const DOMAIN_MIN: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[0])}, ${getWGSLFloatLiteral(indexBounds[1])});
const DOMAIN_MAX: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[2])}, ${getWGSLFloatLiteral(indexBounds[3])});`;
    const hasEdgeCorrection = edgeCorrection !== 'none';
    const weightBindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      ...(props.weights
        ? [{name: 'inputWeights', view: props.weights, type: 'f32', access: 'read'} as const]
        : []),
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'pointWeights', view: pointWeights, type: 'f32', access: 'read_write'},
      ...(needsRawWeights
        ? [{name: 'rawWeights', view: rawWeights, type: 'f32', access: 'read_write'} as const]
        : []),
      ...(excluded
        ? [{name: 'excluded', view: excluded, type: 'u32', access: 'read_write'} as const]
        : [])
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-point-weights`,
        operation: OPERATION,
        variant: `point-weights-${edgeCorrection}`,
        bindings: weightBindings,
        invocationCount: pointCount,
        declarations: `${domainConstants}
${GRID_COORDINATE_WGSL}
${edgeCorrection === 'diggle' ? getWindowMassWGSL(kernel) : ''}`,
        body: `let extentMin = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let bandwidth = params[paramsOffset + 4u];
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - extentMin;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - extentMin;
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let weight = ${props.weights ? 'inputWeights[inputWeightsOffset + index]' : '1.0'};
  var included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    all(position >= DOMAIN_MIN) && all(position <= DOMAIN_MAX) &&
    isFiniteValue(weight) && weight >= 0.0;
  let relative = position - extentMin;
  ${hasEdgeCorrection ? 'included = included && all(relative >= windowMin) && all(relative <= windowMax);' : ''}
  var corrected = select(0.0, weight, included);
  ${rawWeights !== pointWeights ? 'rawWeights[rawWeightsOffset + index] = corrected;' : ''}
  ${
    edgeCorrection === 'diggle'
      ? `if (included) {
    let mass = getWindowMass(relative, windowMin, windowMax, bandwidth);
    corrected = select(0.0, weight / mass, mass > 1e-6);
  }`
      : ''
  }
  pointWeights[pointWeightsOffset + index] = corrected;
  ${excluded ? 'excluded[excludedOffset + index] = select(1u, 0u, included);' : ''}`
      })
    );
    const weightSum = needsWeightSum
      ? (output.weightSum ?? createTransientView(graph, `${id}-weight-sum-scratch`, 'float32', 1))
      : undefined;
    if (weightSum) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-weight-sum`,
          input: rawWeights,
          output: weightSum,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    if (excluded && output.invalidCount) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-invalid-count`,
          input: excluded,
          output: output.invalidCount,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }

    const gatherBindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'pointWeights', view: pointWeights, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
      {name: 'sortedIds', view: sortedIds, type: 'u32', access: 'read'},
      ...(normalization === 'probability' && weightSum
        ? [{name: 'weightSum', view: weightSum, type: 'f32', access: 'read'} as const]
        : []),
      {name: 'valuesOut', view: output.values, type: 'f32', access: 'read_write'}
    ];
    const supportRadiusFactor = kernel === 'gaussian' ? GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION : 1;
    const visit = `let pointWeight = pointWeights[pointWeightsOffset + sample];
        if (!(pointWeight > 0.0)) {
          continue;
        }
        let position = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]);
        // Distances come from local-origin offsets so large absolute coordinates cancel exactly.
        let delta = (position - extentMin) - cellOffset;
        let distanceSquared = dot(delta, delta);
        if (!(distanceSquared < supportSquared)) {
          continue;
        }
        total += pointWeight * getKernelValue(distanceSquared, bandwidth);`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gather`,
        operation: OPERATION,
        variant: `gather-${kernel}-${edgeCorrection}-${normalization}`,
        bindings: gatherBindings,
        invocationCount: cellCount,
        workgroupSize: 64,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const INDEX_WIDTH: u32 = ${indexGridSize[0]}u;
const INDEX_HEIGHT: u32 = ${indexGridSize[1]}u;
const SUPPORT_FACTOR: f32 = ${getWGSLFloatLiteral(supportRadiusFactor)};
${domainConstants}
${GRID_COORDINATE_WGSL}
${getKernelValueWGSL(kernel)}
${edgeCorrection === 'none' ? '' : getWindowMassWGSL(kernel)}`,
        body: `let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let extentMin = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let extentMax = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let bandwidth = params[paramsOffset + 4u];
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - extentMin;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - extentMin;
  let cellSize = (extentMax - extentMin) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = extentMin + cellOffset;
  let radius = SUPPORT_FACTOR * bandwidth;
  let supportSquared = radius * radius;
  let queryMin = center - vec2f(radius);
  let queryMax = center + vec2f(radius);
  let valid = isFiniteValue(center.x) && isFiniteValue(center.y) &&
    isFiniteValue(bandwidth) && bandwidth > 0.0 && isFiniteValue(radius) &&
    all(extentMax > extentMin);
  var total = 0.0;
  if (valid && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX)) {
    let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    // Widen by one index cell per side to absorb f32 rounding in the binning pass; the exact
    // distance test rejects the extra candidates.
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
    ${getRectangleScanWGSL(visit)}
  }
  var result = total;
  if (!valid) {
    result = getNaN();
  }
  ${
    edgeCorrection === 'none'
      ? ''
      : `if (valid) {
    if (!(all(cellOffset >= windowMin) && all(cellOffset <= windowMax))) {
      result = getNaN();
    }${
      edgeCorrection === 'uniform'
        ? ` else {
      let mass = getWindowMass(cellOffset, windowMin, windowMax, bandwidth);
      result = select(getNaN(), total / mass, mass > 1e-6);
    }`
        : ''
    }
  }`
  }
  ${
    normalization === 'probability'
      ? `if (valid) {
    let totalWeight = weightSum[weightSumOffset];
    result = select(getNaN(), result / totalWeight, totalWeight > 0.0);
  }`
      : ''
  }
  valuesOut[valuesOutOffset + index] = result;`
      })
    );
    return nodes;
  }
}
