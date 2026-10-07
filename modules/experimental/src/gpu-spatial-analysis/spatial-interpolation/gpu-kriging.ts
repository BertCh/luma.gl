// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGridIndex,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GRID_COORDINATE_WGSL} from './gpu-inverse-distance-weighting';
import {getRingScanWGSL} from './grid-neighbor-scan';
import {GPU_KRIGING_PARAMETER_LENGTH} from './kriging-parameters';

const OPERATION = 'GPUKriging';
/** Largest compile-time neighborhood; the per-cell system is `(k + 1) x (k + 1)`. */
const MAXIMUM_NEIGHBOR_CAPACITY = 16;

/** Caller-owned row-major raster outputs of {@link GPUKriging}. */
export type GPUKrigingOutput = {
  /**
   * Predicted value per output cell, at least `width * height` float32 rows, row `y * width + x`.
   * Nodata cells (too few neighbors, or a singular system) are NaN.
   */
  values: GraphDataView<'float32'>;
  /**
   * Optional ordinary kriging variance per cell, same layout. Zero at exact hits, NaN at nodata
   * cells. Clamped at zero.
   */
  variance?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUKriging}.
 *
 * Per-frame (no recompile): the contents of `parameters` (extent, radius, neighborhood size,
 * variogram model and parameters) and of every input buffer. Topology (needs a new graph): `width`,
 * `height`, sample count, `indexGridSize`, `indexBounds`, `maximumNeighborCount`, and whether
 * `mask` and `output.variance` are present.
 */
export type GPUKrigingProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'kriging'`. */
  id?: string;
  /** Packed sample positions in the same units as the extent, radius and variogram range. */
  positions: GraphDataView<'float32x2'>;
  /** Packed float32 sample values. NaN values are skipped. */
  values: GraphDataView<'float32'>;
  /** Optional packed `uint32` sample mask; zero skips the sample. */
  mask?: GraphDataView<'uint32'>;
  /** Per-frame float32 view of at least 12 values written with `getGPUKrigingParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Output raster width in cells. */
  width: number;
  /** Output raster height in cells. */
  height: number;
  /** Dimensions `[columns, rows]` of the internal `GPUGridIndex` over the samples. */
  indexGridSize: readonly [number, number];
  /** Inclusive sample domain `[minX, minY, maxX, maxY]` of the internal grid index. */
  indexBounds: readonly [number, number, number, number];
  /** Compile-time neighborhood capacity in `[1, 16]`. Defaults to 16. */
  maximumNeighborCount?: number;
  /** Caller-owned raster outputs. */
  output: GPUKrigingOutput;
};

/**
 * Local ordinary kriging on a raster: for every output cell the `k` nearest samples within the
 * search radius form a `(k + 1) x (k + 1)` ordinary kriging system under a fitted variogram model,
 * solved in the invocation by Gaussian elimination with partial pivoting.
 *
 * Neighbor selection keeps the `k` smallest `(d^2, row)` pairs, a total order, so results are
 * deterministic without sorting the index cells. Cells are visited in rings of growing radius
 * around the cell's own index cell and the walk stops once the `k`-th distance is within the
 * visited block (`O(k)` candidates per cell even with an unbounded radius). The variogram is normalized by its
 * total sill before solving to keep the f32 system well scaled, and the prediction is accumulated
 * relative to the nearest neighbor's value. An exact hit returns that sample with zero variance;
 * duplicate sample positions inside a neighborhood make the system singular (nodata). The
 * variogram follows `fitVariogramModel`: `gamma(0) = 0`, `gamma(h) = nugget + sill * shape(h /
 * range)`, so with a nugget the predictor stays exact at samples. Matches a float64 CPU oracle to
 * f32 rounding for well-conditioned neighborhoods.
 */
export class GPUKriging implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUKrigingProps;
  /** `width * height`. */
  readonly cellCount: number;
  /** Compile-time neighborhood capacity. */
  readonly maximumNeighborCount: number;

  constructor(props: GPUKrigingProps) {
    this.id = props.id ?? 'kriging';
    this.props = props;
    const {id} = this;
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
    this.maximumNeighborCount = props.maximumNeighborCount ?? MAXIMUM_NEIGHBOR_CAPACITY;
    if (
      !Number.isSafeInteger(this.maximumNeighborCount) ||
      this.maximumNeighborCount < 1 ||
      this.maximumNeighborCount > MAXIMUM_NEIGHBOR_CAPACITY
    ) {
      throw new Error(
        `${id} maximumNeighborCount must be an integer in [1, ${MAXIMUM_NEIGHBOR_CAPACITY}]`
      );
    }
    for (const [name, view] of [
      ['positions', props.positions],
      ['values', props.values],
      ['mask', props.mask],
      ['parameters', props.parameters],
      ['output.values', props.output.values],
      ['output.variance', props.output.variance]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.values, ['float32'], `${id} values`);
    const sampleCount = props.values.length;
    if (sampleCount < 1) {
      throw new Error(`${id} needs at least one sample`);
    }
    if (props.positions.length !== sampleCount) {
      throw new Error(`${id} positions length must equal values length`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== sampleCount) {
        throw new Error(`${id} mask length must equal values length`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_KRIGING_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must hold ${GPU_KRIGING_PARAMETER_LENGTH} float32 values`);
    }
    validatePackedView(props.output.values, ['float32'], `${id} output.values`);
    if (props.output.values.length < this.cellCount) {
      throw new Error(`${id} output.values must hold width * height rows`);
    }
    if (props.output.variance) {
      validatePackedView(props.output.variance, ['float32'], `${id} output.variance`);
      if (props.output.variance.length < this.cellCount) {
        throw new Error(`${id} output.variance must hold width * height rows`);
      }
      if (props.output.variance.buffer === props.output.values.buffer) {
        throw new Error(`${id} outputs must not share buffers`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.values, props.output.variance],
      [props.positions, props.values, props.mask, props.parameters]
    );
  }

  /** Returns the grid index build and the per-cell kriging solve, in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, cellCount, maximumNeighborCount} = this;
    const {output, width, height, indexGridSize, indexBounds} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.values,
      props.mask,
      props.parameters,
      output.values,
      output.variance
    ]);
    const sampleCount = props.values.length;
    const indexCellCount = indexGridSize[0] * indexGridSize[1];
    const cellOffsets = createTransientView(
      graph,
      `${id}-cell-offsets`,
      'uint32',
      indexCellCount + 1
    );
    const objectIds = createTransientView(graph, `${id}-object-ids`, 'uint32', sampleCount);
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

    // No in-cell sort: the neighbor list is keyed by the total order (d^2, row), so the selected
    // neighbors do not depend on the order the index cells are visited, and the grid index's
    // unspecified in-cell order is harmless here (unlike an IDW sum).
    // At most 8 storage bindings: positions, values, [mask], params, offsets, ids, out, [variance].
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'sampleValues', view: props.values, type: 'f32', access: 'read'},
      ...(props.mask
        ? [{name: 'sampleMask', view: props.mask, type: 'u32', access: 'read'} as const]
        : []),
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
      {name: 'sortedIds', view: objectIds, type: 'u32', access: 'read'},
      {name: 'valuesOut', view: output.values, type: 'f32', access: 'read_write'},
      ...(output.variance
        ? [{name: 'varianceOut', view: output.variance, type: 'f32', access: 'read_write'} as const]
        : [])
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-solve`,
        operation: OPERATION,
        variant: 'local-ordinary',
        bindings,
        invocationCount: cellCount,
        workgroupSize: 64,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const INDEX_WIDTH: u32 = ${indexGridSize[0]}u;
const INDEX_HEIGHT: u32 = ${indexGridSize[1]}u;
const MAX_NEIGHBORS: u32 = ${maximumNeighborCount}u;
const SYSTEM_STRIDE: u32 = ${maximumNeighborCount + 2}u;
const DOMAIN_MIN: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[0])}, ${getWGSLFloatLiteral(indexBounds[1])});
const DOMAIN_MAX: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[2])}, ${getWGSLFloatLiteral(indexBounds[3])});
${GRID_COORDINATE_WGSL}
${KRIGING_WGSL}`,
        body: getSolveBody(Boolean(props.mask), Boolean(output.variance))
      })
    );
    return nodes;
  }
}

const KRIGING_WGSL = /* wgsl */ `
// Variogram normalized by its total sill; gamma(0) = 0.
fn evaluateVariogram(distance: f32, modelCode: u32, nugget: f32, sill: f32, range: f32) -> f32 {
  if (distance <= 0.0) { return 0.0; }
  let scaled = distance / range;
  var shape = 1.0;
  if (modelCode == 0u) {
    shape = select(1.5 * scaled - 0.5 * scaled * scaled * scaled, 1.0, scaled >= 1.0);
  } else if (modelCode == 1u) {
    shape = 1.0 - exp(-3.0 * scaled);
  } else {
    shape = 1.0 - exp(-3.0 * scaled * scaled);
  }
  return (nugget + sill * shape) / (nugget + sill);
}

// Gaussian elimination with partial pivoting on an n x (n + 1) augmented system stored row-major
// with SYSTEM_STRIDE columns; leaves the solution in column n. Returns false when singular.
fn solveSystem(system: ptr<function, array<f32, ${(MAXIMUM_NEIGHBOR_CAPACITY + 1) * (MAXIMUM_NEIGHBOR_CAPACITY + 2)}>>, n: u32) -> bool {
  for (var column = 0u; column < n; column++) {
    var pivotRow = column;
    var pivotMagnitude = abs((*system)[column * SYSTEM_STRIDE + column]);
    for (var row = column + 1u; row < n; row++) {
      let magnitude = abs((*system)[row * SYSTEM_STRIDE + column]);
      if (magnitude > pivotMagnitude) {
        pivotMagnitude = magnitude;
        pivotRow = row;
      }
    }
    if (!(pivotMagnitude > 1e-6)) { return false; }
    if (pivotRow != column) {
      for (var k = column; k <= n; k++) {
        let held = (*system)[column * SYSTEM_STRIDE + k];
        (*system)[column * SYSTEM_STRIDE + k] = (*system)[pivotRow * SYSTEM_STRIDE + k];
        (*system)[pivotRow * SYSTEM_STRIDE + k] = held;
      }
    }
    let pivot = (*system)[column * SYSTEM_STRIDE + column];
    for (var row = column + 1u; row < n; row++) {
      let factor = (*system)[row * SYSTEM_STRIDE + column] / pivot;
      for (var k = column; k <= n; k++) {
        (*system)[row * SYSTEM_STRIDE + k] -= factor * (*system)[column * SYSTEM_STRIDE + k];
      }
    }
  }
  for (var step = 0u; step < n; step++) {
    let row = n - 1u - step;
    var sum = (*system)[row * SYSTEM_STRIDE + n];
    for (var k = row + 1u; k < n; k++) {
      sum -= (*system)[row * SYSTEM_STRIDE + k] * (*system)[k * SYSTEM_STRIDE + n];
    }
    (*system)[row * SYSTEM_STRIDE + n] = sum / (*system)[row * SYSTEM_STRIDE + row];
  }
  return true;
}`;

function getSolveBody(hasMask: boolean, hasVariance: boolean): string {
  return `let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let extentMin = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let extentMax = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let radius = params[paramsOffset + 4u];
  let neighborLimitValue = params[paramsOffset + 5u];
  let modelCode = u32(clamp(params[paramsOffset + 6u], 0.0, 2.0));
  let nugget = params[paramsOffset + 7u];
  let sill = params[paramsOffset + 8u];
  let range = params[paramsOffset + 9u];
  let minimumNeighborValue = params[paramsOffset + 10u];
  let neighborLimit = u32(clamp(select(0.0, neighborLimitValue, isFiniteValue(neighborLimitValue)), 0.0, f32(MAX_NEIGHBORS)));
  let minimumNeighbors = max(u32(clamp(select(1.0, minimumNeighborValue, isFiniteValue(minimumNeighborValue)), 0.0, 4294967040.0)), 1u);
  let cellSize = (extentMax - extentMin) / vec2f(f32(WIDTH), f32(HEIGHT));
  let center = extentMin + (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let radiusSquared = radius * radius;
  let queryMin = center - vec2f(radius);
  let queryMax = center + vec2f(radius);
  let variogramValid = nugget >= 0.0 && sill >= 0.0 && nugget + sill > 0.0 && range > 0.0 && isFiniteValue(range);
  let valid = variogramValid && neighborLimit > 0u && isFiniteValue(center.x) && isFiniteValue(center.y) &&
    radius >= 0.0 && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX);
  var nearDistances: array<f32, MAX_NEIGHBORS>;
  var nearRows: array<u32, MAX_NEIGHBORS>;
  var nearCount = 0u;
  var exactRow = 0xffffffffu;
  var exactValue = 0.0;
  if (valid) {
    let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
    ${getRingScanWGSL(`let value = sampleValues[sampleValuesOffset + sample];
        if (isNanValue(value)) {
          continue;
        }
        ${hasMask ? 'if (sampleMask[sampleMaskOffset + sample] == 0u) {\n          continue;\n        }' : ''}
        let position = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]);
        let delta = position - center;
        let distanceSquared = dot(delta, delta);
        if (!(distanceSquared <= radiusSquared)) {
          continue;
        }
        if (distanceSquared == 0.0 && sample < exactRow) {
          exactRow = sample;
          exactValue = value;
        }
        let last = neighborLimit - 1u;
        let isFull = nearCount >= neighborLimit;
        if (!isFull || distanceSquared < nearDistances[last] ||
            (distanceSquared == nearDistances[last] && sample < nearRows[last])) {
          var slotPosition = select(nearCount, last, isFull);
          loop {
            if (slotPosition == 0u) { break; }
            let previousDistance = nearDistances[slotPosition - 1u];
            let previousRow = nearRows[slotPosition - 1u];
            if (previousDistance < distanceSquared ||
                (previousDistance == distanceSquared && previousRow < sample)) {
              break;
            }
            nearDistances[slotPosition] = previousDistance;
            nearRows[slotPosition] = previousRow;
            slotPosition -= 1u;
          }
          nearDistances[slotPosition] = distanceSquared;
          nearRows[slotPosition] = sample;
          nearCount = min(nearCount + 1u, neighborLimit);
        }`)}
  }
  var result = getNaN();
  var variance = getNaN();
  if (exactRow != 0xffffffffu) {
    result = exactValue;
    variance = 0.0;
  } else if (valid && nearCount >= minimumNeighbors) {
    let n = nearCount + 1u;
    var system: array<f32, ${(MAXIMUM_NEIGHBOR_CAPACITY + 1) * (MAXIMUM_NEIGHBOR_CAPACITY + 2)}>;
    var targetGamma: array<f32, MAX_NEIGHBORS>;
    var nearPositions: array<vec2f, MAX_NEIGHBORS>;
    for (var i = 0u; i < nearCount; i++) {
      let row = nearRows[i];
      nearPositions[i] = vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
    }
    for (var i = 0u; i < nearCount; i++) {
      system[i * SYSTEM_STRIDE + i] = 0.0;
      // The variogram matrix is symmetric: one evaluation fills both triangles.
      for (var j = i + 1u; j < nearCount; j++) {
        let separation = nearPositions[j] - nearPositions[i];
        let gamma = evaluateVariogram(sqrt(dot(separation, separation)), modelCode, nugget, sill, range);
        system[i * SYSTEM_STRIDE + j] = gamma;
        system[j * SYSTEM_STRIDE + i] = gamma;
      }
      system[i * SYSTEM_STRIDE + nearCount] = 1.0;
      system[nearCount * SYSTEM_STRIDE + i] = 1.0;
      targetGamma[i] = evaluateVariogram(sqrt(nearDistances[i]), modelCode, nugget, sill, range);
      system[i * SYSTEM_STRIDE + n] = targetGamma[i];
    }
    system[nearCount * SYSTEM_STRIDE + nearCount] = 0.0;
    system[nearCount * SYSTEM_STRIDE + n] = 1.0;
    if (solveSystem(&system, n)) {
      let baseValue = sampleValues[sampleValuesOffset + nearRows[0]];
      var prediction = 0.0;
      var predictionVariance = system[nearCount * SYSTEM_STRIDE + n];
      for (var i = 0u; i < nearCount; i++) {
        let weight = system[i * SYSTEM_STRIDE + n];
        prediction += weight * (sampleValues[sampleValuesOffset + nearRows[i]] - baseValue);
        predictionVariance += weight * targetGamma[i];
      }
      result = baseValue + prediction;
      variance = max(predictionVariance, 0.0) * (nugget + sill);
    }
  }
  valuesOut[valuesOutOffset + index] = result;
  ${hasVariance ? 'varianceOut[varianceOutOffset + index] = variance;' : ''}`;
}
