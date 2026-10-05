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
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createMapGraphKernelNode,
  getWGSLFloatLiteral,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH} from './spatial-interpolation-parameters';

const OPERATION = 'GPUInverseDistanceWeighting';
/** Largest compile-time nearest-neighbor capacity; each invocation keeps a private list this long. */
const MAXIMUM_NEIGHBOR_CAPACITY = 64;

/** Caller-owned row-major raster outputs of {@link GPUInverseDistanceWeighting}. */
export type GPUInverseDistanceWeightingOutput = {
  /**
   * Interpolated value per output cell, at least `width * height` float32 rows, row `y * width +
   * x`. Nodata cells (no exact hit and fewer than `max(minimumNeighborCount, 1)` contributing
   * samples) are NaN, which `GPUTerrainContours` and `GPURasterContours` treat as invalid.
   */
  values: GraphDataView<'float32'>;
  /** Optional number of contributing samples per cell (after the `k` limit), uint32 rows. */
  counts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUInverseDistanceWeighting}.
 *
 * Per-frame (no recompile): the contents of `parameters` (output extent, search radius, power,
 * neighbor limit, minimum neighbor count) and of every input buffer. Topology (needs a new graph):
 * `width`, `height`, sample count, `indexGridSize`, `indexBounds`, `maximumNeighborCount`, and
 * whether `mask` and `output.counts` are present.
 */
export type GPUInverseDistanceWeightingProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'inverse-distance-weighting'`. */
  id?: string;
  /** Packed sample positions in the same units as the extent and search radius. */
  positions: GraphDataView<'float32x2'>;
  /** Packed float32 sample values. NaN values are skipped. */
  values: GraphDataView<'float32'>;
  /** Optional packed `uint32` sample mask; zero skips the sample. */
  mask?: GraphDataView<'uint32'>;
  /** Per-frame float32 view of at least 8 values written with `getGPUInverseDistanceWeightingParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Output raster width in cells. */
  width: number;
  /** Output raster height in cells. */
  height: number;
  /**
   * Dimensions `[columns, rows]` of the internal `GPUGridIndex` over the samples. Aim for a few
   * samples per index cell: the in-cell sort is quadratic in the cell population.
   */
  indexGridSize: readonly [number, number];
  /**
   * Inclusive sample domain `[minX, minY, maxX, maxY]` of the internal grid index. Samples outside
   * it, or with non-finite positions, are ignored.
   */
  indexBounds: readonly [number, number, number, number];
  /**
   * Compile-time capacity of the per-frame nearest-neighbor limit `k`, at most 64. Zero (default)
   * compiles the radius-only kernel and ignores the per-frame `neighborCount`.
   */
  maximumNeighborCount?: number;
  /** Caller-owned raster outputs. */
  output: GPUInverseDistanceWeightingOutput;
};

/**
 * Interpolates scattered samples onto a raster with inverse distance weighting, `w = 1 / d^p`.
 *
 * Gather form, one invocation per output cell, so no atomics touch values. Every encoding rebuilds
 * a `GPUGridIndex` over the samples, sorts the sample IDs inside each index cell ascending (the
 * index leaves in-cell order unspecified), and then visits the index rows overlapping the search
 * square in a fixed order: index row, index column, ascending sample row. Samples with
 * `d^2 <= radius^2` contribute. With a per-frame `k > 0`, the `k` nearest contributors are kept by
 * `(d^2, row)`, so equal distances keep the smallest row, and their weights are summed in that
 * order. An exact hit (`d^2 == 0`) returns that sample's value; several exact hits return the one
 * with the smallest row. Weights are accumulated in log space with a running maximum
 * (`exp2(log2 w - max)`), so tiny distances and large powers do not overflow f32.
 *
 * Output cell `(x, y)` samples its center `extentMin + (x + 0.5, y + 0.5) * cellSize`, with row 0
 * at `minY`. Nodata is NaN. Results are deterministic for a given device; they match a float64
 * CPU oracle to f32 rounding, not bitwise.
 */
export class GPUInverseDistanceWeighting implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'inverse-distance-weighting';
  /** Validated properties. */
  readonly props: GPUInverseDistanceWeightingProps;
  /** `width * height`. */
  readonly cellCount: number;
  /** Compile-time nearest-neighbor capacity. */
  readonly maximumNeighborCount: number;

  constructor(props: GPUInverseDistanceWeightingProps) {
    this.id = props.id ?? this.recipe;
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
    this.maximumNeighborCount = props.maximumNeighborCount ?? 0;
    if (
      !Number.isSafeInteger(this.maximumNeighborCount) ||
      this.maximumNeighborCount < 0 ||
      this.maximumNeighborCount > MAXIMUM_NEIGHBOR_CAPACITY
    ) {
      throw new Error(
        `${id} maximumNeighborCount must be an integer in [0, ${MAXIMUM_NEIGHBOR_CAPACITY}]`
      );
    }
    for (const [name, view] of [
      ['positions', props.positions],
      ['values', props.values],
      ['mask', props.mask],
      ['parameters', props.parameters],
      ['output.values', props.output.values],
      ['output.counts', props.output.counts]
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
    if (props.parameters.length < GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH} float32 values`
      );
    }
    validatePackedView(props.output.values, ['float32'], `${id} output.values`);
    if (props.output.values.length < this.cellCount) {
      throw new Error(`${id} output.values must hold width * height rows`);
    }
    if (props.output.counts) {
      validatePackedUint32View(props.output.counts, `${id} output.counts`);
      if (props.output.counts.length < this.cellCount) {
        throw new Error(`${id} output.counts must hold width * height rows`);
      }
      if (props.output.counts.buffer === props.output.values.buffer) {
        throw new Error(`${id} outputs must not share buffers`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.values, props.output.counts],
      [props.positions, props.values, props.mask, props.parameters]
    );
  }

  /** Returns the grid index build, the in-cell sort, and the per-cell gather, in order. */
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
      output.counts
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
    const sortedIds = createTransientView(graph, `${id}-sorted-ids`, 'uint32', sampleCount);
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
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-sort-cells`,
        operation: OPERATION,
        variant: 'sort-cells',
        bindings: [
          {
            name: 'cellOffsets',
            view: cellOffsets,
            type: 'u32',
            access: 'read'
          },
          {name: 'objectIds', view: objectIds, type: 'u32', access: 'read'},
          {name: 'indexCount', view: indexCount, type: 'u32', access: 'read'},
          {
            name: 'sortedIds',
            view: sortedIds,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: sampleCount,
        declarations: `const INDEX_CELL_COUNT: u32 = ${indexCellCount}u;
const SAMPLE_COUNT: u32 = ${sampleCount}u;`,
        body: `if (index >= min(indexCount[indexCountOffset], SAMPLE_COUNT)) {
    return;
  }
  let sample = objectIds[objectIdsOffset + index];
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
    rank += select(0u, 1u, objectIds[objectIdsOffset + slot] < sample);
  }
  sortedIds[sortedIdsOffset + start + rank] = sample;`
      })
    );

    const bindings: MapGraphKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'sampleValues', view: props.values, type: 'f32', access: 'read'},
      ...(props.mask
        ? [
            {
              name: 'sampleMask',
              view: props.mask,
              type: 'u32',
              access: 'read'
            } as const
          ]
        : []),
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
      {name: 'sortedIds', view: sortedIds, type: 'u32', access: 'read'},
      {
        name: 'valuesOut',
        view: output.values,
        type: 'f32',
        access: 'read_write'
      },
      ...(output.counts
        ? [
            {
              name: 'countsOut',
              view: output.counts,
              type: 'u32',
              access: 'read_write'
            } as const
          ]
        : [])
    ];
    const hasNeighborLimit = maximumNeighborCount > 0;
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-gather`,
        operation: OPERATION,
        variant: hasNeighborLimit ? 'gather-nearest' : 'gather-radius',
        bindings,
        invocationCount: cellCount,
        workgroupSize: 64,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const INDEX_WIDTH: u32 = ${indexGridSize[0]}u;
const INDEX_HEIGHT: u32 = ${indexGridSize[1]}u;
const MAX_NEIGHBORS: u32 = ${maximumNeighborCount}u;
const DOMAIN_MIN: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[0])}, ${getWGSLFloatLiteral(indexBounds[1])});
const DOMAIN_MAX: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[2])}, ${getWGSLFloatLiteral(indexBounds[3])});
${GRID_COORDINATE_WGSL}
${WEIGHT_ACCUMULATOR_WGSL}`,
        body: getGatherBody(Boolean(props.mask), Boolean(output.counts), hasNeighborLimit)
      })
    );
    return nodes;
  }
}

/** Same cell mapping as `GPUGridIndex`, plus finiteness and NaN helpers. */
const GRID_COORDINATE_WGSL = /* wgsl */ `
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }
fn isNanValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
// WGSL rejects NaN constants, so build one from a runtime bit pattern.
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value == minimum) { return 0u; }
  if (value == maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    let scaledValue = value / scale;
    let scaledMinimum = minimum / scale;
    let scaledMaximum = maximum / scale;
    return min(
      u32((scaledValue - scaledMinimum) / (scaledMaximum - scaledMinimum) * f32(size)),
      size - 1u
    );
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}`;

/**
 * Log-space weighted mean: weights are `exp2(logWeight - maximumLogWeight)`, rescaled whenever a
 * larger weight arrives, so every stored weight is in `(0, 1]`.
 */
const WEIGHT_ACCUMULATOR_WGSL = /* wgsl */ `
struct WeightAccumulator {
  maximumLogWeight: f32,
  weightSum: f32,
  weightedSum: f32,
  isEmpty: bool,
}

fn addWeightedSample(accumulator: ptr<function, WeightAccumulator>, power: f32, distanceSquared: f32, value: f32) {
  if (distanceSquared == 0.0) {
    return;
  }
  let logWeight = -0.5 * power * log2(distanceSquared);
  if ((*accumulator).isEmpty) {
    (*accumulator).maximumLogWeight = logWeight;
    (*accumulator).weightSum = 1.0;
    (*accumulator).weightedSum = value;
    (*accumulator).isEmpty = false;
  } else if (logWeight > (*accumulator).maximumLogWeight) {
    let scale = exp2((*accumulator).maximumLogWeight - logWeight);
    (*accumulator).weightSum = (*accumulator).weightSum * scale + 1.0;
    (*accumulator).weightedSum = (*accumulator).weightedSum * scale + value;
    (*accumulator).maximumLogWeight = logWeight;
  } else {
    let weight = exp2(logWeight - (*accumulator).maximumLogWeight);
    (*accumulator).weightSum += weight;
    (*accumulator).weightedSum += weight * value;
  }
}`;

function getGatherBody(hasMask: boolean, hasCounts: boolean, hasNeighborLimit: boolean): string {
  const nearestDeclarations = hasNeighborLimit
    ? `var nearDistances: array<f32, MAX_NEIGHBORS>;
  var nearRows: array<u32, MAX_NEIGHBORS>;
  var nearCount = 0u;`
    : '';
  const contribute = hasNeighborLimit
    ? `if (neighborLimit == 0u) {
          count += 1u;
          addWeightedSample(&accumulator, power, distanceSquared, value);
        } else {
          let last = neighborLimit - 1u;
          let isFull = nearCount >= neighborLimit;
          if (!isFull || distanceSquared < nearDistances[last] ||
              (distanceSquared == nearDistances[last] && sample < nearRows[last])) {
            var position = select(nearCount, last, isFull);
            loop {
              if (position == 0u) { break; }
              let previousDistance = nearDistances[position - 1u];
              let previousRow = nearRows[position - 1u];
              if (previousDistance < distanceSquared ||
                  (previousDistance == distanceSquared && previousRow < sample)) {
                break;
              }
              nearDistances[position] = previousDistance;
              nearRows[position] = previousRow;
              position -= 1u;
            }
            nearDistances[position] = distanceSquared;
            nearRows[position] = sample;
            nearCount = min(nearCount + 1u, neighborLimit);
          }
        }`
    : `count += 1u;
        addWeightedSample(&accumulator, power, distanceSquared, value);`;
  const nearestFinish = hasNeighborLimit
    ? `if (neighborLimit > 0u) {
    count = nearCount;
    for (var near = 0u; near < nearCount; near++) {
      let row = nearRows[near];
      addWeightedSample(&accumulator, power, nearDistances[near], sampleValues[sampleValuesOffset + row]);
    }
  }`
    : '';
  return `let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let extentMin = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let extentMax = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let radius = params[paramsOffset + 4u];
  let power = params[paramsOffset + 5u];
  let neighborLimitValue = params[paramsOffset + 6u];
  let minimumNeighborValue = params[paramsOffset + 7u];
  let neighborLimit = u32(clamp(select(0.0, neighborLimitValue, isFiniteValue(neighborLimitValue)), 0.0, f32(MAX_NEIGHBORS)));
  let minimumNeighbors = max(u32(clamp(select(1.0, minimumNeighborValue, isFiniteValue(minimumNeighborValue)), 0.0, 4294967040.0)), 1u);
  let cellSize = (extentMax - extentMin) / vec2f(f32(WIDTH), f32(HEIGHT));
  let center = extentMin + (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let radiusSquared = radius * radius;
  let queryMin = center - vec2f(radius);
  let queryMax = center + vec2f(radius);
  let valid = isFiniteValue(center.x) && isFiniteValue(center.y) && radius >= 0.0 &&
    isFiniteValue(power) && power >= 0.0 &&
    all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX);
  var accumulator = WeightAccumulator(0.0, 0.0, 0.0, true);
  var count = 0u;
  var exactRow = 0xffffffffu;
  var exactValue = 0.0;
  ${nearestDeclarations}
  if (valid) {
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
    for (var indexRow = rowLow; indexRow <= rowHigh; indexRow++) {
      // Cells of one index row are contiguous in cellOffsets, so the row span is one slot range.
      let start = cellOffsets[cellOffsetsOffset + indexRow * INDEX_WIDTH + columnLow];
      let end = cellOffsets[cellOffsetsOffset + indexRow * INDEX_WIDTH + columnHigh + 1u];
      for (var slot = start; slot < end; slot++) {
        let sample = sortedIds[sortedIdsOffset + slot];
        let value = sampleValues[sampleValuesOffset + sample];
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
        ${contribute}
      }
    }
  }
  ${nearestFinish}
  var result = getNaN();
  if (exactRow != 0xffffffffu) {
    result = exactValue;
  } else if (count >= minimumNeighbors && !accumulator.isEmpty) {
    result = accumulator.weightedSum / accumulator.weightSum;
  }
  valuesOut[valuesOutOffset + index] = result;
  ${hasCounts ? 'countsOut[countsOutOffset + index] = count;' : ''}`;
}
