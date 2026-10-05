// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUScanUint64,
  GPUTranspose,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBufferBand} from '../../gpu-raster/index';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {TERRAIN_WGSL_HELPERS} from '../terrain-analysis/terrain-analysis-utils';

/**
 * Largest quantized magnitude accepted by the quantize kernel. Elevations whose quantized value
 * would leave the signed 32-bit range are treated as invalid.
 *
 * @internal
 */
export const TERRAIN_SUMMED_AREA_MAXIMUM_QUANTIZED = 2147483520;

/**
 * WGSL helpers for modular 64-bit integer arithmetic on `vec2<u32>` (low, high) words, plus the
 * shared elevation quantizer. Every operation wraps modulo 2^64, which is what makes box sums
 * from a modular summed-area table exact. Requires an `INVERSE_QUANTUM` constant.
 *
 * @internal
 */
export const TERRAIN_SUMMED_AREA_WGSL_HELPERS = /* wgsl */ `
${TERRAIN_WGSL_HELPERS}
const MAXIMUM_QUANTIZED: f32 = ${getWGSLFloatLiteral(TERRAIN_SUMMED_AREA_MAXIMUM_QUANTIZED)};
fn add64(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  let low = left.x + right.x;
  return vec2<u32>(low, left.y + right.y + select(0u, 1u, low < left.x));
}
fn subtract64(left: vec2<u32>, right: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(left.x - right.x, left.y - right.y - select(0u, 1u, left.x < right.x));
}
fn multiplyWide32(left: u32, right: u32) -> vec2<u32> {
  let left0 = left & 0xffffu;
  let left1 = left >> 16u;
  let right0 = right & 0xffffu;
  let right1 = right >> 16u;
  let product00 = left0 * right0;
  let product01 = left0 * right1;
  let product10 = left1 * right0;
  let product11 = left1 * right1;
  let middle = (product00 >> 16u) + (product01 & 0xffffu) + (product10 & 0xffffu);
  return vec2<u32>(
    (product00 & 0xffffu) | (middle << 16u),
    product11 + (product01 >> 16u) + (product10 >> 16u) + (middle >> 16u)
  );
}
fn multiply64By32(left: vec2<u32>, right: u32) -> vec2<u32> {
  let wide = multiplyWide32(left.x, right);
  return vec2<u32>(wide.x, wide.y + left.y * right);
}
fn negate64(value: vec2<u32>) -> vec2<u32> {
  return subtract64(vec2<u32>(0u, 0u), value);
}
fn unsigned64ToFloat(value: vec2<u32>) -> f32 {
  return f32(value.y) * 4294967296.0 + f32(value.x);
}
// Converts the magnitude, not the two's-complement words: f32(hi) * 2^32 + f32(lo) of a small
// negative value cancels -2^32 against a low word rounded up to 2^32.
fn signed64ToFloat(value: vec2<u32>) -> f32 {
  if (bitcast<i32>(value.y) < 0i) {
    return -unsigned64ToFloat(negate64(value));
  }
  return unsigned64ToFloat(value);
}
fn isQuantizable(elevation: f32) -> bool {
  return isFiniteValue(elevation) && abs(elevation * INVERSE_QUANTUM) <= MAXIMUM_QUANTIZED;
}
fn quantizeElevation(elevation: f32) -> i32 {
  return i32(round(elevation * INVERSE_QUANTUM));
}`;

/** Result of {@link addTerrainSummedAreaTableNodes}. @internal */
export type TerrainSummedAreaTable<Parameters> = {
  /**
   * Low words: three planes of `width * height` words (sum low, square low, count), each the
   * column-major inclusive prefix of the row-major inclusive prefix. Index
   * `plane * pixelCount + column * height + row`. See {@link TERRAIN_SUMMED_AREA_BOX_WGSL}.
   */
  lowTable: GraphDataView<'uint32'>;
  /** High words: two planes (sum high, square high) with the same layout. */
  highTable: GraphDataView<'uint32'>;
  /** Command nodes in execution order. */
  nodes: GPUCommandNode<Parameters>[];
};

/**
 * Adds the nodes that build an exact modular summed-area table of quantized elevations.
 *
 * Elevations are quantized to signed integers `q = round(z / quantum)`. A quantize kernel writes
 * `q` (sign-extended to 64 bits), `q²` (64 bits), and a valid-cell flag; invalid cells contribute
 * zero. One global row-major inclusive scan (`GPUScanUint64` / `GPUScan`), a `GPUTranspose`, and a
 * second global scan over the column-major data produce the table. No segment restarts are needed:
 * every box sum is a difference of table entries, and modular arithmetic makes those differences
 * exact whenever the true box sum fits in 64 bits, however large the table entries become. A
 * single-cell box returns that cell's quantized value and validity, so consumers need no other
 * elevation binding.
 *
 * Two ping-pong stages of `5 * width * height` words hold the intermediate results; low and high
 * words live in separate buffers as `GPUScanUint64` requires.
 *
 * @internal
 */
export function addTerrainSummedAreaTableNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    elevation: GPURasterBufferBand;
    quantum: number;
  }
): TerrainSummedAreaTable<Parameters> {
  const {id, width, height} = props;
  const pixelCount = width * height;
  const byteLength = 3 * pixelCount * 4;
  if (byteLength > graph.device.limits.maxStorageBufferBindingSize) {
    throw new Error(
      `${id} summed-area table needs ${byteLength}-byte bindings, more than the device ` +
        `maxStorageBufferBindingSize of ${graph.device.limits.maxStorageBufferBindingSize}`
    );
  }
  const createStage = (stage: string) => ({
    low: createTransientView(graph, `${id}-summed-area-${stage}-low`, 'uint32', 3 * pixelCount),
    high: createTransientView(graph, `${id}-summed-area-${stage}-high`, 'uint32', 2 * pixelCount)
  });
  const first = createStage('first');
  const second = createStage('second');
  type Stage = typeof first;
  const getPlane = (view: GraphDataView<'uint32'>, plane: number): GraphDataView<'uint32'> =>
    graph.createDataView(view.buffer, {
      format: 'uint32',
      length: pixelCount,
      byteOffset: view.byteOffset + plane * pixelCount * 4
    });
  const elevationValidity = props.elevation.validity;
  if (!elevationValidity) {
    throw new Error(`${id} summed-area table requires a canonical elevation band`);
  }
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-summed-area-quantize`,
      operation: 'GPUTerrainTopographicPosition',
      variant: 'quantize',
      bindings: [
        {
          name: 'elevationValues',
          view: props.elevation.storage.values,
          type: 'f32',
          access: 'read'
        },
        {name: 'elevationValidity', view: elevationValidity, type: 'u32', access: 'read'},
        {name: 'lowPlanes', view: first.low, type: 'u32', access: 'read_write'},
        {name: 'highPlanes', view: first.high, type: 'u32', access: 'read_write'}
      ],
      invocationCount: pixelCount,
      declarations: `const PIXEL_COUNT: u32 = ${pixelCount}u;
const INVERSE_QUANTUM: f32 = ${getWGSLFloatLiteral(1 / props.quantum)};
${TERRAIN_SUMMED_AREA_WGSL_HELPERS}`,
      body: `let elevation = elevationValues[elevationValuesOffset + index];
  let isValid = elevationValidity[elevationValidityOffset + index] != 0u && isQuantizable(elevation);
  var quantized = 0i;
  if (isValid) {
    quantized = quantizeElevation(elevation);
  }
  let magnitude = u32(abs(quantized));
  let square = multiplyWide32(magnitude, magnitude);
  lowPlanes[lowPlanesOffset + index] = bitcast<u32>(quantized);
  lowPlanes[lowPlanesOffset + PIXEL_COUNT + index] = square.x;
  lowPlanes[lowPlanesOffset + 2u * PIXEL_COUNT + index] = select(0u, 1u, isValid);
  highPlanes[highPlanesOffset + index] = select(0u, 0xffffffffu, quantized < 0i);
  highPlanes[highPlanesOffset + PIXEL_COUNT + index] = square.y;`
    })
  ];
  const addScans = (stage: string, input: Stage, output: Stage) => {
    for (const [name, plane] of [
      ['sum', 0],
      ['square', 1]
    ] as const) {
      nodes.push(
        ...new GPUScanUint64({
          id: `${id}-summed-area-${stage}-${name}`,
          inputLow: getPlane(input.low, plane),
          inputHigh: getPlane(input.high, plane),
          outputLow: getPlane(output.low, plane),
          outputHigh: getPlane(output.high, plane)
        }).getCommandNodes(graph)
      );
    }
    nodes.push(
      ...new GPUScan({
        id: `${id}-summed-area-${stage}-count`,
        input: getPlane(input.low, 2),
        output: getPlane(output.low, 2),
        mode: 'inclusive'
      }).getCommandNodes(graph)
    );
  };
  // Row-major inclusive prefix: first -> second.
  addScans('rows', first, second);
  // Transpose every plane: second -> first (column-major).
  for (const [word, planeCount] of [
    ['low', 3],
    ['high', 2]
  ] as const) {
    for (let plane = 0; plane < planeCount; plane++) {
      nodes.push(
        ...new GPUTranspose({
          id: `${id}-summed-area-transpose-${word}-${plane}`,
          input: getPlane(second[word], plane),
          output: getPlane(first[word], plane),
          rows: height,
          columns: width
        }).getCommandNodes(graph)
      );
    }
  }
  // Column-major inclusive prefix of the row prefixes: first -> second.
  addScans('columns', first, second);
  return {lowTable: second.low, highTable: second.high, nodes};
}

/**
 * WGSL box-sum reader for a table built by {@link addTerrainSummedAreaTableNodes}.
 *
 * Requires bindings named `lowTable` and `highTable`, constants `WIDTH`, `HEIGHT`, `PIXEL_COUNT`, and
 * {@link TERRAIN_SUMMED_AREA_WGSL_HELPERS}. `readBoxSums(firstColumn, lastColumn, firstRow,
 * lastRow)` returns the exact sums over the inclusive, in-raster box.
 *
 * Derivation: with `P` the row-major inclusive prefix and `Q` the column-major inclusive prefix of
 * `P`, a column range `sum_{r=r0}^{r1} P[r, c]` is `Q[cH + r1] - Q[cH + r0 - 1]`, and a box is
 * the difference of two column ranges. The left column range for `firstColumn = 0` is the linear
 * predecessor `P[r - 1, W - 1]`, which is column `W - 1` shifted up one row.
 *
 * @internal
 */
export const TERRAIN_SUMMED_AREA_BOX_WGSL = /* wgsl */ `
struct TerrainBoxSums {
  sum: vec2<u32>,
  square: vec2<u32>,
  count: u32,
}
fn readTablePlane64(plane: u32, linearIndex: i32) -> vec2<u32> {
  if (linearIndex < 0i) {
    return vec2<u32>(0u, 0u);
  }
  let offset = plane * PIXEL_COUNT + u32(linearIndex);
  return vec2<u32>(lowTable[lowTableOffset + offset], highTable[highTableOffset + offset]);
}
fn readTableCount(linearIndex: i32) -> u32 {
  if (linearIndex < 0i) {
    return 0u;
  }
  return lowTable[lowTableOffset + 2u * PIXEL_COUNT + u32(linearIndex)];
}
fn readColumnRange(column: i32, firstRow: i32, lastRow: i32) -> TerrainBoxSums {
  var upper: i32;
  var lower: i32;
  if (column >= 0i) {
    upper = column * i32(HEIGHT) + lastRow;
    lower = column * i32(HEIGHT) + firstRow - 1i;
  } else {
    let lastColumnStart = (i32(WIDTH) - 1i) * i32(HEIGHT);
    upper = lastColumnStart + lastRow - 1i;
    lower = lastColumnStart + max(firstRow, 1i) - 2i;
  }
  return TerrainBoxSums(
    subtract64(readTablePlane64(0u, upper), readTablePlane64(0u, lower)),
    subtract64(readTablePlane64(1u, upper), readTablePlane64(1u, lower)),
    readTableCount(upper) - readTableCount(lower)
  );
}
fn readBoxSums(firstColumn: i32, lastColumn: i32, firstRow: i32, lastRow: i32) -> TerrainBoxSums {
  let right = readColumnRange(lastColumn, firstRow, lastRow);
  let left = readColumnRange(firstColumn - 1i, firstRow, lastRow);
  return TerrainBoxSums(
    subtract64(right.sum, left.sum),
    subtract64(right.square, left.square),
    right.count - left.count
  );
}`;
