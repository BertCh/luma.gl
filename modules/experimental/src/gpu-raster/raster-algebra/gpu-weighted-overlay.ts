// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getBreakSearchWGSL,
  getRasterAlgebraValueWGSL,
  validateRasterAlgebraAliasing,
  validateRasterAlgebraCount,
  validateRasterAlgebraGraph,
  validateRasterAlgebraNoData,
  validateRasterAlgebraView
} from './raster-algebra-utils';
import {
  getGPUWeightedOverlayParameterLength,
  GPU_WEIGHTED_OVERLAY_MAXIMUM_LAYER_COUNT
} from './weighted-overlay-parameters';

const OPERATION = 'GPUWeightedOverlay';

/** Caller-owned outputs of {@link GPUWeightedOverlay}. `score` is required. */
export type GPUWeightedOverlayOutput = {
  /** Weighted score per cell; NaN for nodata or restricted cells. */
  score: GraphDataView<'float32'>;
  /** Optional `1` where `score` is defined, else `0`. */
  validity?: GraphDataView<'uint32'>;
  /** Optional two rows `[minimum, maximum]` of the defined scores; NaN when none is defined. */
  scoreRange?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUWeightedOverlay}.
 *
 * Per-frame (no recompile): stack contents, `parameters` (weights, remap modes and ranges, break
 * counts, normalization, nodata policy), and the remap tables. Topology: `layerCount`,
 * `cellCount`, `maximumBreakCount`, `noDataValue`, and which views exist.
 */
export type GPUWeightedOverlayProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'weighted-overlay'`. */
  id?: string;
  /**
   * Band-sequential float32 stack: layer `i` occupies rows `[i * cellCount, (i + 1) * cellCount)`.
   * NaN cells are nodata.
   */
  stack: GraphDataView<'float32'>;
  /** Layer count in `[1, 16]`. */
  layerCount: number;
  /** Cells per layer. */
  cellCount: number;
  /** Optional finite nodata sentinel compared exactly against input values. */
  noDataValue?: number;
  /**
   * Per-frame float32 view of at least `getGPUWeightedOverlayParameterLength(layerCount)` values
   * written with `getGPUWeightedOverlayParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /** Table-mode breaks, `layerCount * maximumBreakCount` rows; layer `i` uses its own block. */
  remapBreaks?: GraphDataView<'float32'>;
  /** Table-mode class values, `layerCount * (maximumBreakCount + 1)` rows. */
  remapValues?: GraphDataView<'float32'>;
  /** Compile-time table capacity per layer. Required with remap tables. */
  maximumBreakCount?: number;
  /** Caller-owned outputs. */
  output: GPUWeightedOverlayOutput;
};

/**
 * Weighted overlay (suitability analysis) of up to 16 float32 rasters.
 *
 * Each layer value is remapped, either linearly from `[inputMin, inputMax]` to `[0, 1]` (clamped,
 * optionally inverted) or through a per-layer break table to a class value, then
 * `score = sum(weight[i] * remap[i](value[i]))` is accumulated in f32 in fixed layer order, so the
 * result is deterministic. A NaN class value marks a restricted class (ArcGIS "Restricted"): the
 * cell's score is NaN. With `normalizeWeights` the sum is divided by the sum of absolute weights of
 * the contributing layers. Weights, remap tables, and policies are per-frame parameters, so weight
 * sliders never recompile. `scoreRange` reduces the defined scores with order-preserving integer
 * atomics.
 */
export class GPUWeightedOverlay implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUWeightedOverlayProps;
  /** Compile-time table capacity per layer (0 without tables). */
  readonly maximumBreakCount: number;

  constructor(props: GPUWeightedOverlayProps) {
    this.id = props.id ?? 'weighted-overlay';
    this.props = props;
    const {id} = this;
    const {output, layerCount, cellCount} = props;
    validateRasterAlgebraCount(
      id,
      'layerCount',
      layerCount,
      1,
      GPU_WEIGHTED_OVERLAY_MAXIMUM_LAYER_COUNT
    );
    validateRasterAlgebraCount(id, 'cellCount', cellCount, 0, 0xffffffff);
    if (layerCount * cellCount > 0xffffffff) {
      throw new Error(`${id} layerCount * cellCount must fit in uint32`);
    }
    validateRasterAlgebraNoData(id, props.noDataValue);
    validateRasterAlgebraView(id, 'stack', props.stack, 'float32', layerCount * cellCount);
    validateRasterAlgebraView(
      id,
      'parameters',
      props.parameters,
      'float32',
      getGPUWeightedOverlayParameterLength(layerCount)
    );
    if (Boolean(props.remapBreaks) !== Boolean(props.remapValues)) {
      throw new Error(`${id} remapBreaks and remapValues must be provided together`);
    }
    this.maximumBreakCount = props.remapBreaks ? (props.maximumBreakCount ?? 0) : 0;
    if (props.remapBreaks) {
      validateRasterAlgebraCount(id, 'maximumBreakCount', this.maximumBreakCount, 1, 1 << 16);
      validateRasterAlgebraView(
        id,
        'remapBreaks',
        props.remapBreaks,
        'float32',
        layerCount * this.maximumBreakCount
      );
      validateRasterAlgebraView(
        id,
        'remapValues',
        props.remapValues,
        'float32',
        layerCount * (this.maximumBreakCount + 1)
      );
    }
    if (!output?.score) {
      throw new Error(`${id} needs output.score`);
    }
    validateRasterAlgebraView(id, 'output.score', output.score, 'float32', cellCount);
    validateRasterAlgebraView(id, 'output.validity', output.validity, 'uint32', cellCount);
    validateRasterAlgebraView(id, 'output.scoreRange', output.scoreRange, 'float32', 2);
    validateRasterAlgebraAliasing(
      id,
      [output.score, output.validity, output.scoreRange],
      [props.stack, props.parameters, props.remapBreaks, props.remapValues]
    );
  }

  /** Returns the score kernel and, with `scoreRange`, a key clear and a decode kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumBreakCount} = this;
    const {output, layerCount, cellCount} = props;
    validateRasterAlgebraGraph(id, graph, [
      props.stack,
      props.parameters,
      props.remapBreaks,
      props.remapValues,
      output.score,
      output.validity,
      output.scoreRange
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const rangeKeys = output.scoreRange
      ? createTransientView(graph, `${id}-range-keys`, 'uint32', 2)
      : undefined;
    if (rangeKeys) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-clear-range`,
          operation: OPERATION,
          variant: 'clear-range',
          bindings: [{name: 'keys', view: rangeKeys, type: 'u32', access: 'read_write'}],
          invocationCount: 1,
          body: `keys[keysOffset] = 0xffffffffu;
  keys[keysOffset + 1u] = 0u;`
        })
      );
    }
    const hasTables = Boolean(props.remapBreaks);
    const bindings: WGSLKernelBinding[] = [
      {name: 'stack', view: props.stack, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (hasTables) {
      bindings.push(
        {name: 'remapBreaks', view: props.remapBreaks!, type: 'f32', access: 'read'},
        {name: 'remapValues', view: props.remapValues!, type: 'f32', access: 'read'}
      );
    }
    bindings.push({name: 'scoreOut', view: output.score, type: 'f32', access: 'read_write'});
    if (output.validity) {
      bindings.push({
        name: 'validityOut',
        view: output.validity,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (rangeKeys) {
      bindings.push({
        name: 'rangeKeys',
        view: rangeKeys,
        type: 'atomic<u32>',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-score`,
        operation: OPERATION,
        variant: 'score',
        bindings,
        invocationCount: cellCount,
        // The range reduction needs workgroup barriers, so every invocation runs the body.
        guardIndex: !rangeKeys,
        declarations: `const LAYER_COUNT: u32 = ${layerCount}u;
const CELL_COUNT: u32 = ${cellCount}u;
const MAXIMUM_BREAK_COUNT: u32 = ${maximumBreakCount}u;
${
  rangeKeys
    ? `var<workgroup> tileMinimum: array<u32, 256>;
var<workgroup> tileMaximum: array<u32, 256>;`
    : ''
}
${getRasterAlgebraValueWGSL(props.noDataValue)}
${hasTables ? getBreakSearchWGSL('countBreaksBelow', 'remapBreaks') : ''}
fn getOrderedKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return bits ^ select(0x80000000u, 0xffffffffu, (bits >> 31u) != 0u);
}`,
        body: `${rangeKeys ? 'var minimumKey = 0xffffffffu;\n  var maximumKey = 0u;\n  if (index < CELL_COUNT) {' : ''}
  let normalizeWeights = params[paramsOffset] != 0.0;
  let ignoreNoData = params[paramsOffset + 1u] != 0.0;
  var sum = 0.0;
  var weightSum = 0.0;
  var usedCount = 0u;
  var isDefined = true;
  for (var layer = 0u; layer < LAYER_COUNT; layer++) {
    let base = paramsOffset + 8u + layer * 8u;
    let weight = params[base];
    let value = stack[stackOffset + layer * CELL_COUNT + index];
    if (isNoDataValue(value)) {
      if (!ignoreNoData) {
        isDefined = false;
      }
      continue;
    }
    var remapped = 0.0;
    if (params[base + 1u] == 0.0) {
      let inputMin = params[base + 2u];
      if (params[base + 7u] != 0.0) {
        remapped = select(0.0, 1.0, value >= inputMin);
      } else {
        remapped = clamp((value - inputMin) * params[base + 3u], 0.0, 1.0);
      }
      if (params[base + 4u] != 0.0) {
        remapped = 1.0 - remapped;
      }
    } else {
      ${
        hasTables
          ? `let breakCountValue = params[base + 5u];
      let breakCount = min(u32(clamp(select(0.0, breakCountValue, isFiniteValue(breakCountValue)), 0.0, 4294967040.0)), MAXIMUM_BREAK_COUNT);
      let classIndex = countBreaksBelow(layer * MAXIMUM_BREAK_COUNT, breakCount, value, params[base + 6u] != 0.0);
      remapped = remapValues[remapValuesOffset + layer * (MAXIMUM_BREAK_COUNT + 1u) + classIndex];`
          : 'remapped = getNaN();'
      }
    }
    if (isNaNValue(remapped)) {
      // Restricted class: the cell is excluded whatever the nodata policy.
      isDefined = false;
    }
    sum = sum + weight * remapped;
    weightSum = weightSum + abs(weight);
    usedCount = usedCount + 1u;
  }
  isDefined = isDefined && usedCount > 0u;
  var score = sum;
  if (normalizeWeights) {
    isDefined = isDefined && weightSum > 0.0;
    score = sum / weightSum;
  }
  score = select(getNaN(), score, isDefined && !isNaNValue(score));
  let isScoreDefined = !isNaNValue(score);
  scoreOut[scoreOutOffset + index] = score;
  ${output.validity ? 'validityOut[validityOutOffset + index] = select(0u, 1u, isScoreDefined);' : ''}
  ${
    rangeKeys
      ? `if (isScoreDefined) {
    let key = getOrderedKey(score);
    minimumKey = key;
    maximumKey = key;
  }
  }
  // Combine the workgroup's keys in a tree; one pair of global atomics per workgroup, not per cell.
  tileMinimum[localInvocationIndex] = minimumKey;
  tileMaximum[localInvocationIndex] = maximumKey;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      tileMinimum[localInvocationIndex] = min(tileMinimum[localInvocationIndex], tileMinimum[localInvocationIndex + stride]);
      tileMaximum[localInvocationIndex] = max(tileMaximum[localInvocationIndex], tileMaximum[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && tileMinimum[0] <= tileMaximum[0]) {
    atomicMin(&rangeKeys[rangeKeysOffset], tileMinimum[0]);
    atomicMax(&rangeKeys[rangeKeysOffset + 1u], tileMaximum[0]);
  }`
      : ''
  }`
      })
    );
    if (rangeKeys && output.scoreRange) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-decode-range`,
          operation: OPERATION,
          variant: 'decode-range',
          bindings: [
            {name: 'keys', view: rangeKeys, type: 'u32', access: 'read'},
            {name: 'rangeOut', view: output.scoreRange, type: 'f32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: `fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}
fn decodeOrderedKey(key: u32) -> f32 {
  return bitcast<f32>(key ^ select(0xffffffffu, 0x80000000u, (key >> 31u) != 0u));
}`,
          body: `let minimumKey = keys[keysOffset];
  let maximumKey = keys[keysOffset + 1u];
  let isEmpty = minimumKey > maximumKey;
  rangeOut[rangeOutOffset] = select(decodeOrderedKey(minimumKey), getNaN(), isEmpty);
  rangeOut[rangeOutOffset + 1u] = select(decodeOrderedKey(maximumKey), getNaN(), isEmpty);`
        })
      );
    }
    return nodes;
  }
}
