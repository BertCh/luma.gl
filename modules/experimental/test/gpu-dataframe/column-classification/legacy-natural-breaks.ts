// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  CLASS_BREAKS_BIN_WGSL,
  CLASS_BREAKS_EXTREMES,
  getClassBreaksDispatchWGSL,
  getClassBreaksIndirectGate
} from '../../../src/gpu-dataframe/column-classification/class-breaks-kernels';
import {COLUMN_ORDERED_KEY_WGSL} from '../../../src/gpu-dataframe/column-classification/column-classification-shared';
import {
  createWGSLKernelNode,
  getAutomaticWGSLKernelWorkgroupSize,
  getWGSLFloatLiteral
} from '../../../src/utils/wgsl-kernel-nodes';

/**
 * The natural-breaks pipeline as it was before the monotone divide and conquer: an equal-width
 * histogram, an O(B^2) table of class costs (Welford per start bin), one O(B) scan per bin and per
 * layer, and a backtrack whose breaks snap to bin edges. Test-only, for the same-run A/B timing
 * in gpu-class-breaks.spec.ts; it computes `e[0..k]` of one column for a fixed class count and has
 * none of GPUClassBreaks' other methods, masks or degenerate-data rules beyond a constant column.
 */
export class LegacyNaturalBreaks implements GPUCommandNodeProducer {
  constructor(
    readonly props: {
      id: string;
      values: GraphDataView<'float32'>;
      classCount: number;
      binCount: number;
      /** `classCount + 1` rows receiving `e[0..k]` as f32. */
      breaks: GraphDataView<'float32'>;
    }
  ) {}

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, values, classCount, binCount, breaks} = this.props;
    const rowCount = values.length;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const extremes = u32('extremes', 4);
    const histogram = u32('histogram', binCount);
    const costs = createTransientView(graph, `${id}-costs`, 'float32', binCount * binCount);
    const totals = createTransientView(graph, `${id}-totals`, 'float32', classCount * binCount);
    const starts = u32('starts', classCount * binCount);
    const layerState = u32('layer-state', 1);
    const layerDispatch = createTransientView(
      graph,
      `${id}-layer-dispatch`,
      'uint32',
      3,
      Buffer.STORAGE | Buffer.INDIRECT
    );
    const operation = 'LegacyNaturalBreaks';
    const binDeclarations = `${COLUMN_ORDERED_KEY_WGSL}
const BIN_COUNT: u32 = ${binCount}u;
const INVERSE_BIN_COUNT: f32 = ${getWGSLFloatLiteral(1 / binCount)};
${CLASS_BREAKS_BIN_WGSL}`;
    const extremesRead = `let minimum = decodeOrderedKey(extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u]);
  let maximum = decodeOrderedKey(extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u]);
  let width = (maximum - minimum) * INVERSE_BIN_COUNT;`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reset`,
        operation,
        bindings: [
          {name: 'extremes', view: extremes, type: 'u32', access: 'read_write'},
          {name: 'layerState', view: layerState, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u] = 0xffffffffu;
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u] = 0u;
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.validCount}u] = 0u;
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.finiteCount}u] = 0u;
  layerState[layerStateOffset] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-extremes`,
        operation,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'extremes', view: extremes, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: COLUMN_ORDERED_KEY_WGSL,
        body: `let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return;
  }
  let key = getOrderedKey(value);
  atomicMin(&extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u], key);
  atomicMax(&extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u], key);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation,
        bindings: [{name: 'histogram', view: histogram, type: 'u32', access: 'read_write'}],
        invocationCount: binCount,
        body: 'histogram[histogramOffset + index] = 0u;'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-histogram`,
        operation,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'extremes', view: extremes, type: 'u32', access: 'read'},
          {name: 'histogram', view: histogram, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: binDeclarations,
        body: `let value = values[valuesOffset + index];
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return;
  }
  ${extremesRead}
  var bin = 0u;
  if (width > 0.0) {
    bin = getBinIndex(value, minimum, width);
  }
  atomicAdd(&histogram[histogramOffset + bin], 1u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-class-costs`,
        operation,
        bindings: [
          {name: 'histogram', view: histogram, type: 'u32', access: 'read'},
          {name: 'costs', view: costs, type: 'f32', access: 'read_write'}
        ],
        invocationCount: binCount,
        declarations: `const BIN_COUNT: u32 = ${binCount}u;`,
        body: `let start = index;
  var weight = 0.0;
  var mean = 0.0;
  var squares = 0.0;
  for (var end = start; end < BIN_COUNT; end = end + 1u) {
    let count = f32(histogram[histogramOffset + end]);
    if (count > 0.0) {
      let center = f32(end);
      let total = weight + count;
      let delta = center - mean;
      mean = mean + delta * (count / total);
      squares = squares + count * delta * (center - mean);
      weight = total;
    }
    costs[costsOffset + start * BIN_COUNT + end] = squares;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-layer-0`,
        operation,
        bindings: [
          {name: 'costs', view: costs, type: 'f32', access: 'read'},
          {name: 'totals', view: totals, type: 'f32', access: 'read_write'},
          {name: 'starts', view: starts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: binCount,
        declarations: `const BIN_COUNT: u32 = ${binCount}u;`,
        body: `totals[totalsOffset + index] = costs[costsOffset + index];
  starts[startsOffset + index] = 0u;`
      })
    );
    const layerWorkgroupSize = getAutomaticWGSLKernelWorkgroupSize(binCount);
    for (let layer = 1; layer < classCount; layer++) {
      const layerId = `${id}-layer-${layer}`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${layerId}-advance`,
          operation,
          bindings: [
            {name: 'layerState', view: layerState, type: 'u32', access: 'read_write'},
            {name: 'dispatch', view: layerDispatch, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          body: `layerState[layerStateOffset] = layerState[layerStateOffset] + 1u;
  ${getClassBreaksDispatchWGSL(graph, {
    operation,
    slot: 0,
    invocationCount: binCount,
    workgroupSize: layerWorkgroupSize,
    activeExpression: 'true'
  })}`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: layerId,
          operation,
          bindings: [
            {name: 'layerState', view: layerState, type: 'u32', access: 'read'},
            {name: 'costs', view: costs, type: 'f32', access: 'read'},
            {name: 'totals', view: totals, type: 'f32', access: 'read_write'},
            {name: 'starts', view: starts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: binCount,
          workgroupSize: layerWorkgroupSize,
          ...getClassBreaksIndirectGate<Parameters>(layerId, layerDispatch, 0),
          declarations: `const BIN_COUNT: u32 = ${binCount}u;`,
          body: `let layer = layerState[layerStateOffset];
  let end = index;
  var best = 3.0e38;
  var bestStart = layer;
  for (var start = layer; start <= end; start = start + 1u) {
    let candidate = totals[totalsOffset + (layer - 1u) * BIN_COUNT + start - 1u] + costs[costsOffset + start * BIN_COUNT + end];
    if (candidate < best) {
      best = candidate;
      bestStart = start;
    }
  }
  totals[totalsOffset + layer * BIN_COUNT + end] = best;
  starts[startsOffset + layer * BIN_COUNT + end] = bestStart;`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-backtrack`,
        operation,
        bindings: [
          {name: 'extremes', view: extremes, type: 'u32', access: 'read'},
          {name: 'starts', view: starts, type: 'u32', access: 'read'},
          {name: 'breaks', view: breaks, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: binDeclarations,
        body: `${extremesRead}
  var end = BIN_COUNT - 1u;
  breaks[breaksOffset] = minimum;
  breaks[breaksOffset + ${classCount}u] = maximum;
  for (var layer = ${classCount - 1}u; layer > 0u; layer = layer - 1u) {
    let start = starts[startsOffset + layer * BIN_COUNT + end];
    breaks[breaksOffset + layer] = minimum + f32(start) * width;
    end = start - 1u;
  }`
      })
    );
    return nodes;
  }
}
