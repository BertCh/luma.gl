// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';

/** Workgroup size of the per-row reductions. Fixed, so the summation tree is fixed. @internal */
export const ACCESSIBILITY_ROW_WORKGROUP_SIZE = 256;

/** u32 sentinel for "no edge" and "no node". @internal */
export const ACCESSIBILITY_NONE = 0xffffffff;

/**
 * WGSL helpers shared by the scoring kernels. `parameters` holds
 * `[threshold, decay, beta, minimumCost]`; `decay` is 0 (none), 1 (negative exponential) or 2
 * (power). A cost contributes only when it is finite and `<= threshold`.
 *
 * @internal
 */
export const ACCESSIBILITY_DECAY_WGSL = /* wgsl */ `
fn isWithinThreshold(cost: f32, threshold: f32) -> bool {
  // Unreached entries are +Infinity; an infinite threshold must still reject them.
  return (bitcast<u32>(cost) & 0x7f800000u) != 0x7f800000u && cost >= 0.0 && cost <= threshold;
}
fn getDecay(cost: f32, decay: u32, beta: f32, minimumCost: f32) -> f32 {
  if (decay == 1u) {
    return exp(-beta * cost);
  }
  if (decay == 2u) {
    return pow(max(cost, minimumCost), -beta);
  }
  return 1.0;
}`;

const READ_PARAMETERS = `let threshold = parameters[parametersOffset];
  let decay = u32(max(parameters[parametersOffset + 1u], 0.0));
  let beta = parameters[parametersOffset + 2u];
  let minimumCost = parameters[parametersOffset + 3u];`;

/**
 * Per-column gather over a row-major `[rowCount x columnCount]` cost matrix: one invocation per
 * column sums `rowWeights[row] * f(cost)` over rows in ascending order, so the sum is bitwise
 * reproducible. Writes the step-weighted sum to `cumulative` and the decayed sum to `gravity`.
 *
 * @internal
 */
export function createAccessibilityColumnGatherNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rowCount: number;
    columnCount: number;
    costs: GraphDataView<'float32'>;
    rowWeights: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    cumulative?: GraphDataView<'float32'>;
    gravity?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: MapGraphKernelBinding[] = [
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
    {name: 'rowWeights', view: props.rowWeights, type: 'f32', access: 'read'},
    {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'}
  ];
  if (props.cumulative) {
    bindings.push({
      name: 'cumulative',
      view: props.cumulative,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.gravity) {
    bindings.push({
      name: 'gravity',
      view: props.gravity,
      type: 'f32',
      access: 'read_write'
    });
  }
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'column-gather',
    bindings,
    invocationCount: props.columnCount,
    declarations: `const ROW_COUNT: u32 = ${props.rowCount}u;
const COLUMN_COUNT: u32 = ${props.columnCount}u;
${ACCESSIBILITY_DECAY_WGSL}`,
    body: `${READ_PARAMETERS}
  var cumulativeSum = 0.0;
  var gravitySum = 0.0;
  for (var row = 0u; row < ROW_COUNT; row++) {
    let cost = costs[costsOffset + row * COLUMN_COUNT + index];
    if (!isWithinThreshold(cost, threshold)) {
      continue;
    }
    let weight = rowWeights[rowWeightsOffset + row];
    cumulativeSum += weight;
    gravitySum += weight * getDecay(cost, decay, beta, minimumCost);
  }
  ${props.cumulative ? 'cumulative[cumulativeOffset + index] = cumulativeSum;' : ''}
  ${props.gravity ? 'gravity[gravityOffset + index] = gravitySum;' : ''}`
  });
}

/**
 * Per-row reduction over a row-major `[rowCount x columnCount]` cost matrix with one 256-thread
 * workgroup per row: strided per-thread partial sums of `columnWeights[column] * f(cost)` and a
 * fixed binary tree, so the result is bitwise reproducible.
 *
 * Writes the step-weighted sum to `cumulative` and the decayed sum to `gravity`. With
 * `rowNumerators`, writes `ratios[row] = rowNumerators[row] / gravitySum` (0 when the sum is 0)
 * instead: step one of the two-step floating catchment area.
 *
 * @internal
 */
export function createAccessibilityRowReduceNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    variant: string;
    rowCount: number;
    columnCount: number;
    costs: GraphDataView<'float32'>;
    columnWeights: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    cumulative?: GraphDataView<'float32'>;
    gravity?: GraphDataView<'float32'>;
    rowNumerators?: GraphDataView<'float32'>;
    ratios?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const size = ACCESSIBILITY_ROW_WORKGROUP_SIZE;
  const bindings: MapGraphKernelBinding[] = [
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
    {
      name: 'columnWeights',
      view: props.columnWeights,
      type: 'f32',
      access: 'read'
    },
    {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'}
  ];
  if (props.cumulative) {
    bindings.push({
      name: 'cumulative',
      view: props.cumulative,
      type: 'f32',
      access: 'read_write'
    });
  }
  if (props.gravity) {
    bindings.push({
      name: 'gravity',
      view: props.gravity,
      type: 'f32',
      access: 'read_write'
    });
  }
  const ratios = props.rowNumerators && props.ratios;
  if (props.rowNumerators && props.ratios) {
    bindings.push(
      {
        name: 'rowNumerators',
        view: props.rowNumerators,
        type: 'f32',
        access: 'read'
      },
      {name: 'ratios', view: props.ratios, type: 'f32', access: 'read_write'}
    );
  }
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: props.variant,
    bindings,
    workgroupSize: size,
    invocationCount: props.rowCount * size,
    guardIndex: false,
    declarations: `const COLUMN_COUNT: u32 = ${props.columnCount}u;
var<workgroup> cumulativePartials: array<f32, ${size}>;
var<workgroup> gravityPartials: array<f32, ${size}>;
${ACCESSIBILITY_DECAY_WGSL}`,
    // No early return: every invocation of a workgroup must reach the barriers.
    body: `let row = index / ${size}u;
  let isInRange = index < INVOCATION_COUNT;
  ${READ_PARAMETERS}
  var cumulativeSum = 0.0;
  var gravitySum = 0.0;
  if (isInRange) {
    for (var column = localInvocationIndex; column < COLUMN_COUNT; column += ${size}u) {
      let cost = costs[costsOffset + row * COLUMN_COUNT + column];
      if (isWithinThreshold(cost, threshold)) {
        let weight = columnWeights[columnWeightsOffset + column];
        cumulativeSum += weight;
        gravitySum += weight * getDecay(cost, decay, beta, minimumCost);
      }
    }
  }
  cumulativePartials[localInvocationIndex] = cumulativeSum;
  gravityPartials[localInvocationIndex] = gravitySum;
  workgroupBarrier();
  for (var stride = ${size / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      cumulativePartials[localInvocationIndex] += cumulativePartials[localInvocationIndex + stride];
      gravityPartials[localInvocationIndex] += gravityPartials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    ${props.cumulative ? 'cumulative[cumulativeOffset + row] = cumulativePartials[0];' : ''}
    ${props.gravity ? 'gravity[gravityOffset + row] = gravityPartials[0];' : ''}
    ${
      ratios
        ? `let denominator = gravityPartials[0];
    ratios[ratiosOffset + row] = select(0.0, rowNumerators[rowNumeratorsOffset + row] / denominator, denominator > 0.0);`
        : ''
    }
  }`
  });
}
