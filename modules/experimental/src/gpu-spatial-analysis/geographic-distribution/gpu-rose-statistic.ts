// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUCircularStatisticsContract} from '../contracts/index';

/** Five float32 values per group written by {@link GPURoseStatistic}. */
export const GPU_ROSE_STATISTIC_SUMMARY_STRIDE = 5;

/** Summary offsets: valid count, mean angle, resultant length, circular variance, Rayleigh z. */
export const GPU_ROSE_STATISTIC_SUMMARY = {
  count: 0,
  meanAngle: 1,
  resultantLength: 2,
  circularVariance: 3,
  rayleighZ: 4,
  stride: GPU_ROSE_STATISTIC_SUMMARY_STRIDE
} as const;

export type GPURoseStatisticProps = {
  id?: string;
  starts: GraphDataView<'float32x2'>;
  ends: GraphDataView<'float32x2'>;
  weights?: GraphDataView<'float32'>;
  groupIds?: GraphDataView<'uint32'>;
  groupCount?: number;
  mask?: GraphDataView<'uint32'>;
  circular: GPUCircularStatisticsContract;
  /** `groupCount * GPU_ROSE_STATISTIC_SUMMARY_STRIDE` float32 values. */
  summary: GraphDataView<'float32'>;
  /** `groupCount * circular.binCount` weighted observations in angle-bin order. */
  bins: GraphDataView<'float32'>;
};

/**
 * Directional or axial Rose statistic for planar line segments. Directional angles cover
 * `[0, 2 pi)`; axial angles identify opposite directions and cover `[0, pi)`. Equal-width bins
 * use a left-closed/right-open convention. The mean is weighted by `weights` (one by default),
 * zero-length/non-finite lines are excluded, and all sums run in source-row order.
 */
export class GPURoseStatistic implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURoseStatisticProps;
  readonly groupCount: number;

  constructor(props: GPURoseStatisticProps) {
    this.id = props.id ?? 'rose-statistic';
    this.props = props;
    this.groupCount = props.groupCount ?? 1;
    const {id, groupCount} = this;
    validatePackedView(props.starts, ['float32x2'], `${id} starts`);
    validatePackedView(props.ends, ['float32x2'], `${id} ends`);
    if (props.ends.length !== props.starts.length)
      throw new Error(`${id} ends length must equal starts length`);
    for (const [name, view] of [
      ['weights', props.weights],
      ['groupIds', props.groupIds],
      ['mask', props.mask]
    ] as const) {
      if (!view) continue;
      if (name === 'weights') validatePackedView(view, ['float32'], `${id} ${name}`);
      else validatePackedUint32View(view as GraphDataView<'uint32'>, `${id} ${name}`);
      if (view.length !== props.starts.length)
        throw new Error(`${id} ${name} length must equal starts length`);
    }
    if (!Number.isSafeInteger(groupCount) || groupCount < 1)
      throw new Error(`${id} groupCount must be positive`);
    if (!props.groupIds && groupCount !== 1)
      throw new Error(`${id} groupIds are required for multiple groups`);
    if (props.circular.mode !== 'directional' && props.circular.mode !== 'axial')
      throw new Error(`${id} circular.mode is invalid`);
    if (
      props.circular.angleUnit !== 'radians' ||
      props.circular.origin !== 'positive-x-counter-clockwise'
    ) {
      throw new Error(`${id} circular angle convention is unsupported`);
    }
    if (
      !Number.isSafeInteger(props.circular.binCount) ||
      props.circular.binCount < 2 ||
      props.circular.binCount > 360
    ) {
      throw new Error(`${id} circular.binCount must be an integer in [2, 360]`);
    }
    validatePackedView(props.summary, ['float32'], `${id} summary`);
    validatePackedView(props.bins, ['float32'], `${id} bins`);
    if (props.summary.length < groupCount * GPU_ROSE_STATISTIC_SUMMARY_STRIDE)
      throw new Error(`${id} summary is too short`);
    if (props.bins.length < groupCount * props.circular.binCount)
      throw new Error(`${id} bins are too short`);
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.summary, props.bins],
      [props.starts, props.ends, props.weights, props.groupIds, props.mask]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, groupCount} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.starts,
      props.ends,
      props.weights,
      props.groupIds,
      props.mask,
      props.summary,
      props.bins
    ]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'starts', view: props.starts, type: 'f32', access: 'read'},
      {name: 'ends', view: props.ends, type: 'f32', access: 'read'},
      ...(props.weights
        ? [{name: 'weights', view: props.weights, type: 'f32' as const, access: 'read' as const}]
        : []),
      ...(props.groupIds
        ? [{name: 'groupIds', view: props.groupIds, type: 'u32' as const, access: 'read' as const}]
        : []),
      ...(props.mask
        ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
        : []),
      {name: 'summary', view: props.summary, type: 'f32', access: 'read_write'},
      {name: 'bins', view: props.bins, type: 'f32', access: 'read_write'}
    ];
    const axial = props.circular.mode === 'axial';
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-reduce`,
        operation: 'GPURoseStatistic',
        variant: props.circular.mode,
        invocationCount: groupCount,
        bindings,
        declarations: `const ROWS: u32 = ${props.starts.length}u;\nconst BIN_COUNT: u32 = ${props.circular.binCount}u;\nconst PI: f32 = 3.141592653589793;\nconst PERIOD: f32 = ${axial ? 'PI' : '2.0 * PI'};\nfn isFiniteFloat(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }`,
        body: `let group = index;
  let binBase = binsOffset + group * BIN_COUNT;
  for (var bin = 0u; bin < BIN_COUNT; bin++) { bins[binBase + bin] = 0.0; }
  var count = 0u;
  var weightTotal = 0.0;
  var cosine = 0.0;
  var sine = 0.0;
  for (var row = 0u; row < ROWS; row++) {
    if (${props.mask ? 'mask[maskOffset + row] == 0u || ' : ''}${props.groupIds ? 'groupIds[groupIdsOffset + row] != group || ' : ''}false) { continue; }
    let start = vec2f(starts[startsOffset + row * 2u], starts[startsOffset + row * 2u + 1u]);
    let end = vec2f(ends[endsOffset + row * 2u], ends[endsOffset + row * 2u + 1u]);
    let delta = end - start;
    let lengthSquared = dot(delta, delta);
    let weight = ${props.weights ? 'weights[weightsOffset + row]' : '1.0'};
    if (!(lengthSquared > 0.0) || !(weight > 0.0) || !isFiniteFloat(lengthSquared) || !isFiniteFloat(weight)) { continue; }
    var angle = atan2(delta.y, delta.x);
    if (angle < 0.0) { angle += 2.0 * PI; }
    ${axial ? 'if (angle >= PI) { angle -= PI; }' : ''}
    let circularAngle = ${axial ? '2.0 * angle' : 'angle'};
    cosine += weight * cos(circularAngle);
    sine += weight * sin(circularAngle);
    weightTotal += weight;
    count++;
    let bin = min(u32(floor(angle * f32(BIN_COUNT) / PERIOD + 1e-6)), BIN_COUNT - 1u);
    bins[binBase + bin] += weight;
  }
  let base = summaryOffset + group * ${GPU_ROSE_STATISTIC_SUMMARY_STRIDE}u;
  summary[base] = f32(count);
  if (weightTotal > 0.0) {
    var meanAngle = atan2(sine, cosine);
    if (meanAngle < 0.0) { meanAngle += 2.0 * PI; }
    ${axial ? 'meanAngle *= 0.5;' : ''}
    let resultant = sqrt(cosine * cosine + sine * sine) / weightTotal;
    summary[base + 1u] = meanAngle;
    summary[base + 2u] = resultant;
    summary[base + 3u] = 1.0 - resultant;
    summary[base + 4u] = weightTotal * resultant * resultant;
  } else {
    let nan = bitcast<f32>(0x7fc00000u | (group & 0u));
    for (var slot = 1u; slot < ${GPU_ROSE_STATISTIC_SUMMARY_STRIDE}u; slot++) { summary[base + slot] = nan; }
  }`
      })
    ];
  }
}
