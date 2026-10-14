// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {validateGPUStatusPort} from '../../analysis-core/contracts/index';
import type {GPURegionMaskOutput} from './region-statistics-types';

/** Properties for {@link GPUPickRegionMask}. */
export type GPUPickRegionMaskProps = {
  /** ID prefix. Defaults to `'pick-region-mask'`. */
  id?: string;
  /**
   * Packed `GPUIndexPickingTarget.addRegionPass` result `[count, overflow, (objectIndex,
   * batchIndex)...]`. Its length must be at least 4.
   */
  picks: GraphDataView<'uint32'>;
  /** Compile-time batch filter in `[0, 2^31 - 1]`. When omitted, every batch is accepted. */
  batchIndex?: number;
  /**
   * Caller-owned outputs: `mask`, whose length is the source row count (cleared, then set to 1
   * for picked rows), and `status.overflow`, which copies the pick result's overflow word as 0 or
   * 1. Both always written.
   */
  output: GPURegionMaskOutput;
};

/**
 * Converts a capacity-bounded index-picking region result into a deduplicated source-aligned 0/1
 * mask. The application adds `GPUIndexPickingTarget.addRegionPass` before this contributor; the graph
 * orders the two through the hazard on `picks`.
 */
export class GPUPickRegionMask implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPickRegionMaskProps;

  constructor(props: GPUPickRegionMaskProps) {
    this.id = props.id ?? 'pick-region-mask';
    this.props = props;
    const {id} = this;
    const {picks, output} = props;
    validatePackedUint32View(picks, `${id} picks`);
    if (picks.length < 4) {
      throw new Error(`${id} picks must hold at least one picked pair`);
    }
    if (props.batchIndex !== undefined) {
      validateCapacity(id, 'batchIndex', props.batchIndex, 0, 0x7fffffff);
    }
    validatePackedUint32View(output.mask, `${id} output.mask`);
    validateGPUStatusPort(`${id} output.status`, output.status);
    if (output.mask.length < 1 || !output.status.overflow) {
      throw new Error(`${id} output.mask and output.status.overflow are required`);
    }
    validateGraphOutputsDisjointFromInputs(id, [output.mask, output.status.overflow], [picks]);
  }

  /** Returns clear, scatter, and overflow nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {picks: result, batchIndex} = props;
    const outputMask = props.output.mask;
    const {overflow} = props.output.status;
    validateGraphViewsBelongToGraph(id, graph, [result, outputMask, overflow]);
    const pairCapacity = Math.floor((result.length - 2) / 2);
    return [
      createFillNode<Parameters>(graph, {
        id: `${id}-clear`,
        operation: 'GPUPickRegionMask',
        view: outputMask,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scatter`,
        operation: 'GPUPickRegionMask',
        variant: 'scatter',
        bindings: [
          {name: 'result', view: result, type: 'u32', access: 'read'},
          {name: 'outputMask', view: outputMask, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: `const ROW_COUNT: u32 = ${outputMask.length}u;
const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
        body: `if (index >= min(result[resultOffset], PAIR_CAPACITY)) { return; }
  let objectIndex = bitcast<i32>(result[resultOffset + 2u + index * 2u]);
  ${
    batchIndex === undefined
      ? ''
      : `let batchIndex = bitcast<i32>(result[resultOffset + 3u + index * 2u]);
  if (batchIndex != ${batchIndex}i) { return; }`
  }
  if (objectIndex < 0 || u32(objectIndex) >= ROW_COUNT) { return; }
  atomicStore(&outputMask[outputMaskOffset + u32(objectIndex)], 1u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-overflow`,
        operation: 'GPUPickRegionMask',
        variant: 'overflow',
        bindings: [
          {name: 'result', view: result, type: 'u32', access: 'read'},
          {name: 'overflow', view: overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: 'overflow[overflowOffset] = select(0u, 1u, result[resultOffset + 1u] != 0u);'
      })
    ];
  }
}
