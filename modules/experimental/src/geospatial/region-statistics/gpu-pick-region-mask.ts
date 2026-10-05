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
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';

/** Properties for {@link GPUPickRegionMask}. */
export type GPUPickRegionMaskProps = {
  /** ID prefix. Defaults to `'pick-region-mask'`. */
  id?: string;
  /** Packed `GPUIndexPickingTarget.addRegionPass` result. Its length must be at least 4. */
  result: GraphDataView<'uint32'>;
  /** Compile-time batch filter in `[0, 2^31 - 1]`. When omitted, every batch is accepted. */
  batchIndex?: number;
  /** Packed mask whose length is the source row count. Cleared, then set to 1 for picked rows. */
  outputMask: GraphDataView<'uint32'>;
  /** One-row flag that copies the region result's overflow word as 0 or 1. Always written. */
  overflow: GraphDataView<'uint32'>;
};

/**
 * Converts a capacity-bounded index-picking region result into a deduplicated source-aligned 0/1
 * mask. The application adds `GPUIndexPickingTarget.addRegionPass` before this contributor; the graph
 * orders the two through the hazard on `result`.
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
    validatePackedUint32View(props.result, `${id} result`);
    if (props.result.length < 4) {
      throw new Error(`${id} result must hold at least one picked pair`);
    }
    if (
      props.batchIndex !== undefined &&
      (!Number.isSafeInteger(props.batchIndex) ||
        props.batchIndex < 0 ||
        props.batchIndex > 0x7fffffff)
    ) {
      throw new Error(`${id} batchIndex must be an integer in [0, 2^31 - 1]`);
    }
    validatePackedUint32View(props.outputMask, `${id} outputMask`);
    validatePackedUint32View(props.overflow, `${id} overflow`);
    if (props.outputMask.length < 1 || props.overflow.length < 1) {
      throw new Error(`${id} outputMask and overflow must contain at least one row`);
    }
    if (
      props.result.buffer === props.outputMask.buffer ||
      props.result.buffer === props.overflow.buffer
    ) {
      throw new Error(`${id} outputs must not share buffers with inputs`);
    }
  }

  /** Returns clear, scatter, and overflow nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {result, outputMask, overflow, batchIndex} = props;
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
