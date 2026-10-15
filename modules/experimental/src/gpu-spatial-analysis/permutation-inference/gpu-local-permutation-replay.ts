// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getLocalPermutationKernelWGSL,
  type GPULocalPermutationTest
} from './gpu-local-permutation-test';

/** Recording-only inputs for one row's GPU replay. @internal */
export type GPULocalPermutationReplayProps = {
  /** Node prefix; defaults to `${test.id}-replay`. */
  id?: string;
  /** Add this test to the same recording graph before adding the replay. */
  test: GPULocalPermutationTest;
  /** One packed uint32 row index, rewritable each frame. */
  row: GraphDataView<'uint32'>;
  /**
   * Key space of the test's rows in the caller's record (for example `tract`); the replay nodes'
   * datum is the pinned row in it. Defaults to `row`, which the caller's record must then declare.
   */
  keySpace?: string;
  /** Caller-owned sinks. */
  output: {
    /** At least maximumPermutations floats; only the reported prefix is valid. */
    nullDistribution: GraphDataView<'float32'>;
    /**
     * At least three words: [raw observed f32, replay count f32, pinned row u32 bits].
     * Count is zero for an invalid/unavailable row; its distribution must not be displayed.
     * The third word is internal state, read as uint32 rather than float32.
     */
    replayInfo: GraphDataView<'float32'>;
  };
};

/**
 * Replays all P raw local terms for one pinned row with the test's identical Philox draws.
 *
 * Borrows the test's prepared views instead of rebuilding its compaction, moments or observed
 * statistic. Add the test first, then this producer, on a graph with `recording` enabled.
 * For post-run inspection, the scenario lists `test.replayInputs.bindings[*].view.buffer.id`
 * and `test.replayInputs.observedRaw.buffer.id` in `recording.retain`. Ordinary graph dependencies
 * already preserve these inputs until replay finishes; retain additionally permits later captures.
 * Sequential tests still replay all P draws: compare only the first permutationCounts[row] terms
 * when explaining their stopping decision. Folded tests need only one replay sweep.
 *
 * Captures must use provenance `gpu-replay` and meaning
 * `null distribution (raw local term) of row <r>`; views draw this provenance hatched.
 * This instrument is deliberately absent from every public barrel. @internal
 */
export class GPULocalPermutationReplay implements GPUCommandNodeProducer {
  /** Prefix for setup and draw nodes. */
  readonly id: string;
  /** Validated replay inputs and sinks. */
  readonly props: GPULocalPermutationReplayProps;

  constructor(props: GPULocalPermutationReplayProps) {
    this.id = props.id ?? `${props.test.id}-replay`;
    this.props = props;
    validatePackedUint32View(props.row, `${this.id} row`);
    validateCapacity(this.id, 'row length', props.row.length, 1, 1);
    validatePackedView(props.output.nullDistribution, ['float32'], `${this.id} nullDistribution`);
    validateCapacity(
      this.id,
      'nullDistribution length',
      props.output.nullDistribution.length,
      props.test.props.maximumPermutations
    );
    validatePackedView(props.output.replayInfo, ['float32'], `${this.id} replayInfo`);
    validateCapacity(this.id, 'replayInfo length', props.output.replayInfo.length, 3);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [props.row]);
  }

  /** Returns setup and a single-invocation replay, after the test's preparation nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const inputs = props.test.replayInputs;
    // Replay is opt-in instrumentation and requires the test to have been lowered first.
    assert(graph.recording && inputs);
    const reads = [...inputs.bindings.map(binding => binding.view), inputs.observedRaw, props.row];
    validateGraphViewsBelongToGraph(id, graph, [...reads, ...Object.values(props.output)]);
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), reads);
    const {declarations, body} = getLocalPermutationKernelWGSL(inputs.kernel, 'replay');
    const replayInfo = {
      name: 'replayInfo',
      view: props.output.replayInfo,
      type: 'u32' as const,
      access: 'read_write' as const
    };
    const record = {
      namespace: 'replay',
      // The replay redraws the null distribution of the raw local term; it computes no formula output.
      datum: {keySpace: props.keySpace ?? 'row'},
      attrs: {provenance: 'gpu-replay', rowResource: props.row.buffer.id}
    };
    return [
      // Packing the pin and observed term leaves the draw kernel at eight storage bindings.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-setup`,
        operation: 'GPULocalPermutationReplay',
        variant: 'setup',
        record,
        bindings: [
          {name: 'row', view: props.row, type: 'u32', access: 'read'},
          {name: 'observedRaw', view: inputs.observedRaw, type: 'f32', access: 'read'},
          replayInfo
        ],
        invocationCount: 1,
        body: `let pinnedRow = row[rowOffset];
  replayInfo[replayInfoOffset] = 0x7fc00000u;
  replayInfo[replayInfoOffset + 1u] = 0u;
  replayInfo[replayInfoOffset + 2u] = pinnedRow;
  if (pinnedRow < ${props.test.props.values.length}u) {
    replayInfo[replayInfoOffset] = bitcast<u32>(observedRaw[observedRawOffset + pinnedRow]);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-permute`,
        operation: 'GPULocalPermutationReplay',
        variant: 'replay',
        record,
        bindings: [
          ...inputs.bindings,
          {
            name: 'nullDistribution',
            view: props.output.nullDistribution,
            type: 'f32',
            access: 'read_write'
          },
          replayInfo
        ],
        invocationCount: 1,
        // The shared loop's index is the pinned row, never the replay dispatch's invocation zero.
        declarations: `${declarations}
fn replayRow(index: u32) {
  ${body}
}`,
        body: 'replayRow(replayInfo[replayInfoOffset + 2u]);'
      })
    ];
  }
}
