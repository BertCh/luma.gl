// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {validateCapacity, validatePackedView} from '../../utils/gpu-contributor-utils';
import {createWGSLDispatchGate, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';

/**
 * Permutations (slots) per batch of the Besag-Clifford stopping path, before the batch count cap.
 * Slot 0 is the observed statistic and rides in batch 0. @internal
 */
export const SPACE_TIME_STOPPING_BATCH_SIZE = 64;

/** Largest number of batches of the stopping path; the batch size doubles to respect it. @internal */
export const SPACE_TIME_MAXIMUM_STOPPING_BATCHES = 256;

/** Workgroup size of the per-batch reduce kernel. @internal */
const REDUCE_WORKGROUP_SIZE = 64;

/** Words of the GPU stopping state. @internal */
export const SPACE_TIME_STATE_LENGTH = 8;

/**
 * Word indices of the GPU stopping state (a transient `uint32` view). @internal
 */
export const SPACE_TIME_STATE = {
  /** Index of the batch the next batch kernels process (rule 12: never in WGSL text). */
  cursor: 0,
  /** Exceedances counted in permutation-index order over the batches scanned so far. */
  exceedances: 1,
  /** The stopping index `L` (1-based permutation), 0 while the rule has not fired. */
  stopIndex: 2,
  /** Mantel only: float32 bits of the mean spatial distance (the centring shift). */
  shiftDistance: 4,
  /** Mantel only: float32 bits of the mean time difference (the centring shift). */
  shiftTime: 5
} as const;

/**
 * Returns the number of slots per batch for `slotCount = maximumPermutations + 1` slots: 64,
 * doubled until the batch count is at most {@link SPACE_TIME_MAXIMUM_STOPPING_BATCHES}. The
 * stopping index never depends on it (the flags are scanned in permutation order).
 *
 * @internal
 */
export function getSpaceTimeStoppingBatchSize(slotCount: number): number {
  let size = SPACE_TIME_STOPPING_BATCH_SIZE;
  while (Math.ceil(slotCount / size) > SPACE_TIME_MAXIMUM_STOPPING_BATCHES) {
    size *= 2;
  }
  return size;
}

/**
 * Validates the Besag-Clifford props shared by `GPUKnoxTest` and `GPUMantelTest`:
 * `stopAfterExceedances` is an integer in `[1, maximumPermutations]` and `output.permutationCount`
 * is one packed `uint32` word, required exactly when stopping is enabled.
 *
 * @internal
 */
export function validateSpaceTimeStopping(
  id: string,
  stopAfterExceedances: number | undefined,
  maximumPermutations: number,
  permutationCount: GraphDataView<'uint32'> | undefined
): void {
  if (stopAfterExceedances === undefined) {
    if (permutationCount) {
      throw new Error(`${id} output.permutationCount needs stopAfterExceedances`);
    }
    return;
  }
  validateCapacity(id, 'stopAfterExceedances', stopAfterExceedances, 1, maximumPermutations);
  if (!permutationCount) {
    throw new Error(`${id} stopAfterExceedances needs output.permutationCount`);
  }
  validatePackedView(permutationCount, ['uint32'], `${id} output.permutationCount`);
  if (permutationCount.length < 1) {
    throw new Error(`${id} output.permutationCount needs 1 row`);
  }
}

/** Node ID and gate of one batch kernel, passed to the callbacks of {@link createSpaceTimeStoppingNodes}. @internal */
export type SpaceTimeBatchNodeProps<Parameters> = {
  id: string;
  condition: GPUCommandGraphNodeCondition<Parameters>;
  extraResources: GraphBufferUse[];
};

/** @internal */
export type SpaceTimeStoppingProps<Parameters> = {
  id: string;
  operation: string;
  stopAfterExceedances: number;
  /** `maximumPermutations + 1`: slot 0 is the observed statistic. */
  slotCount: number;
  groupsPerSlot: number;
  workgroupSize: number;
  statisticType: 'u32' | 'f32';
  parameters: GraphDataView<'uint32'>;
  statistics: GraphDataView<'uint32' | 'float32'>;
  /** Transient `uint32` view of {@link SPACE_TIME_STATE_LENGTH} words. */
  state: GraphDataView<'uint32'>;
  /** Output of `getSpaceTimeCommonWGSL`. */
  common: string;
  /** Creates the batch partials kernel, gated on slot 0 (`batchSize * groupsPerSlot` workgroups). */
  createPartialsNode: (batch: SpaceTimeBatchNodeProps<Parameters>) => GPUCommandNode<Parameters>;
  /** Creates the batch reduce kernel, gated on slot 1 (`batchSize` invocations of workgroup 64). */
  createReduceNode: (batch: SpaceTimeBatchNodeProps<Parameters>) => GPUCommandNode<Parameters>;
};

/**
 * Creates the Besag-Clifford batch loop shared by the Knox and Mantel tests:
 *
 * - `stop-reset` zeroes the cursor, exceedance count and stopping index and arms the dispatch gate.
 * - Per batch `c`: the gated partials and reduce kernels compute the statistics of slots
 *   `c * B .. c * B + B - 1` (slot 0, the observed statistic, is in batch 0), then the ungated
 *   one-thread `scan` walks the batch in permutation order, counts exceedances (`statistic >=
 *   observed`), records the first `L` with `e(L) = h`, advances the cursor and disarms the gate
 *   when the rule fired or the permutations are exhausted. The cursor lives in a GPU word, so the
 *   batch kernels are one pipeline however many batches there are (rule 12).
 * - `stop-finalize` zeroes the statistics past the used permutations.
 *
 * @internal
 */
export function createSpaceTimeStoppingNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: SpaceTimeStoppingProps<Parameters>
): GPUCommandNode<Parameters>[] {
  const {id, operation, state, statistics, parameters, common} = props;
  const batchSize = getSpaceTimeStoppingBatchSize(props.slotCount);
  const batchCount = Math.ceil(props.slotCount / batchSize);
  const gate = createWGSLDispatchGate<Parameters>(graph, {
    id: `${id}-stop-dispatch`,
    operation,
    shapes: [
      [batchSize * props.groupsPerSlot * props.workgroupSize, props.workgroupSize],
      [batchSize, REDUCE_WORKGROUP_SIZE]
    ]
  });
  const stateBinding = (access: 'read' | 'read_write') => ({
    name: 'state',
    view: state,
    type: 'u32' as const,
    access
  });
  const zero = props.statisticType === 'u32' ? '0u' : '0.0';
  const gateNode = (nodeId: string, slot: number): SpaceTimeBatchNodeProps<Parameters> => {
    const {condition, extraResources} = gate.getGate(slot);
    return {id: nodeId, condition: {...condition, id: `${nodeId}-gate`}, extraResources};
  };
  const words = SPACE_TIME_STATE;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-stop-reset`,
      operation,
      variant: 'stop-reset',
      bindings: [stateBinding('read_write'), gate.binding],
      invocationCount: 1,
      body: `state[stateOffset + ${words.cursor}u] = 0u;
  state[stateOffset + ${words.exceedances}u] = 0u;
  state[stateOffset + ${words.stopIndex}u] = 0u;
  ${gate.armWGSL}`
    })
  ];
  for (let batch = 0; batch < batchCount; batch++) {
    nodes.push(
      props.createPartialsNode(gateNode(`${id}-batch-${batch}-partials`, 0)),
      props.createReduceNode(gateNode(`${id}-batch-${batch}-reduce`, 1)),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-batch-${batch}-scan`,
        operation,
        variant: 'stop-scan',
        bindings: [
          {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
          {name: 'statistics', view: statistics, type: props.statisticType, access: 'read'},
          stateBinding('read_write'),
          gate.binding
        ],
        invocationCount: 1,
        declarations: `${common}
const BATCH_SIZE: u32 = ${batchSize}u;
const STOP_AFTER: u32 = ${props.stopAfterExceedances}u;`,
        // The only writer of the cursor and of the dispatch gate, so it is ungated. A scan after
        // the rule fired, or past the last permutation, does nothing. The walk is in permutation
        // order, so L does not depend on the batch size.
        body: `let permutations = readPermutationCount();
  let cursor = state[stateOffset + ${words.cursor}u];
  if (state[stateOffset + ${words.stopIndex}u] != 0u || cursor * BATCH_SIZE > permutations) {
    return;
  }
  let observed = statistics[statisticsOffset];
  var exceedances = state[stateOffset + ${words.exceedances}u];
  var stopIndex = 0u;
  for (var offset = 0u; offset < BATCH_SIZE; offset++) {
    let slot = cursor * BATCH_SIZE + offset;
    if (slot > permutations) {
      break;
    }
    if (slot > 0u && statistics[statisticsOffset + slot] >= observed) {
      exceedances++;
      if (exceedances == STOP_AFTER) {
        stopIndex = slot;
        break;
      }
    }
  }
  state[stateOffset + ${words.exceedances}u] = exceedances;
  state[stateOffset + ${words.stopIndex}u] = stopIndex;
  state[stateOffset + ${words.cursor}u] = cursor + 1u;
  if (stopIndex != 0u || (cursor + 1u) * BATCH_SIZE > permutations) {
    ${gate.disarmWGSL}
  }`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-stop-finalize`,
      operation,
      variant: 'stop-finalize',
      bindings: [
        {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
        stateBinding('read'),
        {name: 'statistics', view: statistics, type: props.statisticType, access: 'read_write'}
      ],
      invocationCount: statistics.length,
      declarations: common,
      // Rows past the used permutations are 0, as past `P` without stopping. Rows of the stopping
      // batch past L were computed but are not part of the first L permutations.
      body: `let stopIndex = state[stateOffset + ${words.stopIndex}u];
  let used = select(readPermutationCount(), stopIndex, (stopIndex != 0u));
  if (index > used) {
    statistics[statisticsOffset + index] = ${zero};
  }`
    })
  );
  return nodes;
}

/** Creates the transient stopping state view. @internal */
export function createSpaceTimeStateView<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string
): GraphDataView<'uint32'> {
  return createTransientView(graph, `${id}-stop-state`, 'uint32', SPACE_TIME_STATE_LENGTH);
}
