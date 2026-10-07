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

/** Largest group count the chunked reduction supports: one lane per group in a 256-lane workgroup. */
export const GROUPED_SUM_MAXIMUM_GROUPS = 256;

/** Workgroup size of both passes (also the tile width). */
const WORKGROUP_SIZE = 256;
/** Largest number of lane slices that split one group's work inside a tile. */
const MAXIMUM_SLICES = 64;
/** Largest number of row chunks (workgroups) the partial pass launches. */
const MAXIMUM_CHUNKS = 1024;
/** Most quantities one reduction sums, bounded by workgroup memory. */
const MAXIMUM_QUANTITIES = 4;

/** Properties of {@link getGroupedSumNodes}. @internal */
export type GroupedSumProps = {
  /** Prefix of the generated node and transient IDs. */
  id: string;
  /** Operation name reported in workload estimates. */
  operation: string;
  /** Number of rows. */
  rows: number;
  /** Number of groups, at most {@link GROUPED_SUM_MAXIMUM_GROUPS}. */
  groupCount: number;
  /** Read-only bindings used by `rowSnippet`. */
  bindings: WGSLKernelBinding[];
  /** Module-scope WGSL (helpers, constants) used by `rowSnippet`. */
  declarations?: string;
  /**
   * WGSL statements run once per row with `row: u32` in scope. They must assign
   * `group` (a `u32`; any value at or above `GROUP_COUNT` excludes the row) and
   * `value0 ... value{n-1}` (`f32`, the quantities to sum). The variables are pre-declared.
   * `GROUP_COUNT` and `NO_GROUP` are available.
   */
  rowSnippet: string;
  /** One per-group f32 output per summed quantity, in `value0, value1, ...` order. */
  outputs: readonly GraphDataView<'float32'>[];
  /** Optional per-group count of rows that were assigned to the group. */
  countOutput?: GraphDataView<'uint32'>;
  /**
   * Optional step that runs on lane 0 of each group's fold workgroup after the sums are final, so
   * a small per-group update fuses with the reduction. `finishBody` sees `group: u32`,
   * `sum0 ... sum{n-1}: f32` and `count: u32`.
   */
  finish?: {bindings: WGSLKernelBinding[]; declarations?: string; body: string};
  /** Suffix of the fold node ID (`<id>-<suffix>`). Defaults to `'fold'`. */
  foldSuffix?: string;
};

/**
 * Per-group sums of up to four per-row quantities without sorting or atomics, for at most 256
 * groups. Replaces "sort by group, gather, one workgroup per group" for the common case of few
 * groups, where the sorted segmented sum leaves almost the whole GPU idle (a single group is one
 * workgroup walking every row).
 *
 * Rows are cut into contiguous chunks, one workgroup each. A workgroup streams its chunk through
 * shared-memory tiles; `groupCount * slices` lanes each own one (group, slice) pair and add the
 * tile entries of their group in a fixed order, then the slices fold in a fixed order into the
 * chunk's per-group partials. A second kernel (one workgroup per group) folds the chunk partials
 * with a strided partial sum and a fixed binary tree. Every order depends only on the sizes, so
 * the sums are bitwise reproducible. Work is `O(rows * groupCount)` shared-memory reads, depth
 * `O(rows / chunks)`, and the rows are read once.
 *
 * @internal
 */
export function getGroupedSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GroupedSumProps
): GPUCommandNode<Parameters>[] {
  const {id, operation, rows, groupCount, outputs} = props;
  const quantityCount = outputs.length;
  if (groupCount < 1 || groupCount > GROUPED_SUM_MAXIMUM_GROUPS) {
    throw new Error(`${id} supports 1 to ${GROUPED_SUM_MAXIMUM_GROUPS} groups`);
  }
  if (quantityCount < 1 || quantityCount > MAXIMUM_QUANTITIES) {
    throw new Error(`${id} sums 1 to ${MAXIMUM_QUANTITIES} quantities`);
  }
  const chunkCount = Math.min(MAXIMUM_CHUNKS, Math.max(1, Math.ceil(rows / WORKGROUP_SIZE)));
  const chunkSize = Math.ceil(rows / chunkCount / WORKGROUP_SIZE) * WORKGROUP_SIZE;
  const slices = Math.min(MAXIMUM_SLICES, Math.floor(WORKGROUP_SIZE / groupCount));
  const lanes = groupCount * slices;
  const quantities = Array.from({length: quantityCount}, (_, quantity) => quantity);
  const partialSums = createTransientView(
    graph,
    `${id}-partial-sums`,
    'float32',
    quantityCount * groupCount * chunkCount
  );
  const partialCounts = props.countOutput
    ? createTransientView(graph, `${id}-partial-counts`, 'uint32', groupCount * chunkCount)
    : undefined;
  const hasCounts = Boolean(partialCounts);

  const partialNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-partial`,
    operation,
    variant: 'grouped-partial',
    bindings: [
      ...props.bindings,
      {name: 'partialSums', view: partialSums, type: 'f32', access: 'read_write'},
      ...(partialCounts
        ? [{name: 'partialCounts', view: partialCounts, type: 'u32', access: 'read_write'} as const]
        : [])
    ],
    workgroupSize: WORKGROUP_SIZE,
    invocationCount: chunkCount * WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `${props.declarations ?? ''}
const GROUP_COUNT: u32 = ${groupCount}u;
const NO_GROUP: u32 = 0xffffffffu;
const ROWS: u32 = ${rows}u;
const CHUNK_SIZE: u32 = ${chunkSize}u;
const CHUNK_COUNT: u32 = ${chunkCount}u;
const TILE_COUNT: u32 = ${chunkSize / WORKGROUP_SIZE}u;
const SLICES: u32 = ${slices}u;
const LANES: u32 = ${lanes}u;
var<workgroup> tileGroups: array<u32, ${WORKGROUP_SIZE}>;
var<workgroup> tileValues: array<f32, ${WORKGROUP_SIZE * quantityCount}>;
var<workgroup> laneSums: array<f32, ${WORKGROUP_SIZE * quantityCount}>;
var<workgroup> laneCounts: array<u32, ${WORKGROUP_SIZE}>;`,
    // Barriers sit in uniform control flow: the tile loop has a constant trip count and
    // out-of-range rows only change the data (they get NO_GROUP, which matches no lane).
    body: `let chunk = index / ${WORKGROUP_SIZE}u;
  let begin = chunk * CHUNK_SIZE;
  let end = min(begin + CHUNK_SIZE, ROWS);
  let lane = localInvocationIndex;
  let laneGroup = lane % GROUP_COUNT;
  let laneSlice = lane / GROUP_COUNT;
  ${quantities.map(quantity => `var sum${quantity} = 0.0;`).join('\n  ')}
  var count = 0u;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let row = begin + tile * ${WORKGROUP_SIZE}u + lane;
    var group = NO_GROUP;
    ${quantities.map(quantity => `var value${quantity} = 0.0;`).join('\n    ')}
    if (row < end) {
      ${props.rowSnippet}
    }
    if (group >= GROUP_COUNT) {
      group = NO_GROUP;
    }
    tileGroups[lane] = group;
    ${quantities.map(quantity => `tileValues[${quantity * WORKGROUP_SIZE}u + lane] = value${quantity};`).join('\n    ')}
    workgroupBarrier();
    if (lane < LANES) {
      // Lane (slice, group) owns tile entries slice, slice + SLICES, ...: a fixed order.
      for (var entry = laneSlice; entry < ${WORKGROUP_SIZE}u; entry += SLICES) {
        if (tileGroups[entry] == laneGroup) {
          count++;
          ${quantities.map(quantity => `sum${quantity} += tileValues[${quantity * WORKGROUP_SIZE}u + entry];`).join('\n          ')}
        }
      }
    }
    workgroupBarrier();
  }
  ${quantities.map(quantity => `laneSums[${quantity * WORKGROUP_SIZE}u + lane] = sum${quantity};`).join('\n  ')}
  laneCounts[lane] = count;
  workgroupBarrier();
  if (lane < GROUP_COUNT) {
    var chunkCountTotal = 0u;
    ${quantities.map(quantity => `var chunkSum${quantity} = 0.0;`).join('\n    ')}
    for (var slice = 0u; slice < SLICES; slice++) {
      chunkCountTotal += laneCounts[slice * GROUP_COUNT + lane];
      ${quantities.map(quantity => `chunkSum${quantity} += laneSums[${quantity * WORKGROUP_SIZE}u + slice * GROUP_COUNT + lane];`).join('\n      ')}
    }
    ${quantities.map(quantity => `partialSums[partialSumsOffset + (${quantity}u * GROUP_COUNT + lane) * CHUNK_COUNT + chunk] = chunkSum${quantity};`).join('\n    ')}
    ${hasCounts ? 'partialCounts[partialCountsOffset + lane * CHUNK_COUNT + chunk] = chunkCountTotal;' : ''}
  }`
  });

  const finish = props.finish;
  const foldNode = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-${props.foldSuffix ?? 'fold'}`,
    operation,
    variant: 'grouped-fold',
    bindings: [
      {name: 'partialSums', view: partialSums, type: 'f32', access: 'read'},
      ...(partialCounts
        ? [{name: 'partialCounts', view: partialCounts, type: 'u32', access: 'read'} as const]
        : []),
      ...outputs.map(
        (view, quantity) =>
          ({name: `output${quantity}`, view, type: 'f32', access: 'read_write'}) as const
      ),
      ...(props.countOutput
        ? [
            {
              name: 'countOutput',
              view: props.countOutput,
              type: 'u32',
              access: 'read_write'
            } as const
          ]
        : []),
      ...(finish?.bindings ?? [])
    ],
    workgroupSize: WORKGROUP_SIZE,
    invocationCount: groupCount * WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `${finish?.declarations ?? ''}
const GROUP_COUNT: u32 = ${groupCount}u;
const CHUNK_COUNT: u32 = ${chunkCount}u;
var<workgroup> reducedSums: array<f32, ${WORKGROUP_SIZE * quantityCount}>;
var<workgroup> reducedCounts: array<u32, ${WORKGROUP_SIZE}>;`,
    // One workgroup per group: a strided per-lane sum over the chunk partials, then a fixed tree.
    body: `let group = index / ${WORKGROUP_SIZE}u;
  let lane = localInvocationIndex;
  ${quantities.map(quantity => `var sum${quantity} = 0.0;`).join('\n  ')}
  var count = 0u;
  for (var chunk = lane; chunk < CHUNK_COUNT; chunk += ${WORKGROUP_SIZE}u) {
    ${quantities.map(quantity => `sum${quantity} += partialSums[partialSumsOffset + (${quantity}u * GROUP_COUNT + group) * CHUNK_COUNT + chunk];`).join('\n    ')}
    ${hasCounts ? 'count += partialCounts[partialCountsOffset + group * CHUNK_COUNT + chunk];' : ''}
  }
  ${quantities.map(quantity => `reducedSums[${quantity * WORKGROUP_SIZE}u + lane] = sum${quantity};`).join('\n  ')}
  reducedCounts[lane] = count;
  workgroupBarrier();
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      ${quantities.map(quantity => `reducedSums[${quantity * WORKGROUP_SIZE}u + lane] += reducedSums[${quantity * WORKGROUP_SIZE}u + lane + stride];`).join('\n      ')}
      reducedCounts[lane] += reducedCounts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    ${quantities.map(quantity => `sum${quantity} = reducedSums[${quantity * WORKGROUP_SIZE}u];`).join('\n    ')}
    count = reducedCounts[0];
    ${quantities.map(quantity => `output${quantity}[output${quantity}Offset + group] = sum${quantity};`).join('\n    ')}
    ${props.countOutput ? 'countOutput[countOutputOffset + group] = count;' : ''}
    ${finish?.body ?? ''}
  }`
  });
  return [partialNode, foldNode];
}
