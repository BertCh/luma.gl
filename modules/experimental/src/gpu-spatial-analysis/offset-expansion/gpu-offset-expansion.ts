// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCardinalityPort, GPUOverflowStatusPort} from '../../analysis-core/contracts/index';

const OPERATION = 'GPUOffsetExpansion';

/** Value written to `owners` rows that have no owner: unused capacity or rows before `offsets[0]`. */
export const GPU_OFFSET_EXPANSION_NO_OWNER = 0xffffffff;

/** Caller-owned outputs of {@link GPUOffsetExpansion}. */
export type GPUOffsetExpansionOutput = {
  /**
   * Owner (parent) index per child row. Its length is the child row capacity. Rows at or beyond
   * `status.count` hold {@link GPU_OFFSET_EXPANSION_NO_OWNER}.
   */
  owners: GraphDataView<'uint32'>;
  /** Optional index of each child row inside its owner; same length as `owners`. */
  localIndex?: GraphDataView<'uint32'>;
  /**
   * One-row status scalars: `count` receives `min(offsets[last], owners.length)`, `overflow`
   * receives `1` when `offsets[last] > owners.length` (otherwise `0`), and the optional
   * `requiredCount` receives the unclamped child row count `offsets[last]`.
   */
  status: GPUCardinalityPort & GPUOverflowStatusPort;
};

/**
 * Properties for {@link GPUOffsetExpansion}.
 *
 * Per-frame (no recompile): the contents of `offsets` and `ownerMap`. Compile-time: view lengths
 * and which optional views are present.
 */
export type GPUOffsetExpansionProps = {
  /** Prefix for generated node IDs. Defaults to `'offset-expansion'`. */
  id?: string;
  /**
   * `ownerCount + 1` monotonic non-decreasing offsets: owner `p` owns child rows
   * `[offsets[p], offsets[p + 1])`, the GeoArrow offsets convention (`lineOffsets`,
   * `featureOffsets`, `polygonOffsets`, `ringOffsets`). Monotonicity is not validated on the GPU.
   */
  offsets: GraphDataView<'uint32'>;
  /**
   * Optional per-owner remap. When set, `owners[i]` becomes `ownerMap[owner]`, which chains
   * levels: expand vertices to rings with `ringOffsets` and pass the ring-to-polygon owner rows
   * (the `owners` output of a second instance over `polygonOffsets`) as the map, giving
   * the polygon of every vertex. Length at least the owner count.
   */
  ownerMap?: GraphDataView<'uint32'>;
  /** Caller-owned bounded outputs. */
  output: GPUOffsetExpansionOutput;
};

/**
 * Expands an offsets view into one owner index (and local index) per child row: the GPU
 * equivalent of Shapely `get_coordinates(..., return_index=True)`, `get_parts(...,
 * return_index=True)` and GeoPandas `explode(index_parts=True)` index maps.
 *
 * Child row `i` belongs to the last owner `p` with `offsets[p] <= i`, found by binary search per
 * row, so owners with an empty range produce no rows (Shapely `get_coordinates` drops empty
 * geometries; `get_parts` and `explode` keep an empty polygon as one part, which is what
 * `get_num_geometries` offsets already say). `localIndex[i] = i - offsets[p]`. Deterministic, no
 * atomics. Output is bounded by `output.owners.length`; surplus rows are counted in `status.requiredCount`
 * and flagged by `status.overflow`.
 *
 * Chain levels by expanding each offsets view and feeding the owner rows of the coarser level as
 * `ownerMap`, or gather per-owner attribute columns with `GPUGather` using `owners` as indices
 * (explode attribute replication).
 */
export class GPUOffsetExpansion implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUOffsetExpansionProps;

  constructor(props: GPUOffsetExpansionProps) {
    this.id = props.id ?? 'offset-expansion';
    this.props = props;
    const {id} = this;
    const {output} = props;
    validatePackedView(props.offsets, ['uint32'], `${id} offsets`);
    if (props.offsets.length < 2) {
      throw new Error(`${id} offsets needs two rows`);
    }
    if (props.ownerMap) {
      validatePackedView(props.ownerMap, ['uint32'], `${id} ownerMap`);
      if (props.ownerMap.length < props.offsets.length - 1) {
        throw new Error(`${id} ownerMap must hold one row per owner`);
      }
    }
    validatePackedView(output.owners, ['uint32'], `${id} output.owners`);
    if (output.owners.length < 1) {
      throw new Error(`${id} output.owners must be non-empty`);
    }
    if (output.localIndex) {
      validatePackedView(output.localIndex, ['uint32'], `${id} output.localIndex`);
      if (output.localIndex.length !== output.owners.length) {
        throw new Error(`${id} output.localIndex length must equal output.owners length`);
      }
    }
    for (const [name, scalar] of [
      ['count', output.status.count],
      ['overflow', output.status.overflow],
      ['requiredCount', output.status.requiredCount]
    ] as const) {
      if (scalar) {
        validatePackedView(scalar, ['uint32'], `${id} output.status.${name}`);
        if (scalar.length < 1) {
          throw new Error(`${id} output.status.${name} must contain one uint32 row`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.owners,
        output.localIndex,
        output.status.count,
        output.status.overflow,
        output.status.requiredCount
      ],
      [props.offsets, props.ownerMap]
    );
  }

  /** Returns the single expansion kernel node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.offsets,
      props.ownerMap,
      output.owners,
      output.localIndex,
      output.status.count,
      output.status.overflow,
      output.status.requiredCount
    ]);
    const ownerCount = props.offsets.length - 1;
    const bindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'ownersOut', view: output.owners, type: 'u32', access: 'read_write'},
      {name: 'countOut', view: output.status.count, type: 'u32', access: 'read_write'},
      {name: 'overflowOut', view: output.status.overflow, type: 'u32', access: 'read_write'}
    ];
    if (props.ownerMap) {
      bindings.push({name: 'ownerMap', view: props.ownerMap, type: 'u32', access: 'read'});
    }
    if (output.localIndex) {
      bindings.push({
        name: 'localOut',
        view: output.localIndex,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (output.status.requiredCount) {
      bindings.push({
        name: 'totalOut',
        view: output.status.requiredCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-expand`,
        operation: OPERATION,
        bindings,
        invocationCount: output.owners.length,
        declarations: `const OWNER_COUNT: u32 = ${ownerCount}u;
const CAPACITY: u32 = ${output.owners.length}u;
const NO_OWNER: u32 = ${GPU_OFFSET_EXPANSION_NO_OWNER}u;`,
        body: `let total = offsets[offsetsOffset + OWNER_COUNT];
  if (index == 0u) {
    countOut[countOutOffset] = min(total, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
    ${output.status.requiredCount ? 'totalOut[totalOutOffset] = total;' : ''}
  }
  // Count the owners whose first row is at or before this row (offsets are non-decreasing).
  var low = 0u;
  var high = OWNER_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  var owner = NO_OWNER;
  var local = NO_OWNER;
  if (index < total && low > 0u) {
    let ownerRow = low - 1u;
    owner = ${props.ownerMap ? 'ownerMap[ownerMapOffset + ownerRow]' : 'ownerRow'};
    local = index - offsets[offsetsOffset + ownerRow];
  }
  ownersOut[ownersOutOffset + index] = owner;
  ${output.localIndex ? 'localOut[localOutOffset + index] = local;' : ''}`
      })
    ];
  }
}
