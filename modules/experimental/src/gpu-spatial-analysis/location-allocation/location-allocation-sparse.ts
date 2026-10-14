// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createLoadBalancedForEachNode} from '../../utils/load-balanced-for-each';
import {createSortedSegmentOffsetsNode} from '../../utils/sorted-segment-offsets';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

/**
 * Sparse demand-by-facility costs in compressed sparse row (CSR) form for
 * {@link GPULocationAllocation}. A demand-facility pair that is not listed is unreachable
 * (`+Infinity`), exactly as if the dense matrix held `+Infinity` there.
 */
export type GPULocationAllocationSparseCosts = {
  /**
   * `demandCount + 1` non-decreasing row offsets into `facilities` and `costs`, starting at 0 and
   * ending at the entry count.
   */
  offsets: GraphDataView<'uint32'>;
  /**
   * Candidate facility column of every entry. Within one demand row the columns must be strictly
   * ascending (sorted, no duplicates) and below `facilityCount`.
   */
  facilities: GraphDataView<'uint32'>;
  /** Cost of every entry, as long as `facilities`. `+Infinity` is allowed; NaN and negatives are invalid. */
  costs: GraphDataView<'float32'>;
};

/** Views of the facility-major (CSC) transpose of {@link GPULocationAllocationSparseCosts}. */
export type SparseTranspose = {
  /** `facilityCount + 1` offsets of each facility's entries. */
  columnOffsets: GraphDataView<'uint32'>;
  /** Per entry two words: the demand row and the float32 cost bits; ascending demand within a facility. */
  columnEntries: GraphDataView<'uint32'>;
};

/** Bindings of the CSR cost views, in the names the sparse WGSL helpers expect. */
export function getSparseRowBindings(
  sparse: GPULocationAllocationSparseCosts
): WGSLKernelBinding[] {
  return [
    {name: 'rowOffsets', view: sparse.offsets, type: 'u32', access: 'read'},
    {name: 'rowFacilities', view: sparse.facilities, type: 'u32', access: 'read'},
    {name: 'rowCosts', view: sparse.costs, type: 'f32', access: 'read'}
  ];
}

/** Bindings of the transpose views, in the names the sparse WGSL helpers expect. */
export function getSparseColumnBindings(transpose: SparseTranspose): WGSLKernelBinding[] {
  return [
    {name: 'columnOffsets', view: transpose.columnOffsets, type: 'u32', access: 'read'},
    {name: 'columnEntries', view: transpose.columnEntries, type: 'u32', access: 'read'}
  ];
}

/**
 * WGSL helper `getSparseRowCost(row, facility) -> f32` over the `rowOffsets`, `rowFacilities` and
 * `rowCosts` bindings: a binary search of the strictly ascending row, `+Infinity` when the pair is
 * not listed.
 */
export const SPARSE_ROW_COST_WGSL = /* wgsl */ `
fn getSparseRowCost(row: u32, facility: u32) -> f32 {
  var low = rowOffsets[rowOffsetsOffset + row];
  let end = rowOffsets[rowOffsetsOffset + row + 1u];
  var high = end;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (rowFacilities[rowFacilitiesOffset + middle] < facility) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && rowFacilities[rowFacilitiesOffset + low] == facility) {
    return rowCosts[rowCostsOffset + low];
  }
  // A runtime value: a constant-expression bitcast to infinity does not compile.
  return bitcast<f32>(0x7f800000u | (facility & 0u));
}`;

/**
 * Validation nodes of sparse input: row offsets that start at 0, never decrease and end at the
 * entry count; every facility below `facilityCount` and strictly ascending within its row (one
 * load-balanced pass, so a hub row does not serialize one thread); every cost non-NaN and
 * non-negative. Each violation adds one to `status` word 3.
 */
export function createSparseValidationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    sparse: GPULocationAllocationSparseCosts;
    declarations: string;
    status: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {sparse, status} = props;
  const entryCount = sparse.facilities.length;
  const rowCount = sparse.offsets.length - 1;
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-validate-offsets`,
      operation: props.operation,
      variant: 'sparse-validate-offsets',
      invocationCount: Math.max(entryCount, rowCount + 1),
      bindings: [
        {name: 'rowOffsets', view: sparse.offsets, type: 'u32', access: 'read'},
        {name: 'rowCosts', view: sparse.costs, type: 'f32', access: 'read'},
        {name: 'status', view: status, type: 'atomic<u32>', access: 'read_write'}
      ],
      declarations: props.declarations,
      body: `if (index <= DEMAND_COUNT) {
    let offset = rowOffsets[rowOffsetsOffset + index];
    var bad = false;
    if (index == 0u) { bad = bad || offset != 0u; }
    if (index == DEMAND_COUNT) {
      bad = bad || offset != SPARSE_ENTRY_COUNT;
    } else {
      bad = bad || offset > rowOffsets[rowOffsetsOffset + index + 1u];
    }
    if (bad) {
      atomicAdd(&status[statusOffset + 3u], 1u);
    }
  }
  if (index < SPARSE_ENTRY_COUNT && isInvalid(rowCosts[rowCostsOffset + index])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }`
    }),
    createLoadBalancedForEachNode<Parameters>(graph, {
      id: `${props.id}-validate-rows`,
      operation: props.operation,
      variant: 'sparse-validate-rows',
      offsets: sparse.offsets,
      itemCapacity: entryCount,
      bindings: [
        {name: 'rowFacilities', view: sparse.facilities, type: 'u32', access: 'read'},
        {name: 'status', view: status, type: 'atomic<u32>', access: 'read_write'}
      ],
      declarations: `${props.declarations}
fn visitLoadBalancedItem(row: u32, item: u32, lane: u32) {
  let facility = rowFacilities[rowFacilitiesOffset + item];
  var bad = facility >= FACILITY_COUNT;
  if (item > loadBalancedOffsets[loadBalancedOffsetsOffset + row] &&
      rowFacilities[rowFacilitiesOffset + item - 1u] >= facility) {
    bad = true;
  }
  if (bad) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }
}`
    })
  ];
}

/**
 * Builds the facility-major transpose of sparse costs on the GPU: a load-balanced pass writes each
 * entry's demand row, a stable radix sort groups the entries by facility (so demand rows stay
 * ascending within a facility and every sum over a column has a fixed order), segment offsets come
 * from a binary search of the sorted keys, and one gather writes the demand row and cost bits of
 * the entries in sorted order. Entries with an invalid facility sort last and are excluded.
 */
export function createSparseTransposeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    sparse: GPULocationAllocationSparseCosts;
    facilityCount: number;
    declarations: string;
  }
): {nodes: GPUCommandNode<Parameters>[]; transpose: SparseTranspose} {
  const {id, sparse, facilityCount} = props;
  const entryCount = sparse.facilities.length;
  const view = (name: string, length: number) =>
    createTransientView(graph, `${id}-transpose-${name}`, 'uint32', length);
  const entryRows = view('entry-rows', entryCount);
  const entryIds = view('entry-ids', entryCount);
  const entryKeys = view('entry-keys', entryCount);
  const sortedKeys = view('sorted-keys', entryCount);
  const sortedEntryIds = view('sorted-entry-ids', entryCount);
  const columnOffsets = view('column-offsets', facilityCount + 1);
  const columnEntries = view('column-entries', 2 * entryCount);
  const nodes: GPUCommandNode<Parameters>[] = [
    createLoadBalancedForEachNode<Parameters>(graph, {
      id: `${id}-transpose-rows`,
      operation: props.operation,
      variant: 'sparse-transpose-rows',
      offsets: sparse.offsets,
      itemCapacity: entryCount,
      bindings: [
        {name: 'rowFacilities', view: sparse.facilities, type: 'u32', access: 'read'},
        {name: 'entryRows', view: entryRows, type: 'u32', access: 'read_write'},
        {name: 'entryIds', view: entryIds, type: 'u32', access: 'read_write'},
        {name: 'entryKeys', view: entryKeys, type: 'u32', access: 'read_write'}
      ],
      declarations: `${props.declarations}
fn visitLoadBalancedItem(row: u32, item: u32, lane: u32) {
  entryRows[entryRowsOffset + item] = row;
  entryIds[entryIdsOffset + item] = item;
  // Invalid facilities (reported by validation) get the excluded key, which sorts last.
  entryKeys[entryKeysOffset + item] = min(rowFacilities[rowFacilitiesOffset + item], FACILITY_COUNT);
}`
    }),
    ...new GPUSort({
      id: `${id}-transpose-sort`,
      keys: entryKeys,
      values: entryIds,
      outputKeys: sortedKeys,
      outputValues: sortedEntryIds,
      // Keys range over [0, facilityCount], the last being the excluded key.
      keyBits: Math.max(1, facilityCount.toString(2).length)
    }).getCommandNodes(graph),
    createSortedSegmentOffsetsNode<Parameters>(graph, {
      id: `${id}-transpose-offsets`,
      operation: props.operation,
      variant: 'sparse-transpose-offsets',
      segmentCount: facilityCount,
      sortedKeys,
      segmentOffsets: columnOffsets
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-transpose-gather`,
      operation: props.operation,
      variant: 'sparse-transpose-gather',
      invocationCount: entryCount,
      bindings: [
        {name: 'rowCosts', view: sparse.costs, type: 'f32', access: 'read'},
        {name: 'entryRows', view: entryRows, type: 'u32', access: 'read'},
        {name: 'sortedEntryIds', view: sortedEntryIds, type: 'u32', access: 'read'},
        {name: 'columnEntries', view: columnEntries, type: 'u32', access: 'read_write'}
      ],
      body: `let entry = sortedEntryIds[sortedEntryIdsOffset + index];
  columnEntries[columnEntriesOffset + 2u * index] = entryRows[entryRowsOffset + entry];
  columnEntries[columnEntriesOffset + 2u * index + 1u] = bitcast<u32>(rowCosts[rowCostsOffset + entry]);`
    })
  ];
  return {nodes, transpose: {columnOffsets, columnEntries}};
}
