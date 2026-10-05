// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, createPublishNode} from '../../utils/wgsl-kernel-nodes';
import {createGatherNode, GROUP_NONE, readBinding, writeBinding} from './group-statistics-common';

/**
 * Sorted-group layout shared by every column reduction.
 *
 * Valid rows (mask set, key not all-ones) are stable-sorted by key; each group is a contiguous
 * range `[offsets[g], offsets[g + 1])` of the sorted order, for `g < min(groupTotal, capacity)`.
 *
 * @internal
 */
export type GroupStructure = {
  rowCount: number;
  capacity: number;
  twoWords: boolean;
  /** Source row of each sorted position (stable, so ties keep ascending row order). */
  permutation: GraphDataView<'uint32'>;
  /** Low key word per source row; invalid rows hold `0xffffffff`. */
  keyLow: GraphDataView<'uint32'>;
  /** High key word per source row (64-bit keys only). */
  keyHigh?: GraphDataView<'uint32'>;
  /** Row indices `0..rowCount-1`. */
  rowIds: GraphDataView<'uint32'>;
  /** `capacity + 1` sorted-order group start offsets (valid for `g <= groupCount`). */
  offsets: GraphDataView<'uint32'>;
  /** One-row unclamped group total. */
  groupTotal: GraphDataView<'uint32'>;
  /** Group index of each sorted position, or {@link GROUP_NONE}. */
  positionGroups: GraphDataView<'uint32'>;
  /** Group index of each source row, or {@link GROUP_NONE}. */
  rowGroups: GraphDataView<'uint32'>;
};

/**
 * Appends a stable LSD sort by `(high, low)` key words that starts from an existing row order.
 *
 * When `initialRows` is omitted the identity order is used. Because every pass is stable, rows
 * that share a key keep the order of `initialRows` (for value-sorted columns: ascending value,
 * then ascending row). Returns the final row permutation.
 *
 * @internal
 */
export function getKeyOrderNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    keyLow: GraphDataView<'uint32'>;
    keyHigh?: GraphDataView<'uint32'>;
    rowIds: GraphDataView<'uint32'>;
    initialRows?: GraphDataView<'uint32'>;
  }
): {
  nodes: GPUCommandNode<Parameters>[];
  permutation: GraphDataView<'uint32'>;
} {
  const {id, operation, keyLow, keyHigh, rowIds} = props;
  const rowCount = keyLow.length;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const u32 = (name: string) => createTransientView(graph, `${id}-${name}`, 'uint32', rowCount);

  let rows = props.initialRows ?? rowIds;
  let lowKeys = keyLow;
  if (props.initialRows) {
    lowKeys = u32('low-by-rows');
    nodes.push(
      createGatherNode(graph, {
        id: `${id}-gather-low`,
        operation,
        indices: props.initialRows,
        source: keyLow,
        target: lowKeys
      })
    );
  }
  const afterLow = u32('rows-by-low');
  nodes.push(
    ...new GPUSort({
      id: `${id}-sort-low`,
      keys: lowKeys,
      values: rows,
      outputKeys: u32('sorted-low-scratch'),
      outputValues: afterLow,
      keyBits: 32
    }).getCommandNodes(graph)
  );
  rows = afterLow;
  if (keyHigh) {
    const highKeys = u32('high-by-rows');
    nodes.push(
      createGatherNode(graph, {
        id: `${id}-gather-high`,
        operation,
        indices: rows,
        source: keyHigh,
        target: highKeys
      })
    );
    const afterHigh = u32('rows-by-high');
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort-high`,
        keys: highKeys,
        values: rows,
        outputKeys: u32('sorted-high-scratch'),
        outputValues: afterHigh,
        keyBits: 32
      }).getCommandNodes(graph)
    );
    rows = afterHigh;
  }
  return {nodes, permutation: rows};
}

/**
 * Builds the key words, the stable key sort, group boundaries and offsets, per-row group indices,
 * and writes `output.keys`, `output.counts` and the clamped count / overflow / total.
 *
 * @internal
 */
export function getGroupStructureNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    keys: GraphDataView<'uint32'> | GraphDataView<'uint32x2'>;
    mask?: GraphDataView<'uint32'>;
    output: {
      keys: GraphDataView<'uint32'> | GraphDataView<'uint32x2'>;
      counts: GraphDataView<'uint32'>;
      count: GraphDataView<'uint32'>;
      overflow: GraphDataView<'uint32'>;
      totalCount?: GraphDataView<'uint32'>;
    };
  }
): {nodes: GPUCommandNode<Parameters>[]; structure: GroupStructure} {
  const {id, operation, output} = props;
  const rowCount = props.keys.length;
  const capacity = output.counts.length;
  const twoWords = props.keys.format === 'uint32x2';
  const u32 = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const nodes: GPUCommandNode<Parameters>[] = [];

  // 1. Key words and row ids. Invalid rows get the all-ones key so they sort last.
  const keyLow = u32('key-low', rowCount);
  const keyHigh = twoWords ? u32('key-high', rowCount) : undefined;
  const rowIds = u32('row-ids', rowCount);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-prepare`,
      operation,
      variant: 'prepare',
      bindings: [
        readBinding('keys', props.keys),
        ...(props.mask ? [readBinding('mask', props.mask)] : []),
        writeBinding('keyLow', keyLow),
        ...(keyHigh ? [writeBinding('keyHigh', keyHigh)] : []),
        writeBinding('rowIds', rowIds)
      ],
      invocationCount: rowCount,
      body: `${
        twoWords
          ? `var low = keys[keysOffset + 2u * index];
  var high = keys[keysOffset + 2u * index + 1u];
  var isValid = !(low == 0xffffffffu && high == 0xffffffffu);`
          : `var low = keys[keysOffset + index];
  var isValid = low != 0xffffffffu;`
      }
  ${props.mask ? 'if (mask[maskOffset + index] == 0u) {\n    isValid = false;\n  }' : ''}
  if (!isValid) {
    low = 0xffffffffu;
    ${twoWords ? 'high = 0xffffffffu;' : ''}
  }
  keyLow[keyLowOffset + index] = low;
  ${twoWords ? 'keyHigh[keyHighOffset + index] = high;' : ''}
  rowIds[rowIdsOffset + index] = index;`
    })
  );

  // 2. Stable sort by key.
  const order = getKeyOrderNodes<Parameters>(graph, {
    id: `${id}-order`,
    operation,
    keyLow,
    keyHigh,
    rowIds
  });
  nodes.push(...order.nodes);
  const permutation = order.permutation;
  const sortedLow = u32('sorted-low', rowCount);
  const sortedHigh = twoWords ? u32('sorted-high', rowCount) : undefined;
  nodes.push(
    createGatherNode(graph, {
      id: `${id}-gather-sorted-low`,
      operation,
      indices: permutation,
      source: keyLow,
      target: sortedLow
    })
  );
  if (sortedHigh && keyHigh) {
    nodes.push(
      createGatherNode(graph, {
        id: `${id}-gather-sorted-high`,
        operation,
        indices: permutation,
        source: keyHigh,
        target: sortedHigh
      })
    );
  }
  const sortedBindings = () => [
    readBinding('sortedLow', sortedLow),
    ...(sortedHigh ? [readBinding('sortedHigh', sortedHigh)] : [])
  ];
  const keyAt = (row: string) =>
    twoWords
      ? `vec2u(sortedHigh[sortedHighOffset + ${row}], sortedLow[sortedLowOffset + ${row}])`
      : `vec2u(0u, sortedLow[sortedLowOffset + ${row}])`;
  const keyDeclarations = `const TWO_WORDS: bool = ${twoWords};
fn isInvalidKey(key: vec2u) -> bool {
  return key.y == 0xffffffffu && (!TWO_WORDS || key.x == 0xffffffffu);
}`;

  // 3. Group heads and the number of valid rows (a prefix of the sorted order).
  const heads = u32('heads', rowCount);
  const validCount = u32('valid-count', 1);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-heads`,
      operation,
      variant: 'heads',
      bindings: [
        ...sortedBindings(),
        writeBinding('heads', heads),
        writeBinding('validCount', validCount)
      ],
      invocationCount: rowCount,
      declarations: `const ROW_COUNT: u32 = ${rowCount}u;
${keyDeclarations}`,
      body: `let key = ${keyAt('index')};
  let isValid = !isInvalidKey(key);
  var isHead = isValid;
  if (isValid && index > 0u) {
    isHead = any(key != ${keyAt('index - 1u')});
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);
  let isLastValid = isValid && (index + 1u == ROW_COUNT || isInvalidKey(${keyAt('index + 1u')}));
  if (isLastValid) {
    validCount[validCountOffset] = index + 1u;
  } else if (index == 0u && !isValid) {
    validCount[validCountOffset] = 0u;
  }`
    })
  );

  // 4. Exclusive scan of heads gives each head its group index.
  const groupIndices = u32('group-indices', rowCount);
  nodes.push(
    ...new GPUScan({
      id: `${id}-scan`,
      input: heads,
      output: groupIndices,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );

  // 5. Offsets of the first `capacity + 1` groups, the unclamped total, and per-position groups.
  const offsets = u32('offsets', capacity + 1);
  const groupTotal = u32('group-total', 1);
  const positionGroups = u32('position-groups', rowCount);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-offsets`,
      operation,
      variant: 'offsets',
      bindings: [
        readBinding('heads', heads),
        readBinding('groupIndices', groupIndices),
        readBinding('validCount', validCount),
        writeBinding('offsets', offsets),
        writeBinding('groupTotal', groupTotal),
        writeBinding('positionGroups', positionGroups)
      ],
      invocationCount: rowCount,
      declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const CAPACITY: u32 = ${capacity}u;`,
      body: `let isHead = heads[headsOffset + index] != 0u;
  let group = groupIndices[groupIndicesOffset + index] + select(0u, 1u, isHead) - 1u;
  let isValid = index < validCount[validCountOffset];
  positionGroups[positionGroupsOffset + index] = select(0xffffffffu, group, isValid && group < CAPACITY);
  if (isHead && group <= CAPACITY) {
    offsets[offsetsOffset + group] = index;
  }
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    let total = groupIndices[groupIndicesOffset + last] + heads[headsOffset + last];
    groupTotal[groupTotalOffset] = total;
    if (total <= CAPACITY) {
      offsets[offsetsOffset + total] = validCount[validCountOffset];
    }
  }`
    })
  );

  // 6. Group index per source row (every row appears once in the permutation).
  const rowGroups = u32('row-groups', rowCount);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-row-groups`,
      operation,
      variant: 'row-groups',
      bindings: [
        readBinding('permutation', permutation),
        readBinding('positionGroups', positionGroups),
        writeBinding('rowGroups', rowGroups)
      ],
      invocationCount: rowCount,
      body: `rowGroups[rowGroupsOffset + permutation[permutationOffset + index]] =
    positionGroups[positionGroupsOffset + index];`
    })
  );

  // 7. Output keys and counts, one invocation per output row.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-groups`,
      operation,
      variant: 'groups',
      bindings: [
        readBinding('offsets', offsets),
        readBinding('groupTotal', groupTotal),
        ...sortedBindings(),
        writeBinding('keysOut', output.keys),
        writeBinding('countsOut', output.counts)
      ],
      invocationCount: capacity,
      declarations: `const CAPACITY: u32 = ${capacity}u;
const GROUP_NONE: u32 = ${GROUP_NONE}u;`,
      body: `if (index >= min(groupTotal[groupTotalOffset], CAPACITY)) {
    ${twoWords ? 'keysOut[keysOutOffset + 2u * index] = 0xffffffffu;\n    keysOut[keysOutOffset + 2u * index + 1u] = 0xffffffffu;' : 'keysOut[keysOutOffset + index] = 0xffffffffu;'}
    countsOut[countsOutOffset + index] = 0u;
    return;
  }
  let begin = offsets[offsetsOffset + index];
  let key = ${keyAt('begin')};
  ${
    twoWords
      ? 'keysOut[keysOutOffset + 2u * index] = key.y;\n  keysOut[keysOutOffset + 2u * index + 1u] = key.x;'
      : 'keysOut[keysOutOffset + index] = key.y;'
  }
  countsOut[countsOutOffset + index] = offsets[offsetsOffset + index + 1u] - begin;`
    })
  );

  // 8. Clamped count, overflow and total.
  nodes.push(
    createPublishNode<Parameters>(graph, {
      id: `${id}-publish`,
      operation,
      totalCount: groupTotal,
      output: {
        ids: output.counts,
        count: output.count,
        overflow: output.overflow,
        totalCount: output.totalCount
      }
    })
  );

  return {
    nodes,
    structure: {
      rowCount,
      capacity,
      twoWords,
      permutation,
      keyLow,
      keyHigh,
      rowIds,
      offsets,
      groupTotal,
      positionGroups,
      rowGroups
    }
  };
}
