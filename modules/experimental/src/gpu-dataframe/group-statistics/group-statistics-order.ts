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
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {ORDERED_KEY_WGSL} from '../../gpu-spatial-analysis/cell-aggregation/cell-table';
import {
  atomicBinding,
  createGatherNode,
  GROUP_VALUE_KEY_WGSL,
  readBinding,
  writeBinding
} from './group-statistics-common';
import {getKeyOrderNodes, type GroupStructure} from './group-statistics-sort';

/** Output views of one column written by the order statistics. @internal */
export type GroupColumnOrderOutput = {
  medians?: GraphDataView<'float32'>;
  percentiles?: GraphDataView<'float32'>;
  modes?: GraphDataView<'float32'>;
  uniqueCounts?: GraphDataView<'uint32'>;
};

/**
 * Linear-interpolation quantile of the first `n` sorted keys of a group (numpy `linear`,
 * `d3.quantile`): `h = (n - 1) p`, `lo + (hi - lo) * (h - floor h)` in f32.
 */
const QUANTILE_WGSL = /* wgsl */ `
// Identity that compilers cannot fold: \`zero\` is always 0 at run time (counts stay below 2^31)
// but unknown at compile time, so the bit operation stops multiply-add fusion and every f32 step
// rounds exactly like the CPU oracle.
fn roundStep(value: f32, zero: u32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) | zero);
}

fn getQuantile(begin: u32, count: u32, rawP: f32) -> f32 {
  if (count == 0u || !isFiniteBits(bitcast<u32>(rawP))) {
    return getNaN();
  }
  let p = clamp(rawP, 0.0, 1.0);
  let zero = count >> 31u;
  let h = roundStep(f32(count - 1u) * p, zero);
  let floorH = floor(h);
  let lowIndex = min(u32(floorH), count - 1u);
  let low = decodeOrderedKey(sortedKeys[sortedKeysOffset + begin + lowIndex]);
  if (lowIndex + 1u >= count) {
    return low;
  }
  let high = decodeOrderedKey(sortedKeys[sortedKeysOffset + begin + lowIndex + 1u]);
  return low + roundStep((high - low) * (h - floorH), zero);
}
`;

/**
 * Order statistics of one column: stable LSD sort by (value key, then group key words), so each
 * group's finite values are ascending at the front of the group's range, then quantiles by direct
 * lookup, and modes / unique counts from run heads, a scan and integer-atomic run reductions.
 *
 * `percentiles` is read from a parameter view at encode time (clamped to [0, 1]).
 *
 * @internal
 */
export function getColumnOrderNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    structure: GroupStructure;
    valueBits: GraphDataView<'uint32'>;
    finiteCounts: GraphDataView<'uint32'>;
    percentiles?: GraphDataView<'float32'>;
    output: GroupColumnOrderOutput;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, structure, valueBits, finiteCounts, output} = props;
  const {rowCount, capacity} = structure;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const u32 = (name: string, length: number = rowCount) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);

  // 1. Sort rows by value (ascending, non-finite last), then by key words (stable).
  const valueKeys = u32('value-keys');
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-value-keys`,
      operation,
      variant: 'value-keys',
      bindings: [readBinding('valueBits', valueBits), writeBinding('valueKeys', valueKeys)],
      invocationCount: rowCount,
      declarations: GROUP_VALUE_KEY_WGSL,
      body: 'valueKeys[valueKeysOffset + index] = getValueKey(valueBits[valueBitsOffset + index]);'
    })
  );
  const byValue = u32('rows-by-value');
  nodes.push(
    ...new GPUSort({
      id: `${id}-sort-value`,
      keys: valueKeys,
      values: structure.rowIds,
      outputKeys: u32('sorted-value-scratch'),
      outputValues: byValue,
      keyBits: 32
    }).getCommandNodes(graph)
  );
  const order = getKeyOrderNodes<Parameters>(graph, {
    id: `${id}-order`,
    operation,
    keyLow: structure.keyLow,
    keyHigh: structure.keyHigh,
    rowIds: structure.rowIds,
    initialRows: byValue
  });
  nodes.push(...order.nodes);
  const sortedKeys = u32('sorted-keys');
  nodes.push(
    createGatherNode(graph, {
      id: `${id}-gather-sorted-keys`,
      operation,
      indices: order.permutation,
      source: valueKeys,
      target: sortedKeys
    })
  );

  // 2. Quantiles.
  const quantileDeclarations = `${ORDERED_KEY_WGSL}
${GROUP_VALUE_KEY_WGSL}
${QUANTILE_WGSL}`;
  if (output.medians) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-medians`,
        operation,
        variant: 'medians',
        bindings: [
          readBinding('finiteCounts', finiteCounts),
          readBinding('offsets', structure.offsets),
          readBinding('sortedKeys', sortedKeys),
          writeBinding('mediansOut', output.medians, 'f32')
        ],
        invocationCount: capacity,
        declarations: quantileDeclarations,
        body: `let count = finiteCounts[finiteCountsOffset + index];
  var begin = 0u;
  if (count > 0u) {
    begin = offsets[offsetsOffset + index];
  }
  mediansOut[mediansOutOffset + index] = getQuantile(begin, count, 0.5);`
      })
    );
  }
  if (output.percentiles && props.percentiles) {
    const percentileCount = props.percentiles.length;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-percentiles`,
        operation,
        variant: 'percentiles',
        bindings: [
          readBinding('finiteCounts', finiteCounts),
          readBinding('offsets', structure.offsets),
          readBinding('sortedKeys', sortedKeys),
          readBinding('fractions', props.percentiles, 'f32'),
          writeBinding('percentilesOut', output.percentiles, 'f32')
        ],
        invocationCount: capacity * percentileCount,
        declarations: `${quantileDeclarations}
const PERCENTILE_COUNT: u32 = ${percentileCount}u;`,
        body: `let group = index / PERCENTILE_COUNT;
  let count = finiteCounts[finiteCountsOffset + group];
  var begin = 0u;
  if (count > 0u) {
    begin = offsets[offsetsOffset + group];
  }
  percentilesOut[percentilesOutOffset + index] =
    getQuantile(begin, count, fractions[fractionsOffset + index % PERCENTILE_COUNT]);`
      })
    );
  }

  // 3. Modes and unique counts from runs of equal values.
  const wantsMode = Boolean(output.modes);
  const wantsUnique = Boolean(output.uniqueCounts);
  if (!wantsMode && !wantsUnique) {
    return nodes;
  }
  const heads = u32('run-heads');
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-run-heads`,
      operation,
      variant: 'run-heads',
      bindings: [
        readBinding('sortedKeys', sortedKeys),
        readBinding('positionGroups', structure.positionGroups),
        readBinding('offsets', structure.offsets),
        writeBinding('heads', heads)
      ],
      invocationCount: rowCount,
      body: `let group = positionGroups[positionGroupsOffset + index];
  let key = sortedKeys[sortedKeysOffset + index];
  var isHead = group != 0xffffffffu && key != 0xffffffffu;
  if (isHead && index != offsets[offsetsOffset + group]) {
    isHead = key != sortedKeys[sortedKeysOffset + index - 1u];
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);`
    })
  );
  const uniqueCounts = u32('unique-counts', capacity);
  const maximumLengths = wantsMode ? u32('maximum-run-lengths', capacity) : undefined;
  const modeRuns = wantsMode ? u32('mode-runs', capacity) : undefined;
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-run-clear`,
      operation,
      variant: 'run-clear',
      bindings: [
        writeBinding('uniqueCounts', uniqueCounts),
        ...(maximumLengths ? [writeBinding('maximumLengths', maximumLengths)] : []),
        ...(modeRuns ? [writeBinding('modeRuns', modeRuns)] : [])
      ],
      invocationCount: capacity,
      body: `uniqueCounts[uniqueCountsOffset + index] = 0u;
  ${maximumLengths ? 'maximumLengths[maximumLengthsOffset + index] = 0u;' : ''}
  ${modeRuns ? 'modeRuns[modeRunsOffset + index] = 0xffffffffu;' : ''}`
    })
  );
  if (!wantsMode) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count-runs`,
        operation,
        variant: 'count-runs',
        bindings: [
          readBinding('heads', heads),
          readBinding('positionGroups', structure.positionGroups),
          atomicBinding('uniqueCounts', uniqueCounts)
        ],
        invocationCount: rowCount,
        body: `if (heads[headsOffset + index] != 0u) {
    atomicAdd(&uniqueCounts[uniqueCountsOffset + positionGroups[positionGroupsOffset + index]], 1u);
  }`
      })
    );
  } else if (maximumLengths && modeRuns) {
    const runIndices = u32('run-indices');
    nodes.push(
      ...new GPUScan({
        id: `${id}-run-scan`,
        input: heads,
        output: runIndices,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    const runStarts = u32('run-starts');
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-run-starts`,
        operation,
        variant: 'run-starts',
        bindings: [
          readBinding('heads', heads),
          readBinding('runIndices', runIndices),
          writeBinding('runStarts', runStarts)
        ],
        invocationCount: rowCount,
        body: `if (heads[headsOffset + index] != 0u) {
    runStarts[runStartsOffset + runIndices[runIndicesOffset + index]] = index;
  }`
      })
    );
    const runBindings = () => [
      readBinding('heads', heads),
      readBinding('runIndices', runIndices),
      readBinding('runStarts', runStarts),
      readBinding('positionGroups', structure.positionGroups),
      readBinding('offsets', structure.offsets),
      readBinding('finiteCounts', finiteCounts)
    ];
    const runDeclarations = `const ROW_COUNT: u32 = ${rowCount}u;`;
    const runLengthSource = `let run = runIndices[runIndicesOffset + index];
  let group = positionGroups[positionGroupsOffset + index];
  let totalRuns = runIndices[runIndicesOffset + ROW_COUNT - 1u] + heads[headsOffset + ROW_COUNT - 1u];
  var nextStart = ROW_COUNT;
  if (run + 1u < totalRuns) {
    nextStart = runStarts[runStartsOffset + run + 1u];
  }
  let runEnd = min(nextStart, offsets[offsetsOffset + group] + finiteCounts[finiteCountsOffset + group]);
  let runLength = runEnd - index;`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-run-lengths`,
        operation,
        variant: 'run-lengths',
        bindings: [
          ...runBindings(),
          atomicBinding('maximumLengths', maximumLengths),
          ...(wantsUnique ? [atomicBinding('uniqueCounts', uniqueCounts)] : [])
        ],
        invocationCount: rowCount,
        declarations: runDeclarations,
        body: `if (heads[headsOffset + index] == 0u) {
    return;
  }
  ${runLengthSource}
  atomicMax(&maximumLengths[maximumLengthsOffset + group], runLength);
  ${wantsUnique ? 'atomicAdd(&uniqueCounts[uniqueCountsOffset + group], 1u);' : ''}`
      })
    );
    // Ties in length resolve to the smallest run, which holds the smallest value.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mode-runs`,
        operation,
        variant: 'mode-runs',
        bindings: [
          ...runBindings(),
          atomicBinding('maximumLengths', maximumLengths),
          atomicBinding('modeRuns', modeRuns)
        ],
        invocationCount: rowCount,
        declarations: runDeclarations,
        body: `if (heads[headsOffset + index] == 0u) {
    return;
  }
  ${runLengthSource}
  if (runLength == atomicLoad(&maximumLengths[maximumLengthsOffset + group])) {
    atomicMin(&modeRuns[modeRunsOffset + group], run);
  }`
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish-modes`,
        operation,
        variant: 'finish-modes',
        bindings: [
          readBinding('finiteCounts', finiteCounts),
          readBinding('modeRuns', modeRuns),
          readBinding('runStarts', runStarts),
          readBinding('sortedKeys', sortedKeys),
          writeBinding('modesOut', output.modes!, 'f32')
        ],
        invocationCount: capacity,
        declarations: ORDERED_KEY_WGSL,
        body: `var mode = getNaN();
  if (finiteCounts[finiteCountsOffset + index] > 0u) {
    mode = decodeOrderedKey(
      sortedKeys[sortedKeysOffset + runStarts[runStartsOffset + modeRuns[modeRunsOffset + index]]]
    );
  }
  modesOut[modesOutOffset + index] = mode;`
      })
    );
  }
  if (output.uniqueCounts) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish-unique`,
        operation,
        variant: 'finish-unique',
        bindings: [
          readBinding('uniqueCounts', uniqueCounts),
          writeBinding('uniqueOut', output.uniqueCounts)
        ],
        invocationCount: capacity,
        body: 'uniqueOut[uniqueOutOffset + index] = uniqueCounts[uniqueCountsOffset + index];'
      })
    );
  }
  return nodes;
}
