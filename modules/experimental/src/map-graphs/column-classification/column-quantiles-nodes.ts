// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import {validateGraphViewsBelongToGraph} from '../map-graph-utils';
import {COLUMN_ORDERED_KEY_WGSL} from './column-classification-shared';
import {GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH} from './column-quantiles-parameters';

/** Caller-owned outputs of the column quantile nodes. */
export type GPUColumnQuantilesOutput = {
  /** `quantileCount` rows; NaN when no valid rows exist or the probability is NaN or outside `[0, 1]`. */
  quantiles: GraphDataView<'float32'>;
  /** One row: rows that are unmasked and not NaN. */
  validCount: GraphDataView<'uint32'>;
  /** `values.length` rows: 1 when the row is valid and inside the filter bounds, else 0. */
  filterMask?: GraphDataView<'uint32'>;
  /** Two rows: `[filterLowerValue, filterUpperValue]`; NaN when no valid rows exist. */
  filterBounds?: GraphDataView<'float32'>;
};

/** Properties of {@link getColumnQuantileNodes}. @internal */
export type ColumnQuantileNodesProps = {
  /** Prefix of node and transient IDs. */
  id: string;
  /** Operation name reported in workload estimates. */
  operation: string;
  /** Packed float32 column. */
  values: GraphDataView<'float32'>;
  /** Optional row mask; zero skips the row. */
  mask?: GraphDataView<'uint32'>;
  /** Float32 parameters in the `getGPUColumnQuantilesParameterValues` layout. */
  parameters: GraphDataView<'float32'>;
  /** Compile-time probability count, 1 to 64. */
  quantileCount: number;
  /** Caller-owned outputs. */
  output: GPUColumnQuantilesOutput;
  /**
   * Optional per-frame gate: every kernel does nothing unless `gate.view[0] == gate.value`.
   * Outputs of a gated-off frame are left untouched.
   */
  gate?: {view: GraphDataView<'uint32'>; value: number};
};

const WORKGROUP_SIZE = 256;
const BIN_COUNT = 256;
/** Digit passes of 8 bits each over the 32-bit ordered key. */
const PASS_COUNT = 4;
/** Slots whose histogram lives in workgroup memory (8 slots x 256 bins x 4 bytes = 8 KB). */
const LOCAL_SLOT_COUNT = 8;
const ROWS_PER_THREAD = 16;
const MAXIMUM_HISTOGRAM_WORKGROUPS = 256;

/**
 * Builds the radix-select nodes shared by `GPUColumnQuantiles` and `GPUClassBreaks`.
 *
 * Targets are the order-statistic ranks needed: two per probability (the lower and upper
 * neighbor, equal for lower/higher/nearest) plus two for the percentile filter. Four passes of
 * 8-bit digits run MSB first. In each pass one histogram kernel counts the digits of rows whose
 * higher digits equal a target's prefix, and one single-workgroup select kernel scans each
 * target's 256 bins, picks the bucket holding the remaining rank, and narrows `(prefix, rank)`.
 * Histograms are kept per DISTINCT active prefix (targets that share a prefix share a slot); the
 * select kernel publishes the sorted distinct prefixes and each row finds its slot by binary
 * search. Counts are integer atomics (workgroup-local for the first 8 slots, global above), so
 * the result is independent of thread order. After the fourth pass each target holds the exact
 * ordered key of its order statistic. A finish kernel decodes keys, interpolates, and writes the
 * outputs; an optional mask kernel writes the percentile filter mask.
 *
 * Returns init, `histogram-p` and `select-p` for p in 0..3, `finish`, and optionally `filter-mask`.
 *
 * @internal
 */
export function getColumnQuantileNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ColumnQuantileNodesProps
): GPUCommandNode<Parameters>[] {
  const {id, operation, values, mask, parameters, quantileCount, output, gate} = props;
  validateGraphViewsBelongToGraph(id, graph, [
    values,
    mask,
    parameters,
    gate?.view,
    output.quantiles,
    output.validCount,
    output.filterMask,
    output.filterBounds
  ]);
  const rows = values.length;
  const hasFilter = Boolean(output.filterMask || output.filterBounds);
  const targetCount = 2 * quantileCount + (hasFilter ? 2 : 0);
  const histogram = createTransientView(
    graph,
    `${id}-histogram`,
    'uint32',
    targetCount * BIN_COUNT
  );
  const stateLength = 4 * targetCount + 2;
  const state = createTransientView(graph, `${id}-state`, 'uint32', stateLength);
  const histogramWorkgroups = Math.min(
    MAXIMUM_HISTOGRAM_WORKGROUPS,
    Math.max(1, Math.ceil(rows / (WORKGROUP_SIZE * ROWS_PER_THREAD)))
  );

  const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): MapGraphKernelBinding => ({
    name,
    view,
    type,
    access: 'read'
  });
  const write = (
    name: string,
    view: GraphDataView,
    type: 'u32' | 'f32' | 'atomic<u32>' = 'u32'
  ): MapGraphKernelBinding => ({name, view, type, access: 'read_write'});
  const gateBindings = gate ? [read('gate', gate.view, 'u32')] : [];
  const enabledExpression = gate ? `gate[gateOffset] == ${gate.value >>> 0}u` : 'true';

  const declarations = /* wgsl */ `
const QUANTILE_COUNT: u32 = ${quantileCount}u;
const TARGET_COUNT: u32 = ${targetCount}u;
const HEADER_LENGTH: u32 = ${GPU_COLUMN_QUANTILES_PARAMETER_HEADER_LENGTH}u;
const STATE_PREFIX: u32 = 0u;
const STATE_RANK: u32 = ${targetCount}u;
const STATE_SLOT: u32 = ${2 * targetCount}u;
const STATE_ACTIVE_PREFIX: u32 = ${3 * targetCount}u;
const STATE_ACTIVE_COUNT: u32 = ${4 * targetCount}u;
const STATE_VALID_COUNT: u32 = ${4 * targetCount + 1}u;
${COLUMN_ORDERED_KEY_WGSL}
`;
  const rankDeclarations = /* wgsl */ `
fn isValidProbability(p: f32) -> bool {
  return p >= 0.0 && p <= 1.0;
}

// Round half to even; the fraction h - floor(h) is exact in f32.
fn roundHalfEven(h: f32) -> f32 {
  let whole = floor(h);
  let fraction = h - whole;
  if (fraction < 0.5) {
    return whole;
  }
  if (fraction > 0.5) {
    return whole + 1.0;
  }
  return select(whole, whole + 1.0, (u32(whole) & 1u) == 1u);
}

// The product is correctly rounded in WGSL, so the CPU oracle matches it with Math.fround.
fn getQuantileHeight(validCount: u32, p: f32) -> f32 {
  return f32(validCount - 1u) * p;
}

fn getQuantileRanks(validCount: u32, p: f32) -> vec2<u32> {
  let h = getQuantileHeight(validCount, p);
  let lower = floor(h);
  let upper = ceil(h);
  let code = u32(params[paramsOffset]);
  var first = lower;
  var second = lower;
  if (code == 1u) {
    first = upper;
    second = upper;
  } else if (code == 2u) {
    first = roundHalfEven(h);
    second = first;
  } else if (code == 3u) {
    second = min(lower + 1.0, f32(validCount - 1u));
  } else if (code == 4u) {
    second = upper;
  }
  return vec2<u32>(u32(first), u32(second));
}

fn getTargetRank(targetIndex: u32, validCount: u32) -> u32 {
  if (validCount == 0u) {
    return 0u;
  }
  if (targetIndex < 2u * QUANTILE_COUNT) {
    let p = params[paramsOffset + HEADER_LENGTH + targetIndex / 2u];
    if (!isValidProbability(p)) {
      return 0u;
    }
    let ranks = getQuantileRanks(validCount, p);
    return select(ranks.x, ranks.y, (targetIndex & 1u) == 1u);
  }
  let lastIndex = f32(validCount - 1u);
  if (targetIndex == 2u * QUANTILE_COUNT) {
    let lowerFraction = params[paramsOffset + 1u];
    let fraction = select(clamp(lowerFraction, 0.0, 1.0), 0.0, isNanBits(lowerFraction));
    return u32(clamp(floor(f32(validCount) * fraction), 0.0, lastIndex));
  }
  let upperFraction = params[paramsOffset + 2u];
  let fraction = select(clamp(upperFraction, 0.0, 1.0), 1.0, isNanBits(upperFraction));
  return u32(clamp(ceil(f32(validCount) * fraction) - 1.0, 0.0, lastIndex));
}
`;
  const nodes: GPUCommandNode<Parameters>[] = [];

  nodes.push(
    createMapGraphKernelNode<Parameters>(graph, {
      id: `${id}-init`,
      operation,
      variant: 'quantile-init',
      bindings: [write('histogram', histogram), ...gateBindings],
      invocationCount: histogram.length,
      body: `if (${enabledExpression}) {
    histogram[histogramOffset + index] = 0u;
  }`
    })
  );

  for (let pass = 0; pass < PASS_COUNT; pass++) {
    const shift = 24 - 8 * pass;
    const localBinCount = LOCAL_SLOT_COUNT * BIN_COUNT;
    const slotSource =
      pass === 0
        ? 'let slot = 0u;'
        : `let prefixValue = key >> ${shift + 8}u;
      var low = 0u;
      var high = activeCount;
      while (low < high) {
        let middle = (low + high) / 2u;
        if (state[stateOffset + STATE_ACTIVE_PREFIX + middle] < prefixValue) {
          low = middle + 1u;
        } else {
          high = middle;
        }
      }
      if (low >= activeCount || state[stateOffset + STATE_ACTIVE_PREFIX + low] != prefixValue) {
        continue;
      }
      let slot = low;`;
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-histogram-${pass}`,
        operation,
        variant: `quantile-histogram-${pass}`,
        bindings: [
          read('values', values, 'f32'),
          ...(mask ? [read('rowMask', mask, 'u32')] : []),
          read('state', state, 'u32'),
          write('histogram', histogram, 'atomic<u32>'),
          ...gateBindings
        ],
        invocationCount: histogramWorkgroups * WORKGROUP_SIZE,
        workgroupSize: WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${declarations}
const ROW_COUNT: u32 = ${rows}u;
const LOCAL_SLOT_COUNT: u32 = ${LOCAL_SLOT_COUNT}u;
const LOCAL_BIN_COUNT: u32 = ${localBinCount}u;
var<workgroup> localHistogram: array<atomic<u32>, ${localBinCount}>;`,
        body: `for (var bin = localInvocationIndex; bin < LOCAL_BIN_COUNT; bin = bin + ${WORKGROUP_SIZE}u) {
    atomicStore(&localHistogram[bin], 0u);
  }
  workgroupBarrier();
  let isEnabled = ${enabledExpression};
  if (isEnabled) {
    let activeCount = ${pass === 0 ? '1u' : 'state[stateOffset + STATE_ACTIVE_COUNT]'};
    for (var row = index; row < ROW_COUNT; row = row + INVOCATION_COUNT) {
      let value = values[valuesOffset + row];
      if (isNanBits(value)) {
        continue;
      }
      ${mask ? 'if (rowMask[rowMaskOffset + row] == 0u) {\n        continue;\n      }' : ''}
      let key = getOrderedKey(value);
      ${slotSource}
      let digit = (key >> ${shift}u) & 255u;
      if (slot < LOCAL_SLOT_COUNT) {
        atomicAdd(&localHistogram[slot * 256u + digit], 1u);
      } else {
        atomicAdd(&histogram[histogramOffset + slot * 256u + digit], 1u);
      }
    }
  }
  workgroupBarrier();
  if (isEnabled) {
    for (var bin = localInvocationIndex; bin < LOCAL_BIN_COUNT; bin = bin + ${WORKGROUP_SIZE}u) {
      let count = atomicLoad(&localHistogram[bin]);
      if (count != 0u && bin < ${targetCount * BIN_COUNT}u) {
        atomicAdd(&histogram[histogramOffset + bin], count);
      }
    }
  }`
      })
    );

    const isFirstPass = pass === 0;
    const isLastPass = pass === PASS_COUNT - 1;
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-select-${pass}`,
        operation,
        variant: `quantile-select-${pass}`,
        bindings: [
          write('histogram', histogram),
          write('state', state),
          read('params', parameters, 'f32'),
          ...gateBindings
        ],
        invocationCount: WORKGROUP_SIZE,
        workgroupSize: WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${declarations}
${rankDeclarations}
var<workgroup> sharedPrefix: array<u32, ${targetCount}>;
var<workgroup> sharedFirst: array<u32, ${targetCount}>;
var<workgroup> sharedValidCount: u32;`,
        body: `let isEnabled = ${enabledExpression};
  let targetIndex = localInvocationIndex;
  ${
    isFirstPass
      ? `if (targetIndex == 0u) {
    var total = 0u;
    if (isEnabled) {
      for (var bin = 0u; bin < 256u; bin = bin + 1u) {
        total = total + histogram[histogramOffset + bin];
      }
      state[stateOffset + STATE_VALID_COUNT] = total;
    }
    sharedValidCount = total;
  }
  workgroupBarrier();
  let validCount = sharedValidCount;`
      : ''
  }
  if (isEnabled && targetIndex < TARGET_COUNT) {
    var prefix = ${isFirstPass ? '0u' : 'state[stateOffset + STATE_PREFIX + targetIndex]'};
    var remaining = ${isFirstPass ? 'getTargetRank(targetIndex, validCount)' : 'state[stateOffset + STATE_RANK + targetIndex]'};
    let slot = ${isFirstPass ? '0u' : 'state[stateOffset + STATE_SLOT + targetIndex]'};
    var digit = 255u;
    for (var bin = 0u; bin < 256u; bin = bin + 1u) {
      let count = histogram[histogramOffset + slot * 256u + bin];
      if (remaining < count) {
        digit = bin;
        break;
      }
      remaining = remaining - count;
    }
    prefix = (prefix << 8u) | digit;
    sharedPrefix[targetIndex] = prefix;
    state[stateOffset + STATE_PREFIX + targetIndex] = prefix;
    state[stateOffset + STATE_RANK + targetIndex] = remaining;
  }
  // The histogram of this pass is consumed: clear it for the next pass.
  workgroupBarrier();
  if (isEnabled) {
    for (var cell = targetIndex; cell < ${targetCount * BIN_COUNT}u; cell = cell + ${WORKGROUP_SIZE}u) {
      histogram[histogramOffset + cell] = 0u;
    }
  }
  ${
    isLastPass
      ? ''
      : `if (isEnabled && targetIndex < TARGET_COUNT) {
    var first = 1u;
    for (var other = 0u; other < targetIndex; other = other + 1u) {
      if (sharedPrefix[other] == sharedPrefix[targetIndex]) {
        first = 0u;
        break;
      }
    }
    sharedFirst[targetIndex] = first;
  }
  workgroupBarrier();
  if (isEnabled && targetIndex < TARGET_COUNT) {
    let prefix = sharedPrefix[targetIndex];
    var slot = 0u;
    for (var other = 0u; other < TARGET_COUNT; other = other + 1u) {
      if (sharedFirst[other] == 1u && sharedPrefix[other] < prefix) {
        slot = slot + 1u;
      }
    }
    state[stateOffset + STATE_SLOT + targetIndex] = slot;
    if (sharedFirst[targetIndex] == 1u) {
      state[stateOffset + STATE_ACTIVE_PREFIX + slot] = prefix;
    }
    if (targetIndex == 0u) {
      var distinct = 0u;
      for (var other = 0u; other < TARGET_COUNT; other = other + 1u) {
        distinct = distinct + sharedFirst[other];
      }
      state[stateOffset + STATE_ACTIVE_COUNT] = distinct;
    }
  }`
  }`
      })
    );
  }

  const finishBindings: MapGraphKernelBinding[] = [
    read('state', state, 'u32'),
    read('params', parameters, 'f32'),
    write('quantilesOut', output.quantiles, 'f32'),
    write('validCountOut', output.validCount, 'u32'),
    ...(output.filterBounds ? [write('filterBoundsOut', output.filterBounds, 'f32')] : []),
    ...gateBindings
  ];
  nodes.push(
    createMapGraphKernelNode<Parameters>(graph, {
      id: `${id}-finish`,
      operation,
      variant: 'quantile-finish',
      bindings: finishBindings,
      invocationCount: quantileCount + 1,
      declarations: `${declarations}
${rankDeclarations}`,
      body: `if (!(${enabledExpression})) {
    return;
  }
  let validCount = state[stateOffset + STATE_VALID_COUNT];
  if (index < QUANTILE_COUNT) {
    var result = getNaN();
    let p = params[paramsOffset + HEADER_LENGTH + index];
    if (validCount > 0u && isValidProbability(p)) {
      let first = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * index]);
      let second = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * index + 1u]);
      result = first;
      let code = u32(params[paramsOffset]);
      if (first != second) {
        if (code == 3u) {
          let h = getQuantileHeight(validCount, p);
          let fraction = h - floor(h);
          if (fraction != 0.0) {
            result = first + (second - first) * fraction;
          }
        } else if (code == 4u) {
          result = (first + second) * 0.5;
        }
      }
    }
    quantilesOut[quantilesOutOffset + index] = result;
  } else {
    validCountOut[validCountOutOffset] = validCount;
    ${
      output.filterBounds
        ? `let hasRows = validCount > 0u;
    let lowerBound = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT]);
    let upperBound = decodeOrderedKey(state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT + 1u]);
    filterBoundsOut[filterBoundsOutOffset] = select(getNaN(), lowerBound, hasRows);
    filterBoundsOut[filterBoundsOutOffset + 1u] = select(getNaN(), upperBound, hasRows);`
        : ''
    }
  }`
    })
  );

  if (output.filterMask) {
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-filter-mask`,
        operation,
        variant: 'quantile-filter-mask',
        bindings: [
          read('values', values, 'f32'),
          ...(mask ? [read('rowMask', mask, 'u32')] : []),
          read('state', state, 'u32'),
          write('filterMaskOut', output.filterMask, 'u32'),
          ...gateBindings
        ],
        invocationCount: rows,
        declarations,
        body: `if (!(${enabledExpression})) {
    return;
  }
  var inside = 0u;
  let value = values[valuesOffset + index];
  if (state[stateOffset + STATE_VALID_COUNT] > 0u && !isNanBits(value)) {
    ${mask ? 'if (rowMask[rowMaskOffset + index] != 0u) {' : '{'}
      let key = getOrderedKey(value);
      let lowerKey = state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT];
      let upperKey = state[stateOffset + STATE_PREFIX + 2u * QUANTILE_COUNT + 1u];
      inside = select(0u, 1u, key >= lowerKey && key <= upperKey);
    }
  }
  filterMaskOut[filterMaskOutOffset + index] = inside;`
      })
    );
  }
  return nodes;
}
