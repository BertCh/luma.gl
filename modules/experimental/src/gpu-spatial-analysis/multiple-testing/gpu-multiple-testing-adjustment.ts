// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {GET_NAN_WGSL} from '../../utils/wgsl-common';
import type {GPUMultipleTestingMethod} from '../../analysis-core/contracts/index';

const OPERATION = 'GPUMultipleTestingAdjustment';

/** Rows per scan workgroup; fixed by the 256-wide workgroup-memory scan. */
const SCAN_BLOCK_SIZE = 256;

/** Sort key of an untested row: one above the bit pattern of 1.0f, so it sorts after every p-value. */
const UNTESTED_KEY = 0x3f800001;

/** Sort key width: `UNTESTED_KEY` needs 30 bits. */
const SORT_KEY_BITS = 30;

/**
 * Largest supported row count (exclusive): ranks and family sizes are held as exact f32 integers,
 * so the row count must stay below 2^24.
 */
export const GPU_MULTIPLE_TESTING_MAXIMUM_ROWS = 1 << 24;

const METHODS: readonly GPUMultipleTestingMethod[] = [
  'none',
  'bonferroni',
  'holm',
  'benjamini-hochberg',
  'benjamini-yekutieli'
];

/** Caller-owned outputs of {@link GPUMultipleTestingAdjustment}. */
export type GPUMultipleTestingAdjustmentOutput = {
  /**
   * Same length as `pValues`, must not alias it. Adjusted p-value in `[0, 1]`
   * (the definitions of statsmodels `multipletests`: `bonferroni`, `holm`, `fdr_bh`, `fdr_by`), NaN
   * where not tested. For every method a row is significant at level `alpha` exactly when
   * `adjustedPValue <= alpha`, so multi-level bins (0.10 / 0.05 / 0.01) are consistent by
   * construction. Tied p-values receive identical adjusted values.
   */
  adjustedPValues: GraphDataView<'float32'>;
  /** Optional one-row `uint32`: the family size `m` actually used. */
  testedCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUMultipleTestingAdjustment}.
 *
 * Per-frame (no rebuild): the contents of `pValues`. Compile-time: the row count, `method`,
 * `minimumFamilySize` and which optional views exist.
 */
export type GPUMultipleTestingAdjustmentProps = {
  /** Prefix for node and transient IDs. Defaults to `'multiple-testing-adjustment'`. */
  id?: string;
  /**
   * p-values, one per row. A row is in the family (tested) when its p-value is finite and in
   * `[0, 1]`; NaN (or anything else) means "not tested" and is excluded. There is no mask prop by
   * design: the family is every tested row. Run the statistic and this node on the full analysis
   * set and apply viewport masks only when rendering, otherwise panning would change `m` and make
   * the remaining rows look more or less significant.
   */
  pValues: GraphDataView<'float32'>;
  /**
   * Compile-time. `'none' | 'bonferroni' | 'holm' | 'benjamini-hochberg' | 'benjamini-yekutieli'`.
   * Use `'benjamini-yekutieli'` for local spatial statistics: neighboring local statistics share
   * observations, so their p-values are dependent, and Benjamini-Hochberg only controls the false
   * discovery rate under independence or positive regression dependence. Benjamini-Yekutieli
   * (BH times `c(m) = sum_{i=1..m} 1/i`) controls it under arbitrary dependence, at a cost in power
   * of about `ln m`. `'holm'` controls the family-wise error rate, `'bonferroni'` is its simpler,
   * more conservative form.
   */
  method: GPUMultipleTestingMethod;
  /**
   * Compile-time, optional non-negative integer below 2^24. The family size used is
   * `m = max(testedRows, minimumFamilySize)`: rows that have no p-value count as `p = 1`. Use it to
   * pin `m` when the tested set can shrink (for example a statistic whose analysis mask follows a
   * viewport) so masking cannot make the remaining rows more significant. Defaults to 0 (`m` is the
   * tested row count).
   */
  minimumFamilySize?: number;
  /** Caller-owned outputs. */
  output: GPUMultipleTestingAdjustmentOutput;
};

/** Validates a method name (throws on an unknown one) and returns it; `undefined` becomes `'none'`. */
export function resolveGPUMultipleTestingMethod(
  id: string,
  method: GPUMultipleTestingMethod | undefined
): GPUMultipleTestingMethod {
  const value = method ?? 'none';
  if (!METHODS.includes(value)) {
    throw new Error(`${id} method must be one of ${METHODS.join(', ')}, got ${String(method)}`);
  }
  return value;
}

function validateMinimumFamilySize(id: string, minimumFamilySize: number | undefined): number {
  const value = minimumFamilySize ?? 0;
  validateCapacity(id, 'minimumFamilySize', value, 0, GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1);
  return value;
}

/**
 * CPU reference (float64) of {@link GPUMultipleTestingAdjustment}, exported for oracles and tests.
 *
 * Rows whose p-value is not finite or not in `[0, 1]` are untested and return NaN. With `m` the
 * family size (`max(testedRows, minimumFamilySize)`), `p_(k)` the k-th smallest tested p-value and
 * `c(m) = sum_{i=1..m} 1/i`:
 * - `'none'`: `p`.
 * - `'bonferroni'`: `min(m p, 1)`.
 * - `'holm'`: `(m - k + 1) p_(k)`, cumulative maximum from the smallest rank, clamped to 1.
 * - `'benjamini-hochberg'`: `p_(k) m / k`, cumulative minimum from the largest tested rank down,
 *   clamped to 1.
 * - `'benjamini-yekutieli'`: the Benjamini-Hochberg value times `c(m)`, clamped to 1.
 *
 * These equal statsmodels `multipletests` (`bonferroni`, `holm`, `fdr_bh`, `fdr_by`) on the tested
 * rows, with `minimumFamilySize - testedRows` extra p-values of 1 appended and then dropped.
 */
export function getMultipleTestingAdjustedPValues(
  pValues: ArrayLike<number>,
  method: GPUMultipleTestingMethod,
  minimumFamilySize: number = 0
): Float64Array {
  resolveGPUMultipleTestingMethod('getMultipleTestingAdjustedPValues', method);
  validateMinimumFamilySize('getMultipleTestingAdjustedPValues', minimumFamilySize);
  const rows = pValues.length;
  const result = new Float64Array(rows).fill(Number.NaN);
  const tested: number[] = [];
  for (let row = 0; row < rows; row++) {
    const p = pValues[row];
    if (Number.isFinite(p) && p >= 0 && p <= 1) {
      tested.push(row);
    }
  }
  const testedCount = tested.length;
  const m = Math.max(testedCount, minimumFamilySize);
  if (method === 'none') {
    for (const row of tested) {
      result[row] = pValues[row];
    }
    return result;
  }
  if (method === 'bonferroni') {
    for (const row of tested) {
      result[row] = Math.min(pValues[row] * m, 1);
    }
    return result;
  }
  tested.sort((left, right) => pValues[left] - pValues[right] || left - right);
  if (method === 'holm') {
    let running = 0;
    for (let rank = 0; rank < testedCount; rank++) {
      running = Math.max(running, (m - rank) * pValues[tested[rank]]);
      result[tested[rank]] = Math.min(running, 1);
    }
    return result;
  }
  let scale = 1;
  if (method === 'benjamini-yekutieli') {
    scale = 0;
    for (let i = m; i >= 1; i--) {
      scale += 1 / i;
    }
  }
  let running = 1;
  for (let rank = testedCount - 1; rank >= 0; rank--) {
    running = Math.min(running, (pValues[tested[rank]] * m * scale) / (rank + 1));
    result[tested[rank]] = Math.min(running, 1);
  }
  return result;
}

/**
 * Multiple-testing adjustment of per-row p-values on the GPU: Bonferroni, Holm, Benjamini-Hochberg
 * and Benjamini-Yekutieli (see {@link getMultipleTestingAdjustedPValues} for the exact formulas,
 * which match statsmodels `multipletests`).
 *
 * The family is every tested row (finite p-value in `[0, 1]`); there is deliberately no mask
 * prop, so a display or viewport mask can never change the family. Pin the family size with
 * `minimumFamilySize` when the tested set can shrink. Rows that are not tested get NaN. A row is
 * significant at level `alpha` iff `adjustedPValue <= alpha`, for every method.
 *
 * Algorithm: a pass counts the tested rows (`atomicAdd`, exact integer) and, for the step
 * procedures, builds sort keys (non-negative f32 bit patterns order like `u32`; untested rows get a
 * key that sorts last). `GPUSort` orders the keys, a tiny kernel derives `m` and, for
 * Benjamini-Yekutieli, `c(m)` (an exact fixed-order sum for `m <= 64`, else
 * `ln m + gamma + 1/(2m) - 1/(12 m^2)`, accurate to f32 rounding). The raw step values are
 * combined by a cumulative maximum (Holm, forward) or minimum (Benjamini procedures, from the
 * largest rank) over the sorted order, with a deterministic three-kernel scan: block-local scans in
 * workgroup memory, one workgroup scanning the block totals, and an apply pass that also scatters
 * to the original rows. Minimum and maximum are exact in any order, so the result does not depend
 * on scheduling. Tied p-values get identical adjusted values through the cumulative extremum.
 *
 * Cost: `none` and Bonferroni are O(n). The step procedures add one 30-bit `GPUSort` of `n` keys
 * plus about 6 linear kernels; memory is a handful of `n`-sized `u32`/`f32` scratch columns. Row
 * counts are limited to below 2^24 so ranks are exact f32 integers.
 */
export class GPUMultipleTestingAdjustment implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUMultipleTestingAdjustmentProps;

  constructor(props: GPUMultipleTestingAdjustmentProps) {
    const id = props.id ?? 'multiple-testing-adjustment';
    this.id = id;
    this.props = props;
    resolveGPUMultipleTestingMethod(id, props.method);
    validateMinimumFamilySize(id, props.minimumFamilySize);
    validatePackedView(props.pValues, ['float32'], `${id} pValues`);
    validatePackedView(props.output.adjustedPValues, ['float32'], `${id} output.adjustedPValues`);
    const rows = props.pValues.length;
    validateCapacity(id, 'pValues length', rows, 1, GPU_MULTIPLE_TESTING_MAXIMUM_ROWS - 1);
    if (props.output.adjustedPValues.length !== rows) {
      throw new Error(`${id} adjustedPValues length must equal the pValues length`);
    }
    if (props.output.testedCount) {
      validatePackedUint32View(props.output.testedCount, `${id} output.testedCount`);
      validateCapacity(id, 'output.testedCount length', props.output.testedCount.length);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.adjustedPValues, props.output.testedCount],
      [props.pValues]
    );
  }

  /** Returns the adjustment nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.pValues,
      props.output.adjustedPValues,
      props.output.testedCount
    ]);
    const method = resolveGPUMultipleTestingMethod(id, props.method);
    const minimumFamilySize = validateMinimumFamilySize(id, props.minimumFamilySize);
    const rows = props.pValues.length;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const testedWGSL = `${GET_NAN_WGSL}
// A p-value is tested when it is in [0, 1]: +0..1.0 as positive bit patterns, or -0.
fn isTestedBits(bits: u32) -> bool {
  return bits <= 0x3f800000u || bits == 0x80000000u;
}`;

    if (method === 'none' && !props.output.testedCount) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-apply`,
          operation: OPERATION,
          variant: 'none',
          bindings: [
            {name: 'pValues', view: props.pValues, type: 'f32', access: 'read'},
            {
              name: 'adjusted',
              view: props.output.adjustedPValues,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: rows,
          declarations: testedWGSL,
          body: `let p = pValues[pValuesOffset + index];
  adjusted[adjustedOffset + index] = select(getNaN(), p + 0.0, isTestedBits(bitcast<u32>(p)));`
        })
      );
      return nodes;
    }

    const isStepProcedure =
      method === 'holm' || method === 'benjamini-hochberg' || method === 'benjamini-yekutieli';
    const counters = createTransientView(graph, `${id}-counters`, 'uint32', 1);
    // family[0] = tested rows, family[1] = family size m.
    const family = createTransientView(graph, `${id}-family`, 'uint32', 2);
    const familyScale = createTransientView(graph, `${id}-family-scale`, 'float32', 1);
    const keys = isStepProcedure
      ? createTransientView(graph, `${id}-keys`, 'uint32', rows)
      : undefined;
    const rowIds = isStepProcedure
      ? createTransientView(graph, `${id}-row-ids`, 'uint32', rows)
      : undefined;

    const countBindings: WGSLKernelBinding[] = [
      {name: 'pValues', view: props.pValues, type: 'f32', access: 'read'},
      {name: 'counters', view: counters, type: 'atomic<u32>', access: 'read_write'}
    ];
    if (keys && rowIds) {
      countBindings.push(
        {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
        {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
      );
    }
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-counters-clear`,
        operation: OPERATION,
        view: counters,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: keys ? 'count-keys' : 'count',
        bindings: countBindings,
        invocationCount: rows,
        declarations: `${testedWGSL}
const UNTESTED_KEY: u32 = ${UNTESTED_KEY}u;`,
        // Non-negative f32 bit patterns order like u32, so the bits are the sort key; -0 maps to 0.
        body: `let bits = bitcast<u32>(pValues[pValuesOffset + index]);
  let tested = isTestedBits(bits);
  if (tested) {
    atomicAdd(&counters[countersOffset], 1u);
  }
  ${
    keys
      ? `keys[keysOffset + index] = select(UNTESTED_KEY, bits & 0x7fffffffu, tested);
  rowIds[rowIdsOffset + index] = index;`
      : ''
  }`
      })
    );

    const familyBindings: WGSLKernelBinding[] = [
      {name: 'counters', view: counters, type: 'u32', access: 'read'},
      {name: 'family', view: family, type: 'u32', access: 'read_write'},
      {name: 'familyScale', view: familyScale, type: 'f32', access: 'read_write'}
    ];
    if (props.output.testedCount) {
      familyBindings.push({
        name: 'testedCount',
        view: props.output.testedCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-family`,
        operation: OPERATION,
        variant: 'family',
        bindings: familyBindings,
        invocationCount: 1,
        declarations: `const MINIMUM_FAMILY_SIZE: u32 = ${minimumFamilySize}u;
const EULER_MASCHERONI: f32 = 0.5772157;
// Harmonic number c(m) = sum_{i=1..m} 1/i. Exact fixed-order sum (smallest terms first) for small
// m; asymptotic series ln m + gamma + 1/(2m) - 1/(12 m^2) (error < 1/(120 m^4)) beyond.
fn getHarmonicNumber(m: u32) -> f32 {
  if (m <= 64u) {
    var sum = 0.0;
    for (var i = m; i >= 1u; i--) {
      sum += 1.0 / f32(i);
    }
    return sum;
  }
  let x = f32(m);
  return log(x) + EULER_MASCHERONI + 0.5 / x - 1.0 / (12.0 * x * x);
}`,
        body: `let tested = counters[countersOffset];
  let m = max(tested, MINIMUM_FAMILY_SIZE);
  family[familyOffset] = tested;
  family[familyOffset + 1u] = m;
  familyScale[familyScaleOffset] = ${method === 'benjamini-yekutieli' ? 'getHarmonicNumber(m)' : '1.0'};
  ${props.output.testedCount ? 'testedCount[testedCountOffset] = m;' : ''}`
      })
    );

    if (method === 'none' || method === 'bonferroni') {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-apply`,
          operation: OPERATION,
          variant: method,
          bindings: [
            {name: 'pValues', view: props.pValues, type: 'f32', access: 'read'},
            {name: 'family', view: family, type: 'u32', access: 'read'},
            {
              name: 'adjusted',
              view: props.output.adjustedPValues,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: rows,
          declarations: testedWGSL,
          body: `let p = pValues[pValuesOffset + index];
  let m = f32(family[familyOffset + 1u]);
  let value = ${method === 'none' ? 'p + 0.0' : 'min(p * m, 1.0)'};
  adjusted[adjustedOffset + index] = select(getNaN(), value, isTestedBits(bitcast<u32>(p)));`
        })
      );
      return nodes;
    }

    // Step procedures: sort, scan, scatter.
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', rows);
    const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', rows);
    const blockCount = Math.ceil(rows / SCAN_BLOCK_SIZE);
    const blockChunk = Math.ceil(blockCount / SCAN_BLOCK_SIZE);
    const scanned = createTransientView(graph, `${id}-scanned`, 'float32', rows);
    const blockTotals = createTransientView(graph, `${id}-block-totals`, 'float32', blockCount);
    // Holm accumulates a maximum upward from the smallest rank; the Benjamini procedures
    // accumulate a minimum downward from the largest rank, so their scan runs on reversed order.
    const isMaximumScan = method === 'holm';
    const reverse = !isMaximumScan;
    const scanWGSL = `const ROWS: u32 = ${rows}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const BLOCK_CHUNK: u32 = ${blockChunk}u;
const REVERSE: bool = ${reverse};
const IDENTITY: f32 = ${isMaximumScan ? '0.0' : '3.0e38'};
var<workgroup> scanValues: array<f32, ${SCAN_BLOCK_SIZE}>;

fn combine(left: f32, right: f32) -> f32 {
  return ${isMaximumScan ? 'max(left, right)' : 'min(left, right)'};
}

// Raw step value of sorted rank r (rank k = r + 1) given tested count, family size m and c(m);
// IDENTITY beyond the tested rows.
fn getStepValue(r: u32, key: u32, tested: u32, m: u32, scale: f32) -> f32 {
  if (r >= tested) {
    return IDENTITY;
  }
  let p = bitcast<f32>(key);
  ${isMaximumScan ? 'return f32(m - r) * p;' : 'return ((p * f32(m)) / f32(r + 1u)) * scale;'}
}

// Hillis-Steele inclusive scan of scanValues across the workgroup. Every invocation must call it.
fn scanWorkgroup(localIndex: u32) {
  workgroupBarrier();
  for (var offset = 1u; offset < ${SCAN_BLOCK_SIZE}u; offset = offset * 2u) {
    var other = IDENTITY;
    if (localIndex >= offset) {
      other = scanValues[localIndex - offset];
    }
    let combined = combine(scanValues[localIndex], other);
    workgroupBarrier();
    scanValues[localIndex] = combined;
    workgroupBarrier();
  }
}`;
    nodes.push(
      ...new GPUSort({
        id: `${id}-sort`,
        keys: keys!,
        values: rowIds!,
        outputKeys: sortedKeys,
        outputValues: sortedRows,
        keyBits: SORT_KEY_BITS
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-block-scan`,
        operation: OPERATION,
        variant: 'block-scan',
        bindings: [
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'family', view: family, type: 'u32', access: 'read'},
          {name: 'familyScale', view: familyScale, type: 'f32', access: 'read'},
          {name: 'scanned', view: scanned, type: 'f32', access: 'read_write'},
          {name: 'blockTotals', view: blockTotals, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        workgroupSize: SCAN_BLOCK_SIZE,
        guardIndex: false,
        declarations: scanWGSL,
        body: `// Scan position j; the sorted rank it reads is r.
  let j = index;
  var value = IDENTITY;
  if (j < ROWS) {
    let r = select(j, ROWS - 1u - j, REVERSE);
    value = getStepValue(r, sortedKeys[sortedKeysOffset + r], family[familyOffset], family[familyOffset + 1u], familyScale[familyScaleOffset]);
  }
  scanValues[localInvocationIndex] = value;
  scanWorkgroup(localInvocationIndex);
  if (j < ROWS) {
    scanned[scannedOffset + j] = scanValues[localInvocationIndex];
  }
  if (localInvocationIndex == ${SCAN_BLOCK_SIZE - 1}u) {
    blockTotals[blockTotalsOffset + workgroupIndex] = scanValues[localInvocationIndex];
  }`
      })
    );
    if (blockCount > 1) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-total-scan`,
          operation: OPERATION,
          variant: 'total-scan',
          bindings: [{name: 'blockTotals', view: blockTotals, type: 'f32', access: 'read_write'}],
          invocationCount: SCAN_BLOCK_SIZE,
          workgroupSize: SCAN_BLOCK_SIZE,
          guardIndex: false,
          declarations: scanWGSL,
          // One workgroup: each invocation scans a contiguous chunk of block totals, the chunk
          // totals are scanned across the workgroup, and each chunk then applies the prefix of
          // the chunks before it.
          body: `let begin = min(localInvocationIndex * BLOCK_CHUNK, BLOCK_COUNT);
  let end = min(begin + BLOCK_CHUNK, BLOCK_COUNT);
  var running = IDENTITY;
  for (var block = begin; block < end; block++) {
    running = combine(running, blockTotals[blockTotalsOffset + block]);
    blockTotals[blockTotalsOffset + block] = running;
  }
  scanValues[localInvocationIndex] = running;
  scanWorkgroup(localInvocationIndex);
  var prefix = IDENTITY;
  if (localInvocationIndex > 0u) {
    prefix = scanValues[localInvocationIndex - 1u];
  }
  for (var block = begin; block < end; block++) {
    blockTotals[blockTotalsOffset + block] = combine(blockTotals[blockTotalsOffset + block], prefix);
  }`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scatter`,
        operation: OPERATION,
        variant: 'scatter',
        bindings: [
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'family', view: family, type: 'u32', access: 'read'},
          {name: 'scanned', view: scanned, type: 'f32', access: 'read'},
          {name: 'blockTotals', view: blockTotals, type: 'f32', access: 'read'},
          {name: 'adjusted', view: props.output.adjustedPValues, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${GET_NAN_WGSL}
const ROWS: u32 = ${rows}u;
const REVERSE: bool = ${reverse};
fn combine(left: f32, right: f32) -> f32 {
  return ${isMaximumScan ? 'max(left, right)' : 'min(left, right)'};
}`,
        body: `let j = select(index, ROWS - 1u - index, REVERSE);
  var value = scanned[scannedOffset + j];
  let block = j / ${SCAN_BLOCK_SIZE}u;
  if (block > 0u) {
    value = combine(value, blockTotals[blockTotalsOffset + block - 1u]);
  }
  adjusted[adjustedOffset + sortedRows[sortedRowsOffset + index]] =
    select(getNaN(), min(value, 1.0), index < family[familyOffset]);`
      })
    );
    return nodes;
  }
}
