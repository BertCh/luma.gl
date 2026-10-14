// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Workgroup size of kernels built with {@link getWorkgroupCounterKernelSource}. @internal */
export const WORKGROUP_COUNTER_WORKGROUP_SIZE = 256;

/** One workgroup counter and the global word it flushes into. @internal */
export type WorkgroupCounter = {
  /** Name of the `atomic<u32>` storage binding the counter flushes into. */
  target: string;
  /** Word of `target`, relative to its view. */
  word: number;
  /** `'add'` (default) sums the rows' values; `'min'` keeps their minimum. */
  operation?: 'add' | 'min';
};

/** Properties of {@link getWorkgroupCounterKernelSource}. @internal */
export type WorkgroupCounterKernelSourceProps = {
  /** Counters in index order: row code updates counter `i` of `counters[i]`. */
  counters: readonly WorkgroupCounter[];
  /** WGSL expression of the number of rows (`index < rowCount` runs `processRow`). */
  rowCount: string;
  /**
   * WGSL statements of `fn processRow(index: u32)`. They may `return` early and call
   * `addCounter(counter: u32, value: u32)` or `minCounter(counter: u32, value: u32)` instead of
   * an atomic on a global word.
   */
  rowSource: string;
};

/**
 * Builds the declarations and body of a kernel whose rows update a few fixed summary words.
 *
 * Library rule 14 (review E-14): rows never issue atomics onto a handful of global words. Each row
 * updates workgroup-memory counters, and after a barrier one invocation flushes every workgroup
 * total with a single global atomic, so a summary word takes one atomic per workgroup instead of
 * one per row. Integer addition and minimum are associative, so the results are unchanged. Use
 * the result with `guardIndex: false` and `workgroupSize: WORKGROUP_COUNTER_WORKGROUP_SIZE`; the
 * row work runs in a function, so it may return early without skipping the barriers.
 *
 * @internal
 */
export function getWorkgroupCounterKernelSource(props: WorkgroupCounterKernelSourceProps): {
  declarations: string;
  body: string;
} {
  const {counters} = props;
  const getEmptyValue = (counter: WorkgroupCounter) =>
    counter.operation === 'min' ? '0xffffffffu' : '0u';
  const resets = counters
    .map(
      (counter, index) => `atomicStore(&workgroupCounters[${index}], ${getEmptyValue(counter)});`
    )
    .join('\n    ');
  const flushes = counters
    .map((counter, index) => {
      const atomic = counter.operation === 'min' ? 'atomicMin' : 'atomicAdd';
      return `let total${index} = atomicLoad(&workgroupCounters[${index}]);
    if (total${index} != ${getEmptyValue(counter)}) {
      ${atomic}(&${counter.target}[${counter.target}Offset + ${counter.word}u], total${index});
    }`;
    })
    .join('\n    ');
  return {
    declarations: /* wgsl */ `
var<workgroup> workgroupCounters: array<atomic<u32>, ${counters.length}>;
fn addCounter(counter: u32, value: u32) {
  if (value > 0u) {
    atomicAdd(&workgroupCounters[counter], value);
  }
}
fn minCounter(counter: u32, value: u32) {
  atomicMin(&workgroupCounters[counter], value);
}
fn processRow(index: u32) {
  ${props.rowSource}
}`,
    body: /* wgsl */ `if (localInvocationIndex == 0u) {
    ${resets}
  }
  workgroupBarrier();
  if (index < ${props.rowCount}) {
    processRow(index);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    ${flushes}
  }`
  };
}
