// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Workgroup size of kernels that use {@link getBalancedSearchSource}. @internal */
export const BALANCED_SEARCH_WORKGROUP_SIZE = 128;

/**
 * WGSL for load-balanced expansion: one invocation per OUTPUT slot finds the input row that owns
 * it, instead of one invocation per input row looping over that row's (very uneven) output count.
 *
 * Rows own consecutive output ranges `[start(row), start(row) + count(row))` from an exclusive scan
 * of the counts. A workgroup binary-searches the owner of its first and last slot once, so every
 * lane then searches only the short window of rows between them (at most one row per slot), which
 * is a merge-path style partition: `O(log rows)` per workgroup plus `O(log window)` per lane, with
 * no divergence from long rows.
 *
 * The caller declares `fn balancedStart(row: u32) -> u32` and `fn balancedCount(row: u32) -> u32`
 * and builds the kernel with `workgroupSize: BALANCED_SEARCH_WORKGROUP_SIZE` and
 * `guardIndex: false`. `findBalancedOwner` contains workgroup barriers, so it must be the first
 * statement of the body (before any early return). It returns `BALANCED_NONE` for slots past the
 * last output.
 *
 * @internal
 */
export function getBalancedSearchSource(rowCount: number): string {
  return /* wgsl */ `
const BALANCED_ROWS: u32 = ${rowCount}u;
const BALANCED_LANES: u32 = ${BALANCED_SEARCH_WORKGROUP_SIZE}u;
const BALANCED_NONE: u32 = 0xffffffffu;
var<workgroup> balancedWindow: array<u32, 2>;

// Last row in [low, high] whose output range starts at or before the slot.
fn balancedLastAtMost(slot: u32, lowRow: u32, highRow: u32) -> u32 {
  var low = lowRow;
  var high = highRow;
  while (low < high) {
    let middle = (low + high + 1u) >> 1u;
    if (balancedStart(middle) <= slot) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  return low;
}

fn findBalancedOwner(slot: u32, firstSlot: u32, lane: u32) -> u32 {
  if (lane == 0u) {
    let lowRow = balancedLastAtMost(firstSlot, 0u, BALANCED_ROWS - 1u);
    balancedWindow[0] = lowRow;
    balancedWindow[1] = balancedLastAtMost(firstSlot + (BALANCED_LANES - 1u), lowRow, BALANCED_ROWS - 1u);
  }
  let windowLow = workgroupUniformLoad(&balancedWindow[0]);
  let windowHigh = workgroupUniformLoad(&balancedWindow[1]);
  let total = balancedStart(BALANCED_ROWS - 1u) + balancedCount(BALANCED_ROWS - 1u);
  if (slot >= total) {
    return BALANCED_NONE;
  }
  return balancedLastAtMost(slot, windowLow, windowHigh);
}
`;
}
