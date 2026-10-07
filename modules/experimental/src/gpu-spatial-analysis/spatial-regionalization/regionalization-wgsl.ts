// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL helpers shared by the regionalization contributors. Every helper expects bindings named
 * `offsets` and `neighbors` (a CSR spatial-weights matrix) declared by `createWGSLKernelNode`.
 *
 * @internal
 */
export const REGIONALIZATION_CSR_WGSL = /* wgsl */ `
// Slot of row \`row\` whose neighbor is \`needle\`, or NONE. Rows are strictly ascending.
fn findSlot(row: u32, needle: u32) -> u32 {
  let rowEnd = offsets[offsetsOffset + row + 1u];
  var low = offsets[offsetsOffset + row];
  var high = rowEnd;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (neighbors[neighborsOffset + middle] < needle) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < rowEnd && neighbors[neighborsOffset + low] == needle) {
    return low;
  }
  return NONE;
}

// First row whose slot range ends after \`slot\` (the row that owns the slot).
fn getRowOfSlot(slot: u32) -> u32 {
  var low = 0u;
  var high = ROWS - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (offsets[offsetsOffset + middle + 1u] > slot) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  return low;
}
`;

/** Sentinel slot or ID shared by the regionalization kernels. @internal */
export const REGIONALIZATION_NONE_WGSL = 'const NONE: u32 = 0xffffffffu;';
