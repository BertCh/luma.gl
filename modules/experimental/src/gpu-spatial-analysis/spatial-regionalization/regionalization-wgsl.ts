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

/**
 * Workgroup-wide lexicographic argmin over `(score, key, value)` tuples for cooperative
 * regionalization kernels. The order is total on distinct tuples, so the winner does not depend
 * on the reduction tree. `reduceArgmin` contains barriers: call it from uniform control flow
 * with every invocation of the workgroup. It returns the winning `value` to all invocations.
 *
 * @internal
 */
export function getWorkgroupArgminWGSL(workgroupSize: number): string {
  return /* wgsl */ `
var<workgroup> argminScores: array<f32, ${workgroupSize}>;
var<workgroup> argminKeys: array<u32, ${workgroupSize}>;
var<workgroup> argminValues: array<u32, ${workgroupSize}>;

fn isArgminBetter(score: f32, key: u32, value: u32, bestScore: f32, bestKey: u32, bestValue: u32) -> bool {
  return score < bestScore || (score == bestScore && (key < bestKey || (key == bestKey && value < bestValue)));
}

fn reduceArgmin(lane: u32, score: f32, key: u32, value: u32) -> u32 {
  argminScores[lane] = score;
  argminKeys[lane] = key;
  argminValues[lane] = value;
  workgroupBarrier();
  for (var stride = ${workgroupSize / 2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      let other = lane + stride;
      if (isArgminBetter(argminScores[other], argminKeys[other], argminValues[other],
          argminScores[lane], argminKeys[lane], argminValues[lane])) {
        argminScores[lane] = argminScores[other];
        argminKeys[lane] = argminKeys[other];
        argminValues[lane] = argminValues[other];
      }
    }
    workgroupBarrier();
  }
  let winner = workgroupUniformLoad(&argminValues[0]);
  // Keep the next call's writes from racing this read of slot 0.
  workgroupBarrier();
  return winner;
}
`;
}
