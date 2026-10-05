// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** WGSL finite-float test shared by the neighbor-search kernels. @internal */
export const NEIGHBOR_SEARCH_FLOAT_WGSL = /* wgsl */ `
fn isFiniteFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
`;
