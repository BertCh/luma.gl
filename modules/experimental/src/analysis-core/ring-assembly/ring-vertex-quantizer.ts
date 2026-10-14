// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';

/**
 * The vertex cell size grows with the frame extent so that no cell index exceeds `2^22` in
 * magnitude (f32 holds integers exactly up to `2^23`): `cellSize = max(tolerance, extent * 2^-22)`
 * times {@link GPU_RING_CELL_SLACK}. A cell no smaller than `vertexTolerance` makes the 3 x 3 cell
 * lookup complete, because candidates are still compared by their true coordinate distance
 * against `vertexTolerance`.
 */
export const GPU_RING_CELL_RESOLUTION_BITS = 22;
/** Cells are stored as `u32` biased by this value so that negative offsets stay unsigned. */
export const GPU_RING_CELL_BIAS = 8388608;
/** Cell size margin that absorbs f32 rounding of `(v - origin) / cellSize` near cell edges. */
export const GPU_RING_CELL_SLACK = 1.0625;

/** CPU mirror of the WGSL vertex quantizer used by `GPUSegmentRingAssembly`. */
export type GPURingVertexCell = {
  /** Biased `u32` cell index (`GPU_RING_CELL_BIAS` for the origin cell). */
  cell: number;
  /** True for a non-finite coordinate (status `overflow`); such vertices are never clamped into a real cell. */
  overflow: boolean;
};

/**
 * Cell size of a frame: `max(tolerance, extent * 2^-22)` times the rounding slack, where `extent`
 * is the largest `|x - originX|` or `|y - originY|` of any endpoint.
 */
export function getRingCellSize(tolerance: number, extent: number): number {
  return Math.fround(
    GPU_RING_CELL_SLACK *
      Math.max(
        Math.fround(tolerance),
        Math.fround(Math.fround(extent) * 2 ** -GPU_RING_CELL_RESOLUTION_BITS),
        Math.fround(1e-30)
      )
  );
}

/**
 * Quantizes one coordinate relative to a local origin exactly as the ring assembly kernels do
 * (f32 arithmetic): `floor((value - origin) / cellSize)`, biased into `u32`. Relative quantization
 * with an extent-adaptive cell keeps projected coordinates (UTM, Web Mercator) from saturating or
 * colliding. Non-finite input reports overflow.
 */
export function quantizeRingVertexCoordinate(
  value: number,
  origin: number,
  tolerance: number,
  extent: number = Math.abs(value - origin)
): GPURingVertexCell {
  const offset = Math.fround(Math.fround(value) - Math.fround(origin));
  if (!Number.isFinite(offset)) {
    return {cell: GPU_RING_CELL_BIAS, overflow: true};
  }
  const cell = Math.floor(
    Math.fround(offset * Math.fround(1 / getRingCellSize(tolerance, extent)))
  );
  return {cell: cell + GPU_RING_CELL_BIAS, overflow: false};
}

/**
 * WGSL helpers shared by the hash, cancel and match kernels. They read the origin (the start
 * vertex of segment row 0) from the `endpoints` binding and the frame extent (f32 bits) from the
 * `counters` binding of the kernel they are declared in. `atomicCounters` says whether that kernel
 * binds `counters` as `array<atomic<u32>>`.
 */
export function getRingQuantizerWGSL(
  toleranceLiteral: string,
  atomicCounters: boolean,
  extentIndex: number
): string {
  const load = atomicCounters
    ? `atomicLoad(&counters[countersOffset + ${extentIndex}u])`
    : `counters[countersOffset + ${extentIndex}u]`;
  return `${IS_FINITE_VALUE_WGSL}
const TOLERANCE_CELL: f32 = ${toleranceLiteral};
const CELL_SLACK: f32 = ${GPU_RING_CELL_SLACK};
const CELL_BIAS: f32 = ${GPU_RING_CELL_BIAS}.0;
fn ringOrigin() -> vec2f { return vec2f(endpoints[endpointsOffset], endpoints[endpointsOffset + 1u]); }
// Largest |x - originX| or |y - originY| of any endpoint, measured on the GPU.
fn ringFrameExtent() -> f32 { return bitcast<f32>(${load}); }
fn ringInverseCell() -> f32 {
  // The 1e-30 floor keeps a zero extent with the derived (zero) tolerance finite.
  return 1.0 / (CELL_SLACK * max(max(TOLERANCE_CELL, ringFrameExtent() * ${2 ** -GPU_RING_CELL_RESOLUTION_BITS}), 1.0e-30));
}
fn ringQuantize(value: f32, origin: f32) -> u32 {
  let offset = value - origin;
  if (!isFiniteValue(offset)) { return u32(CELL_BIAS); }
  return u32(floor(offset * ringInverseCell()) + CELL_BIAS);
}
fn ringHash(qx: u32, qy: u32, group: u32) -> u32 {
  var h = qx * 0x9E3779B1u;
  h = h ^ (h >> 15u);
  h = (h + qy * 0x85EBCA77u) * 0xC2B2AE3Du;
  h = h ^ (h >> 13u);
  h = (h + group * 0x27D4EB2Fu) * 0x165667B1u;
  h = h ^ (h >> 16u);
  return select(h, 0xfffffffeu, h == 0xffffffffu);
}`;
}
