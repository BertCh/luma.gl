// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Shared WGSL helpers of the geographic-distribution kernels. @internal */
export const GEOGRAPHIC_DISTRIBUTION_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;
const TWO_PI: f32 = 6.28318530717959;
// Bit test: NaN and infinity have an all-ones exponent, and a float comparison may be folded away.
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn getQuietNaN(seed: u32) -> f32 { return bitcast<f32>(0x7fc00000u | (seed & 0u)); }
`;

/**
 * Squared distance under which a point is treated as coinciding with the Weiszfeld iterate and is
 * skipped for that step (the standard guard against division by zero). @internal
 */
export const WEISZFELD_COINCIDENT_DISTANCE_SQUARED = 1e-12;

/** Largest number of rows (`2^24 - 1`), so float32 row counts stay exact. @internal */
export const GEOGRAPHIC_DISTRIBUTION_MAXIMUM_ROWS = 2 ** 24 - 1;
