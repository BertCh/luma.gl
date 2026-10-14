// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GET_QUIET_NAN_SEEDED_WGSL, IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';

/** Shared WGSL helpers of the geographic-distribution kernels. @internal */
export const GEOGRAPHIC_DISTRIBUTION_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;
const TWO_PI: f32 = 6.28318530717959;
// Bit test: NaN and infinity have an all-ones exponent, and a float comparison may be folded away.
${IS_FINITE_VALUE_WGSL}
${GET_QUIET_NAN_SEEDED_WGSL}
`;

/**
 * Squared distance under which a point is treated as coinciding with the Weiszfeld iterate and is
 * skipped for that step (the standard guard against division by zero). @internal
 */
export const WEISZFELD_COINCIDENT_DISTANCE_SQUARED = 1e-12;

/** Largest number of rows (`2^24 - 1`), so float32 row counts stay exact. @internal */
export const GEOGRAPHIC_DISTRIBUTION_MAXIMUM_ROWS = 2 ** 24 - 1;
