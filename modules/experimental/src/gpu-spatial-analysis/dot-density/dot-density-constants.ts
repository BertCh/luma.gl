// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PHILOX_PURPOSES} from '../../utils/philox-random';

/** Philox key word 1 of the per-slot Bernoulli remainder stream. */
export const DOT_DENSITY_REMAINDER_PURPOSE = PHILOX_PURPOSES.dotDensityRemainder.id;
/** Philox key word 1 of the candidate position stream. */
export const DOT_DENSITY_CANDIDATE_PURPOSE = PHILOX_PURPOSES.dotDensityCandidate.id;

/**
 * Largest dots per slot. Keeps the u32 prefix sum from wrapping: any slot clamped here forces the
 * overflow flag.
 */
export function getDotSlotCountLimit(slotCount: number, capacity: number): number {
  return Math.min(capacity + 1, Math.floor(0xffffffff / Math.max(slotCount, 1)), 2 ** 31);
}
