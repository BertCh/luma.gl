// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getInt64TimeWords} from '../time-window-filter/time-words';

/** Smallest instant the transition table can hold: Int64 minimum epoch milliseconds. */
const INT64_MINIMUM = -(2n ** 63n);

/** Sampling step of the transition search: one day, see {@link getGPUTimeZoneTransitionTable}. */
const SAMPLE_STEP_MILLISECONDS = 86_400_000;

/** Result of {@link getGPUTimeZoneTransitionTable}. */
export type GPUTimeZoneTransitionTable = {
  /**
   * Strictly increasing Int64 epoch-millisecond instants as `(low, high)` words, two `uint32` per
   * transition, ready for a `uint32x2` view. A zone without a transition in the range gets one
   * sentinel transition at the Int64 minimum so the table is never empty.
   */
  transitions: Uint32Array;
  /**
   * UTC offset in seconds. `offsets[0]` applies before the first transition and `offsets[i + 1]`
   * from transition `i` on, so the length is the transition count plus one.
   */
  offsets: Int32Array;
};

/** Parses `GMT`, `GMT+05:45` or `GMT-00:19:32` (ICU `longOffset`) to seconds. */
function parseLongOffset(name: string): number {
  if (name === 'GMT') {
    return 0;
  }
  const match = /^GMT([+\-−])(\d{2}):(\d{2})(?::(\d{2}))?$/.exec(name);
  if (!match) {
    throw new Error(`Unrecognized time zone offset "${name}"`);
  }
  const seconds = Number(match[2]) * 3600 + Number(match[3]) * 60 + Number(match[4] ?? 0);
  return match[1] === '+' ? seconds : -seconds;
}

/**
 * Builds the transition table that `GPUCalendarBuckets` reads through its `timeZone` input, from
 * the platform time zone database (`Intl.DateTimeFormat`). This is input preparation, not a
 * per-row computation: it runs once per zone and range on the CPU.
 *
 * Method: the offset is read at every UTC midnight-aligned sample of `range` (one day apart),
 * and each change between neighbouring samples is bisected to the exact first millisecond of the
 * new offset. One day cannot miss a transition pair because real zones keep at least several
 * weeks between consecutive transitions (the closest in the IANA database are Morocco's
 * Ramadan shifts and historical double-summer-time steps, still weeks apart). Offsets that go
 * out and back within one sample step would be missed, so do not use this for synthetic zones.
 *
 * Offset resolution: V8 formats the offset with seconds when the zone has them, for example
 * `GMT+05:41:16` for Asia/Kathmandu in 1900, so local mean time before standardization is exact
 * to the second. The result is as exact as the ICU data bundled with the runtime; it can differ
 * from another tz database release (a runtime older than the `tzdata` the oracle used may place
 * a rule change differently), and `Intl` rejects ranges outside the `Date` limits.
 *
 * @param timeZone IANA zone name such as `'America/New_York'`.
 * @param range Epoch-millisecond range `[start, end]`. Instants before `start` use the offset at
 * `start`, instants after the last transition use the offset in force at `end`.
 * @returns Transition words and offsets in seconds.
 * @throws If the zone is unknown, the range is not finite or reversed, or the offset text is
 * not recognized.
 */
export function getGPUTimeZoneTransitionTable(
  timeZone: string,
  range: {start: number; end: number}
): GPUTimeZoneTransitionTable {
  if (!Number.isFinite(range.start) || !Number.isFinite(range.end) || range.end < range.start) {
    throw new Error('Time zone range must be finite with start <= end');
  }
  let format: Intl.DateTimeFormat;
  try {
    format = new Intl.DateTimeFormat('en-US', {timeZone, timeZoneName: 'longOffset'});
  } catch {
    throw new Error(`Unknown time zone "${timeZone}"`);
  }
  const getOffsetSeconds = (instant: number): number => {
    const part = format.formatToParts(instant).find(entry => entry.type === 'timeZoneName');
    return parseLongOffset(part ? part.value : '');
  };

  const start = Math.floor(range.start);
  const end = Math.floor(range.end);
  const transitions: bigint[] = [];
  const offsets: number[] = [getOffsetSeconds(start)];
  let previousInstant = start;
  let previousOffset = offsets[0];
  const sample = (instant: number): void => {
    const offset = getOffsetSeconds(instant);
    if (offset === previousOffset) {
      previousInstant = instant;
      return;
    }
    // Invariant: the offset is previousOffset at `low` and different at `high`.
    let low = previousInstant;
    let high = instant;
    while (high - low > 1) {
      const middle = low + Math.floor((high - low) / 2);
      if (getOffsetSeconds(middle) === previousOffset) {
        low = middle;
      } else {
        high = middle;
      }
    }
    transitions.push(BigInt(high));
    offsets.push(getOffsetSeconds(high));
    previousOffset = offsets[offsets.length - 1];
    previousInstant = high;
    if (high < instant) {
      // A second change inside the same step would be missed; re-check the step end.
      sample(instant);
    }
  };
  for (
    let instant = start + SAMPLE_STEP_MILLISECONDS;
    instant < end;
    instant += SAMPLE_STEP_MILLISECONDS
  ) {
    sample(instant);
  }
  if (end > start) {
    sample(end);
  }
  if (transitions.length === 0) {
    transitions.push(INT64_MINIMUM);
    offsets.push(offsets[0]);
  }
  return {
    transitions: getInt64TimeWords(BigInt64Array.from(transitions)),
    offsets: Int32Array.from(offsets)
  };
}
