// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Decoding of the JSON fixtures written by the `test/oracles/*.py` generators (quality roadmap
 * rule 22). JSON has no NaN, infinities or negative zero, so the generators write them as
 * `null` (NaN), `"Infinity"`, `"-Infinity"` and `"-0"`. GPU specs import the JSON (no `node:fs`)
 * and pass it through {@link decodeOracleFixture} to get plain numbers back.
 */

/** Returns `fixture` with `null`, `"Infinity"`, `"-Infinity"` and `"-0"` replaced by numbers. */
export function decodeOracleFixture<T>(fixture: unknown): T {
  return decodeValue(fixture) as T;
}

function decodeValue(value: unknown): unknown {
  if (value === null) {
    return NaN;
  }
  if (value === 'Infinity') {
    return Infinity;
  }
  if (value === '-Infinity') {
    return -Infinity;
  }
  if (value === '-0') {
    return -0;
  }
  if (Array.isArray(value)) {
    return value.map(decodeValue);
  }
  if (typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, entry]) => [
        key,
        decodeValue(entry)
      ])
    );
  }
  return value;
}
