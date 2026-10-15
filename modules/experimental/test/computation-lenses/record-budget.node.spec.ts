// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, statSync} from 'node:fs';
import {join} from 'node:path';
import {expect, test} from 'vitest';

// Size budget of the recorded lens records (`docs/lens-records/<scenario>.json`, compact JSON with
// the lossless encoding of computation-lenses/record-encoding.ts). A record is shrunk honestly:
// key frames instead of every iteration, only the captures a view reads, docs-sized inputs. Never
// by dropping evidence a figure shows.
const RECORDS_DIRECTORY = new URL('../../../../docs/lens-records/', import.meta.url).pathname;
const RECORD_BUDGET_BYTES = 512 * 1024;
/** Backstop for all records together; the per-record budget is the real control (~300 elements). */
const TOTAL_BUDGET_BYTES = 64_000_000;
/** A listed record may vary this much between capture runs (adapter label, timestamps). */
const ALLOWLIST_TOLERANCE = 1.02;

/**
 * Records allowed over {@link RECORD_BUDGET_BYTES}: id → size when listed and why it cannot shrink
 * honestly. An entry fails once its record is within budget (remove it) or grows past its size.
 */
const OVER_BUDGET: Readonly<Record<string, {bytes: number; reason: string}>> = {
  'gpu-spatial-analysis.buffer-surface': {
    bytes: 539_356,
    reason:
      'The 574-node graph (ring compaction, noding, repair and face assembly) and its 227 bound WGSL kernel sources are the element itself: they are 93% of the record (263 KB and 241 KB after lossless encoding) and do not shrink with a smaller input; captures are 34 KB.'
  }
};

function listRecordSizes(): Map<string, number> {
  const sizes = new Map<string, number>();
  if (!existsSync(RECORDS_DIRECTORY)) return sizes;
  for (const file of readdirSync(RECORDS_DIRECTORY, {withFileTypes: true})) {
    if (!file.isFile() || !file.name.endsWith('.json')) continue;
    sizes.set(file.name.slice(0, -5), statSync(join(RECORDS_DIRECTORY, file.name)).size);
  }
  return sizes;
}

const sizes = listRecordSizes();

test(`each lens record is at most ${RECORD_BUDGET_BYTES} bytes unless allowlisted`, () => {
  const failures: string[] = [];
  for (const [id, bytes] of sizes) {
    const allowed = OVER_BUDGET[id];
    if (!allowed) {
      if (bytes > RECORD_BUDGET_BYTES)
        failures.push(`${id}: ${bytes} bytes exceeds the ${RECORD_BUDGET_BYTES}-byte budget`);
      continue;
    }
    if (bytes <= RECORD_BUDGET_BYTES)
      failures.push(`${id}: ${bytes} bytes is within budget; remove it from OVER_BUDGET`);
    else if (bytes > allowed.bytes * ALLOWLIST_TOLERANCE)
      failures.push(`${id}: ${bytes} bytes grew past its allowlisted ${allowed.bytes} bytes`);
  }
  expect(failures).toEqual([]);
});

test('every allowlisted record exists and gives a reason', () => {
  for (const [id, {bytes, reason}] of Object.entries(OVER_BUDGET)) {
    expect(sizes.has(id), `${id} has no recorded file; remove it from OVER_BUDGET`).toBe(true);
    expect(bytes).toBeGreaterThan(RECORD_BUDGET_BYTES);
    expect(reason.trim().length).toBeGreaterThan(0);
  }
});

test(`all lens records together stay within ${TOTAL_BUDGET_BYTES} bytes`, () => {
  const total = [...sizes.values()].reduce((sum, bytes) => sum + bytes, 0);
  expect(total).toBeLessThanOrEqual(TOTAL_BUDGET_BYTES);
});
