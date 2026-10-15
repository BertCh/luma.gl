// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  formatKeyNumber,
  generateLensDescription,
  describeLensView
} from '../../../src/computation-views/twin/description';
import type {DescriptionFacts, MarkProvenance} from '../../../src/computation-views/types';
import {TABLE_VIEW} from '../../../src/computation-views/views/table-view';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {makeScanRecord} from '../fixtures/scan-record';

const facts: DescriptionFacts = {
  subject: 'Scan',
  keyNumber: {label: 'nodes', value: 1234},
  facts: ['Three passes', 'Two buffers'],
  caveats: ['Timing changes encoding'],
  provenance: 'readback',
  adapterLabel: 'Apple M3'
};

test('one fact source feeds alt, live and Markdown', () => {
  const description = generateLensDescription(facts, {caption: 'A scan'});
  expect(description.alt).toBe(
    'Scan. nodes: 1,234. values read back from the GPU run. Three passes.'
  );
  expect(description.live).toBe('nodes 1,234; Three passes.');
  expect(description.llms).toContain('**A scan**\nScan');
  for (const text of [
    'nodes: 1,234',
    'Three passes',
    'Two buffers',
    'Timing changes encoding',
    'Apple M3'
  ])
    expect(description.llms).toContain(text);
  expect(formatKeyNumber({label: 'duration', value: 1.23456, unit: 'ms'})).toBe('1.235 ms');
  expect(formatKeyNumber({label: 'nodes', value: 0})).toBe('0 nodes');
  expect(formatKeyNumber({label: 'status', value: 'pending'})).toBe('pending status');
  expect(
    generateLensDescription({...facts, subject: 'long '.repeat(100)}).alt.length
  ).toBeLessThanOrEqual(250);
});

test('provenance phrases and disabled reasons are explicit', () => {
  const phrases: Record<MarkProvenance, string> = {
    readback: 'values read back from the GPU run',
    replay: 'values recomputed by a GPU replay',
    derived: 'derived values',
    metadata: 'structure metadata only',
    unknown: 'provenance unknown'
  };
  for (const [provenance, phrase] of Object.entries(phrases))
    expect(
      generateLensDescription({...facts, provenance: provenance as MarkProvenance}).alt
    ).toContain(phrase);
  const description = generateLensDescription(facts, {
    rung: 'kernel',
    disabledReason: 'source absent'
  });
  for (const text of Object.values(description)) expect(text).toContain('source absent');
  const context = createViewContext(makeScanRecord());
  expect(describeLensView(TABLE_VIEW, TABLE_VIEW.build(context), context)).toEqual(
    generateLensDescription(TABLE_VIEW.describe(TABLE_VIEW.build(context), context))
  );
});
