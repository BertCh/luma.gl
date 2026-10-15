// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {STANDARD_GUARDS} from '../../../src/computation-views/guards/standard-guards';
import {
  evaluateGuards,
  getBlockingVerdict,
  createGuardContext
} from '../../../src/computation-views/guards/guards';
import type {GuardContext} from '../../../src/computation-views/guards/guards';
import {
  renderVerdictChip,
  renderBlockedGlyph
} from '../../../src/computation-views/views/verdict-chip';
import {getViewNodeText, viewNodeToHtml} from '../../../src/computation-views/view-node';
import {makeScanRecord} from '../fixtures/scan-record';

const passing: GuardContext['metadata'][] = [
  {coordinateModel: 'projected'},
  {mappedVariableExtent: 'extensive', denominator: 'population'},
  {sumBefore: 100, sumAfter: 100.00001},
  {pCorrection: 'fdr'},
  {significanceAvailable: true},
  {weightsStandardisation: 'row'}
];
const failing: GuardContext['metadata'][] = [
  {coordinateModel: 'geographic'},
  {mappedVariableExtent: 'extensive'},
  {sumBefore: 100, sumAfter: 101},
  {pCorrection: 'none'},
  {significanceAvailable: false},
  {weightsStandardisation: 'binary'}
];
const missingKeys = [
  'coordinateModel',
  'mappedVariableExtent',
  'sumBefore',
  'pCorrection',
  'significanceAvailable',
  'weightsStandardisation'
];

describe('standard guards', () => {
  STANDARD_GUARDS.forEach((guard, index) => {
    test(`${guard.id}: pass, fail and unknown`, () => {
      expect(guard.evaluate({metadata: passing[index]}).status).toBe('pass');
      expect(guard.evaluate({metadata: failing[index]}).status).toBe('fail');
      expect(guard.evaluate({metadata: {}})).toMatchObject({
        status: 'unknown',
        reason: expect.stringContaining(missingKeys[index])
      });
    });
  });
  test('unknown blocks and renders distinctly from fail', () => {
    const verdicts = evaluateGuards(STANDARD_GUARDS, {metadata: {}});
    expect(getBlockingVerdict(verdicts, 'support-ring')).toBe(verdicts[0]);
    expect(getBlockingVerdict(verdicts, 'unrelated')).toBeUndefined();
    expect(getViewNodeText(renderVerdictChip(verdicts[0]))).toContain(
      '? Coordinate model: unknown —'
    );
    expect(getViewNodeText(renderBlockedGlyph(verdicts[0], 'Support ring'))).toContain(
      'Support ring withheld'
    );
    expect(viewNodeToHtml(renderVerdictChip(verdicts[0]))).toContain('role="note"');
    const passed = evaluateGuards([STANDARD_GUARDS[0]], {metadata: passing[0]});
    expect(getBlockingVerdict(passed, 'support-ring')).toBeUndefined();
    expect(getViewNodeText(renderVerdictChip(passed[0]))).toBe('✓ Coordinate model');
    expect(
      getViewNodeText(
        renderVerdictChip(evaluateGuards([STANDARD_GUARDS[0]], {metadata: failing[0]})[0])
      )
    ).toContain('✕ Coordinate model: metric glyphs');
    // Geodesic measures are metric on lon/lat (M8 exception).
    expect(
      STANDARD_GUARDS[0].evaluate({metadata: {coordinateModel: 'geographic', metric: 'geodesic'}})
        .status
    ).toBe('pass');
  });
  test('metadata merges explicitly, and conservation handles zero and tolerance overrides', () => {
    const context = createGuardContext(makeScanRecord(), [
      {
        id: 'a',
        metadata: {coordinateModel: 'geographic'},
        variableKinds: {count: {level: 'ratio', extent: 'extensive'}}
      },
      {id: 'b', metadata: {coordinateModel: 'projected'}}
    ]);
    expect(context.metadata['coordinateModel']).toBe('projected');
    expect(context.variableKinds?.['count']?.extent).toBe('extensive');
    const guard = STANDARD_GUARDS[2];
    expect(guard.evaluate({metadata: {sumBefore: 0, sumAfter: 0}}).status).toBe('pass');
    expect(guard.evaluate({metadata: {sumBefore: 0, sumAfter: 0.0001}}).status).toBe('fail');
    expect(
      guard.evaluate({metadata: {sumBefore: 100, sumAfter: 101, conservationTolerance: 0.02}})
        .status
    ).toBe('pass');
    expect(guard.evaluate({metadata: {sumBefore: 100}}).reason).toContain('sumAfter');
    expect(
      STANDARD_GUARDS[1].evaluate({metadata: {mappedVariableExtent: 'intensive'}}).status
    ).toBe('pass');
  });
});
