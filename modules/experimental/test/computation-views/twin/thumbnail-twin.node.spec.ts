// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  buildLensTwin,
  renderLensTwinHtml,
  validateLensTwin
} from '../../../src/computation-views/twin/static-twin';
import {toRecordedLensFile} from '../../../src/computation-lenses';
import {LENS_RUNG_IDS} from '../../../src/computation-views/types';
import {makeV8ResultRecord} from '../fixtures/v8-result-record';
import {makeScanRecord} from '../fixtures/scan-record';

function recording(record = makeV8ResultRecord()) {
  return toRecordedLensFile(record, {
    scenarioId: 'gpu-core.geometry',
    recordedAt: '2026-10-14T00:00:00.000Z',
    adapterLabel: 'Fixture adapter'
  });
}

describe('thumbnail twins', () => {
  test('width 160, short caption, mandatory key number and omitted receipt/formula', () => {
    const recordFile = recording();
    const directive = {scenario: recordFile.scenarioId, rung: 'result'};
    const full = buildLensTwin(recordFile, directive);
    const thumbnail = buildLensTwin(recordFile, directive, {size: 'thumbnail'});
    expect(thumbnail).toMatchObject({
      size: 'thumbnail',
      figure: {width: 160, height: 96},
      keyNumber: {label: 'polygons', value: '2'}
    });
    expect(thumbnail).not.toHaveProperty('receipt');
    expect(thumbnail).not.toHaveProperty('formula');
    expect(thumbnail.caption.length).toBeLessThanOrEqual(80);
    expect(validateLensTwin(thumbnail)).toEqual([]);
    expect(full.size).toBe('full');
    expect(full.figure.width).toBe(640);
    expect(full.figure.html).not.toEqual(thumbnail.figure.html);
  });
  test('every rung still produces a valid thumbnail including fallbacks', () => {
    const recordFile = recording(makeScanRecord());
    for (const rung of LENS_RUNG_IDS) {
      const twin = buildLensTwin(
        recordFile,
        {scenario: recordFile.scenarioId, rung},
        {size: 'thumbnail'}
      );
      expect(validateLensTwin(twin), rung).toEqual([]);
      expect(twin.figure.width).toBe(160);
      expect(twin).not.toHaveProperty('receipt');
      expect(twin).not.toHaveProperty('formula');
    }
    const full = buildLensTwin(recordFile, {scenario: recordFile.scenarioId, rung: 'table'});
    expect(full.receipt).toBeDefined();
    expect(full.formula).toBeDefined();
  });
  test('extracts source and prints escaped code directly after the result caption', () => {
    const recordFile = recording();
    const example = '// #region example\n  const result = input < 2;\n// #endregion example';
    const twin = buildLensTwin(
      recordFile,
      {scenario: recordFile.scenarioId, rung: 'result'},
      {example}
    );
    expect(twin.example).toEqual({code: 'const result = input < 2;', language: 'ts', lineCount: 1});
    const html = renderLensTwinHtml(twin);
    expect(html).toContain(
      '</figcaption><pre class="lens-example"><code class="language-ts">const result = input &lt; 2;</code></pre>'
    );
    expect(renderLensTwinHtml({...twin, rung: 'table'})).not.toContain('lens-example');
    expect(() =>
      buildLensTwin(
        recordFile,
        {scenario: recordFile.scenarioId, rung: 'result'},
        {example: 'no region'}
      )
    ).toThrow('Missing');
  });
  test('caps an unusually long thumbnail caption', () => {
    const twin = buildLensTwin(
      recording(),
      {scenario: 'gpu-core.geometry', rung: 'result', caption: 'A'.repeat(120)},
      {size: 'thumbnail'}
    );
    expect(twin.caption.length).toBeLessThanOrEqual(80);
  });
});
