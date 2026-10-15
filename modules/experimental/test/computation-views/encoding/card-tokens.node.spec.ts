// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {describe, expect, test} from 'vitest';
import {
  CARD_COLORS,
  getCardPaletteColors,
  getCardTokenDeclarations,
  getCardTokenCss,
  getCardPaletteForVariable,
  getCardMarkAttributes
} from '../../../src/computation-views/encoding/card-tokens';
import {contrastRatio} from '../../../src/computation-views/encoding/provenance';
import {LENS_CSS} from '../../../src/computation-views/hosts/lens-styles';
import {createViewContext} from '../../../src/computation-views/registry/view-registry';
import {
  createDivergingScaleSpec,
  ScaleRegistry
} from '../../../src/computation-views/scales/scale-registry';
import {makeY2Points} from '../fixtures/y2-card-records';

function block(source: string, selector: string): string {
  const start = source.indexOf('{', source.indexOf(selector));
  let depth = 1;
  let end = start + 1;
  while (depth && end < source.length) {
    if (source[end] === '{') depth++;
    if (source[end] === '}') depth--;
    end++;
  }
  return source.slice(start + 1, end - 1);
}

function declarations(source: string): string[] {
  return [...source.matchAll(/(--cv-[\w-]+)\s*:\s*([^;{}]+);/g)]
    .map(match => `${match[1]}: ${match[2].trim().replace(/\s+/g, ' ')}`)
    .sort();
}

describe('card tokens', () => {
  test('samples the exact CARD-DESIGN palette literals on both grounds', () => {
    const expected = {
      seq: {
        light: '#c6dbef #8abfdd #4a98c9 #1c6bb0 #083d7f',
        dark: '#1764ab #4695c8 #8abfdd #c8ddf0 #edf5fc'
      },
      div: {
        light: '#053061 #6bacd1 #f7f7f7 #e58368 #67001f',
        dark: '#9eb0ff #286886 #190c09 #7c331d #ffadad'
      },
      time: {
        light: '#dadaeb #adacd2 #807dba #5f3c99 #3f007d',
        dark: '#6a51a3 #8f8cc1 #bcbddc #e5e4f0 #fcfbfd'
      }
    };
    for (const palette of ['seq', 'div', 'time'] as const)
      for (const ground of ['light', 'dark'] as const) {
        expect(getCardPaletteColors(palette, ground)).toEqual(expected[palette][ground].split(' '));
      }
  });
  test('host CSS copies are byte-identical and contain exactly the library declarations', () => {
    const website = readFileSync('website/src/css/lens-card.css', 'utf8');
    const showcase = readFileSync(
      'examples/deck/spatial-analysis-showcase/src/docs/lens-card.css',
      'utf8'
    );
    expect(website).toBe(showcase);
    for (const source of [website, getCardTokenCss()]) {
      expect(declarations(block(source, ':root {'))).toEqual(
        declarations(getCardTokenDeclarations('light').join('\n'))
      );
      for (const selector of [":root[data-theme='dark']", ':root:not([data-theme])']) {
        expect(declarations(block(source, selector))).toEqual(
          declarations(getCardTokenDeclarations('dark').join('\n'))
        );
      }
    }
    expect(getCardTokenCss().match(/:root/g)).toHaveLength(3);
    expect(LENS_CSS.startsWith(getCardTokenCss())).toBe(true);
    expect(LENS_CSS).toContain(":root[data-theme='dark'] .ladder-lens-frame");
    expect(LENS_CSS).toContain(':root:not([data-theme]) .lens-twin');
    expect(LENS_CSS).toContain('.lens-twin { background: var(--cv-paper); }');
  });
  test('role marks meet graphical contrast on both grounds', () => {
    for (const ground of ['light', 'dark'] as const)
      for (const role of ['input', 'output', 'highlight'] as const) {
        expect(
          contrastRatio(CARD_COLORS[ground][role], CARD_COLORS[ground]['paper'])
        ).toBeGreaterThanOrEqual(3);
      }
    expect(getCardMarkAttributes('highlight', 'point')).toMatchObject({
      r: '4.5',
      stroke: '#ffffff',
      'stroke-width': '2'
    });
    expect(getCardMarkAttributes('input', 'cell')).toMatchObject({'fill-opacity': '0.35'});
  });
  test('uses nominal, explicit midpoint, iteration, then sequential metadata without sign inference', () => {
    const record = makeY2Points([-2, 0, 2]);
    const capture = record.captures!.snapshots.at(-1)!;
    const context = createViewContext(record);
    expect(getCardPaletteForVariable(context, 'values', capture)).toBe('seq');
    expect(
      getCardPaletteForVariable(context, 'values', {...capture, meaning: 'iteration stamp'})
    ).toBe('time');
    expect(
      getCardPaletteForVariable(
        {...context, packs: [{id: 'nominal', variableKinds: {values: {level: 'nominal'}}}]},
        'values',
        capture
      )
    ).toBe('nom');
    const scales = new ScaleRegistry({runId: record.runId});
    scales.register(
      createDivergingScaleSpec({
        id: 'signed',
        variable: 'values',
        variableKind: {level: 'interval'},
        midpoint: 0,
        ground: 'light',
        noDataColor: '#d9d9d9',
        finalStateValues: capture.values
      })
    );
    expect(getCardPaletteForVariable({...context, scales}, 'values', capture)).toBe('div');
  });
});
