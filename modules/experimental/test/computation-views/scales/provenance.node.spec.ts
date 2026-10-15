// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  coarsenValues,
  contrastRatio,
  crispnessAttributes,
  crispnessDefs,
  getSelectionInk,
  PROVENANCE_HATCH_STROKE,
  PROVENANCE_LABEL,
  PROVENANCE_TEXTURE,
  provenanceBadge,
  provenanceMarkAttributes,
  provenancePatternDefs,
  VIEW_GROUND_COLOR
} from '../../../src/computation-views/encoding';
import {isAchromatic} from '../../../src/computation-views/scales';
import {
  findViewNodes,
  getViewNodeText,
  viewNodeToHtml
} from '../../../src/computation-views/view-node';

describe('provenance and crispness encodings', () => {
  test('lineage is texture, with explicit unknown outlines', () => {
    expect(PROVENANCE_TEXTURE).toEqual({
      readback: 'solid',
      replay: 'hatched',
      derived: 'hatched',
      metadata: 'outline',
      unknown: 'outline'
    });
    for (const provenance of ['readback', 'replay', 'derived', 'metadata', 'unknown'] as const) {
      const attributes = provenanceMarkAttributes(provenance, '#123456', 'test');
      expect(attributes['data-provenance']).toBe(provenance);
      expect(getViewNodeText(provenanceBadge(provenance))).toBe(PROVENANCE_LABEL[provenance]);
      expect(provenanceBadge(provenance).attributes['title']).toBeTruthy();
      expect(provenanceBadge(provenance).attributes['class']).toBe(
        `lens-provenance lens-provenance--${provenance}`
      );
    }
    expect(provenanceMarkAttributes('readback', '#123456', 'test')).toMatchObject({
      fill: '#123456'
    });
    expect(provenanceMarkAttributes('replay', '#123456', 'test')).toMatchObject({
      fill: 'url(#test-hatched)',
      stroke: '#123456',
      'stroke-width': '1'
    });
    expect(provenanceMarkAttributes('metadata', '#123456', 'test')).toMatchObject({
      fill: 'none',
      stroke: '#123456'
    });
    expect(provenanceMarkAttributes('unknown', '#123456', 'test')['stroke-dasharray']).toBe('3 2');
  });

  test.each([
    'light',
    'dark'
  ] as const)('hatches and selection strokes contrast on %s ground', ground => {
    expect(
      contrastRatio(PROVENANCE_HATCH_STROKE[ground], VIEW_GROUND_COLOR[ground])
    ).toBeGreaterThanOrEqual(3);
    for (const ink of Object.values(getSelectionInk(ground))) {
      expect(isAchromatic(ink)).toBe(true);
      expect(contrastRatio(ink, VIEW_GROUND_COLOR[ground])).toBeGreaterThanOrEqual(3);
    }
    const tree = provenancePatternDefs('example', ground);
    const html = viewNodeToHtml(tree);
    expect(html).toContain('<defs>');
    expect(html).toContain('id="example-hatched"');
    expect(html).toContain('patternUnits="userSpaceOnUse"');
    expect(html).toContain('patternTransform="rotate(45)"');
    expect(html).toContain(`stroke="${PROVENANCE_HATCH_STROKE[ground]}"`);
    expect(findViewNodes(tree, node => node.tag === 'pattern')).toHaveLength(1);
    expect(findViewNodes(tree, node => node.tag === 'line')).toHaveLength(1);
  });

  test('contrast uses linearized sRGB and is symmetric', () => {
    expect(contrastRatio('#000', '#fff')).toBeCloseTo(21);
    expect(contrastRatio('#777777', '#ffffff')).toBeCloseTo(4.478, 3);
    expect(contrastRatio('#777777', '#ffffff')).toBe(contrastRatio('#ffffff', '#777777'));
    expect(contrastRatio('#123456', '#123456')).toBe(1);
  });

  test('crispness never changes fill, saturation or alpha', () => {
    expect(crispnessAttributes('final', 'test')).toEqual({});
    for (const state of ['provisional', 'unconverged'] as const) {
      expect(crispnessAttributes(state, 'test')).toEqual({
        filter: `url(#test-${state})`,
        'data-provisional': state
      });
      expect(crispnessAttributes(state, 'test')).not.toHaveProperty('fill');
    }
    const tree = crispnessDefs('test');
    expect(
      findViewNodes(tree, node => node.tag === 'filter').map(node => node.attributes['id'])
    ).toEqual(['test-provisional', 'test-unconverged']);
    expect(
      findViewNodes(tree, node => node.tag === 'feGaussianBlur').map(
        node => node.attributes['stdDeviation']
      )
    ).toEqual([1.2, 2.4]);
  });

  test('coarsening uses significant figures and preserves final and missing values', () => {
    const values = new Float64Array([1234, 0.01234, -987.6, 0, NaN, Infinity]);
    expect(coarsenValues(values, 'provisional')).toEqual([1200, 0.012, -990, 0, NaN, Infinity]);
    expect(coarsenValues(values, 'unconverged')).toEqual([1000, 0.01, -1000, 0, NaN, Infinity]);
    expect(coarsenValues(values, 'final')).toEqual(Array.from(values));
    expect(values[0]).toBe(1234);
  });
});
