// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import * as showcase from '../../../../../examples/deck/spatial-analysis-showcase/src/engine/ramps';
import {
  directionFor,
  sampleRamp,
  getClassColors,
  getClassPaletteHex,
  makeClassTable,
  getClassTableLegend
} from '../../../src/computation-views/scales';

describe('copied cartography and charts', () => {
  test('ramp directions and samples remain equal to the showcase', () => {
    expect(directionFor('dark', 'magma')).toEqual(showcase.directionFor('dark', 'magma'));
    for (const ground of ['light', 'dark'] as const) {
      for (const ramp of ['magma', 'rdbu', 'blues', 'romao'] as const) {
        const {reverse} = directionFor(ground, ramp);
        expect({reverse}).toEqual(showcase.directionFor(ground, ramp));
        for (const value of [0, 0.125, 0.5, 0.875, 1])
          expect(sampleRamp(ramp, value, reverse)).toEqual(
            showcase.sampleRamp(ramp, value, reverse)
          );
        expect(getClassColors(ramp, 5, reverse)).toEqual(showcase.getClassColors(ramp, 5, reverse));
      }
    }
  });

  test('published tables and local legend adapter remain usable', () => {
    expect(getClassPaletteHex('Blues', 3)).toEqual(['#deebf7', '#9ecae1', '#3182bd']);
    const table = makeClassTable({breaks: [1, 2], scheme: 'Blues', extent: [0, 3]});
    expect(getClassTableLegend(table, {title: 'Values'})).toMatchObject({
      kind: 'classes',
      title: 'Values',
      table
    });
  });

  test('chart entry imports with DOM globals guarded against access', async () => {
    const globals = ['document', 'window', 'customElements', 'HTMLElement', 'Node', 'DOMParser'];
    const descriptors = new Map(
      globals.map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)])
    );
    for (const name of globals)
      Object.defineProperty(globalThis, name, {
        configurable: true,
        get: () => {
          throw new Error(`Import accessed ${name}`);
        }
      });
    let charts: typeof import('../../../src/computation-views/charts');
    try {
      vi.resetModules();
      charts = await import('../../../src/computation-views/charts');
    } finally {
      for (const name of globals) {
        const descriptor = descriptors.get(name);
        if (descriptor) Object.defineProperty(globalThis, name, descriptor);
        else Reflect.deleteProperty(globalThis, name);
      }
    }
    expect(typeof charts.renderChart).toBe('function');
    expect(typeof charts.ensureChartStyles).toBe('function');
    expect(charts.CHART_CSS).toContain('.chart');
  });

  test('styles install once per owner document', async () => {
    const {ensureChartStyles, CHART_CSS} = await import('../../../src/computation-views/charts');
    const createDocument = () => {
      const styles: {textContent: string; attributes: Record<string, string>}[] = [];
      return {
        styles,
        document: {
          querySelector: () => styles[0],
          createElement: () => ({
            textContent: '',
            attributes: {} as Record<string, string>,
            setAttribute(name: string, value: string) {
              this.attributes[name] = value;
            }
          }),
          head: {appendChild: (style: (typeof styles)[number]) => styles.push(style)}
        }
      };
    };
    for (const owner of [createDocument(), createDocument()]) {
      ensureChartStyles(owner.document as unknown as Document);
      ensureChartStyles(owner.document as unknown as Document);
      expect(owner.styles).toHaveLength(1);
      expect(owner.styles[0].textContent).toBe(CHART_CSS);
      expect(owner.styles[0].attributes).toHaveProperty('data-luma-chart-styles', '');
    }
  });
});
