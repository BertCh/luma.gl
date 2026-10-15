// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {
  defineLadderLensElement,
  LADDER_LENS_ATTRIBUTES
} from '../../../src/computation-views/hosts/ladder-lens-element';
import {toRecordedLensFile} from '../../../src/computation-lenses/registry';
import {makeV7Record} from '../fixtures/v7-lens-record';

const recorded = () =>
  toRecordedLensFile(makeV7Record(), {
    scenarioId: 'gpu-core.double',
    recordedAt: '2026-10-14T00:00:00Z',
    adapterLabel: 'Fixture GPU'
  });

describe('ladder-lens DOM host', () => {
  test('defines idempotently, preserves the static twin and loads only on Explore', async () => {
    const loadRecord = vi.fn(async () => recorded());
    const tagName = 'ladder-lens-v7-test';
    const options = {tagName, loadRecord};
    defineLadderLensElement(options);
    const elementConstructor = customElements.get(tagName);
    defineLadderLensElement(options);
    expect(customElements.get(tagName)).toBe(elementConstructor);
    expect(LADDER_LENS_ATTRIBUTES).toEqual([
      'scenario',
      'rung',
      'bind',
      'datum',
      'iteration',
      'caption',
      'view'
    ]);
    const element = document.createElement(tagName);
    element.setAttribute('scenario', 'gpu-core.double');
    element.setAttribute('rung', 'table');
    const twin = document.createElement('figure');
    twin.textContent = 'Static recorded twin: 16 rows';
    element.append(twin);
    document.body.append(element);
    try {
      expect(element.firstElementChild).toBe(twin);
      expect(twin.hidden).toBe(false);
      expect(loadRecord).not.toHaveBeenCalled();
      expect(element.querySelector('.ladder-lens-frame')).toBeNull();
      element.querySelector<HTMLButtonElement>('[data-ladder="explore"]')!.click();
      await vi.waitFor(() => expect(element.querySelector('.ladder-lens-frame')).not.toBeNull());
      expect(loadRecord).toHaveBeenCalledExactlyOnceWith('gpu-core.double');
      expect(element.contains(twin)).toBe(true);
      expect(twin.hidden).toBe(true);
      expect(twin.textContent).toContain('16 rows');
      element.setAttribute('datum', '5');
      expect(element.querySelector('.lens-axis-header')?.textContent).toContain('row 5 pinned');
      element.removeAttribute('datum');
      expect(element.querySelector('.lens-axis-header')?.textContent).not.toContain('row 5 pinned');
      expect(element.querySelectorAll('[tabindex="0"]')).toHaveLength(1);
    } finally {
      element.remove();
    }
    expect(twin.hidden).toBe(false);
    expect(element.querySelector('.ladder-lens-controls')).toBeNull();
  });

  test('a late recorded load cannot remount a disconnected element', async () => {
    let finish: (value: unknown) => void = () => {};
    const loadRecord = vi.fn(
      () =>
        new Promise<unknown>(resolve => {
          finish = resolve;
        })
    );
    defineLadderLensElement({tagName: 'ladder-lens-v7-late', loadRecord});
    const element = document.createElement('ladder-lens-v7-late');
    element.setAttribute('scenario', 'gpu-core.double');
    const twin = document.createElement('figure');
    element.append(twin);
    document.body.append(element);
    element.querySelector<HTMLButtonElement>('[data-ladder="explore"]')!.click();
    element.remove();
    finish(recorded());
    await Promise.resolve();
    await Promise.resolve();
    expect(element.querySelector('.ladder-lens-frame')).toBeNull();
    expect(twin.hidden).toBe(false);
  });
});
