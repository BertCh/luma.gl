// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {attachRovingTabindex} from '../../../src/computation-views/hosts/roving-tabindex';

// A minimal event surface keeps these focus-contract tests runnable without a browser or DOM library.
function createFocusSurface() {
  const ownerDocument = {activeElement: null as unknown};
  const listeners = new Map<string, (event: unknown) => void>();
  const items = Array.from({length: 3}, () => {
    const attributes = new Map<string, string>();
    const classes = new Set<string>();
    const item = {
      get tabIndex() {
        return Number(attributes.get('tabindex') ?? -1);
      },
      set tabIndex(value: number) {
        attributes.set('tabindex', String(value));
      },
      getAttribute: (name: string) => attributes.get(name) ?? null,
      setAttribute: (name: string, value: string) => {
        attributes.set(name, value);
      },
      removeAttribute: (name: string) => {
        attributes.delete(name);
      },
      classList: {
        contains: (name: string) => classes.has(name),
        add: (name: string) => {
          classes.add(name);
        },
        toggle: (name: string, enabled: boolean) => {
          if (enabled) classes.add(name);
          else classes.delete(name);
        }
      },
      contains: (target: unknown) => item === target,
      focus: vi.fn(() => {
        ownerDocument.activeElement = item;
        listeners.get('focusin')?.({});
      }),
      scrollIntoView: vi.fn()
    };
    return item;
  });
  const root = {
    ownerDocument,
    querySelectorAll: () => items,
    addEventListener: (name: string, listener: (event: unknown) => void) => {
      listeners.set(name, listener);
    },
    removeEventListener: (name: string) => {
      listeners.delete(name);
    }
  };
  const press = (key: string) => {
    const event = {key, preventDefault: vi.fn()};
    listeners.get('keydown')?.(event);
    return event;
  };
  return {root: root as unknown as HTMLElement, items, press, ownerDocument};
}

describe('roving tabindex', () => {
  test('has one tab stop, moves focus, marks current and scrolls it into view', () => {
    const {root, items, press, ownerDocument} = createFocusSurface();
    const onNavigate = vi.fn();
    const detach = attachRovingTabindex(root, {itemSelector: '[data-key]', onNavigate});
    expect(items.map(item => item.tabIndex)).toEqual([0, -1, -1]);
    const event = press('ArrowRight');
    expect(event.preventDefault).toHaveBeenCalledOnce();
    expect(onNavigate).toHaveBeenCalledWith('ArrowRight', event);
    expect(items.map(item => item.tabIndex)).toEqual([-1, 0, -1]);
    expect(ownerDocument.activeElement).toBe(items[1]);
    expect(items[1].getAttribute('aria-current')).toBe('true');
    expect(items[0].getAttribute('aria-current')).toBeNull();
    expect(items[1].classList.contains('lens-focus-visible')).toBe(true);
    expect(items[1].scrollIntoView).toHaveBeenCalledWith({block: 'nearest'});
    expect(press('Tab').preventDefault).not.toHaveBeenCalled();
    detach();
    expect(items.map(item => item.getAttribute('tabindex'))).toEqual([null, null, null]);
    expect(items[1].getAttribute('aria-current')).toBeNull();
    expect(items[1].classList.contains('lens-focus-visible')).toBe(false);
    press('ArrowRight');
    expect(onNavigate).toHaveBeenCalledOnce();
  });
  test('honors the host graph target and synchronizes focus entering a different item', () => {
    const {root, items, press, ownerDocument} = createFocusSurface();
    items[1].setAttribute('aria-current', 'true');
    const detach = attachRovingTabindex(root, {
      itemSelector: '[data-key]',
      onNavigate() {
        items[0].setAttribute('aria-current', 'true');
      }
    });
    expect(items.map(item => item.tabIndex)).toEqual([-1, 0, -1]);
    press('ArrowRight');
    expect(ownerDocument.activeElement).toBe(items[0]);
    expect(items.map(item => item.tabIndex)).toEqual([0, -1, -1]);
    items[2].focus();
    expect(items.map(item => item.tabIndex)).toEqual([-1, -1, 0]);
    expect(items[2].classList.contains('lens-focus-visible')).toBe(true);
    detach();
    expect(items[1].getAttribute('aria-current')).toBe('true');
  });
  test('tolerates a temporarily empty item list', () => {
    const {root, items, press} = createFocusSurface();
    items.splice(0);
    const detach = attachRovingTabindex(root, {itemSelector: '[data-key]', onNavigate() {}});
    expect(() => press('ArrowDown')).not.toThrow();
    detach();
  });
});
