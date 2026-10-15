// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {isLadderNavigationKey, type LadderNavigationKey} from './navigation-graph';

/** Makes a lens one tab stop; hosts may set aria-current in onNavigate to select a graph target. */
export function attachRovingTabindex(
  root: HTMLElement,
  options: {itemSelector: string; onNavigate(key: LadderNavigationKey, event: KeyboardEvent): void}
): () => void {
  const getItems = () => Array.from(root.querySelectorAll<HTMLElement>(options.itemSelector));
  const original = new Map<
    HTMLElement,
    {tabIndex: string | null; current: string | null; focusVisible: boolean}
  >();
  let active: HTMLElement | undefined;
  const synchronize = (item: HTMLElement | undefined, focus: boolean) => {
    const items = getItems();
    active = item && items.includes(item) ? item : items[0];
    for (const candidate of items) {
      if (!original.has(candidate))
        original.set(candidate, {
          tabIndex: candidate.getAttribute('tabindex'),
          current: candidate.getAttribute('aria-current'),
          focusVisible: candidate.classList.contains('lens-focus-visible')
        });
      candidate.tabIndex = candidate === active ? 0 : -1;
      if (candidate === active) candidate.setAttribute('aria-current', 'true');
      else candidate.removeAttribute('aria-current');
      candidate.classList.toggle(
        'lens-focus-visible',
        candidate === active && (focus || root.ownerDocument.activeElement === candidate)
      );
    }
    if (focus && active) {
      active.focus({preventScroll: true});
      active.scrollIntoView({block: 'nearest'});
    }
  };
  const initial = getItems();
  synchronize(
    initial.find(item => item.getAttribute('aria-current') === 'true') ??
      initial.find(item => item.tabIndex === 0),
    false
  );
  const onFocus = () => {
    const focused = getItems().find(
      item =>
        item === root.ownerDocument.activeElement || item.contains(root.ownerDocument.activeElement)
    );
    if (focused) {
      synchronize(focused, false);
      focused.classList.add('lens-focus-visible');
      focused.scrollIntoView({block: 'nearest'});
    }
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if (!isLadderNavigationKey(event.key)) return;
    event.preventDefault();
    const previous = active;
    const previousIndex = getItems().indexOf(previous!);
    options.onNavigate(event.key, event);
    const items = getItems();
    // The graph host can mark a new active item synchronously. Otherwise use linear movement.
    let target = items.find(
      item =>
        item !== previous && (item.getAttribute('aria-current') === 'true' || item.tabIndex === 0)
    );
    if (!target) {
      const offset =
        event.key === 'ArrowLeft' || event.key === 'ArrowUp'
          ? -1
          : event.key === 'ArrowRight' || event.key === 'ArrowDown'
            ? 1
            : 0;
      const index =
        event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : previousIndex + offset;
      target = items[Math.max(0, Math.min(items.length - 1, index))];
    }
    synchronize(target, true);
  };
  root.addEventListener('keydown', onKeyDown);
  root.addEventListener('focusin', onFocus);
  return () => {
    root.removeEventListener('keydown', onKeyDown);
    root.removeEventListener('focusin', onFocus);
    for (const [item, attributes] of original) {
      if (attributes.tabIndex === null) item.removeAttribute('tabindex');
      else item.setAttribute('tabindex', attributes.tabIndex);
      if (attributes.current === null) item.removeAttribute('aria-current');
      else item.setAttribute('aria-current', attributes.current);
      item.classList.toggle('lens-focus-visible', attributes.focusVisible);
    }
    original.clear();
  };
}
