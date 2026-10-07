// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './shortcut-sheet.css';
import {h, icon} from './dom';

/** One row of the shortcut sheet. */
export type Shortcut = {
  /** Key combination; separate alternatives and chords with spaces ("Shift + ?", "Left Right"). */
  keys: string;
  /** What the shortcut does. */
  action: string;
};

let openDialog: HTMLDialogElement | null = null;

/**
 * Opens a modal dialog listing keyboard shortcuts. Focus stays inside the dialog (native modal
 * behaviour), Escape or a click on the backdrop closes it, and focus returns to the element that
 * had it. Calling it while the sheet is open does nothing.
 */
export function openShortcutSheet(shortcuts: readonly Shortcut[]): void {
  if (openDialog) return;
  const previousFocus =
    document.activeElement instanceof HTMLElement ? document.activeElement : null;
  const closeButton = h(
    'button',
    {
      type: 'button',
      class: 'shortcut-sheet-close',
      'aria-label': 'Close',
      on: {click: () => dialog.close()}
    },
    icon('close', 16)
  );
  const dialog = h(
    'dialog',
    {class: 'shortcut-sheet', 'aria-labelledby': 'shortcut-sheet-title'},
    h(
      'div',
      {class: 'shortcut-sheet-panel'},
      h(
        'div',
        {class: 'shortcut-sheet-header'},
        h('h2', {id: 'shortcut-sheet-title'}, 'Keyboard shortcuts'),
        closeButton
      ),
      h(
        'dl',
        {class: 'shortcut-sheet-list'},
        shortcuts.map(shortcut => [
          h('dt', null, renderKeys(shortcut.keys)),
          h('dd', null, shortcut.action)
        ])
      )
    )
  );
  // The dialog has no padding, so a click whose target is the dialog itself hit the backdrop.
  dialog.addEventListener('click', event => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener('close', () => {
    dialog.remove();
    openDialog = null;
    previousFocus?.focus({preventScroll: true});
  });
  document.body.append(dialog);
  openDialog = dialog;
  dialog.showModal();
}

function renderKeys(keys: string): Node[] {
  return keys
    .split(/\s+/)
    .filter(Boolean)
    .map(key =>
      key === '+' || key === '/' || key.toLowerCase() === 'or'
        ? h('span', {class: 'shortcut-sheet-join'}, key)
        : h('kbd', null, key)
    );
}
