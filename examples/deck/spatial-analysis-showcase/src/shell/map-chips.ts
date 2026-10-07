// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './map-chips.css';
import {h, icon} from './dom';

/** Tone of an engine status chip. */
export type MapChipTone = 'info' | 'warn';

/**
 * Small status chips stacked over the map (by default top-right, under the north arrow):
 * the "Modified" reset chip, the map-ground note and engine status flags. The container is empty
 * (and takes no space or pointer events) while no chip shows. Override the offset with the CSS
 * custom properties `--map-chips-top` and `--map-chips-right` on a parent.
 */
export class MapChips {
  /** Container; append it to the map slot. */
  readonly element: HTMLElement;
  private modifiedChip: HTMLButtonElement | null = null;
  private modifiedHandler: (() => void) | null = null;
  private groundChip: HTMLElement | null = null;
  private readonly statusChips = new Map<string, HTMLElement>();

  constructor() {
    this.element = h('div', {class: 'map-chips', role: 'status'});
  }

  /**
   * Shows or hides the "Modified · Reset to step" chip. `onReset` runs when it is clicked; it
   * replaces the previous handler.
   */
  setModified(modified: boolean, onReset: () => void): void {
    this.modifiedHandler = onReset;
    if (!modified) {
      this.modifiedChip?.remove();
      this.modifiedChip = null;
      return;
    }
    if (this.modifiedChip) return;
    this.modifiedChip = h(
      'button',
      {
        type: 'button',
        class: 'map-chip is-action',
        title: 'Put every option back to the values of this step',
        on: {click: () => this.modifiedHandler?.()}
      },
      h('span', {class: 'map-chip-strong'}, 'Modified'),
      h('span', {class: 'map-chip-dot', 'aria-hidden': 'true'}, '·'),
      icon('reset', 12),
      'Reset to step'
    );
    this.element.prepend(this.modifiedChip);
  }

  /** Shows a note such as "Map: dark ground" (when it differs from the page theme); `null` hides it. */
  setGroundNote(text: string | null): void {
    if (text === null) {
      this.groundChip?.remove();
      this.groundChip = null;
      return;
    }
    if (!this.groundChip) {
      this.groundChip = h('div', {class: 'map-chip'});
      this.modifiedChip
        ? this.modifiedChip.after(this.groundChip)
        : this.element.prepend(this.groundChip);
    }
    this.groundChip.textContent = text;
  }

  /**
   * Shows an engine status chip under `key` ("overflow", "compiling"); `null` text removes it.
   * Chips keep the order in which their keys first appeared.
   */
  setStatus(key: string, text: string | null, tone: MapChipTone = 'info'): void {
    let chip = this.statusChips.get(key);
    if (text === null) {
      chip?.remove();
      this.statusChips.delete(key);
      return;
    }
    if (!chip) {
      chip = h('div', {class: 'map-chip'});
      this.statusChips.set(key, chip);
      this.element.append(chip);
    }
    chip.dataset.tone = tone;
    chip.textContent = text;
  }

  /** Removes the element. */
  destroy(): void {
    this.modifiedHandler = null;
    this.statusChips.clear();
    this.element.remove();
  }
}
