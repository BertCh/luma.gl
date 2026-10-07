// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './map-inset.css';
import {h, icon} from './dom';

/** Screen corner an inset is anchored to. */
export type MapInsetCorner = 'top-right' | 'bottom-right' | 'top-left' | 'bottom-left';

/** Options of {@link MapInset}. */
export type MapInsetOptions = {
  /** Header text, used until `setContent` passes another title. */
  title: string;
  /** Map corner the card is anchored to. */
  corner: MapInsetCorner;
};

/**
 * Screen-anchored chart card on the map (a calendar matrix, Hovmoeller axes). Map chrome like the
 * legend: translucent, collapsible from its header, at most `min(360px, 45%)` wide. On phones it
 * starts collapsed as a pill.
 */
export class MapInset {
  /** Card root; append it to the map slot. */
  readonly element: HTMLElement;
  private readonly titleElement: HTMLElement;
  private readonly body: HTMLElement;
  private readonly toggle: HTMLButtonElement;

  constructor(options: MapInsetOptions) {
    this.titleElement = h('span', {class: 'map-inset-title'}, options.title);
    this.toggle = h(
      'button',
      {
        type: 'button',
        class: 'map-inset-toggle',
        'aria-expanded': 'true',
        on: {click: () => this.setCollapsed(!this.isCollapsed())}
      },
      this.titleElement,
      icon('chevronDown', 14)
    );
    this.body = h('div', {class: 'map-inset-body'});
    this.element = h(
      'section',
      {class: 'map-inset', dataset: {corner: options.corner}, 'aria-label': options.title},
      this.toggle,
      this.body
    );
    this.element.hidden = true;
    if (window.matchMedia('(max-width: 900px)').matches) this.setCollapsed(true);
  }

  /**
   * Sets the chart element (the output of `renderChart`); `null` hides the card. `title` replaces
   * the header text.
   */
  setContent(content: HTMLElement | null, title?: string): void {
    if (title !== undefined) {
      this.titleElement.textContent = title;
      this.element.setAttribute('aria-label', title);
    }
    this.body.replaceChildren(...(content ? [content] : []));
    this.element.hidden = content === null;
  }

  /** Collapses the card to its header (a pill on phones) or expands it. */
  setCollapsed(collapsed: boolean): void {
    this.element.classList.toggle('is-collapsed', collapsed);
    this.toggle.setAttribute('aria-expanded', String(!collapsed));
  }

  /** Removes the element. */
  destroy(): void {
    this.body.replaceChildren();
    this.element.remove();
  }

  private isCollapsed(): boolean {
    return this.element.classList.contains('is-collapsed');
  }
}
