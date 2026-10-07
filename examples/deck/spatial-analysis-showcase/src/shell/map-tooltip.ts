// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './map-tooltip.css';
import type {LegendColor, TooltipContent} from '../scenes/scene';
import {h, icon} from './dom';

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';
/** Distance between the pointer and the card, in CSS pixels. */
const POINTER_OFFSET = 14;
/** Minimum gap kept between the card and the edge of the map slot. */
const EDGE_MARGIN = 6;
const SPARK_WIDTH = 48;
const SPARK_HEIGHT = 16;

type Point = {x: number; y: number};

/** Options of {@link MapTooltip.show}. */
export type MapTooltipShowOptions = {
  /** Projected `TooltipContent.anchor`, in CSS px of the map slot. Wins over the pointer. */
  anchor?: Point;
  /** `PointerEvent.pointerType` of the hover. `'touch'` pins the card with a close button. */
  pointerType?: string;
};

const cssColor = (color: LegendColor) =>
  `rgb(${color[0]} ${color[1]} ${color[2]} / ${((color[3] ?? 255) / 255).toFixed(2)})`;

/**
 * Structured hover card of the map (a {@link TooltipContent} or plain text).
 *
 * The shell appends {@link MapTooltip.element} to the map slot and calls {@link MapTooltip.show}
 * with slot-relative pixels. The card ignores the pointer unless pinned, flips at the slot edges,
 * follows the page theme and mirrors its text into a visually hidden live region.
 */
export class MapTooltip {
  /** Absolutely positioned root; append it to the map slot. */
  readonly element: HTMLElement;
  /** Called after the close button or Escape closed the card (the shell clears its highlight). */
  onClose: (() => void) | null = null;
  private readonly card: HTMLElement;
  private readonly closeButton: HTMLButtonElement;
  private readonly mirror: HTMLElement;
  private isPinned = false;
  private isDestroyed = false;
  private lastMirrorText = '';

  constructor() {
    this.closeButton = h(
      'button',
      {
        type: 'button',
        class: 'map-tooltip-close',
        'aria-label': 'Close',
        on: {click: () => this.close()}
      },
      icon('close', 14)
    );
    this.card = h('div', {class: 'map-tooltip-card', 'aria-hidden': 'true'});
    this.mirror = h('div', {class: 'map-tooltip-mirror', 'aria-live': 'polite'});
    this.element = h(
      'div',
      {class: 'map-tooltip', role: 'tooltip'},
      this.card,
      this.closeButton,
      this.mirror
    );
  }

  /** Whether the card is pinned (it stays until closed and takes pointer events). */
  get pinned(): boolean {
    return this.isPinned;
  }

  /**
   * Shows `content` at a pixel of the map slot (CSS px). `options.anchor` wins over `position`.
   * While pinned, mouse updates are ignored; a touch update replaces the pinned card.
   */
  show(
    content: TooltipContent | string,
    position: Point,
    options: MapTooltipShowOptions = {}
  ): void {
    if (this.isDestroyed) return;
    const isTouch = options.pointerType === 'touch';
    if (this.isPinned && !isTouch) return;

    this.renderContent(typeof content === 'string' ? {title: content} : content);
    // Measure once, with the final content, then place.
    const slot = this.element.offsetParent ?? this.element.parentElement;
    const slotWidth = slot?.clientWidth ?? window.innerWidth;
    const slotHeight = slot?.clientHeight ?? window.innerHeight;
    const width = this.element.offsetWidth;
    const height = this.element.offsetHeight;
    const origin = options.anchor ?? position;
    const gap = options.anchor ? POINTER_OFFSET / 2 : POINTER_OFFSET;
    let x = origin.x + gap;
    let y = origin.y + gap;
    if (x + width > slotWidth - EDGE_MARGIN) x = origin.x - gap - width;
    if (y + height > slotHeight - EDGE_MARGIN) y = origin.y - gap - height;
    x = Math.max(EDGE_MARGIN, Math.min(x, slotWidth - width - EDGE_MARGIN));
    y = Math.max(EDGE_MARGIN, Math.min(y, slotHeight - height - EDGE_MARGIN));
    this.element.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
    this.element.classList.add('is-visible');
    if (isTouch) this.pin();
  }

  /** Hides the card. A no-op while pinned. */
  hide(): void {
    if (this.isPinned) return;
    this.element.classList.remove('is-visible');
  }

  /** Pins the card: it takes pointer events, shows a close button, Escape closes it. */
  pin(): void {
    if (this.isPinned || this.isDestroyed) return;
    this.isPinned = true;
    this.element.classList.add('is-pinned');
    document.addEventListener('keydown', this.onKeyDown);
  }

  /** Releases a pinned card (it stays visible until the next {@link MapTooltip.hide}). */
  unpin(): void {
    if (!this.isPinned) return;
    this.isPinned = false;
    this.element.classList.remove('is-pinned');
    document.removeEventListener('keydown', this.onKeyDown);
  }

  /** Removes the element and every listener. */
  destroy(): void {
    this.unpin();
    this.isDestroyed = true;
    this.element.remove();
  }

  private close(): void {
    this.unpin();
    this.hide();
    this.onClose?.();
  }

  private readonly onKeyDown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape') this.close();
  };

  private renderContent(content: TooltipContent): void {
    const parts: (Node | false | undefined)[] = [
      h('div', {class: 'map-tooltip-title'}, content.title),
      content.subtitle ? h('div', {class: 'map-tooltip-subtitle'}, content.subtitle) : undefined
    ];
    if (content.rows?.length) {
      parts.push(
        h(
          'div',
          {class: 'map-tooltip-rows'},
          content.rows.map(row =>
            h(
              'div',
              {class: row.emphasis ? 'map-tooltip-row is-emphasis' : 'map-tooltip-row'},
              h('span', {
                class: 'map-tooltip-swatch',
                style: row.swatch ? {background: cssColor(row.swatch)} : {visibility: 'hidden'}
              }),
              h('span', {class: 'map-tooltip-label'}, row.label),
              h(
                'span',
                {class: 'map-tooltip-value'},
                String(row.value),
                row.unit && h('span', {class: 'map-tooltip-unit'}, ` ${row.unit}`)
              )
            )
          )
        )
      );
    }
    if (content.spark && content.spark.length > 1) {
      parts.push(createSparkline(content.spark, content.sparkHighlight));
    }
    if (content.note) parts.push(h('div', {class: 'map-tooltip-note'}, content.note));
    this.card.replaceChildren(...parts.filter((part): part is Node => Boolean(part)));

    const text = describeContent(content);
    if (text !== this.lastMirrorText) {
      this.lastMirrorText = text;
      this.mirror.textContent = text;
    }
  }
}

function describeContent(content: TooltipContent): string {
  const rows = (content.rows ?? []).map(
    row => `${row.label} ${row.value}${row.unit ? ` ${row.unit}` : ''}`
  );
  return [content.title, content.subtitle, ...rows, content.note].filter(Boolean).join('. ');
}

function createSparkline(values: readonly number[], highlight: number | undefined): SVGSVGElement {
  const svg = document.createElementNS(SVG_NAMESPACE, 'svg');
  svg.setAttribute('class', 'map-tooltip-spark');
  svg.setAttribute('width', String(SPARK_WIDTH));
  svg.setAttribute('height', String(SPARK_HEIGHT));
  svg.setAttribute('viewBox', `0 0 ${SPARK_WIDTH} ${SPARK_HEIGHT}`);
  svg.setAttribute('aria-hidden', 'true');
  const finite = values.filter(Number.isFinite);
  const low = Math.min(...finite);
  const high = Math.max(...finite);
  const span = high - low || 1;
  const pad = 2;
  const pointAt = (value: number, index: number): [number, number] => [
    pad + (index / (values.length - 1)) * (SPARK_WIDTH - 2 * pad),
    SPARK_HEIGHT -
      pad -
      ((Number.isFinite(value) ? value - low : 0) / span) * (SPARK_HEIGHT - 2 * pad)
  ];
  const line = document.createElementNS(SVG_NAMESPACE, 'polyline');
  line.setAttribute(
    'points',
    values.map((value, index) => pointAt(value, index).join(',')).join(' ')
  );
  svg.append(line);
  if (highlight !== undefined && highlight >= 0 && highlight < values.length) {
    const [cx, cy] = pointAt(values[highlight], highlight);
    const dot = document.createElementNS(SVG_NAMESPACE, 'circle');
    dot.setAttribute('cx', cx.toFixed(1));
    dot.setAttribute('cy', cy.toFixed(1));
    dot.setAttribute('r', '2');
    svg.append(dot);
  }
  return svg;
}
