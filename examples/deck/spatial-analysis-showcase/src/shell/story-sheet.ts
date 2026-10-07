// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {prefersReducedMotion} from '../engine/motion';

/** Resting heights of the phone sheet. */
export type SheetDetent = 'peek' | 'half' | 'full';

const DETENTS: readonly SheetDetent[] = ['peek', 'half', 'full'];
/** Pointer travel (CSS px) below which a press on the handle is a tap, not a drag. */
const TAP_DISTANCE = 6;
/** How far ahead (ms) the release velocity projects the sheet when choosing a detent. */
const VELOCITY_LOOKAHEAD = 160;
const SETTLE_MILLISECONDS = 220;

/** Options of {@link SheetController}. */
export type SheetControllerOptions = {
  /** The panel that becomes the sheet (`.story-panel`). */
  panel: HTMLElement;
  /** The drag handle button inside it. */
  handle: HTMLButtonElement;
  /** The map slot, which receives `--sheet-offset` so corner stacks and the time bar clear the sheet. */
  mapSlot: HTMLElement;
  /** Called with the sheet height in CSS px when a detent settles (and on resize), `0` on desktop. */
  onSettle: (heightPixels: number) => void;
};

/**
 * The phone bottom sheet (three detents). Heights are computed, not measured: peek is
 * `max(112px, 14dvh)`, half `46dvh`, full the page below the header minus 8 px. The panel height
 * is the `--sheet-h` custom property (CSS animates it over 200 ms); during a drag it follows the
 * pointer and `--sheet-offset` on the map slot follows it, so no layout is read per frame. On
 * release the velocity picks the detent. A tap cycles peek, half, full.
 */
export class SheetController {
  private readonly options: SheetControllerOptions;
  private detent: SheetDetent = 'half';
  private enabled = false;
  private dragStartY = 0;
  private dragStartHeight = 0;
  private dragPointer: number | null = null;
  private dragMoved = false;
  private lastMove: {y: number; time: number; velocity: number} | null = null;
  private suppressClick = false;
  private settleTimer = 0;
  private headerHeight = 52;

  constructor(options: SheetControllerOptions) {
    this.options = options;
    const {handle} = options;
    handle.addEventListener('pointerdown', this.onPointerDown);
    handle.addEventListener('pointermove', this.onPointerMove);
    handle.addEventListener('pointerup', this.onPointerUp);
    handle.addEventListener('pointercancel', this.onPointerUp);
    handle.addEventListener('click', this.onClick);
  }

  /** The detent the sheet rests at. */
  getDetent(): SheetDetent {
    return this.detent;
  }

  /** Sheet height in CSS px at the current detent; `0` while the sheet is off (desktop). */
  getHeight(): number {
    return this.enabled ? this.getDetentHeight(this.detent) : 0;
  }

  /** Turns the sheet behaviour on (phone width) or off (desktop side panel). */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    const {panel, mapSlot} = this.options;
    if (!enabled) {
      panel.removeAttribute('data-detent');
      panel.style.removeProperty('--sheet-h');
      mapSlot.style.removeProperty('--sheet-offset');
      this.options.handle.setAttribute('aria-expanded', 'true');
      this.options.onSettle(0);
      return;
    }
    this.headerHeight = this.readHeaderHeight();
    this.setDetent(this.detent, false);
  }

  /** Moves to a detent (animated by CSS), keeping the map's offset in step. */
  setDetent(detent: SheetDetent, animate = true): void {
    this.detent = detent;
    const {panel, mapSlot, handle} = this.options;
    panel.dataset['detent'] = detent;
    panel.classList.remove('is-dragging');
    const height = this.getDetentHeight(detent);
    panel.style.setProperty('--sheet-h', `${height}px`);
    mapSlot.style.setProperty('--sheet-offset', `${height}px`);
    handle.setAttribute('aria-expanded', String(detent !== 'peek'));
    window.clearTimeout(this.settleTimer);
    const wait = animate && !prefersReducedMotion() ? SETTLE_MILLISECONDS + 30 : 0;
    this.settleTimer = window.setTimeout(() => this.options.onSettle(height), wait);
  }

  /** Recomputes pixel heights after the window resized. */
  refresh(): void {
    if (!this.enabled) return;
    this.headerHeight = this.readHeaderHeight();
    this.setDetent(this.detent, false);
  }

  /** Removes listeners and timers. */
  destroy(): void {
    const {handle} = this.options;
    handle.removeEventListener('pointerdown', this.onPointerDown);
    handle.removeEventListener('pointermove', this.onPointerMove);
    handle.removeEventListener('pointerup', this.onPointerUp);
    handle.removeEventListener('pointercancel', this.onPointerUp);
    handle.removeEventListener('click', this.onClick);
    window.clearTimeout(this.settleTimer);
  }

  private readHeaderHeight(): number {
    const value = Number.parseFloat(
      getComputedStyle(document.documentElement).getPropertyValue('--header-height')
    );
    return Number.isFinite(value) ? value : 52;
  }

  private getDetentHeight(detent: SheetDetent): number {
    const viewportHeight = window.innerHeight;
    switch (detent) {
      case 'peek':
        return Math.max(112, Math.round(viewportHeight * 0.14));
      case 'half':
        return Math.round(viewportHeight * 0.46);
      case 'full':
        return Math.max(160, viewportHeight - this.headerHeight - 8);
    }
  }

  private readonly onClick = (): void => {
    if (!this.enabled) return;
    if (this.suppressClick) {
      this.suppressClick = false;
      return;
    }
    const next = DETENTS[(DETENTS.indexOf(this.detent) + 1) % DETENTS.length];
    this.setDetent(next);
  };

  private readonly onPointerDown = (event: PointerEvent): void => {
    if (!this.enabled || this.dragPointer !== null) return;
    this.dragPointer = event.pointerId;
    this.dragStartY = event.clientY;
    this.dragStartHeight = this.getDetentHeight(this.detent);
    this.dragMoved = false;
    this.suppressClick = false;
    this.lastMove = {y: event.clientY, time: event.timeStamp, velocity: 0};
    this.options.handle.setPointerCapture(event.pointerId);
  };

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (event.pointerId !== this.dragPointer) return;
    const travel = this.dragStartY - event.clientY;
    if (!this.dragMoved && Math.abs(travel) < TAP_DISTANCE) return;
    this.dragMoved = true;
    const {panel, mapSlot} = this.options;
    panel.classList.add('is-dragging');
    const minimum = this.getDetentHeight('peek');
    const maximum = this.getDetentHeight('full');
    const height = Math.min(Math.max(this.dragStartHeight + travel, minimum), maximum);
    panel.style.setProperty('--sheet-h', `${height}px`);
    mapSlot.style.setProperty('--sheet-offset', `${height}px`);
    const previous = this.lastMove;
    if (previous && event.timeStamp > previous.time) {
      // Upward travel is positive: pixels per millisecond of sheet growth.
      const velocity = (previous.y - event.clientY) / (event.timeStamp - previous.time);
      this.lastMove = {y: event.clientY, time: event.timeStamp, velocity};
    }
  };

  private readonly onPointerUp = (event: PointerEvent): void => {
    if (event.pointerId !== this.dragPointer) return;
    this.dragPointer = null;
    if (this.options.handle.hasPointerCapture(event.pointerId)) {
      this.options.handle.releasePointerCapture(event.pointerId);
    }
    if (!this.dragMoved) return;
    this.suppressClick = true;
    const travel = this.dragStartY - event.clientY;
    const velocity = this.lastMove?.velocity ?? 0;
    const projected = this.dragStartHeight + travel + velocity * VELOCITY_LOOKAHEAD;
    let nearest: SheetDetent = this.detent;
    let nearestDistance = Number.POSITIVE_INFINITY;
    for (const detent of DETENTS) {
      const distance = Math.abs(this.getDetentHeight(detent) - projected);
      if (distance < nearestDistance) {
        nearest = detent;
        nearestDistance = distance;
      }
    }
    this.setDetent(nearest);
  };
}
