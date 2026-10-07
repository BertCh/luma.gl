// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './compare-divider.css';
import type {CompareSpec} from '../scenes/scene';
import {h, icon} from './dom';

const KEY_STEP = 0.02;
const KEY_STEP_LARGE = 0.1;

/** Compare state reported to the scene (`instance.onCompareChange`). */
export type CompareState = {
  /** Divider position as a fraction of the map width (swipe mode). */
  position: number;
  /** `'both'` while swiping (a on the left, b on the right); `'a'`/`'b'` in hold-to-compare. */
  showing: 'a' | 'b' | 'both';
};

/**
 * G11 swipe and hold-to-compare overlay. The element covers the map slot; only the handle (swipe)
 * or the button (toggle) takes pointer events. `onChange` is rAF-coalesced and fires once when a
 * spec is set (the initial state), on every change, and once with `showing: 'both'` when the spec
 * is cleared after being active.
 */
export class CompareDivider {
  /** Overlay covering the map slot; append it above the map. */
  readonly element: HTMLElement;
  private readonly onChange: (state: CompareState) => void;
  private spec: CompareSpec | null = null;
  private current: CompareState | null = null;
  private frame = 0;
  private dragPointer: number | null = null;
  private readonly line: HTMLElement;
  private readonly handle: HTMLElement;
  private readonly labelA: HTMLElement;
  private readonly labelB: HTMLElement;
  private readonly holdButton: HTMLButtonElement;
  private readonly releaseHold = (): void => this.setShowing('a');

  /** @param onChange Called (once per frame at most) with the new state. */
  constructor(onChange: (state: CompareState) => void) {
    this.onChange = onChange;
    this.labelA = h('div', {class: 'compare-label is-a'});
    this.labelB = h('div', {class: 'compare-label is-b'});
    this.handle = h(
      'div',
      {
        class: 'compare-handle',
        role: 'slider',
        tabIndex: 0,
        'aria-orientation': 'horizontal',
        'aria-valuemin': '0',
        'aria-valuemax': '100',
        on: {
          pointerdown: event => this.onHandleDown(event),
          pointermove: event => this.onHandleMove(event),
          pointerup: event => this.onHandleUp(event),
          pointercancel: event => this.onHandleUp(event),
          keydown: event => this.onHandleKey(event)
        }
      },
      icon('chevronLeft', 14),
      icon('chevronRight', 14)
    );
    this.line = h('div', {class: 'compare-line'}, this.labelA, this.labelB, this.handle);
    this.holdButton = h('button', {
      type: 'button',
      class: 'compare-hold',
      on: {
        pointerdown: event => {
          event.preventDefault();
          this.holdButton.setPointerCapture(event.pointerId);
          this.holdButton.focus({preventScroll: true});
          this.setShowing('b');
        },
        pointerup: this.releaseHold,
        pointercancel: this.releaseHold,
        keydown: event => {
          if (event.key !== ' ') return;
          event.preventDefault();
          if (!event.repeat) this.setShowing('b');
        },
        keyup: event => {
          if (event.key === ' ') this.releaseHold();
        },
        blur: this.releaseHold,
        click: event => event.preventDefault()
      }
    });
    this.element = h('div', {class: 'compare-divider', hidden: true}, this.line, this.holdButton);
  }

  /** Current state, or `null` when no comparison is active. */
  get state(): CompareState | null {
    return this.current;
  }

  /** Shows the comparison of a story step; `null` hides it. */
  setSpec(spec: CompareSpec | null): void {
    const wasActive = this.spec !== null;
    this.spec = spec;
    this.element.hidden = !spec;
    if (!spec) {
      this.current = null;
      if (wasActive) this.emit({position: 0.5, showing: 'both'});
      return;
    }
    const isToggle = spec.mode === 'toggle';
    this.element.dataset.mode = isToggle ? 'toggle' : 'swipe';
    this.labelA.textContent = spec.labels[0];
    this.labelB.textContent = spec.labels[1];
    this.holdButton.textContent = `Hold to compare: ${spec.labels[1]}`;
    this.holdButton.setAttribute('aria-label', `Hold to show ${spec.labels[1]}`);
    this.handle.setAttribute('aria-label', `Compare ${spec.labels[0]} and ${spec.labels[1]}`);
    const position = clamp(spec.position ?? 0.5);
    this.current = {position, showing: isToggle ? 'a' : 'both'};
    this.renderPosition();
    this.emit(this.current);
  }

  /** Removes the element and cancels pending notifications. */
  destroy(): void {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.spec = null;
    this.current = null;
    this.element.remove();
  }

  private setShowing(showing: 'a' | 'b'): void {
    if (!this.current || this.spec?.mode !== 'toggle' || this.current.showing === showing) return;
    this.current = {...this.current, showing};
    this.holdButton.setAttribute('aria-pressed', String(showing === 'b'));
    this.element.classList.toggle('is-held', showing === 'b');
    this.emit(this.current);
  }

  private setPosition(position: number): void {
    if (!this.current) return;
    const next = clamp(position);
    if (next === this.current.position) return;
    this.current = {...this.current, position: next};
    this.renderPosition();
    this.emit(this.current);
  }

  private renderPosition(): void {
    if (!this.current) return;
    const percent = this.current.position * 100;
    this.line.style.left = `${percent}%`;
    this.handle.setAttribute('aria-valuenow', String(Math.round(percent)));
    this.handle.setAttribute('aria-valuetext', `${Math.round(percent)} percent`);
  }

  private emit(state: CompareState): void {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      // Report the latest state, not the one that scheduled the frame.
      this.onChange(this.current ?? state);
    });
  }

  private fractionOf(event: PointerEvent): number {
    const bounds = this.element.getBoundingClientRect();
    return (event.clientX - bounds.left) / (bounds.width || 1);
  }

  private onHandleDown(event: PointerEvent): void {
    if (event.button !== 0 && event.pointerType === 'mouse') return;
    event.preventDefault();
    this.dragPointer = event.pointerId;
    this.handle.setPointerCapture(event.pointerId);
    this.handle.focus({preventScroll: true});
    this.element.classList.add('is-dragging');
  }

  private onHandleMove(event: PointerEvent): void {
    if (this.dragPointer === event.pointerId) this.setPosition(this.fractionOf(event));
  }

  private onHandleUp(event: PointerEvent): void {
    if (this.dragPointer !== event.pointerId) return;
    this.dragPointer = null;
    this.element.classList.remove('is-dragging');
  }

  private onHandleKey(event: KeyboardEvent): void {
    if (!this.current) return;
    const step = event.shiftKey ? KEY_STEP_LARGE : KEY_STEP;
    const position = this.current.position;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') this.setPosition(position - step);
    else if (event.key === 'ArrowRight' || event.key === 'ArrowUp')
      this.setPosition(position + step);
    else if (event.key === 'Home') this.setPosition(0);
    else if (event.key === 'End') this.setPosition(1);
    else return;
    event.preventDefault();
  }
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}
