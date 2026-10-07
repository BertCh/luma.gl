// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './time-bar.css';
import type {OptionSpec, TimelineSpec} from '../scenes/scene';
import type {OptionsStore} from '../scenes/options-state';
import {h} from './dom';

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';
const DEFAULT_SPEEDS = [0.5, 1, 2, 4];
const PLAY_PATH = 'M7 4.5v15l12-7.5z';
const PAUSE_PATH = 'M7 5h3.5v14H7zM13.5 5H17v14h-3.5z';

type Spec<Kind extends OptionSpec<never>['kind']> = Extract<OptionSpec<never>, {kind: Kind}>;

/** Data the bar draws behind the scrubber, in time-option units. */
export type TimeBarData = {
  /** Visible `[start, end]`. Defaults to the time slider's `min..max`. */
  domain?: readonly [number, number];
  /** Bar heights (any scale); bins are spread evenly over the domain. */
  histogram?: readonly number[];
  /** Small ticks on the scrubber with a tooltip label; click one to jump there. */
  events?: readonly {at: number; label?: string}[];
};

type DragTarget = 'time' | 'windowStart' | 'windowEnd';

/**
 * Docked playback bar bound to the options named by a `TimelineSpec`: play/pause (`play`), speed
 * (`speed`), the playhead (`time`) and an optional window (`window`). Every write goes through
 * `store.set`; the bar follows the store (playback writes every frame), redrawing once per frame.
 * Dock {@link TimeBar.element} at the bottom of the map slot.
 */
export class TimeBar {
  /** Root element, docked at the bottom of the map slot. */
  readonly element: HTMLElement;
  private readonly store: OptionsStore;
  private readonly spec: TimelineSpec;
  private readonly timeSpec: Spec<'slider'> | undefined;
  private readonly windowSpec: Spec<'range'> | undefined;
  private domain: readonly [number, number];
  private data: TimeBarData = {};
  private readonly playButton: HTMLButtonElement | null = null;
  private readonly speedSelect: HTMLSelectElement | null = null;
  private readonly scrubber: HTMLElement;
  private readonly track: HTMLElement;
  private readonly histogramLayer: HTMLElement;
  private readonly eventLayer: HTMLElement;
  private readonly fadeStart: HTMLElement;
  private readonly fadeEnd: HTMLElement;
  private readonly handleStart: HTMLElement;
  private readonly handleEnd: HTMLElement;
  private readonly playhead: HTMLElement;
  private readonly bandRow: HTMLElement;
  private readonly tickRow: HTMLElement;
  private readonly valueLabel: HTMLElement;
  private readonly unsubscribe: () => void;
  private frame = 0;
  private dragging: DragTarget | null = null;
  private lastValueText = '';
  private isDestroyed = false;

  /**
   * @param store Option store of the active scene.
   * @param spec Option ids the bar is bound to, plus bands, ticks and the label format.
   */
  constructor(store: OptionsStore, spec: TimelineSpec) {
    this.store = store;
    this.spec = spec;
    this.timeSpec = this.findSpec(spec.time, 'slider');
    this.windowSpec = spec.window ? this.findSpec(spec.window, 'range') : undefined;
    this.domain = this.timeSpec ? [this.timeSpec.min, this.timeSpec.max] : [0, 1];

    const children: Node[] = [];
    if (spec.play && this.findSpec(spec.play, 'toggle')) {
      this.playButton = h('button', {
        type: 'button',
        class: 'time-bar-play',
        'aria-label': 'Play',
        'aria-pressed': 'false',
        on: {click: () => this.togglePlay()}
      });
      children.push(this.playButton);
    }
    const speeds = this.getSpeedChoices();
    if (speeds.length > 1) {
      this.speedSelect = h(
        'select',
        {
          class: 'time-bar-speed',
          'aria-label': 'Speed',
          on: {change: () => this.writeSpeed(this.speedSelect?.value ?? '')}
        },
        speeds.map(choice => h('option', {value: String(choice.value)}, choice.label))
      );
      children.push(this.speedSelect);
    }

    this.histogramLayer = h('div', {class: 'time-bar-histogram'});
    this.eventLayer = h('div', {class: 'time-bar-events'});
    this.fadeStart = h('div', {class: 'time-bar-fade is-start'});
    this.fadeEnd = h('div', {class: 'time-bar-fade is-end'});
    this.handleStart = this.createWindowHandle('windowStart', 'Window start');
    this.handleEnd = this.createWindowHandle('windowEnd', 'Window end');
    this.playhead = h('div', {class: 'time-bar-playhead'}, h('div', {class: 'time-bar-grip'}));
    this.track = h(
      'div',
      {class: 'time-bar-track'},
      this.histogramLayer,
      this.eventLayer,
      this.windowSpec && [this.fadeStart, this.fadeEnd, this.handleStart, this.handleEnd],
      this.playhead
    );
    this.bandRow = h('div', {class: 'time-bar-bands'});
    this.tickRow = h('div', {class: 'time-bar-ticks'});
    this.scrubber = h(
      'div',
      {
        class: 'time-bar-scrubber',
        role: 'slider',
        tabIndex: 0,
        'aria-label': 'Time',
        on: {pointerdown: event => this.onScrubberPointerDown(event)}
      },
      this.track,
      this.bandRow,
      this.tickRow
    );
    children.push(this.scrubber);
    this.valueLabel = h('div', {class: 'time-bar-value', 'aria-hidden': 'true'});
    children.push(this.valueLabel);

    this.element = h(
      'div',
      {
        class: 'time-bar',
        role: 'group',
        'aria-label': 'Time controls',
        on: {keydown: event => this.onKeyDown(event)}
      },
      children
    );
    this.element.addEventListener('pointermove', this.onPointerMove);
    this.element.addEventListener('pointerup', this.onPointerEnd);
    this.element.addEventListener('pointercancel', this.onPointerEnd);
    this.element.addEventListener('lostpointercapture', this.onPointerEnd);

    this.renderStatic();
    this.render();
    this.unsubscribe = store.subscribe(id => {
      if (this.isRelevant(id)) this.scheduleRender();
    });
  }

  /**
   * Supplies the histogram, a domain that replaces the slider's `min..max`, and event ticks.
   * Call again whenever the data changes.
   */
  setData(data: TimeBarData): void {
    this.data = data;
    if (data.domain && data.domain[1] > data.domain[0]) this.domain = data.domain;
    else if (this.timeSpec) this.domain = [this.timeSpec.min, this.timeSpec.max];
    this.renderStatic();
    this.scheduleRender();
  }

  /** Removes the element and stops following the store. */
  destroy(): void {
    this.isDestroyed = true;
    this.unsubscribe();
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.element.remove();
  }

  private findSpec<Kind extends OptionSpec<never>['kind']>(
    id: string,
    kind: Kind
  ): Spec<Kind> | undefined {
    const found = this.store.specs.find(
      option => option.kind === kind && 'id' in option && option.id === id
    );
    return found as Spec<Kind> | undefined;
  }

  private isRelevant(id: string): boolean {
    const {time, play, speed, window: windowId} = this.spec;
    return id === time || id === play || id === speed || id === windowId;
  }

  private getSpeedChoices(): {value: string | number; label: string}[] {
    const id = this.spec.speed;
    if (!id) return [];
    const select = this.findSpec(id, 'select');
    if (select) return select.options.map(option => ({value: option.value, label: option.label}));
    const slider = this.findSpec(id, 'slider');
    if (!slider) return [];
    const inRange = DEFAULT_SPEEDS.filter(speed => speed >= slider.min && speed <= slider.max);
    return (inRange.length ? inRange : [slider.default]).map(speed => ({
      value: speed,
      label: `${speed}x`
    }));
  }

  private writeSpeed(raw: string): void {
    const id = this.spec.speed;
    if (!id) return;
    const isNumeric = typeof this.store.get(id) === 'number';
    this.store.set(id, isNumeric ? Number(raw) : raw);
  }

  private togglePlay(): void {
    const id = this.spec.play;
    if (id) this.store.set(id, !this.store.get(id));
  }

  private createWindowHandle(target: DragTarget, label: string): HTMLElement {
    return h('div', {
      class: 'time-bar-window-handle',
      role: 'slider',
      tabIndex: 0,
      'aria-label': label,
      dataset: {handle: target},
      on: {pointerdown: event => this.startDrag(event, target)}
    });
  }

  private scheduleRender(): void {
    if (this.frame || this.isDestroyed) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.render();
    });
  }

  private toFraction(value: number): number {
    const [start, end] = this.domain;
    return Math.max(0, Math.min(1, (value - start) / (end - start || 1)));
  }

  private fromPointer(event: PointerEvent): number {
    const bounds = this.track.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (event.clientX - bounds.left) / (bounds.width || 1)));
    const [start, end] = this.domain;
    return this.snap(start + fraction * (end - start));
  }

  private snap(value: number): number {
    const spec = this.timeSpec;
    const [start, end] = this.domain;
    let snapped = value;
    if (spec?.step) snapped = spec.min + Math.round((value - spec.min) / spec.step) * spec.step;
    snapped = Math.max(start, Math.min(end, snapped));
    // Remove floating point noise from fractional steps.
    return Number(snapped.toFixed(10));
  }

  private formatValue(value: number): string {
    if (this.spec.format) return this.spec.format(value);
    const spec = this.timeSpec;
    if (spec?.format) return spec.format(value);
    const text = Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.?0+$/, '');
    return spec?.unit ? `${text} ${spec.unit}` : text;
  }

  private onScrubberPointerDown(event: PointerEvent): void {
    if ((event.target as HTMLElement).closest('.time-bar-window-handle')) return;
    this.startDrag(event, 'time');
  }

  private startDrag(event: PointerEvent, target: DragTarget): void {
    if (event.button !== 0 && event.pointerType === 'mouse') return;
    event.preventDefault();
    event.stopPropagation();
    this.dragging = target;
    this.element.setPointerCapture(event.pointerId);
    this.element.classList.add('is-dragging');
    if (target === 'time') this.scrubber.focus({preventScroll: true});
    this.applyDrag(event);
  }

  private readonly onPointerMove = (event: PointerEvent): void => {
    if (this.dragging) this.applyDrag(event);
  };

  private readonly onPointerEnd = (): void => {
    if (!this.dragging) return;
    this.dragging = null;
    this.element.classList.remove('is-dragging');
  };

  private applyDrag(event: PointerEvent): void {
    const value = this.fromPointer(event);
    if (this.dragging === 'time') this.store.set(this.spec.time, value);
    else this.writeWindow(this.dragging === 'windowStart' ? 0 : 1, value);
  }

  private writeWindow(index: 0 | 1, value: number): void {
    const id = this.spec.window;
    const current = id ? this.store.get(id) : undefined;
    if (!id || !Array.isArray(current)) return;
    const next: [number, number] = [current[0], current[1]];
    next[index] = value;
    if (next[0] > next[1]) next[index === 0 ? 0 : 1] = next[index === 0 ? 1 : 0];
    this.store.set(id, next);
  }

  private onKeyDown(event: KeyboardEvent): void {
    const target = event.target as HTMLElement;
    const isControl = target.closest('button, select');
    if (event.key === ' ' && !isControl && this.spec.play) {
      event.preventDefault();
      this.togglePlay();
      return;
    }
    if (isControl) return;
    const handle = target.closest<HTMLElement>('.time-bar-window-handle');
    const step = (this.timeSpec?.step ?? 1) * (event.shiftKey ? 10 : 1);
    let delta = 0;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') delta = -step;
    else if (event.key === 'ArrowRight' || event.key === 'ArrowUp') delta = step;
    else if (event.key !== 'Home' && event.key !== 'End') return;
    event.preventDefault();
    const [start, end] = this.domain;
    const edge = event.key === 'Home' ? start : end;
    const move = (value: number) =>
      this.snap(event.key === 'Home' || event.key === 'End' ? edge : value + delta);
    if (handle && this.spec.window) {
      const index = handle.dataset.handle === 'windowStart' ? 0 : 1;
      const range = this.store.get(this.spec.window);
      if (Array.isArray(range)) this.writeWindow(index, move(range[index]));
      return;
    }
    this.store.set(this.spec.time, move(Number(this.store.get(this.spec.time))));
  }

  /** Rebuilds the parts that depend on data, bands and ticks (not on the playhead). */
  private renderStatic(): void {
    this.histogramLayer.replaceChildren(createHistogram(this.data.histogram));
    this.eventLayer.replaceChildren(
      ...(this.data.events ?? []).map(event =>
        h('button', {
          type: 'button',
          class: 'time-bar-event',
          style: {left: `${this.toFraction(event.at) * 100}%`},
          title: event.label ?? this.formatValue(event.at),
          'aria-label': event.label ?? this.formatValue(event.at),
          on: {
            pointerdown: pointerEvent => pointerEvent.stopPropagation(),
            click: () => this.store.set(this.spec.time, this.snap(event.at))
          }
        })
      )
    );
    const bands = this.spec.bands ?? [];
    this.bandRow.hidden = bands.length === 0;
    this.bandRow.replaceChildren(
      ...bands.map((band, index) => {
        const from = this.toFraction(band.from);
        const to = this.toFraction(band.to);
        return h(
          'div',
          {
            class: 'time-bar-band',
            dataset: {tone: String(index % 4)},
            style: {left: `${from * 100}%`, width: `${(to - from) * 100}%`},
            title: band.label
          },
          band.label && h('span', null, band.label)
        );
      })
    );
    const ticks = this.spec.ticks ?? [];
    this.tickRow.hidden = ticks.length === 0;
    this.tickRow.replaceChildren(
      ...ticks.map(tick =>
        h(
          'span',
          {class: 'time-bar-tick', style: {left: `${this.toFraction(tick.at) * 100}%`}},
          tick.label
        )
      )
    );
  }

  /** Moves the playhead, the window and the labels to the current option values. */
  private render(): void {
    if (this.isDestroyed) return;
    const value = Number(this.store.get(this.spec.time));
    this.playhead.style.left = `${this.toFraction(value) * 100}%`;
    const text = this.formatValue(value);
    if (text !== this.lastValueText) {
      this.lastValueText = text;
      this.valueLabel.textContent = text;
      this.scrubber.setAttribute('aria-valuetext', text);
    }
    this.scrubber.setAttribute('aria-valuemin', String(this.domain[0]));
    this.scrubber.setAttribute('aria-valuemax', String(this.domain[1]));
    this.scrubber.setAttribute('aria-valuenow', String(value));

    if (this.playButton && this.spec.play) {
      const playing = Boolean(this.store.get(this.spec.play));
      if (
        this.playButton.getAttribute('aria-pressed') !== String(playing) ||
        !this.playButton.firstChild
      ) {
        this.playButton.setAttribute('aria-pressed', String(playing));
        this.playButton.setAttribute('aria-label', playing ? 'Pause' : 'Play');
        this.playButton.replaceChildren(createPlayIcon(playing));
      }
    }
    if (this.speedSelect && this.spec.speed) {
      const speed = String(this.store.get(this.spec.speed));
      if (this.speedSelect.value !== speed) this.speedSelect.value = speed;
    }
    if (this.windowSpec && this.spec.window) {
      const range = this.store.get(this.spec.window);
      if (Array.isArray(range)) {
        const low = this.toFraction(range[0]);
        const high = this.toFraction(range[1]);
        this.fadeStart.style.width = `${low * 100}%`;
        this.fadeEnd.style.left = `${high * 100}%`;
        this.handleStart.style.left = `${low * 100}%`;
        this.handleEnd.style.left = `${high * 100}%`;
        this.handleStart.setAttribute('aria-valuenow', String(range[0]));
        this.handleEnd.setAttribute('aria-valuenow', String(range[1]));
        for (const handle of [this.handleStart, this.handleEnd]) {
          handle.setAttribute('aria-valuemin', String(this.domain[0]));
          handle.setAttribute('aria-valuemax', String(this.domain[1]));
        }
      }
    }
  }
}

function createPlayIcon(playing: boolean): SVGSVGElement {
  const svg = document.createElementNS(SVG_NAMESPACE, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', '18');
  svg.setAttribute('height', '18');
  svg.setAttribute('fill', 'currentColor');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS(SVG_NAMESPACE, 'path');
  path.setAttribute('d', playing ? PAUSE_PATH : PLAY_PATH);
  svg.append(path);
  return svg;
}

function createHistogram(histogram: readonly number[] | undefined): SVGSVGElement {
  const svg = document.createElementNS(SVG_NAMESPACE, 'svg');
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('aria-hidden', 'true');
  const bins = histogram ?? [];
  svg.setAttribute('viewBox', `0 0 ${Math.max(bins.length, 1)} 1`);
  const peak = Math.max(0, ...bins.filter(Number.isFinite));
  bins.forEach((count, index) => {
    if (!(count > 0) || !peak) return;
    const rect = document.createElementNS(SVG_NAMESPACE, 'rect');
    const height = count / peak;
    rect.setAttribute('x', String(index));
    rect.setAttribute('y', String(1 - height));
    rect.setAttribute('width', '0.86');
    rect.setAttribute('height', String(height));
    svg.append(rect);
  });
  return svg;
}
