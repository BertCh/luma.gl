// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {fromRecordedLensFile, getLensScenario} from '../../computation-lenses/registry';
import type {LensRecord} from '../registry/view-registration';
import type {ViewRegistry} from '../registry/view-registry';
import {LENS_RUNG_IDS, type AxisBindings, type LensRungId, type RenderTeX} from '../types';
import {mountLadderLens, type MountedLadderLens} from './mount-lens';
import {getPageScheduler} from './page-scheduler';
import {getAnalysisDevice} from './device-profile';
import {ensureLensStyles} from './lens-styles';

/** Loader and optional live execution facilities supplied by the embedding application. */
export type DefineLadderLensElementOptions = {
  tagName?: string;
  loadRecord(scenarioId: string): Promise<unknown>;
  renderTeX?: RenderTeX;
  registry?: ViewRegistry;
  createDevice?: (props: {
    type: 'webgpu';
    optionalFeatures: string[];
    requiredLimits?: Record<string, number>;
  }) => Promise<unknown>;
  runScenario?: (scenarioId: string, device: unknown) => Promise<LensRecord>;
};

/** String-only attributes shared with static-twin directive adapters. */
export const LADDER_LENS_ATTRIBUTES: readonly string[] = [
  'scenario',
  'rung',
  'bind',
  'datum',
  'iteration',
  'caption',
  'view'
];

let elementCount = 0;

/** Defines the opt-in lens element once; importing this module never accesses the DOM. */
export function defineLadderLensElement(options: DefineLadderLensElementOptions): void {
  if (typeof customElements === 'undefined') return;
  const tagName = options.tagName ?? 'ladder-lens';
  if (customElements.get(tagName)) return;

  class LadderLensElement extends HTMLElement {
    static get observedAttributes(): string[] {
      return [...LADDER_LENS_ATTRIBUTES];
    }
    private lensId = `ladder-lens-${++elementCount}`;
    private mounted?: MountedLadderLens;
    private mountedRecord?: LensRecord;
    private controls?: HTMLDivElement;
    private container?: HTMLDivElement;
    private message?: HTMLParagraphElement;
    private twinChildren = new Map<Element, boolean>();
    private observer?: IntersectionObserver;
    private releasePause?: () => void;
    private generation = 0;
    private busy = false;
    private visible = true;

    connectedCallback(): void {
      if (this.controls) return;
      ensureLensStyles(this.ownerDocument);
      this.twinChildren = new Map(
        [...this.children].map(child => [child, child.hasAttribute('hidden')])
      );
      this.controls = this.ownerDocument.createElement('div');
      this.controls.className = 'ladder-lens-controls';
      this.appendButton('explore', 'Explore the recorded run');
      this.refreshLiveButton();
      this.message = this.ownerDocument.createElement('p');
      this.message.className = 'ladder-lens-message';
      this.message.hidden = true;
      this.controls.append(this.message);
      this.container = this.ownerDocument.createElement('div');
      this.container.className = 'ladder-lens-interactive';
      this.append(this.controls, this.container);
      this.controls.addEventListener('click', this.handleClick);
      const scheduler = getPageScheduler();
      this.releasePause = scheduler.onPause(lensId => {
        if (lensId === this.lensId) this.resetMount();
      });
      this.ownerDocument.addEventListener('visibilitychange', this.handleVisibility);
      scheduler.setPageHidden(this.ownerDocument.hidden);
      const Observer = this.ownerDocument.defaultView?.IntersectionObserver;
      if (Observer) {
        this.observer = new Observer(
          entries => {
            for (const entry of entries) {
              if (entry.target !== this) continue;
              this.visible = entry.isIntersecting;
              scheduler.setVisible(this.lensId, this.visible);
              if (!this.visible) this.resetMount();
            }
          },
          {rootMargin: '200% 0px'}
        );
        this.observer.observe(this);
      }
    }

    attributeChangedCallback(name: string, previous: string | null, value: string | null): void {
      if (previous === value) return;
      if (name === 'scenario') {
        this.resetMount();
        this.refreshLiveButton();
        return;
      }
      if (!this.mounted) return;
      if (name === 'rung') this.mounted.dispatch({type: 'set-rung', rung: this.getRung()});
      else if (name === 'datum' || name === 'iteration') {
        const axis = name === 'datum' ? 'entity' : 'iteration';
        const binding = parseIndex(value);
        this.mounted.dispatch(
          binding === undefined ? {type: 'free', axis} : {type: 'bind', axis, value: binding}
        );
      } else if (this.mountedRecord && (name === 'caption' || name === 'view' || name === 'bind')) {
        const state = this.mounted.controller.getState();
        this.mounted.dispose();
        this.mounted = undefined;
        this.mountRecord(this.mountedRecord, state.bindings);
      }
    }

    disconnectedCallback(): void {
      this.resetMount();
      this.observer?.disconnect();
      this.observer = undefined;
      this.releasePause?.();
      this.releasePause = undefined;
      this.ownerDocument.removeEventListener('visibilitychange', this.handleVisibility);
      this.controls?.removeEventListener('click', this.handleClick);
      this.controls?.remove();
      this.container?.remove();
      this.controls = undefined;
      this.container = undefined;
      this.message = undefined;
      this.twinChildren.clear();
    }

    private getScenario(): string {
      return this.getAttribute('scenario') ?? '';
    }
    private getRung(): LensRungId {
      const value = this.getAttribute('rung');
      return LENS_RUNG_IDS.find(rung => rung === value) ?? 'table';
    }
    private appendButton(action: string, label: string): void {
      const button = this.ownerDocument.createElement('button');
      button.type = 'button';
      button.dataset['ladder'] = action;
      button.textContent = label;
      this.controls?.append(button);
    }
    private refreshLiveButton(): void {
      this.controls?.querySelector('[data-ladder="live"]')?.remove();
      if (
        this.ownerDocument.defaultView?.navigator.gpu &&
        options.createDevice &&
        (options.runScenario || getLensScenario(this.getScenario()))
      )
        this.appendButton('live', 'Run on this GPU');
    }
    private handleVisibility = (): void => {
      getPageScheduler().setPageHidden(this.ownerDocument.hidden);
    };
    private handleClick = (event: Event): void => {
      const ownerWindow = this.ownerDocument.defaultView;
      if (!ownerWindow || !(event.target instanceof ownerWindow.Element)) return;
      const button = event.target.closest('[data-ladder]');
      if (!button || this.busy) return;
      const action = button.getAttribute('data-ladder');
      if (action === 'explore' || action === 'live') void this.activate(action);
    };
    private async activate(mode: 'explore' | 'live'): Promise<void> {
      const scheduler = getPageScheduler();
      const scenario = this.getScenario();
      const generation = ++this.generation;
      const current = () => generation === this.generation && this.isConnected;
      if (mode === 'live') {
        const reservation = scheduler.requestLive(this.lensId, 16 * 2 ** 20);
        if ('reason' in reservation) {
          this.showError(reservation.reason);
          return;
        }
        scheduler.setVisible(this.lensId, this.visible);
        if (!current()) return;
        this.mounted?.markStale();
      }
      this.busy = true;
      this.setBusy(true);
      if (this.message) this.message.hidden = true;
      try {
        let record: LensRecord;
        if (mode === 'explore') {
          scheduler.release(this.lensId);
          record = fromRecordedLensFile(await options.loadRecord(scenario)).record;
        } else {
          record = await scheduler.enqueueRun(this.lensId, async () => {
            if (!current() || !options.createDevice)
              throw new Error('Live run is no longer active');
            const device = await getAnalysisDevice(options.createDevice);
            if (!current()) throw new Error('Live run is no longer active');
            if (options.runScenario) return options.runScenario(scenario, device);
            const descriptor = getLensScenario(scenario);
            if (!descriptor) throw new Error(`No live scenario registered for ${scenario}`);
            const scenarioModule = await descriptor.load();
            if (!current()) throw new Error('Live run is no longer active');
            // The host factory is the device boundary; registered scenarios require luma Device.
            return scenarioModule.run(device as Device);
          });
        }
        if (!current()) return;
        if (this.mounted && mode === 'live') this.mounted.update(record);
        else {
          this.mounted?.dispose();
          this.mounted = undefined;
          const mounted = this.mountRecord(record);
          if (mode === 'live') mounted?.update(record);
        }
        this.mountedRecord = record;
        this.hideTwin();
      } catch (error) {
        if (current()) {
          this.resetMount();
          this.showError(error instanceof Error ? error.message : String(error));
        }
      } finally {
        if (current()) {
          this.busy = false;
          this.setBusy(false);
        }
      }
    }
    private mountRecord(
      record: LensRecord,
      bindings?: AxisBindings
    ): MountedLadderLens | undefined {
      if (!this.container) return;
      this.mountedRecord = record;
      this.mounted = mountLadderLens(this.container, {
        record,
        lensId: this.lensId,
        rung: this.getRung(),
        bindings: bindings ?? {
          entity: parseIndex(this.getAttribute('datum')),
          iteration: parseIndex(this.getAttribute('iteration'))
        },
        caption: this.getAttribute('caption') ?? undefined,
        bind: this.getAttribute('bind') ?? undefined,
        viewId: this.getAttribute('view') ?? undefined,
        scenarioId: this.getScenario(),
        renderTeX: options.renderTeX,
        registry: options.registry
      });
      return this.mounted;
    }
    private hideTwin(): void {
      for (const child of this.twinChildren.keys()) {
        child.setAttribute('hidden', '');
        child.setAttribute('data-ladder-twin', '');
      }
    }
    private resetMount(): void {
      this.generation++;
      this.busy = false;
      this.setBusy(false);
      this.mounted?.dispose();
      this.mounted = undefined;
      this.mountedRecord = undefined;
      getPageScheduler().release(this.lensId);
      for (const [child, hidden] of this.twinChildren) {
        child.toggleAttribute('hidden', hidden);
        child.removeAttribute('data-ladder-twin');
      }
    }
    private setBusy(busy: boolean): void {
      for (const button of this.controls?.querySelectorAll('button') ?? []) button.disabled = busy;
    }
    private showError(reason: string): void {
      if (!this.message) return;
      this.message.hidden = false;
      this.message.setAttribute('role', 'alert');
      this.message.textContent = reason;
    }
  }
  customElements.define(tagName, LadderLensElement);
}

function parseIndex(value: string | null): number | undefined {
  if (value === null || !value.trim()) return undefined;
  const index = Number(value);
  return Number.isInteger(index) && index >= 0 ? index : undefined;
}
