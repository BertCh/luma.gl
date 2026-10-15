// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LensRecord} from '../registry/view-registration';
import type {ViewRegistry} from '../registry/view-registry';
import {createDefaultViewRegistry} from '../registry/default-registry';
import {getLensScenario} from '../../computation-lenses/registry';
import {getPageCoordinator, type LadderCoordinator} from '../coordinator/ladder-coordinator';
import {LENS_RUNG_IDS, type LensRungId} from '../types';
import {viewNodeToDom} from '../view-node';
import {
  LensController,
  type LensAction,
  type LensControllerOptions,
  type LensFrame
} from './lens-controller';
import {createLiveRegionAnnouncer} from './live-region';
import {attachRovingTabindex} from './roving-tabindex';
import {isLadderNavigationKey} from './navigation-graph';
import {
  decodeLensUrlState,
  encodeLensUrlState,
  readLensStateFromSearch,
  writeLensStateToSearch
} from './url-state';
import {ensureLensStyles} from './lens-styles';

/** Host options for one DOM lens; a page coordinator is shared by bind name. */
export type MountLadderLensOptions = Omit<
  LensControllerOptions,
  'record' | 'registry' | 'lensId' | 'coordinator'
> & {
  record: LensRecord;
  registry?: ViewRegistry;
  bind?: string;
  coordinator?: LadderCoordinator;
  lensId?: string;
  urlState?: boolean;
  /**
   * Scenario shown; with `element` it selects the packs' guards for this element when `guards`
   * is not given (`ViewPackRegistration.guardsFor`).
   */
  scenarioId?: string;
  /** Exported element name; defaults to the registered scenario's `element`. */
  element?: string;
};

/** Mounted lens lifecycle; disposing never destroys a shared page coordinator. */
export type MountedLadderLens = {
  controller: LensController;
  update(record: LensRecord): void;
  markStale(): void;
  dispatch(action: LensAction): void;
  dispose(): void;
};

let lensCount = 0;

/** Mounts the pure controller with delegated DOM events, history and a polite live region. */
export function mountLadderLens(
  element: HTMLElement,
  options: MountLadderLensOptions
): MountedLadderLens {
  const ownerDocument = element.ownerDocument;
  const ownerWindow = ownerDocument.defaultView;
  ensureLensStyles(ownerDocument);
  const container = ownerDocument.createElement('div');
  container.className = 'ladder-lens-mount';
  element.replaceChildren(container);
  const announcer = createLiveRegionAnnouncer(element);
  const registry = options.registry ?? createDefaultViewRegistry();
  const controller = new LensController({
    ...options,
    registry,
    guards:
      options.guards ??
      registry.getGuardsFor({
        record: options.record,
        scenarioId: options.scenarioId,
        element:
          options.element ??
          (options.scenarioId ? getLensScenario(options.scenarioId)?.element : undefined)
      }),
    coordinator: options.coordinator ?? getPageCoordinator(options.bind ?? 'page'),
    lensId: options.lensId ?? `lens-${++lensCount}`,
    ground:
      options.ground ??
      (ownerWindow?.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
    reducedMotion:
      options.reducedMotion ??
      ownerWindow?.matchMedia?.('(prefers-reduced-motion: reduce)').matches ??
      false
  });
  const initialState = controller.getState();
  let disposed = false;
  let detachRoving = () => {};
  let warning: string | undefined;

  const readUrl = (): LensFrame => {
    warning = undefined;
    const encoded = ownerWindow && readLensStateFromSearch(ownerWindow.location.search);
    if (!encoded) return controller.dispatch({type: 'apply-url-state', state: initialState});
    const decoded = decodeLensUrlState(encoded, {expectedRunId: controller.getState().runId});
    if (decoded.status === 'invalid') {
      warning = decoded.reason;
      return {...controller.getFrame(), historyAction: 'none', announcement: undefined};
    }
    if (decoded.status === 'run-mismatch') {
      warning = `This link was made for run ${decoded.state.runId}; showing run ${decoded.expectedRunId}.`;
      const current = controller.getState();
      return controller.dispatch({
        type: 'apply-url-state',
        state: {...current, rung: decoded.state.rung, bindings: decoded.state.bindings}
      });
    }
    return controller.dispatch({type: 'apply-url-state', state: decoded.state});
  };

  const render = (frame: LensFrame) => {
    if (disposed) return;
    const focused = ownerDocument.activeElement;
    const hadFocus = Boolean(focused && container.contains(focused));
    const scrubber = focused?.matches('input.lens-scrubber') ? focused : undefined;
    detachRoving();
    container.replaceChildren(viewNodeToDom(frame.tree, ownerDocument));
    if (warning) {
      const banner = ownerDocument.createElement('p');
      banner.className = 'lens-permalink-warning';
      banner.setAttribute('role', 'alert');
      banner.textContent = warning;
      container.prepend(banner);
    }
    // Keep the native range control during a drag; replacing it would end pointer capture.
    if (scrubber) {
      const replacement = container.querySelector('input.lens-scrubber');
      if (replacement) {
        for (const attribute of Array.from(replacement.attributes))
          scrubber.setAttribute(attribute.name, attribute.value);
        replacement.replaceWith(scrubber);
      }
    }
    const rail = container.querySelector<HTMLElement>('.lens-rung-rail');
    if (rail) {
      detachRoving = attachRovingTabindex(rail, {
        itemSelector: 'button:not([aria-disabled="true"])',
        onNavigate(key, event) {
          event.stopPropagation();
          dispatch({type: 'navigate', key});
        }
      });
      for (const button of rail.querySelectorAll<HTMLButtonElement>('button')) {
        button.tabIndex = -1;
        if (button.dataset['value'] === frame.state.rung)
          button.setAttribute('aria-current', 'true');
        else button.removeAttribute('aria-current');
      }
    }
    if (hadFocus) {
      const target = scrubber ?? container.querySelector<HTMLElement>('.ladder-lens-frame');
      if (target && 'focus' in target) (target as HTMLElement).focus({preventScroll: true});
    }
    if (frame.announcement) announcer.announce(frame.announcement);
    if (options.urlState && ownerWindow && frame.historyAction !== 'none') {
      const search = writeLensStateToSearch(
        ownerWindow.location.search,
        encodeLensUrlState(frame.state)
      );
      ownerWindow.history[frame.historyAction === 'push' ? 'pushState' : 'replaceState'](
        null,
        '',
        `${search}${ownerWindow.location.hash}`
      );
    }
  };
  const dispatch = (action: LensAction) => {
    if (!disposed) render(controller.dispatch(action));
  };
  const actionTarget = (event: Event): Element | undefined => {
    const target = event.target;
    if (!ownerWindow || !(target instanceof ownerWindow.Element)) return undefined;
    const mark = target.closest('[data-action]');
    return mark && container.contains(mark) ? mark : undefined;
  };
  const keysFor = (mark: Element): number[] => {
    const first = Number(mark.getAttribute('data-key'));
    const last = Number(mark.getAttribute('data-key-end') ?? first);
    if (
      !mark.hasAttribute('data-key') ||
      !Number.isInteger(first) ||
      !Number.isInteger(last) ||
      first < 0 ||
      last < first
    )
      return [];
    return Array.from({length: last - first + 1}, (_, index) => first + index);
  };
  const click = (event: Event) => {
    const mark = actionTarget(event);
    if (!mark || mark.closest('[aria-disabled="true"]')) return;
    const action = mark.getAttribute('data-action');
    const axis = mark.getAttribute('data-axis') ?? 'entity';
    const value = mark.getAttribute('data-value') ?? mark.getAttribute('data-key') ?? '';
    switch (action) {
      case 'rung':
        if (LENS_RUNG_IDS.includes(value as LensRungId))
          dispatch({type: 'set-rung', rung: value as LensRungId});
        break;
      case 'pick':
        dispatch({
          type: 'pick',
          axis,
          keys: keysFor(mark),
          label: mark.getAttribute('aria-label') ?? undefined
        });
        break;
      case 'bind': {
        const binding = axis === 'node' || axis === 'buffer' ? value : Number.parseInt(value, 10);
        if (typeof binding === 'string' || Number.isFinite(binding))
          dispatch({type: 'bind', axis, value: binding});
        break;
      }
      case 'free':
        dispatch({type: 'free', axis});
        break;
      case 'toggle':
        dispatch({type: 'toggle', axis, value});
        break;
      case 'pin-inspector':
        dispatch({type: 'pin-inspector', unitId: value});
        break;
      case 'close-inspector':
        dispatch({type: 'close-inspector'});
        break;
      case 'request-update':
        dispatch({type: 'request-update'});
        break;
    }
  };
  let hovered: string | undefined;
  const pointerOver = (event: Event) => {
    const mark = actionTarget(event);
    if (mark?.getAttribute('data-action') !== 'pick' || mark.closest('[aria-disabled="true"]'))
      return;
    const axis = mark.getAttribute('data-axis') ?? 'entity';
    const keys = keysFor(mark);
    const identity = `${axis}:${keys.join(',')}`;
    if (hovered === identity) return;
    hovered = identity;
    dispatch({type: 'hover', axis, keys});
  };
  const pointerLeave = (event: Event) => {
    const mark = actionTarget(event);
    if (event.target !== container && mark?.getAttribute('data-action') !== 'pick') return;
    if (hovered !== undefined) {
      hovered = undefined;
      dispatch({type: 'hover-end'});
    }
  };
  const input = (event: Event) => {
    if (
      ownerWindow &&
      event.target instanceof ownerWindow.HTMLInputElement &&
      event.target.matches('.lens-scrubber')
    ) {
      const value = event.target.valueAsNumber;
      if (Number.isFinite(value)) dispatch({type: 'scrub-preview', value});
    }
  };
  const change = (event: Event) => {
    if (
      ownerWindow &&
      event.target instanceof ownerWindow.HTMLInputElement &&
      event.target.matches('.lens-scrubber')
    )
      dispatch({type: 'scrub-commit'});
  };
  const keydown = (event: KeyboardEvent) => {
    if (event.defaultPrevented || !isLadderNavigationKey(event.key)) return;
    if (ownerWindow && event.target instanceof ownerWindow.HTMLInputElement) return;
    event.preventDefault();
    dispatch({type: 'navigate', key: event.key});
  };
  const focusin = () => {
    for (const button of container.querySelectorAll<HTMLElement>('.lens-rung-rail button')) {
      button.tabIndex = -1;
      if (button.dataset['value'] === controller.getState().rung)
        button.setAttribute('aria-current', 'true');
      else button.removeAttribute('aria-current');
    }
  };
  const popstate = () => render(readUrl());
  container.addEventListener('click', click);
  container.addEventListener('pointerover', pointerOver);
  container.addEventListener('pointerleave', pointerLeave, true);
  container.addEventListener('input', input);
  container.addEventListener('change', change);
  container.addEventListener('keydown', keydown);
  container.addEventListener('focusin', focusin);
  if (options.urlState) ownerWindow?.addEventListener('popstate', popstate);
  const unsubscribe = controller.onCoordinatorChange(() => render(controller.getFrame()));
  render(options.urlState ? readUrl() : controller.getFrame());
  return {
    controller,
    update: record => dispatch({type: 'set-record', record}),
    markStale: () => dispatch({type: 'mark-stale'}),
    dispatch,
    dispose() {
      if (disposed) return;
      disposed = true;
      unsubscribe();
      detachRoving();
      container.removeEventListener('click', click);
      container.removeEventListener('pointerover', pointerOver);
      container.removeEventListener('pointerleave', pointerLeave, true);
      container.removeEventListener('input', input);
      container.removeEventListener('change', change);
      container.removeEventListener('keydown', keydown);
      container.removeEventListener('focusin', focusin);
      ownerWindow?.removeEventListener('popstate', popstate);
      controller.destroy();
      announcer.destroy();
      container.remove();
    }
  };
}
