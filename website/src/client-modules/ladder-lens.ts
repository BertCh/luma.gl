// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ClientModule} from '@docusaurus/types';
import {mountLiveLens, type LiveLensAttributes} from './live-lens-adapter';

type LensPageState = {
  generation: number;
  listenerInstalled: boolean;
  coordinators: Map<string, unknown>;
  disposers: Set<() => void>;
};

let pageState: LensPageState | undefined;

function getPageState(): LensPageState {
  pageState ??= {
    generation: 0,
    listenerInstalled: false,
    // Reserved per-bind coordinator slots until the live host publishes that interface.
    coordinators: new Map<string, unknown>(),
    disposers: new Set<() => void>()
  };
  return pageState;
}

function readAttributes(element: HTMLElement): LiveLensAttributes {
  const attributes: LiveLensAttributes = {
    scenario: element.getAttribute('scenario') || '',
    rung: element.getAttribute('rung') || '',
    bind: element.getAttribute('bind') ?? 'page'
  };
  for (const name of ['datum', 'iteration'] as const) {
    const value = element.getAttribute(name);
    if (value !== null) {
      attributes[name] = Number(value);
    }
  }
  const view = element.getAttribute('view');
  if (view !== null) {
    attributes.view = view;
  }
  return attributes;
}

function showError(figure: Element, error: unknown): void {
  if (typeof window === 'undefined') {
    return;
  }
  figure.querySelector('.lens-twin__error')?.remove();
  const message = document.createElement('p');
  message.className = 'lens-twin__error';
  message.setAttribute('role', 'status');
  message.textContent = error instanceof Error ? error.message : String(error);
  figure.append(message);
}

async function handleLiveClick(event: MouseEvent): Promise<void> {
  if (
    typeof window === 'undefined' ||
    !('gpu' in navigator) ||
    !(event.target instanceof Element)
  ) {
    return;
  }
  const button = event.target.closest<HTMLButtonElement>('.lens-twin__live-button');
  const figure = button?.closest('figure.lens-twin');
  const element = figure?.querySelector<HTMLElement>('ladder-lens');
  if (!button || button.disabled || !figure || !element) {
    return;
  }
  const state = getPageState();
  const generation = state.generation;
  const attributes = readAttributes(element);
  const recordedMarkup = element.innerHTML;
  const buttonText = button.textContent;
  const restoreTwin = () => {
    element.innerHTML = recordedMarkup;
    button.disabled = false;
    button.textContent = buttonText;
  };
  // Register restoration immediately, including mounts still awaiting imports on navigation.
  let disposeMount: (() => void) | undefined;
  const dispose = () => {
    try {
      disposeMount?.();
    } finally {
      restoreTwin();
    }
  };
  state.disposers.add(dispose);
  if (!state.coordinators.has(attributes.bind)) {
    state.coordinators.set(attributes.bind, undefined);
  }
  button.disabled = true;
  button.textContent = 'Starting…';
  figure.querySelector('.lens-twin__error')?.remove();
  try {
    const [views, lenses] = await Promise.all([
      import('@luma.gl/experimental/computation-views'),
      import('@luma.gl/experimental/computation-lenses')
    ]);
    if (generation !== state.generation || !element.isConnected) {
      state.disposers.delete(dispose);
      restoreTwin();
      return;
    }
    disposeMount = await mountLiveLens(element, attributes, {views, lenses});
    if (generation !== state.generation || !element.isConnected) {
      state.disposers.delete(dispose);
      dispose();
      return;
    }
    button.textContent = 'Live on this GPU';
  } catch (error) {
    state.disposers.delete(dispose);
    restoreTwin();
    if (generation === state.generation && element.isConnected) {
      showError(figure, error);
    }
  }
}

function createFramePlayer(container: HTMLElement | Document) {
  const players = new Map<HTMLButtonElement, {stop: () => void}>();
  const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  for (const button of container.querySelectorAll<HTMLButtonElement>(
    'figure.lens-twin button[data-cv-play]'
  )) {
    if (reducedMotion()) button.textContent = 'Step ▸';
  }
  const handleClick = (event: MouseEvent) => {
    if (!(event.target instanceof Element)) return;
    const button = event.target.closest<HTMLButtonElement>('button[data-cv-play]');
    const figure = button?.closest('figure.lens-twin');
    if (!button || button.disabled || !figure || !container.contains(button)) return;
    const running = players.get(button);
    if (running) {
      running.stop();
      return;
    }
    const frames = [...figure.querySelectorAll<SVGGElement>('svg g.cv-frame[data-cv-frame]')].sort(
      (left, right) => Number(left.dataset.cvFrame) - Number(right.dataset.cvFrame)
    );
    if (!frames.length) return;
    let status = figure.querySelector<HTMLElement>('.cv-play-status');
    if (!status) {
      status = document.createElement('span');
      status.className = 'cv-play-status';
      (figure.querySelector('figcaption') ?? figure).append(status);
    }
    status.setAttribute('aria-live', 'polite');
    const showFrame = (index: number) => {
      for (const [position, frame] of frames.entries())
        frame.toggleAttribute('hidden', position !== index);
      status.textContent = `round ${index + 1} of ${frames.length}`;
    };
    if (reducedMotion()) {
      button.textContent = 'Step ▸';
      button.setAttribute('aria-pressed', 'false');
      const visible = frames.findIndex(frame => !frame.hasAttribute('hidden'));
      showFrame((visible + 1) % frames.length);
      return;
    }
    let frameIndex = 0;
    let timer: ReturnType<typeof setInterval> | undefined;
    const stop = () => {
      if (timer !== undefined) clearInterval(timer);
      players.delete(button);
      showFrame(frames.length - 1);
      button.setAttribute('aria-pressed', 'false');
    };
    button.setAttribute('aria-pressed', 'true');
    showFrame(frameIndex);
    if (frames.length === 1) {
      stop();
      return;
    }
    players.set(button, {stop});
    timer = setInterval(
      () => {
        if (!figure.isConnected || ++frameIndex === frames.length - 1) stop();
        else showFrame(frameIndex);
      },
      Math.min(600, 5000 / frames.length)
    );
  };
  return {
    handleClick,
    stop() {
      for (const player of players.values()) player.stop();
      players.clear();
    }
  };
}

let framePlayer: ReturnType<typeof createFramePlayer> | undefined;

function revealHashTarget(): void {
  let identifier: string;
  try {
    identifier = decodeURIComponent(window.location.hash.slice(1));
  } catch {
    return;
  }
  const target = document.getElementById(identifier);
  let parent = target?.parentElement;
  let opened = false;
  while (parent) {
    if (
      parent instanceof HTMLDetailsElement &&
      parent.matches('details.lens-how') &&
      !parent.open
    ) {
      parent.open = true;
      opened = true;
    }
    parent = parent.parentElement;
  }
  if (opened) target?.scrollIntoView();
}

const ladderLensClientModule: ClientModule = {
  onRouteDidUpdate() {
    if (typeof window === 'undefined') {
      return;
    }
    const state = getPageState();
    framePlayer?.stop();
    state.generation++;
    for (const dispose of state.disposers) {
      try {
        dispose();
      } catch {
        // A failed disposal must not prevent cleanup of the remaining mounted lenses.
      }
    }
    state.disposers.clear();
    state.coordinators.clear();
    if (!state.listenerInstalled) {
      document.addEventListener('click', handleLiveClick);
      framePlayer = createFramePlayer(document);
      document.addEventListener('click', framePlayer.handleClick);
      window.addEventListener('hashchange', revealHashTarget);
      state.listenerInstalled = true;
    }
    revealHashTarget();
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      for (const button of document.querySelectorAll('figure.lens-twin button[data-cv-play]')) {
        button.textContent = 'Step ▸';
      }
    }
    if ('gpu' in navigator) {
      for (const button of document.querySelectorAll<HTMLButtonElement>(
        '.lens-twin__live-button'
      )) {
        button.hidden = false;
      }
    }
  }
};

export default ladderLensClientModule;
