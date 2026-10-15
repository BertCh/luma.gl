// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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

/** Activates recorded frame playback and returns cleanup for its listener and timers. */
export function activateFramePlayers(container: HTMLElement): () => void {
  const player = createFramePlayer(container);
  container.addEventListener('click', player.handleClick);
  return () => {
    container.removeEventListener('click', player.handleClick);
    player.stop();
  };
}
