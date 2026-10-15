// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Injectable timers keep the debouncer independent of the DOM and test clocks. */
export type AnnouncerOptions = {
  debounceMs?: number;
  setTimeout?: (callback: () => void, delay: number) => unknown;
  clearTimeout?: (handle: unknown) => void;
};

/** A cancellable, flushable stream of polite announcements. */
export type LiveRegionAnnouncer = {announce(text: string): void; flush(): void; destroy(): void};

/** Emits only the last text after quiet, suppressing consecutive identical announcements. */
export function createDebouncedAnnouncer(
  emit: (text: string) => void,
  options: AnnouncerOptions = {}
): LiveRegionAnnouncer {
  const schedule =
    options.setTimeout ?? ((callback, delay) => globalThis.setTimeout(callback, delay));
  const cancel =
    options.clearTimeout ??
    (handle => globalThis.clearTimeout(handle as ReturnType<typeof globalThis.setTimeout>));
  let timer: unknown;
  let scheduled = false;
  let pending: string | undefined;
  let previous: string | undefined;
  let destroyed = false;
  const clear = () => {
    if (scheduled) cancel(timer);
    scheduled = false;
  };
  const flush = () => {
    clear();
    if (destroyed || pending === undefined) return;
    const text = pending;
    pending = undefined;
    if (text !== previous) {
      previous = text;
      emit(text);
    }
  };
  return {
    announce(text) {
      if (destroyed) return;
      clear();
      pending = text;
      if (text !== previous) {
        scheduled = true;
        timer = schedule(flush, options.debounceMs ?? 600);
      }
    },
    flush,
    destroy() {
      clear();
      pending = undefined;
      destroyed = true;
    }
  };
}

/** Creates one visually hidden polite status node, touching the DOM only when called. */
export function createLiveRegionAnnouncer(
  container: HTMLElement,
  options: AnnouncerOptions = {}
): LiveRegionAnnouncer {
  const region = container.ownerDocument.createElement('div');
  region.setAttribute('role', 'status');
  region.setAttribute('aria-live', 'polite');
  region.setAttribute('aria-atomic', 'true');
  Object.assign(region.style, {
    position: 'absolute',
    width: '1px',
    height: '1px',
    padding: '0',
    margin: '-1px',
    overflow: 'hidden',
    clipPath: 'inset(50%)',
    whiteSpace: 'nowrap',
    border: '0'
  });
  container.appendChild(region);
  const announcer = createDebouncedAnnouncer(text => {
    region.textContent = text;
  }, options);
  return {
    ...announcer,
    destroy() {
      announcer.destroy();
      region.remove();
    }
  };
}
