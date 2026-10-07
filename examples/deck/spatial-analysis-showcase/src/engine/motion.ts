// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Whether the reader asked the system for reduced motion (`prefers-reduced-motion: reduce`).
 * Camera flights, ground cross-fades and other transitions become jump cuts when it is true.
 * Read on every call (not cached) because the preference can change while the page is open.
 */
export function prefersReducedMotion(): boolean {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch {
    return false;
  }
}
