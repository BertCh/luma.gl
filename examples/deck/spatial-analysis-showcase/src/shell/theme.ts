// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'spatial-analysis-showcase-theme';
const listeners = new Set<(theme: Theme) => void>();
let currentTheme: Theme = 'dark';

function readStoredTheme(): Theme | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored === 'light' || stored === 'dark' ? stored : null;
  } catch {
    return null;
  }
}

/** Resolves the initial theme (stored choice, else the system preference) and applies it. */
export function initializeTheme(): Theme {
  const system: Theme = window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light';
  applyTheme(readStoredTheme() ?? system);
  // Follow the system preference until the user makes an explicit choice.
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
    if (!readStoredTheme()) applyTheme(event.matches ? 'dark' : 'light');
  });
  return currentTheme;
}

function applyTheme(theme: Theme): void {
  currentTheme = theme;
  document.documentElement.dataset['theme'] = theme;
  for (const listener of listeners) listener(theme);
}

/** The theme currently shown. */
export function getTheme(): Theme {
  return currentTheme;
}

/** Switches theme and remembers the choice. */
export function setTheme(theme: Theme): void {
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Storage can be blocked; the theme still applies for this page.
  }
  applyTheme(theme);
}

/** Toggles between light and dark. */
export function toggleTheme(): void {
  setTheme(currentTheme === 'dark' ? 'light' : 'dark');
}

/** Subscribes to theme changes. Returns an unsubscribe function. */
export function onThemeChange(listener: (theme: Theme) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
