// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Parsed hash route. */
export type Route =
  | {name: 'home'}
  | {name: 'story'; sceneId: string; stepId: string | null; options: string | null}
  | {name: 'reference'; entry: string | null}
  | {name: 'guide'; guideId: string}
  | {name: 'data'}
  | {name: 'about'};

/** Parses `location.hash` (`#/story/<scene>[/<step>]?o=<state>` and friends). */
export function parseHash(hash: string): Route {
  const trimmed = hash.replace(/^#/, '');
  const [path, query = ''] = trimmed.split('?');
  const segments = path.split('/').filter(Boolean).map(decodeURIComponent);
  const parameters = new URLSearchParams(query);
  switch (segments[0]) {
    case 'story':
      if (!segments[1]) return {name: 'home'};
      return {
        name: 'story',
        sceneId: segments[1],
        stepId: segments[2] ?? null,
        options: parameters.get('o')
      };
    case 'reference':
      return {name: 'reference', entry: segments[1] ?? null};
    case 'guide':
      return segments[1] ? {name: 'guide', guideId: segments[1]} : {name: 'about'};
    case 'data':
      return {name: 'data'};
    case 'about':
      return {name: 'about'};
    default:
      return {name: 'home'};
  }
}

/** Builds the hash of a story location. */
export function getStoryHash(sceneId: string, stepId?: string | null, options?: string): string {
  const path = `#/story/${encodeURIComponent(sceneId)}${stepId ? `/${encodeURIComponent(stepId)}` : ''}`;
  return options ? `${path}?o=${options}` : path;
}

/** Builds the hash of a reference entry. */
export function getReferenceHash(entry?: string | null): string {
  return entry ? `#/reference/${encodeURIComponent(entry)}` : '#/reference';
}

/** Navigates by setting the hash. */
export function navigate(hash: string): void {
  if (window.location.hash === hash) window.dispatchEvent(new HashChangeEvent('hashchange'));
  else window.location.hash = hash;
}

/** Rewrites the hash without a navigation (no `hashchange`), for option and step state. */
export function replaceHash(hash: string): void {
  history.replaceState(null, '', hash);
}

/** Calls `listener` for the current hash and every change. Returns an unsubscribe function. */
export function startRouter(listener: (route: Route) => void): () => void {
  const handle = () => listener(parseHash(window.location.hash));
  window.addEventListener('hashchange', handle);
  handle();
  return () => window.removeEventListener('hashchange', handle);
}
