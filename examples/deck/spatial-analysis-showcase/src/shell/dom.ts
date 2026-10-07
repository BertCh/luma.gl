// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

type Child = Node | string | number | null | undefined | false | readonly Child[];

type Attributes = {
  class?: string;
  /** Sets `innerHTML`. Only pass trusted, build-time or markdown-it rendered strings. */
  html?: string;
  style?: Partial<CSSStyleDeclaration> | string;
  dataset?: Record<string, string>;
  on?: {[Event in keyof HTMLElementEventMap]?: (event: HTMLElementEventMap[Event]) => void};
  [attribute: string]: unknown;
};

/**
 * Creates an element: `h('button', {class: 'btn', on: {click}}, 'Label')`. Boolean attributes are
 * set as properties, `false`/`null` children are skipped, arrays are flattened.
 */
export function h<Tag extends keyof HTMLElementTagNameMap>(
  tag: Tag,
  attributes?: Attributes | null,
  ...children: Child[]
): HTMLElementTagNameMap[Tag] {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes ?? {})) {
    if (value === undefined || value === null || value === false) continue;
    if (name === 'class') element.className = String(value);
    else if (name === 'html') element.innerHTML = String(value);
    else if (name === 'style') {
      if (typeof value === 'string') element.setAttribute('style', value);
      else Object.assign(element.style, value);
    } else if (name === 'dataset') Object.assign(element.dataset, value as Record<string, string>);
    else if (name === 'on') {
      for (const [eventName, handler] of Object.entries(value as Record<string, EventListener>)) {
        element.addEventListener(eventName, handler);
      }
    } else if (name in element && typeof value !== 'string') {
      (element as unknown as Record<string, unknown>)[name] = value;
    } else element.setAttribute(name, value === true ? '' : String(value));
  }
  appendChildren(element, children);
  return element;
}

function appendChildren(parent: ParentNode, children: readonly Child[]): void {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    if (Array.isArray(child)) appendChildren(parent, child);
    else parent.append(child instanceof Node ? child : String(child));
  }
}

/** Removes all children of a node. */
export function clearElement(element: Element): void {
  element.replaceChildren();
}

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';

/** Icon paths (24x24, stroke based). */
const ICONS = {
  sun: 'M12 4V2m0 20v-2m8-8h2M2 12h2m13.66-5.66 1.41-1.41M4.93 19.07l1.41-1.41m0-11.32L4.93 4.93m14.14 14.14-1.41-1.41M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z',
  moon: 'M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z',
  chevronLeft: 'm15 18-6-6 6-6',
  chevronRight: 'm9 18 6-6-6-6',
  chevronDown: 'm6 9 6 6 6-6',
  copy: 'M9 9h11v11H9zM5 15H4V4h11v1',
  check: 'm5 12 5 5L20 7',
  reset: 'M3 12a9 9 0 1 0 3-6.7L3 8m0-5v5h5',
  external: 'M14 4h6v6m0-6L10 14M20 14v6H4V4h6',
  search: 'm21 21-4.3-4.3M17 10.5a6.5 6.5 0 1 1-13 0 6.5 6.5 0 0 1 13 0z',
  layers: 'm12 3 9 5-9 5-9-5 9-5zm-9 9 9 5 9-5m-18 4 9 5 9-5',
  close: 'M6 6l12 12M18 6 6 18'
} as const;

/** Name of an icon in the built-in set. */
export type IconName = keyof typeof ICONS;

/** Creates a 16px stroke icon. */
export function icon(name: IconName, size = 16): SVGSVGElement {
  const svg = document.createElementNS(SVG_NAMESPACE, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', String(size));
  svg.setAttribute('height', String(size));
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS(SVG_NAMESPACE, 'path');
  path.setAttribute('d', ICONS[name]);
  svg.append(path);
  return svg;
}

/** Copies text to the clipboard, falling back to a hidden textarea. Resolves true on success. */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = h('textarea', {value: text, style: 'position:fixed;opacity:0'});
    document.body.append(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    return ok;
  }
}

/** Formats a byte count, for example `2.1 MB`. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/** Escapes text for use in HTML. */
export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
