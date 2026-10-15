// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import tableOfContents from '../../../../../docs/table-of-contents.json';
import type {LensTwinFile} from './lens-directive';

/** Markdown source with its path relative to docs/. */
export type LensCatalogPage = {path: string; text: string};
/** One result card, with a canonical documentation URL. */
export type LensCatalogTile = {
  scenario: string;
  element: string;
  summary: string;
  path: string;
  href: string;
  thumbnail?: LensTwinFile;
};
/** One page's result cards in authored order. */
export type LensCatalogFamily = {family: string; tiles: LensCatalogTile[]};

/** Validates the catalog's sole optional attribute. */
export function parseLensCatalogAttributes(
  attributes: Record<string, unknown> = {}
): string | undefined {
  for (const name of Object.keys(attributes)) {
    if (name !== 'entry') throw new Error(`Unknown lens-catalog attribute: ${name}`);
  }
  if (
    attributes.entry !== undefined &&
    (typeof attributes.entry !== 'string' || !/^[a-z0-9-]+$/.test(attributes.entry))
  ) {
    throw new Error('Invalid lens-catalog entry; expected kebab-case.');
  }
  return attributes.entry as string | undefined;
}

function parseAttributes(source: string): Record<string, string> {
  const attributes: Record<string, string> = Object.create(null);
  const pattern = /([A-Za-z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`{}]+))(?:\s+|$)/y;
  let position = /^\s*/.exec(source)![0].length;
  while (position < source.length) {
    pattern.lastIndex = position;
    const match = pattern.exec(source);
    if (!match) throw new Error(`Invalid lens attribute list at column ${position + 1}`);
    const [, name, doubleQuoted, singleQuoted, unquoted] = match;
    if (Object.hasOwn(attributes, name)) throw new Error(`Duplicate lens attribute: ${name}`);
    if (name === 'caption' && unquoted !== undefined)
      throw new Error('Lens caption must be quoted');
    attributes[name] = doubleQuoted ?? singleQuoted ?? unquoted;
    position = pattern.lastIndex;
  }
  return attributes;
}

function readVisibleLines(source: string): string[] {
  let fence: string | undefined;
  let comment = false;
  return source.split(/\r?\n/).map(line => {
    if (fence) {
      const closing = /^ {0,3}(`{3,}|~{3,})\s*$/.exec(line)?.[1];
      if (closing?.[0] === fence[0] && closing.length >= fence.length) fence = undefined;
      return '';
    }
    const opening = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
    if (opening && (opening[1][0] !== '`' || !opening[2].includes('`'))) {
      fence = opening[1];
      return '';
    }
    if (comment) {
      if (line.includes('-->')) comment = false;
      return '';
    }
    if (line.includes('<!--') && !line.includes('-->')) {
      comment = true;
      return '';
    }
    return line;
  });
}

function getPageOrder(value: unknown, paths: string[] = []): string[] {
  if (typeof value === 'string') paths.push(value);
  else if (Array.isArray(value)) for (const child of value) getPageOrder(child, paths);
  else if (value && typeof value === 'object') {
    const category = value as {items?: unknown; id?: unknown};
    if (category.id) getPageOrder(category.id, paths);
    if (category.items) getPageOrder(category.items, paths);
  }
  return paths;
}

function getSummary(line: string): string {
  const summary = line
    .trim()
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/[`*_]/g, '');
  if (summary.length <= 120) return summary;
  const prefix = summary.slice(0, 119);
  const boundary = prefix.lastIndexOf(' ');
  return `${boundary < 0 ? '' : prefix.slice(0, boundary)}…`;
}

/** Builds a catalog from authored result lenses, excluding fenced examples. */
export function buildLensCatalogModel(
  pages: LensCatalogPage[],
  twins: Map<string, LensTwinFile>,
  entry?: string
): LensCatalogFamily[] {
  const order = getPageOrder(tableOfContents);
  const rank = (path: string) => {
    const position = order.indexOf(path.replace(/^docs\//, '').replace(/\.md$/, ''));
    return position < 0 ? Number.MAX_SAFE_INTEGER : position;
  };
  const families: LensCatalogFamily[] = [];
  for (const page of [...pages].sort(
    (left, right) => rank(left.path) - rank(right.path) || left.path.localeCompare(right.path)
  )) {
    const frontMatter = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(page.text);
    const title = frontMatter?.[1].match(/^title:\s*(.*?)\s*$/m)?.[1];
    const lines = readVisibleLines(page.text.slice(frontMatter?.[0].length ?? 0));
    const family = (title ?? lines.find(line => /^# /.test(line))?.slice(2) ?? page.path)
      .replace(/^['"]|['"]$/g, '')
      .replace(/`/g, '');
    const tiles: LensCatalogTile[] = [];
    let headingName = '';
    let markerName = '';
    let summary = '';
    let inSection = false;
    for (const [lineIndex, line] of lines.entries()) {
      const heading = /^ {0,3}(#{1,3})\s+(.*)$/.exec(line);
      if (heading) {
        inSection = heading[1].length === 3;
        headingName = inSection ? (/`([A-Za-z_][\w.]*)`/.exec(heading[2])?.[1] ?? '') : '';
        markerName = '';
        const following = lines.slice(lineIndex + 1);
        const nextHeading = following.findIndex(candidate =>
          /^ {0,3}#{1,3}(?:\s|$)/.test(candidate)
        );
        const sectionLines = nextHeading < 0 ? following : following.slice(0, nextHeading);
        summary = inSection
          ? getSummary(
              sectionLines.find(
                candidate =>
                  candidate.trim() && !/^\s*(::|<!--|\{\/\*|\||#|<|[-*_]{3,}\s*$)/.test(candidate)
              ) ?? ''
            )
          : '';
        continue;
      }
      if (inSection) {
        const markers = [...line.matchAll(/<!--\s*lens:([A-Za-z0-9_-]+)\s*-->/g)];
        if (markers.length) markerName = markers.at(-1)![1];
        if (!summary && line.trim() && !/^\s*(::|<!--|\{\/\*|\||#|<|[-*_]{3,}\s*$)/.test(line)) {
          summary = getSummary(line);
        }
      }
      const directive = /^ {0,3}::lens\{(.*)\}\s*$/.exec(line);
      if (!directive) continue;
      let attributes: Record<string, string>;
      try {
        attributes = parseAttributes(directive[1]);
      } catch (error) {
        throw new Error(`${page.path}:${lineIndex + 1}: ${String(error)}`);
      }
      const scenario = attributes.scenario;
      if (attributes.rung !== 'result' || !scenario || (entry && scenario.split('.')[0] !== entry))
        continue;
      const route = `/docs/${page.path
        .replace(/^docs\//, '')
        .replace(/\.md$/, '')
        .replace(/(?:^|\/)README$/, '')}`;
      tiles.push({
        scenario,
        element: markerName || headingName || scenario.split('.')[1],
        summary,
        path: page.path,
        href: `${route}#lens-${scenario}-result`,
        thumbnail: twins.get(`${scenario}~result~thumbnail`)
      });
    }
    if (tiles.length) families.push({family, tiles});
  }
  return families;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Renders the host-independent catalog; a resolver may remove unavailable page links. */
export function renderLensCatalogHtml(
  families: LensCatalogFamily[],
  entry?: string,
  resolveHref: (tile: LensCatalogTile) => string | null = tile => tile.href
): string {
  if (!families.length)
    return '<p class="lens-catalog lens-catalog--empty">No catalog entries yet.</p>';
  return `<nav class="lens-catalog" aria-label="Catalog${entry ? ` of ${escapeHtml(entry)}` : ''}">${families
    .map(
      ({family, tiles}) =>
        `<section class="lens-catalog__family"><h3 class="lens-catalog__family-title">${escapeHtml(family)}</h3><ul class="lens-catalog__grid" role="list">${tiles
          .map(tile => {
            const href = resolveHref(tile);
            const tag = href ? 'a' : 'div';
            return `<li><${tag} class="lens-catalog__tile"${href ? ` href="${escapeHtml(href)}"` : ''} aria-label="${escapeHtml(`${tile.element} — ${tile.summary}`)}"><span class="lens-catalog__thumb" aria-hidden="true">${tile.thumbnail?.figure?.html ?? '<span class="lens-catalog__thumb--pending"></span>'}</span><code class="lens-catalog__name">${escapeHtml(tile.element)}</code><span class="lens-catalog__summary">${escapeHtml(tile.summary)}</span></${tag}></li>`;
          })
          .join('')}</ul></section>`
    )
    .join('')}</nav>`;
}

let catalogPages: LensCatalogPage[] = [];
let resolveCatalogHref: (tile: LensCatalogTile) => string | null = () => null;

/** Supplies the already-loaded reference sources and the host's existing route resolver. */
export function setLensCatalogPages(pages: LensCatalogPage[]): void {
  catalogPages = pages;
}

/** Installs the markdown host's route resolver without introducing a second link index. */
export function setLensCatalogResolver(
  resolveHref: (tile: LensCatalogTile) => string | null
): void {
  resolveCatalogHref = resolveHref;
}

/** Renders the current bundled reference catalog using the existing twin map. */
export function renderReferenceLensCatalog(
  twins: Map<string, LensTwinFile>,
  entry?: string
): string {
  return renderLensCatalogHtml(
    buildLensCatalogModel(catalogPages, twins, entry),
    entry,
    resolveCatalogHref
  );
}
