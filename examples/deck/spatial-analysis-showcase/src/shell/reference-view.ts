// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  loadReferenceIndex,
  renderDocOverview,
  renderSectionBody,
  resolveEntry,
  type ReferenceDoc,
  type ReferenceIndex,
  type ReferenceSection
} from '../docs/reference-index';
import {getGuides} from '../docs/guides';
import {findScenesByContributor} from '../scenes/registry';
import {copyText, h, icon} from './dom';
import {getReferenceHash, getStoryHash} from './router';

/** Reference index (searchable) and reference pages. */
export async function renderReference(root: HTMLElement, entry: string | null): Promise<void> {
  root.replaceChildren(h('div', {class: 'page-loading'}, 'Loading reference…'));
  const index = await loadReferenceIndex();
  if (!entry) return renderIndex(root, index);
  const resolved = resolveEntry(index, entry);
  if (!resolved) {
    root.replaceChildren(
      h(
        'div',
        {class: 'page narrow'},
        h('h1', {}, 'Not in the reference'),
        h('p', {}, `There is no reference entry called "${entry}".`),
        h('a', {class: 'btn', href: getReferenceHash()}, 'Open the reference index')
      )
    );
    return;
  }
  if (resolved.kind === 'section') await renderSectionPage(root, index, resolved.section, entry);
  else renderDocPage(root, index, resolved.doc);
}

function renderIndex(root: HTMLElement, index: ReferenceIndex): void {
  const search = h('input', {
    type: 'search',
    class: 'search-input',
    placeholder: 'Search classes, or a tool you know (turf, PostGIS, PySAL…)',
    'aria-label': 'Search the reference'
  });
  const results = h('div', {class: 'reference-groups'});
  const guides = getGuides();

  const renderGroups = (query: string) => {
    const needle = query.trim().toLowerCase();
    const groups = index.docs
      .map(doc => {
        const entries = doc.apiSections.filter(
          section =>
            !needle ||
            section.names.some(name => name.toLowerCase().includes(needle)) ||
            section.heading.toLowerCase().includes(needle) ||
            section.summary.toLowerCase().includes(needle)
        );
        return {doc, entries};
      })
      .filter(group => group.entries.length > 0 || (!needle && group.doc.apiSections.length === 0));
    const children: HTMLElement[] = groups.map(({doc, entries}) =>
      h(
        'section',
        {class: 'reference-group'},
        h('h2', {}, h('a', {href: getReferenceHash(doc.id)}, doc.title)),
        h('p', {class: 'muted small'}, doc.entryPoint),
        entries.length
          ? h(
              'ul',
              {class: 'name-list'},
              entries.map(section =>
                h(
                  'li',
                  {},
                  h(
                    'a',
                    {href: getReferenceHash(section.names[0])},
                    h('span', {class: 'name'}, section.names.join(', ')),
                    h('span', {class: 'muted'}, section.summary)
                  )
                )
              )
            )
          : h(
              'p',
              {},
              h('a', {class: 'btn btn-small', href: getReferenceHash(doc.id)}, 'Open the page')
            )
      )
    );
    if (needle) {
      const crossReference = index.docs.find(
        doc => doc.id === 'gpu-spatial-analysis-cross-reference'
      );
      const rows = crossReference ? findCrossReferenceRows(crossReference, needle) : [];
      if (rows.length) {
        children.push(
          h(
            'section',
            {class: 'reference-group'},
            h('h2', {}, 'In the tool cross-reference'),
            h(
              'ul',
              {class: 'name-list'},
              rows
                .slice(0, 40)
                .map(row =>
                  h(
                    'li',
                    {},
                    h(
                      'a',
                      {href: getReferenceHash('gpu-spatial-analysis-cross-reference')},
                      h('span', {class: 'muted'}, row)
                    )
                  )
                )
            ),
            rows.length > 40
              ? h(
                  'p',
                  {class: 'muted small'},
                  `${rows.length - 40} more rows on the cross-reference page.`
                )
              : null
          )
        );
      }
    }
    if (!children.length) children.push(h('p', {class: 'muted'}, `Nothing matches “${query}”.`));
    results.replaceChildren(...children);
  };
  search.addEventListener('input', () => renderGroups(search.value));
  renderGroups('');

  root.replaceChildren(
    h(
      'div',
      {class: 'page'},
      h('h1', {}, 'Reference'),
      h(
        'p',
        {class: 'lede'},
        'The API docs for the analysis contributors, indexed by class. Each page lists the stories that feature it.'
      ),
      guides.length
        ? h(
            'div',
            {class: 'guide-strip'},
            guides.map(guide =>
              h(
                'a',
                {class: 'guide-link', href: `#/guide/${guide.id}`},
                h('strong', {}, guide.title),
                h('span', {class: 'muted small'}, guide.summary)
              )
            )
          )
        : null,
      h('div', {class: 'search-bar'}, icon('search'), search),
      results
    )
  );
}

/** Table rows of the cross-reference doc whose text contains `needle`. */
function findCrossReferenceRows(doc: ReferenceDoc, needle: string): string[] {
  const rows: string[] = [];
  for (const section of doc.plainSections) {
    for (const line of section.body.split('\n')) {
      if (line.startsWith('|') && !/^\|\s*-/.test(line) && line.toLowerCase().includes(needle)) {
        rows.push(line.replace(/[|`*]/g, ' ').replace(/\s+/g, ' ').trim());
      }
    }
  }
  return rows;
}

async function renderSectionPage(
  root: HTMLElement,
  index: ReferenceIndex,
  section: ReferenceSection,
  requested: string
): Promise<void> {
  const doc = index.docs.find(candidate => candidate.id === section.docId)!;
  const body = h('div', {class: 'prose doc', html: renderSectionBody(section)});
  decorateCode(body);
  const featured = h(
    'section',
    {class: 'card featured'},
    h('h3', {class: 'section-title'}, 'Featured in'),
    h('p', {class: 'muted small'}, 'Loading…')
  );
  const siblings = doc.apiSections;
  const position = siblings.indexOf(section);
  root.replaceChildren(
    h(
      'div',
      {class: 'page doc-page'},
      h(
        'nav',
        {class: 'breadcrumb'},
        h('a', {href: getReferenceHash()}, 'Reference'),
        h('span', {}, '/'),
        h('a', {href: getReferenceHash(doc.id)}, doc.title)
      ),
      h(
        'h1',
        {},
        section.names
          .map(name => h('code', {class: name === requested ? 'current-name' : ''}, name))
          .reduce<(HTMLElement | string)[]>(
            (all, element, i) => (i ? [...all, ' ', element] : [element]),
            []
          )
      ),
      h('p', {class: 'muted small'}, `${doc.entryPoint} · ${section.group}`),
      featured,
      body,
      h(
        'div',
        {class: 'step-nav'},
        position > 0
          ? h(
              'a',
              {class: 'btn', href: getReferenceHash(siblings[position - 1].names[0])},
              icon('chevronLeft'),
              siblings[position - 1].names[0]
            )
          : h('span', {}),
        position < siblings.length - 1
          ? h(
              'a',
              {class: 'btn', href: getReferenceHash(siblings[position + 1].names[0])},
              siblings[position + 1].names[0],
              icon('chevronRight')
            )
          : h('span', {})
      )
    )
  );
  enableAnchorScroll(body);
  const scenes = (await Promise.all(section.names.map(findScenesByContributor))).flat();
  const unique = [...new Map(scenes.map(scene => [scene.id, scene])).values()];
  featured.replaceChildren(
    h('h3', {class: 'section-title'}, 'Featured in'),
    unique.length
      ? h(
          'ul',
          {class: 'link-list'},
          unique.map(scene =>
            h(
              'li',
              {},
              h('a', {href: getStoryHash(scene.id)}, scene.title),
              h('span', {class: 'muted'}, ` — ${scene.summary}`)
            )
          )
        )
      : h('p', {class: 'muted small'}, 'No story features this yet.')
  );
}

function renderDocPage(root: HTMLElement, index: ReferenceIndex, doc: ReferenceDoc): void {
  const body = h('div', {class: 'prose doc', html: renderDocOverview(doc)});
  decorateCode(body);
  const filter = h('input', {
    type: 'search',
    class: 'search-input',
    placeholder: 'Filter this page',
    'aria-label': 'Filter tables on this page'
  });
  filter.addEventListener('input', () => {
    const needle = filter.value.trim().toLowerCase();
    for (const row of body.querySelectorAll('tbody tr')) {
      (row as HTMLElement).hidden =
        Boolean(needle) && !(row.textContent ?? '').toLowerCase().includes(needle);
    }
  });
  root.replaceChildren(
    h(
      'div',
      {class: 'page doc-page'},
      h(
        'nav',
        {class: 'breadcrumb'},
        h('a', {href: getReferenceHash()}, 'Reference'),
        h('span', {}, '/'),
        h('span', {}, doc.title)
      ),
      h('p', {class: 'muted small'}, doc.entryPoint),
      doc.id.endsWith('cross-reference')
        ? h('div', {class: 'search-bar'}, icon('search'), filter)
        : null,
      body,
      doc.apiSections.length
        ? h(
            'section',
            {},
            h('h2', {}, 'API'),
            h(
              'ul',
              {class: 'name-list'},
              doc.apiSections.map(section =>
                h(
                  'li',
                  {},
                  h(
                    'a',
                    {href: getReferenceHash(section.names[0])},
                    h('span', {class: 'name'}, section.names.join(', ')),
                    h('span', {class: 'muted'}, section.summary)
                  )
                )
              )
            )
          )
        : null
    )
  );
  enableAnchorScroll(body);
  void index;
}

/** Adds copy buttons to code blocks. */
export function decorateCode(container: HTMLElement): void {
  for (const block of container.querySelectorAll('.code-block')) {
    const code = block.querySelector('code')?.textContent ?? '';
    const button = h(
      'button',
      {class: 'btn btn-ghost btn-small copy-button', type: 'button', 'aria-label': 'Copy code'},
      icon('copy', 14),
      'Copy'
    );
    button.addEventListener('click', async () => {
      if (await copyText(code)) {
        button.replaceChildren(icon('check', 14), 'Copied');
        setTimeout(() => button.replaceChildren(icon('copy', 14), 'Copy'), 1400);
      }
    });
    block.prepend(h('div', {class: 'code-toolbar'}, button));
  }
}

/** In-page `#heading` links scroll instead of changing the route. */
export function enableAnchorScroll(container: HTMLElement): void {
  container.addEventListener('click', event => {
    const anchor = (event.target as HTMLElement).closest('a');
    const href = anchor?.getAttribute('href');
    if (href && href.startsWith('#') && !href.startsWith('#/')) {
      event.preventDefault();
      container.querySelector(`[id="${CSS.escape(href.slice(1))}"]`)?.scrollIntoView();
    }
  });
}
