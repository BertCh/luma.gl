// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGuides, renderGuide} from '../docs/guides';
import {decorateCode, enableAnchorScroll} from './reference-view';
import {h} from './dom';

/** About page: what the showcase is, and the concept guides. */
export function renderAbout(root: HTMLElement): void {
  const guides = getGuides();
  root.replaceChildren(
    h(
      'div',
      {class: 'page narrow'},
      h('h1', {}, 'About'),
      h(
        'div',
        {class: 'prose'},
        h(
          'p',
          {},
          'This site applies luma.gl spatial-analysis contributors to observed, modeled and synthetic datasets. A contributor is a node in a GPU command graph: the graph is compiled once, parameters are updated when inputs change, and GPU-backed layers consume its output buffers.'
        ),
        h(
          'p',
          {},
          'The map uses deck.gl (WebGPU) over a MapLibre basemap. Large analysis outputs remain GPU-resident. Small asynchronous summary readbacks and scene-specific CPU preparation or post-processing are identified in the story and detail panel.'
        ),
        h(
          'p',
          {},
          'Requires a browser with WebGPU (recent Chrome or Edge, or Safari with WebGPU enabled). The reference and data pages work everywhere.'
        )
      ),
      h('h2', {}, 'Guides'),
      h(
        'ul',
        {class: 'link-list'},
        guides.map(guide =>
          h(
            'li',
            {},
            h('a', {href: `#/guide/${guide.id}`}, guide.title),
            h('span', {class: 'muted'}, ` — ${guide.summary}`)
          )
        )
      )
    )
  );
}

/** One concept guide. */
export function renderGuidePage(root: HTMLElement, guideId: string): void {
  const guides = getGuides();
  const guide = guides.find(candidate => candidate.id === guideId);
  if (!guide) {
    root.replaceChildren(
      h(
        'div',
        {class: 'page narrow'},
        h('h1', {}, 'Guide not found'),
        h('a', {class: 'btn', href: '#/about'}, 'All guides')
      )
    );
    return;
  }
  const body = h('div', {class: 'prose doc', html: renderGuide(guide)});
  decorateCode(body);
  enableAnchorScroll(body);
  root.replaceChildren(
    h(
      'div',
      {class: 'page narrow'},
      h(
        'nav',
        {class: 'breadcrumb'},
        h('a', {href: '#/about'}, 'Guides'),
        h('span', {}, '/'),
        h('span', {}, guide.title)
      ),
      h('h1', {}, guide.title),
      body,
      h(
        'nav',
        {class: 'guide-nav'},
        guides
          .filter(other => other.id !== guide.id)
          .map(other => h('a', {class: 'btn btn-small', href: `#/guide/${other.id}`}, other.title))
      )
    )
  );
}
