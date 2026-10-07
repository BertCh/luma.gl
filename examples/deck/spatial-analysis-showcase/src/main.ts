// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import './styles.css';
import {createDataCatalog} from './data/catalog';
import {DeckHost} from './engine/deck-host';
import {renderAbout, renderGuidePage} from './shell/about-view';
import {renderData} from './shell/data-view';
import {h, icon} from './shell/dom';
import {renderHome} from './shell/home-view';
import {renderReference} from './shell/reference-view';
import {getStoryHash, navigate, startRouter, type Route} from './shell/router';
import {StoryView} from './shell/story-view';
import {getTheme, initializeTheme, onThemeChange, toggleTheme} from './shell/theme';

declare global {
  // eslint-disable-next-line no-var
  var spatialAnalysisShowcase:
    | {
        sceneId: string | null;
        ready: boolean;
        error: string | null;
        rebuildCount: number;
        selectScene: (id: string) => void;
      }
    | undefined;
}

const searchParameters = new URLSearchParams(window.location.search);
const forceSynthetic = searchParameters.get('data') === 'synthetic';
const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu);

// `?scene=<id>` deep link for headless captures: jump to the story when no hash is given.
const sceneParameter = searchParameters.get('scene');
if (sceneParameter && (!window.location.hash || window.location.hash === '#/')) {
  history.replaceState(
    null,
    '',
    `${window.location.pathname}${window.location.search}${getStoryHash(sceneParameter)}`
  );
}

initializeTheme();
const catalog = createDataCatalog({forceSynthetic});
let host: DeckHost | null = null;

function getHost(): DeckHost {
  if (!host) {
    host = new DeckHost({theme: getTheme(), initialView: {longitude: 0, latitude: 20, zoom: 2}});
    host.onStateChange = publishState;
  }
  return host;
}

const hooks = {
  sceneId: null as string | null,
  ready: false,
  error: null as string | null,
  rebuildCount: 0,
  selectScene: (id: string) => navigate(getStoryHash(id))
};
globalThis.spatialAnalysisShowcase = hooks;

function publishState(): void {
  const state = host?.state;
  hooks.sceneId = state?.sceneId ?? null;
  hooks.ready = Boolean(state?.ready);
  hooks.error = state?.error ?? null;
  hooks.rebuildCount = state?.rebuildCount ?? 0;
  const {dataset} = document.body;
  if (hooks.sceneId) dataset['showcaseScene'] = hooks.sceneId;
  else delete dataset['showcaseScene'];
  if (hooks.ready) dataset['showcaseReady'] = 'true';
  else delete dataset['showcaseReady'];
  if (hooks.error) dataset['showcaseError'] = hooks.error;
  else delete dataset['showcaseError'];
}

// Compile-time rebuilds are counted by new compiled-graph objects; keep the test hooks current.
setInterval(() => {
  if (host?.state.sceneId) {
    host.pollRebuilds();
    publishState();
  }
}, 500);

// Shell.
const navLinks = [
  {name: 'home', label: 'Stories', href: '#/'},
  {name: 'reference', label: 'Reference', href: '#/reference'},
  {name: 'data', label: 'Data', href: '#/data'},
  {name: 'about', label: 'About', href: '#/about'}
] as const;
const nav = h(
  'nav',
  {class: 'site-nav', 'aria-label': 'Main'},
  navLinks.map(link => h('a', {href: link.href, dataset: {route: link.name}}, link.label))
);
const themeButton = h('button', {
  class: 'btn btn-ghost icon-button',
  type: 'button',
  'aria-label': 'Toggle light and dark theme'
});
const updateThemeButton = () =>
  themeButton.replaceChildren(icon(getTheme() === 'dark' ? 'sun' : 'moon', 18));
themeButton.addEventListener('click', toggleTheme);
updateThemeButton();
onThemeChange(updateThemeButton);
const view = h('main', {id: 'view', class: 'view'});
document.body.append(
  h(
    'header',
    {class: 'site-header'},
    h(
      'a',
      {class: 'logo', href: '#/'},
      h('span', {class: 'logo-mark', 'aria-hidden': 'true'}, icon('layers', 20)),
      h('span', {}, 'luma.gl ', h('strong', {}, 'Spatial Analysis'))
    ),
    nav,
    h('span', {class: 'spacer'}),
    themeButton
  ),
  view
);

const storyView = new StoryView({getHost, catalog, hasWebGPU});
let activeKind: Route['name'] | null = null;
let renderToken = 0;

function leaveStory(): void {
  storyView.unmount();
  document.body.classList.remove('is-story');
  publishState();
}

startRouter(async route => {
  const token = ++renderToken;
  for (const link of nav.querySelectorAll('a')) {
    const kind =
      route.name === 'story' || route.name === 'home'
        ? 'home'
        : route.name === 'guide'
          ? 'about'
          : route.name;
    link.classList.toggle('is-active', link.dataset['route'] === kind);
  }
  if (route.name === 'story') {
    document.body.classList.add('is-story');
    if (activeKind === 'story' && storyView.sceneId === route.sceneId) {
      storyView.update(route);
      return;
    }
    if (activeKind === 'story') storyView.unmount();
    activeKind = 'story';
    await storyView.mount(view, route);
    return;
  }
  if (activeKind === 'story') leaveStory();
  activeKind = route.name;
  window.scrollTo(0, 0);
  switch (route.name) {
    case 'home':
      await renderHome(
        view,
        new URLSearchParams(window.location.hash.split('?')[1] ?? '').get('chapter')
      );
      break;
    case 'reference':
      await renderReference(view, route.entry);
      break;
    case 'guide':
      renderGuidePage(view, route.guideId);
      break;
    case 'data':
      await renderData(view);
      break;
    case 'about':
      renderAbout(view);
      break;
  }
  if (token !== renderToken) return;
});
