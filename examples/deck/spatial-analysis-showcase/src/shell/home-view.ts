// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CHAPTERS} from '../scenes/chapters';
import {loadScenes} from '../scenes/registry';
import type {AnyScene} from '../scenes/scene';
import {h} from './dom';
import {createThumbnail} from './thumbnails';
import {getReferenceHash, getStoryHash} from './router';

/** Home: hero plus the chapter gallery. */
export async function renderHome(root: HTMLElement, focusChapter: string | null): Promise<void> {
  root.replaceChildren(h('div', {class: 'page-loading'}, 'Loading stories…'));
  const scenes = await loadScenes();
  const total = scenes.length;
  const validatedStart = scenes.find(scene => scene.id === 'rainfall-interpolation') ?? scenes[0];
  const hero = h(
    'section',
    {class: 'hero'},
    h('p', {class: 'eyebrow'}, 'luma.gl · WebGPU'),
    h('h1', {}, 'Spatial analysis on the GPU'),
    h(
      'p',
      {class: 'lede'},
      'Start with a spatial question, follow the evidence across maps and linked charts, then test the assumptions with live controls. Each story names its data, method and limits while GPU command graphs feed the analysis directly into the view.'
    ),
    h(
      'div',
      {class: 'hero-actions'},
      validatedStart
        ? h(
            'a',
            {class: 'btn btn-primary', href: getStoryHash(validatedStart.id)},
            'Open a validated example'
          )
        : null,
      h('a', {class: 'btn', href: getReferenceHash()}, 'Browse the reference'),
      h(
        'a',
        {class: 'btn btn-ghost', href: '#/guide/how-contributors-work'},
        'How contributors work'
      )
    ),
    h(
      'p',
      {class: 'muted small'},
      `${total} ${total === 1 ? 'story' : 'stories'} in ${CHAPTERS.filter(chapter => scenes.some(scene => scene.chapter === chapter.id)).length} of ${CHAPTERS.length} chapters`
    )
  );

  const chapters = CHAPTERS.map(chapter => {
    const chapterScenes = scenes.filter(scene => scene.chapter === chapter.id);
    return h(
      'section',
      {class: 'chapter', id: `chapter-${chapter.id}`},
      h(
        'div',
        {class: 'chapter-head'},
        h('h2', {}, chapter.title),
        h('p', {class: 'muted'}, chapter.summary)
      ),
      chapterScenes.length
        ? h('div', {class: 'card-grid'}, chapterScenes.map(renderSceneCard))
        : h('p', {class: 'coming-soon'}, 'Stories for this chapter are on the way.')
    );
  });
  root.replaceChildren(h('div', {class: 'page'}, hero, ...chapters));
  if (focusChapter) document.getElementById(`chapter-${focusChapter}`)?.scrollIntoView();
}

function renderSceneCard(scene: AnyScene): HTMLElement {
  const chartCount = (scene.readouts ?? []).filter(readout => readout.kind === 'chart').length;
  const storyLabel = `${scene.story.length}-step story`;
  const chartLabel = `${chartCount} ${chartCount === 1 ? 'chart' : 'charts'}`;
  const datasetLabel = `${scene.datasets.length} ${scene.datasets.length === 1 ? 'dataset' : 'datasets'}`;
  return h(
    'a',
    {class: 'scene-card', href: getStoryHash(scene.id)},
    h(
      'div',
      {class: 'thumb', 'data-scene': scene.id, 'aria-hidden': 'true'},
      createThumbnail(scene.chapter, scene.id)
    ),
    h(
      'div',
      {class: 'scene-card-body'},
      h('h3', {}, scene.title),
      h('p', {}, scene.summary),
      h(
        'p',
        {class: 'scene-card-meta', 'aria-label': `${storyLabel}, ${chartLabel}, ${datasetLabel}`},
        h('span', {}, storyLabel),
        h('span', {}, chartLabel),
        h('span', {}, datasetLabel)
      ),
      h(
        'div',
        {class: 'chips'},
        scene.contributors.slice(0, 4).map(name => h('span', {class: 'chip chip-static'}, name)),
        scene.contributors.length > 4
          ? h('span', {class: 'chip chip-static'}, `+${scene.contributors.length - 4}`)
          : null
      )
    )
  );
}
