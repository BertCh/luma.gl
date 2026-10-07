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
  const hero = h(
    'section',
    {class: 'hero'},
    h('p', {class: 'eyebrow'}, 'luma.gl · WebGPU'),
    h('h1', {}, 'Spatial analysis on the GPU'),
    h(
      'p',
      {class: 'lede'},
      'Real datasets, guided stories and live controls for luma.gl’s analysis contributors. Every map you see is computed in WebGPU compute shaders and drawn straight from GPU buffers, with no readback in the frame loop.'
    ),
    h(
      'div',
      {class: 'hero-actions'},
      total
        ? h(
            'a',
            {class: 'btn btn-primary', href: getStoryHash(scenes[0].id)},
            'Start with the first story'
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
