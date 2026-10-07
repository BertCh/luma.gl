// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CHAPTERS} from './chapters';
import type {AnyScene} from './scene';

type SceneModule = {default: AnyScene};

/**
 * Every `src/scenes/<folder>/<id>.scene.ts` is a scene. There is no registry to edit: adding a
 * file adds the scene. Modules load lazily; keep a scene file light (metadata, options, story) and
 * `import()` heavy compute code inside `create` so the gallery loads quickly.
 */
const sceneModules = import.meta.glob<SceneModule>('./*/*.scene.ts');

const sceneLoaders = new Map<string, () => Promise<SceneModule>>();
for (const [path, load] of Object.entries(sceneModules)) {
  const id = path.replace(/^.*\//, '').replace(/\.scene\.ts$/, '');
  sceneLoaders.set(id, load);
}

let scenesPromise: Promise<AnyScene[]> | null = null;

/** Ids of every scene file, without loading them. */
export function getSceneIds(): string[] {
  return [...sceneLoaders.keys()];
}

/** Loads every scene definition, sorted by chapter order then `order`, then title. */
export function loadScenes(): Promise<AnyScene[]> {
  scenesPromise ??= Promise.all(
    [...sceneLoaders.entries()].map(async ([id, load]) => {
      const scene = (await load()).default;
      if (scene.id !== id) {
        // biome-ignore lint/suspicious/noConsole: authoring mistakes must be visible to scene builders
        console.warn(`Scene file "${id}.scene.ts" declares id "${scene.id}"; they must match.`);
      }
      if (!CHAPTERS.some(chapter => chapter.id === scene.chapter)) {
        // biome-ignore lint/suspicious/noConsole: authoring mistakes must be visible to scene builders
        console.warn(`Scene "${scene.id}" uses unknown chapter "${scene.chapter}".`);
      }
      return scene;
    })
  ).then(scenes =>
    scenes.sort((a, b) => {
      const chapterOrder =
        CHAPTERS.findIndex(chapter => chapter.id === a.chapter) -
        CHAPTERS.findIndex(chapter => chapter.id === b.chapter);
      return chapterOrder || a.order - b.order || a.title.localeCompare(b.title);
    })
  );
  return scenesPromise;
}

/** Loads one scene by id, or `undefined`. */
export async function loadScene(id: string): Promise<AnyScene | undefined> {
  const load = sceneLoaders.get(id);
  return load ? (await load()).default : undefined;
}

/** Scenes whose `contributors` include `name`. */
export async function findScenesByContributor(name: string): Promise<AnyScene[]> {
  return (await loadScenes()).filter(scene => scene.contributors.includes(name));
}

/** Scenes that use a dataset id. */
export async function findScenesByDataset(id: string): Promise<AnyScene[]> {
  return (await loadScenes()).filter(scene => scene.datasets.some(ref => ref.id === id));
}
