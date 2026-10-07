// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Checks every scene's story steps against its options and readouts: each step lists explicit
 * `controls`, every listed control and readout exists, a step lists at most five controls, and the
 * first listed control is not disabled when the step opens (a later one may be, when an earlier
 * listed control enables it). Scenes are bundled with esbuild; package imports are stubbed, so a
 * scene that statically imports runtime code may fail to load and is reported as `??`.
 *
 * Usage: `node scripts/check-story-steps.mjs [chapter ...]` (default: every chapter). Exit code 1
 * on problems.
 */
import {readdirSync, readFileSync} from 'node:fs';
import {dirname, join, relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';

const MAXIMUM_STEP_CONTROLS = 5;
const SCENES_DIRECTORY = join(dirname(fileURLToPath(import.meta.url)), '../src/scenes');
const STUB_SOURCE = `const stub = new Proxy(function () {}, {
  get: (target, key) => (key === '__esModule' ? true : key === 'prototype' ? {} : stub),
  construct: () => stub,
  apply: () => stub
});
module.exports = stub;`;

const stubPlugin = {
  name: 'stub-packages',
  setup(builder) {
    builder.onResolve({filter: /\?raw$/}, args => ({
      path: join(args.resolveDir, args.path.replace('?raw', '')),
      namespace: 'raw'
    }));
    builder.onLoad({filter: /.*/, namespace: 'raw'}, args => ({
      contents: readFileSync(args.path, 'utf8'),
      loader: 'text'
    }));
    builder.onResolve({filter: /^[^./]/}, args => ({path: args.path, namespace: 'stub'}));
    builder.onLoad({filter: /.*/, namespace: 'stub'}, () => ({contents: STUB_SOURCE, loader: 'js'}));
  }
};

async function loadScene(file) {
  const result = await build({
    entryPoints: [file],
    bundle: true,
    write: false,
    format: 'cjs',
    platform: 'node',
    plugins: [stubPlugin],
    logLevel: 'silent'
  });
  const module = {exports: {}};
  new Function('module', 'exports', 'require', result.outputFiles[0].text)(
    module,
    module.exports,
    () => ({})
  );
  return module.exports.default;
}

function checkScene(scene) {
  const problems = [];
  const optionIds = new Set(scene.options.map(option => option.id));
  const readoutIds = new Set((scene.readouts ?? []).map(readout => readout.id));
  const state = {};
  for (const option of scene.options) {
    if (option.kind !== 'button') state[option.id] = option.default;
  }
  scene.story.forEach((step, index) => {
    Object.assign(state, step.options ?? {});
    if (!step.controls) problems.push(`step ${index + 1} "${step.id}" has no explicit controls`);
    const controls = step.controls ?? [];
    if (controls.length > MAXIMUM_STEP_CONTROLS) {
      problems.push(`step "${step.id}" lists ${controls.length} controls (max ${MAXIMUM_STEP_CONTROLS})`);
    }
    controls.forEach((id, position) => {
      const spec = scene.options.find(option => option.id === id);
      if (!spec) problems.push(`step "${step.id}" control "${id}" is not an option`);
      else if (position === 0 && spec.disabledWhen?.(state)) {
        problems.push(`step "${step.id}" control "${id}" opens disabled with no control before it`);
      }
    });
    for (const id of step.readouts ?? []) {
      if (!readoutIds.has(id)) problems.push(`step "${step.id}" readout "${id}" is not a readout`);
    }
    for (const id of Object.keys(step.options ?? {})) {
      if (!optionIds.has(id)) problems.push(`step "${step.id}" sets unknown option "${id}"`);
    }
  });
  return problems;
}

const chapters = process.argv.slice(2);
let problemCount = 0;
for (const chapter of readdirSync(SCENES_DIRECTORY, {withFileTypes: true})) {
  if (!chapter.isDirectory() || (chapters.length && !chapters.includes(chapter.name))) continue;
  const directory = join(SCENES_DIRECTORY, chapter.name);
  for (const fileName of readdirSync(directory).filter(name => name.endsWith('.scene.ts')).sort()) {
    const file = join(directory, fileName);
    const name = relative(SCENES_DIRECTORY, file);
    let scene;
    try {
      scene = await loadScene(file);
    } catch (error) {
      console.log(`??   ${name}: could not load (${String(error).slice(0, 100)}); check it by reading`);
      continue;
    }
    const problems = checkScene(scene);
    problemCount += problems.length;
    console.log(`${problems.length ? 'FAIL' : 'ok  '} ${name}`);
    for (const problem of problems) console.log(`     - ${problem}`);
  }
}
process.exit(problemCount ? 1 : 0);
