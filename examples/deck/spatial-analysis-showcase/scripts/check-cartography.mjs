// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Lints the mechanical cartography rules of `src/scenes/CARTOGRAPHY-GUIDE.md` (SYNTHESIS sections 0
 * and 3.1) over every scene: the loaded scene object (basemap, furniture, steps, annotations,
 * legends for the default state) and the scene folder's source text (ramps, pitch, unicode charts).
 *
 * Findings are warnings by default so the existing scenes keep passing; a chapter that has been
 * redesigned runs it with `--strict`, which turns every finding into a failure.
 *
 * Usage: `node scripts/check-cartography.mjs [--strict] [chapter ...]`.
 *
 * Escape hatch: a rule can be waived in a scene file with a comment that names the rule and says
 * why, for example `// cartography-allow: viridis (the "before" map of the rainbow lesson)` or
 * `// cartography-allow: pitch (satellite altitudes are real z)`. Waivers are listed in the output.
 */
import {readdirSync, readFileSync} from 'node:fs';
import {dirname, join, relative} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';

const SCENES_DIRECTORY = join(dirname(fileURLToPath(import.meta.url)), '../src/scenes');
const MAXIMUM_NOTES = 3;
const MAXIMUM_PLACE_LABELS = 6;
const MAXIMUM_FIRST_STEP_LABELS = 12;
const STEP_RANGE = [4, 7];
const MAXIMUM_STEP_READOUTS = 4;
const PLACE_KINDS = new Set(['point', 'area', 'water', 'landform']);
const RAINBOW_RAMPS = ['viridis', 'spectral', 'jet', 'rainbow', 'turbo', 'plasma'];
const DARK_ONLY_RAMPS = ['magma', 'inferno'];
const DIVERGING_RAMPS = new Set([
  'diverging',
  'rdbu',
  'rdylbu',
  'spectral',
  'brbg',
  'puor',
  'piyg',
  'prgn',
  'vik',
  'roma',
  'berlin',
  'vanimo'
]);
const UNICODE_CHART_CHARACTERS = /[▁▂▃▄▅▆▇█▏▎▍▌▋▊▉]/;

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

/** Source text of the scene file and of the chapter files it is likely to use (same id prefix). */
function readSceneSources(directory, fileName) {
  const id = fileName.replace(/\.scene\.ts$/, '');
  return readdirSync(directory)
    .filter(name => name.startsWith(id) && /\.(ts|md)$/.test(name))
    .map(name => ({name, text: readFileSync(join(directory, name), 'utf8')}));
}

/** Waived rule names found in `// cartography-allow: <rule> (<reason>)` comments. */
function getWaivers(sources) {
  const waivers = new Map();
  for (const {text} of sources) {
    for (const match of text.matchAll(/cartography-allow:\s*([\w-]+)\s*(\(([^)]*)\))?/g)) {
      waivers.set(match[1], match[3] ?? '');
    }
  }
  return waivers;
}

function getDefaultState(scene) {
  const state = {};
  for (const option of scene.options) {
    if (option.kind !== 'button' && option.kind !== 'preset') state[option.id] = option.default;
  }
  return state;
}

function countAnnotations(annotations) {
  let notes = 0;
  let places = 0;
  for (const annotation of annotations ?? []) {
    if (annotation?.kind === 'note') notes++;
    else if (PLACE_KINDS.has(annotation?.kind)) places++;
  }
  return {notes, places};
}

function checkScene(scene, sources) {
  const findings = [];
  const waivers = getWaivers(sources);
  const add = (rule, message) => {
    if (!waivers.has(rule)) findings.push(`[${rule}] ${message}`);
  };

  // Rule 1 / 3.1-4: the ground is declared on purpose.
  if (!scene.basemap) add('basemap', 'scene declares no basemap (use a ground preset)');
  // 3.1-6: title cartouche and credit on every step.
  const furniture = scene.furniture ?? {};
  if (!furniture.title) add('cartouche', 'scene has no title cartouche (furniture.title)');
  if (!furniture.credit) add('credit', 'scene has no source credit (furniture.credit)');

  // Rule 8 / 3.1-10: 5-6 steps.
  const stepCount = scene.story.length;
  if (stepCount < STEP_RANGE[0] || stepCount > STEP_RANGE[1]) {
    add('steps', `${stepCount} steps (aim for 5-6)`);
  }

  // Rule 11: pitch 0 unless the layer has real z.
  const pitched = [scene.initialView, ...scene.story.map(step => step.camera ?? {})].some(
    view => (view?.pitch ?? 0) > 0
  );
  if (pitched) add('pitch', 'camera is pitched: only for layers with real z (waive with a reason)');

  const sceneAnnotations = countAnnotations(scene.annotations);
  scene.story.forEach((step, index) => {
    // Rule 6: at most 3 finding notes and 6 place labels per step.
    const stepAnnotations = countAnnotations(step.annotations);
    const notes = sceneAnnotations.notes + stepAnnotations.notes;
    const places = sceneAnnotations.places + stepAnnotations.places;
    const placeLimit = index === 0 ? MAXIMUM_FIRST_STEP_LABELS : MAXIMUM_PLACE_LABELS;
    if (notes > MAXIMUM_NOTES) add('annotations', `step "${step.id}" has ${notes} notes (max 3)`);
    if (places > placeLimit) {
      add('annotations', `step "${step.id}" has ${places} place labels (max ${placeLimit})`);
    }
    // Rule 10 / 3.1-11: 3-4 readouts per step.
    const readouts = step.readouts?.length ?? 0;
    if (readouts > MAXIMUM_STEP_READOUTS) {
      add('readouts', `step "${step.id}" shows ${readouts} readouts (max 4)`);
    }
    // Rule 8: no text-only limits step.
    if (/^limits?$|caveats?$/.test(step.id) && !step.options && !step.annotations && !step.camera) {
      add('limits-step', `step "${step.id}" looks like a text-only limits step`);
    }
  });
  const withoutHeadline = scene.story.filter(step => !step.headline).map(step => step.id);
  if (withoutHeadline.length) {
    add('headline', `${withoutHeadline.length} of ${stepCount} steps have no finding headline`);
  }

  // Legends of the default state.
  let legends = [];
  try {
    legends = scene.legends(getDefaultState(scene), {}) ?? [];
  } catch {
    // Legends that need GPU data may throw on empty data; nothing to check then.
  }
  for (const legend of legends) {
    if (legend.kind === 'classes') {
      if (!legend.unit && !legend.table?.unit) add('legend-unit', `classes legend "${legend.title}" has no unit`);
      if (!legend.noData && !legend.table?.noData) {
        add('legend-no-data', `classes legend "${legend.title}" has no no-data entry`);
      }
    }
    if (legend.kind === 'ramp') {
      if (!legend.unit && !legend.labels) add('legend-unit', `ramp legend "${legend.title}" has no unit`);
      if (DIVERGING_RAMPS.has(legend.ramp) && legend.midpoint === undefined) {
        add('legend-midpoint', `diverging legend "${legend.title}" declares no midpoint`);
      }
    }
  }

  // Source-level rules.
  for (const {name, text} of sources) {
    // Rule 3: no rainbow or viridis as a default.
    for (const ramp of RAINBOW_RAMPS) {
      const pattern = new RegExp(`(colormap|ramp|default)\\s*:\\s*'${ramp}'`);
      if (pattern.test(text)) add('viridis', `${name} uses '${ramp}' as a ramp (rule 3)`);
    }
    for (const ramp of DARK_ONLY_RAMPS) {
      const pattern = new RegExp(`(colormap|ramp)\\s*:\\s*'${ramp}'`);
      const lightGround = /style:\s*'(positron|voyager)'|ground\('paper/.test(text);
      if (pattern.test(text) && lightGround) {
        add('dark-only-ramp', `${name} uses '${ramp}' with a light ground (rule 3)`);
      }
    }
    // 3.1-11: no unicode sparklines or block characters.
    if (UNICODE_CHART_CHARACTERS.test(text)) {
      add('unicode-chart', `${name} draws a unicode sparkline or block chart`);
    }
  }
  return {findings, waivers};
}

const argumentsList = process.argv.slice(2);
const strict = argumentsList.includes('--strict');
const chapters = argumentsList.filter(argument => !argument.startsWith('--'));
let findingCount = 0;
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
    const {findings, waivers} = checkScene(scene, readSceneSources(directory, fileName));
    findingCount += findings.length;
    const status = findings.length ? (strict ? 'FAIL' : 'warn') : 'ok  ';
    console.log(`${status} ${name}`);
    for (const finding of findings) console.log(`     - ${finding}`);
    for (const [rule, reason] of waivers) console.log(`     ~ waived ${rule}${reason ? `: ${reason}` : ''}`);
  }
}
if (findingCount) {
  console.log(`\n${findingCount} finding(s)${strict ? '' : ' (warnings; pass --strict to fail on them)'}`);
}
process.exit(strict && findingCount ? 1 : 0);
