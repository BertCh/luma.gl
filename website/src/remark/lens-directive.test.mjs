// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import remarkLensDirective from './lens-directive.js';
import builders from './lens-twin-html.js';

const {toHtml} = await import('hast-util-to-html').catch(() => ({}));
const SCENARIO = 'gpu-spatial-analysis.local-moran';
const TWIN = {
  schema: 'luma.lens-twin/1',
  scenario: SCENARIO,
  rung: 'datum',
  datum: 0,
  iteration: 0,
  caption: 'Generated caption <with text>',
  description: 'Local Moran values & neighbors',
  keyNumber: {label: 'Local Moran I', value: '0.42', unit: 'score'},
  figure: {html: '<svg viewBox="0 0 10 10"><path d="M0 0L10 10"/></svg>', width: 10, height: 10},
  receipt: {
    columns: ['Source', 'Value'],
    rows: [
      ['GPU', '0.42'],
      ['CPU', ''],
      ['Formula', '0.42'],
      ['Adapter', 'Test GPU']
    ],
    provenance: ['readback', 'replay', 'derived', 'metadata']
  },
  formula: {tex: '\\term{x} = 0.42', ascii: 'x = 0.42', derivation: 'derived'},
  provenance: {
    adapter: 'Test GPU',
    variant: 'reference',
    date: '2026-10-14',
    runId: 'test',
    observerEffects: ['readback', 'timing']
  }
};

function createFixture(context, twin = TWIN) {
  const docsDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'lens-directive-'));
  context.after(() => fs.rmSync(docsDirectory, {recursive: true, force: true}));
  const twinsDirectory = path.join(docsDirectory, 'lens-records', 'twins');
  fs.mkdirSync(twinsDirectory, {recursive: true});
  const twinPath = path.join(twinsDirectory, `${SCENARIO}~datum~d0~i0.json`);
  if (twin) {
    fs.writeFileSync(twinPath, JSON.stringify(twin));
  }
  return {docsDirectory, twinPath, transform: remarkLensDirective({docsDirectory})};
}

function createTree(attributes = {}) {
  return {
    type: 'root',
    children: [
      {
        type: 'leafDirective',
        name: 'lens',
        attributes: {scenario: SCENARIO, rung: 'datum', datum: '0', iteration: '0', ...attributes},
        children: [],
        position: {start: {line: 7, column: 1, offset: 0}, end: {line: 7, column: 90, offset: 89}}
      }
    ]
  };
}

function getFigure(tree) {
  const {hName, hProperties, hChildren} = tree.children[0].data;
  return {type: 'element', tagName: hName, properties: hProperties, children: hChildren};
}

function findElements(node, tagName) {
  return [
    ...(node.tagName === tagName ? [node] : []),
    ...(node.children || []).flatMap(child => findElements(child, tagName))
  ];
}

function getText(node) {
  return node.type === 'text' ? node.value : (node.children || []).map(getText).join('');
}

test('recorded twin produces figure, caption, receipt, SVG and accessible KaTeX HAST', async context => {
  const {transform} = createFixture(context);
  const tree = createTree();
  await transform(tree, {path: '/docs/example.md'});
  const figure = getFigure(tree);
  assert.equal(figure.tagName, 'figure');
  assert.deepEqual(figure.properties.className, ['lens-twin']);
  assert.equal(figure.properties.dataLensKey, `${SCENARIO}~datum~d0~i0`);
  assert.equal(figure.properties.id, `lens-${SCENARIO}-datum-d0-i0`);
  const caption = getText(findElements(figure, 'figcaption')[0]);
  assert.match(caption, /Generated caption <with text>/);
  assert.match(caption, /Local Moran I: 0.42 score/);
  assert.match(caption, /Recorded on Test GPU, reference, 2026-10-14; readback, timing/);
  assert.equal(findElements(figure, 'table').length, 1);
  assert.equal(findElements(figure, 'svg').length, 1);
  assert.equal(findElements(figure, 'math').length, 1);
  assert.ok(
    findElements(figure, 'span').some(node => node.properties.className?.includes('lens-term'))
  );
  assert.ok(findElements(figure, 'td').some(node => getText(node) === '—'));
  assert.deepEqual(
    findElements(figure, 'tr')
      .slice(1)
      .map(node => node.properties.dataProvenance),
    TWIN.receipt.provenance
  );
  assert.equal(findElements(figure, 'ladder-lens')[0].properties.recorded, 'true');
  assert.equal(findElements(figure, 'ladder-lens')[0].properties.bind, 'page');
  assert.equal(findElements(figure, 'button')[0].properties.hidden, true);
  if (toHtml) {
    const html = toHtml(figure);
    assert.match(html, /<math/);
    assert.doesNotMatch(html, /<with text>/);
    assert.match(html, /<ladder-lens[^>]* datum="0" iteration="0" recorded="true"/);
  }
  assert.equal(tree.children[0].type, 'leafDirective');
});

test('missing twin remains pending even when a scenario record exists', async context => {
  const {docsDirectory, transform} = createFixture(context, null);
  for (const hasRecord of [false, true]) {
    if (hasRecord) {
      fs.writeFileSync(path.join(docsDirectory, 'lens-records', `${SCENARIO}.json`), '{}');
    }
    const tree = createTree();
    await transform(tree, {path: '/docs/pending.md'});
    const figure = getFigure(tree);
    assert.deepEqual(figure.properties.className, ['lens-twin', 'lens-twin--pending']);
    assert.match(getText(figure), /Record pending: no recorded run/);
    assert.match(getText(figure), /How gpu-spatial-analysis.local-moran computes \(datum\)/);
    assert.match(getText(figure), /Key number: pending/);
    assert.equal(findElements(figure, 'ladder-lens')[0].properties.recorded, 'false');
    assert.equal(findElements(figure, 'table').length, 0);
    assert.equal(findElements(figure, 'math').length, 0);
  }
});

test('invalid attributes report the source path and line', async context => {
  const {transform} = createFixture(context);
  for (const attributes of [
    {rung: 'invalid'},
    {scenario: '../escape'},
    {unknown: 'value'},
    {datum: '-1'},
    {iteration: '1.5'},
    {datum: '9007199254740992'},
    {bind: null}
  ]) {
    await assert.rejects(
      transform(createTree(attributes), {path: '/docs/invalid.md'}),
      /\/docs\/invalid\.md:7:/
    );
  }
});

test('author captions override generated and pending captions, including empty captions', async context => {
  const {transform, twinPath} = createFixture(context);
  for (const pending of [false, true]) {
    if (pending) {
      fs.unlinkSync(twinPath);
    }
    for (const caption of ['Author <caption> & details', '']) {
      const tree = createTree({caption});
      await transform(tree, {path: '/docs/caption.md'});
      const figure = getFigure(tree);
      const captionNode = findElements(figure, 'figcaption')[0].children[0];
      assert.equal(getText(captionNode), caption);
      if (pending) {
        assert.equal(findElements(figure, 'ladder-lens')[0].properties.ariaLabel, caption);
      }
    }
  }
});

test('optional recorded parts and selectors are omitted', async context => {
  const twin = {...TWIN, figure: undefined, formula: undefined, receipt: undefined};
  const {transform} = createFixture(context, twin);
  const tree = createTree();
  await transform(tree, {path: '/docs/optional.md'});
  const figure = getFigure(tree);
  for (const tagName of ['table', 'svg', 'math']) {
    assert.equal(findElements(figure, tagName).length, 0);
  }
  const directive = builders.parseLensAttributes({scenario: SCENARIO, rung: 'scope'});
  assert.equal(builders.twinKey(directive), `${SCENARIO}~scope`);
  const pendingLens = findElements(builders.buildPendingTwin(directive), 'ladder-lens')[0];
  assert.equal(Object.hasOwn(pendingLens.properties, 'datum'), false);
  assert.equal(Object.hasOwn(pendingLens.properties, 'iteration'), false);
});

test('reads twins again after edits without a process cache', async context => {
  const {transform, twinPath} = createFixture(context);
  await transform(createTree(), {path: '/docs/uncached.md'});
  fs.writeFileSync(twinPath, JSON.stringify({...TWIN, caption: 'Updated twin'}));
  const tree = createTree();
  await transform(tree, {path: '/docs/uncached.md'});
  assert.match(getText(getFigure(tree)), /Updated twin/);
});

test('ignores unrelated and non-leaf directives while walking nested content', async context => {
  const {transform} = createFixture(context, null);
  const leaf = createTree().children[0];
  const unrelated = {...leaf, name: 'other'};
  const container = {...leaf, type: 'containerDirective'};
  const tree = {
    type: 'root',
    children: [{type: 'blockquote', children: [leaf]}, unrelated, container]
  };
  await transform(tree, {path: '/docs/nested.md'});
  assert.equal(leaf.data.hName, 'figure');
  assert.equal(unrelated.data, undefined);
  assert.equal(container.data, undefined);
});

test('result examples follow the caption and pending results retain their aspect class', async context => {
  const {docsDirectory} = createFixture(context);
  const directive = builders.parseLensAttributes({scenario: SCENARIO, rung: 'result'});
  const recorded = builders.buildRecordedTwin(
    directive,
    {
      ...TWIN,
      formula: undefined,
      example: {code: 'const output = "<ready>"; // &', language: 'ts', lineCount: 1}
    },
    () => '',
    () => []
  );
  assert.equal(recorded.properties.dataRung, 'result');
  assert.equal(recorded.children[1].tagName, 'figcaption');
  assert.equal(recorded.children[2].tagName, 'pre');
  assert.deepEqual(recorded.children[2].properties.className, ['lens-example']);
  assert.equal(
    recorded.children[2].children[0].children[0].value,
    'const output = "<ready>"; // &'
  );
  const pending = builders.buildPendingTwin(directive);
  assert.ok(
    findElements(pending, 'div')[0].properties.className.includes('lens-twin__figure--result')
  );
  assert.equal(findElements(pending, 'pre').length, 0);
  const {default: collapse} = await import('./lens-how-collapse.js');
  const {default: catalog} = await import('./lens-catalog.js');
  const tree = {
    type: 'root',
    children: [
      {type: 'heading', depth: 4, children: [{type: 'text', value: 'How it computes'}]},
      createTree().children[0],
      {type: 'thematicBreak'},
      {type: 'leafDirective', name: 'lens-catalog', attributes: {}, children: []}
    ]
  };
  collapse()(tree);
  assert.equal(tree.children[0].data.hName, 'details');
  assert.equal(tree.children[0].children[0].data.hChildren[1].children[0].value, ' · 1 view');
  assert.equal(tree.children[1].type, 'thematicBreak');
  await catalog({docsDirectory})(tree, {path: 'cards.md'});
  assert.deepEqual(tree.children[2].data.hProperties.className, [
    'lens-catalog',
    'lens-catalog--empty'
  ]);
});

test('catalog scans docs and thumbnails and reports authoring errors with source locations', async context => {
  const {docsDirectory} = createFixture(context);
  const {default: catalog} = await import('./lens-catalog.js');
  const directory = path.join(docsDirectory, 'api-reference', 'experimental');
  fs.mkdirSync(directory, {recursive: true});
  fs.writeFileSync(
    path.join(directory, 'weights.md'),
    '# Weights\n### `GPUWeights`\nConnect inputs.\n::lens{scenario=test.weights rung=result}\n'
  );
  fs.writeFileSync(
    path.join(docsDirectory, 'lens-records', 'twins', 'test.weights~result~thumbnail.json'),
    JSON.stringify(TWIN)
  );
  const node = {
    type: 'leafDirective',
    name: 'lens-catalog',
    attributes: {entry: 'test'},
    children: [],
    position: {start: {line: 9}}
  };
  const tree = {type: 'root', children: [node]};
  await catalog({docsDirectory})(tree, {path: 'catalog.md'});
  const figure = getFigure(tree);
  assert.equal(figure.tagName, 'nav');
  assert.match(getText(figure), /WeightsGPUWeightsConnect inputs/);
  assert.equal(findElements(figure, 'svg').length, 1);
  assert.equal(
    findElements(figure, 'a')[0].properties.href,
    '/docs/api-reference/experimental/weights#lens-test.weights-result'
  );
  node.attributes = {entry: 'Invalid'};
  await assert.rejects(
    catalog({docsDirectory})(tree, {path: 'catalog.md'}),
    /catalog.md:9: Invalid lens-catalog entry/
  );
});

test('MDX twins and catalogs compile and render generated HTML as opaque component props', async context => {
  const {evaluate} = await import('@mdx-js/mdx');
  const {default: remarkDirective} = await import('remark-directive');
  const {default: collapse} = await import('./lens-how-collapse.js');
  const {default: catalog} = await import('./lens-catalog.js');
  const {fromHtml} = await import('hast-util-from-html');
  const runtime = await import('react/jsx-runtime');
  const {createElement} = await import('react');
  const {renderToStaticMarkup} = await import('react-dom/server');
  const {docsDirectory} = createFixture(context, {
    ...TWIN,
    figure: {
      html: '<!-- recorded SVG --><svg viewBox="0 0 10 10"><text>{value} &amp; "quote"</text></svg>'
    }
  });
  const directory = path.join(docsDirectory, 'api-reference', 'experimental');
  fs.mkdirSync(directory, {recursive: true});
  fs.writeFileSync(
    path.join(directory, 'weights.md'),
    '# Weights\n### Element\n{/* lens:GPUWeights */}\nConnect inputs.\n::lens{scenario=test.weights rung=result}\n'
  );
  fs.writeFileSync(
    path.join(docsDirectory, 'lens-records', 'twins', 'test.weights~result~thumbnail.json'),
    JSON.stringify({
      ...TWIN,
      figure: {html: '<!-- thumbnail --><svg><text>{thumbnail}</text></svg>'}
    })
  );
  // Exercise .md imports and .mdx syntax without imports, as well as an empty catalog.
  for (const [filename, preamble] of [
    ['example.md', "import {Fragment} from 'react';\n\n<Fragment />\n\n"],
    ['example.mdx', '']
  ]) {
    const markup = [];
    const {default: Content} = await evaluate(
      {
        path: filename,
        value: `${preamble}{/* lens:GPUExample */}\n\n#### How it computes\n\n::lens{scenario=${SCENARIO} rung=datum datum=0 iteration=0}\n\n::lens{scenario=test.missing rung=result}\n\n---\n\n::lens-catalog{entry=test}\n\n::lens-catalog{entry=missing}`
      },
      {
        ...runtime,
        format: 'mdx',
        baseUrl: import.meta.url,
        remarkPlugins: [
          remarkDirective,
          [remarkLensDirective, {docsDirectory}],
          collapse,
          [catalog, {docsDirectory}]
        ]
      }
    );
    const html = renderToStaticMarkup(
      createElement(Content, {
        components: {
          LensHtml: ({html}) => {
            markup.push(html);
            return createElement('div', {dangerouslySetInnerHTML: {__html: html}});
          }
        }
      })
    );
    assert.equal(markup.length, 4);
    assert.match(html, /How it computes.*2 views/);
    assert.match(markup[0], /<!-- ?recorded SVG ?-->/);
    assert.match(markup[0], /<math/);
    assert.match(markup[0], /<ladder-lens/);
    assert.match(markup[1], /lens-twin--pending/);
    assert.match(markup[2], /<!-- thumbnail -->/);
    assert.match(markup[2], /GPUWeights/);
    assert.match(markup[3], /lens-catalog--empty/);
    const recorded = fromHtml(markup[0], {fragment: true});
    assert.equal(getText(findElements(recorded, 'text')[0]), '{value} & "quote"');
    assert.equal(findElements(recorded, 'table').length, 1);
  }
});
