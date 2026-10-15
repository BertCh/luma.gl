// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {describe, expect, it, vi} from 'vitest';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import LensHtml from '../../../../website/src/components/docs/lens-html';
import {
  buildLensCatalogModel,
  parseLensCatalogAttributes,
  renderLensCatalogHtml,
  setLensCatalogPages,
  type LensCatalogPage
} from '../../../../examples/deck/spatial-analysis-showcase/src/docs/lens-catalog';
import {
  parseLensAttributes,
  renderLensTwinHtml,
  type LensTwinFile
} from '../../../../examples/deck/spatial-analysis-showcase/src/docs/lens-directive';
import {activateFramePlayers} from '../../../../examples/deck/spatial-analysis-showcase/src/docs/lens-frame-player';
import {
  renderMarkdown,
  setLinkIndex
} from '../../../../examples/deck/spatial-analysis-showcase/src/shell/markdown';

const requireFromRoot = createRequire(new URL('../../../../package.json', import.meta.url));
const websiteCatalog = requireFromRoot('./website/src/remark/lens-catalog.js');
const websiteCollapse = requireFromRoot('./website/src/remark/lens-how-collapse.js');
const websiteDirective = requireFromRoot('./website/src/remark/lens-directive.js');
const websiteTwin = requireFromRoot('./website/src/remark/lens-twin-html.js');

it('preserves generated HTML verbatim in the registered MDX lens component', () => {
  const html =
    '<figure><!-- recorded --><svg><text>{value}</text></svg><math><mi>x</mi></math></figure>';
  expect(renderToStaticMarkup(createElement(LensHtml, {html}))).toBe(
    `<div class="lens-html">${html}</div>`
  );
});

async function loadModule(name: string) {
  return import(/* @vite-ignore */ pathToFileURL(requireFromRoot.resolve(name)).href);
}

const pages: LensCatalogPage[] = [
  {
    path: 'api-reference/experimental/gpu-spatial-analysis/operators-statistics.md',
    text: `# Statistics
### \`GPUScore\`
Score the input.
::lens{scenario=spatial.score rung=result}
### No identifier
::lens{scenario=network.walk rung=result}
Walk the network.
`
  },
  {
    path: 'api-reference/experimental/gpu-spatial-analysis/operators-weights.md',
    text: `---
title: 'Weights'
---
# Ignored title
### \`GPUFallback\`
Connect nearby inputs & preserve "weights".
<!-- lens:GPUContiguityWeights -->
::lens{scenario=spatial.weights rung=result}
<!-- lens:GPUOtherWeights -->
::lens{scenario=spatial.other rung=result}
::lens{scenario=spatial.weights rung=datum}
\`\`\`md
### \`GPUFake\`
::lens{scenario=spatial.fake rung=result}
\`\`\`
~~~md
::lens{scenario=spatial.tilde rung=result}
~~~
### \`GPUHeading\`
Use a heading when the previous marker belongs to another section.
::lens{scenario=spatial.heading rung=result}
`
  }
];
const thumbnail: LensTwinFile = {
  schema: 'luma.lens-twin/1',
  scenario: 'spatial.weights',
  rung: 'result',
  size: 'thumbnail',
  caption: 'Weights',
  description: 'Weights',
  keyNumber: {label: 'Count', value: '3'},
  figure: {
    html: '<svg viewBox="0 0 160 96"><path d="M0 0L1 1"></path></svg>',
    width: 160,
    height: 96
  },
  provenance: {adapter: 'Test', date: '2026-10-14', runId: 'test', observerEffects: []}
};
const twins = new Map([['spatial.weights~result~thumbnail', thumbnail]]);

describe('element card host contracts', () => {
  it('adds result markers, pending aspect boxes and escaped examples after the caption', async () => {
    const [{fromHtml}, {toHtml}] = await Promise.all([
      loadModule('hast-util-from-html'),
      loadModule('hast-util-to-html')
    ]);
    const directive = parseLensAttributes({scenario: 'spatial.weights', rung: 'result'});
    const example = {
      code: 'const output = "<ready>";\n// & result',
      language: 'ts' as const,
      lineCount: 2
    };
    const recorded = {...thumbnail, example};
    for (const twin of [recorded, null]) {
      const html = renderLensTwinHtml(directive, twin);
      const website = twin
        ? websiteTwin.buildRecordedTwin(
            directive,
            twin,
            () => '',
            (source: string) => fromHtml(source, {fragment: true}).children
          )
        : websiteTwin.buildPendingTwin(directive);
      expect(html).toContain('data-lens-key="spatial.weights~result" data-rung="result"');
      const normalize = (source: string) => toHtml(fromHtml(source, {fragment: true}));
      expect(normalize(html)).toBe(normalize(toHtml(website)));
      if (twin) {
        expect(html).toContain('</figcaption><pre class="lens-example"><code class="language-ts">');
        expect(html).toContain('&lt;ready&gt;');
        expect(html).not.toContain('<ready>');
      } else {
        expect(html).toContain('lens-twin__figure--pending lens-twin__figure--result');
        expect(html).not.toContain('lens-example');
      }
    }
  });

  it('matches collapsed mdast→hast structure and preserves the next section', async () => {
    const [
      {unified},
      {default: remarkParse},
      {default: remarkDirective},
      {toHast},
      {toHtml},
      {fromHtml}
    ] = await Promise.all([
      loadModule('unified'),
      loadModule('remark-parse'),
      loadModule('remark-directive'),
      loadModule('mdast-util-to-hast'),
      loadModule('hast-util-to-html'),
      loadModule('hast-util-from-html')
    ]);
    const source =
      '### X\n\n::lens{scenario=test.card rung=result}\n\nProse.\n\n#### How it computes\n\n::lens{scenario=test.card rung=datum}\n\n::lens{scenario=test.card rung=formula}\n\n### Y\n\nAfter.';
    const processor = unified().use(remarkParse).use(remarkDirective);
    const tree = processor.parse(source);
    // Production appends collapse after lens: leafDirective nodes remain countable.
    await websiteDirective({docsDirectory: '/nonexistent-card-fixtures'})(tree, {path: 'test.md'});
    websiteCollapse()(tree);
    const normalize = (html: string) => {
      const root = fromHtml(html, {fragment: true});
      function clean(node: {
        position?: unknown;
        properties?: {id?: string};
        children?: unknown[];
        type: string;
        value?: string;
      }): unknown {
        delete node.position;
        if (node.properties?.id && !node.properties.id.startsWith('lens-'))
          delete node.properties.id;
        if (node.children)
          node.children = node.children.map(child => clean(child as typeof node)).filter(Boolean);
        if (node.type === 'text' && !node.value?.trim()) return null;
        return node;
      }
      return clean(root);
    };
    const websiteHtml = toHtml(toHast(tree));
    const showcaseHtml = renderMarkdown(source);
    expect(normalize(showcaseHtml)).toEqual(normalize(websiteHtml));
    expect(showcaseHtml).toContain(' · 2 views</span></summary>');
    expect(showcaseHtml).toContain('</details><h3 id="y">');
    expect(showcaseHtml).not.toContain('<details open');
  });

  it('closes disclosures at rules, sibling headings and parent ends, counting nested lenses', async () => {
    for (const [body, count] of [
      ['Prose.', ''],
      ['::lens{scenario=test.one rung=datum}', ' · 1 view'],
      [
        '> ::lens{scenario=test.one rung=datum}\n>\n> ::lens{scenario=test.two rung=datum}',
        ' · 2 views'
      ]
    ]) {
      const html = renderMarkdown(`#### How it computes\n\n${body}\n\n---\n\nAfter`);
      expect(html).toContain('</details><hr>');
      if (count) expect(html).toContain(`${count}</span>`);
      else expect(html).not.toContain('lens-how__count');
    }
    expect(renderMarkdown('> #### How it computes\n>\n> Body\n\nOutside')).toContain(
      '</details></blockquote>'
    );
    expect(renderMarkdown('#### **How it computes**\n\nBody')).not.toContain('<details');
    expect(renderMarkdown('### How it computes\n\nBody')).not.toContain('<details');
    expect(renderMarkdown('#### How it computes\n\n##### Kept\n\nBody')).toContain('<h5');
  });
});

describe('catalog host parity', () => {
  it('extracts names, summaries, thumbnails and sidebar order with byte-identical HTML', () => {
    const model = buildLensCatalogModel(pages, twins);
    expect(model.map(family => family.family)).toEqual(['Weights', 'Statistics']);
    expect(model[0].tiles.map(tile => tile.element)).toEqual([
      'GPUContiguityWeights',
      'GPUOtherWeights',
      'GPUHeading'
    ]);
    expect(model[1].tiles.map(tile => tile.element)).toEqual(['GPUScore', 'walk']);
    expect(model[1].tiles[1].summary).toBe('Walk the network.');
    expect(model[0].tiles[0].thumbnail).toBe(thumbnail);
    expect(model[0].tiles[0].href).toBe(
      '/docs/api-reference/experimental/gpu-spatial-analysis/operators-weights#lens-spatial.weights-result'
    );
    expect(model).toEqual(websiteCatalog.buildLensCatalogModel(pages, twins));
    expect(renderLensCatalogHtml(model)).toBe(websiteCatalog.renderLensCatalogHtml(model));
    expect(renderLensCatalogHtml(model)).toContain(thumbnail.figure!.html);
    expect(renderLensCatalogHtml(model)).toContain('lens-catalog__thumb--pending');
    expect(renderLensCatalogHtml(model)).toContain('inputs &amp; preserve &quot;weights&quot;');
    const filtered = buildLensCatalogModel(pages, twins, 'network');
    expect(filtered.flatMap(family => family.tiles).map(tile => tile.scenario)).toEqual([
      'network.walk'
    ]);
    expect(renderLensCatalogHtml(filtered, 'network')).toContain('aria-label="Catalog of network"');
  });

  it('handles empty catalogs, truncation, README routes and unresolved showcase links', () => {
    const model = buildLensCatalogModel(
      [
        {
          path: 'api-reference/experimental/demo/README.md',
          text: `# Demo\n### \`GPUExample\`\n${'Words '.repeat(30)}\n::lens{scenario=demo.example rung=result}`
        }
      ],
      new Map()
    );
    expect(model[0].tiles[0].summary.length).toBeLessThanOrEqual(120);
    expect(model[0].tiles[0].summary).toMatch(/Words…$/);
    expect(model[0].tiles[0].href).toBe(
      '/docs/api-reference/experimental/demo#lens-demo.example-result'
    );
    expect(renderLensCatalogHtml(model, undefined, () => null)).toContain(
      '<div class="lens-catalog__tile" aria-label='
    );
    expect(renderLensCatalogHtml([])).toBe(
      '<p class="lens-catalog lens-catalog--empty">No catalog entries yet.</p>'
    );
    setLensCatalogPages(pages);
    setLinkIndex({
      names: new Set(),
      namesBySlug: new Map(),
      docIdsByPath: new Map([['gpu-spatial-analysis/operators-weights', 'weights']])
    });
    try {
      const html = renderMarkdown('::lens-catalog{entry=spatial}');
      expect(html).toContain('href="#/reference/weights?anchor=lens-spatial.weights-result"');
      expect(html).toContain('<div class="lens-catalog__tile"');
      expect(renderMarkdown('::lens-catalog{unknown=x}')).toContain('Lens directive error');
      expect(renderMarkdown('::lens-catalog{entry="Bad Entry"}')).toContain('Lens directive error');
      expect(renderMarkdown('```md\n::lens-catalog{}\n```')).not.toContain('<nav');
    } finally {
      setLensCatalogPages([]);
      setLinkIndex({names: new Set(), namesBySlug: new Map(), docIdsByPath: new Map()});
    }
  });

  it('rejects malformed, duplicate and unknown attributes', () => {
    for (const attributes of [{entry: 'Bad'}, {entry: ''}, {entry: null}, {other: 'x'}]) {
      expect(() => parseLensCatalogAttributes(attributes)).toThrow();
      expect(() => websiteCatalog.parseLensCatalogAttributes(attributes)).toThrow();
    }
    expect(parseLensCatalogAttributes({})).toBeUndefined();
    expect(renderMarkdown('::lens-catalog{entry=a entry=b}')).toContain('Duplicate lens attribute');
    expect(() =>
      buildLensCatalogModel(
        [{path: 'test.md', text: '::lens{scenario=test.one rung=result rung=datum}'}],
        new Map()
      )
    ).toThrow('Duplicate lens attribute');
  });
});

// A small DOM fixture exercises delegated events and SVG attributes in the node project.
class FrameElement {
  attributes = new Map<string, string>();
  dataset: {cvFrame?: string} = {};
  textContent = '';
  disabled = false;
  isConnected = true;
  className = '';
  parent: FrameElement | null = null;
  children: FrameElement[] = [];
  listeners = new Set<(event: unknown) => void>();

  constructor(readonly kind: string) {}
  setAttribute(name: string, value: string) {
    this.attributes.set(name, value);
  }
  hasAttribute(name: string) {
    return this.attributes.has(name);
  }
  toggleAttribute(name: string, force: boolean) {
    if (force) this.attributes.set(name, '');
    else this.attributes.delete(name);
  }
  append(child: FrameElement) {
    child.parent = this;
    this.children.push(child);
  }
  contains(child: FrameElement): boolean {
    return child === this || this.children.some(item => item.contains(child));
  }
  closest(selector: string): FrameElement | null {
    if (
      (selector === 'button[data-cv-play]' && this.kind === 'button') ||
      (selector === 'figure.lens-twin' && this.kind === 'figure')
    )
      return this;
    return this.parent?.closest(selector) ?? null;
  }
  querySelectorAll(selector: string): FrameElement[] {
    const kind = selector.includes('data-cv-frame')
      ? 'frame'
      : selector.includes('data-cv-play')
        ? 'button'
        : 'none';
    return this.children.flatMap(child => [
      ...(child.kind === kind ? [child] : []),
      ...child.querySelectorAll(selector)
    ]);
  }
  querySelector(selector: string): FrameElement | null {
    if (selector === 'figcaption')
      return this.children.find(child => child.kind === 'caption') ?? null;
    if (selector === '.cv-play-status') {
      for (const child of this.children) {
        if (child.className === 'cv-play-status') return child;
        const nested = child.querySelector(selector);
        if (nested) return nested;
      }
    }
    return null;
  }
  addEventListener(type: string, listener: (event: unknown) => void) {
    if (type === 'click') this.listeners.add(listener);
  }
  removeEventListener(type: string, listener: (event: unknown) => void) {
    if (type === 'click') this.listeners.delete(listener);
  }
  getElementById(): FrameElement | null {
    return null;
  }
  get parentElement(): FrameElement | null {
    return this.parent;
  }
  scrollIntoView() {}
  matches(selector: string): boolean {
    return selector === 'details.lens-how' && this.kind === 'details';
  }
}

function createFrameFixture(reducedMotion = false, count = 3) {
  const container = new FrameElement('container');
  const figure = new FrameElement('figure');
  const button = new FrameElement('button');
  const frames = Array.from({length: count}, (_, index) => {
    const frame = new FrameElement('frame');
    frame.dataset.cvFrame = String(index * 10);
    frame.toggleAttribute('hidden', index !== count - 1);
    return frame;
  });
  // Reverse DOM order to ensure playback uses numeric frame order.
  for (const frame of [...frames].reverse()) figure.append(frame);
  figure.append(button);
  figure.append(new FrameElement('caption'));
  container.append(figure);
  vi.stubGlobal('Element', FrameElement);
  vi.stubGlobal(
    'document',
    Object.assign(container, {createElement: () => new FrameElement('span')})
  );
  vi.stubGlobal('window', {
    location: {hash: ''},
    matchMedia: () => ({matches: reducedMotion}),
    addEventListener: () => {}
  });
  vi.stubGlobal('navigator', {});
  return {
    container,
    figure,
    button,
    frames,
    click: () => {
      for (const listener of container.listeners) listener({target: button});
    },
    visible: () => frames.findIndex(frame => !frame.hasAttribute('hidden'))
  };
}

describe('recorded frame playback without a browser or GPU', () => {
  it('plays once in numeric order, stops at final on click, and clears timers on disposal', () => {
    vi.useFakeTimers();
    const fixture = createFrameFixture();
    const dispose = activateFramePlayers(fixture.container as unknown as HTMLElement);
    try {
      expect(fixture.visible()).toBe(2);
      fixture.click();
      expect(fixture.visible()).toBe(0);
      expect(fixture.button.attributes.get('aria-pressed')).toBe('true');
      vi.advanceTimersByTime(600);
      expect(fixture.visible()).toBe(1);
      expect(fixture.figure.querySelector('.cv-play-status')?.textContent).toBe('round 2 of 3');
      vi.advanceTimersByTime(600);
      expect(fixture.visible()).toBe(2);
      expect(fixture.button.attributes.get('aria-pressed')).toBe('false');
      expect(vi.getTimerCount()).toBe(0);
      fixture.click();
      fixture.click();
      expect(fixture.visible()).toBe(2);
      expect(vi.getTimerCount()).toBe(0);
      fixture.click();
      dispose();
      expect(vi.getTimerCount()).toBe(0);
      expect(fixture.container.listeners.size).toBe(0);
    } finally {
      dispose();
      vi.useRealTimers();
      vi.unstubAllGlobals();
    }
  });

  it('steps and wraps with reduced motion, and limits a longer sequence to five seconds', () => {
    vi.useFakeTimers();
    let dispose = () => {};
    try {
      const fixture = createFrameFixture(true);
      dispose = activateFramePlayers(fixture.container as unknown as HTMLElement);
      expect(fixture.button.textContent).toBe('Step ▸');
      for (const expected of [0, 1, 2, 0]) {
        fixture.click();
        expect(fixture.visible()).toBe(expected);
      }
      expect(vi.getTimerCount()).toBe(0);
      dispose();
      const longFixture = createFrameFixture(false, 12);
      dispose = activateFramePlayers(longFixture.container as unknown as HTMLElement);
      longFixture.click();
      vi.advanceTimersByTime(5000);
      expect(longFixture.visible()).toBe(11);
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      dispose();
      vi.useRealTimers();
      vi.unstubAllGlobals();
    }
  });

  it('installs the website listener once and stops its playback on route updates', async () => {
    vi.useFakeTimers();
    const fixture = createFrameFixture();
    try {
      const {default: client} = await import('../../../../website/src/client-modules/ladder-lens');
      client.onRouteDidUpdate?.({} as never);
      fixture.click();
      expect(fixture.visible()).toBe(0);
      vi.advanceTimersByTime(600);
      expect(fixture.visible()).toBe(1);
      const listeners = fixture.container.listeners.size;
      client.onRouteDidUpdate?.({} as never);
      expect(fixture.container.listeners.size).toBe(listeners);
      expect(fixture.visible()).toBe(2);
      expect(vi.getTimerCount()).toBe(0);
      class DetailsElement extends FrameElement {
        open = false;
      }
      vi.stubGlobal('HTMLDetailsElement', DetailsElement);
      const details = new DetailsElement('details');
      const target = new FrameElement('target');
      details.append(target);
      fixture.container.getElementById = () => target;
      const scroll = vi.spyOn(target, 'scrollIntoView');
      window.location.hash = '#lens-test.one-datum';
      client.onRouteDidUpdate?.({} as never);
      expect(details.open).toBe(true);
      expect(scroll).toHaveBeenCalledOnce();
    } finally {
      vi.useRealTimers();
      vi.unstubAllGlobals();
    }
  });
});
