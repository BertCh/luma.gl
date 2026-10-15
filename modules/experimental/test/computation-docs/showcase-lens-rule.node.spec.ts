// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
import {describe, expect, it} from 'vitest';
import MarkdownIt from 'markdown-it';
import katex from 'katex';
import {
  parseLensAttributeList,
  parseLensAttributes,
  registerLensDirective,
  renderLensTwinHtml,
  type LensTwinFile
} from '../../../../examples/deck/spatial-analysis-showcase/src/docs/lens-directive';
import {
  renderMarkdown,
  setLinkIndex
} from '../../../../examples/deck/spatial-analysis-showcase/src/shell/markdown';

const requireFromRoot = createRequire(new URL('../../../../package.json', import.meta.url));
const websiteHost = requireFromRoot('./website/src/remark/lens-twin-html.js');
const attributes = {
  scenario: 'gpu-spatial-analysis.local-moran',
  rung: 'datum',
  datum: '0',
  iteration: '2',
  bind: 'page'
};
const directive = parseLensAttributes(attributes);
const twin: LensTwinFile = {
  schema: 'luma.lens-twin/1',
  scenario: directive.scenario,
  rung: directive.rung,
  datum: 0,
  iteration: 2,
  caption: 'Recorded caption <with> & symbols',
  description: 'A "recorded" diagram & receipt',
  keyNumber: {label: 'Local I', value: '0.42', unit: 'units'},
  figure: {
    html: '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><circle cx="5" cy="5" r="2"></circle></svg>',
    width: 10,
    height: 10
  },
  formula: {tex: '\\term{x} = \\frac{1}{2}', ascii: 'x = 1/2', derivation: 'derived'},
  receipt: {
    columns: ['Datum', 'Value & units'],
    rows: [
      ['0', '0.42'],
      ['1', ''],
      ['2', '<unknown>'],
      ['3', 'metadata'],
      ['4', 'no provenance']
    ],
    provenance: ['readback', 'replay', 'derived', 'metadata']
  },
  provenance: {
    adapter: 'Test GPU',
    date: '2026-10-14',
    variant: 'reference',
    runId: 'test-run',
    observerEffects: ['readback', 'timing']
  }
};

function createMarkdown(): MarkdownIt {
  const markdown = new MarkdownIt();
  registerLensDirective(markdown);
  return markdown;
}

describe('showcase lens directive', () => {
  it('renders recorded figures, captions, key numbers, receipts and MathML', () => {
    const html = renderLensTwinHtml(directive, twin);
    expect(html).toContain('<figure class="lens-twin"');
    expect(html).toContain('recorded="true"');
    expect(html).toContain(twin.figure!.html);
    expect(html).toContain('<figcaption class="lens-twin__caption">');
    expect(html).toContain('Recorded caption &lt;with&gt; &amp; symbols');
    expect(html).toContain('<strong>Local I:</strong> <data value="0.42">0.42</data> units');
    expect(html).toContain('<table class="lens-twin__receipt">');
    expect(html).toContain('<td>—</td>');
    expect(html).toContain('<math');
    expect(html).toContain('class="enclosing lens-term"');
    expect(html).toContain('data-lens-key="gpu-spatial-analysis.local-moran~datum~d0~i2"');
    expect(html).toContain(' hidden>Run live on this GPU');
  });

  it('uses pending twins when no twin file exists', () => {
    const html = createMarkdown().render('::lens{scenario=test.missing rung=table}');
    expect(html).toContain('class="lens-twin lens-twin--pending"');
    expect(html).toContain('recorded="false"');
    expect(html).toContain('Record pending: no recorded run of <code>test.missing</code> yet.');
    expect(html).toContain('How test.missing computes (table)');
  });

  it('prints escaped error asides for invalid attributes and malformed lists', () => {
    const markdown = createMarkdown();
    expect(markdown.render('::lens{scenario=test.missing rung="<bad>"}')).toBe(
      '<aside class="admonition admonition-danger"><p class="admonition-title">Lens directive error</p><p>Invalid lens rung: &lt;bad&gt;</p></aside>'
    );
    for (const source of [
      'scenario=bad rung=datum',
      'scenario=test.missing rung=datum unknown=value',
      'scenario=test.missing rung=datum datum=-1',
      'scenario=test.missing rung=datum iteration=9007199254740992',
      'scenario=test.missing rung=datum caption="unclosed',
      'scenario=test.missing rung=datum caption=has=equals'
    ]) {
      expect(markdown.render(`::lens{${source}}`)).toContain('Lens directive error');
    }
  });

  it('parses quoted spaces and lets an author caption override the recorded caption', () => {
    const parsed = parseLensAttributeList(
      " scenario=test.missing rung=datum caption=\"Tract 17's local Moran I\" bind='shared page' "
    );
    expect(parsed).toEqual({
      scenario: 'test.missing',
      rung: 'datum',
      caption: "Tract 17's local Moran I",
      bind: 'shared page'
    });
    const html = renderLensTwinHtml({...directive, caption: 'Author <caption>'}, twin);
    expect(html).toContain('Author &lt;caption&gt;');
    expect(html).not.toContain('Recorded caption');
    expect(renderLensTwinHtml({...directive, caption: ''}, twin)).toContain(
      '<span class="lens-twin__caption-text"></span>'
    );
  });

  it('emits a block token, interrupts paragraphs and leaves code fences and inline text alone', () => {
    // The showcase's minimal markdown-it declaration omits its token parser.
    const markdown = createMarkdown() as MarkdownIt & {
      parse: (
        source: string,
        environment: Record<string, unknown>
      ) => {type: string; meta: unknown}[];
    };
    const source = '::lens{scenario=test.missing rung=datum}';
    const tokens = markdown.parse(`Paragraph\n   ${source}`, {});
    expect(tokens.at(-1)?.type).toBe('lens_directive');
    expect(tokens.at(-1)?.meta).toEqual({scenario: 'test.missing', rung: 'datum'});
    for (const literal of [
      `\`\`\`md\n${source}\n\`\`\``,
      `    ${source}`,
      `Inline ${source}`,
      `${source} extra`
    ]) {
      expect(markdown.render(literal)).not.toContain('<figure');
    }
    expect(renderMarkdown(`:::note Example\n${source}\n:::`)).toContain('<figure');
  });

  it('matches website validation and both recorded and pending HTML contracts', async context => {
    let toHtmlPath: string;
    let fromHtmlPath: string;
    try {
      toHtmlPath = requireFromRoot.resolve('hast-util-to-html');
      fromHtmlPath = requireFromRoot.resolve('hast-util-from-html');
    } catch {
      context.skip(
        'Website parity requires hast-util-to-html and hast-util-from-html in root node_modules.'
      );
      return;
    }
    const [{toHtml}, {fromHtml}] = await Promise.all([
      import(/* @vite-ignore */ pathToFileURL(toHtmlPath).href),
      import(/* @vite-ignore */ pathToFileURL(fromHtmlPath).href)
    ]);
    const parseHtml = (html: string) => fromHtml(html, {fragment: true}).children;
    const canonicalize = (html: string) =>
      JSON.parse(
        JSON.stringify(fromHtml(html, {fragment: true}), (name, value) =>
          name === 'position'
            ? undefined
            : name === 'value' && typeof value === 'string'
              ? value.replace(/\s+/g, ' ')
              : value
        )
      );
    expect(directive).toEqual(websiteHost.parseLensAttributes(attributes));
    const withView = {...attributes, view: 'neighbourhood-view'};
    expect(parseLensAttributes(withView)).toEqual(websiteHost.parseLensAttributes(withView));
    expect(websiteHost.twinKey(parseLensAttributes(withView))).toBe(
      'gpu-spatial-analysis.local-moran~datum~d0~i2~vneighbourhood-view'
    );
    expect(() => parseLensAttributes({...attributes, view: 'Bad View'})).toThrow(
      'Invalid lens view'
    );
    const variants = [undefined, 'Author "caption" & <details>', ''].flatMap(caption => [
      {...directive, caption},
      {...parseLensAttributes(withView), caption}
    ]);
    for (const selected of variants) {
      const minimalTwin = {...twin, figure: undefined, formula: undefined, receipt: undefined};
      const selectedTwins: (LensTwinFile | null)[] = [
        twin,
        minimalTwin,
        {
          ...twin,
          size: 'full',
          example: {code: 'const output = a < b && "ready";', language: 'ts', lineCount: 1}
        },
        {...twin, disabled: {reason: 'Missing <metrics>'}},
        {
          ...twin,
          verdicts: [
            {guardId: 'coordinate-model', label: 'Coordinate model', status: 'pass', reason: 'ok'},
            {guardId: 'uncorrected-p', label: 'P correction', status: 'fail', reason: '<none>'},
            {guardId: 'conservation', label: 'Conservation', status: 'unknown', reason: 'sum?'}
          ]
        },
        null
      ];
      for (const selectedTwin of selectedTwins) {
        const websiteTree = selectedTwin
          ? websiteHost.buildRecordedTwin(selected, selectedTwin, katex.renderToString, parseHtml)
          : websiteHost.buildPendingTwin(selected);
        expect(canonicalize(renderLensTwinHtml(selected, selectedTwin))).toEqual(
          canonicalize(toHtml(websiteTree))
        );
      }
    }
  });
});

describe('showcase markdown context', () => {
  it('resolves sibling paths against the page directory and prioritizes API anchors', () => {
    setLinkIndex({
      names: new Set(['GPUExample']),
      namesBySlug: new Map([['gpu-example', 'GPUExample']]),
      docIdsByPath: new Map([
        ['gpu-spatial-analysis/concepts', 'spatial-concepts'],
        ['gpu-network/concepts', 'network-concepts'],
        ['gpu-network/operators', 'network-operators'],
        ['gpu-spatial-analysis/operators-weights', 'spatial-weights'],
        ['gpu-network', 'network-overview']
      ])
    });
    try {
      const context = {docPath: 'gpu-spatial-analysis/operators-weights.md'};
      for (const [target, route] of [
        ['./concepts.md#overview', 'spatial-concepts'],
        ['./operators-weights.md#weights', 'spatial-weights'],
        ['../gpu-network/operators.md#overview', 'network-operators'],
        ['../gpu-network/README.md', 'network-overview'],
        ['./nested/../concepts.md', 'spatial-concepts'],
        ['./concepts.md#gpu-example', 'GPUExample']
      ]) {
        expect(renderMarkdown(`[link](${target})`, context)).toContain(
          `href="#/reference/${route}"`
        );
      }
      expect(
        renderMarkdown('[link](./concepts.md)', {docPath: 'gpu-network/operators.md'})
      ).toContain('href="#/reference/network-concepts"');
      expect(renderMarkdown('[link](../gpu-raster/README.md)', context)).toContain(
        'href="https://luma.gl/docs/api-reference/experimental/gpu-raster"'
      );
      expect(renderMarkdown('[link](../unknown/concepts.md)', context)).toContain(
        'href="https://luma.gl/docs/api-reference/experimental/unknown/concepts"'
      );
      expect(renderMarkdown(':::note\n[link](./concepts.md)\n:::', context)).toContain(
        'href="#/reference/spatial-concepts"'
      );
      expect(renderMarkdown('[link](gpu-network/concepts.md)')).toContain(
        'href="#/reference/network-concepts"'
      );
    } finally {
      setLinkIndex({names: new Set(), namesBySlug: new Map(), docIdsByPath: new Map()});
    }
  });

  it('strips standalone MDX comments outside fences and preserves directives', () => {
    const source = '::lens{scenario=test.missing rung=datum}';
    const html = renderMarkdown(
      `{/* generated marker */}\n${source}\n\n\`\`\`md\n{/* fenced marker */}\n\`\`\`\n\n~~~md\n{/* tilde marker */}\n~~~`
    );
    expect(html).not.toContain('generated marker');
    expect(html).toContain('fenced marker');
    expect(html).toContain('tilde marker');
    expect(html).toContain('<figure');
  });

  it.each([
    '<!-- lens:GPUExample -->',
    '{/* lens:GPUExample */}'
  ])('accepts the lens slot marker %s in the standalone rule', marker => {
    const html = createMarkdown().render(
      `${marker}\n\n::lens{scenario=test.missing rung=datum}\n\n~~~md\n${marker}\n~~~`
    );
    expect(html).toContain('<figure');
    expect(html).not.toContain('<p>{/*');
    expect(html).toContain('<pre><code class="language-md">');
    expect(html.slice(html.indexOf('<pre>'))).toContain('lens:GPUExample');
  });
});
