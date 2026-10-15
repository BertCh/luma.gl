// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {parseLensCatalogAttributes, renderReferenceLensCatalog} from './lens-catalog';
import type MarkdownIt from 'markdown-it';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import './lens-twin.css';
import './lens-card.css';
import {mountLiveLens, type LiveLensAttributes} from './live-lens-adapter';

/** Validated lens selectors and optional author caption. */
export type LensDirective = LiveLensAttributes & {caption?: string};

/** Read-only materialized twin shared with the website host. */
export type LensTwinFile = {
  schema: 'luma.lens-twin/1';
  scenario: string;
  rung: string;
  size?: 'full' | 'thumbnail';
  example?: {code: string; language: 'ts'; lineCount: number};
  datum?: number;
  iteration?: number;
  caption: string;
  description: string;
  disabled?: {reason: string};
  keyNumber: {label: string; value: string; unit?: string};
  figure?: {html: string; width: number; height: number};
  receipt?: {columns: string[]; rows: string[][]; provenance?: string[]};
  formula?: {tex: string; ascii: string; derivation: 'derived' | 'curated'};
  verdicts?: {
    guardId: string;
    label: string;
    status: 'pass' | 'fail' | 'unknown';
    reason: string;
  }[];
  provenance: {
    adapter: string;
    date: string;
    variant?: string;
    runId: string;
    observerEffects: string[];
  };
};

const TWIN_FILES = import.meta.glob<LensTwinFile>('../../../../../docs/lens-records/twins/*.json', {
  eager: true,
  import: 'default'
});
const TWINS = new Map(
  Object.entries(TWIN_FILES).map(([path, twin]) => [path.split('/').pop()!.slice(0, -5), twin])
);
const ATTRIBUTE_NAMES = ['scenario', 'rung', 'bind', 'datum', 'iteration', 'caption', 'view'];
const RUNG_NAMES = [
  'result',
  'formula',
  'datum',
  'iterations',
  'kernel',
  'graph',
  'timing',
  'compare',
  'scope',
  'table'
];

/** Parses whitespace-separated directive attributes, preserving quoted values. */
export function parseLensAttributeList(source: string): Record<string, string> {
  const attributes: Record<string, string> = Object.create(null);
  const attribute = /([A-Za-z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`}]+))(?:\s+|$)/y;
  let position = /^\s*/.exec(source)![0].length;
  while (position < source.length) {
    attribute.lastIndex = position;
    const match = attribute.exec(source);
    if (!match) throw new Error('Invalid lens attribute list.');
    if (Object.hasOwn(attributes, match[1]))
      throw new Error(`Duplicate lens attribute: ${match[1]}`);
    attributes[match[1]] = match[2] ?? match[3] ?? match[4];
    position = attribute.lastIndex;
  }
  return attributes;
}

/** Validates author attributes using the website host's rules and error messages. */
export function parseLensAttributes(attributes: Record<string, unknown> = {}): LensDirective {
  for (const [name, value] of Object.entries(attributes)) {
    if (!ATTRIBUTE_NAMES.includes(name)) throw new Error(`Unknown lens attribute: ${name}`);
    if (typeof value !== 'string') throw new Error(`Lens attribute ${name} must be a string.`);
  }
  const {scenario, rung, bind, caption, view} = attributes;
  if (typeof scenario !== 'string' || !/^[a-z0-9-]+(\.[a-z0-9-]+){1,2}$/.test(scenario)) {
    throw new Error('Invalid lens scenario; expected <entry>.<element>[.<variant>].');
  }
  if (typeof rung !== 'string' || !RUNG_NAMES.includes(rung)) {
    throw new Error(`Invalid lens rung: ${rung}`);
  }
  if (view !== undefined && (typeof view !== 'string' || !/^[a-z0-9]+(-[a-z0-9]+)*$/.test(view))) {
    throw new Error(`Invalid lens view: ${view}`);
  }
  const directive: LensDirective = {
    scenario,
    rung,
    bind: typeof bind === 'string' ? bind : 'page',
    ...(typeof caption === 'string' ? {caption} : {}),
    ...(typeof view === 'string' ? {view} : {})
  };
  for (const name of ['datum', 'iteration'] as const) {
    const value = attributes[name];
    if (value !== undefined) {
      if (
        typeof value !== 'string' ||
        !/^\d+$/.test(value) ||
        !Number.isSafeInteger(Number(value))
      ) {
        throw new Error(`Lens attribute ${name} must be a non-negative safe integer.`);
      }
      directive[name] = Number(value);
    }
  }
  return directive;
}

function getTwinKey(directive: LensDirective): string {
  return `${directive.scenario}~${directive.rung}${directive.datum === undefined ? '' : `~d${directive.datum}`}${directive.iteration === undefined ? '' : `~i${directive.iteration}`}${directive.view === undefined ? '' : `~v${directive.view}`}`;
}

/** Verdict chip text, identical to computation-views `renderVerdictChip`. */
function getVerdictText(verdict: NonNullable<LensTwinFile['verdicts']>[number]): string {
  return verdict.status === 'pass'
    ? `✓ ${verdict.label}`
    : verdict.status === 'fail'
      ? `✕ ${verdict.label}: ${verdict.reason}`
      : `? ${verdict.label}: unknown — ${verdict.reason}`;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Renders the shared static HTML contract with only generated figure markup and TeX output inserted verbatim. */
export function renderLensTwinHtml(
  directive: LensDirective,
  twin: LensTwinFile | null,
  renderTeX: typeof katex.renderToString = katex.renderToString
): string {
  const key = getTwinKey(directive);
  const caption =
    directive.caption ?? twin?.caption ?? `How ${directive.scenario} computes (${directive.rung})`;
  const description = twin?.description ?? caption;
  const selectors = (['datum', 'iteration', 'view'] as const)
    .map(name => (directive[name] === undefined ? '' : ` ${name}="${directive[name]}"`))
    .join('');
  const figure = twin
    ? `<div class="lens-twin__figure" role="img" aria-label="${escapeHtml(description)}">${twin.figure?.html ?? ''}</div>`
    : `<div class="lens-twin__figure lens-twin__figure--pending${directive.rung === 'result' ? ' lens-twin__figure--result' : ''}">Record pending: no recorded run of <code>${escapeHtml(directive.scenario)}</code> yet.</div>`;
  const keyNumber = twin
    ? `<strong>${escapeHtml(twin.keyNumber.label)}:</strong> <data value="${escapeHtml(twin.keyNumber.value)}">${escapeHtml(twin.keyNumber.value)}</data>${twin.keyNumber.unit ? ` ${escapeHtml(twin.keyNumber.unit)}` : ''}`
    : '<strong>Key number:</strong> pending';
  const provenance = twin?.provenance;
  const provenanceHtml = provenance
    ? `<span class="lens-twin__provenance">${escapeHtml(`Recorded on ${provenance.adapter}${provenance.variant ? `, ${provenance.variant}` : ''}, ${provenance.date}${provenance.observerEffects.length ? `; ${provenance.observerEffects.join(', ')}` : ''}`)}</span>`
    : '';
  const verdictsHtml = twin?.verdicts?.length
    ? `<span class="lens-twin__verdicts">${twin.verdicts.map(verdict => `<span class="lens-verdict lens-verdict--${escapeHtml(verdict.status)}" role="note" title="${escapeHtml(verdict.reason)}">${escapeHtml(getVerdictText(verdict))}</span>`).join('')}</span>`
    : '';
  const exampleHtml = twin?.example
    ? `<pre class="lens-example"><code class="language-ts">${escapeHtml(twin.example.code)}</code></pre>`
    : '';
  const disabledHtml = twin?.disabled
    ? `<p class="lens-twin__disabled">Rung unavailable: ${escapeHtml(twin.disabled.reason)}</p>`
    : '';
  const formula = twin?.formula;
  const formulaHtml = formula
    ? `<div class="lens-twin__formula" data-derivation="${escapeHtml(formula.derivation)}">${renderTeX(
        formula.tex,
        {
          displayMode: true,
          output: 'htmlAndMathml',
          throwOnError: false,
          strict: 'ignore',
          trust: ({command}) => command === '\\htmlClass' || command === '\\htmlData',
          macros: {'\\term': '\\htmlClass{lens-term}{#1}'}
        }
      )}</div>`
    : '';
  const receipt = twin?.receipt;
  const receiptHtml = receipt
    ? `<table class="lens-twin__receipt"><thead><tr>${receipt.columns.map(column => `<th>${escapeHtml(column)}</th>`).join('')}</tr></thead><tbody>${receipt.rows.map((row, rowIndex) => `<tr${receipt.provenance?.[rowIndex] ? ` data-provenance="${escapeHtml(receipt.provenance[rowIndex])}"` : ''}>${row.map(cell => `<td>${escapeHtml(cell === '' ? '—' : cell)}</td>`).join('')}</tr>`).join('')}</tbody></table>`
    : '';
  return `<figure class="lens-twin${twin ? '' : ' lens-twin--pending'}" id="lens-${escapeHtml(key.replaceAll('~', '-'))}" data-lens-key="${escapeHtml(key)}" data-rung="${escapeHtml(directive.rung)}"><ladder-lens scenario="${escapeHtml(directive.scenario)}" rung="${escapeHtml(directive.rung)}" bind="${escapeHtml(directive.bind)}"${selectors} recorded="${Boolean(twin)}" aria-label="${escapeHtml(description)}">${figure}</ladder-lens><figcaption class="lens-twin__caption"><span class="lens-twin__caption-text">${escapeHtml(caption)}</span><span class="lens-twin__key-number">${keyNumber}</span>${provenanceHtml}${verdictsHtml}</figcaption>${exampleHtml}${disabledHtml}${formulaHtml}${receiptHtml}<p class="lens-twin__live"><button type="button" class="lens-twin__live-button" data-lens-key="${escapeHtml(key)}" hidden>Run live on this GPU</button></p></figure>`;
}

// The showcase's minimal markdown-it typings omit block plugin state and token metadata.
type LensBlockState = {
  src: string;
  bMarks: number[];
  eMarks: number[];
  tShift: number[];
  sCount: number[];
  blkIndent: number;
  line: number;
  push: (
    type: string,
    tag: string,
    nesting: number
  ) => {
    block: boolean;
    map: [number, number];
    meta: unknown;
  };
};
type LensBlockParser = {
  ruler: {
    before: (
      beforeName: string,
      ruleName: string,
      rule: (state: LensBlockState, startLine: number, endLine: number, silent: boolean) => boolean,
      options: {alt: string[]}
    ) => void;
  };
};

/** Installs the whole-line leaf directive and its non-throwing HTML renderer. */
export function registerLensDirective(markdown: MarkdownIt): void {
  const blockParser = (markdown as MarkdownIt & {block: LensBlockParser}).block;
  blockParser.ruler.before(
    'paragraph',
    'lens_directive',
    (state, startLine, endLine, silent) => {
      if (state.sCount[startLine] - state.blkIndent > 3) return false;
      const line = state.src.slice(
        state.bMarks[startLine] + state.tShift[startLine],
        state.eMarks[startLine]
      );
      if (
        /^(?:<!--\s*lens:[A-Za-z0-9_-]+\s*-->|\{\/\*\s*lens:[A-Za-z0-9_-]+\s*\*\/\})[ \t]*$/.test(
          line
        )
      ) {
        if (!silent) state.line = startLine + 1;
        return true;
      }
      const match = /^::(lens|lens-catalog)\{(.*)\}[ \t]*$/.exec(line);
      if (!match) return false;
      if (silent) return true;
      const token = state.push(match[1] === 'lens' ? 'lens_directive' : 'lens_catalog', '', 0);
      token.block = true;
      token.map = [startLine, startLine + 1];
      try {
        token.meta = parseLensAttributeList(match[2]);
      } catch (error) {
        token.meta = error;
      }
      state.line = startLine + 1;
      return true;
    },
    {alt: ['paragraph', 'reference', 'blockquote', 'list']}
  );
  const renderDirective: NonNullable<(typeof markdown.renderer.rules)[string]> = (
    tokens,
    index
  ) => {
    try {
      const attributes = (
        tokens[index] as (typeof tokens)[number] & {
          meta: Record<string, unknown> | Error;
        }
      ).meta;
      if (attributes instanceof Error) throw attributes;
      if (tokens[index].type === 'lens_catalog') {
        return renderReferenceLensCatalog(TWINS, parseLensCatalogAttributes(attributes));
      }
      const directive = parseLensAttributes(attributes);
      return renderLensTwinHtml(directive, TWINS.get(getTwinKey(directive)) ?? null);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return `<aside class="admonition admonition-danger"><p class="admonition-title">Lens directive error</p><p>${escapeHtml(message)}</p></aside>`;
    }
  };
  markdown.renderer.rules['lens_directive'] = renderDirective;
  markdown.renderer.rules['lens_catalog'] = renderDirective;
}

/** Enables opt-in live lenses and returns cleanup for mounts, pending work and the listener. */
export function activateLensButtons(container: HTMLElement): () => void {
  let disposed = false;
  const disposers = new Set<() => void>();
  // Reserved per-bind coordinator slots until the live host publishes that interface.
  const coordinators = new Map<string, unknown>();
  if ('gpu' in navigator) {
    for (const button of container.querySelectorAll<HTMLButtonElement>('.lens-twin__live-button')) {
      button.hidden = false;
    }
  }
  const handleClick = async (event: MouseEvent) => {
    if (disposed || !('gpu' in navigator) || !(event.target instanceof Element)) return;
    const button = event.target.closest<HTMLButtonElement>('.lens-twin__live-button');
    const figure = button?.closest('figure.lens-twin');
    const element = figure?.querySelector<HTMLElement>('ladder-lens');
    if (!button || button.disabled || !figure || !element || !container.contains(button)) return;
    const attributes: LiveLensAttributes = {
      scenario: element.getAttribute('scenario') ?? '',
      rung: element.getAttribute('rung') ?? '',
      bind: element.getAttribute('bind') ?? 'page'
    };
    for (const name of ['datum', 'iteration'] as const) {
      const value = element.getAttribute(name);
      if (value !== null) attributes[name] = Number(value);
    }
    const view = element.getAttribute('view');
    if (view !== null) attributes.view = view;
    const recordedMarkup = element.innerHTML;
    const buttonText = button.textContent;
    const restoreTwin = () => {
      element.innerHTML = recordedMarkup;
      button.disabled = false;
      button.textContent = buttonText;
    };
    let disposeMount: (() => void) | undefined;
    const dispose = () => {
      try {
        const mounted = disposeMount;
        disposeMount = undefined;
        mounted?.();
      } finally {
        restoreTwin();
      }
    };
    disposers.add(dispose);
    if (!coordinators.has(attributes.bind)) coordinators.set(attributes.bind, undefined);
    button.disabled = true;
    button.textContent = 'Starting…';
    figure.querySelector('.lens-twin__error')?.remove();
    try {
      const [views, lenses] = await Promise.all([
        import('@luma.gl/experimental/computation-views'),
        import('@luma.gl/experimental/computation-lenses')
      ]);
      if (disposed || !element.isConnected) {
        disposers.delete(dispose);
        restoreTwin();
        return;
      }
      disposeMount = await mountLiveLens(element, attributes, {views, lenses});
      if (disposed || !element.isConnected) {
        disposers.delete(dispose);
        dispose();
        return;
      }
      button.textContent = 'Live on this GPU';
    } catch (error) {
      disposers.delete(dispose);
      dispose();
      if (!disposed && element.isConnected) {
        const message = document.createElement('p');
        message.className = 'lens-twin__error';
        message.setAttribute('role', 'status');
        message.textContent = error instanceof Error ? error.message : String(error);
        figure.append(message);
      }
    }
  };
  container.addEventListener('click', handleClick);
  return () => {
    disposed = true;
    container.removeEventListener('click', handleClick);
    for (const dispose of disposers) {
      try {
        dispose();
      } catch {
        // A failed disposal must not prevent cleanup of the remaining mounted lenses.
      }
    }
    disposers.clear();
    coordinators.clear();
  };
}

// Local structural types supplement the showcase's intentionally small markdown-it declaration.
type CollapseToken = {
  type: string;
  tag: string;
  content: string;
  level: number;
  block: boolean;
  nesting: number;
};
type CollapseState = {
  tokens: CollapseToken[];
  Token: new (type: string, tag: string, nesting: number) => CollapseToken;
};

/** Collapses exact level-four computation headings within their token parent. */
export function registerLensCollapse(markdown: MarkdownIt): void {
  const parser = markdown as MarkdownIt & {
    core: {
      ruler: {after: (name: string, rule: string, run: (state: CollapseState) => void) => void};
    };
  };
  parser.core.ruler.after('inline', 'lens_how', state => {
    const {tokens} = state;
    for (let index = 0; index < tokens.length; index++) {
      const heading = tokens[index];
      const inline = tokens[index + 1];
      if (
        heading.type !== 'heading_open' ||
        heading.tag !== 'h4' ||
        inline?.type !== 'inline' ||
        inline.content !== 'How it computes'
      )
        continue;
      let end = index + 3;
      while (end < tokens.length) {
        const token = tokens[end];
        if (
          token.level < heading.level ||
          (token.level === heading.level &&
            (token.type === 'hr' || (token.type === 'heading_open' && /^h[1-4]$/.test(token.tag))))
        )
          break;
        end++;
      }
      const count = tokens
        .slice(index + 3, end)
        .filter(token => token.type === 'lens_directive').length;
      const opening = new state.Token('html_block', '', 0);
      opening.block = true;
      opening.level = heading.level;
      opening.content = `<details class="lens-how"><summary class="lens-how__summary">How it computes${count ? `<span class="lens-how__count"> · ${count} ${count === 1 ? 'view' : 'views'}</span>` : ''}</summary>`;
      const closing = new state.Token('html_block', '', 0);
      closing.block = true;
      closing.level = heading.level;
      closing.content = '</details>';
      tokens.splice(end, 0, closing);
      tokens.splice(index, 3, opening);
    }
  });
}
