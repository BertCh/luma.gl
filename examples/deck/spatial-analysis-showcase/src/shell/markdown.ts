// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import MarkdownIt from 'markdown-it';
import {highlightCode} from './highlight';

/**
 * Link knowledge the renderer uses to turn doc links and inline `GPUClass` names into reference
 * routes. The reference index fills it when it loads; before that, links render unresolved.
 */
export type LinkIndex = {
  /** Class and API names with a reference page. */
  names: ReadonlySet<string>;
  /** Doc ids by lowercase path tail, for example `gpu-network` or `gpu-raster/operations-analysis`. */
  docIdsByPath: ReadonlyMap<string, string>;
  /** Heading slug to reference name. */
  namesBySlug: ReadonlyMap<string, string>;
};

let linkIndex: LinkIndex = {names: new Set(), docIdsByPath: new Map(), namesBySlug: new Map()};

/** Installs the link index used by every renderer. */
export function setLinkIndex(index: LinkIndex): void {
  linkIndex = index;
}

/** Heading text to a Docusaurus-like slug. */
export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/`/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

const md = new MarkdownIt({html: true, linkify: false, typographer: false});

const defaultFence = md.renderer.rules['fence'];
md.renderer.rules['fence'] = (tokens, index, options, env, self) => {
  const token = tokens[index];
  const language = token.info.trim().split(/\s+/)[0];
  const html = ['', 'ts', 'typescript', 'js', 'javascript', 'json', 'wgsl', 'tsx'].includes(
    language
  )
    ? highlightCode(token.content)
    : md.utils.escapeHtml(token.content);
  if (!defaultFence) return `<pre><code>${html}</code></pre>`;
  return `<div class="code-block"><pre><code class="language-${md.utils.escapeHtml(language)}">${html}</code></pre></div>`;
};

md.renderer.rules['heading_open'] = (tokens, index, options, env, self) => {
  const inline = tokens[index + 1];
  if (inline && inline.type === 'inline') tokens[index].attrSet('id', slugify(inline.content));
  return self.renderToken(tokens, index, options);
};

const defaultCodeInline = md.renderer.rules['code_inline'];
md.renderer.rules['code_inline'] = (tokens, index, options, env, self) => {
  const name = tokens[index].content;
  if (linkIndex.names.has(name)) {
    return `<a class="name-link" href="#/reference/${encodeURIComponent(name)}"><code>${md.utils.escapeHtml(name)}</code></a>`;
  }
  return defaultCodeInline
    ? defaultCodeInline(tokens, index, options, env, self)
    : `<code>${md.utils.escapeHtml(name)}</code>`;
};

const defaultLinkOpen = md.renderer.rules['link_open'];
md.renderer.rules['link_open'] = (tokens, index, options, env, self) => {
  const token = tokens[index];
  const href = token.attrGet('href') ?? '';
  const resolved = resolveLink(href);
  if (resolved.kind === 'route') {
    token.attrSet('href', resolved.href);
  } else if (resolved.kind === 'external') {
    token.attrSet('href', resolved.href);
    token.attrSet('target', '_blank');
    token.attrSet('rel', 'noreferrer noopener');
  }
  return defaultLinkOpen
    ? defaultLinkOpen(tokens, index, options, env, self)
    : self.renderToken(tokens, index, options);
};

const DOCS_SITE = 'https://luma.gl/docs/api-reference/experimental';

/** Route of a bundled doc for a path such as `/docs/api-reference/experimental/gpu-network`. */
function resolveKnownDoc(path: string): string | null {
  const clean = path.replace(/\.md$/, '');
  const tail = clean.split('/').slice(-2).join('/').toLowerCase();
  const docId =
    linkIndex.docIdsByPath.get(tail) ?? linkIndex.docIdsByPath.get(tail.split('/').pop() ?? '');
  return docId ? `#/reference/${encodeURIComponent(docId)}` : null;
}

/** Maps a markdown link target to a showcase route or an external URL. */
function resolveLink(href: string): {kind: 'route' | 'external' | 'keep'; href: string} {
  if (/^(https?:|mailto:)/.test(href)) return {kind: 'external', href};
  if (href.startsWith('#/')) return {kind: 'keep', href};
  if (href.startsWith('/docs/')) {
    const known = resolveKnownDoc(href.split('#')[0]);
    return known
      ? {kind: 'route', href: known}
      : {kind: 'external', href: `https://luma.gl${href}`};
  }
  const [path, anchor] = href.split('#');
  if (!path) {
    const name = anchor ? linkIndex.namesBySlug.get(anchor) : undefined;
    return name
      ? {kind: 'route', href: `#/reference/${encodeURIComponent(name)}`}
      : {kind: 'keep', href};
  }
  if (path.endsWith('.md') || path.endsWith('/') || !path.includes('.')) {
    const clean = path
      .replace(/^(\.\.?\/)+/, '')
      .replace(/\.md$/, '')
      .replace(/\/README$/, '');
    const tail = clean.split('/').slice(-2).join('/').toLowerCase();
    const docId =
      linkIndex.docIdsByPath.get(tail) ?? linkIndex.docIdsByPath.get(tail.split('/').pop() ?? '');
    const name = anchor ? linkIndex.namesBySlug.get(anchor) : undefined;
    if (name) return {kind: 'route', href: `#/reference/${encodeURIComponent(name)}`};
    if (docId) return {kind: 'route', href: `#/reference/${encodeURIComponent(docId)}`};
    return {kind: 'external', href: `${DOCS_SITE}/${clean}`};
  }
  return {kind: 'keep', href};
}

const ADMONITION = /^:::(\w+)[ \t]*(.*)$/;

/** Renders Docusaurus-flavoured markdown (front matter, admonitions, MDX imports removed). */
export function renderMarkdown(source: string): string {
  const cleaned = stripMdx(stripFrontMatter(source));
  const lines = cleaned.split('\n');
  const output: string[] = [];
  let buffer: string[] = [];
  let fence = false;
  let admonition: {kind: string; title: string; lines: string[]} | null = null;
  const flush = () => {
    if (buffer.length) output.push(md.render(buffer.join('\n')));
    buffer = [];
  };
  for (const line of lines) {
    if (/^\s*```/.test(line)) fence = !fence;
    const open = fence ? null : ADMONITION.exec(line);
    if (!admonition && open) {
      flush();
      admonition = {kind: open[1], title: open[2] || open[1], lines: []};
    } else if (admonition && !fence && line.trim() === ':::') {
      output.push(
        `<aside class="admonition admonition-${md.utils.escapeHtml(admonition.kind)}"><p class="admonition-title">${md.utils.escapeHtml(admonition.title)}</p>${md.render(admonition.lines.join('\n'))}</aside>`
      );
      admonition = null;
    } else if (admonition) {
      admonition.lines.push(line);
    } else {
      buffer.push(line);
    }
  }
  flush();
  return output.join('\n');
}

/** Renders a short markdown snippet inline (no wrapping paragraph). */
export function renderInlineMarkdown(source: string): string {
  return md.renderInline(source);
}

/** Returns the `title:` of a front-matter block and the body without it. */
export function parseFrontMatter(source: string): {data: Record<string, string>; body: string} {
  const match = /^---\n([\s\S]*?)\n---\n?/.exec(source);
  if (!match) return {data: {}, body: source};
  const data: Record<string, string> = {};
  for (const line of match[1].split('\n')) {
    const separator = line.indexOf(':');
    if (separator > 0) data[line.slice(0, separator).trim()] = line.slice(separator + 1).trim();
  }
  return {data, body: source.slice(match[0].length)};
}

function stripFrontMatter(source: string): string {
  return parseFrontMatter(source).body;
}

/** Removes MDX `import` lines and self-closing JSX component lines. */
function stripMdx(source: string): string {
  let fence = false;
  return source
    .split('\n')
    .filter(line => {
      if (/^\s*```/.test(line)) fence = !fence;
      if (fence) return true;
      return (
        !/^import\s.+from\s+['"].+['"];?\s*$/.test(line) && !/^<[A-Z]\w*\b[^>]*\/>\s*$/.test(line)
      );
    })
    .join('\n');
}
