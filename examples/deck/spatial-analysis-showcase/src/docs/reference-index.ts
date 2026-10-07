// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {parseFrontMatter, renderMarkdown, setLinkIndex, slugify} from '../shell/markdown';

/** Docs bundled into the Reference section, loaded on first use. */
const DOC_LOADERS = import.meta.glob(
  [
    '../../../../../docs/api-reference/experimental/gpu-spatial-analysis.md',
    '../../../../../docs/api-reference/experimental/gpu-spatial-analysis-cross-reference.md',
    '../../../../../docs/api-reference/experimental/gpu-network.md',
    '../../../../../docs/api-reference/experimental/gpu-terrain.md',
    '../../../../../docs/api-reference/experimental/gpu-dataframe-analysis.md',
    '../../../../../docs/api-reference/experimental/gpu-raster/operations-analysis.md',
    '../../../../../docs/api-reference/experimental/gpu-graph.md',
    '../../../../../docs/api-reference/experimental/gpu-graph-topology.md',
    '../../../../../docs/api-reference/experimental/gpu-graph-metrics.md',
    '../../../../../docs/api-reference/experimental/gpu-graph-connectivity.md',
    '../../../../../docs/api-reference/experimental/gpu-graph-traversal.md',
    '../../../../../docs/api-reference/experimental/gpu-graph-layouts.md',
    '../../../../../docs/api-reference/experimental/gpu-graph-operations.md'
  ],
  {query: '?raw', import: 'default'}
) as Record<string, () => Promise<string>>;

/** Display order, ids and labels of the bundled docs. */
const DOC_DEFINITIONS = [
  {
    path: 'gpu-spatial-analysis.md',
    id: 'gpu-spatial-analysis',
    entryPoint: '@luma.gl/experimental/gpu-spatial-analysis'
  },
  {
    path: 'gpu-spatial-analysis-cross-reference.md',
    id: 'gpu-spatial-analysis-cross-reference',
    entryPoint: 'Tool cross-reference'
  },
  {path: 'gpu-network.md', id: 'gpu-network', entryPoint: '@luma.gl/experimental/gpu-network'},
  {path: 'gpu-terrain.md', id: 'gpu-terrain', entryPoint: '@luma.gl/experimental/gpu-terrain'},
  {
    path: 'gpu-dataframe-analysis.md',
    id: 'gpu-dataframe-analysis',
    entryPoint: '@luma.gl/experimental/gpu-dataframe'
  },
  {
    path: 'gpu-raster/operations-analysis.md',
    id: 'gpu-raster-operations-analysis',
    entryPoint: '@luma.gl/experimental/gpu-raster'
  },
  {
    path: 'gpu-graph.md',
    id: 'gpu-graph',
    entryPoint: '@luma.gl/gpgpu/gpu-graph',
    bareClassNames: true
  },
  {
    path: 'gpu-graph-topology.md',
    id: 'gpu-graph-topology',
    entryPoint: '@luma.gl/gpgpu/gpu-graph',
    bareClassNames: true
  },
  {
    path: 'gpu-graph-metrics.md',
    id: 'gpu-graph-metrics',
    entryPoint: '@luma.gl/gpgpu/gpu-graph',
    bareClassNames: true
  },
  {
    path: 'gpu-graph-connectivity.md',
    id: 'gpu-graph-connectivity',
    entryPoint: '@luma.gl/gpgpu/gpu-graph',
    bareClassNames: true
  },
  {
    path: 'gpu-graph-traversal.md',
    id: 'gpu-graph-traversal',
    entryPoint: '@luma.gl/gpgpu/gpu-graph',
    bareClassNames: true
  },
  {
    path: 'gpu-graph-layouts.md',
    id: 'gpu-graph-layouts',
    entryPoint: '@luma.gl/gpgpu/gpu-graph',
    bareClassNames: true
  },
  {
    path: 'gpu-graph-operations.md',
    id: 'gpu-graph-operations',
    entryPoint: '@luma.gl/gpgpu/gpu-graph',
    bareClassNames: true
  }
] as const;

/** One heading-delimited part of a doc. */
export type ReferenceSection = {
  docId: string;
  /** Heading text without the leading `#`s. */
  heading: string;
  level: 2 | 3;
  /** Backticked identifiers in the heading; the first is the primary route name. */
  names: string[];
  /** The `##` group this section sits under. */
  group: string;
  /** Markdown body without the heading. */
  body: string;
  /** First sentence of the body, plain text. */
  summary: string;
};

/** One bundled doc. */
export type ReferenceDoc = {
  id: string;
  title: string;
  entryPoint: string;
  /** Sections without class names (overview, conventions, tables), in order. */
  plainSections: ReferenceSection[];
  /** Sections that document classes or functions. */
  apiSections: ReferenceSection[];
};

/** All reference docs with name lookup. */
export type ReferenceIndex = {
  docs: ReferenceDoc[];
  /** Section by every identifier in its heading. */
  byName: Map<string, ReferenceSection>;
};

let indexPromise: Promise<ReferenceIndex> | null = null;

/** Parses every bundled doc once and installs the link index for markdown rendering. */
export function loadReferenceIndex(): Promise<ReferenceIndex> {
  indexPromise ??= buildIndex();
  return indexPromise;
}

async function buildIndex(): Promise<ReferenceIndex> {
  const docs: ReferenceDoc[] = [];
  for (const definition of DOC_DEFINITIONS) {
    const key = Object.keys(DOC_LOADERS).find(path => path.endsWith(`/${definition.path}`));
    if (!key) continue;
    const raw = await DOC_LOADERS[key]();
    docs.push(parseDoc(definition.id, definition.entryPoint, raw, 'bareClassNames' in definition));
  }
  const byName = new Map<string, ReferenceSection>();
  const namesBySlug = new Map<string, string>();
  const docIdsByPath = new Map<string, string>();
  for (const doc of docs) {
    for (const section of doc.apiSections) {
      for (const name of section.names) {
        if (!byName.has(name)) byName.set(name, section);
      }
      namesBySlug.set(slugify(section.heading), section.names[0]);
      for (const name of section.names) namesBySlug.set(slugify(name), section.names[0]);
    }
  }
  for (const definition of DOC_DEFINITIONS) {
    const path = definition.path.replace(/\.md$/, '');
    docIdsByPath.set(path, definition.id);
    docIdsByPath.set(path.split('/').pop() ?? path, definition.id);
  }
  // Raster docs link to `gpu-raster/README.md` and friends that are not bundled: they stay external.
  setLinkIndex({names: new Set(byName.keys()), docIdsByPath, namesBySlug});
  return {docs, byName};
}

/**
 * Splits a doc into sections. Headings name their classes in backticks; docs whose headings name
 * them bare (`## Rank incoming influence with GPUGraphPageRank`) set `bareClassNames`.
 */
function parseDoc(
  id: string,
  entryPoint: string,
  raw: string,
  bareClassNames = false
): ReferenceDoc {
  const {data, body} = parseFrontMatter(raw);
  const lines = body.split('\n');
  const sections: ReferenceSection[] = [];
  let current: {heading: string; level: 2 | 3; group: string; lines: string[]} | null = null;
  let group = '';
  let title = data['title'] ?? id;
  let preamble: string[] = [];
  let fence = false;
  const finish = () => {
    if (!current) return;
    const pattern = bareClassNames ? /\b(GPU[A-Z]\w*)\b/g : /`([A-Za-z_][\w.]*)`/g;
    const names = [...current.heading.matchAll(pattern)].map(match => match[1]);
    const text = current.lines.join('\n').trim();
    sections.push({
      docId: id,
      heading: current.heading.replace(/`/g, ''),
      level: current.level,
      names,
      group: current.group,
      body: text,
      summary: getSummary(text)
    });
    current = null;
  };
  for (const line of lines) {
    if (/^\s*```/.test(line)) fence = !fence;
    const heading = fence ? null : /^(#{1,3}) (.+?)\s*$/.exec(line);
    if (heading && heading[1].length === 1) {
      title = heading[2].replace(/`/g, '');
    } else if (heading) {
      finish();
      const level = heading[1].length as 2 | 3;
      if (level === 2) group = heading[2].replace(/`/g, '');
      current = {heading: heading[2], level, group, lines: []};
    } else if (current) {
      current.lines.push(line);
    } else {
      preamble.push(line);
    }
  }
  finish();
  const overview: ReferenceSection = {
    docId: id,
    heading: 'Introduction',
    level: 2,
    names: [],
    group: title,
    body: preamble.join('\n').trim(),
    summary: getSummary(preamble.join('\n'))
  };
  return {
    id,
    title,
    entryPoint,
    plainSections: [overview, ...sections.filter(section => section.names.length === 0)],
    apiSections: sections.filter(section => section.names.length > 0)
  };
}

function getSummary(markdown: string): string {
  const paragraph = markdown
    .split(/\n\s*\n/)
    .map(part => part.trim())
    .find(part => part && !/^(:::|```|\||import |<|#)/.test(part));
  if (!paragraph) return '';
  const text = paragraph
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/[`*_]/g, '')
    .replace(/\s+/g, ' ');
  const sentence = /^(.+?[.!?])(\s|$)/.exec(text);
  return (sentence ? sentence[1] : text).slice(0, 220);
}

/** Renders one section's body to HTML. */
export function renderSectionBody(section: ReferenceSection): string {
  return renderMarkdown(section.body);
}

/** Renders a doc's plain sections (overview, conventions, tables) to HTML. */
export function renderDocOverview(doc: ReferenceDoc): string {
  return doc.plainSections
    .map(section =>
      section.heading === 'Introduction'
        ? renderMarkdown(section.body)
        : renderMarkdown(`${'#'.repeat(section.level)} ${section.heading}\n\n${section.body}`)
    )
    .join('\n');
}

/** Resolves a route entry: an API name or a doc id. */
export function resolveEntry(
  index: ReferenceIndex,
  entry: string
): {kind: 'section'; section: ReferenceSection} | {kind: 'doc'; doc: ReferenceDoc} | null {
  const section = index.byName.get(entry);
  if (section) return {kind: 'section', section};
  const doc = index.docs.find(candidate => candidate.id === entry);
  return doc ? {kind: 'doc', doc} : null;
}
