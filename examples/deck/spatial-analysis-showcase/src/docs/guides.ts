// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {parseFrontMatter, renderMarkdown} from '../shell/markdown';

const guideSources = import.meta.glob('./guides/*.md', {
  query: '?raw',
  import: 'default',
  eager: true
}) as Record<string, string>;

/** A hand-written concept guide from `src/docs/guides/*.md`. */
export type Guide = {
  /** File name without extension. */
  id: string;
  title: string;
  summary: string;
  order: number;
  /** Markdown body without front matter. */
  body: string;
};

/** Guides sorted by their `order` front-matter field. */
export function getGuides(): Guide[] {
  return Object.entries(guideSources)
    .map(([path, source]) => {
      const {data, body} = parseFrontMatter(source);
      const id = path.replace(/^.*\//, '').replace(/\.md$/, '');
      return {
        id,
        title: data['title'] ?? id,
        summary: data['summary'] ?? '',
        order: Number(data['order'] ?? 99),
        body
      };
    })
    .sort((a, b) => a.order - b.order);
}

/** Renders a guide body to HTML. */
export function renderGuide(guide: Guide): string {
  return renderMarkdown(guide.body);
}
