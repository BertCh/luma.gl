// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {StoryStep} from './scene';

/**
 * Splits markdown into story steps: each `## Title {#step-id}` heading starts a step whose body is
 * the text below it. `extras` adds camera, options and callouts by step id, so long narratives can
 * live in a sibling `<id>.md` imported with `?raw`.
 */
export function storyFromMarkdown<O extends object>(
  markdown: string,
  extras: Record<string, Omit<StoryStep<O>, 'id' | 'title' | 'body'> | undefined> = {}
): StoryStep<O>[] {
  const steps: StoryStep<O>[] = [];
  let current: {id: string; title: string; lines: string[]} | null = null;
  const finish = () => {
    if (current) {
      steps.push({
        id: current.id,
        title: current.title,
        body: current.lines.join('\n').trim(),
        ...extras[current.id]
      });
    }
  };
  for (const line of markdown.split('\n')) {
    const heading = /^## (.+?)(?:\s*\{#([\w-]+)\})?\s*$/.exec(line);
    if (heading) {
      finish();
      const title = heading[1];
      const id =
        heading[2] ??
        title
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, '-')
          .replace(/^-+|-+$/g, '');
      current = {id, title, lines: []};
    } else if (current) {
      current.lines.push(line);
    }
  }
  finish();
  return steps;
}
