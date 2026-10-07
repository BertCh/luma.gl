// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {escapeHtml} from './dom';

const KEYWORDS = new Set(
  (
    'const let var function return if else for while of in new import from export default type interface ' +
    'class extends async await true false null undefined as typeof void struct fn vec2 vec3 vec4 f32 u32 i32 ' +
    'array storage uniform read read_write'
  ).split(' ')
);

const TOKEN =
  /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`)|(\b\d[\d_]*(?:\.\d+)?(?:e[+-]?\d+)?\b)|(\b[A-Za-z_$][\w$]*\b)/g;

/**
 * Tiny syntax highlighter for TypeScript, JSON and WGSL snippets: comments, strings, numbers,
 * keywords and `GPU*` class names. Returns HTML-safe markup.
 */
export function highlightCode(code: string): string {
  let result = '';
  let last = 0;
  for (const match of code.matchAll(TOKEN)) {
    const index = match.index ?? 0;
    result += escapeHtml(code.slice(last, index));
    const [text, comment, string, number, word] = match;
    if (comment) result += `<span class="tok-comment">${escapeHtml(text)}</span>`;
    else if (string) result += `<span class="tok-string">${escapeHtml(text)}</span>`;
    else if (number) result += `<span class="tok-number">${escapeHtml(text)}</span>`;
    else if (word && KEYWORDS.has(word)) result += `<span class="tok-keyword">${word}</span>`;
    else if (word && /^GPU[A-Z]/.test(word)) result += `<span class="tok-class">${word}</span>`;
    else result += escapeHtml(text);
    last = index + text.length;
  }
  return result + escapeHtml(code.slice(last));
}
