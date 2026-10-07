// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Scans `src/**` TypeScript template strings for WGSL declarations that use a reserved word or
 * keyword as an identifier (`let target`, `fn filter(...)`, a parameter named `from`). WGSL
 * compilation errors for these appear only at runtime, in the browser.
 *
 * Usage: `node scripts/check-wgsl-identifiers.mjs [dir ...]` (default `src`). Exit code 1 on hits.
 */
import {readdirSync, readFileSync, statSync} from 'node:fs';
import {join, relative} from 'node:path';

// Keywords and reserved words of the WGSL specification (section "Keyword Summary" and
// "Reserved Words").
const KEYWORDS = `alias break case const const_assert continue continuing default diagnostic discard
else enable false fn for if let loop override requires return struct switch true var while`;
const RESERVED = `NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto
await become binding_array cast catch class co_await co_return co_yield coherent column_major common
compile compile_fragment concept const_cast consteval constexpr constinit crate debugger decltype
delete demote demote_to_helper do dynamic_cast enum explicit export extends extern external
fallthrough filter final finally friend from fxgroup get goto groupshared highp impl implements
import inline instanceof interface layout lowp macro macro_rules match mediump meta mod module move
mut mutable namespace new nil noexcept noinline nointerpolation noperspective null nullptr of
operator package packoffset partition pass patch pixelfragment precise precision premerge priv
protected pub public readonly ref regardless register reinterpret_cast require resource restrict
self set shared sizeof smooth snorm static static_assert static_cast std subroutine super target
template this thread_local throw trait try type typedef typeid typename typeof union unless unorm
unsafe unsized use using varying virtual volatile wgsl where with writeonly yield`;
// `flat` is an @interpolate keyword, not a reserved word, but it has broken builders' shaders.
const FORBIDDEN = new Set(`${KEYWORDS} ${RESERVED} flat`.split(/\s+/));
const roots = process.argv.slice(2).length ? process.argv.slice(2) : ['src'];
const hits = [];
let scannedTemplates = 0;

function walk(directory) {
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    const stats = statSync(path);
    if (stats.isDirectory()) {
      if (entry !== 'node_modules') walk(path);
    } else if (/\.(ts|tsx|wgsl)$/.test(entry) && !entry.endsWith('.d.ts')) {
      scan(path);
    }
  }
}

/** Yields `{text, startLine}` for each backtick template literal (without `${}` bodies). */
function* templates(source) {
  let line = 1;
  for (let index = 0; index < source.length; index++) {
    const character = source[index];
    if (character === '\n') line++;
    else if (character === '/' && source[index + 1] === '/') {
      while (index < source.length && source[index] !== '\n') index++;
      line++;
    } else if (character === '/' && source[index + 1] === '*') {
      const end = source.indexOf('*/', index + 2);
      const stop = end < 0 ? source.length : end + 2;
      for (let k = index; k < stop; k++) if (source[k] === '\n') line++;
      index = stop - 1;
    } else if (character === "'" || character === '"') {
      index++;
      while (index < source.length && source[index] !== character && source[index] !== '\n') {
        if (source[index] === '\\') index++;
        index++;
      }
    } else if (character === '`') {
      const startLine = line;
      let text = '';
      let depth = 0;
      index++;
      for (; index < source.length; index++) {
        const next = source[index];
        if (next === '\n') line++;
        if (depth === 0 && next === '`') break;
        if (next === '\\') {
          index++;
          continue;
        }
        if (next === '$' && source[index + 1] === '{') {
          depth++;
          index++;
          text += '0';
          continue;
        }
        if (depth > 0) {
          if (next === '{') depth++;
          else if (next === '}') depth--;
          continue;
        }
        text += next;
      }
      yield {text, startLine};
    }
  }
}

function scan(path) {
  const source = readFileSync(path, 'utf8');
  for (const {text, startLine} of templates(source)) {
    scannedTemplates++;
    if (!/\b(fn|@compute|@vertex|@fragment|var<|struct)\b|@group/.test(text)) continue;
    let inStruct = false;
    text.split('\n').forEach((lineText, offset) => {
      const code = lineText.replace(/\/\/.*$/, '');
      const report = (name, kind) => {
        if (FORBIDDEN.has(name)) {
          hits.push({file: relative(process.cwd(), path), line: startLine + offset, name, kind, code: code.trim()});
        }
      };
      for (const match of code.matchAll(/\b(?:let|var(?:<[^>]*>)?|const|override)\s+([A-Za-z_]\w*)/g)) {
        report(match[1], 'declaration');
      }
      for (const match of code.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(([^)]*)\)?/g)) {
        report(match[1], 'function name');
        for (const parameter of match[2].matchAll(/(?:^|,)\s*(?:@\w+(?:\([^)]*\))?\s*)*([A-Za-z_]\w*)\s*:/g)) {
          report(parameter[1], 'parameter');
        }
      }
      if (/\bstruct\s+\w+\s*\{/.test(code)) inStruct = true;
      if (inStruct) {
        for (const member of code.matchAll(/(?:^|[{,])\s*(?:@\w+(?:\([^)]*\))?\s*)*([A-Za-z_]\w*)\s*:/g)) {
          report(member[1], 'struct member');
        }
        if (code.includes('}')) inStruct = false;
      }
    });
  }
}

for (const root of roots) walk(root);

if (hits.length === 0) {
  console.log('check-wgsl-identifiers: no reserved WGSL identifiers found (' + scannedTemplates + ' templates scanned).');
} else {
  const byFile = new Map();
  for (const hit of hits) byFile.set(hit.file, [...(byFile.get(hit.file) ?? []), hit]);
  for (const [file, fileHits] of byFile) {
    console.log(file);
    for (const hit of fileHits) {
      console.log(`  ${hit.line}: "${hit.name}" (${hit.kind})  ${hit.code}`);
    }
  }
  console.log(`\n${hits.length} reserved-word identifier(s) in ${byFile.size} file(s).`);
  process.exitCode = 1;
}
