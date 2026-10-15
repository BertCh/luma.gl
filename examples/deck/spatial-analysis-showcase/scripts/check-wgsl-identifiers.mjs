// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Scans `src/**` TypeScript template strings for WGSL declarations that use a reserved word or
 * keyword as an identifier (`let target`, `fn filter(...)`, a parameter named `from`). WGSL
 * compilation errors for these appear only at runtime, in the browser.
 *
 * Also rejects constant non-finite f32 bitcasts, including standalone kernel body snippets.
 * Usage: `node scripts/check-wgsl-identifiers.mjs [dir ...]`. Defaults to all module sources and
 * showcase sources, independent of the working directory. Exit code 1 on hits.
 */
import {readdirSync, readFileSync, statSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';

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
const scriptPath = fileURLToPath(import.meta.url);
const showcaseDirectory = resolve(dirname(scriptPath), '..');
const modulesDirectory = resolve(showcaseDirectory, '../../../modules');
/** Shared by the CLI and the repository's node test gate. */
export function getSourceRoots() {
  return [
    join(showcaseDirectory, 'src'),
    ...readdirSync(modulesDirectory)
      .map(name => join(modulesDirectory, name, 'src'))
      .filter(path => statSync(path, {throwIfNoEntry: false})?.isDirectory())
  ];
}

export function scanDirectories(roots = getSourceRoots()) {
  const hits = [];
  let scannedTemplates = 0;
  function walk(directory) {
    for (const entry of readdirSync(directory, {withFileTypes: true})) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        if (entry.name !== 'node_modules') walk(path);
      } else if (/\.(?:[cm]?[jt]sx?|wgsl)$/.test(entry.name) && !entry.name.endsWith('.d.ts')) {
        const result = scanSource(readFileSync(path, 'utf8'), path);
        hits.push(...result.hits);
        scannedTemplates += result.scannedTemplates;
      }
    }
  }
  for (const root of roots) walk(root);
  return {hits, scannedTemplates};
}

/** Extract literal portions, preserving lines and visiting nested template expressions too. */
function* templates(source) {
  const sourceFile = typescript.createSourceFile(
    'source.ts',
    source,
    typescript.ScriptTarget.Latest
  );
  function* visit(node) {
    if (typescript.isNoSubstitutionTemplateLiteral(node) || typescript.isTemplateExpression(node)) {
      const start = node.getStart(sourceFile);
      let text = source.slice(start + 1, node.end - 1);
      if (typescript.isTemplateExpression(node)) {
        // Mask from each opening ${ through its closing }, retaining line offsets.
        const characters = text.split('');
        let literal = node.head;
        for (const span of node.templateSpans) {
          for (let index = literal.end - 2; index <= span.literal.pos; index++) {
            if (characters[index - start - 1] !== '\n') characters[index - start - 1] = ' ';
          }
          literal = span.literal;
        }
        text = characters.join('');
      }
      yield {
        text,
        startLine: sourceFile.getLineAndCharacterOfPosition(start).line + 1,
        tagged: /(?:\/\*\s*wgsl\s*\*\/|\bwgsl)\s*$/.test(source.slice(0, start))
      };
    }
    const children = [];
    typescript.forEachChild(node, child => {
      children.push(child);
    });
    for (const child of children) yield* visit(child);
  }
  yield* visit(sourceFile);
}

/** Remove WGSL comments, including nested block comments, without shifting diagnostics. */
function stripComments(source) {
  const characters = source.split('');
  let depth = 0;
  let lineComment = false;
  for (let index = 0; index < source.length; index++) {
    const pair = source.slice(index, index + 2);
    if (source[index] === '\n') lineComment = false;
    if (!lineComment && pair === '/*') {
      depth++;
      characters[index++] = ' ';
      characters[index] = ' ';
    } else if (!lineComment && depth > 0 && pair === '*/') {
      depth--;
      characters[index++] = ' ';
      characters[index] = ' ';
    } else {
      if (depth === 0 && pair === '//') lineComment = true;
      if ((depth > 0 || lineComment) && source[index] !== '\n') characters[index] = ' ';
    }
  }
  return characters.join('');
}

/**
 * Deliberately a source heuristic, not a WGSL evaluator: reject non-finite bit patterns in
 * constant-only bitcast operands (scalar or vector). Resolve simple const aliases and quoted
 * template alternatives, but leave let/var, parameters and buffer loads as runtime operands.
 * Scan raw source so interpolations, quoted fragments and bodies without `fn` are checked too.
 */
export function findNonFiniteBitcasts(source) {
  const code = stripComments(source);
  const constants = new Map(
    [...code.matchAll(/\bconst\s+(\w+)(?:\s*:\s*(?:u32|i32))?\s*=\s*([^;\n`]+);/g)].map(match => [
      match[1],
      match[2].replace(/^(['"])(.*)\1$/, '$2')
    ])
  );
  const expandConstants = expression => {
    for (let depth = 0; depth < 8; depth++) {
      const expanded = expression.replace(/\b[A-Za-z_]\w*\b/g, name => constants.get(name) ?? name);
      if (expanded === expression) break;
      expression = expanded;
    }
    return expression;
  };
  // Bounded nesting keeps this regex small; enough for vector constructors and bitwise expressions.
  let operand = '[^()]';
  for (let depth = 0; depth < 4; depth++) operand = `(?:[^()]|\\(${operand}*\\))`;
  const bitcasts = new RegExp(
    `bitcast\\s*<\\s*(?:f32|vec[234](?:f|\\s*<\\s*f32\\s*>))\\s*>\\s*\\((${operand}*)\\)`,
    'g'
  );
  const matches = [];
  for (const match of code.matchAll(bitcasts)) {
    const expression = expandConstants(
      match[1].replace(/\$\{([^{}]*)\}/g, (_match, interpolation) => {
        const alternatives = [...interpolation.matchAll(/['"]([^'"]+)['"]/g)];
        return alternatives.length
          ? alternatives.map(alternative => alternative[1]).join(',')
          : expandConstants(interpolation);
      })
    );
    const withoutNumbers = expression
      .replace(/\b(?:0x[\da-f]+|\d+)[ui]?\b/gi, '')
      .replace(/\b(?:vec[234][ui]?|u32|i32|true|false|select)\b/g, '');
    if (/[A-Za-z_$\[\]]/.test(withoutNumbers)) continue;
    const hasNonFiniteBits = [...expression.matchAll(/\b(0x[\da-f]+|\d+)[ui]?\b/gi)].some(
      literal => (Number(literal[1]) & 0x7f800000) === 0x7f800000
    );
    if (hasNonFiniteBits) matches.push({index: match.index, code: match[0]});
  }
  return matches;
}

/** Check WGSL declarations across lines; attributes may contain their own parentheses. */
export function findReservedIdentifiers(source) {
  const code = stripComments(source);
  const hits = [];
  const report = (match, kind, offset = 0) => {
    if (FORBIDDEN.has(match[1])) hits.push({index: offset + match.index, name: match[1], kind});
  };
  for (const match of code.matchAll(
    /\b(?:let|var(?:\s*<[^>]*>)?|const|override)\s+([A-Za-z_]\w*)/g
  )) {
    report(match, 'declaration');
  }
  for (const match of code.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)) {
    report(match, 'function name');
    const start = match.index + match[0].length;
    let end = start;
    let depth = 1;
    while (end < code.length && depth > 0) {
      if (code[end] === '(') depth++;
      if (code[end] === ')') depth--;
      end++;
    }
    for (const parameter of code.slice(start, end - 1).matchAll(/\b([A-Za-z_]\w*)\s*:/g)) {
      report(parameter, 'parameter', start);
    }
  }
  for (const match of code.matchAll(/\bstruct\s+[A-Za-z_]\w*\s*\{([^}]*)\}/g)) {
    const start = match.index + match[0].indexOf('{') + 1;
    for (const member of match[1].matchAll(/\b([A-Za-z_]\w*)\s*:/g)) {
      report(member, 'struct member', start);
    }
  }
  return hits;
}

export function scanSource(source, path = 'source.ts') {
  const hits = [];
  const file = relative(process.cwd(), path);
  for (const match of findNonFiniteBitcasts(source)) {
    hits.push({
      file,
      line: source.slice(0, match.index).split('\n').length,
      name: 'non-finite bitcast',
      kind: 'constant operand; use a runtime let/var bit pattern',
      code: match.code.replace(/\s+/g, ' ')
    });
  }
  const snippets = path.endsWith('.wgsl') ? [{text: source, startLine: 1}] : templates(source);
  let scannedTemplates = 0;
  for (const {text, startLine, tagged} of snippets) {
    scannedTemplates++;
    // Exclude embedded JavaScript examples; include typed WGSL and tagged kernel bodies.
    if (
      !tagged &&
      !path.endsWith('.wgsl') &&
      !/\bfn\s|\bstruct\s|\bvar\s*<|@(?:group|compute|vertex|fragment)\b|:\s*(?:f32|u32|i32|vec[234]|mat[234])\b|\b(?:let|var|const)\s+\w+\s*=[^;]*\b\d+[ui]\b/.test(
        stripComments(text)
      )
    )
      continue;
    for (const match of findReservedIdentifiers(text)) {
      const offset = text.slice(0, match.index).split('\n').length - 1;
      hits.push({
        file,
        line: startLine + offset,
        name: match.name,
        kind: match.kind,
        code: text.split('\n')[offset].trim()
      });
    }
  }
  return {hits, scannedTemplates};
}

if (process.argv[1] && resolve(process.argv[1]) === scriptPath) {
  const roots = process.argv.slice(2);
  const {hits, scannedTemplates} = scanDirectories(roots.length ? roots : getSourceRoots());

  if (hits.length === 0) {
    console.log(
      'check-wgsl-identifiers: no reserved WGSL identifiers or constant non-finite bitcasts found (' +
        scannedTemplates +
        ' templates scanned).'
    );
  } else {
    const byFile = new Map();
    for (const hit of hits) byFile.set(hit.file, [...(byFile.get(hit.file) ?? []), hit]);
    for (const [file, fileHits] of byFile) {
      console.log(file);
      for (const hit of fileHits) {
        console.log(`  ${hit.line}: "${hit.name}" (${hit.kind})  ${hit.code}`);
      }
    }
    console.log(`\n${hits.length} WGSL source issue(s) in ${byFile.size} file(s).`);
    process.exitCode = 1;
  }
}
