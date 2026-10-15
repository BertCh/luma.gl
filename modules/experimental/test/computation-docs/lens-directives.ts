// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {readdirSync, readFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';

/** Absolute repository root, independent of the test runner's working directory. */
export const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
/** Root of authored documentation and recorded lenses. */
export const DOCS_ROOT = join(REPOSITORY_ROOT, 'docs');

/** A marker's one-based, inclusive section boundaries. Preambles and `##` sections are supported. */
export type LensMarker = {name: string; line: number; section: {start: number; end: number}};
/** One authored directive and the markers sharing its section. */
export type ScannedLensDirective = {
  file: string;
  line: number;
  attributes: Record<string, string>;
  sectionClasses: string[];
};

/** Lists all Markdown documentation except the generated API subtree, deterministically. */
export function listDocsMarkdownFiles(): string[] {
  const files: string[] = [];
  function visitDirectory(directory: string): void {
    for (const entry of readdirSync(directory, {withFileTypes: true})) {
      const path = join(directory, entry.name);
      if (path === join(DOCS_ROOT, 'api-reference/generated')) continue;
      if (entry.isDirectory()) visitDirectory(path);
      else if (entry.isFile() && /\.mdx?$/.test(entry.name)) files.push(path);
    }
  }
  visitDirectory(DOCS_ROOT);
  return files.sort();
}

/** Parses the whitespace-separated attribute body; malformed or duplicate attributes throw. */
export function parseDirectiveAttributes(text: string): Record<string, string> {
  const attributes: Record<string, string> = Object.create(null);
  const pattern = /([A-Za-z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`{}]+))(?:\s+|$)/y;
  let position = /^\s*/.exec(text)![0].length;
  while (position < text.length) {
    pattern.lastIndex = position;
    const match = pattern.exec(text);
    if (!match) throw new Error(`Invalid lens attribute list at column ${position + 1}`);
    const [, name, doubleQuoted, singleQuoted, unquoted] = match;
    if (Object.hasOwn(attributes, name)) throw new Error(`Duplicate lens attribute: ${name}`);
    if (name === 'caption' && unquoted !== undefined) {
      throw new Error('Lens caption must be quoted');
    }
    attributes[name] = doubleQuoted ?? singleQuoted ?? unquoted;
    position = pattern.lastIndex;
  }
  return attributes;
}

function readVisibleLines(text: string): {line: number; text: string}[] {
  const visible: {line: number; text: string}[] = [];
  let fence: {character: string; length: number} | undefined;
  for (const [index, source] of text.split(/\r?\n/).entries()) {
    if (fence) {
      const closing = /^ {0,3}(`{3,}|~{3,})\s*$/.exec(source);
      if (closing && closing[1][0] === fence.character && closing[1].length >= fence.length) {
        fence = undefined;
      }
      continue;
    }
    const opening = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(source);
    if (opening && (opening[1][0] !== '`' || !opening[2].includes('`'))) {
      fence = {character: opening[1][0], length: opening[1].length};
      continue;
    }
    visible.push({line: index + 1, text: source});
  }
  return visible;
}

/** Finds real marker slots, ignoring fenced examples. Optional text supports in-memory fixtures. */
export function scanLensMarkers(file: string, text = readFileSync(file, 'utf8')): LensMarker[] {
  const visible = readVisibleLines(text);
  const boundaries = visible
    .filter(source => /^ {0,3}#{1,3}(?:\s|$)/.test(source.text))
    .map(source => source.line);
  const lastLine = text.split(/\r?\n/).length;
  const markers: LensMarker[] = [];
  for (const source of visible) {
    for (const match of source.text.matchAll(
      /<!--\s*lens:([A-Za-z0-9_-]+)\s*-->|\{\/\*\s*lens:([A-Za-z0-9_-]+)\s*\*\/\}/g
    )) {
      const start = boundaries.filter(line => line <= source.line).at(-1) ?? 1;
      const end = (boundaries.find(line => line > source.line) ?? lastLine + 1) - 1;
      markers.push({name: match[1] ?? match[2], line: source.line, section: {start, end}});
    }
  }
  return markers;
}

/** Scans leaf directives outside fences and associates markers anywhere in the same section. */
export function scanLensDirectives(
  file: string,
  text = readFileSync(file, 'utf8')
): ScannedLensDirective[] {
  const markers = scanLensMarkers(file, text);
  const directives: ScannedLensDirective[] = [];
  for (const source of readVisibleLines(text)) {
    const match = /^\s{0,3}::lens\{(.*)\}\s*$/.exec(source.text);
    if (!match) continue;
    try {
      directives.push({
        file,
        line: source.line,
        attributes: parseDirectiveAttributes(match[1]),
        sectionClasses: markers
          .filter(
            marker => source.line >= marker.section.start && source.line <= marker.section.end
          )
          .map(marker => marker.name)
      });
    } catch (error) {
      throw new Error(`${file}:${source.line}: ${String(error)}`);
    }
  }
  return directives;
}

/**
 * Scenario `element` segment of an exported contributor or recipe: the name without its `GPU` or `add`
 * prefix, in kebab case (`GPULocalMoran` → `local-moran`, `GPULISAMarkov` → `lisa-markov`,
 * `addHotSpotAnalysisRecipe` → `hot-spot-analysis-recipe`).
 */
export function getLensElementId(name: string): string {
  return (
    name
      .replace(/^(?:GPU|add|createGPU|create)(?=[A-Z])/, '')
      .replace(/([a-z])([A-Z])/g, '$1-$2')
      // A digit splits only before a word (`FFT1D` stays `fft1d`).
      .replace(/([0-9])([A-Z][a-z])/g, '$1-$2')
      .replace(/([A-Z]+)([A-Z][a-z])/g, '$1-$2')
      .toLowerCase()
  );
}

/**
 * Canonical twin basename; numeric pins are normalized and zero is never omitted. A chosen view
 * (`view=<id>`) is part of the key, so one rung with and without a domain view never collides.
 */
export function twinKey(attributes: Record<string, string>): string {
  return `${attributes['scenario']}~${attributes['rung']}${
    attributes['datum'] === undefined ? '' : `~d${Number(attributes['datum'])}`
  }${attributes['iteration'] === undefined ? '' : `~i${Number(attributes['iteration'])}`}${
    attributes['view'] === undefined ? '' : `~v${attributes['view']}`
  }`;
}

type EntryElements = {contributors: string[]; recipes: string[]; otherClasses: string[]};
const entryElements = new Map<string, EntryElements>();

/**
 * Discovers exported contributors (including inherited methods), recipe functions and other
 * classes. A class is a contributor when its instances have one of `contributorMethods`.
 */
export function listEntryElements(
  barrelPath: string,
  contributorMethods: readonly string[] = ['getCommandNodes']
): EntryElements {
  const path = resolve(REPOSITORY_ROOT, barrelPath);
  const cacheKey = `${path}#${contributorMethods.join(',')}`;
  const cached = entryElements.get(cacheKey);
  if (cached) return cached;
  const configuration = typescript.readConfigFile(
    join(REPOSITORY_ROOT, 'tsconfig.json'),
    typescript.sys.readFile
  );
  // Discovery must use the repository paths so inherited @luma.gl methods resolve.
  assert(!configuration.error);
  const {options, errors} = typescript.parseJsonConfigFileContent(
    configuration.config,
    typescript.sys,
    REPOSITORY_ROOT
  );
  // Invalid compiler options must not silently turn contributors into ordinary classes.
  assert(errors.length === 0);
  const program = typescript.createProgram([path], {
    ...options,
    noEmit: true,
    composite: false,
    incremental: false
  });
  const checker = program.getTypeChecker();
  const source = program.getSourceFile(path);
  const moduleSymbol = source && checker.getSymbolAtLocation(source);
  // Missing barrels must fail instead of reporting an empty entry.
  assert(moduleSymbol);
  const result: EntryElements = {contributors: [], recipes: [], otherClasses: []};
  for (const exportedSymbol of checker.getExportsOfModule(moduleSymbol)) {
    const symbol =
      exportedSymbol.flags & typescript.SymbolFlags.Alias
        ? checker.getAliasedSymbol(exportedSymbol)
        : exportedSymbol;
    if (symbol.flags & typescript.SymbolFlags.Class) {
      const instance = checker.getDeclaredTypeOfSymbol(symbol);
      const names = contributorMethods.some(method => instance.getProperty(method))
        ? result.contributors
        : result.otherClasses;
      names.push(exportedSymbol.name);
    } else if (
      symbol.flags & typescript.SymbolFlags.Function &&
      /^add.*Recipe$/.test(exportedSymbol.name)
    ) {
      result.recipes.push(exportedSymbol.name);
    }
  }
  result.contributors.sort();
  result.recipes.sort();
  result.otherClasses.sort();
  entryElements.set(cacheKey, result);
  return result;
}
