// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, readFileSync, statSync} from 'node:fs';
import {dirname, join, relative, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {describe, expect, it} from 'vitest';

// Rule 20 (A-4, A-13): shared contracts live below every analysis entry, the entries form a DAG,
// and each entry barrel is the union of its subdirectory indexes.

const SOURCE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../src');
const DOCS_ROOT = resolve(SOURCE_ROOT, '../../../docs/api-reference/experimental');

/** Shared layers that every analysis entry may import. They import no analysis entry. */
const SHARED_LAYERS = ['utils', 'analysis-core'] as const;

/**
 * Analysis entries from lowest to highest. An entry may import shared layers and entries listed
 * before it, never one listed after it.
 */
const ENTRY_ORDER = [
  'geospatial',
  'gpu-dataframe',
  'gpu-raster',
  'gpu-terrain',
  'gpu-spatial-analysis',
  'gpu-network'
] as const;

/** Upstream luma entries the analysis entries may also import (they import no analysis entry). */
const UPSTREAM_ENTRIES = new Set(['gpu-project', 'gpu-tables']);

/** Top-level source files each barrel re-exports explicitly; everything else comes from `export *`. */
const EXPLICIT_BARREL_SOURCES: Record<(typeof ENTRY_ORDER)[number], RegExp> = {
  geospatial: /^\.\/[a-z0-9-]+$|^@luma\.gl\/gpgpu\/gpu-core$|^\.\.\/utils\//,
  'gpu-dataframe': /^\.\/gpu-[a-z0-9-]+$|^\.\.\/utils\//,
  'gpu-raster':
    /^\.\/(gpu-raster[a-z0-9-]*|types|raster-device-limits)$|^\.\.\/(utils|analysis-core)\//,
  'gpu-terrain': /^\.\.\/(utils|analysis-core)\//,
  'gpu-spatial-analysis': /^\.\.\/(utils|analysis-core)\//,
  'gpu-network': /^\.\.\/utils\//
};

/** Documentation pages whose `###` headings name exports of one entry. */
const DOCUMENTED_ENTRIES: Record<string, (typeof ENTRY_ORDER)[number]> = {
  'geospatial.md': 'geospatial',
  'gpu-dataframe-analysis.md': 'gpu-dataframe',
  'gpu-raster/operations-analysis.md': 'gpu-raster',
  'gpu-terrain.md': 'gpu-terrain',
  'gpu-spatial-analysis.md': 'gpu-spatial-analysis',
  'gpu-network.md': 'gpu-network'
};

function listSourceFiles(directory: string): string[] {
  const files: string[] = [];
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) {
      files.push(...listSourceFiles(path));
    } else if (name.endsWith('.ts')) {
      files.push(path);
    }
  }
  return files;
}

function getRelativeImportLayers(file: string): Set<string> {
  const layers = new Set<string>();
  const source = readFileSync(file, 'utf8');
  for (const match of source.matchAll(/(?:from|import)\s*\(?\s*'(\.{1,2}\/[^']*)'/g)) {
    const target = relative(SOURCE_ROOT, resolve(dirname(file), match[1]));
    const layer = target.split(sep)[0];
    if (!layer.startsWith('..')) {
      layers.add(layer.replace(/\.ts$/, ''));
    }
  }
  return layers;
}

function getLayerImports(layer: string): Map<string, string[]> {
  const imports = new Map<string, string[]>();
  for (const file of listSourceFiles(join(SOURCE_ROOT, layer))) {
    for (const target of getRelativeImportLayers(file)) {
      if (target !== layer) {
        imports.set(target, [...(imports.get(target) || []), relative(SOURCE_ROOT, file)]);
      }
    }
  }
  return imports;
}

function getEntryExports(): Map<string, Set<string>> {
  const files: string[] = [];
  for (const entry of ENTRY_ORDER) {
    files.push(join(SOURCE_ROOT, entry, 'index.ts'));
    for (const index of getSubdirectoryIndexes(entry)) {
      files.push(index);
    }
  }
  const program = typescript.createProgram(files, {
    noEmit: true,
    skipLibCheck: true,
    target: typescript.ScriptTarget.ES2022,
    module: typescript.ModuleKind.ESNext,
    moduleResolution: typescript.ModuleResolutionKind.Bundler
  });
  const checker = program.getTypeChecker();
  const exportsByFile = new Map<string, Set<string>>();
  for (const file of files) {
    const sourceFile = program.getSourceFile(file);
    const moduleSymbol = sourceFile && checker.getSymbolAtLocation(sourceFile);
    exportsByFile.set(
      relative(SOURCE_ROOT, file),
      new Set(moduleSymbol ? checker.getExportsOfModule(moduleSymbol).map(({name}) => name) : [])
    );
  }
  return exportsByFile;
}

function getSubdirectoryIndexes(entry: string): string[] {
  return readdirSync(join(SOURCE_ROOT, entry))
    .map(name => join(SOURCE_ROOT, entry, name, 'index.ts'))
    .filter(index => {
      try {
        return statSync(index).isFile();
      } catch {
        return false;
      }
    });
}

describe('analysis layering', () => {
  it('keeps the shared layers below every analysis entry', () => {
    for (const layer of SHARED_LAYERS) {
      const imports = getLayerImports(layer);
      const allowed = SHARED_LAYERS.slice(0, SHARED_LAYERS.indexOf(layer) + 1);
      for (const [target, files] of imports) {
        expect(allowed as readonly string[], `${layer} imports ${target} in ${files[0]}`).toContain(
          target
        );
      }
    }
  });

  it('orders the analysis entries as a DAG', () => {
    for (const [position, entry] of ENTRY_ORDER.entries()) {
      const allowed = new Set<string>([
        ...SHARED_LAYERS,
        ...UPSTREAM_ENTRIES,
        ...ENTRY_ORDER.slice(0, position)
      ]);
      for (const [target, files] of getLayerImports(entry)) {
        expect(allowed.has(target), `${entry} imports ${target} in ${files[0]}`).toBe(true);
      }
    }
  });
});

describe('analysis entry barrels', () => {
  const exportsByFile = getEntryExports();

  it('re-export every subdirectory index whole', () => {
    for (const entry of ENTRY_ORDER) {
      const barrel = readFileSync(join(SOURCE_ROOT, entry, 'index.ts'), 'utf8');
      for (const index of getSubdirectoryIndexes(entry)) {
        const subdirectory = relative(join(SOURCE_ROOT, entry), dirname(index));
        expect(barrel, `${entry} barrel`).toContain(`export * from './${subdirectory}/index';`);
      }
      for (const match of barrel.matchAll(/export\s+(?:type\s+)?\{[^}]*\}\s+from\s+'([^']*)'/g)) {
        expect(match[1], `${entry} explicit re-export`).toMatch(EXPLICIT_BARREL_SOURCES[entry]);
      }
    }
  });

  it('exports the union of the subdirectory exports', () => {
    for (const entry of ENTRY_ORDER) {
      const entryExports = exportsByFile.get(join(entry, 'index.ts'))!;
      for (const index of getSubdirectoryIndexes(entry)) {
        const subdirectoryExports = exportsByFile.get(relative(SOURCE_ROOT, index))!;
        const missing = [...subdirectoryExports].filter(name => !entryExports.has(name));
        expect(missing, relative(SOURCE_ROOT, index)).toEqual([]);
      }
    }
  });

  it('reaches every documented contributor and recipe from its entry', () => {
    for (const [page, entry] of Object.entries(DOCUMENTED_ENTRIES)) {
      const markdown = readFileSync(join(DOCS_ROOT, page), 'utf8');
      const entryExports = exportsByFile.get(join(entry, 'index.ts'))!;
      const documented = new Set<string>();
      for (const heading of markdown.match(/^#{2,4} .*$/gm) || []) {
        for (const match of heading.matchAll(/`((?:GPU[A-Z]\w+)|(?:add\w+Recipe))`/g)) {
          documented.add(match[1]);
        }
      }
      expect(documented.size, page).toBeGreaterThan(0);
      const unreachable = [...documented].filter(name => !entryExports.has(name));
      expect(unreachable, `${page} names missing from ${entry}`).toEqual([]);
    }
  });
});
