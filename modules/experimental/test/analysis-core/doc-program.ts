// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';

// Shared by the docs specs (A-7, A-22): the analysis doc pages, the entry barrels they import,
// and a TypeScript program over the repo's own compiler options (the root `tsconfig.json` paths
// map `@luma.gl/*` package names to `modules/*/src`) with optional in-memory files.

export const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
export const DOCS_ROOT = join(REPOSITORY_ROOT, 'docs/api-reference/experimental');

/**
 * The analysis doc pages, relative to `DOCS_ROOT`. Upstream-owned raster pages (concepts,
 * filters and morphology, overviews, pixel operations) and the dataframe core pages are not
 * analysis pages and stay out.
 */
export const ANALYSIS_DOC_PAGES = [
  'gpu-spatial-analysis/README.md',
  'gpu-spatial-analysis/concepts.md',
  'gpu-spatial-analysis/operators.md',
  'gpu-spatial-analysis/operators-weights.md',
  'gpu-spatial-analysis/operators-statistics.md',
  'gpu-spatial-analysis/operators-dynamics.md',
  'gpu-spatial-analysis/operators-accessibility.md',
  'gpu-spatial-analysis/operators-joins.md',
  'gpu-spatial-analysis/operators-validity.md',
  'gpu-spatial-analysis/operators-density.md',
  'gpu-spatial-analysis/operators-cells.md',
  'gpu-spatial-analysis/operators-lines.md',
  'gpu-spatial-analysis/operators-geometry.md',
  'gpu-spatial-analysis/operators-regression.md',
  'gpu-spatial-analysis/recipes.md',
  'gpu-spatial-analysis-cross-reference.md',
  'gpu-network/README.md',
  'gpu-network/concepts.md',
  'gpu-network/operators.md',
  'gpu-terrain/README.md',
  'gpu-terrain/concepts.md',
  'gpu-terrain/operators.md',
  'gpu-dataframe-analysis.md',
  'geospatial.md',
  'gpu-raster/README.md',
  'gpu-raster/operations.md',
  'gpu-raster/operations-analysis.md',
  'gpu-raster/operations-regions-contours.md',
  'gpu-raster/operations-statistics.md'
] as const;

/** Package entries an analysis page may import, in the order free names are resolved. */
export const ANALYSIS_ENTRIES = [
  '@luma.gl/experimental/gpu-spatial-analysis',
  '@luma.gl/experimental/gpu-network',
  '@luma.gl/experimental/gpu-terrain',
  '@luma.gl/experimental/gpu-raster',
  '@luma.gl/experimental/gpu-dataframe',
  '@luma.gl/experimental/geospatial',
  '@luma.gl/experimental/gpu-tables',
  '@luma.gl/gpgpu/gpu-core',
  '@luma.gl/core'
] as const;

/** Reads the root compiler options (strict, bundler resolution, `@luma.gl/*` paths). */
export function getRepositoryCompilerOptions(): typescript.CompilerOptions {
  const configPath = join(REPOSITORY_ROOT, 'tsconfig.json');
  const config = typescript.readConfigFile(configPath, typescript.sys.readFile);
  const parsed = typescript.parseJsonConfigFileContent(
    config.config,
    typescript.sys,
    REPOSITORY_ROOT
  );
  return {
    ...parsed.options,
    noEmit: true,
    composite: false,
    incremental: false,
    declaration: false,
    // Snippets show isolated steps; an unused local is not an API error.
    noUnusedLocals: false
  };
}

/**
 * Creates a program whose root files are the given in-memory files (absolute path → text). Every
 * other file is read from disk, so package-name imports resolve to the current sources.
 */
export function createVirtualProgram(
  virtualFiles: Map<string, string>,
  options: typescript.CompilerOptions = getRepositoryCompilerOptions(),
  oldProgram?: typescript.Program
): typescript.Program {
  const host = typescript.createCompilerHost(options, true);
  const readFile = host.readFile.bind(host);
  const fileExists = host.fileExists.bind(host);
  const getSourceFile = host.getSourceFile.bind(host);
  host.readFile = fileName => virtualFiles.get(fileName) ?? readFile(fileName);
  host.fileExists = fileName => virtualFiles.has(fileName) || fileExists(fileName);
  host.getSourceFile = (fileName, languageVersion, onError, shouldCreate) => {
    const text = virtualFiles.get(fileName);
    return text === undefined
      ? getSourceFile(fileName, languageVersion, onError, shouldCreate)
      : typescript.createSourceFile(fileName, text, languageVersion, true);
  };
  return typescript.createProgram([...virtualFiles.keys()], options, host, oldProgram);
}

/** Reads one analysis doc page. */
export function readDocPage(page: string): string {
  const path = join(DOCS_ROOT, page);
  if (!existsSync(path)) {
    throw new Error(`missing doc page ${page}`);
  }
  return readFileSync(path, 'utf8');
}

/** Formats a diagnostic as `message` with its line and column in the file. */
export function formatDiagnostic(diagnostic: typescript.Diagnostic): string {
  const message = typescript.flattenDiagnosticMessageText(diagnostic.messageText, '\n');
  if (!diagnostic.file || diagnostic.start === undefined) {
    return message;
  }
  const {line, character} = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start);
  return `${line + 1}:${character + 1} TS${diagnostic.code} ${message}`;
}
