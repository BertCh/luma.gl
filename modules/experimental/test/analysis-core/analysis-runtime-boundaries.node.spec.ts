// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {dirname, join, relative, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {expect, it} from 'vitest';

const SOURCE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../src');
const ANALYSIS_ENTRIES = [
  'geospatial',
  'gpu-dataframe',
  'gpu-raster',
  'gpu-terrain',
  'gpu-spatial-analysis',
  'gpu-network'
];

function getRuntimeSpecifiers(source: string): string[] {
  const sourceFile = typescript.createSourceFile(
    'source.ts',
    source,
    typescript.ScriptTarget.Latest
  );
  const specifiers: string[] = [];
  function visit(node: typescript.Node): void {
    if (typescript.isImportDeclaration(node)) {
      const clause = node.importClause;
      const bindings = clause?.namedBindings;
      const onlyTypes =
        clause?.isTypeOnly ||
        (!clause?.name &&
          bindings &&
          typescript.isNamedImports(bindings) &&
          bindings.elements.length > 0 &&
          bindings.elements.every(element => element.isTypeOnly));
      if (!onlyTypes && typescript.isStringLiteral(node.moduleSpecifier)) {
        specifiers.push(node.moduleSpecifier.text);
      }
    } else if (typescript.isExportDeclaration(node)) {
      const clause = node.exportClause;
      const onlyTypes =
        node.isTypeOnly ||
        (clause &&
          typescript.isNamedExports(clause) &&
          clause.elements.length > 0 &&
          clause.elements.every(element => element.isTypeOnly));
      if (!onlyTypes && node.moduleSpecifier && typescript.isStringLiteral(node.moduleSpecifier)) {
        specifiers.push(node.moduleSpecifier.text);
      }
    } else if (
      typescript.isCallExpression(node) &&
      (node.expression.kind === typescript.SyntaxKind.ImportKeyword ||
        (typescript.isIdentifier(node.expression) && node.expression.text === 'require')) &&
      node.arguments[0] &&
      typescript.isStringLiteral(node.arguments[0])
    ) {
      specifiers.push(node.arguments[0].text);
    }
    typescript.forEachChild(node, visit);
  }
  visit(sourceFile);
  return specifiers;
}

it('recognizes runtime imports and re-exports while allowing type-only dependencies', () => {
  expect(
    getRuntimeSpecifiers(`
      import type {ViewPack} from './type-import';
      import {type ViewPack} from './inline-type-import';
      export type * from './type-export';
      export {type ViewPack} from './inline-type-export';
      import './registration';
      import {type ViewPack, register} from './mixed-import';
      export * from './barrel';
      export {register} from './named-export';
      void import('./lazy-view');
      require('./required-view');
    `)
  ).toEqual([
    './registration',
    './mixed-import',
    './barrel',
    './named-export',
    './lazy-view',
    './required-view'
  ]);
});

it.each(ANALYSIS_ENTRIES)('%s never reaches visualization adapters at runtime', entry => {
  const visited = new Set<string>();
  function visit(file: string, chain: string[]): void {
    if (visited.has(file)) return;
    visited.add(file);
    for (const specifier of getRuntimeSpecifiers(readFileSync(file, 'utf8'))) {
      const target = specifier.startsWith('.')
        ? resolve(dirname(file), specifier)
        : specifier.startsWith('@luma.gl/experimental/')
          ? resolve(SOURCE_ROOT, specifier.slice('@luma.gl/experimental/'.length))
          : undefined;
      if (!target) continue;
      const targetPath = relative(SOURCE_ROOT, target);
      const segments = targetPath.split(sep);
      const nextChain = [...chain, targetPath];
      expect(
        segments.some(segment =>
          ['computation-pack', 'computation-views', 'computation-lenses'].includes(segment)
        ),
        nextChain.join(' -> ')
      ).toBe(false);
      const resolvedFile = [target, `${target}.ts`, `${target}.tsx`, join(target, 'index.ts')].find(
        candidate =>
          candidate.endsWith('.ts') || candidate.endsWith('.tsx') ? existsSync(candidate) : false
      );
      if (resolvedFile) visit(resolvedFile, nextChain);
    }
  }
  visit(join(SOURCE_ROOT, entry, 'index.ts'), [entry]);
});
