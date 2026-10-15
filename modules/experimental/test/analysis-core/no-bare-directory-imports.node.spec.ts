// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {describe, expect, it} from 'vitest';

const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const SOURCE_ROOTS = ['modules/experimental/src', 'modules/gpgpu/src'];

function listSourceFiles(directory: string): string[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const filePath = join(directory, entry.name);
    return entry.isDirectory()
      ? listSourceFiles(filePath)
      : /\.[cm]?[jt]sx?$/.test(entry.name)
        ? [filePath]
        : [];
  });
}

function getImportSpecifiers(sourceFile: typescript.SourceFile): typescript.StringLiteralLike[] {
  const specifiers: typescript.StringLiteralLike[] = [];
  function visitNode(node: typescript.Node): void {
    let specifier: typescript.Node | undefined;
    if (typescript.isImportDeclaration(node) || typescript.isExportDeclaration(node)) {
      specifier = node.moduleSpecifier;
    } else if (
      typescript.isCallExpression(node) &&
      node.expression.kind === typescript.SyntaxKind.ImportKeyword
    ) {
      specifier = node.arguments[0];
    } else if (typescript.isImportTypeNode(node) && typescript.isLiteralTypeNode(node.argument)) {
      specifier = node.argument.literal;
    } else if (typescript.isExternalModuleReference(node)) {
      specifier = node.expression;
    }
    if (specifier && typescript.isStringLiteralLike(specifier)) {
      specifiers.push(specifier);
    }
    typescript.forEachChild(node, visitNode);
  }
  visitNode(sourceFile);
  return specifiers;
}

describe('explicit directory index imports', () => {
  it('recognizes static, dynamic, and type-only specifiers without matching comments or strings', () => {
    const sourceFile = typescript.createSourceFile(
      'imports.ts',
      [
        "import {value} from './named';",
        "import type {Value} from './type';",
        "import './side-effect';",
        "export {value} from './export';",
        "export * from './star';",
        "export type {Value} from './export-type';",
        'const promise = import("./dynamic");',
        'const templatePromise = import(`./template`);',
        "type Value = import('./import-type').Value;",
        "import legacy = require('./legacy');",
        "// import './comment';",
        '`import "./string";`;'
      ].join('\n'),
      typescript.ScriptTarget.Latest,
      true
    );
    expect(getImportSpecifiers(sourceFile).map(specifier => specifier.text)).toEqual([
      './named',
      './type',
      './side-effect',
      './export',
      './star',
      './export-type',
      './dynamic',
      './template',
      './import-type',
      './legacy'
    ]);
  });

  it.each(SOURCE_ROOTS)('uses /index for every relative directory import in %s', sourceRoot => {
    const violations: string[] = [];
    const sourceFiles = listSourceFiles(resolve(REPOSITORY_ROOT, sourceRoot));
    expect(sourceFiles.length).toBeGreaterThan(0);
    for (const filePath of sourceFiles) {
      const sourceFile = typescript.createSourceFile(
        filePath,
        readFileSync(filePath, 'utf8'),
        typescript.ScriptTarget.Latest,
        true
      );
      for (const specifier of getImportSpecifiers(sourceFile)) {
        if (
          /^(\.{1,2})(\/|$)/.test(specifier.text) &&
          existsSync(resolve(dirname(filePath), specifier.text, 'index.ts'))
        ) {
          const {line} = sourceFile.getLineAndCharacterOfPosition(specifier.getStart(sourceFile));
          violations.push(`${relative(REPOSITORY_ROOT, filePath)}:${line + 1}: ${specifier.text}`);
        }
      }
    }
    // The ESM build appends .js to specifiers; directory imports must explicitly name index.
    expect(violations).toEqual([]);
  });
});
