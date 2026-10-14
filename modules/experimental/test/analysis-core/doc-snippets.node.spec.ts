// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {join} from 'node:path';
import typescript from 'typescript';
import {expect, it} from 'vitest';
import {
  ANALYSIS_DOC_PAGES,
  ANALYSIS_ENTRIES,
  DOCS_ROOT,
  createVirtualProgram,
  formatDiagnostic,
  getRepositoryCompilerOptions,
  readDocPage
} from './doc-program';

// A-7: every ```ts block on the analysis doc pages type-checks against the current barrels.
//
// A block is a module of its own and keeps its own imports. Most blocks are fragments of a larger
// program, so the check resolves the names a block uses without declaring them:
// - a name exported by an analysis entry (`ANALYSIS_ENTRIES`, first match wins) is imported from it;
// - `device`, `graph` and `commandEncoder` have their real types (`TYPED_FREE_NAMES`), so graph
//   construction, `compileAsync()` and `encode()` are checked;
// - any other name is a value (or type) the reader already has, declared as `any`.
// Prop names, required props, nested port shapes and packer settings are therefore checked exactly,
// while the data views passed in are not. A block that cannot compile on its own (a signature
// listing, an elided `...` fragment) opts out with `{/* no-typecheck: reason */}` (MDX; the
// `<!-- no-typecheck: reason -->` HTML form is also read, for pages not built as MDX) on the line
// before its fence; the reason is required.

type DocSnippet = {
  /** Doc page relative to `DOCS_ROOT`. */
  page: string;
  /** 1-based line of the opening fence. */
  fenceLine: number;
  /** Block text between the fences. */
  code: string;
  /** Reason given by a `no-typecheck` opt-out comment, if any. */
  optOutReason?: string;
};

const OPT_OUT =
  /^(?:<!--\s*no-typecheck(?::\s*(.*?))?\s*-->|\{\/\*\s*no-typecheck(?::\s*(.*?))?\s*\*\/\})$/;

/** Free names with real types. Values are `[import specifier, type expression]`. */
const TYPED_FREE_NAMES: Record<string, [string, string]> = {
  device: ['@luma.gl/core', 'Device'],
  graph: ['@luma.gl/gpgpu/gpu-core', 'GPUCommandGraph'],
  commandEncoder: ['@luma.gl/core', 'CommandEncoder']
};

/** Diagnostics that report a name with no declaration. */
const MISSING_NAME_CODES = new Set([2304, 2552, 2749, 18004, 2503]);

/** Extracts the ```ts blocks of one page. */
function extractSnippets(page: string): DocSnippet[] {
  const lines = readDocPage(page).split('\n');
  const snippets: DocSnippet[] = [];
  for (let index = 0; index < lines.length; index++) {
    if (lines[index].trim() !== '```ts') {
      continue;
    }
    const fenceLine = index + 1;
    let previous = index - 1;
    while (previous >= 0 && lines[previous].trim() === '') {
      previous--;
    }
    const optOut = previous >= 0 ? OPT_OUT.exec(lines[previous].trim()) : null;
    const body: string[] = [];
    for (index++; index < lines.length && lines[index].trim() !== '```'; index++) {
      body.push(lines[index]);
    }
    snippets.push({
      page,
      fenceLine,
      code: body.join('\n'),
      ...(optOut ? {optOutReason: optOut[1] ?? optOut[2] ?? ''} : {})
    });
  }
  return snippets;
}

function getSnippetPath(snippet: DocSnippet): string {
  return join(
    DOCS_ROOT,
    '__doc_snippets__',
    `${snippet.page.replace(/[/.]/g, '-')}-${snippet.fenceLine}.ts`
  );
}

/** Returns the identifier a missing-name diagnostic points at and whether it is a type. */
function getMissingName(diagnostic: typescript.Diagnostic): {name: string; isType: boolean} | null {
  if (!diagnostic.file || diagnostic.start === undefined || !diagnostic.length) {
    return null;
  }
  const name = diagnostic.file.text.slice(diagnostic.start, diagnostic.start + diagnostic.length);
  if (!/^[A-Za-z_$][\w$]*$/.test(name)) {
    return null;
  }
  // Start above the identifier itself, which is an expression node in every position.
  let node: typescript.Node | undefined = findNodeAt(diagnostic.file, diagnostic.start).parent;
  let isType = false;
  while (node && !typescript.isSourceFile(node)) {
    if (typescript.isTypeNode(node)) {
      isType = true;
      break;
    }
    if (typescript.isExpression(node) || typescript.isStatement(node)) {
      break;
    }
    node = node.parent;
  }
  return {name, isType};
}

function findNodeAt(sourceFile: typescript.SourceFile, position: number): typescript.Node {
  let found: typescript.Node = sourceFile;
  const visit = (node: typescript.Node): void => {
    if (node.getStart(sourceFile) <= position && position < node.getEnd()) {
      found = node;
      node.forEachChild(visit);
    }
  };
  sourceFile.forEachChild(visit);
  return found;
}

/**
 * Names a block uses that only resolve to a browser global value such as `name`, `status` or
 * `parent` (a `declare var` of a non-callable type in a default library). In a docs fragment they
 * stand for the reader's own variable, so they are declared like any other free name.
 */
function getLibraryGlobalNames(
  program: typescript.Program,
  sourceFile: typescript.SourceFile | undefined
): Set<string> {
  const names = new Set<string>();
  if (!sourceFile) {
    return names;
  }
  const checker = program.getTypeChecker();
  const visit = (node: typescript.Node): void => {
    if (typescript.isIdentifier(node) && !names.has(node.text)) {
      const symbol = checker.getSymbolAtLocation(node);
      const declarations = symbol?.declarations ?? [];
      const isLibraryValue =
        declarations.length > 0 &&
        declarations.every(
          declaration =>
            program.isSourceFileDefaultLibrary(declaration.getSourceFile()) &&
            typescript.isVariableDeclaration(declaration)
        );
      if (isLibraryValue && symbol) {
        const type = checker.getTypeOfSymbolAtLocation(symbol, node);
        if (type.getCallSignatures().length === 0 && type.getConstructSignatures().length === 0) {
          names.add(node.text);
        }
      }
    }
    node.forEachChild(visit);
  };
  sourceFile.forEachChild(visit);
  return names;
}

/** An analysis entry export: the first entry that exports the name, and its meanings. */
type EntryExport = {entry: string; isType: boolean; isValue: boolean};

/** Maps each export of the analysis entries to the first entry that exports it. */
function getEntryExports(options: typescript.CompilerOptions): Map<string, EntryExport> {
  const probePath = join(DOCS_ROOT, '__doc_snippets__', 'entries.ts');
  const probe = ANALYSIS_ENTRIES.map(
    (entry, index) => `export * as entry${index} from '${entry}';`
  ).join('\n');
  const program = createVirtualProgram(new Map([[probePath, probe]]), options);
  const checker = program.getTypeChecker();
  const sourceFile = program.getSourceFile(probePath);
  if (!sourceFile) {
    throw new Error('entry probe missing');
  }
  const entryExports = new Map<string, EntryExport>();
  sourceFile.statements.forEach((statement, index) => {
    if (
      !typescript.isExportDeclaration(statement) ||
      !statement.moduleSpecifier ||
      !typescript.isStringLiteral(statement.moduleSpecifier)
    ) {
      return;
    }
    const moduleSymbol = checker.getSymbolAtLocation(statement.moduleSpecifier);
    if (!moduleSymbol) {
      throw new Error(`cannot resolve ${ANALYSIS_ENTRIES[index]}`);
    }
    for (const symbol of checker.getExportsOfModule(moduleSymbol)) {
      if (!entryExports.has(symbol.name)) {
        const target =
          symbol.flags & typescript.SymbolFlags.Alias ? checker.getAliasedSymbol(symbol) : symbol;
        entryExports.set(symbol.name, {
          entry: ANALYSIS_ENTRIES[index],
          isType: Boolean(target.flags & typescript.SymbolFlags.Type),
          isValue: Boolean(target.flags & typescript.SymbolFlags.Value)
        });
      }
    }
  });
  return entryExports;
}

/** Builds the prelude that declares a block's free names. */
function getPrelude(
  missingNames: Map<string, boolean>,
  entryExports: Map<string, EntryExport>
): string[] {
  const imports = new Map<string, Set<string>>();
  const declarations: string[] = [];
  const addImport = (specifier: string, name: string): void => {
    const names = imports.get(specifier) ?? new Set<string>();
    names.add(name);
    imports.set(specifier, names);
  };
  for (const [name, isType] of missingNames) {
    const typed = TYPED_FREE_NAMES[name];
    const entryExport = entryExports.get(name);
    if (typed && !isType) {
      addImport(typed[0], `type ${typed[1]} as Snippet${typed[1]}`);
      declarations.push(`declare const ${name}: Snippet${typed[1]};`);
    } else if (entryExport && (isType ? entryExport.isType : entryExport.isValue)) {
      addImport(entryExport.entry, name);
    } else if (isType) {
      declarations.push(`type ${name} = any;`);
    } else {
      declarations.push(`declare const ${name}: any;`);
    }
  }
  return [
    ...[...imports].map(
      ([specifier, names]) => `import {${[...names].join(', ')}} from '${specifier}';`
    ),
    ...declarations
  ];
}

it('analysis doc ```ts blocks type-check against the barrels', () => {
  const options = getRepositoryCompilerOptions();
  const snippets = ANALYSIS_DOC_PAGES.flatMap(extractSnippets);
  const optOuts = snippets.filter(snippet => snippet.optOutReason !== undefined);
  expect(
    optOuts.filter(snippet => !snippet.optOutReason).map(s => `${s.page}:${s.fenceLine}`),
    'every no-typecheck opt-out gives a reason'
  ).toEqual([]);
  const checked = snippets.filter(snippet => snippet.optOutReason === undefined);
  const entryExports = getEntryExports(options);

  // Pass 1: find the free names of each block.
  const firstFiles = new Map(
    checked.map(snippet => [getSnippetPath(snippet), `${snippet.code}\nexport {};\n`])
  );
  const firstProgram = createVirtualProgram(firstFiles, options);
  const missingNamesByPath = new Map<string, Map<string, boolean>>();
  for (const path of firstFiles.keys()) {
    const sourceFile = firstProgram.getSourceFile(path);
    const missingNames = new Map<string, boolean>();
    for (const diagnostic of firstProgram.getSemanticDiagnostics(sourceFile)) {
      const missing = MISSING_NAME_CODES.has(diagnostic.code) ? getMissingName(diagnostic) : null;
      if (missing) {
        missingNames.set(missing.name, (missingNames.get(missing.name) ?? false) || missing.isType);
      }
    }
    for (const name of getLibraryGlobalNames(firstProgram, sourceFile)) {
      missingNames.set(name, missingNames.get(name) ?? false);
    }
    missingNamesByPath.set(path, missingNames);
  }

  // Pass 2: declare them and report every remaining diagnostic at its doc line.
  const preludeLengths = new Map<string, number>();
  const secondFiles = new Map<string, string>();
  for (const [path, text] of firstFiles) {
    const prelude = getPrelude(missingNamesByPath.get(path) ?? new Map(), entryExports);
    preludeLengths.set(path, prelude.length);
    secondFiles.set(path, [...prelude, text].join('\n'));
  }
  const secondProgram = createVirtualProgram(secondFiles, options, firstProgram);
  const failures: string[] = [];
  for (const snippet of checked) {
    const path = getSnippetPath(snippet);
    const sourceFile = secondProgram.getSourceFile(path);
    const diagnostics = [
      ...secondProgram.getSyntacticDiagnostics(sourceFile),
      ...secondProgram.getSemanticDiagnostics(sourceFile)
    ];
    for (const diagnostic of diagnostics) {
      const formatted = formatDiagnostic(diagnostic);
      const [, line, rest] = /^(\d+):\d+ (.*)$/s.exec(formatted) ?? [undefined, '0', formatted];
      const docLine = snippet.fenceLine + Number(line) - (preludeLengths.get(path) ?? 0);
      failures.push(`${snippet.page}:${docLine} ${rest.split('\n')[0]}`);
    }
  }
  // Count shown in the handoff; a growing opt-out list needs review.
  console.log(
    `doc snippets: ${checked.length} checked, ${optOuts.length} opted out\n` +
      optOuts.map(s => `  ${s.page}:${s.fenceLine} ${s.optOutReason}`).join('\n')
  );
  expect(failures).toEqual([]);
}, 600_000);
