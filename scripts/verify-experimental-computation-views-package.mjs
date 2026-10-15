// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

import typescript from 'typescript';

const require = createRequire(import.meta.url);
const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const packageJson = require('../modules/experimental/package.json');
const entryExports = {
  'computation-views': ['buildLensTwin', 'ViewRegistry'],
  'computation-lenses': ['registerLensScenario']
};
const ecmaScriptRootModule = await import('@luma.gl/experimental');
const commonJsRootModule = require('@luma.gl/experimental');
for (const [subpath, exportNames] of Object.entries(entryExports)) {
  assert.deepEqual(packageJson.exports?.[`./${subpath}`], {
    types: `./dist/${subpath}/index.d.ts`,
    import: `./dist/${subpath}/index.js`,
    require: `./dist/${subpath}/index.cjs`
  });
  const ecmaScriptEntryModule = await import(`@luma.gl/experimental/${subpath}`);
  const commonJsEntryModule = require(`@luma.gl/experimental/${subpath}`);
  for (const exportName of exportNames) {
    assert.equal(typeof ecmaScriptEntryModule[exportName], 'function', exportName);
    assert.equal(typeof commonJsEntryModule[exportName], 'function', exportName);
    assert.equal(exportName in ecmaScriptRootModule, false, `${exportName} leaked into the root`);
    assert.equal(exportName in commonJsRootModule, false, `${exportName} leaked into the root`);
  }
}

const temporaryDirectory = mkdtempSync(path.join(repositoryRoot, '.computation-views-package-'));
try {
  const typeTestPath = path.join(temporaryDirectory, 'computation-views.mts');
  writeFileSync(
    typeTestPath,
    `import {buildLensTwin, ViewRegistry, type LensDirective, type LensTwin, type ViewContext, type ViewRegistration} from '@luma.gl/experimental/computation-views';
import {registerLensScenario, type RecordedLensFile, type LensScenarioDescriptor} from '@luma.gl/experimental/computation-lenses';
import type {GPUComputationRecordWithSidecars} from '@luma.gl/gpgpu';

declare const recording: RecordedLensFile;
declare const directive: LensDirective;
declare const scenario: LensScenarioDescriptor;
declare const context: ViewContext;
declare const view: ViewRegistration;
const record: GPUComputationRecordWithSidecars = recording.record;
const twin: LensTwin = buildLensTwin(recording, directive, {registry: new ViewRegistry()});
registerLensScenario(scenario);
void record;
void twin;
void context;
void view;
// @ts-expect-error Views stay isolated from the experimental root.
import {buildLensTwin as RootTwin} from '@luma.gl/experimental';
// @ts-expect-error Views stay isolated from the experimental root.
import {ViewRegistry as RootRegistry} from '@luma.gl/experimental';
// @ts-expect-error Scenarios stay isolated from the experimental root.
import {registerLensScenario as rootRegister} from '@luma.gl/experimental';
void RootTwin;
void RootRegistry;
void rootRegister;
`
  );
  const program = typescript.createProgram([typeTestPath], {
    module: typescript.ModuleKind.NodeNext,
    moduleResolution: typescript.ModuleResolutionKind.NodeNext,
    noEmit: true,
    skipLibCheck: true,
    strict: true,
    target: typescript.ScriptTarget.ES2022,
    types: []
  });
  const diagnostics = typescript.getPreEmitDiagnostics(program);
  assert.equal(
    diagnostics.length,
    0,
    typescript.formatDiagnosticsWithColorAndContext(diagnostics, {
      getCanonicalFileName: fileName => fileName,
      getCurrentDirectory: () => temporaryDirectory,
      getNewLine: () => '\n'
    })
  );
} finally {
  rmSync(temporaryDirectory, {force: true, recursive: true});
}

process.stdout.write('Verified computation views and lenses ESM, CJS, and declaration imports.\n');
