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

assert.deepEqual(packageJson.exports?.['./gpu-spatial-analysis'], {
  types: './dist/gpu-spatial-analysis/index.d.ts',
  import: './dist/gpu-spatial-analysis/index.js',
  require: './dist/gpu-spatial-analysis/index.cjs'
});

const ecmaScriptEntryModule = await import('@luma.gl/experimental/gpu-spatial-analysis');
const commonJsEntryModule = require('@luma.gl/experimental/gpu-spatial-analysis');
const ecmaScriptGeospatialModule = await import('@luma.gl/experimental/geospatial');
const commonJsGeospatialModule = require('@luma.gl/experimental/geospatial');
const ecmaScriptRootModule = await import('@luma.gl/experimental');
const commonJsRootModule = require('@luma.gl/experimental');

assert.deepEqual(
  commonJsEntryModule.GPU_SPATIAL_ANALYSIS_OPERATORS,
  ecmaScriptEntryModule.GPU_SPATIAL_ANALYSIS_OPERATORS,
  'ESM and CJS operator metadata differ'
);

// The ontology also describes related entries. Validate each operator against its declared owner,
// including declaration imports. Helpers that are not analytical operators remain explicit.
const exportNamesByPackage = new Map([
  ['@luma.gl/experimental/gpu-spatial-analysis', new Set(['getGPUGridCornerCount'])]
]);
for (const operator of ecmaScriptEntryModule.GPU_SPATIAL_ANALYSIS_OPERATORS) {
  const exportNames = exportNamesByPackage.get(operator.package) ?? new Set();
  exportNames.add(operator.exportName);
  exportNamesByPackage.set(operator.package, exportNames);
}

const contributorImports = [];
const contributorExportNames = [];
for (const [packageName, exportNames] of exportNamesByPackage) {
  const ecmaScriptOwnerModule = await import(packageName);
  const commonJsOwnerModule = require(packageName);
  const sortedExportNames = [...exportNames].sort();
  contributorImports.push(`import {${sortedExportNames.join(', ')}} from '${packageName}';`);
  contributorExportNames.push(...sortedExportNames);
  for (const exportName of sortedExportNames) {
    const qualifiedName = `${packageName}: ${exportName}`;
    assert.equal(typeof ecmaScriptOwnerModule[exportName], 'function', qualifiedName);
    assert.equal(typeof commonJsOwnerModule[exportName], 'function', qualifiedName);
    assert.equal(exportName in ecmaScriptRootModule, false, `${exportName} leaked into the root`);
    assert.equal(exportName in commonJsRootModule, false, `${exportName} leaked into the root`);
  }
}

// Analysis contributors live only here; the upstream geospatial entry stays kernels-only.
for (const exportName of ['GPUPointDensity', 'GPUNeighborSearch', 'GPUParameterBuffer']) {
  assert.equal(exportName in ecmaScriptGeospatialModule, false, `${exportName} leaked into geospatial`);
  assert.equal(exportName in commonJsGeospatialModule, false, `${exportName} leaked into geospatial`);
}

const temporaryDirectory = mkdtempSync(path.join(repositoryRoot, '.gpu-spatial-analysis-package-'));
try {

  const contributorTypeTestPath = path.join(temporaryDirectory, 'contributors.mts');
  writeFileSync(
    contributorTypeTestPath,
    `${contributorImports.join('\n')}
import {
  type GPUCompactOutput,
  type GPUGeometryCleanupPointOutput,
  type GPUUint32Rows
} from '@luma.gl/experimental/gpu-spatial-analysis';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';

const contributorConstructors = [
  ${contributorExportNames.join(',\n  ')}
];
declare const output: GPUCompactOutput;
declare const cleanupPointOutput: GPUGeometryCleanupPointOutput;
declare const rows: GPUUint32Rows;
declare const contributor: GPUCommandNodeProducer;
void contributorConstructors;
void output;
void cleanupPointOutput;
void rows;
void contributor;

// @ts-expect-error Analysis contributors stay isolated from the experimental root.
import {GPUPointDensity as RootContributor} from '@luma.gl/experimental';
void RootContributor;
`
  );
  const contributorProgram = typescript.createProgram([contributorTypeTestPath], {
    module: typescript.ModuleKind.NodeNext,
    moduleResolution: typescript.ModuleResolutionKind.NodeNext,
    noEmit: true,
    skipLibCheck: true,
    strict: true,
    target: typescript.ScriptTarget.ES2022,
    types: []
  });
  const contributorDiagnostics = typescript.getPreEmitDiagnostics(contributorProgram);
  assert.equal(
    contributorDiagnostics.length,
    0,
    typescript.formatDiagnosticsWithColorAndContext(contributorDiagnostics, {
      getCanonicalFileName: fileName => fileName,
      getCurrentDirectory: () => temporaryDirectory,
      getNewLine: () => '\n'
    })
  );
} finally {
  rmSync(temporaryDirectory, {force: true, recursive: true});
}

console.log('Verified @luma.gl/experimental/gpu-spatial-analysis ESM, CJS, and declaration imports.');
