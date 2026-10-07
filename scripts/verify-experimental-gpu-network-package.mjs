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

assert.deepEqual(packageJson.exports?.['./gpu-network'], {
  types: './dist/gpu-network/index.d.ts',
  import: './dist/gpu-network/index.js',
  require: './dist/gpu-network/index.cjs'
});

const ecmaScriptEntryModule = await import('@luma.gl/experimental/gpu-network');
const commonJsEntryModule = require('@luma.gl/experimental/gpu-network');
const ecmaScriptRootModule = await import('@luma.gl/experimental');
const commonJsRootModule = require('@luma.gl/experimental');

const contributorExportNames = [
  'GPUNetworkReachability',
  'GPUNetworkPathExtraction',
  'GPUNetworkServiceAreas',
  'GPUNetworkNeighborhood',
  'GPUNetworkAnalyticsColumns',
  'GPUFlowAggregation',
  'GPUNetworkStatistics',
  'GPUEdgeBundling',
  'GPUNetworkCoarsening',
  'GPUAdjacencyMatrix',
  'GPUAdjacencyMatrixOrder',
  'GPUNetworkSubgraphFilter',
  'GPUNetworkSnapping',
  'GPUNetworkCostMatrix',
  'GPUNetworkAccessibility',
  'GPUParameterBuffer',
  'GPUNetworkIsochrones',
  'GPUNetworkNoding',
  'GPUNetworkKFunction',
  'GPUNetworkLineGraph',
  'GPUMapMatching'
];

for (const exportName of contributorExportNames) {
  assert.equal(typeof ecmaScriptEntryModule[exportName], 'function', exportName);
  assert.equal(typeof commonJsEntryModule[exportName], 'function', exportName);
  assert.equal(exportName in ecmaScriptRootModule, false, `${exportName} leaked into the root`);
  assert.equal(exportName in commonJsRootModule, false, `${exportName} leaked into the root`);
}

const temporaryDirectory = mkdtempSync(path.join(repositoryRoot, '.gpu-network-package-'));
try {

  const contributorTypeTestPath = path.join(temporaryDirectory, 'contributors.mts');
  writeFileSync(
    contributorTypeTestPath,
    `import {
  ${contributorExportNames.join(',\n  ')},
  type GPUCompactOutput,
  type GPUUint32Rows
} from '@luma.gl/experimental/gpu-network';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';

const contributorConstructors = [
  ${contributorExportNames.join(',\n  ')}
];
declare const output: GPUCompactOutput;
declare const rows: GPUUint32Rows;
declare const contributor: GPUCommandNodeProducer;
void contributorConstructors;
void output;
void rows;
void contributor;

// @ts-expect-error Analysis contributors stay isolated from the experimental root.
import {GPUNetworkReachability as RootContributor} from '@luma.gl/experimental';
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

console.log('Verified @luma.gl/experimental/gpu-network ESM, CJS, and declaration imports.');
