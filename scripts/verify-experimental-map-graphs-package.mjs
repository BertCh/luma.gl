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
const recipeExportNames = [
  'GPUTimeWindowFilter',
  'GPUPointDensity',
  'GPURegionStatistics',
  'GPURegionMask',
  'GPUPickRegionMask',
  'GPUPointInPolygonJoin',
  'GPUNearestFeatureJoin',
  'GPUNetworkReachability',
  'GPUTerrainDerivatives',
  'GPUTerrainContours',
  'GPUTerrainViewshed',
  'GPUTileLODSelection',
  'GPUBufferSelection',
  'GPUZonalStatistics',
  'GPUNetworkPathExtraction',
  'GPUNetworkServiceAreas',
  'GPUNetworkNeighborhood',
  'GPUNetworkAnalyticsColumns',
  'GPUTerrainFlow',
  'GPUCostDistance',
  'GPUCostDistancePath',
  'GPURasterZonalStatistics',
  'GPUFlowAggregation',
  'GPUSpatialClustering',
  'GPUTrajectoryMetrics',
  'GPUNetworkStatistics',
  'GPUEdgeBundling',
  'GPUAttributeCrossfilter',
  'GPUResidencyArena',
  'GPUResidentRowSelection',
  'ResidencyArenaAllocator',
  'GPUTemporalReduction',
  'GPUNetworkCoarsening',
  'GPUAdjacencyMatrix',
  'GPUAdjacencyMatrixOrder',
  'GPUNetworkSubgraphFilter',
  'GPUDistanceField',
  'GPUInverseDistanceWeighting',
  'GPUFocalStatistics',
  'GPUPolygonRasterization',
  'GPURasterJoin',
  'GPUNetworkSnapping',
  'GPUNetworkCostMatrix',
  'GPUNetworkAccessibility',
  'GPUHotSpotAnalysis',
  'GPULocalMoran',
  'GPUTrajectoryPlayhead',
  'GPUTrajectoryResample',
  'GPULineSimplification',
  'GPUCellAggregation',
  'GPUCellLevelSelection',
  'GPUCellPyramid',
  'GPUCellRollup',
  'GPUPointToCell',
  'GPUCellGeometry',
  'GPUCellTopology',
  'GPUCellCompaction',
  'GPUColumnQuantiles',
  'GPUClassBreaks',
  'GPUColorScale',
  'GPUBivariateClassification',
  'GPUColumnProfile',
  'GPUCellTableCompare',
  'GPULineSegmentize',
  'GPUGreatCircleArcs',
  'GPULineSmooth',
  'GPULineChunk',
  'GPUGeometryMeasures',
  'GPUGeodesicPairs',
  'GPUGeodesicDestination',
  'GPULinearReferencing',
  'GPULineLocate',
  'GPUReliefShading',
  'GPUSolarPosition',
  'GPUSolarShadowMask',
  'GPUTerrainHorizon',
  'GPUTextureShading',
  'GPURasterArithmetic',
  'GPURasterCellStatistics',
  'GPURasterConditional',
  'GPURasterReclassify',
  'GPUWeightedOverlay',
  'GPURasterStretch',
  'GPUIsobands',
  'GPUIsolines',
  'GPURasterProfile',
  'GPURasterSampling',
  'GPUParticleAdvection',
  'GPULineIntegralConvolution',
  'GPUStreamlines',
  'GPUDotDensity',
  'GPURandomPointsInPolygon',
  'GPUNeighborSearch',
  'GPUGlobalSpatialStatistics',
  'GPUGlobalPermutationTest',
  'GPULocalPermutationTest',
  'GPUVariogram',
  'GPUSpatialCorrelogram',
  'GPURipley',
  'GPUPointPatternIndices',
  'GPUGeographicDistribution',
  'GPUEmergingHotSpots',
  'GPUCalendarBuckets',
  'GPUChangeDetection',
  'GPUCellCover',
  'GPUGroupStatistics',
  'GPUKeyJoin',
  'GPUOrdinaryLeastSquares',
  'GPUGeographicallyWeightedRegression',
  'GPUCompositeScore',
  'GPUInequality',
  'GPUMapGraphParameterBuffer'
];

assert.deepEqual(packageJson.exports?.['./map-graphs'], {
  types: './dist/map-graphs/index.d.ts',
  import: './dist/map-graphs/index.js',
  require: './dist/map-graphs/index.cjs'
});

const ecmaScriptModule = await import('@luma.gl/experimental/map-graphs');
const commonJsModule = require('@luma.gl/experimental/map-graphs');
const ecmaScriptRootModule = await import('@luma.gl/experimental');
const commonJsRootModule = require('@luma.gl/experimental');

for (const exportName of recipeExportNames) {
  assert.equal(typeof ecmaScriptModule[exportName], 'function', exportName);
  assert.equal(typeof commonJsModule[exportName], 'function', exportName);
  assert.equal(exportName in ecmaScriptRootModule, false, `${exportName} leaked into the root`);
  assert.equal(exportName in commonJsRootModule, false, `${exportName} leaked into the root`);
}

const temporaryDirectory = mkdtempSync(path.join(repositoryRoot, '.map-graphs-package-'));
try {
  const typeTestPath = path.join(temporaryDirectory, 'index.mts');
  writeFileSync(
    typeTestPath,
    `import {
  GPUNetworkReachability,
  GPUPointDensity,
  GPUTileLODSelection,
  GPUTimeWindowFilter,
  type GPUMapGraphCompactOutput,
  type GPUMapGraphRecipe,
  type GPUPointDensityProps,
  type GPUTileLODSelectionProps
} from '@luma.gl/experimental/map-graphs';

declare const densityProps: GPUPointDensityProps;
declare const tileProps: GPUTileLODSelectionProps;
declare const output: GPUMapGraphCompactOutput;
const recipes: GPUMapGraphRecipe[] = [new GPUPointDensity(densityProps), new GPUTileLODSelection(tileProps)];
void recipes;
void output;
void GPUNetworkReachability;
void GPUTimeWindowFilter;

// @ts-expect-error Map-graph recipes stay isolated from the experimental root.
import {GPUPointDensity as RootGPUPointDensity} from '@luma.gl/experimental';
void RootGPUPointDensity;
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

console.log('Verified @luma.gl/experimental/map-graphs ESM, CJS, and declaration imports.');
