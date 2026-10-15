// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {describe, expect, it} from 'vitest';
import * as spatialAnalysis from '../../../src/gpu-spatial-analysis/index';
import {listTexTerms} from '../../../src/computation-views/views/formula-tex';
import type {OperatorReduction, OperatorScope} from '../../../src/computation-views/types';
import {LINES_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/lines';
import {SPATIAL_ANALYSIS_STAGES} from '../../../src/gpu-spatial-analysis/computation-pack/stages';
import {SPATIAL_GUARDS} from '../../../src/gpu-spatial-analysis/computation-pack/guards';
import {SPATIAL_VARIABLE_KINDS} from '../../../src/gpu-spatial-analysis/computation-pack/variable-kinds';

const ELEMENTS = [
  'GPULineSegmentize',
  'GPUGreatCircleArcs',
  'GPULineSmooth',
  'GPULineChunk',
  'GPUGeometryMeasures',
  'GPUGeodesicPairs',
  'GPUGeodesicDestination',
  'GPULocalFrameProjection',
  'GPULinearReferencing',
  'GPULineLocate',
  'GPULineSplit',
  'GPULineMerge',
  'GPULineClipByPolygon',
  'GPUSharedPaths',
  'GPULineSimplification',
  'GPUCoverageSimplification',
  'GPUCoverageDissolve',
  'GPUTrajectoryMetrics',
  'GPUTrajectoryPlayhead',
  'GPUTrajectoryResample',
  'GPUZoneEvents',
  'GPUTrajectoryEncounters',
  'GPUTrajectoryClosestApproach',
  'GPUTrackSimilarity',
  'addFleetDwellZoneEventsRecipe'
] as const;

type Element = (typeof ELEMENTS)[number];
type Contributor = Exclude<Element, 'addFleetDwellZoneEventsRecipe'>;

const SOURCE_ROOT = 'modules/experimental/src/gpu-spatial-analysis/';
const REPOSITORY_ROOT = new URL('../../../../../', import.meta.url);

// Explicit source ownership, including delegated WGSL generators where the contributor composes them.
const ELEMENT_KERNEL_SOURCES: Record<Contributor, string[]> = {
  GPULineSegmentize: [
    `${SOURCE_ROOT}line-segmentize/line-segmentize-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`
  ],
  GPUGreatCircleArcs: [
    `${SOURCE_ROOT}line-segmentize/line-segmentize-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`
  ],
  GPULineSmooth: [`${SOURCE_ROOT}line-segmentize/line-smooth-kernels.ts`],
  GPULineChunk: [
    `${SOURCE_ROOT}line-segmentize/line-chunk-kernels.ts`,
    `${SOURCE_ROOT}line-segmentize/line-segmentize-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`
  ],
  GPUGeometryMeasures: [
    `${SOURCE_ROOT}geometry-measures/geometry-measures-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/feature-measures-cooperative.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`
  ],
  GPUGeodesicPairs: [
    `${SOURCE_ROOT}geometry-measures/gpu-geodesic-pairs.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`,
    `${SOURCE_ROOT}geometry-measures/rhumb-wgsl.ts`
  ],
  GPUGeodesicDestination: [
    `${SOURCE_ROOT}geometry-measures/gpu-geodesic-destination.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`,
    `${SOURCE_ROOT}geometry-measures/rhumb-wgsl.ts`
  ],
  GPULocalFrameProjection: ['modules/experimental/src/gpu-project/projection-shader.ts'],
  GPULinearReferencing: [
    `${SOURCE_ROOT}linear-referencing/linear-referencing-kernels.ts`,
    `${SOURCE_ROOT}line-segmentize/line-segmentize-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`
  ],
  GPULineLocate: [
    `${SOURCE_ROOT}linear-referencing/linear-referencing-kernels.ts`,
    `${SOURCE_ROOT}line-segmentize/line-segmentize-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`
  ],
  GPULineSplit: [`${SOURCE_ROOT}line-split/gpu-line-split.ts`],
  GPULineMerge: [`${SOURCE_ROOT}line-merge/gpu-line-merge.ts`],
  GPULineClipByPolygon: [`${SOURCE_ROOT}line-clip/gpu-line-clip-by-polygon.ts`],
  GPUSharedPaths: [`${SOURCE_ROOT}line-clip/gpu-shared-paths.ts`],
  GPULineSimplification: [
    `${SOURCE_ROOT}line-simplification/line-simplification-kernels.ts`,
    `${SOURCE_ROOT}line-simplification/visvalingam-kernels.ts`
  ],
  GPUCoverageSimplification: [
    `${SOURCE_ROOT}polygon-coverage-simplification/gpu-coverage-simplification.ts`,
    `${SOURCE_ROOT}line-simplification/line-simplification-kernels.ts`
  ],
  GPUCoverageDissolve: [
    `${SOURCE_ROOT}polygon-coverage-topology/gpu-coverage-dissolve.ts`,
    `${SOURCE_ROOT}polygon-coverage-topology/coverage-topology-kernels.ts`,
    'modules/experimental/src/analysis-core/ring-assembly/gpu-segment-ring-assembly.ts'
  ],
  GPUTrajectoryMetrics: [
    `${SOURCE_ROOT}trajectory-analysis/trajectory-metrics-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`
  ],
  GPUTrajectoryPlayhead: [
    `${SOURCE_ROOT}trajectory-interpolation/trajectory-interpolation-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`
  ],
  GPUTrajectoryResample: [
    `${SOURCE_ROOT}trajectory-interpolation/trajectory-interpolation-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`
  ],
  GPUZoneEvents: [`${SOURCE_ROOT}trajectory-zones/zone-events-kernels.ts`],
  GPUTrajectoryEncounters: [
    `${SOURCE_ROOT}trajectory-encounters/encounter-kernels.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`
  ],
  GPUTrajectoryClosestApproach: [`${SOURCE_ROOT}trajectory-encounters/closest-approach-kernels.ts`],
  GPUTrackSimilarity: [
    `${SOURCE_ROOT}track-similarity/gpu-track-similarity.ts`,
    `${SOURCE_ROOT}track-similarity/track-similarity-dynamic-programs.ts`,
    `${SOURCE_ROOT}geometry-measures/geodesic-wgsl.ts`
  ]
};

const SCOPES = [
  'local',
  'focal',
  'block',
  'zonal',
  'global',
  'frontier',
  'pairwise',
  'solver'
] as const satisfies readonly OperatorScope[];
const REDUCTIONS = [
  'none',
  'count',
  'sum',
  'weighted-mean',
  'argmin',
  'histogram',
  'mode',
  'select',
  'scan-sort',
  'solve'
] as const satisfies readonly OperatorReduction[];
const DATUM_KINDS = [
  'feature',
  'vertex',
  'segment',
  'cell',
  'line',
  'track',
  'fix',
  'pair',
  'triangle',
  'ring'
];

function expectBalancedBraces(tex: string): void {
  let depth = 0;
  for (let index = 0; index < tex.length; index++) {
    if (tex[index] === '\\') {
      index++;
    } else if (tex[index] === '{') {
      depth++;
    } else if (tex[index] === '}') {
      depth--;
      expect(depth).toBeGreaterThanOrEqual(0);
    }
  }
  expect(depth).toBe(0);
}

describe('S3 lines computation pack', () => {
  it('covers exactly the 25 requested elements and has unique formula ids', () => {
    expect(Object.keys(LINES_OPERATORS).sort()).toEqual([...ELEMENTS].sort());
    const identifiers = ELEMENTS.flatMap(element => {
      const formula = LINES_OPERATORS[element].formula;
      return 'id' in formula ? [formula.id] : [];
    });
    expect(new Set(identifiers).size).toBe(identifiers.length);
  });

  for (const element of ELEMENTS) {
    describe(element, () => {
      it('declares a valid lines descriptor and spatial vocabulary', () => {
        const entry = LINES_OPERATORS[element];
        expect(entry).toBeDefined();
        expect(entry.family).toBe('lines');
        expect(SCOPES).toContain(entry.descriptor.scope);
        if (entry.descriptor.scopeSecondary)
          expect(SCOPES).toContain(entry.descriptor.scopeSecondary);
        expect(REDUCTIONS).toContain(entry.descriptor.reduction);
        expect(Object.keys(entry.variables).length).toBeGreaterThan(0);
        for (const kind of Object.values(entry.variables))
          expect(SPATIAL_VARIABLE_KINDS).toHaveProperty(kind);
        expect(entry.stages.length).toBeGreaterThan(0);
        let previousStage = -1;
        for (const stage of entry.stages) {
          const stageIndex = SPATIAL_ANALYSIS_STAGES.indexOf(stage);
          expect(stageIndex).toBeGreaterThan(previousStage);
          previousStage = stageIndex;
        }
        expect(entry.datumKinds.length).toBeGreaterThan(0);
        for (const datum of entry.datumKinds) {
          expect(DATUM_KINDS).toContain(datum.kind);
          expect(datum.label.trim()).not.toBe('');
          expect(['1d', 'csr', '2d', 'batch']).toContain(datum.indexSpace);
        }
        for (const guard of entry.guards) expect(SPATIAL_GUARDS).toHaveProperty(guard);
        for (const choice of entry.choices) {
          expect(SPATIAL_ANALYSIS_STAGES).toContain(choice.stage);
          expect(choice.name.trim()).not.toBe('');
          expect(typeof choice.sweepable).toBe('boolean');
        }
      });

      it('fulfills the curated formula or explicit none contract', () => {
        const formula = LINES_OPERATORS[element].formula;
        if ('formula' in formula) {
          expect(formula.formula).toBe('none');
          expect(formula.reason.trim()).not.toBe('');
          return;
        }
        expect(formula.id).toMatch(/^s3\.[a-z]+(?:-[a-z]+)*$/);
        expect(formula.derivation).toBe('curated');
        expect(formula.tex.trim()).not.toBe('');
        expect(formula.ascii.trim()).not.toBe('');
        expectBalancedBraces(formula.tex);
        const terms = listTexTerms(formula.tex);
        expect(terms.length).toBeGreaterThan(0);
        expect(terms.length).toBe([...formula.tex.matchAll(/\\term\{/g)].length);
        const symbolIds = formula.symbols.map(symbol => symbol.id);
        expect(new Set(symbolIds).size).toBe(symbolIds.length);
        for (const term of terms) expect(symbolIds).toContain(term.symbolId);
        for (const symbol of formula.symbols) {
          if (['input', 'output', 'parameter'].includes(symbol.role)) {
            expect(terms.map(term => term.symbolId)).toContain(symbol.id);
            expect(symbol.binding).toMatch(/^wgsl:[A-Za-z_][A-Za-z_0-9]*$/);
          }
        }
      });

      if (element !== 'addFleetDwellZoneEventsRecipe') {
        it('binds symbols to identifiers in existing contributor kernel sources', () => {
          const paths = ELEMENT_KERNEL_SOURCES[element];
          expect(paths.length).toBeGreaterThan(0);
          const source = paths
            .map(path => {
              const absolutePath = fileURLToPath(new URL(path, REPOSITORY_ROOT));
              expect(existsSync(absolutePath), path).toBe(true);
              return readFileSync(absolutePath, 'utf8');
            })
            .join('\n');
          const formula = LINES_OPERATORS[element].formula;
          if ('formula' in formula) return;
          for (const symbol of formula.symbols) {
            if (!symbol.binding?.startsWith('wgsl:')) continue;
            const identifier = symbol.binding.slice(5);
            expect(identifier).toMatch(/^[A-Za-z_][A-Za-z_0-9]*$/);
            expect(source, `${element}: ${symbol.id} -> ${identifier}`).toMatch(
              new RegExp(`\\b${identifier}\\b`)
            );
          }
        });
      }

      it('is a real function exported by the spatial analysis barrel', () => {
        expect(typeof spatialAnalysis[element]).toBe('function');
      });
    });
  }

  it('describes the fleet recipe at contributor grain in the lines family', () => {
    const recipe = LINES_OPERATORS['addFleetDwellZoneEventsRecipe'];
    expect(recipe.descriptor).toEqual({
      scope: 'zonal',
      scopeSecondary: 'pairwise',
      reduction: 'sum'
    });
    expect(recipe.formula).toEqual({
      formula: 'none',
      reason: 'a recipe composes contributors; its graph rung shows them'
    });
  });
});
