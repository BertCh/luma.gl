// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {describe, expect, test} from 'vitest';
import * as spatialAnalysis from '../../../src/gpu-spatial-analysis/index';
import {CELLS_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/cells';
import {SPATIAL_ANALYSIS_STAGES} from '../../../src/gpu-spatial-analysis/computation-pack/stages';
import {SPATIAL_GUARDS} from '../../../src/gpu-spatial-analysis/computation-pack/guards';
import {SPATIAL_VARIABLE_KINDS} from '../../../src/gpu-spatial-analysis/computation-pack/variable-kinds';
import {listTexTerms} from '../../../src/computation-views/views/formula-tex';
import type {OperatorReduction, OperatorScope} from '../../../src/computation-views/types';

const ELEMENTS = [
  'GPUCellAggregation',
  'GPUCellRollup',
  'GPUCellPyramid',
  'GPUCellLevelSelection',
  'GPUPointToCell',
  'GPUCellGeometry',
  'GPUCellTopology',
  'GPUCellCompaction',
  'GPUCellGridPath',
  'GPUCellMeasures',
  'GPUCellCover',
  'GPUCellSetOutline',
  'GPUSegmentRingAssembly',
  'GPULineNoding',
  'GPUPolygonize',
  'GPUMakeValid',
  'GPUCellTableCompare'
] as const;

type CellsElement = (typeof ELEMENTS)[number];

const SOURCE_ROOT = 'modules/experimental/src/gpu-spatial-analysis';
const ELEMENT_KERNEL_SOURCES: Record<CellsElement, string[]> = {
  GPUCellAggregation: [
    `${SOURCE_ROOT}/cell-aggregation/gpu-cell-aggregation.ts`,
    `${SOURCE_ROOT}/cell-aggregation/cell-keys.ts`,
    `${SOURCE_ROOT}/cell-aggregation/cell-table.ts`,
    'modules/experimental/src/analysis-core/cell-key-wgsl.ts'
  ],
  GPUCellRollup: [
    `${SOURCE_ROOT}/cell-aggregation/gpu-cell-rollup.ts`,
    `${SOURCE_ROOT}/cell-aggregation/cell-keys.ts`,
    `${SOURCE_ROOT}/cell-aggregation/cell-table.ts`,
    'modules/experimental/src/analysis-core/cell-key-wgsl.ts'
  ],
  GPUCellPyramid: [
    `${SOURCE_ROOT}/cell-aggregation/gpu-cell-pyramid.ts`,
    `${SOURCE_ROOT}/cell-aggregation/cell-keys.ts`,
    `${SOURCE_ROOT}/cell-aggregation/cell-table.ts`,
    'modules/experimental/src/analysis-core/cell-key-wgsl.ts'
  ],
  GPUCellLevelSelection: [`${SOURCE_ROOT}/cell-aggregation/gpu-cell-level-selection.ts`],
  GPUPointToCell: [
    `${SOURCE_ROOT}/cell-indexing/gpu-point-to-cell.ts`,
    `${SOURCE_ROOT}/cell-indexing/cell-index-wgsl.ts`,
    `${SOURCE_ROOT}/cell-indexing/h3-index-wgsl.ts`
  ],
  GPUCellGeometry: [
    `${SOURCE_ROOT}/cell-indexing/gpu-cell-geometry.ts`,
    `${SOURCE_ROOT}/cell-indexing/cell-index-wgsl.ts`,
    `${SOURCE_ROOT}/cell-indexing/h3-boundary-wgsl.ts`
  ],
  GPUCellTopology: [
    `${SOURCE_ROOT}/cell-topology/gpu-cell-topology.ts`,
    `${SOURCE_ROOT}/cell-topology/cell-topology-wgsl.ts`,
    `${SOURCE_ROOT}/cell-topology/h3-neighbor-wgsl.ts`,
    `${SOURCE_ROOT}/cell-topology/h3-local-ij-wgsl.ts`
  ],
  GPUCellCompaction: [`${SOURCE_ROOT}/cell-topology/gpu-cell-compaction.ts`],
  GPUCellGridPath: [
    `${SOURCE_ROOT}/cell-topology/gpu-cell-grid-path.ts`,
    `${SOURCE_ROOT}/cell-topology/cell-topology-wgsl.ts`,
    `${SOURCE_ROOT}/cell-topology/h3-local-ij-wgsl.ts`
  ],
  GPUCellMeasures: [
    `${SOURCE_ROOT}/cell-indexing/gpu-cell-measures.ts`,
    `${SOURCE_ROOT}/cell-indexing/h3-measures-wgsl.ts`,
    `${SOURCE_ROOT}/cell-indexing/cell-index-wgsl.ts`
  ],
  GPUCellCover: [
    `${SOURCE_ROOT}/cell-cover/gpu-cell-cover.ts`,
    `${SOURCE_ROOT}/cell-cover/cell-cover-kernels.ts`,
    `${SOURCE_ROOT}/cell-cover/cell-cover-scanline.ts`
  ],
  GPUCellSetOutline: [`${SOURCE_ROOT}/cell-set-outline/gpu-cell-set-outline.ts`],
  GPUSegmentRingAssembly: [
    'modules/experimental/src/analysis-core/ring-assembly/gpu-segment-ring-assembly.ts'
  ],
  GPULineNoding: [`${SOURCE_ROOT}/geometry-topology/gpu-line-noding.ts`],
  GPUPolygonize: [`${SOURCE_ROOT}/geometry-topology/gpu-polygonize.ts`],
  GPUMakeValid: [`${SOURCE_ROOT}/geometry-topology/gpu-make-valid.ts`],
  GPUCellTableCompare: [`${SOURCE_ROOT}/cell-table-compare/gpu-cell-table-compare.ts`]
};

const SCOPES: readonly OperatorScope[] = [
  'local',
  'focal',
  'block',
  'zonal',
  'global',
  'frontier',
  'pairwise',
  'solver'
];
const REDUCTIONS: readonly OperatorReduction[] = [
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
];
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

describe('S3 cells computation pack', () => {
  test('contains exactly the requested 17 cells entries and unique curated formula IDs', () => {
    expect(Object.keys(CELLS_OPERATORS).sort()).toEqual([...ELEMENTS].sort());
    const identifiers = ELEMENTS.flatMap(element => {
      const formula = CELLS_OPERATORS[element].formula;
      return 'formula' in formula ? [] : [formula.id];
    });
    expect(new Set(identifiers).size).toBe(identifiers.length);
    expect(identifiers.length).toBeGreaterThan(0);
  });

  for (const element of ELEMENTS) {
    describe(element, () => {
      test('is exported and declares valid family, algebra, stages, data and guards', () => {
        expect(typeof spatialAnalysis[element]).toBe('function');
        const entry = CELLS_OPERATORS[element];
        expect(entry).toBeDefined();
        expect(entry.family).toBe('cells');
        expect(SCOPES).toContain(entry.descriptor.scope);
        if (entry.descriptor.scopeSecondary) {
          expect(SCOPES).toContain(entry.descriptor.scopeSecondary);
          expect(entry.descriptor.scopeSecondary).not.toBe(entry.descriptor.scope);
        }
        expect(REDUCTIONS).toContain(entry.descriptor.reduction);
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
          expect(datum.label.trim().length).toBeGreaterThan(0);
          expect(['1d', 'csr', '2d', 'batch']).toContain(datum.indexSpace);
        }
        for (const guard of entry.guards) {
          expect(Object.hasOwn(SPATIAL_GUARDS, guard)).toBe(true);
        }
        for (const choice of entry.choices) {
          expect(SPATIAL_ANALYSIS_STAGES).toContain(choice.stage);
          expect(entry.stages).toContain(choice.stage);
          expect(choice.name.trim().length).toBeGreaterThan(0);
          expect(typeof choice.sweepable).toBe('boolean');
        }
      });

      test('declares primary variables with shared kinds', () => {
        const variables = CELLS_OPERATORS[element].variables;
        expect(Object.keys(variables).length).toBeGreaterThan(0);
        for (const [name, kind] of Object.entries(variables)) {
          expect(name.trim().length).toBeGreaterThan(0);
          expect(Object.hasOwn(SPATIAL_VARIABLE_KINDS, kind), `${name}: ${kind}`).toBe(true);
        }
      });

      test('has an explicit procedural reason or balanced, tagged curated formula', () => {
        const formula = CELLS_OPERATORS[element].formula;
        if ('formula' in formula) {
          expect(formula.formula).toBe('none');
          expect(formula.reason.trim().length).toBeGreaterThan(0);
          return;
        }
        expect(formula.id).toMatch(/^s3\.[a-z]+(?:-[a-z]+)*$/);
        expect(formula.derivation).toBe('curated');
        expect(formula.tex.trim().length).toBeGreaterThan(0);
        expect(formula.ascii.trim().length).toBeGreaterThan(0);
        const symbolIdentifiers = new Set(formula.symbols.map(symbol => symbol.id));
        expect(symbolIdentifiers.size).toBe(formula.symbols.length);
        const terms = listTexTerms(formula.tex);
        const termIdentifiers = new Set(terms.map(term => term.symbolId));
        expect(terms.length).toBeGreaterThan(0);
        for (const term of terms) {
          expect(symbolIdentifiers.has(term.symbolId), term.symbolId).toBe(true);
        }
        for (const symbol of formula.symbols) {
          if (['input', 'output', 'parameter'].includes(symbol.role) || symbol.binding) {
            expect(termIdentifiers.has(symbol.id), symbol.id).toBe(true);
          }
        }
        let braceDepth = 0;
        for (const character of formula.tex) {
          if (character === '{') braceDepth++;
          if (character === '}') braceDepth--;
          expect(braceDepth).toBeGreaterThanOrEqual(0);
        }
        expect(braceDepth).toBe(0);
        // A malformed term must not silently disappear from the parsed-term contract.
        expect(terms.length).toBe((formula.tex.match(/\\term\{/g) ?? []).length);
      });

      test('binds formula symbols to identifiers in existing kernel sources', () => {
        expect(ELEMENT_KERNEL_SOURCES[element].length).toBeGreaterThan(0);
        const source = ELEMENT_KERNEL_SOURCES[element]
          .map(path => {
            expect(existsSync(path), path).toBe(true);
            return readFileSync(path, 'utf8');
          })
          .join('\n');
        const formula = CELLS_OPERATORS[element].formula;
        if ('formula' in formula) return;
        for (const symbol of formula.symbols) {
          if (!symbol.binding) continue;
          expect(symbol.binding).toMatch(/^wgsl:[A-Za-z_][A-Za-z_0-9]*$/);
          const identifier = symbol.binding.slice('wgsl:'.length);
          expect(source, `${element}: ${symbol.id} -> ${symbol.binding}`).toMatch(
            new RegExp(`\\b${identifier}\\b`)
          );
        }
      });
    });
  }

  test('guards additive cell reductions with conservation and rate checks', () => {
    for (const element of ['GPUCellAggregation', 'GPUCellRollup', 'GPUCellPyramid'] as const) {
      expect(CELLS_OPERATORS[element].guards, element).toContain('conservation');
    }
    expect(CELLS_OPERATORS['GPUCellAggregation'].guards).toContain('count-vs-rate');
  });

  test('declares procedural topology builders without closed forms', () => {
    for (const element of ['GPUSegmentRingAssembly', 'GPULineNoding', 'GPUPolygonize'] as const) {
      expect(CELLS_OPERATORS[element].formula, element).toMatchObject({formula: 'none'});
    }
  });
});
