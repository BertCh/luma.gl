// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {describe, expect, it} from 'vitest';
import type {OperatorReduction, OperatorScope} from '../../../src/computation-views/types';
import {FORMULA_TEX_MACROS, listTexTerms} from '../../../src/computation-views/views/formula-tex';
import {DATAFRAME_GUARDS} from '../../../src/gpu-dataframe/computation-pack/guards';
import {DATAFRAME_ANALYSIS_STAGES} from '../../../src/gpu-dataframe/computation-pack/stages';
import {CLASSIFICATION_OPERATORS} from '../../../src/gpu-dataframe/computation-pack/operators/classification';
import {TEMPORAL_OPERATORS} from '../../../src/gpu-dataframe/computation-pack/operators/temporal';
import {GROUPS_JOINS_OPERATORS} from '../../../src/gpu-dataframe/computation-pack/operators/groups-joins';
import {PAIR_STATISTICS_OPERATORS} from '../../../src/gpu-dataframe/computation-pack/operators/pair-statistics';
import {INDICATORS_OPERATORS} from '../../../src/gpu-dataframe/computation-pack/operators/indicators';

const ROOT = fileURLToPath(new URL('../../../../../', import.meta.url));
const EXPECTED_ELEMENTS = [
  'GPUColumnQuantiles',
  'GPUClassBreaks',
  'GPUColorScale',
  'GPUBivariateClassification',
  'GPUColumnProfile',
  'GPUTimeWindowFilter',
  'GPUTemporalReduction',
  'GPUCalendarBuckets',
  'GPUGroupStatistics',
  'GPUKeyJoin',
  'GPUVariogram',
  'GPUSpatialCorrelogram',
  'GPURipley',
  'GPURipleyDistanceFunctions',
  'GPURipleyEnvelope',
  'GPUPointPatternIndices',
  'GPUCompositeScore',
  'GPUInequality'
];

// Exhaustive records keep the runtime validation vocabulary tied to the type-only contract.
const SCOPES: Record<OperatorScope, true> = {
  local: true,
  focal: true,
  block: true,
  zonal: true,
  global: true,
  frontier: true,
  pairwise: true,
  solver: true
};
const REDUCTIONS: Record<OperatorReduction, true> = {
  none: true,
  count: true,
  sum: true,
  'weighted-mean': true,
  argmin: true,
  histogram: true,
  mode: true,
  select: true,
  'scan-sort': true,
  solve: true
};
const FAMILIES = {
  classification: CLASSIFICATION_OPERATORS,
  temporal: TEMPORAL_OPERATORS,
  'groups-joins': GROUPS_JOINS_OPERATORS,
  'pair-statistics': PAIR_STATISTICS_OPERATORS,
  indicators: INDICATORS_OPERATORS
};
const ENTRIES = Object.values(FAMILIES).flatMap(table => Object.entries(table));

type FormulaRenderer = {
  renderToString(
    tex: string,
    options: {macros: Readonly<Record<string, string>>; trust: boolean; throwOnError: boolean}
  ): string;
};
const requireFromRoot = createRequire(resolve(ROOT, 'package.json'));
let katex: FormulaRenderer | undefined;
let katexPath: string | undefined;
try {
  katexPath = requireFromRoot.resolve('katex');
} catch (error) {
  // Absence is optional; a broken installed renderer must fail the spec rather than be skipped.
  if ((error as NodeJS.ErrnoException).code !== 'MODULE_NOT_FOUND') throw error;
}
if (katexPath) katex = requireFromRoot(katexPath) as FormulaRenderer;

function readPublicExportSources(path: string, visited = new Set<string>()): string {
  if (visited.has(path)) return '';
  visited.add(path);
  const source = readFileSync(path, 'utf8');
  const reexports = [...source.matchAll(/export\s*\*\s*from\s*'([^']+)'/g)];
  return [
    source,
    ...reexports.map(match =>
      readPublicExportSources(resolve(dirname(path), `${match[1]}.ts`), visited)
    )
  ].join('\n');
}

describe('dataframe curated analysis formulas', () => {
  it('covers exactly the 18 public analysis contributors without duplicates', () => {
    expect(ENTRIES.map(([name]) => name).sort()).toEqual([...EXPECTED_ELEMENTS].sort());
    expect(new Set(ENTRIES.map(([name]) => name)).size).toBe(18);
    const publicEntry = readPublicExportSources(
      resolve(ROOT, 'modules/experimental/src/gpu-dataframe/index.ts')
    );
    for (const name of EXPECTED_ELEMENTS) {
      expect(publicEntry).toMatch(new RegExp(`export\\s*\\{[^}]*\\b${name}\\b[^}]*\\}\\s*from`));
    }
  });

  for (const [family, table] of Object.entries(FAMILIES)) {
    for (const [name, entry] of Object.entries(table)) {
      it(`${name} has valid descriptors, stages, guards and kernel-bound terms`, () => {
        expect(entry.family).toBe(family);
        expect(Object.hasOwn(SCOPES, entry.descriptor.scope)).toBe(true);
        expect(Object.hasOwn(REDUCTIONS, entry.descriptor.reduction)).toBe(true);
        if (entry.descriptor.scopeSecondary)
          expect(Object.hasOwn(SCOPES, entry.descriptor.scopeSecondary)).toBe(true);
        expect(entry.stages.length).toBeGreaterThan(0);
        for (const stage of entry.stages) expect(DATAFRAME_ANALYSIS_STAGES).toContain(stage);
        for (const guard of entry.guards) expect(Object.hasOwn(DATAFRAME_GUARDS, guard)).toBe(true);
        expect(entry.datumKinds.length).toBeGreaterThan(0);
        for (const datum of entry.datumKinds) {
          expect(datum.kind.trim()).not.toBe('');
          expect(datum.label.trim()).not.toBe('');
          expect(datum.indexSpace).toBe('1d');
        }
        expect(entry.sourceFiles.length).toBeGreaterThan(0);
        const sourceTexts = entry.sourceFiles.map(path => {
          expect(path).toMatch(/^modules\/.*\.ts$/);
          return readFileSync(resolve(ROOT, path), 'utf8');
        });
        const formula = entry.formula;
        if ('formula' in formula) {
          expect(formula.formula).toBe('none');
          expect(formula.reason.trim()).not.toBe('');
          return;
        }
        expect(formula.id).toBe(`gpu-dataframe:${name}`);
        expect(formula.derivation).toBe('curated');
        expect(formula.tex.trim()).not.toBe('');
        expect(formula.ascii.trim()).not.toBe('');
        const symbolIds = formula.symbols.map(symbol => symbol.id);
        expect(symbolIds.length).toBeGreaterThan(0);
        expect(new Set(symbolIds).size).toBe(symbolIds.length);
        expect([...new Set(listTexTerms(formula.tex).map(term => term.symbolId))].sort()).toEqual(
          [...symbolIds].sort()
        );
        for (const symbol of formula.symbols) {
          expect(symbol.binding).toMatch(/^wgsl:[A-Za-z_][A-Za-z_0-9]*$/);
          expect(['input', 'output', 'parameter', 'index', 'intermediate', 'constant']).toContain(
            symbol.role
          );
          const identifier = symbol.binding!.slice('wgsl:'.length);
          expect(
            sourceTexts.some(source => new RegExp(`\\b${identifier}\\b`).test(source)),
            `${name}: ${symbol.binding}`
          ).toBe(true);
        }
      });
      it.skipIf(!katex)(`${name} TeX parses with the term macro`, () => {
        const formula = entry.formula;
        if ('formula' in formula) return;
        expect(() =>
          katex!.renderToString(formula.tex, {
            macros: FORMULA_TEX_MACROS,
            trust: true,
            throwOnError: true
          })
        ).not.toThrow();
      });
    }
  }
});
