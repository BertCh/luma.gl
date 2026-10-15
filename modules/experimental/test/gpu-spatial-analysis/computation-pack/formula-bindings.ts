// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect} from 'vitest';
import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';
import {listTexTerms} from '../../../src/computation-views/views/formula-tex';
import {SPATIAL_GUARDS} from '../../../src/gpu-spatial-analysis/computation-pack/guards';
import {isSpatialAnalysisStage} from '../../../src/gpu-spatial-analysis/computation-pack/stages';
import {SPATIAL_VARIABLE_KINDS} from '../../../src/gpu-spatial-analysis/computation-pack/variable-kinds';
import type {SpatialOperatorEntry} from '../../../src/gpu-spatial-analysis/computation-pack/types';

/** Checks curated symbols against whole identifiers in repo-relative WGSL generator sources. */
export function expectFormulaBindingsExist(
  formula: GPUComputationFormulaRef,
  sourceFiles: string[]
): void {
  const sources = sourceFiles.map(path => readFileSync(path, 'utf8'));
  expect(formula.derivation).toBe('curated');
  expect(formula.ascii.trim()).not.toBe('');
  const symbols = formula.symbols.map(symbol => symbol.id);
  expect(new Set(symbols).size).toBe(symbols.length);
  expect([...new Set(listTexTerms(formula.tex).map(term => term.symbolId))].sort()).toEqual(
    [...symbols].sort()
  );
  for (const symbol of formula.symbols) {
    if (!symbol.binding?.startsWith('wgsl:')) continue;
    const identifier = symbol.binding.slice(5);
    expect(identifier).toMatch(/^[A-Za-z_][A-Za-z0-9_]*$/);
    const pattern = new RegExp(`(?<![A-Za-z0-9_])${identifier}(?![A-Za-z0-9_])`);
    expect(
      sources.some(source => pattern.test(source)),
      `${formula.id}: ${symbol.binding}`
    ).toBe(true);
  }
}

/** Checks an entry; use `family/Element` as name to check its owning family file too. */
export function expectSpatialEntryWellFormed(name: string, entry: SpatialOperatorEntry): void {
  const [family] = name.split('/');
  if (name.includes('/')) expect(entry.family, name).toBe(family);
  expect(entry.stages.length, name).toBeGreaterThan(0);
  for (const stage of entry.stages) expect(isSpatialAnalysisStage(stage), name).toBe(true);
  for (const guard of entry.guards) expect(guard in SPATIAL_GUARDS, name).toBe(true);
  for (const kind of Object.values(entry.variables))
    expect(kind in SPATIAL_VARIABLE_KINDS, name).toBe(true);
  for (const choice of entry.choices) expect(isSpatialAnalysisStage(choice.stage), name).toBe(true);
  if ('formula' in entry.formula) {
    expect(entry.formula.formula).toBe('none');
    expect(entry.formula.reason.trim(), name).not.toBe('');
  }
}
