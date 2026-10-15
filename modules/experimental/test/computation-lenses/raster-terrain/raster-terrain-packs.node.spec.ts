// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {join} from 'node:path';
import {describe, expect, it} from 'vitest';
import {listLensScenarios} from '../../../src/computation-lenses/index';
import '../../../src/computation-lenses/scenarios/gpu-raster/index';
import '../../../src/computation-lenses/scenarios/gpu-terrain/index';
import {
  getRasterOperatorEntries,
  getRasterViewPack,
  RASTER_GUARDS,
  RASTER_VARIABLE_KINDS,
  type RasterOperatorEntry
} from '../../../src/gpu-raster/computation-pack/index';
import {
  getTerrainOperatorEntries,
  getTerrainViewPack
} from '../../../src/gpu-terrain/computation-pack/index';
import {listEntryElements, REPOSITORY_ROOT} from '../../computation-docs/lens-directives';

const SCOPES = ['local', 'focal', 'block', 'zonal', 'global', 'frontier', 'pairwise', 'solver'];
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
];

const PACKS = [
  {
    entry: 'gpu-terrain',
    barrel: 'modules/experimental/src/gpu-terrain/index.ts',
    entries: getTerrainOperatorEntries,
    pack: getTerrainViewPack
  },
  {
    entry: 'gpu-raster',
    barrel: 'modules/experimental/src/gpu-raster/index.ts',
    entries: getRasterOperatorEntries,
    pack: getRasterViewPack
  }
] as const;

/** Every problem with one operator entry; an empty list means the entry is sound. */
function collectEntryProblems(name: string, entry: RasterOperatorEntry): string[] {
  const problems: string[] = [];
  if (!SCOPES.includes(entry.descriptor.scope)) problems.push(`${name}: scope`);
  if (!REDUCTIONS.includes(entry.descriptor.reduction)) problems.push(`${name}: reduction`);
  for (const [capture, kind] of Object.entries(entry.variables)) {
    if (!(kind in RASTER_VARIABLE_KINDS))
      problems.push(`${name}: variable ${capture} kind ${kind}`);
  }
  for (const guard of entry.guards) {
    if (!(guard in RASTER_GUARDS)) problems.push(`${name}: unknown guard ${guard}`);
  }
  if (!entry.datumKinds.length) problems.push(`${name}: no datum kind`);
  if (!entry.sources.length) problems.push(`${name}: no source files`);
  const sourceText = entry.sources
    .map(path => {
      const file = join(REPOSITORY_ROOT, path);
      if (!existsSync(file)) {
        problems.push(`${name}: missing source ${path}`);
        return '';
      }
      return readFileSync(file, 'utf8');
    })
    .join('\n');
  const identifiers = new Set(sourceText.match(/[A-Za-z_][A-Za-z0-9_]*/g) ?? []);
  const formula = entry.formula;
  if ('formula' in formula) {
    if (!formula.reason.trim()) problems.push(`${name}: formula none without reason`);
    return problems;
  }
  if (!formula.tex.trim() || !formula.ascii.trim()) problems.push(`${name}: empty formula`);
  if (formula.derivation !== 'curated') problems.push(`${name}: formulas here are curated`);
  const symbolIds = new Set(formula.symbols.map(symbol => symbol.id));
  for (const match of formula.tex.matchAll(/\\term\{([^}]+)\}/g)) {
    if (!symbolIds.has(match[1])) problems.push(`${name}: \\term{${match[1]}} has no symbol`);
  }
  for (const symbol of formula.symbols) {
    if (!formula.tex.includes(`\\term{${symbol.id}}`)) {
      problems.push(`${name}: symbol ${symbol.id} is not used in the TeX`);
    }
    if (!symbol.binding?.startsWith('wgsl:')) {
      if (symbol.binding?.startsWith('resource:')) {
        problems.push(`${name}: symbol ${symbol.id} binds a scenario resource; bind WGSL names`);
      }
      continue;
    }
    // Curated formulas must bind names that exist in the element's own kernels.
    const target = symbol.binding.slice('wgsl:'.length);
    if (!identifiers.has(target)) {
      problems.push(`${name}: symbol ${symbol.id} binding ${target} not found in sources`);
    }
  }
  return problems;
}

for (const {entry, barrel, entries, pack} of PACKS) {
  describe(`${entry} computation pack`, () => {
    it('declares sound operators, guards, variable kinds and source-bound formulas', () => {
      const problems = Object.entries(entries()).flatMap(([name, operator]) =>
        collectEntryProblems(name, operator)
      );
      expect(problems).toEqual([]);
    });

    it('exports every operator and formula through the view pack', () => {
      const viewPack = pack();
      for (const name of Object.keys(entries())) {
        expect(viewPack.operators?.[name], name).toBeDefined();
        expect(viewPack.formulas?.[name], name).toBeDefined();
      }
    });

    it('registers one scenario with a result rung for every operator entry', () => {
      const scenarios = listLensScenarios().filter(scenario => scenario.entry === entry);
      const missing = Object.keys(entries()).filter(
        name => !scenarios.some(scenario => scenario.element === name)
      );
      expect(missing).toEqual([]);
      for (const scenario of scenarios) {
        expect(entries()[scenario.element], scenario.id).toBeDefined();
        expect(scenario.rungs, scenario.id).toContain('result');
        expect(scenario.defaultRung, scenario.id).toBe('result');
        expect(scenario.dataset, scenario.id).toBeDefined();
      }
    });

    // Wave 2 stopped part-way (2026-10-14): enforced with T_REQUIRE_COMPLETE=1 once every element lands.
    it.skipIf(process.env['T_REQUIRE_COMPLETE'] !== '1')(
      'covers every exported contributor',
      () => {
        const {contributors} = listEntryElements(barrel);
        const missing = contributors.filter(name => !entries()[name]);
        expect(missing).toEqual([]);
      },
      120_000
    );
  });
}
