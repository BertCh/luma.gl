// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {STATISTICS_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/statistics';
import {expectFormulaBindingsExist, expectSpatialEntryWellFormed} from './formula-bindings';
import {STATISTICS_INFERENCE_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/statistics-inference';
import {STATISTICS_SUMMARY_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/statistics-summaries';
import {STATISTICS_REGION_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/statistics-regions';
import {STATISTICS_CLUSTERING_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/statistics-clustering';

const kernelSources: Record<string, string[]> = {
  GPULocalMoran: [
    'modules/experimental/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-moran.ts',
    'modules/experimental/src/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-kernels.ts'
  ]
};

// Sub-family tables are checked by their own `statistics-*-pack.node.spec.ts`; append one spread per table.
const SUB_FAMILY_ELEMENTS = new Set<string>([
  ...Object.keys(STATISTICS_INFERENCE_OPERATORS),
  ...Object.keys(STATISTICS_SUMMARY_OPERATORS),
  ...Object.keys(STATISTICS_REGION_OPERATORS),
  ...Object.keys(STATISTICS_CLUSTERING_OPERATORS)
]);

for (const [name, entry] of Object.entries(STATISTICS_OPERATORS)) {
  if (SUB_FAMILY_ELEMENTS.has(name)) continue;
  test(`${name}: statistics pack and curated kernel bindings`, () => {
    expectSpatialEntryWellFormed(`statistics/${name}`, entry);
    if ('formula' in entry.formula) return;
    expect(kernelSources[name], `Add ${name}'s WGSL generator sources`).toBeDefined();
    expectFormulaBindingsExist(entry.formula, kernelSources[name]);
  });
}

test('formula binding checks reject stale identifiers and unmatched TeX terms', () => {
  const formula = STATISTICS_OPERATORS['GPULocalMoran'].formula;
  if ('formula' in formula) throw new Error('Local Moran must have a curated formula');
  expect(() =>
    expectFormulaBindingsExist(
      {
        ...formula,
        symbols: formula.symbols.map((symbol, index) =>
          index === 0 ? {...symbol, binding: 'wgsl:missingKernelIdentifier'} : symbol
        )
      },
      kernelSources['GPULocalMoran']
    )
  ).toThrow();
  expect(() =>
    expectFormulaBindingsExist(
      {...formula, tex: `${formula.tex} \\term{missing}{x}`},
      kernelSources['GPULocalMoran']
    )
  ).toThrow();
});
