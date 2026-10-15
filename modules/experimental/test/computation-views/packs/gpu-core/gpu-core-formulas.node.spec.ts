// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import {describe, expect, test} from 'vitest';
import {GPU_CORE_OPERATORS} from '../../../../src/computation-views/packs/gpu-core/operators';
import {
  createGPUComputationRecord,
  validateGPUComputationRecord,
  type GPUComputationFormulaRef
} from '@luma.gl/gpgpu';
import {
  GPU_CORE_FORMULAS,
  GPU_CORE_FORMULA_REFS,
  mergeGPUCoreFormulaRefs
} from '../../../../src/computation-views/packs/gpu-core/formulas';
import {
  FORMULA_TEX_MACROS,
  isTrustedFormulaCommand,
  listTexTerms
} from '../../../../src/computation-views/views/formula-tex';

const ELEMENTS = [
  ['GPUScan', 'scan', 'global', 'scan-sort', ['gpu-scan']],
  ['GPUFlagOffsets', 'flag-offsets', 'global', 'scan-sort', ['gpu-flag-offsets', 'gpu-scan']],
  ['GPUCompaction', 'compaction', 'global', 'select', ['gpu-compaction', 'gpu-scan']],
  ['GPUMask', 'mask', 'local', 'none', ['gpu-mask']],
  ['GPUReduction', 'reduction', 'global', 'sum', ['gpu-reduction']],
  ['GPUSegmentedReduction', 'segmented-reduction', 'zonal', 'sum', ['gpu-segmented-reduction']],
  ['GPUSort', 'sort', 'global', 'scan-sort', ['gpu-sort']],
  ['GPUSegmentedSort', 'segmented-sort', 'zonal', 'scan-sort', ['gpu-segmented-sort']],
  ['GPUGather', 'gather', 'local', 'select', ['gpu-gather']],
  ['GPUHistogram', 'histogram', 'global', 'histogram', ['gpu-histogram']],
  ['GPUElementwise', 'elementwise', 'local', 'none', ['gpu-elementwise']],
  [
    'GPUGridIndex',
    'grid-index',
    'zonal',
    'scan-sort',
    ['gpu-grid-index', 'gpu-grid-index-internals']
  ],
  ['GPUGridBinning', 'grid-binning', 'zonal', 'count', ['gpu-grid-binning', 'gpu-spatial-utils']],
  [
    'GPUGridAggregation',
    'grid-aggregation',
    'zonal',
    'sum',
    ['gpu-grid-aggregation', 'gpu-spatial-utils']
  ],
  ['GPUGroupAggregation', 'group-aggregation', 'zonal', 'sum', ['gpu-group-aggregation']],
  ['GPUHashIndex', 'hash-index', 'global', 'none', ['gpu-hash-index']],
  ['GPUHashJoin', 'hash-join', 'pairwise', 'select', ['gpu-hash-join']],
  ['GPUMatVec', 'mat-vec', 'zonal', 'sum', ['gpu-matvec']],
  ['GPUMatMul', 'mat-mul', 'block', 'sum', ['gpu-matmul']],
  ['GPUConvolution', 'convolution', 'focal', 'sum', ['gpu-convolution']],
  ['GPUFFT1D', 'fft1d', 'global', 'sum', ['gpu-fft1d', 'gpu-fft-utils']],
  ['GPUFFT2D', 'fft2d', 'global', 'sum', ['gpu-fft2d', 'gpu-fft2d-shaders', 'gpu-fft-utils']],
  [
    'createGPUConjugateGradientProgram',
    'conjugate-gradient',
    'solver',
    'solve',
    [
      'gpu-conjugate-gradient-program',
      'gpu-semantic-vector-operation',
      'gpu-semantic-scalar-operation',
      'gpu-elementwise-scalar',
      'gpu-scalar-operation',
      'gpu-scalar'
    ]
  ]
] as const;
const PROCEDURAL = new Set([
  'GPUSort',
  'GPUSegmentedSort',
  'GPUGridIndex',
  'GPUHashIndex',
  'GPUHashJoin'
]);
const requireFromRoot = createRequire(resolve('package.json'));
let katex: {renderToString: (tex: string, options: object) => string} | undefined;
try {
  katex = requireFromRoot('katex');
} catch {
  /* Optional root dependency. */
}

describe('GPU Core curated formula coverage', () => {
  test('covers exactly the element table with unique base scenario ids', () => {
    expect(Object.keys(GPU_CORE_OPERATORS).sort()).toEqual(ELEMENTS.map(row => row[0]).sort());
    expect(Object.keys(GPU_CORE_FORMULAS).sort()).toEqual(ELEMENTS.map(row => row[0]).sort());
    const identifiers = Object.values(GPU_CORE_FORMULAS).flatMap(entry =>
      'id' in entry ? [entry.id] : []
    );
    expect(new Set(identifiers).size).toBe(identifiers.length);
  });
  for (const [element, scenario, scope, reduction, files] of ELEMENTS) {
    test(`${element}: scope, formula, and real shader identifiers`, () => {
      expect(GPU_CORE_OPERATORS[element]).toEqual({scope, reduction});
      const entry = GPU_CORE_FORMULAS[element];
      expect(entry).toBeDefined();
      if ('formula' in entry) {
        expect(PROCEDURAL.has(element)).toBe(true);
        expect(entry.formula).toBe('none');
        expect(entry.reason.split(/\s+/).length).toBeLessThanOrEqual(20);
        expect(entry.reason).toMatch(/procedural/i);
        expect(entry.reason).toMatch(/SkeletonView\/graph shows the steps/);
        return;
      }
      expect(PROCEDURAL.has(element)).toBe(false);
      expect(entry.id).toBe(`gpu-core.${scenario}`);
      expect(entry.derivation).toBe('curated');
      expect(entry.tex.trim()).not.toBe('');
      expect(entry.ascii.trim()).not.toBe('');
      expect(entry.symbols.length).toBeGreaterThan(0);
      const source = files
        .map(file => readFileSync(resolve(`modules/gpgpu/src/gpu-core/${file}.ts`), 'utf8'))
        .join('\n');
      const identifiers = new Set(source.match(/[A-Za-z_][A-Za-z_0-9]*/g));
      for (const symbol of entry.symbols) {
        expect(symbol.role).toMatch(/^(input|output|parameter|index|intermediate|constant)$/);
        expect(symbol.binding).toMatch(/^wgsl:[A-Za-z_][A-Za-z_0-9]*$/);
        expect(identifiers.has(symbol.binding!.slice(5)), `${element}/${symbol.binding}`).toBe(
          true
        );
      }
      for (const term of listTexTerms(entry.tex)) {
        expect(
          entry.symbols.some(symbol => symbol.id === term.symbolId),
          term.symbolId
        ).toBe(true);
      }
    });
  }
  test.skipIf(!katex)('all curated TeX renders with the shared KaTeX conventions', () => {
    for (const entry of Object.values(GPU_CORE_FORMULAS)) {
      if ('formula' in entry) continue;
      expect(() =>
        katex!.renderToString(entry.tex, {
          throwOnError: true,
          displayMode: true,
          macros: FORMULA_TEX_MACROS,
          trust: isTrustedFormulaCommand
        })
      ).not.toThrow();
    }
  });
});

describe('composed GPU Core formula refs', () => {
  test('records of other entries that compose a core primitive validate after merging', () => {
    const graph = {
      id: 'network-compiled',
      form: 'compiled' as const,
      domain: 'gpu-network',
      label: 'network',
      nodes: [
        {
          id: 'histogram-count',
          label: 'histogram count',
          namespace: '',
          kind: 'compute',
          formula: 'gpu-core.histogram',
          reads: [],
          writes: [],
          attrs: {}
        }
      ],
      edges: []
    };
    const recordWith = (formulas: GPUComputationFormulaRef[]) =>
      createGPUComputationRecord({
        id: 'gpu-network.composed',
        runId: 'composed-run',
        createdAt: '2026-10-14T00:00:00.000Z',
        adapter: {backend: 'webgpu'},
        domains: ['gpu-network'],
        graphs: [graph],
        formulas
      });
    expect(validateGPUComputationRecord(recordWith([])).valid).toBe(false);
    const merged = mergeGPUCoreFormulaRefs(['gpu-core.histogram', 'gpu-network.unknown']);
    expect(merged.map(formula => formula.id)).toEqual(['gpu-core.histogram']);
    expect(merged[0]).toBe(GPU_CORE_FORMULA_REFS['gpu-core.histogram']);
    expect(
      validateGPUComputationRecord(recordWith(merged)).issues.filter(
        issue => issue.level === 'error'
      )
    ).toEqual([]);
    // A record's own ref wins over the core table.
    const own = {...merged[0], ascii: 'local'};
    expect(mergeGPUCoreFormulaRefs(['gpu-core.histogram'], [own])).toEqual([own]);
  });
});
