// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {describe, expect, it} from 'vitest';
// Share the showcase CLI's word list and checks so the two gates cannot drift.
import {
  findNonFiniteBitcasts,
  findReservedIdentifiers,
  getSourceRoots,
  scanDirectories,
  scanSource
} from '../../../../examples/deck/spatial-analysis-showcase/scripts/check-wgsl-identifiers.mjs';

function makeTemplate(source: string): string {
  return `const shader = /* wgsl */ \`${source}\`;`;
}

describe('WGSL source safety', () => {
  it('scans every module source directory and the showcase without a GPU', () => {
    const roots = getSourceRoots();
    expect(roots.some(root => root.endsWith('/modules/gpgpu/src'))).toBe(true);
    expect(roots.some(root => root.endsWith('/modules/experimental/src'))).toBe(true);
    expect(roots.some(root => root.endsWith('/spatial-analysis-showcase/src'))).toBe(true);
    const {hits, scannedTemplates} = scanDirectories(roots);
    expect(scannedTemplates).toBeGreaterThan(0);
    expect(hits).toEqual([]);
  }, 30000);

  it('rejects the original geodesic target parameter regression', () => {
    const source = readFileSync(
      new URL('../../src/gpu-spatial-analysis/geometry-measures/geodesic-wgsl.ts', import.meta.url),
      'utf8'
    );
    expect(source).toContain('fn geodesicLongitudeDelta(source: f32, destination: f32)');
    const regression = source.replace(
      'fn geodesicLongitudeDelta(source: f32, destination: f32)',
      'fn geodesicLongitudeDelta(source: f32, target: f32)'
    );
    expect(scanSource(regression).hits).toEqual([
      expect.objectContaining({name: 'target', kind: 'parameter'})
    ]);
  });

  it.each([
    ['let target = 0u;', 'target', 'declaration'],
    ['var<private> filter: f32;', 'filter', 'declaration'],
    ['var <workgroup> shared: array<u32, 4>;', 'shared', 'declaration'],
    ['const common: f32 = 1.0;', 'common', 'declaration'],
    ['fn target() {}', 'target', 'function name'],
    ['fn helper(\n source: f32,\n target: f32\n) {}', 'target', 'parameter'],
    ['fn main(@builtin(global_invocation_id) target: vec3<u32>) {}', 'target', 'parameter'],
    ['fn helper(values: array<vec2<f32>, 4>, from: f32) {}', 'from', 'parameter'],
    ['struct Values\n{\n @align(16) target: f32,\n}', 'target', 'struct member'],
    ['struct Values { value: f32, filter: f32, }', 'filter', 'struct member'],
    ['let return = 1u;', 'return', 'declaration']
  ])('rejects %s', (source, name, kind) => {
    expect(scanSource(makeTemplate(source)).hits).toEqual([expect.objectContaining({name, kind})]);
  });

  it('preserves lines across interpolations and visits nested templates', () => {
    const source =
      'const shader = `fn valid() {\n${\n condition ? `let target: f32 = 0.0;` : ""\n}\nlet filter: f32 = 0.0;\n}`;';
    expect(scanSource(source).hits).toEqual([
      expect.objectContaining({name: 'filter', line: 5}),
      expect.objectContaining({name: 'target', line: 3})
    ]);
    expect(scanSource('fn helper(target: f32) {}', 'shader.wgsl').hits).toHaveLength(1);
  });

  it('ignores comments, interpolation expressions, JavaScript examples and valid identifiers', () => {
    expect(
      findReservedIdentifiers(`
      /* fn filter(target: f32) { /* nested */ let common = 0u; } */
      // struct Values { target: f32 }
      fn helper(destination: f32) { let targetValue = destination; }
      struct Values { @interpolate(flat) value: u32, }
    `)
    ).toEqual([]);
    expect(scanSource('const example = `const filter = new GPUCrossfilter();`;').hits).toEqual([]);
    expect(
      scanSource(
        'const shader = `fn helper() { ${(() => { const target = 1; return target; })()} }`;'
      ).hits
    ).toEqual([]);
  });

  it.each([
    'bitcast<f32>(0x7f800000u)',
    'bitcast<f32>(0xff800000u)',
    'bitcast<f32>(0x7fc00000u)',
    'bitcast<vec2<f32>>(vec2<u32>(0u, 0xffc00000u))',
    'const bits: u32 = 2139095040u; const aliasBits: u32 = bits; bitcast<f32>(aliasBits)',
    'bitcast<f32>(${positive ? "0x7f800000u" : "0xff800000u"})'
  ])('rejects constant non-finite bitcasts: %s', source => {
    expect(scanSource(makeTemplate(source)).hits).toEqual([
      expect.objectContaining({name: 'non-finite bitcast'})
    ]);
  });

  it.each([
    'let bits = 0x7f800000u; bitcast<f32>(bits)',
    'var bits = 0x7fc00000u; bitcast<f32>(bits)',
    'bitcast<f32>(input[index])',
    'bitcast<f32>(0x7f7fffffu)',
    '// bitcast<f32>(0x7f800000u)'
  ])('allows runtime operands and finite constants: %s', source => {
    expect(findNonFiniteBitcasts(source)).toEqual([]);
  });
});
