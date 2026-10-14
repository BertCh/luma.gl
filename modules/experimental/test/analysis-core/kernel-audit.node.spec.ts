// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, readFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {describe, expect, it} from 'vitest';
import * as geospatial from '../../src/geospatial';
import * as dataframe from '../../src/gpu-dataframe';
import * as network from '../../src/gpu-network';
import * as raster from '../../src/gpu-raster';
import * as spatialAnalysis from '../../src/gpu-spatial-analysis';
import * as terrain from '../../src/gpu-terrain';
import {KERNEL_AUDIT_CASES} from './kernel-audit/kernel-audit-cases';
import {KERNEL_AUDIT_REJECTIONS} from './kernel-audit/kernel-audit-rejections';
import {auditWGSLModule} from './kernel-audit/kernel-audit-wgsl';

// Library-wide kernel audit (quality rule 16 and the BRIEF rule 6 compile traps). Every public
// contributor of the six analysis entries is built at maximal documented options on a fake 'core'
// device; every generated WGSL module is checked for core limits (8 storage buffers per stage,
// 8192 B private and per-function memory, 16384 B workgroup memory, workgroup size), binding and
// buffer sizes, `capacity x lanes <= 2^32 - 1`, WGSL reserved words used as identifiers, NaN
// self-compares, const-expression non-finite bitcasts and calls to undeclared functions. These
// are exactly the shader-creation errors that `wgsl_reflect` accepts and Dawn rejects.

/** Analysis entry barrels, as published. */
const BARRELS: Readonly<Record<string, Record<string, unknown>>> = {
  geospatial,
  'gpu-dataframe': dataframe,
  'gpu-raster': raster,
  'gpu-terrain': terrain,
  'gpu-spatial-analysis': spatialAnalysis,
  'gpu-network': network
};

/**
 * Contributors the audit cannot build, with the reason. Keep it empty: a contributor that needs
 * real inputs gets a fake view in its case instead.
 */
const SKIPPED_CONTRIBUTORS: Readonly<Record<string, string>> = {};

/**
 * Public contributors: classes with `getCommandNodes` (rule 21) or the older `addToGraph`
 * (upstream raster operations), and `add*Recipe` functions. `GPUDataFrame` queries are not
 * graph contributors (they compile against real device buffers) and are audited by their own specs.
 */
function getPublicContributors(): Map<string, string> {
  const contributors = new Map<string, string>();
  for (const [entry, barrel] of Object.entries(BARRELS)) {
    for (const [name, value] of Object.entries(barrel)) {
      if (typeof value !== 'function') continue;
      const prototype = (value as {prototype?: object}).prototype;
      const methods = prototype ? Object.getOwnPropertyNames(prototype) : [];
      const isContributor =
        methods.includes('getCommandNodes') ||
        methods.includes('addToGraph') ||
        /^add\w+Recipe$/.test(name);
      if (isContributor && !contributors.has(name)) contributors.set(name, entry);
    }
  }
  return contributors;
}

const contributors = getPublicContributors();
const cases = Object.values(KERNEL_AUDIT_CASES).flat();

describe('kernel audit checks', () => {
  const checks = (source: string) => auditWGSLModule(source).map(finding => finding.check);

  it('rejects reserved words and keywords as identifiers', () => {
    expect(checks('fn f() { let target = 1u; }')).toEqual(['reserved word']);
    expect(checks('var<private> active: u32;')).toEqual(['reserved word']);
    expect(checks('struct S { filter: u32 }')).toEqual(['reserved word']);
    expect(checks('fn f(loop: u32) {}')).toEqual(['keyword identifier']);
    expect(checks('fn f() { switch (1u) { default: {} } }')).toEqual([]);
  });

  it('rejects NaN self-compares only when the operands are the whole comparison', () => {
    expect(checks('fn f(x: f32) -> bool { return x != x; }')).toEqual(['NaN self-compare']);
    expect(checks('fn f(v: vec2f) -> bool { return all(v.x == v.x); }')).toEqual([
      'NaN self-compare'
    ]);
    expect(checks('fn f(a: f32, x: f32) -> bool { return a + x == x; }')).toEqual([]);
    expect(checks('fn f(a: f32, b: f32) -> bool { return a == b; }')).toEqual([]);
  });

  it('rejects const-expression non-finite bitcasts and divisions by zero', () => {
    expect(checks('fn f() -> f32 { return bitcast<f32>(0x7f800000u); }')).toEqual([
      'const non-finite bitcast'
    ]);
    expect(
      checks('const NAN_BITS = 0x7fc00000u; fn f() -> f32 { return bitcast<f32>(NAN_BITS); }')
    ).toEqual(['const non-finite bitcast']);
    expect(checks('fn f() -> f32 { return 1.0 / 0.0; }')).toEqual(['const division by zero']);
    expect(checks('fn f(bits: u32) -> f32 { return bitcast<f32>(bits); }')).toEqual([]);
    expect(checks('fn f() -> f32 { return bitcast<f32>(0x3f800000u); }')).toEqual([]);
  });

  it('rejects calls to undeclared functions', () => {
    expect(checks('fn f(a: f32) -> f32 { return orientExact(a); }')).toEqual(['undeclared call']);
    expect(
      checks('fn g() -> f32 { return 1.0; } fn f() -> f32 { return max(g(), f32(1)); }')
    ).toEqual([]);
    expect(checks('fn f() -> vec2<f32> { return vec2<f32>(array<f32, 2>(1.0, 2.0)[0]); }')).toEqual(
      []
    );
  });

  it('accounts struct, override-sized and per-function memory', () => {
    expect(checks('var<private> big: array<vec4<f32>, 600>;')).toEqual(['private bytes']);
    const workgroup = `struct Pair { a: vec3<f32>, b: f32 }
override GROUP: u32 = 64u;
var<workgroup> table: array<Pair, GROUP * 32u>;`;
    expect(checks(workgroup)).toEqual(['workgroup bytes']);
    expect(auditWGSLModule(workgroup, {GROUP: 16})).toEqual([]);
    expect(checks('fn f() { var scratch: array<u32, 2049>; }')).toEqual(['function bytes']);
    expect(checks('@compute @workgroup_size(512) fn main() {}')).toEqual([
      'workgroup invocations',
      'workgroup size x'
    ]);
    const storage = Array.from(
      {length: 9},
      (_, index) => `@group(0) @binding(${index}) var<storage, read> b${index}: array<u32>;`
    ).join('\n');
    expect(checks(storage)).toEqual(['storage buffers per stage']);
  });
});

const SOURCE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../src');
const AUDITED_SOURCE_DIRECTORIES = [...Object.keys(BARRELS), 'utils', 'analysis-core'];

function listSourceFiles(directory: string): string[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? listSourceFiles(path) : entry.name.endsWith('.ts') ? [path] : [];
  });
}

describe('kernel audit sources', () => {
  it('analysis sources write no NaN self-compare (upstream idioms are filtered by text)', () => {
    // TypeScript compares with `===`, so `a == a` / `${a} != ${a}` in source text is WGSL.
    const selfCompare = /(?<![=!<>\w.\]])([A-Za-z_][\w.]*|\$\{[^}]+\})\s*(==|!=)\s*\1(?![\w(.[=])/g;
    const found: string[] = [];
    for (const directory of AUDITED_SOURCE_DIRECTORIES) {
      for (const file of listSourceFiles(join(SOURCE_ROOT, directory))) {
        readFileSync(file, 'utf8')
          .split('\n')
          .forEach((line, index) => {
            const code = line.replace(/\/\/.*$/, '');
            if (/^\s*(\*|\/\*)/.test(code)) return;
            for (const match of code.matchAll(selfCompare)) {
              found.push(`${file.slice(SOURCE_ROOT.length + 1)}:${index + 1} ${match[0]}`);
            }
          });
      }
    }
    expect(found).toEqual([]);
  });
});

describe('kernel audit coverage', () => {
  it('builds every public contributor or records why not', () => {
    const covered = new Set(cases.map(auditCase => auditCase.exportName));
    const missing = [...contributors.keys()].filter(
      name => !covered.has(name) && !(name in SKIPPED_CONTRIBUTORS)
    );
    expect(missing, 'contributors without a kernel-audit case').toEqual([]);
  });

  it('has no stale cases or skips', () => {
    const stale = [
      ...cases.map(auditCase => auditCase.exportName),
      ...Object.keys(SKIPPED_CONTRIBUTORS)
    ].filter(name => !contributors.has(name));
    expect([...new Set(stale)], 'cases or skips naming no public contributor').toEqual([]);
    const labels = cases.map(auditCase => `${auditCase.exportName} ${auditCase.name}`);
    expect(labels.length, 'duplicate case labels').toBe(new Set(labels).size);
    const staleRejections = KERNEL_AUDIT_REJECTIONS.map(
      ({exportName, name}) => `${exportName} ${name}`
    ).filter(label => !labels.includes(label));
    expect(staleRejections, 'rejections naming no case').toEqual([]);
  });
});
