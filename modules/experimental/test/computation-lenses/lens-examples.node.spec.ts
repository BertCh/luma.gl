// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {describe, expect, test} from 'vitest';
import {
  extractLensExample,
  findNonPublicImports,
  getLensExamplePath,
  LENS_EXAMPLE_MAX_LINES,
  listLensScenarios
} from '../../src/computation-lenses';

const region = (code: string) => `// #region example\n${code}\n// #endregion example`;

describe('lens example mechanism', () => {
  test('extracts and dedents only the shown block, including CRLF', () => {
    expect(
      extractLensExample(
        `import {Device} from '@luma.gl/core';\r\n${region('    const result = await run();\n      return result;').replace(/\n/g, '\r\n')}`
      )
    ).toEqual({
      code: 'const result = await run();\n  return result;',
      language: 'ts',
      lineCount: 2
    });
    expect(
      extractLensExample(region(Array.from({length: 25}, () => 'run();').join('\n'))).lineCount
    ).toBe(LENS_EXAMPLE_MAX_LINES);
  });
  test('rejects missing, duplicated, empty, reversed and overlong regions', () => {
    expect(() => extractLensExample('const value = 1;')).toThrow(/missing/i);
    expect(() => extractLensExample(`${region('run();')}\n${region('run();')}`)).toThrow(
      /duplicated/i
    );
    expect(() =>
      extractLensExample(region(Array.from({length: 26}, () => 'run();').join('\n')))
    ).toThrow(/25 lines/);
    expect(() => extractLensExample(region(''))).toThrow(/empty/i);
    expect(() => extractLensExample(region(' \n  '))).toThrow(/empty/i);
    expect(() => extractLensExample(region('\nrun();'))).toThrow(/blank first or last/);
    expect(() => extractLensExample(region('run();\n'))).toThrow(/blank first or last/);
    expect(() => extractLensExample('// #endregion example\n// #region example')).toThrow(
      /precedes/
    );
  });
  test('accepts only public entry imports, including dynamic imports and re-exports', () => {
    expect(findNonPublicImports('import(`./private`);')).toEqual(['./private']);
    expect(
      findNonPublicImports(`import type {Device} from '@luma.gl/core';
      import {Model} from '@luma.gl/engine';
      import '@luma.gl/gpgpu';
      import {GPUData} from '@luma.gl/gpgpu/gpu-data';
      export {Result} from '@luma.gl/experimental/computation-views';
      const module = import('@luma.gl/experimental/gpu-raster');`)
    ).toEqual([]);
    expect(
      findNonPublicImports(`import './local';
      import type {Hidden} from '@luma.gl/experimental/src/private';
      export * from '@luma.gl/gpgpu/dist/index';
      import('@luma.gl/gpgpu/test/fixture');
      import {readFile} from 'node:fs';
      import '@luma.gl/experimental';
      import '/absolute';
      const local = require('../private');`)
    ).toEqual([
      './local',
      '@luma.gl/experimental/src/private',
      '@luma.gl/gpgpu/dist/index',
      '@luma.gl/gpgpu/test/fixture',
      'node:fs',
      '@luma.gl/experimental',
      '/absolute',
      '../private'
    ]);
    expect(
      findNonPublicImports(
        `// import './comment';\n/* import './block'; */\nconst text = "import './string';";`
      )
    ).toEqual([]);
  });
  test('derives sibling paths from entry and element, ignoring variant', () => {
    expect(getLensExamplePath('gpu-spatial-analysis.local-moran.variant')).toBe(
      'scenarios/gpu-spatial-analysis/local-moran.example.ts'
    );
    expect(() => getLensExamplePath('../private')).toThrow('Invalid scenario id');
  });
  test('every registered scenario has a checked public-API example', async () => {
    const root = new URL('../../src/computation-lenses/', import.meta.url);
    const index = new URL('scenarios/index.ts', root);
    if (existsSync(index)) await import(index.href);
    const scenarios = listLensScenarios();
    expect(Array.isArray(scenarios)).toBe(true);
    for (const scenario of scenarios) {
      const path = new URL(scenario.example?.path ?? getLensExamplePath(scenario.id), root);
      expect(existsSync(path), `${scenario.id}: ${path.pathname}`).toBe(true);
      const source = readFileSync(path, 'utf8');
      expect(extractLensExample(source).lineCount, scenario.id).toBeLessThanOrEqual(
        LENS_EXAMPLE_MAX_LINES
      );
      expect(findNonPublicImports(source), scenario.id).toEqual([]);
      expect(source, scenario.id).toContain('export async function runExample');
    }
  });
});
