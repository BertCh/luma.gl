// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, readFileSync, statSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';
import {PHILOX_PURPOSES} from '../../src/utils/philox-random';

// B-22: one registry of Philox purpose IDs, so one user seed never gives two features one stream.

const SOURCE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../src');

function listSourceFiles(directory: string): string[] {
  return readdirSync(directory).flatMap(name => {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) {
      return listSourceFiles(path);
    }
    return path.endsWith('.ts') ? [path] : [];
  });
}

it('PHILOX_PURPOSES holds unique nonzero uint32 IDs', () => {
  const ids = Object.values(PHILOX_PURPOSES).map(purpose => purpose.id);
  for (const id of ids) {
    expect(Number.isInteger(id) && id > 0 && id <= 0xffffffff, `${id}`).toBe(true);
  }
  expect(new Set(ids).size).toBe(ids.length);
});

it('Philox callers take their purpose from the registry', () => {
  // A purpose literal next to a Philox call is a second registry. Constants must be defined as
  // `PHILOX_PURPOSES.<name>.id`, and WGSL keys must not end in a literal purpose word.
  const literalConstant =
    /^\s*(export )?const \w*(PURPOSE|_STREAM|_TAG|_KEY)\s*=\s*(0x[0-9a-f]+|\d+)\s*;/im;
  const literalKey = /vec2<u32>\([^()]*,\s*(0x[0-9a-f]+|\d+)u\s*\)/i;
  const offenders: string[] = [];
  for (const file of listSourceFiles(SOURCE_ROOT)) {
    const source = readFileSync(file, 'utf8');
    if (!/philox/i.test(source) || file.endsWith('philox-random.ts')) {
      continue;
    }
    for (const [lineIndex, line] of source.split('\n').entries()) {
      if (literalConstant.test(line) || (/philox4x32/.test(line) && literalKey.test(line))) {
        offenders.push(`${relative(SOURCE_ROOT, file)}:${lineIndex + 1}`);
      }
    }
  }
  expect(offenders).toEqual([]);
});
