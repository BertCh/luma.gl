// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {computeRunId} from '../../../src/computation-views/hosts/run-id';

const props = {
  structureIds: ['scan/up', 'scan/down'],
  datasetVersion: '2026-10',
  seed: 17,
  adapter: 'Test adapter',
  variant: 'subgroups'
};

describe('computeRunId', () => {
  test('matches an independent BigInt FNV-1a oracle (runtime implementation uses no BigInt)', () => {
    for (const input of [
      props,
      {...props, adapter: '東京 🌎'},
      {...props, structureIds: [''], seed: 0}
    ]) {
      const canonical = JSON.stringify([
        input.structureIds,
        input.datasetVersion,
        input.seed,
        input.adapter,
        input.variant
      ]);
      let hash = 0xcbf29ce484222325n;
      for (const byte of new TextEncoder().encode(canonical))
        hash = BigInt.asUintN(64, (hash ^ BigInt(byte)) * 0x100000001b3n);
      expect(computeRunId(input)).toBe(`run-${hash.toString(16).padStart(16, '0')}`);
    }
  });
  test('is deterministic, ordered, typed and sensitive to every identity field', () => {
    const original = computeRunId(props);
    expect(original).toMatch(/^run-[0-9a-f]{16}$/);
    expect(computeRunId({...props})).toBe(original);
    for (const changed of [
      {...props, structureIds: [...props.structureIds].reverse()},
      {...props, datasetVersion: 'next'},
      {...props, seed: '17'},
      {...props, seed: undefined},
      {...props, adapter: 'other'},
      {...props, variant: undefined}
    ])
      expect(computeRunId(changed)).not.toBe(original);
    expect(computeRunId({...props, structureIds: ['a,b', 'c']})).not.toBe(
      computeRunId({...props, structureIds: ['a', 'b,c']})
    );
  });
});
