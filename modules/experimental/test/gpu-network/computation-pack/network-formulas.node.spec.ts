// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {NETWORK_FORMULA_REFS} from '../../../src/gpu-network/computation-pack/network-formulas';
import {describeStructure} from '../../../src/computation-lenses/scenarios/gpu-network/network-reachability.lens';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

test('reachability formula symbols bind to identifiers in its recorded generated WGSL', async () => {
  const device = createNullWebGPUDevice();
  try {
    const record = await describeStructure(device);
    const sources = record.refs.sources.map(source => source.text ?? '');
    expect(sources.length).toBeGreaterThan(0);
    for (const formula of Object.values(NETWORK_FORMULA_REFS).filter(reference =>
      reference.id.startsWith('gpu-network.reachability.')
    )) {
      expect(record.refs.formulas.some(reference => reference.id === formula.id)).toBe(true);
      for (const symbol of formula.symbols) {
        expect(symbol.binding).toMatch(/^wgsl:/);
        const identifier = symbol.binding!.slice(5);
        expect(
          sources.some(source => new RegExp(`\\b${identifier}\\b`).test(source)),
          `${formula.id}: ${identifier}`
        ).toBe(true);
      }
    }
  } finally {
    device.destroy();
  }
});
