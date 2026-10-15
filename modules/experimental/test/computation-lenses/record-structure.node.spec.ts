// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import {fromRecordedLensFile, listLensScenarios} from '../../src/computation-lenses/index';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

import '../../src/computation-lenses/scenarios/index';
const scenarios = listLensScenarios();
test('lens scenario registry is available without a GPU', () => {
  expect(Array.isArray(scenarios)).toBe(true);
});

for (const scenario of scenarios) {
  const recordPath = new URL(`../../../../docs/lens-records/${scenario.id}.json`, import.meta.url);
  if (!existsSync(recordPath)) {
    test.skip(`${scenario.id}: no recorded file`, () => {});
    continue;
  }
  test(`${scenario.id}: recorded graph and node ids match structure`, async context => {
    const loaded = await scenario.load();
    if (!loaded.describeStructure) {
      context.skip('scenario has no describeStructure');
      return;
    }
    const device = createNullWebGPUDevice();
    try {
      let structure: Awaited<ReturnType<typeof loaded.describeStructure>>;
      try {
        structure = await loaded.describeStructure(device);
      } catch (error) {
        // Some contributors create compute pipelines while building their graph; the headless
        // record-structure.spec.ts runs the same comparison on a real device.
        if (String(error).includes('not supported on NullDevice')) {
          context.skip('describeStructure needs a GPU device (checked headless)');
          return;
        }
        throw error;
      }
      const recorded = fromRecordedLensFile(JSON.parse(readFileSync(recordPath, 'utf8')));
      expect(structure.graphs.map(graph => graph.id)).toEqual(
        recorded.record.graphs.map(graph => graph.id)
      );
      for (const graph of structure.graphs)
        expect(graph.nodes.map(node => node.id)).toEqual(
          recorded.record.graphs
            .find(candidate => candidate.id === graph.id)
            ?.nodes.map(node => node.id)
        );
    } finally {
      device.destroy();
    }
  });
}
