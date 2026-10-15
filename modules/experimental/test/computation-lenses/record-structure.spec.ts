// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {fromRecordedLensFile, listLensScenarios} from '../../src/computation-lenses/index';

import '../../src/computation-lenses/scenarios/index';

// Same comparison as record-structure.node.spec.ts, for scenarios whose structure can only be
// described on a real device (their contributors create compute pipelines while building).
// `import.meta.glob` is a Vite transform (it must stay a literal call); the package tsconfig does
// not load vite/client types.
declare global {
  interface ImportMeta {
    glob(pattern: string, options: {query: string; import: string}): Record<string, unknown>;
  }
}
const recordFiles = import.meta.glob('../../../../docs/lens-records/*.json', {
  query: '?raw',
  import: 'default'
}) as Record<string, () => Promise<string>>;

for (const scenario of listLensScenarios()) {
  const loadRecordFile = recordFiles[`../../../../docs/lens-records/${scenario.id}.json`];
  if (!loadRecordFile) continue;
  test(`${scenario.id}: recorded graph and node ids match structure on a GPU`, async context => {
    const loaded = await scenario.load();
    if (!loaded.describeStructure) {
      context.skip('scenario has no describeStructure');
      return;
    }
    const device = await getWebGPUTestDevice();
    if (!device) {
      context.skip('requires WebGPU');
      return;
    }
    const structure = await loaded.describeStructure(device);
    const recorded = fromRecordedLensFile(JSON.parse(await loadRecordFile()));
    expect(structure.graphs.map(graph => graph.id)).toEqual(
      recorded.record.graphs.map(graph => graph.id)
    );
    for (const graph of structure.graphs)
      expect(graph.nodes.map(node => node.id)).toEqual(
        recorded.record.graphs
          .find(candidate => candidate.id === graph.id)
          ?.nodes.map(node => node.id)
      );
  });
}
