// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUTerrainFlow, type GPUTerrainFlowProps} from '../../../src/gpu-terrain/hydrology';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createFlow(
  getOverrides: (graph: GPUCommandGraph) => Partial<GPUTerrainFlowProps> = () => ({})
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const flow = new GPUTerrainFlow({
    id: 'flow',
    width: 6,
    height: 5,
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {kind: 'buffer', values: createTransientView(graph, 'elevation', 'float32', 30)}
    },
    settings: createTransientView(graph, 'settings', 'float32', 8),
    flowDirections: createTransientView(graph, 'directions', 'uint32', 30),
    ...getOverrides(graph)
  });
  return {graph, flow};
}

it('GPUTerrainFlow resolveFlats adds the flat resolution nodes after the direction node', () => {
  const {graph, flow} = createFlow(() => ({resolveFlats: true, maxFlatIterations: 3}));
  const ids = flow.getCommandNodes(graph).map(node => node.id);
  const direction = ids.indexOf('flow-flow-direction');
  expect(direction).toBeGreaterThanOrEqual(0);
  expect(ids[direction + 1]).toBe('flow-flats-prepare');
  expect(ids).toContain('flow-flats-lower-relax-reset');
  expect(ids).toContain('flow-flats-higher-relax-2');
  expect(ids).toContain('flow-flats-maximum-relax-gate-2');
  expect(ids).not.toContain('flow-flats-lower-relax-3');
  expect(ids.indexOf('flow-flats-prepare-maximum')).toBeGreaterThan(
    ids.indexOf('flow-flats-higher-relax-gate-2')
  );
  expect(ids.indexOf('flow-flats-mask')).toBeLessThan(ids.indexOf('flow-flats-route'));
  expect(ids).not.toContain('flow-flats-finalize');
});

it('GPUTerrainFlow resolveFlats node count is two resets plus two nodes per iteration per pass', () => {
  const baseline = createFlow();
  const baselineCount = baseline.flow.getCommandNodes(baseline.graph).length;
  const {graph, flow} = createFlow(() => ({resolveFlats: true, maxFlatIterations: 4}));
  const withFlats = flow.getCommandNodes(graph).length;
  // classes become a transient; two prepare kernels, mask, route, and three relaxation passes.
  expect(withFlats - baselineCount).toBe(2 + 1 + 1 + 3 * (2 + 2 * 4));
});

it('GPUTerrainFlow flatsConverged adds a finalize node and requires resolveFlats', () => {
  const plain = createFlow(() => ({resolveFlats: true, maxFlatIterations: 2}));
  const plainCount = plain.flow.getCommandNodes(plain.graph).length;
  const flagged = createFlow(graph => ({
    resolveFlats: true,
    maxFlatIterations: 2,
    flatsConverged: createTransientView(graph, 'flag', 'uint32', 1)
  }));
  const ids = flagged.flow.getCommandNodes(flagged.graph).map(node => node.id);
  expect(ids.length).toBe(plainCount + 1);
  expect(ids[ids.length - 1]).toBe('flow-flats-finalize');
  expect(() =>
    createFlow(graph => ({flatsConverged: createTransientView(graph, 'flag', 'uint32', 1)}))
  ).toThrow(/flatsConverged requires resolveFlats/);
});

it('GPUTerrainFlow validates maxFlatIterations', () => {
  expect(() => createFlow(() => ({resolveFlats: true, maxFlatIterations: 0}))).toThrow(
    /maxFlatIterations/
  );
});
