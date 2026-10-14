// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUTerrainFlow,
  type GPUTerrainFlowOutput,
  type GPUTerrainFlowProps
} from '../../../src/gpu-terrain/hydrology';
import {getGPURasterRelaxationContinuationStateLength} from '../../../src/gpu-raster/cost-distance/raster-relaxation';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

const GRID = makeGPURasterMetadata({width: 6, height: 5});

/** Props overrides whose `output` members are merged over the default `flowDirections` output. */
type FlowOverrides = Partial<Omit<GPUTerrainFlowProps, 'output'>> & {
  output?: Partial<GPUTerrainFlowOutput>;
};

function createFlow(getOverrides: (graph: GPUCommandGraph) => FlowOverrides = () => ({})) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const {output, ...overrides} = getOverrides(graph);
  const flow = new GPUTerrainFlow({
    id: 'flow',
    grid: GRID,
    elevation: makeGPURasterBand(createTransientView(graph, 'elevation', 'float32', 30), GRID, {
      id: 'elevation'
    }),
    parameters: createTransientView(graph, 'settings', 'float32', 8),
    ...overrides,
    output: {flowDirections: createTransientView(graph, 'directions', 'uint32', 30), ...output}
  });
  return {graph, flow};
}

it('GPUTerrainFlow resolveFlats adds the flat resolution nodes after the direction node', () => {
  const {graph, flow} = createFlow(() => ({resolveFlats: true, maximumFlatIterations: 3}));
  const ids = flow.getCommandNodes(graph).map(node => node.id);
  const direction = ids.indexOf('flow-flow-direction');
  expect(direction).toBeGreaterThanOrEqual(0);
  expect(ids[direction + 1]).toBe('flow-flats-prepare');
  expect(ids).toContain('flow-flats-lower-relax-reset');
  expect(ids).toContain('flow-flats-higher-relax-2');
  expect(ids).toContain('flow-flats-maximum-relax-gate-0');
  expect(ids).not.toContain('flow-flats-lower-relax-3');
  expect(ids.indexOf('flow-flats-prepare-maximum')).toBeGreaterThan(
    ids.indexOf('flow-flats-higher-relax-gate-0')
  );
  expect(ids.indexOf('flow-flats-mask')).toBeLessThan(ids.indexOf('flow-flats-route'));
  expect(ids).not.toContain('flow-flats-finalize');
});

it('GPUTerrainFlow resolveFlats node count is two resets plus one node per iteration (and one seed kernel) and one gate per four iterations per pass', () => {
  const baseline = createFlow();
  const baselineCount = baseline.flow.getCommandNodes(baseline.graph).length;
  const {graph, flow} = createFlow(() => ({resolveFlats: true, maximumFlatIterations: 4}));
  const withFlats = flow.getCommandNodes(graph).length;
  // classes become a transient; two prepare kernels, one edge seed kernel, mask, route, and three
  // relaxation passes of two resets, a gate slot table write, four iterations and one gate, plus
  // the unresolved-output arm and gated cell pass (review C-09).
  expect(withFlats - baselineCount).toBe(2 + 1 + 1 + 1 + 3 * (2 + 1 + 4 + 1) + 2);
});

it('GPUTerrainFlow flatsConverged adds a finalize node and requires resolveFlats', () => {
  const plain = createFlow(() => ({resolveFlats: true, maximumFlatIterations: 2}));
  const plainCount = plain.flow.getCommandNodes(plain.graph).length;
  const flagged = createFlow(graph => ({
    resolveFlats: true,
    maximumFlatIterations: 2,
    output: {status: {flatsConverged: createTransientView(graph, 'flag', 'uint32', 1)}}
  }));
  const ids = flagged.flow.getCommandNodes(flagged.graph).map(node => node.id);
  expect(ids.length).toBe(plainCount + 1);
  // The finalize node precedes the unresolved-output arm and gated pass, which always close the graph.
  expect(ids.slice(-3)).toEqual([
    'flow-flats-finalize',
    'flow-unresolved-arm',
    'flow-unresolved-cells'
  ]);
  expect(() =>
    createFlow(graph => ({
      output: {status: {flatsConverged: createTransientView(graph, 'flag', 'uint32', 1)}}
    }))
  ).toThrow(/flatsConverged and flatsIterationCount require resolveFlats/);
});

it('GPUTerrainFlow validates maximumFlatIterations', () => {
  expect(() => createFlow(() => ({resolveFlats: true, maximumFlatIterations: 0}))).toThrow(
    /maximumFlatIterations/
  );
});

it('GPUTerrainFlow iteration outputs add finalize nodes and need their stage', () => {
  const plain = createFlow(() => ({resolveFlats: true, fillDepressions: true}));
  const plainIds = plain.flow.getCommandNodes(plain.graph).map(node => node.id);
  const counted = createFlow(graph => ({
    resolveFlats: true,
    fillDepressions: true,
    output: {
      status: {
        fillIterationCount: createTransientView(graph, 'fill-iterations', 'uint32', 1),
        flatsIterationCount: createTransientView(graph, 'flats-iterations', 'uint32', 1)
      }
    }
  }));
  const ids = counted.flow.getCommandNodes(counted.graph).map(node => node.id);
  // The fill status is written by the spill fill's publish node, which always runs; only the
  // flats add a finalize node.
  expect(ids.length).toBe(plainIds.length + 1);
  expect(ids).toContain('flow-spill-publish');
  expect(ids).toContain('flow-flats-finalize');
  expect(() =>
    createFlow(graph => ({
      output: {
        status: {fillIterationCount: createTransientView(graph, 'fill-iterations', 'uint32', 1)}
      }
    }))
  ).toThrow(/fillIterationCount requires fillDepressions/);
  expect(() =>
    createFlow(graph => ({
      output: {
        status: {flatsIterationCount: createTransientView(graph, 'flats-iterations', 'uint32', 1)}
      }
    }))
  ).toThrow(/require resolveFlats/);
});

function createContinuation(graph: GPUCommandGraph) {
  const stage = (name: string) => ({
    state: createTransientView(
      graph,
      `${name}-state`,
      'uint32',
      getGPURasterRelaxationContinuationStateLength(GRID)
    ),
    values: createTransientView(graph, `${name}-values`, 'float32', 30)
  });
  return {lower: stage('lower'), higher: stage('higher'), maximum: stage('maximum')};
}

it('flats continuation gates initialization, freezes completed stages and delays maximum until higher converges', () => {
  const {graph, flow} = createFlow(graph => ({
    resolveFlats: true,
    flatContinuation: createContinuation(graph)
  }));
  const nodes = flow.getCommandNodes(graph);
  const sources = getParsedKernelSources(nodes);
  const ids = nodes.map(node => node.id);
  expect(ids).toContain('flow-flats-lower-relax-63');
  expect(ids).not.toContain('flow-flats-lower-relax-64');
  for (const suffix of ['lower', 'higher']) {
    expect(
      nodes.find(node => node.id === `flow-flats-${suffix}-initialize`)?.condition
    ).toMatchObject({source: 'gpu', mode: 'indirect'});
    expect(sources.get(`flow-flats-${suffix}-restart-gate`)).toContain('skipStage');
  }
  expect(sources.get('flow-flats-maximum-restart-gate')).toContain(
    'prerequisite[prerequisiteOffset + 3u] == 0u'
  );
  expect(ids.indexOf('flow-flats-maximum-relax-reset')).toBeGreaterThan(
    ids.indexOf('flow-flats-higher-relax-63')
  );
  expect(nodes.find(node => node.id === 'flow-flats-prepare-maximum')?.condition).toMatchObject({
    source: 'gpu',
    mode: 'indirect'
  });
});

it('flats continuation keeps the pipeline count independent of rounds and validates persistent views', () => {
  const counts = [4, 5].map(maximumFlatIterations => {
    const {graph, flow} = createFlow(graph => ({
      resolveFlats: true,
      maximumFlatIterations,
      flatContinuation: createContinuation(graph)
    }));
    return getGPUCommandNodePipelineKeys(flow.getCommandNodes(graph)).size;
  });
  expect(counts[0]).toBe(counts[1]);
  expect(() => createFlow(graph => ({flatContinuation: createContinuation(graph)}))).toThrow(
    /requires resolveFlats/
  );
  expect(() =>
    createFlow(graph => {
      const continuation = createContinuation(graph);
      continuation.maximum.values = continuation.higher.values;
      return {resolveFlats: true, flatContinuation: continuation};
    })
  ).toThrow(/distinct/);
});
