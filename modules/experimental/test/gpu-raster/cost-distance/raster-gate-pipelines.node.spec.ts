// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUCostDistance} from '../../../src/gpu-raster/cost-distance';
import {GPUTerrainFlow, type GPUTerrainFlowProps} from '../../../src/gpu-terrain/hydrology';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {countRasterPipelines, getRasterPipelines} from '../raster-pipeline-count';

const WIDTH = 40;
const HEIGHT = 36;
const CELLS = WIDTH * HEIGHT;
const GRID = makeGPURasterMetadata({width: WIDTH, height: HEIGHT});

function buildFlowNodes(iterations: number, overrides: Partial<GPUTerrainFlowProps> = {}) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length = CELLS
  ) => createTransientView(graph, name, format, length);
  return new GPUTerrainFlow({
    grid: GRID,
    elevation: makeGPURasterBand(view('elevation', 'float32'), GRID, {id: 'elevation'}),
    parameters: view('settings', 'float32', 8),
    fillDepressions: true,
    resolveFlats: true,
    maximumFillIterations: iterations,
    maximumFlatIterations: iterations,
    maximumAccumulationIterations: iterations,
    output: {
      flowDirections: view('directions', 'uint32'),
      accumulation: view('accumulation', 'float32'),
      streams: view('streams', 'uint32')
    },
    ...overrides
  }).getCommandNodes(graph);
}

/** Distinct (loop, pipeline) pairs: round and gate suffixes `-<n>` are dropped from the node id. */
function countLoopPipelines(nodes: readonly unknown[]): number {
  return new Set(
    getRasterPipelines(nodes).map(
      pipeline => `${pipeline.id.replace(/-\d+$/, '')}\n${pipeline.key}`
    )
  ).size;
}

function buildCostDistanceNodes(iterations: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  return new GPUCostDistance({
    grid: GRID,
    friction: makeGPURasterBand(createTransientView(graph, 'friction', 'float32', CELLS), GRID, {
      id: 'friction'
    }),
    parameters: createTransientView(graph, 'settings', 'float32', 8),
    sources: createTransientView(graph, 'sources', 'uint32', 3),
    output: {
      costs: createTransientView(graph, 'costs', 'float32', CELLS),
      backLinks: createTransientView(graph, 'links', 'uint32', CELLS)
    },
    maximumIterations: iterations,
    sweepAfterIteration: 2
  }).getCommandNodes(graph);
}

it('review E-03: gated raster loops compile one pipeline per kernel variant, whatever the round count', () => {
  // Rule 12: the gate-group position of a round comes from a slot word, so N and N + 1 rounds
  // (which change the last group's size and every round's slot) compile the same pipelines.
  for (const routing of ['d8', 'd-infinity', 'mfd-quinn'] as const) {
    // Counted per loop (node id without its round suffix): a gate bakes its loop's round limit, so
    // two different loops may share a gate source by coincidence at some limits.
    const counts = [11, 12, 13, 14, 15, 16, 17].map(iterations =>
      countLoopPipelines(buildFlowNodes(iterations, {flowRouting: routing}))
    );
    expect(new Set(counts).size, routing).toBe(1);
  }
  const costCounts = [9, 10, 11, 12, 13].map(iterations =>
    countRasterPipelines(buildCostDistanceNodes(iterations))
  );
  expect(new Set(costCounts).size).toBe(1);
});

it('review E-03: each tiled relaxation compiles its five pass variants once', () => {
  const pipelines = getRasterPipelines(buildFlowNodes(32));
  const relaxationKeys = new Set(
    pipelines.filter(pipeline => /-relax-\d+$/.test(pipeline.id)).map(pipeline => pipeline.key)
  );
  // Fill, and the lower, higher and maximum flat passes: the fill uses the tiled pass only below
  // its sweep threshold, so count per relaxation instead of in total.
  for (const prefix of ['flats-lower', 'flats-higher', 'flats-maximum']) {
    const keys = new Set(
      pipelines
        .filter(pipeline => new RegExp(`${prefix}-relax-\\d+$`).test(pipeline.id))
        .map(pipeline => pipeline.key)
    );
    expect(keys.size, prefix).toBeLessThanOrEqual(5);
  }
  // Before E-03 each pass variant compiled once per gate slot (four slots): up to 20 per relaxation.
  expect(relaxationKeys.size).toBeLessThanOrEqual(4 * 5);
  const roundPipelines = pipelines.filter(pipeline => /-round-\d+$/.test(pipeline.id));
  expect(roundPipelines.length).toBeGreaterThan(0);
  // Every round of one loop shares one pipeline.
  for (const loop of new Set(roundPipelines.map(pipeline => pipeline.id.replace(/-\d+$/, '')))) {
    const loopKeys = roundPipelines
      .filter(pipeline => pipeline.id.replace(/-\d+$/, '') === loop)
      .map(pipeline => pipeline.key);
    expect(new Set(loopKeys).size, loop).toBe(1);
  }
  const gatePipelines = pipelines.filter(pipeline => /-gate-\d+$/.test(pipeline.id));
  // One gate source per loop (the iteration limit is the only baked value).
  const gateLoops = new Set(gatePipelines.map(pipeline => pipeline.id.replace(/-\d+$/, '')));
  expect(new Set(gatePipelines.map(pipeline => pipeline.key)).size).toBeLessThanOrEqual(
    gateLoops.size
  );
  for (const pipeline of pipelines) {
    expect(pipeline.key).not.toMatch(/GROUP_OFFSET/);
  }
});

it('review C-09: a cost relaxation stopped by maximumIterations marks passable costs unresolved', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUCostDistance({
    grid: GRID,
    friction: makeGPURasterBand(createTransientView(graph, 'friction', 'float32', CELLS), GRID, {
      id: 'friction'
    }),
    parameters: createTransientView(graph, 'settings', 'float32', 8),
    sources: createTransientView(graph, 'sources', 'uint32', 3),
    costBreaks: createTransientView(graph, 'thresholds', 'float32', 2),
    output: {
      costs: createTransientView(graph, 'costs', 'float32', CELLS),
      bands: createTransientView(graph, 'bands', 'uint32', CELLS),
      bandCounts: createTransientView(graph, 'band-counts', 'uint32', 2)
    },
    maximumIterations: 4
  }).getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  const arm = ids.indexOf('cost-distance-unresolved-arm');
  const invalidate = ids.indexOf('cost-distance-unresolved-costs');
  // After the last relaxation gate, before the bands read the costs.
  expect(arm).toBeGreaterThan(ids.lastIndexOf('cost-distance-relax-gate-0'));
  expect(invalidate).toBe(arm + 1);
  expect(invalidate).toBeLessThan(ids.indexOf('cost-distance-bands'));
  // The pass over the grid is gated: a converged run dispatches nothing.
  expect(nodes[invalidate].condition).toMatchObject({source: 'gpu', mode: 'indirect'});
  expect(ids.at(-1)).toBe('cost-distance-band-counts-unresolved');
  const sources = new Map(
    getRasterPipelines(nodes).map(pipeline => [pipeline.id, pipeline.source])
  );
  expect(sources.get('cost-distance-unresolved-costs')).toMatch(/0x7fc00000u/);
  expect(sources.get('cost-distance-bands')).toMatch(/band = 4294967294u/);
});
