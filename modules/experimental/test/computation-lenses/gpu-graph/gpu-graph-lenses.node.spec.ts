// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, statSync} from 'node:fs';
import {afterEach, describe, expect, it, vi} from 'vitest';
import {ComputePipeline, type ComputePipelineProps} from '@luma.gl/core';
import {NullDevice} from '@luma.gl/test-utils';
import * as graphExports from '@luma.gl/gpgpu/gpu-graph';
import {
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type GPUComputationCapture
} from '@luma.gl/gpgpu';
import {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import * as captureModule from '../../../../gpgpu/src/computation-record/capture-command-graph';
import {
  extractLensExample,
  findNonPublicImports,
  getLensExamplePath,
  listLensScenarios
} from '../../../src/computation-lenses';
import {
  GPU_GRAPH_COMPUTATION_PACK,
  GPU_GRAPH_FORMULAS
} from '../../../src/computation-lenses/scenarios/gpu-graph/computation-pack';
import {createRailGraphFixture} from '../../../src/computation-lenses/scenarios/gpu-graph/rail-graph-fixture';
import {
  buildGPUGraphScenario,
  getGPUGraphCaptureView,
  recordGPUGraphScenario,
  selectGPUGraphIterations
} from '../../../src/computation-lenses/scenarios/gpu-graph/gpu-graph-scenario';
import {runExample} from '../../../src/computation-lenses/scenarios/gpu-graph/graph-page-rank.example';
import railData from '../../../src/computation-lenses/data/g-nl-rail-graph.json';
import '../../../src/computation-lenses/scenarios/gpu-graph/index';

afterEach(() => vi.restoreAllMocks());

const scenarios = listLensScenarios().filter(scenario => scenario.id.startsWith('gpu-graph.'));
const sourceRoot = new URL('../../../src/computation-lenses/', import.meta.url);

// Match the gpu-graph recording specs: the null backend supplies allocation and shader objects,
// while WebGPU capability metadata and a no-op pipeline permit construction without a GPU.
class StructurePipeline extends ComputePipeline {
  readonly handle = null;
  constructor(
    readonly device: NullDevice,
    props: ComputePipelineProps
  ) {
    super(device, props);
  }
  setBindings(): void {}
}

function createStructureDevice(id: string): NullDevice {
  const device = new NullDevice({id});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxStorageBuffersPerShaderStage: 8
    }
  });
  vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props => new StructurePipeline(device, props!)
  );
  return device;
}

describe('gpu-graph lens substrate', () => {
  it('preserves the real, compact CC0 station and edge dataset', () => {
    expect(statSync(new URL('data/g-nl-rail-graph.json', sourceRoot)).size).toBeLessThanOrEqual(
      150_000
    );
    expect(railData.stationNames).toHaveLength(485);
    expect(railData.lonLat).toHaveLength(970);
    expect(railData.edges.source).toHaveLength(1411);
    expect(railData.edges.target).toHaveLength(1411);
    expect(railData.edges.travelTimeSeconds).toHaveLength(1411);
    expect(railData.stationNames[409]).toBe('Utrecht Centraal');
    expect(railData.licence).toBe('CC0');
    expect(railData.serviceDate).toBe('2026-10-09');
    expect(railData.edges.travelTimeSeconds.every(Number.isInteger)).toBe(true);
    expect(railData.lonLat.every(value => value === Number(value.toFixed(5)))).toBe(true);
  });

  it('deduplicates undirected source pairs and reserves complete symmetric weighted CSR', () => {
    const device = createStructureDevice('rail-undirected');
    const submit = vi.spyOn(device, 'submit');
    const fixture = createRailGraphFixture(device, {
      id: 'rail-undirected',
      directed: false,
      weights: true
    });
    try {
      const pairs = new Set(
        railData.edges.source.flatMap((source, index) => {
          const target = railData.edges.target[index];
          return source === target
            ? []
            : [`${Math.min(source, target)}:${Math.max(source, target)}`];
        })
      );
      expect(fixture.edgeCount).toBe(pairs.size);
      expect(fixture.graph.directed).toBe(false);
      expect(fixture.railTopology.reverse).toBeUndefined();
      expect(fixture.railTopology.forward.neighbors.length).toBe(pairs.size * 2);
      expect(fixture.railTopology.forward.edgeWeights?.length).toBe(pairs.size * 2);
      expect(fixture.graph.edgeWeights?.length).toBe(pairs.size);
      expect(fixture.railTopology.forward.offsets.length).toBe(486);
      expect(fixture.railTopology.invalidEdgeCount.length).toBe(1);
      expect(submit).not.toHaveBeenCalled();
    } finally {
      fixture.destroy();
      device.destroy();
    }
  });

  it('reuses the actual imported output handle and destroys owned columns exactly once', async () => {
    const device = createStructureDevice('rail-output-handle');
    const submit = vi.spyOn(device, 'submit');
    const {fixture, outputs} = await buildGPUGraphScenario(device, {
      id: 'rail-output-handle',
      directed: true,
      reverse: true,
      example: runExample
    });
    const destroyRanks = vi.spyOn(outputs.ranks, 'destroy');
    const destroyBuffer = vi.spyOn(outputs.ranks.data[0].buffer, 'destroy');
    const first = getGPUGraphCaptureView(fixture, 'observation-a', outputs.ranks);
    const second = getGPUGraphCaptureView(fixture, 'observation-b', outputs.ranks);
    expect(first.buffer).toBe(second.buffer);
    const compiled = fixture.commandGraph.compile();
    try {
      const description = compiled.describe({domain: 'gpu-graph'});
      expect(first.buffer.id).toBe('gpu-graph-page-rank-output');
      expect(
        description.graph.nodes.some(
          node => node.op === 'GPUGraphPageRank' && node.writes.includes(first.buffer.id)
        )
      ).toBe(true);
      expect(fixture.railTopology.reverse?.neighbors.length).toBe(1411);
      expect(submit).not.toHaveBeenCalled();
    } finally {
      compiled.destroy();
      fixture.destroy();
      fixture.destroy();
      device.destroy();
    }
    expect(destroyRanks).toHaveBeenCalledTimes(1);
    expect(destroyBuffer).toHaveBeenCalledTimes(1);
  });

  it('keeps early powers of two and the final round within eight deterministic frames', () => {
    expect(selectGPUGraphIterations(Array.from({length: 40}, (_, iteration) => iteration))).toEqual(
      [0, 1, 2, 4, 8, 16, 32, 39]
    );
    expect(
      selectGPUGraphIterations(Array.from({length: 1024}, (_, iteration) => iteration))
    ).toEqual([0, 1, 2, 4, 8, 16, 32, 1023]);
    expect(selectGPUGraphIterations([0])).toEqual([0]);
    expect(selectGPUGraphIterations([])).toEqual([]);
  });
});

describe('gpu-graph scenario contracts', () => {
  it('registers at least the PageRank worked example', () => {
    expect(scenarios.some(scenario => scenario.element === 'GPUGraphPageRank')).toBe(true);
  });
  for (const scenario of scenarios) {
    it(`${scenario.id} has a public example, pack entry and coherent descriptor`, () => {
      expect(scenario.entry).toBe('gpu-graph');
      expect(scenario.element in graphExports).toBe(true);
      expect(scenario.audience).toBe('engineer');
      expect(scenario.defaultRung).toBe('result');
      expect(scenario.rungs).toContain('result');
      expect(scenario.dataset).toMatchObject({id: 'g-nl-rail-graph', licence: 'CC0'});
      expect(GPU_GRAPH_COMPUTATION_PACK.operators?.[scenario.element]).toBeDefined();
      const formula = GPU_GRAPH_FORMULAS[scenario.element];
      expect(formula).toBeDefined();
      expect(scenario.rungs.includes('formula')).toBe(
        !('formula' in formula) && formula.derivation === 'curated'
      );
      const source = readFileSync(
        new URL(scenario.example?.path ?? getLensExamplePath(scenario.id), sourceRoot),
        'utf8'
      );
      expect(extractLensExample(source).lineCount).toBeLessThanOrEqual(25);
      expect(findNonPublicImports(source)).toEqual([]);
      expect(source).toMatch(/export async function runExample\b/);
    });

    it(`${scenario.id} binds each curated term to its own recorded WGSL or resources`, async () => {
      const device = createStructureDevice(`${scenario.id}-structure`);
      const submit = vi.spyOn(device, 'submit');
      try {
        const loaded = await scenario.load();
        expect(loaded.describeStructure).toBeDefined();
        const record = await loaded.describeStructure!(device);
        expect(submit).not.toHaveBeenCalled();
        expect(validateGPUComputationRecord(record).issues).toEqual([]);
        const graph = record.graphs.find(candidate => candidate.form === 'compiled')!;
        const nodes = graph.nodes.filter(node => node.op === scenario.element);
        expect(nodes.length).toBeGreaterThan(0);
        const formula = GPU_GRAPH_FORMULAS[scenario.element];
        if ('formula' in formula) return;
        const texts = nodes.map(
          node => record.refs.sources.find(source => source.id === node.source)?.text ?? ''
        );
        for (const symbol of formula.symbols) {
          expect(symbol.binding, symbol.id).toMatch(/^(wgsl|resource):.+/);
          const [kind, ...parts] = symbol.binding!.split(':');
          const identifier = parts.join(':');
          if (kind === 'wgsl') {
            // Strict even before G-1 lands: missing source.text is a real integration failure.
            const escaped = identifier.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            expect(
              texts.some(text => new RegExp(`\\b${escaped}\\b`).test(text)),
              symbol.binding
            ).toBe(true);
          } else {
            expect(
              graph.resources?.some(resource => resource.id === identifier),
              symbol.binding
            ).toBe(true);
          }
        }
        const termIds = [...formula.tex.matchAll(/\\term\{([^}]+)\}/g)].map(match => match[1]);
        expect(termIds.length).toBeGreaterThan(0);
        for (const identifier of termIds)
          expect(formula.symbols.some(symbol => symbol.id === identifier)).toBe(true);
      } finally {
        device.destroy();
      }
    });
  }
});

describe('gpu-graph recorder plumbing (mock readback)', () => {
  it('retains completed rounds, real capture identities, geometry, highlight, residual and observer effects', async () => {
    const device = createStructureDevice('record-plumbing');
    // Spy on the defining module because shared workers may have already loaded the scenario.
    // Only the transport is mocked; these synthetic values never leave tests.
    const capture = vi.spyOn(captureModule, 'captureGPUCommandGraph');
    let destroyCompiled: ReturnType<typeof vi.spyOn> | undefined;
    let frameCount = 0;
    let copiesPerFrame = 0;
    capture.mockImplementationOnce(async (compiled, options) => {
      destroyCompiled = vi.spyOn(compiled, 'destroy');
      expect(options.at).toBe('step');
      expect(options.encodeOptions.coalesceComputePasses).toBe(false);
      const nodes = compiled.describe({domain: 'gpu-graph'}).graph.nodes;
      frameCount = nodes.length;
      copiesPerFrame = Object.keys(options.views).length;
      const captures: GPUComputationCapture[] = nodes.flatMap(node =>
        Object.entries(options.views).map(([name, view]) => {
          expect(view).toBeInstanceOf(GraphDataView);
          if (!(view instanceof GraphDataView)) throw new Error('Expected packed view');
          const components = view.format === 'float32x2' ? 2 : 1;
          const values = new Float32Array(view.length * components);
          values.fill(node.loop?.iteration ?? 0);
          if (name === 'page-rank' || name === 'iteration:page-rank') values[409] = 100;
          if (name === 'page-rank-residual') values[0] = 0.125;
          return {
            id: `${name}-${node.id}`,
            name,
            graph: options.graphId,
            resource: view.buffer.id,
            afterNode: node.id,
            elementType: 'f32',
            values,
            components,
            provenance: 'gpu-readback'
          };
        })
      );
      return {captures, stepCount: nodes.length, snapshotCount: nodes.length};
    });
    try {
      const {run} = await import(
        '../../../src/computation-lenses/scenarios/gpu-graph/graph-page-rank.run'
      );
      const record = await run(device);
      expect(validateGPUComputationSidecars(record, record).issues).toEqual([]);
      const snapshots = record.captures!.snapshots;
      const frames = snapshots.filter(snapshot => snapshot.name === 'page-rank');
      expect(frames.map(frame => frame.iteration)).toEqual([0, 1, 2, 4, 8, 16, 32, 39]);
      for (const frame of frames) {
        const lastNode = record.graphs[0].nodes
          .filter(
            node => node.op === 'GPUGraphPageRank' && node.loop?.iteration === frame.iteration
          )
          .at(-1);
        expect(frame.afterNode).toBe(lastNode?.id);
        expect(frame.values[0]).toBe(frame.iteration);
        expect(frame.keySpace).toBe('vertex');
      }
      expect(snapshots.find(snapshot => snapshot.id === 'csr-offsets')).toMatchObject({
        role: 'input',
        geometry: {
          kind: 'graph',
          offsetsCapture: 'csr-offsets',
          targetsCapture: 'csr-neighbors',
          layoutCapture: 'station-positions',
          coordinateModel: 'geographic'
        }
      });
      expect(snapshots.find(snapshot => snapshot.id === 'page-rank-stations')).toMatchObject({
        role: 'output',
        components: 2,
        keySpace: 'vertex',
        geometry: {kind: 'points', coordinateModel: 'geographic', valuesCapture: 'page-rank'},
        highlight: {key: 409, label: 'Utrecht Centraal: rank 1'}
      });
      expect(snapshots.every(snapshot => snapshot.provenance === 'gpu-readback')).toBe(true);
      expect(record.metrics![0].encoding).toEqual({
        coalesced: false,
        timestamped: false,
        taps: frameCount * copiesPerFrame,
        lensTwin: true,
        retained: []
      });
      expect(record.metrics![0].nodes).toEqual({});
      expect(record.metrics![0].series).toEqual([
        {name: 'final L1 residual', axis: 'publication', values: [0.125]}
      ]);
      expect(destroyCompiled).toHaveBeenCalledOnce();
    } finally {
      device.destroy();
    }
  });

  it('uses end capture for noniterative work and always releases compiled and fixture resources on failure', async () => {
    const device = createStructureDevice('failed-record');
    const fixture = createRailGraphFixture(device, {id: 'failed-record', directed: true});
    const compiled = fixture.commandGraph.compile();
    const destroyFixture = vi.spyOn(fixture, 'destroy');
    const destroyCompiled = vi.spyOn(compiled, 'destroy');
    vi.spyOn(captureModule, 'captureGPUCommandGraph').mockImplementationOnce(
      async (_compiled, options) => {
        expect(options.at).toBe('end');
        expect(options.encodeOptions.coalesceComputePasses).toBe(true);
        throw new Error('readback failed');
      }
    );
    try {
      await expect(
        recordGPUGraphScenario({
          scenarioId: 'gpu-graph.failed-record',
          element: 'GPUGraphTopology',
          fixture,
          compiled,
          outputCaptures: {},
          choices: []
        })
      ).rejects.toThrow('readback failed');
      expect(destroyCompiled).toHaveBeenCalledOnce();
      expect(destroyFixture).toHaveBeenCalledOnce();
    } finally {
      device.destroy();
    }
  });
});
