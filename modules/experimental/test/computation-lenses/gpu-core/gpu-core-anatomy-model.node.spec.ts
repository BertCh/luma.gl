// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {describe, expect, test} from 'vitest';

import {fromRecordedLensFile} from '../../../src/computation-lenses/registry';
import {
  buildGPUCoreAnatomyModel,
  type GPUCoreAnatomyModel
} from '../../../../../website/src/components/docs/gpu-core-compiler-model';

const MODEL_PATH = 'website/src/components/docs/gpu-core-compiler-model.ts';
const RECORD_PATH = 'docs/lens-records/gpu-core.scan.anatomy.json';

function makeNode(
  id: string,
  namespace: string,
  op: string,
  reads: string[],
  writes: string[],
  invocationCount: number
) {
  return {
    id,
    label: id,
    namespace,
    kind: 'compute',
    op,
    source: `wgsl:${id}`,
    reads,
    writes,
    dispatch: {
      workgroups: [Math.ceil(invocationCount / 256), 1, 1],
      invocationCount,
      indirect: false
    },
    attrs: {}
  };
}

function makeFile() {
  return {
    schema: 'luma.lens-record/1',
    scenarioId: 'gpu-core.scan.anatomy',
    recordedAt: '2026-10-15T00:00:00.000Z',
    adapterLabel: 'test adapter',
    record: {
      schema: 'luma.computation/1',
      runId: 'run-test',
      refs: {sources: [{id: 'wgsl:scan', moduleId: 'GPUScan/level-0'}]},
      graphs: [
        {
          id: 'g',
          form: 'compiled',
          nodes: [
            makeNode('mask', 'gpu-mask', 'GPUMask', ['a'], ['t1'], 1024),
            makeNode('scan', 'gpu-compaction', 'GPUScan', ['t1'], ['t2'], 1024),
            makeNode('scatter', 'gpu-compaction', 'GPUCompaction', ['t1', 't2'], ['out'], 1024),
            makeNode('sum', 'gpu-reduction', 'GPUReduction', ['t1'], ['t3'], 1024),
            makeNode('final', 'gpu-reduction', 'GPUReduction', ['t3'], ['total'], 1)
          ].map(node => (node.id === 'scan' ? {...node, source: 'wgsl:scan'} : node)),
          edges: [
            {from: 'mask', to: 'scan', kind: 'raw', resource: 't1'},
            {from: 'mask', to: 'sum', kind: 'raw', resource: 't1'},
            {from: 'scan', to: 'scatter', kind: 'raw', resource: 't2'},
            {from: 'sum', to: 'final', kind: 'raw', resource: 't3'},
            {from: 'scatter', to: 'final', kind: 'explicit'}
          ],
          resources: [
            {
              id: 'a',
              kind: 'buffer',
              bytes: 4096,
              ownership: 'imported',
              firstNode: 'mask',
              lastNode: 'mask'
            },
            {
              id: 't1',
              kind: 'buffer',
              bytes: 4096,
              ownership: 'transient',
              allocation: 'alloc-0',
              allocationBytes: 4096,
              firstNode: 'mask',
              lastNode: 'scatter'
            },
            {
              id: 't2',
              kind: 'buffer',
              bytes: 16,
              ownership: 'transient',
              allocation: 'alloc-1',
              allocationBytes: 16,
              firstNode: 'scan',
              lastNode: 'scatter'
            },
            {
              id: 't3',
              kind: 'buffer',
              bytes: 4,
              ownership: 'transient',
              allocation: 'alloc-1',
              allocationBytes: 16,
              firstNode: 'sum',
              lastNode: 'final'
            }
          ]
        }
      ],
      metrics: [
        {
          series: [
            {name: 'slices@512', axis: 'node', values: [0, 1, 2, 3, 4]},
            {name: 'slices@4096', axis: 'node', values: [0, 0, 0, 0, 0]}
          ]
        }
      ]
    }
  };
}

describe('buildGPUCoreAnatomyModel', () => {
  test('builds every stage from a recorded file', () => {
    const model = buildGPUCoreAnatomyModel(makeFile()) as GPUCoreAnatomyModel;
    expect(model.status).toBe('ready');
    expect(model.provenance).toEqual({
      recordedAt: '2026-10-15T00:00:00.000Z',
      adapterLabel: 'test adapter',
      runId: 'run-test'
    });
    // declared
    expect(model.nodes.map(node => node.id)).toEqual(['mask', 'scan', 'scatter', 'sum', 'final']);
    expect(model.contributors.map(contributor => contributor.label)).toEqual([
      'GPUMask',
      'GPUCompaction',
      'GPUReduction'
    ]);
    expect(model.nodes[1].contributorPath).toBe('GPUCompaction › GPUScan');
    expect(model.nodes[1].moduleId).toBe('GPUScan/level-0');
    expect(model.nodes[0].moduleId).toBeUndefined();
    // hazards
    expect(model.edges.map(edge => edge.kind)).toEqual(['raw', 'raw', 'raw', 'raw', 'explicit']);
    expect(model.edges[4].resource).toBeUndefined();
    // lifetimes
    const t1 = model.resources.find(resource => resource.id === 't1')!;
    expect([t1.firstNode, t1.lastNode, t1.byteLength]).toEqual([0, 2, 4096]);
    expect(model.resources.find(resource => resource.id === 'a')!.ownership).toBe('imported');
    // allocations
    expect(model.allocations.map(group => group.resourceIds)).toEqual([['t1'], ['t2', 't3']]);
    expect(model.allocationSentence).toBe('t2 and t3 share physical allocation alloc-1.');
    // encoded
    expect(model.encodedPasses).toBeUndefined();
    expect(model.encodedNote).toContain('not recorded');
    // slices
    expect(model.sliceSeries.map(series => series.budget)).toEqual([512, 4096]);
    expect(model.sliceSeries[0].slices.map(slice => slice.nodeIds)).toEqual([
      ['mask'],
      ['scan'],
      ['scatter'],
      ['sum'],
      ['final']
    ]);
    expect(model.sliceSeries[0].slices.map(slice => slice.oversized)).toEqual([
      true,
      true,
      true,
      true,
      false
    ]);
    expect(model.sliceSeries[1].slices).toHaveLength(1);
    expect(model.sliceSeries[1].slices[0].invocationCount).toBe(4097);
  });

  test('says when no transients share an allocation', () => {
    const file = makeFile();
    (file.record.graphs[0].resources[3] as {allocation: string}).allocation = 'alloc-2';
    const model = buildGPUCoreAnatomyModel(file) as GPUCoreAnatomyModel;
    expect(model.allocationSentence).toBe(
      'In this run no logical transients share a physical allocation.'
    );
  });

  test('is pending without a record and invalid for a wrong schema', () => {
    expect(buildGPUCoreAnatomyModel(undefined).status).toBe('pending');
    expect(buildGPUCoreAnatomyModel(null).status).toBe('pending');
    expect(buildGPUCoreAnatomyModel({schema: 'nope'}).status).toBe('invalid');
    expect(
      buildGPUCoreAnatomyModel({schema: 'luma.lens-record/1', record: {graphs: []}}).status
    ).toBe('invalid');
  });

  test('keeps no hand-written node ids in the model source', () => {
    const source = readFileSync(MODEL_PATH, 'utf8');
    for (const id of [
      'classify',
      'scan-blocks',
      'scan-carry',
      'scatter',
      'style',
      'instances',
      'render'
    ]) {
      expect(source, id).not.toMatch(new RegExp(`['"\`]${id}['"\`]`));
    }
  });

  test('rebuilds the real recorded run', () => {
    const decoded = fromRecordedLensFile(JSON.parse(readFileSync(RECORD_PATH, 'utf8')));
    const model = buildGPUCoreAnatomyModel(decoded) as GPUCoreAnatomyModel;
    expect(model.status).toBe('ready');
    const compiled = decoded.record.graphs.find(graph => graph.form === 'compiled')!;
    expect(model.nodes).toHaveLength(compiled.nodes.length);
    expect(model.contributors.length).toBeGreaterThan(0);
    expect(model.edges.length).toBeGreaterThan(0);
    expect(model.resources.length).toBeGreaterThan(0);
    expect(model.allocations.length).toBeGreaterThan(0);
    expect(model.allocationSentence.length).toBeGreaterThan(0);
    expect(model.nodes.every(node => node.workgroups?.length === 3)).toBe(true);
    expect(model.sliceSeries.map(series => series.budget)).toEqual([512, 2048, 8192]);
    for (const series of model.sliceSeries) {
      expect(series.slices.length).toBeGreaterThan(0);
      expect(series.slices.flatMap(slice => slice.nodeIds)).toHaveLength(model.nodes.length);
    }
  });
});
