// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {NullDevice} from '@luma.gl/test-utils';
import {
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type GPUComputationCapture
} from '@luma.gl/gpgpu';
import * as captureModule from '../../../../gpgpu/src/computation-record/capture-command-graph';
import {afterEach, expect, test, vi} from 'vitest';
import {getLensScenario} from '../../../src/computation-lenses/registry';
import {buildDatumReceiptModel} from '../../../src/computation-views/views/datum-receipt';
import {buildDispatchViewModel} from '../../../src/computation-views/views/dispatch-view';
import '../../../src/computation-lenses/scenarios/gpu-core/index';

const SCENARIO_ID = 'gpu-core.conjugate-gradient-program';

afterEach(() => vi.restoreAllMocks());

function createDevice() {
  const device = new NullDevice({_cacheShaders: false, _cachePipelines: false});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxComputeWorkgroupsPerDimension: 65535,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 32768,
      maxStorageBuffersPerShaderStage: 8,
      maxStorageBufferBindingSize: 2 ** 30
    }
  });
  vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props => ({id: props?.id, hash: '', destroy() {}}) as never
  );
  return device;
}

test('CG card declares kernel, graph, and datum in engineer order', () => {
  expect(getLensScenario(SCENARIO_ID)!.rungs).toEqual([
    'result',
    'kernel',
    'graph',
    'datum',
    'iterations',
    'formula',
    'table'
  ]);
  const document = readFileSync(
    'docs/api-reference/experimental/gpu-core/gpu-preconditioned-conjugate-gradient.md',
    'utf8'
  );
  const directives = [...document.matchAll(/::lens\{[^}]*rung=(\w+)[^}]*caption="([^"]+)"\}/g)];
  expect(directives.map(directive => directive[1])).toEqual([
    'result',
    'kernel',
    'graph',
    'datum',
    'iterations',
    'formula'
  ]);
  for (const directive of directives)
    expect(directive[2].split(/\s+/).length).toBeLessThanOrEqual(18);
});

test('CG dispatches retain actual SpMV and dot-product sources', async () => {
  const device = createDevice();
  const submit = vi.spyOn(device, 'submit');
  try {
    const scenario = await getLensScenario(SCENARIO_ID)!.load();
    const record = await scenario.describeStructure!(device);
    expect(validateGPUComputationRecord(record)).toMatchObject({valid: true});
    const defaultKernel = buildDispatchViewModel({
      record,
      bindings: {},
      ground: 'light',
      idPrefix: 'cg-default-kernel',
      reducedMotion: true
    });
    expect(
      record.graphs[1].nodes.find(node => node.id === defaultKernel.nodeId)?.derivedFrom
    ).toContain('smooth-intent:smooth-initial-spmv');
    expect(defaultKernel.source).not.toHaveProperty('status');
    for (const intent of ['smooth-initial-spmv', 'smooth-rr0', 'smooth-spmv', 'smooth-pq']) {
      const node = record.graphs[1].nodes.find(
        candidate => candidate.source && candidate.derivedFrom?.includes(`smooth-intent:${intent}`)
      );
      expect(node, intent).toBeDefined();
      const model = buildDispatchViewModel({
        record,
        bindings: {node: `${record.graphs[1].id}:${node!.id}`},
        ground: 'light',
        idPrefix: intent,
        reducedMotion: true
      });
      expect(model.nodeId).toBe(node!.id);
      expect(model.workgroupCount).toBeGreaterThan(0);
      expect(model.source).not.toHaveProperty('status');
      expect(record.refs.sources.find(source => source.id === node!.source)?.text).toContain(
        '@compute'
      );
      expect(node!.bindings?.map(binding => binding.name)).toEqual(
        intent.endsWith('spmv')
          ? [
              'rowBegins',
              'rowEnds',
              'columnIndices',
              'matrixValues',
              'vectorValues',
              'outputValues'
            ]
          : intent === 'smooth-rr0'
            ? ['inputValues', 'gpuValues']
            : ['inputValues', 'inputBValues', 'gpuValues']
      );
      if (intent === 'smooth-spmv' || intent === 'smooth-pq') expect(model.indirect).toBe(true);
    }
    expect(
      record.graphs[1].resources?.filter(resource => resource.retained).map(resource => resource.id)
    ).toEqual(expect.arrayContaining(['gpu-value-arena-buffer', 'smooth-r', 'smooth-p']));
    expect(submit).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

test('CG receipt selects GPU snapshot boundaries and excludes gated iterations', async () => {
  const device = createDevice();
  // Synthetic readbacks are a test oracle only. Compilation and capture selection remain real.
  const capture = vi
    .spyOn(captureModule, 'captureGPUCommandGraph')
    .mockImplementation(async (compiled, props) => {
      const graph = compiled.describe().graph;
      const captures = graph.nodes.flatMap((node, nodeIndex) =>
        Object.entries(props.views).map(([name, view]) => {
          const length = 'length' in view ? view.length : view.byteLength / 4;
          const iteration = node.loop?.iteration;
          const value =
            name === 'residualSquared'
              ? iteration === undefined
                ? 100000
                : iteration === 0
                  ? 50000
                  : 25000
              : nodeIndex + 1;
          return {
            id: `${node.id}:${name}`,
            name,
            afterNode: node.id,
            graph: props.graphId,
            resource: 'buffer' in view ? view.buffer.id : view.id,
            elementType: 'f32' as const,
            values: new Float32Array(length).fill(value),
            provenance: 'gpu-readback' as const
          };
        })
      );
      return {captures, stepCount: graph.nodes.length, snapshotCount: graph.nodes.length};
    });
  try {
    const scenario = await getLensScenario(SCENARIO_ID)!.load();
    const record = await scenario.run(device);
    expect(capture).toHaveBeenCalledOnce();
    expect(validateGPUComputationSidecars(record, record)).toMatchObject({valid: true});
    const model = buildDatumReceiptModel({
      record,
      bindings: {entity: 0},
      ground: 'light',
      idPrefix: 'cg-receipt',
      reducedMotion: true
    });
    expect(model.keySpaceLabel).toContain('x[134]');
    for (const name of ['solution', 'residual', 'residualSquared']) {
      expect(model.rows.filter(row => row.name === name).map(row => row.iteration)).toEqual([
        0, 1, 2
      ]);
    }
    for (const name of ['alpha', 'searchBeforeUpdate']) {
      const rows = model.rows.filter(row => row.name === name);
      expect(rows.map(row => row.iteration)).toEqual([1, 2]);
      for (const row of rows) {
        const node = record.graphs[1].nodes.find(candidate => candidate.id === row.afterNode)!;
        expect(node.derivedFrom).toContain('smooth-intent:smooth-alpha');
        expect(node.loop?.iteration).toBe(row.iteration! - 1);
      }
    }
    for (const name of ['matrixColumns', 'matrixValues']) {
      const row = model.rows.find(candidate => candidate.name === name)!;
      expect(row.value.split(', ')).toHaveLength(5);
    }
    expect(model.rows.every(row => row.provenance === 'readback')).toBe(true);
    const rawCaptures: GPUComputationCapture[] = (await capture.mock.results[0].value).captures;
    for (const snapshot of record.captures!.snapshots) {
      expect(snapshot.values).toBe(rawCaptures.find(raw => raw.id === snapshot.id)!.values);
    }
    expect(
      record.captures!.snapshots.filter(snapshot => snapshot.role).map(snapshot => snapshot.name)
    ).toEqual(['elevation', 'smoothed']);
    expect(record.metrics![0].series![0].values).toHaveLength(3);
  } finally {
    device.destroy();
  }
});
