// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  Buffer,
  ComputePass,
  type Bindings,
  type BindingsByGroup,
  type ComputePassProps,
  type ComputePipeline
} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUCommandGraphEncoding,
  type GPUCommandGraphComputeNode
} from '@luma.gl/gpgpu/gpu-core';
import {NullDevice} from '@luma.gl/test-utils';
import {afterEach, describe, expect, test, vi} from 'vitest';
import type {CompiledNode} from '../../src/gpu-core/gpu-command-graph-compiler';

const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const cleanup of cleanups.splice(0).reverse()) cleanup();
});

class RecordingComputePass extends ComputePass {
  constructor(
    device: NullDevice,
    props: ComputePassProps,
    private readonly labels: string[]
  ) {
    super(device, props);
  }
  override destroy(): void {
    this.destroyResource();
  }
  override end(): void {
    this.destroy();
  }
  override setPipeline(_pipeline: ComputePipeline): void {}
  override setBindings(_bindings: Bindings | BindingsByGroup): void {}
  override dispatch(_x: number, _y = 1, _z = 1): void {}
  override dispatchIndirect(_buffer: Buffer, _offset = 0): void {}
  override pushDebugGroup(label: string): void {
    this.labels.push(label);
  }
  override popDebugGroup(): void {}
  override insertDebugMarker(_label: string): void {}
}

function createFixture(recording: boolean) {
  const device = new NullDevice({id: 'recording-device'});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  const graph = new GPUCommandGraph(device, {
    id: 'recording-graph',
    ...(recording ? {recording: {}} : {})
  });
  const commandEncoder = device.createCommandEncoder();
  const labels: string[] = [];
  vi.spyOn(commandEncoder, 'beginComputePass').mockImplementation(
    (props = {}) => new RecordingComputePass(device, props, labels)
  );
  cleanups.push(
    () => device.destroy(),
    () => commandEncoder.destroy()
  );
  return {device, graph, commandEncoder, labels};
}

function compileGraph(graph: GPUCommandGraph) {
  const compiled = graph.compile();
  cleanups.push(() => compiled.destroy());
  return compiled;
}

describe('GPUCommandGraph recording', () => {
  test('recording changes no scheduling, allocation, preflight or pass behavior', () => {
    const build = (recording: boolean) => {
      const fixture = createFixture(recording);
      const buffer = fixture.graph.createTransientBuffer({
        id: 'scratch',
        byteLength: 16,
        usage: Buffer.STORAGE
      });
      const nodes: GPUCommandGraphComputeNode[] = ['write', 'read'].map((id, index) => ({
        id,
        type: 'compute',
        resources: [{buffer, usage: index ? 'storage-read' : 'storage-write'}],
        annotation: {label: 'supplied unconditionally', namespace: 'producer'},
        workload: {operation: 'test', commandCount: 1},
        compile: () => ({encode: () => {}})
      }));
      fixture.graph.add({getCommandNodes: () => nodes});
      fixture.graph.addComputePass({
        id: 'undefined-annotation',
        annotation: undefined,
        compile: () => ({encode: () => {}})
      });
      const compiled = compileGraph(fixture.graph);
      const encoding = compiled.encode(fixture.commandEncoder, {parameters: undefined});
      return {compiled, encoding};
    };
    const production = build(false);
    const recorded = build(true);
    expect(production.compiled.stats).toEqual(recorded.compiled.stats);
    expect(production.compiled.preflight).toEqual(recorded.compiled.preflight);
    expect(production.encoding.stats.nodes.map(node => node.id)).toEqual(
      recorded.encoding.stats.nodes.map(node => node.id)
    );
    expect(production.encoding.stats.computePassCount).toBe(
      recorded.encoding.stats.computePassCount
    );
    const scheduledNodes = Reflect.get(
      production.compiled,
      'compiledNodes'
    ) as CompiledNode<void>[];
    expect(scheduledNodes).toHaveLength(3);
    for (const {node} of scheduledNodes) expect(Object.hasOwn(node, 'annotation')).toBe(false);
    expect(production.compiled.describe().graph.nodes.every(node => node.namespace === '')).toBe(
      true
    );
  });

  test.each([false, true])('uses scoped debug labels only when recording (%s)', recording => {
    const fixture = createFixture(recording);
    fixture.graph.withRecordScope({namespace: 'scope'}, () => {
      fixture.graph.addComputePass({id: 'compute', compile: () => ({encode: () => {}})});
      fixture.graph.addRenderPass({id: 'render', compile: () => ({encode: () => {}})});
    });
    const beginRenderPass = fixture.commandEncoder.beginRenderPass.bind(fixture.commandEncoder);
    vi.spyOn(fixture.commandEncoder, 'beginRenderPass').mockImplementation(props => {
      const pass = beginRenderPass(props);
      vi.spyOn(pass, 'pushDebugGroup').mockImplementation(label => {
        fixture.labels.push(label);
      });
      return pass;
    });
    compileGraph(fixture.graph).encode(fixture.commandEncoder, {parameters: undefined});
    expect(fixture.labels).toEqual(
      recording ? ['scope/compute', 'scope/render'] : ['compute', 'render']
    );
  });

  test('scopes named producers and groups without duplicating their expansion', () => {
    const fixture = createFixture(true);
    class Producer {
      readonly id = 'producer';
      getCommandNodes(): GPUCommandGraphComputeNode[] {
        return [{id: 'compute', type: 'compute', compile: () => ({encode: () => {}})}];
      }
    }
    const group = {id: 'group', getNodes: vi.fn(() => [new Producer()])};
    fixture.graph.add(group);
    expect(group.getNodes).toHaveBeenCalledTimes(1);
    expect(compileGraph(fixture.graph).describe().graph.nodes[0]).toMatchObject({
      namespace: 'group/producer',
      op: 'Producer'
    });
  });

  test.each([
    true,
    false
  ])('describes pass coalescing, copies, renders and skipped nodes (%s)', coalesceComputePasses => {
    const fixture = createFixture(true);
    for (const id of ['first', 'second'])
      fixture.graph.addComputePass({id, compile: () => ({encode: () => {}})});
    fixture.graph.addComputePass({
      id: 'skipped',
      condition: {id: 'disabled', source: 'cpu', evaluate: () => false},
      compile: () => ({encode: () => {}})
    });
    fixture.graph.addCopyPass({id: 'copy', compile: () => ({encode: () => {}})});
    fixture.graph.addRenderPass({id: 'render', compile: () => ({encode: () => {}})});
    fixture.graph.addComputePass({id: 'last', compile: () => ({encode: () => {}})});
    const encoding = compileGraph(fixture.graph).encode(fixture.commandEncoder, {
      parameters: undefined,
      coalesceComputePasses
    });
    expect(encoding.stats.nodes.map(node => node.passIndex)).toEqual(
      coalesceComputePasses
        ? [0, 0, undefined, undefined, 1, 2]
        : [0, 1, undefined, undefined, 2, 3]
    );
    expect(Object.hasOwn(encoding.stats.nodes[2], 'passIndex')).toBe(false);
    expect(Object.hasOwn(encoding.stats.nodes[3], 'passIndex')).toBe(false);
    const description = encoding.describe({compiledGraphId: 'compiled'});
    expect(description).toMatchObject({
      id: 'compiled-encoded',
      form: 'encoded',
      domain: 'gpu-core',
      edges: [],
      attrs: {skipped: 'skipped'}
    });
    expect(description.nodes[0]).toEqual({
      id: 'pass-0',
      label: 'compute pass 0',
      kind: 'pass',
      namespace: '',
      derivedFrom: coalesceComputePasses
        ? ['compiled:first', 'compiled:second']
        : ['compiled:first'],
      reads: [],
      writes: [],
      attrs: {passType: 'compute', nodeCount: coalesceComputePasses ? 2 : 1}
    });
    expect(description.nodes.find(node => node.id === 'copy-copy')).toMatchObject({
      kind: 'copy',
      derivedFrom: ['compiled:copy']
    });
    expect(description.nodes.map(node => node.id)).toEqual(
      coalesceComputePasses
        ? ['pass-0', 'copy-copy', 'pass-1', 'pass-2']
        : ['pass-0', 'pass-1', 'copy-copy', 'pass-2', 'pass-3']
    );
    expect(() => encoding.describe({compiledGraphId: 'compiled', graphId: 'bad:id'})).toThrow(
      'must not contain'
    );
  });

  test('reads raw timestamp ticks while preserving the duration API result', async () => {
    const fixture = createFixture(true);
    const querySet = fixture.device.createQuerySet({type: 'timestamp', count: 8});
    cleanups.push(() => querySet.destroy());
    const readResults = vi
      .spyOn(querySet, 'readResults')
      .mockResolvedValue([1000000n, 2000000n, 3000000n]);
    const readDuration = vi.spyOn(querySet, 'readTimestampDuration').mockResolvedValue(2);
    const encoding = new GPUCommandGraphEncoding(
      [
        {
          stats: {
            id: 'timed',
            type: 'compute',
            cpuEncodeTimeMilliseconds: 1,
            hasGPUTimestamps: true
          },
          timestamp: {querySet, beginIndex: 2, endIndex: 4}
        },
        {stats: {id: 'copy', type: 'copy', cpuEncodeTimeMilliseconds: 1, hasGPUTimestamps: false}}
      ],
      2
    );
    expect(readResults).not.toHaveBeenCalled();
    const timings = await encoding.readTimings();
    expect(readDuration).toHaveBeenCalledWith(2, 4);
    expect(readResults).toHaveBeenCalledWith({firstQuery: 2, queryCount: 3});
    expect(timings.nodes[0]).toMatchObject({
      gpuTimeMilliseconds: 2,
      gpuBeginNanoseconds: 1000000,
      gpuEndNanoseconds: 3000000
    });
    expect(timings.gpuTimeMilliseconds).toBe(2);
    expect(Object.hasOwn(timings.nodes[1], 'gpuBeginNanoseconds')).toBe(false);
  });
});
