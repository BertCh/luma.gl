// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, Texture} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type GPUCommandGraphNode,
  type GraphResourceUse
} from '@luma.gl/gpgpu/gpu-core';
import {NullDevice} from '@luma.gl/test-utils';
import {afterEach, describe, expect, test, vi} from 'vitest';
import {inferGPUCommandGraphDependencies} from '../../src/gpu-core/gpu-command-graph-compiler';
import {setGPUComputeDispatchWorkgroups} from '../../src/gpu-core/gpu-command-dispatch-metadata';

const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const cleanup of cleanups.splice(0).reverse()) cleanup();
});

function createGraph(retain?: readonly string[], recording = true): GPUCommandGraph {
  const device = new NullDevice({id: 'description-device'});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  cleanups.push(() => device.destroy());
  return new GPUCommandGraph(device, {
    id: 'description',
    ...(recording ? {recording: {retain}} : {})
  });
}

function compileGraph(graph: GPUCommandGraph) {
  const compiled = graph.compile();
  cleanups.push(() => compiled.destroy());
  return compiled;
}

function addNode(
  graph: GPUCommandGraph,
  id: string,
  resources: GraphResourceUse[],
  dependsOn?: string[]
) {
  const node: GPUCommandGraphNode = {
    id,
    type: 'copy',
    resources,
    dependsOn,
    compile: () => ({encode: () => {}})
  };
  graph.add(node);
  return node;
}

function createBuffer(graph: GPUCommandGraph, id: string) {
  return graph.createTransientBuffer({
    id,
    byteLength: 16,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.INDIRECT
  });
}

describe('GPUCommandGraph description', () => {
  test('reports distinct hazard kinds with the original dependency pairs', () => {
    const graph = createGraph();
    const buffer = createBuffer(graph, 'a');
    const insertionNodes = [
      addNode(graph, 'write', [{buffer, usage: 'storage-write'}]),
      addNode(graph, 'read', [
        {buffer, usage: 'storage-read'},
        {buffer, usage: 'storage-read'}
      ]),
      addNode(graph, 'rewrite', [{buffer, usage: 'storage-write'}]),
      addNode(graph, 'read-write', [{buffer, usage: 'storage-read-write'}], ['write'])
    ];
    const conditional: GPUCommandGraphNode = {
      id: 'indirect',
      type: 'compute',
      resources: [{buffer, usage: 'indirect'}],
      condition: {id: 'gate', source: 'gpu', mode: 'indirect', buffer},
      compile: () => ({encode: () => {}})
    };
    graph.add(conditional);
    insertionNodes.push(conditional);
    const compiled = compileGraph(graph);
    const description = compiled.describe();
    expect(description.graph.edges).toEqual([
      {from: 'write', to: 'read', kind: 'raw', resource: 'a'},
      {from: 'write', to: 'rewrite', kind: 'waw', resource: 'a'},
      {from: 'read', to: 'rewrite', kind: 'war', resource: 'a'},
      {from: 'write', to: 'read-write', kind: 'explicit'},
      {from: 'rewrite', to: 'read-write', kind: 'raw', resource: 'a'},
      {from: 'read-write', to: 'indirect', kind: 'indirect', resource: 'a'}
    ]);
    const dependencies = inferGPUCommandGraphDependencies(insertionNodes);
    expect(new Set(description.graph.edges.map(edge => `${edge.from}>${edge.to}`))).toEqual(
      new Set([...dependencies].flatMap(([to, from]) => [...from].map(id => `${id}>${to}`)))
    );
    expect(description.graph.nodes.at(-1)?.condition).toEqual({
      id: 'gate',
      source: 'gpu',
      mode: 'indirect',
      resource: 'a'
    });
    expect(description.graph.nodes[3].reads).toEqual(['a']);
    expect(description.graph.nodes[3].writes).toEqual(['a']);
    expect(description.graph.id).toBe('description-compiled');
    expect(
      compiled.describe({graphId: 'custom', label: 'Custom', domain: 'custom-domain'}).graph
    ).toMatchObject({id: 'custom', label: 'Custom', domain: 'custom-domain'});
    expect(() => compiled.describe({graphId: 'bad:id'})).toThrow('must not contain');
    expect(description).toEqual(JSON.parse(JSON.stringify(description)));
  });

  test('never emits self edges, including explicit and repeated read-write uses', () => {
    const graph = createGraph();
    const buffer = createBuffer(graph, 'a');
    const node = addNode(
      graph,
      'self',
      [
        {buffer, usage: 'storage-read-write'},
        {buffer, usage: 'storage-read-write'}
      ],
      ['self']
    );
    const collect = vi.fn();
    expect(inferGPUCommandGraphDependencies([node], collect).get('self')).toEqual(new Set());
    expect(collect).not.toHaveBeenCalled();
  });

  test.each([
    false,
    true
  ])('reports lifetimes, imports and retained allocation reuse (%s)', retain => {
    const graph = createGraph(retain ? ['b'] : undefined);
    const first = createBuffer(graph, 'b');
    const overlap = createBuffer(graph, 'overlap');
    const later = createBuffer(graph, 'later');
    const imported = graph.importBuffer({id: 'input', byteLength: 16, usage: Buffer.COPY_SRC});
    addNode(graph, 'first', [
      {buffer: first, usage: 'copy-destination'},
      {buffer: overlap, usage: 'copy-destination'},
      {buffer: imported, usage: 'copy-source'}
    ]);
    addNode(graph, 'last-first', [{buffer: first, usage: 'copy-source'}]);
    addNode(
      graph,
      'later',
      [
        {buffer: later, usage: 'copy-destination'},
        {buffer: overlap, usage: 'copy-source'}
      ],
      ['last-first']
    );
    const compiled = compileGraph(graph);
    const [firstResource, overlapResource, laterResource, importedResource] =
      compiled.describe().graph.resources!;
    expect(firstResource).toMatchObject({
      id: 'b',
      firstNode: 'first',
      lastNode: 'last-first',
      allocationBytes: 16
    });
    expect(overlapResource.allocation).not.toBe(firstResource.allocation);
    expect(importedResource).toEqual({
      id: 'input',
      kind: 'buffer',
      bytes: 16,
      ownership: 'imported',
      firstNode: 'first',
      lastNode: 'first'
    });
    if (retain) {
      expect(laterResource.allocation).not.toBe(firstResource.allocation);
      expect(firstResource.retained).toBe(true);
      expect(compiled.describe().graph.attrs).toEqual({retained: 1});
      expect(compiled.getPhysicalBuffer(first)).toBeDefined();
    } else {
      expect(laterResource.allocation).toBe(firstResource.allocation);
      expect(() => compiled.getPhysicalBuffer(first)).toThrow('aliased and not retained');
    }
  });

  test('retained handles may reuse earlier allocations but prevent later reuse', async () => {
    const graph = createGraph(['middle']);
    for (const [index, id] of ['first', 'middle', 'last'].entries()) {
      addNode(
        graph,
        id,
        [{buffer: createBuffer(graph, id), usage: 'copy-destination'}],
        index ? [['first', 'middle'][index - 1]] : []
      );
    }
    const compiled = await graph.compileAsync();
    cleanups.push(() => compiled.destroy());
    const [first, middle, last] = compiled.describe().graph.resources!;
    expect(first.allocation).toBe(middle.allocation);
    expect(last.allocation).not.toBe(middle.allocation);
  });

  test('adds COPY_SRC to observed storage-only transients as a stated observer effect', () => {
    const storageOnly = (graph: GPUCommandGraph, id: string) =>
      graph.createTransientBuffer({id, byteLength: 16, usage: Buffer.STORAGE});
    const buildGraph = (retain?: readonly string[], recording = true, tap = false) => {
      const graph = createGraph(retain, recording);
      const retained = storageOnly(graph, 'retained');
      const tapped = storageOnly(graph, 'tapped');
      const sink = createBuffer(graph, 'sink');
      addNode(graph, 'write', [
        {buffer: retained, usage: 'storage-write'},
        {buffer: tapped, usage: 'storage-write'}
      ]);
      if (tap) graph.addObservationTap(tapped, {afterNode: 'write', sink});
      addNode(
        graph,
        'read',
        [
          {buffer: sink, usage: 'storage-read'},
          {buffer: retained, usage: 'storage-read'}
        ],
        ['write']
      );
      const compiled = compileGraph(graph);
      return {compiled, retained, tapped, resources: compiled.describe().graph.resources!};
    };

    const production = buildGraph(undefined, false);
    expect(production.compiled.getPhysicalBuffer(production.retained).usage & Buffer.COPY_SRC).toBe(
      0
    );
    expect(production.resources.some(resource => resource.attrs?.['observerUsage'])).toBe(false);

    const observed = buildGraph(['retained'], true, true);
    expect(observed.compiled.getPhysicalBuffer(observed.retained).usage & Buffer.COPY_SRC).toBe(
      Buffer.COPY_SRC
    );
    const byId = new Map(observed.resources.map(resource => [resource.id, resource]));
    expect(byId.get('retained')?.attrs).toEqual({observerUsage: 'copy-src'});
    expect(byId.get('tapped')?.attrs).toEqual({observerUsage: 'copy-src'});
    // Already-readable buffers are not reported: their usage did not change.
    expect(byId.get('sink')?.attrs).toBeUndefined();
  });

  test.each(['unknown', 'imported'])('rejects invalid retain id %s', id => {
    const graph = createGraph([id]);
    graph.importBuffer({id: 'imported', byteLength: 4, usage: Buffer.COPY_SRC});
    expect(() => graph.compile()).toThrow(
      `GPUCommandGraph retain id "${id}" is not a transient resource`
    );
  });

  test('describes overlapping texture hazards, allocation estimates and external lifetimes', () => {
    const graph = createGraph(['texture']);
    const descriptor = {
      format: 'rgba8unorm' as const,
      width: 4,
      height: 4,
      mipLevels: 2,
      usage: Texture.COPY_SRC | Texture.COPY_DST
    };
    const texture = graph.createTransientTexture({id: 'texture', ...descriptor});
    const later = graph.createTransientTexture({id: 'later', ...descriptor});
    const firstMip = graph.createTextureView(texture, {mipLevelCount: 1});
    const secondMip = graph.createTextureView(texture, {baseMipLevel: 1, mipLevelCount: 1});
    addNode(graph, 'write', [{texture: firstMip, usage: 'copy-destination'}]);
    addNode(graph, 'separate', [{texture: secondMip, usage: 'copy-source'}], ['write']);
    addNode(graph, 'read', [{texture: firstMip, usage: 'copy-source'}], ['separate']);
    addNode(graph, 'rewrite', [{texture: firstMip, usage: 'copy-destination'}]);
    addNode(graph, 'later', [{texture: later, usage: 'copy-destination'}], ['rewrite']);
    const external = graph.importExternalTexture({id: 'external', width: 4, height: 4});
    graph.addRenderPass({
      id: 'sample',
      resources: [{externalTexture: external, usage: 'sampled'}],
      compile: () => ({encode: () => {}})
    });
    const description = compileGraph(graph).describe().graph;
    expect(description.edges.filter(edge => edge.resource)).toEqual([
      {from: 'write', to: 'read', kind: 'raw', resource: 'texture'},
      {from: 'write', to: 'rewrite', kind: 'waw', resource: 'texture'},
      {from: 'read', to: 'rewrite', kind: 'war', resource: 'texture'}
    ]);
    expect(description.resources![0]).toMatchObject({
      bytes: 80,
      allocationBytes: 80,
      retained: true,
      format: 'rgba8unorm',
      attrs: {width: 4, height: 4, depth: 1, mipLevels: 2, dimension: '2d'}
    });
    expect(description.resources![0].allocation).not.toBe(description.resources![1].allocation);
    expect(description.resources![2]).toMatchObject({
      kind: 'external-texture',
      ownership: 'imported',
      firstNode: 'sample',
      lastNode: 'sample'
    });
  });

  test('merges nested annotations and deduplicates sources in schedule order', () => {
    const graph = createGraph();
    const source = {
      id: 'wgsl:hash',
      language: 'wgsl' as const,
      moduleId: 'kernel',
      hash: 'hash',
      text: 'fn main() {}'
    };
    graph.withRecordScope({namespace: 'a', derivedFrom: ['intent:x']}, () => {
      graph.withRecordScope({namespace: 'b', loop: {id: 'l', iteration: 2}}, () => {
        for (const id of ['first', 'second'])
          graph.addComputePass(
            setGPUComputeDispatchWorkgroups(
              {
                id,
                annotation: {
                  source,
                  dedupName: 'kernel',
                  formula: 'formula',
                  workgroupSize: [64, 1, 1] as const,
                  bindings: [],
                  datum: {keySpace: 'rows'}
                },
                workload: {operation: 'scan', commandCount: 1},
                compile: () => ({encode: () => {}})
              },
              [2, 1, 1]
            )
          );
      });
    });
    const description = compileGraph(graph).describe();
    expect(description.sources).toEqual([source]);
    expect(description.graph.nodes[0]).toMatchObject({
      label: 'kernel',
      namespace: 'a/b',
      derivedFrom: ['intent:x'],
      loop: {id: 'l', iteration: 2},
      dispatch: {workgroups: [2, 1, 1]},
      op: 'scan',
      source: 'wgsl:hash',
      formula: 'formula',
      bindings: [],
      datum: {keySpace: 'rows'}
    });
  });

  test('rejects a source id reused with different text', () => {
    const graph = createGraph();
    for (const id of ['first', 'second'])
      graph.addCopyPass({
        id,
        annotation: {
          source: {id: 'same', language: 'wgsl', moduleId: 'kernel', hash: 'hash', text: id}
        },
        compile: () => ({encode: () => {}})
      });
    expect(() => compileGraph(graph).describe()).toThrow('conflicting text');
  });

  test.each([
    false,
    true
  ])('inserts stable hazard-correct taps, including without recording (%s)', recording => {
    const graph = createGraph(undefined, recording);
    const source = createBuffer(graph, 'source');
    const sink = createBuffer(graph, 'sink');
    addNode(graph, 'writer', [{buffer: source, usage: 'copy-destination'}]);
    addNode(graph, 'later-writer', [{buffer: source, usage: 'copy-destination'}]);
    const sourceView = graph.createDataView(source, {format: 'uint32', byteOffset: 4, length: 2});
    const sinkView = graph.createDataView(sink, {format: 'uint32', byteOffset: 8, length: 2});
    const tapId = graph.addObservationTap(sourceView, {afterNode: 'writer', sink: sinkView});
    graph.addObservationTap(sourceView, {afterNode: 'writer', sink: sinkView, id: 'second-tap'});
    const compiled = compileGraph(graph);
    const description = compiled.describe().graph;
    expect(compiled.stats.nodeOrder).toEqual(['writer', tapId, 'second-tap', 'later-writer']);
    expect(description.nodes[1].attrs['observationTap']).toBe(true);
    expect(description.attrs).toEqual({taps: 2});
    expect(description.edges).toContainEqual({
      from: 'writer',
      to: tapId,
      kind: 'raw',
      resource: 'source'
    });
    expect(description.edges).toContainEqual({
      from: tapId,
      to: 'later-writer',
      kind: 'war',
      resource: 'source'
    });
    const encoder = graph.device.createCommandEncoder();
    cleanups.push(() => encoder.destroy());
    const copy = vi.spyOn(encoder, 'copyBufferToBuffer');
    compiled.encode(encoder, {parameters: undefined});
    expect(copy).toHaveBeenCalledWith({
      sourceBuffer: compiled.getPhysicalBuffer(source),
      sourceOffset: 4,
      destinationBuffer: compiled.getPhysicalBuffer(sink),
      destinationOffset: 8,
      size: 8
    });
  });

  test('validates taps without reserving failed ids', () => {
    const graph = createGraph();
    const source = createBuffer(graph, 'source');
    const sink = createBuffer(graph, 'sink');
    addNode(graph, 'writer', []);
    expect(() => graph.addObservationTap(source, {afterNode: 'missing', sink})).toThrow(
      'unknown afterNode'
    );
    const small = graph.createDataView(sink, {format: 'uint8', length: 3});
    expect(() =>
      graph.addObservationTap(source, {afterNode: 'writer', sink: small, id: 'invalid'})
    ).toThrow('invalid');
    expect(() =>
      graph.addObservationTap(small, {afterNode: 'writer', sink, id: 'invalid'})
    ).toThrow('invalid');
    const unaligned = graph.createDataView(source, {format: 'uint8', byteOffset: 1, length: 4});
    expect(() =>
      graph.addObservationTap(unaligned, {afterNode: 'writer', sink, id: 'invalid'})
    ).toThrow('invalid');
    graph.addObservationTap(source, {afterNode: 'writer', sink, id: 'invalid'});
    expect(() =>
      graph.addObservationTap(source, {afterNode: 'writer', sink, id: 'invalid'})
    ).toThrow('already in use');
  });

  test('resolves physical imports from overrides before defaults', () => {
    const graph = createGraph();
    const defaults = graph.device.createBuffer({byteLength: 16, usage: Buffer.COPY_SRC});
    const replacement = graph.device.createBuffer({byteLength: 16, usage: Buffer.COPY_SRC});
    cleanups.push(
      () => defaults.destroy(),
      () => replacement.destroy()
    );
    const handle = graph.importBuffer(
      {id: 'input', byteLength: 16, usage: Buffer.COPY_SRC},
      defaults
    );
    const compiled = compileGraph(graph);
    expect(compiled.getPhysicalBuffer(handle)).toBe(defaults);
    expect(compiled.getPhysicalBuffer(handle, {input: replacement})).toBe(replacement);
  });
});
