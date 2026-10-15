// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUProgram,
  GPUProgramCompiler,
  GPUProgramScalarLiteral,
  GPUCompositeOperation,
  GPULoopOperation,
  GPUConditionalOperation
} from '@luma.gl/gpgpu/gpu-core';
import type {
  GPUComputationGraph,
  GPUComputationSourceRef
} from '../../src/computation-record/computation-record-types';
import {getGPUOperationTree} from '../../src/gpu-core/gpu-operation';
import {
  describeGPUProgramCompilation,
  getGPUProgramIntentGraphId
} from '../../src/gpu-core/gpu-program-describe';
import type {GPUCommandNode} from '../../src/gpu-core/gpu-command-node';
import type {GPUCommandGraph} from '../../src/gpu-core/gpu-command-graph';
import {NullDevice} from '@luma.gl/test-utils';
import {describe, expect, test, vi} from 'vitest';

// Inspect lowering annotations independently of the parallel compiled describe implementation.
function getNodes(graph: GPUCommandGraph): readonly GPUCommandNode[] {
  return (graph as unknown as {nodes: GPUCommandNode[]}).nodes;
}

function makeDevice(): NullDevice {
  const device = new NullDevice({id: 'program-description-device'});
  Object.defineProperty(device, 'type', {value: 'webgpu'});
  return device;
}

function makeProgram(): GPUProgram {
  const program = new GPUProgram({id: 'recorded-program'});
  const active = program.scalar('active', 'uint32');
  const output = program.scalar('output', 'float32');
  const write = Object.assign(new GPUProgramScalarLiteral({id: 'write', output, value: 2}), {
    metadata: {inputs: [{name: 'active'}], outputs: [{name: 'output'}], workload: {rows: 1}}
  });
  program.add(
    new GPUCompositeOperation({
      id: 'root',
      operations: [
        Object.assign(new GPUProgramScalarLiteral({id: 'initialize', output: active, value: 1}), {
          metadata: {outputs: [{name: 'active'}]}
        }),
        new GPULoopOperation({
          id: 'repeat',
          maximumIterations: 3,
          minimumIterations: 1,
          predicate: {id: 'is-active', source: 'gpu', value: active, expression: 'active != 0'},
          body: write
        }),
        new GPUProgramScalarLiteral({id: 'write', output, value: 3})
      ]
    })
  );
  return program;
}

describe('GPU program descriptions', () => {
  test('records intent hierarchy, metadata dependencies and chosen lowering rules', () => {
    const device = makeDevice();
    try {
      const compilation = new GPUProgramCompiler(device).compile(
        makeProgram(),
        {},
        {recording: {}}
      );
      expect(compilation.lowering.operations[0].children?.[1].loop).toEqual({
        maximumIterations: 3,
        minimumIterations: 1,
        lowering: 'auto',
        predicate: 'active != 0'
      });
      const description = describeGPUProgramCompilation(compilation, {domain: 'test-domain'});
      expect(getGPUProgramIntentGraphId(compilation.program)).toBe('recorded-program-intent');
      expect(description.graphs[0]).toMatchObject({
        form: 'intent',
        domain: 'test-domain',
        label: 'recorded-program'
      });
      expect(
        description.graphs[0].nodes.map(({id, namespace, kind}) => ({id, namespace, kind}))
      ).toEqual([
        {id: 'root', namespace: '', kind: 'composite'},
        {id: 'initialize', namespace: 'root', kind: 'operation'},
        {id: 'repeat', namespace: 'root', kind: 'loop'},
        {id: 'write', namespace: 'root/repeat', kind: 'operation'},
        {id: 'write#1', namespace: 'root', kind: 'operation'}
      ]);
      expect(description.graphs[0].nodes[3]).toMatchObject({
        op: 'scalar-literal',
        attrs: {rows: 1},
        reads: ['active'],
        writes: ['output']
      });
      expect(description.graphs[0].edges).toEqual([
        {from: 'initialize', to: 'write', kind: 'dataflow', resource: 'active'}
      ]);
      expect(description.decisions).toHaveLength(compilation.lowering.decisions.length);
      expect(description.decisions.find(decision => decision.id === 'lowering:repeat#0')).toEqual({
        id: 'lowering:repeat#0',
        subject: 'recorded-program-intent:repeat',
        kind: 'lowering',
        chosen: 'gpu-gated-bounded-sequence',
        reason: 'WebGPU has indirect dispatch but no native graph loop',
        method: 'rule',
        candidates: [
          {
            id: 'gpu-gated-bounded-sequence',
            supported: true,
            reason: 'WebGPU has indirect dispatch but no native graph loop'
          }
        ]
      });
      expect(description.decisions.at(-1)?.subject).toBe('recorded-program-intent:write#1');
      const graph: GPUComputationGraph = {
        id: 'recorded-program-compiled',
        form: 'compiled',
        domain: 'gpu-core',
        nodes: [],
        edges: []
      };
      const sources: readonly GPUComputationSourceRef[] = [];
      const compiled = {describe: vi.fn(() => ({graph, sources}))};
      expect(describeGPUProgramCompilation(compilation, {compiled})).toMatchObject({
        graphs: [{...description.graphs[0], domain: 'gpu-core'}, graph],
        sources
      });
      expect(compiled.describe).toHaveBeenCalledWith({graphId: 'recorded-program-compiled'});
    } finally {
      device.destroy();
    }
  });

  test('recording preserves command order and stamps loop copies, gates and duplicate names', () => {
    const device = makeDevice();
    try {
      const compiler = new GPUProgramCompiler(device);
      const program = makeProgram();
      const recorded = compiler.compile(program, {}, {recording: {}});
      const plain = compiler.compile(program);
      // Scheduling and allocation are real; only GPU pipeline creation is stubbed on NullDevice.
      for (const node of [...getNodes(recorded.graph), ...getNodes(plain.graph)]) {
        vi.spyOn(node, 'compile').mockReturnValue({encode: () => {}});
      }
      const recordedGraph = recorded.graph.compile();
      const plainGraph = plain.graph.compile();
      try {
        expect(recordedGraph.stats.nodeOrder).toEqual(plainGraph.stats.nodeOrder);
        expect(getNodes(plain.graph).every(node => node.annotation === undefined)).toBe(true);
        expect(
          getNodes(recorded.graph).find(node => node.id === 'write-iteration-1')?.annotation
        ).toMatchObject({
          namespace: 'root/repeat/iteration-1/write',
          derivedFrom: ['recorded-program-intent:write'],
          dedupName: 'write',
          loop: {id: 'repeat', iteration: 1, phase: 'gated', maximumIterations: 3}
        });
        expect(
          getNodes(recorded.graph).find(node => node.id === 'write')?.annotation?.loop?.phase
        ).toBe('unconditional');
        expect(
          getNodes(recorded.graph).filter(
            node => node.annotation?.attrs?.['role'] === 'predicate-gate'
          )
        ).toHaveLength(2);
        expect(getNodes(recorded.graph).at(-1)?.annotation?.derivedFrom).toEqual([
          'recorded-program-intent:write#1'
        ]);
      } finally {
        recordedGraph.destroy();
        plainGraph.destroy();
      }
    } finally {
      device.destroy();
    }
  });

  test('nested loops reuse duplicate intent ids and stamp predicate conjunctions', () => {
    const device = makeDevice();
    try {
      const program = new GPUProgram({id: 'nested'});
      const active = program.scalar('active', 'uint32');
      const output = program.scalar('output', 'float32');
      const predicate = {id: 'active', source: 'gpu' as const, value: active};
      program.add(
        new GPULoopOperation({
          id: 'loop',
          maximumIterations: 2,
          predicate,
          body: new GPULoopOperation({
            id: 'loop',
            maximumIterations: 2,
            predicate,
            body: [
              new GPUProgramScalarLiteral({id: 'write', output, value: 1}),
              new GPUProgramScalarLiteral({id: 'write', output, value: 2})
            ]
          })
        })
      );
      const compilation = new GPUProgramCompiler(device).compile(program, {}, {recording: {}});
      const writes = getNodes(compilation.graph).filter(
        node => node.annotation?.attrs?.['role'] === undefined
      );
      expect(writes).toHaveLength(8);
      expect(writes.map(node => node.annotation?.derivedFrom)).toEqual(
        Array.from({length: 4}, () => [['nested-intent:write'], ['nested-intent:write#1']]).flat()
      );
      expect(writes.at(-1)?.annotation).toMatchObject({
        namespace: 'loop/iteration-1/loop#1/iteration-1/write#1',
        loop: {id: 'loop#1', iteration: 1, phase: 'gated'}
      });
      expect(
        getNodes(compilation.graph).filter(
          node => node.annotation?.attrs?.['role'] === 'predicate-conjunction'
        )
      ).toHaveLength(8);
    } finally {
      device.destroy();
    }
  });

  test('describes conditionals, predicate ids, explicit unroll and producer intent hooks', () => {
    const device = makeDevice();
    try {
      const program = new GPUProgram({id: 'hooks'});
      const active = program.scalar('active', 'uint32');
      const predicate = {id: 'predicate-id', source: 'gpu' as const, value: active};
      const producer = {
        getNodes: () => [],
        describeIntent: () => ({id: 'custom', type: 'custom-intent'})
      };
      expect(getGPUOperationTree(producer)).toEqual({id: 'custom', type: 'custom-intent'});
      expect(getGPUOperationTree({getNodes: () => []}).type).toBe('command-node-producer');
      const conditional = new GPUConditionalOperation({id: 'condition', predicate, body: producer});
      expect(getGPUOperationTree(conditional).conditional).toEqual({
        lowering: 'auto',
        predicate: 'predicate-id'
      });
      program.add([
        conditional,
        new GPULoopOperation({
          id: 'unroll',
          predicate,
          lowering: 'unroll',
          maximumIterations: 2,
          body: new GPUProgramScalarLiteral({id: 'set', output: active, value: 1})
        }),
        {getNodes: () => []}
      ]);
      const compilation = new GPUProgramCompiler(device).compile(program, {}, {recording: {}});
      expect(getNodes(compilation.graph).map(node => node.annotation?.loop?.phase)).toEqual([
        'unconditional',
        'unconditional'
      ]);
      expect(describeGPUProgramCompilation(compilation).graphs[0].nodes.at(-1)).toMatchObject({
        kind: 'operation',
        attrs: {opaque: true}
      });
    } finally {
      device.destroy();
    }
  });
});
