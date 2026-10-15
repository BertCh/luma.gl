// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  printGPUComputationGraph,
  printGPUComputationRecord,
  toChromeTraceEvents,
  toGraphvizDOT,
  toModelExplorerGraph
} from '../../src/computation-record';
import {
  computationRecordFixture as record,
  computationMetricsFixture as metrics
} from './computation-record-fixture';

const intentText = `graph intent form=intent domain=gpu-core nodes=3 edges=1
  [operation] prepare ns=- op=prepare loop=-
    [operation] relax ns=solve op=relax loop=-
    [loop] repeat ns=solve op=- loop=-
edges:
  prepare -dataflow-> relax`;
const compiledText = `graph compiled form=compiled domain=gpu-core nodes=5 edges=4
  [compute] init ns=- op=prepare loop=- w=first <- intent:prepare
      [compute] step:0 ns=solve/body op=relax loop=repeat#0 r=first w=first <- intent:relax,intent:repeat
      [compute] step:1 ns=solve/body op=relax loop=repeat#1 w=second <- intent:relax,intent:repeat
    [compute] finish ns=solve op=- loop=- r=second w=second,output <- intent:relax
  [copy] copy ns=- op=- loop=- r=output
edges:
  init -raw[first]-> step:0
  step:0 -war[first]-> step:1
  step:1 -waw[second]-> finish
  finish -explicit-> copy
resources:
  first transient 16B alloc=scratch life=init..step:0
  second transient 16B alloc=scratch life=step:1..finish
  output imported 16B alloc=- life=finish..copy`;

describe('computation record text exports', () => {
  it('prints graph forms with stable indentation and omission rules', () => {
    expect(printGPUComputationGraph(record.graphs[0])).toBe(intentText);
    expect(printGPUComputationGraph(record.graphs[1])).toBe(compiledText);
  });
  it('prints the complete record and semantic tables', () => {
    expect(
      printGPUComputationRecord(record)
    ).toBe(`record relaxation run=run-fixture adapter=webgpu/portable domains=gpu-core
${intentText}
${compiledText}
decisions:
  strategy chosen=portable
    id supported score reason
    portable true 2 Fits workload
    wide true 1 More padding
    subgroups false - Missing feature
choices:
  solve.iterations=2
  input.weights="uniform"
axes:
  lane kind=execution parent=workgroup keySpace=invocation
  workgroup kind=execution parent=- keySpace=-
  iteration kind=time parent=- keySpace=-
  keySpace invocation label="Invocation" size=4
  keySpace row label="Row" size=4
relations:
  identity invocation->row 1:1 kind=function timeDependent=false
  frontier row->row N:M kind=unknown timeDependent=true`);
  });
  it('prints deterministic nested monochrome DOT', () => {
    expect(toGraphvizDOT(record)).toBe(`digraph "compiled" {
  "init" [label="Initialize"];
  "copy" [label="Copy"];
  subgraph cluster_0 {
    label="solve";
    "finish" [label="Finish"];
    subgraph cluster_1 {
      label="body";
      "step:0" [label="Relax 0"];
      "step:1" [label="Relax 1"];
    }
  }
  "init" -> "step:0" [style=solid, label="first"];
  "step:0" -> "step:1" [style=dashed, label="first"];
  "step:1" -> "finish" [style=bold, label="second"];
  "finish" -> "copy" [style=dotted];
}`);
    expect(toGraphvizDOT(record, {graph: 'intent'})).toContain(
      '"prepare" -> "relax" [style=solid]'
    );
    const resourceDot = toGraphvizDOT(record, {includeResources: true});
    expect(resourceDot).toContain('"resource_0" [shape=note, label="first"]');
    expect(resourceDot).toContain('"resource_0" -> "step:0" [label="read", color=gray]');
    expect(resourceDot).toContain('"init" -> "resource_0" [label="write", color=gray]');
  });
  it('escapes DOT labels, handles indirect edges and avoids resource id collisions', () => {
    const graph = record.graphs[1];
    const changed = {
      ...record,
      graphs: [
        {
          ...graph,
          nodes: [
            ...graph.nodes,
            {...graph.nodes[0], id: 'resource_0', label: 'Quote " slash \\ newline\n'}
          ],
          edges: [
            ...graph.edges,
            {from: 'init', to: 'step:1', kind: 'indirect' as const, resource: 'first'}
          ]
        }
      ]
    };
    const dot = toGraphvizDOT(changed, {includeResources: true});
    expect(dot).toContain('label="Quote \\" slash \\\\ newline\\n"');
    expect(dot).toContain('style=dashed, arrowhead=odiamond');
    expect(dot).toContain('"_resource_0" [shape=note');
    expect(toGraphvizDOT({...record, graphs: []})).toBe('digraph computation {\n}');
  });
});

describe('computation record JSON exports', () => {
  it('exports Chrome thread metadata, microseconds, counters and RAW flows', () => {
    const trace = toChromeTraceEvents(record, [metrics]);
    expect(trace.displayTimeUnit).toBe('ns');
    expect(trace.metadata).toEqual({
      schema: record.schema,
      runId: record.runId,
      adapter: record.adapter,
      encoding: metrics.encoding
    });
    expect(trace.traceEvents.filter(event => event.ph === 'M')).toEqual([
      {name: 'process_name', ph: 'M', ts: 0, pid: 1, tid: 0, args: {name: 'relaxation'}},
      ...[0, 1, 2].map(level => ({
        name: 'thread_name',
        ph: 'M',
        ts: 0,
        pid: 1,
        tid: level,
        args: {name: `level ${level}`}
      }))
    ]);
    const durations = trace.traceEvents.filter(event => event.ph === 'X');
    expect(durations).toHaveLength(4);
    expect(durations[1]).toEqual({
      name: 'Relax 0',
      ph: 'X',
      ts: 2,
      dur: 2,
      pid: 1,
      tid: 2,
      cat: 'compute',
      args: {
        nodeId: 'step:0',
        graph: 'compiled',
        namespace: 'solve/body',
        op: 'relax',
        passIndex: 1
      }
    });
    expect(trace.traceEvents.filter(event => event.ph === 'C')).toEqual([
      {name: 'residual', ph: 'C', ts: 2, pid: 1, tid: 0, args: {residual: 0.5}},
      {name: 'residual', ph: 'C', ts: 4, pid: 1, tid: 0, args: {residual: 0.1}}
    ]);
    expect(trace.traceEvents.filter(event => event.ph === 's' || event.ph === 'f')).toEqual([
      {name: 'raw', ph: 's', ts: 1, pid: 1, tid: 0, id: 0, cat: 'raw'},
      {name: 'raw', ph: 'f', ts: 2, pid: 1, tid: 2, id: 0, cat: 'raw', bp: 'e'}
    ]);
    expect(
      toChromeTraceEvents(record, [metrics], {flows: false}).traceEvents.some(
        event => event.ph === 's' || event.ph === 'f'
      )
    ).toBe(false);
  });
  it('handles untimed counters, milliseconds and multiple metric sets', () => {
    const changed = {
      ...metrics,
      timebase: 'ms' as const,
      series: [{name: 'count', axis: 'unknown', values: [3, 4]}]
    };
    const trace = toChromeTraceEvents(record, [changed]);
    expect(trace.traceEvents.find(event => event.ph === 'X')?.dur).toBe(1000000);
    expect(trace.traceEvents.filter(event => event.ph === 'C').map(event => event.ts)).toEqual([
      0, 0
    ]);
    const multiple = toChromeTraceEvents(record, [metrics, changed]);
    expect(multiple.metadata['encoding']).toEqual([metrics.encoding, changed.encoding]);
    expect(multiple.traceEvents.filter(event => event.ph === 's').map(event => event.id)).toEqual([
      0, 1
    ]);
    expect(toChromeTraceEvents(record, []).metadata['encoding']).toEqual([]);
  });
  it('exports Model Explorer forms, incoming edges and string attributes', () => {
    const collection = toModelExplorerGraph(record);
    expect(collection.label).toBe(record.id);
    expect(collection.graphs.map(graph => graph.id)).toEqual(['intent', 'compiled']);
    const node = collection.graphs[1].nodes[1];
    expect(node).toMatchObject({
      id: 'step:0',
      label: 'Relax 0',
      namespace: 'solve/body',
      incomingEdges: [
        {sourceNodeId: 'init', sourceNodeOutputId: '0', targetNodeInputId: 'raw:first'}
      ]
    });
    expect(node.attrs).toEqual([
      {key: 'kind', value: 'compute'},
      {key: 'op', value: 'relax'},
      {key: 'form', value: 'compiled'},
      {key: 'derivedFrom', value: 'intent:relax,intent:repeat'},
      {key: 'loop', value: 'repeat#0'},
      {key: 'reads', value: 'first'},
      {key: 'writes', value: 'first'},
      {key: 'workload.maximumInvocationCount', value: '4'},
      {key: 'variant', value: 'portable'}
    ]);
  });
  it.each([
    ['graph text', () => printGPUComputationGraph(record.graphs[1])],
    ['record text', () => printGPUComputationRecord(record)],
    ['DOT', () => toGraphvizDOT(record)],
    ['DOT resources', () => toGraphvizDOT(record, {includeResources: true})],
    ['Chrome trace', () => toChromeTraceEvents(record, [metrics])],
    ['Model Explorer', () => toModelExplorerGraph(record)]
  ] as const)('%s is deterministic and has no undefined values', (_name, exportRecord) => {
    const output = exportRecord();
    expect(exportRecord()).toEqual(output);
    const visit = (value: unknown): void => {
      expect(value).not.toBeUndefined();
      if (value && typeof value === 'object') Object.values(value).forEach(visit);
    };
    visit(output);
    expect(JSON.stringify(output)).not.toContain('undefined');
  });
});
