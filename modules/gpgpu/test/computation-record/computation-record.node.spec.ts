// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  createGPUComputationRecord,
  getGPUComputationRunId,
  getGPUComputationAdapter,
  parseGPUComputationNodeRef,
  formatGPUComputationNodeRef,
  createGPUComputationMetrics,
  createGPUComputationCaptures,
  validateGPUComputationRecord,
  validateGPUComputationSidecars
} from '../../src/computation-record';
import {
  computationRecordFixture as record,
  computationMetricsFixture as metrics,
  computationCapturesFixture as captures
} from './computation-record-fixture';

function replacePath<Value>(original: Value, path: string, value: unknown): Value {
  const copy = structuredClone(original);
  const segments = path.replace(/\[(\d+)\]/g, '.$1').split('.');
  let parent = copy as Record<string, unknown>;
  for (const segment of segments.slice(0, -1)) parent = parent[segment] as Record<string, unknown>;
  parent[segments[segments.length - 1]] = value;
  return copy;
}

describe('computation record validation', () => {
  it('accepts the complete fixture and its sidecars', () => {
    expect(validateGPUComputationRecord(record)).toEqual({valid: true, issues: []});
    expect(validateGPUComputationSidecars(record, {metrics: [metrics], captures})).toEqual({
      valid: true,
      issues: []
    });
  });
  it.each([
    ['schema', 'other'],
    ['id', undefined],
    ['runId', 4],
    ['createdAt', ''],
    ['adapter.backend', undefined],
    ['domains', {}],
    ['domains[0]', 1],
    ['graphs', null],
    ['graphs[1].id', 'intent'],
    ['graphs[1].id', 'bad:id'],
    ['graphs[1].domain', undefined],
    ['graphs[1].form', 'invalid'],
    ['graphs[1].nodes[1].id', 'init'],
    ['graphs[1].nodes[0].label', undefined],
    ['graphs[1].nodes[0].namespace', null],
    ['graphs[1].nodes[0].kind', undefined],
    ['graphs[1].nodes[0].reads', null],
    ['graphs[1].nodes[0].writes', {}],
    ['graphs[1].nodes[0].attrs', []],
    ['graphs[1].edges[0].from', 'missing'],
    ['graphs[1].edges[0].to', 'missing'],
    ['graphs[1].edges[0].kind', 'invalid'],
    ['graphs[1].nodes[0].derivedFrom[0]', 'invalid'],
    ['graphs[1].nodes[0].derivedFrom[0]', ':empty'],
    ['graphs[1].nodes[0].derivedFrom[0]', 'intent:'],
    ['graphs[1].nodes[0].derivedFrom[0]', 'absent:node'],
    ['graphs[1].nodes[0].derivedFrom[0]', 'intent:absent'],
    ['graphs[1].nodes[0].derivedFrom[0]', 'compiled:step:0'],
    ['graphs[1].nodes[0].source', 'missing'],
    ['graphs[1].nodes[1].formula', 'missing'],
    ['graphs[1].nodes[1].reads[0]', 'missing'],
    ['graphs[1].nodes[0].writes[0]', 'missing'],
    ['graphs[1].nodes[1].bindings[0].resource', 'missing'],
    ['graphs[1].resources[0].firstNode', 'missing'],
    ['graphs[1].resources[0].lastNode', 'missing'],
    ['graphs[1].resources[1].id', 'first'],
    ['graphs[1].nodes[1].loop.iteration', -1],
    ['graphs[1].nodes[1].loop.iteration', 0.5],
    ['graphs[1].nodes[1].loop.iteration', 2],
    ['graphs[1].nodes[1].loop.maximumIterations', -1],
    ['axes.axes[1].id', 'lane'],
    ['axes.keySpaces[1].id', 'invocation'],
    ['axes.relations[1].id', 'identity'],
    ['axes.axes[0].parent', 'missing'],
    ['axes.axes[0].keySpace', 'missing'],
    ['axes.relations[0].from', 'missing'],
    ['axes.relations[0].to', 'missing'],
    ['axes.relations[0].ref.function', undefined],
    ['graphs[1].nodes[1].datum.keySpace', 'missing'],
    ['graphs[1].nodes[1].datum.relation', 'missing'],
    ['decisions[0].chosen', 'missing'],
    ['workflow.invocations[0].graph', 'missing'],
    ['refs.sources[0].moduleId', undefined],
    ['refs.formulas[0].ascii', undefined],
    ['choices[0].stage', undefined]
  ])('reports the invalid field %s (%j)', (path, value) => {
    const report = validateGPUComputationRecord(replacePath(record, path, value));
    expect(report.valid).toBe(false);
    expect(report.issues).toContainEqual(expect.objectContaining({level: 'error', path}));
  });
  it.each([
    null,
    undefined,
    1,
    'text',
    [],
    {},
    {graphs: [null]}
  ])('never throws for malformed JSON %j', value => {
    expect(() => validateGPUComputationRecord(value)).not.toThrow();
    expect(validateGPUComputationRecord(value).valid).toBe(false);
  });
  it('rejects references to later forms', () => {
    const changed = replacePath(record, 'graphs[0].nodes[0].derivedFrom', ['compiled:init']);
    expect(validateGPUComputationRecord(changed).issues).toContainEqual(
      expect.objectContaining({path: 'graphs[0].nodes[0].derivedFrom[0]'})
    );
  });
  it('checks required strings in optional node metadata', () => {
    for (const [field, metadata, required] of [
      ['condition', {source: 'gpu', mode: 'indirect'}, 'id'],
      ['publication', {id: 'result'}, 'completeness']
    ] as const) {
      const changed = replacePath(record, `graphs[1].nodes[0].${field}`, metadata);
      expect(validateGPUComputationRecord(changed).issues).toContainEqual(
        expect.objectContaining({path: `graphs[1].nodes[0].${field}.${required}`})
      );
    }
  });
  it('checks inclusive alias lifetimes and their order', () => {
    const overlap = replacePath(record, 'graphs[1].resources[1].firstNode', 'step:0');
    expect(validateGPUComputationRecord(overlap).issues).toContainEqual(
      expect.objectContaining({path: 'graphs[1].resources[1].allocation'})
    );
    expect(
      validateGPUComputationRecord(replacePath(overlap, 'graphs[1].resources[1].retained', true))
        .valid
    ).toBe(true);
    expect(
      validateGPUComputationRecord(replacePath(record, 'graphs[1].resources[0].lastNode', 'init'))
        .valid
    ).toBe(true);
    expect(
      validateGPUComputationRecord(replacePath(record, 'graphs[1].resources[1].lastNode', 'init'))
        .issues
    ).toContainEqual(expect.objectContaining({path: 'graphs[1].resources[1].lastNode'}));
  });
  it.each(['sources', 'formulas'] as const)('rejects duplicate refs in %s', table => {
    const changed = replacePath(record, `refs.${table}`, [
      record.refs[table][0],
      record.refs[table][0]
    ]);
    expect(validateGPUComputationRecord(changed).issues).toContainEqual(
      expect.objectContaining({path: `refs.${table}[1].id`})
    );
  });
  it.each([
    ['graphs[1].resources', undefined],
    ['refs.formulas[0].symbols[0].binding', undefined],
    ['axes.relations[1].ref', {function: 'identity'}]
  ])('warns without failing for %s', (path, value) => {
    const report = validateGPUComputationRecord(replacePath(record, path, value));
    expect(report.valid).toBe(true);
    expect(report.issues).toContainEqual(expect.objectContaining({level: 'warning', path}));
  });
});

describe('record construction and references', () => {
  const props = {
    id: record.id,
    adapter: record.adapter,
    domains: record.domains,
    graphs: record.graphs
  };
  it('fills defaults and preserves explicit identity', () => {
    const built = createGPUComputationRecord({...props, graphs: []});
    expect(built.axes).toEqual({axes: [], keySpaces: [], relations: []});
    expect(built.refs).toEqual({formulas: [], sources: []});
    expect(built.decisions).toEqual([]);
    expect(built.choices).toEqual([]);
    expect(built).not.toHaveProperty('workflow');
    expect(Number.isNaN(Date.parse(built.createdAt))).toBe(false);
    expect(built.runId).toMatch(/^run-[a-f0-9]{8}$/);
    expect(validateGPUComputationRecord(built).valid).toBe(true);
    expect(
      createGPUComputationRecord({...props, runId: 'explicit', createdAt: 'today'})
    ).toMatchObject({runId: 'explicit', createdAt: 'today'});
  });
  it('deduplicates graph and supplied sources, retains available text, rejects conflicts', () => {
    const source = record.refs.sources[0];
    const {text: sourceText, ...withoutText} = source;
    const built = createGPUComputationRecord({
      ...props,
      sources: [withoutText],
      graphs: [{graph: record.graphs[1], sources: [source, withoutText]}]
    });
    expect(built.refs.sources).toEqual([source]);
    expect(built.graphs).toEqual([record.graphs[1]]);
    expect(() =>
      createGPUComputationRecord({
        ...props,
        sources: [source, {...source, text: `${sourceText} different`}]
      })
    ).toThrow('Conflicting source text');
  });
  it('hashes structure, adapter, dataset and typed seed deterministically', () => {
    const runId = getGPUComputationRunId(props);
    expect(getGPUComputationRunId(props)).toBe(runId);
    expect(getGPUComputationRunId({...props, graphs: [...props.graphs].reverse()})).not.toBe(runId);
    expect(getGPUComputationRunId({...props, graphs: [record.graphs[0]]})).not.toBe(runId);
    expect(
      getGPUComputationRunId({
        ...props,
        graphs: [
          {...record.graphs[0], nodes: [...record.graphs[0].nodes].reverse()},
          record.graphs[1]
        ]
      })
    ).not.toBe(runId);
    expect(
      getGPUComputationRunId({...props, adapter: {...props.adapter, backend: 'null'}})
    ).not.toBe(runId);
    expect(
      getGPUComputationRunId({...props, adapter: {...props.adapter, variant: 'other'}})
    ).not.toBe(runId);
    expect(getGPUComputationRunId({...props, datasetVersion: '2'})).not.toBe(runId);
    expect(getGPUComputationRunId({...props, seed: 1})).not.toBe(runId);
    expect(getGPUComputationRunId({...props, seed: 1})).not.toBe(
      getGPUComputationRunId({...props, seed: '1'})
    );
  });
  it('reads actual DeviceInfo field names without requiring a Device', () => {
    expect(
      getGPUComputationAdapter(
        {
          type: 'webgpu',
          info: {vendor: 'vendor', gpu: 'apple', gpuArchitecture: 'common-3', renderer: 'renderer'}
        },
        'portable'
      )
    ).toEqual({
      backend: 'webgpu',
      vendor: 'vendor',
      architecture: 'common-3',
      description: 'renderer',
      variant: 'portable'
    });
    expect(getGPUComputationAdapter({type: 'null', info: {}})).toEqual({backend: 'null'});
    expect(getGPUComputationAdapter({type: 'webgpu', info: {gpu: 'apple'}}).architecture).toBe(
      'apple'
    );
  });
  it('splits node refs only at the first colon', () => {
    expect(formatGPUComputationNodeRef('compiled', 'step:0')).toBe('compiled:step:0');
    expect(parseGPUComputationNodeRef('compiled:step:0')).toEqual({
      graph: 'compiled',
      node: 'step:0'
    });
    for (const invalid of ['', 'missing', ':node', 'graph:', ':'])
      expect(() => parseGPUComputationNodeRef(invalid)).toThrow();
  });
});

describe('metrics and captures', () => {
  it('normalizes times, preserves CPU times and maps conditional outcomes', () => {
    const built = createGPUComputationMetrics({
      runId: record.runId,
      graph: 'compiled',
      encoding: metrics.encoding,
      adapter: record.adapter,
      series: metrics.series,
      counters: {init: {count: 4}},
      timing: {
        nodes: [
          {
            id: 'init',
            cpuEncodeTimeMilliseconds: 0.1,
            gpuBeginNanoseconds: 2000,
            gpuEndNanoseconds: 3000,
            passIndex: 0
          },
          {
            id: 'early',
            cpuEncodeTimeMilliseconds: 0.2,
            gpuBeginNanoseconds: 1000,
            gpuEndNanoseconds: 1500,
            condition: {outcome: 'executed'}
          },
          {id: 'skip', cpuEncodeTimeMilliseconds: 0, condition: {outcome: 'skipped'}},
          {id: 'gate', cpuEncodeTimeMilliseconds: 0, condition: {outcome: 'gpu-resolved'}},
          {id: 'indirect', cpuEncodeTimeMilliseconds: 0, condition: {outcome: 'indirect'}},
          {id: 'condition', cpuEncodeTimeMilliseconds: 0, condition: {}},
          {id: 'copy', cpuEncodeTimeMilliseconds: 0.1}
        ]
      }
    });
    expect(built.timebase).toBe('ns');
    expect(built.nodes['init']).toEqual({
      cpuEncodeMs: 0.1,
      gpuBegin: 1000,
      gpuEnd: 2000,
      passIndex: 0,
      counters: {count: 4}
    });
    expect(built.nodes['early']).toMatchObject({gpuBegin: 0, gpuEnd: 500, outcome: 'executed'});
    expect(built.nodes['skip']).toEqual({cpuEncodeMs: 0, outcome: 'skipped'});
    expect(built.nodes['gate'].outcome).toBe('indirect');
    expect(built.nodes['indirect'].outcome).toBe('indirect');
    expect(built.nodes['condition'].outcome).toBe('executed');
    expect(built.nodes['copy']).toEqual({cpuEncodeMs: 0.1});
    expect(built.adapter).toEqual(record.adapter);
    expect(built.series).toEqual(metrics.series);
    expect(
      createGPUComputationMetrics({
        runId: 'run',
        graph: 'graph',
        encoding: metrics.encoding,
        timing: {nodes: [{id: 'copy', cpuEncodeTimeMilliseconds: 1}]}
      }).nodes['copy']
    ).toEqual({cpuEncodeMs: 1});
    expect(createGPUComputationCaptures({runId: record.runId})).toEqual({
      schema: 'luma.computation-captures/1',
      runId: record.runId,
      snapshots: [],
      stamps: [],
      replays: []
    });
  });
  it.each([
    ['metrics[0].schema', 'other'],
    ['metrics[0].runId', 'other'],
    ['metrics[0].graph', 'missing'],
    ['metrics[0].nodes.init.gpuEnd', -1],
    ['captures.schema', 'other'],
    ['captures.runId', 'other'],
    ['captures.snapshots[0].graph', 'missing'],
    ['captures.snapshots[0].resource', 'missing'],
    ['captures.snapshots[0].afterNode', 'missing'],
    ['captures.snapshots[0].keySpace', 'missing'],
    ['captures.snapshots[0].values', [1, 2, 3]],
    ['captures.snapshots[0].components', 0]
  ])('reports sidecar error at %s', (path, value) => {
    const sidecars = replacePath({metrics: [metrics], captures}, path, value);
    const report = validateGPUComputationSidecars(record, sidecars);
    expect(report.valid).toBe(false);
    expect(report.issues).toContainEqual(
      expect.objectContaining({path: path.replace('.nodes.init.', '.nodes["init"].')})
    );
  });
  it('rejects unknown metrics node ids', () => {
    expect(
      validateGPUComputationSidecars(record, {
        metrics: [{...metrics, nodes: {missing: {cpuEncodeMs: 1}}}]
      }).issues
    ).toContainEqual(expect.objectContaining({path: 'metrics[0].nodes["missing"]'}));
  });
  it('accepts typed captures and resolves omitted graph within the record', () => {
    const typed = replacePath(captures, 'snapshots[0].values', new Float32Array([1, 2]));
    expect(
      validateGPUComputationSidecars(record, {
        captures: replacePath(typed, 'snapshots[0].graph', undefined)
      }).valid
    ).toBe(true);
    expect(validateGPUComputationSidecars(record, {}).valid).toBe(true);
  });
});
