// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUComputationRecord,
  GPUComputationMetrics,
  GPUComputationCaptures
} from '../../src/computation-record';

/** Hand-built structure with two graph forms, unrolling, aliasing and semantic metadata. */
export const computationRecordFixture: GPUComputationRecord = {
  schema: 'luma.computation/1',
  id: 'relaxation',
  runId: 'run-fixture',
  createdAt: '2026-10-14T00:00:00.000Z',
  adapter: {backend: 'webgpu', vendor: 'test', architecture: 'test', variant: 'portable'},
  domains: ['gpu-core'],
  graphs: [
    {
      id: 'intent',
      form: 'intent',
      domain: 'gpu-core',
      nodes: [
        {
          id: 'prepare',
          label: 'Prepare',
          namespace: '',
          kind: 'operation',
          op: 'prepare',
          reads: [],
          writes: [],
          attrs: {}
        },
        {
          id: 'relax',
          label: 'Relax',
          namespace: 'solve',
          kind: 'operation',
          op: 'relax',
          reads: [],
          writes: [],
          attrs: {}
        },
        {
          id: 'repeat',
          label: 'Repeat',
          namespace: 'solve',
          kind: 'loop',
          reads: [],
          writes: [],
          attrs: {maximumIterations: 2}
        }
      ],
      edges: [{from: 'prepare', to: 'relax', kind: 'dataflow'}]
    },
    {
      id: 'compiled',
      form: 'compiled',
      domain: 'gpu-core',
      nodes: [
        {
          id: 'init',
          label: 'Initialize',
          namespace: '',
          kind: 'compute',
          op: 'prepare',
          derivedFrom: ['intent:prepare'],
          source: 'wgsl:fixture',
          reads: [],
          writes: ['first'],
          attrs: {}
        },
        {
          id: 'step:0',
          label: 'Relax 0',
          namespace: 'solve/body',
          kind: 'compute',
          op: 'relax',
          loop: {id: 'repeat', iteration: 0, maximumIterations: 2},
          dedupName: 'relax',
          derivedFrom: ['intent:relax', 'intent:repeat'],
          source: 'wgsl:fixture',
          formula: 'update',
          datum: {keySpace: 'row', relation: 'identity'},
          bindings: [{name: 'values', resource: 'first', access: 'read_write'}],
          reads: ['first'],
          writes: ['first'],
          workload: {maximumInvocationCount: 4},
          attrs: {variant: 'portable'}
        },
        {
          id: 'step:1',
          label: 'Relax 1',
          namespace: 'solve/body',
          kind: 'compute',
          op: 'relax',
          loop: {id: 'repeat', iteration: 1, maximumIterations: 2},
          dedupName: 'relax',
          derivedFrom: ['intent:relax', 'intent:repeat'],
          source: 'wgsl:fixture',
          reads: [],
          writes: ['second'],
          attrs: {}
        },
        {
          id: 'finish',
          label: 'Finish',
          namespace: 'solve',
          kind: 'compute',
          derivedFrom: ['intent:relax'],
          reads: ['second'],
          writes: ['second', 'output'],
          attrs: {}
        },
        {
          id: 'copy',
          label: 'Copy',
          namespace: '',
          kind: 'copy',
          reads: ['output'],
          writes: [],
          attrs: {}
        }
      ],
      edges: [
        {from: 'init', to: 'step:0', kind: 'raw', resource: 'first'},
        {from: 'step:0', to: 'step:1', kind: 'war', resource: 'first'},
        {from: 'step:1', to: 'finish', kind: 'waw', resource: 'second'},
        {from: 'finish', to: 'copy', kind: 'explicit'}
      ],
      resources: [
        {
          id: 'first',
          kind: 'buffer',
          bytes: 16,
          ownership: 'transient',
          allocation: 'scratch',
          firstNode: 'init',
          lastNode: 'step:0'
        },
        {
          id: 'second',
          kind: 'buffer',
          bytes: 16,
          ownership: 'transient',
          allocation: 'scratch',
          firstNode: 'step:1',
          lastNode: 'finish'
        },
        {
          id: 'output',
          kind: 'buffer',
          bytes: 16,
          ownership: 'imported',
          firstNode: 'finish',
          lastNode: 'copy'
        }
      ]
    }
  ],
  axes: {
    axes: [
      {id: 'lane', label: 'Lane', kind: 'execution', parent: 'workgroup', keySpace: 'invocation'},
      {id: 'workgroup', label: 'Workgroup', kind: 'execution'},
      {id: 'iteration', label: 'Iteration', kind: 'time'}
    ],
    keySpaces: [
      {id: 'invocation', label: 'Invocation', size: 4},
      {id: 'row', label: 'Row', size: 4}
    ],
    relations: [
      {
        id: 'identity',
        from: 'invocation',
        to: 'row',
        cardinality: '1:1',
        kind: 'function',
        ref: {function: 'identity'},
        timeDependent: false
      },
      {
        id: 'frontier',
        from: 'row',
        to: 'row',
        cardinality: 'N:M',
        kind: 'unknown',
        timeDependent: true
      }
    ]
  },
  refs: {
    sources: [
      {
        id: 'wgsl:fixture',
        language: 'wgsl',
        moduleId: 'relax',
        hash: 'fixture',
        text: 'fn main() {}'
      }
    ],
    formulas: [
      {
        id: 'update',
        tex: 'x',
        ascii: 'x',
        derivation: 'curated',
        symbols: [{id: 'x', role: 'input', binding: 'resource:first'}]
      }
    ]
  },
  decisions: [
    {
      id: 'strategy',
      subject: 'intent:relax',
      kind: 'strategy',
      chosen: 'portable',
      candidates: [
        {id: 'portable', supported: true, score: 2, reason: 'Fits workload'},
        {id: 'wide', supported: true, score: 1, reason: 'More padding'},
        {id: 'subgroups', supported: false, reason: 'Missing feature', failingFeature: 'subgroups'}
      ]
    }
  ],
  choices: [
    {stage: 'solve', name: 'iterations', value: 2, alternatives: [1, 2, 4], sweepable: true},
    {stage: 'input', name: 'weights', value: 'uniform', sweepable: false}
  ],
  workflow: {id: 'solve', invocations: [{graph: 'compiled', reason: 'Solve input'}]}
};

/** Measurements with explicit observer effects and an iteration counter series. */
export const computationMetricsFixture: GPUComputationMetrics = {
  schema: 'luma.computation-metrics/1',
  runId: 'run-fixture',
  graph: 'compiled',
  timebase: 'ns',
  encoding: {coalesced: false, timestamped: true, taps: 1, lensTwin: true, retained: ['output']},
  nodes: {
    init: {cpuEncodeMs: 0.1, gpuBegin: 0, gpuEnd: 1000, passIndex: 0},
    'step:0': {cpuEncodeMs: 0.2, gpuBegin: 2000, gpuEnd: 4000, passIndex: 1, outcome: 'executed'},
    'step:1': {cpuEncodeMs: 0.2, gpuBegin: 4000, gpuEnd: 6000, passIndex: 2, outcome: 'indirect'},
    finish: {cpuEncodeMs: 0.1, gpuBegin: 7000, gpuEnd: 8000, passIndex: 3},
    copy: {cpuEncodeMs: 0.1}
  },
  series: [{name: 'residual', axis: 'iteration', values: [0.5, 0.1]}]
};

/** One resource snapshot and empty stamp/replay collections. */
export const computationCapturesFixture: GPUComputationCaptures = {
  schema: 'luma.computation-captures/1',
  runId: 'run-fixture',
  snapshots: [
    {
      id: 'result',
      name: 'result',
      graph: 'compiled',
      resource: 'output',
      afterNode: 'copy',
      keySpace: 'row',
      elementType: 'f32',
      components: 2,
      values: [1, 2, 3, 4],
      provenance: 'gpu-readback'
    }
  ],
  stamps: [],
  replays: []
};
