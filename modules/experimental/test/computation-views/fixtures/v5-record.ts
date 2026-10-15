// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationNode, GPUComputationCapture} from '@luma.gl/gpgpu';
import type {
  LensRecord,
  ViewContext
} from '../../../src/computation-views/registry/view-registration';

/** A small dispatch node with explicit datum identity. */
export function makeNode(overrides: Partial<GPUComputationNode> = {}): GPUComputationNode {
  return {
    id: 'relax-0',
    label: 'Relax',
    namespace: 'solver/body',
    kind: 'compute',
    reads: ['state'],
    writes: ['state'],
    attrs: {},
    datum: {keySpace: 'rows'},
    loop: {id: 'solve', iteration: 0, maximumIterations: 16},
    dedupName: 'relax',
    source: 'shader',
    dispatch: {workgroups: [3, 1, 1], invocationCount: 10},
    workgroupSize: [4, 1, 1],
    bindings: [{name: 'values', resource: 'state', access: 'read_write', elementType: 'f32'}],
    ...overrides
  };
}

/** One captured scalar per row. */
export function makeCapture(overrides: Partial<GPUComputationCapture> = {}): GPUComputationCapture {
  return {
    id: 'state-0',
    name: 'state',
    iteration: 0,
    keySpace: 'rows',
    elementType: 'f32',
    values: [9, 8, 7],
    provenance: 'gpu-readback',
    ...overrides
  };
}

/** A deterministic recorded loop with readbacks, an id probe, and timing sidecar. */
export function makeRecord(): LensRecord {
  return {
    schema: 'luma.computation/1',
    id: 'v5',
    runId: 'run-v5',
    createdAt: '2026-10-14T00:00:00Z',
    adapter: {backend: 'webgpu', description: 'Test adapter'},
    domains: [],
    decisions: [],
    choices: [],
    graphs: [
      {
        id: 'compiled',
        label: 'Solver',
        form: 'compiled',
        domain: 'gpu-core',
        nodes: [
          makeNode(),
          makeNode({
            id: 'relax-1',
            loop: {id: 'solve', iteration: 1, maximumIterations: 16, phase: 'gated'}
          }),
          makeNode({
            id: 'finish',
            label: 'Finish',
            namespace: 'solver',
            loop: undefined,
            dedupName: undefined
          })
        ],
        edges: [
          {from: 'relax-0', to: 'relax-1', kind: 'raw'},
          {from: 'relax-1', to: 'finish', kind: 'explicit'}
        ]
      }
    ],
    axes: {
      axes: [{id: 'entity', label: 'Rows', kind: 'data', keySpace: 'rows'}],
      keySpaces: [{id: 'rows', label: 'Rows', size: 12}],
      relations: []
    },
    refs: {
      formulas: [],
      sources: [
        {
          id: 'shader',
          language: 'wgsl',
          moduleId: 'Relax',
          hash: 'abc123',
          entryPoint: 'main',
          text: '@group(0) @binding(0) var<storage, read_write> values: array<f32>;\n@compute @workgroup_size(4)\nfn main() {}'
        }
      ]
    },
    metrics: [
      {
        schema: 'luma.computation-metrics/1',
        runId: 'run-v5',
        graph: 'compiled',
        timebase: 'ns',
        encoding: {
          coalesced: false,
          timestamped: true,
          taps: 1,
          lensTwin: true,
          retained: ['state']
        },
        nodes: {
          'relax-0': {
            gpuBegin: 0,
            gpuEnd: 2000,
            cpuEncodeMs: 0.2,
            outcome: 'executed',
            passIndex: 0
          },
          'relax-1': {gpuBegin: 2000, gpuEnd: 5000, cpuEncodeMs: 0.3, passIndex: 1},
          finish: {cpuEncodeMs: 0.1, passIndex: 2}
        },
        series: [{name: 'residual', axis: 'iteration', values: [1000, 10, 1], unit: 'error'}]
      }
    ],
    captures: {
      schema: 'luma.computation-captures/1',
      runId: 'run-v5',
      snapshots: [
        ...[0, 1, 2, 3, 4, 8, 9].map(iteration =>
          makeCapture({
            id: `state-${iteration}`,
            iteration,
            values: [9 - iteration, 10 - iteration, 11 - iteration]
          })
        ),
        makeCapture({
          id: 'probe',
          name: 'id-probe',
          iteration: undefined,
          afterNode: 'relax-0',
          values: Array.from({length: 12}, (_, index) => index + 100)
        })
      ],
      stamps: [
        makeCapture({
          id: 'settled',
          name: 'settledAt',
          iteration: undefined,
          values: [2, 4, 8],
          meaning: 'settledAt round'
        })
      ],
      replays: []
    }
  };
}

/** Default pure render context; overrides never mutate another fixture. */
export function makeContext(overrides: Partial<ViewContext> = {}): ViewContext {
  return {
    record: makeRecord(),
    bindings: {},
    ground: 'light',
    idPrefix: 'v5',
    reducedMotion: false,
    ...overrides
  };
}
