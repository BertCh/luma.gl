// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LensRecord} from '../../../src/computation-views/registry/view-registration';

/** Small recorded computation with explicit row keys, a formula, values and measurements. */
export function makeV7Record(): LensRecord {
  return {
    schema: 'luma.computation/1',
    id: 'v7',
    runId: 'v7-run',
    createdAt: '2026-10-14T00:00:00Z',
    adapter: {backend: 'webgpu', description: 'Fixture GPU'},
    domains: ['gpu-core'],
    graphs: [
      {
        id: 'compiled',
        form: 'compiled',
        domain: 'gpu-core',
        label: 'Double rows',
        nodes: [
          {
            id: 'double',
            label: 'Double',
            namespace: 'rows',
            kind: 'compute',
            datum: {keySpace: 'row'},
            formula: 'double-formula',
            reads: ['input'],
            writes: ['output'],
            attrs: {},
            workgroupSize: [16, 1, 1],
            dispatch: {workgroups: [1, 1, 1], invocationCount: 16}
          }
        ],
        edges: [],
        resources: [{id: 'output', kind: 'buffer', bytes: 64, ownership: 'imported'}]
      }
    ],
    axes: {
      axes: [
        {id: 'entity', label: 'row', kind: 'data', keySpace: 'row', size: 16},
        {id: 'iteration', label: 'iteration', kind: 'time', size: 4}
      ],
      keySpaces: [{id: 'row', label: 'row', size: 16, indexSpace: '1d'}],
      relations: []
    },
    refs: {
      formulas: [
        {
          id: 'double-formula',
          tex: 'y_i = 2x_i',
          ascii: 'y[i] = 2 * x[i]',
          symbols: [],
          derivation: 'derived'
        }
      ],
      sources: []
    },
    decisions: [],
    choices: [],
    captures: {
      schema: 'luma.computation-captures/1',
      runId: 'v7-run',
      snapshots: [
        {
          id: 'values',
          name: 'values',
          resource: 'output',
          graph: 'compiled',
          afterNode: 'double',
          keySpace: 'row',
          elementType: 'f32',
          values: Array.from({length: 16}, (_, index) => index * 2),
          provenance: 'gpu-readback'
        }
      ],
      stamps: [],
      replays: []
    },
    metrics: [
      {
        schema: 'luma.computation-metrics/1',
        runId: 'v7-run',
        graph: 'compiled',
        timebase: 'ms',
        encoding: {coalesced: false, timestamped: true, taps: 1, lensTwin: true},
        nodes: {double: {cpuEncodeMs: 0.1, gpuBegin: 0, gpuEnd: 1}},
        series: [{name: 'residual', axis: 'iteration', values: [8, 4, 2, 1]}]
      }
    ]
  };
}
