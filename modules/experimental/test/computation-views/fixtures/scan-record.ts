// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LensRecord} from '../../../src/computation-views/registry/view-registration';

/** Small deterministic scan recording with real-shaped structure and capture sidecars. */
export function makeScanRecord(): LensRecord {
  const identifiers = ['scan/level-0/upsweep', 'scan/level-0/downsweep', 'scan/add-partials'];
  return {
    schema: 'luma.computation/1',
    id: 'gpu-core.scan',
    runId: 'scan-run-1',
    createdAt: '2026-10-14T00:00:00.000Z',
    adapter: {backend: 'webgpu', description: 'Apple M3', variant: 'metal-3'},
    domains: ['gpu-core'],
    graphs: [
      {
        id: 'compiled',
        form: 'compiled',
        domain: 'gpu-core',
        label: 'Compiled scan',
        nodes: identifiers.map((id, index) => ({
          id,
          label: id.split('/').at(-1)!,
          namespace: index < 2 ? 'scan/level-0' : 'scan',
          kind: 'compute',
          op: 'GPUScan',
          reads: [index === 0 ? 'input' : 'partial-a'],
          writes: [index === 2 ? 'output' : 'partial-a'],
          attrs: {level: index, inclusive: true},
          loop: {id: 'levels', iteration: index},
          dispatch: {workgroups: [1, 1, 1], invocationCount: 16},
          datum: {keySpace: 'row'},
          bindings: [{name: 'values', resource: 'input', access: 'read', elementType: 'u32'}],
          formula: 'scan-formula',
          source: 'wgsl:scan'
        })),
        edges: [
          {from: identifiers[0], to: identifiers[1], kind: 'raw', resource: 'partial-a'},
          {from: identifiers[1], to: identifiers[2], kind: 'raw', resource: 'partial-a'}
        ],
        resources: ['input', 'partial-a', 'partial-b', 'output'].map((id, index) => ({
          id,
          kind: 'buffer',
          bytes: 64,
          ownership: index === 1 || index === 2 ? 'transient' : 'imported',
          allocation: index === 1 || index === 2 ? 'allocation-0' : undefined,
          firstNode: identifiers[index === 2 ? 2 : 0],
          lastNode: identifiers[index === 1 ? 1 : 2],
          retained: false
        }))
      }
    ],
    axes: {
      axes: [
        {id: 'entity', label: 'row', kind: 'data', keySpace: 'row', size: 16},
        {id: 'iteration', label: 'iteration', kind: 'time', size: 64}
      ],
      keySpaces: [{id: 'row', label: 'row', size: 16}],
      relations: []
    },
    refs: {
      formulas: [
        {
          id: 'scan-formula',
          tex: 'y_i = \\sum_{j=0}^i x_j',
          ascii: 'y[i] = sum(x[0..i])',
          symbols: [],
          derivation: 'curated'
        }
      ],
      sources: [
        {
          id: 'wgsl:scan',
          language: 'wgsl',
          moduleId: 'GPUScan',
          hash: 'scan',
          text: '@compute @workgroup_size(16) fn main() {}'
        }
      ]
    },
    decisions: [
      {
        id: 'strategy',
        subject: 'GPUScan',
        kind: 'strategy',
        chosen: 'hierarchical',
        candidates: [
          {id: 'hierarchical', score: 1, supported: true, reason: 'supported'},
          {id: 'subgroup', supported: false, reason: 'missing subgroups'}
        ]
      }
    ],
    choices: [
      {stage: 'scan', name: 'inclusive', value: true, alternatives: [false], sweepable: false}
    ],
    metrics: [
      {
        schema: 'luma.computation-metrics/1',
        runId: 'scan-run-1',
        graph: 'compiled',
        timebase: 'ns',
        encoding: {
          coalesced: false,
          timestamped: true,
          taps: 0,
          lensTwin: true,
          retained: ['partial-a', 'partial-b']
        },
        nodes: Object.fromEntries(
          identifiers.map((id, index) => [
            id,
            {cpuEncodeMs: 0.1, gpuBegin: index * 100, gpuEnd: (index + 1) * 100}
          ])
        )
      }
    ],
    captures: {
      schema: 'luma.computation-captures/1',
      runId: 'scan-run-1',
      snapshots: ['input', 'output'].map((name, index) => ({
        id: name,
        name,
        keySpace: 'row',
        elementType: 'u32',
        values: new Uint32Array(
          Array.from({length: 16}, (_, value) => (index ? (value * (value + 1)) / 2 : value))
        ),
        provenance: 'gpu-readback'
      })),
      stamps: [
        {
          id: 'stamp',
          name: 'settledAt',
          keySpace: 'row',
          elementType: 'u32',
          values: new Uint32Array(16).fill(2),
          provenance: 'gpu-readback',
          meaning: 'settledAt round'
        }
      ],
      replays: []
    }
  };
}
