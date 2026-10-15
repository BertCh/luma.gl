// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationGraph, GPUComputationNode} from '@luma.gl/gpgpu';
import type {
  LensRecord,
  ViewContext
} from '../../../src/computation-views/registry/view-registration';

/** Forty scheduled nodes, including a ten-iteration loop and deep scan namespaces. */
export function makeV6Graph(): GPUComputationGraph {
  const nodes: GPUComputationNode[] = Array.from({length: 40}, (_, index) => ({
    id: `node-${index}`,
    label: `Pass ${index}`,
    namespace:
      index < 10
        ? 'scan/level-0/upsweep'
        : index < 20
          ? 'scan/level-1/downsweep'
          : index < 30
            ? 'solver/iteration'
            : 'finish/output',
    kind: 'compute',
    reads: index === 0 ? ['input'] : ['scratch-a'],
    writes: index === 39 ? ['output'] : ['scratch-b'],
    attrs: {size: 256, stride: index, caption: 'non-numeric'},
    workload: {maximumInvocationCount: 256},
    ...(index >= 20 && index < 30
      ? {loop: {id: 'relaxation', iteration: index - 20}, dedupName: 'relax'}
      : {}),
    ...(index === 39 ? {publication: {id: 'result', completeness: 'complete' as const}} : {}),
    derivedFrom: ['intent:scan']
  }));
  return {
    id: 'compiled',
    label: 'Scan and relaxation',
    form: 'compiled',
    domain: 'gpu-core',
    nodes,
    edges: [
      ...nodes.slice(1).map((node, index) => ({
        from: nodes[index].id,
        to: node.id,
        kind: 'raw' as const,
        resource: 'scratch-a'
      })),
      {from: 'node-2', to: 'node-21', kind: 'war', resource: 'scratch-b'},
      {from: 'node-2', to: 'node-22', kind: 'war', resource: 'scratch-b'},
      {from: 'node-2', to: 'node-23', kind: 'waw', resource: 'scratch-b'},
      {from: 'node-2', to: 'node-24', kind: 'explicit'},
      {from: 'node-2', to: 'node-25', kind: 'indirect', resource: 'scratch-a'},
      {from: 'node-2', to: 'node-26', kind: 'dataflow'}
    ],
    resources: [
      {id: 'input', kind: 'buffer', bytes: 256, ownership: 'imported'},
      {id: 'output', kind: 'buffer', bytes: 256, ownership: 'imported'},
      {
        id: 'scratch-a',
        kind: 'buffer',
        bytes: 128,
        ownership: 'transient',
        allocation: 'shared',
        allocationBytes: 128,
        firstNode: 'node-0',
        lastNode: 'node-9'
      },
      {
        id: 'scratch-b',
        kind: 'buffer',
        bytes: 128,
        ownership: 'transient',
        allocation: 'shared',
        allocationBytes: 128,
        firstNode: 'node-10',
        lastNode: 'node-29'
      },
      {
        id: 'observed',
        kind: 'buffer',
        bytes: 64,
        ownership: 'transient',
        allocation: 'retained',
        allocationBytes: 64,
        firstNode: 'node-3',
        lastNode: 'node-5',
        retained: true
      }
    ]
  };
}

/** Self-contained multi-form record with focused and non-focused readbacks. */
export function makeV6Record(): LensRecord {
  return {
    schema: 'luma.computation/1',
    id: 'v6',
    runId: 'v6-run',
    createdAt: '2026-10-14T00:00:00Z',
    adapter: {backend: 'webgpu', description: 'Fixture adapter'},
    domains: ['gpu-core'],
    graphs: [
      {
        id: 'intent',
        form: 'intent',
        label: 'Declared scan',
        domain: 'gpu-core',
        nodes: [
          {
            id: 'scan',
            label: 'Scan',
            namespace: '',
            kind: 'operation',
            reads: [],
            writes: [],
            attrs: {}
          }
        ],
        edges: []
      },
      makeV6Graph()
    ],
    axes: {axes: [], keySpaces: [], relations: []},
    refs: {formulas: [], sources: []},
    decisions: [],
    choices: [],
    captures: {
      schema: 'luma.computation-captures/1',
      runId: 'v6-run',
      snapshots: [
        {
          id: 'focus-capture',
          name: 'values',
          graph: 'compiled',
          afterNode: 'node-2',
          elementType: 'f32',
          values: [1, 2, 3, 4],
          provenance: 'gpu-readback'
        },
        {
          id: 'other-capture',
          name: 'values',
          graph: 'compiled',
          afterNode: 'node-39',
          elementType: 'f32',
          values: [2, 3, 4, 5],
          provenance: 'gpu-replay'
        }
      ],
      stamps: [],
      replays: []
    }
  };
}

/** Stable view context for model and static-render tests. */
export function makeV6Context(overrides: Partial<ViewContext> = {}): ViewContext {
  return {
    record: makeV6Record(),
    bindings: {node: 'node-2'},
    ground: 'light',
    idPrefix: 'v6',
    reducedMotion: true,
    ...overrides
  };
}
