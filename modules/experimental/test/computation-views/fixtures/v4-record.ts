// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import type {
  LensRecord,
  ViewContext
} from '../../../src/computation-views/registry/view-registration';

/** Creates an independent recorded capture for view tests. */
export function makeCapture(overrides: Partial<GPUComputationCapture> = {}): GPUComputationCapture {
  return {
    id: 'result-1',
    name: 'result',
    resource: 'output',
    graph: 'compiled',
    afterNode: 'finish',
    iteration: 1,
    keySpace: 'cells',
    elementType: 'f32',
    values: [2, 4, 6, 8, 10, 12, 14, 16, 18],
    provenance: 'gpu-readback',
    ...overrides
  };
}

/** Hand-built deterministic record with two iterations, a formula and a raster key space. */
export function makeRecord(): LensRecord {
  return {
    schema: 'luma.computation/1',
    id: 'v4-record',
    runId: 'v4-run',
    createdAt: '2026-10-14T00:00:00Z',
    adapter: {backend: 'webgpu', description: 'Fixture adapter'},
    domains: ['test'],
    graphs: [
      {
        id: 'compiled',
        form: 'compiled',
        domain: 'test',
        nodes: [
          {
            id: 'start',
            label: 'Input',
            namespace: 'prepare/input',
            kind: 'compute',
            reads: [],
            writes: ['input'],
            attrs: {}
          },
          {
            id: 'finish',
            label: 'Double',
            namespace: 'compute/finish',
            kind: 'compute',
            op: 'double',
            formula: 'double',
            source: 'source',
            bindings: [{name: 'weights', resource: 'input', access: 'read'}],
            reads: ['input'],
            writes: ['output'],
            attrs: {}
          }
        ],
        edges: [{from: 'start', to: 'finish', kind: 'dataflow'}],
        resources: [{id: 'unused', kind: 'buffer', bytes: 4, ownership: 'imported'}]
      }
    ],
    axes: {
      axes: [{id: 'entity', label: 'Cell', kind: 'data', keySpace: 'cells'}],
      keySpaces: [{id: 'cells', label: 'Cell', size: 9, indexSpace: '2d', shape: [3, 3]}],
      relations: []
    },
    refs: {
      formulas: [
        {
          id: 'double',
          tex: '\\term{output}{y}=\\term{factor}{a}\\term{input}{x}',
          ascii: 'y = a x',
          derivation: 'curated',
          symbols: [
            {id: 'input', label: 'x', role: 'input', binding: 'resource:input'},
            {id: 'factor', label: 'a', role: 'parameter', binding: '2'},
            {id: 'output', label: 'y', role: 'output', binding: 'resource:output'},
            {id: 'unbound', role: 'index'},
            {id: 'missing', role: 'intermediate', binding: 'resource:missing'}
          ]
        }
      ],
      sources: [
        {
          id: 'source',
          language: 'wgsl',
          moduleId: 'double',
          hash: 'fixture',
          text: 'let output = sourceValue * 2;'
        }
      ]
    },
    decisions: [],
    choices: [],
    captures: {
      schema: 'luma.computation-captures/1',
      runId: 'v4-run',
      snapshots: [
        makeCapture({
          id: 'input-0',
          name: 'input',
          resource: 'input',
          afterNode: 'start',
          iteration: undefined,
          values: [1, 2, 3, 4, 5, 6, 7, 8, 9]
        }),
        makeCapture({
          id: 'factor',
          name: 'factor',
          resource: undefined,
          afterNode: 'start',
          iteration: undefined,
          values: [2, 2, 2, 2, 2, 2, 2, 2, 2]
        }),
        makeCapture({id: 'result-0', iteration: 0, values: [1, 3, 5, 7, 9, 11, 13, 15, 17]}),
        makeCapture()
      ],
      stamps: [],
      replays: []
    }
  };
}

/** A fresh view context and domain pack, avoiding shared mutable test state. */
export function makeContext(overrides: Partial<ViewContext> = {}): ViewContext {
  return {
    record: makeRecord(),
    bindings: {entity: 1},
    ground: 'light',
    idPrefix: 'v4-test',
    reducedMotion: true,
    packs: [
      {
        id: 'test',
        analysisStages: ['prepare', 'compute'],
        captureStages: {input: 'prepare', factor: 'prepare', result: 'compute'},
        operators: {double: {scope: 'focal', reduction: 'sum'}},
        neighbourhoods: {cells: {kind: 'stencil-3x3', keySpace: 'cells'}}
      }
    ],
    ...overrides
  };
}
