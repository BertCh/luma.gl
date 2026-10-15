// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import type {GPUComputationRelation, GPUComputationCapture} from '@luma.gl/gpgpu';
import {createKeyRelationGraphFromRecord} from '../../../src/computation-views/hosts/record-relations';
import {makeV7Record} from '../fixtures/v7-lens-record';

function makeGraph(relations: GPUComputationRelation[], captures: GPUComputationCapture[] = []) {
  const record = makeV7Record();
  record.axes = {
    ...record.axes,
    keySpaces: [
      ...record.axes.keySpaces,
      {id: 'tile', label: 'tile', size: 4},
      {id: 'cell', label: 'cell', size: 16}
    ],
    relations
  };
  record.captures = {...record.captures!, snapshots: captures};
  return createKeyRelationGraphFromRecord(record);
}
const identity: GPUComputationRelation = {
  id: 'identity',
  from: 'row',
  to: 'cell',
  cardinality: '1:1',
  kind: 'function',
  ref: {function: 'identity'},
  timeDependent: false
};
const divide: GPUComputationRelation = {
  id: 'divide',
  from: 'row',
  to: 'tile',
  cardinality: 'N:1',
  kind: 'function',
  ref: {function: 'divide', params: {divisor: 4}},
  timeDependent: false
};
const targets: GPUComputationCapture = {
  id: 'target-id',
  name: 'neighbours',
  resource: 'targets',
  elementType: 'u32',
  values: [1, 2, 3],
  provenance: 'gpu-readback'
};
const offsets: GPUComputationCapture = {
  ...targets,
  id: 'offset-id',
  name: 'neighbours-offsets',
  resource: 'offsets',
  values: [0, 2, 3]
};
const csr: GPUComputationRelation = {
  id: 'csr',
  from: 'row',
  to: 'cell',
  cardinality: 'N:M',
  kind: 'buffer',
  ref: {layout: 'csr', resource: 'targets', offsetsResource: 'offsets', capture: 'target-id'},
  timeDependent: false
};

describe('record key relations', () => {
  test('evaluates identity and divide in both directions within declared domains', () => {
    const graph = makeGraph([identity, divide]);
    expect(graph.mapKeys([0, 3, 15, 16], 'row', 'cell')).toEqual({
      status: 'ok',
      keys: new Set([0, 3, 15])
    });
    expect(graph.mapKeys([0, 3, 4, 15], 'row', 'tile')).toEqual({
      status: 'ok',
      keys: new Set([0, 1, 3])
    });
    expect(graph.mapKeys([1], 'tile', 'row')).toEqual({status: 'ok', keys: new Set([4, 5, 6, 7])});
  });
  test('evaluates row-major tiles including partial edge tiles', () => {
    const graph = makeGraph([
      {...divide, ref: {function: 'tile-2d', params: {width: 5, tileWidth: 2, tileHeight: 2}}}
    ]);
    expect(graph.mapKeys([0, 4, 9, 10, 14], 'row', 'tile')).toEqual({
      status: 'ok',
      keys: new Set([0, 2, 3, 5])
    });
  });
  test('joins CSR captures by id and offsets resource, with name fallbacks', () => {
    const graph = makeGraph([csr], [targets, {...offsets, name: 'not-the-fallback'}]);
    expect(graph.mapKeys([0], 'row', 'cell')).toEqual({status: 'ok', keys: new Set([1, 2])});
    expect(graph.mapKeys([2, 3], 'cell', 'row')).toEqual({status: 'ok', keys: new Set([0, 1])});
    const fallback = makeGraph(
      [{...csr, ref: {layout: 'csr', resource: 'targets', capture: 'neighbours'}}],
      [targets, offsets]
    );
    expect(fallback.mapKeys([1], 'row', 'cell')).toEqual({status: 'ok', keys: new Set([3])});
  });
  test('captured index relations are usable even when time-dependent', () => {
    const graph = makeGraph(
      [
        {
          ...csr,
          timeDependent: true,
          ref: {layout: 'index', resource: 'targets', capture: 'target-id'}
        }
      ],
      [targets]
    );
    expect(graph.mapKeys([0, 2, 8], 'row', 'cell')).toEqual({status: 'ok', keys: new Set([1, 3])});
  });
  test('does not guess unknown or uncaptured time-dependent relations', () => {
    for (const relation of [
      {...identity, kind: 'unknown' as const},
      {...identity, timeDependent: true},
      {...csr, timeDependent: true}
    ]) {
      expect(makeGraph([relation]).mapKeys([0], 'row', 'cell')).toEqual({
        status: 'unknown',
        reason: 'time-dependent relation not captured'
      });
    }
    expect(makeGraph([csr]).mapKeys([0], 'row', 'cell')).toEqual({
      status: 'unknown',
      reason: 'relation values not captured'
    });
    expect(makeGraph([csr], [targets]).mapKeys([0], 'row', 'cell')).toMatchObject({
      status: 'unknown'
    });
  });
  test('reports unsupported functions and unknown domain sizes', () => {
    expect(
      makeGraph([{...identity, ref: {function: 'custom'}}]).mapKeys([0], 'row', 'cell')
    ).toEqual({status: 'unknown', reason: 'function custom not evaluable in views'});
    const record = makeV7Record();
    record.axes = {...record.axes, keySpaces: [{id: 'row', label: 'row'}], relations: [identity]};
    expect(createKeyRelationGraphFromRecord(record).mapKeys([0], 'row', 'cell')).toEqual({
      status: 'unknown',
      reason: 'key space size unknown'
    });
  });
  test('maps a two-hop star through its declared primary key', () => {
    const graph = makeGraph([identity, divide]);
    expect(graph.findPath('cell', 'tile')).toHaveLength(2);
    expect(graph.mapKeys([5, 6], 'cell', 'tile')).toEqual({status: 'ok', keys: new Set([1])});
    expect(graph.mapKeys([1], 'tile', 'cell')).toEqual({status: 'ok', keys: new Set([4, 5, 6, 7])});
  });
  test('never invents an invocation identity from node datum bindings', () => {
    expect(
      createKeyRelationGraphFromRecord(makeV7Record()).mapKeys([0], 'invocation', 'row')
    ).toEqual({status: 'unlinked'});
  });
});
