// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  KeyRelationGraph,
  createFunctionRelation,
  createCSRRelation,
  createUnavailableRelation
} from '../../../src/computation-views/coordinator/key-relations';
import {LadderCoordinator} from '../../../src/computation-views/coordinator/ladder-coordinator';

const makeRowsToCells = () =>
  createFunctionRelation({
    id: 'rows-cells',
    from: 'rows',
    to: 'cells',
    cardinality: 'N:1',
    fromCount: 8,
    map: key => Math.floor(key / 4)
  });

describe('KeyRelationGraph', () => {
  test('function images, scanned preimages and explicit inverses are sets', () => {
    const relation = makeRowsToCells();
    expect(relation.forward([0, 1, 4, -1, 8])).toEqual(new Set([0, 1]));
    expect(relation.inverse([1])).toEqual(new Set([4, 5, 6, 7]));
    const multiple = createFunctionRelation({
      id: 'multiple',
      from: 'a',
      to: 'b',
      cardinality: '1:N',
      fromCount: 3,
      map: key => [key * 2, key * 2 + 1],
      inverseMap: key => Math.floor(key / 2)
    });
    expect(multiple.forward([1])).toEqual(new Set([2, 3]));
    expect(multiple.inverse([2, 3])).toEqual(new Set([1]));
  });

  test('CSR supports empty rows, duplicate edges and a cached inverse', () => {
    const relation = createCSRRelation({
      id: 'csr',
      from: 'rows',
      to: 'cells',
      offsets: [0, 2, 2, 5],
      targets: [0, 1, 1, 2, 2],
      toCount: 3
    });
    expect(relation.forward([0, 1])).toEqual(new Set([0, 1]));
    expect(relation.forward([2])).toEqual(new Set([1, 2]));
    expect(relation.inverse([1])).toEqual(new Set([0, 2]));
    expect(relation.inverse([2])).toEqual(new Set([2]));
    expect(relation.preimageSize?.(2)).toBe(1);
    expect(relation.inverse([99])).toEqual(new Set());
  });

  test('two-hop paths work in both directions, including cycles and identity', () => {
    const graph = new KeyRelationGraph();
    graph.add(makeRowsToCells());
    graph.add(
      createFunctionRelation({
        id: 'cells-tiles',
        from: 'cells',
        to: 'tiles',
        cardinality: 'N:1',
        fromCount: 2,
        map: () => 7
      })
    );
    graph.add(
      createFunctionRelation({
        id: 'branch',
        from: 'cells',
        to: 'nodes',
        cardinality: '1:1',
        fromCount: 2,
        map: key => key
      })
    );
    expect(graph.findPath('rows', 'tiles')).toHaveLength(2);
    expect(graph.mapKeys([1, 7], 'rows', 'tiles')).toEqual({status: 'ok', keys: new Set([7])});
    expect(graph.mapKeys([7], 'tiles', 'rows')).toEqual({
      status: 'ok',
      keys: new Set([0, 1, 2, 3, 4, 5, 6, 7])
    });
    expect(graph.mapKeys([1], 'rows', 'rows')).toEqual({status: 'ok', keys: new Set([1])});
    expect(graph.mapKeys([1], 'rows', 'disconnected')).toEqual({status: 'unlinked'});
  });

  test('unavailable relation reports unknown without executing the mapping', () => {
    const graph = new KeyRelationGraph();
    const relation = createUnavailableRelation({
      id: 'missing',
      from: 'rows',
      to: 'cells',
      cardinality: 'N:M',
      reason: 'Time-dependent relation not captured'
    });
    graph.add(relation);
    expect(graph.mapKeys([0], 'rows', 'cells')).toEqual({
      status: 'unknown',
      reason: 'Time-dependent relation not captured'
    });
    expect(graph.mapKeys([0], 'cells', 'rows').status).toBe('unknown');
    expect(() => relation.forward([0])).toThrow();
    expect(() => relation.inverse([0])).toThrow();
  });

  test('highlight shows one of four rows, pick lowers marks to preimages, and off-view is explicit', () => {
    const coordinator = new LadderCoordinator();
    const relation = makeRowsToCells();
    coordinator.addClause('selection', {
      source: 'table',
      keySpace: 'rows',
      predicate: {kind: 'keys', keys: [0, 7, 8]},
      label: 'Three rows'
    });
    const view = {viewId: 'map', keySpace: 'rows', keyCount: 9, markRelation: relation};
    expect(coordinator.highlight('selection', view)).toEqual({
      status: 'ok',
      marks: new Map([
        [0, {covered: 1, total: 4}],
        [1, {covered: 1, total: 4}]
      ]),
      offView: {count: 1}
    });
    expect(coordinator.highlight('selection', {...view, visibleKeys: [0]})).toEqual({
      status: 'ok',
      marks: new Map([[0, {covered: 1, total: 4}]]),
      offView: {count: 2}
    });
    const picked = coordinator.pick('selection', view, [1], {label: 'Cell 1'});
    expect(picked.predicate).toEqual({kind: 'keys', keys: [4, 5, 6, 7]});
    expect(coordinator.resolve('selection', 'rows', {keyCount: 9, viewId: 'table'})).toEqual({
      status: 'ok',
      keys: new Set([4, 5, 6, 7])
    });
    const additive = coordinator.pick('selection', view, [0], {additive: true});
    expect(additive.predicate.kind === 'keys' && new Set(additive.predicate.keys)).toEqual(
      new Set([0, 1, 2, 3, 4, 5, 6, 7])
    );
  });

  test('cross-space clauses map before intersection; missing mark evidence stays unknown', () => {
    const relations = new KeyRelationGraph();
    relations.add(makeRowsToCells());
    const coordinator = new LadderCoordinator({relations});
    coordinator.addClause('filter', {
      source: 'table',
      keySpace: 'rows',
      predicate: {kind: 'keys', keys: [5]},
      label: 'Row 5'
    });
    coordinator.addClause('filter', {
      source: 'map',
      keySpace: 'cells',
      predicate: {kind: 'keys', keys: [1]},
      label: 'Cell 1'
    });
    expect(coordinator.resolve('filter', 'rows', {keyCount: 8})).toEqual({
      status: 'ok',
      keys: new Set([5])
    });
    const missing = createUnavailableRelation({
      id: 'missing',
      from: 'rows',
      to: 'marks',
      cardinality: '1:N',
      reason: 'No capture'
    });
    expect(
      coordinator.highlight('filter', {
        viewId: 'other',
        keySpace: 'rows',
        keyCount: 8,
        markRelation: missing
      })
    ).toEqual({status: 'unknown', reason: 'No capture'});
    expect(() =>
      coordinator.pick('selection', {viewId: 'other', keySpace: 'rows', markRelation: missing}, [1])
    ).toThrow();
  });
});
