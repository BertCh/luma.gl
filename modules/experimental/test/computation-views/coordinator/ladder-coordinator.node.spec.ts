// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  LadderCoordinator,
  getPageCoordinator,
  resetPageCoordinators,
  type CoordinatorEvent
} from '../../../src/computation-views/coordinator/ladder-coordinator';
import {CPUSelectionBackend} from '../../../src/computation-views/coordinator/selection-backend';
import {
  isGPUSelectionBackend,
  type GPUSelectionBackend
} from '../../../src/computation-views/coordinator/gpu-selection-backend';
import {
  KeyRelationGraph,
  createFunctionRelation
} from '../../../src/computation-views/coordinator/key-relations';
import type {SelectionClause} from '../../../src/computation-views/types';

function clause(source: string, keys: number[], clients?: string[]): SelectionClause {
  return {
    source,
    keySpace: 'rows',
    predicate: {kind: 'keys', keys},
    label: `Rows ${keys.join(', ')}`,
    clients
  };
}
function resolved(coordinator: LadderCoordinator, selectionId = 'filter', viewId?: string) {
  return coordinator.resolve(selectionId, 'rows', {keyCount: 8, viewId});
}

describe('LadderCoordinator', () => {
  test('replaces by source and resolves single, intersect and union', () => {
    const coordinator = new LadderCoordinator();
    coordinator.addClause('filter', clause('a', [1, 2]));
    coordinator.addClause('filter', clause('b', [2, 3]));
    expect(resolved(coordinator)).toEqual({status: 'ok', keys: new Set([2])});
    coordinator.addClause('filter', clause('a', [3, 4]));
    expect(coordinator.getClauses('filter')).toHaveLength(2);
    expect(resolved(coordinator)).toEqual({status: 'ok', keys: new Set([3])});
    coordinator.defineSelection('filter', 'union');
    expect(resolved(coordinator)).toEqual({status: 'ok', keys: new Set([2, 3, 4])});
    coordinator.defineSelection('filter', 'single');
    expect(resolved(coordinator)).toEqual({status: 'ok', keys: new Set([3, 4])});
    coordinator.clearClauses('filter', 'a');
    expect(resolved(coordinator)).toEqual({status: 'ok', keys: new Set([2, 3])});
    coordinator.clearClauses('filter');
    expect(resolved(coordinator)).toEqual({status: 'empty'});
    coordinator.addClause('filter', clause('a', []));
    expect(resolved(coordinator)).toEqual({status: 'ok', keys: new Set()});
  });

  test('self-excludes and respects client scopes before resolving single', () => {
    const coordinator = new LadderCoordinator();
    coordinator.addClause('selection', clause('a', [1, 2]));
    coordinator.addClause('selection', clause('b', [2, 3]));
    expect(resolved(coordinator, 'selection', 'b')).toEqual({status: 'ok', keys: new Set([1, 2])});
    coordinator.addClause('filter', clause('a', [1], ['target']));
    expect(resolved(coordinator, 'filter', 'a')).toEqual({status: 'empty'});
    expect(resolved(coordinator, 'filter', 'other')).toEqual({status: 'empty'});
    expect(resolved(coordinator, 'filter', 'target')).toEqual({status: 'ok', keys: new Set([1])});
  });

  test('keeps all four interaction states independent', () => {
    const coordinator = new LadderCoordinator();
    coordinator.addClause('hover', clause('view', [1]));
    coordinator.addClause('selection', clause('view', [2]));
    coordinator.addClause('filter', clause('view', [3]));
    coordinator.setParam('focus', 4);
    expect(resolved(coordinator, 'hover')).toEqual({status: 'ok', keys: new Set([1])});
    expect(resolved(coordinator, 'selection')).toEqual({status: 'ok', keys: new Set([2])});
    expect(resolved(coordinator)).toEqual({status: 'ok', keys: new Set([3])});
    expect(coordinator.getParam('focus')).toBe(4);
    coordinator.clearClauses('hover');
    expect(coordinator.getClauses('selection')).toHaveLength(1);
  });

  test('switches writer before change, suppresses origin echo and guards delayed writes', () => {
    const coordinator = new LadderCoordinator();
    const events: CoordinatorEvent[] = [];
    const originEvents: CoordinatorEvent[] = [];
    const unsubscribe = coordinator.subscribe(event => events.push(event));
    coordinator.subscribe(event => originEvents.push(event), {viewId: 'view'});
    coordinator.setParam('iteration', 2, {writer: 'pointer', origin: 'view', heat: 'hot'});
    expect(events.map(event => event.type)).toEqual(['writer', 'param']);
    expect(originEvents.map(event => event.type)).toEqual(['writer']);
    expect(coordinator.setParam('iteration', 9, {writer: 'narrative', ifWriter: 'narrative'})).toBe(
      false
    );
    expect(coordinator.getParam('iteration')).toBe(2);
    expect(coordinator.getWriter()).toBe('pointer');
    expect(events).toHaveLength(2);
    coordinator.addClause('selection', clause('view', [2]), {origin: 'view'});
    expect(originEvents).toHaveLength(1);
    unsubscribe();
    coordinator.setWriter('url');
    expect(events).toHaveLength(3);
  });

  test('stays stale until record arrival and decouples only over the threshold', () => {
    const coordinator = new LadderCoordinator();
    coordinator.subscribe(() => {}, {viewId: 'dependent'});
    coordinator.setParam('parameter', 1, {heat: 'warm'});
    expect(coordinator.getCoupling('dependent')).toBe('stale');
    expect(coordinator.reportUpdate('dependent', 99)).toBe('stale');
    coordinator.requestUpdate('dependent');
    expect(coordinator.getCoupling('dependent')).toBe('stale');
    coordinator.setRecord('run-1');
    expect(coordinator.getCoupling('dependent')).toBe('linked');
    expect(coordinator.reportUpdate('view', 99)).toBe('linked');
    expect(coordinator.reportUpdate('view', 100)).toBe('linked');
    expect(coordinator.reportUpdate('view', 101)).toBe('decoupled');
    expect(coordinator.reportUpdate('view', 1)).toBe('decoupled');
    coordinator.setRecord('run-2');
    expect(coordinator.getCoupling('view')).toBe('decoupled');
    coordinator.requestUpdate('view');
    expect(coordinator.getCoupling('view')).toBe('linked');
  });

  test('preview is hot and commits one labelled warm action; log is bounded', () => {
    const coordinator = new LadderCoordinator({now: () => 42});
    const events: CoordinatorEvent[] = [];
    coordinator.subscribe(event => events.push(event));
    coordinator.previewParam('iteration', 1);
    coordinator.previewParam('iteration', 2);
    expect(coordinator.getActionLog()).toEqual([]);
    expect(coordinator.getParam('iteration')).toBe(2);
    coordinator.commitParam('iteration', {label: 'Inspect iteration 2'});
    expect(events.filter(event => event.type === 'param').map(event => event.heat)).toEqual([
      'hot',
      'hot',
      'warm'
    ]);
    expect(coordinator.getActionLog()).toEqual([
      {sequence: 1, kind: 'param', label: 'Inspect iteration 2', writer: 'narrative', time: 42}
    ]);
    expect(coordinator.commitParam('iteration')).toBe(false);
    coordinator.addClause('selection', clause('a', [1]));
    expect(coordinator.getActionLog()[1].label).toBe('Rows 1');
    for (let index = 0; index < 500; index++) coordinator.setParam('iteration', index);
    expect(coordinator.getActionLog()).toHaveLength(500);
    expect(coordinator.getActionLog()[0].sequence).toBe(3);
  });

  test('requires source metadata for cross-space all/range and does not read GPU masks', () => {
    const relations = new KeyRelationGraph();
    relations.add(
      createFunctionRelation({
        id: 'groups',
        from: 'rows',
        to: 'cells',
        cardinality: 'N:1',
        fromCount: 4,
        map: key => Math.floor(key / 2)
      })
    );
    const coordinator = new LadderCoordinator({relations});
    coordinator.addClause('filter', {
      source: 'range',
      keySpace: 'rows',
      predicate: {kind: 'range', field: 'value', range: [2, 3]},
      label: 'Range 2–3'
    });
    expect(coordinator.resolve('filter', 'cells', {keyCount: 2}).status).toBe('unknown');
    coordinator.registerKeySpace('rows', 4, field =>
      field === 'value' ? [0, 2, 3, 9] : undefined
    );
    expect(coordinator.resolve('filter', 'cells', {keyCount: 2})).toEqual({
      status: 'ok',
      keys: new Set([0, 1])
    });
    const backend: GPUSelectionBackend = {
      kind: 'gpu',
      accepts: () => true,
      setClause() {
        throw Error('interface only');
      },
      clearSource() {},
      getMask() {
        throw Error('no readback');
      },
      transferMask() {
        throw Error('interface only');
      },
      readCounts: async () => ({})
    };
    const gpuCoordinator = new LadderCoordinator({backends: [backend]});
    gpuCoordinator.addClause('filter', clause('a', [0]));
    expect(resolved(gpuCoordinator).status).toBe('unknown');
    expect(isGPUSelectionBackend(backend)).toBe(true);
    expect(isGPUSelectionBackend(new CPUSelectionBackend())).toBe(false);
  });

  test('owns clause snapshots and shares coordinators by bind name', () => {
    resetPageCoordinators();
    const coordinator = getPageCoordinator('page');
    expect(getPageCoordinator('page')).toBe(coordinator);
    expect(getPageCoordinator('other')).not.toBe(coordinator);
    const input = clause('a', [1]);
    coordinator.addClause('selection', input);
    input.label = 'mutated';
    expect(coordinator.getClauses('selection')[0].label).toBe('Rows 1');
    const snapshot = coordinator.getClauses('selection')[0];
    snapshot.label = 'mutated again';
    expect(coordinator.getClauses('selection')[0].label).toBe('Rows 1');
    resetPageCoordinators();
    expect(() => coordinator.setParam('focus', 1)).toThrow();
    expect(getPageCoordinator('page')).not.toBe(coordinator);
    resetPageCoordinators();
  });
});

describe('CPUSelectionBackend', () => {
  test('evaluates keys, all and inclusive ranges within its budget', () => {
    const backend = new CPUSelectionBackend({maxKeyCount: 4});
    expect(backend.accepts('rows', 4)).toBe(true);
    expect(backend.accepts('rows', 5)).toBe(false);
    expect(backend.evaluate(clause('a', [0, 0, 3, 4, -1, 1.5]), 4)).toEqual(new Set([0, 3]));
    expect(backend.evaluate({...clause('a', []), predicate: {kind: 'all'}}, 4)).toEqual(
      new Set([0, 1, 2, 3])
    );
    const range: SelectionClause = {
      ...clause('a', []),
      predicate: {kind: 'range', field: 'value', range: [1, 2]}
    };
    expect(backend.evaluate(range, 4, () => [0, 1, 2, 3])).toEqual(new Set([1, 2]));
    expect(backend.evaluate(range, 4)).toBe('unknown');
    expect(backend.evaluate(range, 4, () => [1])).toBe('unknown');
    expect(backend.evaluate(clause('a', [0]), 5)).toBe('unknown');
    expect(new CPUSelectionBackend().accepts('rows', 65536)).toBe(true);
    expect(new CPUSelectionBackend().accepts('rows', 65537)).toBe(false);
  });
});
