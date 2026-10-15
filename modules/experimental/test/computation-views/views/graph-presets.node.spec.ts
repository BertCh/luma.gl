// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  buildMemoryAliasing,
  buildPassResourceLadder
} from '../../../src/computation-views/views/graph-presets';
import {makeV6Graph} from '../fixtures/v6-command-graph';

describe('command graph presets', () => {
  test('ladder preserves compiler order and merges read/write accesses', () => {
    const graph = makeV6Graph();
    graph.nodes = [
      {...graph.nodes[0], reads: ['input', 'input'], writes: ['input', 'undeclared']},
      ...graph.nodes.slice(1)
    ];
    const ladder = buildPassResourceLadder(graph);
    expect(ladder.passes.map(pass => pass.nodeId)).toEqual(graph.nodes.map(node => node.id));
    expect(ladder.cells.filter(cell => cell.nodeId === 'node-0')).toEqual([
      {nodeId: 'node-0', resourceId: 'input', access: 'read-write'},
      {nodeId: 'node-0', resourceId: 'undeclared', access: 'write'}
    ]);
    expect(ladder.resources.at(-1)).toEqual({id: 'undeclared', ownership: 'unknown'});
    expect(ladder).toEqual(buildPassResourceLadder(graph));
  });

  test('aliased resources share allocations; savings and inclusive peak account for retention', () => {
    const aliasing = buildMemoryAliasing(makeV6Graph());
    expect(aliasing.scheduleLength).toBe(40);
    expect(aliasing.allocations).toHaveLength(2);
    expect(
      aliasing.allocations[0].resources.map(resource => [resource.firstIndex, resource.lastIndex])
    ).toEqual([
      [0, 9],
      [10, 29]
    ]);
    expect(aliasing.aliasedSavingsBytes).toBe(128);
    expect(aliasing.peakTransientBytes).toBe(192);
    expect(aliasing.totalsKnown).toBe(true);
    expect(aliasing.imported).toHaveLength(2);
    expect(aliasing).toEqual(buildMemoryAliasing(makeV6Graph()));
  });

  test('missing, unresolvable and reversed lifetimes stay unknown', () => {
    const graph = makeV6Graph();
    graph.resources = graph.resources!.map(resource => ({...resource, firstNode: undefined}));
    const aliasing = buildMemoryAliasing(graph);
    expect(
      aliasing.allocations
        .flatMap(allocation => allocation.resources)
        .every(
          resource =>
            resource.lifetime === 'unknown' &&
            resource.firstIndex === undefined &&
            resource.lastIndex === undefined
        )
    ).toBe(true);
    expect(aliasing.totalsKnown).toBe(false);
    expect(aliasing.savingsKnown).toBe(true);
    graph.resources = [
      {
        id: 'bad',
        kind: 'buffer',
        bytes: 12,
        ownership: 'transient',
        firstNode: 'absent',
        lastNode: 'node-1'
      }
    ];
    expect(buildMemoryAliasing(graph).allocations[0]).toMatchObject({
      allocation: 'unknown',
      allocationKnown: false,
      resources: [{lifetime: 'unknown'}]
    });
    graph.resources = [{...graph.resources[0], firstNode: 'node-9', lastNode: 'node-1'}];
    expect(buildMemoryAliasing(graph).allocations[0].resources[0].lifetime).toBe('unknown');
  });

  test('absent resource metadata differs from a recorded empty allocation set', () => {
    const graph = makeV6Graph();
    graph.resources = undefined;
    expect(buildMemoryAliasing(graph)).toMatchObject({totalsKnown: false, savingsKnown: false});
    graph.resources = [];
    expect(buildMemoryAliasing(graph)).toMatchObject({
      totalsKnown: true,
      savingsKnown: true,
      peakTransientBytes: 0,
      aliasedSavingsBytes: 0
    });
  });

  test('peak uses schedule positions, counts an allocation once, and extends retained storage', () => {
    const graph = makeV6Graph();
    graph.resources = [
      {
        id: 'early',
        kind: 'buffer',
        bytes: 64,
        ownership: 'transient',
        allocation: 'early',
        allocationBytes: 64,
        firstNode: 'node-0',
        lastNode: 'node-1',
        retained: true
      },
      {
        id: 'late',
        kind: 'buffer',
        bytes: 128,
        ownership: 'transient',
        allocation: 'late',
        allocationBytes: 128,
        firstNode: 'node-38',
        lastNode: 'node-39'
      }
    ];
    expect(buildMemoryAliasing(graph).peakTransientBytes).toBe(192);
    graph.resources = graph.resources.map(resource => ({...resource, retained: false}));
    expect(buildMemoryAliasing(graph).peakTransientBytes).toBe(128);
  });
});
