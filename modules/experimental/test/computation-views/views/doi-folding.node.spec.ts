// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {foldGraphByDegreeOfInterest} from '../../../src/computation-views/views/doi-folding';
import {makeV6Graph} from '../fixtures/v6-command-graph';

describe('degree-of-interest folding', () => {
  test('preserves a leaf partition, budget, deep focus, loop multiplicity and typed dependencies', () => {
    const graph = makeV6Graph();
    const result = foldGraphByDegreeOfInterest(graph, {focus: 'node-2'});
    expect(graph.nodes).toHaveLength(40);
    expect(result.units.length).toBeLessThanOrEqual(24);
    expect(result.focusUnitId).toBe('node:node-2');
    expect(result.units.some(unit => unit.id === 'group:scan/level-0/upsweep')).toBe(false);
    expect(result.units.find(unit => unit.id === result.focusUnitId)?.tier).toBe('subject');
    expect(result.units.find(unit => unit.id === 'node:node-1')?.tier).toBe('input');
    expect(result.units.find(unit => unit.id === 'node:node-8')?.tier).toBe('context');
    const loop = result.units.find(unit => unit.kind === 'loop')!;
    expect(loop.label).toBe('relax ×10');
    expect(loop.memberIds).toHaveLength(10);
    expect(loop.tier).toBe('input');
    expect(
      result.edges.find(edge => edge.from === 'node:node-2' && edge.to === loop.id)
    ).toMatchObject({
      kinds: {war: 2, waw: 1, explicit: 1, indirect: 1, dataflow: 1},
      resources: ['scratch-b', 'scratch-a']
    });
    expect(result.edges.every(edge => edge.from !== edge.to)).toBe(true);
    expect(result.units.flatMap(unit => unit.memberIds).sort()).toEqual(
      graph.nodes.map(node => node.id).sort()
    );
    expect(result.hiddenNodeCount).toBe(
      40 - result.units.filter(unit => unit.kind === 'node').length
    );
    expect(result).toEqual(foldGraphByDegreeOfInterest(graph, {focus: 'node-2'}));
  });

  test('mandatory focus and reader expansions may exceed the budget without hiding their leaves', () => {
    const focused = foldGraphByDegreeOfInterest(makeV6Graph(), {focus: 'node-2', budget: 2});
    expect(focused.units.length).toBeGreaterThan(2);
    expect(focused.focusUnitId).toBe('node:node-2');
    const expanded = foldGraphByDegreeOfInterest(makeV6Graph(), {
      budget: 2,
      expanded: ['scan/level-0/upsweep']
    });
    expect(expanded.units.filter(unit => unit.kind === 'node')).toHaveLength(10);
    expect(expanded.hiddenNodeCount).toBe(30);
  });

  test('focus unfolds its loop, while disabling loop folding exposes each iteration', () => {
    const focused = foldGraphByDegreeOfInterest(makeV6Graph(), {focus: 'node-24', budget: 2});
    expect(focused.focusUnitId).toBe('node:node-24');
    expect(focused.units.filter(unit => unit.kind === 'loop')).toHaveLength(0);
    const unfolded = foldGraphByDegreeOfInterest(makeV6Graph(), {foldLoops: false, budget: 100});
    expect(unfolded.units).toHaveLength(40);
    expect(unfolded.hiddenNodeCount).toBe(0);
    expect(unfolded.units.every(unit => unit.tier === 'subject')).toBe(true);
  });

  test('group focus remains visible, with namespace distance and publication/writer bonuses', () => {
    const result = foldGraphByDegreeOfInterest(makeV6Graph(), {focus: 'group:finish', budget: 3});
    const finish = result.units.find(unit => unit.id === 'group:finish')!;
    expect(result.focusUnitId).toBe(finish.id);
    expect(finish.doi).toBe(1.5);
    expect(finish.tier).toBe('subject');
    expect(result.units.find(unit => unit.id === 'group:scan')?.doi).toBe(-2);
  });

  test('reader expansion reaches a loop through folded ancestors, and cross-namespace loops partition once', () => {
    const graph = makeV6Graph();
    const expanded = foldGraphByDegreeOfInterest(graph, {
      budget: 2,
      expanded: ['loop:relaxation/relax']
    });
    expect(expanded.units.filter(unit => unit.kind === 'node')).toHaveLength(10);
    expect(expanded.units.some(unit => unit.kind === 'loop')).toBe(false);
    graph.nodes = graph.nodes.map(node =>
      node.loop ? {...node, namespace: `solver/iteration-${node.loop.iteration}/body`} : node
    );
    const folded = foldGraphByDegreeOfInterest(graph, {budget: 24});
    expect(folded.units.find(unit => unit.kind === 'loop')?.memberIds).toHaveLength(10);
    expect(new Set(folded.units.flatMap(unit => unit.memberIds)).size).toBe(40);
    expect(folded.units.flatMap(unit => unit.memberIds)).toHaveLength(40);
  });

  test('bounds a wide unnamespaced schedule while preserving every leaf and dependency count', () => {
    const graph = makeV6Graph();
    graph.nodes = graph.nodes.map(node => ({...node, namespace: '', loop: undefined}));
    const folded = foldGraphByDegreeOfInterest(graph, {budget: 8});
    expect(folded.units.length).toBeLessThanOrEqual(8);
    expect(folded.units.flatMap(unit => unit.memberIds)).toEqual(graph.nodes.map(node => node.id));
    expect(folded.units.some(unit => unit.label.startsWith('Schedule range:'))).toBe(true);
    const owners = new Map(
      folded.units.flatMap(unit => unit.memberIds.map(member => [member, unit.id]))
    );
    expect(
      folded.edges.reduce(
        (sum, edge) => sum + Object.values(edge.kinds).reduce((count, value) => count + value, 0),
        0
      )
    ).toBe(graph.edges.filter(edge => owners.get(edge.from) !== owners.get(edge.to)).length);
  });
  test('greedy expansion prefers DOI then schedule order, even when one candidate cannot fit', () => {
    const graph = makeV6Graph();
    const result = foldGraphByDegreeOfInterest(graph, {budget: 12});
    expect(result.units.some(unit => unit.id === 'node:node-39')).toBe(true);
    expect(result.units.length).toBeLessThanOrEqual(12);
    expect(result.units.map(unit => unit.order)).toEqual(
      result.units.map(unit => unit.order).sort((left, right) => left - right)
    );
  });
});
