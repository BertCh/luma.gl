// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {layerGraph, foldLoops} from '../../../src/computation-views/views/graph-layering';
import {makeRecord, makeNode} from '../fixtures/v5-record';

describe('graph layering', () => {
  test('uses longest paths, stable input order, and counts ignored backwards edges', () => {
    const result = layerGraph(
      ['a', 'b', 'c', 'd'].map(id => ({id})),
      [
        {from: 'a', to: 'c'},
        {from: 'c', to: 'd'},
        {from: 'a', to: 'd'},
        {from: 'd', to: 'b'},
        {from: 'b', to: 'b'},
        {from: 'missing', to: 'd'}
      ]
    );
    expect(result.layers).toEqual([['a', 'b'], ['c'], ['d']]);
    expect(result.depth).toBe(3);
    expect(result.backEdgeCount).toBe(2);
    expect(result.order.get('c')).toBe(2);
    expect(layerGraph([], []).depth).toBe(0);
  });
  test('folds only matching loop identities and names, deduplicating edge kinds', () => {
    const graph = makeRecord().graphs[0];
    const folded = foldLoops({
      ...graph,
      edges: [
        ...graph.edges,
        {from: 'relax-0', to: 'finish', kind: 'raw'},
        {from: 'relax-0', to: 'finish', kind: 'raw'}
      ]
    });
    expect(folded.nodes[0]).toMatchObject({
      id: 'loop:solve/relax',
      memberIds: ['relax-0', 'relax-1'],
      iterationCount: 2,
      maximumIterations: 16
    });
    expect(folded.edges).toEqual([
      {from: 'loop:solve/relax', to: 'finish', kinds: ['explicit', 'raw']}
    ]);
    expect(
      foldLoops({
        ...graph,
        nodes: [makeNode(), makeNode({id: 'other', loop: {id: 'other-loop', iteration: 0}})]
      }).nodes
    ).toHaveLength(2);
  });
});
