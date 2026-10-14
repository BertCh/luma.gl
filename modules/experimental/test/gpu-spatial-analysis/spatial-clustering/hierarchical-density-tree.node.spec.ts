// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createParallelDensityTree,
  createSerialDensityTree,
  selectDensityTreeParallel,
  selectDensityTreeSerial,
  type DensityTreeEdge
} from './hierarchical-density-tree-model';

it('parallel parent splicing and Euler ranking reproduce Kruskal under interleaved atomics', () => {
  let randomState = 982451653;
  const random = (limit: number) => {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    return randomState % limit;
  };
  for (let trial = 0; trial < 200; trial++) {
    const rows = 2 + random(127);
    const edges: DensityTreeEdge[] = Array.from({length: rows - 1}, (_, index) => [
      index + 1,
      random(index + 1),
      1
    ]);
    for (let index = edges.length - 1; index > 0; index--) {
      const other = random(index + 1);
      [edges[index], edges[other]] = [edges[other], edges[index]];
    }
    // A sorted forest also exercises isolated invalid rows and empty slots.
    if (trial % 3 === 0) edges.length = random(edges.length);
    expect(createParallelDensityTree(rows, edges, randomState)).toEqual(
      createSerialDensityTree(rows, edges)
    );
  }
});

it('parallel dendrogram handles empty, singleton, chain and star trees without depth buckets', () => {
  for (const rows of [1, 2, 257, 2049]) {
    for (const star of [false, true]) {
      const edges: DensityTreeEdge[] = Array.from({length: rows - 1}, (_, index) => [
        star ? 0 : index,
        index + 1,
        Math.floor(index / 7)
      ]);
      expect(createParallelDensityTree(rows, edges)).toEqual(createSerialDensityTree(rows, edges));
    }
  }
});

it('heavy-path EOM agrees with serial EOM on balanced and deep comb trees, ties and infinity', () => {
  for (const comb of [false, true]) {
    for (const leaves of [1, 2, 17, 513]) {
      const parents = Array<number>(2 * leaves - 1).fill(0xffffffff);
      const sizes: number[] = parents.map((_, cluster) => (cluster < leaves ? 2 : 0));
      const pending = Array.from({length: leaves}, (_, cluster) => cluster);
      for (let cluster = leaves; cluster < parents.length; cluster++) {
        const first = pending.shift()!;
        const second = pending.shift()!;
        parents[first] = parents[second] = cluster;
        sizes[cluster] = sizes[first] + sizes[second];
        if (comb) pending.unshift(cluster);
        else pending.push(cluster);
      }
      for (const mode of ['zero', 'ties', 'varied', 'infinity']) {
        const stability = parents.map((_, cluster) =>
          mode === 'zero'
            ? 0
            : mode === 'ties'
              ? sizes[cluster]
              : mode === 'infinity'
                ? Infinity
                : (cluster * 104729) % 4096
        );
        const input = {parents, sizes, stability};
        expect(selectDensityTreeParallel(input), `${comb}/${leaves}/${mode}`).toEqual(
          selectDensityTreeSerial(input)
        );
      }
    }
  }
});
