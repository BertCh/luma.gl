// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getAdjacency,
  getCSRFromAdjacency,
  getTieKey,
  getWithinSumOfSquares,
  isRegionConnected,
  isRemovalSafeMirror,
  runBatchedAzpMirror,
  runContiguityMirror,
  selectIndependentMoves,
  type AzpAdjacency
} from './azp-batched-mirror';
import statisticsAzpFixture from '../../oracles/fixtures/statistics-azp.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {createRandomValues} from './large-lattice-invariants';

/**
 * Largest accepted ratio of the batched greedy objective to spopt AZP's. Both are local searches
 * from the same start and end at different local optima; the float64 mirror measured ratios of
 * 0.83 to 1.33 on the six fixture starts, and 1.5 leaves margin for the float32 GPU path reaching
 * yet another local optimum.
 */
const SPOPT_QUALITY_FACTOR = 1.5;

/** Small deterministic generator (mulberry32) so failures reproduce. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

type GraphKind = 'rook' | 'queen' | 'rook-with-long-links';

function createGraph(kind: GraphKind, width: number, height: number, random: () => number) {
  const sets = Array.from({length: width * height}, () => new Set<number>());
  const link = (a: number, b: number) => {
    if (a !== b) {
      sets[a].add(b);
      sets[b].add(a);
    }
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const row = y * width + x;
      if (x + 1 < width) link(row, row + 1);
      if (y + 1 < height) link(row, row + width);
      if (kind === 'queen') {
        if (x + 1 < width && y + 1 < height) link(row, row + width + 1);
        if (x > 0 && y + 1 < height) link(row, row + width - 1);
      }
      if (kind === 'rook-with-long-links' && random() < 0.05) {
        link(row, Math.floor(random() * width * height));
      }
    }
  }
  return sets.map(set => [...set].sort((a, b) => a - b));
}

/** Multi-source breadth-first growth: every region is connected by construction. */
function createRandomPartition(adjacency: AzpAdjacency, regionCount: number, random: () => number) {
  const labels = new Array<number>(adjacency.length).fill(-1);
  const frontier: number[][] = [];
  for (let region = 0; region < regionCount; ) {
    const row = Math.floor(random() * adjacency.length);
    if (labels[row] === -1) {
      labels[row] = region;
      frontier.push([row]);
      region++;
    }
  }
  let remaining = adjacency.length - regionCount;
  while (remaining > 0) {
    const region = Math.floor(random() * regionCount);
    const candidates = frontier[region].filter(row => adjacency[row].some(n => labels[n] === -1));
    frontier[region] = candidates;
    if (candidates.length === 0) {
      if (frontier.every(list => list.length === 0)) break;
      continue;
    }
    const from = candidates[Math.floor(random() * candidates.length)];
    const open = adjacency[from].filter(n => labels[n] === -1);
    const row = open[Math.floor(random() * open.length)];
    labels[row] = region;
    frontier[region].push(row);
    remaining--;
  }
  return labels;
}

const KINDS: GraphKind[] = ['rook', 'queen', 'rook-with-long-links'];

it('the local articulation test never accepts a move that disconnects its region', () => {
  let accepted = 0;
  let refusedSafe = 0;
  let refusedUnsafe = 0;
  for (const [kindIndex, kind] of KINDS.entries()) {
    for (let trial = 0; trial < 12; trial++) {
      const random = createRandom(1000 * kindIndex + trial);
      const adjacency = createGraph(kind, 14, 11, random);
      const regionCount = 3 + (trial % 9);
      const labels = createRandomPartition(adjacency, regionCount, random);
      for (let region = 0; region < regionCount; region++) {
        expect(isRegionConnected(adjacency, labels, region)).toBe(true);
      }
      for (let row = 0; row < adjacency.length; row++) {
        const source = labels[row];
        const sourceSize = labels.filter(label => label === source).length;
        if (sourceSize < 2) continue;
        const safe = isRemovalSafeMirror(adjacency, labels, row);
        const connectedAfter = isRegionConnected(adjacency, labels, source, row);
        if (safe) {
          accepted++;
          expect(connectedAfter, `${kind} trial ${trial} row ${row}`).toBe(true);
        } else if (connectedAfter) {
          refusedSafe++;
        } else {
          refusedUnsafe++;
        }
      }
    }
  }
  // The test is sufficient, not necessary: report how many safe moves it refuses.
  const refusedShare = refusedSafe / (refusedSafe + accepted);
  console.log(
    `AZP local articulation test: accepted ${accepted}, refused-but-safe ${refusedSafe} ` +
      `(${(100 * refusedShare).toFixed(2)}% of safe moves), refused-and-unsafe ${refusedUnsafe}`
  );
  expect(accepted).toBeGreaterThan(1000);
  expect(refusedUnsafe).toBeGreaterThan(0);
  // Lattice regions reconnect within two steps almost always; a regression would show here.
  expect(refusedShare).toBeLessThan(0.1);
});

it('the local articulation test refuses when its bounds are exceeded and decides small cases', () => {
  // A path 0-1-2: the middle row is an articulation point, an end row is a leaf.
  const path: AzpAdjacency = [[1], [0, 2], [1]];
  expect(isRemovalSafeMirror(path, [0, 0, 0], 1)).toBe(false);
  expect(isRemovalSafeMirror(path, [0, 0, 0], 0)).toBe(true);
  // A 4-cycle: removing any row leaves a path.
  const cycle: AzpAdjacency = [
    [1, 3],
    [0, 2],
    [1, 3],
    [0, 2]
  ];
  expect(isRemovalSafeMirror(cycle, [0, 0, 0, 0], 0)).toBe(true);
  // A ring of 6 around a hub: the ring members connect through each other, but a capacity of 5
  // rows overflows, and the test then answers "not proven" instead of guessing.
  const wheel: AzpAdjacency = [
    [1, 2, 3, 4, 5, 6],
    [0, 2, 6],
    [0, 1, 3],
    [0, 2, 4],
    [0, 3, 5],
    [0, 4, 6],
    [0, 5, 1]
  ];
  const labels = new Array(7).fill(0);
  expect(isRemovalSafeMirror(wheel, labels, 0)).toBe(true);
  expect(isRemovalSafeMirror(wheel, labels, 0, 5)).toBe(false);
  expect(isRemovalSafeMirror(wheel, labels, 0, 128, 2)).toBe(false);
  // Two arms joined only by a path longer than two steps from the removed row: refused though safe.
  const longLoop: AzpAdjacency = [
    [1, 5], // 0: the removed row, neighbors 1 and 5
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 0]
  ];
  expect(isRegionConnected(longLoop, [0, 0, 0, 0, 0, 0], 0, 0)).toBe(true);
  expect(isRemovalSafeMirror(longLoop, [0, 0, 0, 0, 0, 0], 0)).toBe(false);
});

it('the tie key is a bijection, so one move holds the largest (gain, key) of each region', () => {
  for (const seed of [0, 1, 0xdeadbeef]) {
    const keys = new Set<number>();
    for (let row = 0; row < 50000; row++) keys.add(getTieKey(row, seed));
    expect(keys.size).toBe(50000);
  }
});

it('batched selection is independent, contains the best move, and equal gains resolve by key', () => {
  for (let trial = 0; trial < 200; trial++) {
    const random = createRandom(trial);
    const regionCapacity = 2 + Math.floor(random() * 12);
    const moves = Array.from({length: 5 + Math.floor(random() * 40)}, (_, row) => {
      const source = Math.floor(random() * regionCapacity);
      let destination = Math.floor(random() * regionCapacity);
      if (destination === source) destination = (destination + 1) % regionCapacity;
      // Few distinct gains, so ties are common.
      return {row, source, destination, gain: 1 + Math.floor(random() * 4)};
    });
    const seed = trial * 7919;
    const selected = selectIndependentMoves(moves, seed, regionCapacity);
    const touched = selected.flatMap(move => [move.source, move.destination]);
    expect(new Set(touched).size).toBe(touched.length);
    const best = moves.reduce((winner, move) =>
      move.gain > winner.gain ||
      (move.gain === winner.gain && getTieKey(move.row, seed) > getTieKey(winner.row, seed))
        ? move
        : winner
    );
    expect(selected).toContainEqual(best);
  }
});

it('batched rounds lower the objective strictly, add up exactly, and keep every region connected', () => {
  let totalRounds = 0;
  let totalMoves = 0;
  let singleMoveRounds = 0;
  for (const [kindIndex, kind] of KINDS.entries()) {
    for (let trial = 0; trial < 4; trial++) {
      const random = createRandom(77 + 100 * kindIndex + trial);
      const adjacency = createGraph(kind, 18, 14, random);
      const regionCount = 6 + 3 * trial;
      const initial = createRandomPartition(adjacency, regionCount, random);
      const values = createRandomValues(adjacency.length, 5 + trial);
      const mirror = runBatchedAzpMirror({
        adjacency,
        values,
        labels: initial,
        regionCapacity: regionCount,
        seed: trial,
        minimumRegionSize: trial % 2 ? 2 : 1,
        maximumRounds: 256
      });
      expect(mirror.converged, `${kind} ${trial}`).toBe(true);
      // Replay the rounds on a copy to check contiguity after every round.
      const labels = [...initial];
      for (const round of mirror.rounds) {
        for (const move of round.selected) labels[move.row] = move.destination;
        for (let region = 0; region < regionCount; region++) {
          expect(isRegionConnected(adjacency, labels, region), `${kind} ${trial}`).toBe(true);
        }
        if (round.selected.length === 0) break;
        // Independent selection: gains are exactly additive and the objective falls strictly.
        const gain = round.selected.reduce((sum, move) => sum + move.gain, 0);
        expect(round.objectiveAfter).toBeLessThan(round.objectiveBefore);
        expect(round.objectiveBefore - round.objectiveAfter).toBeCloseTo(gain, 6);
        totalMoves += round.selected.length;
        if (round.selected.length === 1) singleMoveRounds++;
      }
      totalRounds += mirror.rounds.length;
      expect(labels).toEqual(mirror.labels);
      // At the local optimum no move passes the test and improves.
      expect(mirror.rounds.at(-1)!.moves).toHaveLength(0);
    }
  }
  console.log(
    `AZP batched mirror: ${totalMoves} moves in ${totalRounds} rounds ` +
      `(${(totalMoves / totalRounds).toFixed(1)} per round, ${singleMoveRounds} single-move rounds)`
  );
});

it('the mirrored objective helper matches a direct within-region sum of squares', () => {
  const values = [1, 2, 4, 8];
  const labels = [0, 0, 1, 1];
  expect(getWithinSumOfSquares(values, labels, 2)).toBeCloseTo(0.5 + 8, 10);
  const {offsets, neighbors} = getCSRFromAdjacency([[1], [0, 2], [1, 3], [2]]);
  expect(getAdjacency(offsets, neighbors)).toEqual([[1], [0, 2], [1, 3], [2]]);
});

it('the batched search reaches the quality of spopt AZP from the same starts (spopt oracle)', () => {
  const fixture = decodeOracleFixture<{
    instances: {
      name: string;
      regionCount: number;
      offsets: number[];
      neighbors: number[];
      values: number[];
      initialLabels: number[];
      initialObjective: number;
      spoptObjective: number;
    }[];
  }>(statisticsAzpFixture);
  for (const instance of fixture.instances) {
    const adjacency = getAdjacency(
      Uint32Array.from(instance.offsets),
      Uint32Array.from(instance.neighbors)
    );
    const mirror = runBatchedAzpMirror({
      adjacency,
      values: instance.values,
      labels: instance.initialLabels,
      regionCapacity: instance.regionCount,
      maximumRounds: 256
    });
    const objective = getWithinSumOfSquares(instance.values, mirror.labels, instance.regionCount);
    console.log(
      `${instance.name}: initial ${instance.initialObjective.toFixed(2)}, ` +
        `batched ${objective.toFixed(2)} in ${mirror.rounds.length} rounds, ` +
        `spopt ${instance.spoptObjective.toFixed(2)}`
    );
    expect(mirror.converged).toBe(true);
    expect(objective).toBeLessThanOrEqual(instance.initialObjective);
    expect(objective).toBeLessThanOrEqual(SPOPT_QUALITY_FACTOR * instance.spoptObjective);
  }
});

/** Number of connected components of each region, by full search (the oracle). */
function countExtraComponents(adjacency: AzpAdjacency, labels: number[], regionCapacity: number) {
  let extra = 0;
  const seen = new Set<number>();
  for (let region = 0; region < regionCapacity; region++) {
    let components = 0;
    for (let row = 0; row < labels.length; row++) {
      if (labels[row] !== region || seen.has(row)) continue;
      components++;
      const stack = [row];
      seen.add(row);
      while (stack.length) {
        for (const neighbor of adjacency[stack.pop()!]) {
          if (labels[neighbor] === region && !seen.has(neighbor)) {
            seen.add(neighbor);
            stack.push(neighbor);
          }
        }
      }
    }
    extra += Math.max(components - 1, 0);
  }
  return extra;
}

it('the initial contiguity check counts extra components exactly within the round bound', () => {
  // A snake: one long region winding through a lattice (a path of 400 rows), with labels in
  // decreasing order along the path, the slowest case for min-label propagation.
  const width = 20;
  const snakeRows: number[] = [];
  for (let y = 0; y < width; y++) {
    for (let x = 0; x < width; x++) snakeRows.push(y * width + (y % 2 ? width - 1 - x : x));
  }
  const path = Array.from({length: width * width}, () => [] as number[]);
  for (let i = 0; i + 1 < snakeRows.length; i++) {
    path[snakeRows[i]].push(snakeRows[i + 1]);
    path[snakeRows[i + 1]].push(snakeRows[i]);
  }
  const descending = new Array<number>(width * width).fill(0);
  const connected = runContiguityMirror(path, descending, 1);
  expect(connected.invalidCount).toBe(0);
  expect(connected.activeRounds).toBeLessThanOrEqual(2 * Math.ceil(Math.log2(width * width)) + 8);
  // Cut the snake in the middle: the region has two components, one extra.
  const cut = descending.slice();
  cut[snakeRows[200]] = 1;
  expect(runContiguityMirror(path, cut, 2).invalidCount).toBe(
    countExtraComponents(path, cut, 2) + 0
  );
  // A bound of one round cannot converge on the snake and must be reported as invalid, not pass.
  expect(runContiguityMirror(path, descending, 1, 1).invalidCount).toBeGreaterThan(0);
  // Random (mostly disconnected) partitions: the count equals the number of extra components.
  let maximumRounds = 0;
  for (let trial = 0; trial < 60; trial++) {
    const random = createRandom(31 + trial);
    const adjacency = createGraph(KINDS[trial % 3], 16, 12, random);
    const regionCapacity = 2 + (trial % 7);
    const labels = adjacency.map(() => Math.floor(random() * regionCapacity));
    const result = runContiguityMirror(adjacency, labels, regionCapacity);
    // Each extra component contributes exactly one extra root and no unconverged edge.
    expect(result.invalidCount, `trial ${trial}`).toBe(
      countExtraComponents(adjacency, labels, regionCapacity)
    );
    maximumRounds = Math.max(maximumRounds, result.activeRounds);
    // Connected partitions pass.
    const grown = createRandomPartition(adjacency, regionCapacity, random);
    expect(runContiguityMirror(adjacency, grown, regionCapacity).invalidCount).toBe(0);
  }
  console.log(
    `AZP contiguity mirror: most active hook rounds on random partitions ${maximumRounds}`
  );
});

/** Relabels the rows of a graph by a permutation (`newId[oldRow]`). */
function relabelGraph(adjacency: AzpAdjacency, newId: number[]): number[][] {
  const result = adjacency.map(() => [] as number[]);
  adjacency.forEach((list, row) => (result[newId[row]] = list.map(n => newId[n])));
  return result;
}

function createPathGraph(order: number[], count: number): number[][] {
  const lists = Array.from({length: count}, () => [] as number[]);
  for (let i = 0; i + 1 < order.length; i++) {
    lists[order[i]].push(order[i + 1]);
    lists[order[i + 1]].push(order[i]);
  }
  return lists;
}

it('the contiguity check converges within the round bound on adversarial row orders', () => {
  const cases: {name: string; adjacency: number[][]}[] = [];
  for (const count of [2048, 8192, 20000]) {
    const random = createRandom(count);
    const identity = Array.from({length: count}, (_, i) => i);
    const shuffled = identity.slice().sort(() => random() - 0.5);
    // Zigzag ids: the path alternates between the low and the high half.
    const zigzag = identity.map(i => (i % 2 ? count - 1 - (i >> 1) : i >> 1));
    // Valley ids: ids fall to a minimum in the middle and rise again.
    const mid = count >> 1;
    const byDistance = identity
      .slice()
      .sort((p, q) => Math.abs(p - mid) - Math.abs(q - mid) || p - q);
    const valleyOrder = new Array<number>(count);
    byDistance.forEach((position, rank) => (valleyOrder[position] = rank));
    cases.push(
      {name: `path ascending ${count}`, adjacency: createPathGraph(identity, count)},
      {
        name: `path descending ${count}`,
        adjacency: createPathGraph(identity.slice().reverse(), count)
      },
      {name: `path random ids ${count}`, adjacency: createPathGraph(shuffled, count)},
      {name: `path zigzag ids ${count}`, adjacency: createPathGraph(zigzag, count)},
      {name: `path valley ids ${count}`, adjacency: createPathGraph(valleyOrder, count)}
    );
    // Star with the largest id at the centre, and with the smallest.
    for (const center of [count - 1, 0]) {
      const star = Array.from({length: count}, () => [] as number[]);
      for (let row = 0; row < count; row++) {
        if (row !== center) {
          star[row].push(center);
          star[center].push(row);
        }
      }
      cases.push({name: `star centre ${center} ${count}`, adjacency: star});
    }
    // Comb: a spine of sqrt(n) rows, each carrying a tooth of the same length; ids by tooth.
    const side = Math.floor(Math.sqrt(count));
    const comb = Array.from({length: side * side}, () => [] as number[]);
    const link = (a: number, b: number) => {
      comb[a].push(b);
      comb[b].push(a);
    };
    for (let tooth = 0; tooth < side; tooth++) {
      if (tooth + 1 < side) link(tooth, tooth + 1);
      for (let step = 1; step < side; step++) link((step - 1) * side + tooth, step * side + tooth);
    }
    cases.push({name: `comb ${side}`, adjacency: comb});
    // Spiral through a square lattice, ids in spiral order and in random order.
    const width = Math.floor(Math.sqrt(count));
    const spiral: number[] = [];
    let [x, y, dx, dy] = [0, 0, 1, 0];
    const seen = new Set<number>();
    for (let i = 0; i < width * width; i++) {
      spiral.push(y * width + x);
      seen.add(y * width + x);
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= width || ny >= width || seen.has(ny * width + nx)) {
        [dx, dy] = [-dy, dx];
      }
      x += dx;
      y += dy;
    }
    const spiralPath = createPathGraph(spiral, width * width);
    cases.push({name: `spiral ${width}`, adjacency: spiralPath});
    const permutation = Array.from({length: width * width}, (_, i) => i).sort(() => random() - 0.5);
    cases.push({
      name: `spiral random ids ${width}`,
      adjacency: relabelGraph(spiralPath, permutation)
    });
    cases.push({
      name: `rook lattice random ids ${width}`,
      adjacency: relabelGraph(createGraph('rook', width, width, random), permutation)
    });
  }
  let worstRatio = 0;
  const lines: string[] = [];
  for (const {name, adjacency} of cases) {
    const rowCount = adjacency.length;
    const bound = 2 * Math.ceil(Math.log2(rowCount)) + 8;
    const result = runContiguityMirror(adjacency, new Array(rowCount).fill(0), 1);
    expect(result.invalidCount, name).toBe(0);
    expect(result.activeRounds, name).toBeLessThanOrEqual(bound - 4);
    worstRatio = Math.max(worstRatio, result.activeRounds / Math.log2(rowCount));
    lines.push(
      `${name}: ${result.activeRounds} rounds (log2 n = ${Math.log2(rowCount).toFixed(1)})`
    );
  }
  console.log(
    `AZP contiguity adversarial rounds:\n${lines.join('\n')}\nworst rounds / log2 n = ${worstRatio.toFixed(2)}`
  );
});
