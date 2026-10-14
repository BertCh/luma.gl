// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Sorted MST edge (endpoints and weight); exclusively a test input. */
export type DensityTreeEdge = readonly [number, number, number];
const NONE = 0xffffffff;

/** Serial Kruskal oracle, independent of the concurrent parent splice. */
export function createSerialDensityTree(rows: number, edges: readonly DensityTreeEdge[]) {
  const parents = Array<number>(rows * 2).fill(NONE);
  const sizes = Array.from({length: rows * 2}, (_, index) => Number(index < rows));
  const children = Array.from({length: rows * 2}, () => [NONE, NONE]);
  const roots = Array.from({length: rows}, (_, index) => index);
  const tops = [...roots];
  const find = (point: number) => {
    let current = point;
    while (roots[current] !== current) current = roots[current];
    return current;
  };
  edges.forEach(([first, second], rank) => {
    const firstRoot = find(first);
    const secondRoot = find(second);
    const node = rows + rank;
    children[node] = [tops[firstRoot], tops[secondRoot]].sort((left, right) => left - right);
    for (const child of children[node]) parents[child] = node;
    sizes[node] = sizes[tops[firstRoot]] + sizes[tops[secondRoot]];
    roots[secondRoot] = firstRoot;
    tops[firstRoot] = node;
  });
  return {parents, sizes, children};
}

/** Interleaves individual atomicMin steps, then models the GPU Euler-tour doubling passes. */
export function createParallelDensityTree(
  rows: number,
  edges: readonly DensityTreeEdge[],
  seed = 1
) {
  const parents = Array<number>(rows * 2).fill(NONE);
  const tasks = edges.flatMap(([first, second], rank) => [
    [first, rows + rank],
    [second, rows + rank]
  ]);
  let randomState = seed;
  while (tasks.length) {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    const slot = randomState % tasks.length;
    const [first, second] = tasks[slot];
    tasks[slot] = tasks[tasks.length - 1];
    tasks.pop();
    if (first === second) continue;
    const lower = Math.min(first, second);
    const higher = Math.max(first, second);
    const previous = parents[lower];
    parents[lower] = Math.min(previous, higher);
    if (previous !== NONE && previous !== higher) tasks.push([previous, higher]);
  }
  const children = Array.from({length: rows * 2}, () => [NONE, NONE]);
  parents.forEach((parent, node) => {
    if (parent !== NONE) {
      const current = children[parent].filter(child => child !== NONE);
      current.push(node);
      children[parent] = current.sort((left, right) => left - right);
    }
  });
  let tour = parents.flatMap((parent, node) => {
    const first = children[node][0];
    const enter = first === NONE ? 2 * node + 1 : 2 * first;
    const exit =
      parent === NONE
        ? NONE
        : children[parent][0] === node
          ? 2 * children[parent][1]
          : 2 * parent + 1;
    return [
      [enter, Number(first === NONE && node < rows)],
      [exit, 0]
    ];
  });
  while (tour.some(([next]) => next !== NONE)) {
    tour = tour.map(([next, total]) =>
      next === NONE ? [next, total] : [tour[next][0], total + tour[next][1]]
    );
  }
  const sizes = parents.map((_, node) => tour[2 * node][1] - tour[2 * node + 1][1]);
  return {parents, sizes, children};
}

/** Condensed binary tree test input (IDs ascend from children to parents). */
export type DensitySelectionInput = {parents: number[]; sizes: number[]; stability: number[]};

/** Serial pre-change EOM, including float32 accumulation and exact parent-wins ties. */
export function selectDensityTreeSerial(input: DensitySelectionInput): number[] {
  const childBest = input.parents.map(() => 0);
  const selected = input.parents.map(() => false);
  input.parents.forEach((parent, cluster) => {
    if (parent === NONE) return;
    selected[cluster] = !(childBest[cluster] > input.stability[cluster]);
    const best = selected[cluster] ? input.stability[cluster] : childBest[cluster];
    childBest[parent] = Math.fround(childBest[parent] + best);
  });
  return resolveSelection(input.parents, selected);
}

function resolveSelection(parents: number[], selected: boolean[]): number[] {
  const representatives = parents.map(() => NONE);
  for (let cluster = parents.length - 1; cluster >= 0; cluster--) {
    const parent = parents[cluster];
    if (parent !== NONE && representatives[parent] !== NONE) {
      representatives[cluster] = representatives[parent];
    } else if (selected[cluster]) {
      representatives[cluster] = cluster;
    }
  }
  return representatives;
}

/** Mirrors the heavy/light max-plus contraction using float32 at every arithmetic step. */
export function selectDensityTreeParallel(input: DensitySelectionInput): number[] {
  const children = input.parents.map(() => [] as number[]);
  input.parents.forEach((parent, cluster) => {
    if (parent !== NONE) children[parent].push(cluster);
  });
  const heavy = children.map(([first, second]) =>
    first === undefined ? NONE : input.sizes[second] > input.sizes[first] ? second : first
  );
  const depth = input.parents.map(() => 0);
  for (let cluster = depth.length - 1; cluster >= 0; cluster--) {
    const parent = input.parents[cluster];
    if (parent !== NONE) depth[cluster] = depth[parent] + Number(heavy[parent] !== cluster);
  }
  const best = input.parents.map(() => 0);
  for (let level = Math.max(...depth); level >= 0; level--) {
    let functions = input.parents.map((_, cluster) => {
      const light = children[cluster].find(child => child !== heavy[cluster]);
      return [heavy[cluster], input.stability[cluster], light === undefined ? 0 : best[light]];
    });
    while (functions.some(([next], cluster) => depth[cluster] === level && next !== NONE)) {
      functions = functions.map(([next, floorValue, addition], cluster) => {
        if (depth[cluster] !== level || next === NONE) return [next, floorValue, addition];
        const [successor, childFloor, childAddition] = functions[next];
        return [
          successor,
          Math.max(floorValue, Math.fround(addition + childFloor)),
          Math.fround(addition + childAddition)
        ];
      });
    }
    functions.forEach(([, floorValue], cluster) => {
      if (depth[cluster] === level) best[cluster] = floorValue;
    });
  }
  const selected = children.map((childIds, cluster) => {
    const childBest = childIds.length ? Math.fround(best[childIds[0]] + best[childIds[1]]) : 0;
    return input.parents[cluster] !== NONE && !(childBest > input.stability[cluster]);
  });
  return resolveSelection(input.parents, selected);
}

/** Condensed tree / f32 stability derived from a serial dendrogram, for stage parity fixtures. */
export function createDensitySelectionInput(
  rows: number,
  edges: readonly DensityTreeEdge[],
  minimumClusterSize: number
): DensitySelectionInput {
  const tree = createSerialDensityTree(rows, edges);
  const isBig = (node: number) => node !== NONE && tree.sizes[node] >= minimumClusterSize;
  const isStart = (node: number) =>
    isBig(node) && (tree.parents[node] === NONE || tree.children[tree.parents[node]].every(isBig));
  const starts = tree.parents.map((_, node) => node).filter(isStart);
  const clusterIds = new Map(starts.map((node, cluster) => [node, cluster]));
  const startOf = (node: number) => {
    let current = node;
    while (current !== NONE && !isStart(current)) current = tree.parents[current];
    return current;
  };
  const lambda = (node: number) => {
    const weight = node < rows ? 0 : Math.fround(edges[node - rows][2]);
    return weight > 0 ? Math.fround(1 / weight) : Math.fround(3.402823e38);
  };
  const parents = starts.map(node =>
    tree.parents[node] === NONE ? NONE : clusterIds.get(startOf(tree.parents[node]))!
  );
  const birth = starts.map(node => (tree.parents[node] === NONE ? 0 : lambda(tree.parents[node])));
  const stability = starts.map(() => 0);
  for (let node = 0; node < rows; node++) {
    let lowest = node;
    while (lowest !== NONE && !isBig(lowest)) lowest = tree.parents[lowest];
    if (lowest === NONE) continue;
    const cluster = clusterIds.get(startOf(lowest))!;
    stability[cluster] = Math.fround(
      stability[cluster] + Math.fround(lambda(lowest) - birth[cluster])
    );
  }
  parents.forEach((parent, cluster) => {
    if (parent !== NONE)
      stability[parent] = Math.fround(
        stability[parent] +
          Math.fround(tree.sizes[starts[cluster]] * Math.fround(birth[cluster] - birth[parent]))
      );
  });
  return {parents, sizes: starts.map(node => tree.sizes[node]), stability};
}
