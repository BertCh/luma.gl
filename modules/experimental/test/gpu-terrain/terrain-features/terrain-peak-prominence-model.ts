// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Test-only sorted spill graph, preserving stable cell/direction order. */
export function getProminenceEdges(testCase: {
  width: number;
  height: number;
  elevation: (number | string)[];
}) {
  const values = testCase.elevation.map(Number);
  const edges: {first: number; second: number; saddle: number}[] = [];
  for (let cell = 0; cell < values.length; cell++) {
    const row = Math.floor(cell / testCase.width);
    const column = cell % testCase.width;
    for (const [columnOffset, rowOffset] of [
      [1, 0],
      [1, 1],
      [0, 1],
      [-1, 1]
    ]) {
      const nextRow = row + rowOffset;
      const nextColumn = column + columnOffset;
      if (nextRow >= testCase.height || nextColumn < 0 || nextColumn >= testCase.width) continue;
      const neighbor = nextRow * testCase.width + nextColumn;
      if (Number.isFinite(values[cell]) && Number.isFinite(values[neighbor]))
        edges.push({
          first: cell,
          second: neighbor,
          saddle: Math.min(values[cell], values[neighbor])
        });
    }
  }
  edges.sort((first, second) => second.saddle - first.saddle);
  return {values, edges};
}

/** Frozen serial sorted-edge elder union oracle, including exact parentPeak ties. */
export function getSerialProminence(testCase: Parameters<typeof getProminenceEdges>[0]) {
  const {values, edges} = getProminenceEdges(testCase);
  const parents = values.map((_, index) => index);
  const sizes = values.map(() => 1);
  const peaks = [...parents];
  const parentPeaks = values.map(() => 0xffffffff);
  const saddles = values.map(value => (Number.isFinite(value) ? -Infinity : NaN));
  const root = (cell: number): number => {
    while (parents[cell] !== cell) cell = parents[cell];
    return cell;
  };
  for (const edge of edges) {
    let first = root(edge.first);
    let second = root(edge.second);
    if (first === second) continue;
    const firstPeak = peaks[first];
    const secondPeak = peaks[second];
    const firstWins =
      values[firstPeak] > values[secondPeak] ||
      (values[firstPeak] === values[secondPeak] && firstPeak < secondPeak);
    const winner = firstWins ? firstPeak : secondPeak;
    const loser = firstWins ? secondPeak : firstPeak;
    saddles[loser] = edge.saddle;
    parentPeaks[loser] = winner;
    if (sizes[first] < sizes[second]) [first, second] = [second, first];
    parents[second] = first;
    sizes[first] += sizes[second];
    peaks[first] = winner;
  }
  return {
    saddles,
    parentPeaks,
    prominence: saddles.map((saddle, cell) => Math.fround(values[cell] - saddle))
  };
}

/** Randomly interleaved atomic-min splicing and scheduled ancestor-doubling reduction. */
export function getParallelProminence(
  testCase: Parameters<typeof getProminenceEdges>[0],
  seed = 1
) {
  const {values, edges} = getProminenceEdges(testCase);
  const count = values.length;
  const none = 0xffffffff;
  const parents = Array<number>(count * 5).fill(none);
  let randomState = seed;
  const random = (limit: number) => {
    randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
    return randomState % limit;
  };
  const tasks = edges.flatMap((edge, rank) => [
    [edge.first, count + rank],
    [edge.second, count + rank]
  ]);
  while (tasks.length) {
    const slot = random(tasks.length);
    const [first, second] = tasks[slot];
    tasks[slot] = tasks[tasks.length - 1];
    tasks.pop();
    if (first === second) continue;
    const lower = Math.min(first, second);
    const higher = Math.max(first, second);
    const previous = parents[lower];
    parents[lower] = Math.min(previous, higher);
    if (previous !== none && previous !== higher) tasks.push([previous, higher]);
  }
  const children = parents.map(() => [] as number[]);
  parents.forEach((parent, node) => {
    if (parent !== none) children[parent].push(node);
  });
  if (children.some(nodes => nodes.length > 2)) throw new Error('non-binary merge');
  const elders = parents.map((_, node) =>
    node < count && Number.isFinite(values[node]) ? node : none
  );
  const chooseElder = (first: number, second: number) => {
    if (first === none) return second;
    if (second === none) return first;
    return values[first] > values[second] || (values[first] === values[second] && first < second)
      ? first
      : second;
  };
  let pointers = [...parents];
  for (let round = 0; round < Math.ceil(Math.log2(parents.length)) + 1; round++) {
    if (pointers.every(parent => parent === none)) break;
    const order = parents.map((_, node) => node);
    for (let slot = order.length - 1; slot > 0; slot--) {
      const other = random(slot + 1);
      [order[slot], order[other]] = [order[other], order[slot]];
    }
    // Alternate snapshot reads (worst-case propagation) with in-dispatch early arrivals.
    const source = seed % 2 ? [...elders] : elders;
    for (const node of order) {
      const ancestor = pointers[node];
      if (ancestor !== none) elders[ancestor] = chooseElder(elders[ancestor], source[node]);
    }
    pointers = pointers.map(parent => (parent === none ? none : pointers[parent]));
  }
  const saddles = values.map(value => (Number.isFinite(value) ? -Infinity : NaN));
  const parentPeaks = values.map(() => none);
  const recorded = new Set<number>();
  edges.forEach((edge, rank) => {
    const childNodes = children[count + rank];
    if (childNodes.length !== 2) return;
    const first = elders[childNodes[0]];
    const second = elders[childNodes[1]];
    const winner = chooseElder(first, second);
    const loser = winner === first ? second : first;
    if (recorded.has(loser)) throw new Error('duplicate losing leaf');
    recorded.add(loser);
    saddles[loser] = edge.saddle;
    parentPeaks[loser] = winner;
  });
  return {
    saddles,
    parentPeaks,
    prominence: saddles.map((saddle, cell) => Math.fround(values[cell] - saddle))
  };
}
