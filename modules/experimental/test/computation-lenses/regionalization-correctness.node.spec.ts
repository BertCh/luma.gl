// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {fromRecordedLensFile} from '../../src/computation-lenses/registry';
import {
  getAdjacency,
  runBatchedAzpMirror
} from '../gpu-spatial-analysis/spatial-regionalization/azp-batched-mirror';
import {
  computeComponentLabels,
  evaluatePartition,
  standardizeColumns
} from '../gpu-spatial-analysis/spatial-regionalization/spatial-regionalization-oracle';

/** Read captured inputs only; CPU oracles never supply scenario output or figures. */
function readInputs(element: string) {
  const {record} = fromRecordedLensFile(
    JSON.parse(readFileSync(`docs/lens-records/gpu-spatial-analysis.${element}.json`, 'utf8'))
  );
  return (name: string): number[] => {
    const snapshot = record.captures?.snapshots?.find(capture => capture.name === name);
    if (!snapshot) throw new Error(`Missing captured input: ${name}`);
    return Array.from(snapshot.values, Number);
  };
}

it('Chicago AZP keeps improving beyond 32 rounds and eventually exhausts admissible moves', () => {
  const input = readInputs('azp-regions');
  const properties = {
    adjacency: getAdjacency(
      Uint32Array.from(input('weightsRowOffsets')),
      Uint32Array.from(input('weightsNeighbors'))
    ),
    values: standardizeColumns(input('values'), 1),
    labels: input('initialLabels'),
    regionCapacity: 8,
    seed: 0
  };
  for (const maximumRounds of [16, 32]) {
    const result = runBatchedAzpMirror({...properties, maximumRounds});
    expect(result.converged).toBe(false);
    expect(result.rounds).toHaveLength(maximumRounds);
    expect(result.rounds.at(-1)!.selected.length).toBeGreaterThan(0);
  }
  const result = runBatchedAzpMirror({...properties, maximumRounds: 64});
  expect(result.converged).toBe(true);
  expect(result.rounds).toHaveLength(42);
  expect(result.rounds.at(-1)!.moves).toHaveLength(0);
  for (const round of result.rounds.slice(0, -1)) {
    expect(round.objectiveAfter).toBeLessThan(round.objectiveBefore);
  }
});

it('AZP needs a no-move round after the last improvement to certify convergence', () => {
  // A path with one misplaced value: the first round moves row 1 into region 1.
  const properties = {
    adjacency: [[1], [0, 2], [1, 3], [2]],
    values: [0, 10, 11, 20],
    labels: [0, 0, 1, 1],
    regionCapacity: 2
  };
  const limited = runBatchedAzpMirror({...properties, maximumRounds: 1});
  const complete = runBatchedAzpMirror({...properties, maximumRounds: 2});
  expect(limited.labels).toEqual([0, 1, 1, 1]);
  expect(limited.converged).toBe(false);
  expect(complete.labels).toEqual(limited.labels);
  expect(complete.converged).toBe(true);
  expect(complete.rounds.at(-1)!.moves).toHaveLength(0);
  expect(complete.rounds[0].objectiveAfter).toBeCloseTo(182 / 3, 10);
});

/** Exhaustive CPU cut oracle: recompute components and their SSD for each possible tree cut. */
function cutTree(
  values: number[],
  edges: [number, number][],
  regionCount: number,
  minimumRegionSize: number
) {
  let remainingEdges = edges.slice();
  let labels = computeComponentLabels(values.length, remainingEdges);
  const cuts: [number, number][] = [];
  while (new Set(labels).size < regionCount) {
    let bestScore = Infinity;
    let bestIndex = -1;
    let bestLabels = labels;
    for (let index = 0; index < remainingEdges.length; index++) {
      const candidateLabels = computeComponentLabels(
        values.length,
        remainingEdges.filter((_, edgeIndex) => edgeIndex !== index)
      );
      const candidate = evaluatePartition(values, 1, candidateLabels);
      if (candidate.minimumSize >= minimumRegionSize && candidate.withinSsd < bestScore) {
        bestScore = candidate.withinSsd;
        bestIndex = index;
        bestLabels = candidateLabels;
      }
    }
    if (bestIndex < 0) break;
    cuts.push(remainingEdges[bestIndex]);
    remainingEdges = remainingEdges.filter((_, index) => index !== bestIndex);
    labels = bestLabels;
  }
  return {labels, cuts, ...evaluatePartition(values, 1, labels)};
}

it('SKATER may isolate an outlier at floor one but must score both cut sides against the floor', () => {
  const values = [0, 0, 0, 0, 10];
  const edges: [number, number][] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4]
  ];
  const unrestricted = cutTree(values, edges, 2, 1);
  expect(unrestricted.cuts).toEqual([[3, 4]]);
  expect(unrestricted.minimumSize).toBe(1);
  expect(unrestricted.withinSsd).toBe(0);
  const constrained = cutTree(values, edges, 2, 2);
  expect(constrained.cuts).toEqual([[2, 3]]);
  expect(constrained.minimumSize).toBe(2);
  expect(constrained.withinSsd).toBe(50);
  const infeasible = cutTree(values, edges, 2, 3);
  expect(infeasible.cuts).toHaveLength(0);
  expect(infeasible.regionCount).toBe(1);
});

it('Chicago SKATER permits a singleton at floor one and admits five regions at floor five', () => {
  const input = readInputs('skater-regions');
  const offsets = input('weightsRowOffsets');
  const neighbors = input('weightsNeighbors');
  const flags = input('treeEdgeFlags');
  const values = input('standardizedValues');
  const edges: [number, number][] = [];
  for (let row = 0; row < values.length; row++) {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      if (flags[slot]) edges.push([row, neighbors[slot]]);
    }
  }
  expect(edges).toHaveLength(values.length - 1);
  const unrestricted = cutTree(values, edges, 5, 1);
  expect(unrestricted.regionCount).toBe(5);
  expect(unrestricted.minimumSize).toBe(1);
  const constrained = cutTree(values, edges, 5, 5);
  expect(constrained.regionCount).toBe(5);
  expect(constrained.minimumSize).toBeGreaterThanOrEqual(5);
  expect(constrained.withinSsd).toBeGreaterThan(unrestricted.withinSsd);
});
