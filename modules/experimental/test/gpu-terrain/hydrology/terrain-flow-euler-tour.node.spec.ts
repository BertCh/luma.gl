// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPU_TERRAIN_FLOW_NONE,
  GPU_TERRAIN_FLOW_STREAM_UNRESOLVED,
  GPU_TERRAIN_FLOW_UNRESOLVED
} from '../../../src/gpu-terrain/hydrology';
import {getTerrainFlowTourRoundCount} from '../../../src/gpu-terrain/hydrology/terrain-flow-euler-tour';
import fixture from '../../oracles/fixtures/terrain-hydrology.json';
import {createSeededRandom} from '../terrain-oracle-utils';
import {computeTerrainFlow} from './terrain-flow-oracle';
import {
  computeStrahlerOrderRecursive,
  computeSubtreeSizesRecursive,
  getCodes,
  getFlowReceiverOfCode,
  linkTour,
  makeCaterpillar,
  makeRandomForest,
  makeStem,
  makeStrahlerTree,
  modelStrahlerOrder,
  modelTourAccumulation,
  rankTour,
  TOUR_NONE,
  type TourForest
} from './terrain-flow-euler-tour-model';

/**
 * The CPU model repeats the GPU algorithm step by step (tour links, gated Wyllie rounds, positions,
 * levels), so these checks stand in for the GPU specs that run later:
 * - subtree sizes against a direct recursive oracle and against pysheds accumulation;
 * - Strahler orders against the recursive pull definition of the previous implementation.
 */

it('the tour ranks a 1 x 100000 stem exactly in log rounds, where 64 pull rounds left NaN', () => {
  const length = 100000;
  const stem = makeStem(length);
  const result = modelTourAccumulation(length, 1, stem.parents, new Array(length).fill(true));
  expect(result.converged).toBe(1);
  // ceil(log2(2e5)) = 18 doublings, plus the confirming round, in groups of four.
  expect(getTerrainFlowTourRoundCount(length)).toBe(20);
  expect(result.roundsExecuted).toBeLessThanOrEqual(20);
  expect(result.roundsExecuted).toBeGreaterThanOrEqual(18);
  for (const cell of [0, 1, 63, 64, 65, 4095, 50000, length - 1]) {
    expect(result.accumulation[cell]).toBe(cell + 1);
  }
  const strahler = modelStrahlerOrder({
    width: length,
    height: 1,
    flowDirections: getCodes(stem),
    streams: new Uint32Array(length).fill(1),
    unresolvedStream: GPU_TERRAIN_FLOW_STREAM_UNRESOLVED
  });
  expect(strahler.converged).toBe(1);
  expect(strahler.order.every(order => order === 1)).toBe(true);
});

it('round counts follow ceil(log2(2 n)) + 1 in groups of four', () => {
  expect(getTerrainFlowTourRoundCount(1)).toBe(4);
  expect(getTerrainFlowTourRoundCount(30)).toBe(8);
  expect(getTerrainFlowTourRoundCount(2880)).toBe(16);
  expect(getTerrainFlowTourRoundCount(2 ** 24)).toBe(28);
});

it('subtree sizes match a recursive oracle on caterpillars, stems, trees, many roots and holes', () => {
  const trees = [1, 2, 3, 4, 5].map(order => makeStrahlerTree(order));
  const forests: [string, TourForest, boolean[]?][] = [
    ['single cell', makeStem(1)],
    ['two cells', makeStem(2)],
    ['stem', makeStem(997)],
    ['caterpillar', makeCaterpillar(41)],
    ...trees.map((tree, index): [string, TourForest] => [`binary tree ${index + 1}`, tree]),
    ['many roots', {width: 12, height: 9, parents: new Uint32Array(108).fill(TOUR_NONE)}]
  ];
  for (let seed = 1; seed <= 12; seed++) {
    const random = makeRandomForest(15 + seed, 11 + (seed % 5), seed, seed % 3 === 0 ? 0.25 : 0);
    forests.push([`random ${seed}`, random.forest, random.valid]);
  }
  for (const [name, forest, valid] of forests) {
    const cellCount = forest.width * forest.height;
    const expected = computeSubtreeSizesRecursive(forest);
    const actual = modelTourAccumulation(
      forest.width,
      forest.height,
      forest.parents,
      valid ?? new Array(cellCount).fill(true)
    );
    expect(actual.converged, name).toBe(1);
    for (let cell = 0; cell < cellCount; cell++) {
      if (valid && !valid[cell]) {
        expect(actual.accumulation[cell], `${name} hole ${cell}`).toBeNaN();
      } else {
        expect(actual.accumulation[cell], `${name} cell ${cell}`).toBe(expected[cell]);
      }
    }
    const ranked = rankTour(linkTour(forest), cellCount);
    expect(ranked.roundsExecuted, name).toBeLessThanOrEqual(
      getTerrainFlowTourRoundCount(cellCount)
    );
  }
});

it('a receiver cycle leaves its component NaN and clears the converged flag', () => {
  // Cells 0 -> 1 -> 2 -> 0 form a cycle, cell 3 drains into it; cells 4 and 5 are a separate stem.
  const parents = Uint32Array.from([1, 2, 0, 0, 5, TOUR_NONE]);
  const result = modelTourAccumulation(6, 1, parents, new Array(6).fill(true));
  expect(result.converged).toBe(0);
  for (const cell of [0, 1, 2, 3]) expect(result.accumulation[cell]).toBeNaN();
  expect(Array.from(result.accumulation.slice(4))).toEqual([1, 2]);
});

it('a power-of-two cycle does not read as a root (self-jumping edges are rejected)', () => {
  // A four-cell cycle: after two doublings every edge points at itself.
  const parents = Uint32Array.from([1, 2, 3, 0]);
  const result = modelTourAccumulation(4, 1, parents, new Array(4).fill(true));
  expect(result.converged).toBe(0);
  expect(Array.from(result.accumulation).every(Number.isNaN)).toBe(true);
});

it('accumulation matches pysheds on the fixture DEMs', () => {
  for (const testCase of fixture.cases) {
    const {width, height} = testCase;
    const receivers = new Uint32Array(width * height).fill(TOUR_NONE);
    for (let cell = 0; cell < receivers.length; cell++) {
      const code = testCase.directions[cell];
      if (code !== 0) receivers[cell] = getFlowReceiverOfCode(width, height, cell, code);
    }
    const result = modelTourAccumulation(
      width,
      height,
      receivers,
      new Array(receivers.length).fill(true)
    );
    expect(Array.from(result.accumulation), testCase.name).toEqual(testCase.accumulation);
  }
});

it('accumulation matches the flow oracle across holes, plateaus and filled depressions', () => {
  for (let seed = 1; seed <= 10; seed++) {
    const random = createSeededRandom(100 + seed);
    const width = 14 + seed;
    const height = 12 + (seed % 4);
    // Few distinct heights create plateaus; NaN cells are holes.
    const elevation = Float32Array.from({length: width * height}, () =>
      random() < 0.08 ? NaN : Math.floor(random() * 6) + random() * (seed % 2 ? 0 : 0.4)
    );
    const oracle = computeTerrainFlow({
      elevation,
      grid: makeGPURasterMetadata({width, height}),
      settings: {fillEpsilon: 0.01},
      fillDepressions: true,
      resolveFlats: true
    });
    const valid = Array.from(elevation, value => Number.isFinite(value));
    const result = modelTourAccumulation(width, height, oracle.receivers, valid);
    expect(result.converged, `seed ${seed}`).toBe(1);
    for (let cell = 0; cell < elevation.length; cell++) {
      if (Number.isNaN(oracle.accumulation[cell])) {
        expect(result.accumulation[cell], `seed ${seed} cell ${cell}`).toBeNaN();
      } else {
        expect(result.accumulation[cell], `seed ${seed} cell ${cell}`).toBe(
          oracle.accumulation[cell]
        );
      }
    }
  }
});

it('float32 conversion is exact below 2^24 and rounds to nearest above', () => {
  // Models the u32 -> f32 conversion of the count kernel.
  expect(Math.fround(16777217)).toBe(16777216);
  expect(Math.fround(16777219)).toBe(16777220);
  expect(Math.fround(16777215)).toBe(16777215);
});

const UNRESOLVED = GPU_TERRAIN_FLOW_STREAM_UNRESOLVED;

function checkStrahler(
  name: string,
  width: number,
  height: number,
  flowDirections: Uint32Array,
  streams: Uint32Array
) {
  const props = {width, height, flowDirections, streams, unresolvedStream: UNRESOLVED};
  const actual = modelStrahlerOrder(props);
  const expected = computeStrahlerOrderRecursive(props);
  expect(Array.from(actual.order), name).toEqual(Array.from(expected));
  return actual;
}

it('Strahler order of the complete binary trees equals the tree order', () => {
  for (let order = 1; order <= 5; order++) {
    const tree = makeStrahlerTree(order);
    const streams = new Uint32Array(tree.parents.length);
    for (let cell = 0; cell < streams.length; cell++) {
      const isMember = tree.parents[cell] !== TOUR_NONE || cell === tree.root;
      streams[cell] = isMember ? 1 : 0;
    }
    const result = checkStrahler(`tree ${order}`, tree.width, tree.height, getCodes(tree), streams);
    expect(result.order[tree.root]).toBe(order);
    expect(result.converged).toBe(1);
  }
});

it('Strahler order of a caterpillar is 2 on the spine and 1 on the leaves', () => {
  const caterpillar = makeCaterpillar(30);
  const result = checkStrahler(
    'caterpillar',
    caterpillar.width,
    caterpillar.height,
    getCodes(caterpillar),
    new Uint32Array(caterpillar.parents.length).fill(1)
  );
  for (let column = 0; column < 30; column++) {
    expect(result.order[column]).toBe(1);
    expect(result.order[30 + column]).toBe(2);
    expect(result.order[60 + column]).toBe(1);
  }
});

it('Strahler order matches the pull definition on random forests with partial streams', () => {
  for (let seed = 1; seed <= 16; seed++) {
    const {forest, valid} = makeRandomForest(18, 14, 500 + seed, 0.12);
    const random = createSeededRandom(900 + seed);
    const codes = getCodes(forest);
    const streams = new Uint32Array(codes.length);
    for (let cell = 0; cell < codes.length; cell++) {
      if (!valid[cell]) {
        codes[cell] = GPU_TERRAIN_FLOW_NONE;
      } else {
        streams[cell] = random() < 0.75 ? 1 : 0;
      }
    }
    checkStrahler(`random ${seed}`, forest.width, forest.height, codes, streams);
  }
});

it('unresolved streams and directions poison themselves and everything downstream', () => {
  for (let seed = 1; seed <= 16; seed++) {
    const {forest, valid} = makeRandomForest(16, 12, 700 + seed, 0.1);
    const random = createSeededRandom(1300 + seed);
    const codes = getCodes(forest);
    const streams = new Uint32Array(codes.length);
    for (let cell = 0; cell < codes.length; cell++) {
      if (!valid[cell]) {
        codes[cell] = GPU_TERRAIN_FLOW_NONE;
        continue;
      }
      streams[cell] = random() < 0.8 ? 1 : 0;
      const roll = random();
      if (roll < 0.03) codes[cell] = GPU_TERRAIN_FLOW_UNRESOLVED;
      else if (roll < 0.06 && streams[cell]) streams[cell] = UNRESOLVED;
    }
    const result = checkStrahler(`unresolved ${seed}`, forest.width, forest.height, codes, streams);
    expect(
      result.order.some(order => order === TOUR_NONE),
      `unknown cells exist ${seed}`
    ).toBe(true);
  }
});

it('a stream cycle leaves its component unknown and clears converged', () => {
  // Cells 0 -> 1 -> 0 (codes east, west); cell 2 drains into cell 1; cells 4, 5 are separate.
  const codes = Uint32Array.from([1, 16, 16, 0, 1, 0]);
  const streams = new Uint32Array(6).fill(1);
  streams[3] = 0;
  const result = modelStrahlerOrder({
    width: 6,
    height: 1,
    flowDirections: codes,
    streams,
    unresolvedStream: UNRESOLVED
  });
  expect(result.converged).toBe(0);
  expect(Array.from(result.order)).toEqual([TOUR_NONE, TOUR_NONE, TOUR_NONE, 0, 1, 1]);
});
