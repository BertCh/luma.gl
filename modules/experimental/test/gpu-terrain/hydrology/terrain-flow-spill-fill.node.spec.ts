// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import fixture from '../../oracles/fixtures/terrain-fill-depressions.json';
import {computeTerrainFlow} from './terrain-flow-oracle';
import {
  getSpillKey,
  getSpillValue,
  modelSpillFill,
  modelTileFill,
  SPILL_NONE
} from './terrain-flow-spill-fill-model';
import {
  fillDepressionsPriorityFlood,
  makeSpillFillCases,
  makeSpiralCase
} from './terrain-flow-spill-fill-oracle';
import {getTerrainFlowSpillBoruvkaRoundCount} from '../../../src/gpu-terrain/hydrology/terrain-flow-spill-fill';

/**
 * The CPU model repeats the GPU spill graph fill step by step (tile fill and labels, implicit
 * spill edges, Boruvka rounds with pointer jumping, Euler tour ranking, path maximum), so these
 * checks stand in for the GPU specs that run later. Expected values are bit-identical: the filled
 * surface is the unique minimax fixpoint, so Planchon-Darboux, Priority-Flood and pysheds agree.
 */

function expectSameBits(actual: Float32Array, expected: Float32Array, label: string) {
  expect(actual.length, label).toBe(expected.length);
  const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  for (let cell = 0; cell < actual.length; cell++) {
    if (Number.isNaN(expected[cell])) {
      if (!Number.isNaN(actual[cell])) throw new Error(`${label}: cell ${cell} should be NaN`);
    } else if (actualBits[cell] !== expectedBits[cell]) {
      throw new Error(`${label}: cell ${cell} is ${actual[cell]}, expected ${expected[cell]}`);
    }
  }
}

const cases = makeSpillFillCases();

it('order-preserving keys round trip and sort like the floats', () => {
  const values = [-1e30, -2.5, -1, -1e-30, 0, 1e-30, 1, 2.5, 3.4e38];
  for (const value of values) {
    expect(getSpillValue(getSpillKey(value))).toBe(Math.fround(value));
  }
  const keys = values.map(getSpillKey);
  expect([...keys].sort((left, right) => left - right)).toEqual(keys);
});

it('the spill fill model equals Priority-Flood on every adversarial DEM, for both tile sizes', () => {
  for (const testCase of cases) {
    const expected = fillDepressionsPriorityFlood(
      testCase.elevation,
      testCase.width,
      testCase.height
    );
    for (const tileSize of [32, 16]) {
      const result = modelSpillFill(testCase.elevation, testCase.width, testCase.height, {
        tileSize
      });
      expectSameBits(result.filled, expected, `${testCase.name} T=${tileSize}`);
      expect(result.converged, `${testCase.name} T=${tileSize}`).toBe(true);
      expect(result.boruvkaRounds).toBeLessThanOrEqual(
        getTerrainFlowSpillBoruvkaRoundCount(testCase.width * testCase.height)
      );
    }
  }
});

it('the spill fill model equals the Planchon-Darboux relaxation oracle', () => {
  for (const testCase of cases.filter(({width, height}) => width * height <= 4000)) {
    const grid = makeGPURasterMetadata({width: testCase.width, height: testCase.height});
    const oracle = computeTerrainFlow({
      elevation: testCase.elevation,
      grid,
      settings: {},
      fillDepressions: true
    }).filled;
    const result = modelSpillFill(testCase.elevation, testCase.width, testCase.height);
    expectSameBits(result.filled, oracle, testCase.name);
  }
});

it('the spill fill model equals pysheds fill_depressions', () => {
  expect(fixture.versions.pysheds).toBeTruthy();
  for (const testCase of fixture.cases) {
    const elevation = Float32Array.from(testCase.elevation);
    const result = modelSpillFill(elevation, testCase.width, testCase.height);
    expectSameBits(result.filled, Float32Array.from(testCase.filled), testCase.name);
    // The comparison must exercise filling: strips cannot hold a depression, the rest do.
    if (!/^strip-(1x|41x1)/.test(testCase.name)) {
      expect(testCase.raisedCells, testCase.name).toBeGreaterThan(0);
    }
  }
});

it('the result does not depend on the order of the atomic operations', () => {
  for (const testCase of cases.filter(({width, height}) => width * height <= 4000)) {
    const reference = modelSpillFill(testCase.elevation, testCase.width, testCase.height);
    for (const shuffleSeed of [1, 2, 3]) {
      const shuffled = modelSpillFill(testCase.elevation, testCase.width, testCase.height, {
        shuffleSeed
      });
      expectSameBits(shuffled.filled, reference.filled, `${testCase.name} seed ${shuffleSeed}`);
    }
  }
});

it('labels follow the outlet and tie rules', () => {
  // A flat 40 x 40 plateau: every perimeter cell of a tile is an outlet with its own label, and
  // every interior cell takes the smallest label reachable at equal height.
  const flat = new Float32Array(40 * 40).fill(5);
  const tile = modelTileFill(flat, 40, 40, 32);
  for (let cell = 0; cell < flat.length; cell++) {
    expect(tile.labels[cell]).not.toBe(SPILL_NONE);
    expect(tile.localFill[cell]).toBe(5);
  }
  // Tile (0, 0) interior cell (5, 5): smallest outlet of the tile is cell 0 (a boundary outlet).
  expect(tile.labels[5 * 40 + 5]).toBe(0);
  // Invalid cells have no label and are NaN.
  const holed = Float32Array.from(flat);
  holed[100] = NaN;
  const holedTile = modelTileFill(holed, 40, 40, 32);
  expect(holedTile.labels[100]).toBe(SPILL_NONE);
  expect(Number.isNaN(holedTile.localFill[100])).toBe(true);
});

it('a corridor of width 1 closed in a spiral fills to the spill height of every ring', () => {
  const spiral = makeSpiralCase(61, 1);
  const result = modelSpillFill(spiral.elevation, spiral.width, spiral.height);
  const expected = fillDepressionsPriorityFlood(spiral.elevation, spiral.width, spiral.height);
  expectSameBits(result.filled, expected, spiral.name);
  // The spiral raises corridors (the fill is not a no-op) ...
  const raised = result.filled.reduce(
    (count, value, cell) => count + (value > spiral.elevation[cell] ? 1 : 0),
    0
  );
  expect(raised).toBeGreaterThan(500);
  // ... and the tree needs only logarithmically many rounds.
  expect(result.boruvkaRounds).toBeLessThanOrEqual(
    getTerrainFlowSpillBoruvkaRoundCount(spiral.width * spiral.height)
  );
});

it('Boruvka rounds halve the components: the round count is logarithmic in the cell count', () => {
  const small = modelSpillFill(
    ...(([c]) => [c.elevation, c.width, c.height] as const)([makeSpiralCase(23, 1)])
  );
  const large = modelSpillFill(
    ...(([c]) => [c.elevation, c.width, c.height] as const)([makeSpiralCase(61, 1)])
  );
  expect(small.treeEdgeCount).toBeLessThan(large.treeEdgeCount);
  expect(large.boruvkaRounds - small.boruvkaRounds).toBeLessThanOrEqual(4);
});
