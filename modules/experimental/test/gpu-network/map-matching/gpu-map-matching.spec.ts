// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUMapMatching,
  GPU_MAP_MATCHING_NONE,
  encodeGPUMapMatchingParameters,
  type GPUMapMatchingParameters
} from '../../../src/gpu-network/map-matching/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCSR,
  createGridFixture,
  NONE
} from '../network-accessibility/network-accessibility-oracle';
import {matchOracle} from './map-matching-oracle';

const WIDTH = 10;
const SPACING = 100;
const GRID = createGridFixture(5, WIDTH, WIDTH);
const POSITIONS = GRID.positions.map(value => value * SPACING);
const CSR = buildCSR(WIDTH * WIDTH, GRID.edges);
const CELL_SIZE = 60;
const CANDIDATE_COUNT = 8;

/** Deterministic LCG with a Box-Muller normal. */
function createRandom(seed: number) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 0.5) / 2 ** 32;
  };
  return {
    uniform,
    normal: () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform())
  };
}

/**
 * Random walks along grid roads sampled every 45 m with 12 m Gaussian GPS noise. Track 3 jumps
 * across the map mid-way (a break), track 4 has an outlier 400 m off any road, track 5 is a single
 * point and track 6 is empty.
 */
function createTracks(): {points: Float32Array; offsets: number[]} {
  const random = createRandom(11);
  const coordinates: number[] = [];
  const offsets = [0];
  const walk = (startNode: number, stepCount: number, jumpAt = -1, outlierAt = -1) => {
    let x = (startNode % WIDTH) * SPACING;
    let y = Math.floor(startNode / WIDTH) * SPACING;
    let direction = random.uniform() < 0.5 ? [1, 0] : [0, 1];
    for (let step = 0; step < stepCount; step++) {
      if (step === jumpAt) {
        x = 800;
        y = 100;
      }
      let px = x + 12 * random.normal();
      let py = y + 12 * random.normal();
      if (step === outlierAt) {
        px = -400;
        py = 450;
      }
      coordinates.push(px, py);
      x += direction[0] * 45;
      y += direction[1] * 45;
      if (x < 0 || x > 900 || y < 0 || y > 900) {
        x = Math.min(900, Math.max(0, x - direction[0] * 90));
        y = Math.min(900, Math.max(0, y - direction[1] * 90));
        direction = [-direction[0], -direction[1]];
      }
      // Turn only at the nearest junction so the walk stays on roads.
      const isAtJunction = Math.abs(x / SPACING - Math.round(x / SPACING)) < 0.2;
      const isAtJunctionY = Math.abs(y / SPACING - Math.round(y / SPACING)) < 0.2;
      if (isAtJunction && isAtJunctionY && random.uniform() < 0.4) {
        x = Math.round(x / SPACING) * SPACING;
        y = Math.round(y / SPACING) * SPACING;
        direction = direction[0] === 0 ? [random.uniform() < 0.5 ? 1 : -1, 0] : [0, 1];
      }
    }
    offsets.push(coordinates.length / 2);
  };
  walk(0, 25);
  walk(44, 30);
  walk(23, 20, 10);
  walk(66, 15, -1, 7);
  walk(12, 1);
  offsets.push(coordinates.length / 2);
  return {points: Float32Array.from(coordinates), offsets};
}

const TRACKS = createTracks();
const POINT_COUNT = TRACKS.points.length / 2;
const TRACK_COUNT = TRACKS.offsets.length - 1;

type MatchResult = {
  edges: number[];
  fractions: number[];
  offsets: number[];
  distances: number[];
  positions: number[];
  breaks: number[];
  logLikelihoods: number[];
  matchedCount: number;
  breakCount: number;
  overflow: number;
};

async function runMatching(
  device: Device,
  parameters: GPUMapMatchingParameters
): Promise<MatchResult> {
  const graph = new GPUCommandGraph(device, {id: 'map-matching'});
  const buffers: Buffer[] = [];
  const outputs: Record<string, Buffer> = {};
  const input = <Format extends 'uint32' | 'float32x2'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      id,
      buffer,
      format,
      format === 'float32x2' ? values.length / 2 : values.length
    );
  };
  const output = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, format === 'float32x2' ? length * 2 : length);
    buffers.push(buffer);
    outputs[id] = buffer;
    return importGraphBuffer(graph, id, buffer, format, length);
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'map-matching-parameters',
    format: 'float32',
    length: 8,
    values: encodeGPUMapMatchingParameters(parameters)
  });
  graph.add(
    new GPUMapMatching({
      id: 'match',
      points: input('points', TRACKS.points, 'float32x2'),
      trackOffsets: input('track-offsets', Uint32Array.from(TRACKS.offsets), 'uint32'),
      nodePositions: input('nodes', POSITIONS, 'float32x2'),
      offsets: input('offsets', CSR.offsets, 'uint32'),
      edgeTargets: input('targets', CSR.neighbors, 'uint32'),
      parameters: parameterBuffer.importToGraph(graph),
      candidateCount: CANDIDATE_COUNT,
      cellSize: CELL_SIZE,
      bounds: {minimum: [-500, -500], maximum: [1000, 1000]},
      output: {
        matchedEdges: output('edges', 'uint32', POINT_COUNT),
        matchedFractions: output('fractions', 'float32', POINT_COUNT),
        matchedOffsets: output('offsets-out', 'float32', POINT_COUNT),
        snapDistances: output('distances', 'float32', POINT_COUNT),
        snappedPositions: output('positions', 'float32x2', POINT_COUNT),
        breaks: output('breaks', 'uint32', POINT_COUNT),
        trackLogLikelihoods: output('ll', 'float32', TRACK_COUNT),
        matchedCount: output('matched-count', 'uint32', 1),
        breakCount: output('break-count', 'uint32', 1),
        overflow: output('overflow', 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: MatchResult = {
    edges: await readUint32(outputs['edges'], POINT_COUNT),
    fractions: await readFloat32(outputs['fractions'], POINT_COUNT),
    offsets: await readFloat32(outputs['offsets-out'], POINT_COUNT),
    distances: await readFloat32(outputs['distances'], POINT_COUNT),
    positions: await readFloat32(outputs['positions'], POINT_COUNT * 2),
    breaks: await readUint32(outputs['breaks'], POINT_COUNT),
    logLikelihoods: await readFloat32(outputs['ll'], TRACK_COUNT),
    matchedCount: (await readUint32(outputs['matched-count'], 1))[0],
    breakCount: (await readUint32(outputs['break-count'], 1))[0],
    overflow: (await readUint32(outputs['overflow'], 1))[0]
  };
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

/** Two directions of one road share a key. */
function getRoadKey(edge: number): string {
  if (edge === NONE) {
    return 'none';
  }
  const source = CSR.sources[edge];
  const target = CSR.neighbors[edge];
  return `${Math.min(source, target)}-${Math.max(source, target)}`;
}

const BASE: GPUMapMatchingParameters = {
  sigma: 12,
  beta: 20,
  searchRadius: 55,
  routeFactor: 3,
  routeSlack: 50
};

function expectMatchesOracle(result: MatchResult, parameters: GPUMapMatchingParameters): void {
  const expected = matchOracle(
    TRACKS.points,
    TRACKS.offsets,
    POSITIONS,
    CSR,
    {
      sigma: parameters.sigma,
      beta: parameters.beta,
      searchRadius: parameters.searchRadius,
      routeFactor: parameters.routeFactor ?? 3,
      routeSlack: parameters.routeSlack ?? 50
    },
    CANDIDATE_COUNT,
    CELL_SIZE
  );
  expect(result.overflow).toBe(0);
  expect(result.edges.map(getRoadKey)).toEqual(expected.edges.map(getRoadKey));
  expect(result.breaks).toEqual(expected.breaks);
  expected.trackLogLikelihoods.forEach((value, track) => {
    expect(result.logLikelihoods[track]).toBeCloseTo(value, 1);
  });
  expect(result.matchedCount).toBe(expected.edges.filter(edge => edge !== NONE).length);
  expect(result.breakCount).toBe(expected.breaks.reduce((sum, value) => sum + value, 0));
  expected.edges.forEach((edge, point) => {
    if (edge === NONE) {
      expect(result.edges[point]).toBe(GPU_MAP_MATCHING_NONE);
      expect(result.fractions[point]).toBe(-1);
      expect(result.offsets[point]).toBe(-1);
      expect(result.distances[point]).toBe(-1);
      expect(result.positions[point * 2]).toBeCloseTo(TRACKS.points[point * 2], 3);
      return;
    }
    // The snapped position lies on the road and within the radius of the raw point.
    const sx = result.positions[point * 2];
    const sy = result.positions[point * 2 + 1];
    expect(
      Math.hypot(sx - TRACKS.points[point * 2], sy - TRACKS.points[point * 2 + 1])
    ).toBeCloseTo(result.distances[point], 2);
    const source = CSR.sources[result.edges[point]];
    const target = CSR.neighbors[result.edges[point]];
    const ax = POSITIONS[source * 2];
    const ay = POSITIONS[source * 2 + 1];
    const bx = POSITIONS[target * 2];
    const by = POSITIONS[target * 2 + 1];
    expect(sx).toBeCloseTo(ax + result.fractions[point] * (bx - ax), 2);
    expect(sy).toBeCloseTo(ay + result.fractions[point] * (by - ay), 2);
    expect(result.offsets[point]).toBeCloseTo(
      result.fractions[point] * Math.hypot(bx - ax, by - ay),
      2
    );
  });
}

it('GPUMapMatching matches the Newson-Krumm oracle on noisy grid tracks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const result = await runMatching(device, BASE);
  expectMatchesOracle(result, BASE);
  // The fixture exercises matched points, the jump break and the unmatched outlier.
  expect(result.matchedCount).toBeGreaterThan(POINT_COUNT * 0.8);
  expect(result.breakCount).toBeGreaterThan(0);
  expect(result.edges.some(edge => edge === GPU_MAP_MATCHING_NONE)).toBe(true);
});

it('GPUMapMatching responds to per-frame sigma, beta and radius without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const parameters of [
    {...BASE, sigma: 25, beta: 60},
    {...BASE, sigma: 6, beta: 8, searchRadius: 40},
    {...BASE, routeFactor: 1.2, routeSlack: 5}
  ]) {
    expectMatchesOracle(await runMatching(device, parameters), parameters);
  }
});
