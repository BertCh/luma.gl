// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNetworkCostTable} from '../../../src/gpu-network/network-cost-table/index';
import {
  GPUMapMatching,
  GPU_MAP_MATCHING_NONE,
  getGPUMapMatchingParameterValues,
  type GPUMapMatchingSettings,
  type GPUMapMatchingProps
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
import {
  createGreatCircleNetworkMetric,
  getHaversineDistance,
  wrapLongitudeDelta
} from '../network-metric-oracle';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '../../../src/analysis-core/contracts/earth';
import {matchOracle} from './map-matching-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

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
  /** Overflow flag of the route table, when one was built. */
  tableOverflow: number;
};

/** A `GPUNetworkCostTable` built in the same graph and passed as `routeTable`. */
type RouteTableOptions = {
  /** Cost bound of the table, in meters. */
  costLimit: number;
  /** Entry capacity. Defaults to the full node-pair count, which cannot overflow. */
  capacity?: number;
};

/** Coordinates of a matching run; defaults to the planar grid in meters. */
type MatchFixture = {
  points: Float32Array;
  nodePositions: Float32Array;
  cellSize: number;
  bounds: {minimum: readonly [number, number]; maximum: readonly [number, number]};
  spatialContext?: GPUMapMatchingProps['spatialContext'];
};

const PLANAR_FIXTURE: MatchFixture = {
  points: TRACKS.points,
  nodePositions: POSITIONS,
  cellSize: CELL_SIZE,
  bounds: {minimum: [-500, -500], maximum: [1000, 1000]}
};

async function runMatching(
  device: Device,
  parameters: GPUMapMatchingSettings,
  fixture: MatchFixture = PLANAR_FIXTURE,
  routeTableOptions?: RouteTableOptions
): Promise<MatchResult> {
  const graph = new GPUCommandGraph(device, {id: 'map-matching'});
  const buffers: Buffer[] = [];
  const outputs: Record<string, Buffer> = {};
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
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
    values: getGPUMapMatchingParameterValues(parameters)
  });
  const topology = {
    nodePositions: input('nodes', fixture.nodePositions, 'float32x2'),
    offsets: input('offsets', CSR.offsets, 'uint32'),
    neighbors: input('targets', CSR.neighbors, 'uint32')
  };
  let routeTable: GPUMapMatchingProps['routeTable'];
  if (routeTableOptions) {
    // The table's weights are the edge lengths the matcher measures from the node positions.
    const nodeCount = WIDTH * WIDTH;
    const capacity = routeTableOptions.capacity ?? nodeCount * nodeCount;
    const lengths = Float32Array.from(CSR.neighbors, (target, edge) => {
      const source = CSR.sources[edge];
      return Math.hypot(
        fixture.nodePositions[target * 2] - fixture.nodePositions[source * 2],
        fixture.nodePositions[target * 2 + 1] - fixture.nodePositions[source * 2 + 1]
      );
    });
    routeTable = {
      offsets: output('table-offsets', 'uint32', nodeCount + 1),
      targets: output('table-targets', 'uint32', capacity),
      costs: output('table-costs', 'float32', capacity),
      costLimit: input('table-cost-limit', Float32Array.of(routeTableOptions.costLimit), 'float32'),
      status: {overflow: output('table-overflow', 'uint32', 1)}
    };
    graph.add(
      new GPUNetworkCostTable({
        id: 'table',
        topology: {
          offsets: topology.offsets,
          neighbors: topology.neighbors,
          weights: input('table-weights', lengths, 'float32')
        },
        costLimit: routeTable.costLimit,
        laneCount: 32,
        output: {
          offsets: routeTable.offsets,
          targets: routeTable.targets,
          costs: routeTable.costs,
          status: {overflow: routeTable.status?.overflow}
        }
      })
    );
  }
  graph.add(
    new GPUMapMatching({
      id: 'match',
      routeTable,
      spatialContext: fixture.spatialContext,
      positions: input('points', fixture.points, 'float32x2'),
      trackOffsets: input('track-offsets', Uint32Array.from(TRACKS.offsets), 'uint32'),
      topology,
      parameters: parameterBuffer.importToGraph(graph),
      candidateCount: CANDIDATE_COUNT,
      cellSize: fixture.cellSize,
      bounds: fixture.bounds,
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
        status: {overflow: output('overflow', 'uint32', 1)}
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
    overflow: (await readUint32(outputs['overflow'], 1))[0],
    tableOverflow: routeTable ? (await readUint32(outputs['table-overflow'], 1))[0] : 0
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

const BASE: GPUMapMatchingSettings = {
  sigma: 12,
  beta: 20,
  searchRadius: 55,
  routeFactor: 3,
  routeSlack: 50
};

function expectMatchesOracle(result: MatchResult, parameters: GPUMapMatchingSettings): void {
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
  requireTestDevice(device);
  const result = await runMatching(device, BASE);
  expectMatchesOracle(result, BASE);
  // The fixture exercises matched points, the jump break and the unmatched outlier.
  expect(result.matchedCount).toBeGreaterThan(POINT_COUNT * 0.8);
  expect(result.breakCount).toBeGreaterThan(0);
  expect(result.edges.some(edge => edge === GPU_MAP_MATCHING_NONE)).toBe(true);
});

it('GPUMapMatching responds to per-frame sigma, beta and radius without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const parameters of [
    {...BASE, sigma: 25, beta: 60},
    {...BASE, sigma: 6, beta: 8, searchRadius: 40},
    {...BASE, routeFactor: 1.2, routeSlack: 5}
  ]) {
    expectMatchesOracle(await runMatching(device, parameters), parameters);
  }
});

const METERS_PER_DEGREE = GPU_EARTH_MEAN_RADIUS_METERS * (Math.PI / 180);
const ORIGIN_LONGITUDE = 179.997;
const ORIGIN_LATITUDE = 60;

/** Lays planar meters onto longitude/latitude at 60 N, east across the antimeridian. */
function toLongitudeLatitude(planar: Float32Array): Float32Array {
  const lngLats = new Float32Array(planar.length);
  const metersPerLongitude = METERS_PER_DEGREE * Math.cos((ORIGIN_LATITUDE * Math.PI) / 180);
  for (let index = 0; index < planar.length; index += 2) {
    lngLats[index] = wrapLongitudeDelta(ORIGIN_LONGITUDE + planar[index] / metersPerLongitude);
    lngLats[index + 1] = ORIGIN_LATITUDE + planar[index + 1] / METERS_PER_DEGREE;
  }
  return lngLats;
}

it('GPUMapMatching great-circle path matches the f64 oracle across the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The planar fixture laid onto longitude/latitude: 100 m blocks, tracks and noise in meters.
  const cellSizeDegrees = CELL_SIZE / METERS_PER_DEGREE;
  const fixture: MatchFixture = {
    points: toLongitudeLatitude(TRACKS.points),
    nodePositions: toLongitudeLatitude(POSITIONS),
    cellSize: cellSizeDegrees,
    // Grid bounds across the antimeridian: longitudes 179.97 to 180.03.
    bounds: {minimum: [179.97, 59.99], maximum: [180.03, 60.02]},
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
  };
  const metric = createGreatCircleNetworkMetric();
  for (const parameters of [BASE, {...BASE, sigma: 6, beta: 8, searchRadius: 40}]) {
    const result = await runMatching(device, parameters, fixture);
    const kilometers = await runMatching(
      device,
      {
        ...parameters,
        sigma: parameters.sigma / 1000,
        beta: parameters.beta / 1000,
        searchRadius: parameters.searchRadius / 1000,
        routeSlack: (parameters.routeSlack ?? 50) / 1000
      },
      {
        ...fixture,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'kilometers'
        }
      }
    );
    expect(kilometers.edges).toEqual(result.edges);
    expect(kilometers.breaks).toEqual(result.breaks);
    expect(kilometers.overflow).toBe(0);
    result.edges.forEach((edge, point) => {
      if (edge !== GPU_MAP_MATCHING_NONE) {
        expect(kilometers.distances[point]).toBeCloseTo(result.distances[point] / 1000, 5);
        expect(kilometers.offsets[point]).toBeCloseTo(result.offsets[point] / 1000, 5);
      }
    });
    const expected = matchOracle(
      fixture.points,
      TRACKS.offsets,
      fixture.nodePositions,
      CSR,
      {
        sigma: parameters.sigma,
        beta: parameters.beta,
        searchRadius: parameters.searchRadius,
        routeFactor: parameters.routeFactor ?? 3,
        routeSlack: parameters.routeSlack ?? 50
      },
      CANDIDATE_COUNT,
      cellSizeDegrees,
      metric,
      CELL_SIZE
    );
    expect(result.overflow).toBe(0);
    expect(result.edges.map(getRoadKey)).toEqual(expected.edges.map(getRoadKey));
    expect(result.breaks).toEqual(expected.breaks);
    expected.trackLogLikelihoods.forEach((value, track) => {
      expect(result.logLikelihoods[track]).toBeCloseTo(value, 1);
    });
    expect(result.matchedCount).toBeGreaterThan(POINT_COUNT * 0.8);
    result.edges.forEach((edge, point) => {
      if (edge === GPU_MAP_MATCHING_NONE) {
        return;
      }
      const source = CSR.sources[edge];
      const target = CSR.neighbors[edge];
      const start = [fixture.nodePositions[source * 2], fixture.nodePositions[source * 2 + 1]];
      const end = [fixture.nodePositions[target * 2], fixture.nodePositions[target * 2 + 1]];
      const raw = [fixture.points[point * 2], fixture.points[point * 2 + 1]];
      const snapped = [result.positions[point * 2], result.positions[point * 2 + 1]];
      // Snapped longitudes near 180 are f32 (1.5e-5 degrees, 0.85 m at 60 N), so the distance
      // recomputed from the snapped position is good to about a meter; the reported one is exact.
      expect(Math.abs(getHaversineDistance(raw, snapped) - result.distances[point])).toBeLessThan(
        1
      );
      const onEdge = metric.interpolate(start, end, result.fractions[point]);
      expect(getHaversineDistance(onEdge, snapped)).toBeLessThan(1);
      expect(result.offsets[point]).toBeCloseTo(
        result.fractions[point] * getHaversineDistance(start, end),
        1
      );
    });
  }
});

// A route table must not change what the matcher decides. Route costs are the same shortest paths,
// summed in f32 along the search's own order: the table is the fixpoint of the relaxations and A*
// settles in key order, so a route of `h` hops differs by at most about `h * 2^-24` relative, that
// is under 1e-3 m for the 100 m blocks and routes of a few hundred meters here. A route enters the
// log-likelihood as `|straight - route| / beta` (beta 20 m) once per transition, at most 30
// transitions per track, so the track log-likelihoods agree to 1e-3. Matched edges, breaks and
// counts must be identical, because no candidate pair sits within that error of a decision.
function expectSameAsAStar(table: MatchResult, aStar: MatchResult): void {
  expect(aStar.overflow).toBe(0);
  expect(table.overflow).toBe(0);
  expect(table.edges).toEqual(aStar.edges);
  expect(table.breaks).toEqual(aStar.breaks);
  expect(table.matchedCount).toBe(aStar.matchedCount);
  expect(table.breakCount).toBe(aStar.breakCount);
  aStar.logLikelihoods.forEach((value, track) => {
    expect(Math.abs(table.logLikelihoods[track] - value)).toBeLessThan(1e-3);
  });
  table.fractions.forEach((value, point) => {
    expect(value).toBeCloseTo(aStar.fractions[point], 5);
  });
}

it('GPUMapMatching with a route table equals the A* search on the noisy grid tracks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 400 m covers every bound of the scene (3 * 45 m + 50 m plus the partial edges): all rows come
  // from the table.
  for (const parameters of [BASE, {...BASE, sigma: 25, beta: 60}]) {
    const aStar = await runMatching(device, parameters);
    const table = await runMatching(device, parameters, PLANAR_FIXTURE, {costLimit: 400});
    expect(table.tableOverflow).toBe(0);
    expectSameAsAStar(table, aStar);
    expectMatchesOracle(table, parameters);
  }
});

it('GPUMapMatching falls back to A* for bounds above the table limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Bounds are 3 * straight + 50 m minus the tail of the source edge: about 100 to 190 m between
  // neighboring samples and 2 km at the jump of track 2. A 120 m table answers the short
  // transitions and A* the rest in the same encoding; 60 m answers almost none; 0 none.
  const aStar = await runMatching(device, BASE);
  for (const costLimit of [120, 60, 0]) {
    const table = await runMatching(device, BASE, PLANAR_FIXTURE, {costLimit});
    expect(table.tableOverflow).toBe(0);
    expectSameAsAStar(table, aStar);
  }
});

it('GPUMapMatching ignores a route table that overflowed', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const aStar = await runMatching(device, BASE);
  // 50 entries cannot hold the 400 m neighborhoods: the table reports overflow and every route
  // uses A*, so the result is the A*-only result bit for bit.
  const table = await runMatching(device, BASE, PLANAR_FIXTURE, {costLimit: 400, capacity: 50});
  expect(table.tableOverflow).toBe(1);
  expect(table.edges).toEqual(aStar.edges);
  expect(table.logLikelihoods).toEqual(aStar.logLikelihoods);
  expect(table.breaks).toEqual(aStar.breaks);
});
