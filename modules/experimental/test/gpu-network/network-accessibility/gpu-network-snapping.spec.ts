// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUNetworkSnapping,
  GPU_NETWORK_SNAPPING_NONE,
  type GPUNetworkSnappingProps,
  type GPUNetworkSnappingStartDirection
} from '../../../src/gpu-network/network-accessibility';
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
  NONE,
  snapOracle,
  type AccessibilityCSR,
  type AccessibilityEdge
} from './network-accessibility-oracle';
import {
  createGreatCircleNetworkMetric,
  getHaversineDistance,
  type NetworkMetricOracle
} from '../network-metric-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type SnappingResult = {
  edges: number[];
  fractions: number[];
  distances: number[];
  sourceCosts: number[];
  targetCosts: number[];
  positions: number[];
  startNodes: number[];
  startCosts: number[];
  overflow?: number;
};

type SnappingOptions = {
  edgeCosts?: boolean;
  maximumSnapDistance?: number;
  candidateCapacity?: number;
  startDirection?: GPUNetworkSnappingStartDirection;
  /** Optional fixture and spatial context; defaults to the planar grid below. */
  fixture?: SnappingFixture;
};

type SnappingFixture = {
  points: Float32Array;
  positions: Float32Array;
  csr: AccessibilityCSR;
  spatialContext?: GPUNetworkSnappingProps['spatialContext'];
};

/** Grid fixture plus a CSR; `edges` keeps both directions of every link. */
const WIDTH = 8;
const NODE_COUNT = WIDTH * WIDTH;
const GRID = createGridFixture(21, WIDTH, WIDTH);
const CSR = buildCSR(NODE_COUNT, GRID.edges);

/**
 * Points: on an edge interior, exactly on a node (tie among incident edges), on a node at the grid
 * corner, near an edge endpoint, outside the grid, and far away (beyond a 1.5 snap distance).
 */
const POINTS = Float32Array.from([
  0.25, 0.1, 3, 3, 0, 0, 6.98, 5, 7.6, 2.5, 20, 20, 4.5, 6.5, 2.2, 0.9, 1, 7, 5.5, 5.5
]);

async function runSnapping(device: Device, options: SnappingOptions): Promise<SnappingResult> {
  const fixture = options.fixture ?? {points: POINTS, positions: GRID.positions, csr: CSR};
  const pointCount = fixture.points.length / 2;
  const graph = new GPUCommandGraph(device, {id: 'snapping'});
  const buffers: Buffer[] = [];
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
  const outputs: Record<string, Buffer> = {};
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
  const maximumSnapDistance =
    options.maximumSnapDistance !== undefined
      ? new GPUParameterBuffer(device, {
          id: 'max-snap-distance',
          format: 'float32',
          length: 1,
          values: Float32Array.from([options.maximumSnapDistance])
        })
      : undefined;
  const props: GPUNetworkSnappingProps = {
    id: 'snap',
    spatialContext: fixture.spatialContext,
    positions: input('points', fixture.points, 'float32x2'),
    topology: {
      offsets: input('offsets', fixture.csr.offsets, 'uint32'),
      neighbors: input('targets', fixture.csr.neighbors, 'uint32'),
      weights: options.edgeCosts ? input('costs', fixture.csr.weights, 'float32') : undefined,
      nodePositions: input('positions', fixture.positions, 'float32x2')
    },
    maximumSnapDistance: maximumSnapDistance?.importToGraph(graph),
    candidateCapacity: options.candidateCapacity,
    startDirection: options.startDirection,
    output: {
      snappedEdges: output('edges', 'uint32', pointCount),
      snapFractions: output('fractions', 'float32', pointCount),
      snapDistances: output('distances', 'float32', pointCount),
      sourceCosts: output('sourceCosts', 'float32', pointCount),
      targetCosts: output('targetCosts', 'float32', pointCount),
      snappedPositions: output('positions-out', 'float32x2', pointCount),
      startNodes: output('startNodes', 'uint32', pointCount * 2),
      startCosts: output('startCosts', 'float32', pointCount * 2),
      status: {
        overflow: options.candidateCapacity ? output('overflow', 'uint32', 1) : undefined
      }
    }
  };
  graph.add(new GPUNetworkSnapping(props));
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: SnappingResult = {
    edges: await readUint32(outputs['edges'], pointCount),
    fractions: await readFloat32(outputs['fractions'], pointCount),
    distances: await readFloat32(outputs['distances'], pointCount),
    sourceCosts: await readFloat32(outputs['sourceCosts'], pointCount),
    targetCosts: await readFloat32(outputs['targetCosts'], pointCount),
    positions: await readFloat32(outputs['positions-out'], pointCount * 2),
    startNodes: await readUint32(outputs['startNodes'], pointCount * 2),
    startCosts: await readFloat32(outputs['startCosts'], pointCount * 2),
    overflow: outputs['overflow'] ? (await readUint32(outputs['overflow'], 1))[0] : undefined
  };
  compiled.destroy();
  maximumSnapDistance?.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectSnapsMatchOracle(
  result: SnappingResult,
  options: {
    edgeCosts?: boolean;
    maximumSnapDistance?: number;
    startDirection?: string;
  }
): void {
  const expected = snapOracle(POINTS, GRID.positions, CSR.sources, CSR.neighbors, {
    edgeCosts: options.edgeCosts ? CSR.weights : undefined,
    maximumSnapDistance: options.maximumSnapDistance
  });
  expect(result.edges).toEqual(expected.map(snap => snap.edge));
  const direction = options.startDirection ?? 'both';
  expected.forEach((snap, point) => {
    for (const [actual, value] of [
      [result.fractions[point], snap.fraction],
      [result.distances[point], snap.distance],
      [result.sourceCosts[point], snap.sourceCost],
      [result.targetCosts[point], snap.targetCost]
    ]) {
      expect(actual).toBeCloseTo(value, 4);
    }
    const isSnapped = snap.edge !== NONE;
    const useSource = isSnapped && direction !== 'forward';
    const useTarget = isSnapped && direction !== 'reverse';
    expect(result.startNodes[point * 2]).toBe(useSource ? CSR.sources[snap.edge] : NONE);
    expect(result.startNodes[point * 2 + 1]).toBe(useTarget ? CSR.neighbors[snap.edge] : NONE);
    expect(result.startCosts[point * 2]).toBeCloseTo(useSource ? snap.sourceCost : -1, 4);
    expect(result.startCosts[point * 2 + 1]).toBeCloseTo(useTarget ? snap.targetCost : -1, 4);
    if (isSnapped) {
      const source = CSR.sources[snap.edge];
      const target = CSR.neighbors[snap.edge];
      for (const axis of [0, 1]) {
        const start = GRID.positions[source * 2 + axis];
        const end = GRID.positions[target * 2 + axis];
        expect(result.positions[point * 2 + axis]).toBeCloseTo(
          start + snap.fraction * (end - start),
          4
        );
      }
    } else {
      expect(result.positions[point * 2]).toBeCloseTo(POINTS[point * 2], 5);
    }
  });
}

it('GPUNetworkSnapping exhaustive scan matches the oracle, including endpoint ties', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runSnapping(device, {});
  expectSnapsMatchOracle(result, {});
  // (3, 3) is node 27: every incident edge is at distance 0 and the smallest CSR row wins. That is
  // the in-edge 19 -> 27, so the point sits at fraction 1 with a zero cost to the edge target.
  const incident = Array.from(CSR.neighbors.keys()).filter(
    edge => CSR.sources[edge] === 27 || CSR.neighbors[edge] === 27
  );
  expect(result.edges[1]).toBe(Math.min(...incident));
  expect(CSR.sources[result.edges[1]]).toBe(19);
  expect(result.fractions[1]).toBe(1);
  expect(result.distances[1]).toBe(0);
  expect(result.targetCosts[1]).toBe(0);
  // The corner node (0, 0) snaps onto its first out-edge at fraction 0.
  expect(result.edges[2]).toBe(0);
  // Unlimited: even the far point snaps.
  expect(result.edges[5]).not.toBe(GPU_NETWORK_SNAPPING_NONE);
});

it('GPUNetworkSnapping supports edge costs, a snap limit, and start directions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const startDirection of ['both', 'forward', 'reverse'] as const) {
    const options = {
      edgeCosts: true,
      maximumSnapDistance: 1.5,
      startDirection
    };
    const result = await runSnapping(device, options);
    expectSnapsMatchOracle(result, options);
    expect(result.edges[5]).toBe(GPU_NETWORK_SNAPPING_NONE);
    expect(result.distances[5]).toBe(-1);
  }
});

it('GPUNetworkSnapping BVH candidate path matches the exhaustive scan', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const options = {
    edgeCosts: true,
    maximumSnapDistance: 1.5,
    candidateCapacity: 1024
  };
  const joined = await runSnapping(device, options);
  expectSnapsMatchOracle(joined, options);
  expect(joined.overflow).toBe(0);
  const scanned = await runSnapping(device, {
    ...options,
    candidateCapacity: undefined
  });
  expect(joined.edges).toEqual(scanned.edges);
  expect(joined.fractions).toEqual(scanned.fractions);
  expect(joined.distances).toEqual(scanned.distances);

  const overflowing = await runSnapping(device, {
    ...options,
    candidateCapacity: 4
  });
  expect(overflowing.overflow).toBe(1);
});

/**
 * 3 x 3 longitude/latitude grid at 60 N straddling the antimeridian (longitudes 179.985, 179.995,
 * -179.995; latitudes 59.99, 60, 60.01), one direction per link so that no two edge rows tie.
 */
function createAntimeridianFixture(): SnappingFixture {
  const longitudes = [179.985, 179.995, -179.995];
  const latitudes = [59.99, 60, 60.01];
  const positions: number[] = [];
  for (const latitude of latitudes) {
    for (const longitude of longitudes) {
      positions.push(longitude, latitude);
    }
  }
  const edges: AccessibilityEdge[] = [];
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column < 3; column++) {
      const node = row * 3 + column;
      if (column < 2) {
        edges.push([node, node + 1, 1]);
      }
      if (row < 2) {
        edges.push([node, node + 3, 1]);
      }
    }
  }
  return {
    // The first point snaps onto the link that crosses the antimeridian at 60 N.
    points: Float32Array.from([
      179.999, 60.0015, -179.9951, 59.9952, 179.987, 60.0101, 179.99, 60.0003, -179.999, 59.995
    ]),
    positions: Float32Array.from(positions),
    csr: buildCSR(9, edges),
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
  };
}

it('GPUNetworkSnapping great-circle scan matches the f64 oracle across the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createAntimeridianFixture();
  const metric: NetworkMetricOracle = createGreatCircleNetworkMetric();
  const result = await runSnapping(device, {fixture});
  const expected = snapOracle(
    fixture.points,
    fixture.positions,
    fixture.csr.sources,
    fixture.csr.neighbors,
    {metric}
  );
  expect(result.edges).toEqual(expected.map(snap => snap.edge));
  const crossing = expected[0].edge;
  expect([fixture.csr.sources[crossing], fixture.csr.neighbors[crossing]]).toEqual([4, 5]);
  // 0.0015 degrees of latitude: 166.79 m, far from the 0.0015-degree planar reading.
  expect(result.distances[0]).toBeCloseTo(
    getHaversineDistance([179.999, 60.0015], [179.999, 60]),
    0
  );
  expected.forEach((snap, point) => {
    // f32 trigonometry on f32 inputs: centimeters on distances and costs of hundreds of meters.
    expect(Math.abs(result.distances[point] - snap.distance)).toBeLessThan(0.05);
    expect(Math.abs(result.sourceCosts[point] - snap.sourceCost)).toBeLessThan(0.05);
    expect(Math.abs(result.targetCosts[point] - snap.targetCost)).toBeLessThan(0.05);
    expect(result.fractions[point]).toBeCloseTo(snap.fraction, 4);
    const start = [
      fixture.positions[fixture.csr.sources[snap.edge] * 2],
      fixture.positions[fixture.csr.sources[snap.edge] * 2 + 1]
    ];
    const end = [
      fixture.positions[fixture.csr.neighbors[snap.edge] * 2],
      fixture.positions[fixture.csr.neighbors[snap.edge] * 2 + 1]
    ];
    const position = metric.interpolate(start, end, snap.fraction);
    // One f32 ulp of a longitude near 180 is 1.5e-5 degrees.
    const longitudeError = result.positions[point * 2] - position[0];
    expect(Math.abs(longitudeError - 360 * Math.round(longitudeError / 360))).toBeLessThan(3e-5);
    expect(result.positions[point * 2 + 1]).toBeCloseTo(position[1], 4);
  });
});

it('GPUNetworkSnapping great-circle BVH candidate path matches the great-circle scan', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createAntimeridianFixture();
  // 400 m covers every fixture point; the join's great-circle search ignores the capacity.
  const scan = await runSnapping(device, {fixture, maximumSnapDistance: 400});
  const joined = await runSnapping(device, {
    fixture,
    maximumSnapDistance: 400,
    candidateCapacity: 4
  });
  expect(joined.edges).toEqual(scan.edges);
  expect(joined.edges.every(edge => edge !== GPU_NETWORK_SNAPPING_NONE)).toBe(true);
  joined.distances.forEach((distance, point) => {
    expect(Math.abs(distance - scan.distances[point])).toBeLessThan(1e-3);
  });
  expect(joined.overflow).toBe(0);
});

it('GPUNetworkSnapping ignores CSR rows past offsets[nodeCount]', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Review B-04: the edge-source expansion wrote only the live rows, so a tail row read a zero (or
  // an aliased transient's word) as its source and became a phantom edge from node 0. The tail
  // rows below target node 63 and node 7: phantom diagonals 0 -> 63 and 0 -> 7 pass through
  // (5.5, 5.5) and near (2.2, 0.9), which would snap there at distance 0 (or closer than a road).
  const tailCSR: AccessibilityCSR = {
    offsets: CSR.offsets,
    neighbors: Uint32Array.from([...CSR.neighbors, 63, 7]),
    weights: Float32Array.from([...CSR.weights, 1, 1]),
    sources: Uint32Array.from([...CSR.sources, 0, 0])
  };
  const liveEdgeCount = CSR.neighbors.length;
  for (const candidateCapacity of [undefined, 64]) {
    const options = {
      edgeCosts: true,
      maximumSnapDistance: 1.5,
      candidateCapacity
    };
    const expected = await runSnapping(device, options);
    const result = await runSnapping(device, {
      ...options,
      fixture: {points: POINTS, positions: GRID.positions, csr: tailCSR}
    });
    expect(result.edges).toEqual(expected.edges);
    expect(result.distances).toEqual(expected.distances);
    for (const edge of result.edges) {
      expect(edge === GPU_NETWORK_SNAPPING_NONE || edge < liveEdgeCount).toBe(true);
    }
  }
});
