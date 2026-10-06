// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {cellsToMultiPolygon, gridDisk, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUSegmentRingAssembly,
  GPU_SEGMENT_RING_ASSEMBLY_NONE
} from '../../../src/gpu-spatial-analysis/ring-assembly';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline';
import {
  GPUPointInPolygonJoin,
  GPU_SPATIAL_JOIN_NO_FEATURE
} from '../../../src/gpu-spatial-analysis/spatial-join';
import {
  bigIntToH3,
  h3ToBigInt,
  quadbinTileToCell,
  splitCellKey
} from '../cell-aggregation/cell-aggregation-oracle';
import {webMercatorTileBounds} from '../cell-indexing/cell-indexing-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  assembleRingsOnCPU,
  type OracleRingOptions,
  type OracleRingResult,
  containsPoint,
  type OracleSegment
} from './ring-assembly-oracle';

const NONE = GPU_SEGMENT_RING_ASSEMBLY_NONE;

type RingRun = {
  count: number;
  overflow: number;
  total: number;
  open: number;
  touching: number;
  offsets: number[];
  positions: [number, number][];
  areas: number[];
  isHole: number[];
  shells: number[];
  groups: number[];
  segmentRings: number[];
  segmentVertices: number[];
  flags: number[];
};

type AssemblyOptions = {
  ringCapacity: number;
  vertexCapacity: number;
  tolerance?: number;
  interiorSide?: 'left' | 'right';
  normalizeWinding?: boolean;
  geographic?: boolean;
  splitTouchingRings?: boolean;
  count?: number;
  groups?: number[];
};

type Allocated = {track: <B extends Buffer>(buffer: B) => B; release: () => void};

function createTracker(): Allocated {
  const buffers: Buffer[] = [];
  return {
    track: buffer => {
      buffers.push(buffer);
      return buffer;
    },
    release: () => buffers.forEach(buffer => buffer.destroy())
  };
}

function createRingBuffers(
  device: Device,
  tracker: Allocated,
  segments: number,
  options: AssemblyOptions
) {
  const {track} = tracker;
  return {
    ringOffsets: track(createOutputBuffer(device, options.ringCapacity + 1)),
    positions: track(createOutputBuffer(device, 2 * options.vertexCapacity)),
    areas: track(createOutputBuffer(device, options.ringCapacity)),
    isHole: track(createOutputBuffer(device, options.ringCapacity)),
    shells: track(createOutputBuffer(device, options.ringCapacity)),
    groups: track(createOutputBuffer(device, options.ringCapacity)),
    segmentRings: track(createOutputBuffer(device, segments)),
    segmentVertices: track(createOutputBuffer(device, segments)),
    flags: track(createOutputBuffer(device, segments)),
    count: track(createOutputBuffer(device, 1)),
    overflow: track(createOutputBuffer(device, 1)),
    total: track(createOutputBuffer(device, 1)),
    open: track(createOutputBuffer(device, 1)),
    touching: track(createOutputBuffer(device, 1))
  };
}

function getRingOutput(
  graph: GPUCommandGraph,
  buffers: ReturnType<typeof createRingBuffers>,
  segments: number,
  options: AssemblyOptions,
  withGroups: boolean
) {
  return {
    ringOffsets: importGraphBuffer(
      graph,
      'ring-offsets',
      buffers.ringOffsets,
      'uint32',
      options.ringCapacity + 1
    ),
    positions: importGraphBuffer(
      graph,
      'ring-positions',
      buffers.positions,
      'float32x2',
      options.vertexCapacity
    ),
    ringAreas: importGraphBuffer(
      graph,
      'ring-areas',
      buffers.areas,
      'float32',
      options.ringCapacity
    ),
    ringIsHole: importGraphBuffer(
      graph,
      'ring-is-hole',
      buffers.isHole,
      'uint32',
      options.ringCapacity
    ),
    ringShells: importGraphBuffer(
      graph,
      'ring-shells',
      buffers.shells,
      'uint32',
      options.ringCapacity
    ),
    ringGroups: withGroups
      ? importGraphBuffer(graph, 'ring-groups', buffers.groups, 'uint32', options.ringCapacity)
      : undefined,
    segmentRings: importGraphBuffer(
      graph,
      'segment-rings',
      buffers.segmentRings,
      'uint32',
      segments
    ),
    segmentVertices: importGraphBuffer(
      graph,
      'segment-vertices',
      buffers.segmentVertices,
      'uint32',
      segments
    ),
    segmentFlags: importGraphBuffer(graph, 'segment-flags', buffers.flags, 'uint32', segments),
    count: importGraphBuffer(graph, 'ring-count', buffers.count, 'uint32', 1),
    overflow: importGraphBuffer(graph, 'ring-overflow', buffers.overflow, 'uint32', 1),
    totalCount: importGraphBuffer(graph, 'ring-total', buffers.total, 'uint32', 1),
    openSegmentCount: importGraphBuffer(graph, 'ring-open', buffers.open, 'uint32', 1),
    touchingSegmentCount: importGraphBuffer(graph, 'ring-touching', buffers.touching, 'uint32', 1)
  };
}

async function readRingRun(
  buffers: ReturnType<typeof createRingBuffers>,
  segments: number,
  options: AssemblyOptions
): Promise<RingRun> {
  const [count] = await readUint32(buffers.count, 1);
  const [overflow] = await readUint32(buffers.overflow, 1);
  const [total] = await readUint32(buffers.total, 1);
  const [open] = await readUint32(buffers.open, 1);
  const [touching] = await readUint32(buffers.touching, 1);
  const offsets = await readUint32(buffers.ringOffsets, options.ringCapacity + 1);
  const coordinates = await readFloat32(buffers.positions, 2 * options.vertexCapacity);
  const positions: [number, number][] = [];
  for (let index = 0; index < options.vertexCapacity; index++) {
    positions.push([coordinates[2 * index], coordinates[2 * index + 1]]);
  }
  return {
    count,
    overflow,
    total,
    open,
    touching,
    offsets,
    positions,
    areas: await readFloat32(buffers.areas, options.ringCapacity),
    isHole: await readUint32(buffers.isHole, options.ringCapacity),
    shells: await readUint32(buffers.shells, options.ringCapacity),
    groups: await readUint32(buffers.groups, options.ringCapacity),
    segmentRings: await readUint32(buffers.segmentRings, segments),
    segmentVertices: await readUint32(buffers.segmentVertices, segments),
    flags: await readUint32(buffers.flags, segments)
  };
}

async function runAssembly(
  device: Device,
  segments: OracleSegment[],
  options: AssemblyOptions
): Promise<RingRun> {
  const tracker = createTracker();
  const {track} = tracker;
  const buffers = createRingBuffers(device, tracker, segments.length, options);
  const graph = new GPUCommandGraph(device, {id: 'ring-assembly-graph'});
  const withGroups = Boolean(options.groups);
  graph.add(
    new GPUSegmentRingAssembly({
      endpoints: importGraphBuffer(
        graph,
        'endpoints',
        track(createInputBuffer(device, Float32Array.from(segments.flat()))),
        'float32x4',
        segments.length
      ),
      count:
        options.count === undefined
          ? undefined
          : importGraphBuffer(
              graph,
              'count-in',
              track(createInputBuffer(device, Uint32Array.of(options.count))),
              'uint32',
              1
            ),
      groups: options.groups
        ? importGraphBuffer(
            graph,
            'groups-in',
            track(createInputBuffer(device, Uint32Array.from(options.groups))),
            'uint32',
            segments.length
          )
        : undefined,
      vertexTolerance: options.tolerance,
      interiorSide: options.interiorSide,
      normalizeWinding: options.normalizeWinding,
      geographic: options.geographic,
      splitTouchingRings: options.splitTouchingRings,
      output: getRingOutput(graph, buffers, segments.length, options, withGroups)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const run = await readRingRun(buffers, segments.length, options);
  compiled.destroy?.();
  tracker.release();
  return run;
}

type OutlineRingRun = {
  segments: OracleSegment[];
  groups: number[];
  ring: RingRun;
};

async function runOutlineRings(
  device: Device,
  family: 'h3' | 'quadbin',
  cells: bigint[],
  options: {
    ringCapacity: number;
    vertexCapacity: number;
    segmentCapacity: number;
    groups?: Map<bigint, number>;
    normalizeWinding?: boolean;
  }
): Promise<OutlineRingRun> {
  const tracker = createTracker();
  const {track, release} = tracker;
  const capacity = options.segmentCapacity;
  const words = new Uint32Array(cells.length * 2);
  cells.forEach((cell, row) => {
    const [low, high] = splitCellKey(cell);
    words[2 * row] = low;
    words[2 * row + 1] = high;
  });
  const out = {
    rows: track(createOutputBuffer(device, capacity)),
    cells: track(createOutputBuffer(device, 2 * capacity)),
    edges: track(createOutputBuffer(device, capacity)),
    endpoints: track(createOutputBuffer(device, 4 * capacity)),
    groups: track(createOutputBuffer(device, capacity)),
    count: track(createOutputBuffer(device, 1)),
    overflow: track(createOutputBuffer(device, 1))
  };
  const ringOptions: AssemblyOptions = {
    ringCapacity: options.ringCapacity,
    vertexCapacity: options.vertexCapacity
  };
  const buffers = createRingBuffers(device, tracker, capacity, ringOptions);
  const graph = new GPUCommandGraph(device, {id: 'outline-rings-graph'});
  const groupValues = options.groups
    ? Uint32Array.from(cells.map(cell => options.groups!.get(cell) ?? 0))
    : undefined;
  graph.add(
    new GPUCellSetOutline({
      family,
      cells: importGraphBuffer(
        graph,
        'cells',
        track(createInputBuffer(device, words)),
        'uint32x2',
        cells.length
      ),
      groups: groupValues
        ? importGraphBuffer(
            graph,
            'groups',
            track(createInputBuffer(device, groupValues)),
            'uint32',
            cells.length
          )
        : undefined,
      output: {
        rows: importGraphBuffer(graph, 'out-rows', out.rows, 'uint32', capacity),
        cells: importGraphBuffer(graph, 'out-cells', out.cells, 'uint32x2', capacity),
        edgeIndices: importGraphBuffer(graph, 'out-edges', out.edges, 'uint32', capacity),
        endpoints: importGraphBuffer(graph, 'out-endpoints', out.endpoints, 'float32x4', capacity),
        groups: groupValues
          ? importGraphBuffer(graph, 'out-groups', out.groups, 'uint32', capacity)
          : undefined,
        count: importGraphBuffer(graph, 'out-count', out.count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'out-overflow', out.overflow, 'uint32', 1)
      },
      rings: {
        normalizeWinding: options.normalizeWinding,
        output: getRingOutput(graph, buffers, capacity, ringOptions, Boolean(groupValues))
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(out.count, 1);
  const endpoints = await readFloat32(out.endpoints, 4 * capacity);
  const groups = (await readUint32(out.groups, capacity)).slice(0, count);
  const segments: OracleSegment[] = [];
  for (let row = 0; row < count; row++) {
    segments.push(endpoints.slice(4 * row, 4 * row + 4) as OracleSegment);
  }
  const ring = await readRingRun(buffers, capacity, ringOptions);
  compiled.destroy?.();
  release();
  return {segments, groups, ring};
}

/** Asserts a GPU run equals the oracle exactly (positions are copies of input endpoints). */
function expectMatchesOracle(
  run: RingRun,
  oracle: OracleRingResult,
  segmentCount: number,
  label: string
) {
  expect(run.total, `${label} total`).toBe(oracle.rings.length);
  expect(run.count, `${label} count`).toBe(oracle.rings.length);
  expect(run.overflow, `${label} overflow`).toBe(0);
  expect(run.open, `${label} open`).toBe(oracle.openSegments);
  expect(run.touching, `${label} touching`).toBe(oracle.touchingSegments);
  let cursor = 0;
  oracle.rings.forEach((ring, index) => {
    expect(run.offsets[index], `${label} offset ${index}`).toBe(cursor);
    expect(run.offsets[index + 1], `${label} offset ${index + 1}`).toBe(
      cursor + ring.vertices.length
    );
    ring.vertices.forEach((vertex, position) => {
      const actual = run.positions[cursor + position];
      expect(actual[0], `${label} ring ${index} vertex ${position} x`).toBe(Math.fround(vertex[0]));
      expect(actual[1], `${label} ring ${index} vertex ${position} y`).toBe(Math.fround(vertex[1]));
    });
    cursor += ring.vertices.length;
    const tolerance = 1e-3 * Math.abs(ring.area) + 1e-12;
    expect(Math.abs(run.areas[index] - ring.area), `${label} area ${index}`).toBeLessThanOrEqual(
      tolerance
    );
    expect(run.isHole[index], `${label} hole ${index}`).toBe(ring.isHole ? 1 : 0);
    expect(run.shells[index], `${label} shell ${index}`).toBe(ring.shell ?? NONE);
  });
  for (let segment = 0; segment < segmentCount; segment++) {
    const ring = oracle.segmentRing[segment];
    expect(run.segmentRings[segment], `${label} segment ring ${segment}`).toBe(
      ring < 0 ? NONE : ring
    );
  }
}

function polygonSegments(
  ring: [number, number][],
  offset: [number, number] = [0, 0]
): OracleSegment[] {
  const result: OracleSegment[] = [];
  for (let index = 0; index < ring.length; index++) {
    const a = ring[index];
    const b = ring[(index + 1) % ring.length];
    result.push([a[0] + offset[0], a[1] + offset[1], b[0] + offset[0], b[1] + offset[1]]);
  }
  return result;
}

function square(x0: number, y0: number, x1: number, y1: number, counterClockwise: boolean) {
  const ring: [number, number][] = [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1]
  ];
  return counterClockwise ? ring : ring.reverse();
}

/** Deterministic shuffle so ring order and leaders do not follow construction order. */
function shuffle<T>(items: T[]): T[] {
  const result = [...items];
  let state = 12345;
  for (let index = result.length - 1; index > 0; index--) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    const other = state % (index + 1);
    [result[index], result[other]] = [result[other], result[index]];
  }
  return result;
}

function createPlanarScene(): OracleSegment[] {
  return shuffle([
    ...polygonSegments(square(0, 0, 10, 10, true)),
    ...polygonSegments(square(2, 2, 8, 8, false)),
    ...polygonSegments(square(4, 4, 6, 6, true)),
    ...polygonSegments(square(4.5, 4.5, 5.5, 5.5, false)),
    // Two squares touching at the vertex (21, 1).
    ...polygonSegments(square(20, 0, 21, 1, true)),
    ...polygonSegments(square(21, 1, 22, 2, true)),
    // A second shell with a hole whose two diagonal pockets touch at (53, 3).
    ...polygonSegments(square(50, 0, 56, 6, true)),
    ...polygonSegments(square(51, 1, 53, 3, false)),
    ...polygonSegments(square(53, 3, 55, 5, false))
  ]);
}

it('GPUSegmentRingAssembly chains shells, holes, islands and touching corners', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createPlanarScene();
  const options: AssemblyOptions = {
    ringCapacity: 16,
    vertexCapacity: scene.length + 16,
    tolerance: 0.01,
    geographic: false
  };
  const oracle = assembleRingsOnCPU(scene, {
    ...options,
    tolerance: 0.01,
    interiorSide: 'left',
    geographic: false
  });
  const run = await runAssembly(device, scene, options);
  expectMatchesOracle(run, oracle, scene.length, 'planar');

  // Hand-checked structure: A, hole, island, island hole, two touching squares, B, two B holes.
  expect(run.count).toBe(9);
  expect(run.isHole.slice(0, 9).reduce((sum, value) => sum + value, 0)).toBe(4);
  const sizes = Array.from({length: 9}, (_, ring) => run.offsets[ring + 1] - run.offsets[ring]);
  expect(sizes.every(size => size === 5)).toBe(true);
  // Touching squares stay separate rings (4 segments each) and the shared vertex is flagged.
  expect(run.touching).toBe(4);
  expect(run.open).toBe(0);
  // Nested hole assignment: every hole belongs to the smallest enclosing shell.
  const expectedShellAreaByHoleArea = new Map([
    [36, 100],
    [1, 4],
    [4, 36]
  ]);
  let holesChecked = 0;
  for (let ring = 0; ring < 9; ring++) {
    if (run.isHole[ring]) {
      expect(run.shells[ring], `hole ${ring} has a shell`).not.toBe(NONE);
      expect(run.isHole[run.shells[ring]]).toBe(0);
      expect(Math.abs(run.areas[run.shells[ring]])).toBeCloseTo(
        expectedShellAreaByHoleArea.get(Math.round(Math.abs(run.areas[ring])))!,
        3
      );
      holesChecked++;
    }
  }
  expect(holesChecked).toBe(4);

  // Deterministic: a second run is identical.
  const again = await runAssembly(device, scene, options);
  expect(again.positions).toEqual(run.positions);
  expect(again.offsets).toEqual(run.offsets);
});

it('GPUSegmentRingAssembly honors interiorSide, normalizeWinding and reversed input', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createPlanarScene();
  const reversed = scene.map(([x0, y0, x1, y1]): OracleSegment => [x1, y1, x0, y0]);
  for (const normalizeWinding of [false, true]) {
    const options: AssemblyOptions = {
      ringCapacity: 16,
      vertexCapacity: scene.length + 16,
      tolerance: 0.01,
      geographic: false,
      interiorSide: 'right',
      normalizeWinding
    };
    const oracle = assembleRingsOnCPU(reversed, {
      ...options,
      interiorSide: 'right'
    } as OracleRingOptions);
    const run = await runAssembly(device, reversed, options);
    expectMatchesOracle(run, oracle, reversed.length, `right ${normalizeWinding}`);
    expect(run.count).toBe(9);
    expect(run.isHole.reduce((sum, value) => sum + value, 0)).toBe(4);
    // Shells are clockwise unless normalized; holes are the opposite of their shells.
    for (let ring = 0; ring < run.count; ring++) {
      const shellPositive = normalizeWinding;
      expect(run.areas[ring] > 0 === (run.isHole[ring] ? !shellPositive : shellPositive)).toBe(
        true
      );
    }
  }
});

it('GPUSegmentRingAssembly leaves unclosed chains open and flags them', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const closed = polygonSegments(square(0, 0, 4, 4, true));
  const chain: OracleSegment[] = [
    [10, 0, 11, 0],
    [11, 0, 11, 1],
    [11, 1, 12, 1]
  ];
  // A near-miss: the end is outside the tolerance of the next start.
  const gap: OracleSegment[] = [
    [20, 0, 21, 0],
    [21.5, 0, 21.5, 1]
  ];
  const scene = [...chain, ...closed, ...gap];
  const options: AssemblyOptions = {
    ringCapacity: 4,
    vertexCapacity: 16,
    tolerance: 0.01,
    geographic: false
  };
  const run = await runAssembly(device, scene, options);
  const oracle = assembleRingsOnCPU(scene, {
    ...options,
    tolerance: 0.01,
    interiorSide: 'left',
    geographic: false
  });
  expectMatchesOracle(run, oracle, scene.length, 'open');
  expect(run.count).toBe(1);
  expect(run.open).toBe(5);
  expect(run.flags.slice(0, 3)).toEqual([0, 0, 2]);
  expect(run.segmentRings.slice(0, 3)).toEqual([NONE, NONE, NONE]);
  expect(run.segmentRings.slice(3, 7)).toEqual([0, 0, 0, 0]);

  // A smaller count removes trailing rows: the ring loses a segment and no longer closes.
  const truncated = await runAssembly(device, scene, {...options, count: 6});
  expect(truncated.count).toBe(0);
  expect(truncated.open).toBe(6);
});

it('GPUSegmentRingAssembly writes whole rings only when capacity is short', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createPlanarScene();
  const full = await runAssembly(device, scene, {
    ringCapacity: 16,
    vertexCapacity: 64,
    tolerance: 0.01,
    geographic: false
  });
  expect(full.count).toBe(9);
  for (const [ringCapacity, vertexCapacity, expectedCount] of [
    [3, 64, 3],
    [16, 12, 2],
    [16, 4, 0]
  ]) {
    const run = await runAssembly(device, scene, {
      ringCapacity,
      vertexCapacity,
      tolerance: 0.01,
      geographic: false
    });
    const label = `rings ${ringCapacity} vertices ${vertexCapacity}`;
    expect(run.total, label).toBe(9);
    expect(run.count, label).toBe(expectedCount);
    expect(run.overflow, label).toBe(1);
    // Written rings are a prefix of the full result and the offsets stay flat afterwards.
    for (let ring = 0; ring <= expectedCount; ring++) {
      expect(run.offsets[ring], label).toBe(full.offsets[ring]);
    }
    for (let ring = expectedCount; ring <= ringCapacity; ring++) {
      expect(run.offsets[ring], label).toBe(full.offsets[expectedCount]);
    }
    const written = full.offsets[expectedCount];
    expect(run.positions.slice(0, written), label).toEqual(full.positions.slice(0, written));
    for (let segment = 0; segment < scene.length; segment++) {
      const ring = run.segmentRings[segment];
      expect(ring === NONE || ring < expectedCount, label).toBe(true);
    }
  }
});

function sortKeys(cells: bigint[]): bigint[] {
  return [...new Set(cells)].sort((left, right) => (left < right ? -1 : left > right ? 1 : 0));
}

function createH3Scene(resolution: number): bigint[] {
  const first = gridDisk(latLngToCell(47.2, 8.5, resolution), 6).filter(
    (_, index) => index % 5 !== 3
  );
  const second = gridDisk(latLngToCell(39.7, -100.2, resolution), 3);
  // Island inside a hole of the first blob.
  const origin = latLngToCell(47.2, 8.5, resolution);
  const third = gridDisk(origin, 6).filter(cell => cell === origin);
  return sortKeys([...first, ...second, ...third].map(h3ToBigInt));
}

it('GPUCellSetOutline rings of an H3 set equal h3-js cellsToMultiPolygon', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const resolution of [5, 6]) {
    const cells = createH3Scene(resolution);
    const polygons = cellsToMultiPolygon(cells.map(bigIntToH3), true) as [number, number][][][];
    const expectedRings = polygons.flatMap((polygon, polygonIndex) =>
      polygon.map((ring, ringIndex) => ({ring, polygonIndex, isHole: ringIndex > 0}))
    );
    const holeCount = expectedRings.filter(item => item.isHole).length;
    expect(holeCount, 'scene has holes').toBeGreaterThan(0);
    expect(polygons.length, 'scene has several polygons').toBeGreaterThan(1);
    const result = await runOutlineRings(device, 'h3', cells, {
      ringCapacity: 64,
      vertexCapacity: 4096,
      segmentCapacity: 4096
    });
    const {ring: run} = result;
    const label = `h3 res ${resolution}`;
    expect(run.overflow, label).toBe(0);
    expect(run.open, label).toBe(0);
    expect(run.touching, label).toBe(0);
    expect(run.count, label).toBe(expectedRings.length);
    // Match each GPU ring to the h3-js ring with the same vertex set.
    const matched = new Map<number, number>();
    for (let ring = 0; ring < run.count; ring++) {
      const vertices = run.positions.slice(run.offsets[ring], run.offsets[ring + 1] - 1);
      expect(run.positions[run.offsets[ring + 1] - 1], `${label} closed`).toEqual(
        run.positions[run.offsets[ring]]
      );
      const found = expectedRings.findIndex(
        (item, index) =>
          !new Set(matched.values()).has(index) &&
          item.ring.length - 1 === vertices.length &&
          vertices.every(vertex =>
            item.ring.some(
              point =>
                Math.abs(point[0] - vertex[0]) < 1e-4 && Math.abs(point[1] - vertex[1]) < 1e-4
            )
          )
      );
      expect(found, `${label} ring ${ring} has an h3-js counterpart`).toBeGreaterThanOrEqual(0);
      matched.set(ring, found);
      expect(run.isHole[ring], `${label} hole flag ${ring}`).toBe(
        expectedRings[found].isHole ? 1 : 0
      );
      expect(run.areas[ring] > 0, `${label} orientation ${ring}`).toBe(
        !expectedRings[found].isHole
      );
    }
    // Shell assignment equals the h3-js polygon grouping.
    for (let ring = 0; ring < run.count; ring++) {
      const expected = expectedRings[matched.get(ring)!];
      const shell = run.shells[ring];
      expect(shell, `${label} shell of ring ${ring}`).not.toBe(NONE);
      expect(
        expectedRings[matched.get(shell)!].polygonIndex,
        `${label} polygon of ring ${ring}`
      ).toBe(expected.polygonIndex);
      expect(expectedRings[matched.get(shell)!].isHole).toBe(false);
    }
  }
});

function getTileArea(x: number, y: number, z: number): number {
  const [west, south, east, north] = webMercatorTileBounds(x, y, z);
  return (east - west) * (north - south);
}

function countComponents(tiles: Set<string>): number {
  const seen = new Set<string>();
  let components = 0;
  for (const tile of tiles) {
    if (seen.has(tile)) {
      continue;
    }
    components++;
    const stack = [tile];
    seen.add(tile);
    while (stack.length) {
      const [x, y] = stack.pop()!.split(',').map(Number);
      for (const key of [`${x + 1},${y}`, `${x - 1},${y}`, `${x},${y + 1}`, `${x},${y - 1}`]) {
        if (tiles.has(key) && !seen.has(key)) {
          seen.add(key);
          stack.push(key);
        }
      }
    }
  }
  return components;
}

it('GPUCellSetOutline rings of a Quadbin set handle holes and touching corners', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const z = 6;
  const tiles = new Set<string>();
  const addBlock = (x0: number, y0: number, x1: number, y1: number) => {
    for (let x = x0; x <= x1; x++) {
      for (let y = y0; y <= y1; y++) {
        tiles.add(`${x},${y}`);
      }
    }
  };
  addBlock(10, 10, 17, 17);
  // Hole of two tiles and an island inside a larger hole.
  for (const tile of ['12,12', '13,12']) {
    tiles.delete(tile);
  }
  for (const tile of ['15,14', '15,15', '14,14', '14,15', '16,14']) {
    tiles.delete(tile);
  }
  tiles.add('15,15');
  // Diagonal-only neighbors.
  tiles.add('30,30');
  tiles.add('31,31');
  tiles.add('32,30');
  // Block with two holes that touch at a corner.
  addBlock(40, 40, 46, 46);
  tiles.delete('42,42');
  tiles.delete('43,43');
  // A tile touching the shell of another block at a single corner.
  tiles.add('18,18');
  const cells = sortKeys(
    [...tiles].map(tile => {
      const [x, y] = tile.split(',').map(Number);
      return quadbinTileToCell(x, y, z);
    })
  );
  const expectedShells = countComponents(tiles);
  const expectedArea = [...tiles].reduce((sum, tile) => {
    const [x, y] = tile.split(',').map(Number);
    return sum + getTileArea(x, y, z);
  }, 0);

  for (const normalizeWinding of [false, true]) {
    const result = await runOutlineRings(device, 'quadbin', cells, {
      ringCapacity: 64,
      vertexCapacity: 4096,
      segmentCapacity: 2048,
      normalizeWinding
    });
    const {ring: run, segments} = result;
    const label = `quadbin normalize ${normalizeWinding}`;
    const oracle = assembleRingsOnCPU(segments, {
      tolerance: 5e-5,
      interiorSide: 'right',
      normalizeWinding
    });
    expectMatchesOracle(run, oracle, segments.length, label);
    expect(run.open, label).toBe(0);
    expect(run.touching, `${label} touching`).toBeGreaterThan(0);
    // Shells are exactly the 4-connected components: corner contact never joins rings.
    const shells = run.isHole.slice(0, run.count).filter(value => value === 0).length;
    expect(shells, `${label} shells`).toBe(expectedShells);
    for (let ring = 0; ring < run.count; ring++) {
      if (run.isHole[ring]) {
        expect(run.shells[ring], `${label} hole ${ring} shell`).not.toBe(NONE);
      }
    }
    if (normalizeWinding) {
      // Signed ring areas (shells positive, holes negative) add up to the filled tile area.
      const total = run.areas.slice(0, run.count).reduce((sum, value) => sum + value, 0);
      expect(Math.abs(total - expectedArea) / expectedArea).toBeLessThan(1e-3);
    }
  }
});

it('GPUCellSetOutline rings never mix groups', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) {
    return;
  }
  const cells = createH3Scene(5);
  const groups = new Map(cells.map(cell => [cell, Number((cell >> 3n) % 3n)]));
  const result = await runOutlineRings(device, 'h3', cells, {
    ringCapacity: 256,
    vertexCapacity: 8192,
    segmentCapacity: 4096,
    groups
  });
  const {ring: run, segments, groups: segmentGroups} = result;
  const oracle = assembleRingsOnCPU(segments, {
    tolerance: 5e-5,
    interiorSide: 'left',
    groups: segmentGroups
  });
  expectMatchesOracle(run, oracle, segments.length, 'h3 groups');
  expect(run.count).toBeGreaterThan(3);
  expect(run.open).toBe(0);
  for (let segment = 0; segment < segments.length; segment++) {
    const ring = run.segmentRings[segment];
    expect(ring).not.toBe(NONE);
    expect(run.groups[ring]).toBe(segmentGroups[segment]);
  }
});

it('GPUSegmentRingAssembly without splitting joins holes that touch at a corner', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createPlanarScene();
  const options: AssemblyOptions = {
    ringCapacity: 16,
    vertexCapacity: scene.length + 16,
    tolerance: 0.01,
    geographic: false,
    splitTouchingRings: false
  };
  const oracle = assembleRingsOnCPU(scene, {
    tolerance: 0.01,
    interiorSide: 'left',
    geographic: false,
    splitTouchingRings: false
  });
  const run = await runAssembly(device, scene, options);
  expectMatchesOracle(run, oracle, scene.length, 'tight');
  // The two pockets of the second shell become one self-touching hole ring (8 instead of 9 rings).
  expect(run.count).toBe(8);
  expect(run.isHole.slice(0, 8).reduce((sum, value) => sum + value, 0)).toBe(3);
});

type PolygonRun = {
  ringOffsets: number[];
  polygonOffsets: number[];
  featureOffsets: number[];
  positions: [number, number][];
  pointFeatureIds: number[];
  joinOverflow: number;
  uncertain: number;
};

async function runPolygons(
  device: Device,
  segments: OracleSegment[],
  options: AssemblyOptions,
  points: [number, number][]
): Promise<PolygonRun> {
  const tracker = createTracker();
  const {track} = tracker;
  const buffers = createRingBuffers(device, tracker, segments.length, options);
  const polygonBuffers = {
    positions: track(createOutputBuffer(device, 2 * options.vertexCapacity)),
    ringOffsets: track(createOutputBuffer(device, options.ringCapacity + 1)),
    polygonOffsets: track(createOutputBuffer(device, options.ringCapacity + 1)),
    featureOffsets: track(createOutputBuffer(device, 2)),
    pointFeatureIds: track(createOutputBuffer(device, points.length)),
    overflow: track(createOutputBuffer(device, 1)),
    uncertain: track(createOutputBuffer(device, 1))
  };
  const graph = new GPUCommandGraph(device, {id: 'ring-polygons-graph'});
  const polygons = {
    positions: importGraphBuffer(
      graph,
      'polygon-positions',
      polygonBuffers.positions,
      'float32x2',
      options.vertexCapacity
    ),
    ringOffsets: importGraphBuffer(
      graph,
      'polygon-ring-offsets',
      polygonBuffers.ringOffsets,
      'uint32',
      options.ringCapacity + 1
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      'polygon-offsets',
      polygonBuffers.polygonOffsets,
      'uint32',
      options.ringCapacity + 1
    ),
    featureOffsets: importGraphBuffer(
      graph,
      'feature-offsets',
      polygonBuffers.featureOffsets,
      'uint32',
      2
    )
  };
  graph.add(
    new GPUSegmentRingAssembly({
      endpoints: importGraphBuffer(
        graph,
        'endpoints',
        track(createInputBuffer(device, Float32Array.from(segments.flat()))),
        'float32x4',
        segments.length
      ),
      vertexTolerance: options.tolerance,
      interiorSide: options.interiorSide,
      normalizeWinding: options.normalizeWinding,
      geographic: options.geographic,
      output: {...getRingOutput(graph, buffers, segments.length, options, false), polygons}
    })
  );
  graph.add(
    new GPUPointInPolygonJoin({
      points: importGraphBuffer(
        graph,
        'points',
        track(createInputBuffer(device, Float32Array.from(points.flat()))),
        'float32x2',
        points.length
      ),
      polygonPositions: polygons.positions,
      featureOffsets: polygons.featureOffsets,
      polygonOffsets: polygons.polygonOffsets,
      ringOffsets: polygons.ringOffsets,
      candidateCapacity: points.length * 2,
      pointFeatureIds: importGraphBuffer(
        graph,
        'point-features',
        polygonBuffers.pointFeatureIds,
        'uint32',
        points.length
      ),
      overflow: importGraphBuffer(graph, 'join-overflow', polygonBuffers.overflow, 'uint32', 1),
      uncertainCount: importGraphBuffer(
        graph,
        'join-uncertain',
        polygonBuffers.uncertain,
        'uint32',
        1
      )
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const coordinates = await readFloat32(polygonBuffers.positions, 2 * options.vertexCapacity);
  const result: PolygonRun = {
    ringOffsets: await readUint32(polygonBuffers.ringOffsets, options.ringCapacity + 1),
    polygonOffsets: await readUint32(polygonBuffers.polygonOffsets, options.ringCapacity + 1),
    featureOffsets: await readUint32(polygonBuffers.featureOffsets, 2),
    positions: Array.from({length: options.vertexCapacity}, (_, index) => [
      coordinates[2 * index],
      coordinates[2 * index + 1]
    ]),
    pointFeatureIds: await readUint32(polygonBuffers.pointFeatureIds, points.length),
    joinOverflow: (await readUint32(polygonBuffers.overflow, 1))[0],
    uncertain: (await readUint32(polygonBuffers.uncertain, 1))[0]
  };
  compiled.destroy?.();
  tracker.release();
  return result;
}

it('GPUSegmentRingAssembly polygon layout feeds GPUPointInPolygonJoin', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createPlanarScene();
  const options: AssemblyOptions = {
    ringCapacity: 16,
    vertexCapacity: 64,
    tolerance: 0.01,
    geographic: false
  };
  const oracle = assembleRingsOnCPU(scene, {...options, interiorSide: 'left'} as OracleRingOptions);
  // Expected layout: rings ordered by (shell, hole flag, ring index), closing vertex dropped.
  const order = oracle.rings
    .map((ring, index) => ({ring, index}))
    .filter(({ring}) => ring.shell !== null)
    .sort(
      (a, b) =>
        a.ring.shell! * 2 +
          (a.ring.isHole ? 1 : 0) -
          (b.ring.shell! * 2 + (b.ring.isHole ? 1 : 0)) || a.index - b.index
    );
  const expectedRingOffsets: number[] = [0];
  const expectedPositions: [number, number][] = [];
  const shellStarts: number[] = [];
  order.forEach(({ring}, position) => {
    if (!ring.isHole) {
      shellStarts.push(position);
    }
    expectedPositions.push(...ring.vertices.slice(0, -1));
    expectedRingOffsets.push(expectedPositions.length);
  });

  // Random points away from edges (coordinates off the integer and half grids).
  const points: [number, number][] = [];
  let state = 99;
  const random = () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
  for (let index = 0; index < 400; index++) {
    points.push([
      Math.fround(-1 + random() * 59 + 0.0137),
      Math.fround(-1 + random() * 8 + 0.0291)
    ]);
  }
  const run = await runPolygons(device, scene, options, points);
  expect(run.joinOverflow).toBe(0);
  expect(run.uncertain).toBe(0);
  expect(run.featureOffsets).toEqual([0, shellStarts.length]);
  expect(run.ringOffsets.slice(0, order.length + 1)).toEqual(expectedRingOffsets);
  for (let ring = order.length; ring <= options.ringCapacity; ring++) {
    expect(run.ringOffsets[ring]).toBe(expectedRingOffsets[order.length]);
  }
  expect(run.polygonOffsets.slice(0, shellStarts.length)).toEqual(shellStarts);
  for (let polygon = shellStarts.length; polygon <= options.ringCapacity; polygon++) {
    expect(run.polygonOffsets[polygon]).toBe(order.length);
  }
  expect(run.positions.slice(0, expectedPositions.length)).toEqual(
    expectedPositions.map(([x, y]) => [Math.fround(x), Math.fround(y)])
  );

  // Join result equals even-odd containment against every ring of the oracle.
  let inside = 0;
  points.forEach((point, row) => {
    const expected = oracle.rings.some(ring => !ring.isHole && containsPoint(ring.vertices, point))
      ? !oracle.rings.some(ring => ring.isHole && containsPoint(ring.vertices, point))
      : false;
    // A point inside a hole that sits inside an island shell is inside again: use parity.
    const parity =
      oracle.rings.reduce((sum, ring) => sum + (containsPoint(ring.vertices, point) ? 1 : 0), 0) %
        2 ===
      1;
    void expected;
    expect(
      run.pointFeatureIds[row] !== GPU_SPATIAL_JOIN_NO_FEATURE,
      `point ${row} ${point} feature ${run.pointFeatureIds[row]}`
    ).toBe(parity);
    inside += parity ? 1 : 0;
  });
  expect(inside).toBeGreaterThan(20);
  expect(inside).toBeLessThan(points.length - 20);
});
