// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_SEGMENT_INTERSECTION_KIND,
  GPUSegmentIntersection
} from '../../../src/gpu-spatial-analysis/segment-intersection/index';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createSegmentGeometry} from './segment-geometry-harness';
import {SHAPELY_SEGMENT_FIXTURE} from './shapely-segment-fixture';
import {
  createRandom,
  intersectWithOracle,
  KIND_CODES,
  roundPoint,
  type OracleHit,
  type OraclePoint,
  type OracleSegmentFeature
} from './segment-intersection-oracle';

type GPUHit = {
  left: number;
  right: number;
  kind: number;
  point: OraclePoint;
  endPoint: OraclePoint;
  leftFeature: number;
  rightFeature: number;
  leftRing: number;
  rightRing: number;
};

type RunResult = {
  hits: GPUHit[];
  count: number;
  total: number;
  overflow: number;
  uncertain: number;
};

type RunOptions = {
  capacity?: number;
  sameFeatureOnly?: boolean;
  leafCapacity?: number;
  spatialSort?: boolean;
};

async function runIntersection(
  device: Device,
  left: OracleSegmentFeature[],
  right: OracleSegmentFeature[] | undefined,
  options: RunOptions = {}
): Promise<RunResult> {
  const graph = new GPUCommandGraph(device, {id: 'segment-intersection'});
  const buffers: Buffer[] = [];
  const leftGeometry = createSegmentGeometry(device, graph, 'left', left, buffers);
  const rightGeometry = right
    ? createSegmentGeometry(device, graph, 'right', right, buffers)
    : undefined;
  const capacity = options.capacity ?? 4096;
  const output = (name: string, length: number, format: 'uint32' | 'float32x2' = 'uint32') => {
    const words = format === 'uint32' ? length : length * 2;
    const buffer = createOutputBuffer(device, words);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length) as never};
  };
  const leftIds = output('left-ids', capacity);
  const rightIds = output('right-ids', capacity);
  const count = output('count', 1);
  const overflow = output('overflow', 1);
  const total = output('total', 1);
  const uncertain = output('uncertain', 1);
  const kinds = output('kinds', capacity);
  const points = output('points', capacity, 'float32x2');
  const endPoints = output('end-points', capacity, 'float32x2');
  const leftFeatures = output('left-features', capacity);
  const rightFeatures = output('right-features', capacity);
  const leftRings = output('left-rings', capacity);
  const rightRings = output('right-rings', capacity);
  graph.add(
    new GPUSegmentIntersection({
      left: leftGeometry,
      right: rightGeometry,
      sameFeatureOnly: options.sameFeatureOnly,
      leafCapacity: options.leafCapacity,
      spatialSort: options.spatialSort,
      pairs: {
        leftIds: leftIds.view,
        rightIds: rightIds.view,
        count: count.view,
        overflow: overflow.view,
        totalCount: total.view
      },
      uncertainCount: uncertain.view,
      kinds: kinds.view,
      points: points.view,
      endPoints: endPoints.view,
      leftFeatures: leftFeatures.view,
      rightFeatures: rightFeatures.view,
      leftRings: leftRings.view,
      rightRings: rightRings.view
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [countValue] = await readUint32(count.buffer, 1);
  const columns = {
    left: await readUint32(leftIds.buffer, capacity),
    right: await readUint32(rightIds.buffer, capacity),
    kind: await readUint32(kinds.buffer, capacity),
    point: await readFloat32(points.buffer, capacity * 2),
    endPoint: await readFloat32(endPoints.buffer, capacity * 2),
    leftFeature: await readUint32(leftFeatures.buffer, capacity),
    rightFeature: await readUint32(rightFeatures.buffer, capacity),
    leftRing: await readUint32(leftRings.buffer, capacity),
    rightRing: await readUint32(rightRings.buffer, capacity)
  };
  const hits: GPUHit[] = [];
  for (let slot = 0; slot < countValue; slot++) {
    hits.push({
      left: columns.left[slot],
      right: columns.right[slot],
      kind: columns.kind[slot],
      point: [columns.point[slot * 2], columns.point[slot * 2 + 1]],
      endPoint: [columns.endPoint[slot * 2], columns.endPoint[slot * 2 + 1]],
      leftFeature: columns.leftFeature[slot],
      rightFeature: columns.rightFeature[slot],
      leftRing: columns.leftRing[slot],
      rightRing: columns.rightRing[slot]
    });
  }
  const result: RunResult = {
    hits,
    count: countValue,
    total: (await readUint32(total.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0],
    uncertain: (await readUint32(uncertain.buffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectNear(actual: OraclePoint, expected: OraclePoint, label: string) {
  for (const axis of [0, 1]) {
    const tolerance = 2e-6 * Math.max(1, Math.abs(expected[axis]));
    expect(Math.abs(actual[axis] - expected[axis]), `${label} axis ${axis}`).toBeLessThanOrEqual(
      tolerance
    );
  }
}

/** Asserts the GPU output equals the oracle, row by row, and returns the oracle hits. */
function expectMatchesOracle(
  result: RunResult,
  expected: OracleHit[],
  label: string,
  limit = expected.length
): void {
  expect(result.total, `${label} total`).toBe(expected.length);
  expect(result.count, `${label} count`).toBe(Math.min(expected.length, limit));
  expect(result.uncertain, `${label} uncertain`).toBe(0);
  const prefix = expected.slice(0, limit);
  expect(
    result.hits.map(hit => [hit.left, hit.right, hit.kind]),
    `${label} pairs`
  ).toEqual(prefix.map(hit => [hit.left, hit.right, KIND_CODES[hit.kind]]));
  result.hits.forEach((hit, slot) => {
    const want = prefix[slot];
    const where = `${label} pair ${hit.left},${hit.right}`;
    if (want.kind === 'proper') {
      expectNear(hit.point, want.point, where);
    } else {
      expect(hit.point, `${where} point`).toEqual(want.point);
      expect(hit.endPoint, `${where} endPoint`).toEqual(want.endPoint);
    }
    expect([hit.leftFeature, hit.rightFeature, hit.leftRing, hit.rightRing], where).toEqual([
      want.leftFeature,
      want.rightFeature,
      want.leftRing,
      want.rightRing
    ]);
  });
}

const line = (...vertices: OraclePoint[]): OracleSegmentFeature => ({kind: 'lines', vertices});
const polygon = (...rings: OraclePoint[][]): OracleSegmentFeature => ({
  kind: 'polygons',
  polygons: [rings]
});

it('GPUSegmentIntersection classifies proper, touch, collinear and overlap pairs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const left = [
    line([0, 0], [4, 4]), // 0..1: crosses, touches, overlaps below
    line([10, 10], [12, 10]),
    line([20, 0], [24, 0]),
    line([30, 0], [30, 4])
  ];
  const right = [
    line([0, 4], [4, 0]), // proper crossing with left 0
    line([2, 2], [6, 0]), // touches left 0 at its endpoint (2,2)? (2,2) is on left 0 interior: T-junction
    line([4, 4], [8, 8]), // collinear with left 0, shares endpoint (4,4)
    line([1, 1], [3, 3]), // overlaps left 0 on (1,1)-(3,3)
    line([0, 7], [4, 7]), // disjoint
    line([12, 10], [14, 10]), // collinear touch with left 1
    line([30, 4], [34, 8]), // touch at shared endpoint with left 3
    line([22, 0], [30, 0]) // overlap with left 2 on (22,0)-(24,0)
  ];
  const result = await runIntersection(device, left, right);
  const expected = intersectWithOracle(left, right);
  expect(expected.length).toBeGreaterThan(5);
  expectMatchesOracle(result, expected, 'hand');
  const kinds = new Set(result.hits.map(hit => hit.kind));
  for (const kind of Object.values(GPU_SEGMENT_INTERSECTION_KIND)) {
    if (kind !== GPU_SEGMENT_INTERSECTION_KIND.uncertain) {
      expect(kinds.has(kind), `kind ${kind} appears`).toBe(true);
    }
  }
});

it('GPUSegmentIntersection is exact for nearly collinear random segments', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(7);
  // Points that are almost, but not exactly, on the line of another segment decide the orientation
  // by one ulp; a naive f32 determinant misses these. A small coordinate lattice adds exact ties.
  const make = (count: number): OracleSegmentFeature[] => {
    const features: OracleSegmentFeature[] = [];
    for (let row = 0; row < count; row++) {
      const a = roundPoint([random() * 100, random() * 100]);
      const b = roundPoint([a[0] + (random() - 0.5) * 60, a[1] + (random() - 0.5) * 60]);
      const t = random();
      const mode = row % 3;
      const c: OraclePoint =
        mode === 0
          ? roundPoint([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
          : mode === 1
            ? roundPoint([Math.round(random() * 8) * 4, Math.round(random() * 8) * 4])
            : roundPoint([random() * 100, random() * 100]);
      const d = roundPoint([c[0] + (b[0] - a[0]) * 0.5, c[1] + (b[1] - a[1]) * 0.5]);
      features.push(mode === 0 ? line(c, d) : line(a, b));
    }
    return features;
  };
  const left = make(150);
  const right = make(150).concat(
    left.slice(0, 30).map(feature => {
      // Duplicates of left segments, shifted along their own line, create overlaps and touches.
      const {vertices} = feature as {vertices: OraclePoint[]};
      const [a, b] = vertices;
      return line(roundPoint([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]), b);
    })
  );
  const expected = intersectWithOracle(left, right);
  expect(expected.length).toBeGreaterThan(50);
  for (const spatialSort of [false, true]) {
    const result = await runIntersection(device, left, right, {capacity: 20000, spatialSort});
    expectMatchesOracle(result, expected, `random sort=${spatialSort}`);
  }
});

it('GPUSegmentIntersection self mode finds kinks and skips adjacent segments', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const polygons: OracleSegmentFeature[] = [
    // 0: bowtie, edges (0,0)-(4,4) and (4,0)-(0,4) cross.
    polygon([
      [0, 0],
      [4, 4],
      [4, 0],
      [0, 4]
    ]),
    // 1: simple square, adjacent edges meet only at vertices.
    polygon([
      [10, 0],
      [14, 0],
      [14, 4],
      [10, 4]
    ]),
    // 2: square with a repeated vertex and an explicit closing vertex.
    polygon([
      [20, 0],
      [24, 0],
      [24, 0],
      [24, 4],
      [20, 4],
      [20, 0]
    ]),
    // 3: ring touching itself at a non-adjacent vertex.
    polygon([
      [70, 0],
      [74, 0],
      [72, 2],
      [74, 4],
      [70, 4],
      [72, 2]
    ]),
    // 4: ring that doubles back along an edge (a spike).
    polygon([
      [80, 0],
      [84, 0],
      [82, 0],
      [82, 4]
    ])
  ];
  const lines: OracleSegmentFeature[] = [
    // 0: spike, the line doubles back over itself.
    line([30, 0], [34, 0], [32, 0]),
    // 1: figure eight.
    line([40, 0], [44, 4], [44, 0], [40, 4]),
    // 2: closed square (no kink).
    line([50, 0], [54, 0], [54, 4], [50, 4], [50, 0]),
    // 3: closed line that touches itself at a vertex.
    line([60, 0], [64, 0], [64, 4], [60, 4], [60, 0], [64, 4]),
    // 4: plain polyline.
    line([90, 0], [94, 0], [94, 4])
  ];
  for (const [name, scene, kinked, clean] of [
    ['polygons', polygons, [0, 3, 4], [1, 2]],
    ['lines', lines, [0, 1, 3], [2, 4]]
  ] as const) {
    const expected = intersectWithOracle(scene);
    const result = await runIntersection(device, scene, undefined);
    expectMatchesOracle(result, expected, `self ${name}`);
    const hitFeatures = new Set(result.hits.map(hit => hit.leftFeature));
    for (const feature of kinked) {
      expect(hitFeatures.has(feature), `${name} feature ${feature} is flagged`).toBe(true);
    }
    for (const feature of clean) {
      expect(hitFeatures.has(feature), `${name} feature ${feature} is clean`).toBe(false);
    }
  }
});

it('GPUSegmentIntersection self mode with sameFeatureOnly and between features', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(11);
  const features: OracleSegmentFeature[] = [];
  for (let row = 0; row < 60; row++) {
    const base: OraclePoint = [random() * 40, random() * 40];
    const vertices: OraclePoint[] = [roundPoint(base)];
    for (let step = 0; step < 4 + (row % 4); step++) {
      const last = vertices[vertices.length - 1];
      vertices.push(
        roundPoint([
          Math.round((last[0] + (random() - 0.5) * 16) * 2) / 2,
          Math.round((last[1] + (random() - 0.5) * 16) * 2) / 2
        ])
      );
    }
    features.push(row % 2 === 0 ? line(...vertices) : polygon(vertices));
  }
  // Mixed kinds are not allowed in one geometry; keep each kind in its own scene.
  const lines = features.filter(feature => feature.kind === 'lines');
  const polygons = features.filter(feature => feature.kind === 'polygons');
  for (const [name, scene] of [
    ['lines', lines],
    ['polygons', polygons]
  ] as const) {
    const all = await runIntersection(device, scene, undefined, {capacity: 20000});
    expectMatchesOracle(all, intersectWithOracle(scene), `${name} all`);
    const sorted = await runIntersection(device, scene, undefined, {
      capacity: 20000,
      spatialSort: true
    });
    expectMatchesOracle(sorted, intersectWithOracle(scene), `${name} all, spatial sort`);
    expect(all.total, `${name} has hits`).toBeGreaterThan(10);
    const same = await runIntersection(device, scene, undefined, {
      capacity: 20000,
      sameFeatureOnly: true
    });
    const expectedSame = intersectWithOracle(scene, undefined, {sameFeatureOnly: true});
    expectMatchesOracle(same, expectedSame, `${name} same feature`);
    expect(same.total).toBeLessThan(all.total);
  }
});

it('GPUSegmentIntersection intersects lines with polygons across kinds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const lines = [line([-2, 2], [10, 2]), line([2, -2], [2, 10]), line([0, 0], [8, 0], [8, 8])];
  const polygons = [
    polygon(
      [
        [0, 0],
        [8, 0],
        [8, 8],
        [0, 8]
      ],
      [
        [2, 2],
        [2, 6],
        [6, 6],
        [6, 2]
      ]
    ),
    polygon([
      [100, 100],
      [101, 100],
      [101, 101]
    ])
  ];
  const result = await runIntersection(device, lines, polygons);
  const expected = intersectWithOracle(lines, polygons);
  expect(expected.length).toBeGreaterThan(8);
  expectMatchesOracle(result, expected, 'line-polygon');
  const swapped = await runIntersection(device, polygons, lines);
  expectMatchesOracle(swapped, intersectWithOracle(polygons, lines), 'polygon-line');
});

it('GPUSegmentIntersection reports overflow and keeps a sorted prefix', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const left: OracleSegmentFeature[] = [];
  const right: OracleSegmentFeature[] = [];
  for (let row = 0; row < 12; row++) {
    left.push(line([0, row], [20, row + 0.5]));
    right.push(line([row, -5], [row + 0.5, 30]));
  }
  const expected = intersectWithOracle(left, right);
  expect(expected.length).toBe(144);
  const result = await runIntersection(device, left, right, {capacity: 50});
  expect(result.overflow).toBe(1);
  expectMatchesOracle(result, expected, 'overflow', 50);
  const roomy = await runIntersection(device, left, right, {capacity: 144});
  expect(roomy.overflow).toBe(0);
  expectMatchesOracle(roomy, expected, 'exact capacity');
});

it('GPUSegmentIntersection flags pairs it cannot certify as uncertain', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // The first right segment forces an orientation whose terms span about 2^226: not certifiable.
  const left = [line([1e-30, 1e-30], [2e-30, 2e-30]), line([0, 0], [4, 4])];
  const right = [line([1.5e-30, 1.5e-30], [1e38, 1e38]), line([0, 4], [4, 0])];
  const result = await runIntersection(device, left, right);
  expect(result.uncertain).toBe(2);
  expect(
    result.hits.map(hit => [hit.left, hit.right, hit.kind]),
    'uncertain pairs are listed, not dropped'
  ).toEqual([
    [0, 0, GPU_SEGMENT_INTERSECTION_KIND.uncertain],
    [2, 0, GPU_SEGMENT_INTERSECTION_KIND.uncertain],
    [2, 2, GPU_SEGMENT_INTERSECTION_KIND.proper]
  ]);
});

it('GPUSegmentIntersection skips degenerate and non-finite segments and validates props', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const left = [line([0, 0], [0, 0], [4, 4]), line([0, 0], [Number.NaN, 4], [1, 1])];
  const right = [line([0, 4], [4, 0]), line([2, 2], [2, 2])];
  const result = await runIntersection(device, left, right);
  expectMatchesOracle(result, intersectWithOracle(left, right), 'degenerate');
  expect(result.total).toBe(1);

  const graph = new GPUCommandGraph(device, {id: 'segment-intersection-validation'});
  const buffers: Buffer[] = [];
  const geometry = createSegmentGeometry(device, graph, 'left', left, buffers);
  const view = (name: string, length: number) =>
    importGraphBuffer(graph, name, createOutputBuffer(device, length), 'uint32', length);
  const pairs = {
    leftIds: view('a', 4),
    rightIds: view('b', 4),
    count: view('c', 1),
    overflow: view('d', 1)
  };
  expect(
    () =>
      new GPUSegmentIntersection({left: geometry, pairs, sameFeatureOnly: true, right: geometry})
  ).toThrow(/self mode/);
  expect(() => new GPUSegmentIntersection({left: geometry, pairs, leafCapacity: 3})).toThrow(
    /power of two/
  );
  expect(
    () =>
      new GPUSegmentIntersection({
        left: geometry,
        pairs,
        kinds: view('e', 3)
      })
  ).toThrow(/pair capacity/);
  for (const buffer of buffers) {
    buffer.destroy();
  }
});

it('GPUSegmentIntersection agrees with Shapely on a pinned lattice scene', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const toLines = (segments: readonly (readonly (readonly number[])[])[]) =>
    segments.map(([a, b]) => line([a[0], a[1]], [b[0], b[1]]));
  const left = toLines(SHAPELY_SEGMENT_FIXTURE.left);
  const right = toLines(SHAPELY_SEGMENT_FIXTURE.right);
  const result = await runIntersection(device, left, right, {capacity: 2000});
  expect(result.overflow).toBe(0);
  expect(result.total).toBe(SHAPELY_SEGMENT_FIXTURE.pairs.length);
  expect(result.hits.map(hit => [hit.left, hit.right, hit.kind])).toEqual(
    SHAPELY_SEGMENT_FIXTURE.pairs.map(([leftId, rightId, kind]) => [
      leftId,
      rightId,
      KIND_CODES[kind as keyof typeof KIND_CODES]
    ])
  );
});

it('GPUSegmentIntersection self mode on a shuffled multi-block scene equals the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Short random two-vertex lines in no spatial order: 4400 vertices (more than one bounds
  // reduction block), spatialSort on, so self-mode probes walk the tree in leaf order and the
  // sorted-order output must still equal the row-order oracle. A few long lines cross many.
  const random = createRandom(23);
  const scene: OracleSegmentFeature[] = [];
  for (let row = 0; row < 2200; row++) {
    const a = roundPoint([random() * 300, random() * 300]);
    const b = roundPoint([a[0] + (random() - 0.5) * 8, a[1] + (random() - 0.5) * 8]);
    scene.push(line(a, b));
  }
  for (const row of [100, 1000, 2100]) {
    scene[row] = line(roundPoint([0, random() * 300]), roundPoint([300, random() * 300]));
  }
  const expected = intersectWithOracle(scene);
  expect(expected.length).toBeGreaterThan(30);
  for (const spatialSort of [true, false]) {
    const result = await runIntersection(device, scene, undefined, {capacity: 20000, spatialSort});
    expectMatchesOracle(result, expected, `shuffled self sort=${spatialSort}`);
  }
}, 120000);
