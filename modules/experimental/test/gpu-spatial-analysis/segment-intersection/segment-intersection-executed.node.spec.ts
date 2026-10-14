// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Runs the real GPUSegmentIntersection WGSL on the CPU (wgsl_reflect interpreter, see
// wgsl-graph-executor.ts) on small fixtures and compares it with segment-intersection-oracle.ts.
// This also proves the CPU stand-ins the other executed specs rely on: GPUBVH (complete binary
// layout), GPUSort/GPUScan (Morton sort, probe scan) and the bounds reduction.

import {useCPUExecutorTestHooks} from './cpu-executor-test-hooks';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {GPUSegmentIntersection} from '../../../src/gpu-spatial-analysis/segment-intersection/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  buildArrays,
  intersectWithOracle,
  KIND_CODES,
  type OracleHit,
  type OraclePoint,
  type OracleSegmentFeature
} from './segment-intersection-oracle';
import {createCPUGraphMemory, executeCommandNodesOnCPU} from './wgsl-graph-executor';

useCPUExecutorTestHooks();

const line = (...vertices: OraclePoint[]): OracleSegmentFeature => ({kind: 'lines', vertices});
const polygon = (...rings: OraclePoint[][]): OracleSegmentFeature => ({
  kind: 'polygons',
  polygons: [rings]
});

type Options = {spatialSort?: boolean; leafCapacity?: number; sameFeatureOnly?: boolean};

function runOnCPU(
  left: OracleSegmentFeature[],
  right: OracleSegmentFeature[] | undefined,
  options: Options = {}
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'segment-intersection-cpu'});
  const memory = createCPUGraphMemory();
  const upload = (name: string, features: OracleSegmentFeature[]) => {
    const arrays = buildArrays(features);
    const positions = createTransientView(
      graph,
      `${name}-positions`,
      'float32x2',
      arrays.positions.length / 2
    );
    memory.write(positions, arrays.positions);
    const offsets = (suffix: string, data: Uint32Array) => {
      const view = createTransientView(graph, `${name}-${suffix}`, 'uint32', data.length);
      memory.write(view, data);
      return view;
    };
    return arrays.kind === 'lines'
      ? ({
          kind: 'lines',
          positions,
          lineOffsets: offsets('line-offsets', arrays.lineOffsets)
        } as const)
      : ({
          kind: 'polygons',
          positions,
          featureOffsets: offsets('feature-offsets', arrays.featureOffsets),
          polygonOffsets: offsets('polygon-offsets', arrays.polygonOffsets),
          ringOffsets: offsets('ring-offsets', arrays.ringOffsets)
        } as const);
  };
  const capacity = 256;
  const column = (name: string) => createTransientView(graph, name, 'uint32', capacity);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const view = {
    leftIds: column('left-ids'),
    rightIds: column('right-ids'),
    count: scalar('count'),
    overflow: scalar('overflow'),
    total: scalar('total'),
    uncertain: scalar('uncertain'),
    kinds: column('kinds'),
    points: createTransientView(graph, 'points', 'float32x2', capacity),
    leftFeatures: column('left-features'),
    rightFeatures: column('right-features'),
    leftRings: column('left-rings'),
    rightRings: column('right-rings')
  };
  const contributor = new GPUSegmentIntersection({
    left: upload('left', left),
    right: right && upload('right', right),
    ...options,
    output: {
      pairs: {
        leftIds: view.leftIds,
        rightIds: view.rightIds,
        status: {count: view.count, overflow: view.overflow, requiredCount: view.total}
      },
      status: {uncertainCount: view.uncertain},
      kinds: view.kinds,
      points: view.points,
      leftFeatures: view.leftFeatures,
      rightFeatures: view.rightFeatures,
      leftRings: view.leftRings,
      rightRings: view.rightRings
    }
  });
  executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
  const read = (target: GraphDataView<'uint32'>) => memory.readUint32(target)[0];
  const count = read(view.count);
  const words = (target: GraphDataView<'uint32'>) => memory.readUint32(target, count);
  const points = memory.readFloat32(view.points, 2 * count);
  return {
    count,
    total: read(view.total),
    overflow: read(view.overflow),
    uncertain: read(view.uncertain),
    left: words(view.leftIds),
    right: words(view.rightIds),
    kind: words(view.kinds),
    leftFeature: words(view.leftFeatures),
    rightFeature: words(view.rightFeatures),
    leftRing: words(view.leftRings),
    rightRing: words(view.rightRings),
    points
  };
}

function expectMatchesOracle(result: ReturnType<typeof runOnCPU>, expected: OracleHit[]) {
  expect(result.total).toBe(expected.length);
  expect(result.count).toBe(expected.length);
  expect(result.overflow).toBe(0);
  expect(result.uncertain).toBe(0);
  const rows = (hits: OracleHit[]) => hits.map(hit => [hit.left, hit.right, KIND_CODES[hit.kind]]);
  expect(
    Array.from(result.left, (left, row) => [left, result.right[row], result.kind[row]])
  ).toEqual(rows(expected));
  expected.forEach((hit, row) => {
    expect([
      result.leftFeature[row],
      result.rightFeature[row],
      result.leftRing[row],
      result.rightRing[row]
    ]).toEqual([hit.leftFeature, hit.rightFeature, hit.leftRing, hit.rightRing]);
    for (const axis of [0, 1]) {
      const wanted = hit.point[axis];
      // Proper crossings are rounded to f32 on the GPU; other points are exact input vertices.
      const tolerance = hit.kind === 'proper' ? 2e-6 * Math.max(1, Math.abs(wanted)) : 0;
      expect(Math.abs(result.points[row * 2 + axis] - wanted)).toBeLessThanOrEqual(tolerance);
    }
  });
}

const HAND_LEFT = [
  line([0, 0], [4, 4]),
  line([10, 10], [12, 10]),
  line([20, 0], [24, 0]),
  line([30, 0], [30, 4])
];
const HAND_RIGHT = [
  line([0, 4], [4, 0]), // proper crossing with left 0
  line([2, 2], [6, 0]), // T junction on left 0
  line([4, 4], [8, 8]), // collinear touch at (4,4)
  line([1, 1], [3, 3]), // overlap of left 0
  line([0, 7], [4, 7]), // disjoint
  line([12, 10], [14, 10]), // collinear touch with left 1
  line([30, 4], [34, 8]), // shared endpoint with left 3
  line([22, 0], [30, 0]) // overlap with left 2
];

describe('GPUSegmentIntersection WGSL executed on the CPU', () => {
  it('two-sided: proper, touch, collinear touch and overlap pairs match the oracle', () => {
    const expected = intersectWithOracle(HAND_LEFT, HAND_RIGHT);
    expect(new Set(expected.map(hit => hit.kind)).size).toBe(4);
    expectMatchesOracle(runOnCPU(HAND_LEFT, HAND_RIGHT), expected);
  });

  it('self mode on polygon rings skips adjacent segments and finds the bow tie', () => {
    const scene = [
      polygon([
        [0, 0],
        [4, 4],
        [4, 0],
        [0, 4],
        [0, 0]
      ]),
      polygon([
        [10, 0],
        [14, 0],
        [14, 4],
        [10, 4]
      ])
    ];
    const expected = intersectWithOracle(scene);
    expect(expected.length).toBeGreaterThan(0);
    expectMatchesOracle(runOnCPU(scene, undefined), expected);
  });

  it('self mode with the Morton sort, a wider leaf capacity and lattice ties', () => {
    // 12 segments on a 4-unit lattice: many shared endpoints and exact ties.
    const scene: OracleSegmentFeature[] = [];
    for (let row = 0; row < 6; row++) {
      const a: OraclePoint = [((row * 3) % 8) * 4, ((row * 5) % 8) * 4];
      const b: OraclePoint = [((row * 7 + 2) % 8) * 4, ((row * 3 + 1) % 8) * 4];
      scene.push(line(a, b, [((row * 5 + 4) % 8) * 4, ((row * 2) % 8) * 4]));
    }
    const expected = intersectWithOracle(scene);
    expect(expected.length).toBeGreaterThan(0);
    expectMatchesOracle(runOnCPU(scene, undefined, {spatialSort: true}), expected);
    expectMatchesOracle(
      runOnCPU(scene, undefined, {spatialSort: false, leafCapacity: 32}),
      expected
    );
  });
});
