// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {afterAll, expect, it} from 'vitest';

type CapturedKernel = {
  id: string;
  operation: string;
  /** The complete compiled WGSL module (module-scope declarations and the entry body). */
  source: string;
};

const captured: CapturedKernel[] = [];

const {GPUNearestFeatureJoin, GPUPointInPolygonJoin, GPUSpatialPredicateJoin} = await import(
  '../../../src/gpu-spatial-analysis/spatial-join'
);
const {createNullWebGPUDevice, getGPUCommandNodeKernelSources} = await import(
  '../../utils/gpu-contributor-test-utils'
);

/** Compiles the nodes on a recording device and keeps every generated kernel. */
function record(nodes: readonly unknown[]): void {
  captured.push(...getGPUCommandNodeKernelSources(nodes));
}

function parseCaptured(operations: string[]): number {
  let parsed = 0;
  for (const kernel of captured) {
    if (!operations.includes(kernel.operation)) {
      continue;
    }
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    parsed++;
  }
  return parsed;
}

const GEOMETRY_KINDS = ['points', 'lines', 'polygons'] as const;
type Kind = (typeof GEOMETRY_KINDS)[number];

let serial = 0;

function createGeometry(graph: GPUCommandGraph, kind: Kind) {
  const prefix = `geometry-${serial++}`;
  const view = <Format extends 'uint32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${prefix}-${name}`, format, length);
  if (kind === 'points') {
    return {kind, positions: view('positions', 'float32x2', 4)} as const;
  }
  if (kind === 'lines') {
    return {
      kind,
      positions: view('positions', 'float32x2', 8),
      lineOffsets: view('line-offsets', 'uint32', 3)
    } as const;
  }
  return {
    kind,
    positions: view('positions', 'float32x2', 12),
    featureOffsets: view('feature-offsets', 'uint32', 3),
    polygonOffsets: view('polygon-offsets', 'uint32', 3),
    ringOffsets: view('ring-offsets', 'uint32', 4)
  } as const;
}

it('GPUSpatialPredicateJoin kernels parse for every kind pair, engine and metric', () => {
  captured.length = 0;
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const greatCircle = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  for (const leftKind of GEOMETRY_KINDS) {
    for (const rightKind of GEOMETRY_KINDS) {
      const variants: Record<string, object> = {
        'fast-intersects': {predicate: 'intersects', engine: 'fast'},
        'fast-contains': {predicate: 'contains', engine: 'fast'},
        'fast-dwithin': {predicate: 'dwithin', distance: 3, engine: 'fast'},
        'planar-workgroup-dwithin': {predicate: 'dwithin', distance: 3, engine: 'relate'},
        'great-circle-dwithin': {
          predicate: 'dwithin',
          distance: 1500,
          spatialContext: greatCircle
        },
        'great-circle-dwithin-view': {
          predicate: 'dwithin',
          distance: createTransientView(graph, `distance-${serial++}`, 'float32', 1),
          spatialContext: greatCircle
        },
        'degrees-touches': {
          predicate: 'touches',
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
        },
        'relate-intersects': {predicate: 'intersects', engine: 'relate'}
      };
      for (const [name, overrides] of Object.entries(variants)) {
        const id = `p${serial++}`;
        const view = (suffix: string) =>
          createTransientView(graph, `${id}-${suffix}`, 'uint32', 4096);
        record(
          new GPUSpatialPredicateJoin({
            id,
            left: createGeometry(graph, leftKind),
            right: createGeometry(graph, rightKind),
            candidateCapacity: 4096,
            pairs: {leftIds: view('left-ids'), rightIds: view('right-ids')},
            overflow: createTransientView(graph, `${id}-overflow`, 'uint32', 1),
            ...overrides
          } as never).getCommandNodes(graph)
        );
        expect(name.length).toBeGreaterThan(0);
      }
    }
  }
  // 9 kind pairs by 8 variants, most with several kernels each.
  expect(parseCaptured(['GPUSpatialPredicateJoin', 'GPUSpatialJoinCandidates'])).toBeGreaterThan(
    72
  );
  device.destroy();
});

it('great-circle dwithin kernels use the straight-segment distance and a longitude-widened probe', () => {
  captured.length = 0;
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const id = 'gc';
  record(
    new GPUSpatialPredicateJoin({
      id,
      left: createGeometry(graph, 'points'),
      right: createGeometry(graph, 'polygons'),
      predicate: 'dwithin',
      distance: 1500,
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
      candidateCapacity: 64,
      pairs: {
        leftIds: createTransientView(graph, 'gc-left-ids', 'uint32', 64),
        rightIds: createTransientView(graph, 'gc-right-ids', 'uint32', 64)
      },
      overflow: createTransientView(graph, 'gc-overflow', 'uint32', 1)
    }).getCommandNodes(graph)
  );
  const exact = captured.find(kernel => kernel.id === 'gc-exact');
  expect(exact?.source).toContain('geodesicSegmentDistance');
  expect(exact?.source).toContain('fn pointSegmentDistanceSq(p: vec2f, a: vec2f, b: vec2f)');
  // 1500 m on a sphere of radius 6371009 (f32) as a central angle.
  expect(exact?.source).toMatch(/fn distanceLimitSq\(\) -> f32 \{ return 0\.000235\d+; \}/);
  const probe = captured.find(kernel => kernel.id === 'gc-count-candidates');
  expect(probe?.source).toContain('WRAP_SHIFT');
  expect(probe?.source).toContain('wrapDown');
  device.destroy();
});

it('GPUPointInPolygonJoin kernels parse for every boundary rule', () => {
  captured.length = 0;
  const device = createNullWebGPUDevice();
  for (const boundaryRule of ['half-open', 'closed', 'open'] as const) {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    record(
      new GPUPointInPolygonJoin({
        id: `pip-${boundaryRule}`,
        points: view('points', 'float32x2', 10),
        polygonPositions: view('polygon-positions', 'float32x2', 20),
        featureOffsets: view('feature-offsets', 'uint32', 5),
        polygonOffsets: view('polygon-offsets', 'uint32', 5),
        ringOffsets: view('ring-offsets', 'uint32', 6),
        candidateCapacity: 16,
        pointFeatureIds: view('point-feature-ids', 'uint32', 10),
        overflow: view('overflow', 'uint32', 1),
        boundaryRule
      }).getCommandNodes(graph)
    );
  }
  expect(parseCaptured(['GPUPointInPolygonJoin'])).toBeGreaterThan(9);
  const resolve = captured.find(kernel => kernel.id === 'pip-open-resolve');
  expect(resolve?.source).toContain('const BOUNDARY_CLOSED: bool = false;');
  expect(resolve?.source).toContain('const BOUNDARY_HALF_OPEN: bool = false;');
  device.destroy();
});

it('GPUNearestFeatureJoin kernels parse with exact orientation for every feature kind', () => {
  captured.length = 0;
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  for (const kind of ['points', 'lines', 'polygons'] as const) {
    const id = `nn-${kind}`;
    const slotCount = 3 * 2;
    record(
      new GPUNearestFeatureJoin({
        id,
        points: createTransientView(graph, `${id}-points`, 'float32x2', 3),
        features: createGeometry(graph, kind),
        k: 2,
        ties: 'all',
        neighborCapacity: 2,
        neighborIds: createTransientView(graph, `${id}-ids`, 'uint32', slotCount),
        neighborCounts: createTransientView(graph, `${id}-counts`, 'uint32', 3),
        neighborDistances: createTransientView(graph, `${id}-distances`, 'float32', slotCount),
        overflow: createTransientView(graph, `${id}-overflow`, 'uint32', 1)
      }).getCommandNodes(graph)
    );
  }
  expect(parseCaptured(['GPUNearestFeatureJoin'])).toBeGreaterThan(3);
  const traverse = captured.find(kernel => kernel.id.endsWith('traverse'));
  expect(traverse?.source).toContain('orientSign');
  device.destroy();
});

it('great-circle GPUNearestFeatureJoin kernels parse for every feature kind and both modes', () => {
  captured.length = 0;
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const spatialContext = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  for (const kind of ['points', 'lines', 'polygons'] as const) {
    const id = `gnn-${kind}`;
    const slotCount = 3 * 2;
    record(
      new GPUNearestFeatureJoin({
        id,
        spatialContext,
        points: createTransientView(graph, `${id}-points`, 'float32x2', 3),
        features: createGeometry(graph, kind),
        k: 2,
        ties: 'all',
        neighborCapacity: 2,
        maxDistance: createTransientView(graph, `${id}-max`, 'float32', 1),
        neighborIds: createTransientView(graph, `${id}-ids`, 'uint32', slotCount),
        neighborCounts: createTransientView(graph, `${id}-counts`, 'uint32', 3),
        neighborDistances: createTransientView(graph, `${id}-distances`, 'float32', slotCount),
        neighborFootPoints: createTransientView(graph, `${id}-foot`, 'float32x2', slotCount),
        neighborQueryPoints: createTransientView(graph, `${id}-query`, 'float32x2', slotCount),
        neighborSegmentIndices: createTransientView(graph, `${id}-segments`, 'uint32', slotCount),
        overflow: createTransientView(graph, `${id}-overflow`, 'uint32', 1)
      }).getCommandNodes(graph)
    );
  }
  const nearestId = 'gnn-nearest';
  record(
    new GPUNearestFeatureJoin({
      id: nearestId,
      spatialContext,
      points: createTransientView(graph, `${nearestId}-points`, 'float32x2', 3),
      features: {
        kind: 'segments',
        starts: createTransientView(graph, `${nearestId}-starts`, 'float32x2', 4),
        ends: createTransientView(graph, `${nearestId}-ends`, 'float32x2', 4)
      },
      radius: createTransientView(graph, `${nearestId}-radius`, 'float32', 1),
      nearestFeatureIds: createTransientView(graph, `${nearestId}-ids`, 'uint32', 3),
      nearestDistances: createTransientView(graph, `${nearestId}-distances`, 'float32', 3),
      featureCounts: createTransientView(graph, `${nearestId}-counts`, 'uint32', 4),
      overflow: createTransientView(graph, `${nearestId}-overflow`, 'uint32', 1)
    }).getCommandNodes(graph)
  );
  expect(parseCaptured(['GPUNearestFeatureJoin'])).toBeGreaterThan(12);
  const traverse = captured.find(kernel => kernel.id === 'gnn-lines-traverse');
  expect(traverse?.source).toContain('fn boxLowerAngle');
  expect(traverse?.source).toContain('geodesicSegmentDistance');
  expect(traverse?.source).toContain('orientSign');
  // One exact-orientation body per kernel.
  expect(traverse?.source.match(/fn orientExact/g)?.length ?? 0).toBeLessThanOrEqual(1);
  const clear = captured.find(kernel => kernel.id === 'gnn-lines-clear');
  expect(clear?.source).toContain('SPHERE_RADIUS');
  expect(clear?.source).toContain('limit / SPHERE_RADIUS');
  device.destroy();
});

it('planar GPUNearestFeatureJoin kernels carry no great-circle code', () => {
  captured.length = 0;
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  record(
    new GPUNearestFeatureJoin({
      id: 'pnn',
      points: createTransientView(graph, 'pnn-points', 'float32x2', 3),
      features: createGeometry(graph, 'lines'),
      neighborIds: createTransientView(graph, 'pnn-ids', 'uint32', 3),
      neighborCounts: createTransientView(graph, 'pnn-counts', 'uint32', 3),
      overflow: createTransientView(graph, 'pnn-overflow', 'uint32', 1)
    }).getCommandNodes(graph)
  );
  for (const kernel of captured.filter(kernel => kernel.operation === 'GPUNearestFeatureJoin')) {
    expect(kernel.source, kernel.id).not.toContain('geodesic');
    expect(kernel.source, kernel.id).not.toContain('boxLowerAngle');
  }
  device.destroy();
});

afterAll(() => {
  captured.length = 0;
});
