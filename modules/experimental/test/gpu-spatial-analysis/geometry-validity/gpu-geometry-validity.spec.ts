// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_GEOMETRY_VALIDITY_BIT,
  GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK,
  GPUGeometryValidity,
  type GPUGeometryValidityOrientation,
  type GPUGeometryValidityRingClosure
} from '../../../src/gpu-spatial-analysis/geometry-validity/index';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createSegmentGeometry} from '../segment-intersection/segment-geometry-harness';
import type {OracleSegmentFeature} from '../segment-intersection/segment-intersection-oracle';
import {SHAPELY_VALIDITY_FIXTURE} from './shapely-validity-fixture';

const BIT = GPU_GEOMETRY_VALIDITY_BIT;
type Point = [number, number];

/** Expected masks for the Shapely fixture under the default options (implicit, ccw shells). */
const EXPECTED_MASKS: Record<string, number> = {
  valid_square: 0,
  valid_hole: 0,
  cw_square: BIT.badOrientation,
  bowtie: BIT.selfIntersection,
  hole_outside: BIT.holeOutsideShell,
  hole_crossing_shell: BIT.crossingRings,
  holes_overlap: BIT.crossingRings,
  hole_touches_shell: 0,
  self_touch_ring: BIT.selfIntersection,
  repeated_vertex: BIT.repeatedVertex,
  spike_ring: BIT.selfIntersection,
  multi_disjoint: 0,
  multi_overlap: BIT.crossingRings,
  multi_shared_edge: BIT.crossingRings,
  multi_touch_point: 0
};

/** Cases Shapely cannot express: pinned by hand with their expected masks per closure mode. */
const MANUAL_CASES: {
  name: string;
  polygons: Point[][][];
  implicit: number;
  explicit: number;
}[] = [
  {
    name: 'two_vertex_ring',
    polygons: [
      [
        [
          [0, 0],
          [4, 0]
        ]
      ]
    ],
    implicit: BIT.shortRing,
    explicit: BIT.shortRing | BIT.unclosedRing
  },
  {
    name: 'nan_vertex',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [Number.NaN, 4],
          [0, 4],
          [0, 0]
        ]
      ]
    ],
    implicit: BIT.nonFinite,
    explicit: BIT.nonFinite
  },
  {
    name: 'unclosed_square',
    polygons: [
      [
        [
          [0, 0],
          [4, 0],
          [4, 4],
          [0, 4]
        ]
      ]
    ],
    implicit: 0,
    explicit: BIT.unclosedRing
  }
];

type Scene = {names: string[]; features: OracleSegmentFeature[]};

function buildScene(): Scene {
  const names: string[] = [];
  const features: OracleSegmentFeature[] = [];
  for (const entry of SHAPELY_VALIDITY_FIXTURE) {
    names.push(entry.name);
    features.push({kind: 'polygons', polygons: entry.polygons});
  }
  for (const entry of MANUAL_CASES) {
    names.push(entry.name);
    features.push({kind: 'polygons', polygons: entry.polygons});
  }
  return {names, features};
}

async function runValidity(
  device: Device,
  features: OracleSegmentFeature[],
  options: {
    ringClosure?: GPUGeometryValidityRingClosure;
    orientation?: GPUGeometryValidityOrientation;
    intersectionCapacity?: number;
  } = {}
): Promise<{mask: number[]; overflow: number; intersectionCount: number}> {
  const graph = new GPUCommandGraph(device, {id: 'geometry-validity'});
  const buffers: Buffer[] = [];
  const geometry = createSegmentGeometry(device, graph, 'polygons', features, buffers);
  if (geometry.kind !== 'polygons') {
    throw new Error('expected polygons');
  }
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const mask = output('mask', features.length);
  const overflow = output('overflow', 1);
  const intersectionCount = output('intersection-count', 1);
  graph.add(
    new GPUGeometryValidity({
      polygons: geometry,
      mask: mask.view,
      overflow: overflow.view,
      intersectionCount: intersectionCount.view,
      intersectionCapacity: options.intersectionCapacity ?? 256,
      ringClosure: options.ringClosure,
      orientation: options.orientation
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    mask: await readUint32(mask.buffer, features.length),
    overflow: (await readUint32(overflow.buffer, 1))[0],
    intersectionCount: (await readUint32(intersectionCount.buffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUGeometryValidity masks match pinned expectations and Shapely validity', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {names, features} = buildScene();
  const result = await runValidity(device, features);
  expect(result.overflow).toBe(0);
  const byName = (name: string) => result.mask[names.indexOf(name)];
  for (const [name, expected] of Object.entries(EXPECTED_MASKS)) {
    expect(byName(name), `${name} mask`).toBe(expected);
  }
  for (const entry of MANUAL_CASES) {
    expect(byName(entry.name), `${entry.name} mask`).toBe(entry.implicit);
  }
  // Agreement with Shapely: structural validity equals is_valid, except repeated vertices, which
  // OGC accepts and this mask reports.
  let shapelyInvalid = 0;
  for (const entry of SHAPELY_VALIDITY_FIXTURE) {
    const structurallyValid = (byName(entry.name) & GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK) === 0;
    if (entry.name === 'repeated_vertex') {
      expect(structurallyValid, 'repeated vertices are reported').toBe(false);
      continue;
    }
    shapelyInvalid += entry.shapelyValid ? 0 : 1;
    expect(structurallyValid, `${entry.name}: ${entry.reason}`).toBe(entry.shapelyValid);
  }
  expect(shapelyInvalid, 'fixture contains invalid polygons').toBeGreaterThanOrEqual(7);
  expect(result.intersectionCount, 'invalid features produce intersections').toBeGreaterThan(5);
});

it('GPUGeometryValidity honors ring closure and orientation conventions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {names, features} = buildScene();
  const explicit = await runValidity(device, features, {ringClosure: 'explicit'});
  for (const entry of MANUAL_CASES) {
    expect(explicit.mask[names.indexOf(entry.name)], `${entry.name} explicit`).toBe(entry.explicit);
  }
  // Fixture rings are explicitly closed, so explicit mode changes nothing for them.
  const implicit = await runValidity(device, features);
  for (const entry of SHAPELY_VALIDITY_FIXTURE) {
    const row = names.indexOf(entry.name);
    expect(explicit.mask[row], `${entry.name} closure`).toBe(implicit.mask[row]);
  }
  const clockwise = await runValidity(device, features, {orientation: 'clockwise-shell'});
  const ignored = await runValidity(device, features, {orientation: 'ignore'});
  const square = names.indexOf('valid_square');
  const clockwiseSquare = names.indexOf('cw_square');
  const hole = names.indexOf('valid_hole');
  expect(clockwise.mask[square]).toBe(BIT.badOrientation);
  expect(clockwise.mask[clockwiseSquare]).toBe(0);
  expect(clockwise.mask[hole]).toBe(BIT.badOrientation);
  for (const mask of ignored.mask) {
    expect(mask & BIT.badOrientation).toBe(0);
  }
  expect(ignored.mask[clockwiseSquare]).toBe(0);
  expect(ignored.mask[names.indexOf('bowtie')]).toBe(BIT.selfIntersection);
});

it('GPUGeometryValidity finds a self-intersection among many valid polygons', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // A grid of valid squares with holes (touching neighbors share only corners), plus a few bowties.
  const features: OracleSegmentFeature[] = [];
  const badRows = new Set<number>();
  for (let row = 0; row < 24; row++) {
    for (let column = 0; column < 24; column++) {
      const x = column * 4;
      const y = row * 4;
      const index = features.length;
      const shell: Point[] = [
        [x, y],
        [x + 4, y],
        [x + 4, y + 4],
        [x, y + 4]
      ];
      const hole: Point[] = [
        [x + 1, y + 1],
        [x + 1, y + 3],
        [x + 3, y + 3],
        [x + 3, y + 1]
      ];
      if (index % 97 === 5) {
        badRows.add(index);
        features.push({
          kind: 'polygons',
          polygons: [[[shell[0], shell[2], shell[1], shell[3]]]]
        });
      } else {
        features.push({kind: 'polygons', polygons: [[shell, hole]]});
      }
    }
  }
  expect(badRows.size).toBeGreaterThan(3);
  const result = await runValidity(device, features, {intersectionCapacity: 512});
  expect(result.overflow).toBe(0);
  result.mask.forEach((mask, row) => {
    expect(mask, `feature ${row}`).toBe(badRows.has(row) ? BIT.selfIntersection : 0);
  });
  // Features that only touch at corners contribute intersections that are allowed (touch kinds).
  const small = await runValidity(device, features, {intersectionCapacity: 1});
  expect(small.overflow).toBe(1);
  expect(small.intersectionCount).toBeGreaterThan(1);
});

it('GPUGeometryValidity validates its props', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'geometry-validity-props'});
  const buffers: Buffer[] = [];
  const features: OracleSegmentFeature[] = [
    {
      kind: 'polygons',
      polygons: [
        [
          [
            [0, 0],
            [1, 0],
            [1, 1]
          ]
        ]
      ]
    }
  ];
  const geometry = createSegmentGeometry(device, graph, 'p', features, buffers);
  if (geometry.kind !== 'polygons') {
    throw new Error('expected polygons');
  }
  const view = (name: string, length: number) =>
    importGraphBuffer(graph, name, createOutputBuffer(device, length), 'uint32', length);
  const base = {
    polygons: geometry,
    mask: view('mask', 1),
    overflow: view('overflow', 1),
    intersectionCapacity: 4
  };
  expect(() => new GPUGeometryValidity({...base, mask: view('wrong-mask', 3)})).toThrow(
    /feature count/
  );
  expect(() => new GPUGeometryValidity({...base, intersectionCapacity: 0})).toThrow(
    /intersectionCapacity/
  );
  expect(() => new GPUGeometryValidity({...base, orientation: 'sideways' as 'ignore'})).toThrow(
    /orientation/
  );
  for (const buffer of buffers) {
    buffer.destroy();
  }
});
