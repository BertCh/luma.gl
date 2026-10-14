// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPolygonOverlay against shapely 2.2 / GEOS 3.14 with a fixed precision model
// (fixtures/overlay-snap-rounding.json, written by test/oracles/geometry-overlay-snap-rounding.py).
// The fixture inputs are RAW (unrounded). The overlay snaps vertices to the lattice first and then nodes the
// rounded segments, so the oracle is shapely on pointwise-rounded input (`oracleInputs`; GEOS nodes unrounded
// input, which differs by a pixel off the lattice). Every case uses a power-of-two grid size, so the overlay
// lattice equals the shapely grid and the
// result must agree EXACTLY in lattice integers: twice the signed area, the ring count and the corner
// vertices of every ring (rings with collinear vertices removed; GEOS and the GPU may keep different
// intermediate nodes on a straight run, so raw vertices are not compared). Cases whose `method` is
// `faces` are self-overlapping dissolves under a fill rule; their reference is documented in the
// generator. `snap-rounded-faces` cases have invalid rounded operands: their independent exact
// noding/winding oracle replaces undefined GEOS polygon-overlay behavior, with the original GEOS
// result and an explicit counterexample retained in fixture metadata. Needs a WebGPU device.

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {describe, expect, it} from 'vitest';
import type {
  GPUPolygonOverlayFillRule,
  GPUPolygonOverlayOperation
} from '../../../src/gpu-spatial-analysis/polygon-overlay/index';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtureJson from './fixtures/overlay-snap-rounding.json';
import {
  FIXTURE_CAPACITY,
  runOverlay,
  planOverlayCapacity,
  summarizeReadout,
  type OverlayLattice,
  type OverlayOperand
} from './overlay-gpu-harness';

type Ring = [number, number][];
type FixtureCase = {
  name: string;
  template: string;
  gridSize: number;
  lattice: {origin: [number, number]; cell: number};
  operation: GPUPolygonOverlayOperation;
  fillRule: GPUPolygonOverlayFillRule;
  method: 'shapely' | 'faces' | 'snap-rounded-faces';
  left: OverlayOperand;
  right: OverlayOperand | null;
  expected: {
    rings: Ring[];
    corners: Ring[];
    area2: number;
    ringCount: number;
    polygonCount: number;
  };
};

const FIXTURE = fixtureJson as unknown as {
  versions: {shapely: string; geos: string};
  cases: FixtureCase[];
};

const canonical = (rings: Ring[]) =>
  [...rings].sort((a, b) => {
    const keyA = JSON.stringify(a);
    const keyB = JSON.stringify(b);
    return keyA < keyB ? -1 : keyA > keyB ? 1 : 0;
  });

describe(`GPUPolygonOverlay snap-rounding fixtures (shapely ${FIXTURE.versions.shapely}, GEOS ${FIXTURE.versions.geos})`, () => {
  for (const fixture of FIXTURE.cases) {
    it(fixture.name, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const lattice: OverlayLattice = fixture.lattice;
      const readout = await runOverlay(
        device,
        {
          implementation: 'current',
          operation: fixture.operation,
          fillRule: fixture.fillRule,
          left: fixture.left,
          right: fixture.right ?? undefined,
          // The snap grid; a power of two, so the lattice cell equals it.
          vertexTolerance: fixture.gridSize,
          capacity: FIXTURE_CAPACITY
        },
        `snap-${fixture.template}`
      );
      expect(readout.overflow, 'overflow').toBe(0);
      expect(readout.candidateOverflow, 'candidateOverflow').toBe(0);
      const actual = summarizeReadout(readout, lattice);
      expect(actual.area2, 'twice the area in lattice units').toBe(fixture.expected.area2);
      expect(actual.ringCount, 'ring count').toBe(fixture.expected.ringCount);
      expect(readout.requiredRingCount, 'required ring count').toBe(fixture.expected.ringCount);
      expect(actual.corners, 'corner vertices').toEqual(canonical(fixture.expected.corners));
      // Every vertex GEOS reports is a vertex of the GPU ring as well, up to intermediate nodes.
      const actualVertices = new Set(actual.rings.flat().map(vertex => vertex.join(',')));
      for (const vertex of fixture.expected.corners.flat()) {
        expect(actualVertices.has(vertex.join(',')), `corner ${vertex}`).toBe(true);
      }
    });
  }
});

it('sizes all overlay outputs from unclamped GPU requirements after ring overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const left: OverlayOperand = [0, 4, 8].map(offset => [
    [
      [
        [offset, 0],
        [offset + 2, 0],
        [offset + 2, 2],
        [offset, 2]
      ]
    ]
  ]);
  const props = {
    implementation: 'current' as const,
    operation: 'dissolve' as const,
    left,
    vertexTolerance: 1,
    capacity: {intersections: 48, nodedSegments: 96, boundary: 96, vertices: 96, rings: 1}
  };
  const limited = await runOverlay(device, props, 'ring-overflow');
  expect(limited).toMatchObject({
    overflow: 1,
    candidateOverflow: 0,
    boundaryOverflow: 0,
    requiredRingCount: 3,
    boundaryRequiredCount: 12,
    nodedSegmentRequiredCount: 12
  });
  const capacity = planOverlayCapacity(props.capacity, limited);
  const complete = await runOverlay(device, {...props, capacity}, 'ring-sized');
  expect(complete).toMatchObject({overflow: 0, requiredRingCount: 3});
  expect(complete.rings).toHaveLength(3);
  expect(summarizeReadout(complete, {origin: [-1, -1], cell: 1}).area2).toBe(24);
});
