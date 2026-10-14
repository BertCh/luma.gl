// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../utils/require-test-device';
import {
  prepareOverlay,
  planOverlayCapacity,
  type OverlayRunCapacity,
  type OverlayRunProps
} from '../overlay-gpu-harness';

import {createOverlayABScene} from './overlay-ab-scene';

// Same-run A/B of the legacy overlay (f32 noding, every piece ray-cast against every ring) against the
// snap-rounded overlay (lattice noding, face labelling). Run with A5_AB=1. Compile time is excluded;
// each side reports the best of three submits, each timed to the readback of the published feature
// count after one untimed warm-up. Sizes are input vertices per operand: 1e3, 1e4, 1e5. The legacy
// side runs only up to 1e4 (its membership probe is O(pieces x vertices): 1e5 would run for minutes
// and exceed the default device timeout); the table prints "infeasible" there.
//
// Scene: a grid of small star polygons (each 20 vertices) per operand, the right operand shifted by
// half a cell and rotated, so every star overlaps a few neighbours (about 40 crossings per star
// pair). `intersection` takes the two operands; `dissolve` takes the left operand's polygons as
// separate features (neighbours overlap, so the union merges them into large components).
//
// Area tolerance. The new side snaps every vertex to a lattice of cell `c` (the power of two at least
// the grid size) and the legacy side keeps f32 coordinates, so the areas differ by the snapping
// displacement. A vertex moves by at most c/2 per axis (half a cell, round half up), so each boundary
// edge of length L moves by at most c * sqrt(2)/2 in the normal direction, changing the area by at most
// L * c * sqrt(2)/2 per edge on each side of the result. The perimeter P of the result is bounded by
// the total length of the input edges that survive, which is at most the sum of the lengths of all input
// edges, `perimeterBound`. Crossings are re-rounded too (at most c/2 per axis), which adds the same
// order again, so the bound used is 2 * perimeterBound * c * sqrt(2)/2 = perimeterBound * c * sqrt(2).
// The legacy output has its own f32 rounding (about 6e-8 relative, far below c), neglected. The test
// asserts |areaNew - areaLegacy| <= perimeterBound * c * sqrt(2) and, where the legacy side is skipped,
// only that the new side is within the same bound of the analytic upper bound (the area of the union
// of the two operands' bounding squares).

const ENABLED = Reflect.get(process.env, 'A5_AB') === '1';
const LEGACY_MAXIMUM_VERTICES = 10000;
const STAR_VERTICES = 20;

async function measure(
  device: Parameters<typeof prepareOverlay>[0],
  props: OverlayRunProps,
  name: string
) {
  let prepared = prepareOverlay(device, props, name);
  let capacity = props.capacity;
  try {
    await prepared.submitOnce(); // untimed sizing/compilation pass
    const plan = await prepared.read();
    if (props.implementation === 'current') {
      // The old stars * 4 ring guess is too small (521 rings at n=1000 intersection).
      // Plan from the contributor's unclamped counts before recording any timings. This also
      // removes the enormous unused noded/boundary tails from the measured snap-rounded graph.
      capacity = planOverlayCapacity(capacity, plan);
      prepared.destroy();
      prepared = prepareOverlay(device, {...props, capacity}, `${name}-sized`);
      await prepared.submitOnce();
    } else if (plan.overflow && !plan.boundaryOverflow && !plan.candidateOverflow) {
      // Legacy ring counts can differ after snapping. Keep its scratch budget and grow outputs.
      capacity = {...capacity, rings: Math.max(capacity.rings, plan.requiredRingCount)};
      prepared.destroy();
      prepared = prepareOverlay(device, {...props, capacity}, `${name}-sized`);
      await prepared.submitOnce();
    }
    const {rings, ...validation} = await prepared.read();
    expect(validation.overflow, `sized ${name}: ${JSON.stringify(validation)}`).toBe(0);
    expect(validation.candidateOverflow).toBe(0);
    let best = Number.POSITIVE_INFINITY;
    for (let attempt = 0; attempt < 3; attempt++) {
      best = Math.min(best, (await prepared.submitOnce()).milliseconds);
    }
    const readout = await prepared.read();
    const area = readout.rings.reduce((sum, ring) => {
      const doubled = ring.vertices.reduce((acc, [x, y], i) => {
        const [nextX, nextY] = ring.vertices[(i + 1) % ring.vertices.length];
        return acc + (x * nextY - nextX * y);
      }, 0);
      // Normalized orientation is not guaranteed by the legacy side: shells add, holes subtract.
      return sum + (ring.isShell ? Math.abs(doubled) : -Math.abs(doubled)) / 2;
    }, 0);
    return {best, compileMilliseconds: prepared.compileMilliseconds, readout, area, capacity};
  } finally {
    prepared.destroy();
  }
}

function getCapacity(vertexCount: number): OverlayRunCapacity {
  // About 40 crossings per star pair and 3 pairs per star; generous headroom.
  const stars = Math.ceil(vertexCount / STAR_VERTICES);
  return {
    intersections: Math.max(4096, stars * 400),
    nodedSegments: Math.max(16384, stars * 1600),
    vertices: Math.max(8192, stars * 400),
    rings: Math.max(512, stars * 4),
    boundary: Math.max(8192, stars * 800)
  };
}

for (const vertexCount of [1000, 10000, 100000]) {
  for (const operation of ['intersection', 'dissolve'] as const) {
    it.skipIf(!ENABLED)(
      `overlay A/B ${operation} at ${vertexCount} vertices per operand`,
      async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const scene = createOverlayABScene(vertexCount);
        const capacity = getCapacity(vertexCount);
        const base = {
          operation,
          left: scene.left,
          right: operation === 'dissolve' ? undefined : scene.right,
          // Power-of-two grid fine enough to keep the stars' shape: 1/64 of a unit.
          vertexTolerance: 1 / 64,
          capacity
        };
        const current = await measure(
          device,
          {...base, implementation: 'current'},
          `ab-current-${vertexCount}`
        );
        expect(current.readout.overflow, 'new overflow').toBe(0);
        expect(current.readout.candidateOverflow, 'new candidateOverflow').toBe(0);
        // Cell of the new lattice: the power of two at least max(gridSize, extent * 2^-22).
        const cell = 2 ** Math.ceil(Math.log2(Math.max(1 / 64, scene.extent * 2 ** -22)));
        const tolerance = scene.perimeterBound * cell * Math.SQRT2;
        let legacyText = 'legacy infeasible (O(pieces x vertices) membership)';
        let ratioText = 'n/a';
        if (vertexCount <= LEGACY_MAXIMUM_VERTICES) {
          const legacy = await measure(
            device,
            {...base, implementation: 'legacy'},
            `ab-legacy-${vertexCount}`
          );
          expect(legacy.readout.overflow, 'legacy overflow').toBe(0);
          expect(
            Math.abs(current.area - legacy.area),
            'area difference within the snapping bound'
          ).toBeLessThanOrEqual(tolerance);
          legacyText = `legacy ${legacy.best.toFixed(1)} ms`;
          ratioText = `${(legacy.best / current.best).toFixed(1)}x`;
        } else {
          // No legacy reference: the area must at least be positive and inside the scene.
          expect(current.area).toBeGreaterThan(0);
          expect(current.area).toBeLessThanOrEqual(scene.extent * scene.extent);
        }
        // eslint-disable-next-line no-console
        console.log(
          `overlay-ab ${operation} n=${vertexCount} ${legacyText} | snap-rounded ${current.best.toFixed(1)} ms | speedup ${ratioText} | compile ${current.compileMilliseconds.toFixed(0)} ms | capacity ${JSON.stringify(current.capacity)}`
        );
      },
      1_800_000
    );
  }
}
