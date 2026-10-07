// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPULineSimplification} from '../../../src/gpu-spatial-analysis/line-simplification';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeVisvalingamImportance, getKeptRowsAboveTolerance} from './visvalingam-oracle';
import {VISVALINGAM_REFERENCE_SCENES} from './visvalingam-reference';

/** Symmetric difference sizes between the CPU mirror and `simplify_coords_vw`, per scene. */
function measureMismatch(name: string, radius: number): {mismatched: number; total: number} {
  const scene = VISVALINGAM_REFERENCE_SCENES[name];
  const positions = Float32Array.from(scene.positions);
  const rowCount = positions.length / 2;
  const {importanceBits, converged} = computeVisvalingamImportance(
    positions,
    Uint32Array.of(0, rowCount),
    radius,
    1024
  );
  expect(converged).toBe(true);
  let mismatched = 0;
  for (const [index, tolerance] of scene.tolerances.entries()) {
    const kept = new Set(getKeptRowsAboveTolerance(importanceBits, tolerance));
    const reference = new Set(scene.kept[index]);
    for (const row of new Set([...kept, ...reference])) {
      mismatched += kept.has(row) === reference.has(row) ? 0 : 1;
    }
  }
  return {mismatched, total: scene.tolerances.length * rowCount};
}

it('Visvalingam rounds reproduce simplify_coords_vw exactly on the pinned small scenes', () => {
  for (const name of ['hand', 'walk60', 'smooth120']) {
    expect(measureMismatch(name, 3), name).toEqual({
      mismatched: 0,
      total:
        VISVALINGAM_REFERENCE_SCENES[name].tolerances.length *
        (VISVALINGAM_REFERENCE_SCENES[name].positions.length / 2)
    });
  }
});

it('Visvalingam rounds stay within 1 percent of simplify_coords_vw on a 300-vertex walk', () => {
  const byRadius = [1, 2, 3, 4].map(radius => measureMismatch('walk300', radius));
  console.log(
    `visvalingam walk300 mismatched vertices per 6 tolerances by radius 1..4: ${byRadius
      .map(result => result.mismatched)
      .join(', ')} of ${byRadius[0].total}`
  );
  expect(byRadius[2].mismatched / byRadius[2].total).toBeLessThan(0.01);
  // Wider neighborhoods follow the sequential order at least as closely.
  expect(byRadius[2].mismatched).toBeLessThanOrEqual(byRadius[0].mismatched);
});

it('Visvalingam round cap leaves undecided rows at +Infinity (superset)', () => {
  const scene = VISVALINGAM_REFERENCE_SCENES.walk300;
  const positions = Float32Array.from(scene.positions);
  const capped = computeVisvalingamImportance(positions, Uint32Array.of(0, 300), 3, 2);
  expect(capped.converged).toBe(false);
  const full = computeVisvalingamImportance(positions, Uint32Array.of(0, 300), 3, 1024);
  for (let row = 0; row < 300; row++) {
    expect(capped.importanceBits[row]).toBeGreaterThanOrEqual(full.importanceBits[row]);
  }
});

it('GPULineSimplification validates Visvalingam options', async () => {
  const device = await createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'visvalingam-validation'});
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const create = (overrides: Record<string, unknown>) =>
    new GPULineSimplification({
      positions: view('float32x2', 5),
      trackOffsets: view('uint32', 2),
      importance: view('float32', 5),
      method: 'visvalingam',
      ...overrides
    });
  expect(() => create({})).not.toThrow();
  expect(() => create({neighborhoodRadius: 0})).toThrow(/neighborhoodRadius/);
  expect(() => create({neighborhoodRadius: 9})).toThrow(/neighborhoodRadius/);
  expect(() => create({neighborhoodRadius: 1.5})).toThrow(/neighborhoodRadius/);
  expect(() => create({metric: 'time-ratio'})).toThrow(/visvalingam/);
  expect(() => create({method: 'other'})).toThrow(/method/);
  expect(create({}).maximumRounds).toBe(256);
  expect(create({maximumRounds: 8}).maximumRounds).toBe(8);
  const nodes = create({}).getCommandNodes(graph);
  expect(nodes.length).toBeGreaterThan(256 * 4);
});
