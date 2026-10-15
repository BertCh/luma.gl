// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture} from '@luma.gl/gpgpu';
import {describe, expect, it} from 'vitest';
import {toRecordedLensFile} from '../../../src/computation-lenses/registry';
import type {LensRecord} from '../../../src/computation-views/registry/view-registration';
import {buildLensTwin, validateLensTwin} from '../../../src/computation-views/twin/static-twin';
import '../../../src/gpu-terrain/computation-pack/index';
import {
  RASTER_CONVERGENCE_GUARD,
  RASTER_METRIC_CELL_SIZE_GUARD
} from '../../../src/gpu-raster/computation-pack/guards';

const raster = {
  kind: 'raster',
  shape: [3, 3],
  coordinateModel: 'projected',
  cellSize: 120,
  unit: 'm'
} as const;

function capture(
  name: string,
  values: number[],
  metadata: Partial<GPUComputationCapture> = {}
): GPUComputationCapture {
  return {
    id: `${name}-0`,
    name,
    values,
    elementType: 'f32',
    provenance: 'gpu-readback',
    keySpace: 'cells',
    ...metadata
  };
}

/** A 3×3 bowl draining south-west from the centre: the GPU chose SW (code 8). */
function makeRecord(snapshots: GPUComputationCapture[]): LensRecord {
  return {
    schema: 'luma.computation/1',
    id: 'gpu-terrain.terrain-flow',
    runId: 'run-d8',
    createdAt: '2026-10-14T00:00:00.000Z',
    adapter: {backend: 'webgpu', description: 'Fixture adapter'},
    domains: ['gpu-terrain'],
    graphs: [],
    axes: {
      axes: [],
      keySpaces: [{id: 'cells', label: 'grid cells', size: 9, indexSpace: '2d', shape: [3, 3]}],
      relations: []
    },
    refs: {formulas: [], sources: []},
    decisions: [],
    choices: [{stage: 'support', name: 'cellSizeMode', value: 'uniform', sweepable: false}],
    captures: {
      schema: 'luma.computation-captures/1',
      runId: 'run-d8',
      snapshots,
      stamps: [],
      replays: []
    }
  };
}

const ELEVATION = [105, 104, 103, 104, 100, 101, 95, 102, 103];

describe('terrain D8 stencil view (M14)', () => {
  const record = makeRecord([
    capture('dem', ELEVATION, {role: 'input', geometry: raster, meaning: 'elevation (m)'}),
    capture('flowDirections', [0, 0, 0, 0, 8, 0, 0, 0, 0], {
      elementType: 'u32',
      role: 'intermediate',
      geometry: raster,
      meaning: 'D8 flow direction (ESRI code)'
    })
  ]);
  const file = toRecordedLensFile(record, {
    scenarioId: 'gpu-terrain.terrain-flow',
    recordedAt: '2026-10-14',
    adapterLabel: 'Fixture adapter'
  });

  it('wins the datum rung, shows √2 diagonals and the GPU receiver', () => {
    const twin = buildLensTwin(file, {
      scenario: 'gpu-terrain.terrain-flow',
      rung: 'datum',
      datum: '4',
      caption: 'D8 stencil'
    });
    expect(validateLensTwin(twin)).toEqual([]);
    expect(twin.viewId).toBe('terrain-d8-stencil');
    expect(twin.keyNumber.value).toBe('SW');
    expect(twin.figure.html).toContain('×1.414');
    const southWest = twin.receipt?.rows.find(row => row[0] === 'SW');
    // Drop 5 m over √2 × 120 m.
    expect(Number(southWest?.[3])).toBeCloseTo(5 / (Math.SQRT2 * 120), 4);
    expect(southWest?.[4]).toBe('receiver');
    expect(twin.description).toContain('1.414');
  });

  it('stays out of records without a D8 capture', () => {
    const plain = toRecordedLensFile(
      makeRecord([
        capture('dem', ELEVATION, {role: 'input', geometry: raster, meaning: 'elevation (m)'})
      ]),
      {
        scenarioId: 'gpu-terrain.terrain-flow',
        recordedAt: '2026-10-14',
        adapterLabel: 'Fixture adapter'
      }
    );
    const twin = buildLensTwin(plain, {
      scenario: 'gpu-terrain.terrain-flow',
      rung: 'datum',
      datum: '4',
      caption: 'x'
    });
    expect(twin.viewId).not.toBe('terrain-d8-stencil');
  });
});

describe('raster guards', () => {
  it('fails uniform cell sizes on a geographic input and passes projected ones', () => {
    const projected = makeRecord([capture('dem', ELEVATION, {role: 'input', geometry: raster})]);
    expect(RASTER_METRIC_CELL_SIZE_GUARD.evaluate({metadata: {}, record: projected}).status).toBe(
      'pass'
    );
    const geographic = makeRecord([
      capture('dem', ELEVATION, {
        role: 'input',
        geometry: {...raster, coordinateModel: 'geographic'}
      })
    ]);
    expect(RASTER_METRIC_CELL_SIZE_GUARD.evaluate({metadata: {}, record: geographic}).status).toBe(
      'fail'
    );
    expect(
      RASTER_METRIC_CELL_SIZE_GUARD.evaluate({metadata: {}, record: {...projected, choices: []}})
        .status
    ).toBe('unknown');
  });

  it('reads convergence from captured GPU status words', () => {
    const converged = makeRecord([capture('fillConverged', [1], {meaning: 'converged: fill'})]);
    const stopped = makeRecord([capture('fillConverged', [0], {meaning: 'converged: fill'})]);
    expect(RASTER_CONVERGENCE_GUARD.evaluate({metadata: {}, record: converged}).status).toBe(
      'pass'
    );
    expect(RASTER_CONVERGENCE_GUARD.evaluate({metadata: {}, record: stopped}).status).toBe('fail');
    expect(RASTER_CONVERGENCE_GUARD.evaluate({metadata: {}, record: makeRecord([])}).status).toBe(
      'unknown'
    );
  });
});
