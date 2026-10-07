// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUVertexSnapParameterValues,
  GPU_VERTEX_SNAP_NO_REFERENCE,
  GPUVertexSnap
} from '../../../src/gpu-spatial-analysis/vertex-snap/index';
import {createGeometryFixture} from './geometry-fixture';
import {VERTEX_SNAP_REFERENCE_CASES, VERTEX_SNAP_TOLERANCE} from './offset-curve-reference';

it('GPUVertexSnap matches shapely.snap on vertex-only cases', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  let snappedTotal = 0;
  for (const referenceCase of VERTEX_SNAP_REFERENCE_CASES) {
    const fixture = createGeometryFixture(device, {
      inputs: {
        positions: {values: new Float32Array(referenceCase.vertices.flat()), format: 'float32x2'},
        references: {values: new Float32Array(referenceCase.refs.flat()), format: 'float32x2'}
      },
      outputs: {
        output: {format: 'float32x2', length: referenceCase.vertices.length},
        rows: {format: 'uint32', length: referenceCase.vertices.length}
      },
      create: ({inputs, outputs, parameters}) =>
        new GPUVertexSnap({
          positions: inputs['positions'] as never,
          referencePositions: inputs['references'] as never,
          parameters,
          output: {positions: outputs['output'] as never, referenceRows: outputs['rows'] as never}
        })
    });
    const result = await fixture.run(
      getGPUVertexSnapParameterValues({tolerance: VERTEX_SNAP_TOLERANCE})
    );
    referenceCase.expected.forEach((point, index) => {
      expect(Math.abs(result['output'][2 * index] - point[0]), `x ${index}`).toBeLessThan(1e-5);
      expect(Math.abs(result['output'][2 * index + 1] - point[1]), `y ${index}`).toBeLessThan(1e-5);
      if (result['rows'][index] !== GPU_VERTEX_SNAP_NO_REFERENCE) {
        snappedTotal++;
      }
    });
    fixture.destroy();
  }
  expect(VERTEX_SNAP_REFERENCE_CASES.length).toBeGreaterThan(50);
  expect(snappedTotal).toBeGreaterThan(50);
});

it('GPUVertexSnap picks the nearest reference, lowest row on ties, and supports pairwise features', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Feature 0 (rows 0-1) owns references 0-1, feature 1 (row 2) owns reference 2. Row 1 sits
  // exactly between references 0 and 1 mirrored about x = 5, so the lower row 0 wins; the farther
  // reference of row 0 (0.4 away) loses to the nearer one (0.1).
  const positions = new Float32Array([0, 0, 5, 0, 20, 20]);
  const references = new Float32Array([0, 0.4, 0, 0.1, 5, 0.25, 5, -0.25, 20.1, 20]);
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {values: positions, format: 'float32x2'},
      references: {values: references, format: 'float32x2'},
      featureOffsets: {values: new Uint32Array([0, 2, 3]), format: 'uint32'},
      referenceOffsets: {values: new Uint32Array([0, 4, 5]), format: 'uint32'}
    },
    outputs: {
      output: {format: 'float32x2', length: 3},
      rows: {format: 'uint32', length: 3}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUVertexSnap({
        positions: inputs['positions'] as never,
        referencePositions: inputs['references'] as never,
        featureOffsets: inputs['featureOffsets'] as never,
        referenceOffsets: inputs['referenceOffsets'] as never,
        parameters,
        output: {positions: outputs['output'] as never, referenceRows: outputs['rows'] as never}
      })
  });
  const first = await fixture.run(getGPUVertexSnapParameterValues({tolerance: 0.5}));
  expect(first['rows']).toEqual([1, 2, 4]);
  expect(first['output'][0]).toBeCloseTo(0, 6);
  expect(first['output'][1]).toBeCloseTo(0.1, 6);
  expect(first['output'][3]).toBeCloseTo(0.25, 6);
  expect(first['output'][4]).toBeCloseTo(20.1, 4);
  // Per-frame tolerance: zero snaps nothing and keeps inputs bit-for-bit; no recompile.
  const second = await fixture.run(getGPUVertexSnapParameterValues({tolerance: 0}));
  expect(second['rows']).toEqual([
    GPU_VERTEX_SNAP_NO_REFERENCE,
    GPU_VERTEX_SNAP_NO_REFERENCE,
    GPU_VERTEX_SNAP_NO_REFERENCE
  ]);
  expect(second['output']).toEqual([0, 0, 5, 0, 20, 20]);
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPUVertexSnap validates the tolerance', () => {
  expect(() => getGPUVertexSnapParameterValues({tolerance: -1})).toThrow();
  expect(Array.from(getGPUVertexSnapParameterValues({tolerance: 2}))).toEqual([2, 0, 0, 0]);
});

it('GPUVertexSnap BVH path (many references) equals a linear scan, including duplicate and tied references', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  let seed = 12345;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 0x100000000;
  };
  // 301 references on an integer-ish grid so exact ties are common; every 7th repeats an earlier one.
  const referenceCount = 301;
  const references = new Float32Array(referenceCount * 2);
  for (let row = 0; row < referenceCount; row++) {
    if (row % 7 === 6) {
      const source = Math.floor(random() * row);
      references[row * 2] = references[source * 2];
      references[row * 2 + 1] = references[source * 2 + 1];
    } else {
      references[row * 2] = Math.floor(random() * 20);
      references[row * 2 + 1] = Math.floor(random() * 20);
    }
  }
  // Vertices: half exactly on reference-lattice midpoints (ties), half random; size not a multiple of 256.
  const vertexCount = 517;
  const positions = new Float32Array(vertexCount * 2);
  for (let row = 0; row < vertexCount; row++) {
    positions[row * 2] = row % 2 ? Math.floor(random() * 40) / 2 : random() * 20;
    positions[row * 2 + 1] = row % 2 ? Math.floor(random() * 40) / 2 : random() * 20;
  }
  const tolerance = 1.25;
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {values: positions, format: 'float32x2'},
      references: {values: references, format: 'float32x2'}
    },
    outputs: {
      output: {format: 'float32x2', length: vertexCount},
      rows: {format: 'uint32', length: vertexCount}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUVertexSnap({
        positions: inputs['positions'] as never,
        referencePositions: inputs['references'] as never,
        parameters,
        output: {positions: outputs['output'] as never, referenceRows: outputs['rows'] as never}
      })
  });
  const result = await fixture.run(getGPUVertexSnapParameterValues({tolerance}));
  const toleranceSquared = Math.fround(tolerance * tolerance);
  let snapped = 0;
  for (let row = 0; row < vertexCount; row++) {
    let bestRow = GPU_VERTEX_SNAP_NO_REFERENCE;
    let best = toleranceSquared;
    for (let reference = 0; reference < referenceCount; reference++) {
      const dx = Math.fround(references[reference * 2] - positions[row * 2]);
      const dy = Math.fround(references[reference * 2 + 1] - positions[row * 2 + 1]);
      const distanceSquared = Math.fround(Math.fround(dx * dx) + Math.fround(dy * dy));
      if (distanceSquared < best) {
        best = distanceSquared;
        bestRow = reference;
      }
    }
    if (bestRow !== GPU_VERTEX_SNAP_NO_REFERENCE) {
      snapped++;
    }
    expect(result['rows'][row], `row ${row}`).toBe(bestRow);
  }
  expect(snapped).toBeGreaterThan(100);
  fixture.destroy();
});

it('GPUVertexSnap BVH path skips non-finite references exactly like the linear scan', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  let seed = 777;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 0x100000000;
  };
  const referenceCount = 200;
  const references = new Float32Array(referenceCount * 2);
  for (let row = 0; row < referenceCount; row++) {
    references[row * 2] = random() * 10;
    references[row * 2 + 1] = random() * 10;
  }
  // Non-finite references, including mixed infinities in one box and components of either axis.
  const bad: [number, number][] = [
    [Infinity, 1],
    [-Infinity, 2],
    [3, Infinity],
    [4, -Infinity],
    [NaN, 5],
    [6, NaN],
    [NaN, NaN],
    [Infinity, -Infinity],
    [-Infinity, Infinity]
  ];
  bad.forEach(([x, y], index) => {
    const row = 5 + index * 19;
    references[row * 2] = x;
    references[row * 2 + 1] = y;
  });
  // Finite vertices placed 0.05 from finite references, plus random ones.
  const vertexCount = 300;
  const positions = new Float32Array(vertexCount * 2);
  for (let row = 0; row < vertexCount; row++) {
    if (row < 150) {
      const reference = (row * 3) % referenceCount;
      const x = references[reference * 2];
      const y = references[reference * 2 + 1];
      positions[row * 2] = Number.isFinite(x) ? x + 0.05 : random() * 10;
      positions[row * 2 + 1] = Number.isFinite(y) ? y - 0.05 : random() * 10;
    } else {
      positions[row * 2] = random() * 10;
      positions[row * 2 + 1] = random() * 10;
    }
  }
  const tolerance = 0.3;
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {values: positions, format: 'float32x2'},
      references: {values: references, format: 'float32x2'}
    },
    outputs: {
      output: {format: 'float32x2', length: vertexCount},
      rows: {format: 'uint32', length: vertexCount}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUVertexSnap({
        positions: inputs['positions'] as never,
        referencePositions: inputs['references'] as never,
        parameters,
        output: {positions: outputs['output'] as never, referenceRows: outputs['rows'] as never}
      })
  });
  const result = await fixture.run(getGPUVertexSnapParameterValues({tolerance}));
  const toleranceSquared = Math.fround(tolerance * tolerance);
  let snapped = 0;
  for (let row = 0; row < vertexCount; row++) {
    let bestRow = GPU_VERTEX_SNAP_NO_REFERENCE;
    let best = toleranceSquared;
    for (let reference = 0; reference < referenceCount; reference++) {
      const dx = Math.fround(references[reference * 2] - positions[row * 2]);
      const dy = Math.fround(references[reference * 2 + 1] - positions[row * 2 + 1]);
      const distanceSquared = Math.fround(Math.fround(dx * dx) + Math.fround(dy * dy));
      if (distanceSquared < best) {
        best = distanceSquared;
        bestRow = reference;
      }
    }
    if (bestRow !== GPU_VERTEX_SNAP_NO_REFERENCE) {
      snapped++;
    }
    expect(result['rows'][row], `row ${row}`).toBe(bestRow);
  }
  expect(snapped).toBeGreaterThan(100);
  fixture.destroy();
});
