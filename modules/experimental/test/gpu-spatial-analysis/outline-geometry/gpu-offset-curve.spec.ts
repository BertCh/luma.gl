// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUOffsetCurveParameterValues,
  getGPUOffsetCurveRowsPerVertex,
  GPUOffsetCurve
} from '../../../src/gpu-spatial-analysis/outline-geometry/index';
import {createGeometryFixture} from './geometry-fixture';
import {OFFSET_CURVE_REFERENCE_CASES} from './offset-curve-reference';

// Reference values are pinned from shapely 2.1.2 (GEOS) `offset_curve`; see offset-curve-reference.ts.
for (const referenceCase of OFFSET_CURVE_REFERENCE_CASES) {
  it(`GPUOffsetCurve matches shapely.offset_curve: ${referenceCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const {vertices, joinStyle, quadSegments, ring} = referenceCase;
    const rowsPerVertex = getGPUOffsetCurveRowsPerVertex(joinStyle, quadSegments);
    const fixture = createGeometryFixture(device, {
      inputs: {
        positions: {values: new Float32Array(vertices.flat()), format: 'float32x2'},
        pathOffsets: {values: new Uint32Array([0, vertices.length]), format: 'uint32'}
      },
      outputs: {
        output: {format: 'float32x2', length: vertices.length * rowsPerVertex},
        counts: {format: 'uint32', length: vertices.length}
      },
      create: ({inputs, outputs, parameters}) =>
        new GPUOffsetCurve({
          positions: inputs['positions'] as never,
          pathOffsets: inputs['pathOffsets'] as never,
          geometryType: ring ? 'rings' : 'lines',
          joinStyle,
          quadSegments,
          parameters,
          output: {positions: outputs['output'] as never, counts: outputs['counts'] as never}
        })
    });
    const result = await fixture.run(
      getGPUOffsetCurveParameterValues({
        distance: referenceCase.distance,
        mitreLimit: referenceCase.mitreLimit
      })
    );
    const actual: number[][] = [];
    for (let vertex = 0; vertex < vertices.length; vertex++) {
      for (let slot = 0; slot < result['counts'][vertex]; slot++) {
        const row = vertex * rowsPerVertex + slot;
        actual.push([result['output'][2 * row], result['output'][2 * row + 1]]);
      }
    }
    // GEOS repeats the first point at the end of a closed ring; the contributor emits each corner once.
    const expected = ring ? referenceCase.expected.slice(0, -1) : referenceCase.expected;
    const matches = (a: number[], b: number[]) =>
      Math.abs(a[0] - b[0]) < 2e-4 && Math.abs(a[1] - b[1]) < 2e-4;
    if (ring && referenceCase.distance < 0) {
      // On the outside of a ring GEOS leaves the corner at the ring's first vertex open and its
      // mitre output drops it; the contributor closes it, so GEOS's points must all be present.
      expect(actual.length).toBeGreaterThanOrEqual(expected.length);
      for (const point of expected) {
        expect(
          actual.some(candidate => matches(candidate, point)),
          `missing ${point}`
        ).toBe(true);
      }
    } else if (ring) {
      expect(actual.length, 'point count').toBe(expected.length);
      for (const point of expected) {
        expect(
          actual.some(candidate => matches(candidate, point)),
          `missing ${point}`
        ).toBe(true);
      }
    } else {
      expect(actual.length, 'point count').toBe(expected.length);
      expected.forEach((point, index) => {
        expect(matches(actual[index], point), `point ${index} ${actual[index]} vs ${point}`).toBe(
          true
        );
      });
    }
    // Padding rows repeat the last real point, so drawing the whole buffer adds no geometry.
    if (!ring) {
      const lastVertex = vertices.length - 1;
      const count = result['counts'][lastVertex];
      for (let slot = count; slot < rowsPerVertex; slot++) {
        const row = lastVertex * rowsPerVertex + slot;
        const lastReal = lastVertex * rowsPerVertex + count - 1;
        expect(result['output'][2 * row]).toBe(result['output'][2 * lastReal]);
      }
    }
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  });
}

it('GPUOffsetCurve validates parameters and output sizes', () => {
  expect(() => getGPUOffsetCurveParameterValues({distance: 1, mitreLimit: 0.5})).toThrow();
  expect(() => getGPUOffsetCurveParameterValues({distance: Number.NaN})).toThrow();
  expect(Array.from(getGPUOffsetCurveParameterValues({distance: -2}))).toEqual([-2, 5, 0, 0]);
  expect(getGPUOffsetCurveRowsPerVertex('round', 8)).toBe(17);
  expect(getGPUOffsetCurveRowsPerVertex('mitre', 8)).toBe(2);
});
