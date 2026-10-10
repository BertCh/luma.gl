// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUAffineTransformParameters,
  getGPUGeometryCleanupParameterValues,
  getGPUGeometryOrientationParameterValues,
  GPUAffineTransform,
  GPUGeometryCleanup,
  GPUGeometryOrientation
} from '../../../src/gpu-spatial-analysis/geometry-edit/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GEOMETRY_EDIT_FIXTURES} from './geometry-edit-fixtures';

let serial = 0;

it('getGPUAffineTransformParameters matches shapely scale, skew, rotate, translate', () => {
  for (const testCase of GEOMETRY_EDIT_FIXTURES.packerCases) {
    const values = getGPUAffineTransformParameters({
      rotate: testCase.rotate,
      scale: testCase.scale as [number, number],
      skew: testCase.skew as [number, number],
      translate: testCase.translate as [number, number],
      origin: testCase.origin as [number, number]
    });
    expect(values).toHaveLength(12);
    const [a, b, d, e, translateX, translateY, originX, originY, originMode] = values;
    expect(originMode).toBe(0);
    testCase.probe.forEach(([x, y], vertex) => {
      const relativeX = x - originX;
      const relativeY = y - originY;
      const actual = [
        a * relativeX + b * relativeY + originX + translateX,
        d * relativeX + e * relativeY + originY + translateY
      ];
      expect(actual[0]).toBeCloseTo(testCase.expected[vertex][0], 5);
      expect(actual[1]).toBeCloseTo(testCase.expected[vertex][1], 5);
    });
  }
});

it('geometry edit parameter packers encode and validate values', () => {
  expect(Array.from(getGPUAffineTransformParameters({origin: 'center'}).slice(0, 9))).toEqual([
    1, 0, 0, 1, 0, 0, 0, 0, 1
  ]);
  expect(getGPUAffineTransformParameters({origin: 'centroid'})[8]).toBe(2);
  const radians = getGPUAffineTransformParameters({rotate: Math.PI / 2, useRadians: true});
  expect(radians[0]).toBeCloseTo(0, 6);
  expect(radians[2]).toBeCloseTo(1, 6);
  expect(() => getGPUAffineTransformParameters({rotate: Number.NaN})).toThrow(/finite/);
  expect(() => getGPUAffineTransformParameters({}, new Float32Array(4))).toThrow(/12/);
  expect(Array.from(getGPUGeometryOrientationParameterValues({exteriorClockwise: true}))).toEqual([
    1, 0, 0, 0
  ]);
  expect(Array.from(getGPUGeometryCleanupParameterValues({gridSize: 0.5, tolerance: 2}))).toEqual([
    0.5, 2, 1, 0
  ]);
  expect(getGPUGeometryCleanupParameterValues({removeRepeatedPoints: false})[2]).toBe(0);
  expect(() => getGPUGeometryCleanupParameterValues({tolerance: -1})).toThrow(/non-negative/);
});

it('geometry edit contributors validate props and declare nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'geometry-edit-node'});
  const view = <Format extends 'float32x2' | 'uint32' | 'float32' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `edit-${serial++}`, format, length);

  const orientation = {
    positions: view('float32x2', 12),
    ringOffsets: view('uint32', 3),
    polygonOffsets: view('uint32', 2),
    parameters: view('float32', 4),
    output: {positions: view('float32x2', 12)}
  };
  expect(
    new GPUGeometryOrientation({...orientation, mode: 'orient-polygons'}).getCommandNodes(graph)
  ).toHaveLength(2);
  expect(
    () =>
      new GPUGeometryOrientation({
        positions: orientation.positions,
        ringOffsets: orientation.ringOffsets,
        mode: 'orient-polygons',
        output: orientation.output
      })
  ).toThrow(/requires polygonOffsets/);
  expect(
    () =>
      new GPUGeometryOrientation({
        ...orientation,
        mode: 'reverse',
        output: {positions: view('float32x2', 5)}
      })
  ).toThrow(/length must equal/);
  expect(
    () =>
      new GPUGeometryOrientation({...orientation, mode: 'reverse', output: orientation as never})
  ).toThrow();

  const cleanup = {
    positions: view('float32x2', 12),
    ringOffsets: view('uint32', 3),
    geometryType: 'polygons' as const,
    parameters: view('float32', 4),
    output: {
      positions: view('float32x2', 12),
      ringOffsets: view('uint32', 3),
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      collapsedRings: view('uint32', 1)
    }
  };
  // Clear x2, flags, scan (at least one node), emit, publish.
  expect(new GPUGeometryCleanup(cleanup).getCommandNodes(graph).length).toBeGreaterThanOrEqual(6);
  expect(
    () =>
      new GPUGeometryCleanup({
        ...cleanup,
        output: {...cleanup.output, ringOffsets: view('uint32', 2)}
      })
  ).toThrow(/ringOffsets length/);
  expect(
    () => new GPUGeometryCleanup({...cleanup, ringOffsets: cleanup.output.ringOffsets})
  ).toThrow(/disjoint|alias|share/i);

  const pointCleanup = {
    positions: view('float32x2', 12),
    geometryType: 'points' as const,
    parameters: view('float32', 4),
    output: {
      positions: view('float32x2', 8),
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1)
    }
  };
  expect(new GPUGeometryCleanup(pointCleanup).getCommandNodes(graph)).toHaveLength(1);
  expect(
    () =>
      new GPUGeometryCleanup({
        ...pointCleanup,
        ringOffsets: view('uint32', 2)
      } as never)
  ).toThrow(/point geometry.*offsets/);
  expect(
    () =>
      new GPUGeometryCleanup({
        ...pointCleanup,
        output: {...pointCleanup.output, ringOffsets: view('uint32', 2)}
      } as never)
  ).toThrow(/point geometry.*offsets/);
  expect(
    () =>
      new GPUGeometryCleanup({
        ...pointCleanup,
        output: {...pointCleanup.output, collapsedRings: view('uint32', 1)}
      } as never)
  ).toThrow(/point geometry.*offsets/);

  const affine = {
    positions: view('float32x2', 12),
    parameters: view('float32', 12),
    output: {positions: view('float32x2', 12)}
  };
  expect(new GPUAffineTransform(affine).getCommandNodes(graph)).toHaveLength(1);
  expect(() => new GPUAffineTransform({...affine, featureTransforms: view('float32', 6)})).toThrow(
    /exactly one/
  );
  expect(() => new GPUAffineTransform({...affine, origins: ['center']})).toThrow(/ringOffsets/);
  const withRings = {...affine, ringOffsets: view('uint32', 3)};
  expect(
    new GPUAffineTransform({...withRings, origins: ['center', 'centroid']}).getCommandNodes(graph)
      .length
  ).toBeGreaterThan(1);
  expect(
    () => new GPUAffineTransform({...withRings, origins: ['centroid'], geometryType: 'points'})
  ).toThrow(/centroid/);
  expect(
    () =>
      new GPUAffineTransform({
        positions: affine.positions,
        ringOffsets: withRings.ringOffsets,
        featureTransforms: view('float32', 5),
        output: affine.output
      })
  ).toThrow(/6 rows per feature/);
});
