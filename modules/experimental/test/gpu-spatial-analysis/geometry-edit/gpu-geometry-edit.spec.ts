// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUAffineTransformParameters,
  getGPUGeometryCleanupParameterValues,
  getGPUGeometryOrientationParameterValues,
  GPUAffineTransform,
  GPUGeometryCleanup,
  GPUGeometryOrientation
} from '../../../src/gpu-spatial-analysis/geometry-edit/index';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {GEOMETRY_EDIT_FIXTURES} from './geometry-edit-fixtures';

type Ring = number[][];
type Polygon = Ring[];

/** Flattens polygons into GeoArrow-style positions and offsets. */
function flattenPolygons(polygons: Polygon[]) {
  const positions: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  for (const polygon of polygons) {
    for (const ring of polygon) {
      for (const [x, y] of ring) {
        positions.push(x, y);
      }
      ringOffsets.push(positions.length / 2);
    }
    polygonOffsets.push(ringOffsets.length - 1);
  }
  return {
    positions: new Float32Array(positions),
    ringOffsets: new Uint32Array(ringOffsets),
    polygonOffsets: new Uint32Array(polygonOffsets)
  };
}

/** Flattens rings (or paths) into positions and ring offsets. */
function flattenRings(rings: Ring[]) {
  return flattenPolygons(rings.map(ring => [ring]));
}

function toRows(values: number[], count: number): number[][] {
  return Array.from({length: count}, (_, row) => [values[2 * row], values[2 * row + 1]]);
}

/** Compiles one contributor with imported inputs and returns a runner. */
function createRun(
  device: Device,
  inputs: Record<string, Float32Array | Uint32Array>,
  parameterLength: number
) {
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'geometry-edit-spec'});
  const views: Record<string, GraphDataView> = {};
  for (const [name, values] of Object.entries(inputs)) {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    const format = values instanceof Float32Array ? 'float32x2' : 'uint32';
    views[name] = importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  }
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: parameterLength
  });
  const output = (id: string, format: 'float32x2' | 'uint32', length: number) => {
    const buffer = createOutputBuffer(device, format === 'float32x2' ? 2 * length : length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, id, buffer, format, length)};
  };
  return {
    graph,
    views,
    buffers,
    parameterBuffer,
    output,
    parameters: parameterBuffer.importToGraph(graph),
    destroy(compiled?: {destroy(): void}) {
      compiled?.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectPolygonsClose(
  actualRows: number[][],
  expected: Polygon[],
  tolerance: number,
  label: string
) {
  const flat = expected.flat().flat();
  expect(actualRows.length, label).toBe(flat.length);
  let row = 0;
  for (const polygon of expected) {
    for (const ring of polygon) {
      for (const [x, y] of ring) {
        expect(Math.abs(actualRows[row][0] - x), `${label} x ${row}`).toBeLessThanOrEqual(
          tolerance
        );
        expect(Math.abs(actualRows[row][1] - y), `${label} y ${row}`).toBeLessThanOrEqual(
          tolerance
        );
        row++;
      }
    }
  }
}

it('GPUGeometryOrientation reverses rings and orients polygons like shapely', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) {
    return;
  }
  const geometry = flattenPolygons(GEOMETRY_EDIT_FIXTURES.orientationInput);
  const vertexCount = geometry.positions.length / 2;
  const ringCount = geometry.ringOffsets.length - 1;
  const run = createRun(device, geometry, 4);
  const {graph, views} = run;
  const reversedOutput = run.output('out-reversed', 'float32x2', vertexCount);
  const orientedOutput = run.output('out-oriented', 'float32x2', vertexCount);
  const flags = run.output('out-flags', 'uint32', ringCount);
  graph.add(
    new GPUGeometryOrientation({
      id: 'reverse',
      positions: views['positions'] as GraphDataView<'float32x2'>,
      ringOffsets: views['ringOffsets'] as GraphDataView<'uint32'>,
      mode: 'reverse',
      output: {positions: reversedOutput.view as GraphDataView<'float32x2'>}
    })
  );
  graph.add(
    new GPUGeometryOrientation({
      id: 'orient',
      positions: views['positions'] as GraphDataView<'float32x2'>,
      ringOffsets: views['ringOffsets'] as GraphDataView<'uint32'>,
      polygonOffsets: views['polygonOffsets'] as GraphDataView<'uint32'>,
      mode: 'orient-polygons',
      parameters: run.parameters,
      output: {
        positions: orientedOutput.view as GraphDataView<'float32x2'>,
        reversedRings: flags.view as GraphDataView<'uint32'>
      }
    })
  );
  const compiled = graph.compile();
  const orientationRuns = [
    {exteriorClockwise: false, expected: GEOMETRY_EDIT_FIXTURES.orientedCCW},
    {exteriorClockwise: true, expected: GEOMETRY_EDIT_FIXTURES.orientedCW}
  ];
  for (const {exteriorClockwise, expected} of orientationRuns) {
    run.parameterBuffer.write(getGPUGeometryOrientationParameterValues({exteriorClockwise}));
    submitGraph(device, compiled, undefined);
    const reversed = toRows(await readFloat32(reversedOutput.buffer, 2 * vertexCount), vertexCount);
    expectPolygonsClose(reversed, GEOMETRY_EDIT_FIXTURES.reversed, 0, 'reverse');
    const oriented = toRows(await readFloat32(orientedOutput.buffer, 2 * vertexCount), vertexCount);
    expectPolygonsClose(oriented, expected, 0, `orient cw=${exteriorClockwise}`);
    const reversedRings = await readUint32(flags.buffer, ringCount);
    expect(reversedRings.some(flag => flag === 1)).toBe(true);
    expect(reversedRings.some(flag => flag === 0)).toBe(true);
  }
  run.destroy(compiled);
});

it('GPUAffineTransform matches shapely for fixed, center and centroid origins and per-feature matrices', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) {
    return;
  }
  const fixtures = GEOMETRY_EDIT_FIXTURES;
  const geometry = flattenPolygons(fixtures.affineInput);
  const vertexCount = geometry.positions.length / 2;
  const matrices = new Float32Array(fixtures.affineMatrices.flat());
  const run = createRun(device, {...geometry, matrices}, 12);
  const {graph, views} = run;
  const globalOutput = run.output('out-global', 'float32x2', vertexCount);
  const featureOutput = run.output('out-feature', 'float32x2', vertexCount);
  graph.add(
    new GPUAffineTransform({
      id: 'global',
      positions: views['positions'] as GraphDataView<'float32x2'>,
      ringOffsets: views['ringOffsets'] as GraphDataView<'uint32'>,
      featureRingOffsets: views['polygonOffsets'] as GraphDataView<'uint32'>,
      geometryType: 'polygons',
      holeRule: 'first-ring-exterior',
      origins: ['center', 'centroid'],
      parameters: run.parameters,
      output: {positions: globalOutput.view as GraphDataView<'float32x2'>}
    })
  );
  const matrixBuffer = createInputBuffer(device, matrices);
  run.buffers.push(matrixBuffer);
  graph.add(
    new GPUAffineTransform({
      id: 'per-feature',
      positions: views['positions'] as GraphDataView<'float32x2'>,
      ringOffsets: views['ringOffsets'] as GraphDataView<'uint32'>,
      featureRingOffsets: views['polygonOffsets'] as GraphDataView<'uint32'>,
      featureTransforms: importGraphBuffer(
        graph,
        'feature-matrices',
        matrixBuffer,
        'float32',
        matrices.length
      ),
      output: {positions: featureOutput.view as GraphDataView<'float32x2'>}
    })
  );
  const compiled = graph.compile();
  const parameters = fixtures.affineParameters;
  const runs = [
    {
      name: 'fixed origin',
      values: getGPUAffineTransformParameters({
        rotate: parameters.rotate,
        scale: parameters.scale as [number, number],
        skew: parameters.skew as [number, number],
        translate: parameters.translate as [number, number],
        origin: fixtures.affineFixedOrigin as [number, number]
      }),
      expected: fixtures.affineFixed
    },
    {
      name: 'center origin',
      values: getGPUAffineTransformParameters({
        rotate: parameters.rotate,
        scale: parameters.scale as [number, number],
        skew: parameters.skew as [number, number],
        translate: parameters.translate as [number, number],
        origin: 'center'
      }),
      expected: fixtures.affineCenter
    },
    {
      name: 'centroid rotate',
      values: getGPUAffineTransformParameters({rotate: 77, origin: 'centroid'}),
      expected: fixtures.affineCentroidRotate
    }
  ];
  // Coordinates near 1e5 have an f32 spacing of 0.0078; allow a few ulps.
  const tolerance = 0.03;
  for (const {name, values, expected} of runs) {
    run.parameterBuffer.write(values);
    submitGraph(device, compiled, undefined);
    const actual = toRows(await readFloat32(globalOutput.buffer, 2 * vertexCount), vertexCount);
    expectPolygonsClose(actual, expected, tolerance, name);
  }
  const perFeature = toRows(await readFloat32(featureOutput.buffer, 2 * vertexCount), vertexCount);
  expectPolygonsClose(perFeature, fixtures.affinePerFeature, 0.05, 'per-feature matrices');
  run.destroy(compiled);
});

it('GPUGeometryCleanup snaps and removes repeated points like shapely, per-frame parameters', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) {
    return;
  }
  for (const geometryType of ['lines', 'polygons'] as const) {
    const paths: Ring[] =
      geometryType === 'lines'
        ? GEOMETRY_EDIT_FIXTURES.cleanupLines
        : GEOMETRY_EDIT_FIXTURES.cleanupRings;
    const geometry = flattenRings(paths);
    const vertexCount = geometry.positions.length / 2;
    const ringCount = geometry.ringOffsets.length - 1;
    const run = createRun(
      device,
      {positions: geometry.positions, ringOffsets: geometry.ringOffsets},
      4
    );
    const positionsOutput = run.output('out-positions', 'float32x2', vertexCount);
    const offsetsOutput = run.output('out-offsets', 'uint32', ringCount + 1);
    const count = run.output('out-count', 'uint32', 1);
    const overflow = run.output('out-overflow', 'uint32', 1);
    const total = run.output('out-total', 'uint32', 1);
    const collapsed = run.output('out-collapsed', 'uint32', 1);
    run.graph.add(
      new GPUGeometryCleanup({
        positions: run.views['positions'] as GraphDataView<'float32x2'>,
        ringOffsets: run.views['ringOffsets'] as GraphDataView<'uint32'>,
        geometryType,
        parameters: run.parameters,
        output: {
          positions: positionsOutput.view as GraphDataView<'float32x2'>,
          ringOffsets: offsetsOutput.view as GraphDataView<'uint32'>,
          count: count.view as GraphDataView<'uint32'>,
          overflow: overflow.view as GraphDataView<'uint32'>,
          requiredCount: total.view as GraphDataView<'uint32'>,
          collapsedRings: collapsed.view as GraphDataView<'uint32'>
        }
      })
    );
    const compiled = run.graph.compile();
    const started = performance.now();
    for (const testCase of GEOMETRY_EDIT_FIXTURES.cleanupCases) {
      const label = `${geometryType} grid=${testCase.grid} tol=${testCase.tol} remove=${testCase.remove}`;
      run.parameterBuffer.write(
        getGPUGeometryCleanupParameterValues({
          gridSize: testCase.grid,
          tolerance: testCase.tol,
          removeRepeatedPoints: testCase.remove
        })
      );
      submitGraph(device, compiled, undefined);
      const expectedPaths: Ring[] = (
        geometryType === 'lines' ? testCase.lines : testCase.rings
      ).map(path => (path === 'collapsed' ? [] : path));
      const expectedCollapsed =
        geometryType === 'lines' ? 0 : testCase.rings.filter(r => r === 'collapsed').length;
      const expected = flattenRings(expectedPaths);
      const expectedCount = expected.positions.length / 2;
      expect(await readUint32(count.buffer, 1), label).toEqual([expectedCount]);
      expect(await readUint32(total.buffer, 1), label).toEqual([expectedCount]);
      expect(await readUint32(overflow.buffer, 1), label).toEqual([0]);
      expect(await readUint32(collapsed.buffer, 1), label).toEqual([expectedCollapsed]);
      expect(await readUint32(offsetsOutput.buffer, ringCount + 1), label).toEqual(
        Array.from(expected.ringOffsets)
      );
      const actual = await readFloat32(positionsOutput.buffer, 2 * expectedCount);
      // `+ 0` folds -0 (GEOS rounds small negatives to -0) into 0.
      expect(
        actual.map(value => value + 0),
        label
      ).toEqual(Array.from(expected.positions, value => value + 0));
    }
    console.log(
      `GPUGeometryCleanup ${geometryType}: ${GEOMETRY_EDIT_FIXTURES.cleanupCases.length} parameter sets in ${(performance.now() - started).toFixed(1)} ms`
    );
    run.destroy(compiled);
  }
});

it('GPUGeometryCleanup snaps Point features without deduplicating rows', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) {
    return;
  }
  const positions = new Float32Array([
    0.25, 0.25, 0.25, 0.25, -0.25, -0.75, 1.24, 1.26, -1.24, -1.26
  ]);
  const pointCount = positions.length / 2;
  const run = createRun(device, {positions}, 4);
  const positionsOutput = run.output('point-positions', 'float32x2', pointCount);
  const count = run.output('point-count', 'uint32', 1);
  const overflow = run.output('point-overflow', 'uint32', 1);
  run.graph.add(
    new GPUGeometryCleanup({
      positions: run.views['positions'] as GraphDataView<'float32x2'>,
      geometryType: 'points',
      parameters: run.parameters,
      output: {
        positions: positionsOutput.view as GraphDataView<'float32x2'>,
        count: count.view as GraphDataView<'uint32'>,
        overflow: overflow.view as GraphDataView<'uint32'>
      }
    })
  );
  const compiled = run.graph.compile();

  run.parameterBuffer.write(
    getGPUGeometryCleanupParameterValues({
      gridSize: 0.5,
      tolerance: 100,
      removeRepeatedPoints: true
    })
  );
  submitGraph(device, compiled, undefined);
  expect(await readFloat32(positionsOutput.buffer, positions.length)).toEqual([
    0.5, 0.5, 0.5, 0.5, 0, -0.5, 1, 1.5, -1, -1.5
  ]);
  expect(await readUint32(count.buffer, 1)).toEqual([pointCount]);
  expect(await readUint32(overflow.buffer, 1)).toEqual([0]);

  run.parameterBuffer.write(
    getGPUGeometryCleanupParameterValues({tolerance: 100, removeRepeatedPoints: true})
  );
  submitGraph(device, compiled, undefined);
  expect(await readFloat32(positionsOutput.buffer, positions.length)).toEqual(
    Array.from(positions)
  );
  expect(await readUint32(count.buffer, 1)).toEqual([pointCount]);
  expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
  run.destroy(compiled);
});

it('GPUGeometryCleanup reports Point capacity overflow', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) {
    return;
  }
  const positions = new Float32Array([0.1, 0.2, 1.1, 1.2, 2.1, 2.2, 3.1, 3.2]);
  const run = createRun(device, {positions}, 4);
  const capacity = 3;
  const positionsOutput = run.output('point-overflow-positions', 'float32x2', capacity);
  const count = run.output('point-overflow-count', 'uint32', 1);
  const overflow = run.output('point-overflow-flag', 'uint32', 1);
  const total = run.output('point-overflow-total', 'uint32', 1);
  run.graph.add(
    new GPUGeometryCleanup({
      positions: run.views['positions'] as GraphDataView<'float32x2'>,
      geometryType: 'points',
      parameters: run.parameters,
      output: {
        positions: positionsOutput.view as GraphDataView<'float32x2'>,
        count: count.view as GraphDataView<'uint32'>,
        overflow: overflow.view as GraphDataView<'uint32'>,
        requiredCount: total.view as GraphDataView<'uint32'>
      }
    })
  );
  const compiled = run.graph.compile();
  run.parameterBuffer.write(getGPUGeometryCleanupParameterValues({}));
  submitGraph(device, compiled, undefined);
  expect(await readFloat32(positionsOutput.buffer, 2 * capacity)).toEqual(
    Array.from(positions.slice(0, 2 * capacity))
  );
  expect(await readUint32(count.buffer, 1)).toEqual([capacity]);
  expect(await readUint32(total.buffer, 1)).toEqual([positions.length / 2]);
  expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
  run.destroy(compiled);
});

it('GPUGeometryCleanup reports overflow with clamped offsets', async () => {
  const device = await getWebGPUTestDevice('core');
  if (!device) {
    return;
  }
  const paths: Ring[] = [
    [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0]
    ],
    [
      [5, 5],
      [6, 5],
      [7, 5]
    ]
  ];
  const geometry = flattenRings(paths);
  const run = createRun(
    device,
    {positions: geometry.positions, ringOffsets: geometry.ringOffsets},
    4
  );
  const capacity = 5;
  const positionsOutput = run.output('out-positions', 'float32x2', capacity);
  const offsetsOutput = run.output('out-offsets', 'uint32', 3);
  const count = run.output('out-count', 'uint32', 1);
  const overflow = run.output('out-overflow', 'uint32', 1);
  const total = run.output('out-total', 'uint32', 1);
  run.graph.add(
    new GPUGeometryCleanup({
      positions: run.views['positions'] as GraphDataView<'float32x2'>,
      ringOffsets: run.views['ringOffsets'] as GraphDataView<'uint32'>,
      geometryType: 'lines',
      parameters: run.parameters,
      output: {
        positions: positionsOutput.view as GraphDataView<'float32x2'>,
        ringOffsets: offsetsOutput.view as GraphDataView<'uint32'>,
        count: count.view as GraphDataView<'uint32'>,
        overflow: overflow.view as GraphDataView<'uint32'>,
        requiredCount: total.view as GraphDataView<'uint32'>
      }
    })
  );
  const compiled = run.graph.compile();
  run.parameterBuffer.write(getGPUGeometryCleanupParameterValues({}));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(count.buffer, 1)).toEqual([capacity]);
  expect(await readUint32(total.buffer, 1)).toEqual([7]);
  expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
  expect(await readUint32(offsetsOutput.buffer, 3)).toEqual([0, 4, 5]);
  run.destroy(compiled);
});
