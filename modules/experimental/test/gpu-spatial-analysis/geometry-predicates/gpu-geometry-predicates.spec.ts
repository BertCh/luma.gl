// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUGeometryPredicatesParameterValues,
  GPUGeometryPredicates
} from '../../../src/gpu-spatial-analysis/geometry-predicates/index';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createPredicateGeometry,
  type PredicateGeometrySpec
} from './gpu-geometry-predicates-harness';
import {SHAPELY_PREDICATES_FIXTURE as FIXTURE} from './shapely-predicates-fixture';

const COLUMNS = [
  'isSimple',
  'isRing',
  'isClosed',
  'isCcw',
  'numPoints',
  'numCoordinates',
  'numInteriorRings',
  'numGeometries'
] as const;

type Expected = Record<(typeof COLUMNS)[number], number[]>;

async function runColumns(
  device: Device,
  spec: PredicateGeometrySpec,
  featureCount: number,
  vertexCount: number
): Promise<{
  columns: Record<string, number[]>;
  unique: number[][][];
  overflow: number;
}> {
  const graph = new GPUCommandGraph(device, {id: 'geometry-predicates'});
  const buffers: Buffer[] = [];
  const geometry = createPredicateGeometry(device, graph, 'geometry', spec, buffers);
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const columns = Object.fromEntries(COLUMNS.map(name => [name, output(name, featureCount)]));
  const overflow = output('overflow', 1);
  const uniqueOffsets = output('unique-offsets', featureCount + 1);
  const uniqueBuffer = createOutputBuffer(device, vertexCount * 2);
  buffers.push(uniqueBuffer);
  const uniquePositions = importGraphBuffer(
    graph,
    'unique-positions',
    uniqueBuffer,
    'float32x2',
    vertexCount
  );
  graph.add(
    new GPUGeometryPredicates({
      geometry,
      isSimple: columns.isSimple.view,
      isRing: columns.isRing.view,
      isClosed: columns.isClosed.view,
      isCcw: columns.isCcw.view,
      numPoints: columns.numPoints.view,
      numCoordinates: columns.numCoordinates.view,
      numInteriorRings: columns.numInteriorRings.view,
      numGeometries: columns.numGeometries.view,
      intersectionCapacity: 4096,
      overflow: overflow.view,
      uniquePositions,
      uniqueOffsets: uniqueOffsets.view
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: Record<string, number[]> = {};
  for (const name of COLUMNS) {
    result[name] = await readUint32(columns[name].buffer, featureCount);
  }
  const offsets = await readUint32(uniqueOffsets.buffer, featureCount + 1);
  const flat = await readFloat32(uniqueBuffer, vertexCount * 2);
  const unique: number[][][] = [];
  for (let feature = 0; feature < featureCount; feature++) {
    const points: number[][] = [];
    for (let row = offsets[feature]; row < offsets[feature + 1]; row++) {
      points.push([flat[row * 2], flat[row * 2 + 1]]);
    }
    unique.push(points);
  }
  const overflowFlag = (await readUint32(overflow.buffer, 1))[0];
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return {columns: result, unique, overflow: overflowFlag};
}

function expectColumns(
  names: readonly string[],
  actual: Record<string, number[]>,
  expected: Expected
): void {
  for (const column of COLUMNS) {
    const mismatches: string[] = [];
    names.forEach((name, row) => {
      if (actual[column][row] !== expected[column][row]) {
        mismatches.push(`${name}: got ${actual[column][row]}, Shapely ${expected[column][row]}`);
      }
    });
    expect(mismatches, `${column} mismatches`).toEqual([]);
  }
}

it('GPUGeometryPredicates linestring columns and unique points match Shapely', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {lines} = FIXTURE;
  const vertexCount = lines.vertices.reduce((sum, vertices) => sum + vertices.length, 0);
  const result = await runColumns(
    device,
    {kind: 'lines', vertices: lines.vertices},
    lines.names.length,
    vertexCount
  );
  expect(result.overflow).toBe(0);
  expectColumns(lines.names, result.columns, lines.expected);
  expect(result.unique).toEqual(lines.unique);
  expect(lines.expected.isRing.some(Boolean) && lines.expected.isCcw.some(Boolean)).toBe(true);
});

it('GPUGeometryPredicates polygon columns and unique points match Shapely', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {polygons} = FIXTURE;
  const vertexCount = polygons.polygons.reduce(
    (sum, feature) =>
      sum +
      feature.reduce(
        (inner, polygon) => inner + polygon.reduce((ring, vertices) => ring + vertices.length, 0),
        0
      ),
    0
  );
  const result = await runColumns(
    device,
    {kind: 'polygons', polygons: polygons.polygons},
    polygons.names.length,
    vertexCount
  );
  expect(result.overflow).toBe(0);
  // numPoints is zero for polygons in Shapely; numGeometries of an empty polygon is not exercised.
  expectColumns(polygons.names, result.columns, polygons.expected);
  expect(result.unique).toEqual(polygons.unique);
});

it('GPUGeometryPredicates point columns are constants', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {points} = FIXTURE;
  const count = points.names.length;
  const result = await runColumns(
    device,
    {kind: 'points', positions: points.positions},
    count,
    count
  );
  expect(result.columns.isSimple).toEqual(new Array(count).fill(1));
  expect(result.columns.numGeometries).toEqual(new Array(count).fill(1));
  expect(result.columns.numCoordinates).toEqual(new Array(count).fill(1));
  for (const name of ['isRing', 'isClosed', 'isCcw', 'numPoints', 'numInteriorRings']) {
    expect(result.columns[name], name).toEqual(new Array(count).fill(0));
  }
});

/** Python `str(float)` key used by the fixture generator. */
function toleranceKey(tolerance: number): string {
  return Number.isInteger(tolerance) ? tolerance.toFixed(1) : String(tolerance);
}

async function runEquality(
  device: Device,
  firstSpec: PredicateGeometrySpec,
  secondSpec: PredicateGeometrySpec,
  featureCount: number
): Promise<{exact: Record<string, number[]>; identical: number[]}> {
  const graph = new GPUCommandGraph(device, {id: 'geometry-predicates-equality'});
  const buffers: Buffer[] = [];
  const geometry = createPredicateGeometry(device, graph, 'first', firstSpec, buffers);
  const other = createPredicateGeometry(device, graph, 'second', secondSpec, buffers);
  const exact = createOutputBuffer(device, featureCount);
  const identical = createOutputBuffer(device, featureCount);
  buffers.push(exact, identical);
  const parameters = new GPUParameterBuffer(device, {
    id: 'predicates-parameters',
    format: 'float32',
    length: 4
  });
  graph.add(
    new GPUGeometryPredicates({
      geometry,
      other,
      parameters: parameters.importToGraph(graph),
      equalsExact: importGraphBuffer(graph, 'exact', exact, 'uint32', featureCount),
      equalsIdentical: importGraphBuffer(graph, 'identical', identical, 'uint32', featureCount)
    })
  );
  const compiled = graph.compile();
  const exactByTolerance: Record<string, number[]> = {};
  for (const tolerance of FIXTURE.tolerances) {
    parameters.write(getGPUGeometryPredicatesParameterValues({tolerance}));
    submitGraph(device, compiled, undefined);
    exactByTolerance[toleranceKey(tolerance)] = await readUint32(exact, featureCount);
  }
  const identicalResult = await readUint32(identical, featureCount);
  compiled.destroy();
  parameters.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return {exact: exactByTolerance, identical: identicalResult};
}

it('GPUGeometryPredicates equalsExact and equalsIdentical match Shapely on aligned pairs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const lines = FIXTURE.equalLines;
  const lineResult = await runEquality(
    device,
    {kind: 'lines', vertices: lines.pairs.map(pair => pair[0])},
    {kind: 'lines', vertices: lines.pairs.map(pair => pair[1])},
    lines.pairs.length
  );
  expect(lineResult.identical).toEqual(lines.identical);
  for (const tolerance of FIXTURE.tolerances) {
    expect(lineResult.exact[toleranceKey(tolerance)], `lines tolerance ${tolerance}`).toEqual(
      lines.exact[toleranceKey(tolerance)]
    );
  }
  const polygons = FIXTURE.equalPolygons;
  const polygonResult = await runEquality(
    device,
    {kind: 'polygons', polygons: polygons.pairs.map(pair => pair[0])},
    {kind: 'polygons', polygons: polygons.pairs.map(pair => pair[1])},
    polygons.pairs.length
  );
  expect(polygonResult.identical).toEqual(polygons.identical);
  for (const tolerance of FIXTURE.tolerances) {
    expect(polygonResult.exact[toleranceKey(tolerance)], `polygons tolerance ${tolerance}`).toEqual(
      polygons.exact[toleranceKey(tolerance)]
    );
  }
});

it('GPUGeometryPredicates unique points scale past one workgroup and report timing', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Pseudo-random lines on a coarse grid so most vertices repeat; oracle is a first-occurrence Map.
  let seed = 12345;
  const next = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed;
  };
  const vertices: number[][][] = [];
  for (let line = 0; line < 400; line++) {
    const count = 2 + (next() % 150);
    const row: number[][] = [];
    for (let k = 0; k < count; k++) {
      row.push([(next() % 40) - 20, (next() % 40) - 20 + (k % 7 === 0 ? 0.5 : 0)]);
    }
    vertices.push(row);
  }
  const expected = vertices.map(row => {
    const seen = new Set<string>();
    const unique: number[][] = [];
    for (const vertex of row) {
      const key = vertex.join(',');
      if (!seen.has(key)) {
        seen.add(key);
        unique.push(vertex);
      }
    }
    return unique;
  });
  const vertexCount = vertices.reduce((sum, row) => sum + row.length, 0);
  const start = performance.now();
  const result = await runColumns(device, {kind: 'lines', vertices}, vertices.length, vertexCount);
  const elapsed = performance.now() - start;
  console.log(
    `GPUGeometryPredicates ${vertexCount} vertices, all columns + unique: ${elapsed.toFixed(0)} ms`
  );
  expect(result.unique).toEqual(expected);
});
