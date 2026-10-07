// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPULinearReferencing,
  GPULineLocate
} from '../../../src/gpu-spatial-analysis/linear-referencing';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  PLANAR_INTERPOLATED_EXPECTED,
  PLANAR_INTERPOLATED_FRACTIONS,
  PLANAR_NORMALIZED_EXPECTED,
  PLANAR_NORMALIZED_QUERIES,
  SPHERICAL_EVENTS,
  SPHERICAL_PATH_LENGTHS,
  SPHERICAL_PATH_OFFSETS,
  SPHERICAL_POINTS,
  SPHERICAL_POSITIONS,
  SPHERICAL_PROJECTIONS
} from './spherical-oracle-values';

const NO_PATH = 0xffffffff;
/** f32 longitude/latitude inputs plus f32 trigonometry: meters of slack on every length. */
const METER_TOLERANCE = 3;

function createGraphHelpers(device: Device, id: string) {
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id});
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  return {
    graph,
    track,
    input<Format extends 'float32x2' | 'uint32' | 'float32'>(
      name: string,
      values: Float32Array | Uint32Array,
      format: Format
    ): GraphDataView<Format> {
      return importGraphBuffer(
        graph,
        name,
        track(createInputBuffer(device, values)),
        format,
        format === 'float32x2' ? values.length / 2 : values.length
      );
    },
    output<Format extends 'float32x2' | 'uint32' | 'float32' | 'sint32'>(
      name: string,
      format: Format,
      length: number
    ): {view: GraphDataView<Format>; buffer: Buffer} {
      const buffer = track(
        createOutputBuffer(device, format === 'float32x2' ? 2 * length : length)
      );
      return {view: importGraphBuffer(graph, name, buffer, format, length), buffer};
    },
    destroy() {
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

async function runProjection(
  device: Device,
  coordinateSystem: 'planar' | 'spherical',
  positions: Float32Array,
  pathOffsets: Uint32Array,
  points: Float32Array,
  searchRadius: number
) {
  const helpers = createGraphHelpers(device, `linear-referencing-${coordinateSystem}`);
  const pointCount = points.length / 2;
  const radius = new GPUParameterBuffer(device, {id: 'radius', format: 'float32', length: 1});
  const columns = {
    pathIndices: helpers.output('o-path', 'uint32', pointCount),
    segmentIndices: helpers.output('o-segment', 'uint32', pointCount),
    fractions: helpers.output('o-fraction', 'float32', pointCount),
    footPoints: helpers.output('o-foot', 'float32x2', pointCount),
    distances: helpers.output('o-distance', 'float32', pointCount),
    measures: helpers.output('o-measure', 'float32', pointCount),
    normalizedMeasures: helpers.output('o-normalized', 'float32', pointCount),
    sides: helpers.output('o-side', 'sint32', pointCount)
  };
  const overflow = helpers.output('o-overflow', 'uint32', 1);
  helpers.graph.add(
    new GPULinearReferencing({
      coordinateSystem,
      points: helpers.input('points', points, 'float32x2'),
      positions: helpers.input('positions', positions, 'float32x2'),
      pathOffsets: helpers.input('path-offsets', pathOffsets, 'uint32'),
      radius: radius.importToGraph(helpers.graph),
      candidateCapacity: coordinateSystem === 'planar' ? 4096 : undefined,
      output: Object.fromEntries(
        Object.entries(columns).map(([name, column]) => [name, column.view])
      ),
      overflow: overflow.view
    })
  );
  const compiled = helpers.graph.compile();
  radius.write(new Float32Array([searchRadius]));
  submitGraph(device, compiled, undefined);
  const sideBytes = await columns.sides.buffer.readAsync();
  const foot = await readFloat32(columns.footPoints.buffer, 2 * pointCount);
  const result = {
    pathIndices: await readUint32(columns.pathIndices.buffer, pointCount),
    segmentIndices: await readUint32(columns.segmentIndices.buffer, pointCount),
    fractions: await readFloat32(columns.fractions.buffer, pointCount),
    foot,
    distances: await readFloat32(columns.distances.buffer, pointCount),
    measures: await readFloat32(columns.measures.buffer, pointCount),
    normalizedMeasures: await readFloat32(columns.normalizedMeasures.buffer, pointCount),
    sides: Array.from(new Int32Array(sideBytes.buffer, sideBytes.byteOffset, pointCount)),
    overflow: (await readUint32(overflow.buffer, 1))[0]
  };
  compiled.destroy();
  radius.destroy();
  helpers.destroy();
  return result;
}

it('GPULinearReferencing spherical matches pyproj Geod closest points on great-circle arcs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = new Float32Array(SPHERICAL_POSITIONS);
  const pathOffsets = new Uint32Array(SPHERICAL_PATH_OFFSETS);
  const points = new Float32Array(SPHERICAL_POINTS);
  const started = performance.now();
  const actual = await runProjection(device, 'spherical', positions, pathOffsets, points, 1e9);
  const elapsed = performance.now() - started;
  expect(actual.overflow).toBe(0);
  let worst = {distance: 0, measure: 0, foot: 0, normalized: 0};
  SPHERICAL_PROJECTIONS.forEach((expected, row) => {
    const label = `point ${row}`;
    expect(actual.pathIndices[row], label).toBe(expected.path);
    const distanceError = Math.abs(actual.distances[row] - expected.distance);
    const measureError = Math.abs(actual.measures[row] - expected.measure);
    const footError =
      Math.hypot(
        (((actual.foot[2 * row] - expected.foot[0] + 540) % 360) - 180) *
          Math.cos((expected.foot[1] * Math.PI) / 180),
        actual.foot[2 * row + 1] - expected.foot[1]
      ) * 111194.9;
    const length = SPHERICAL_PATH_LENGTHS[expected.path];
    const normalizedError = Math.abs(actual.normalizedMeasures[row] - expected.normalized);
    worst = {
      distance: Math.max(worst.distance, distanceError),
      measure: Math.max(worst.measure, measureError),
      foot: Math.max(worst.foot, footError),
      normalized: Math.max(worst.normalized, normalizedError * length)
    };
    expect(distanceError, `${label} distance`).toBeLessThanOrEqual(METER_TOLERANCE);
    expect(measureError, `${label} measure`).toBeLessThanOrEqual(METER_TOLERANCE);
    expect(footError, `${label} foot`).toBeLessThanOrEqual(METER_TOLERANCE);
    expect(normalizedError * length, `${label} normalized`).toBeLessThanOrEqual(METER_TOLERANCE);
    // Vertex ties may legitimately pick either neighbouring segment.
    if (actual.segmentIndices[row] === expected.segment && expected.distance > 50) {
      expect(actual.sides[row], `${label} side`).toBe(expected.side);
    }
  });
  console.log(
    `spherical projection ${SPHERICAL_PROJECTIONS.length} points, worst error (m): ` +
      `distance ${worst.distance.toFixed(2)}, measure ${worst.measure.toFixed(2)}, ` +
      `foot ${worst.foot.toFixed(2)}, normalized ${worst.normalized.toFixed(2)}; ${elapsed.toFixed(0)} ms`
  );
  // A per-frame radius in meters drops the far point (last row) but keeps near ones.
  const nearRadius = await runProjection(device, 'spherical', positions, pathOffsets, points, 5e5);
  const last = SPHERICAL_PROJECTIONS.length - 1;
  expect(SPHERICAL_PROJECTIONS[last].distance).toBeGreaterThan(5e5);
  expect(nearRadius.pathIndices[last]).toBe(NO_PATH);
  expect(nearRadius.distances[last]).toBe(-1);
  expect(nearRadius.sides[last]).toBe(0);
  expect(nearRadius.normalizedMeasures[last]).toBeNaN();
  expect(nearRadius.pathIndices[0]).toBe(SPHERICAL_PROJECTIONS[0].path);
  const nothing = await runProjection(
    device,
    'spherical',
    positions,
    pathOffsets,
    points,
    Number.NaN
  );
  expect(nothing.pathIndices.every(path => path === NO_PATH)).toBe(true);
});

it('GPULinearReferencing planar normalizedMeasures matches shapely line_locate_point(normalized=True)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const actual = await runProjection(
    device,
    'planar',
    new Float32Array([0, 0, 10, 0, 10, 10, 20, 10]),
    new Uint32Array([0, 4]),
    new Float32Array(PLANAR_NORMALIZED_QUERIES.flat()),
    1000
  );
  PLANAR_NORMALIZED_EXPECTED.forEach((expected, row) => {
    expect(actual.normalizedMeasures[row], `point ${row}`).toBeCloseTo(expected, 6);
  });
});

async function runLocate(
  device: Device,
  coordinateSystem: 'planar' | 'spherical',
  measureMode: 'distance' | 'fraction',
  positions: Float32Array,
  pathOffsets: Uint32Array,
  eventPaths: number[],
  eventMeasures: number[],
  eventOffsets?: number[]
) {
  const helpers = createGraphHelpers(device, `line-locate-${coordinateSystem}`);
  const eventCount = eventPaths.length;
  const output = {
    positions: helpers.output('o-positions', 'float32x2', eventCount),
    segmentIndices: helpers.output('o-segments', 'uint32', eventCount),
    tangents: helpers.output('o-tangents', 'float32x2', eventCount),
    angles: helpers.output('o-angles', 'float32', eventCount)
  };
  helpers.graph.add(
    new GPULineLocate({
      coordinateSystem,
      measureMode,
      positions: helpers.input('positions', positions, 'float32x2'),
      pathOffsets: helpers.input('path-offsets', pathOffsets, 'uint32'),
      eventPaths: helpers.input('event-paths', new Uint32Array(eventPaths), 'uint32'),
      eventMeasures: helpers.input('event-measures', new Float32Array(eventMeasures), 'float32'),
      eventOffsets: eventOffsets
        ? helpers.input('event-offsets', new Float32Array(eventOffsets), 'float32')
        : undefined,
      output: Object.fromEntries(
        Object.entries(output).map(([name, column]) => [name, column.view])
      ) as {
        positions: GraphDataView<'float32x2'>;
      }
    })
  );
  const compiled = helpers.graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    positions: await readFloat32(output.positions.buffer, 2 * eventCount),
    tangents: await readFloat32(output.tangents.buffer, 2 * eventCount),
    angles: await readFloat32(output.angles.buffer, eventCount)
  };
  compiled.destroy();
  helpers.destroy();
  return result;
}

it('GPULineLocate spherical interpolates by meters and by fraction like pyproj Geod.fwd', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = new Float32Array(SPHERICAL_POSITIONS);
  const pathOffsets = new Uint32Array(SPHERICAL_PATH_OFFSETS);
  const eventPaths = SPHERICAL_EVENTS.map(event => event.path);
  const offsets = SPHERICAL_EVENTS.map(event => event.offset);
  for (const mode of ['fraction', 'distance'] as const) {
    const measures = SPHERICAL_EVENTS.map(event =>
      mode === 'fraction' ? event.fraction : event.meters
    );
    const actual = await runLocate(
      device,
      'spherical',
      mode,
      positions,
      pathOffsets,
      eventPaths,
      measures,
      offsets
    );
    let worst = 0;
    SPHERICAL_EVENTS.forEach((expected, row) => {
      const longitudeDelta = ((actual.positions[2 * row] - expected.position[0] + 540) % 360) - 180;
      const error =
        Math.hypot(
          longitudeDelta * Math.cos((expected.position[1] * Math.PI) / 180),
          actual.positions[2 * row + 1] - expected.position[1]
        ) * 111194.9;
      worst = Math.max(worst, error);
      // 1 m of f32 slack per 1e6 m of measure: the f32 measure itself rounds at that scale.
      const slack = METER_TOLERANCE + expected.meters * 1e-6 * 2;
      expect(error, `${mode} event ${row}`).toBeLessThanOrEqual(slack);
      if (expected.offset === 0) {
        // Compass azimuth of the arc: tangent is (east, north).
        const azimuth = (expected.azimuth * Math.PI) / 180;
        const tangentError = Math.hypot(
          actual.tangents[2 * row] - Math.sin(azimuth),
          actual.tangents[2 * row + 1] - Math.cos(azimuth)
        );
        expect(tangentError, `${mode} event ${row} tangent`).toBeLessThan(2e-3);
      }
    });
    console.log(`spherical locate (${mode}) worst error: ${worst.toFixed(2)} m`);
  }
});

it('GPULineLocate planar fraction matches shapely line_interpolate_point(normalized=True)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const actual = await runLocate(
    device,
    'planar',
    'fraction',
    new Float32Array([0, 0, 10, 0, 10, 10, 20, 10]),
    new Uint32Array([0, 4]),
    PLANAR_INTERPOLATED_FRACTIONS.map(() => 0),
    PLANAR_INTERPOLATED_FRACTIONS
  );
  PLANAR_INTERPOLATED_EXPECTED.forEach(([x, y], row) => {
    expect(actual.positions[2 * row], `x ${row}`).toBeCloseTo(x, 4);
    expect(actual.positions[2 * row + 1], `y ${row}`).toBeCloseTo(y, 4);
  });
});

/** f64 distance in meters from a point to the great-circle arc `a..b` (endpoints included). */
function getArcDistanceOracle(
  point: readonly number[],
  a: readonly number[],
  b: readonly number[]
): number {
  const toVector = ([longitude, latitude]: readonly number[]) => {
    const lambda = (longitude * Math.PI) / 180;
    const phi = (latitude * Math.PI) / 180;
    return [Math.cos(phi) * Math.cos(lambda), Math.cos(phi) * Math.sin(lambda), Math.sin(phi)];
  };
  const dot = (u: number[], v: number[]) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const cross = (u: number[], v: number[]) => [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0]
  ];
  const angle = (u: number[], v: number[]) => Math.atan2(Math.hypot(...cross(u, v)), dot(u, v));
  const p = toVector(point);
  const start = toVector(a);
  const end = toVector(b);
  let best = Math.min(angle(p, start), angle(p, end));
  const normal = cross(start, end);
  const normalLength = Math.hypot(...normal);
  if (normalLength > 1e-12) {
    // The point's foot on the great circle lies on the arc when it is between both endpoints.
    const unit = normal.map(value => value / normalLength);
    const foot = cross(unit, cross(p, unit));
    const arcAngle = angle(start, end);
    if (angle(start, foot) <= arcAngle && angle(foot, end) <= arcAngle) {
      best = Math.min(best, Math.abs(Math.asin(Math.max(-1, Math.min(1, dot(p, unit))))));
    }
  }
  return best * 6371008.8;
}

it('GPULinearReferencing spherical prunes by bounding caps without changing nearest segments', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Many short and a few very long arcs spread over the globe, including the antimeridian, the
  // high latitudes and a zero-length segment, so cap pruning meets large and tiny half-angles.
  let state = 12345;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const pathOffsets = [0];
  const vertices: number[][] = [];
  for (let path = 0; path < 12; path++) {
    let longitude = random() * 360 - 180;
    let latitude = random() * 150 - 75;
    const count = 4 + Math.floor(random() * 60);
    const step = path % 4 === 0 ? 25 : 0.4;
    for (let vertex = 0; vertex < count; vertex++) {
      vertices.push([longitude, latitude]);
      if (vertex === 2) {
        vertices.push([longitude, latitude]);
      }
      longitude = ((longitude + (random() - 0.4) * step + 540) % 360) - 180;
      latitude = Math.max(-80, Math.min(80, latitude + (random() - 0.5) * step));
    }
    pathOffsets.push(vertices.length);
  }
  const points: number[][] = [];
  for (let index = 0; index < 150; index++) {
    const near = vertices[Math.floor(random() * vertices.length)];
    points.push(
      index % 3 === 0
        ? [random() * 360 - 180, random() * 160 - 80]
        : [
            ((near[0] + (random() - 0.5) * 2 + 540) % 360) - 180,
            Math.max(-89, Math.min(89, near[1] + (random() - 0.5) * 2))
          ]
    );
  }
  const actual = await runProjection(
    device,
    'spherical',
    new Float32Array(vertices.flat()),
    new Uint32Array(pathOffsets),
    new Float32Array(points.flat()),
    1e9
  );
  const rowCount = vertices.length;
  let ambiguous = 0;
  points.forEach((point, index) => {
    const distances: {row: number; distance: number}[] = [];
    for (let path = 0; path + 1 < pathOffsets.length; path++) {
      for (let row = pathOffsets[path]; row + 1 < pathOffsets[path + 1]; row++) {
        distances.push({
          row,
          distance: getArcDistanceOracle(point, vertices[row], vertices[row + 1])
        });
      }
    }
    distances.sort((left, right) => left.distance - right.distance);
    expect(rowCount).toBeGreaterThan(100);
    expect(
      Math.abs(actual.distances[index] - distances[0].distance),
      `point ${index}`
    ).toBeLessThanOrEqual(METER_TOLERANCE * 4);
    if (distances[1].distance - distances[0].distance > 15) {
      const path = pathOffsets.findIndex((offset, p) => distances[0].row < pathOffsets[p + 1]);
      expect(actual.pathIndices[index], `point ${index} path`).toBe(path);
      expect(actual.segmentIndices[index], `point ${index} segment`).toBe(
        distances[0].row - pathOffsets[path]
      );
    } else {
      ambiguous++;
    }
  });
  expect(ambiguous).toBeLessThan(points.length * 0.8);
  device.destroy?.();
});
