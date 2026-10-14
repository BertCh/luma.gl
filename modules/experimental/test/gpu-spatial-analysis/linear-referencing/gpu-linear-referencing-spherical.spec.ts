// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  createSphericalBVHScene,
  getArcDistanceOracle,
  getNearestArcCandidates
} from './spherical-arc-bvh-oracle';
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
  SPHERICAL_LINE_LENGTHS,
  SPHERICAL_LINE_OFFSETS,
  SPHERICAL_POINTS,
  SPHERICAL_POSITIONS,
  SPHERICAL_PROJECTIONS
} from './spherical-oracle-values';
import {requireTestDevice} from '../../utils/require-test-device';

const NO_LINE = 0xffffffff;
/**
 * f32 longitude/latitude inputs plus f32 trigonometry: meters of slack on every length. One f32 ulp
 * of a longitude in [128, 180] degrees is 2^-16 degree = 1.7 m at the equator, so rounding the
 * inputs and the output foot point alone costs up to ~1.7 m; the measured worst error is 1.7 m
 * (foot) and 1.5 m (distance), so 3 m is under two ulps and cannot be tightened meaningfully.
 */
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
  lineOffsets: Uint32Array,
  points: Float32Array,
  searchRadius: number
) {
  const helpers = createGraphHelpers(device, `linear-referencing-${coordinateSystem}`);
  const pointCount = points.length / 2;
  const radius = new GPUParameterBuffer(device, {id: 'radius', format: 'float32', length: 1});
  const columns = {
    lineIndices: helpers.output('o-line', 'uint32', pointCount),
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
      spatialContext:
        coordinateSystem === 'spherical'
          ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
          : undefined,
      queryPositions: helpers.input('points', points, 'float32x2'),
      positions: helpers.input('positions', positions, 'float32x2'),
      lineOffsets: helpers.input('line-offsets', lineOffsets, 'uint32'),
      radius: radius.importToGraph(helpers.graph),
      candidateCapacity: coordinateSystem === 'planar' ? 4096 : undefined,
      output: {
        ...Object.fromEntries(Object.entries(columns).map(([name, column]) => [name, column.view])),
        status: {overflow: overflow.view}
      }
    })
  );
  const compiled = helpers.graph.compile();
  radius.write(new Float32Array([searchRadius]));
  submitGraph(device, compiled, undefined);
  const sideBytes = await columns.sides.buffer.readAsync();
  const foot = await readFloat32(columns.footPoints.buffer, 2 * pointCount);
  const result = {
    lineIndices: await readUint32(columns.lineIndices.buffer, pointCount),
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
  requireTestDevice(device);
  const positions = new Float32Array(SPHERICAL_POSITIONS);
  const lineOffsets = new Uint32Array(SPHERICAL_LINE_OFFSETS);
  const points = new Float32Array(SPHERICAL_POINTS);
  const started = performance.now();
  const actual = await runProjection(device, 'spherical', positions, lineOffsets, points, 1e9);
  const elapsed = performance.now() - started;
  expect(actual.overflow).toBe(0);
  let worst = {distance: 0, measure: 0, foot: 0, normalized: 0};
  SPHERICAL_PROJECTIONS.forEach((expected, row) => {
    const label = `point ${row}`;
    expect(actual.lineIndices[row], label).toBe(expected.line);
    const distanceError = Math.abs(actual.distances[row] - expected.distance);
    const measureError = Math.abs(actual.measures[row] - expected.measure);
    const footError =
      Math.hypot(
        (((actual.foot[2 * row] - expected.foot[0] + 540) % 360) - 180) *
          Math.cos((expected.foot[1] * Math.PI) / 180),
        actual.foot[2 * row + 1] - expected.foot[1]
      ) * 111194.9;
    const length = SPHERICAL_LINE_LENGTHS[expected.line];
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
  const nearRadius = await runProjection(device, 'spherical', positions, lineOffsets, points, 5e5);
  const last = SPHERICAL_PROJECTIONS.length - 1;
  expect(SPHERICAL_PROJECTIONS[last].distance).toBeGreaterThan(5e5);
  expect(nearRadius.lineIndices[last]).toBe(NO_LINE);
  expect(nearRadius.distances[last]).toBe(-1);
  expect(nearRadius.sides[last]).toBe(0);
  expect(nearRadius.normalizedMeasures[last]).toBeNaN();
  expect(nearRadius.lineIndices[0]).toBe(SPHERICAL_PROJECTIONS[0].line);
  const nothing = await runProjection(
    device,
    'spherical',
    positions,
    lineOffsets,
    points,
    Number.NaN
  );
  expect(nothing.lineIndices.every(line => line === NO_LINE)).toBe(true);
});

it('GPULinearReferencing planar normalizedMeasures matches shapely line_locate_point(normalized=True)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
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
  lineOffsets: Uint32Array,
  eventLineIndices: number[],
  eventMeasures: number[],
  eventOffsets?: number[]
) {
  const helpers = createGraphHelpers(device, `line-locate-${coordinateSystem}`);
  const eventCount = eventLineIndices.length;
  const output = {
    positions: helpers.output('o-positions', 'float32x2', eventCount),
    segmentIndices: helpers.output('o-segments', 'uint32', eventCount),
    tangents: helpers.output('o-tangents', 'float32x2', eventCount),
    angles: helpers.output('o-angles', 'float32', eventCount)
  };
  helpers.graph.add(
    new GPULineLocate({
      spatialContext:
        coordinateSystem === 'spherical'
          ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
          : undefined,
      measureMode,
      positions: helpers.input('positions', positions, 'float32x2'),
      lineOffsets: helpers.input('line-offsets', lineOffsets, 'uint32'),
      eventLineIndices: helpers.input('event-lines', new Uint32Array(eventLineIndices), 'uint32'),
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
  requireTestDevice(device);
  const positions = new Float32Array(SPHERICAL_POSITIONS);
  const lineOffsets = new Uint32Array(SPHERICAL_LINE_OFFSETS);
  const eventLineIndices = SPHERICAL_EVENTS.map(event => event.line);
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
      lineOffsets,
      eventLineIndices,
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
  requireTestDevice(device);
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

it('GPULinearReferencing spherical prunes by bounding caps without changing nearest segments', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Many short and a few very long arcs spread over the globe, including the antimeridian, the
  // high latitudes and a zero-length segment, so cap pruning meets large and tiny half-angles.
  let state = 12345;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const lineOffsets = [0];
  const vertices: number[][] = [];
  for (let line = 0; line < 12; line++) {
    let longitude = random() * 360 - 180;
    let latitude = random() * 150 - 75;
    const count = 4 + Math.floor(random() * 60);
    const step = line % 4 === 0 ? 25 : 0.4;
    for (let vertex = 0; vertex < count; vertex++) {
      vertices.push([longitude, latitude]);
      if (vertex === 2) {
        vertices.push([longitude, latitude]);
      }
      longitude = ((longitude + (random() - 0.4) * step + 540) % 360) - 180;
      latitude = Math.max(-80, Math.min(80, latitude + (random() - 0.5) * step));
    }
    lineOffsets.push(vertices.length);
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
    new Uint32Array(lineOffsets),
    new Float32Array(points.flat()),
    1e9
  );
  const rowCount = vertices.length;
  let ambiguous = 0;
  points.forEach((point, index) => {
    const distances: {row: number; distance: number}[] = [];
    for (let line = 0; line + 1 < lineOffsets.length; line++) {
      for (let row = lineOffsets[line]; row + 1 < lineOffsets[line + 1]; row++) {
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
      // The oracle here is a different spherical arc-distance formula over many more points, so allow
      // four times the single-point bound.
    ).toBeLessThanOrEqual(METER_TOLERANCE * 4);
    if (distances[1].distance - distances[0].distance > 15) {
      const line = lineOffsets.findIndex((offset, p) => distances[0].row < lineOffsets[p + 1]);
      expect(actual.lineIndices[index], `point ${index} line`).toBe(line);
      expect(actual.segmentIndices[index], `point ${index} segment`).toBe(
        distances[0].row - lineOffsets[line]
      );
    } else {
      ambiguous++;
    }
  });
  expect(ambiguous).toBeLessThan(points.length * 0.8);
  device.destroy?.();
});

it('GPULinearReferencing spherical arc BVH matches a brute-force scan on 5000 arcs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {vertices, lineOffsets, points, bulgeLine, tieLine} = createSphericalBVHScene();
  for (const searchRadius of [1e9, 150_000]) {
    const actual = await runProjection(
      device,
      'spherical',
      new Float32Array(vertices.flat()),
      new Uint32Array(lineOffsets),
      new Float32Array(points.flat()),
      searchRadius
    );
    expect(actual.overflow).toBe(0);
    let ambiguous = 0;
    points.forEach((point, index) => {
      const {best, secondDistance, competingLineDistance} = getNearestArcCandidates(
        point,
        vertices,
        lineOffsets
      );
      const label = `radius ${searchRadius} point ${index}`;
      if (best.distance > searchRadius + METER_TOLERANCE * 4) {
        expect(actual.lineIndices[index], label).toBe(NO_LINE);
        return;
      }
      if (best.distance > searchRadius - METER_TOLERANCE * 4) {
        // Within rounding of the radius: either answer is valid.
        return;
      }
      expect(Math.abs(actual.distances[index] - best.distance), label).toBeLessThanOrEqual(
        METER_TOLERANCE * 4
      );
      const selectedLine = actual.lineIndices[index];
      const selectedRow = lineOffsets[selectedLine] + actual.segmentIndices[index];
      expect(selectedLine, `${label} selected line`).toBeLessThan(lineOffsets.length - 1);
      expect(selectedRow + 1, `${label} selected segment`).toBeLessThan(
        lineOffsets[selectedLine + 1]
      );
      expect(
        getArcDistanceOracle(point, vertices[selectedRow], vertices[selectedRow + 1]) -
          best.distance,
        `${label} selected segment oracle distance`
      ).toBeLessThanOrEqual(METER_TOLERANCE * 4);
      // Adjacent segments share endpoints, so their distances can be exactly tied over a
      // whole region. Only competing lines count toward the fixture's ambiguity fraction.
      if (competingLineDistance - best.distance > 15) {
        expect(actual.lineIndices[index], `${label} line`).toBe(best.line);
      } else {
        ambiguous++;
      }
      if (secondDistance - best.distance > 15) {
        expect(actual.segmentIndices[index], `${label} segment`).toBe(
          best.row - lineOffsets[best.line]
        );
      }
    });
    expect(ambiguous).toBeLessThan(points.length * 0.2);
    if (searchRadius > 1e8) {
      expect(actual.lineIndices[0], 'bulge apex').toBe(bulgeLine);
      // Exact duplicates tie bitwise: the lower row (the first copy) wins.
      expect(actual.lineIndices[2], 'duplicate tie').toBe(tieLine);
    }
  }
  device.destroy?.();
}, 120000);
