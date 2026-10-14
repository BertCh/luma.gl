// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUGreatCircleArcsParameterValues,
  getGPULineSegmentizeParameterValues,
  GPUGreatCircleArcs,
  GPULineSegmentize
} from '../../../src/gpu-spatial-analysis/line-segmentize';
import {
  createTransientView,
  GPUCommandGraph,
  GPUScan,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createLinePrefixNode,
  createSegmentizeCountNode,
  createSegmentizeEmitNode
} from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-kernels';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getCentralAngle} from '../geometry-measures/geodesic-oracle';
import {
  createFlatLines,
  createRandom,
  getLinearInterpolationErrorBound,
  segmentizeLines,
  tessellateGreatCircleArcs,
  type FlatLines
} from './line-segmentize-oracle';
import {
  createLineLineFixture,
  expectLineLineParity,
  getSphericalErrorMeters
} from './line-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const EARTH_RADIUS = 6371008.8;
// One f32 ulp of a longitude near 180 degrees is about 1.7 m on the equator; f32 unit vectors add
// about 1e-7 rad (0.6 m). Errors are great-circle distances to the f64 slerp.
const SPHERICAL_TOLERANCE_METERS = 5;
// f32 unit vectors fix the great circle to about 1e-7 / sin(angle) rad, so arcs that are far from
// antipodal (angle below 2.6 rad, about 16,500 km) stay within a few meters of the f64 slerp.
const GLOBAL_ARC_TOLERANCE_METERS = 10;
const MAXIMUM_TEST_ARC_ANGLE = 2.6;

function createSegmentizeFixture(
  device: Device,
  lines: FlatLines,
  options: {capacity: number; spherical?: boolean; maximumPieces?: number; closedLines?: boolean}
) {
  return createLineLineFixture(device, {
    inputs: {
      positions: {values: lines.positions, format: 'float32x2'},
      lineOffsets: {values: lines.lineOffsets, format: 'uint32'}
    },
    capacity: options.capacity,
    lineCapacity: lines.lineOffsets.length - 1,
    createContributor: (inputs, parameters, output) =>
      new GPULineSegmentize({
        positions: inputs['positions'] as GraphDataView<'float32x2'>,
        lineOffsets: inputs['lineOffsets'] as GraphDataView<'uint32'>,
        spatialContext: options.spherical
          ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
          : undefined,
        maximumPieceCount: options.maximumPieces,
        closedLines: options.closedLines,
        parameters,
        output
      })
  });
}

function createRandomPlanarLines(seed: number, lineCount: number): FlatLines {
  const random = createRandom(seed);
  const lines = Array.from({length: lineCount}, (_, lineIndex) => {
    // Include empty and single-vertex lines.
    const length =
      lineIndex % 17 === 3 ? 0 : lineIndex % 13 === 5 ? 1 : 2 + Math.floor(random() * 8);
    let x = random() * 100;
    let y = random() * 100;
    return Array.from({length}, (__, vertexIndex) => {
      if (vertexIndex > 0 && random() > 0.1) {
        x += (random() - 0.5) * 20;
        y += (random() - 0.5) * 20;
      }
      return [x, y];
    });
  });
  return createFlatLines(lines);
}

it('GPULineSegmentize densifies planar lines like the f64 oracle and changes length per frame', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createRandomPlanarLines(3, 400);
  const capacity = 40000;
  const fixture = createSegmentizeFixture(device, lines, {
    capacity,
    maximumPieces: 64
  });
  for (const maximumSegmentLength of [0, 2.718281828, 0.7071067, 13.37, 0.01]) {
    const actual = await fixture.run(getGPULineSegmentizeParameterValues({maximumSegmentLength}));
    const expected = segmentizeLines(lines, {
      maximumSegmentLength,
      maximumPieces: 64
    });
    expectLineLineParity(actual, expected, capacity, 2e-5 * 100, 1e-5);
    expect(actual.lineCount).toBe(400);
    expect(actual.sourceLineIndices).toEqual(Array.from({length: 400}, (_, index) => index));
  }
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPULineSegmentize clamps line offsets and reports overflow at capacity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createRandomPlanarLines(5, 60);
  const capacity = 300;
  const fixture = createSegmentizeFixture(device, lines, {capacity});
  const maximumSegmentLength = 0.5;
  const actual = await fixture.run(getGPULineSegmentizeParameterValues({maximumSegmentLength}));
  const expected = segmentizeLines(lines, {
    maximumSegmentLength,
    maximumPieces: 1024
  });
  expect(expected.positions.length).toBeGreaterThan(capacity);
  expectLineLineParity(actual, expected, capacity, 2e-3, 1e-5);
  expect(actual.overflow).toBe(1);
  const relaxed = await fixture.run(getGPULineSegmentizeParameterValues({maximumSegmentLength: 0}));
  expect(relaxed.overflow).toBe(0);
  expect(relaxed.count).toBe(lines.positions.length / 2);
  fixture.destroy();
});

it('GPULineSegmentize slerps spherical lines with continuous longitudes across the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    // Pacific crossing, west to east across 180.
    [
      [170, 10],
      [-175, 20],
      [-160, 15],
      [175, 5]
    ],
    // High latitude.
    [
      [-30, 70],
      [60, 80]
    ],
    // Short city-scale line (meters).
    [
      [2.35, 48.85],
      [2.351, 48.8505],
      [2.3525, 48.851]
    ],
    [],
    [[10, 10]]
  ]);
  const capacity = 4096;
  const fixture = createSegmentizeFixture(device, lines, {
    capacity,
    spherical: true
  });
  let maximumError = 0;
  for (const maximumSegmentLength of [100000, 333333, 25]) {
    const actual = await fixture.run(getGPULineSegmentizeParameterValues({maximumSegmentLength}));
    const expected = segmentizeLines(lines, {
      maximumSegmentLength,
      maximumPieces: 1024,
      spherical: true,
      radius: EARTH_RADIUS
    });
    maximumError = Math.max(
      maximumError,
      expectLineLineParity(
        actual,
        expected,
        capacity,
        SPHERICAL_TOLERANCE_METERS,
        1e-5,
        getSphericalErrorMeters
      )
    );
    // Continuity: no jump of more than 180 degrees inside a line.
    for (let line = 0; line < 5; line++) {
      for (let row = actual.lineOffsets[line] + 1; row < actual.lineOffsets[line + 1]; row++) {
        expect(Math.abs(actual.positions[row][0] - actual.positions[row - 1][0])).toBeLessThan(180);
      }
    }
  }
  console.log(`GPULineSegmentize spherical max error ${maximumError} m`);
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPUGreatCircleArcs matches the f64 slerp for origin/destination pairs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(11);
  const sources: number[][] = [
    [139.69, 35.69], // Tokyo
    [-0.1276, 51.5072], // London
    [0, 0],
    [12, 41],
    [-170, -60]
  ];
  const targets: number[][] = [
    [-122.42, 37.77], // San Francisco: crosses the antimeridian
    [-74.006, 40.7128], // New York
    [0, 0], // zero length
    [12.0001, 41.0001], // ~14 m
    [170, -65]
  ];
  while (sources.length < 300) {
    const source = [random() * 360 - 180, random() * 160 - 80];
    const target = [random() * 360 - 180, random() * 160 - 80];
    if (getCentralAngle(source, target) < MAXIMUM_TEST_ARC_ANGLE) {
      sources.push(source);
      targets.push(target);
    }
  }
  const f32 = (values: number[][]) => values.map(([x, y]) => [Math.fround(x), Math.fround(y)]);
  const sourceValues = new Float32Array(sources.flat());
  const targetValues = new Float32Array(targets.flat());
  const capacity = 300 * 66;
  const maximumSegmentCount = 64;
  const fixture = createLineLineFixture(device, {
    inputs: {
      sources: {values: sourceValues, format: 'float32x2'},
      targets: {values: targetValues, format: 'float32x2'}
    },
    capacity,
    lineCapacity: sources.length,
    createContributor: (inputs, parameters, output) =>
      new GPUGreatCircleArcs({
        sources: inputs['sources'] as GraphDataView<'float32x2'>,
        targets: inputs['targets'] as GraphDataView<'float32x2'>,
        maximumSegmentCount,
        parameters,
        output
      })
  });
  let maximumError = 0;
  for (const [maximumSegmentLength, minimumSegments] of [
    [500000, 1],
    [0, 16],
    [1234567, 4]
  ]) {
    const actual = await fixture.run(
      getGPUGreatCircleArcsParameterValues({
        maximumSegmentLength,
        minimumSegments
      })
    );
    const expected = tessellateGreatCircleArcs(f32(sources), f32(targets), {
      maximumSegmentLength,
      minimumSegments,
      maximumSegmentCount,
      radius: EARTH_RADIUS
    });
    maximumError = Math.max(
      maximumError,
      expectLineLineParity(
        actual,
        expected,
        capacity,
        GLOBAL_ARC_TOLERANCE_METERS,
        2e-5,
        getSphericalErrorMeters
      )
    );
    // Tokyo -> San Francisco ends at the unwrapped longitude.
    const tokyoEnd = actual.positions[actual.lineOffsets[1] - 1];
    expect(tokyoEnd[0]).toBeCloseTo(-122.42 + 360, 3);
  }
  console.log(`GPUGreatCircleArcs max error ${maximumError} m`);
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPUGreatCircleArcs reports overflow and clamps offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sources = new Float32Array([0, 0, 10, 10, 20, 20]);
  const targets = new Float32Array([90, 0, 10, 50, -160, 20]);
  const capacity = 20;
  const fixture = createLineLineFixture(device, {
    inputs: {
      sources: {values: sources, format: 'float32x2'},
      targets: {values: targets, format: 'float32x2'}
    },
    capacity,
    lineCapacity: 3,
    createContributor: (inputs, parameters, output) =>
      new GPUGreatCircleArcs({
        sources: inputs['sources'] as GraphDataView<'float32x2'>,
        targets: inputs['targets'] as GraphDataView<'float32x2'>,
        parameters,
        output
      })
  });
  const actual = await fixture.run(
    getGPUGreatCircleArcsParameterValues({
      maximumSegmentLength: 0,
      minimumSegments: 8
    })
  );
  expect(actual.requiredCount).toBe(27);
  expect(actual.count).toBe(20);
  expect(actual.overflow).toBe(1);
  expect(actual.lineOffsets).toEqual([0, 9, 18, 20]);
  fixture.destroy();
});

it('GPULineSegmentize matches the oracle for long lines and one segment cut into many pieces', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(31);
  const walk = (length: number) => {
    let x = random() * 50;
    let y = random() * 50;
    return Array.from({length}, () => {
      x += (random() - 0.5) * 8;
      y += (random() - 0.5) * 8;
      return [x, y];
    });
  };
  // Row counts around the 64-row prefix tile, one 2000-row line, and a 2-vertex line of length
  // 5000 that is cut into 1000 pieces by the segment length below.
  const lines = createFlatLines([
    walk(63),
    walk(64),
    walk(65),
    walk(2000),
    [
      [0, 0],
      [5000, 0]
    ],
    walk(3)
  ]);
  const capacity = 60000;
  const fixture = createSegmentizeFixture(device, lines, {capacity, maximumPieces: 1024});
  for (const maximumSegmentLength of [5, 0, 41.3]) {
    const actual = await fixture.run(getGPULineSegmentizeParameterValues({maximumSegmentLength}));
    const expected = segmentizeLines(lines, {maximumSegmentLength, maximumPieces: 1024});
    expectLineLineParity(actual, expected, capacity, 2e-3, 2e-5);
  }
  fixture.destroy();
});

it('GPULineSegmentize densifies implicit and explicit planar and spherical closing edges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const spherical of [false, true]) {
    const ring = spherical
      ? [
          [177, -20],
          [-178, -20],
          [-178, -15],
          [177, -15]
        ]
      : [
          [0, 0],
          [2, 0],
          [2, 2],
          [0, 2]
        ];
    const lines = createFlatLines([
      ring,
      [...ring, ring[0]],
      [],
      [[1, 1]],
      [
        [3, 4],
        [3, 4]
      ]
    ]);
    const maximumSegmentLength = spherical ? 150000 : 0.5;
    const fixture = createSegmentizeFixture(device, lines, {
      capacity: 512,
      spherical,
      closedLines: true
    });
    try {
      const actual = await fixture.run(getGPULineSegmentizeParameterValues({maximumSegmentLength}));
      const expected = segmentizeLines(lines, {
        maximumSegmentLength,
        maximumPieces: 1024,
        spherical,
        radius: EARTH_RADIUS,
        closedLines: true
      });
      expectLineLineParity(
        actual,
        expected,
        512,
        spherical ? SPHERICAL_TOLERANCE_METERS : 1e-5,
        1e-5,
        spherical ? getSphericalErrorMeters : undefined
      );
      const ringLength = actual.lineOffsets[1];
      expect(actual.lineOffsets[2]).toBe(2 * ringLength);
      expect(actual.positions.slice(ringLength, 2 * ringLength)).toEqual(
        actual.positions.slice(0, ringLength)
      );
      expect(actual.positions[ringLength - 1]).not.toEqual(actual.positions[0]);
      expect(actual.sourceRows[ringLength - 1]).toBe(3);
    } finally {
      fixture.destroy();
    }
  }
});

/** Direct factory fixture used by polygon preparation: integer turns and a caller-cleared flag. */
function createDegreeStepFixture(
  device: Device,
  lines: FlatLines,
  options: {
    capacity: number;
    maximumPieceCount?: number;
    closedLines?: boolean;
    interpolation?: 'linear-degrees' | 'great-circle';
  }
) {
  const graph = new GPUCommandGraph(device, {id: 'degree-step'});
  const rowCount = lines.positions.length / 2;
  const positionBuffer = createInputBuffer(device, lines.positions);
  const offsetBuffer = createInputBuffer(device, lines.lineOffsets);
  const outputBuffer = createOutputBuffer(device, 2 * options.capacity);
  const turnBuffer = createOutputBuffer(device, options.capacity);
  const countBuffer = createOutputBuffer(device, rowCount);
  const approximationBuffer = createOutputBuffer(device, 1);
  const positions = importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', rowCount);
  const lineOffsets = importGraphBuffer(
    graph,
    'offsets',
    offsetBuffer,
    'uint32',
    lines.lineOffsets.length
  );
  const counts = importGraphBuffer(graph, 'counts', countBuffer, 'uint32', rowCount);
  const starts = createTransientView(graph, 'starts', 'uint32', rowCount);
  const rowTurns = createTransientView(graph, 'row-turns', 'sint32', rowCount);
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 4
  });
  const common = {
    operation: 'GPULineSegmentize',
    positions,
    lineOffsets,
    coordinateSystem: 'spherical' as const,
    radius: 1,
    closedLines: options.closedLines
  };
  graph.add(createLinePrefixNode(graph, {...common, id: 'prefix', rowTurns}));
  graph.add(
    createSegmentizeCountNode(graph, {
      ...common,
      id: 'count',
      counts,
      parameters: parameters.importToGraph(graph),
      parameterIndex: 2,
      pieceCountModel: 'degree-step',
      interpolation: options.interpolation,
      maximumPieceCount: options.maximumPieceCount ?? 1024,
      approximation: importGraphBuffer(graph, 'approximation', approximationBuffer, 'uint32', 1)
    })
  );
  graph.add(new GPUScan({id: 'scan', input: counts, output: starts, mode: 'exclusive'}));
  graph.add(
    createSegmentizeEmitNode(graph, {
      ...common,
      id: 'emit',
      counts,
      starts,
      rowTurns,
      interpolation: options.interpolation,
      wrapOutputLongitudes: true,
      outputPositions: importGraphBuffer(
        graph,
        'output',
        outputBuffer,
        'float32x2',
        options.capacity
      ),
      outputTurns: importGraphBuffer(graph, 'turns', turnBuffer, 'sint32', options.capacity)
    })
  );
  const compiled = graph.compile();
  return {
    async run(maximumSegmentDegrees: number) {
      approximationBuffer.write(new Uint32Array([0]));
      parameters.write(new Float32Array([99, 99, maximumSegmentDegrees, 99]));
      submitGraph(device, compiled, undefined);
      const countsReadback = await readUint32(countBuffer, rowCount);
      const count = Math.min(
        countsReadback.reduce((sum, value) => sum + value, 0),
        options.capacity
      );
      const values = await readFloat32(outputBuffer, 2 * count);
      const turns = (await readUint32(turnBuffer, count)).map(value => value | 0);
      return {
        positions: Array.from({length: count}, (_, row) => values.slice(2 * row, 2 * row + 2)),
        turns,
        counts: countsReadback,
        approximation: (await readUint32(approximationBuffer, 1))[0]
      };
    },
    destroy() {
      compiled.destroy();
      parameters.destroy();
      for (const buffer of [
        positionBuffer,
        offsetBuffer,
        outputBuffer,
        turnBuffer,
        countBuffer,
        approximationBuffer
      ])
        buffer.destroy();
    }
  };
}

it('degree-step great-circle kernels emit 600 pieces for 60 degrees at 45N, within 1e-6 radians', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    [
      [150, 45],
      [-150, 45]
    ]
  ]);
  const fixture = createDegreeStepFixture(device, lines, {capacity: 1025});
  try {
    const actual = await fixture.run(0.1);
    const expected = segmentizeLines(lines, {
      maximumSegmentLength: 0.1,
      maximumPieces: 1024,
      spherical: true,
      pieceCountModel: 'degree-step',
      wrapOutputLongitudes: true
    });
    expect(actual.counts).toEqual([600, 1]);
    expect(actual.approximation).toBe(0);
    expect(actual.positions[0]).toEqual([150, 45]);
    expect(actual.positions[600]).toEqual([-150, 45]);
    expect(actual.turns[600]).toBe(1);
    actual.positions.forEach((position, row) => {
      expect(position[0]).toBeGreaterThanOrEqual(-180);
      expect(position[0]).toBeLessThanOrEqual(180);
      // The audit bounds angular slerp error, not longitude ULPs: at 45N, 1e-6 rad
      // permits about 8.1e-5 degrees of longitude (dLambda*cos(latitude)).
      expect(getCentralAngle(position, expected.positions[row])).toBeLessThanOrEqual(1e-6);
      // Angular distance is periodic; separately reject a wrong whole-turn prefix.
      expect(
        Math.abs(
          position[0] +
            360 * actual.turns[row] -
            (expected.positions[row][0] + 360 * expected.turns[row])
        )
      ).toBeLessThan(180);
    });
  } finally {
    fixture.destroy();
  }
});

it('degree-step kernels retain wrapped vertex bits and integer turns across long prefixes and clamping', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const interpolation of ['linear-degrees', 'great-circle'] as const) {
    // More than 64 rows exercises the cooperative prefix, including many whole turns.
    const ring = Array.from(
      {length: 130},
      (_, row) =>
        [
          [170, 70],
          [-170, 71],
          [-10, 72],
          [150, 71]
        ][row % 4]
    );
    const lines = createFlatLines([
      ring,
      [
        [-0, 10],
        [120, 10],
        [180, 10],
        [-180, 10],
        [-120, 10]
      ],
      [
        [177.1234567, -17.234567],
        [-178.7654321, -16.345678],
        [-176.112233, -15.998877]
      ]
    ]);
    const fixture = createDegreeStepFixture(device, lines, {
      capacity: 2048,
      closedLines: true,
      maximumPieceCount: 8,
      interpolation
    });
    try {
      for (const maximumSegmentDegrees of [0, 0.1, 180]) {
        const actual = await fixture.run(maximumSegmentDegrees);
        const expected = segmentizeLines(lines, {
          maximumSegmentLength: maximumSegmentDegrees,
          maximumPieces: 8,
          spherical: true,
          closedLines: true,
          pieceCountModel: 'degree-step',
          interpolation,
          wrapOutputLongitudes: true
        });
        expect(actual.positions).toHaveLength(expected.positions.length);
        expect(actual.approximation).toBe(maximumSegmentDegrees === 0.1 ? 1 : 0);
        let outputRow = 0;
        const keptRows = new Set<number>();
        for (let row = 0; row < actual.counts.length; row++) {
          if (actual.counts[row] > 0) {
            keptRows.add(outputRow);
            expect(new Uint32Array(new Float32Array(actual.positions[outputRow]).buffer)).toEqual(
              new Uint32Array(lines.positions.buffer, row * 8, 2)
            );
            expect(actual.turns[outputRow]).toBe(expected.turns[outputRow]);
            expect(actual.positions[outputRow][0] + 360 * actual.turns[outputRow]).toBe(
              lines.positions[2 * row] + 360 * expected.turns[outputRow]
            );
          }
          outputRow += actual.counts[row];
        }
        actual.positions.forEach((position, row) => {
          expect(position[0]).toBeGreaterThanOrEqual(-180);
          expect(position[0]).toBeLessThanOrEqual(180);
          const expectedLongitude = expected.positions[row][0] + 360 * expected.turns[row];
          const actualLongitude = position[0] + 360 * actual.turns[row];
          if (keptRows.has(row)) return;
          if (interpolation === 'linear-degrees') {
            const sourceRow = expected.sourceRows[row];
            const line = Array.from(lines.lineOffsets).findIndex(offset => offset > sourceRow) - 1;
            const nextRow =
              sourceRow + 1 < lines.lineOffsets[line + 1] ? sourceRow + 1 : lines.lineOffsets[line];
            // For u=2^-24, the longitude lerp bound is
            // u*(|rawStep| + 8*|wrappedStep| + |start|)/(1-8u), also covering
            // the weighted mix form (see the oracle helper). Integer turns add no f32 error.
            for (const axis of [0, 1]) {
              expect(
                Math.abs(
                  axis === 0
                    ? actualLongitude - expectedLongitude
                    : position[1] - expected.positions[row][1]
                )
              ).toBeLessThanOrEqual(
                getLinearInterpolationErrorBound(
                  lines.positions[2 * sourceRow + axis],
                  lines.positions[2 * nextRow + axis],
                  axis === 0
                )
              );
            }
          } else {
            // Slerp has the audit's 1e-6 rad angular budget; a longitude ULP is not
            // an angular bound, especially on the 70N prefix. Check turns separately.
            expect(Math.abs(actualLongitude - expectedLongitude)).toBeLessThan(180);
          }
          expect(getCentralAngle(position, expected.positions[row])).toBeLessThanOrEqual(1e-6);
        });
      }
    } finally {
      fixture.destroy();
    }
  }
});
