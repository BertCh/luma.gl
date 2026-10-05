// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  cellToBoundary,
  cellToChildren,
  getHexagonEdgeLengthAvg,
  getPentagons,
  getRes0Cells,
  gridDisk
} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {H3_BOUNDARY_WGSL} from '../../../src/geospatial/cell-indexing/h3-boundary-wgsl';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {h3ToBigInt, splitCellKey} from '../cell-aggregation/cell-aggregation-oracle';
import {createRandom} from '../cell-aggregation/cell-aggregation-points';
import {createInputBuffer, createOutputBuffer} from '../../utils/gpu-contributor-test-utils';

const CAPACITY = 8192;
const MAXIMUM_VERTICES = 10;
/** Absolute vertex tolerance in degrees (f32 longitude rounding plus GPU trig error). */
const MAXIMUM_DEGREE_ERROR = 3e-5;

type BoundaryRunner = {
  /** Runs `cellIndexH3GetBoundary` for each cell (h3-js hex strings). */
  run(cells: string[]): Promise<{counts: number[]; points: number[]}>;
  destroy(): void;
};

/** Compiles one reusable kernel that decodes up to CAPACITY H3 boundaries. */
function createRunner(device: Device): BoundaryRunner {
  const cells = createInputBuffer(device, new Uint32Array(2 * CAPACITY));
  const counts = createOutputBuffer(device, CAPACITY);
  const points = createOutputBuffer(device, CAPACITY * MAXIMUM_VERTICES * 2);
  const graph = new GPUCommandGraph(device, {id: 'h3-boundary-graph'});
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'h3-boundary-kernel',
      operation: 'H3Boundary',
      bindings: [
        {
          name: 'cells',
          view: importGraphBuffer(graph, 'cells', cells, 'uint32x2', CAPACITY),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'counts',
          view: importGraphBuffer(graph, 'counts', counts, 'uint32', CAPACITY),
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'points',
          view: importGraphBuffer(
            graph,
            'points',
            points,
            'float32x2',
            CAPACITY * MAXIMUM_VERTICES
          ),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: CAPACITY,
      declarations: `${dggs.source}\n${H3_BOUNDARY_WGSL}`,
      body: `let cell = vec2u(cells[cellsOffset + 2u * index + 1u], cells[cellsOffset + 2u * index]);
  let boundary = cellIndexH3GetBoundary(cell);
  counts[countsOffset + index] = boundary.count;
  for (var vertex = 0u; vertex < ${MAXIMUM_VERTICES}u; vertex++) {
    var point = vec2f(0.0);
    if (vertex < boundary.count) {
      point = boundary.points[vertex];
    }
    let slot = 2u * (index * ${MAXIMUM_VERTICES}u + vertex);
    points[pointsOffset + slot] = point.x;
    points[pointsOffset + slot + 1u] = point.y;
  }`
    })
  );
  const compiled = graph.compile();
  return {
    async run(cellStrings) {
      const result = {counts: [] as number[], points: [] as number[]};
      for (let start = 0; start < cellStrings.length; start += CAPACITY) {
        const batch = cellStrings.slice(start, start + CAPACITY);
        const words = new Uint32Array(2 * CAPACITY);
        batch.forEach((cell, row) => words.set(splitCellKey(h3ToBigInt(cell)), 2 * row));
        cells.write(words);
        submitGraph(device, compiled, undefined);
        const countBytes = await counts.readAsync();
        const pointBytes = await points.readAsync();
        const countWords = new Uint32Array(countBytes.buffer, countBytes.byteOffset, batch.length);
        const pointWords = new Float32Array(
          pointBytes.buffer,
          pointBytes.byteOffset,
          batch.length * MAXIMUM_VERTICES * 2
        );
        countWords.forEach(count => result.counts.push(count));
        pointWords.forEach(value => result.points.push(value));
      }
      return result;
    },
    destroy() {
      compiled.destroy();
      cells.destroy();
      counts.destroy();
      points.destroy();
    }
  };
}

/** Planar-corrected angular distance in degrees, wrapping longitude across the antimeridian. */
function getDegreeError(
  longitude: number,
  latitude: number,
  expectedLongitude: number,
  expectedLatitude: number
): number {
  let deltaLongitude = Math.abs(longitude - expectedLongitude);
  deltaLongitude = Math.min(deltaLongitude, 360 - deltaLongitude);
  const scale = Math.cos((expectedLatitude * Math.PI) / 180);
  return Math.hypot(deltaLongitude * scale, latitude - expectedLatitude);
}

/** Per-resolution comparison accumulator. */
type Report = {rows: number; countMismatches: number; maximumError: number; worst: string};

function createReports(): Map<number, Report> {
  return new Map();
}

/** Compares every cell with h3-js; vertex counts exactly, vertices within tolerance. */
async function compareCells(
  runner: BoundaryRunner,
  cells: string[],
  reports: Map<number, Report>,
  resolutionOf: (cell: string) => number
): Promise<void> {
  const result = await runner.run(cells);
  cells.forEach((cell, row) => {
    const resolution = resolutionOf(cell);
    const report = reports.get(resolution) ?? {
      rows: 0,
      countMismatches: 0,
      maximumError: 0,
      worst: ''
    };
    reports.set(resolution, report);
    report.rows++;
    const expected = cellToBoundary(cell);
    if (result.counts[row] !== expected.length) {
      report.countMismatches++;
      report.worst = `${cell} count ${result.counts[row]} vs ${expected.length}`;
      return;
    }
    // The f32 floor: longitudes near +-180 round at 1.5e-5 degrees, and the GPU atan2 adds a few
    // ulps; the measured maximum over all resolutions is about 2.7e-5 (see the console report).
    const edgeDegrees = getHexagonEdgeLengthAvg(resolution, 'km' as never) / 111.32;
    const allowed = MAXIMUM_DEGREE_ERROR + 0.02 * edgeDegrees;
    expected.forEach(([latitude, longitude], vertex) => {
      const slot = 2 * (row * MAXIMUM_VERTICES + vertex);
      const error = getDegreeError(
        result.points[slot],
        result.points[slot + 1],
        longitude,
        latitude
      );
      if (error > report.maximumError) {
        report.maximumError = error;
        report.worst = `${cell} vertex ${vertex}`;
      }
      expect(error, `${cell} res ${resolution} vertex ${vertex}`).toBeLessThanOrEqual(allowed);
    });
  });
}

function expectAllCountsMatch(reports: Map<number, Report>, label: string): void {
  for (const [resolution, report] of reports) {
    expect(report.countMismatches, `${label} res ${resolution} count mismatches`).toBe(0);
  }
  const lines = [...reports]
    .sort(([left], [right]) => left - right)
    .map(
      ([resolution, report]) =>
        `${label} res ${resolution}: ${report.rows} cells, max error ${report.maximumError.toExponential(2)} deg (${report.worst}) count mismatches ${report.countMismatches}`
    );
  console.log(lines.join('\n'));
}

it('H3_BOUNDARY_WGSL matches h3-js for every res 0-2 cell and all pentagons', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const runner = createRunner(device);
  const resolutionOf = (cell: string) => Number.parseInt(cell[1], 16);
  const reports = createReports();
  const base = getRes0Cells();
  const cells = new Set<string>(base);
  for (const baseCell of base) {
    for (const child of cellToChildren(baseCell, 1)) {
      cells.add(child);
    }
    for (const child of cellToChildren(baseCell, 2)) {
      cells.add(child);
    }
  }
  for (let resolution = 0; resolution <= 15; resolution++) {
    getPentagons(resolution).forEach(pentagon => cells.add(pentagon));
  }
  await compareCells(runner, [...cells], reports, resolutionOf);
  expectAllCountsMatch(reports, 'res0-2+pentagons');
  expect(reports.get(0)!.rows).toBe(122);
  expect(reports.get(1)!.rows).toBe(842);
  expect(reports.get(2)!.rows).toBe(5882);
  runner.destroy();
});

it('H3_BOUNDARY_WGSL matches h3-js around every pentagon at res 3-8', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const runner = createRunner(device);
  const resolutionOf = (cell: string) => Number.parseInt(cell[1], 16);
  const reports = createReports();
  for (let resolution = 3; resolution <= 8; resolution++) {
    const cells = new Set<string>();
    for (const pentagon of getPentagons(resolution)) {
      gridDisk(pentagon, 2).forEach(cell => cells.add(cell));
    }
    await compareCells(runner, [...cells], reports, resolutionOf);
  }
  expectAllCountsMatch(reports, 'pentagon rings');
  runner.destroy();
});

it('H3_BOUNDARY_WGSL matches h3-js for random cells at res 3-15', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const runner = createRunner(device);
  const resolutionOf = (cell: string) => Number.parseInt(cell[1], 16);
  const reports = createReports();
  const random = createRandom(7);
  for (let resolution = 3; resolution <= 15; resolution++) {
    const cells: string[] = [];
    for (let count = 0; count < 2000; count++) {
      cells.push(getRandomCell(random, resolution));
    }
    await compareCells(runner, cells, reports, resolutionOf);
  }
  expectAllCountsMatch(reports, 'random');
  runner.destroy();
});

/** A uniformly random valid H3 cell at `resolution` (random base cell and digits). */
function getRandomCell(random: () => number, resolution: number): string {
  const baseCells = getRes0Cells();
  let cell = baseCells[Math.floor(random() * baseCells.length)];
  for (let level = 1; level <= resolution; level++) {
    const children = cellToChildren(cell, level);
    cell = children[Math.floor(random() * children.length)];
  }
  return cell;
}
