// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUCellMeasures} from '../../../src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {h3ToBigInt, splitCellKey} from '../cell-aggregation/cell-aggregation-oracle';
import {H3_MEASURES_FIXTURES} from './h3-measures-fixtures';

const MAXIMUM_EDGE_COUNT = 10;
const INVALID_CELLS = [0n, 1n, 0xffffffffffffffffn];

/** Fallback cells (pentagons, cells across an icosahedron edge) are only asserted up to here. */
const FALLBACK_MAXIMUM_RESOLUTION = 4;

/**
 * Relative error limit. Single-face hexagons use exact lattice offsets (measured worst 9.6e-7 over
 * resolutions 0-15); fallback cells measured 6e-7 at resolution 0, 2e-5 at 2, 5e-5 at 3 and 2.5e-4
 * at 4, growing about 40x per two resolutions.
 */
function getRelativeTolerance(isExactPath: boolean): number {
  return isExactPath ? 2e-6 : 1e-3;
}

it('GPUCellMeasures area, perimeter, edge lengths and pentagon flags match python h3 4.5.0', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cells = [...H3_MEASURES_FIXTURES.map(row => h3ToBigInt(row[0])), ...INVALID_CELLS];
  const rows = cells.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const words = new Uint32Array(2 * rows);
  cells.forEach((cell, row) => words.set(splitCellKey(cell), 2 * row));
  const graph = new GPUCommandGraph(device, {id: 'cell-measures-graph'});
  const cellsBuffer = track(createInputBuffer(device, words));
  const areasBuffer = track(createOutputBuffer(device, rows));
  const perimetersBuffer = track(createOutputBuffer(device, rows));
  const edgeLengthsBuffer = track(createOutputBuffer(device, rows * MAXIMUM_EDGE_COUNT));
  const edgeCountsBuffer = track(createOutputBuffer(device, rows));
  const pentagonsBuffer = track(createOutputBuffer(device, rows));
  graph.add(
    new GPUCellMeasures({
      family: 'h3',
      cells: importGraphBuffer(graph, 'cells', cellsBuffer, 'uint32x2', rows),
      maximumEdgeCount: MAXIMUM_EDGE_COUNT,
      output: {
        areas: importGraphBuffer(graph, 'areas', areasBuffer, 'float32', rows),
        perimeters: importGraphBuffer(graph, 'perimeters', perimetersBuffer, 'float32', rows),
        edgeLengths: importGraphBuffer(
          graph,
          'edge-lengths',
          edgeLengthsBuffer,
          'float32',
          rows * MAXIMUM_EDGE_COUNT
        ),
        edgeCounts: importGraphBuffer(graph, 'edge-counts', edgeCountsBuffer, 'uint32', rows),
        pentagons: importGraphBuffer(graph, 'pentagons', pentagonsBuffer, 'uint32', rows)
      }
    })
  );
  const compiled = graph.compile();
  const startTime = performance.now();
  submitGraph(device, compiled, undefined);
  const areas = await readFloat32(areasBuffer, rows);
  const elapsed = performance.now() - startTime;
  const perimeters = await readFloat32(perimetersBuffer, rows);
  const edgeLengths = await readFloat32(edgeLengthsBuffer, rows * MAXIMUM_EDGE_COUNT);
  const edgeCounts = await readUint32(edgeCountsBuffer, rows);
  const pentagons = await readUint32(pentagonsBuffer, rows);

  let worstExact = 0;
  let worstFallback = 0;
  let exactCount = 0;
  H3_MEASURES_FIXTURES.forEach(
    ([cell, areaKm2, perimeterKm, edgeCount, isSingleFace, isPentagon, expectedEdges], row) => {
      const resolution = Number((h3ToBigInt(cell) >> 52n) & 15n);
      const label = `${cell} (res ${resolution})`;
      expect(pentagons[row], `${label} pentagon`).toBe(isPentagon);
      expect(edgeCounts[row], `${label} edge count`).toBe(edgeCount);
      const isExactPath = isSingleFace === 1 && isPentagon === 0;
      const tolerance = getRelativeTolerance(isExactPath);
      const areaError = Math.abs(areas[row] - areaKm2) / areaKm2;
      const perimeterError = Math.abs(perimeters[row] - perimeterKm) / perimeterKm;
      let edgeError = 0;
      expectedEdges.forEach((expectedEdge, edge) => {
        const actual = edgeLengths[row * MAXIMUM_EDGE_COUNT + edge];
        edgeError = Math.max(edgeError, Math.abs(actual - expectedEdge) / expectedEdge);
      });
      for (let edge = edgeCount; edge < MAXIMUM_EDGE_COUNT; edge++) {
        expect(edgeLengths[row * MAXIMUM_EDGE_COUNT + edge], `${label} padding`).toBe(0);
      }
      const error = Math.max(areaError, perimeterError, edgeError);
      if (isExactPath) {
        exactCount++;
        worstExact = Math.max(worstExact, error);
      } else if (resolution <= FALLBACK_MAXIMUM_RESOLUTION) {
        worstFallback = Math.max(worstFallback, error);
      }
      if (isExactPath || resolution <= FALLBACK_MAXIMUM_RESOLUTION) {
        expect(
          error,
          `${label} relative error (area ${areaError.toExponential(1)}, perimeter ${perimeterError.toExponential(1)}, edges ${edgeError.toExponential(1)})`
        ).toBeLessThan(tolerance);
      }
    }
  );
  for (let row = H3_MEASURES_FIXTURES.length; row < rows; row++) {
    expect(areas[row]).toBe(0);
    expect(perimeters[row]).toBe(0);
    expect(edgeCounts[row]).toBe(0);
    expect(pentagons[row]).toBe(0);
  }
  console.log(
    `GPUCellMeasures ${rows} cells in ${elapsed.toFixed(0)} ms: worst relative error ${worstExact.toExponential(2)} over ${exactCount} single-face hexagons, ${worstFallback.toExponential(2)} over fallback cells of resolution <= ${FALLBACK_MAXIMUM_RESOLUTION}`
  );
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
}, 120_000);

it('GPUCellMeasures validates its properties', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'cell-measures-validation'});
  const buffer = createInputBuffer(device, new Uint32Array(4));
  const outBuffer = createOutputBuffer(device, 2);
  const cells = importGraphBuffer(graph, 'cells', buffer, 'uint32x2', 2);
  const areas = importGraphBuffer(graph, 'areas', outBuffer, 'float32', 2);
  expect(() => new GPUCellMeasures({family: 'h3', cells, output: {}})).toThrow(/at least one/);
  expect(
    () => new GPUCellMeasures({family: 'h3', cells, maximumEdgeCount: 5, output: {areas}})
  ).toThrow(/maximumEdgeCount/);
  expect(
    () => new GPUCellMeasures({family: 'quadbin' as unknown as 'h3', cells, output: {areas}})
  ).toThrow(/family/);
  buffer.destroy();
  outBuffer.destroy();
});
