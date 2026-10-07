// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it, vi} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPULineSimplificationParameterValues,
  GPULineSimplification
} from '../../../src/gpu-spatial-analysis/line-simplification';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeVisvalingamImportance, getKeptRowsAboveTolerance} from './visvalingam-oracle';
import {VISVALINGAM_REFERENCE_SCENES} from './visvalingam-reference';

type VisvalingamRun = {
  importanceBits: Uint32Array;
  converged: number;
  roundCount: number;
  /** Kept rows (ascending) at each tolerance, from one compiled graph. */
  kept: number[][];
  keptLineCounts: number[][];
  compileCount: number;
};

/** Runs importance and the selection at each tolerance in ONE graph and parameter buffer. */
async function runVisvalingam(
  device: Device,
  positions: Float32Array,
  trackOffsets: Uint32Array,
  tolerances: readonly number[],
  options: {neighborhoodRadius?: number; maximumRounds?: number} = {}
): Promise<VisvalingamRun> {
  const rowCount = positions.length / 2;
  const lineCount = trackOffsets.length - 1;
  const buffers: Buffer[] = [];
  const track = <B extends Buffer>(buffer: B): B => {
    buffers.push(buffer);
    return buffer;
  };
  const positionBuffer = track(createInputBuffer(device, positions));
  const offsetBuffer = track(createInputBuffer(device, trackOffsets));
  const importance = track(createOutputBuffer(device, rowCount));
  const converged = track(createOutputBuffer(device, 1));
  const roundCount = track(createOutputBuffer(device, 1));
  const ids = track(createOutputBuffer(device, rowCount));
  const count = track(createOutputBuffer(device, 1));
  const overflow = track(createOutputBuffer(device, 1));
  const lineCounts = track(createOutputBuffer(device, lineCount));
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'visvalingam-tolerance',
    format: 'float32',
    length: 4
  });
  const graph = new GPUCommandGraph(device, {id: 'visvalingam-graph'});
  graph.add(
    new GPULineSimplification({
      id: 'visvalingam',
      method: 'visvalingam',
      ...options,
      positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', rowCount),
      trackOffsets: importGraphBuffer(graph, 'offsets', offsetBuffer, 'uint32', lineCount + 1),
      importance: importGraphBuffer(graph, 'importance', importance, 'float32', rowCount),
      status: {
        converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1),
        roundCount: importGraphBuffer(graph, 'round-count', roundCount, 'uint32', 1)
      },
      parameters: parameterBuffer.importToGraph(graph),
      selection: {
        output: {
          ids: importGraphBuffer(graph, 'ids', ids, 'uint32', rowCount),
          count: importGraphBuffer(graph, 'count', count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
        },
        lineCounts: importGraphBuffer(graph, 'line-counts', lineCounts, 'uint32', lineCount)
      }
    })
  );
  const compiled = graph.compile();
  const compileSpy = vi.spyOn(graph, 'compile');
  const kept: number[][] = [];
  const keptLineCounts: number[][] = [];
  let importanceBits = new Uint32Array(0);
  for (const tolerance of tolerances) {
    parameterBuffer.write(getGPULineSimplificationParameterValues({tolerance}));
    submitGraph(device, compiled, undefined);
    const [keptCount] = await readUint32(count, 1);
    kept.push(await readUint32(ids, keptCount));
    keptLineCounts.push(await readUint32(lineCounts, lineCount));
    importanceBits = Uint32Array.from(await readUint32(importance, rowCount));
  }
  const result: VisvalingamRun = {
    importanceBits,
    converged: (await readUint32(converged, 1))[0],
    roundCount: (await readUint32(roundCount, 1))[0],
    kept,
    keptLineCounts,
    compileCount: compileSpy.mock.calls.length
  };
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPULineSimplification visvalingam equals the CPU mirror and simplify_coords_vw', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const name of ['hand', 'walk60', 'smooth120', 'walk300']) {
    const scene = VISVALINGAM_REFERENCE_SCENES[name];
    const positions = Float32Array.from(scene.positions);
    const rowCount = positions.length / 2;
    const trackOffsets = Uint32Array.of(0, rowCount);
    const run = await runVisvalingam(device, positions, trackOffsets, scene.tolerances);
    const expected = computeVisvalingamImportance(positions, trackOffsets, 3, 256);
    expect(Array.from(run.importanceBits), name).toEqual(Array.from(expected.importanceBits));
    expect(run.converged).toBe(1);
    expect(run.roundCount).toBe(expected.roundCount);
    expect(run.compileCount).toBe(0);
    let mismatched = 0;
    for (const [index, tolerance] of scene.tolerances.entries()) {
      expect(run.kept[index], `${name} tolerance ${tolerance}`).toEqual(
        getKeptRowsAboveTolerance(expected.importanceBits, tolerance)
      );
      expect(run.keptLineCounts[index][0]).toBe(run.kept[index].length);
      const reference = new Set(scene.kept[index]);
      const actual = new Set(run.kept[index]);
      for (const row of new Set([...reference, ...actual])) {
        mismatched += reference.has(row) === actual.has(row) ? 0 : 1;
      }
    }
    // Pinned parity with the Rust reference: exact on the small scenes, below 1 percent otherwise.
    console.log(
      `visvalingam ${name}: ${rowCount} rows, ${run.roundCount} rounds, ${mismatched} rows differ from simplify_coords_vw`
    );
    if (name === 'walk300') {
      expect(mismatched / (rowCount * scene.tolerances.length)).toBeLessThan(0.01);
    } else {
      expect(mismatched, name).toBe(0);
    }
  }
});

it('GPULineSimplification visvalingam handles many lines, degenerate lines, and a round cap', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const lines = [
    VISVALINGAM_REFERENCE_SCENES.walk60.positions,
    [],
    [5, 5],
    [0, 0, 3, 4],
    // Collinear (areas 0) and duplicated vertices.
    [0, 0, 1, 0, 2, 0, 2, 0, 3, 0, 4, 0],
    VISVALINGAM_REFERENCE_SCENES.smooth120.positions
  ];
  const trackOffsets = new Uint32Array(lines.length + 1);
  const flat: number[] = [];
  for (const [lineIndex, line] of lines.entries()) {
    flat.push(...line);
    trackOffsets[lineIndex + 1] = flat.length / 2;
  }
  const positions = Float32Array.from(flat);
  const tolerances = [0, 0.5, 20, 200];
  const run = await runVisvalingam(device, positions, trackOffsets, tolerances);
  const expected = computeVisvalingamImportance(positions, trackOffsets, 3, 256);
  expect(Array.from(run.importanceBits)).toEqual(Array.from(expected.importanceBits));
  for (const [index, tolerance] of tolerances.entries()) {
    expect(run.kept[index]).toEqual(getKeptRowsAboveTolerance(expected.importanceBits, tolerance));
  }
  // Collinear line keeps only its endpoints at tolerance 0 (area 0 is not above 0).
  expect(run.keptLineCounts[0][4]).toBe(2);
  expect(run.keptLineCounts[0].slice(1, 4)).toEqual([0, 1, 2]);

  // A tight round cap leaves undecided rows at +Infinity, never removing a kept row.
  const capped = await runVisvalingam(device, positions, trackOffsets, [20], {maximumRounds: 3});
  const cappedExpected = computeVisvalingamImportance(positions, trackOffsets, 3, 3);
  expect(capped.converged).toBe(0);
  expect(cappedExpected.converged).toBe(false);
  expect(Array.from(capped.importanceBits)).toEqual(Array.from(cappedExpected.importanceBits));
  for (const row of run.kept[2]) {
    expect(capped.kept[0]).toContain(row);
  }
});

it('GPULineSimplification visvalingam radius changes the result consistently with the mirror', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const positions = Float32Array.from(VISVALINGAM_REFERENCE_SCENES.walk300.positions);
  const trackOffsets = Uint32Array.of(0, 300);
  for (const neighborhoodRadius of [1, 5]) {
    const run = await runVisvalingam(device, positions, trackOffsets, [10], {neighborhoodRadius});
    const expected = computeVisvalingamImportance(positions, trackOffsets, neighborhoodRadius, 256);
    expect(Array.from(run.importanceBits), `radius ${neighborhoodRadius}`).toEqual(
      Array.from(expected.importanceBits)
    );
  }
});
