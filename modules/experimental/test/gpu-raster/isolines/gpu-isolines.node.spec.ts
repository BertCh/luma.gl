// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUIsolines} from '../../../src/gpu-raster/isolines/gpu-isolines';
import {
  getGPUIsolinesParameterValues,
  GPU_ISOLINES_PARAMETER_LENGTH
} from '../../../src/gpu-raster/isolines/isolines-parameters';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeIsolinesOnCPU,
  getPolylineVertices,
  stitchIsolinesOnCPU,
  type IsolinesScene
} from './isolines-oracle';

function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

function createRandomScene(seed: number, width: number, height: number): IsolinesScene {
  const random = createRandom(seed);
  const values = new Float32Array(width * height);
  for (let index = 0; index < values.length; index++) {
    const roll = random();
    // Mostly integers, so levels land exactly on samples, plus some NaN holes.
    values[index] = roll < 0.04 ? NaN : roll < 0.7 ? Math.floor(random() * 10) : random() * 10;
  }
  return {
    width,
    height,
    values,
    levels: [2, 4.5, 5, 7.25],
    extent: [100, -50, 100 + width * 2, -50 + height * 3]
  };
}

it('getGPUIsolinesParameterValues packs and validates', () => {
  const grid = makeGPURasterMetadata({width: 4, height: 5});
  const values = getGPUIsolinesParameterValues({levelCount: 3, extent: [10, 20, 18, 30]}, grid);
  expect(values.length).toBe(GPU_ISOLINES_PARAMETER_LENGTH);
  expect(Array.from(values)).toEqual([3, 10, 20, 2, 2, 0, 0, 0]);
  const settings = {levelCount: 1, extent: [0, 0, 1, 1]} as const;
  const square = makeGPURasterMetadata({width: 4, height: 4});
  expect(() =>
    getGPUIsolinesParameterValues(settings, makeGPURasterMetadata({width: 1, height: 4}))
  ).toThrow(/width/);
  expect(() => getGPUIsolinesParameterValues({...settings, levelCount: -1}, square)).toThrow(
    /levelCount/
  );
  expect(() => getGPUIsolinesParameterValues({...settings, extent: [0, 0, 0, 1]}, square)).toThrow(
    /extent/
  );
  // y-down extent
  expect(() => getGPUIsolinesParameterValues({...settings, extent: [0, 1, 1, 0]}, square)).toThrow(
    /extent/
  );
  expect(() => getGPUIsolinesParameterValues(settings, square, new Float32Array(2))).toThrow(
    /target/
  );
});

it('GPUIsolines validates views and schedules deterministic nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x4' | 'float32x2' | 'uint32x2'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${name}-${serial++}`, format, length);
  const output = () => ({
    segments: view('segments', 'float32x4', 40),
    segmentLevels: view('segment-levels', 'uint32', 40),
    status: {overflow: view('overflow', 'uint32', 1), count: view('count', 'uint32', 1)}
  });
  const base = {
    grid: makeGPURasterMetadata({width: 5, height: 4}),
    values: view('raster', 'float32', 20),
    levels: view('levels', 'float32', 3),
    parameters: view('parameters', 'float32', GPU_ISOLINES_PARAMETER_LENGTH)
  };
  const segmentsOnly = new GPUIsolines({...base, output: output()});
  expect(segmentsOnly.maximumLevelCount).toBe(3);
  expect(segmentsOnly.segmentCapacity).toBe(40);
  const segmentNodes = segmentsOnly.getCommandNodes(graph).map(node => node.id);
  expect(segmentNodes).toContain('isolines-count');
  expect(segmentNodes).toContain('isolines-scatter');
  expect(segmentNodes).toContain('isolines-publish');
  expect(segmentNodes.some(id => id.includes('jump'))).toBe(false);

  const stitched = new GPUIsolines({
    ...base,
    id: 'stitched',
    output: {
      ...output(),
      polylines: {
        vertices: view('vertices', 'float32x2', 80),
        polylineOffsets: view('polyline-offsets', 'uint32', 41),
        polylineLevels: view('polyline-levels', 'uint32', 40),
        polylineClosed: view('polyline-closed', 'uint32', 40),
        polylineCount: view('polyline-count', 'uint32', 1),
        vertexCount: view('vertex-count', 'uint32', 1)
      },
      status: {
        count: view('count', 'uint32', 1),
        overflow: view('overflow', 'uint32', 1),
        polylineOverflow: view('polyline-overflow', 'uint32', 1)
      }
    }
  });
  const stitchedNodes = stitched.getCommandNodes(graph).map(node => node.id);
  // ceil(log2(40)) + 1 = 7 rounds, rounded up to 8 because rounds are gated in pairs.
  expect(stitchedNodes.filter(id => /^stitched-jump-\d+$/.test(id))).toHaveLength(8);
  expect(stitchedNodes.filter(id => /^stitched-jump-gate-\d+$/.test(id))).toHaveLength(4);
  expect(new Set(stitchedNodes).size).toBe(stitchedNodes.length);

  expect(
    () =>
      new GPUIsolines({
        ...base,
        grid: makeGPURasterMetadata({width: 1, height: 4}),
        output: output()
      })
  ).toThrow(/width/);
  expect(
    () => new GPUIsolines({...base, values: view('small', 'float32', 5), output: output()})
  ).toThrow(/values/);
  expect(
    () =>
      new GPUIsolines({...base, output: {...output(), segmentLevels: view('short', 'uint32', 3)}})
  ).toThrow(/segmentLevels/);
  expect(() => new GPUIsolines({...base, noDataValue: NaN, output: output()})).toThrow(
    /noDataValue/
  );
  device.destroy();
});

it('oracle: every interior segment end has exactly one successor, borders and nodata none', () => {
  for (let seed = 1; seed <= 12; seed++) {
    const width = 9 + (seed % 5) * 4;
    const height = 7 + (seed % 3) * 5;
    // Integer levels (2, 5) hit integer samples exactly: zero-length bridges must keep chains.
    const scene = createRandomScene(seed, width, height);
    const segments = computeIsolinesOnCPU(scene, scene.levels.length);
    expect(segments.length).toBeGreaterThan(0);
    const {nextOf, previousOf, polylines} = stitchIsolinesOnCPU(segments);
    const horizontalCount = height * (width - 1);
    const cellColumns = width - 1;
    // Cells that own an edge, by brute force.
    const cellsOfEdge = new Map<number, number[]>();
    for (let cell = 0; cell < cellColumns * (height - 1); cell++) {
      const cx = cell % cellColumns;
      const cy = Math.floor(cell / cellColumns);
      for (const edge of [
        cy * cellColumns + cx,
        (cy + 1) * cellColumns + cx,
        horizontalCount + cy * width + cx,
        horizontalCount + cy * width + cx + 1
      ]) {
        cellsOfEdge.set(edge, [...(cellsOfEdge.get(edge) ?? []), cell]);
      }
    }
    const emittingCells = new Set(segments.map(segment => segment.cell));
    const hasNodata = (cell: number) => {
      const cx = cell % cellColumns;
      const cy = Math.floor(cell / cellColumns);
      return [
        cy * width + cx,
        cy * width + cx + 1,
        (cy + 1) * width + cx,
        (cy + 1) * width + cx + 1
      ].some(sample => !Number.isFinite(scene.values[sample]));
    };
    let interior = 0;
    segments.forEach((segment, index) => {
      const cells = cellsOfEdge.get(segment.endEdge)!;
      const neighbour = cells.find(cell => cell !== segment.cell);
      if (neighbour === undefined || hasNodata(neighbour)) {
        expect(nextOf[index]).toBe(-1);
      } else {
        interior++;
        expect(
          nextOf[index],
          `segment ${index} (seed ${seed}) has a successor`
        ).toBeGreaterThanOrEqual(0);
        expect(segments[nextOf[index]].cell).toBe(neighbour);
        expect(segments[nextOf[index]].level).toBe(segment.level);
        expect(previousOf[nextOf[index]]).toBe(index);
      }
    });
    expect(interior).toBeGreaterThan(0);
    expect(emittingCells.size).toBeGreaterThan(0);
    // Every segment is in exactly one polyline.
    const seen = polylines.flatMap(polyline => polyline.segmentIndices).sort((a, b) => a - b);
    expect(seen).toEqual(segments.map((_, index) => index));
  }
});

it('oracle: both saddle types resolve by the centre and orient high to the left', () => {
  // 2x2 cell: edge ids e0 bottom 0, e2 top 1, e3 left 2, e1 right 3.
  const makeCell = (v0: number, v1: number, v2: number, v3: number): IsolinesScene => ({
    width: 2,
    height: 2,
    // Row-major: row 0 holds v0, v1; row 1 holds v3, v2.
    values: Float32Array.from([v0, v1, v3, v2]),
    levels: [1],
    extent: [0, 0, 2, 2]
  });
  const edges = (scene: IsolinesScene) =>
    computeIsolinesOnCPU(scene, 1).map(segment => [segment.startEdge, segment.endEdge]);
  // Case 5 (v0, v2 high). Centre above: joined, the segments cut off the low corners v1 and v3.
  expect(edges(makeCell(3, 0, 3, 0))).toEqual([
    [0, 3],
    [1, 2]
  ]);
  // Centre below: separated, the segments cut off the high corners v0 and v2.
  expect(edges(makeCell(2, -1, 2, -1))).toEqual([
    [0, 2],
    [1, 3]
  ]);
  // Case 10 (v1, v3 high).
  expect(edges(makeCell(0, 3, 0, 3))).toEqual([
    [3, 1],
    [2, 0]
  ]);
  expect(edges(makeCell(-1, 2, -1, 2))).toEqual([
    [3, 0],
    [2, 1]
  ]);
  // Only v0 high: e0 -> e3, high (v0) on the left of the direction of travel.
  const single = computeIsolinesOnCPU(makeCell(2, 0, 0, 0), 1);
  expect(single).toHaveLength(1);
  const [segment] = single;
  expect([segment.startEdge, segment.endEdge]).toEqual([0, 2]);
  const direction = [segment.p1[0] - segment.p0[0], segment.p1[1] - segment.p0[1]];
  const towardHigh = [0.5 - segment.p0[0], 0.5 - segment.p0[1]];
  // Sample v0 sits at (0.5, 0.5) in world coordinates: cross(direction, toward) > 0 means left.
  expect(direction[0] * towardHigh[1] - direction[1] * towardHigh[0]).toBeGreaterThan(0);
  // Level equal to every corner: all high, nothing emitted (plateau).
  expect(edges(makeCell(1, 1, 1, 1))).toEqual([]);
});

it('oracle: non-finite corners are nodata, so no NaN vertices or flipped saddles', () => {
  const makeScene = (corners: number[]): IsolinesScene => ({
    width: 2,
    height: 2,
    // Row-major: row 0 holds v0, v1; row 1 holds v3, v2.
    values: Float32Array.from([corners[0], corners[1], corners[3], corners[2]]),
    levels: [1],
    extent: [0, 0, 2, 2]
  });
  for (const corners of [
    [Infinity, 0, Infinity, 0],
    [Infinity, 0, -Infinity, 0],
    [2, -Infinity, 2, 0],
    [3, 0, 3, Infinity]
  ]) {
    expect(computeIsolinesOnCPU(makeScene(corners), 1)).toEqual([]);
  }
  // A finite cell next to an infinite one still emits finite vertices.
  const scene: IsolinesScene = {
    width: 3,
    height: 2,
    values: Float32Array.from([0, 2, Infinity, 0, 2, -Infinity]),
    levels: [1],
    extent: [0, 0, 3, 2]
  };
  const segments = computeIsolinesOnCPU(scene, 1);
  expect(segments).toHaveLength(1);
  expect(segments[0].cell).toBe(0);
  expect(segments[0].p0.concat(segments[0].p1).every(Number.isFinite)).toBe(true);
});

it('oracle: zero-length bridges keep chains and polyline vertices have no duplicates', () => {
  // Integer raster, integer levels: corners sit exactly on levels.
  const width = 6;
  const height = 5;
  const values = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      values[row * width + column] = (column * 2 + row * 3) % 5;
    }
  }
  const scene: IsolinesScene = {width, height, values, levels: [0, 1, 2, 3], extent: [0, 0, 6, 5]};
  const segments = computeIsolinesOnCPU(scene, 4);
  const degenerate = segments.filter(
    segment => segment.p0[0] === segment.p1[0] && segment.p0[1] === segment.p1[1]
  );
  expect(degenerate.length).toBeGreaterThan(0);
  const {polylines} = stitchIsolinesOnCPU(segments);
  const {vertices, offsets} = getPolylineVertices(polylines, segments);
  for (let polyline = 0; polyline < polylines.length; polyline++) {
    for (let vertex = offsets[polyline] + 1; vertex < offsets[polyline + 1]; vertex++) {
      const same =
        vertices[2 * vertex] === vertices[2 * vertex - 2] &&
        vertices[2 * vertex + 1] === vertices[2 * vertex - 1];
      expect(same, `polyline ${polyline} vertex ${vertex}`).toBe(false);
    }
  }
  // A bridge joins its neighbours into one polyline: more segments than polyline vertices - 1.
  expect(vertices.length / 2).toBeLessThan(segments.length + polylines.length);
});
