// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUIsobandRings} from '../../../src/gpu-raster/isolines/gpu-isoband-rings';
import {getGPUIsobandsParameterValues} from '../../../src/gpu-raster/isolines/isobands-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {buildIsobandTrianglesOnCPU, type IsobandsScene} from './isobands-oracle';

const WIDTH = 24;
const HEIGHT = 20;
const MAXIMUM_BREAK_COUNT = 6;
const EDGE_CAPACITY = 6000;
const RING_CAPACITY = 400;
const VERTEX_CAPACITY = 6000;
const NONE = 0xffffffff;

/** A radial bump with a crater (so bands have holes), two peaks and one nodata cell. */
function createScene(integerSamples: boolean): IsobandsScene {
  const values = new Float32Array(WIDTH * HEIGHT);
  for (let row = 0; row < HEIGHT; row++) {
    for (let column = 0; column < WIDTH; column++) {
      const dx = column - 8.3;
      const dy = row - 9.7;
      const radius = Math.hypot(dx, dy);
      const ridge = 5 * Math.exp(-((radius - 5) ** 2) / 6);
      const peak = 4 * Math.exp(-((column - 18.4) ** 2 + (row - 5.2) ** 2) / 9);
      const value = ridge + peak + 0.1 * Math.sin(column * 0.7);
      // Integer samples equal the integer breaks exactly, the worst case for crossings at corners.
      values[row * WIDTH + column] = Math.fround(integerSamples ? Math.round(value) : value);
    }
  }
  const validity = new Uint32Array(WIDTH * HEIGHT).fill(1);
  validity[14 * WIDTH + 17] = 0;
  if (integerSamples) {
    // Scattered nodata samples make band regions touch themselves at single vertices.
    for (let sample = 0; sample < validity.length; sample++) {
      if ((sample * 2654435761) % 11 === 0) validity[sample] = 0;
    }
  }
  return {
    width: WIDTH,
    height: HEIGHT,
    values,
    validity,
    noDataValue: undefined,
    breaks: integerSamples
      ? Float32Array.of(1, 2, 3, 4, 5, 0)
      : Float32Array.of(0.5, 1.5, 2.5, 3.5, 4.5, 0)
  } as IsobandsScene;
}

function quantize(x: number, y: number): string {
  return `${Math.round(x * 100)},${Math.round(y * 100)}`;
}

for (const integerSamples of [false, true]) {
  it(`GPUIsobandRings closes band boundary edges into rings that match the isoband triangles (${integerSamples ? 'integer samples' : 'smooth samples'})`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const scene = createScene(integerSamples);
    const sampleCount = WIDTH * HEIGHT;
    const parameterBuffer = new GPUParameterBuffer(device, {
      id: 'isoband-rings-parameters',
      format: 'float32',
      length: 12
    });
    const valuesBuffer = createInputBuffer(device, scene.values);
    const validityBuffer = createInputBuffer(device, scene.validity!);
    const breaksBuffer = createInputBuffer(device, scene.breaks);
    const outputs = {
      ringOffsets: createOutputBuffer(device, RING_CAPACITY + 1),
      positions: createOutputBuffer(device, VERTEX_CAPACITY * 2),
      ringAreas: createOutputBuffer(device, RING_CAPACITY),
      ringIsHole: createOutputBuffer(device, RING_CAPACITY),
      ringShells: createOutputBuffer(device, RING_CAPACITY),
      ringGroups: createOutputBuffer(device, RING_CAPACITY),
      count: createOutputBuffer(device, 1),
      overflow: createOutputBuffer(device, 1),
      openSegmentCount: createOutputBuffer(device, 1),
      edgeCount: createOutputBuffer(device, 1),
      edgeOverflow: createOutputBuffer(device, 1),
      polygonPositions: createOutputBuffer(device, VERTEX_CAPACITY * 2),
      polygonRingOffsets: createOutputBuffer(device, RING_CAPACITY + 1),
      polygonOffsets: createOutputBuffer(device, RING_CAPACITY + 1),
      featureOffsets: createOutputBuffer(device, RING_CAPACITY + 1),
      sourceIds: createOutputBuffer(device, RING_CAPACITY)
    };
    const graph = new GPUCommandGraph(device, {id: 'isoband-rings-graph'});
    graph.add(
      new GPUIsobandRings({
        id: 'rings',
        width: WIDTH,
        height: HEIGHT,
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', sampleCount),
        validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', sampleCount),
        breaks: importGraphBuffer(graph, 'breaks', breaksBuffer, 'float32', MAXIMUM_BREAK_COUNT),
        parameters: parameterBuffer.importToGraph(graph),
        edgeCapacity: EDGE_CAPACITY,
        output: {
          ringOffsets: importGraphBuffer(
            graph,
            'ro',
            outputs.ringOffsets,
            'uint32',
            RING_CAPACITY + 1
          ),
          positions: importGraphBuffer(graph, 'p', outputs.positions, 'float32x2', VERTEX_CAPACITY),
          ringAreas: importGraphBuffer(graph, 'ra', outputs.ringAreas, 'float32', RING_CAPACITY),
          ringIsHole: importGraphBuffer(graph, 'rh', outputs.ringIsHole, 'uint32', RING_CAPACITY),
          ringShells: importGraphBuffer(graph, 'rs', outputs.ringShells, 'uint32', RING_CAPACITY),
          ringGroups: importGraphBuffer(graph, 'rg', outputs.ringGroups, 'uint32', RING_CAPACITY),
          count: importGraphBuffer(graph, 'c', outputs.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'o', outputs.overflow, 'uint32', 1),
          openSegmentCount: importGraphBuffer(graph, 'os', outputs.openSegmentCount, 'uint32', 1),
          edgeCount: importGraphBuffer(graph, 'ec', outputs.edgeCount, 'uint32', 1),
          edgeOverflow: importGraphBuffer(graph, 'eo', outputs.edgeOverflow, 'uint32', 1),
          polygons: {
            kind: 'polygons',
            positions: importGraphBuffer(
              graph,
              'pp',
              outputs.polygonPositions,
              'float32x2',
              VERTEX_CAPACITY
            ),
            ringOffsets: importGraphBuffer(
              graph,
              'pro',
              outputs.polygonRingOffsets,
              'uint32',
              RING_CAPACITY + 1
            ),
            polygonOffsets: importGraphBuffer(
              graph,
              'ppo',
              outputs.polygonOffsets,
              'uint32',
              RING_CAPACITY + 1
            ),
            featureOffsets: importGraphBuffer(
              graph,
              'pfo',
              outputs.featureOffsets,
              'uint32',
              RING_CAPACITY + 1
            ),
            sourceIds: importGraphBuffer(graph, 'pg', outputs.sourceIds, 'uint32', RING_CAPACITY)
          }
        }
      })
    );
    const compiled = graph.compile();
    const frames = [
      {breakCount: 5, firstBand: 0, lastBand: 5},
      {breakCount: 3, firstBand: 0, lastBand: 3},
      {breakCount: 5, firstBand: 2, lastBand: 4}
    ];
    for (const [frameIndex, frame] of frames.entries()) {
      const parameters = getGPUIsobandsParameterValues({
        ...frame,
        width: WIDTH,
        height: HEIGHT,
        extent: [10, -5, 58, 35]
      });
      parameterBuffer.write(parameters);
      submitGraph(device, compiled, undefined);
      const label = `frame ${frameIndex}`;
      const expected = buildIsobandTrianglesOnCPU(scene, parameters, MAXIMUM_BREAK_COUNT);

      // Oracle edges: triangle edges that no triangle of the same band traverses backwards.
      const edgeKeys = new Map<string, number>();
      const edgeKey = (band: number, ax: number, ay: number, bx: number, by: number) =>
        `${band}:${quantize(ax, ay)}>${quantize(bx, by)}`;
      for (let triangle = 0; triangle < expected.bands.length; triangle++) {
        const band = expected.bands[triangle];
        for (let corner = 0; corner < 3; corner++) {
          const next = (corner + 1) % 3;
          const ax = expected.triangles[triangle * 6 + corner * 2];
          const ay = expected.triangles[triangle * 6 + corner * 2 + 1];
          const bx = expected.triangles[triangle * 6 + next * 2];
          const by = expected.triangles[triangle * 6 + next * 2 + 1];
          if (quantize(ax, ay) === quantize(bx, by)) continue;
          const key = edgeKey(band, ax, ay, bx, by);
          const reverse = edgeKey(band, bx, by, ax, ay);
          if (edgeKeys.get(reverse)) {
            edgeKeys.set(reverse, edgeKeys.get(reverse)! - 1);
          } else {
            edgeKeys.set(key, (edgeKeys.get(key) ?? 0) + 1);
          }
        }
      }
      const expectedEdgeCount = [...edgeKeys.values()].reduce((sum, value) => sum + value, 0);
      expect(await readUint32(outputs.edgeOverflow, 1), `${label} edge overflow`).toEqual([0]);
      if (!integerSamples) {
        expect(await readUint32(outputs.edgeCount, 1), `${label} edge count`).toEqual([
          expectedEdgeCount
        ]);
      }

      expect(await readUint32(outputs.overflow, 1), `${label} overflow`).toEqual([0]);
      expect(await readUint32(outputs.openSegmentCount, 1), `${label} open segments`).toEqual([0]);
      const [ringCount] = await readUint32(outputs.count, 1);
      expect(ringCount, `${label} rings`).toBeGreaterThan(0);
      const areas = await readFloat32(outputs.ringAreas, ringCount);
      const bands = await readUint32(outputs.ringGroups, ringCount);
      const isHole = await readUint32(outputs.ringIsHole, ringCount);
      const shells = await readUint32(outputs.ringShells, ringCount);
      const ringOffsets = await readUint32(outputs.ringOffsets, ringCount + 1);

      // Per-band signed ring area (shells positive, holes negative) equals the triangle area.
      const ringAreaByBand = new Map<number, number>();
      for (let ring = 0; ring < ringCount; ring++) {
        ringAreaByBand.set(bands[ring], (ringAreaByBand.get(bands[ring]) ?? 0) + areas[ring]);
        expect(isHole[ring] === 1, `${label} ring ${ring} hole sign`).toBe(areas[ring] < 0);
        if (isHole[ring]) {
          expect(shells[ring], `${label} hole ${ring} has a shell`).not.toBe(NONE);
          expect(bands[shells[ring]], `${label} hole ${ring} shell band`).toBe(bands[ring]);
        }
      }
      const triangleAreaByBand = new Map<number, number>();
      for (let triangle = 0; triangle < expected.bands.length; triangle++) {
        const t = expected.triangles.slice(triangle * 6, triangle * 6 + 6);
        const area = ((t[2] - t[0]) * (t[5] - t[1]) - (t[4] - t[0]) * (t[3] - t[1])) / 2;
        const band = expected.bands[triangle];
        triangleAreaByBand.set(band, (triangleAreaByBand.get(band) ?? 0) + area);
      }
      // Bands whose triangles all have zero area (a ridge on samples equal to a break) have no ring.
      for (const [band, area] of triangleAreaByBand) {
        if (Math.abs(area) < 1e-6) triangleAreaByBand.delete(band);
      }
      expect([...ringAreaByBand.keys()].sort(), `${label} bands`).toEqual(
        [...triangleAreaByBand.keys()].sort()
      );
      for (const [band, area] of triangleAreaByBand) {
        expect(ringAreaByBand.get(band)!, `${label} band ${band} area`).toBeCloseTo(area, -0.5);
        expect(
          Math.abs(ringAreaByBand.get(band)! - area) / area,
          `${label} band ${band}`
        ).toBeLessThan(2e-3);
      }
      // Rings are closed (the assembly repeats the first vertex) and non-trivial.
      for (let ring = 0; ring < ringCount; ring++) {
        expect(
          ringOffsets[ring + 1] - ringOffsets[ring],
          `${label} ring ${ring} size`
        ).toBeGreaterThanOrEqual(4);
      }
      if (frameIndex === 0) {
        if (!integerSamples) {
          expect(
            isHole.some(value => value === 1),
            'the crater produces holes'
          ).toBe(true);
        }
        const polygonFeatureOffsets = await readUint32(outputs.featureOffsets, RING_CAPACITY + 1);
        const polygonCount = polygonFeatureOffsets[RING_CAPACITY];
        expect(polygonFeatureOffsets[0]).toBe(0);
        expect(polygonFeatureOffsets.slice(0, polygonCount + 1)).toEqual(
          Array.from({length: polygonCount + 1}, (_, index) => index)
        );
        expect(polygonCount).toBe(isHole.filter(value => value === 0).length);
        const sourceIds = await readUint32(outputs.sourceIds, RING_CAPACITY);
        expect(sourceIds.slice(0, polygonCount).sort()).toEqual(
          bands.filter((_, ring) => isHole[ring] === 0).sort()
        );
        expect(sourceIds.slice(polygonCount).every(value => value === NONE)).toBe(true);
      }
    }
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of [valuesBuffer, validityBuffer, breaksBuffer, ...Object.values(outputs)]) {
      buffer.destroy();
    }
  });
}
