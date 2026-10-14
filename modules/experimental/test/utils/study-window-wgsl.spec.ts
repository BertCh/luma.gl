// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  getPolygonEdgeSlabBindings,
  getPolygonEdgeSlabNodes
} from '../../src/utils/polygon-edge-slabs';
import {
  getStudyWindowBinding,
  getStudyWindowPackedLayout,
  getStudyWindowWGSL,
  type StudyWindowKernel
} from '../../src/utils/study-window-wgsl';
import {createWGSLKernelNode} from '../../src/utils/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {EXACT_ORIENTATION_WGSL} from '../../src/utils/exact-orientation-wgsl';
import {EXACT_GEOMETRY_WGSL} from '../../src/utils/exact-geometry-wgsl';
import {
  ORACLE_RECTANGLE,
  ORACLE_L_SHAPE,
  ORACLE_SQUARE_WITH_HOLE,
  ORACLE_TWO_ISLANDS,
  ORACLE_CLOCKWISE_SHELL_CCW_HOLE,
  getOracleOrientedEdges,
  getOracleWindowArea,
  getOracleWindowPerimeter,
  getOracleWindowBounds,
  isOracleInsideWindow,
  getOracleBoundaryDistance,
  getOracleCircleFractionFan,
  getOracleKernelMassScheme,
  getOracleRectangleIsotropicFraction,
  type OracleStudyWindow,
  type OraclePoint
} from '../analysis-core/study-window-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';

// Subdivision forces multiple slabs even for the four-edge rectangle. Oriented edge rows still
// retain the source vertex index; CW-shell normalization must not change that lookup.
function subdivideWindow(window: OracleStudyWindow): OracleStudyWindow {
  const positions: number[] = [];
  const ringOffsets = [0];
  for (let ring = 0; ring + 1 < window.ringOffsets.length; ring++) {
    const begin = window.ringOffsets[ring];
    const end = window.ringOffsets[ring + 1];
    for (let vertex = begin; vertex < end; vertex++) {
      const next = vertex + 1 === end ? begin : vertex + 1;
      for (let step = 0; step < 4; step++) {
        for (let axis = 0; axis < 2; axis++) {
          positions.push(
            window.positions[2 * vertex + axis] +
              (step / 4) * (window.positions[2 * next + axis] - window.positions[2 * vertex + axis])
          );
        }
      }
    }
    ringOffsets.push(positions.length / 2);
  }
  return {
    ...window,
    positions: new Float64Array(positions),
    ringOffsets: new Uint32Array(ringOffsets)
  };
}

const QUERY_POINTS: readonly OraclePoint[] = [
  [350, 550],
  [600, 600],
  [900, 500],
  [1500, 1200],
  [1000, 1100],
  [1250, 500],
  [0, 0],
  [700, 700],
  [1000, 0],
  [2000, 1000],
  [-100, 500],
  [0.001, 500]
];
const QUERY_RADII = [25, 300, 800, 1700, 4000];
const OUTPUT_STRIDE = 12;

async function evaluateWindow(
  device: Device,
  window: OracleStudyWindow,
  kernel: StudyWindowKernel,
  slabbed: boolean
): Promise<number[]> {
  const graph = new GPUCommandGraph(device, {id: `window-${kernel}-${slabbed}`});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const vertexCount = window.positions.length / 2;
  const positions = importGraphBuffer(
    graph,
    'positions',
    input(new Float32Array(window.positions)),
    'float32x2',
    vertexCount
  );
  const featureOffsets =
    window.featureOffsets ?? new Uint32Array([0, window.polygonOffsets.length - 1]);
  const {nodes, slabs} = getPolygonEdgeSlabNodes(graph, {
    id: 'slabs',
    operation: 'StudyWindowTest',
    owner: 'feature',
    positions,
    featureOffsets: importGraphBuffer(
      graph,
      'features',
      input(featureOffsets),
      'uint32',
      featureOffsets.length
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      'polygons',
      input(window.polygonOffsets),
      'uint32',
      window.polygonOffsets.length
    ),
    ringOffsets: importGraphBuffer(
      graph,
      'rings',
      input(window.ringOffsets),
      'uint32',
      window.ringOffsets.length
    ),
    minimumVertexCount: 1,
    entryCapacity: 8 * vertexCount
  });
  const layout = getStudyWindowPackedLayout({
    vertexCount,
    slabOffsetCount: slabs.slabOffsets.length,
    slabEntryCount: slabs.slabEntries.length,
    sampleCellCapacity: 20
  });
  // Exercise a nonzero view offset, in addition to every nonzero packed-table offset.
  const prefixLength = 64;
  const packedWords = new Uint32Array(prefixLength + layout.length);
  const packedFloats = new Float32Array(packedWords.buffer);
  const bounds = getOracleWindowBounds(window);
  packedFloats.set(
    [...bounds, getOracleWindowArea(window), getOracleWindowPerimeter(window), vertexCount, 0],
    prefixLength + layout.summary
  );
  getOracleOrientedEdges(window).forEach((edge, vertex) =>
    packedFloats.set(edge, prefixLength + layout.edges + vertex * 4)
  );
  packedWords[prefixLength + layout.slabParams + 4] = 0;
  packedWords[prefixLength + layout.slabParams + 5] = vertexCount;
  for (let ring = 0; ring + 1 < window.ringOffsets.length; ring++) {
    const begin = window.ringOffsets[ring];
    const end = window.ringOffsets[ring + 1];
    for (let vertex = begin; vertex < end; vertex++) {
      packedWords[prefixLength + layout.slabEntries + vertex * 2] = vertex;
      packedWords[prefixLength + layout.slabEntries + vertex * 2 + 1] =
        vertex + 1 === end ? begin : vertex + 1;
    }
  }
  const packedBuffer = input(packedWords);
  const packedHandle = graph.importBuffer(
    {id: 'packed', byteLength: packedBuffer.byteLength, usage: packedBuffer.usage},
    packedBuffer
  );
  const packed = graph.createDataView(packedHandle, {
    format: 'uint32',
    length: layout.length,
    byteOffset: prefixLength * 4
  });
  if (slabbed) {
    for (const node of nodes) {
      graph.add(node);
    }
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'pack-slabs',
        operation: 'StudyWindowTest',
        invocationCount: Math.max(6, slabs.slabOffsets.length, 2 * slabs.slabEntries.length),
        bindings: [
          ...getPolygonEdgeSlabBindings(slabs),
          {name: 'packed', view: packed, type: 'u32', access: 'read_write'}
        ],
        body: `if (index < 6u) { packed[packedOffset + ${layout.slabParams}u + index] = edgeSlabParams[edgeSlabParamsOffset + index]; }
        if (index < ${slabs.slabOffsets.length}u) { packed[packedOffset + ${layout.slabOffsets}u + index] = edgeSlabOffsets[edgeSlabOffsetsOffset + index]; }
        if (index < ${slabs.slabEntries.length * 2}u) { packed[packedOffset + ${layout.slabEntries}u + index] = edgeSlabEntries[edgeSlabEntriesOffset + index]; }`
      })
    );
  }
  const queryValues: number[] = [];
  for (const point of QUERY_POINTS) {
    for (const radius of QUERY_RADII) {
      queryValues.push(...point, radius, radius / 2);
    }
  }
  queryValues.push(NaN, 500, 100, 50, Infinity, 500, 100, 50);
  const queryCount = queryValues.length / 4;
  const queryInput = importGraphBuffer(
    graph,
    'queries',
    input(new Float32Array(queryValues)),
    'float32',
    queryValues.length
  );
  const outputBuffer = createOutputBuffer(device, OUTPUT_STRIDE * queryCount);
  buffers.push(outputBuffer);
  const output = importGraphBuffer(
    graph,
    'output',
    outputBuffer,
    'float32',
    OUTPUT_STRIDE * queryCount
  );
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'query-window',
      operation: 'StudyWindowTest',
      invocationCount: queryCount,
      bindings: [
        getStudyWindowBinding({packed}),
        {name: 'queries', view: queryInput, type: 'f32', access: 'read'},
        {name: 'output', view: output, type: 'f32', access: 'read_write'}
      ],
      declarations: `${EXACT_ORIENTATION_WGSL}${EXACT_GEOMETRY_WGSL}${getStudyWindowWGSL({layout, vertexCount, entryCapacity: slabs.entryCapacity, kernel, sampleGrid: [5, 4]})}`,
      body: `let query = queriesOffset + index * 4u;
      let point = vec2f(queries[query], queries[query + 1u]);
      let radius = queries[query + 2u];
      let bandwidth = queries[query + 3u];
      let base = outputOffset + index * ${OUTPUT_STRIDE}u;
      output[base] = select(0.0, 1.0, isInsideStudyWindow(point));
      output[base + 1u] = getStudyWindowBoundaryDistance(point, radius);
      output[base + 2u] = getStudyWindowCircleFraction(point, radius);
      output[base + 3u] = getStudyWindowKernelMass(point, bandwidth);
      output[base + 4u] = select(0.0, 1.0, isStudyWindowUsable());
      output[base + 5u] = studyWindowSummaryValue(4u);
      let cell = getStudyWindowSampleCell((index % 20u) | 0x80000000u);
      output[base + 6u] = cell.x;
      output[base + 7u] = cell.y;
      output[base + 8u] = cell.z;
      output[base + 9u] = cell.w;
      output[base + 10u] = f32(studyWindow[studyWindowOffset + ${layout.slabParams}u]);
      output[base + 11u] = studyWindowSummaryValue(5u);`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return await readFloat32(outputBuffer, OUTPUT_STRIDE * queryCount);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

for (const kernel of ['gaussian', 'quartic', 'epanechnikov'] as const) {
  it(`packed study window matches binary64 and walked/slabbed agree (${kernel})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const fixture of [
      ORACLE_RECTANGLE,
      ORACLE_L_SHAPE,
      ORACLE_SQUARE_WITH_HOLE,
      ORACLE_TWO_ISLANDS,
      ORACLE_CLOCKWISE_SHELL_CCW_HOLE
    ]) {
      const window = subdivideWindow(fixture);
      const walked = await evaluateWindow(device, window, kernel, false);
      const slabbed = await evaluateWindow(device, window, kernel, true);
      const bounds = getOracleWindowBounds(window);
      const sizeX = Math.fround(Math.fround(bounds[2] - bounds[0]) / 5);
      const sizeY = Math.fround(Math.fround(bounds[3] - bounds[1]) / 4);
      let query = 0;
      for (const originalPoint of QUERY_POINTS) {
        const point: OraclePoint = [Math.fround(originalPoint[0]), Math.fround(originalPoint[1])];
        for (const radius of QUERY_RADII) {
          const base = query * OUTPUT_STRIDE;
          for (const result of [walked, slabbed]) {
            expect(result[base]).toBe(Number(isOracleInsideWindow(window, point)));
            const distance = getOracleBoundaryDistance(window, point, radius);
            if (distance === Infinity) {
              expect(result[base + 1]).toBe(Infinity);
            } else {
              expect(Math.abs(result[base + 1] - distance)).toBeLessThanOrEqual(
                1e-4 * Math.max(distance, 1e-4)
              );
            }
            expect(
              Math.abs(result[base + 2] - getOracleCircleFractionFan(window, point, radius))
            ).toBeLessThanOrEqual(1e-5);
            if (
              fixture === ORACLE_RECTANGLE &&
              point[0] >= 0 &&
              point[0] <= 2000 &&
              point[1] >= 0 &&
              point[1] <= 1000
            ) {
              expect(
                Math.abs(
                  result[base + 2] - getOracleRectangleIsotropicFraction(bounds, point, radius)
                )
              ).toBeLessThanOrEqual(1e-5);
            }
            expect(
              Math.abs(
                result[base + 3] - getOracleKernelMassScheme(window, point, radius / 2, kernel)
              )
            ).toBeLessThanOrEqual(2e-5);
            expect(result[base + 4]).toBe(1);
            expect(result[base + 5]).toBe(getOracleWindowArea(window));
            const cell = query % 20;
            expect(result.slice(base + 6, base + 10)).toEqual([
              Math.fround(bounds[0] + Math.fround(sizeX * (cell % 5))),
              Math.fround(bounds[1] + Math.fround(sizeY * Math.floor(cell / 5))),
              sizeX,
              sizeY
            ]);
            expect(result[base + 11]).toBe(getOracleWindowPerimeter(window));
          }
          expect(walked[base + 10]).toBe(0);
          expect(slabbed[base + 10]).toBeGreaterThan(1);
          for (let slot = 0; slot < OUTPUT_STRIDE; slot++) {
            if (slot === 10) {
              continue;
            }
            if (walked[base + slot] === Infinity) {
              expect(slabbed[base + slot]).toBe(Infinity);
            } else {
              expect(Math.abs(walked[base + slot] - slabbed[base + slot])).toBeLessThanOrEqual(
                1e-6
              );
            }
          }
          query++;
        }
      }
      for (const result of [walked, slabbed]) {
        for (let invalid = 0; invalid < 2; invalid++) {
          const base = (query + invalid) * OUTPUT_STRIDE;
          expect(result.slice(base, base + 4)).toEqual([0, Infinity, 0, 0]);
        }
      }
    }
  }, 120000);
}
