// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {cellsToMultiPolygon, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochroneParameterValues,
  type GPUNetworkIsochroneSettings
} from '../../../src/gpu-network/network-isochrones';
import {assembleRingsOnCPU} from '../../gpu-spatial-analysis/ring-assembly/ring-assembly-oracle';
import {
  bigIntToH3,
  h3ToBigInt,
  joinCellKey,
  quadbinPointToCell
} from '../../gpu-spatial-analysis/cell-aggregation/cell-aggregation-oracle';
import {
  matchSegments,
  outlineH3OnCPU,
  outlineQuadbinOnCPU
} from '../../gpu-spatial-analysis/cell-set-outline/cell-set-outline-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCSR,
  createGridNetwork,
  dijkstra
} from '../network-reachability/network-reachability-oracle';
import {
  getBandAreaBounds,
  getTriangleAreasPerBand,
  splatIsochroneRasterOnCPU
} from './network-isochrones-oracle';

const GRID = 9;
const NODE_COUNT = GRID * GRID;
const BREAKS = [8, 16, 28];
const WIDTH = 72;
const HEIGHT = 72;

/** Grid network with node `(x, y)` at `origin + spacing * (x, y)`. */
function createScene(origin: [number, number], spacing: number) {
  const csr = buildCSR(NODE_COUNT, createGridNetwork(5, GRID, GRID));
  const positions = new Float32Array(2 * NODE_COUNT);
  for (let node = 0; node < NODE_COUNT; node++) {
    positions[2 * node] = origin[0] + spacing * (node % GRID);
    positions[2 * node + 1] = origin[1] + spacing * Math.floor(node / GRID);
  }
  return {csr, positions};
}

type Run = {
  buffers: Buffer[];
  parameters: GPUParameterBuffer[];
};

function destroyRun(run: Run) {
  for (const parameter of run.parameters) parameter.destroy();
  for (const buffer of run.buffers) buffer.destroy();
}

async function runRaster(
  device: Device,
  options: {
    mode: 'min' | 'max';
    settings: GPUNetworkIsochroneSettings;
    maximumBufferPixels?: number;
    triangleCapacity?: number;
  }
) {
  const {csr, positions} = createScene([0, 0], 100);
  const run: Run = {buffers: [], parameters: []};
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    run.buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    run.buffers.push(buffer);
    return buffer;
  };
  const triangleCapacity = options.triangleCapacity ?? 40000;
  const out = {
    costs: output(NODE_COUNT),
    values: output(WIDTH * HEIGHT),
    triangles: output(6 * triangleCapacity),
    bands: output(triangleCapacity),
    count: output(1),
    overflow: output(1),
    total: output(1)
  };
  const graph = new GPUCommandGraph(device, {id: 'isochrones-raster'});
  const parameters = new GPUParameterBuffer(device, {
    id: 'isochrone-parameters',
    format: 'float32',
    length: 12,
    values: getGPUNetworkIsochroneParameterValues(options.settings)
  });
  const breaks = new GPUParameterBuffer(device, {
    id: 'isochrone-breaks',
    format: 'float32',
    length: BREAKS.length,
    values: Float32Array.from(BREAKS)
  });
  run.parameters.push(parameters, breaks);
  graph.add(
    new GPUNetworkIsochrones({
      offsets: importGraphBuffer(graph, 'offsets', input(csr.offsets), 'uint32', NODE_COUNT + 1),
      neighbors: importGraphBuffer(
        graph,
        'neighbors',
        input(csr.neighbors),
        'uint32',
        csr.neighbors.length
      ),
      weights: importGraphBuffer(
        graph,
        'weights',
        input(csr.weights),
        'float32',
        csr.weights.length
      ),
      nodePositions: importGraphBuffer(
        graph,
        'positions',
        input(positions),
        'float32x2',
        NODE_COUNT
      ),
      costs: importGraphBuffer(graph, 'costs', out.costs, 'float32', NODE_COUNT),
      sources: importGraphBuffer(graph, 'sources', input(Uint32Array.of(0)), 'uint32', 1),
      maxIterations: 128,
      breaks: breaks.importToGraph(graph),
      parameters: parameters.importToGraph(graph),
      raster: {
        width: WIDTH,
        height: HEIGHT,
        mode: options.mode,
        maximumBufferPixels: options.maximumBufferPixels,
        output: {
          values: importGraphBuffer(graph, 'values', out.values, 'float32', WIDTH * HEIGHT),
          triangles: importGraphBuffer(
            graph,
            'triangles',
            out.triangles,
            'float32x2',
            3 * triangleCapacity
          ),
          triangleBands: importGraphBuffer(graph, 'bands', out.bands, 'uint32', triangleCapacity),
          count: importGraphBuffer(graph, 'count', out.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', out.overflow, 'uint32', 1),
          requiredCount: importGraphBuffer(graph, 'total', out.total, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const costs = Float32Array.from(await readFloat32(out.costs, NODE_COUNT));
  const values = Float32Array.from(await readFloat32(out.values, WIDTH * HEIGHT));
  const [count] = await readUint32(out.count, 1);
  const [overflow] = await readUint32(out.overflow, 1);
  const triangles = (await readFloat32(out.triangles, 6 * Math.max(count, 1))).slice(0, 6 * count);
  const bands = (await readUint32(out.bands, Math.max(count, 1))).slice(0, count);
  compiled.destroy?.();
  destroyRun(run);
  return {csr, positions, costs, values, count, overflow, triangles, bands};
}

for (const mode of ['min', 'max'] as const) {
  it(`GPUNetworkIsochrones raster (${mode}) matches the CPU splat and the band areas`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const extent: [number, number, number, number] = [-100, -100, 900, 900];
    const settings: GPUNetworkIsochroneSettings = {
      breakCount: BREAKS.length,
      extent,
      bufferRadius: 30,
      walkCostPerUnit: 0.05
    };
    const result = await runRaster(device, {mode, settings});
    // Search costs equal Dijkstra over the same network.
    const expectedCosts = dijkstra(result.csr, NODE_COUNT, [{node: 0, cost: 0}]);
    for (let node = 0; node < NODE_COUNT; node++) {
      expect(result.costs[node]).toBe(expectedCosts[node]);
    }
    const expected = splatIsochroneRasterOnCPU(result.csr, result.positions, expectedCosts, {
      width: WIDTH,
      height: HEIGHT,
      extent,
      bufferRadius: 30,
      walkCostPerUnit: 0.05,
      mode,
      maximumBufferPixels: 6,
      maximumSamplesPerEdge: 64,
      unreachedCost: 1e30
    });
    let mismatched = 0;
    let reached = 0;
    for (let pixel = 0; pixel < expected.length; pixel++) {
      const isReached = expected[pixel] < 1e29;
      reached += isReached ? 1 : 0;
      const gpuReached = result.values[pixel] < 1e29;
      if (
        isReached !== gpuReached ||
        (isReached && Math.abs(result.values[pixel] - expected[pixel]) > 0.05)
      ) {
        mismatched++;
      }
    }
    expect(reached, 'reached pixels').toBeGreaterThan(WIDTH * HEIGHT * 0.2);
    // Pixels on the buffer circle may flip with f32 sample positions; the bulk must agree.
    expect(mismatched / expected.length, 'mismatched pixel fraction').toBeLessThan(0.01);

    // Isobands: triangle area per band lies between the all-corners and any-corner cell areas.
    expect(result.overflow).toBe(0);
    expect(result.count).toBeGreaterThan(100);
    const cellArea = ((extent[2] - extent[0]) / WIDTH) * ((extent[3] - extent[1]) / HEIGHT);
    const bounds = getBandAreaBounds(result.values, WIDTH, HEIGHT, BREAKS, cellArea);
    const areas = getTriangleAreasPerBand(
      Array.from(result.triangles),
      Array.from(result.bands),
      BREAKS.length + 1
    );
    for (let band = 0; band < BREAKS.length; band++) {
      if (mode === 'min') {
        expect(bounds.lower[band], `band ${band} has full cells`).toBeGreaterThan(0);
      }
      expect(areas[band], `band ${band} area above lower bound`).toBeGreaterThanOrEqual(
        bounds.lower[band] * 0.999
      );
      expect(areas[band], `band ${band} area below upper bound`).toBeLessThanOrEqual(
        bounds.upper[band] * 1.001
      );
    }
    // Max mode raises every pixel next to a costly edge, so only the total is guaranteed nonzero.
    expect(areas[1] + areas[2] + areas[0], 'covered area').toBeGreaterThan(cellArea * 20);
    // Default band window leaves the unreached band empty.
    expect(areas[BREAKS.length]).toBe(0);
    // Larger bands nest: more cost allowance covers more ground.
    expect(areas[0] + areas[1] + areas[2]).toBeGreaterThan(areas[0]);
  });
}

it('GPUNetworkIsochrones min is never above max and the buffer widens the area', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const extent: [number, number, number, number] = [-100, -100, 900, 900];
  const base = {breakCount: BREAKS.length, extent, walkCostPerUnit: 0.05};
  const min = await runRaster(device, {mode: 'min', settings: {...base, bufferRadius: 30}});
  const max = await runRaster(device, {mode: 'max', settings: {...base, bufferRadius: 30}});
  const wide = await runRaster(device, {mode: 'min', settings: {...base, bufferRadius: 60}});
  let strictlyLess = 0;
  for (let pixel = 0; pixel < min.values.length; pixel++) {
    expect(min.values[pixel]).toBeLessThanOrEqual(max.values[pixel]);
    strictlyLess += min.values[pixel] < max.values[pixel] ? 1 : 0;
  }
  expect(strictlyLess).toBeGreaterThan(50);
  const reachedPixels = (values: Float32Array) => values.filter(value => value < 1e29).length;
  expect(reachedPixels(wide.values)).toBeGreaterThan(reachedPixels(min.values));
});

type CellRun = {
  segments: {cell: bigint; a: [number, number]; b: [number, number]}[];
  count: number;
  overflow: number;
  rings: {
    count: number;
    overflow: number;
    open: number;
    offsets: number[];
    positions: number[];
    areas: number[];
    isHole: number[];
    shells: number[];
  };
  endpoints: number[];
  cellKeys: bigint[];
};

async function runCellOutline(
  device: Device,
  family: 'h3' | 'quadbin',
  resolution: number,
  cellCostLimit: number
): Promise<CellRun & {expected: ReturnType<typeof outlineQuadbinOnCPU>; reachedCells: number}> {
  const ringCapacity = 128;
  const vertexCapacity = 4096;
  const origin: [number, number] = [8.4, 47.1];
  const spacing = 0.01;
  const {csr, positions} = createScene(origin, spacing);
  const run: Run = {buffers: [], parameters: []};
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    run.buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    run.buffers.push(buffer);
    return buffer;
  };
  const capacity = 4096;
  const out = {
    costs: output(NODE_COUNT),
    rows: output(capacity),
    cells: output(2 * capacity),
    edges: output(capacity),
    endpoints: output(4 * capacity),
    count: output(1),
    overflow: output(1),
    ringOffsets: output(ringCapacity + 1),
    ringPositions: output(2 * vertexCapacity),
    ringAreas: output(ringCapacity),
    ringIsHole: output(ringCapacity),
    ringShells: output(ringCapacity),
    ringCount: output(1),
    ringOverflow: output(1),
    ringOpen: output(1)
  };
  const graph = new GPUCommandGraph(device, {id: 'isochrones-cells'});
  const parameters = new GPUParameterBuffer(device, {
    id: 'isochrone-cell-parameters',
    format: 'float32',
    length: 12,
    values: getGPUNetworkIsochroneParameterValues({
      breakCount: 1,
      extent: [8.3, 47, 8.6, 47.3],
      cellCostLimit
    })
  });
  const breaks = new GPUParameterBuffer(device, {
    id: 'isochrone-cell-breaks',
    format: 'float32',
    length: 1,
    values: Float32Array.of(10)
  });
  run.parameters.push(parameters, breaks);
  graph.add(
    new GPUNetworkIsochrones({
      offsets: importGraphBuffer(graph, 'offsets', input(csr.offsets), 'uint32', NODE_COUNT + 1),
      neighbors: importGraphBuffer(
        graph,
        'neighbors',
        input(csr.neighbors),
        'uint32',
        csr.neighbors.length
      ),
      weights: importGraphBuffer(
        graph,
        'weights',
        input(csr.weights),
        'float32',
        csr.weights.length
      ),
      nodePositions: importGraphBuffer(
        graph,
        'positions',
        input(positions),
        'float32x2',
        NODE_COUNT
      ),
      costs: importGraphBuffer(graph, 'costs', out.costs, 'float32', NODE_COUNT),
      sources: importGraphBuffer(graph, 'sources', input(Uint32Array.of(40)), 'uint32', 1),
      maxIterations: 128,
      breaks: breaks.importToGraph(graph),
      parameters: parameters.importToGraph(graph),
      cellOutline: {
        family,
        resolution,
        output: {
          rows: importGraphBuffer(graph, 'rows', out.rows, 'uint32', capacity),
          cells: importGraphBuffer(graph, 'out-cells', out.cells, 'uint32x2', capacity),
          edgeIndices: importGraphBuffer(graph, 'edges', out.edges, 'uint32', capacity),
          endpoints: importGraphBuffer(graph, 'endpoints', out.endpoints, 'float32x4', capacity),
          count: importGraphBuffer(graph, 'count', out.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', out.overflow, 'uint32', 1)
        },
        rings: {
          normalizeWinding: true,
          output: {
            ringOffsets: importGraphBuffer(
              graph,
              'ring-offsets',
              out.ringOffsets,
              'uint32',
              ringCapacity + 1
            ),
            positions: importGraphBuffer(
              graph,
              'ring-positions',
              out.ringPositions,
              'float32x2',
              vertexCapacity
            ),
            ringAreas: importGraphBuffer(
              graph,
              'ring-areas',
              out.ringAreas,
              'float32',
              ringCapacity
            ),
            ringIsHole: importGraphBuffer(
              graph,
              'ring-is-hole',
              out.ringIsHole,
              'uint32',
              ringCapacity
            ),
            ringShells: importGraphBuffer(
              graph,
              'ring-shells',
              out.ringShells,
              'uint32',
              ringCapacity
            ),
            count: importGraphBuffer(graph, 'ring-count', out.ringCount, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'ring-overflow', out.ringOverflow, 'uint32', 1),
            openSegmentCount: importGraphBuffer(graph, 'ring-open', out.ringOpen, 'uint32', 1)
          }
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(out.count, 1);
  const [overflow] = await readUint32(out.overflow, 1);
  const words = await readUint32(out.cells, 2 * capacity);
  const endpoints = await readFloat32(out.endpoints, 4 * capacity);
  const segments: CellRun['segments'] = [];
  for (let row = 0; row < count; row++) {
    segments.push({
      cell: joinCellKey(words[2 * row], words[2 * row + 1]),
      a: [endpoints[4 * row], endpoints[4 * row + 1]],
      b: [endpoints[4 * row + 2], endpoints[4 * row + 3]]
    });
  }
  const rings = {
    count: (await readUint32(out.ringCount, 1))[0],
    overflow: (await readUint32(out.ringOverflow, 1))[0],
    open: (await readUint32(out.ringOpen, 1))[0],
    offsets: await readUint32(out.ringOffsets, ringCapacity + 1),
    positions: await readFloat32(out.ringPositions, 2 * vertexCapacity),
    areas: await readFloat32(out.ringAreas, ringCapacity),
    isHole: await readUint32(out.ringIsHole, ringCapacity),
    shells: await readUint32(out.ringShells, ringCapacity)
  };
  compiled.destroy?.();
  destroyRun(run);

  // Oracle: reached nodes -> cells -> sorted distinct -> outline segments.
  const costs = dijkstra(csr, NODE_COUNT, [{node: 40, cost: 0}]);
  const keys = new Set<bigint>();
  for (let node = 0; node < NODE_COUNT; node++) {
    if (costs[node] <= cellCostLimit) {
      const lng = positions[2 * node];
      const lat = positions[2 * node + 1];
      keys.add(
        family === 'quadbin'
          ? quadbinPointToCell(lng, lat, resolution)
          : h3ToBigInt(latLngToCell(lat, lng, resolution))
      );
    }
  }
  const cells = [...keys].sort((left, right) => (left < right ? -1 : 1));
  const expected = family === 'quadbin' ? outlineQuadbinOnCPU(cells) : outlineH3OnCPU(cells);
  return {
    segments,
    count,
    overflow,
    expected,
    reachedCells: cells.length,
    rings,
    endpoints: Array.from(endpoints.slice(0, 4 * count)),
    cellKeys: cells
  };
}

for (const [family, resolution] of [
  ['quadbin', 13],
  ['h3', 8]
] as const) {
  it(`GPUNetworkIsochrones cell outline (${family}) matches the CPU cell outline`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const result = await runCellOutline(device, family, resolution, 9);
    expect(result.overflow).toBe(0);
    expect(result.reachedCells).toBeGreaterThan(3);
    expect(result.expected.length).toBeGreaterThan(8);
    const {unmatchedActual, unmatchedExpected} = matchSegments(result.segments, result.expected);
    expect(unmatchedExpected, 'expected but missing').toEqual([]);
    expect(unmatchedActual, 'unexpected extra').toEqual([]);
  });
}

for (const [family, resolution, limit] of [
  ['quadbin', 13, 9],
  ['h3', 8, 9],
  ['h3', 9, 14]
] as const) {
  it(`GPUNetworkIsochrones cell rings (${family} ${resolution}, limit ${limit}) close the outline`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const result = await runCellOutline(device, family, resolution, limit);
    const {rings} = result;
    expect(rings.overflow).toBe(0);
    expect(rings.open, 'every outline segment is on a closed ring').toBe(0);
    expect(rings.count).toBeGreaterThan(0);
    // Same assembly on the CPU from the GPU outline segments.
    const segments = result.segments.map(
      segment => [...segment.a, ...segment.b] as [number, number, number, number]
    );
    const oracle = assembleRingsOnCPU(segments, {
      tolerance: 5e-5,
      interiorSide: family === 'h3' ? 'left' : 'right',
      normalizeWinding: true
    });
    expect(rings.count).toBe(oracle.rings.length);
    let cursor = 0;
    oracle.rings.forEach((ring, index) => {
      expect(rings.offsets[index]).toBe(cursor);
      cursor += ring.vertices.length;
      expect(rings.offsets[index + 1]).toBe(cursor);
      expect(rings.isHole[index]).toBe(ring.isHole ? 1 : 0);
    });
    // Shells are counter-clockwise (positive area) after normalization.
    for (let ring = 0; ring < rings.count; ring++) {
      expect(rings.areas[ring] > 0).toBe(rings.isHole[ring] === 0);
    }
    if (family === 'h3') {
      const polygons = cellsToMultiPolygon(result.cellKeys.map(bigIntToH3), true);
      expect(rings.count).toBe(polygons.reduce((sum, polygon) => sum + polygon.length, 0));
    }
  });
}
