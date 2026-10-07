// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochroneParameterValues
} from '../../../src/gpu-network/network-isochrones';
import {
  h3ToBigInt,
  joinCellKey,
  quadbinPointToCell
} from '../../gpu-spatial-analysis/cell-aggregation/cell-aggregation-oracle';
import {
  matchSegments,
  outlineH3OnCPU,
  outlineQuadbinOnCPU
} from '../../gpu-spatial-analysis/cell-set-outline/cell-set-outline-oracle';
import {assembleRingsOnCPU} from '../../gpu-spatial-analysis/ring-assembly/ring-assembly-oracle';
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

const GRID = 9;
const NODE_COUNT = GRID * GRID;
const FACILITIES = [10, 40, 70];
const BREAKS = [8, 16, 28];
const WIDTH = 72;
const HEIGHT = 72;
const NONE = 0xffffffff;
const ORIGIN: [number, number] = [8.4, 47.1];
const SPACING = 0.01;
const CELL_COST_LIMIT = 14;

/** Nearest facility per node with the lowest row on cost ties, the allocation oracle. */
function allocateOnCPU(csr: ReturnType<typeof buildCSR>) {
  const perFacility = FACILITIES.map(node => dijkstra(csr, NODE_COUNT, [{node, cost: 0}]));
  const costs = new Float32Array(NODE_COUNT);
  const assignments = new Uint32Array(NODE_COUNT).fill(NONE);
  for (let node = 0; node < NODE_COUNT; node++) {
    costs[node] = Infinity;
    FACILITIES.forEach((_, facility) => {
      if (perFacility[facility][node] < costs[node]) {
        costs[node] = perFacility[facility][node];
        assignments[node] = facility;
      }
    });
  }
  return {costs, assignments};
}

for (const [family, resolution] of [
  ['h3', 8],
  ['quadbin', 13]
] as const) {
  it(`GPUNetworkIsochrones labels ${family} cell rings and raster pixels by facility`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const csr = buildCSR(NODE_COUNT, createGridNetwork(5, GRID, GRID));
    const positions = new Float32Array(2 * NODE_COUNT);
    for (let node = 0; node < NODE_COUNT; node++) {
      positions[2 * node] = ORIGIN[0] + SPACING * (node % GRID);
      positions[2 * node + 1] = ORIGIN[1] + SPACING * Math.floor(node / GRID);
    }
    const buffers: Buffer[] = [];
    const input = (values: Float32Array | Uint32Array) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return buffer;
    };
    const output = (length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return buffer;
    };
    const segmentCapacity = 4096;
    const ringCapacity = 128;
    const vertexCapacity = 4096;
    const triangleCapacity = 40000;
    const extent: [number, number, number, number] = [8.38, 47.08, 8.5, 47.2];
    const out = {
      costs: output(NODE_COUNT),
      assignments: output(NODE_COUNT),
      values: output(WIDTH * HEIGHT),
      pixelFacilities: output(WIDTH * HEIGHT),
      triangles: output(6 * triangleCapacity),
      bands: output(triangleCapacity),
      triangleFacilities: output(triangleCapacity),
      count: output(1),
      overflow: output(1),
      rows: output(segmentCapacity),
      cells: output(2 * segmentCapacity),
      edges: output(segmentCapacity),
      endpoints: output(4 * segmentCapacity),
      groups: output(segmentCapacity),
      outlineCount: output(1),
      outlineOverflow: output(1),
      cellFacilities: output(NODE_COUNT),
      ringOffsets: output(ringCapacity + 1),
      ringPositions: output(2 * vertexCapacity),
      ringGroups: output(ringCapacity),
      ringCount: output(1),
      ringOverflow: output(1),
      ringOpen: output(1)
    };
    const parameters = new GPUParameterBuffer(device, {
      id: 'facility-parameters',
      format: 'float32',
      length: 12,
      values: getGPUNetworkIsochroneParameterValues({
        breakCount: BREAKS.length,
        extent,
        bufferRadius: 0.003,
        cellCostLimit: CELL_COST_LIMIT
      })
    });
    const breaks = new GPUParameterBuffer(device, {
      id: 'facility-breaks',
      format: 'float32',
      length: BREAKS.length,
      values: Float32Array.from(BREAKS)
    });
    const graph = new GPUCommandGraph(device, {id: 'isochrone-facilities'});
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
        sources: importGraphBuffer(
          graph,
          'sources',
          input(Uint32Array.from(FACILITIES)),
          'uint32',
          FACILITIES.length
        ),
        assignments: importGraphBuffer(graph, 'assignments', out.assignments, 'uint32', NODE_COUNT),
        maxIterations: 128,
        breaks: breaks.importToGraph(graph),
        parameters: parameters.importToGraph(graph),
        raster: {
          width: WIDTH,
          height: HEIGHT,
          output: {
            values: importGraphBuffer(graph, 'values', out.values, 'float32', WIDTH * HEIGHT),
            pixelFacilities: importGraphBuffer(
              graph,
              'pixel-facilities',
              out.pixelFacilities,
              'uint32',
              WIDTH * HEIGHT
            ),
            triangles: importGraphBuffer(
              graph,
              'triangles',
              out.triangles,
              'float32x2',
              3 * triangleCapacity
            ),
            triangleBands: importGraphBuffer(graph, 'bands', out.bands, 'uint32', triangleCapacity),
            triangleFacilities: importGraphBuffer(
              graph,
              'triangle-facilities',
              out.triangleFacilities,
              'uint32',
              triangleCapacity
            ),
            count: importGraphBuffer(graph, 'count', out.count, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'overflow', out.overflow, 'uint32', 1)
          }
        },
        cellOutline: {
          family,
          resolution,
          byFacility: true,
          cellFacilities: importGraphBuffer(
            graph,
            'cell-facilities',
            out.cellFacilities,
            'uint32',
            NODE_COUNT
          ),
          output: {
            rows: importGraphBuffer(graph, 'rows', out.rows, 'uint32', segmentCapacity),
            cells: importGraphBuffer(graph, 'out-cells', out.cells, 'uint32x2', segmentCapacity),
            edgeIndices: importGraphBuffer(graph, 'edges', out.edges, 'uint32', segmentCapacity),
            endpoints: importGraphBuffer(
              graph,
              'endpoints',
              out.endpoints,
              'float32x4',
              segmentCapacity
            ),
            groups: importGraphBuffer(graph, 'groups', out.groups, 'uint32', segmentCapacity),
            count: importGraphBuffer(graph, 'outline-count', out.outlineCount, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'outline-overflow', out.outlineOverflow, 'uint32', 1)
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
              ringGroups: importGraphBuffer(
                graph,
                'ring-groups',
                out.ringGroups,
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

    // Allocation equals nearest facility with lowest-row ties.
    const expected = allocateOnCPU(csr);
    expect(Float32Array.from(await readFloat32(out.costs, NODE_COUNT))).toEqual(expected.costs);
    expect(await readUint32(out.assignments, NODE_COUNT)).toEqual(Array.from(expected.assignments));

    // Cell labels: facility of the cheapest reached node of each cell, ties to the lowest row.
    const best = new Map<bigint, {cost: number; facility: number}>();
    for (let node = 0; node < NODE_COUNT; node++) {
      if (!(expected.costs[node] <= CELL_COST_LIMIT)) continue;
      const [lng, lat] = [positions[2 * node], positions[2 * node + 1]];
      const key =
        family === 'quadbin'
          ? quadbinPointToCell(lng, lat, resolution)
          : h3ToBigInt(latLngToCell(lat, lng, resolution));
      const current = best.get(key);
      const candidate = {cost: expected.costs[node], facility: expected.assignments[node]};
      if (
        !current ||
        candidate.cost < current.cost ||
        (candidate.cost === current.cost && candidate.facility < current.facility)
      ) {
        best.set(key, candidate);
      }
    }
    const cells = [...best.keys()].sort((left, right) => (left < right ? -1 : 1));
    const groups = new Map(cells.map(cell => [cell, best.get(cell)!.facility]));
    expect(new Set(groups.values()).size, 'several facilities own cells').toBeGreaterThan(1);
    const cellFacilities = await readUint32(out.cellFacilities, NODE_COUNT);
    expect(cellFacilities.slice(0, cells.length), 'cell facility rows').toEqual(
      cells.map(cell => groups.get(cell))
    );
    expect(cellFacilities.slice(cells.length).every(value => value === NONE)).toBe(true);

    // Outline with group borders, group per segment, and facility-pure rings.
    const [outlineCount] = await readUint32(out.outlineCount, 1);
    expect((await readUint32(out.outlineOverflow, 1))[0]).toBe(0);
    const expectedSegments =
      family === 'quadbin' ? outlineQuadbinOnCPU(cells, groups) : outlineH3OnCPU(cells, groups);
    const words = await readUint32(out.cells, 2 * outlineCount);
    const endpoints = await readFloat32(out.endpoints, 4 * outlineCount);
    const segmentGroups = (await readUint32(out.groups, outlineCount)).slice(0, outlineCount);
    const segments = Array.from({length: outlineCount}, (_, row) => ({
      cell: joinCellKey(words[2 * row], words[2 * row + 1]),
      a: [endpoints[4 * row], endpoints[4 * row + 1]] as [number, number],
      b: [endpoints[4 * row + 2], endpoints[4 * row + 3]] as [number, number]
    }));
    const matching = matchSegments(segments, expectedSegments);
    expect(matching.unmatchedExpected, 'outline missing').toEqual([]);
    expect(matching.unmatchedActual, 'outline extra').toEqual([]);
    segments.forEach((segment, row) => {
      expect(segmentGroups[row]).toBe(groups.get(segment.cell));
    });
    const [ringCount] = await readUint32(out.ringCount, 1);
    expect((await readUint32(out.ringOverflow, 1))[0]).toBe(0);
    expect((await readUint32(out.ringOpen, 1))[0], 'open segments').toBe(0);
    const oracle = assembleRingsOnCPU(
      segments.map(segment => [...segment.a, ...segment.b] as [number, number, number, number]),
      {
        tolerance: 5e-5,
        interiorSide: family === 'h3' ? 'left' : 'right',
        normalizeWinding: true,
        groups: segmentGroups
      }
    );
    expect(ringCount).toBe(oracle.rings.length);
    const ringGroups = await readUint32(out.ringGroups, ringCount);
    expect([...ringGroups].sort()).toEqual(oracle.rings.map(ring => ring.group).sort());
    expect(new Set(ringGroups).size, 'rings of several facilities').toBeGreaterThan(1);

    // Raster: each facility owns the pixel at its own node, labels exist exactly where reached.
    const values = await readFloat32(out.values, WIDTH * HEIGHT);
    const pixelFacilities = await readUint32(out.pixelFacilities, WIDTH * HEIGHT);
    values.forEach((value, pixel) => {
      expect(pixelFacilities[pixel] === NONE, `pixel ${pixel} label vs reach`).toBe(value > 1e29);
    });
    FACILITIES.forEach((node, facility) => {
      const column = Math.floor(
        ((positions[2 * node] - extent[0]) / (extent[2] - extent[0])) * WIDTH
      );
      const row = Math.floor(
        ((positions[2 * node + 1] - extent[1]) / (extent[3] - extent[1])) * HEIGHT
      );
      expect(pixelFacilities[row * WIDTH + column], `facility ${facility} own pixel`).toBe(
        facility
      );
    });
    expect(new Set(pixelFacilities.filter(value => value !== NONE)).size).toBe(FACILITIES.length);

    // Triangles carry a facility within range up to the count, NONE after.
    const [triangleCount] = await readUint32(out.count, 1);
    expect(triangleCount).toBeGreaterThan(100);
    const triangleFacilities = await readUint32(out.triangleFacilities, triangleCount + 1);
    for (let triangle = 0; triangle < triangleCount; triangle++) {
      expect(triangleFacilities[triangle], `triangle ${triangle}`).toBeLessThan(FACILITIES.length);
    }
    expect(triangleFacilities[triangleCount]).toBe(NONE);

    compiled.destroy?.();
    parameters.destroy();
    breaks.destroy();
    for (const buffer of buffers) buffer.destroy();
  });
}
