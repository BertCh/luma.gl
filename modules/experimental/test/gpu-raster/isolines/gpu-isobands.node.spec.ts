// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {GPUIsobands} from '../../../src/gpu-raster/isolines/gpu-isobands';
import {
  getGPUIsobandsParameterValues,
  GPU_ISOBANDS_PARAMETER_LENGTH
} from '../../../src/gpu-raster/isolines/isobands-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from '../raster-algebra/raster-algebra-test-utils';
import {
  buildCellBandPieces,
  buildIsobandTrianglesOnCPU,
  countBreaksAtOrBelow,
  fanTriangulate,
  getCellGeometry,
  getPolygonArea,
  type BandVertex,
  type IsobandsScene
} from './isobands-oracle';

const PARAMETERS = getGPUIsobandsParameterValues({
  breakCount: 2,
  extent: [0, 0, 4, 4],
  width: 4,
  height: 4
});
const GEOMETRY = getCellGeometry(PARAMETERS, 1, 2);
const CELL_AREA = PARAMETERS[8] * PARAMETERS[9];

/** Value inside ternary state 0 (<1), 1 ([1, 2)) or 2 (>= 2) for breaks 1 and 2. */
function sampleState(state: number, random: () => number): number {
  return Math.fround(
    [-1, 1, 2][state] + random() * (state === 2 ? 1 : 0.9) * (state === 0 ? 1 : 1)
  );
}

function isConvexCounterClockwise(piece: readonly BandVertex[]): boolean {
  const tolerance = -1e-5 * CELL_AREA;
  for (let i = 0; i < piece.length; i++) {
    const a = piece[i];
    const b = piece[(i + 1) % piece.length];
    const c = piece[(i + 2) % piece.length];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (cross < tolerance) {
      return false;
    }
  }
  return getPolygonArea(piece) >= tolerance;
}

function getBands(corners: number[], override: {first: boolean; second: boolean} | undefined) {
  return [
    buildCellBandPieces(corners, GEOMETRY, null, 1, {hi: override?.first}),
    buildCellBandPieces(corners, GEOMETRY, 1, 2, {lo: override?.first, hi: override?.second}),
    buildCellBandPieces(corners, GEOMETRY, 2, null, {lo: override?.second})
  ];
}

describe('GPUIsobands oracle', () => {
  it('splits every ternary corner pattern into convex CCW pieces that tile the cell', () => {
    const random = createRandom(11);
    const overrides = [
      undefined,
      {first: true, second: true},
      {first: true, second: false},
      {first: false, second: false}
    ];
    let patternCount = 0;
    const saddleCentres = new Set<string>();
    for (let pattern = 0; pattern < 81; pattern++) {
      const states = [0, 1, 2, 3].map(corner => Math.floor(pattern / 3 ** corner) % 3);
      patternCount++;
      for (let sample = 0; sample < 12; sample++) {
        const corners = states.map(state => sampleState(state, random));
        for (const override of overrides) {
          const bands = getBands(corners, override);
          let areaSum = 0;
          for (const pieces of bands) {
            expect(pieces.length).toBeLessThanOrEqual(2);
            for (const piece of pieces) {
              expect(piece.length).toBeGreaterThanOrEqual(3);
              expect(piece.length).toBeLessThanOrEqual(8);
              expect(
                isConvexCounterClockwise(piece),
                `pattern ${pattern} ${JSON.stringify(piece)}`
              ).toBe(true);
              areaSum += getPolygonArea(piece);
            }
          }
          expect(areaSum / CELL_AREA).toBeCloseTo(1, 4);
        }
        const [v0, v1, v2, v3] = corners;
        if (v0 >= 1 === v2 >= 1 && v1 >= 1 === v3 >= 1 && v0 >= 1 !== v1 >= 1) {
          const centre = (v0 + v1 + (v2 + v3)) * 0.25;
          saddleCentres.add(`${centre >= 1}`);
        }
      }
    }
    expect(patternCount).toBe(81);
    // Both saddle choices occurred naturally.
    expect(saddleCentres.size).toBe(2);
  });

  it('shares crossing vertices between adjacent bands and adjacent cells', () => {
    const random = createRandom(5);
    for (let trial = 0; trial < 300; trial++) {
      const corners = [0, 1, 2, 3].map(() => Math.fround(random() * 3.4 - 0.2));
      const bands = getBands(corners, undefined);
      const key = (vertex: BandVertex) => `${vertex.edge}:${vertex.x}:${vertex.y}`;
      for (const [lower, upper, level] of [
        [0, 1, 1],
        [1, 2, 2]
      ]) {
        const highs = new Set(
          bands[lower]
            .flat()
            .filter(v => v.level === 'hi')
            .map(key)
        );
        const lows = new Set(
          bands[upper]
            .flat()
            .filter(v => v.level === 'lo')
            .map(key)
        );
        expect(Array.from(highs).sort(), `level ${level}`).toEqual(Array.from(lows).sort());
      }
    }
    // Neighbour cells compute the same crossing on a shared edge bit for bit.
    const random2 = createRandom(9);
    const scene: IsobandsScene = {
      width: 9,
      height: 7,
      values: Float32Array.from({length: 63}, () => Math.fround(random2() * 3.5)),
      breaks: Float32Array.from([0.7, 1.9, 2.6])
    };
    const parameters = getGPUIsobandsParameterValues({
      breakCount: 3,
      extent: [10, -5, 37, 16],
      width: 9,
      height: 7
    });
    const crossingsByGlobalEdge = new Map<string, Set<string>>();
    const cellColumns = scene.width - 1;
    for (let cell = 0; cell < cellColumns * (scene.height - 1); cell++) {
      const column = cell % cellColumns;
      const row = Math.floor(cell / cellColumns);
      const geometry = getCellGeometry(parameters, column, row);
      const corners = [
        [column, row],
        [column + 1, row],
        [column + 1, row + 1],
        [column, row + 1]
      ].map(([c, r]) => scene.values[r * scene.width + c]);
      for (let band = 0; band <= 3; band++) {
        const lo = band > 0 ? scene.breaks[band - 1] : null;
        const hi = band < 3 ? scene.breaks[band] : null;
        for (const piece of buildCellBandPieces(corners, geometry, lo, hi)) {
          for (const vertex of piece) {
            if (vertex.kind !== 'crossing') continue;
            const level = vertex.level === 'lo' ? band - 1 : band;
            const edge = vertex.edge!;
            const globalEdge =
              edge === 0
                ? `h${column},${row}`
                : edge === 2
                  ? `h${column},${row + 1}`
                  : edge === 3
                    ? `v${column},${row}`
                    : `v${column + 1},${row}`;
            const key = `${level}:${globalEdge}`;
            if (!crossingsByGlobalEdge.has(key)) crossingsByGlobalEdge.set(key, new Set());
            crossingsByGlobalEdge.get(key)!.add(`${vertex.x},${vertex.y}`);
          }
        }
      }
    }
    expect(crossingsByGlobalEdge.size).toBeGreaterThan(20);
    for (const [key, positions] of crossingsByGlobalEdge) {
      expect(positions.size, key).toBe(1);
    }
  });

  it('tiles the valid cells of a raster and honours nodata, window and capacity', () => {
    const random = createRandom(3);
    const width = 12;
    const height = 9;
    const values = Float32Array.from({length: width * height}, () => Math.fround(random() * 4));
    values[13] = NaN;
    values[50] = -9999;
    const validity = new Uint32Array(width * height).fill(1);
    validity[70] = 0;
    const scene: IsobandsScene = {
      width,
      height,
      values,
      validity,
      noDataValue: -9999,
      breaks: Float32Array.from([1, 2, 3, 0, 0])
    };
    const parameters = getGPUIsobandsParameterValues({
      breakCount: 3,
      extent: [0, 0, 12, 9],
      width,
      height
    });
    const all = buildIsobandTrianglesOnCPU(scene, parameters, 5);
    const cellsWithNodata = new Set<number>();
    for (const bad of [13, 50, 70]) {
      const column = bad % width;
      const row = Math.floor(bad / width);
      for (const cellColumn of [column - 1, column]) {
        for (const cellRow of [row - 1, row]) {
          if (cellColumn >= 0 && cellColumn < width - 1 && cellRow >= 0 && cellRow < height - 1) {
            cellsWithNodata.add(cellRow * (width - 1) + cellColumn);
          }
        }
      }
    }
    let area = 0;
    for (let i = 0; i < all.bands.length; i++) {
      const vertices = [0, 1, 2].map(k => ({
        x: all.triangles[i * 6 + k * 2],
        y: all.triangles[i * 6 + k * 2 + 1]
      }));
      area += getPolygonArea(vertices);
      expect(cellsWithNodata.has(all.cells[i])).toBe(false);
    }
    expect(area).toBeCloseTo((width - 1) * (height - 1) - cellsWithNodata.size, 2);
    expect(new Set(all.bands)).toEqual(new Set([0, 1, 2, 3]));
    const windowParameters = getGPUIsobandsParameterValues({
      breakCount: 3,
      extent: [0, 0, 12, 9],
      width,
      height,
      firstBand: 1,
      lastBand: 2
    });
    const windowed = buildIsobandTrianglesOnCPU(scene, windowParameters, 5);
    expect(new Set(windowed.bands)).toEqual(new Set([1, 2]));
    expect(windowed.requiredCount).toBe(all.bands.filter(band => band === 1 || band === 2).length);
    const clamped = buildIsobandTrianglesOnCPU(scene, parameters, 5, 10);
    expect(clamped.bands.length).toBe(10);
    expect(clamped.requiredCount).toBe(all.requiredCount);
    expect(clamped.triangles).toEqual(all.triangles.slice(0, 60));
  });

  it('counts breaks at or below a value', () => {
    const breaks = Float32Array.from([1, 2, 2, 5]);
    expect([0.5, 1, 1.5, 2, 4.9, 5, 9].map(v => countBreaksAtOrBelow(breaks, 4, v))).toEqual([
      0, 1, 1, 3, 3, 4, 4
    ]);
    expect(countBreaksAtOrBelow(breaks, 2, 9)).toBe(2);
    // Fan triangulation of a quad.
    const quad = buildCellBandPieces([1, 1, 1, 1], GEOMETRY, 1, 2)[0];
    expect(fanTriangulate(quad).length).toBe(2);
  });
});

describe('GPUIsobands validation and wiring', () => {
  const setup = () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `v-${serial++}`, format, length);
    return {device, graph, view};
  };

  it('packs parameters', () => {
    const packed = getGPUIsobandsParameterValues({
      breakCount: 3,
      extent: [0, 10, 8, 14],
      width: 4,
      height: 2,
      firstBand: 1
    });
    expect(packed.length).toBe(GPU_ISOBANDS_PARAMETER_LENGTH);
    expect(Array.from(packed)).toEqual([3, 1, 3, 0, 0, 10, 8, 14, 2, 2, 0, 0]);
    expect(() =>
      getGPUIsobandsParameterValues({breakCount: -1, extent: [0, 0, 1, 1], width: 2, height: 2})
    ).toThrow(/breakCount/);
    expect(() =>
      getGPUIsobandsParameterValues({breakCount: 1, extent: [0, 0, NaN, 1], width: 2, height: 2})
    ).toThrow(/extent/);
  });

  it('validates props', () => {
    const {device, graph, view} = setup();
    const base = {
      width: 4,
      height: 3,
      values: view('float32', 12),
      breaks: view('float32', 3),
      parameters: view('float32', 12)
    };
    expect(() => new GPUIsobands({...base, output: {}})).toThrow(/needs output/);
    expect(
      () => new GPUIsobands({...base, width: 1, output: {bandClasses: view('uint32', 12)}})
    ).toThrow(/width/);
    expect(() => new GPUIsobands({...base, output: {bandClasses: view('uint32', 4)}})).toThrow(
      /rows/
    );
    expect(() => new GPUIsobands({...base, output: {triangles: view('float32x2', 30)}})).toThrow(
      /together/
    );
    expect(
      () =>
        new GPUIsobands({
          ...base,
          output: {
            triangles: view('float32x2', 20),
            triangleBands: view('uint32', 10),
            count: view('uint32', 1),
            overflow: view('uint32', 1)
          }
        })
    ).toThrow(/3 \* triangleBands/);
    expect(() => new GPUIsobands({...base, output: {bandClasses: base.values as never}})).toThrow();
    expect(
      () => new GPUIsobands({...base, noDataValue: NaN, output: {bandClasses: view('uint32', 12)}})
    ).toThrow(/finite/);
    device.destroy();
    void graph;
  });

  it('creates deterministic node ids', () => {
    const {device, graph, view} = setup();
    const contributor = new GPUIsobands({
      id: 'bands',
      width: 4,
      height: 3,
      values: view('float32', 12),
      validity: view('uint32', 12),
      breaks: view('float32', 3),
      parameters: view('float32', 12),
      output: {
        bandClasses: view('uint32', 12),
        triangles: view('float32x2', 60),
        triangleBands: view('uint32', 20),
        count: view('uint32', 1),
        overflow: view('uint32', 1),
        requiredCount: view('uint32', 1),
        vertexCount: view('uint32', 1)
      }
    });
    expect(contributor.cellCount).toBe(6);
    expect(contributor.triangleCapacity).toBe(20);
    const ids = contributor.getCommandNodes(graph).map(node => node.id);
    expect(ids[0]).toBe('bands-classes');
    expect(ids).toContain('bands-count');
    expect(ids).toContain('bands-scatter');
    expect(ids).toContain('bands-publish');
    expect(ids[ids.length - 1]).toBe('bands-vertex-count');
    expect(new Set(ids).size).toBe(ids.length);
    device.destroy();
  });
});
