// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Review C-08: projected rasters (UTM northings near 4.6e6 m) lose up to half a metre when GPU
// positions are absolute float32. These specs emulate the WGSL float32 arithmetic on the packed
// parameters and compare against a float64 reference, with and without `localOrigin`.

import {describe, expect, it} from 'vitest';
import {getGPUDistanceFieldParameterValues} from '../../src/gpu-raster/distance-field/distance-field-parameters';
import {getGPUIsobandsParameterValues} from '../../src/gpu-raster/isolines/isobands-parameters';
import {getGPUIsolinesParameterValues} from '../../src/gpu-raster/isolines/isolines-parameters';
import {
  getGPURasterProfileParameterValues,
  getGPURasterSamplingParameterValues
} from '../../src/gpu-raster/raster-sampling/raster-sampling-parameters';
import {getCellSeeds} from './distance-field/distance-field-oracle';
import {buildCellBandPieces, getCellGeometry} from './isolines/isobands-oracle';
import {sampleRasterOnCPU} from './raster-sampling/raster-sampling-oracle';

const f = Math.fround;

/** UTM zone-scale raster: 96 x 64 samples of 2.5 m, lower-left corner at a non-dyadic position. */
const WIDTH = 96;
const HEIGHT = 64;
const CELL = 2.5;
const MIN_X = 500123.37;
const MIN_Y = 4600456.91;
const EXTENT = [MIN_X, MIN_Y, MIN_X + WIDTH * CELL, MIN_Y + HEIGHT * CELL] as const;
const LOCAL_ORIGIN = [MIN_X, MIN_Y] as const;
/** The required vertex placement: sub-decimetre against float64. */
const SUB_DECIMETRE = 0.1;

function createSmoothValues(width: number, height: number): Float32Array {
  const values = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      values[row * width + column] =
        10 * Math.sin(0.13 * column) + 7 * Math.cos(0.11 * row) + 0.37 * column;
    }
  }
  return values;
}

type Crossing = {
  /** Emulated GPU vertex, float32 arithmetic on the packed parameters. */
  gpu: [number, number];
  /** Float64 crossing on the float64 extent, from the same float32 samples. */
  reference: [number, number];
};

/**
 * Every level crossing on every raster edge, placed exactly like `getEdgeCrossing` in
 * `gpu-isolines.ts`: `t = (level - a) / (b - a)`, `minX + (gridX + 0.5) * cellWidth`, each
 * operation rounded to float32.
 */
function getIsolineCrossings(
  values: Float32Array,
  width: number,
  height: number,
  extent: readonly [number, number, number, number],
  parameters: Float32Array,
  level: number
): Crossing[] {
  const [minX, minY, cellWidth, cellHeight] = [
    parameters[1],
    parameters[2],
    parameters[3],
    parameters[4]
  ];
  const referenceCellWidth = (extent[2] - extent[0]) / width;
  const referenceCellHeight = (extent[3] - extent[1]) / height;
  const crossings: Crossing[] = [];
  const add = (a: number, b: number, column: number, row: number, isHorizontal: boolean) => {
    if (a >= level === b >= level) {
      return;
    }
    const t = f(f(level - a) / f(b - a));
    const gridX = isHorizontal ? f(column + t) : column;
    const gridY = isHorizontal ? row : f(row + t);
    const referenceT = (level - a) / (b - a);
    crossings.push({
      gpu: [f(minX + f(f(gridX + 0.5) * cellWidth)), f(minY + f(f(gridY + 0.5) * cellHeight))],
      reference: [
        extent[0] + (column + (isHorizontal ? referenceT : 0) + 0.5) * referenceCellWidth,
        extent[1] + (row + (isHorizontal ? 0 : referenceT) + 0.5) * referenceCellHeight
      ]
    });
  };
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const value = values[row * width + column];
      if (column + 1 < width) add(value, values[row * width + column + 1], column, row, true);
      if (row + 1 < height) add(value, values[(row + 1) * width + column], column, row, false);
    }
  }
  return crossings;
}

/** Largest float64 distance between `gpu + origin` and the reference, per axis. */
function getMaximumError(crossings: readonly Crossing[], origin: readonly [number, number]) {
  let maximum = 0;
  for (const {gpu, reference} of crossings) {
    maximum = Math.max(
      maximum,
      Math.abs(gpu[0] + origin[0] - reference[0]),
      Math.abs(gpu[1] + origin[1] - reference[1])
    );
  }
  return maximum;
}

describe('localOrigin packing', () => {
  it('subtracts the local origin in float64 before rounding and keeps the default', () => {
    const settings = {width: WIDTH, height: HEIGHT, levelCount: 1, extent: EXTENT};
    const absolute = getGPUIsolinesParameterValues(settings);
    const local = getGPUIsolinesParameterValues({...settings, localOrigin: LOCAL_ORIGIN});
    expect(absolute[1]).toBe(f(MIN_X));
    expect(absolute[2]).toBe(f(MIN_Y));
    expect([local[1], local[2]]).toEqual([0, 0]);
    // Cell sizes come from the float64 extent either way.
    expect([local[3], local[4]]).toEqual([absolute[3], absolute[4]]);
    const shifted = getGPUIsolinesParameterValues({...settings, localOrigin: [500000, 4600000]});
    expect([shifted[1], shifted[2]]).toEqual([f(MIN_X - 500000), f(MIN_Y - 4600000)]);

    const bands = getGPUIsobandsParameterValues({
      breakCount: 2,
      width: WIDTH,
      height: HEIGHT,
      extent: EXTENT,
      localOrigin: [500000, 4600000]
    });
    expect(Array.from(bands.subarray(4, 10))).toEqual([
      f(MIN_X - 500000),
      f(MIN_Y - 4600000),
      f(EXTENT[2] - 500000),
      f(EXTENT[3] - 4600000),
      f(CELL),
      f(CELL)
    ]);

    const sampling = getGPURasterSamplingParameterValues({
      width: WIDTH,
      height: HEIGHT,
      extent: EXTENT,
      localOrigin: LOCAL_ORIGIN
    });
    expect(Array.from(sampling.subarray(0, 4))).toEqual([0, 0, f(WIDTH * CELL), f(HEIGHT * CELL)]);
    const profile = getGPURasterProfileParameterValues({
      width: WIDTH,
      height: HEIGHT,
      extent: EXTENT,
      spacing: 1,
      localOrigin: LOCAL_ORIGIN
    });
    expect(Array.from(profile.subarray(0, 4))).toEqual(Array.from(sampling.subarray(0, 4)));

    const field = getGPUDistanceFieldParameterValues({
      bounds: EXTENT,
      gridSize: [WIDTH, HEIGHT],
      localOrigin: [500000, 4600000]
    });
    expect([field[0], field[1]]).toEqual([f(MIN_X - 500000), f(MIN_Y - 4600000)]);
    const fieldByOrigin = getGPUDistanceFieldParameterValues({
      origin: [MIN_X, MIN_Y],
      cellSize: [CELL, CELL],
      localOrigin: LOCAL_ORIGIN
    });
    expect([fieldByOrigin[0], fieldByOrigin[1]]).toEqual([0, 0]);
  });

  it('rejects a non-finite local origin', () => {
    const settings = {width: 4, height: 4, levelCount: 1, extent: [0, 0, 1, 1] as const};
    expect(() => getGPUIsolinesParameterValues({...settings, localOrigin: [NaN, 0]})).toThrow(
      /localOrigin/
    );
    expect(() =>
      getGPUIsobandsParameterValues({
        breakCount: 1,
        width: 4,
        height: 4,
        extent: [0, 0, 1, 1],
        localOrigin: [0, Infinity]
      })
    ).toThrow(/localOrigin/);
    expect(() =>
      getGPURasterSamplingParameterValues({
        width: 4,
        height: 4,
        extent: [0, 0, 1, 1],
        localOrigin: [Infinity, 0]
      })
    ).toThrow(/localOrigin/);
    expect(() =>
      getGPUDistanceFieldParameterValues({cellSize: [1, 1], localOrigin: [0, NaN]})
    ).toThrow(/localOrigin/);
  });
});

describe('UTM-scale vertex placement against a float64 reference', () => {
  const values = createSmoothValues(WIDTH, HEIGHT);
  const levels = [-3.25, 0.5, 4, 11.75, 19];

  it('places isoline vertices to sub-decimetre with localOrigin at the raster corner', () => {
    const settings = {width: WIDTH, height: HEIGHT, levelCount: levels.length, extent: EXTENT};
    const local = getGPUIsolinesParameterValues({...settings, localOrigin: LOCAL_ORIGIN});
    const absolute = getGPUIsolinesParameterValues(settings);
    let localError = 0;
    let absoluteError = 0;
    let crossingCount = 0;
    for (const level of levels) {
      const localCrossings = getIsolineCrossings(values, WIDTH, HEIGHT, EXTENT, local, level);
      const absoluteCrossings = getIsolineCrossings(values, WIDTH, HEIGHT, EXTENT, absolute, level);
      crossingCount += localCrossings.length;
      localError = Math.max(localError, getMaximumError(localCrossings, LOCAL_ORIGIN));
      absoluteError = Math.max(absoluteError, getMaximumError(absoluteCrossings, [0, 0]));
    }
    expect(crossingCount).toBeGreaterThan(500);
    // Relative coordinates stay below 240 m, where the float32 ulp is 2^-16 m.
    expect(localError).toBeLessThan(SUB_DECIMETRE);
    expect(localError).toBeLessThan(4 * 2 ** -16);
    // The bug being fixed: the absolute default is quantized by the 0.5 m ulp at 4.6e6 m.
    expect(absoluteError).toBeGreaterThan(SUB_DECIMETRE);
  });

  it('keeps about 1.2e-7 of the distance from the origin on a 100 km tile', () => {
    const width = 64;
    const height = 64;
    const cell = 1587.3;
    const extent = [MIN_X, MIN_Y, MIN_X + width * cell, MIN_Y + height * cell] as const;
    const tileValues = createSmoothValues(width, height);
    const parameters = getGPUIsolinesParameterValues({
      width,
      height,
      levelCount: 1,
      extent,
      localOrigin: LOCAL_ORIGIN
    });
    let worstRatio = 0;
    for (const level of levels) {
      for (const {gpu, reference} of getIsolineCrossings(
        tileValues,
        width,
        height,
        extent,
        parameters,
        level
      )) {
        const distance = Math.max(reference[0] - MIN_X, reference[1] - MIN_Y);
        const error = Math.max(
          Math.abs(gpu[0] + MIN_X - reference[0]),
          Math.abs(gpu[1] + MIN_Y - reference[1])
        );
        worstRatio = Math.max(worstRatio, error / distance);
      }
    }
    expect(worstRatio).toBeLessThan(1.2e-7);
  });

  it('places isoband vertices to sub-decimetre with localOrigin at the raster corner', () => {
    const lo = 0.5;
    const hi = 11.75;
    const parameters = getGPUIsobandsParameterValues({
      breakCount: 2,
      width: WIDTH,
      height: HEIGHT,
      extent: EXTENT,
      localOrigin: LOCAL_ORIGIN
    });
    const absoluteParameters = getGPUIsobandsParameterValues({
      breakCount: 2,
      width: WIDTH,
      height: HEIGHT,
      extent: EXTENT
    });
    const edgeStart = [0, 1, 3, 0];
    const edgeEnd = [1, 2, 2, 3];
    const cornerOffsets = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1]
    ];
    let localError = 0;
    let absoluteError = 0;
    let vertexCount = 0;
    for (let row = 0; row + 1 < HEIGHT; row++) {
      for (let column = 0; column + 1 < WIDTH; column++) {
        const corners = [
          values[row * WIDTH + column],
          values[row * WIDTH + column + 1],
          values[(row + 1) * WIDTH + column + 1],
          values[(row + 1) * WIDTH + column]
        ];
        const getReference = (vertex: {
          kind: string;
          corner?: number;
          edge?: number;
          level?: 'lo' | 'hi';
        }): [number, number] => {
          let gridX: number;
          let gridY: number;
          if (vertex.kind === 'corner') {
            const [dx, dy] = cornerOffsets[vertex.corner!];
            gridX = column + dx;
            gridY = row + dy;
          } else {
            const boundary = vertex.level === 'lo' ? lo : hi;
            const a = corners[edgeStart[vertex.edge!]];
            const b = corners[edgeEnd[vertex.edge!]];
            const t = (boundary - a) / (b - a);
            const [startX, startY] = cornerOffsets[edgeStart[vertex.edge!]];
            const horizontal = vertex.edge === 0 || vertex.edge === 2;
            gridX = column + startX + (horizontal ? t : 0);
            gridY = row + startY + (horizontal ? 0 : t);
          }
          return [MIN_X + (gridX + 0.5) * CELL, MIN_Y + (gridY + 0.5) * CELL];
        };
        const localPieces = buildCellBandPieces(
          corners,
          getCellGeometry(parameters, column, row),
          lo,
          hi
        );
        const absolutePieces = buildCellBandPieces(
          corners,
          getCellGeometry(absoluteParameters, column, row),
          lo,
          hi
        );
        for (const [pieceIndex, piece] of localPieces.entries()) {
          for (const [vertexIndex, vertex] of piece.entries()) {
            const reference = getReference(vertex);
            const absoluteVertex = absolutePieces[pieceIndex][vertexIndex];
            localError = Math.max(
              localError,
              Math.abs(vertex.x + MIN_X - reference[0]),
              Math.abs(vertex.y + MIN_Y - reference[1])
            );
            absoluteError = Math.max(
              absoluteError,
              Math.abs(absoluteVertex.x - reference[0]),
              Math.abs(absoluteVertex.y - reference[1])
            );
            vertexCount++;
          }
        }
      }
    }
    expect(vertexCount).toBeGreaterThan(1000);
    expect(localError).toBeLessThan(SUB_DECIMETRE);
    expect(localError).toBeLessThan(4 * 2 ** -16);
    expect(absoluteError).toBeGreaterThan(SUB_DECIMETRE);
  });
});

describe('UTM-scale seed and sample snapping', () => {
  // Points 5 cm inside each cell's upper-right corner: absolute float32 (0.5 m ulp in y) moves
  // many of them across the cell edge, origin-relative float32 never does.
  const insetPoints: [number, number, number, number][] = [];
  for (let row = 0; row < HEIGHT; row += 3) {
    for (let column = 0; column < WIDTH; column += 5) {
      insetPoints.push([
        MIN_X + (column + 1) * CELL - 0.05,
        MIN_Y + (row + 1) * CELL - 0.05,
        column,
        row
      ]);
    }
  }

  it('snaps distance-field seeds to the float64 cell with localOrigin', () => {
    const snap = (
      settings: Float32Array,
      origin: readonly [number, number]
    ): {correct: number; wrong: number} => {
      const positions = new Float32Array(insetPoints.length * 2);
      for (const [index, [x, y]] of insetPoints.entries()) {
        positions[index * 2] = x - origin[0];
        positions[index * 2 + 1] = y - origin[1];
      }
      const cellSeeds = getCellSeeds({width: WIDTH, height: HEIGHT, settings, positions});
      let correct = 0;
      for (const [index, [, , column, row]] of insetPoints.entries()) {
        if (cellSeeds[row * WIDTH + column] === index) correct++;
      }
      return {correct, wrong: insetPoints.length - correct};
    };
    const base = {bounds: EXTENT, gridSize: [WIDTH, HEIGHT] as const};
    const local = snap(
      getGPUDistanceFieldParameterValues({...base, localOrigin: LOCAL_ORIGIN}),
      LOCAL_ORIGIN
    );
    expect(local.wrong).toBe(0);
    const absolute = snap(getGPUDistanceFieldParameterValues(base), [0, 0]);
    expect(absolute.wrong).toBeGreaterThan(0);
  });

  it('samples the float64 cell (nearest) with localOrigin', () => {
    const raster = {
      width: WIDTH,
      height: HEIGHT,
      values: Float32Array.from({length: WIDTH * HEIGHT}, (_, index) => index)
    };
    const count = (parameters: Float32Array, origin: readonly [number, number]) => {
      let wrong = 0;
      for (const [x, y, column, row] of insetPoints) {
        const value = sampleRasterOnCPU(raster, parameters, x - origin[0], y - origin[1]);
        if (value !== row * WIDTH + column) wrong++;
      }
      return wrong;
    };
    const settings = {width: WIDTH, height: HEIGHT, extent: EXTENT, method: 'nearest'} as const;
    expect(
      count(
        getGPURasterSamplingParameterValues({...settings, localOrigin: LOCAL_ORIGIN}),
        LOCAL_ORIGIN
      )
    ).toBe(0);
    expect(count(getGPURasterSamplingParameterValues(settings), [0, 0])).toBeGreaterThan(0);
  });
});
