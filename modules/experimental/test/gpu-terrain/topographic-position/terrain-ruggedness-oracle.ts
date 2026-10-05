// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getOracleGroundCellSize} from '../terrain-oracle-utils';

/** Options mirroring `GPUTerrainRuggedness` topology props. */
export type TerrainRuggednessOracleOptions = {
  edgeMode?: 'nodata' | 'extrapolate';
};

/** Float64 results of the oracle; invalid pixels hold NaN and validity 0. */
export type TerrainRuggednessOracleResult = {
  topographicPositionIndex: number[];
  rileyRuggedness: number[];
  wilsonRuggedness: number[];
  roughness: number[];
  validity: number[];
};

type Sample = number; // NaN encodes nodata, exactly like GDAL with a NaN nodata value.

/** GDAL INTERPOL: nodata when either operand is nodata, else 2a - b. */
function interpolate(a: Sample, b: Sample): Sample {
  return Number.isNaN(a) || Number.isNaN(b) ? Number.NaN : 2 * a - b;
}

/**
 * Literal mirror of gdaldem `GDALGeneric3x3Processing` window assembly plus `ComputeVal`, for
 * a NaN-encoded nodata raster. Returns the 9-sample window or undefined when the pixel is nodata.
 */
export function getGdalWindow(
  samples: ArrayLike<number>,
  width: number,
  height: number,
  row: number,
  column: number,
  edgeMode: 'nodata' | 'extrapolate'
): number[] | undefined {
  const at = (r: number, c: number): Sample => samples[r * width + c];
  let window: Sample[];
  const isEdge = row === 0 || row === height - 1 || column === 0 || column === width - 1;
  if (edgeMode === 'nodata') {
    if (isEdge) {
      return undefined;
    }
    window = [];
    for (let r = row - 1; r <= row + 1; r++) {
      for (let c = column - 1; c <= column + 1; c++) {
        window.push(at(r, c));
      }
    }
  } else if (row === 0) {
    const jmin = column === 0 ? column : column - 1;
    const jmax = column === width - 1 ? column : column + 1;
    window = [
      interpolate(at(0, jmin), at(1, jmin)),
      interpolate(at(0, column), at(1, column)),
      interpolate(at(0, jmax), at(1, jmax)),
      at(0, jmin),
      at(0, column),
      at(0, jmax),
      at(1, jmin),
      at(1, column),
      at(1, jmax)
    ];
  } else if (row === height - 1) {
    const jmin = column === 0 ? column : column - 1;
    const jmax = column === width - 1 ? column : column + 1;
    window = [
      at(row - 1, jmin),
      at(row - 1, column),
      at(row - 1, jmax),
      at(row, jmin),
      at(row, column),
      at(row, jmax),
      interpolate(at(row, jmin), at(row - 1, jmin)),
      interpolate(at(row, column), at(row - 1, column)),
      interpolate(at(row, jmax), at(row - 1, jmax))
    ];
  } else if (column === 0) {
    window = [
      interpolate(at(row - 1, 0), at(row - 1, 1)),
      at(row - 1, 0),
      at(row - 1, 1),
      interpolate(at(row, 0), at(row, 1)),
      at(row, 0),
      at(row, 1),
      interpolate(at(row + 1, 0), at(row + 1, 1)),
      at(row + 1, 0),
      at(row + 1, 1)
    ];
  } else if (column === width - 1) {
    const j = column;
    window = [
      at(row - 1, j - 1),
      at(row - 1, j),
      interpolate(at(row - 1, j), at(row - 1, j - 1)),
      at(row, j - 1),
      at(row, j),
      interpolate(at(row, j), at(row, j - 1)),
      at(row + 1, j - 1),
      at(row + 1, j),
      interpolate(at(row + 1, j), at(row + 1, j - 1))
    ];
  } else {
    window = [];
    for (let r = row - 1; r <= row + 1; r++) {
      for (let c = column - 1; c <= column + 1; c++) {
        window.push(at(r, c));
      }
    }
  }
  // ComputeVal
  if (Number.isNaN(window[4])) {
    return undefined;
  }
  for (let k = 0; k < 9; k++) {
    if (Number.isNaN(window[k])) {
      if (edgeMode === 'extrapolate') {
        window[k] = window[4];
      } else {
        return undefined;
      }
    }
  }
  return window;
}

/** GDAL TPI, TRI (Riley and Wilson) and roughness of one window. */
export function computeWindowRuggedness(window: readonly number[]): {
  topographicPositionIndex: number;
  rileyRuggedness: number;
  wilsonRuggedness: number;
  roughness: number;
} {
  const neighbors = [0, 1, 2, 3, 5, 6, 7, 8].map(k => window[k]);
  const centre = window[4];
  return {
    topographicPositionIndex: centre - neighbors.reduce((sum, value) => sum + value, 0) * 0.125,
    rileyRuggedness: Math.sqrt(neighbors.reduce((sum, v) => sum + (v - centre) ** 2, 0)),
    wilsonRuggedness: neighbors.reduce((sum, v) => sum + Math.abs(v - centre), 0) * 0.125,
    roughness: Math.max(...window) - Math.min(...window)
  };
}

/**
 * Computes gdaldem TPI, TRI (both algorithms) and roughness in float64.
 *
 * `valid` marks usable pixels (undefined means all); non-finite elevations are nodata too.
 */
export function computeTerrainRuggedness(
  values: ArrayLike<number>,
  valid: ArrayLike<number> | undefined,
  width: number,
  height: number,
  options: TerrainRuggednessOracleOptions = {}
): TerrainRuggednessOracleResult {
  const edgeMode = options.edgeMode ?? 'nodata';
  const samples = Array.from(values, (value, index) =>
    Number.isFinite(value) && (valid ? valid[index] !== 0 : true) ? value : Number.NaN
  );
  const result: TerrainRuggednessOracleResult = {
    topographicPositionIndex: [],
    rileyRuggedness: [],
    wilsonRuggedness: [],
    roughness: [],
    validity: []
  };
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const window = getGdalWindow(samples, width, height, row, column, edgeMode);
      const metrics = window ? computeWindowRuggedness(window) : undefined;
      result.topographicPositionIndex.push(metrics?.topographicPositionIndex ?? Number.NaN);
      result.rileyRuggedness.push(metrics?.rileyRuggedness ?? Number.NaN);
      result.wilsonRuggedness.push(metrics?.wilsonRuggedness ?? Number.NaN);
      result.roughness.push(metrics?.roughness ?? Number.NaN);
      result.validity.push(metrics ? 1 : 0);
    }
  }
  return result;
}

/** Options mirroring `GPUTerrainVectorRuggedness` topology props. */
export type VectorRuggednessOracleOptions = {
  radius?: number;
  cellSizeMode?: 'uniform' | 'web-mercator' | 'geographic';
  rowDirection?: 'south' | 'north';
  borderMode?: 'clamp' | 'nodata';
};

/** Float64 mirror of the VRM contributor: Horn normals, then 1 - |sum n| / N over the window. */
export function computeVectorRuggedness(
  values: ArrayLike<number>,
  valid: ArrayLike<number> | undefined,
  width: number,
  height: number,
  settings: ArrayLike<number>,
  options: VectorRuggednessOracleOptions = {}
): {vectorRuggedness: number[]; validity: number[]; normals: (number[] | undefined)[]} {
  const radius = options.radius ?? 1;
  const mode = options.cellSizeMode ?? 'uniform';
  const northSign = (options.rowDirection ?? 'south') === 'south' ? -1 : 1;
  const borderMode = options.borderMode ?? 'clamp';
  const isSampleValid = (row: number, column: number) =>
    (valid ? valid[row * width + column] !== 0 : true) &&
    Number.isFinite(values[row * width + column]);
  const normals: (number[] | undefined)[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      let ok = true;
      const window: number[] = [];
      for (let k = 0; k < 9; k++) {
        let sampleRow = row + Math.floor(k / 3) - 1;
        let sampleColumn = column + (k % 3) - 1;
        if (
          borderMode === 'nodata' &&
          (sampleRow < 0 || sampleRow >= height || sampleColumn < 0 || sampleColumn >= width)
        ) {
          ok = false;
          window.push(0);
          continue;
        }
        sampleRow = Math.min(Math.max(sampleRow, 0), height - 1);
        sampleColumn = Math.min(Math.max(sampleColumn, 0), width - 1);
        ok = ok && isSampleValid(sampleRow, sampleColumn);
        window.push(values[sampleRow * width + sampleColumn]);
      }
      const [cellX, cellY] = getOracleGroundCellSize(settings, row, height, mode);
      const zFactor = settings[2];
      const east =
        (zFactor *
          (window[2] + 2 * window[5] + window[8] - (window[0] + 2 * window[3] + window[6]))) /
        (8 * cellX);
      const north =
        (northSign *
          zFactor *
          (window[6] + 2 * window[7] + window[8] - (window[0] + 2 * window[1] + window[2]))) /
        (8 * cellY);
      if (!ok || !(cellX > 0) || !(cellY > 0) || !Number.isFinite(east + north)) {
        normals.push(undefined);
        continue;
      }
      const length = Math.hypot(east, north, 1);
      normals.push([-east / length, -north / length, 1 / length]);
    }
  }
  const vectorRuggedness: number[] = [];
  const validity: number[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      if (!normals[row * width + column]) {
        vectorRuggedness.push(Number.NaN);
        validity.push(0);
        continue;
      }
      const sum = [0, 0, 0];
      let count = 0;
      for (let r = Math.max(row - radius, 0); r <= Math.min(row + radius, height - 1); r++) {
        for (let c = Math.max(column - radius, 0); c <= Math.min(column + radius, width - 1); c++) {
          const normal = normals[r * width + c];
          if (normal) {
            sum[0] += normal[0];
            sum[1] += normal[1];
            sum[2] += normal[2];
            count++;
          }
        }
      }
      vectorRuggedness.push(
        Math.min(Math.max(1 - Math.hypot(sum[0], sum[1], sum[2]) / count, 0), 1)
      );
      validity.push(1);
    }
  }
  return {vectorRuggedness, validity, normals};
}
