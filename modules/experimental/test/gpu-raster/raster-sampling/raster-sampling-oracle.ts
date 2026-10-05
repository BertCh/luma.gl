// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU oracle for GPURasterSampling and GPURasterProfile. Not exported from src. Every f32 operation
// is rounded with Math.fround in the association of the WGSL kernels. GPUs may contract a*b+c into
// FMA, so bilinear/bicubic/profile results are exact only where the arithmetic is exact.

const f = Math.fround;

/** Raster description for the oracle. */
export type OracleRaster = {
  width: number;
  height: number;
  values: Float32Array;
  validity?: Uint32Array;
  noDataValue?: number;
};

function fetchCell(raster: OracleRaster, column: number, row: number): number {
  const cell = row * raster.width + column;
  let value = raster.values[cell];
  if (raster.validity && raster.validity[cell] === 0) {
    value = NaN;
  }
  if (raster.noDataValue !== undefined && value === f(raster.noDataValue)) {
    value = NaN;
  }
  return value;
}

function fetchClamped(raster: OracleRaster, column: number, row: number): number {
  return fetchCell(
    raster,
    Math.min(Math.max(column, 0), raster.width - 1),
    Math.min(Math.max(row, 0), raster.height - 1)
  );
}

/** Catmull-Rom (a = -0.5) weights in the WGSL association. */
export function getCubicWeightsOnCPU(t: number): [number, number, number, number] {
  const t2 = f(t * t);
  return [
    f(t * f(-0.5 + f(t * f(1 - f(0.5 * t))))),
    f(1 + f(t2 * f(-2.5 + f(1.5 * t)))),
    f(t * f(0.5 + f(t * f(2 - f(1.5 * t))))),
    f(t2 * f(-0.5 + f(0.5 * t)))
  ];
}

function sampleBilinear(
  raster: OracleRaster,
  iu: number,
  iv: number,
  fx: number,
  fy: number,
  renormalize: boolean
): number {
  const wx = [f(1 - fx), fx];
  const wy = [f(1 - fy), fy];
  let sum = 0;
  let weightSum = 0;
  let hasNoData = false;
  for (let j = 0; j < 2; j++) {
    for (let i = 0; i < 2; i++) {
      if (wx[i] !== 0 && wy[j] !== 0) {
        const value = fetchClamped(raster, iu + i, iv + j);
        const weight = f(wx[i] * wy[j]);
        if (Number.isNaN(value)) {
          hasNoData = true;
        } else {
          sum = f(sum + f(weight * value));
          weightSum = f(weightSum + weight);
        }
      }
    }
  }
  if (!hasNoData) {
    return sum;
  }
  return renormalize && weightSum > 0 ? f(sum / weightSum) : NaN;
}

function sampleBicubic(
  raster: OracleRaster,
  iu: number,
  iv: number,
  fx: number,
  fy: number,
  renormalize: boolean
): number {
  const wx = getCubicWeightsOnCPU(fx);
  const wy = getCubicWeightsOnCPU(fy);
  const cells = new Array<number>(16).fill(0);
  let hasNoData = false;
  for (let j = 0; j < 4; j++) {
    for (let i = 0; i < 4; i++) {
      if (wx[i] !== 0 && wy[j] !== 0) {
        const value = fetchClamped(raster, iu - 1 + i, iv - 1 + j);
        cells[j * 4 + i] = value;
        hasNoData ||= Number.isNaN(value);
      }
    }
  }
  if (hasNoData) {
    return sampleBilinear(raster, iu, iv, fx, fy, renormalize);
  }
  let result = 0;
  for (let j = 0; j < 4; j++) {
    if (wy[j] !== 0) {
      let row = 0;
      for (let i = 0; i < 4; i++) {
        if (wx[i] !== 0) {
          row = f(row + f(wx[i] * cells[j * 4 + i]));
        }
      }
      result = f(result + f(wy[j] * row));
    }
  }
  return result;
}

/** Samples `raster` at `(x, y)`; `parameters` is the packed float32 parameter view. */
export function sampleRasterOnCPU(
  raster: OracleRaster,
  parameters: Float32Array,
  x: number,
  y: number
): number {
  if (Number.isNaN(x) || Number.isNaN(y)) {
    return NaN;
  }
  x = f(x);
  y = f(y);
  const [minX, minY, maxX, maxY] = parameters;
  if (x < minX || x > maxX || y < minY || y > maxY) {
    return NaN;
  }
  const method = parameters[8];
  const renormalize = parameters[9] !== 0;
  const tx = f(f(x - minX) * parameters[6]);
  const ty = f(f(y - minY) * parameters[7]);
  if (method === 0) {
    return fetchClamped(raster, Math.floor(tx), Math.floor(ty));
  }
  const u = f(tx - 0.5);
  const v = f(ty - 0.5);
  const u0 = Math.floor(u);
  const v0 = Math.floor(v);
  const sampler = method === 1 ? sampleBilinear : sampleBicubic;
  return sampler(raster, u0, v0, f(u - u0), f(v - v0), renormalize);
}

/** Samples every point of the packed `positions`; rows at or beyond `activeCount` are NaN. */
export function sampleRasterPointsOnCPU(
  raster: OracleRaster,
  parameters: Float32Array,
  positions: Float32Array,
  activeCount = positions.length / 2
): Float32Array {
  const pointCount = positions.length / 2;
  const values = new Float32Array(pointCount).fill(NaN);
  for (let point = 0; point < Math.min(activeCount, pointCount); point++) {
    values[point] = sampleRasterOnCPU(
      raster,
      parameters,
      positions[2 * point],
      positions[2 * point + 1]
    );
  }
  return values;
}

/** Result of {@link profileRasterOnCPU}. All arrays are untruncated by capacity. */
export type RasterProfileOracle = {
  vertexDistances: Float32Array;
  pathLength: Float32Array;
  sampleOffsets: Uint32Array;
  totalCount: number;
  truncated: boolean;
  samplePositions: Float32Array;
  sampleDistances: Float32Array;
  sampleValues: Float32Array;
  samplePathIds: Uint32Array;
  sampleCumulativeGain: Float32Array;
  sampleCumulativeLoss: Float32Array;
  pathGain: Float32Array;
  pathLoss: Float32Array;
  pathMinimum: Float32Array;
  pathMaximum: Float32Array;
};

/** Profile oracle; mirrors the GPU pipeline including the per-path sample limit. */
export function profileRasterOnCPU(
  raster: OracleRaster,
  parameters: Float32Array,
  pathPositions: Float32Array,
  pathOffsets: Uint32Array
): RasterProfileOracle {
  const pathCount = pathOffsets.length - 1;
  const vertexCount = pathPositions.length / 2;
  const spacing = parameters[10];
  const maximumPerPath = Math.max(1, Math.min(1 << 24, Math.floor(0xffffffff / (pathCount + 1))));
  const vertexDistances = new Float32Array(vertexCount);
  const pathLength = new Float32Array(pathCount);
  const counts: number[] = [];
  const ranges: [number, number][] = [];
  let truncated = false;
  for (let path = 0; path < pathCount; path++) {
    const begin = Math.min(pathOffsets[path], vertexCount);
    const end = Math.max(Math.min(pathOffsets[path + 1], vertexCount), begin);
    ranges.push([begin, end]);
    let count = 0;
    if (end > begin) {
      let distance = 0;
      vertexDistances[begin] = 0;
      for (let vertex = begin + 1; vertex < end; vertex++) {
        const deltaX = f(pathPositions[2 * vertex] - pathPositions[2 * vertex - 2]);
        const deltaY = f(pathPositions[2 * vertex + 1] - pathPositions[2 * vertex - 1]);
        distance = f(distance + f(Math.sqrt(f(f(deltaX * deltaX) + f(deltaY * deltaY)))));
        vertexDistances[vertex] = distance;
      }
      pathLength[path] = distance;
      count = 1;
      if (distance > 0) {
        let segments = Math.min(Math.ceil(f(distance / spacing)), maximumPerPath);
        if (segments > 0 && f(f(segments - 1) * spacing) >= distance) {
          segments -= 1;
        }
        if (f(f(segments) * spacing) < distance) {
          segments += 1;
        }
        count = segments + 1;
        if (count > maximumPerPath) {
          count = maximumPerPath;
          truncated = true;
        }
      }
    }
    counts.push(count);
  }
  const sampleOffsets = new Uint32Array(pathCount + 1);
  for (let path = 0; path < pathCount; path++) {
    sampleOffsets[path + 1] = sampleOffsets[path] + counts[path];
  }
  const totalCount = sampleOffsets[pathCount];
  const samplePositions = new Float32Array(totalCount * 2);
  const sampleDistances = new Float32Array(totalCount);
  const sampleValues = new Float32Array(totalCount);
  const samplePathIds = new Uint32Array(totalCount);
  const sampleCumulativeGain = new Float32Array(totalCount);
  const sampleCumulativeLoss = new Float32Array(totalCount);
  const pathGain = new Float32Array(pathCount);
  const pathLoss = new Float32Array(pathCount);
  const pathMinimum = new Float32Array(pathCount).fill(NaN);
  const pathMaximum = new Float32Array(pathCount).fill(NaN);
  for (let path = 0; path < pathCount; path++) {
    const [begin, end] = ranges[path];
    const first = sampleOffsets[path];
    let gain = 0;
    let loss = 0;
    let minimum = NaN;
    let maximum = NaN;
    let previous = 0;
    let hasPrevious = false;
    for (let local = 0; local < counts[path]; local++) {
      const sample = first + local;
      const lastVertex = end - 1;
      const length = vertexDistances[lastVertex];
      let distance = length;
      let positionX = pathPositions[2 * lastVertex];
      let positionY = pathPositions[2 * lastVertex + 1];
      if (local + 1 < counts[path]) {
        distance = Math.min(f(f(local) * spacing), length);
        let low = begin + 1;
        let high = end;
        while (low < high) {
          const middle = (low + high) >>> 1;
          if (vertexDistances[middle] <= distance) {
            low = middle + 1;
          } else {
            high = middle;
          }
        }
        const segmentStart = low - 1;
        if (segmentStart < lastVertex) {
          const startDistance = vertexDistances[segmentStart];
          const t = f(
            f(distance - startDistance) / f(vertexDistances[segmentStart + 1] - startDistance)
          );
          const startX = pathPositions[2 * segmentStart];
          const startY = pathPositions[2 * segmentStart + 1];
          positionX = f(startX + f(f(pathPositions[2 * segmentStart + 2] - startX) * t));
          positionY = f(startY + f(f(pathPositions[2 * segmentStart + 3] - startY) * t));
        }
      }
      samplePositions[2 * sample] = positionX;
      samplePositions[2 * sample + 1] = positionY;
      sampleDistances[sample] = distance;
      samplePathIds[sample] = path;
      const value = sampleRasterOnCPU(raster, parameters, positionX, positionY);
      sampleValues[sample] = value;
      if (Number.isFinite(value)) {
        if (hasPrevious) {
          const delta = f(value - previous);
          if (delta > 0) {
            gain = f(gain + delta);
          } else if (delta < 0) {
            loss = f(loss - delta);
          }
          minimum = value < minimum ? value : minimum;
          maximum = value > maximum ? value : maximum;
        } else {
          minimum = value;
          maximum = value;
        }
        previous = value;
        hasPrevious = true;
      }
      sampleCumulativeGain[sample] = gain;
      sampleCumulativeLoss[sample] = loss;
    }
    pathGain[path] = gain;
    pathLoss[path] = loss;
    pathMinimum[path] = minimum;
    pathMaximum[path] = maximum;
  }
  return {
    vertexDistances,
    pathLength,
    sampleOffsets,
    totalCount,
    truncated,
    samplePositions,
    sampleDistances,
    sampleValues,
    samplePathIds,
    sampleCumulativeGain,
    sampleCumulativeLoss,
    pathGain,
    pathLoss,
    pathMinimum,
    pathMaximum
  };
}
