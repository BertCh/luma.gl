// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUTerrainCellSizeMode} from '../../../src/map-graphs/terrain-analysis/gpu-terrain-derivatives';

const f32 = Math.fround;
const METERS_PER_DEGREE = 111319.49079327357;
const NAN_STATUS_HEIGHT = Number.NaN;

/** Options shared by both oracles. */
export type TerrainFeaturesOracleOptions = {
  cellSizeMode?: GPUTerrainCellSizeMode;
  maximumRadiusPixels?: number;
};

/** Ground cell size for a row, float64 formula rounded to float32. Mirrors the kernel's model. */
export function getOracleCellSize(
  mode: GPUTerrainCellSizeMode,
  row: number,
  height: number,
  cellSizeX: number,
  cellSizeY: number,
  northEdge: number,
  southEdge: number
): [number, number] {
  if (mode === 'uniform') {
    return [cellSizeX, cellSizeY];
  }
  const rowFraction = (row + 0.5) / height;
  const edge = northEdge + (southEdge - northEdge) * rowFraction;
  if (mode === 'web-mercator') {
    const scale = 1 / Math.cosh(Math.PI * (1 - 2 * edge));
    return [f32(cellSizeX * scale), f32(cellSizeY * scale)];
  }
  return [
    f32(cellSizeX * METERS_PER_DEGREE * Math.cos((edge * Math.PI) / 180)),
    f32(cellSizeY * METERS_PER_DEGREE)
  ];
}

/** Metric disc membership with the same float32 rounding steps as the kernel. */
export function isInOracleDisc(
  pixelX: number,
  pixelY: number,
  centreX: number,
  centreY: number,
  cellX: number,
  cellY: number,
  radiusSquared: number
): boolean {
  const scaledX = f32(f32(pixelX - centreX) * cellX);
  const scaledY = f32(f32(pixelY - centreY) * cellY);
  return f32(f32(scaledX * scaledX) + f32(scaledY * scaledY)) <= radiusSquared;
}

/** Whether a disc pixel has an 8-neighbour outside the disc (grid bounds are irrelevant). */
export function isOnOracleRing(
  pixelX: number,
  pixelY: number,
  centreX: number,
  centreY: number,
  cellX: number,
  cellY: number,
  radiusSquared: number
): boolean {
  for (let neighborY = -1; neighborY <= 1; neighborY++) {
    for (let neighborX = -1; neighborX <= 1; neighborX++) {
      if (
        (neighborX !== 0 || neighborY !== 0) &&
        !isInOracleDisc(
          pixelX + neighborX,
          pixelY + neighborY,
          centreX,
          centreY,
          cellX,
          cellY,
          radiusSquared
        )
      ) {
        return true;
      }
    }
  }
  return false;
}

/** Radius clamped so the disc fits `maximumRadiusPixels`, and whether it was clamped. */
export function getOracleEffectiveRadius(
  requested: number,
  cellX: number,
  cellY: number,
  maximumRadiusPixels: number
): {radius: number; clamped: boolean} {
  const maximum = Math.min(f32(maximumRadiusPixels * cellX), f32(maximumRadiusPixels * cellY));
  return {radius: Math.min(requested, maximum), clamped: requested > maximum};
}

/** Result of {@link computeTerrainSummits}. */
export type TerrainSummitsOracleResult = {
  mask: number[];
  drop: number[];
  overflow: number;
  ids: number[];
};

/** Definitional summit oracle: collect the disc, take its (height, -index) maximum and ring. */
export function computeTerrainSummits(
  values: ArrayLike<number>,
  validity: ArrayLike<number> | undefined,
  width: number,
  height: number,
  settings: ArrayLike<number>,
  options: TerrainFeaturesOracleOptions & {incompleteNeighborhood?: 'reject' | 'ignore'} = {}
): TerrainSummitsOracleResult {
  const mode = options.cellSizeMode ?? 'uniform';
  const maximumRadiusPixels = options.maximumRadiusPixels ?? 16;
  const reject = (options.incompleteNeighborhood ?? 'reject') === 'reject';
  const isValid = (column: number, row: number) =>
    column >= 0 &&
    row >= 0 &&
    column < width &&
    row < height &&
    (validity ? validity[row * width + column] !== 0 : true) &&
    Number.isFinite(values[row * width + column]);
  const mask: number[] = new Array(width * height).fill(0);
  const drop: number[] = new Array(width * height).fill(Number.NaN);
  const ids: number[] = [];
  let overflow = 0;
  const requested = settings[0];
  const minimumDrop = settings[1];
  for (let row = 0; row < height; row++) {
    const [cellX, cellY] = getOracleCellSize(
      mode,
      row,
      height,
      settings[2],
      settings[3],
      settings[4],
      settings[5]
    );
    if (!(cellX > 0 && cellY > 0 && Number.isFinite(cellX) && Number.isFinite(cellY))) {
      continue;
    }
    if (!(Number.isFinite(requested) && requested > 0)) {
      continue;
    }
    const {radius, clamped} = getOracleEffectiveRadius(
      requested,
      cellX,
      cellY,
      maximumRadiusPixels
    );
    if (clamped) {
      overflow = 1;
    }
    const radiusSquared = f32(radius * radius);
    const extentX = Math.ceil(radius / cellX) + 2;
    const extentY = Math.ceil(radius / cellY) + 2;
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      if (!isValid(column, row)) {
        continue;
      }
      let incomplete = false;
      let bestIndex = -1;
      let bestHeight = 0;
      const ring: number[] = [];
      for (let neighborY = row - extentY; neighborY <= row + extentY; neighborY++) {
        for (let neighborX = column - extentX; neighborX <= column + extentX; neighborX++) {
          if (!isInOracleDisc(neighborX, neighborY, column, row, cellX, cellY, radiusSquared)) {
            continue;
          }
          if (!isValid(neighborX, neighborY)) {
            incomplete = true;
            continue;
          }
          const neighborIndex = neighborY * width + neighborX;
          const neighborHeight = values[neighborIndex];
          if (
            bestIndex < 0 ||
            neighborHeight > bestHeight ||
            (neighborHeight === bestHeight && neighborIndex < bestIndex)
          ) {
            bestIndex = neighborIndex;
            bestHeight = neighborHeight;
          }
          if (
            neighborIndex !== index &&
            isOnOracleRing(neighborX, neighborY, column, row, cellX, cellY, radiusSquared)
          ) {
            ring.push(neighborHeight);
          }
        }
      }
      if ((reject && incomplete) || bestIndex !== index) {
        continue;
      }
      const ringMaximum = ring.length > 0 ? Math.max(...ring) : undefined;
      const dropValue =
        ringMaximum === undefined ? Number.POSITIVE_INFINITY : f32(values[index] - ringMaximum);
      if (ringMaximum !== undefined && dropValue < minimumDrop) {
        continue;
      }
      mask[index] = 1;
      drop[index] = dropValue;
      ids.push(index);
    }
  }
  return {mask, drop, overflow, ids};
}

/** Result of {@link computeTerrainPeakSnap}. */
export type TerrainPeakSnapOracleResult = {
  positions: number[];
  heights: number[];
  status: number[];
  snapDistance: number[];
  overflow: number;
};

/** Float32 bilinear DEM at an in-grid position; NaN unless all four corners are valid. */
export function sampleOracleElevation(
  values: ArrayLike<number>,
  validity: ArrayLike<number> | undefined,
  width: number,
  height: number,
  x: number,
  y: number
): number {
  const baseX = Math.floor(x);
  const baseY = Math.floor(y);
  const nextX = Math.min(baseX + 1, width - 1);
  const nextY = Math.min(baseY + 1, height - 1);
  const isValid = (column: number, row: number) =>
    (validity ? validity[row * width + column] !== 0 : true) &&
    Number.isFinite(values[row * width + column]);
  if (
    !isValid(baseX, baseY) ||
    !isValid(nextX, baseY) ||
    !isValid(baseX, nextY) ||
    !isValid(nextX, nextY)
  ) {
    return Number.NaN;
  }
  const fractionX = f32(x - baseX);
  const fractionY = f32(y - baseY);
  const mix = (first: number, second: number, fraction: number) =>
    f32(f32(first * f32(1 - fraction)) + f32(second * fraction));
  const top = mix(values[baseY * width + baseX], values[baseY * width + nextX], fractionX);
  const bottom = mix(values[nextY * width + baseX], values[nextY * width + nextX], fractionX);
  return mix(top, bottom, fractionY);
}

/** Definitional peak snap oracle following the documented status precedence. */
export function computeTerrainPeakSnap(
  values: ArrayLike<number>,
  validity: ArrayLike<number> | undefined,
  width: number,
  height: number,
  candidates: ArrayLike<number>,
  candidateHeights: ArrayLike<number> | undefined,
  candidateRadii: ArrayLike<number> | undefined,
  settings: ArrayLike<number>,
  options: TerrainFeaturesOracleOptions = {}
): TerrainPeakSnapOracleResult {
  const mode = options.cellSizeMode ?? 'uniform';
  const maximumRadiusPixels = options.maximumRadiusPixels ?? 16;
  const interior = settings[7] !== 0;
  const candidateCount = candidates.length / 2;
  const isValid = (column: number, row: number) =>
    column >= 0 &&
    row >= 0 &&
    column < width &&
    row < height &&
    (validity ? validity[row * width + column] !== 0 : true) &&
    Number.isFinite(values[row * width + column]);
  const result: TerrainPeakSnapOracleResult = {
    positions: [],
    heights: [],
    status: [],
    snapDistance: [],
    overflow: 0
  };
  for (let candidateIndex = 0; candidateIndex < candidateCount; candidateIndex++) {
    const candidateX = candidates[2 * candidateIndex];
    const candidateY = candidates[2 * candidateIndex + 1];
    let status = 6;
    let positionX = candidateX;
    let positionY = candidateY;
    let outputHeight = NAN_STATUS_HEIGHT;
    let snapDistance = 0;
    const finish = () => {
      result.positions.push(positionX, positionY);
      result.heights.push(outputHeight);
      result.status.push(status);
      result.snapDistance.push(snapDistance);
    };
    if (
      !(
        Number.isFinite(candidateX) &&
        Number.isFinite(candidateY) &&
        candidateX >= 0 &&
        candidateY >= 0 &&
        candidateX <= width - 1 &&
        candidateY <= height - 1
      )
    ) {
      finish();
      continue;
    }
    const demHeight = sampleOracleElevation(
      values,
      validity,
      width,
      height,
      candidateX,
      candidateY
    );
    outputHeight = demHeight;
    status = 5;
    const nearestX = Math.floor(f32(candidateX + 0.5));
    const nearestY = Math.floor(f32(candidateY + 0.5));
    const [cellX, cellY] = getOracleCellSize(
      mode,
      nearestY,
      height,
      settings[3],
      settings[4],
      settings[5],
      settings[6]
    );
    let requested = settings[0];
    const candidateRadius = candidateRadii?.[candidateIndex];
    if (candidateRadius !== undefined && Number.isFinite(candidateRadius) && candidateRadius > 0) {
      requested = candidateRadius;
    }
    if (
      !(
        Number.isFinite(cellX) &&
        Number.isFinite(cellY) &&
        cellX > 0 &&
        cellY > 0 &&
        Number.isFinite(requested) &&
        requested > 0
      )
    ) {
      finish();
      continue;
    }
    const {radius, clamped} = getOracleEffectiveRadius(
      requested,
      cellX,
      cellY,
      maximumRadiusPixels
    );
    if (clamped) {
      result.overflow = 1;
    }
    const radiusSquared = f32(radius * radius);
    const extentX = Math.ceil(radius / cellX) + 2;
    const extentY = Math.ceil(radius / cellY) + 2;
    let bestIndex = -1;
    let bestHeight = 0;
    for (let pixelY = nearestY - extentY; pixelY <= nearestY + extentY; pixelY++) {
      for (let pixelX = nearestX - extentX; pixelX <= nearestX + extentX; pixelX++) {
        if (
          !isValid(pixelX, pixelY) ||
          !isInOracleDisc(pixelX, pixelY, candidateX, candidateY, cellX, cellY, radiusSquared)
        ) {
          continue;
        }
        const pixelIndex = pixelY * width + pixelX;
        const pixelHeight = values[pixelIndex];
        if (
          bestIndex < 0 ||
          pixelHeight > bestHeight ||
          (pixelHeight === bestHeight && pixelIndex < bestIndex)
        ) {
          bestIndex = pixelIndex;
          bestHeight = pixelHeight;
        }
      }
    }
    if (bestIndex < 0) {
      finish();
      continue;
    }
    const bestX = bestIndex % width;
    const bestY = Math.floor(bestIndex / width);
    if (
      interior &&
      isOnOracleRing(bestX, bestY, candidateX, candidateY, cellX, cellY, radiusSquared)
    ) {
      status = 2;
      finish();
      continue;
    }
    if (bestX === nearestX && bestY === nearestY) {
      status = 0;
      finish();
      continue;
    }
    const moveX = f32(f32(bestX - candidateX) * cellX);
    const moveY = f32(f32(bestY - candidateY) * cellY);
    const moveSquared = f32(f32(moveX * moveX) + f32(moveY * moveY));
    if (moveSquared > f32(settings[1] * settings[1])) {
      status = 3;
      finish();
      continue;
    }
    let reference = demHeight;
    const candidateHeight = candidateHeights?.[candidateIndex];
    if (candidateHeight !== undefined && Number.isFinite(candidateHeight)) {
      reference = candidateHeight;
    }
    if (Number.isFinite(reference) && Math.abs(f32(bestHeight - reference)) > settings[2]) {
      status = 4;
      finish();
      continue;
    }
    status = 1;
    positionX = bestX;
    positionY = bestY;
    outputHeight = bestHeight;
    snapDistance = Math.sqrt(moveSquared);
    finish();
  }
  return result;
}
