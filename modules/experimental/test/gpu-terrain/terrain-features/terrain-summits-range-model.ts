// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterCellSizeMode} from '../../../src/analysis-core/raster-grid';
import {getOracleCellSize, isInOracleDisc} from './terrain-summits-oracle';

const f32 = Math.fround;
const FLOAT_MAXIMUM = 3.4028234663852886e38;

/** Inputs shared by the two CPU models. `validity` defaults to all valid. */
export type SummitsModelInput = {
  values: ArrayLike<number>;
  validity?: ArrayLike<number>;
  width: number;
  height: number;
  /** `[radius, minimumDrop, cellSizeX, cellSizeY, northEdge, southEdge, 0, 0]`. */
  settings: ArrayLike<number>;
  cellSizeMode?: GPURasterCellSizeMode;
  maximumRadiusPixels?: number;
  incompleteNeighborhood?: 'reject' | 'ignore';
};

/** Mask, drop and the clamp word of a model run. */
export type SummitsModelResult = {mask: number[]; drop: number[]; radiusClamped: number};

/** Counters of the new algorithm, for the O(R) check. */
export type SummitsModelStatistics = {blocksVisited: number; maximumDepth: number};

type RowContext = {
  cellX: number;
  cellY: number;
  radiusSquared: number;
  extentX: number;
  extentY: number;
};

function getRowContext(
  input: SummitsModelInput,
  row: number
): (RowContext & {clamped: boolean}) | undefined {
  const maximumRadiusPixels = input.maximumRadiusPixels ?? 16;
  const [cellX, cellY] = getOracleCellSize(
    input.cellSizeMode ?? 'uniform',
    row,
    input.height,
    input.settings[2],
    input.settings[3],
    input.settings[4],
    input.settings[5]
  );
  if (!(cellX > 0 && cellY > 0 && Number.isFinite(cellX) && Number.isFinite(cellY))) {
    return undefined;
  }
  const requested = input.settings[0];
  if (!(Number.isFinite(requested) && requested > 0)) {
    return undefined;
  }
  const maximum = Math.min(f32(maximumRadiusPixels * cellX), f32(maximumRadiusPixels * cellY));
  const radius = Math.min(requested, maximum);
  const getExtent = (cellSize: number) =>
    Math.min(Math.floor(f32(f32(radius / cellSize) + 0.5)) + 1, maximumRadiusPixels);
  return {
    cellX,
    cellY,
    radiusSquared: f32(radius * radius),
    extentX: getExtent(cellX),
    extentY: getExtent(cellY),
    clamped: requested > maximum
  };
}

function isValidPixel(input: SummitsModelInput, column: number, row: number): boolean {
  return (
    column >= 0 &&
    row >= 0 &&
    column < input.width &&
    row < input.height &&
    (input.validity ? input.validity[row * input.width + column] !== 0 : true)
  );
}

/** Literal CPU port of the previous O(R^2) kernel (3 x 3 pre-test, box scan, per-pixel ring test). */
export function computeSummitsPreviousKernel(input: SummitsModelInput): SummitsModelResult {
  const {values, width, height} = input;
  const reject = (input.incompleteNeighborhood ?? 'reject') === 'reject';
  const minimumDrop = input.settings[1];
  const mask = new Array<number>(width * height).fill(0);
  const drop = new Array<number>(width * height).fill(Number.NaN);
  let radiusClamped = 0;
  for (let row = 0; row < height; row++) {
    const context = getRowContext(input, row);
    if (!context) {
      continue;
    }
    if (context.clamped) {
      radiusClamped = 1;
    }
    const {cellX, cellY, radiusSquared, extentX, extentY} = context;
    const inDisc = (x: number, y: number, column: number) =>
      isInOracleDisc(x, y, column, row, cellX, cellY, radiusSquared);
    const onRing = (x: number, y: number, column: number) => {
      for (let nY = -1; nY <= 1; nY++) {
        for (let nX = -1; nX <= 1; nX++) {
          if ((nX !== 0 || nY !== 0) && !inDisc(x + nX, y + nY, column)) {
            return true;
          }
        }
      }
      return false;
    };
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      if (!isValidPixel(input, column, row)) {
        continue;
      }
      const own = values[index];
      let ok = true;
      for (let nearY = -1; nearY <= 1 && ok; nearY++) {
        for (let nearX = -1; nearX <= 1 && ok; nearX++) {
          const x = column + nearX;
          const y = row + nearY;
          if ((nearX === 0 && nearY === 0) || !inDisc(x, y, column) || !isValidPixel(input, x, y)) {
            continue;
          }
          const nearIndex = y * width + x;
          if (values[nearIndex] > own || (values[nearIndex] === own && nearIndex < index)) {
            ok = false;
          }
        }
      }
      let ringHeight = 0;
      let hasRing = false;
      for (let offsetY = -extentY; offsetY <= extentY && ok; offsetY++) {
        for (let offsetX = -extentX; offsetX <= extentX && ok; offsetX++) {
          const x = column + offsetX;
          const y = row + offsetY;
          if (!inDisc(x, y, column)) {
            continue;
          }
          if (!isValidPixel(input, x, y)) {
            if (reject) {
              ok = false;
            }
            continue;
          }
          const neighborIndex = y * width + x;
          const neighborHeight = values[neighborIndex];
          if (neighborHeight > own || (neighborHeight === own && neighborIndex < index)) {
            ok = false;
            continue;
          }
          if ((offsetX !== 0 || offsetY !== 0) && onRing(x, y, column)) {
            if (!hasRing || neighborHeight > ringHeight) {
              ringHeight = neighborHeight;
              hasRing = true;
            }
          }
        }
      }
      if (!ok) {
        continue;
      }
      let dropValue = Number.POSITIVE_INFINITY;
      if (hasRing) {
        dropValue = f32(own - ringHeight);
        if (dropValue < minimumDrop) {
          continue;
        }
      }
      mask[index] = 1;
      drop[index] = dropValue;
    }
  }
  return {mask, drop, radiusClamped};
}

/** Cell-footprint max pyramid with first block size `firstBlockSize`, as the GPU builds it. */
export function buildMaximumPyramid(
  values: ArrayLike<number>,
  validity: ArrayLike<number> | undefined,
  width: number,
  height: number,
  firstBlockSize: number
): {levels: {blockSize: number; width: number; height: number; maximum: number[]}[]} {
  const levels: {blockSize: number; width: number; height: number; maximum: number[]}[] = [];
  for (let blockSize = firstBlockSize; ; blockSize *= 2) {
    const levelWidth = Math.ceil(width / blockSize);
    const levelHeight = Math.ceil(height / blockSize);
    const maximum = new Array<number>(levelWidth * levelHeight).fill(-FLOAT_MAXIMUM);
    for (let cellRow = 0; cellRow < levelHeight; cellRow++) {
      for (let cellColumn = 0; cellColumn < levelWidth; cellColumn++) {
        let value = -FLOAT_MAXIMUM;
        for (let y = cellRow * blockSize; y < Math.min((cellRow + 1) * blockSize, height); y++) {
          for (
            let x = cellColumn * blockSize;
            x < Math.min((cellColumn + 1) * blockSize, width);
            x++
          ) {
            const pixel = y * width + x;
            if ((validity ? validity[pixel] !== 0 : true) && Number.isFinite(values[pixel])) {
              if (values[pixel] > value) {
                value = values[pixel];
              }
            }
          }
        }
        maximum[cellRow * levelWidth + cellColumn] = value;
      }
    }
    levels.push({blockSize, width: levelWidth, height: levelHeight, maximum});
    if (levelWidth === 1 && levelHeight === 1) {
      return {levels};
    }
  }
}

/** Largest `dx >= 0` with offset `(dx, dy)` in the disc, or -1 (sqrt seed plus exact fix-up). */
export function getModelDiscRowHalfWidth(
  dy: number,
  cellX: number,
  cellY: number,
  radiusSquared: number,
  maximumRadiusPixels: number
): number {
  const inDisc = (dx: number) => isInOracleDisc(dx, dy, 0, 0, cellX, cellY, radiusSquared);
  const scaledY = f32(dy * cellY);
  const remaining = Math.max(f32(radiusSquared - f32(scaledY * scaledY)), 0);
  let halfWidth = Math.min(
    Math.max(Math.floor(f32(Math.sqrt(remaining) / cellX)), 0),
    maximumRadiusPixels + 1
  );
  while (halfWidth > 0 && !inDisc(halfWidth)) {
    halfWidth--;
  }
  while (halfWidth <= maximumRadiusPixels && inDisc(halfWidth + 1)) {
    halfWidth++;
  }
  return inDisc(halfWidth) ? halfWidth : -1;
}

/** Largest `dy >= 0` with `(0, dy)` in the disc. */
export function getModelDiscColumnHalfHeight(
  cellX: number,
  cellY: number,
  radiusSquared: number,
  maximumRadiusPixels: number
): number {
  const inDisc = (dy: number) => isInOracleDisc(0, dy, 0, 0, cellX, cellY, radiusSquared);
  let halfHeight = Math.min(
    Math.max(Math.floor(f32(Math.sqrt(radiusSquared) / cellY)), 0),
    maximumRadiusPixels + 1
  );
  while (halfHeight > 0 && !inDisc(halfHeight)) {
    halfHeight--;
  }
  while (halfHeight <= maximumRadiusPixels && inDisc(halfHeight + 1)) {
    halfHeight++;
  }
  return halfHeight;
}

/** CPU port of the new kernel: quadtree descent over a max pyramid plus ring runs. */
export function computeSummitsRangeKernel(
  input: SummitsModelInput,
  statistics: SummitsModelStatistics = {blocksVisited: 0, maximumDepth: 0}
): SummitsModelResult {
  const {values, width, height} = input;
  const maximumRadiusPixels = input.maximumRadiusPixels ?? 16;
  const reject = (input.incompleteNeighborhood ?? 'reject') === 'reject';
  const minimumDrop = input.settings[1];
  // 'reject' builds the pyramid on a copy where invalid pixels are +FLT_MAX and valid.
  const pyramidValues = reject
    ? Array.from(values, (value, pixel) =>
        isValidPixel(input, pixel % width, Math.floor(pixel / width)) ? value : FLOAT_MAXIMUM
      )
    : values;
  const pyramid = buildMaximumPyramid(
    pyramidValues,
    reject ? undefined : input.validity,
    width,
    height,
    2
  );
  const mask = new Array<number>(width * height).fill(0);
  const drop = new Array<number>(width * height).fill(Number.NaN);
  let radiusClamped = 0;
  for (let row = 0; row < height; row++) {
    const context = getRowContext(input, row);
    if (!context) {
      continue;
    }
    if (context.clamped) {
      radiusClamped = 1;
    }
    const {cellX, cellY, radiusSquared} = context;
    const inOffset = (dx: number, dy: number) =>
      isInOracleDisc(dx, dy, 0, 0, cellX, cellY, radiusSquared);
    const rowHalfWidth = (dy: number) =>
      getModelDiscRowHalfWidth(dy, cellX, cellY, radiusSquared, maximumRadiusPixels);
    const columnHalfHeight = getModelDiscColumnHalfHeight(
      cellX,
      cellY,
      radiusSquared,
      maximumRadiusPixels
    );
    const ownRowHalfWidth = rowHalfWidth(0);
    for (let column = 0; column < width; column++) {
      const index = row * width + column;
      if (!isValidPixel(input, column, row)) {
        continue;
      }
      const own = values[index];
      const inDisc = (x: number, y: number) =>
        isInOracleDisc(x, y, column, row, cellX, cellY, radiusSquared);
      let ok = true;
      for (let nearY = -1; nearY <= 1 && ok; nearY++) {
        for (let nearX = -1; nearX <= 1 && ok; nearX++) {
          const x = column + nearX;
          const y = row + nearY;
          if ((nearX === 0 && nearY === 0) || !inDisc(x, y) || !isValidPixel(input, x, y)) {
            continue;
          }
          const nearIndex = y * width + x;
          if (values[nearIndex] > own || (values[nearIndex] === own && nearIndex < index)) {
            ok = false;
          }
        }
      }
      if (!ok) {
        continue;
      }
      if (
        reject &&
        (column - ownRowHalfWidth < 0 ||
          column + ownRowHalfWidth >= width ||
          row - columnHalfHeight < 0 ||
          row + columnHalfHeight >= height)
      ) {
        continue;
      }
      const exact = Math.abs(own) >= FLOAT_MAXIMUM;
      const beats = (other: number, hasSmallerIndex: boolean) =>
        hasSmallerIndex ? other >= own : other > own;
      for (let offsetX = -ownRowHalfWidth; offsetX <= ownRowHalfWidth && ok; offsetX++) {
        const x = column + offsetX;
        if (offsetX === 0 || x < 0 || x >= width) {
          continue;
        }
        if (!isValidPixel(input, x, row)) {
          if (reject) {
            ok = false;
          }
          continue;
        }
        if (beats(values[row * width + x], offsetX < 0)) {
          ok = false;
        }
      }
      if (!ok) {
        continue;
      }
      const isHalfDominated = (rowLow: number, rowHigh: number, hasSmallerIndex: boolean) => {
        if (rowLow > rowHigh) {
          return true;
        }
        const stack: [number, number, number][] = [[pyramid.levels.length - 1, 0, 0]];
        while (stack.length > 0) {
          statistics.maximumDepth = Math.max(statistics.maximumDepth, stack.length);
          const [level, cellColumn, cellRow] = stack.pop()!;
          statistics.blocksVisited++;
          const {blockSize} = pyramid.levels[level];
          const x0 = cellColumn * blockSize;
          const y0 = cellRow * blockSize;
          const x1 = Math.min(x0 + blockSize, width) - 1;
          const y1 = Math.min(y0 + blockSize, height) - 1;
          const ya = Math.max(y0, rowLow);
          const yb = Math.min(y1, rowHigh);
          if (ya > yb) {
            continue;
          }
          const dxLow = x0 - column;
          const dxHigh = x1 - column;
          const dyLow = ya - row;
          const dyHigh = yb - row;
          const nearX = dxLow > 0 ? dxLow : dxHigh < 0 ? dxHigh : 0;
          const nearY = dyLow > 0 ? dyLow : dyHigh < 0 ? dyHigh : 0;
          if (!inOffset(nearX, nearY)) {
            continue;
          }
          const farX = Math.max(Math.abs(dxLow), Math.abs(dxHigh));
          const farY = Math.max(Math.abs(dyLow), Math.abs(dyHigh));
          if (!exact && ya === y0 && yb === y1 && inOffset(farX, farY)) {
            const levelData = pyramid.levels[level];
            const blockMaximum =
              levelData.maximum[Math.floor(y0 / blockSize) * levelData.width + cellColumn];
            if (beats(blockMaximum, hasSmallerIndex)) {
              return false;
            }
            continue;
          }
          if (level === 0) {
            for (let y = ya; y <= yb; y++) {
              for (let x = x0; x <= x1; x++) {
                if (!inOffset(x - column, y - row)) {
                  continue;
                }
                if (!isValidPixel(input, x, y)) {
                  if (reject) {
                    return false;
                  }
                  continue;
                }
                if (beats(values[y * width + x], hasSmallerIndex)) {
                  return false;
                }
              }
            }
            continue;
          }
          const childSize = pyramid.levels[level - 1].blockSize;
          for (let child = 0; child < 4; child++) {
            const childColumn = 2 * cellColumn + (child & 1);
            const childRow = 2 * cellRow + (child >> 1);
            if (childColumn * childSize < width && childRow * childSize < height) {
              stack.push([level - 1, childColumn, childRow]);
            }
          }
        }
        return true;
      };
      if (
        !isHalfDominated(Math.max(row - columnHalfHeight, 0), row - 1, true) ||
        !isHalfDominated(row + 1, Math.min(row + columnHalfHeight, height - 1), false)
      ) {
        continue;
      }
      let ringHeight = 0;
      let hasRing = false;
      let previousHalfWidth = -1;
      let currentHalfWidth = rowHalfWidth(-columnHalfHeight);
      for (let offsetY = -columnHalfHeight; offsetY <= columnHalfHeight; offsetY++) {
        const nextHalfWidth = offsetY < columnHalfHeight ? rowHalfWidth(offsetY + 1) : -1;
        const pixelRow = row + offsetY;
        if (pixelRow >= 0 && pixelRow < height) {
          const interiorHalfWidth = Math.min(previousHalfWidth, currentHalfWidth, nextHalfWidth);
          const hasInterior = interiorHalfWidth > 0;
          for (let run = 0; run < 2; run++) {
            if (run === 1 && !hasInterior) {
              break;
            }
            let runStart = -currentHalfWidth;
            let runEnd = currentHalfWidth;
            if (hasInterior) {
              if (run === 0) {
                runEnd = -interiorHalfWidth;
              } else {
                runStart = interiorHalfWidth;
              }
            }
            for (let offsetX = runStart; offsetX <= runEnd; offsetX++) {
              const pixelColumn = column + offsetX;
              if ((offsetX === 0 && offsetY === 0) || pixelColumn < 0 || pixelColumn >= width) {
                continue;
              }
              if (!isValidPixel(input, pixelColumn, pixelRow)) {
                continue;
              }
              const neighborHeight = values[pixelRow * width + pixelColumn];
              if (!hasRing || neighborHeight > ringHeight) {
                ringHeight = neighborHeight;
                hasRing = true;
              }
            }
          }
        }
        previousHalfWidth = currentHalfWidth;
        currentHalfWidth = nextHalfWidth;
      }
      let dropValue = Number.POSITIVE_INFINITY;
      if (hasRing) {
        dropValue = f32(own - ringHeight);
        if (dropValue < minimumDrop) {
          continue;
        }
      }
      mask[index] = 1;
      drop[index] = dropValue;
    }
  }
  return {mask, drop, radiusClamped};
}
