// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Twelve 2D points; every point is at least 0.1 away from every edge used. */
export const POSITIONS = Float32Array.from([
  1, 1, 2, 1, 3, 3, 5, 5, 8, 2, 2, 2.5, 6.5, 8, 9, 9, 4, 2, 0.5, 7, 1, 4.5, 3, 4.5
]);
/** Values aligned with `POSITIONS`; row 5 is NaN. */
export const VALUES = Float32Array.from([10, 20, 30, 40, 50, Number.NaN, 70, 80, -5, 100, 60, 90]);
/** Stable IDs aligned with `POSITIONS`. */
export const SOURCE_IDS = Uint32Array.from([
  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111
]);
/** Concave lasso with a notch at (3, 4). */
export const LASSO = Float32Array.from([0, 0, 6, 0, 6, 6, 3, 4, 0, 6]);
export const RECTANGLE = Float32Array.from([0, 0, 4.25, 4.25]);
export const RECTANGLE_2 = Float32Array.from([4.75, 4.75, 10, 10]);
export const CIRCLE = Float32Array.from([2, 2, 1.6]);
/** World (x, y) to screen pixels (10x, 10y) for a 100x100 viewport. Column-major. */
export const SCREEN_TRANSFORM = Float32Array.from([
  0.2, 0, 0, 0, 0, -0.2, 0, 0, 0, 0, 1, 0, -1, 1, 0, 1, 100, 100, 0, 0
]);

/** Even-odd PNPOLY test, half-open in y. */
export function isInsidePolygon(
  x: number,
  y: number,
  vertices: ArrayLike<number>,
  count: number
): boolean {
  if (count < 3) {
    return false;
  }
  let inside = false;
  let previousX = vertices[(count - 1) * 2];
  let previousY = vertices[(count - 1) * 2 + 1];
  for (let index = 0; index < count; index++) {
    const currentX = vertices[index * 2];
    const currentY = vertices[index * 2 + 1];
    if (currentY > y !== previousY > y) {
      const crossingX =
        ((previousX - currentX) * (y - currentY)) / (previousY - currentY) + currentX;
      if (x < crossingX) {
        inside = !inside;
      }
    }
    previousX = currentX;
    previousY = currentY;
  }
  return inside;
}

/** Inclusive rectangle test. */
export function isInsideRectangle(x: number, y: number, bounds: ArrayLike<number>): boolean {
  return x >= bounds[0] && y >= bounds[1] && x <= bounds[2] && y <= bounds[3];
}

/** Rows of `POSITIONS` selected by a predicate. */
export function selectRows(predicate: (x: number, y: number) => boolean): number[] {
  const rows: number[] = [];
  for (let row = 0; row < POSITIONS.length / 2; row++) {
    if (predicate(POSITIONS[row * 2], POSITIONS[row * 2 + 1])) {
      rows.push(row);
    }
  }
  return rows;
}

/** CPU reference statistics over selected rows, with the `GPUHistogram` binning rule. */
export function computeRegionStatistics(
  selectedRows: number[],
  values: ArrayLike<number>,
  binCount: number,
  domain?: [number, number]
): {
  selectedCount: number;
  valueCount: number;
  sum: number;
  mean: number;
  minimum: number;
  maximum: number;
  histogram: number[];
  histogramOutsideCount: number;
} {
  const finite = selectedRows.map(row => values[row]).filter(Number.isFinite);
  const sum = finite.reduce((total, value) => total + value, 0);
  const minimum = finite.length ? Math.min(...finite) : 0;
  const maximum = finite.length ? Math.max(...finite) : 0;
  const [domainMinimum, domainMaximum] = domain ?? [minimum, maximum];
  const histogram = new Array<number>(binCount).fill(0);
  let binned = 0;
  for (const value of finite) {
    if (value < domainMinimum || value > domainMaximum) {
      continue;
    }
    const bin =
      value === domainMaximum
        ? binCount - 1
        : domainMaximum === domainMinimum
          ? 0
          : Math.min(
              Math.floor(((value - domainMinimum) / (domainMaximum - domainMinimum)) * binCount),
              binCount - 1
            );
    histogram[bin]++;
    binned++;
  }
  return {
    selectedCount: selectedRows.length,
    valueCount: finite.length,
    sum,
    mean: finite.length ? sum / finite.length : 0,
    minimum,
    maximum,
    histogram: binCount ? histogram : [],
    histogramOutsideCount: binCount ? finite.length - binned : 0
  };
}
