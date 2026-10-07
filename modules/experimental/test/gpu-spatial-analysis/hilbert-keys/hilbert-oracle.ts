// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** CPU reference of the Hilbert index of cell `(x, y)` on a `2^order` grid (Wikipedia `xy2d`). */
export function getHilbertIndexReference(order: number, cellX: number, cellY: number): number {
  const size = 2 ** order;
  let x = cellX;
  let y = cellY;
  let d = 0;
  for (let s = size / 2; s >= 1; s /= 2) {
    const rx = (x & s) > 0 ? 1 : 0;
    const ry = (y & s) > 0 ? 1 : 0;
    d += s * s * ((3 * rx) ^ ry);
    if (ry === 0) {
      if (rx === 1) {
        x = s - 1 - x;
        y = s - 1 - y;
      }
      [x, y] = [y, x];
    }
  }
  return d;
}

/** Inverse of {@link getHilbertIndexReference} (Wikipedia `d2xy`), an independent check. */
export function getHilbertCellReference(order: number, index: number): [number, number] {
  const size = 2 ** order;
  let t = index;
  let x = 0;
  let y = 0;
  for (let s = 1; s < size; s *= 2) {
    const rx = 1 & (t / 2);
    const ry = 1 & (t ^ rx);
    if (ry === 0) {
      if (rx === 1) {
        x = s - 1 - x;
        y = s - 1 - y;
      }
      [x, y] = [y, x];
    }
    x += s * rx;
    y += s * ry;
    t = Math.floor(t / 4);
  }
  return [x, y];
}

/** Cell of a coordinate on the grid, in exact arithmetic (callers use exactly representable data). */
export function getCellReference(
  order: number,
  value: number,
  minimum: number,
  maximum: number
): number {
  const extent = maximum - minimum;
  if (!(extent > 0)) return 0;
  const normalized = Math.min(Math.max((value - minimum) / extent, 0), 1);
  return Math.min(Math.floor(normalized * 2 ** order), 2 ** order - 1);
}
