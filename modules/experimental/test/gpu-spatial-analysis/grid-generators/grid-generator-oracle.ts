// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type GridOracleResult = {positions: number[][]; centers: number[][]};

/** f64 reference of `GPUGridGenerator`, one cell at a time. */
export function generateGrid(
  gridType: 'square' | 'hex' | 'triangle' | 'point',
  columns: number,
  rows: number,
  minX: number,
  minY: number,
  w: number,
  h: number
): GridOracleResult {
  const positions: number[][] = [];
  const centers: number[][] = [];
  for (let row = 0; row < rows; row++) {
    const shift = row % 2 === 1 ? w / 2 : 0;
    const perRow = gridType === 'triangle' ? 2 * columns : columns;
    for (let column = 0; column < perRow; column++) {
      if (gridType === 'square' || gridType === 'point') {
        const x = minX + column * w;
        const y = minY + row * h;
        centers.push([x + w / 2, y + h / 2]);
        if (gridType === 'square') {
          positions.push([x, y], [x + w, y], [x + w, y + h], [x, y + h]);
        }
      } else if (gridType === 'hex') {
        const radius = w / Math.sqrt(3);
        const cx = minX + w / 2 + column * w + shift;
        const cy = minY + radius + row * 1.5 * radius;
        centers.push([cx, cy]);
        for (let corner = 0; corner < 6; corner++) {
          const angle = Math.PI / 6 + (Math.PI / 3) * corner;
          positions.push([cx + radius * Math.cos(angle), cy + radius * Math.sin(angle)]);
        }
      } else {
        const x0 = minX + Math.floor(column / 2) * w + shift;
        const y0 = minY + row * h;
        const y1 = y0 + h;
        const triangle =
          column % 2 === 0
            ? [
                [x0, y0],
                [x0 + w, y0],
                [x0 + w / 2, y1]
              ]
            : [
                [x0 + w, y0],
                [x0 + 1.5 * w, y1],
                [x0 + w / 2, y1]
              ];
        positions.push(...triangle);
        centers.push([
          (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3,
          (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3
        ]);
      }
    }
  }
  return {positions, centers};
}
