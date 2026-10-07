// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUPointDensityHexagonCell,
  getGPUPointDensityHexagonCenter,
  getGPUPointDensityHexagonGridSize
} from '@luma.gl/experimental/gpu-spatial-analysis';

/**
 * Zone geometry of the tides scene: the active lattice (hexagons or squares) of one radius, the
 * zone under a point, the polygon of a zone, and the outline segments the scene draws between the
 * zones that carry data. Pure TypeScript over the contributor's own hexagon helpers, so the CPU
 * picture (tooltips, notes, outlines, arrow endpoints) is the lattice the GPU wrote.
 */

/** The zone kind of the lattice. */
export type ZoneKind = 'hexagon' | 'grid';

/** Planar bounds `[minX, minY, maxX, maxY]` in metres. */
export type MetricBounds = [number, number, number, number];

/** The active lattice of one zone size: what the layer and the contributor are both given. */
export type Lattice = {
  kind: ZoneKind;
  /** Hexagon radius or square cell size in metres. */
  size: number;
  /** Lattice bounds; for hexagons `minX, minY` is the centre of hexagon (0, 0). */
  bounds: MetricBounds;
  /** `[columns, rows]` of the active lattice. */
  grid: [number, number];
};

const SQRT3 = Math.sqrt(3);

/** Lattice of `size` metres over the points' bounds, padded so every point has a zone. */
export function getLattice(kind: ZoneKind, size: number, dataBounds: MetricBounds): Lattice {
  const pad = kind === 'hexagon' ? 1.2 * size : 100;
  const bounds: MetricBounds = [
    dataBounds[0] - pad,
    dataBounds[1] - pad,
    dataBounds[2] + pad,
    dataBounds[3] + pad
  ];
  const grid: [number, number] =
    kind === 'hexagon'
      ? getGPUPointDensityHexagonGridSize(bounds, size)
      : [Math.ceil((bounds[2] - bounds[0]) / size), Math.ceil((bounds[3] - bounds[1]) / size)];
  return {kind, size, bounds, grid};
}

/** Width and height of one square cell of a `grid` lattice (the bounds divided by the grid). */
function getCellSize(lattice: Lattice): [number, number] {
  return [
    (lattice.bounds[2] - lattice.bounds[0]) / lattice.grid[0],
    (lattice.bounds[3] - lattice.bounds[1]) / lattice.grid[1]
  ];
}

/** Planar centre of a zone. */
export function getZoneCenter(lattice: Lattice, zone: number): [number, number] {
  const column = zone % lattice.grid[0];
  const row = Math.floor(zone / lattice.grid[0]);
  if (lattice.kind === 'hexagon') {
    return getGPUPointDensityHexagonCenter(
      column,
      row,
      lattice.bounds[0],
      lattice.bounds[1],
      lattice.size
    );
  }
  const [width, height] = getCellSize(lattice);
  return [lattice.bounds[0] + (column + 0.5) * width, lattice.bounds[1] + (row + 0.5) * height];
}

/** The zone under a planar point, or `-1` outside the active lattice. */
export function getZoneAt(lattice: Lattice, x: number, y: number): number {
  let column: number;
  let row: number;
  if (lattice.kind === 'hexagon') {
    [column, row] = getGPUPointDensityHexagonCell(
      x,
      y,
      lattice.bounds[0],
      lattice.bounds[1],
      lattice.size
    );
  } else {
    const [width, height] = getCellSize(lattice);
    column = Math.floor((x - lattice.bounds[0]) / width);
    row = Math.floor((y - lattice.bounds[1]) / height);
  }
  if (column < 0 || row < 0 || column >= lattice.grid[0] || row >= lattice.grid[1]) return -1;
  return row * lattice.grid[0] + column;
}

/** Corners of a zone in planar metres (six for a hexagon, four for a square), counter-clockwise. */
export function getZoneCorners(lattice: Lattice, zone: number): [number, number][] {
  const [x, y] = getZoneCenter(lattice, zone);
  if (lattice.kind === 'hexagon') {
    const radius = lattice.size;
    const half = (SQRT3 / 2) * radius;
    return [
      [x + half, y - radius / 2],
      [x + half, y + radius / 2],
      [x, y + radius],
      [x - half, y + radius / 2],
      [x - half, y - radius / 2],
      [x, y - radius]
    ];
  }
  const [width, height] = getCellSize(lattice);
  return [
    [x - width / 2, y - height / 2],
    [x + width / 2, y - height / 2],
    [x + width / 2, y + height / 2],
    [x - width / 2, y + height / 2]
  ];
}

/**
 * Neighbour zone in one of the six hexagon directions (odd-r offset layout, row grows north),
 * or `-1` off the lattice. Direction order: east, north-east, north-west, west, south-west,
 * south-east.
 */
function getHexagonNeighbor(lattice: Lattice, zone: number, direction: number): number {
  const [columns, rows] = lattice.grid;
  const column = zone % columns;
  const row = Math.floor(zone / columns);
  const odd = row & 1;
  const offsets: [number, number][] = [
    [1, 0],
    [odd ? 1 : 0, 1],
    [odd ? 0 : -1, 1],
    [-1, 0],
    [odd ? 0 : -1, -1],
    [odd ? 1 : 0, -1]
  ];
  const [dc, dr] = offsets[direction];
  const neighborColumn = column + dc;
  const neighborRow = row + dr;
  if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= columns || neighborRow >= rows) {
    return -1;
  }
  return neighborRow * columns + neighborColumn;
}

/**
 * Writes the outline segments (`x0, y0, x1, y1` metres per segment) of the zones for which
 * `hasData(zone)` is true into `target`, every shared edge once, and returns the segment count.
 * `target` needs room for six segments per zone (hexagons) or four (squares).
 *
 * Edges toward east, north-east and north-west (hexagons) or east and north (squares) are always
 * drawn; the others only when the neighbour has no data, so a hairline separates two zones with
 * data exactly once and closes the outline of the data's edge.
 */
export function buildZoneOutlines(
  lattice: Lattice,
  hasData: (zone: number) => boolean,
  target: Float32Array
): number {
  const zoneCount = lattice.grid[0] * lattice.grid[1];
  let count = 0;
  const push = (a: readonly [number, number], b: readonly [number, number]) => {
    target[count * 4] = a[0];
    target[count * 4 + 1] = a[1];
    target[count * 4 + 2] = b[0];
    target[count * 4 + 3] = b[1];
    count++;
  };
  for (let zone = 0; zone < zoneCount; zone++) {
    if (!hasData(zone)) continue;
    const corners = getZoneCorners(lattice, zone);
    if (lattice.kind === 'hexagon') {
      // Corner i to i + 1 is the edge toward direction i (east first), see getZoneCorners.
      for (let direction = 0; direction < 6; direction++) {
        const owner = direction <= 2;
        const neighbor = getHexagonNeighbor(lattice, zone, direction);
        if (owner || neighbor < 0 || !hasData(neighbor)) {
          push(corners[direction], corners[(direction + 1) % 6]);
        }
      }
    } else {
      const columns = lattice.grid[0];
      const column = zone % columns;
      const row = Math.floor(zone / columns);
      const west = column > 0 ? zone - 1 : -1;
      const south = row > 0 ? zone - columns : -1;
      // Corners: south-west, south-east, north-east, north-west; east and north are always drawn.
      if (south < 0 || !hasData(south)) push(corners[0], corners[1]);
      push(corners[1], corners[2]);
      push(corners[2], corners[3]);
      if (west < 0 || !hasData(west)) push(corners[3], corners[0]);
    }
  }
  return count;
}
