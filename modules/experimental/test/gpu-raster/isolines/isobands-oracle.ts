// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const f32 = Math.fround;

/** One vertex of a band piece: a cell corner or a level crossing. */
export type BandVertex = {
  x: number;
  y: number;
  kind: 'corner' | 'crossing';
  /** Corner index 0..3 (`v0` bottom-left, counter-clockwise) for corners. */
  corner?: number;
  /** `'lo'` or `'hi'` break of the band for crossings. */
  level?: 'lo' | 'hi';
  /** Canonical cell edge 0 bottom, 1 right, 2 top, 3 left for crossings. */
  edge?: number;
};

/** Geometry of the cell being built, unpacked from a `getGPUIsobandsParameterValues` array. */
export type IsobandsCellGeometry = {
  cellColumn: number;
  cellRow: number;
  minX: number;
  minY: number;
  cellWidth: number;
  cellHeight: number;
};

/** Optional overrides of the saddle centre rule, for exhaustive tests. */
export type SaddleOverride = {lo?: boolean; hi?: boolean};

type Event = {
  kind: 'corner' | 'entry' | 'exit';
  index: number;
  level: 0 | 1;
};

const EDGE_START = [0, 1, 3, 0];
const EDGE_END = [1, 2, 2, 3];

/** Reads cell sizes and origin from packed parameters. */
export function getCellGeometry(
  parameters: Float32Array,
  cellColumn: number,
  cellRow: number
): IsobandsCellGeometry {
  return {
    cellColumn,
    cellRow,
    minX: parameters[4],
    minY: parameters[5],
    cellWidth: parameters[8],
    cellHeight: parameters[9]
  };
}

function getWorld(geometry: IsobandsCellGeometry, gridX: number, gridY: number): [number, number] {
  return [
    f32(geometry.minX + f32(f32(gridX + 0.5) * geometry.cellWidth)),
    f32(geometry.minY + f32(f32(gridY + 0.5) * geometry.cellHeight))
  ];
}

/** Number of the first `breakCount` breaks that are `<= value`, by binary search. */
export function countBreaksAtOrBelow(
  breaks: ArrayLike<number>,
  breakCount: number,
  value: number
): number {
  let low = 0;
  let high = breakCount;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (breaks[middle] <= value) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

/** Band class of every sample: breaks `<= value`, `0xffffffff` for nodata. */
export function computeBandClassesOnCPU(scene: IsobandsScene, breakCount: number): Uint32Array {
  const result = new Uint32Array(scene.width * scene.height);
  for (let sample = 0; sample < result.length; sample++) {
    result[sample] = isSampleValid(scene, sample)
      ? countBreaksAtOrBelow(scene.breaks, breakCount, scene.values[sample])
      : 0xffffffff;
  }
  return result;
}

/**
 * Builds the convex pieces of band `[lo, hi)` in one marching-squares cell, following the
 * "Band regions" definition: boundary walk, crossings, segment partners, no float clipping.
 *
 * @param cornerValues Float32 values of `v0..v3` (bottom-left, bottom-right, top-right, top-left).
 * @param lo Lower break or `null` for an open lower end.
 * @param hi Upper break or `null` for an open upper end.
 * @param override Forces the saddle choice (`true` joins the high corners) of a level.
 */
export function buildCellBandPieces(
  cornerValues: readonly number[],
  geometry: IsobandsCellGeometry,
  lo: number | null,
  hi: number | null,
  override: SaddleOverride = {}
): BandVertex[][] {
  lo = lo === null ? null : f32(lo);
  hi = hi === null ? null : f32(hi);
  const values = cornerValues.map(f32);
  const states: number[] = values.map(value =>
    lo !== null && value < lo ? 0 : hi !== null && value >= hi ? 2 : 1
  );
  if (lo !== null && hi !== null && lo >= hi) {
    return [];
  }
  const masks = [0, 1].map(level =>
    states.reduce<number>(
      (mask, state, corner) => mask | ((level === 0 ? state >= 1 : state === 2) ? 1 << corner : 0),
      0
    )
  );
  const centre = f32(f32(f32(values[0] + values[1]) + f32(values[2] + values[3])) * 0.25);
  const joined = [
    override.lo ?? (lo !== null && centre >= lo),
    override.hi ?? (hi !== null && centre >= hi)
  ];
  const {cellColumn, cellRow} = geometry;

  const corner = (index: number): BandVertex => {
    const [x, y] = getWorld(
      geometry,
      f32(cellColumn + (index === 1 || index === 2 ? 1 : 0)),
      f32(cellRow + (index >= 2 ? 1 : 0))
    );
    return {x, y, kind: 'corner', corner: index};
  };
  const crossing = (level: 0 | 1, edge: number): BandVertex => {
    const boundary = f32((level === 0 ? lo : hi) as number);
    const startValue = values[EDGE_START[edge]];
    const endValue = values[EDGE_END[edge]];
    const t = f32(f32(boundary - startValue) / f32(endValue - startValue));
    const baseX = f32(cellColumn);
    const baseY = f32(cellRow);
    let gridX: number;
    let gridY: number;
    if (edge === 0) {
      [gridX, gridY] = [f32(baseX + t), baseY];
    } else if (edge === 1) {
      [gridX, gridY] = [f32(cellColumn + 1), f32(baseY + t)];
    } else if (edge === 2) {
      [gridX, gridY] = [f32(baseX + t), f32(cellRow + 1)];
    } else {
      [gridX, gridY] = [baseX, f32(baseY + t)];
    }
    const [x, y] = getWorld(geometry, gridX, gridY);
    return {x, y, kind: 'crossing', level: level === 0 ? 'lo' : 'hi', edge};
  };

  const events: Event[] = [];
  for (let walk = 0; walk < 4; walk++) {
    const startState = states[walk];
    const endState = states[(walk + 1) % 4];
    if (startState === 1) {
      events.push({kind: 'corner', index: walk, level: 0});
    }
    if (startState < endState) {
      if (startState === 0) events.push({kind: 'entry', index: walk, level: 0});
      if (endState === 2) events.push({kind: 'exit', index: walk, level: 1});
    } else if (startState > endState) {
      if (startState === 2) events.push({kind: 'entry', index: walk, level: 1});
      if (endState === 0) events.push({kind: 'exit', index: walk, level: 0});
    }
  }
  const vertexOf = (event: Event): BandVertex =>
    event.kind === 'corner' ? corner(event.index) : crossing(event.level, event.index);

  if (events.every(event => event.kind === 'corner')) {
    return events.length === 4 ? [events.map(vertexOf)] : [];
  }

  const isCrossed = (mask: number, edge: number) =>
    ((mask >> EDGE_START[edge]) & 1) !== ((mask >> EDGE_END[edge]) & 1);
  const getPartnerEdge = (level: 0 | 1, edge: number): number => {
    const mask = masks[level];
    if (mask === 5 || mask === 10) {
      return (mask === 5) === joined[level] ? edge ^ 1 : 3 - edge;
    }
    for (let other = 0; other < 4; other++) {
      if (other !== edge && isCrossed(mask, other)) return other;
    }
    return edge;
  };

  const pieces: BandVertex[][] = [];
  const visited = new Set<number>();
  for (let start = 0; start < events.length; start++) {
    if (events[start].kind !== 'entry' || visited.has(start)) continue;
    const piece: BandVertex[] = [];
    let current = start;
    for (let arc = 0; arc < 8; arc++) {
      visited.add(current);
      let exitIndex = current;
      for (let stepIndex = 0; stepIndex < 12; stepIndex++) {
        piece.push(vertexOf(events[exitIndex]));
        if (events[exitIndex].kind === 'exit') break;
        exitIndex = (exitIndex + 1) % events.length;
      }
      const exit = events[exitIndex];
      const partnerEdge = getPartnerEdge(exit.level, exit.index);
      current = events.findIndex(
        event =>
          event.kind !== 'corner' && event.level === exit.level && event.index === partnerEdge
      );
      if (current === start || current < 0) break;
    }
    pieces.push(piece);
  }
  return pieces;
}

/** Test scene for the oracle. */
export type IsobandsScene = {
  width: number;
  height: number;
  values: Float32Array;
  validity?: Uint32Array;
  noDataValue?: number;
  breaks: Float32Array;
};

function isSampleValid(scene: IsobandsScene, sample: number): boolean {
  const value = scene.values[sample];
  return (
    !Number.isNaN(value) &&
    (scene.noDataValue === undefined || value !== f32(scene.noDataValue)) &&
    (!scene.validity || scene.validity[sample] !== 0)
  );
}

/** Oracle geometry result. */
export type IsobandsGeometry = {
  /** Interleaved x, y of every vertex (3 per triangle), capped at `capacity` triangles. */
  triangles: number[];
  /** Band per emitted triangle. */
  bands: number[];
  /** Cell index per emitted triangle. */
  cells: number[];
  /** Unclamped triangle count. */
  requiredCount: number;
};

/** Converts pieces to fan triangles `(p0, p[i], p[i + 1])`. */
export function fanTriangulate(piece: readonly BandVertex[]): BandVertex[][] {
  const triangles: BandVertex[][] = [];
  for (let i = 1; i + 1 < piece.length; i++) {
    triangles.push([piece[0], piece[i], piece[i + 1]]);
  }
  return triangles;
}

/**
 * Whole-raster band triangles in (cell, band, piece, fan) order, mirroring the GPU contributor.
 *
 * @param parameters Packed `getGPUIsobandsParameterValues` array.
 * @param capacity Triangle capacity; extra triangles only count toward `requiredCount`.
 */
export function buildIsobandTrianglesOnCPU(
  scene: IsobandsScene,
  parameters: Float32Array,
  maximumBreakCount: number,
  capacity: number = Infinity
): IsobandsGeometry {
  const breakCount = Math.min(parameters[0], maximumBreakCount);
  const firstBand = Math.min(parameters[1], maximumBreakCount);
  const lastBand = Math.min(parameters[2], maximumBreakCount, breakCount);
  const result: IsobandsGeometry = {triangles: [], bands: [], cells: [], requiredCount: 0};
  const cellColumns = scene.width - 1;
  for (let cell = 0; cell < cellColumns * (scene.height - 1); cell++) {
    const column = cell % cellColumns;
    const row = Math.floor(cell / cellColumns);
    const samples = [
      row * scene.width + column,
      row * scene.width + column + 1,
      (row + 1) * scene.width + column + 1,
      (row + 1) * scene.width + column
    ];
    if (!samples.every(sample => isSampleValid(scene, sample))) continue;
    const corners = samples.map(sample => scene.values[sample]);
    const classes = corners.map(value => countBreaksAtOrBelow(scene.breaks, breakCount, value));
    const geometry = getCellGeometry(parameters, column, row);
    for (
      let band = Math.max(Math.min(...classes), firstBand);
      band <= Math.min(Math.max(...classes), lastBand);
      band++
    ) {
      const lo = band > 0 ? scene.breaks[band - 1] : null;
      const hi = band < breakCount ? scene.breaks[band] : null;
      for (const piece of buildCellBandPieces(corners, geometry, lo, hi)) {
        for (const triangle of fanTriangulate(piece)) {
          if (result.requiredCount < capacity) {
            for (const vertex of triangle) result.triangles.push(vertex.x, vertex.y);
            result.bands.push(band);
            result.cells.push(cell);
          }
          result.requiredCount++;
        }
      }
    }
  }
  return result;
}

/** Signed area of a polygon given as vertices (positive when counter-clockwise), in float64. */
export function getPolygonArea(vertices: readonly {x: number; y: number}[]): number {
  let area = 0;
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i];
    const b = vertices[(i + 1) % vertices.length];
    area += a.x * b.y - b.x * a.y;
  }
  return area / 2;
}
