// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** No-zone sentinel, matching `GPU_POLYGON_RASTERIZATION_NO_ZONE`. */
export const NO_ZONE = 0xffffffff;

/** GeoArrow-style polygon features: vertex xy pairs plus feature, polygon and ring offsets. */
export type OraclePolygons = {
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
};

/** Raster definition `[originX, originY, cellWidth, cellHeight]` plus fixed dimensions. */
export type OracleRaster = {
  width: number;
  height: number;
  extent: readonly [number, number, number, number];
};

type Edge = {x0: number; y0: number; x1: number; y1: number};

/** Builds one polygon feature collection from nested `feature -> polygon -> ring -> [x, y][]`. */
export function createPolygons(features: number[][][][][]): OraclePolygons {
  const positions: number[] = [];
  const featureOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  for (const polygons of features) {
    for (const rings of polygons) {
      for (const ring of rings) {
        for (const [x, y] of ring) {
          positions.push(x, y);
        }
        ringOffsets.push(positions.length / 2);
      }
      polygonOffsets.push(ringOffsets.length - 1);
    }
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets)
  };
}

/** Returns the edges of one polygon (every ring, implicitly closed), with float32 vertices. */
function getPolygonEdges(polygons: OraclePolygons, polygon: number): Edge[] {
  const {positions, polygonOffsets, ringOffsets} = polygons;
  const edges: Edge[] = [];
  for (let ring = polygonOffsets[polygon]; ring < polygonOffsets[polygon + 1]; ring++) {
    const start = ringOffsets[ring];
    const end = ringOffsets[ring + 1];
    for (let vertex = start; vertex < end; vertex++) {
      const next = vertex + 1 < end ? vertex + 1 : start;
      edges.push({
        x0: positions[vertex * 2],
        y0: positions[vertex * 2 + 1],
        x1: positions[next * 2],
        y1: positions[next * 2 + 1]
      });
    }
  }
  return edges.filter(edge => [edge.x0, edge.y0, edge.x1, edge.y1].every(Number.isFinite));
}

/**
 * Even-odd test with the recipe's half-open rule: an edge counts when
 * `min(y0, y1) <= y < max(y0, y1)` and its crossing is at or left of `x`.
 */
function isInsideEdges(edges: readonly Edge[], x: number, y: number): boolean {
  let inside = false;
  for (const {x0, y0, x1, y1} of edges) {
    const lowerIsFirst = y0 < y1;
    const lowerX = lowerIsFirst ? x0 : x1;
    const lowerY = lowerIsFirst ? y0 : y1;
    const upperX = lowerIsFirst ? x1 : x0;
    const upperY = lowerIsFirst ? y1 : y0;
    if (lowerY <= y && y < upperY) {
      const crossingX = lowerX + ((upperX - lowerX) * (y - lowerY)) / (upperY - lowerY);
      if (crossingX <= x) {
        inside = !inside;
      }
    }
  }
  return inside;
}

/** Distance in cells from `(x, y)` to the nearest crossing or vertex row, to flag rounding ties. */
function getTieDistance(edges: readonly Edge[], x: number, y: number, cellWidth: number): number {
  let distance = Infinity;
  for (const {x0, y0, x1, y1} of edges) {
    distance = Math.min(distance, Math.abs(y0 - y) / cellWidth, Math.abs(y1 - y) / cellWidth);
    const lowerY = Math.min(y0, y1);
    const upperY = Math.max(y0, y1);
    if (lowerY <= y && y < upperY) {
      const t = (y - y0) / (y1 - y0);
      distance = Math.min(distance, Math.abs(x0 + (x1 - x0) * t - x) / cellWidth);
    }
  }
  return distance;
}

/** Smallest feature row whose polygons contain `(x, y)`, or {@link NO_ZONE}. */
export function findContainingFeature(polygons: OraclePolygons, x: number, y: number): number {
  const featureCount = polygons.featureOffsets.length - 1;
  for (let feature = 0; feature < featureCount; feature++) {
    for (
      let polygon = polygons.featureOffsets[feature];
      polygon < polygons.featureOffsets[feature + 1];
      polygon++
    ) {
      if (isInsideEdges(getPolygonEdges(polygons, polygon), x, y)) {
        return feature;
      }
    }
  }
  return NO_ZONE;
}

/** Result of {@link rasterizePolygonsOnCPU}. */
export type OracleRasterization = {
  /** Smallest containing feature row per cell center, or {@link NO_ZONE}. */
  zones: Uint32Array;
  /** 1 where a closed edge segment touches the closed cell. */
  boundary: Uint8Array;
  /** 1 where an edge comes within `loosePadding` cells of the cell (superset of `boundary`). */
  looseBoundary: Uint8Array;
  /** 1 where a crossing or vertex lies within `tieTolerance` cells of the center (float ties). */
  ambiguous: Uint8Array;
  /** Total (edge, row) crossings inside the raster, as counted by the recipe. */
  crossingCount: number;
};

/** Returns whether the segment touches the axis-aligned box `[x0, x1] x [y0, y1]` (Liang-Barsky). */
function doesSegmentTouchBox(
  edge: Edge,
  boxX0: number,
  boxY0: number,
  boxX1: number,
  boxY1: number
): boolean {
  let t0 = 0;
  let t1 = 1;
  const dx = edge.x1 - edge.x0;
  const dy = edge.y1 - edge.y0;
  for (const [p, q] of [
    [-dx, edge.x0 - boxX0],
    [dx, boxX1 - edge.x0],
    [-dy, edge.y0 - boxY0],
    [dy, boxY1 - edge.y0]
  ]) {
    if (p === 0) {
      if (q < 0) {
        return false;
      }
    } else {
      const r = q / p;
      if (p < 0) {
        t0 = Math.max(t0, r);
      } else {
        t1 = Math.min(t1, r);
      }
      if (t0 > t1) {
        return false;
      }
    }
  }
  return true;
}

/**
 * CPU reference for `GPUPolygonRasterization`: point-in-polygon at every cell center (even-odd per
 * polygon, union per feature, smallest feature row wins), plus exact and loose boundary cells.
 */
export function rasterizePolygonsOnCPU(
  polygons: OraclePolygons,
  raster: OracleRaster,
  options: {tieTolerance?: number; loosePadding?: number} = {}
): OracleRasterization {
  const {width, height} = raster;
  const [originX, originY, cellWidth, cellHeight] = raster.extent.map(Math.fround);
  const tieTolerance = options.tieTolerance ?? 1e-4;
  const loosePadding = options.loosePadding ?? 4e-3;
  const cellCount = width * height;
  const zones = new Uint32Array(cellCount).fill(NO_ZONE);
  const boundary = new Uint8Array(cellCount);
  const looseBoundary = new Uint8Array(cellCount);
  const ambiguous = new Uint8Array(cellCount);
  let crossingCount = 0;
  const featureCount = polygons.featureOffsets.length - 1;
  const getRowThreshold = (y: number) =>
    Math.min(Math.max(Math.ceil((y - originY) / cellHeight - 0.5), 0), height);
  for (let feature = featureCount - 1; feature >= 0; feature--) {
    for (
      let polygon = polygons.featureOffsets[feature];
      polygon < polygons.featureOffsets[feature + 1];
      polygon++
    ) {
      const edges = getPolygonEdges(polygons, polygon);
      if (edges.length === 0) {
        continue;
      }
      let minimumX = Infinity;
      let minimumY = Infinity;
      let maximumX = -Infinity;
      let maximumY = -Infinity;
      for (const edge of edges) {
        minimumX = Math.min(minimumX, edge.x0, edge.x1);
        maximumX = Math.max(maximumX, edge.x0, edge.x1);
        minimumY = Math.min(minimumY, edge.y0, edge.y1);
        maximumY = Math.max(maximumY, edge.y0, edge.y1);
        const lowRow = getRowThreshold(Math.min(edge.y0, edge.y1));
        const highRow = getRowThreshold(Math.max(edge.y0, edge.y1));
        crossingCount += Math.max(highRow - lowRow, 0);
      }
      const clampColumn = (value: number) => Math.min(Math.max(value, 0), width - 1);
      const clampRow = (value: number) => Math.min(Math.max(value, 0), height - 1);
      const firstColumn = clampColumn(Math.floor((minimumX - originX) / cellWidth) - 1);
      const lastColumn = clampColumn(Math.floor((maximumX - originX) / cellWidth) + 1);
      const firstRow = clampRow(Math.floor((minimumY - originY) / cellHeight) - 1);
      const lastRow = clampRow(Math.floor((maximumY - originY) / cellHeight) + 1);
      for (let row = firstRow; row <= lastRow; row++) {
        for (let column = firstColumn; column <= lastColumn; column++) {
          const cell = row * width + column;
          const centerX = originX + (column + 0.5) * cellWidth;
          const centerY = originY + (row + 0.5) * cellHeight;
          if (isInsideEdges(edges, centerX, centerY)) {
            zones[cell] = feature;
          }
          if (getTieDistance(edges, centerX, centerY, cellWidth) < tieTolerance) {
            ambiguous[cell] = 1;
          }
          const boxX0 = originX + column * cellWidth;
          const boxY0 = originY + row * cellHeight;
          const boxX1 = boxX0 + cellWidth;
          const boxY1 = boxY0 + cellHeight;
          const padX = loosePadding * cellWidth;
          const padY = loosePadding * cellHeight;
          for (const edge of edges) {
            if (doesSegmentTouchBox(edge, boxX0, boxY0, boxX1, boxY1)) {
              boundary[cell] = 1;
            }
            if (doesSegmentTouchBox(edge, boxX0 - padX, boxY0 - padY, boxX1 + padX, boxY1 + padY)) {
              looseBoundary[cell] = 1;
            }
          }
        }
      }
    }
  }
  return {zones, boundary, looseBoundary, ambiguous, crossingCount};
}

/** Result of {@link joinPointsOnCPU}. */
export type OracleRasterJoin = {
  pointZones: Uint32Array;
  pointBoundaryMask: Uint32Array;
  counts: number[];
  sums: number[];
  boundaryCounts: number[];
  unassignedBoundaryCount: number;
  outsideCount: number;
};

/** CPU reference for `GPURasterJoin` over a given zone raster and boundary flags. */
export function joinPointsOnCPU(props: {
  raster: OracleRaster;
  zones: Uint32Array;
  boundary: ArrayLike<number>;
  zoneCount: number;
  points: Float32Array;
  values?: Float32Array;
}): OracleRasterJoin {
  const {raster, zones, boundary, zoneCount, points, values} = props;
  const [originX, originY, cellWidth, cellHeight] = raster.extent.map(Math.fround);
  const pointCount = points.length / 2;
  const result: OracleRasterJoin = {
    pointZones: new Uint32Array(pointCount).fill(NO_ZONE),
    pointBoundaryMask: new Uint32Array(pointCount),
    counts: new Array(zoneCount).fill(0),
    sums: new Array(zoneCount).fill(0),
    boundaryCounts: new Array(zoneCount).fill(0),
    unassignedBoundaryCount: 0,
    outsideCount: 0
  };
  for (let point = 0; point < pointCount; point++) {
    const x = points[point * 2];
    const y = points[point * 2 + 1];
    const column = Math.floor(Math.fround(Math.fround(x - originX) / cellWidth));
    const row = Math.floor(Math.fround(Math.fround(y - originY) / cellHeight));
    if (
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      column < 0 ||
      row < 0 ||
      column >= raster.width ||
      row >= raster.height
    ) {
      result.outsideCount++;
      continue;
    }
    const cell = row * raster.width + column;
    const zone = zones[cell] < zoneCount ? zones[cell] : NO_ZONE;
    const isBoundary = boundary[cell] !== 0;
    result.pointZones[point] = zone;
    result.pointBoundaryMask[point] = isBoundary ? 1 : 0;
    if (zone === NO_ZONE) {
      if (isBoundary) {
        result.unassignedBoundaryCount++;
      }
      continue;
    }
    result.counts[zone]++;
    if (isBoundary) {
      result.boundaryCounts[zone]++;
    }
    const value = values?.[point] ?? 0;
    if (Number.isFinite(value)) {
      result.sums[zone] += value;
    }
  }
  return result;
}
