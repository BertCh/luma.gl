// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Inputs of the isolines oracle. Mirrors the GPU recipe's topology and per-frame values. */
export type IsolinesScene = {
  width: number;
  height: number;
  values: Float32Array;
  /** Optional validity words; zero is nodata. */
  validity?: Uint32Array;
  noDataValue?: number;
  /** Level values (rounded to float32 here). */
  levels: readonly number[];
  /** `[minX, minY, maxX, maxY]`. */
  extent: readonly [number, number, number, number];
};

/** One oracle segment, in (cell, level, slot) order. */
export type IsolinesOracleSegment = {
  cell: number;
  level: number;
  p0: [number, number];
  p1: [number, number];
  startEdge: number;
  endEdge: number;
};

/** One stitched oracle polyline. */
export type IsolinesOraclePolyline = {
  level: number;
  closed: boolean;
  /** Segment indices in chain order, starting at the head. */
  segmentIndices: number[];
};

const f = Math.fround;

function isNoData(scene: IsolinesScene, cell: number): boolean {
  const value = scene.values[cell];
  return (
    Number.isNaN(value) ||
    (scene.noDataValue !== undefined && value === f(scene.noDataValue)) ||
    (scene.validity !== undefined && scene.validity[cell] === 0)
  );
}

/**
 * Marching squares exactly as in `stream-g/MARCHING-SQUARES.md`, with float32 rounding after
 * every operation. The GPU may contract `a * b + c` into FMA, so coordinates are compared with
 * a stated tolerance; classification, saddles, ordering and edge ids are exact.
 */
export function computeIsolinesOnCPU(
  scene: IsolinesScene,
  levelCount: number
): IsolinesOracleSegment[] {
  const {width, height, values} = scene;
  const [minX, minY, maxX, maxY] = scene.extent.map(f);
  const cellWidth = f((scene.extent[2] - scene.extent[0]) / width);
  const cellHeight = f((scene.extent[3] - scene.extent[1]) / height);
  void maxX;
  void maxY;
  const horizontalCount = height * (width - 1);
  const segments: IsolinesOracleSegment[] = [];
  const activeLevels = Math.min(levelCount, scene.levels.length);
  for (let cy = 0; cy < height - 1; cy++) {
    for (let cx = 0; cx < width - 1; cx++) {
      const cell = cy * (width - 1) + cx;
      const corners = [
        cy * width + cx,
        cy * width + cx + 1,
        (cy + 1) * width + cx + 1,
        (cy + 1) * width + cx
      ];
      if (corners.some(sample => isNoData(scene, sample))) {
        continue;
      }
      const v = corners.map(sample => values[sample]);
      const centre = f(f(f(v[0] + v[1]) + f(v[2] + v[3])) * 0.25);
      const edgeIds = [
        cy * (width - 1) + cx,
        horizontalCount + cy * width + cx + 1,
        (cy + 1) * (width - 1) + cx,
        horizontalCount + cy * width + cx
      ];
      for (let levelIndex = 0; levelIndex < activeLevels; levelIndex++) {
        const level = f(scene.levels[levelIndex]);
        const high = v.map(value => value >= level);
        const exits: number[] = [];
        const entries: number[] = [];
        for (let k = 0; k < 4; k++) {
          const startHigh = high[k];
          const endHigh = high[(k + 1) % 4];
          if (startHigh && !endHigh) exits.push(k);
          if (!startHigh && endHigh) entries.push(k);
        }
        const isJoined = exits.length < 2 || centre >= level;
        const crossing = (k: number): [number, number] => {
          const [a, b] = [
            [v[0], v[1]],
            [v[1], v[2]],
            [v[3], v[2]],
            [v[0], v[3]]
          ][k];
          const t = f(f(level - a) / f(b - a));
          const world = (gridX: number, gridY: number): [number, number] => [
            f(minX + f(f(gridX + 0.5) * cellWidth)),
            f(minY + f(f(gridY + 0.5) * cellHeight))
          ];
          if (k === 0 || k === 2) {
            return world(f(cx + t), cy + (k === 2 ? 1 : 0));
          }
          return world(cx + (k === 1 ? 1 : 0), f(cy + t));
        };
        for (const exit of exits) {
          let partner = exit;
          for (let step = 1; step < 4; step++) {
            const candidate = isJoined ? (exit + step) % 4 : (exit + 4 - step) % 4;
            if (entries.includes(candidate)) {
              partner = candidate;
              break;
            }
          }
          segments.push({
            cell,
            level: levelIndex,
            p0: crossing(exit),
            p1: crossing(partner),
            startEdge: edgeIds[exit],
            endEdge: edgeIds[partner]
          });
        }
      }
    }
  }
  return segments;
}

/**
 * Independent stitching oracle: `next(s)` is the same-level segment whose start edge is the end
 * edge of `s` (edge ids are global, so this equals the neighbour-cell rule). Open chains start at
 * segments without a predecessor, rings at their smallest index; polylines are ordered by head.
 */
export function stitchIsolinesOnCPU(segments: readonly IsolinesOracleSegment[]): {
  polylines: IsolinesOraclePolyline[];
  nextOf: number[];
  previousOf: number[];
} {
  const byStart = new Map<string, number>();
  segments.forEach((segment, index) => byStart.set(`${segment.level}:${segment.startEdge}`, index));
  const nextOf = segments.map(segment => byStart.get(`${segment.level}:${segment.endEdge}`) ?? -1);
  const previousOf = segments.map(() => -1);
  nextOf.forEach((target, index) => {
    if (target >= 0) {
      if (previousOf[target] >= 0) {
        throw new Error('Segment has two predecessors');
      }
      previousOf[target] = index;
    }
  });
  const visited = new Array<boolean>(segments.length).fill(false);
  const polylines: IsolinesOraclePolyline[] = [];
  const walk = (head: number, closed: boolean) => {
    const segmentIndices: number[] = [];
    let current = head;
    while (current >= 0 && !visited[current]) {
      visited[current] = true;
      segmentIndices.push(current);
      current = nextOf[current];
    }
    polylines.push({level: segments[head].level, closed, segmentIndices});
  };
  for (let index = 0; index < segments.length; index++) {
    if (previousOf[index] < 0) walk(index, false);
  }
  for (let index = 0; index < segments.length; index++) {
    if (!visited[index]) walk(index, true);
  }
  polylines.sort((left, right) => left.segmentIndices[0] - right.segmentIndices[0]);
  return {polylines, nextOf, previousOf};
}

/** Flattens oracle polylines into vertices and offsets using the given segment endpoints. */
export function getPolylineVertices(
  polylines: readonly IsolinesOraclePolyline[],
  segments: readonly {p0: readonly number[]; p1: readonly number[]}[]
): {vertices: number[]; offsets: number[]} {
  const vertices: number[] = [];
  const offsets = [0];
  for (const polyline of polylines) {
    for (const index of polyline.segmentIndices) {
      vertices.push(...segments[index].p0);
    }
    const first = segments[polyline.segmentIndices[0]];
    const last = segments[polyline.segmentIndices[polyline.segmentIndices.length - 1]];
    vertices.push(...(polyline.closed ? first.p0 : last.p1));
    offsets.push(vertices.length / 2);
  }
  return {vertices, offsets};
}
