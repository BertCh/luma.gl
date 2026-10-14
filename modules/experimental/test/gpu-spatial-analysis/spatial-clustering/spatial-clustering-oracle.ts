// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Noise label and root, matching `GPU_SPATIAL_CLUSTERING_NOISE`. */
export const ORACLE_NOISE = 0xffffffff;

export type SpatialClusteringOracleParameters = {
  bounds: readonly [number, number, number, number];
  epsilon: number;
  minimumPoints: number;
};

export type SpatialClusteringOracleResult = {
  labels: number[];
  rootRows: number[];
  coreFlags: number[];
  clusterCount: number;
  /** Source ID of each cluster's root, in compact label order (unclamped). */
  clusterRoots: number[];
  /** Core plus border member count of each cluster (unclamped). */
  clusterSizes: number[];
  /** Mean member position of each cluster (unclamped), as `[x, y]` pairs. */
  clusterCentroids: number[];
};

/**
 * Brute-force O(n^2) DBSCAN with the canonical labeling documented on `GPUSpatialClustering`.
 *
 * Inputs are rounded to f32 first, as on the GPU. Distances use float64, which is exact enough
 * because tests avoid pairs near epsilon (see {@link hasNearEpsilonPair}).
 */
export function clusterPointsOracle(
  positions: ArrayLike<number>,
  parameters: SpatialClusteringOracleParameters,
  sourceIds?: ArrayLike<number>
): SpatialClusteringOracleResult {
  const count = positions.length / 2;
  const xs = Array.from({length: count}, (_, row) => Math.fround(positions[row * 2]));
  const ys = Array.from({length: count}, (_, row) => Math.fround(positions[row * 2 + 1]));
  const [minimumX, minimumY, maximumX, maximumY] = parameters.bounds.map(Math.fround);
  const epsilon = Math.fround(parameters.epsilon);
  const parametersValid =
    [minimumX, minimumY, maximumX, maximumY, epsilon].every(Number.isFinite) &&
    epsilon > 0 &&
    maximumX >= minimumX &&
    maximumY >= minimumY;
  const valid = xs.map(
    (x, row) =>
      parametersValid &&
      Number.isFinite(x) &&
      Number.isFinite(ys[row]) &&
      x >= minimumX &&
      x <= maximumX &&
      ys[row] >= minimumY &&
      ys[row] <= maximumY
  );
  const epsilonSquared = epsilon * epsilon;
  const neighbors: number[][] = Array.from({length: count}, () => []);
  for (let first = 0; first < count; first++) {
    if (!valid[first]) {
      continue;
    }
    for (let second = 0; second < count; second++) {
      if (!valid[second]) {
        continue;
      }
      const deltaX = xs[second] - xs[first];
      const deltaY = ys[second] - ys[first];
      if (deltaX * deltaX + deltaY * deltaY <= epsilonSquared) {
        neighbors[first].push(second);
      }
    }
  }
  const coreFlags = neighbors.map((list, row) =>
    valid[row] && list.length >= parameters.minimumPoints ? 1 : 0
  );

  // Connected components of core points; scanning rows ascending makes the first visited row the
  // smallest core row of its component.
  const rootRows: number[] = new Array(count).fill(ORACLE_NOISE);
  for (let start = 0; start < count; start++) {
    if (!coreFlags[start] || rootRows[start] !== ORACLE_NOISE) {
      continue;
    }
    rootRows[start] = start;
    const stack = [start];
    while (stack.length > 0) {
      const row = stack.pop() as number;
      for (const neighbor of neighbors[row]) {
        if (coreFlags[neighbor] && rootRows[neighbor] === ORACLE_NOISE) {
          rootRows[neighbor] = start;
          stack.push(neighbor);
        }
      }
    }
  }
  for (let row = 0; row < count; row++) {
    if (valid[row] && !coreFlags[row]) {
      const roots = neighbors[row].filter(neighbor => coreFlags[neighbor]).map(n => rootRows[n]);
      rootRows[row] = roots.length > 0 ? Math.min(...roots) : ORACLE_NOISE;
    }
  }

  const rankByRoot = new Map<number, number>();
  const clusterRoots: number[] = [];
  for (let row = 0; row < count; row++) {
    if (coreFlags[row] && rootRows[row] === row) {
      rankByRoot.set(row, rankByRoot.size);
      clusterRoots.push(sourceIds ? sourceIds[row] : row);
    }
  }
  const labels = rootRows.map(root =>
    root === ORACLE_NOISE ? ORACLE_NOISE : rankByRoot.get(root)!
  );
  const clusterSizes: number[] = new Array(rankByRoot.size).fill(0);
  const sums: number[] = new Array(rankByRoot.size * 2).fill(0);
  for (let row = 0; row < count; row++) {
    if (labels[row] !== ORACLE_NOISE) {
      clusterSizes[labels[row]]++;
      sums[labels[row] * 2] += xs[row];
      sums[labels[row] * 2 + 1] += ys[row];
    }
  }
  const clusterCentroids = sums.map((sum, index) => sum / clusterSizes[index >> 1]);
  return {
    labels,
    rootRows,
    coreFlags,
    clusterCount: rankByRoot.size,
    clusterRoots,
    clusterSizes,
    clusterCentroids
  };
}

/** Small deterministic generator (mulberry32) returning floats in `[0, 1)`. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Seeded points from gaussian blobs plus uniform noise, as interleaved `[x, y]` f32 values in
 * `[0, extent]^2` (blob samples are clamped into it).
 */
export function createClusteredPoints(
  seed: number,
  props: {
    pointCount: number;
    blobCount: number;
    extent: number;
    blobSigma: number;
    noiseFraction: number;
  }
): Float32Array {
  const random = createSeededRandom(seed);
  const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
  const centers = Array.from({length: props.blobCount}, () => [
    random() * props.extent * 0.8 + props.extent * 0.1,
    random() * props.extent * 0.8 + props.extent * 0.1
  ]);
  const points = new Float32Array(props.pointCount * 2);
  for (let row = 0; row < props.pointCount; row++) {
    if (random() < props.noiseFraction) {
      points[row * 2] = random() * props.extent;
      points[row * 2 + 1] = random() * props.extent;
    } else {
      const center = centers[Math.floor(random() * props.blobCount)];
      points[row * 2] = Math.min(
        Math.max(center[0] + gaussian() * props.blobSigma, 0),
        props.extent
      );
      points[row * 2 + 1] = Math.min(
        Math.max(center[1] + gaussian() * props.blobSigma, 0),
        props.extent
      );
    }
  }
  return points;
}

/**
 * Returns whether any valid pair has `|distanceSquared - epsilon^2| < 1e-3 * epsilon^2`. Such pairs
 * could flip a neighbor decision between f32 and f64 arithmetic, so tests reject those datasets.
 */
export function hasNearEpsilonPair(positions: ArrayLike<number>, epsilon: number): boolean {
  const count = positions.length / 2;
  const epsilonSquared = epsilon * epsilon;
  for (let first = 0; first < count; first++) {
    for (let second = first + 1; second < count; second++) {
      const deltaX = positions[second * 2] - positions[first * 2];
      const deltaY = positions[second * 2 + 1] - positions[first * 2 + 1];
      const distanceSquared = deltaX * deltaX + deltaY * deltaY;
      if (Math.abs(distanceSquared - epsilonSquared) < 1e-3 * epsilonSquared) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Jitters points in place until no pair sits near any of `epsilons` (see
 * {@link hasNearEpsilonPair}), so f32 and f64 arithmetic cannot disagree on a neighbor decision.
 * Points keep their coordinates to within a few percent of the smallest epsilon.
 */
export function separateNearEpsilonPairs(
  points: Float32Array,
  epsilons: readonly number[],
  seed: number = 1
): Float32Array {
  const random = createSeededRandom(seed);
  const count = points.length / 2;
  for (let pass = 0; pass < 500; pass++) {
    let adjusted = 0;
    for (const epsilon of epsilons) {
      const epsilonSquared = epsilon * epsilon;
      for (let first = 0; first < count; first++) {
        for (let second = first + 1; second < count; second++) {
          const deltaX = points[second * 2] - points[first * 2];
          const deltaY = points[second * 2 + 1] - points[first * 2 + 1];
          const distanceSquared = deltaX * deltaX + deltaY * deltaY;
          if (Math.abs(distanceSquared - epsilonSquared) < 2e-3 * epsilonSquared) {
            points[second * 2] += (random() - 0.5) * 0.06 * epsilon;
            points[second * 2 + 1] += (random() - 0.5) * 0.06 * epsilon;
            adjusted++;
          }
        }
      }
    }
    if (adjusted === 0) {
      return points;
    }
  }
  throw new Error('Could not separate near-epsilon pairs');
}

/** Mean Earth radius in meters, as `GPU_EARTH_MEAN_RADIUS_METERS`. */
export const ORACLE_EARTH_RADIUS = 6371008.8;

/** Great-circle distance in meters (haversine, f64) between `[lon, lat]` degree points. */
export function getHaversineDistance(
  a: readonly [number, number],
  b: readonly [number, number],
  radius: number = ORACLE_EARTH_RADIUS
): number {
  const toRadians = Math.PI / 180;
  const deltaLat = (b[1] - a[1]) * toRadians;
  const deltaLon = (b[0] - a[0]) * toRadians;
  const h =
    Math.sin(deltaLat / 2) ** 2 +
    Math.cos(a[1] * toRadians) * Math.cos(b[1] * toRadians) * Math.sin(deltaLon / 2) ** 2;
  return 2 * radius * Math.asin(Math.min(1, Math.sqrt(h)));
}

/**
 * Brute-force DBSCAN over `[lon, lat]` degrees with a great-circle `epsilon` in meters and the
 * same canonical labeling as {@link clusterPointsOracle}: clusters are the components of core
 * points, rooted at their smallest core row; a border point joins the adjacent cluster with the
 * smallest root; compact IDs rank roots by row. Points outside the bounds or non-finite are noise.
 */
export function clusterGeographicPointsOracle(
  positions: ArrayLike<number>,
  parameters: {
    bounds: readonly [number, number, number, number];
    epsilon: number;
    minimumPoints: number;
    radius?: number;
  }
): {labels: number[]; coreFlags: number[]; clusterCount: number} {
  const rows = positions.length / 2;
  const [minX, minY, maxX, maxY] = parameters.bounds;
  const valid = Array.from({length: rows}, (_, row) => {
    const x = positions[2 * row];
    const y = positions[2 * row + 1];
    return (
      Number.isFinite(x) && Number.isFinite(y) && x >= minX && x <= maxX && y >= minY && y <= maxY
    );
  });
  const near = (a: number, b: number) =>
    getHaversineDistance(
      [positions[2 * a], positions[2 * a + 1]],
      [positions[2 * b], positions[2 * b + 1]],
      parameters.radius
    ) <= parameters.epsilon;
  const neighbors: number[][] = Array.from({length: rows}, () => []);
  for (let a = 0; a < rows; a++) {
    if (!valid[a]) continue;
    for (let b = 0; b < rows; b++) if (valid[b] && near(a, b)) neighbors[a].push(b);
  }
  const coreFlags = neighbors.map(list => (list.length >= parameters.minimumPoints ? 1 : 0));
  const root = new Array<number>(rows).fill(-1);
  for (let start = 0; start < rows; start++) {
    if (!coreFlags[start] || root[start] >= 0) continue;
    const stack = [start];
    root[start] = start;
    while (stack.length) {
      const row = stack.pop()!;
      for (const other of neighbors[row]) {
        if (coreFlags[other] && root[other] < 0) {
          root[other] = start;
          stack.push(other);
        }
      }
    }
  }
  for (let row = 0; row < rows; row++) {
    if (!valid[row] || coreFlags[row]) continue;
    const roots = neighbors[row].filter(other => coreFlags[other]).map(other => root[other]);
    if (roots.length) root[row] = Math.min(...roots);
  }
  const rootIds = [...new Set(root.filter(value => value >= 0))].sort((a, b) => a - b);
  const labels = root.map(value => (value < 0 ? ORACLE_NOISE : rootIds.indexOf(value)));
  return {labels, coreFlags, clusterCount: rootIds.length};
}

/**
 * Deterministic longitude/latitude scene: three dense blobs (about 300 m sigma) at latitudes 0, 45
 * and 70, a cross-shaped chain, and scattered noise. Offsets in meters are converted to degrees
 * with the local `cos(latitude)`, so one meter epsilon means the same everywhere.
 */
export function createGeographicBlobScene(): Float32Array {
  const random = createSeededRandom(77);
  const points: number[] = [];
  for (const [lon, lat] of [
    [10, 0],
    [10, 45],
    [-20, 70]
  ]) {
    for (let index = 0; index < 60; index++) {
      const dx = (random() + random() + random() - 1.5) * 600;
      const dy = (random() + random() + random() - 1.5) * 600;
      points.push(
        lon + dx / ((ORACLE_EARTH_RADIUS * Math.PI * Math.cos((lat * Math.PI) / 180)) / 180),
        lat + dy / ((ORACLE_EARTH_RADIUS * Math.PI) / 180)
      );
    }
  }
  for (let index = 0; index < 20; index++) {
    points.push(-50 + random() * 20, -30 + random() * 20);
  }
  return new Float32Array(points);
}
