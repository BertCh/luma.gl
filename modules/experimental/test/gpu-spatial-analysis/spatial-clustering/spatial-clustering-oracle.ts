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
