// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Result of {@link computeKMeansOracle}. */
export type KMeansOracleResult = {
  labels: number[];
  centers: number[];
  sizes: number[];
  squaredDistances: number[];
  /** Total inertia: sum of the finite squared distances. NaN with no valid points. */
  inertia: number;
};

const NOISE = 0xffffffff;

/**
 * f64 Lloyd oracle with the same fixed rules as `GPUKMeans` for `'first-valid'` initialization:
 * first `k` finite points as centers, nearest center with lowest-ID ties, scikit-learn's
 * empty-cluster relocation inside every update (see {@link relocateEmptyClusters}), a final
 * assignment after the last update.
 */
export function computeKMeansOracle(
  positions: Float32Array,
  k: number,
  iterations: number
): KMeansOracleResult {
  const rows = positions.length / 2;
  const initialCenters = new Array<number>(2 * k).fill(NaN);
  let found = 0;
  for (let row = 0; row < rows && found < k; row++) {
    if (Number.isFinite(positions[2 * row]) && Number.isFinite(positions[2 * row + 1])) {
      initialCenters[2 * found] = positions[2 * row];
      initialCenters[2 * found + 1] = positions[2 * row + 1];
      found++;
    }
  }
  return computeKMeansOracleFromCenters(positions, initialCenters, iterations);
}

/**
 * scikit-learn 1.9.1 `_relocate_empty_clusters_dense`, edited in place on the per-cluster `sums` and
 * `counts` taken before the means. The empty clusters (count 0, finite center) in ascending ID take
 * the points with the largest squared distance to their assigned center, farthest first and lowest
 * row among equal distances (scikit-learn's `argpartition` leaves both orders unspecified). Nothing
 * happens when the largest distance is 0. Donor sums and counts lose the moved points.
 */
export function relocateEmptyClusters(
  positions: Float32Array,
  centers: readonly number[],
  labels: readonly number[],
  squaredDistances: readonly number[],
  sums: number[],
  counts: number[]
): void {
  const empty: number[] = [];
  for (let cluster = 0; cluster < counts.length; cluster++) {
    if (counts[cluster] === 0 && Number.isFinite(centers[2 * cluster])) empty.push(cluster);
  }
  if (empty.length === 0) return;
  const rows = Array.from({length: labels.length}, (_, row) => row).filter(
    row => labels[row] !== NOISE && Number.isFinite(squaredDistances[row])
  );
  rows.sort((a, b) => squaredDistances[b] - squaredDistances[a] || a - b);
  if (rows.length === 0 || squaredDistances[rows[0]] === 0) return;
  for (let index = 0; index < Math.min(empty.length, rows.length); index++) {
    const row = rows[index];
    const donor = labels[row];
    sums[2 * donor] -= positions[2 * row];
    sums[2 * donor + 1] -= positions[2 * row + 1];
    counts[donor]--;
    sums[2 * empty[index]] = positions[2 * row];
    sums[2 * empty[index] + 1] = positions[2 * row + 1];
    counts[empty[index]] = 1;
  }
}

/**
 * f64 Lloyd run from caller-supplied initial centers (`2 * k` numbers), with the same assignment,
 * empty-cluster relocation and final-assignment rules as {@link computeKMeansOracle}. A NaN initial
 * center is never relocated (an uninitialized cluster).
 */
export function computeKMeansOracleFromCenters(
  positions: Float32Array,
  initialCenters: readonly number[],
  iterations: number
): KMeansOracleResult {
  const rows = positions.length / 2;
  const k = initialCenters.length / 2;
  const valid = (row: number) =>
    Number.isFinite(positions[2 * row]) && Number.isFinite(positions[2 * row + 1]);
  const centers = Array.from(initialCenters);
  const labels = new Array<number>(rows).fill(NOISE);
  const squaredDistances = new Array<number>(rows).fill(NaN);
  const assign = () => {
    for (let row = 0; row < rows; row++) {
      labels[row] = NOISE;
      squaredDistances[row] = NaN;
      if (!valid(row)) continue;
      for (let center = 0; center < k; center++) {
        const distance =
          (centers[2 * center] - positions[2 * row]) ** 2 +
          (centers[2 * center + 1] - positions[2 * row + 1]) ** 2;
        if (distance === distance && (labels[row] === NOISE || distance < squaredDistances[row])) {
          labels[row] = center;
          squaredDistances[row] = distance;
        }
      }
    }
  };
  for (let iteration = 0; iteration < iterations; iteration++) {
    assign();
    const sums = new Array<number>(2 * k).fill(0);
    const counts = new Array<number>(k).fill(0);
    for (let row = 0; row < rows; row++) {
      if (labels[row] !== NOISE) {
        sums[2 * labels[row]] += positions[2 * row];
        sums[2 * labels[row] + 1] += positions[2 * row + 1];
        counts[labels[row]]++;
      }
    }
    relocateEmptyClusters(positions, centers, labels, squaredDistances, sums, counts);
    for (let center = 0; center < k; center++) {
      if (counts[center] > 0) {
        centers[2 * center] = sums[2 * center] / counts[center];
        centers[2 * center + 1] = sums[2 * center + 1] / counts[center];
      } else if (centers[2 * center] === centers[2 * center]) {
        // Still empty (no relocation possible): the mean of the largest cluster, lowest ID on ties.
        let largest = 0;
        for (let other = 1; other < k; other++)
          if (counts[other] > counts[largest]) largest = other;
        if (counts[largest] > 0) {
          centers[2 * center] = sums[2 * largest] / counts[largest];
          centers[2 * center + 1] = sums[2 * largest + 1] / counts[largest];
        }
      }
    }
  }
  assign();
  const sizes = new Array<number>(k).fill(0);
  for (const label of labels) if (label !== NOISE) sizes[label]++;
  let inertia = NaN;
  for (const distance of squaredDistances) {
    if (Number.isFinite(distance)) inertia = (Number.isNaN(inertia) ? 0 : inertia) + distance;
  }
  return {labels, centers, sizes, squaredDistances, inertia};
}

/**
 * The `GPUKMeans` restart rule: the run with the lowest total inertia wins, strictly, so ties go
 * to the lowest restart index; a finite inertia replaces a NaN one. Returns the winning index.
 */
export function selectBestKMeansRestart(inertias: readonly number[]): number {
  let best = 0;
  for (let restart = 1; restart < inertias.length; restart++) {
    const candidate = inertias[restart];
    const current = inertias[best];
    if (candidate < current || (Number.isNaN(current) && !Number.isNaN(candidate))) {
      best = restart;
    }
  }
  return best;
}

/**
 * Deterministic scene of `blobs` Gaussian-like blobs (sum of three uniforms, sigma 6) centered on a
 * ring of radius 400, `perBlob` points each, built with a fixed LCG. Blobs are so far apart that
 * every sensible initialization converges to the same partition, which makes the final inertia
 * init-independent and comparable to scikit-learn.
 */
export function createSeparatedBlobScene(blobs: number, perBlob: number): Float32Array {
  let state = 12345;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const points: number[] = [];
  for (let blob = 0; blob < blobs; blob++) {
    const angle = (2 * Math.PI * blob) / blobs;
    const centerX = 400 * Math.cos(angle);
    const centerY = 400 * Math.sin(angle);
    for (let index = 0; index < perBlob; index++) {
      const gaussian = () => random() + random() + random() - 1.5;
      points.push(centerX + gaussian() * 6, centerY + gaussian() * 6);
    }
  }
  return new Float32Array(points);
}

/** One frozen scikit-learn run of the empty-cluster relocation, per `max_iter` step count. */
export type SklearnRelocationFixture = {
  name: string;
  k: number;
  /** Interleaved x, y. The first `k` rows are the initial centers (also GPUKMeans 'first-valid'). */
  positions: number[];
  steps: Record<'1' | '2' | '3' | '50', {centers: number[]; labels: number[]; inertia: number}>;
};

// Oracle: scikit-learn 1.9.1 KMeans(n_clusters=k, init=<first k rows>, n_init=1, algorithm='lloyd',
// max_iter=m, tol=0), generated by fix/work/W2A/kmeans/kmeans-relocation-sklearn.py. The duplicated
// initial centers leave clusters empty after the first assignment (lowest ID wins ties), which
// scikit-learn relocates to the farthest points. Integer coordinates keep f32 and f64 identical.
export const SKLEARN_RELOCATION_FIXTURES: SklearnRelocationFixture[] = [
  {
    name: 'one-empty',
    k: 3,
    positions: [
      0, 0, 0, 0, 40, 30, 7, 2, 0, -7, -1, -1, 0, -3, 0, -2, -5, 4, 4, 7, 0, -2, -2, -6, 4, -4, -5,
      -1, 6, 1, 36, 27, 43, 29, 37, 29, 43, 38, 35, 27, 37, 20, 36, 26, 44, 29, 34, 33, 39, 23, 38,
      28, 37, 30, -37, 29, 55, -48
    ],
    steps: {
      1: {
        centers: [
          -1.9333333333333265, 1.1333333333333275, 55, -48, 38.38461538461539, 28.38461538461538
        ],
        labels: [
          0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1
        ],
        inertia: 2850.8205128205127
      },
      2: {
        centers: [
          -1.93333333333333, 1.1333333333333275, 55, -48, 38.38461538461539, 28.38461538461538
        ],
        labels: [
          0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1
        ],
        inertia: 2850.8205128205127
      },
      3: {
        centers: [
          -1.93333333333333, 1.1333333333333275, 55, -48, 38.38461538461539, 28.38461538461538
        ],
        labels: [
          0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1
        ],
        inertia: 2850.8205128205127
      },
      50: {
        centers: [
          -1.93333333333333, 1.1333333333333275, 55, -48, 38.38461538461539, 28.38461538461538
        ],
        labels: [
          0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1
        ],
        inertia: 2850.8205128205127
      }
    }
  },
  {
    name: 'two-empty',
    k: 4,
    positions: [
      0, 0, 0, 0, 0, 0, 40, 30, 7, 2, 0, -7, -1, -1, 0, -3, 0, -2, -5, 4, 4, 7, 0, -2, -2, -6, 4,
      -4, -5, -1, 6, 1, 36, 27, 43, 29, 37, 29, 43, 38, 35, 27, 37, 20, 36, 26, 44, 29, 34, 33, 39,
      23, 38, 28, 37, 30, -37, 29, 55, -48
    ],
    steps: {
      1: {
        centers: [
          0.533333333333335, -0.8000000000000025, 55, -48, -37, 29, 38.38461538461539,
          28.384615384615387
        ],
        labels: [
          0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1
        ],
        inertia: 702.2871794871795
      },
      2: {
        centers: [
          0.533333333333335, -0.8000000000000007, 55, -48, -37, 29, 38.38461538461539,
          28.384615384615387
        ],
        labels: [
          0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1
        ],
        inertia: 702.2871794871796
      },
      3: {
        centers: [
          0.533333333333335, -0.8000000000000007, 55, -48, -37, 29, 38.38461538461539,
          28.384615384615387
        ],
        labels: [
          0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1
        ],
        inertia: 702.2871794871796
      },
      50: {
        centers: [
          0.533333333333335, -0.8000000000000007, 55, -48, -37, 29, 38.38461538461539,
          28.384615384615387
        ],
        labels: [
          0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1
        ],
        inertia: 702.2871794871796
      }
    }
  }
];
