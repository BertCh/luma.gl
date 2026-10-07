// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of uint32 elements in a scan statistic parameter view. */
export const GPU_SCAN_STATISTIC_PARAMETER_LENGTH = 8;

/** Largest compile-time `maximumWindowZones` (one bit per window size in a 32-bit mask). */
export const GPU_SCAN_STATISTIC_MAXIMUM_WINDOW_ZONES = 32;

/** Largest compile-time `timeBuckets`. */
export const GPU_SCAN_STATISTIC_MAXIMUM_TIME_BUCKETS = 32;

/** Largest compile-time `maximumClusters`. */
export const GPU_SCAN_STATISTIC_MAXIMUM_CLUSTERS = 64;

/** Largest compile-time `maximumPermutations`. */
export const GPU_SCAN_STATISTIC_MAXIMUM_PERMUTATIONS = 2 ** 14;

/** Words per cluster in the `clusterIndices` output: center zone, zone count, first and last bucket. */
export const GPU_SCAN_STATISTIC_INDEX_WORDS = 4;

/** Words per cluster in the `clusterStatistics` output, see `GPU_SCAN_STATISTIC_CLUSTER`. */
export const GPU_SCAN_STATISTIC_STATISTIC_WORDS = 8;

/** Number of uint32 elements in the `summary` output. */
export const GPU_SCAN_STATISTIC_SUMMARY_LENGTH = 4;

/** Indices into one `clusterIndices` record. */
export const GPU_SCAN_STATISTIC_CLUSTER_INDEX = {
  /** Zone at the center of the circular window. */
  center: 0,
  /** Number of zones in the window: the center's `zoneCount` nearest zones, itself included. */
  zoneCount: 1,
  /** First time bucket of the cylinder, inclusive. */
  firstBucket: 2,
  /** Last time bucket of the cylinder, inclusive. */
  lastBucket: 3
} as const;

/** Indices into one `clusterStatistics` record. */
export const GPU_SCAN_STATISTIC_CLUSTER = {
  /** Poisson log-likelihood ratio of the cluster. */
  logLikelihoodRatio: 0,
  /** Observed cases in the cluster. */
  observedCases: 1,
  /** Expected cases in the cluster under the null (baseline scaled to the total case count). */
  expectedCases: 2,
  /** Monte Carlo p-value `(1 + #{replicate maxima >= LLR}) / (P + 1)`. */
  pValue: 3,
  /** Radius of the window: the distance from the center to its farthest zone. */
  radius: 4,
  /** `observedCases / expectedCases`. */
  observedOverExpected: 5
} as const;

/** Indices into the `summary` output. */
export const GPU_SCAN_STATISTIC_SUMMARY = {
  /** Clusters written to the outputs (at most `maximumClusters`). */
  clusterCount: 0,
  /** Total observed cases `C` over all zones and buckets. */
  totalCases: 1,
  /** Monte Carlo replicates run. */
  permutations: 2
} as const;

/** Spatial window shapes. */
export const GPU_SCAN_STATISTIC_WINDOW_SHAPE = {
  /** Circles: a window never splits zones at the same distance from the center. */
  circle: 0,
  /** The `k` nearest zones for every `k`, ties broken by zone index. */
  nearest: 1
} as const;

/** Window shape name of {@link GPUSpatialScanStatisticParameters}. */
export type GPUSpatialScanWindowShape = keyof typeof GPU_SCAN_STATISTIC_WINDOW_SHAPE;

/** Per-frame parameters of `GPUSpatialScanStatistic`. */
export type GPUSpatialScanStatisticParameters = {
  /** Random seed, an integer in `[0, 2^53)`. The result is a pure function of seed and inputs. */
  seed: number;
  /** Monte Carlo replicates `P`, at most the contributor's `maximumPermutations`. */
  permutations: number;
  /**
   * Largest spatial window as a fraction of the total baseline, in `(0, 1]`. A window grows until
   * adding the next zone would exceed it. Defaults to 0.5.
   */
  maximumPopulationFraction?: number;
  /** Largest number of zones per window, clamped to the contributor's `maximumWindowZones`. */
  maximumWindowZones?: number;
  /** Longest cylinder in time buckets, clamped to `timeBuckets`. Defaults to all buckets. */
  maximumTimeBuckets?: number;
  /** Window shape. Defaults to `'circle'`. */
  windowShape?: GPUSpatialScanWindowShape;
};

/**
 * Packs {@link GPUSpatialScanStatisticParameters} into the uint32 layout read by the kernels:
 * `[seedLow, seedHigh, permutations, float32 bits of maximumPopulationFraction,
 * maximumWindowZones, maximumTimeBuckets, windowShape, 0]`.
 */
export function getGPUSpatialScanParameterValues(
  parameters: GPUSpatialScanStatisticParameters
): Uint32Array {
  const {seed, permutations} = parameters;
  if (!Number.isSafeInteger(seed) || seed < 0) {
    throw new Error('scan statistic seed must be a non-negative safe integer');
  }
  if (!Number.isInteger(permutations) || permutations < 1 || permutations >= 2 ** 31) {
    throw new Error('scan statistic permutations must be a positive integer below 2^31');
  }
  const fraction = parameters.maximumPopulationFraction ?? 0.5;
  if (!(fraction > 0 && fraction <= 1)) {
    throw new Error('scan statistic maximumPopulationFraction must be in (0, 1]');
  }
  const values = new Uint32Array(GPU_SCAN_STATISTIC_PARAMETER_LENGTH);
  values[0] = seed % 2 ** 32;
  values[1] = Math.floor(seed / 2 ** 32);
  values[2] = permutations;
  values[3] = new Uint32Array(new Float32Array([fraction]).buffer)[0];
  values[4] = Math.max(1, Math.floor(parameters.maximumWindowZones ?? 2 ** 31));
  values[5] = Math.max(1, Math.floor(parameters.maximumTimeBuckets ?? 2 ** 31));
  values[6] = GPU_SCAN_STATISTIC_WINDOW_SHAPE[parameters.windowShape ?? 'circle'];
  return values;
}
