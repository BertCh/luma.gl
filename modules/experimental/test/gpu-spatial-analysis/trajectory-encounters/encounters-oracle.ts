// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU reference for `GPUTrajectoryEncounters`: all-pairs per bucket in float64 over the f32
 * samples, grouped into lowest-ID ordered `(track, partner)` pairs.
 */

/** One encounter pair. */
export type EncounterOraclePair = {
  track: number;
  partner: number;
  firstBucket: number;
  minimumDistance: number;
  bucketCount: number;
};

/** CPU reference. `samples` is `[trackCount * bucketCount * 2]`; NaN marks absent samples. */
export function computeEncountersOracle(
  samples: Float32Array,
  trackCount: number,
  bucketCount: number,
  distance: number,
  trackValid?: Uint32Array,
  bounds?: readonly [number, number, number, number]
): EncounterOraclePair[] {
  const pairs: EncounterOraclePair[] = [];
  const isPresent = (track: number, bucket: number) => {
    const x = samples[2 * (track * bucketCount + bucket)];
    const y = samples[2 * (track * bucketCount + bucket) + 1];
    if (trackValid && trackValid[track] === 0) {
      return false;
    }
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      return false;
    }
    return !bounds || (x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3]);
  };
  for (let track = 0; track < trackCount; track++) {
    for (let partner = track + 1; partner < trackCount; partner++) {
      let firstBucket = -1;
      let minimumDistance = Infinity;
      let bucketCountForPair = 0;
      for (let bucket = 0; bucket < bucketCount; bucket++) {
        if (!isPresent(track, bucket) || !isPresent(partner, bucket)) {
          continue;
        }
        const a = track * bucketCount + bucket;
        const b = partner * bucketCount + bucket;
        const separation = Math.hypot(
          samples[2 * a] - samples[2 * b],
          samples[2 * a + 1] - samples[2 * b + 1]
        );
        if (separation <= distance) {
          firstBucket = firstBucket < 0 ? bucket : firstBucket;
          minimumDistance = Math.min(minimumDistance, separation);
          bucketCountForPair++;
        }
      }
      if (bucketCountForPair > 0) {
        pairs.push({
          track,
          partner,
          firstBucket,
          minimumDistance,
          bucketCount: bucketCountForPair
        });
      }
    }
  }
  return pairs;
}
