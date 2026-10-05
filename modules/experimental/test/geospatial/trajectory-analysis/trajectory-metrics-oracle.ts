// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Packed track rows: positions are `x, y` pairs, tracks are delimited by `trackOffsets`. */
export type TrajectoryTracks = {
  positions: Float32Array;
  timestamps: Float32Array;
  trackOffsets: number[];
};

/** One expected stop. */
export type TrajectoryOracleStop = {
  track: number;
  startRow: number;
  endRow: number;
  duration: number;
  centroid: [number, number];
};

/** Expected per-track metrics and the full (unclamped) stop list in row order. */
export type TrajectoryOracleResult = {
  trackLengths: number[];
  trackDurations: number[];
  averageSpeeds: number[];
  maximumSpeeds: number[];
  trackStopCounts: number[];
  stops: TrajectoryOracleStop[];
};

/**
 * Exact time model for the oracle: `difference(a, b)` is `t[a] - t[b]` for two rows computed
 * exactly (before the f32 rounding the GPU applies) and `isNonDecreasing(a, b)` is exact `t[a] >= t[b]`.
 */
export type TrajectoryOracleTime = {
  difference: (rowA: number, rowB: number) => number;
  isNonDecreasing: (rowA: number, rowB: number) => boolean;
};

const fround = Math.fround;

/** Exact Int64 time model: BigInt subtraction, then converted to a double. */
export function createBigIntOracleTime(times: readonly bigint[]): TrajectoryOracleTime {
  return {
    difference: (rowA, rowB) => Number(times[rowA] - times[rowB]),
    isNonDecreasing: (rowA, rowB) => times[rowA] >= times[rowB]
  };
}

/** Double time model for double-single data: doubles hold epoch-ms integers exactly. */
export function createDoubleOracleTime(times: readonly number[]): TrajectoryOracleTime {
  return {
    difference: (rowA, rowB) => times[rowA] - times[rowB],
    isNonDecreasing: (rowA, rowB) => times[rowA] >= times[rowB]
  };
}

/**
 * CPU reference for `GPUTrajectoryMetrics`, written directly from the documented definitions.
 * Float32 rounding is applied after every operation that the GPU performs in float32.
 */
export function computeTrajectoryOracle(
  tracks: TrajectoryTracks,
  stopSpeedThreshold: number,
  stopMinimumDuration: number,
  time?: TrajectoryOracleTime
): TrajectoryOracleResult {
  const {positions, timestamps, trackOffsets} = tracks;
  const difference = time?.difference ?? ((rowA, rowB) => timestamps[rowA] - timestamps[rowB]);
  const isNonDecreasing =
    time?.isNonDecreasing ?? ((rowA, rowB) => timestamps[rowA] >= timestamps[rowB]);
  const threshold = fround(stopSpeedThreshold);
  const minimumDuration = fround(stopMinimumDuration);
  const trackCount = trackOffsets.length - 1;
  const result: TrajectoryOracleResult = {
    trackLengths: [],
    trackDurations: [],
    averageSpeeds: [],
    maximumSpeeds: [],
    trackStopCounts: new Array(trackCount).fill(0),
    stops: []
  };
  for (let track = 0; track < trackCount; track++) {
    const first = trackOffsets[track];
    const end = trackOffsets[track + 1];
    let trackLength = 0;
    let maximumSpeed = 0;
    const distances: number[] = [];
    const deltaTimes: number[] = [];
    const forward: boolean[] = [];
    for (let row = first + 1; row < end; row++) {
      const deltaX = fround(positions[2 * row] - positions[2 * row - 2]);
      const deltaY = fround(positions[2 * row + 1] - positions[2 * row - 1]);
      const distance = fround(Math.sqrt(fround(fround(deltaX * deltaX) + fround(deltaY * deltaY))));
      const deltaTime = fround(difference(row, row - 1));
      distances.push(distance);
      deltaTimes.push(deltaTime);
      forward.push(isNonDecreasing(row, row - 1));
      trackLength = fround(trackLength + distance);
      if (deltaTime > 0) {
        maximumSpeed = Math.max(maximumSpeed, fround(distance / deltaTime));
      }
    }
    const duration = end - first >= 2 ? fround(difference(end - 1, first)) : 0;
    result.trackLengths.push(trackLength);
    result.trackDurations.push(duration);
    result.averageSpeeds.push(duration > 0 ? fround(trackLength / duration) : 0);
    result.maximumSpeeds.push(maximumSpeed);

    // slow[k] describes step row first + 1 + k.
    const slow = distances.map(
      (distance, step) =>
        forward[step] && (distance === 0 || distance < fround(threshold * deltaTimes[step]))
    );
    let step = 0;
    while (step < slow.length) {
      if (!slow[step]) {
        step++;
        continue;
      }
      let runEnd = step;
      while (runEnd + 1 < slow.length && slow[runEnd + 1]) {
        runEnd++;
      }
      const startRow = first + step; // a - 1 where a = first + 1 + step
      const endRow = first + 1 + runEnd;
      const dwellDuration = fround(difference(endRow, startRow));
      if (dwellDuration >= minimumDuration) {
        let sumX = 0;
        let sumY = 0;
        for (let row = startRow; row <= endRow; row++) {
          sumX += positions[2 * row];
          sumY += positions[2 * row + 1];
        }
        const rowCount = endRow - startRow + 1;
        result.stops.push({
          track,
          startRow,
          endRow,
          duration: dwellDuration,
          centroid: [sumX / rowCount, sumY / rowCount]
        });
        result.trackStopCounts[track]++;
      }
      step = runEnd + 1;
    }
  }
  return result;
}

/** Small deterministic PRNG returning floats in `[0, 1)`. */
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

/** Builds tracks from `[x, y, time]` rows, one array per track. */
export function createTracksFromRows(
  trackRows: [number, number, number][][],
  leadingRowCount: number = 0
): TrajectoryTracks {
  const positions: number[] = new Array(2 * leadingRowCount).fill(7);
  const timestamps: number[] = new Array(leadingRowCount).fill(3);
  const trackOffsets = [leadingRowCount];
  for (const rows of trackRows) {
    for (const [x, y, time] of rows) {
      positions.push(x, y);
      timestamps.push(time);
    }
    trackOffsets.push(timestamps.length);
  }
  return {
    positions: Float32Array.from(positions),
    timestamps: Float32Array.from(timestamps),
    trackOffsets
  };
}

/**
 * Generates tracks of alternating moving and dwell segments. Moving steps run at 3x to 6x the
 * threshold and dwell steps at 0 to 0.3x (some exactly zero), so no step speed is within 20% of
 * `threshold`. Timestamps advance by integers, which keeps durations exact in float32. The track
 * lengths include empty tracks and single-row tracks.
 */
export function generateTrajectoryTracks(
  seed: number,
  trackCount: number,
  minimumRows: number,
  maximumRows: number,
  threshold: number
): TrajectoryTracks {
  const random = createSeededRandom(seed);
  const trackRows: [number, number, number][][] = [];
  for (let track = 0; track < trackCount; track++) {
    const kind = random();
    const rowCount =
      kind < 0.08
        ? 0
        : kind < 0.16
          ? 1
          : minimumRows + Math.floor(random() * (maximumRows - minimumRows + 1));
    const rows: [number, number, number][] = [];
    let x = (random() - 0.5) * 400;
    let y = (random() - 0.5) * 400;
    let time = Math.floor(random() * 50);
    let isDwelling = random() < 0.5;
    let remainingInSegment = 0;
    for (let row = 0; row < rowCount; row++) {
      if (row > 0) {
        if (remainingInSegment === 0) {
          isDwelling = !isDwelling;
          remainingInSegment = 1 + Math.floor(random() * 8);
        }
        remainingInSegment--;
        const deltaTime = 1 + Math.floor(random() * 3);
        const speed = isDwelling
          ? random() < 0.3
            ? 0
            : random() * 0.3 * threshold
          : (3 + random() * 3) * threshold;
        const angle = random() * 2 * Math.PI;
        x += Math.cos(angle) * speed * deltaTime;
        y += Math.sin(angle) * speed * deltaTime;
        time += deltaTime;
      }
      rows.push([x, y, time]);
    }
    trackRows.push(rows);
  }
  return createTracksFromRows(trackRows);
}
