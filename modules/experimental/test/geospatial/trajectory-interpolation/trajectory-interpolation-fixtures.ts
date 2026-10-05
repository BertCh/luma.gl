// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OracleTracks} from './trajectory-interpolation-oracle';

/** One sample: `[x, y, time]`. */
export type FixtureRow = [x: number, y: number, time: number];

/**
 * Edge-case tracks shared by the node and GPU specs:
 *
 * 0. regular square-ish path, times 0, 10, 20, 40
 * 1. duplicate timestamps in the middle (three rows at t=5) and at the end (two rows at t=20)
 * 2. a single sample at t=12
 * 3. empty
 * 4. a 100-unit gap between t=1 and t=101
 * 5. starts late (t=100..110)
 * 6. empty, as the last track
 */
export const EDGE_CASE_TRACKS: FixtureRow[][] = [
  [
    [0, 0, 0],
    [10, 0, 10],
    [10, 20, 20],
    [0, 20, 40]
  ],
  [
    [0, 0, 0],
    [5, 5, 5],
    [7, 7, 5],
    [9, 9, 5],
    [20, 9, 15],
    [21, 9, 20],
    [22, 9, 20]
  ],
  [[3, 4, 12]],
  [],
  [
    [0, 0, 0],
    [1, 0, 1],
    [101, 0, 101],
    [102, 0, 102]
  ],
  [
    [50, 50, 100],
    [60, 50, 110]
  ],
  []
];

/** Packs fixture rows into f32 positions, elevations (`row * 0.5`), times, and offsets. */
export function packTracks(
  trackRows: FixtureRow[][],
  wordBase?: bigint
): OracleTracks & {floatTimes: Float32Array; wordTimes: bigint[]} {
  const positions: number[] = [];
  const times: number[] = [];
  const trackOffsets = [0];
  for (const rows of trackRows) {
    for (const [x, y, time] of rows) {
      positions.push(x, y);
      times.push(time);
    }
    trackOffsets.push(times.length);
  }
  const floatTimes = Float32Array.from(times);
  const wordTimes = times.map(time => (wordBase ?? 0n) + BigInt(time));
  return {
    positions: Float32Array.from(positions),
    elevations: Float32Array.from(times, (_, row) => row * 0.5),
    times:
      wordBase === undefined
        ? {kind: 'float32', values: floatTimes}
        : {kind: 'words', values: wordTimes},
    trackOffsets,
    floatTimes,
    wordTimes
  };
}

/** Deterministic xorshift in [0, 1). */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/**
 * Random tracks with integer times (so f32 and Int64 words describe the same instants), frequent
 * duplicate timestamps, zero-length steps, and some empty and single-row tracks.
 */
export function createRandomTracks(seed: number, trackCount: number): FixtureRow[][] {
  const random = createRandom(seed);
  const tracks: FixtureRow[][] = [];
  for (let track = 0; track < trackCount; track++) {
    const roll = random();
    const rowCount = roll < 0.08 ? 0 : roll < 0.15 ? 1 : 2 + Math.floor(random() * 60);
    let time = Math.floor(random() * 200);
    let x = Math.round(random() * 1000);
    let y = Math.round(random() * 1000);
    const rows: FixtureRow[] = [];
    for (let row = 0; row < rowCount; row++) {
      rows.push([x, y, time]);
      // 20% duplicate times, 10% stationary steps, up to 40-unit gaps.
      time += random() < 0.2 ? 0 : 1 + Math.floor(random() * 40);
      if (random() >= 0.1) {
        x += Math.round((random() - 0.5) * 40);
        y += Math.round((random() - 0.5) * 40);
      }
    }
    tracks.push(rows);
  }
  return tracks;
}
