// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU references for `GPUTrajectoryPlayhead` and `GPUTrajectoryResample`, written from the
 * documented rules. Float32 rounding is applied after every operation the GPU performs in f32, so
 * results match the GPU up to fused multiply-add and transcendental (atan2, sqrt, division) ulps.
 */

import {GPU_TRAJECTORY_PLAYHEAD_STATUS} from '../../../src/geospatial/trajectory-interpolation';

const fround = Math.fround;
const NO_ROW = 0xffffffff;

/** Sample times: f32 relative values or exact Int64 integers. */
export type OracleTimes =
  | {kind: 'float32'; values: Float32Array}
  | {kind: 'words'; values: readonly bigint[]};

/** Playhead: an f32 value, or an Int64 integer plus an f32 fraction in `[0, 1)`. */
export type OraclePlayhead =
  | {kind: 'float32'; value: number}
  | {kind: 'words'; integer: bigint; fraction: number};

/** Packed track rows shared by both oracles. */
export type OracleTracks = {
  positions: Float32Array;
  elevations?: Float32Array;
  times: OracleTimes;
  trackOffsets: readonly number[];
};

/** Per-track playhead result. */
export type PlayheadOracleResult = {
  positions: number[];
  elevations: number[];
  headings: number[];
  speeds: number[];
  status: number[];
  segmentRows: number[];
  segmentFractions: number[];
  activeTracks: number[];
};

/** Dense resample result, `trackCount * sampleCount` rows. */
export type ResampleOracleResult = {
  samples: number[];
  sampleElevations: number[];
  sampleTimes: number[];
};

/** `t[a] - t[b]` rounded to f32 the way the GPU computes it. */
function getRowTimeDifference(times: OracleTimes, a: number, b: number): number {
  if (times.kind === 'words') {
    return fround(Number(times.values[a] - times.values[b]));
  }
  return fround(times.values[a] - times.values[b]);
}

function createPlayheadModel(times: OracleTimes, playhead: OraclePlayhead) {
  if (times.kind === 'words') {
    if (playhead.kind !== 'words') {
      throw new Error('Word times require a word playhead');
    }
    const {integer, fraction} = playhead;
    return {
      isRowAfter: (row: number) => integer < times.values[row],
      isRowBefore: (row: number) =>
        times.values[row] < integer || (times.values[row] === integer && fraction > 0),
      getElapsed: (row: number) =>
        fround(fround(Number(integer - times.values[row])) + fround(fraction))
    };
  }
  if (playhead.kind !== 'float32') {
    throw new Error('Float32 times require a float32 playhead');
  }
  const value = fround(playhead.value);
  return {
    isRowAfter: (row: number) => times.values[row] > value,
    isRowBefore: (row: number) => times.values[row] < value,
    getElapsed: (row: number) => fround(value - times.values[row])
  };
}

function getTrackRange(tracks: OracleTracks, track: number): [number, number] {
  const rowCount = tracks.positions.length / 2;
  const start = Math.min(tracks.trackOffsets[track], rowCount);
  const end = Math.min(Math.max(tracks.trackOffsets[track + 1], start), rowCount);
  return [start, end];
}

function interpolate(start: number, end: number, fraction: number): number {
  return fraction >= 1 ? end : fround(start + fround(fround(end - start) * fraction));
}

/** CPU reference for `GPUTrajectoryPlayhead`. */
export function computePlayheadOracle(
  tracks: OracleTracks,
  playhead: OraclePlayhead,
  maxGap: number = 0
): PlayheadOracleResult {
  const {positions, elevations, times} = tracks;
  const model = createPlayheadModel(times, playhead);
  const gapLimit = fround(maxGap);
  const trackCount = tracks.trackOffsets.length - 1;
  const result: PlayheadOracleResult = {
    positions: [],
    elevations: [],
    headings: [],
    speeds: [],
    status: [],
    segmentRows: [],
    segmentFractions: [],
    activeTracks: []
  };
  for (let track = 0; track < trackCount; track++) {
    const [start, end] = getTrackRange(tracks, track);
    let status: number = GPU_TRAJECTORY_PLAYHEAD_STATUS.empty;
    let segmentRow = NO_ROW;
    let nextRow = NO_ROW;
    let fraction = 0;
    let duration = 0;
    if (end > start) {
      const lastRow = end - 1;
      const lastSegmentRow = lastRow > start ? lastRow - 1 : start;
      if (model.isRowAfter(start)) {
        status = GPU_TRAJECTORY_PLAYHEAD_STATUS.beforeStart;
        segmentRow = start;
        nextRow = Math.min(start + 1, lastRow);
      } else if (model.isRowBefore(lastRow)) {
        status = GPU_TRAJECTORY_PLAYHEAD_STATUS.afterEnd;
        segmentRow = lastSegmentRow;
        nextRow = lastRow;
        fraction = 1;
      } else {
        // Linear upper bound: first row after the playhead (the GPU uses binary search).
        let upper = start + 1;
        while (upper < end && !model.isRowAfter(upper)) {
          upper++;
        }
        status = GPU_TRAJECTORY_PLAYHEAD_STATUS.active;
        if (upper === end) {
          segmentRow = lastSegmentRow;
          nextRow = lastRow;
          fraction = 1;
        } else {
          segmentRow = upper - 1;
          nextRow = upper;
          const interval = getRowTimeDifference(times, nextRow, segmentRow);
          fraction = Math.min(Math.max(fround(model.getElapsed(segmentRow) / interval), 0), 1);
          if (gapLimit > 0 && interval > gapLimit && model.isRowBefore(segmentRow)) {
            status = GPU_TRAJECTORY_PLAYHEAD_STATUS.gap;
            fraction = 0;
          }
        }
      }
      duration = getRowTimeDifference(times, nextRow, segmentRow);
    }
    let x = 0;
    let y = 0;
    let z = 0;
    let heading = 0;
    let speed = 0;
    if (status !== GPU_TRAJECTORY_PLAYHEAD_STATUS.empty) {
      const deltaX = fround(positions[2 * nextRow] - positions[2 * segmentRow]);
      const deltaY = fround(positions[2 * nextRow + 1] - positions[2 * segmentRow + 1]);
      x = interpolate(positions[2 * segmentRow], positions[2 * nextRow], fraction);
      y = interpolate(positions[2 * segmentRow + 1], positions[2 * nextRow + 1], fraction);
      if (elevations) {
        z = interpolate(elevations[segmentRow], elevations[nextRow], fraction);
      }
      if (deltaX !== 0 || deltaY !== 0) {
        heading = fround(Math.atan2(deltaY, deltaX));
      }
      if (status === GPU_TRAJECTORY_PLAYHEAD_STATUS.active && duration > 0) {
        const distance = fround(
          Math.sqrt(fround(fround(deltaX * deltaX) + fround(deltaY * deltaY)))
        );
        speed = fround(distance / duration);
      }
    }
    result.positions.push(x, y);
    result.elevations.push(z);
    result.headings.push(heading);
    result.speeds.push(speed);
    result.status.push(status);
    result.segmentRows.push(segmentRow);
    result.segmentFractions.push(fraction);
    if (status === GPU_TRAJECTORY_PLAYHEAD_STATUS.active) {
      result.activeTracks.push(track);
    }
  }
  return result;
}

/** Moves a non-negative f32 value by `ulps` units in the last place (clamped at 0). */
function shiftUlps(value: number, ulps: number): number {
  if (ulps === 0 || value <= 0) {
    return value;
  }
  const bits = new Uint32Array(new Float32Array([value]).buffer);
  bits[0] = Math.max(bits[0] + ulps, 0);
  return new Float32Array(bits.buffer)[0];
}

/**
 * CPU reference for `GPUTrajectoryResample`.
 *
 * @param targetUlpOffset Shifts every interior sample target by this many f32 ulps. The GPU's
 * `k / (sampleCount - 1)` division is only accurate to 2.5 ulps, which matters only where the
 * resampled path is discontinuous (duplicate timestamps, stationary steps in arc length); tests
 * accept a GPU sample that matches the oracle at any small offset.
 */
export function computeResampleOracle(
  tracks: OracleTracks,
  sampleCount: number,
  spacing: 'time' | 'arc-length' = 'time',
  targetUlpOffset: number = 0
): ResampleOracleResult {
  const {positions, elevations, times} = tracks;
  const rowCount = positions.length / 2;
  const cumulative = new Float32Array(rowCount);
  const trackCount = tracks.trackOffsets.length - 1;
  const result: ResampleOracleResult = {
    samples: [],
    sampleElevations: [],
    sampleTimes: []
  };
  for (let track = 0; track < trackCount; track++) {
    const [start, end] = getTrackRange(tracks, track);
    if (spacing === 'arc-length' && end > start) {
      let total = 0;
      cumulative[start] = 0;
      for (let row = start + 1; row < end; row++) {
        const deltaX = fround(positions[2 * row] - positions[2 * row - 2]);
        const deltaY = fround(positions[2 * row + 1] - positions[2 * row - 1]);
        total = fround(
          total + fround(Math.sqrt(fround(fround(deltaX * deltaX) + fround(deltaY * deltaY))))
        );
        cumulative[row] = total;
      }
    }
    const getProgress = (row: number) =>
      spacing === 'arc-length' ? cumulative[row] : getRowTimeDifference(times, row, start);
    for (let sample = 0; sample < sampleCount; sample++) {
      if (end <= start) {
        result.samples.push(0, 0);
        result.sampleElevations.push(0);
        result.sampleTimes.push(0);
        continue;
      }
      const lastRow = end - 1;
      const total = getProgress(lastRow);
      let target = 0;
      if (sampleCount > 1) {
        target =
          sample === sampleCount - 1
            ? total
            : shiftUlps(fround(total * fround(sample / (sampleCount - 1))), targetUlpOffset);
      }
      let upper = start + 1;
      while (upper < end && !(getProgress(upper) > target)) {
        upper++;
      }
      let row0 = lastRow;
      let row1 = lastRow;
      let fraction = 1;
      if (upper < end) {
        row0 = upper - 1;
        row1 = upper;
        const startProgress = getProgress(row0);
        fraction = Math.min(
          Math.max(
            fround(fround(target - startProgress) / fround(getProgress(row1) - startProgress)),
            0
          ),
          1
        );
      }
      result.samples.push(
        interpolate(positions[2 * row0], positions[2 * row1], fraction),
        interpolate(positions[2 * row0 + 1], positions[2 * row1 + 1], fraction)
      );
      result.sampleElevations.push(
        elevations ? interpolate(elevations[row0], elevations[row1], fraction) : 0
      );
      result.sampleTimes.push(
        spacing === 'arc-length'
          ? interpolate(
              getRowTimeDifference(times, row0, start),
              getRowTimeDifference(times, row1, start),
              fraction
            )
          : target
      );
    }
  }
  return result;
}
