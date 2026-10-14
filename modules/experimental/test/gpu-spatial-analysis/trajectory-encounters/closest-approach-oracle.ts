// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import fixtureJson from '../../oracles/fixtures/trajectory-closest-approach.json';

/** One track of a closest-approach scene: float32-rounded positions and times. */
export type ClosestApproachTrack = {positions: [number, number][]; times: number[]};

/** One exact encounter pair; times relative to the first sample of `track`. */
export type ClosestApproachPair = {
  track: number;
  partner: number;
  minimumDistance: number;
  closestTime: number;
  firstTime: number;
  duration: number;
  /** The minimum is within 1e-3 of the distance, so f32 may decide either way. */
  isNearTangent: boolean;
};

/** Scenes of `test/oracles/trajectory-closest-approach.py`. */
export const CLOSEST_APPROACH_FIXTURE = decodeOracleFixture<{
  tolerances: {
    distanceRelative: number;
    distanceAbsolute: number;
    timeAbsolute: number;
    timeRelative: number;
  };
  scenes: {
    name: string;
    distance: number;
    tracks: ClosestApproachTrack[];
    pairs: ClosestApproachPair[];
  }[];
}>(fixtureJson);

/**
 * float64 brute-force reference over all segment pairs, the algorithm of the Python generator (a
 * second implementation, so the node spec cross-checks the two languages).
 */
export function computeClosestApproachOracle(
  tracks: readonly ClosestApproachTrack[],
  radius: number
): ClosestApproachPair[] {
  const pairs: ClosestApproachPair[] = [];
  for (let track = 0; track < tracks.length; track++) {
    for (let partner = track + 1; partner < tracks.length; partner++) {
      const a = tracks[track];
      const b = tracks[partner];
      const hits: {minimum: number; closest: number; enter: number; exit: number}[] = [];
      for (let i = 0; i + 1 < a.positions.length; i++) {
        if (!(a.times[i + 1] > a.times[i])) {
          continue;
        }
        for (let j = 0; j + 1 < b.positions.length; j++) {
          if (!(b.times[j + 1] > b.times[j])) {
            continue;
          }
          const hit = testSegmentPair(
            a.positions[i],
            a.positions[i + 1],
            a.times[i],
            a.times[i + 1],
            b.positions[j],
            b.positions[j + 1],
            b.times[j],
            b.times[j + 1],
            radius
          );
          if (hit) {
            hits.push(hit);
          }
        }
      }
      if (hits.length === 0) {
        continue;
      }
      const origin = a.times[0];
      let best = hits[0];
      for (const hit of hits) {
        if (
          hit.minimum < best.minimum ||
          (hit.minimum === best.minimum && hit.closest < best.closest)
        ) {
          best = hit;
        }
      }
      pairs.push({
        track,
        partner,
        minimumDistance: best.minimum,
        closestTime: best.closest - origin,
        firstTime: Math.min(...hits.map(hit => hit.enter)) - origin,
        duration: hits.reduce((sum, hit) => sum + (hit.exit - hit.enter), 0),
        isNearTangent: Math.abs(best.minimum - radius) < 1e-3 * Math.max(1, radius)
      });
    }
  }
  return pairs;
}

function testSegmentPair(
  a0: readonly number[],
  a1: readonly number[],
  ta0: number,
  ta1: number,
  b0: readonly number[],
  b1: readonly number[],
  tb0: number,
  tb1: number,
  radius: number
): {minimum: number; closest: number; enter: number; exit: number} | null {
  const start = Math.max(ta0, tb0);
  const end = Math.min(ta1, tb1);
  if (start > end) {
    return null;
  }
  const position = (
    p0: readonly number[],
    p1: readonly number[],
    t0: number,
    t1: number,
    t: number
  ) => {
    const fraction = (t - t0) / (t1 - t0);
    return [p0[0] + (p1[0] - p0[0]) * fraction, p0[1] + (p1[1] - p0[1]) * fraction];
  };
  const pa = position(a0, a1, ta0, ta1, start);
  const pb = position(b0, b1, tb0, tb1, start);
  const qa = position(a0, a1, ta0, ta1, end);
  const qb = position(b0, b1, tb0, tb1, end);
  const d = [pb[0] - pa[0], pb[1] - pa[1]];
  const span = end - start;
  const v = span > 0 ? [(qb[0] - qa[0] - d[0]) / span, (qb[1] - qa[1] - d[1]) / span] : [0, 0];
  const a = v[0] * v[0] + v[1] * v[1];
  const b = d[0] * v[0] + d[1] * v[1];
  const c = d[0] * d[0] + d[1] * d[1] - radius * radius;
  const closest = a > 0 ? Math.min(Math.max(-b / a, 0), span) : 0;
  const minimum = Math.hypot(d[0] + v[0] * closest, d[1] + v[1] * closest);
  let enter = 0;
  let exit = span;
  if (a > 0) {
    const discriminant = b * b - a * c;
    if (discriminant < 0) {
      return null;
    }
    const root = Math.sqrt(discriminant);
    enter = Math.max((-b - root) / a, 0);
    exit = Math.min((-b + root) / a, span);
    if (enter > exit) {
      return null;
    }
  } else if (c > 0) {
    return null;
  }
  return {minimum, closest: start + closest, enter: start + enter, exit: start + exit};
}
