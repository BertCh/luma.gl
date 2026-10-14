// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The Monte Carlo envelope of the point-patterns story: the pointwise min and max of the simulated curves.
 *
 * The GPU part (re-running one compiled graph over a second positions buffer) lives in
 * `point-patterns.compute.ts`; this file accumulates read-back curves for presentation. A pointwise envelope of 39 simulations is a 2 / 40 = 5 % band: the
 * observed curve is outside it at a given radius by chance in 1 of 20 radii.
 */

/** Number of random patterns in the envelope (2 / (39 + 1) = 5 % pointwise). */
export const ENVELOPE_RUNS = 39;

/** `[minX, minY, maxX, maxY]` in local metres. */
export type PatternBounds = [number, number, number, number];

/** One curve family: `L(r) - r`, `G`, `F` or `J`. */
export type EnvelopeCurveName = 'lMinusR' | 'g' | 'f' | 'j';

/** Pointwise minimum and maximum of the simulated curves. */
export type CurveEnvelope = {low: Float32Array; high: Float32Array};

/** The four envelopes of a finished run. */
export type PatternEnvelopes = Record<EnvelopeCurveName, CurveEnvelope>;

const CURVE_NAMES: readonly EnvelopeCurveName[] = ['lMinusR', 'g', 'f', 'j'];

/**
 * Collects the curves of the simulations and reduces them to a pointwise envelope. The band is only
 * available when every run has arrived ({@link EnvelopeAccumulator.isComplete}), so a partial band is
 * never drawn as if it were final.
 */
export class EnvelopeAccumulator {
  private radiusCount = 0;
  private runs = 0;
  private readonly curves: Record<EnvelopeCurveName, Float32Array[]> = {
    lMinusR: [],
    g: [],
    f: [],
    j: []
  };

  /** Number of simulations received so far. */
  get count(): number {
    return this.runs;
  }

  /** True when {@link ENVELOPE_RUNS} simulations have been added. */
  get isComplete(): boolean {
    return this.runs >= ENVELOPE_RUNS;
  }

  /** Forgets every run and expects curves with `radiusCount` radii. */
  reset(radiusCount: number): void {
    this.radiusCount = radiusCount;
    this.runs = 0;
    for (const name of CURVE_NAMES) this.curves[name] = [];
  }

  /** Adds the curves of one simulation (copied; the arrays may be views of a readback). */
  add(curves: Record<EnvelopeCurveName, ArrayLike<number>>): void {
    for (const name of CURVE_NAMES) {
      this.curves[name].push(Float32Array.from(curves[name]));
    }
    this.runs++;
  }

  /** The pointwise envelopes, or `null` while runs are missing. NaN where no run has a value. */
  getEnvelopes(): PatternEnvelopes | null {
    if (!this.isComplete) return null;
    const result = {} as PatternEnvelopes;
    for (const name of CURVE_NAMES) {
      const low = new Float32Array(this.radiusCount).fill(Number.NaN);
      const high = new Float32Array(this.radiusCount).fill(Number.NaN);
      for (const run of this.curves[name]) {
        for (let index = 0; index < this.radiusCount; index++) {
          const value = run[index];
          if (!Number.isFinite(value)) continue;
          if (!(value >= low[index])) low[index] = value;
          if (!(value <= high[index])) high[index] = value;
        }
      }
      result[name] = {low, high};
    }
    return result;
  }
}
