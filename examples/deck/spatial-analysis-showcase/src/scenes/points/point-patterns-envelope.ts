// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The Monte Carlo envelope of the point-patterns story: random (CSR) points, the land they may fall
 * on, and the pointwise min and max of the simulated curves.
 *
 * The GPU part (re-running one compiled graph over a second positions buffer) lives in
 * `point-patterns.compute.ts`; this file is the CPU side that stays testable: sampling, the land
 * raster and the accumulator. A pointwise envelope of 39 simulations is a 2 / 40 = 5 % band: the
 * observed curve is outside it at a given radius by chance in 1 of 20 radii.
 */

/** Number of random patterns in the envelope (2 / (39 + 1) = 5 % pointwise). */
export const ENVELOPE_RUNS = 39;

/** `[minX, minY, maxX, maxY]` in local metres. */
export type PatternBounds = [number, number, number, number];

/** The city land inside a window, as a raster with exact tests along the boundary. */
export type LandSampler = {
  /** Share of the window that is city land (cell centres). */
  landShare: number;
  /** True when local metres `(x, y)` are city land. */
  contains: (x: number, y: number) => boolean;
};

/**
 * Rasterises `containsMeters` over a window so that rejection sampling does not run a polygon test
 * per candidate. Cells whose centre differs from a neighbour's are "mixed": a candidate that falls
 * in one is tested exactly, so the sampler stays exact where the shoreline is.
 *
 * @param containsMeters Point-in-city test in local metres.
 * @param bounds The window.
 * @param maximumCells Upper bound on raster cells (the cell edge grows to fit).
 */
export function createLandSampler(
  containsMeters: (x: number, y: number) => boolean,
  bounds: Readonly<PatternBounds>,
  maximumCells = 30000
): LandSampler {
  const width = bounds[2] - bounds[0];
  const height = bounds[3] - bounds[1];
  const cellSize = Math.max(Math.sqrt((width * height) / maximumCells), 1);
  const columns = Math.max(1, Math.ceil(width / cellSize));
  const rows = Math.max(1, Math.ceil(height / cellSize));
  const inside = new Uint8Array(columns * rows);
  let insideCount = 0;
  for (let row = 0; row < rows; row++) {
    const y = bounds[1] + (row + 0.5) * cellSize;
    for (let column = 0; column < columns; column++) {
      const x = bounds[0] + (column + 0.5) * cellSize;
      if (containsMeters(x, y)) {
        inside[row * columns + column] = 1;
        insideCount++;
      }
    }
  }
  // 0 outside, 1 inside, 2 mixed (a neighbouring centre differs).
  const state = new Uint8Array(columns * rows);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const value = inside[row * columns + column];
      let mixed = false;
      for (let dy = -1; dy <= 1 && !mixed; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const neighborRow = row + dy;
          const neighborColumn = column + dx;
          if (
            neighborRow < 0 ||
            neighborColumn < 0 ||
            neighborRow >= rows ||
            neighborColumn >= columns
          ) {
            continue;
          }
          if (inside[neighborRow * columns + neighborColumn] !== value) {
            mixed = true;
            break;
          }
        }
      }
      state[row * columns + column] = mixed ? 2 : value;
    }
  }
  return {
    landShare: insideCount / (columns * rows),
    contains: (x, y) => {
      if (x < bounds[0] || x > bounds[2] || y < bounds[1] || y > bounds[3]) return false;
      const column = Math.min(columns - 1, Math.floor((x - bounds[0]) / cellSize));
      const row = Math.min(rows - 1, Math.floor((y - bounds[1]) / cellSize));
      const cell = state[row * columns + column];
      return cell === 2 ? containsMeters(x, y) : cell === 1;
    }
  };
}

/** Candidate draws per point before a point falls back to the whole window (a window with no land). */
const MAXIMUM_ATTEMPTS = 200;

/**
 * Writes `count` complete-spatial-randomness points into `target` (interleaved `x, y`): uniform in
 * the window rectangle, or uniform on city land when a `land` sampler is given (rejection sampling:
 * a uniform candidate is kept when it is inside the city).
 *
 * @returns The share of candidates that were accepted (1 for the rectangle).
 */
export function fillRandomPositions(
  target: Float32Array,
  count: number,
  bounds: Readonly<PatternBounds>,
  land: LandSampler | null,
  random: () => number
): number {
  const width = bounds[2] - bounds[0];
  const height = bounds[3] - bounds[1];
  let candidates = 0;
  for (let index = 0; index < count; index++) {
    let x = bounds[0] + random() * width;
    let y = bounds[1] + random() * height;
    candidates++;
    if (land) {
      let attempts = 1;
      while (!land.contains(x, y) && attempts < MAXIMUM_ATTEMPTS) {
        x = bounds[0] + random() * width;
        y = bounds[1] + random() * height;
        attempts++;
        candidates++;
      }
    }
    target[index * 2] = x;
    target[index * 2 + 1] = y;
  }
  return count > 0 ? count / candidates : 1;
}

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
