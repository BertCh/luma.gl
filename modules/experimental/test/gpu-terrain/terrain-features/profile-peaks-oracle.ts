// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const f32 = Math.fround;

/** Options of {@link findProfilePeaksOracle}. */
export type ProfilePeaksOracleOptions = {
  window: number;
  minProminence: number;
  minSide?: number;
  nms?: number;
  wrap?: boolean;
};

/** One kept peak, in profile-local coordinates. */
export type ProfilePeakOracleResult = {
  /** Profile index. */
  profile: number;
  /** Global sample row. */
  row: number;
  /** Local sample index inside the profile. */
  local: number;
  /** Refined fractional local index (wrapped into `[0, n)` in wrap mode). */
  index: number;
  /** Refined value. */
  value: number;
  /** Prominence, one f32 subtraction of f32 values (bit-exact with the GPU). */
  prominence: number;
};

/** Per-sample arrays plus the ordered peak list. */
export type ProfilePeaksOracleResult = {
  peaks: ProfilePeakOracleResult[];
  /** NaN where not a kept peak. */
  prominence: Float64Array;
  refinedIndex: Float64Array;
  refinedValue: Float64Array;
  mask: Uint32Array;
};

/**
 * Float64 CPU oracle of `GPUProfilePeaks`: the mt-image `profilePeaks` reference (sequential greedy
 * suppression) applied to every CSR profile, plus the circular `wrap` extension. Gaps (validity 0
 * or non-finite value) are skipped before any value is read.
 */
export function findProfilePeaksOracle(
  values: Float32Array,
  validity: Uint32Array | undefined,
  offsets: Uint32Array | number[],
  options: ProfilePeaksOracleOptions
): ProfilePeaksOracleResult {
  const minSide = options.minSide ?? 4;
  const nms = options.nms ?? Math.max(2, Math.round(options.window / 4));
  const wrap = options.wrap ?? false;
  const result: ProfilePeaksOracleResult = {
    peaks: [],
    prominence: new Float64Array(values.length).fill(Number.NaN),
    refinedIndex: new Float64Array(values.length).fill(Number.NaN),
    refinedValue: new Float64Array(values.length).fill(Number.NaN),
    mask: new Uint32Array(values.length)
  };
  for (let profile = 0; profile < offsets.length - 1; profile++) {
    const start = offsets[profile];
    const count = Math.min(offsets[profile + 1], values.length) - start;
    if (count <= 0) {
      continue;
    }
    const isGap = (local: number): boolean =>
      (validity !== undefined && validity[start + local] === 0) ||
      !Number.isFinite(values[start + local]);
    const neighbor = (local: number, delta: number): number => {
      const position = local + delta;
      if (wrap) {
        return ((position % count) + count) % count;
      }
      return position < 0 || position >= count ? -1 : position;
    };
    const candidates: {
      local: number;
      prominence: number;
      delta: number;
      value: number;
    }[] = [];
    for (let local = wrap ? 0 : 1; local < (wrap ? count : count - 1); local++) {
      if (isGap(local)) {
        continue;
      }
      const peak = values[start + local];
      let isMaximum = true;
      for (let offset = -2; offset <= 2 && isMaximum; offset++) {
        if (offset === 0) {
          continue;
        }
        const other = neighbor(local, offset);
        if (other < 0 || other === local || isGap(other)) {
          continue;
        }
        const otherValue = values[start + other];
        if (otherValue > peak || (offset < 0 && otherValue === peak)) {
          isMaximum = false;
        }
      }
      if (!isMaximum) {
        continue;
      }
      const walk = (direction: number) => {
        let lowest = peak;
        let valid = 0;
        const limit = wrap ? Math.min(options.window, count - 1) : options.window;
        for (let step = 1; step <= limit; step++) {
          const other = neighbor(local, direction * step);
          if (other < 0 || isGap(other)) {
            break;
          }
          valid++;
          const otherValue = values[start + other];
          if (otherValue > peak) {
            break;
          }
          if (otherValue < lowest) {
            lowest = otherValue;
          }
        }
        return {lowest, valid};
      };
      const left = walk(-1);
      const right = walk(1);
      if (left.valid < minSide || right.valid < minSide) {
        continue;
      }
      const prominence = f32(peak - Math.max(left.lowest, right.lowest));
      if (prominence < options.minProminence) {
        continue;
      }
      const previous = neighbor(local, -1);
      const next = neighbor(local, 1);
      let delta = 0;
      let value = peak;
      if (!isGap(previous) && !isGap(next)) {
        const a = values[start + previous];
        const c = values[start + next];
        const denominator = a - 2 * peak + c;
        if (denominator < 0) {
          delta = Math.max(-0.5, Math.min(0.5, (0.5 * (a - c)) / denominator));
          value = peak - 0.25 * (a - c) * delta;
        }
      }
      candidates.push({local, prominence, delta, value});
    }
    // Greedy suppression: prominence descending, ties by ascending local index.
    const ranked = [...candidates].sort(
      (left, right) => right.prominence - left.prominence || left.local - right.local
    );
    const kept: typeof candidates = [];
    for (const candidate of ranked) {
      const isSuppressed = kept.some(other => {
        let distance = Math.abs(other.local + other.delta - (candidate.local + candidate.delta));
        if (wrap) {
          distance = Math.min(distance, count - distance);
        }
        return distance <= nms;
      });
      if (!isSuppressed) {
        kept.push(candidate);
      }
    }
    for (const candidate of kept.sort((left, right) => left.local - right.local)) {
      let index = candidate.local + candidate.delta;
      if (wrap) {
        if (index < 0) {
          index += count;
        }
        if (index >= count) {
          index -= count;
        }
      }
      const row = start + candidate.local;
      result.peaks.push({
        profile,
        row,
        local: candidate.local,
        index,
        value: candidate.value,
        prominence: candidate.prominence
      });
      result.prominence[row] = candidate.prominence;
      result.refinedIndex[row] = index;
      result.refinedValue[row] = candidate.value;
      result.mask[row] = 1;
    }
  }
  return result;
}

/** Sum of Gaussian bumps `[centre, height, width]` sampled at integer indices, as float32. */
export function createGaussianBumps(
  count: number,
  bumps: readonly (readonly [centre: number, height: number, width: number])[],
  base: number = 0
): Float32Array {
  return Float32Array.from({length: count}, (_, index) =>
    bumps.reduce(
      (sum, [centre, height, width]) =>
        sum + height * Math.exp(-((index - centre) ** 2) / (2 * width * width)),
      base
    )
  );
}

/** Circular Gaussian bumps (distance modulo `count`), as float32. */
export function createCircularGaussianBumps(
  count: number,
  bumps: readonly (readonly [centre: number, height: number, width: number])[],
  base: number = 0
): Float32Array {
  return Float32Array.from({length: count}, (_, index) =>
    bumps.reduce((sum, [centre, height, width]) => {
      const distance = Math.abs(index - centre);
      const circular = Math.min(distance, count - distance);
      return sum + height * Math.exp(-(circular ** 2) / (2 * width * width));
    }, base)
  );
}

/** Concatenates profiles into one value array plus CSR offsets. */
export function concatenateProfiles(profiles: readonly ArrayLike<number>[]): {
  values: Float32Array;
  offsets: Uint32Array;
} {
  const offsets = new Uint32Array(profiles.length + 1);
  for (const [index, profile] of profiles.entries()) {
    offsets[index + 1] = offsets[index] + profile.length;
  }
  const values = new Float32Array(offsets[profiles.length]);
  for (const [index, profile] of profiles.entries()) {
    values.set(Array.from(profile), offsets[index]);
  }
  return {values, offsets};
}
