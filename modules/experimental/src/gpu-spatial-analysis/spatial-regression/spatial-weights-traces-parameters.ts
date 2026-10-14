// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PHILOX_PURPOSES, PHILOX_WGSL} from '../../utils/philox-random';
import {getPhilox4x32} from '../../utils/philox-random';

/** Largest `maximumPower` of `GPUSpatialWeightsTraces`: traces `tr(W^q)` for `q = 1..256`. */
export const GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER = 256;

/** Default number of Rademacher probes of the stochastic mode. */
export const GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT = 64;

/**
 * Largest row count for which `mode` defaults to `'exact'`.
 *
 * Exact mode runs one unit-vector probe per row, so its work is `rows * maximumPower * nnz` and
 * its probe batches (scratch capped at 64 MiB) number `ceil(rows * rows / 8M)`. At 4096 rows and
 * `maximumPower = 256` that is 2 batches, about 520 dispatches and `3e10` multiply-adds for an
 * 8-neighbor `W`, which stays well under a second; at 16384 rows it is 32 batches, 8200
 * dispatches and `5e11` multiply-adds, which is no longer an interactive default.
 */
export const GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT = 4096;

/** Largest row count an explicit `mode: 'exact'` accepts (see the dispatch estimate above). */
export const GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT = 16384;

/**
 * Probe-scratch budget in float32 elements per ping-pong buffer: two buffers of 8M elements are
 * 64 MiB. A probe batch holds `max(1, floor(8M / rows))` probes.
 */
export const GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT = 8 * 1024 * 1024;

/** Number of `gramTraces` values: `[tr(W'W), tr(W'W W), tr((W'W)^2), sum_i c_i^2]`. */
export const GPU_SPATIAL_WEIGHTS_TRACES_GRAM_TRACE_LENGTH = 4;
/** `gramTraces` slot: `tr(W'W) = sum w_ij^2` (exact). */
export const GPU_SPATIAL_WEIGHTS_TRACES_GRAM_TRACE_WTW = 0;
/** `gramTraces` slot: `tr(W'W W)` (exact or Hutchinson estimate, per mode). */
export const GPU_SPATIAL_WEIGHTS_TRACES_GRAM_TRACE_WTWW = 1;
/** `gramTraces` slot: `tr((W'W)^2)` (exact or Hutchinson estimate, per mode). */
export const GPU_SPATIAL_WEIGHTS_TRACES_GRAM_TRACE_WTW_SQUARED = 2;
/** `gramTraces` slot: `sum_i c_i^2` with `c_i = sum_l w_li^2` (exact). */
export const GPU_SPATIAL_WEIGHTS_TRACES_GRAM_TRACE_COLUMN_SQUARES = 3;

/** Philox counter word 3 of the probe stream (`'PROB'`). */
export const GPU_SPATIAL_WEIGHTS_TRACES_PROBE_STREAM =
  PHILOX_PURPOSES.spatialWeightsTracesProbeCounter.id;
/** Philox key word 1 of the probe stream (`'TRAC'`). */
export const GPU_SPATIAL_WEIGHTS_TRACES_PROBE_KEY = PHILOX_PURPOSES.spatialWeightsTracesProbeKey.id;

/** Number of uint32 words of the per-frame trace parameters (`[seed]`). */
export const GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH = 1;

/**
 * Per-frame settings of `GPUSpatialWeightsTraces` (and of the `traceParameters` of the spatial
 * maximum-likelihood and GM estimators), packed by {@link getGPUSpatialWeightsTracesParameterValues}.
 */
export type GPUSpatialWeightsTracesSettings = {
  /** Philox seed of the stochastic probes, an integer in `[0, 2^32)`. Defaults to 0. */
  seed?: number;
};

/**
 * Packs {@link GPUSpatialWeightsTracesSettings} into `GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH`
 * uint32 words. Write them to the `parameters` view each frame; the seed is read on the GPU, so
 * changing it never rebuilds or recompiles the graph.
 */
export function getGPUSpatialWeightsTracesParameterValues(
  settings: GPUSpatialWeightsTracesSettings = {}
): Uint32Array {
  const seed = settings.seed ?? 0;
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error('getGPUSpatialWeightsTracesParameterValues seed must be a uint32');
  }
  return Uint32Array.of(seed);
}

/** Trace estimation mode of `GPUSpatialWeightsTraces`. */
export type GPUSpatialWeightsTracesMode = 'exact' | 'stochastic';

/** CSR weights arrays as read back on the CPU. */
export type SpatialWeightsCSR = {
  offsets: ArrayLike<number>;
  neighbors: ArrayLike<number>;
  weights: ArrayLike<number>;
};

/** Options of {@link getSpatialWeightsTracesOnCPU}. */
export type SpatialWeightsTracesCPUOptions = {
  /** `'exact'` (default) uses unit probes; `'stochastic'` uses Rademacher probes. */
  mode?: GPUSpatialWeightsTracesMode;
  /** Probes of the stochastic mode. Defaults to 64. */
  probeCount?: number;
  /** Philox seed of the stochastic probes. Defaults to 0. */
  seed?: number;
  /** Also compute the Gram traces. */
  includeGramTraces?: boolean;
};

/** Result of {@link getSpatialWeightsTracesOnCPU}. */
export type SpatialWeightsTracesCPUResult = {
  /** `tr((W / s)^q)` for `q = 1..maximumPower`, `s = infinityNorm`. */
  powerTraces: Float64Array;
  /** `1' (W / s)^q 1` for `q = 1..maximumPower` (always exact). */
  powerSums: Float64Array;
  /** `[tr(W'W), tr(W'W W), tr((W'W)^2), sum c_i^2]` of the unscaled `W`, when requested. */
  gramTraces?: Float64Array;
  /** `s = ||W||_inf`, the largest absolute row sum (1 when `W` has no links). */
  infinityNorm: number;
};

/**
 * `||W||_inf`, the largest absolute row sum over slots whose neighbor id is `< rows`; 1 when it
 * is zero or not finite (the scale `GPUSpatialWeightsTraces` divides by).
 */
export function getSpatialWeightsTracesInfinityNorm(weights: SpatialWeightsCSR): number {
  const rows = weights.offsets.length - 1;
  let largest = 0;
  for (let row = 0; row < rows; row++) {
    let sum = 0;
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      if (weights.neighbors[slot] < rows) {
        sum += Math.abs(weights.weights[slot]);
      }
    }
    largest = Math.max(largest, sum);
  }
  return largest > 0 && Number.isFinite(largest) ? largest : 1;
}

/**
 * Rademacher probe value (`+1` or `-1`) of probe `probe` at row `row`: the CPU mirror of the
 * WGSL probe stream, so a seeded stochastic run replays exactly. Bit `row & 3` selects word
 * `row & 3` of the Philox block with counter `(probe, row >> 2, 0, 'PROB')` and key `(seed, 'TRAC')`.
 */
export function getSpatialWeightsTracesProbeValue(
  seed: number,
  probe: number,
  row: number
): 1 | -1 {
  const block = getPhilox4x32(
    [probe, row >>> 2, 0, GPU_SPATIAL_WEIGHTS_TRACES_PROBE_STREAM],
    [seed, GPU_SPATIAL_WEIGHTS_TRACES_PROBE_KEY]
  );
  return (block[row & 3] & 1) === 1 ? -1 : 1;
}

/**
 * Float64 mirror of `GPUSpatialWeightsTraces` on sparse CSR weights. Like the GPU node it runs the
 * chains with `W / s`, `s = ||W||_inf` (returned as `infinityNorm`), so `powerTraces` and
 * `powerSums` are those of the scaled matrix; the Gram traces are of the unscaled `W`.
 *
 * `'exact'` (default) runs one unit-vector chain per row (`O(rows * maximumPower * nnz)`);
 * `'stochastic'` runs `probeCount` seeded Rademacher chains and averages, using the same probes
 * as the GPU. `powerSums` is always one chain from the ones vector. Slots whose neighbor id is
 * `>= rows` are dropped, as the GPU kernels do; self-loops are allowed.
 */
export function getSpatialWeightsTracesOnCPU(
  weights: SpatialWeightsCSR,
  maximumPower: number,
  options: SpatialWeightsTracesCPUOptions = {}
): SpatialWeightsTracesCPUResult {
  const rows = weights.offsets.length - 1;
  const stochastic = options.mode === 'stochastic';
  const probeTotal = stochastic
    ? (options.probeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT)
    : rows;
  const seed = (options.seed ?? 0) >>> 0;
  const includeGram = options.includeGramTraces === true;
  const infinityNorm = getSpatialWeightsTracesInfinityNorm(weights);
  const stepCount = includeGram ? Math.max(maximumPower, 2) : maximumPower;
  const multiply = (vector: Float64Array): Float64Array => {
    const result = new Float64Array(rows);
    for (let row = 0; row < rows; row++) {
      let sum = 0;
      for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
        const neighbor = weights.neighbors[slot];
        if (neighbor < rows) {
          sum += (weights.weights[slot] / infinityNorm) * vector[neighbor];
        }
      }
      result[row] = sum;
    }
    return result;
  };
  const multiplyTranspose = (vector: Float64Array): Float64Array => {
    const result = new Float64Array(rows);
    for (let row = 0; row < rows; row++) {
      for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
        const neighbor = weights.neighbors[slot];
        if (neighbor < rows) {
          result[neighbor] += weights.weights[slot] * vector[row];
        }
      }
    }
    return result;
  };
  const powerTraces = new Float64Array(maximumPower);
  let tracePairs = 0;
  let traceSquare = 0;
  for (let probe = 0; probe < probeTotal; probe++) {
    const z = new Float64Array(rows);
    if (stochastic) {
      for (let row = 0; row < rows; row++) {
        z[row] = getSpatialWeightsTracesProbeValue(seed, probe, row);
      }
    } else {
      z[probe] = 1;
    }
    let vector: Float64Array = z;
    let first: Float64Array = z;
    for (let step = 1; step <= stepCount; step++) {
      const next = multiply(vector);
      if (step <= maximumPower) {
        let dot = 0;
        for (let row = 0; row < rows; row++) {
          dot += z[row] * next[row];
        }
        powerTraces[step - 1] += dot;
      }
      if (step === 1) {
        first = next;
      } else if (includeGram && step === 2) {
        for (let row = 0; row < rows; row++) {
          tracePairs += first[row] * next[row];
        }
        const transposed = multiplyTranspose(first);
        for (let row = 0; row < rows; row++) {
          traceSquare += transposed[row] * transposed[row];
        }
      }
      vector = next;
    }
  }
  const scale = stochastic ? 1 / probeTotal : 1;
  for (let index = 0; index < maximumPower; index++) {
    powerTraces[index] *= scale;
  }
  const powerSums = new Float64Array(maximumPower);
  let ones: Float64Array = new Float64Array(rows).fill(1);
  for (let step = 0; step < maximumPower; step++) {
    ones = multiply(ones);
    let sum = 0;
    for (let row = 0; row < rows; row++) {
      sum += ones[row];
    }
    powerSums[step] = sum;
  }
  if (!includeGram) {
    return {powerTraces, powerSums, infinityNorm};
  }
  let traceGram = 0;
  const columnSquares = new Float64Array(rows);
  for (let row = 0; row < rows; row++) {
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      const neighbor = weights.neighbors[slot];
      if (neighbor < rows) {
        const square = weights.weights[slot] * weights.weights[slot];
        traceGram += square;
        columnSquares[neighbor] += square;
      }
    }
  }
  let sumColumnSquares = 0;
  for (let column = 0; column < rows; column++) {
    sumColumnSquares += columnSquares[column] * columnSquares[column];
  }
  return {
    powerTraces,
    powerSums,
    infinityNorm,
    // The chains hold W / s: tr(W'W W) = s^3 E[v1' v2] and tr((W'W)^2) = s^2 E[|W' v1|^2].
    gramTraces: Float64Array.of(
      traceGram,
      tracePairs * scale * infinityNorm ** 3,
      traceSquare * scale * infinityNorm ** 2,
      sumColumnSquares
    )
  };
}

/**
 * Truncated spatial log-determinant `ln|I - rho W| = -sum_{q=1}^{Q} t^q tr((W/s)^q) / q` with
 * `t = rho * s`. `powerTraces` are the traces of `W / s` and `infinityNorm` is `s` (default 1,
 * for traces of an unscaled `W`).
 *
 * The series converges only for `|t| < 1`; the omitted tail is bounded by
 * {@link getSpatialLogDeterminantTruncationBound}.
 */
export function getSpatialLogDeterminantFromTraces(
  rho: number,
  powerTraces: ArrayLike<number>,
  infinityNorm = 1
): number {
  const t = rho * infinityNorm;
  let sum = 0;
  for (let q = powerTraces.length; q >= 1; q--) {
    sum = sum * t + powerTraces[q - 1] / q;
  }
  return -t * sum;
}

/**
 * Truncated `tr((I - rho W)^-1) = n + sum_{q=1}^{Q} t^q tr((W/s)^q)` with `t = rho * s`, the
 * numerator of the average direct impact multiplier. `infinityNorm` is `s` (default 1).
 * Requires `|t| < 1`.
 */
export function getSpatialMultiplierTraceFromTraces(
  rho: number,
  powerTraces: ArrayLike<number>,
  rowCount: number,
  infinityNorm = 1
): number {
  const t = rho * infinityNorm;
  let sum = 0;
  for (let q = powerTraces.length; q >= 1; q--) {
    sum = sum * t + powerTraces[q - 1];
  }
  return rowCount + t * sum;
}

/**
 * Truncated `1'(I - rho W)^-1 1 = n + sum_{q=1}^{Q} t^q 1'(W/s)^q 1` with `t = rho * s`, the
 * numerator of the average total impact multiplier. `infinityNorm` is `s` (default 1).
 * Requires `|t| < 1`.
 */
export function getSpatialMultiplierTotalFromPowerSums(
  rho: number,
  powerSums: ArrayLike<number>,
  rowCount: number,
  infinityNorm = 1
): number {
  return getSpatialMultiplierTraceFromTraces(rho, powerSums, rowCount, infinityNorm);
}

/**
 * Upper bound of the absolute log-determinant truncation error after `maximumPower` terms,
 * `n |t|^(Q+1) / ((Q+1) (1 - |t|))` with `t = |rho| * scale`. Since `|tr((W/s)^q)| <= n`
 * (spectral radius of `W / s` at most 1), this is the geometric tail of the series. Pass
 * `spectralRadius = ||W||_inf` (the `infinityNorm` of the traces); it is 1 for a row-standardized
 * `W`. Returns `Infinity` when `t >= 1`.
 */
export function getSpatialLogDeterminantTruncationBound(
  rho: number,
  maximumPower: number,
  rowCount: number,
  spectralRadius = 1
): number {
  const scaled = Math.abs(rho) * spectralRadius;
  if (!(scaled < 1)) {
    return Infinity;
  }
  return (rowCount * scaled ** (maximumPower + 1)) / ((maximumPower + 1) * (1 - scaled));
}

/**
 * Upper bound of the absolute truncation error of {@link getSpatialMultiplierTraceFromTraces}
 * (and of the total multiplier numerator), `n |s|^(Q+1) / (1 - |s|)`. Returns `Infinity` when `s >= 1`.
 */
export function getSpatialMultiplierTruncationBound(
  rho: number,
  maximumPower: number,
  rowCount: number,
  spectralRadius = 1
): number {
  const scaled = Math.abs(rho) * spectralRadius;
  if (!(scaled < 1)) {
    return Infinity;
  }
  return (rowCount * scaled ** (maximumPower + 1)) / (1 - scaled);
}

/** Average impacts of one predictor of a spatial lag model. */
export type SpatialLagImpacts = {direct: number; indirect: number; total: number};

/**
 * Float64 mirror of `GPUSpatialLagImpacts` for one coefficient: `direct = beta tr((I - rho W)^-1) / n`,
 * `total = beta 1'(I - rho W)^-1 1 / n`, `indirect = total - direct`, with the series truncated at
 * `powerTraces.length` terms.
 */
export function getSpatialLagImpactsFromTraces(
  rho: number,
  beta: number,
  powerTraces: ArrayLike<number>,
  powerSums: ArrayLike<number>,
  rowCount: number,
  infinityNorm = 1
): SpatialLagImpacts {
  const direct =
    (beta * getSpatialMultiplierTraceFromTraces(rho, powerTraces, rowCount, infinityNorm)) /
    rowCount;
  const total =
    (beta * getSpatialMultiplierTotalFromPowerSums(rho, powerSums, rowCount, infinityNorm)) /
    rowCount;
  return {direct, indirect: total - direct, total};
}

function getSeriesWGSL(
  functionName: string,
  maximumPower: number,
  bindingName: string,
  divideByPower: boolean,
  constantTerm: string,
  prefix: string,
  normExpression: string
): string {
  if (
    !Number.isInteger(maximumPower) ||
    maximumPower < 1 ||
    maximumPower > GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER
  ) {
    throw new Error(
      `${functionName} maximumPower must be an integer in [1, ${GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER}]`
    );
  }
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(bindingName)) {
    throw new Error(`${functionName} bindingName must be a WGSL identifier`);
  }
  const term = `${bindingName}[${bindingName}Offset + q - 1u]${divideByPower ? ' / f32(q)' : ''}`;
  return /* wgsl */ `
fn ${functionName}(rho: f32) -> f32 {
  // The stored powers are those of W / s, so the series runs in t = rho * s.
  let t = rho * (${normExpression});
  var accumulator = 0.0;
  for (var step = 0u; step < ${maximumPower}u; step++) {
    let q = ${maximumPower}u - step;
    accumulator = accumulator * t + ${term};
  }
  return ${prefix}t * accumulator${constantTerm};
}
`;
}

/**
 * WGSL defining `fn getSpatialLogDeterminant(rho: f32) -> f32`, the truncated series
 * `ln|I - rho W| = -sum_{q=1}^{Q} t^q c_q / q` with `t = rho * s`, `c_q = tr((W/s)^q)` read from the f32 storage
 * binding `bindingName` (its element offset is the `${bindingName}Offset` constant that
 * `createWGSLKernelNode` declares). `normExpression` is a WGSL expression for `s = ||W||_inf`
 * (the `infinityNorm` output of `GPUSpatialWeightsTraces`), default `'1.0'` for unscaled traces.
 * Evaluated by Horner's rule from the highest power down.
 *
 * Truncation error is below {@link getSpatialLogDeterminantTruncationBound} (about
 * `n |t|^(Q+1) / (Q+1)`); the series requires `|rho| * s < 1`, so callers must keep `rho` inside
 * that range.
 */
export function getSpatialLogDeterminantWGSL(
  maximumPower: number,
  bindingName: string,
  normExpression = '1.0'
): string {
  return getSeriesWGSL(
    'getSpatialLogDeterminant',
    maximumPower,
    bindingName,
    true,
    '',
    '-',
    normExpression
  );
}

/**
 * WGSL defining `fn getSpatialMultiplierTrace(rho: f32) -> f32`, the truncated
 * `tr((I - rho W)^-1) = n + sum_{q=1}^{Q} t^q c_q` with `t = rho * s`, `c_q = tr((W/s)^q)` read from
 * the f32 storage binding `bindingName` (offset constant `${bindingName}Offset`), by Horner's rule;
 * `normExpression` is a WGSL expression for `s` (default `'1.0'`).
 * `rowCount` is `n`. Truncation error is below {@link getSpatialMultiplierTruncationBound}; the
 * series requires `|rho| * s < 1`.
 */
export function getSpatialMultiplierTraceWGSL(
  maximumPower: number,
  bindingName: string,
  rowCount: number,
  normExpression = '1.0'
): string {
  return getSeriesWGSL(
    'getSpatialMultiplierTrace',
    maximumPower,
    bindingName,
    false,
    ` + f32(${rowCount}u)`,
    '',
    normExpression
  );
}

/**
 * WGSL defining `fn getSpatialMultiplierTotal(rho: f32) -> f32`, the truncated
 * `1'(I - rho W)^-1 1 = n + sum_{q=1}^{Q} t^q c_q` with `c_q = 1'(W/s)^q 1`, `t = rho * s` read from the f32 storage
 * binding `bindingName` (the `powerSums` output of `GPUSpatialWeightsTraces`).
 */
export function getSpatialMultiplierTotalWGSL(
  maximumPower: number,
  bindingName: string,
  rowCount: number,
  normExpression = '1.0'
): string {
  return getSeriesWGSL(
    'getSpatialMultiplierTotal',
    maximumPower,
    bindingName,
    false,
    ` + f32(${rowCount}u)`,
    '',
    normExpression
  );
}

/**
 * WGSL defining `fn getProbeValue(seed: u32, probe: u32, row: u32) -> f32`, the probe vectors of
 * `GPUSpatialWeightsTraces`: unit vectors `e_probe` in exact mode, Rademacher `+-1` entries from
 * Philox (counter `(probe, row >> 2, 0, PROBE_STREAM)`, key `(seed, PROBE_KEY)`, lane `row & 3`, low
 * bit) in stochastic mode. Shared by every probe pass that must draw the same probes (the
 * maximum-likelihood `tr(WA'WA)` pass); the CPU mirror is {@link getSpatialWeightsTracesProbeValue}.
 */
export function getSpatialWeightsTracesProbeWGSL(stochastic: boolean): string {
  if (!stochastic) {
    return /* wgsl */ `fn getProbeValue(seed: u32, probe: u32, row: u32) -> f32 {
  _ = seed;
  return select(0.0, 1.0, row == probe);
}`;
  }
  // The seed is a runtime argument (word 2 of the probe batch table), never WGSL text (rule 12).
  return /* wgsl */ `${PHILOX_WGSL}
fn getProbeValue(seed: u32, probe: u32, row: u32) -> f32 {
  let randomBlock = philox4x32(
    vec4<u32>(probe, row >> 2u, 0u, ${GPU_SPATIAL_WEIGHTS_TRACES_PROBE_STREAM}u),
    vec2<u32>(seed, ${GPU_SPATIAL_WEIGHTS_TRACES_PROBE_KEY}u)
  );
  let word = randomBlock[row & 3u];
  return select(1.0, -1.0, (word & 1u) == 1u);
}`;
}
