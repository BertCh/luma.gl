// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const f32 = Math.fround;

/** A CPU spatial-weights CSR. */
export type CPUWeights = {offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array};

/** Builds a CSR from per-row `[neighbor, weight]` lists, sorted by neighbor. */
export function createWeights(rows: readonly (readonly [number, number][])[]): CPUWeights {
  const offsets = new Uint32Array(rows.length + 1);
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (const [row, entries] of rows.entries()) {
    for (const [neighbor, weight] of [...entries].sort((a, b) => a[0] - b[0])) {
      neighbors.push(neighbor);
      weights.push(weight);
    }
    offsets[row + 1] = neighbors.length;
  }
  return {offsets, neighbors: Uint32Array.from(neighbors), weights: Float32Array.from(weights)};
}

/** Per-row CPU summary. */
export type OracleSummary = {
  count: number;
  weightSum: number;
  sum: number;
  mean: number;
  min: number;
  max: number;
  standardDeviation: number;
  median: number;
  mode: number;
  entropy: number;
};

/** Reference of `GPUNeighborhoodSummary`: member order and f32 operations as on the GPU. */
export function computeNeighborhoodSummaryOracle(input: {
  weights: CPUWeights;
  values: Float32Array;
  categories: Uint32Array;
  mask?: Uint32Array;
  includeFocal?: boolean;
  focalWeight?: number;
  maximumNeighbors?: number;
  /** Re-standardize slot weights over the active members (`weights.info.transform === 'row'`). */
  standardized?: boolean;
  /** `weights.status.overflow` is nonzero: every output is NaN. */
  truncated?: boolean;
}): OracleSummary[] {
  const {weights, values, categories, mask} = input;
  const rows = values.length;
  const maximumNeighbors = input.maximumNeighbors ?? 32;
  const summaries: OracleSummary[] = [];
  for (let row = 0; row < rows; row++) {
    if (!(mask && mask[row] === 0) && input.truncated) {
      summaries.push({
        count: NaN,
        weightSum: NaN,
        sum: NaN,
        mean: NaN,
        min: NaN,
        max: NaN,
        standardDeviation: NaN,
        median: NaN,
        mode: 0xffffffff,
        entropy: NaN
      });
      continue;
    }
    if (mask && mask[row] === 0) {
      summaries.push({
        count: 0,
        weightSum: NaN,
        sum: NaN,
        mean: NaN,
        min: NaN,
        max: NaN,
        standardDeviation: NaN,
        median: NaN,
        mode: 0xffffffff,
        entropy: NaN
      });
      continue;
    }
    const slotMembers: {row: number; weight: number}[] = [];
    // Active slots only: another row with a positive weight.
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      const neighbor = weights.neighbors[slot];
      if (neighbor < rows && neighbor !== row && weights.weights[slot] > 0) {
        slotMembers.push({row: neighbor, weight: weights.weights[slot]});
      }
    }
    // Slots valid for a member filter, re-standardized by their summed weight in slot order.
    const withFocal = (filter: (member: {row: number}) => boolean) => {
      const valid = slotMembers.filter(
        member => (!mask || mask[member.row] !== 0) && filter(member)
      );
      let sum = 0;
      for (const member of valid) {
        sum = f32(sum + member.weight);
      }
      const scale = input.standardized ? (sum > 0 ? f32(1 / sum) : 0) : 1;
      const scaled = valid.map(member => ({
        row: member.row,
        weight: input.standardized ? f32(member.weight * scale) : member.weight
      }));
      const focal =
        input.includeFocal && (!mask || mask[row] !== 0) && filter({row})
          ? [{row, weight: f32(input.focalWeight ?? 1)}]
          : [];
      return [...focal, ...scaled];
    };
    const included = withFocal(() => true);
    const numeric = withFocal(member => Number.isFinite(values[member.row]));
    let weightSum = 0;
    let weightedSum = 0;
    for (const member of numeric) {
      weightSum = f32(weightSum + member.weight);
      weightedSum = f32(weightedSum + f32(member.weight * values[member.row]));
    }
    const mean = f32(weightedSum / weightSum);
    let squares = 0;
    for (const member of numeric) {
      const deviation = f32(values[member.row] - mean);
      squares = f32(squares + f32(f32(member.weight * deviation) * deviation));
    }
    const sorted = numeric.map(member => values[member.row]).sort((a, b) => a - b);
    const k = sorted.length;
    let median = NaN;
    if (k > 0 && k <= maximumNeighbors) {
      median = k % 2 ? sorted[(k - 1) / 2] : f32(0.5 * f32(sorted[k / 2 - 1] + sorted[k / 2]));
    }
    // Categories: any included member.
    let categoryWeightSum = 0;
    const frequencies = new Map<number, number>();
    for (const member of included) {
      categoryWeightSum = f32(categoryWeightSum + member.weight);
      const category = categories[member.row];
      frequencies.set(category, f32((frequencies.get(category) ?? 0) + member.weight));
    }
    let mode = 0xffffffff;
    let bestFrequency = -1;
    let entropy = 0;
    for (const [category, frequency] of frequencies) {
      if (frequency > bestFrequency || (frequency === bestFrequency && category < mode)) {
        bestFrequency = frequency;
        mode = category;
      }
      if (frequency > 0) {
        const share = frequency / categoryWeightSum;
        entropy -= share * Math.log(share);
      }
    }
    summaries.push({
      count: numeric.length,
      weightSum,
      sum: weightedSum,
      mean: weightSum > 0 ? mean : NaN,
      min: k > 0 ? sorted[0] : NaN,
      max: k > 0 ? sorted[k - 1] : NaN,
      standardDeviation: weightSum > 0 ? Math.sqrt(squares / weightSum) : NaN,
      median,
      mode,
      entropy: categoryWeightSum > 0 ? entropy : NaN
    });
  }
  return summaries;
}
