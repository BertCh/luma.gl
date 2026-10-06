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
}): OracleSummary[] {
  const {weights, values, categories, mask} = input;
  const rows = values.length;
  const maximumNeighbors = input.maximumNeighbors ?? 32;
  const summaries: OracleSummary[] = [];
  for (let row = 0; row < rows; row++) {
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
    const members: {row: number; weight: number}[] = [];
    if (input.includeFocal) {
      members.push({row, weight: f32(input.focalWeight ?? 1)});
    }
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      const neighbor = weights.neighbors[slot];
      if (neighbor < rows && neighbor !== row) {
        members.push({row: neighbor, weight: weights.weights[slot]});
      }
    }
    const included = members.filter(member => !mask || mask[member.row] !== 0);
    const numeric = included.filter(member => Number.isFinite(values[member.row]));
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
