// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** A CPU spatial-weights CSR (neighbor IDs ascending within each row). */
export type CPUSpatialWeights = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

/** Statistic and analytic moments of one global statistic (NaN where undefined). */
export type GlobalStatisticOracle = {
  statistic: number;
  expected: number;
  varianceNormality: number;
  zNormality: number;
  pNormality: number;
  varianceRandomization: number;
  zRandomization: number;
  pRandomization: number;
};

/** Join counts and their randomization moments. */
export type JoinCountOracle = {
  blackBlack: number;
  blackWhite: number;
  whiteWhite: number;
  joins: number;
  expectedBlackBlack: number;
  varianceBlackBlack: number;
  zBlackBlack: number;
  pBlackBlack: number;
  expectedBlackWhite: number;
  varianceBlackWhite: number;
  zBlackWhite: number;
  pBlackWhite: number;
  ordered: [number, number, number];
};

/** Every output of `GPUGlobalSpatialStatistics`, in double precision. */
export type GlobalSpatialStatisticsOracle = {
  count: number;
  s0: number;
  s1: number;
  s2: number;
  mean: number;
  variance: number;
  blackCount: number;
  islandCount: number;
  moran: GlobalStatisticOracle;
  geary: GlobalStatisticOracle;
  getisOrdG: GlobalStatisticOracle;
  bivariateMoran: GlobalStatisticOracle;
  joinCount: JoinCountOracle;
};

/** Dense included-pair weights: `[n][n]` over included rows, plus the included row IDs. */
type DenseWeights = {rows: number[]; matrix: number[][]};

/** Two-sided normal p-value (erfc via a high-accuracy series/continued fraction). */
export function getTwoSidedPValue(z: number): number {
  if (!Number.isFinite(z)) {
    return NaN;
  }
  return erfc(Math.abs(z) / Math.SQRT2);
}

function erfc(x: number): number {
  // W. J. Cody's rational approximations are overkill here; use the Numerical Recipes erfcc,
  // which has fractional error below 1.2e-7, ample for comparing f32 GPU output.
  const t = 1 / (1 + 0.5 * x);
  const exponent =
    -x * x -
    1.26551223 +
    t *
      (1.00002368 +
        t *
          (0.37409196 +
            t *
              (0.09678418 +
                t *
                  (-0.18628806 +
                    t *
                      (0.27886807 +
                        t *
                          (-1.13520398 +
                            t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
  return Math.min(t * Math.exp(exponent), 1);
}

/** mulberry32 generator with values in [0, 1). */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Builds a CSR from per-row `[neighbor, weight]` lists, sorting each row by neighbor. */
export function createWeights(
  rows: readonly (readonly (readonly [number, number])[])[],
  slack = 0
): CPUSpatialWeights {
  const offsets = new Uint32Array(rows.length + 1);
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (const [row, entries] of rows.entries()) {
    const sorted = [...entries].sort((a, b) => a[0] - b[0]);
    for (const [neighbor, weight] of sorted) {
      neighbors.push(neighbor);
      weights.push(weight);
    }
    offsets[row + 1] = neighbors.length;
  }
  // Capacity slack filled with garbage the contributor must ignore.
  for (let slot = 0; slot < slack; slot++) {
    neighbors.push(slot % Math.max(rows.length, 1));
    weights.push(1e6);
  }
  return {
    offsets,
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(weights)
  };
}

/** Rook contiguity on a `columns x rows` grid, binary or row standardized. */
export function createGridWeights(
  columns: number,
  rows: number,
  rowStandardize = false,
  queen = false
): CPUSpatialWeights {
  const lists: [number, number][][] = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      const entries: [number, number][] = [];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const rook = Math.abs(dx) + Math.abs(dy) === 1;
          const diagonal = Math.abs(dx) === 1 && Math.abs(dy) === 1;
          if (
            (rook || (queen && diagonal)) &&
            x + dx >= 0 &&
            x + dx < columns &&
            y + dy >= 0 &&
            y + dy < rows
          ) {
            entries.push([(y + dy) * columns + x + dx, 1]);
          }
        }
      }
      lists.push(rowStandardize ? entries.map(([j]) => [j, 1 / entries.length]) : entries);
    }
  }
  return createWeights(lists);
}

/** Asymmetric kNN weights over random points (row standardized when requested). */
export function createKnnWeights(
  positions: Float64Array,
  k: number,
  rowStandardize: boolean,
  inverseDistance = false
): CPUSpatialWeights {
  const count = positions.length / 2;
  const lists: [number, number][][] = [];
  for (let i = 0; i < count; i++) {
    const candidates: [number, number][] = [];
    for (let j = 0; j < count; j++) {
      if (j !== i) {
        candidates.push([
          j,
          Math.hypot(
            positions[j * 2] - positions[i * 2],
            positions[j * 2 + 1] - positions[i * 2 + 1]
          )
        ]);
      }
    }
    candidates.sort((a, b) => a[1] - b[1] || a[0] - b[0]);
    let entries = candidates
      .slice(0, k)
      .map(([j, d]): [number, number] => [j, inverseDistance ? 1 / Math.max(d, 1e-3) : 1]);
    if (rowStandardize) {
      const sum = entries.reduce((total, [, w]) => total + w, 0);
      entries = entries.map(([j, w]) => [j, w / sum]);
    }
    lists.push(entries);
  }
  return createWeights(lists);
}

function getDenseWeights(weights: CPUSpatialWeights, included: boolean[]): DenseWeights {
  const rows = included.flatMap((value, row) => (value ? [row] : []));
  const position = new Map(rows.map((row, index) => [row, index]));
  const matrix = rows.map(() => new Array<number>(rows.length).fill(0));
  const total = weights.offsets[weights.offsets.length - 1];
  for (const row of rows) {
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1] && slot < total; slot++) {
      const neighbor = weights.neighbors[slot];
      const column = position.get(neighbor);
      if (column !== undefined && neighbor !== row) {
        matrix[position.get(row)!][column] = weights.weights[slot];
      }
    }
  }
  return {rows, matrix};
}

function getWeightSums(matrix: number[][]): {
  s0: number;
  s1: number;
  s2: number;
  degreeDeviation: number;
} {
  const n = matrix.length;
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  const degrees: number[] = [];
  for (let i = 0; i < n; i++) {
    let row = 0;
    let column = 0;
    for (let j = 0; j < n; j++) {
      s0 += matrix[i][j];
      s1 += 0.5 * (matrix[i][j] + matrix[j][i]) ** 2;
      row += matrix[i][j];
      column += matrix[j][i];
    }
    s2 += (row + column) ** 2;
    degrees.push(row + column);
  }
  const meanDegree = degrees.reduce((a, b) => a + b, 0) / n;
  const degreeDeviation = degrees.reduce((total, d) => total + (d - meanDegree) ** 2, 0);
  return {s0, s1, s2, degreeDeviation};
}

function getStatistic(
  statistic: number,
  expected: number,
  varianceNormality: number,
  varianceRandomization: number
): GlobalStatisticOracle {
  const z = (variance: number) =>
    variance > 0 ? (statistic - expected) / Math.sqrt(variance) : NaN;
  const zNormality = z(varianceNormality);
  const zRandomization = z(varianceRandomization);
  return {
    statistic,
    expected,
    varianceNormality,
    zNormality,
    pNormality: getTwoSidedPValue(zNormality),
    varianceRandomization,
    zRandomization,
    pRandomization: getTwoSidedPValue(zRandomization)
  };
}

/** esda.Moran moments `[EI, VI_norm, VI_rand]`. */
export function getMoranMoments(
  n: number,
  s0: number,
  s1: number,
  s2: number,
  kurtosis: number
): number[] {
  const n2 = n * n;
  const s02 = s0 * s0;
  const expected = -1 / (n - 1);
  const varianceNormality = (n2 * s1 - n * s2 + 3 * s02) / ((n2 - 1) * s02) - expected ** 2;
  const a = n * ((n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02);
  const b = kurtosis * ((n2 - n) * s1 - 2 * n * s2 + 6 * s02);
  const varianceRandomization = (a - b) / ((n - 1) * (n - 2) * (n - 3) * s02) - expected ** 2;
  return [expected, varianceNormality, varianceRandomization];
}

/** esda.Geary moments `[VC_norm, VC_rand]`. */
export function getGearyMoments(
  n: number,
  s0: number,
  s1: number,
  s2: number,
  kurtosis: number
): number[] {
  const n2 = n * n;
  const s02 = s0 * s0;
  const varianceNormality = ((2 * s1 + s2) * (n - 1) - 4 * s02) / (2 * (n + 1) * s02);
  const a = (n - 1) * s1 * (n2 - 3 * n + 3 - (n - 1) * kurtosis);
  const b = 0.25 * ((n - 1) * s2 * (n2 + 3 * n - 6 - (n2 - n + 2) * kurtosis));
  const c = s02 * (n2 - 3 - (n - 1) ** 2 * kurtosis);
  return [varianceNormality, (a - b + c) / (n * (n - 2) * (n - 3) * s02)];
}

/** esda.G raw-moment variance `VG` (the formula the GPU evaluates in centered form). */
export function getGeneralGVariance(values: number[], s0: number, s1: number, s2: number): number {
  const n = values.length;
  const n2 = n * n;
  const s02 = s0 * s0;
  const b0 = (n2 - 3 * n + 3) * s1 - n * s2 + 3 * s02;
  const b1 = -((n2 - n) * s1 - 2 * n * s2 + 6 * s02);
  const b2 = -(2 * n * s1 - (n + 3) * s2 + 6 * s02);
  const b3 = 4 * (n - 1) * s1 - 2 * (n + 1) * s2 + 8 * s02;
  const b4 = s1 - s2 + s02;
  const sum = (power: number) => values.reduce((total, value) => total + value ** power, 0);
  const sum1 = sum(1);
  const sum2 = sum(2);
  const numerator =
    b0 * sum2 ** 2 + b1 * sum(4) + b2 * sum1 ** 2 * sum2 + b3 * sum1 * sum(3) + b4 * sum1 ** 4;
  const denominator = (sum1 ** 2 - sum2) ** 2 * n * (n - 1) * (n - 2) * (n - 3);
  const expected = s0 / (n * (n - 1));
  return numerator / denominator - expected ** 2;
}

/** Cliff-Ord randomization moments of BB and BW for binarized weights. */
export function getJoinCountMoments(
  n: number,
  blackCount: number,
  s0: number,
  s1: number,
  s2: number
): number[] {
  const n1 = blackCount;
  const n2 = n - n1;
  const fall = (value: number, k: number) => {
    let product = 1;
    for (let index = 0; index < k; index++) {
      product *= value - index;
    }
    return product;
  };
  const expectedBlackBlack = (0.5 * s0 * fall(n1, 2)) / fall(n, 2);
  const varianceBlackBlack =
    0.25 *
      ((s1 * fall(n1, 2)) / fall(n, 2) +
        ((s2 - 2 * s1) * fall(n1, 3)) / fall(n, 3) +
        ((s0 * s0 + s1 - s2) * fall(n1, 4)) / fall(n, 4)) -
    expectedBlackBlack ** 2;
  const expectedBlackWhite = (s0 * n1 * n2) / fall(n, 2);
  const varianceBlackWhite =
    0.25 *
      ((2 * s1 * n1 * n2) / fall(n, 2) +
        ((s2 - 2 * s1) * n1 * n2 * (n1 + n2 - 2)) / fall(n, 3) +
        (4 * (s0 * s0 + s1 - s2) * fall(n1, 2) * fall(n2, 2)) / fall(n, 4)) -
    expectedBlackWhite ** 2;
  return [expectedBlackBlack, varianceBlackBlack, expectedBlackWhite, varianceBlackWhite];
}

/**
 * Reference of `GPUGlobalSpatialStatistics` in double precision with esda's formulas.
 */
export function computeGlobalSpatialStatisticsOracle(input: {
  weights: CPUSpatialWeights;
  values: Float32Array;
  secondValues?: Float32Array;
  mask?: Uint32Array;
}): GlobalSpatialStatisticsOracle {
  const {weights, values, secondValues, mask} = input;
  const rowCount = values.length;
  const included = Array.from(
    {length: rowCount},
    (_, row) =>
      (!mask || mask[row] !== 0) &&
      Number.isFinite(values[row]) &&
      (!secondValues || Number.isFinite(secondValues[row]))
  );
  const dense = getDenseWeights(weights, included);
  const {matrix, rows} = dense;
  const n = rows.length;
  const x = rows.map(row => values[row]);
  const y = rows.map(row => (secondValues ? secondValues[row] : 0));
  const mean = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;
  const z = x.map(value => value - mean);
  const zy = y.map(value => value - meanY);
  const sumSquares = z.reduce((total, value) => total + value * value, 0);
  const sumSquaresY = zy.reduce((total, value) => total + value * value, 0);
  const sumFourths = z.reduce((total, value) => total + value ** 4, 0);
  const kurtosis = (n * sumFourths) / sumSquares ** 2;
  const {s0, s1, s2} = getWeightSums(matrix);
  const valid = n >= 4 && s0 > 0 && sumSquares > 0;
  const orNaN = (value: number) => (valid && Number.isFinite(value) ? value : NaN);

  let moranCross = 0;
  let gearySum = 0;
  let gCross = 0;
  let bivariateCross = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const w = matrix[i][j];
      moranCross += w * z[i] * z[j];
      gearySum += w * (x[i] - x[j]) ** 2;
      gCross += w * x[i] * x[j];
      bivariateCross += w * z[i] * zy[j];
    }
  }
  const [moranExpected, moranNormality, moranRandomization] = getMoranMoments(
    n,
    s0,
    s1,
    s2,
    kurtosis
  );
  const [gearyNormality, gearyRandomization] = getGearyMoments(n, s0, s1, s2, kurtosis);
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumX2 = x.reduce((a, b) => a + b * b, 0);
  const columnLags = Array.from({length: n}, (_, j) =>
    matrix.reduce((total, row, i) => total + row[j] * z[i], 0)
  );
  const meanLag = columnLags.reduce((a, b) => a + b, 0) / n;
  const lagDeviation = columnLags.reduce((total, value) => total + (value - meanLag) ** 2, 0);
  const scale = n / s0;

  // Join counts on binarized weights.
  const binary = matrix.map(row => row.map(w => (w > 0 ? 1 : 0)));
  const black = x.map(value => (value !== 0 ? 1 : 0));
  const blackCount = black.reduce((a: number, b) => a + b, 0);
  const ordered: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (binary[i][j]) {
        ordered[black[i] && black[j] ? 0 : black[i] !== black[j] ? 1 : 2]++;
      }
    }
  }
  const binarySums = getWeightSums(binary);
  const [expectedBlackBlack, rawVarianceBlackBlack, expectedBlackWhite, rawVarianceBlackWhite] =
    getJoinCountMoments(n, blackCount, binarySums.s0, binarySums.s1, binarySums.s2);
  const binaryValid = n >= 4 && binarySums.s0 > 0;
  // One colour: every join count is constant.
  const oneColour = blackCount === 0 || blackCount === n;
  const varianceBlackBlack = oneColour ? 0 : rawVarianceBlackBlack;
  const varianceBlackWhite = oneColour ? 0 : rawVarianceBlackWhite;
  const blackBlack = ordered[0] / 2;
  const blackWhite = ordered[1] / 2;
  const zBlackBlack =
    varianceBlackBlack > 0
      ? (blackBlack - expectedBlackBlack) / Math.sqrt(varianceBlackBlack)
      : NaN;
  const zBlackWhite =
    varianceBlackWhite > 0
      ? (blackWhite - expectedBlackWhite) / Math.sqrt(varianceBlackWhite)
      : NaN;
  const degrees = binary.map(
    (row, i) =>
      row.reduce((a: number, b) => a + b, 0) + binary.reduce((t: number, r) => t + r[i], 0)
  );

  return {
    count: n,
    s0,
    s1,
    s2,
    mean,
    variance: sumSquares / n,
    blackCount,
    islandCount: degrees.filter(degree => degree === 0).length,
    moran: getStatistic(
      (scale * moranCross) / sumSquares,
      moranExpected,
      orNaN(moranNormality),
      orNaN(moranRandomization)
    ),
    geary: getStatistic(
      ((n - 1) * gearySum) / (2 * s0 * sumSquares),
      1,
      orNaN(gearyNormality),
      orNaN(gearyRandomization)
    ),
    getisOrdG: getStatistic(
      gCross / (sumX * sumX - sumX2),
      s0 / (n * (n - 1)),
      NaN,
      orNaN(getGeneralGVariance(x, s0, s1, s2))
    ),
    bivariateMoran: getStatistic(
      (scale * bivariateCross) / Math.sqrt(sumSquares * sumSquaresY),
      0,
      NaN,
      orNaN((scale * scale * lagDeviation) / ((n - 1) * sumSquares))
    ),
    joinCount: {
      blackBlack,
      blackWhite,
      whiteWhite: ordered[2] / 2,
      joins: binarySums.s0 / 2,
      expectedBlackBlack,
      varianceBlackBlack: binaryValid ? varianceBlackBlack : NaN,
      zBlackBlack: binaryValid ? zBlackBlack : NaN,
      pBlackBlack: binaryValid ? getTwoSidedPValue(zBlackBlack) : NaN,
      expectedBlackWhite,
      varianceBlackWhite: binaryValid ? varianceBlackWhite : NaN,
      zBlackWhite: binaryValid ? zBlackWhite : NaN,
      pBlackWhite: binaryValid ? getTwoSidedPValue(zBlackWhite) : NaN,
      ordered
    }
  };
}

/**
 * Exact mean and variance of `statistic(permuted values)` over every permutation of `values`
 * (tiny n only), the gold standard for randomization moments.
 */
export function getPermutationMoments(
  values: number[],
  statistic: (permuted: number[]) => number
): [number, number] {
  let count = 0;
  let sum = 0;
  let sumSquares = 0;
  const permute = (prefix: number[], rest: number[]) => {
    if (rest.length === 0) {
      const value = statistic(prefix);
      count++;
      sum += value;
      sumSquares += value * value;
      return;
    }
    for (let index = 0; index < rest.length; index++) {
      permute([...prefix, rest[index]], [...rest.slice(0, index), ...rest.slice(index + 1)]);
    }
  };
  permute([], values);
  const mean = sum / count;
  return [mean, sumSquares / count - mean * mean];
}

/** Dense helper for the node tests: weights matrix of a CSR over all rows. */
export function getDenseMatrix(weights: CPUSpatialWeights): number[][] {
  return getDenseWeights(
    weights,
    Array.from({length: weights.offsets.length - 1}, () => true)
  ).matrix;
}
