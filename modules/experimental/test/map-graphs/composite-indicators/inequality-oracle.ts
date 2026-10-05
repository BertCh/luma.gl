// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUInequalitySettings} from '../../../src/map-graphs/composite-indicators/inequality-parameters';

/** CPU (float64) description of one `GPUInequality` evaluation. */
export type InequalityOracleProps = {
  values: ArrayLike<number>;
  zoneIds: ArrayLike<number>;
  zoneCount: number;
  weights?: ArrayLike<number>;
  mask?: ArrayLike<number>;
  settings?: GPUInequalitySettings;
  lorenzKnotCount?: number;
};

/** Per-zone and pooled results; NaN where undefined. */
export type InequalityOracleResult = {
  gini: number[];
  theilT: number[];
  theilL: number[];
  atkinson: number[];
  hoover: number[];
  palma: number[];
  mean: number[];
  count: number[];
  /** `zone * lorenzKnotCount + knot`. */
  lorenzKnots: number[];
  /** Same slot order as `GPU_INEQUALITY_GLOBAL_SUMMARY`. */
  globalSummary: number[];
};

type Entry = {x: number; w: number};

/** Lorenz curve of value-sorted entries at population fraction `p`, linear inside each entry. */
export function evaluateLorenz(sorted: Entry[], p: number): number {
  const totalWeight = sorted.reduce((sum, entry) => sum + entry.w, 0);
  const totalIncome = sorted.reduce((sum, entry) => sum + entry.w * entry.x, 0);
  let cumulativeWeight = 0;
  let cumulativeIncome = 0;
  for (const entry of sorted) {
    const nextWeight = cumulativeWeight + entry.w;
    const nextIncome = cumulativeIncome + entry.w * entry.x;
    const nextP = nextWeight / totalWeight;
    if (p <= nextP) {
      const previousP = cumulativeWeight / totalWeight;
      const previousL = cumulativeIncome / totalIncome;
      const nextL = nextIncome / totalIncome;
      return nextP === previousP
        ? nextL
        : previousL + ((p - previousP) / (nextP - previousP)) * (nextL - previousL);
    }
    cumulativeWeight = nextWeight;
    cumulativeIncome = nextIncome;
  }
  return 1;
}

/** Gini by the discrete rank formula, for unit weights. */
export function computeDiscreteGini(values: ArrayLike<number>): number {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const n = sorted.length;
  const sum = sorted.reduce((total, x) => total + x, 0);
  let weighted = 0;
  sorted.forEach((x, index) => {
    weighted += (index + 1) * x;
  });
  return (2 * weighted) / (n * sum) - (n + 1) / n;
}

function computeIndices(
  entries: Entry[],
  epsilon: number,
  palmaTop: number,
  palmaBottom: number,
  knotCount: number
) {
  const nan = Number.NaN;
  const sorted = [...entries].sort((a, b) => a.x - b.x);
  const totalWeight = sorted.reduce((sum, entry) => sum + entry.w, 0);
  const totalIncome = sorted.reduce((sum, entry) => sum + entry.w * entry.x, 0);
  const result = {
    mean: nan,
    gini: nan,
    theilT: nan,
    theilL: nan,
    atkinson: nan,
    hoover: nan,
    palma: nan,
    knots: new Array<number>(knotCount).fill(nan),
    totalWeight,
    totalIncome,
    weightedXLogX: 0
  };
  if (sorted.length === 0) {
    return result;
  }
  result.mean = totalIncome / totalWeight;
  for (const entry of sorted) {
    if (entry.x > 0) {
      result.weightedXLogX += entry.w * entry.x * Math.log(entry.x);
    }
  }
  if (!(totalIncome > 0)) {
    return result;
  }
  const mean = result.mean;
  const hasZero = sorted.some(entry => entry.x === 0);
  let area = 0;
  let cumulativeWeight = 0;
  let cumulativeIncome = 0;
  let theilT = 0;
  let sumLog = 0;
  let sumPower = 0;
  let sumAbsolute = 0;
  const power = 1 - epsilon;
  for (const entry of sorted) {
    const previousP = cumulativeWeight / totalWeight;
    const previousL = cumulativeIncome / totalIncome;
    cumulativeWeight += entry.w;
    cumulativeIncome += entry.w * entry.x;
    area +=
      (cumulativeWeight / totalWeight - previousP) * (previousL + cumulativeIncome / totalIncome);
    sumAbsolute += entry.w * Math.abs(entry.x - mean);
    if (entry.x > 0) {
      const ratio = entry.x / mean;
      theilT += entry.w * ratio * Math.log(ratio);
      sumLog += entry.w * Math.log(ratio);
      sumPower += entry.w * Math.pow(ratio, power);
    }
  }
  result.gini = 1 - area;
  result.theilT = theilT / totalWeight;
  result.theilL = hasZero ? nan : -sumLog / totalWeight;
  if (epsilon === 1) {
    result.atkinson = hasZero ? nan : 1 - Math.exp(sumLog / totalWeight);
  } else if (power > 0 || !hasZero) {
    const atkinson = 1 - Math.pow(sumPower / totalWeight, 1 / power);
    result.atkinson = Number.isFinite(atkinson) ? atkinson : nan;
  }
  result.hoover = sumAbsolute / (2 * totalIncome);
  const bottom = evaluateLorenz(sorted, palmaBottom);
  result.palma = bottom > 0 ? (1 - evaluateLorenz(sorted, 1 - palmaTop)) / bottom : nan;
  for (let knot = 0; knot < knotCount; knot++) {
    result.knots[knot] = knot === 0 ? 0 : evaluateLorenz(sorted, knot / (knotCount - 1));
  }
  return result;
}

/** Float64 reference for `GPUInequality` with the same inclusion rules and definitions. */
export function computeInequalityOnCPU(props: InequalityOracleProps): InequalityOracleResult {
  const {values, zoneIds, zoneCount, weights, mask} = props;
  const knotCount = props.lorenzKnotCount ?? 11;
  const epsilon = Math.fround(props.settings?.epsilon ?? 1);
  const palmaTop = Math.fround(props.settings?.palmaTopShare ?? 0.1);
  const palmaBottom = Math.fround(props.settings?.palmaBottomShare ?? 0.4);
  const zones: Entry[][] = Array.from({length: zoneCount}, () => []);
  const pooled: Entry[] = [];
  for (let row = 0; row < values.length; row++) {
    const x = values[row];
    const w = weights ? weights[row] : 1;
    const zone = zoneIds[row];
    if (
      !Number.isFinite(x) ||
      x < 0 ||
      !(zone < zoneCount) ||
      (mask && mask[row] === 0) ||
      !Number.isFinite(w) ||
      !(w > 0)
    ) {
      continue;
    }
    const entry = {x: x === 0 ? 0 : x, w};
    zones[zone].push(entry);
    pooled.push(entry);
  }
  const result: InequalityOracleResult = {
    gini: [],
    theilT: [],
    theilL: [],
    atkinson: [],
    hoover: [],
    palma: [],
    mean: [],
    count: [],
    lorenzKnots: [],
    globalSummary: []
  };
  const zoneResults = zones.map(entries =>
    computeIndices(entries, epsilon, palmaTop, palmaBottom, knotCount)
  );
  zoneResults.forEach((zoneResult, zone) => {
    result.gini.push(zoneResult.gini);
    result.theilT.push(zoneResult.theilT);
    result.theilL.push(zoneResult.theilL);
    result.atkinson.push(zoneResult.atkinson);
    result.hoover.push(zoneResult.hoover);
    result.palma.push(zoneResult.palma);
    result.mean.push(zoneResult.mean);
    result.count.push(zones[zone].length);
    result.lorenzKnots.push(...zoneResult.knots);
  });
  const pooledResult = computeIndices(pooled, epsilon, palmaTop, palmaBottom, 2);
  let between = 0;
  let within = 0;
  if (pooledResult.totalIncome > 0) {
    for (const zoneResult of zoneResults) {
      if (!(zoneResult.totalIncome > 0)) {
        continue;
      }
      const incomeShare = zoneResult.totalIncome / pooledResult.totalIncome;
      const populationShare = zoneResult.totalWeight / pooledResult.totalWeight;
      between += incomeShare * Math.log(incomeShare / populationShare);
      within += incomeShare * zoneResult.theilT;
    }
  }
  const defined = pooledResult.totalIncome > 0;
  result.globalSummary = [
    pooledResult.theilT,
    defined ? between : Number.NaN,
    defined ? within : Number.NaN,
    pooledResult.gini,
    pooled.length,
    pooledResult.mean,
    pooledResult.totalWeight,
    pooledResult.totalIncome
  ];
  return result;
}
