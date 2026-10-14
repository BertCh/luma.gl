// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getPhilox4x32, getPhiloxUnitFloat, PHILOX_PURPOSES} from '../../../src/utils/philox-random';

const fround = Math.fround;

/** Statistic scale of an envelope. */
export type RipleyEnvelopeOracleStatistic = 'k' | 'l' | 'l-minus-r';

/** CPU result of {@link computeRipleyEnvelopeOnCPU}. */
export type RipleyEnvelopeOracleResult = {
  lower: number[];
  upper: number[];
  globalLower: number[];
  globalUpper: number[];
  theoretical: number[];
  /** `T_j = max_r |L_j(r) - r|` per simulation, NaN when no radius is finite. */
  deviations: number[];
  critical: number;
  observedDeviation: number;
  pValue: number;
};

/**
 * CPU mirror of the `GPURipleyEnvelope` simulation kernel: `count` uniform points in the window
 * from Philox4x32-10 with key `(seed, 'RIPF')` and counter `(row, simulationIndex, 0, 0)`, using
 * f32 arithmetic in the same order as the shader (a GPU fused multiply-add can differ by one ulp).
 */
export function simulateCSROnCPU(
  seed: number,
  simulationIndex: number,
  count: number,
  bounds: readonly [number, number, number, number]
): Float32Array {
  const [minX, minY, maxX, maxY] = bounds.map(fround);
  const positions = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    const words = getPhilox4x32(
      [row, simulationIndex, 0, 0],
      [seed, PHILOX_PURPOSES.ripleyEnvelopeSimulation.id]
    );
    positions[row * 2] = fround(minX + fround(getPhiloxUnitFloat(words[0]) * fround(maxX - minX)));
    positions[row * 2 + 1] = fround(
      minY + fround(getPhiloxUnitFloat(words[1]) * fround(maxY - minY))
    );
  }
  return positions;
}

function getLFromK(kValue: number): number {
  return Math.sqrt(Math.max(kValue, 0) / Math.PI);
}

/**
 * f64 pointwise and global CSR envelopes from simulated K curves, following the definitions in
 * `GPURipleyEnvelope`: NaN entries are skipped per radius / per curve, ranks break ties by
 * simulation index, the global band is `r +- (rank-th largest max |L_j(r) - r|)`.
 */
export function computeRipleyEnvelopeOnCPU(
  simulatedK: readonly (readonly number[])[],
  observedK: readonly number[] | undefined,
  radii: readonly number[],
  rank: number,
  statistic: RipleyEnvelopeOracleStatistic
): RipleyEnvelopeOracleResult {
  const fromK = (kValue: number, radius: number) =>
    statistic === 'k' ? kValue : getLFromK(kValue) - (statistic === 'l-minus-r' ? radius : 0);
  const fromL = (lValue: number, radius: number) =>
    statistic === 'k'
      ? Math.PI * Math.max(lValue, 0) ** 2
      : lValue - (statistic === 'l-minus-r' ? radius : 0);
  const getMaximumDeviation = (curve: readonly number[]) => {
    let deviation = 0;
    let anyFinite = false;
    curve.forEach((kValue, b) => {
      if (Number.isFinite(kValue)) {
        deviation = Math.max(deviation, Math.abs(getLFromK(kValue) - radii[b]));
        anyFinite = true;
      }
    });
    return anyFinite ? deviation : NaN;
  };
  const deviations = simulatedK.map(getMaximumDeviation);
  const finiteDeviations = deviations.filter(Number.isFinite).sort((a, b) => b - a);
  const critical = finiteDeviations.length >= rank ? finiteDeviations[rank - 1] : NaN;
  const lower: number[] = [];
  const upper: number[] = [];
  for (let b = 0; b < radii.length; b++) {
    const column = simulatedK
      .map(curve => curve[b])
      .filter(Number.isFinite)
      .sort((x, y) => x - y);
    const defined = column.length >= 2 * rank;
    lower.push(defined ? fromK(column[rank - 1], radii[b]) : NaN);
    upper.push(defined ? fromK(column[column.length - rank], radii[b]) : NaN);
  }
  const observedDeviation = observedK ? getMaximumDeviation(observedK) : NaN;
  const pValue = Number.isFinite(observedDeviation)
    ? (1 + finiteDeviations.filter(value => value >= observedDeviation).length) /
      (finiteDeviations.length + 1)
    : NaN;
  return {
    lower,
    upper,
    globalLower: radii.map(radius => fromL(radius - critical, radius)),
    globalUpper: radii.map(radius => fromL(radius + critical, radius)),
    theoretical: radii.map(radius => fromL(radius, radius)),
    deviations,
    critical,
    observedDeviation,
    pValue
  };
}

import {
  getOracleWindowBounds,
  getOracleOrientedEdges,
  isOracleInsideWindow,
  type OracleStudyWindow
} from '../../analysis-core/study-window-oracle';
import {STUDY_WINDOW_BOUNDARY_CELL_BIT} from '../../../src/utils/study-window-wgsl';

/** Conservative equal-area cell cover for the rejection-sampler oracle (test only). */
export function getOracleSamplerCells(
  window: OracleStudyWindow,
  grid: readonly [number, number]
): Uint32Array {
  const [minimumX, minimumY, maximumX, maximumY] = getOracleWindowBounds(window);
  const width = (maximumX - minimumX) / grid[0];
  const height = (maximumY - minimumY) / grid[1];
  const edges = getOracleOrientedEdges(window);
  const cells: number[] = [];
  for (let row = 0; row < grid[1]; row++) {
    for (let column = 0; column < grid[0]; column++) {
      const left = minimumX + column * width;
      const bottom = minimumY + row * height;
      const boundary = edges.some(
        edge =>
          Math.max(edge[0], edge[2]) >= left &&
          Math.min(edge[0], edge[2]) <= left + width &&
          Math.max(edge[1], edge[3]) >= bottom &&
          Math.min(edge[1], edge[3]) <= bottom + height
      );
      if (boundary || isOracleInsideWindow(window, [left + width / 2, bottom + height / 2])) {
        cells.push((row * grid[0] + column) | (boundary ? STUDY_WINDOW_BOUNDARY_CELL_BIT : 0));
      }
    }
  }
  return Uint32Array.from(cells);
}

/** Polygon Philox sampler with the GPU counter, cell flag, rejection restart and attempt limit. */
export function simulatePolygonCSROnCPU(
  seed: number,
  simulationIndex: number,
  count: number,
  window: OracleStudyWindow,
  grid: readonly [number, number],
  maximumAttempts = 32,
  cells = getOracleSamplerCells(window, grid)
): {positions: Float32Array; mask: Uint32Array; unplacedCount: number} {
  const [minimumX, minimumY, maximumX, maximumY] = getOracleWindowBounds(window).map(fround);
  const sizeX = fround(fround(maximumX - minimumX) / grid[0]);
  const sizeY = fround(fround(maximumY - minimumY) / grid[1]);
  const positions = new Float32Array(count * 2);
  const mask = new Uint32Array(count);
  let unplacedCount = 0;
  for (let row = 0; row < count; row++) {
    for (let attempt = 0; attempt < maximumAttempts && cells.length; attempt++) {
      const words = getPhilox4x32(
        [row, simulationIndex, attempt, 0],
        [seed, PHILOX_PURPOSES.ripleyEnvelopeSimulation.id]
      );
      const cell =
        cells[
          Math.min(
            Math.floor(fround(getPhiloxUnitFloat(words[0]) * cells.length)),
            cells.length - 1
          )
        ];
      const cellIndex = cell & 0x7fffffff;
      const originX = fround(minimumX + fround(sizeX * (cellIndex % grid[0])));
      const originY = fround(minimumY + fround(sizeY * Math.floor(cellIndex / grid[0])));
      const point = [
        fround(originX + fround(getPhiloxUnitFloat(words[1]) * sizeX)),
        fround(originY + fround(getPhiloxUnitFloat(words[2]) * sizeY))
      ] as const;
      positions.set(point, row * 2);
      if (!(cell & STUDY_WINDOW_BOUNDARY_CELL_BIT) || isOracleInsideWindow(window, point)) {
        mask[row] = 1;
        break;
      }
    }
    if (!mask[row]) unplacedCount++;
  }
  return {positions, mask, unplacedCount};
}
