// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSeededRandom} from './areal-interpolation-harness';

export const NO_ZONE = 0xffffffff;

/** CPU result of areal interpolation in float64. */
export type ArealOracle = {
  offsets: number[];
  neighbors: number[];
  areas: number[];
  extensive: number[];
  intensive: number[];
};

/** Float64 reference for `GPUArealInterpolation`: pair areas, totals and both normalizations. */
export function computeArealOracle(props: {
  sourceZones: ArrayLike<number>;
  targetZones: ArrayLike<number>;
  sourceCount: number;
  targetCount: number;
  cellWeights?: ArrayLike<number>;
  denominator: 'zone' | 'overlap';
}): ArealOracle {
  const {sourceZones, targetZones, sourceCount, targetCount, denominator} = props;
  const pairAreas = new Map<number, number>();
  const sourceTotals = new Float64Array(sourceCount);
  const targetTotals = new Float64Array(targetCount);
  for (let cell = 0; cell < sourceZones.length; cell++) {
    const source = sourceZones[cell];
    const target = targetZones[cell];
    let weight = props.cellWeights ? props.cellWeights[cell] : 1;
    if (!Number.isFinite(weight) || weight <= 0) weight = 0;
    const isPair = source < sourceCount && target < targetCount && weight > 0;
    if (isPair) {
      const key = target * sourceCount + source;
      pairAreas.set(key, (pairAreas.get(key) ?? 0) + weight);
    }
    if (denominator === 'zone' || isPair) {
      if (source < sourceCount) sourceTotals[source] += weight;
      if (target < targetCount) targetTotals[target] += weight;
    }
  }
  const keys = [...pairAreas.keys()].sort((a, b) => a - b);
  const offsets = new Array<number>(targetCount + 1).fill(0);
  const neighbors: number[] = [];
  const areas: number[] = [];
  const extensive: number[] = [];
  const intensive: number[] = [];
  for (const key of keys) {
    const target = Math.floor(key / sourceCount);
    const source = key % sourceCount;
    const area = pairAreas.get(key)!;
    offsets[target + 1]++;
    neighbors.push(source);
    areas.push(area);
    extensive.push(sourceTotals[source] > 0 ? area / sourceTotals[source] : 0);
    intensive.push(targetTotals[target] > 0 ? area / targetTotals[target] : 0);
  }
  for (let row = 0; row < targetCount; row++) offsets[row + 1] += offsets[row];
  return {offsets, neighbors, areas, extensive, intensive};
}

/** `sum_s w_ts v_s` per target. */
export function computeTransferOracle(
  offsets: number[],
  neighbors: number[],
  weights: number[],
  values: ArrayLike<number>
): number[] {
  const result: number[] = [];
  for (let row = 0; row + 1 < offsets.length; row++) {
    let total = 0;
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      total += weights[slot] * values[neighbors[slot]];
    }
    result.push(total);
  }
  return result;
}

/** Block partition of a `width x height` grid with some cells outside the system. */
export function createBlockZones(
  width: number,
  height: number,
  blockWidth: number,
  blockHeight: number,
  options: {shiftX?: number; shiftY?: number; holeSeed?: number; holeRate?: number} = {}
): {zones: Uint32Array; zoneCount: number} {
  const columns = Math.ceil((width + (options.shiftX ?? 0)) / blockWidth);
  const rows = Math.ceil((height + (options.shiftY ?? 0)) / blockHeight);
  const random = createSeededRandom(options.holeSeed ?? 1);
  const zones = new Uint32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const column = Math.floor((x + (options.shiftX ?? 0)) / blockWidth);
      const row = Math.floor((y + (options.shiftY ?? 0)) / blockHeight);
      zones[y * width + x] =
        options.holeRate && random() < options.holeRate ? NO_ZONE : row * columns + column;
    }
  }
  return {zones, zoneCount: columns * rows};
}

/** Float64 reference for `GPUPycnophylactic` (same restore, clamp, rescale cycle). */
export function computePycnophylacticOracle(props: {
  width: number;
  height: number;
  zones: ArrayLike<number>;
  zoneCount: number;
  totals: ArrayLike<number>;
  iterations: number;
  kernel: 'rook' | 'box';
}): number[] {
  const {width, height, zones, zoneCount, iterations, kernel} = props;
  const cellCount = width * height;
  const total = (zone: number) => (props.totals[zone] > 0 ? props.totals[zone] : 0);
  const counts = new Array<number>(zoneCount).fill(0);
  for (let cell = 0; cell < cellCount; cell++) if (zones[cell] < zoneCount) counts[zones[cell]]++;
  let values = new Array<number>(cellCount).fill(0);
  for (let cell = 0; cell < cellCount; cell++) {
    if (zones[cell] < zoneCount) values[cell] = total(zones[cell]) / counts[zones[cell]];
  }
  const sums = () => {
    const result = new Array<number>(zoneCount).fill(0);
    for (let cell = 0; cell < cellCount; cell++)
      if (zones[cell] < zoneCount) result[zones[cell]] += values[cell];
    return result;
  };
  for (let iteration = 0; iteration < iterations; iteration++) {
    const next = new Array<number>(cellCount).fill(0);
    for (let cell = 0; cell < cellCount; cell++) {
      if (zones[cell] >= zoneCount) continue;
      const column = cell % width;
      const row = Math.floor(cell / width);
      let sum = 0;
      let count = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (kernel === 'rook' && (dx === 0) === (dy === 0)) continue;
          const x = column + dx;
          const y = row + dy;
          if (x < 0 || y < 0 || x >= width || y >= height) continue;
          if (zones[y * width + x] < zoneCount) {
            sum += values[y * width + x];
            count++;
          }
        }
      }
      next[cell] = count > 0 ? sum / count : values[cell];
    }
    values = next;
    let zoneSums = sums();
    for (let cell = 0; cell < cellCount; cell++) {
      const zone = zones[cell];
      if (zone < zoneCount)
        values[cell] = Math.max(values[cell] + (total(zone) - zoneSums[zone]) / counts[zone], 0);
    }
    zoneSums = sums();
    for (let cell = 0; cell < cellCount; cell++) {
      const zone = zones[cell];
      if (zone < zoneCount) {
        values[cell] =
          zoneSums[zone] > 0
            ? (values[cell] * total(zone)) / zoneSums[zone]
            : total(zone) / counts[zone];
      }
    }
  }
  return values;
}
