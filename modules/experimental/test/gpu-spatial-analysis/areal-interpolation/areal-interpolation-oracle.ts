// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSeededRandom} from './areal-interpolation-harness';

export const NO_ZONE = 0xffffffff;

/** Denominator choice: one value for both kinds, or per kind (default: extensive zone, intensive overlap). */
export type ArealOracleDenominator =
  | 'zone'
  | 'overlap'
  | {extensive?: 'zone' | 'overlap'; intensive?: 'zone' | 'overlap'};

/** CPU result of areal interpolation in float64. */
export type ArealOracle = {
  offsets: number[];
  neighbors: number[];
  areas: number[];
  extensive: number[];
  intensive: number[];
  /** Raster area of every source after anchoring. */
  sourceAreas: number[];
  /** Overlap area over the extensive denominator; NaN without raster area; 0 for a zero denominator. */
  sourceTransferredFractions: number[];
  /** Sources without raster area. */
  invalidCount: number;
};

/**
 * Claims the anchor cell of every zone without cells, visiting zones by ascending row: the cell
 * must belong to no zone or to a zone that keeps another cell (live counts). Mirrors
 * `getZoneAnchorNodes` and the numpy oracle `pycnophylactic_lib.apply_anchors`.
 */
export function applyZoneAnchors(
  zones: ArrayLike<number>,
  zoneCount: number,
  anchors: ArrayLike<number>
): number[] {
  const result = Array.from(zones);
  const counts = new Array<number>(zoneCount).fill(0);
  for (const zone of result) if (zone < zoneCount) counts[zone]++;
  for (let zone = 0; zone < zoneCount; zone++) {
    const anchor = anchors[zone];
    if (counts[zone] !== 0 || anchor >= result.length) continue;
    const owner = result[anchor];
    if (owner < zoneCount) {
      if (counts[owner] < 2) continue;
      counts[owner]--;
    }
    result[anchor] = zone;
    counts[zone] = 1;
  }
  return result;
}

/**
 * Spherical area of every cell of a longitude/latitude raster: `R^2 dLon (sin(top) - sin(bottom))`
 * in float64, row-major (`width * height`).
 */
export function computeSphericalCellAreas(
  width: number,
  height: number,
  originLatitude: number,
  cellWidth: number,
  cellHeight: number,
  radius = 6371007.180918475
): number[] {
  const toRadians = Math.PI / 180;
  const areas: number[] = [];
  for (let row = 0; row < height; row++) {
    const bottom = (originLatitude + row * cellHeight) * toRadians;
    const top = (originLatitude + (row + 1) * cellHeight) * toRadians;
    const area = radius * radius * cellWidth * toRadians * (Math.sin(top) - Math.sin(bottom));
    for (let column = 0; column < width; column++) areas.push(area);
  }
  return areas;
}

/**
 * Float64 reference for `GPUArealInterpolation`: pair areas, totals and both normalizations,
 * source diagnostics and optional source anchors.
 */
export function computeArealOracle(props: {
  sourceZones: ArrayLike<number>;
  targetZones: ArrayLike<number>;
  sourceCount: number;
  targetCount: number;
  cellWeights?: ArrayLike<number>;
  denominator?: ArealOracleDenominator;
  sourceAnchorCells?: ArrayLike<number>;
}): ArealOracle {
  const {targetZones, sourceCount, targetCount} = props;
  const sourceZones = props.sourceAnchorCells
    ? applyZoneAnchors(props.sourceZones, sourceCount, props.sourceAnchorCells)
    : props.sourceZones;
  const denominator = props.denominator;
  const extensiveDenominator =
    (typeof denominator === 'object' ? denominator.extensive : denominator) ?? 'zone';
  const intensiveDenominator =
    (typeof denominator === 'object' ? denominator.intensive : denominator) ?? 'overlap';
  const pairAreas = new Map<number, number>();
  const zoneSourceTotals = new Float64Array(sourceCount);
  const zoneTargetTotals = new Float64Array(targetCount);
  const overlapSourceTotals = new Float64Array(sourceCount);
  const overlapTargetTotals = new Float64Array(targetCount);
  for (let cell = 0; cell < sourceZones.length; cell++) {
    const source = sourceZones[cell];
    const target = targetZones[cell];
    let weight = props.cellWeights ? props.cellWeights[cell] : 1;
    if (!Number.isFinite(weight) || weight <= 0) weight = 0;
    const isPair = source < sourceCount && target < targetCount && weight > 0;
    if (isPair) {
      const key = target * sourceCount + source;
      pairAreas.set(key, (pairAreas.get(key) ?? 0) + weight);
      overlapSourceTotals[source] += weight;
      overlapTargetTotals[target] += weight;
    }
    if (source < sourceCount) zoneSourceTotals[source] += weight;
    if (target < targetCount) zoneTargetTotals[target] += weight;
  }
  const sourceTotals = extensiveDenominator === 'overlap' ? overlapSourceTotals : zoneSourceTotals;
  const targetTotals = intensiveDenominator === 'overlap' ? overlapTargetTotals : zoneTargetTotals;
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
  const sourceAreas = Array.from(zoneSourceTotals);
  const sourceTransferredFractions = sourceAreas.map((area, source) =>
    area > 0
      ? sourceTotals[source] > 0
        ? overlapSourceTotals[source] / sourceTotals[source]
        : 0
      : Number.NaN
  );
  return {
    offsets,
    neighbors,
    areas,
    extensive,
    intensive,
    sourceAreas,
    sourceTransferredFractions,
    invalidCount: sourceAreas.filter(area => !(area > 0)).length
  };
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

/** Result of {@link computePycnophylacticDetailedOracle}. */
export type PycnophylacticOracleResult = {
  /** Mass per cell (density times cell area). */
  mass: number[];
  /** Density per cell. */
  density: number[];
  /** Whether the tolerance was reached within the iterations. */
  converged: boolean;
  /** 1-based iteration after which the tolerance was reached. */
  convergedAfter?: number;
  /** Largest relative change per executed iteration (only with a tolerance). */
  changes: number[];
  /** Zones without cell area after anchoring. */
  invalidCount: number;
};

/**
 * Float64 reference for `GPUPycnophylactic` (same restore, clamp, rescale cycle) on a density
 * surface: optional cell areas (spherical cells), relative tolerance and zone anchors. Mirrors
 * the numpy oracle `pycnophylactic.py`.
 */
export function computePycnophylacticDetailedOracle(props: {
  width: number;
  height: number;
  zones: ArrayLike<number>;
  zoneCount: number;
  totals: ArrayLike<number>;
  iterations: number;
  kernel: 'rook' | 'box';
  cellAreas?: ArrayLike<number>;
  tolerance?: number;
  zoneAnchorCells?: ArrayLike<number>;
}): PycnophylacticOracleResult {
  const {width, height, zoneCount, iterations, kernel} = props;
  const zones = props.zoneAnchorCells
    ? applyZoneAnchors(props.zones, zoneCount, props.zoneAnchorCells)
    : props.zones;
  const cellCount = width * height;
  const inZone = (cell: number) => zones[cell] < zoneCount;
  const area = (cell: number) => {
    const value = props.cellAreas ? props.cellAreas[cell] : 1;
    return inZone(cell) && value > 0 ? value : 0;
  };
  const total = (zone: number) => (props.totals[zone] > 0 ? props.totals[zone] : 0);
  const zoneAreas = new Array<number>(zoneCount).fill(0);
  for (let cell = 0; cell < cellCount; cell++)
    if (inZone(cell)) zoneAreas[zones[cell]] += area(cell);
  let values = new Array<number>(cellCount).fill(0);
  for (let cell = 0; cell < cellCount; cell++) {
    if (inZone(cell) && zoneAreas[zones[cell]] > 0) {
      values[cell] = total(zones[cell]) / zoneAreas[zones[cell]];
    }
  }
  const masses = () => {
    const result = new Array<number>(zoneCount).fill(0);
    for (let cell = 0; cell < cellCount; cell++) {
      if (inZone(cell)) result[zones[cell]] += values[cell] * area(cell);
    }
    return result;
  };
  let converged = false;
  let convergedAfter: number | undefined;
  const changes: number[] = [];
  for (let iteration = 0; iteration < iterations; iteration++) {
    if (converged) continue;
    const previous = values;
    const next = new Array<number>(cellCount).fill(0);
    for (let cell = 0; cell < cellCount; cell++) {
      if (!inZone(cell)) continue;
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
          if (inZone(y * width + x)) {
            sum += previous[y * width + x];
            count++;
          }
        }
      }
      next[cell] = count > 0 ? sum / count : previous[cell];
    }
    values = next;
    let zoneMasses = masses();
    for (let cell = 0; cell < cellCount; cell++) {
      if (!inZone(cell)) continue;
      const zone = zones[cell];
      const shift = zoneAreas[zone] > 0 ? (total(zone) - zoneMasses[zone]) / zoneAreas[zone] : 0;
      values[cell] = Math.max(values[cell] + shift, 0);
    }
    zoneMasses = masses();
    for (let cell = 0; cell < cellCount; cell++) {
      if (!inZone(cell)) continue;
      const zone = zones[cell];
      values[cell] =
        zoneMasses[zone] > 0
          ? values[cell] * (total(zone) / zoneMasses[zone])
          : zoneAreas[zone] > 0
            ? total(zone) / zoneAreas[zone]
            : 0;
    }
    if (props.tolerance !== undefined) {
      let change = 0;
      for (let cell = 0; cell < cellCount; cell++) {
        if (!inZone(cell) || !(zoneAreas[zones[cell]] > 0)) continue;
        const reference = total(zones[cell]) / zoneAreas[zones[cell]];
        if (reference > 0)
          change = Math.max(change, Math.abs(values[cell] - previous[cell]) / reference);
      }
      changes.push(change);
      if (change <= props.tolerance) {
        converged = true;
        convergedAfter = iteration + 1;
      }
    }
  }
  return {
    mass: values.map((value, cell) => value * area(cell)),
    density: values,
    converged,
    convergedAfter,
    changes,
    invalidCount: zoneAreas.filter(value => !(value > 0)).length
  };
}

/** Float64 reference for `GPUPycnophylactic` (planar, no tolerance or anchors): mass per cell. */
export function computePycnophylacticOracle(props: {
  width: number;
  height: number;
  zones: ArrayLike<number>;
  zoneCount: number;
  totals: ArrayLike<number>;
  iterations: number;
  kernel: 'rook' | 'box';
}): number[] {
  return computePycnophylacticDetailedOracle(props).mass;
}
