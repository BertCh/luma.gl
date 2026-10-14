// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURipleyDistanceFunctionsSettings} from '../../../src/gpu-dataframe/pair-statistics/ripley-distance-parameters';

import {
  getOracleBoundaryDistance,
  getOracleWindowBounds,
  getOracleWindowArea,
  isOracleInsideWindow,
  type OracleStudyWindow
} from '../../analysis-core/study-window-oracle';

/** Result of {@link computeRipleyDistanceFunctionsOnCPU}. */
export type RipleyDistanceOracleResult = {
  radii: number[];
  g: number[];
  f: number[];
  j: number[];
  /** Denominators of G and F per radius (n, or the border-eligible event count). */
  gDenominators: number[];
  fDenominators: number[];
};

/**
 * f64 oracle of `GPURipleyDistanceFunctions`: brute-force nearest-neighbor distances of the
 * included events and of a regular reference lattice, with the border (reduced sample) or no
 * correction.
 */
export function computeRipleyDistanceFunctionsOnCPU(
  scene: {positions: Float32Array; mask?: Uint32Array},
  parameters: GPURipleyDistanceFunctionsSettings,
  radiusCount: number,
  referenceGrid: readonly [number, number],
  studyWindow?: OracleStudyWindow
): RipleyDistanceOracleResult {
  const bounds = studyWindow ? getOracleWindowBounds(studyWindow) : parameters.bounds;
  if (bounds === 'study-window') throw new Error('Oracle needs a polygon for study-window bounds');
  const [minX, minY, maxX, maxY] = bounds;
  const maximumDistance = parameters.maximumDistance;
  const mode = parameters.edgeCorrection ?? 'border';
  const border = mode === 'border';
  const events: [number, number][] = [];
  for (let row = 0; row < scene.positions.length / 2; row++) {
    const x = scene.positions[2 * row];
    const y = scene.positions[2 * row + 1];
    if (
      (!scene.mask || scene.mask[row]) &&
      x >= minX &&
      x <= maxX &&
      y >= minY &&
      y <= maxY &&
      (!studyWindow || isOracleInsideWindow(studyWindow, [x, y]))
    ) {
      events.push([x, y]);
    }
  }
  const getBorderDistance = (x: number, y: number) =>
    studyWindow
      ? getOracleBoundaryDistance(studyWindow, [x, y])
      : Math.min(x - minX, maxX - x, y - minY, maxY - y);
  const getNearest = (x: number, y: number, skip: number): number => {
    let best = Infinity;
    events.forEach(([ex, ey], index) => {
      if (index !== skip) best = Math.min(best, Math.hypot(ex - x, ey - y));
    });
    return best;
  };
  const eventSamples = events.map(([x, y], index) => ({
    distance: getNearest(x, y, index),
    border: getBorderDistance(x, y)
  }));
  const referenceSamples: {distance: number; border: number}[] = [];
  for (let row = 0; row < referenceGrid[1]; row++) {
    for (let column = 0; column < referenceGrid[0]; column++) {
      const x = minX + ((column + 0.5) * (maxX - minX)) / referenceGrid[0];
      const y = minY + ((row + 0.5) * (maxY - minY)) / referenceGrid[1];
      if (studyWindow && !isOracleInsideWindow(studyWindow, [x, y])) continue;
      referenceSamples.push({distance: getNearest(x, y, -1), border: getBorderDistance(x, y)});
    }
  }
  const radii: number[] = [];
  const g: number[] = [];
  const f: number[] = [];
  const j: number[] = [];
  const gDenominators: number[] = [];
  const fDenominators: number[] = [];
  const windowWidth = maxX - minX;
  const windowHeight = maxY - minY;
  const erosion = studyWindow
    ? getReferenceLatticeErosion(studyWindow, referenceGrid, maximumDistance, radiusCount)
    : undefined;
  // Weight |W| / |W eroded by distance|, capped like the GPU kernel.
  const getHanischWeight = (distance: number) => {
    if (erosion) {
      const eroded = erosion.getArea(distance);
      const ratio = eroded > 0 ? Math.min(getOracleWindowArea(studyWindow!) / eroded, 1024) : 1024;
      return Math.floor(ratio * 16384 + 0.5) / 16384;
    }
    const eroded =
      Math.max(windowWidth - 2 * distance, 0) * Math.max(windowHeight - 2 * distance, 0);
    return eroded > 0 ? Math.min((windowWidth * windowHeight) / eroded, 1024) : 1024;
  };
  const step = maximumDistance / radiusCount;
  const getSlot = (distance: number) => Math.max(Math.ceil(distance / step) - 1, 0);
  /** Kaplan-Meier on the radius grid: hazards per slot, events before censoring. */
  const estimateKaplanMeier = (
    samples: {distance: number; border: number}[],
    slotIndex: number
  ) => {
    const events = new Array<number>(radiusCount + 1).fill(0);
    const censored = new Array<number>(radiusCount + 1).fill(0);
    for (const sample of samples) {
      if (sample.distance <= sample.border) {
        events[Math.min(getSlot(sample.distance), radiusCount)]++;
      } else {
        censored[Math.min(getSlot(sample.border), radiusCount)]++;
      }
    }
    let atRisk = samples.length;
    let survival = 1;
    for (let slot = 0; slot <= slotIndex; slot++) {
      if (atRisk >= 1 && events[slot] > 0) survival *= 1 - events[slot] / atRisk;
      atRisk -= events[slot] + censored[slot];
    }
    return {
      hits: (1 - survival) * samples.length,
      denominator: samples.length,
      size: samples.length
    };
  };
  /** Hanisch weighting with the GPU's treatment of points that have no neighbor within the maximum. */
  const estimateHanisch = (samples: {distance: number; border: number}[], radius: number) => {
    let numerator = 0;
    let denominator = 0;
    for (const sample of samples) {
      if (sample.distance > maximumDistance) {
        if (sample.border >= maximumDistance) denominator += getHanischWeight(maximumDistance);
      } else if (sample.distance <= sample.border) {
        const weight = getHanischWeight(sample.distance);
        denominator += weight;
        if (sample.distance <= radius) numerator += weight;
      }
    }
    return {hits: numerator, denominator, size: samples.length};
  };
  const estimate = (samples: {distance: number; border: number}[], radius: number) => {
    if (mode === 'kaplan-meier') {
      return estimateKaplanMeier(samples, Math.round((radius / maximumDistance) * radiusCount) - 1);
    }
    if (mode === 'hanisch') {
      return estimateHanisch(samples, radius);
    }
    const eligible = border ? samples.filter(sample => sample.border >= radius) : samples;
    const hits = eligible.filter(sample => sample.distance <= radius).length;
    return {hits, denominator: eligible.length, size: eligible.length};
  };
  for (let b = 0; b < radiusCount; b++) {
    const radius = (maximumDistance * (b + 1)) / radiusCount;
    radii.push(radius);
    const gEstimate = estimate(eventSamples, radius);
    const fEstimate = estimate(referenceSamples, radius);
    const gValue =
      events.length >= 2 && gEstimate.denominator >= 1
        ? gEstimate.hits / gEstimate.denominator
        : NaN;
    const fValue =
      events.length >= 1 && fEstimate.denominator >= 1
        ? fEstimate.hits / fEstimate.denominator
        : NaN;
    g.push(gValue);
    f.push(fValue);
    j.push(
      Number.isFinite(gValue) && Number.isFinite(fValue) && fValue < 1
        ? (1 - gValue) / (1 - fValue)
        : NaN
    );
    gDenominators.push(gEstimate.size);
    fDenominators.push(fEstimate.size);
  }
  return {radii, g, f, j, gDenominators, fDenominators};
}

/** Same 4 * radiusCount sub-slot histogram and linear interpolation as the polygon GPU estimator. */
export function getReferenceLatticeErosion(
  window: OracleStudyWindow,
  referenceGrid: readonly [number, number],
  maximumDistance: number,
  radiusCount: number
): {count: number; getArea: (distance: number) => number} {
  const [minimumX, minimumY, maximumX, maximumY] = getOracleWindowBounds(window);
  const slotCount = 4 * radiusCount;
  const counts = new Array<number>(slotCount + 1).fill(0);
  for (let row = 0; row < referenceGrid[1]; row++) {
    for (let column = 0; column < referenceGrid[0]; column++) {
      const point = [
        minimumX + ((column + 0.5) * (maximumX - minimumX)) / referenceGrid[0],
        minimumY + ((row + 0.5) * (maximumY - minimumY)) / referenceGrid[1]
      ] as const;
      if (!isOracleInsideWindow(window, point)) continue;
      const distance = Math.min(getOracleBoundaryDistance(window, point), maximumDistance);
      counts[Math.min(Math.floor((distance / maximumDistance) * slotCount), slotCount)]++;
    }
  }
  for (let slot = slotCount - 1; slot >= 0; slot--) counts[slot] += counts[slot + 1];
  return {
    count: counts[0],
    getArea(distance) {
      const scaled = Math.min(Math.max((distance / maximumDistance) * slotCount, 0), slotCount);
      const lower = Math.floor(scaled);
      const fraction = scaled - lower;
      const remaining =
        counts[lower] * (1 - fraction) + counts[Math.min(lower + 1, slotCount)] * fraction;
      return counts[0] ? (getOracleWindowArea(window) * remaining) / counts[0] : 0;
    }
  };
}
