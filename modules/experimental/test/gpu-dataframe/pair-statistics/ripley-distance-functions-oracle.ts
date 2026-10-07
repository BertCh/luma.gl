// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURipleyDistanceParameters} from '../../../src/gpu-dataframe/pair-statistics/ripley-distance-parameters';

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
  parameters: GPURipleyDistanceParameters,
  radiusCount: number,
  referenceGrid: readonly [number, number]
): RipleyDistanceOracleResult {
  const [minX, minY, maxX, maxY] = parameters.bounds;
  const maximumDistance = parameters.maximumDistance;
  const mode = parameters.edgeCorrection ?? 'border';
  const border = mode === 'border';
  const events: [number, number][] = [];
  for (let row = 0; row < scene.positions.length / 2; row++) {
    const x = scene.positions[2 * row];
    const y = scene.positions[2 * row + 1];
    if ((!scene.mask || scene.mask[row]) && x >= minX && x <= maxX && y >= minY && y <= maxY) {
      events.push([x, y]);
    }
  }
  const getBorderDistance = (x: number, y: number) =>
    Math.min(x - minX, maxX - x, y - minY, maxY - y);
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
  // Weight |W| / |W eroded by distance|, capped like the GPU kernel.
  const getHanischWeight = (distance: number) => {
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
