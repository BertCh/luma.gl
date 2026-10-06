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
  const border = (parameters.edgeCorrection ?? 'border') === 'border';
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
  const estimate = (samples: {distance: number; border: number}[], radius: number) => {
    const eligible = border ? samples.filter(sample => sample.border >= radius) : samples;
    const hits = eligible.filter(sample => sample.distance <= radius).length;
    return {hits, denominator: eligible.length};
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
    gDenominators.push(gEstimate.denominator);
    fDenominators.push(fEstimate.denominator);
  }
  return {radii, g, f, j, gDenominators, fDenominators};
}
