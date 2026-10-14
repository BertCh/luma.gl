// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getPhilox4x32, getPhiloxUnitFloat} from '../../../src/utils/philox-random';
import {
  DOT_DENSITY_REMAINDER_PURPOSE,
  getDotSlotCountLimit
} from '../../../src/gpu-spatial-analysis/dot-density/dot-density-constants';

/** GeoArrow polygon columns, as `GPUPolygonRasterization` and `GPUPointInPolygonJoin` take them. */
export type DotDensityCPUPolygons = {
  /** Flattened `(x, y)` vertices. */
  positions: Float32Array;
  /** Feature-to-polygon offsets, `featureCount + 1` entries. */
  featureOffsets: Uint32Array;
  /** Polygon-to-ring offsets with a terminal entry. */
  polygonOffsets: Uint32Array;
  /** Ring-to-vertex offsets with a terminal entry. */
  ringOffsets: Uint32Array;
};

/** Input of {@link generateDotCountsOnCPU}. */
export type DotDensityCPUInput = DotDensityCPUPolygons & {
  /** Float values, `featureCount * categoryCount` rows (dot density), or `undefined` with `counts`. */
  values?: Float32Array;
  /** Integer point counts per feature (random points in polygon). */
  counts?: Uint32Array;
  /** Categories per feature. Defaults to 1. */
  categoryCount?: number;
  /** Parameter words from `getGPUDotDensityParameterValues`. */
  parameters: Uint32Array;
  /** Output capacity in dots. */
  capacity: number;
};

/** Result of {@link generateDotCountsOnCPU}. */
export type DotDensityCPUResult = {
  /** Dots per slot after clamping. */
  slotCounts: Uint32Array;
  /** Exclusive prefix of `slotCounts`. */
  slotOffsets: Uint32Array;
  /** Unclamped total dot count. */
  requiredCount: number;
  /** `min(requiredCount, capacity)`. */
  count: number;
  /** 1 when the total exceeds the capacity or a slot count was clamped. */
  overflow: number;
  /** Feature row per dot. */
  featureIds: Uint32Array;
  /** Category per dot. */
  categories: Uint32Array;
};

const fround = Math.fround;

/**
 * CPU oracle of the dot counts of `GPUDotDensity` (with `values`) and `GPURandomPointsInPolygon`
 * (with `counts`): slot counts, offsets, totals, overflow, feature IDs and categories, all exact.
 *
 * Positions are not generated here: the GPU samples inside a GPU triangulation, so positions are
 * checked by property (all dots placed, inside the feature per shapely) in the GPU spec.
 */
export function generateDotCountsOnCPU(input: DotDensityCPUInput): DotDensityCPUResult {
  const categoryCount = input.categoryCount ?? 1;
  const featureCount = input.featureOffsets.length - 1;
  const slotCount = featureCount * categoryCount;
  const words = input.parameters;
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const seed = words[0];
  const dotsPerUnit = floats[1];
  const limit = getDotSlotCountLimit(slotCount, input.capacity);

  const slotCounts = new Uint32Array(slotCount);
  let clamped = false;
  for (let slot = 0; slot < slotCount; slot++) {
    let dots: number;
    if (input.counts) {
      dots = input.counts[slot];
    } else {
      const value = input.values![slot];
      const scaled = fround(value * dotsPerUnit);
      const remainder = getPhiloxUnitFloat(
        getPhilox4x32([slot, 0, 0, 0], [seed, DOT_DENSITY_REMAINDER_PURPOSE])[0]
      );
      const ceiling = Math.ceil(fround(scaled - remainder));
      // NaN, negative values and a non-positive scale draw nothing.
      dots = Number.isFinite(scaled) && scaled > 0 && ceiling > 0 ? ceiling : 0;
      if (scaled === Infinity) {
        dots = Infinity;
      }
    }
    if (dots > limit) {
      clamped = true;
      dots = limit;
    }
    slotCounts[slot] = dots;
  }
  const slotOffsets = new Uint32Array(slotCount);
  let requiredCount = 0;
  for (let slot = 0; slot < slotCount; slot++) {
    slotOffsets[slot] = requiredCount;
    requiredCount += slotCounts[slot];
  }
  const count = Math.min(requiredCount, input.capacity);
  const featureIds = new Uint32Array(count);
  const categories = new Uint32Array(count);
  let dot = 0;
  for (let slot = 0; slot < slotCount && dot < count; slot++) {
    for (let rank = 0; rank < slotCounts[slot] && dot < count; rank++, dot++) {
      featureIds[dot] = Math.floor(slot / categoryCount);
      categories[dot] = slot % categoryCount;
    }
  }
  return {
    slotCounts,
    slotOffsets,
    requiredCount,
    count,
    overflow: requiredCount > input.capacity || clamped ? 1 : 0,
    featureIds,
    categories
  };
}
