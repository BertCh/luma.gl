// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getPhilox4x32, getPhiloxUnitFloat} from './dot-density-random';

/** Philox key word 1 of the per-slot Bernoulli remainder stream. */
export const DOT_DENSITY_REMAINDER_PURPOSE = 1;
/** Philox key word 1 of the candidate position stream. */
export const DOT_DENSITY_CANDIDATE_PURPOSE = 2;

/** GeoArrow polygon columns, as `GPUPolygonRasterization` and `GPUPointInPolygonJoin` take them. */
export type DotDensityCPUPolygons = {
  /** Flattened `(x, y)` vertices. */
  polygonPositions: Float32Array;
  /** Feature-to-polygon offsets, `featureCount + 1` entries. */
  featureOffsets: Uint32Array;
  /** Polygon-to-ring offsets with a terminal entry. */
  polygonOffsets: Uint32Array;
  /** Ring-to-vertex offsets with a terminal entry. */
  ringOffsets: Uint32Array;
};

/** Input of {@link generateDotsOnCPU}. */
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
  /** Rejection-sampling attempts per dot. */
  maximumAttempts: number;
  /** Optional dasymetric weight raster in `[0, 1]`. */
  mask?: {weights: Float32Array; width: number; height: number};
};

/** Result of {@link generateDotsOnCPU}. */
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
  /** `count * 2` positions; NaN for a dot whose attempts all failed. */
  positions: Float32Array;
  /** Feature row per dot. */
  featureIds: Uint32Array;
  /** Category per dot. */
  categories: Uint32Array;
  /** Number of dots whose attempts all failed. */
  failedCount: number;
};

const fround = Math.fround;

/**
 * Largest dots per slot. Keeps the u32 prefix sum from wrapping: any slot clamped here forces the
 * overflow flag.
 */
export function getDotSlotCountLimit(slotCount: number, capacity: number): number {
  return Math.min(capacity + 1, Math.floor(0xffffffff / Math.max(slotCount, 1)), 2 ** 31);
}

/** Returns the `[ringStart, ringEnd)` range and vertex bounds of one feature. */
export function getDotFeatureRings(
  polygons: DotDensityCPUPolygons,
  feature: number
): {
  ringStart: number;
  ringEnd: number;
  bounds: [number, number, number, number];
} {
  const ringStart = polygons.polygonOffsets[polygons.featureOffsets[feature]];
  const ringEnd = polygons.polygonOffsets[polygons.featureOffsets[feature + 1]];
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  for (
    let vertex = polygons.ringOffsets[ringStart];
    vertex < polygons.ringOffsets[ringEnd];
    vertex++
  ) {
    const x = polygons.polygonPositions[2 * vertex];
    const y = polygons.polygonPositions[2 * vertex + 1];
    bounds[0] = Math.min(bounds[0], x);
    bounds[1] = Math.min(bounds[1], y);
    bounds[2] = Math.max(bounds[2], x);
    bounds[3] = Math.max(bounds[3], y);
  }
  return {ringStart, ringEnd, bounds};
}

/**
 * Even-odd test over every ring of one feature, mirroring the WGSL crossing test. For a valid
 * multipolygon (non-overlapping parts, holes inside their shells) this equals containment.
 */
export function isPointInDotFeature(
  polygons: DotDensityCPUPolygons,
  ringStart: number,
  ringEnd: number,
  x: number,
  y: number
): boolean {
  const positions = polygons.polygonPositions;
  let inside = false;
  for (let ring = ringStart; ring < ringEnd; ring++) {
    const first = polygons.ringOffsets[ring];
    const end = polygons.ringOffsets[ring + 1];
    if (end <= first) {
      continue;
    }
    let previousX = positions[2 * (end - 1)];
    let previousY = positions[2 * (end - 1) + 1];
    for (let vertex = first; vertex < end; vertex++) {
      const currentX = positions[2 * vertex];
      const currentY = positions[2 * vertex + 1];
      if (currentY > y !== previousY > y) {
        const t = fround(fround(y - currentY) / fround(previousY - currentY));
        const crossX = fround(currentX + fround(t * fround(previousX - currentX)));
        if (x < crossX) {
          inside = !inside;
        }
      }
      previousX = currentX;
      previousY = currentY;
    }
  }
  return inside;
}

/** Nearest-cell mask weight, 0 outside the raster or for NaN. */
export function getDotMaskWeight(
  mask: {weights: Float32Array; width: number; height: number},
  extent: ArrayLike<number>,
  x: number,
  y: number
): number {
  const localX = fround(fround(x - extent[0]) / extent[2]);
  const localY = fround(fround(y - extent[1]) / extent[3]);
  if (!(localX >= 0 && localY >= 0 && localX < mask.width && localY < mask.height)) {
    return 0;
  }
  const weight = mask.weights[Math.floor(localY) * mask.width + Math.floor(localX)];
  return Number.isNaN(weight) ? 0 : weight;
}

/**
 * CPU oracle of `GPUDotDensity` (with `values`) and `GPURandomPointsInPolygon` (with `counts`).
 *
 * Counts, offsets, feature IDs, categories and the failed count are exact. Positions match the GPU
 * to f32 rounding of `min + u * (max - min)`, which a GPU may evaluate with a fused multiply-add.
 */
export function generateDotsOnCPU(input: DotDensityCPUInput): DotDensityCPUResult {
  const categoryCount = input.categoryCount ?? 1;
  const featureCount = input.featureOffsets.length - 1;
  const slotCount = featureCount * categoryCount;
  const words = input.parameters;
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const seed = words[0];
  const dotsPerUnit = floats[1];
  const maskExtent = [floats[2], floats[3], floats[4], floats[5]];
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
  const positions = new Float32Array(count * 2);
  const featureIds = new Uint32Array(count);
  const categories = new Uint32Array(count);
  let failedCount = 0;
  let dot = 0;
  for (let slot = 0; slot < slotCount && dot < count; slot++) {
    const feature = Math.floor(slot / categoryCount);
    const {ringStart, ringEnd, bounds} = getDotFeatureRings(input, feature);
    const minX = fround(bounds[0]);
    const minY = fround(bounds[1]);
    const spanX = fround(bounds[2] - bounds[0]);
    const spanY = fround(bounds[3] - bounds[1]);
    for (let rank = 0; rank < slotCounts[slot] && dot < count; rank++, dot++) {
      let x = NaN;
      let y = NaN;
      for (let attempt = 0; attempt < input.maximumAttempts; attempt++) {
        const random = getPhilox4x32(
          [slot, rank, attempt, 0],
          [seed, DOT_DENSITY_CANDIDATE_PURPOSE]
        );
        const candidateX = fround(minX + fround(getPhiloxUnitFloat(random[0]) * spanX));
        const candidateY = fround(minY + fround(getPhiloxUnitFloat(random[1]) * spanY));
        let accepted = isPointInDotFeature(input, ringStart, ringEnd, candidateX, candidateY);
        if (accepted && input.mask) {
          accepted =
            getPhiloxUnitFloat(random[2]) <
            getDotMaskWeight(input.mask, maskExtent, candidateX, candidateY);
        }
        if (accepted) {
          x = candidateX;
          y = candidateY;
          break;
        }
      }
      if (Number.isNaN(x)) {
        failedCount++;
      }
      positions[2 * dot] = x;
      positions[2 * dot + 1] = y;
      featureIds[dot] = feature;
      categories[dot] = slot % categoryCount;
    }
  }
  return {
    slotCounts,
    slotOffsets,
    requiredCount,
    count,
    overflow: requiredCount > input.capacity || clamped ? 1 : 0,
    positions,
    featureIds,
    categories,
    failedCount
  };
}
