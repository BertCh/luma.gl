// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ChicagoTracts} from './spatial-lens-data';

/** Bounds `[minX, minY, maxX, maxY]` of interleaved planar positions. */
export function getPositionBounds(positions: Float32Array): [number, number, number, number] {
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let index = 0; index < positions.length; index += 2) {
    bounds[0] = Math.min(bounds[0], positions[index]);
    bounds[1] = Math.min(bounds[1], positions[index + 1]);
    bounds[2] = Math.max(bounds[2], positions[index]);
    bounds[3] = Math.max(bounds[3], positions[index + 1]);
  }
  return bounds;
}

/** Nearest-rank quantile of one axis (0 for x, 1 for y) of interleaved positions. */
export function getPositionQuantile(
  positions: Float32Array,
  axis: 0 | 1,
  fraction: number
): number {
  const values = Array.from({length: positions.length / 2}, (_, row) => positions[row * 2 + axis]);
  values.sort((left, right) => left - right);
  return values[Math.min(values.length - 1, Math.floor(fraction * values.length))];
}

/**
 * Projects the single-ring tract outlines to local metres and returns each tract's area centroid.
 * These planar points are the inputs of the region elements; no GPU result is derived on the CPU.
 */
export async function projectTractCentroids(tracts: ChicagoTracts): Promise<Float32Array> {
  const runner = await import('./s2-lens-run');
  const frame = runner.createS2ChicagoFrame();
  return runner.computeS2RingCentroids(frame.projectArray(tracts.vertices), tracts.ringOffsets);
}
