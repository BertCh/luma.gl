// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUNetworkIsochrones} parameter view. */
export const GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH = 12;

/** Per-frame settings of {@link GPUNetworkIsochrones}. */
export type GPUNetworkIsochroneSettings = {
  /**
   * Number of active ascending breaks, at most `breaks.length`. `breakCount + 1` bands exist:
   * band `k` holds cost in `[breaks[k - 1], breaks[k])`, band 0 everything below the first break
   * and the last band everything at or above the last active break (including unreached pixels).
   */
  breakCount: number;
  /** Raster extent `[minX, minY, maxX, maxY]` in the units of `nodePositions`. Row 0 is at `minY`. */
  extent: readonly [number, number, number, number];
  /**
   * Walking buffer radius in `nodePositions` units. Pixels within it of an edge sample get that
   * sample's cost plus `walkCostPerUnit` times the distance. The effective radius is at least half
   * a pixel diagonal so every sample marks its own pixel. Defaults to 0.
   */
  bufferRadius?: number;
  /** Cost per distance unit walked off the network inside the buffer. Defaults to 0. */
  walkCostPerUnit?: number;
  /** First band emitted as geometry. Defaults to 0. */
  firstBand?: number;
  /** Last band (inclusive) emitted as geometry. Defaults to `breakCount - 1`, so unreached space has no triangles. */
  lastBand?: number;
  /**
   * Cost limit of the cell-outline path: nodes with cost `<=` it contribute their cell. Defaults
   * to `Infinity`, every reached node.
   */
  cellCostLimit?: number;
};

/**
 * Packs per-frame {@link GPUNetworkIsochrones} parameters.
 *
 * Layout (float32): `[breakCount, minX, minY, maxX, maxY, bufferRadius, walkCostPerUnit,
 * firstBand, lastBand, cellCostLimit, 0, 0]`. `cellCostLimit` is capped to a finite float32 so a
 * limit of `Infinity` stays well defined in WGSL.
 *
 * @param settings Break count, extent, buffer, band window and cell cost limit.
 * @param target Optional destination of at least 12 elements.
 * @throws If the target is too short or a value is invalid.
 */
export function getGPUNetworkIsochroneParameterValues(
  settings: GPUNetworkIsochroneSettings,
  target: Float32Array = new Float32Array(GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH) {
    throw new Error(
      `Isochrones parameter target must hold ${GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH} elements`
    );
  }
  const {breakCount} = settings;
  const firstBand = settings.firstBand ?? 0;
  const lastBand = settings.lastBand ?? Math.max(breakCount - 1, 0);
  for (const [name, count] of [
    ['breakCount', breakCount],
    ['firstBand', firstBand],
    ['lastBand', lastBand]
  ] as const) {
    if (!Number.isSafeInteger(count) || count < 0) {
      throw new Error(`Isochrones ${name} must be a non-negative integer`);
    }
  }
  const [minX, minY, maxX, maxY] = settings.extent;
  if (![minX, minY, maxX, maxY].every(Number.isFinite) || maxX <= minX || maxY <= minY) {
    throw new Error('Isochrones extent must be finite with maxX > minX and maxY > minY');
  }
  const bufferRadius = settings.bufferRadius ?? 0;
  const walkCostPerUnit = settings.walkCostPerUnit ?? 0;
  if (!(bufferRadius >= 0) || !Number.isFinite(bufferRadius)) {
    throw new Error('Isochrones bufferRadius must be a non-negative finite number');
  }
  if (!(walkCostPerUnit >= 0) || !Number.isFinite(walkCostPerUnit)) {
    throw new Error('Isochrones walkCostPerUnit must be a non-negative finite number');
  }
  target.fill(0);
  target[0] = breakCount;
  target[1] = minX;
  target[2] = minY;
  target[3] = maxX;
  target[4] = maxY;
  target[5] = bufferRadius;
  target[6] = walkCostPerUnit;
  target[7] = firstBand;
  target[8] = lastBand;
  target[9] = Math.min(settings.cellCostLimit ?? Infinity, 3.0e38);
  return target;
}
