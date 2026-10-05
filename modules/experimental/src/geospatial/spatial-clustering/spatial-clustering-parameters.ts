// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a spatial-clustering parameter buffer. */
export const GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH = 8;

/** Label and root value of points that belong to no cluster. */
export const GPU_SPATIAL_CLUSTERING_NOISE = 0xffffffff;

/** Largest accepted `minimumPoints`, the largest integer a float32 holds exactly. */
const MAXIMUM_MINIMUM_POINTS = 2 ** 24;

/**
 * CPU description of the per-frame parameters of one {@link GPUSpatialClustering}.
 *
 * Distances use the same planar units as the positions.
 */
export type GPUSpatialClusteringParameters = {
  /**
   * Inclusive `[minX, minY, maxX, maxY]` extent of the data. Points outside it, or with a
   * non-finite coordinate, are excluded and labeled noise. Bounds should cover the data: they
   * size the active cell lattice, and a tighter extent makes the neighbor search cheaper.
   * Bounds with `maxX < minX` or `maxY < minY` exclude every point.
   */
  bounds: readonly [number, number, number, number];
  /** Neighbor radius. Two points are neighbors when their distance is `<= epsilon`. Positive. */
  epsilon: number;
  /**
   * Integer `>= 1`. A point is a core point when its epsilon-neighborhood, including the point
   * itself, holds at least this many points.
   */
  minimumPoints: number;
};

/**
 * Packs spatial-clustering parameters into the 8-element float32 layout read by
 * `GPUSpatialClustering`.
 *
 * Layout: `[minX, minY, maxX, maxY, epsilon, minimumPoints, 0, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, `epsilon <= 0`, `minimumPoints` is not an integer in
 * `[1, 2^24]`, or `target` is too short.
 */
export function getGPUSpatialClusteringParameterValues(
  parameters: GPUSpatialClusteringParameters,
  target: Float32Array = new Float32Array(GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH) {
    throw new Error(
      `Spatial clustering target must hold ${GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH} elements`
    );
  }
  for (const value of [...parameters.bounds, parameters.epsilon]) {
    if (!Number.isFinite(value)) {
      throw new Error('Spatial clustering bounds and epsilon must be finite');
    }
  }
  if (parameters.epsilon <= 0) {
    throw new Error('Spatial clustering epsilon must be positive');
  }
  if (
    !Number.isInteger(parameters.minimumPoints) ||
    parameters.minimumPoints < 1 ||
    parameters.minimumPoints > MAXIMUM_MINIMUM_POINTS
  ) {
    throw new Error(
      `Spatial clustering minimumPoints must be an integer from 1 to ${MAXIMUM_MINIMUM_POINTS}`
    );
  }
  target.set([...parameters.bounds, parameters.epsilon, parameters.minimumPoints, 0, 0]);
  return target;
}
