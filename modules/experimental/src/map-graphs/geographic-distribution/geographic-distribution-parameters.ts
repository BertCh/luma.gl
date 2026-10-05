// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a geographic-distribution parameter buffer. */
export const GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH = 8;

/** Default absolute convergence tolerance of the median centre, in position units. */
export const GPU_GEOGRAPHIC_DISTRIBUTION_DEFAULT_MEDIAN_TOLERANCE = 1e-3;

/** Axis scaling of the standard deviational ellipse. */
export type GPUGeographicDistributionEllipseConvention = 'arcgis' | 'standard';

/**
 * CPU description of the per-frame parameters of `GPUGeographicDistribution`.
 *
 * Every field can change between encodings without rebuilding or recompiling the graph.
 */
export type GPUGeographicDistributionParameters = {
  /**
   * Local origin `[x, y]` subtracted from every position before any sum, so float32 sums keep
   * precision when the planar coordinates are large (for example Web Mercator meters). Choose a
   * point near the data. Defaults to `[0, 0]`.
   */
  origin?: readonly [number, number];
  /**
   * Number of standard deviations that scale `standardDistances`, `ellipses` and the vertex
   * rings (ArcGIS offers 1, 2 and 3). Any positive number is accepted. Defaults to `1`.
   */
  standardDeviations?: number;
  /**
   * Axis scaling of the ellipse: `'arcgis'` scales the axes by `sqrt(2)` as the ArcGIS Directional
   * Distribution tool does, `'standard'` uses the plain standard deviation along each principal
   * axis. Defaults to `'arcgis'`.
   */
  ellipseConvention?: GPUGeographicDistributionEllipseConvention;
  /**
   * Treat lines as undirected (angles are doubled before averaging, so a line and its reverse
   * agree). Only affects `directionalMeans`. Defaults to `false`.
   */
  orientationOnly?: boolean;
  /**
   * Absolute tolerance, in position units, under which the last Weiszfeld step counts as
   * converged for `medianConverged`. Non-negative. Defaults to
   * {@link GPU_GEOGRAPHIC_DISTRIBUTION_DEFAULT_MEDIAN_TOLERANCE}.
   */
  medianTolerance?: number;
};

/**
 * Packs geographic-distribution parameters into the 8-element float32 layout read by
 * `GPUGeographicDistribution`.
 *
 * Layout: `[originX, originY, standardDeviations, ellipseScale, orientationOnly,
 * medianTolerance, 0, 0]`, where `ellipseScale` is `sqrt(2)` for the `'arcgis'` convention and `1`
 * for `'standard'`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 8 elements. A new array is returned when omitted.
 * @throws If a value is not finite, `standardDeviations <= 0`, `medianTolerance < 0`, the
 * convention is unknown, or `target` is too short.
 */
export function getGPUGeographicDistributionParameterValues(
  parameters: GPUGeographicDistributionParameters = {},
  target: Float32Array = new Float32Array(GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH) {
    throw new Error(
      `Geographic distribution target must hold ${GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH} elements`
    );
  }
  const [originX, originY] = parameters.origin ?? [0, 0];
  const standardDeviations = parameters.standardDeviations ?? 1;
  const convention = parameters.ellipseConvention ?? 'arcgis';
  const medianTolerance =
    parameters.medianTolerance ?? GPU_GEOGRAPHIC_DISTRIBUTION_DEFAULT_MEDIAN_TOLERANCE;
  for (const value of [originX, originY, standardDeviations, medianTolerance]) {
    if (!Number.isFinite(value)) {
      throw new Error(
        'Geographic distribution origin, standardDeviations and medianTolerance must be finite'
      );
    }
  }
  if (standardDeviations <= 0) {
    throw new Error('Geographic distribution standardDeviations must be positive');
  }
  if (medianTolerance < 0) {
    throw new Error('Geographic distribution medianTolerance must not be negative');
  }
  if (convention !== 'arcgis' && convention !== 'standard') {
    throw new Error('Geographic distribution ellipseConvention must be arcgis or standard');
  }
  target.set([
    originX,
    originY,
    standardDeviations,
    convention === 'arcgis' ? Math.SQRT2 : 1,
    parameters.orientationOnly ? 1 : 0,
    medianTolerance,
    0,
    0
  ]);
  return target;
}
