// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Logical coordinate spaces consumed by spatial-analysis operations. */
export type GPUSpatialCoordinateSpace =
  | 'planar'
  | 'longitude-latitude'
  | 'discrete-grid'
  | 'topology-only';

/** Metric applied to coordinates, kept separate from their coordinate space. */
export type GPUSpatialMetric = 'none' | 'native' | 'great-circle' | 'rhumb' | 'ellipsoidal';

/** Units of metric results. Native units are meaningful only for a native planar metric. */
export type GPUSpatialUnit = 'native' | 'meters' | 'kilometers';

/** Parameters of an ellipsoidal metric model. */
export type GPUSpatialEllipsoid = {
  semiMajorAxis: number;
  flattening: number;
};

/** Compile-time coordinate and metric descriptor; CRS transformation is deliberately absent. */
export type GPUSpatialContext = {
  coordinateSpace: GPUSpatialCoordinateSpace;
  metric: GPUSpatialMetric;
  units?: GPUSpatialUnit;
  sphereRadius?: number;
  ellipsoid?: GPUSpatialEllipsoid;
  /** Grid family when `coordinateSpace` is `discrete-grid`, for example `h3` or `quadbin`. */
  gridFamily?: string;
  /** Fixed grid resolution when required by an operation. */
  gridResolution?: number;
};

/** Result of checking whether two graph ports share a usable spatial context. */
export type GPUSpatialContextCompatibility =
  | {compatible: true}
  | {compatible: false; reason: string};

/** Validates one spatial context without performing CRS parsing or reprojection. */
export function validateGPUSpatialContext(id: string, context: GPUSpatialContext): void {
  const {coordinateSpace, metric} = context;
  const allowedMetrics: Record<GPUSpatialCoordinateSpace, readonly GPUSpatialMetric[]> = {
    planar: ['none', 'native'],
    'longitude-latitude': ['none', 'great-circle', 'rhumb', 'ellipsoidal'],
    'discrete-grid': ['none'],
    'topology-only': ['none']
  };
  if (!allowedMetrics[coordinateSpace]?.includes(metric)) {
    // Spatial analysis does not silently reinterpret or reproject coordinates.
    throw new Error(`${id} metric ${metric} is incompatible with ${coordinateSpace}`);
  }
  if (metric === 'native' && context.units && context.units !== 'native') {
    throw new Error(`${id} native metrics must use native units`);
  }
  if (
    metric !== 'none' &&
    metric !== 'native' &&
    context.units !== undefined &&
    context.units === 'native'
  ) {
    throw new Error(`${id} geographic metrics cannot use native units`);
  }
  if (context.sphereRadius !== undefined) {
    if (!['great-circle', 'rhumb'].includes(metric) || !isPositiveFinite(context.sphereRadius)) {
      throw new Error(`${id} sphereRadius needs a spherical metric and a positive finite value`);
    }
  }
  if (context.ellipsoid !== undefined) {
    const {semiMajorAxis, flattening} = context.ellipsoid;
    if (
      metric !== 'ellipsoidal' ||
      !isPositiveFinite(semiMajorAxis) ||
      !Number.isFinite(flattening) ||
      flattening < 0 ||
      flattening >= 1
    ) {
      throw new Error(`${id} ellipsoid needs valid ellipsoidal metric parameters`);
    }
  }
  if (coordinateSpace === 'discrete-grid') {
    if (!context.gridFamily) {
      throw new Error(`${id} discrete-grid contexts need gridFamily`);
    }
    if (
      context.gridResolution !== undefined &&
      (!Number.isInteger(context.gridResolution) || context.gridResolution < 0)
    ) {
      throw new Error(`${id} gridResolution must be a non-negative integer`);
    }
  } else if (context.gridFamily !== undefined || context.gridResolution !== undefined) {
    throw new Error(`${id} grid metadata needs a discrete-grid coordinate space`);
  }
}

/** Checks graph-construction compatibility without changing either context. */
export function getGPUSpatialContextCompatibility(
  source: GPUSpatialContext,
  target: GPUSpatialContext
): GPUSpatialContextCompatibility {
  validateGPUSpatialContext('source spatial context', source);
  validateGPUSpatialContext('target spatial context', target);
  if (source.coordinateSpace !== target.coordinateSpace) {
    return {
      compatible: false,
      reason: `coordinate spaces differ (${source.coordinateSpace} and ${target.coordinateSpace})`
    };
  }
  if (source.metric !== 'none' && target.metric !== 'none' && source.metric !== target.metric) {
    return {compatible: false, reason: `metrics differ (${source.metric} and ${target.metric})`};
  }
  if (source.units && target.units && source.units !== target.units) {
    return {compatible: false, reason: `units differ (${source.units} and ${target.units})`};
  }
  if (!equalOptionalNumber(source.sphereRadius, target.sphereRadius)) {
    return {compatible: false, reason: 'sphere radii differ'};
  }
  if (!equalOptionalEllipsoid(source.ellipsoid, target.ellipsoid)) {
    return {compatible: false, reason: 'ellipsoids differ'};
  }
  if (
    source.gridFamily &&
    target.gridFamily &&
    source.gridFamily.toLowerCase() !== target.gridFamily.toLowerCase()
  ) {
    return {compatible: false, reason: 'grid families differ'};
  }
  if (!equalOptionalNumber(source.gridResolution, target.gridResolution)) {
    return {compatible: false, reason: 'grid resolutions differ'};
  }
  return {compatible: true};
}

/** Throws a short graph-construction error when two contexts are incompatible. */
export function assertCompatibleGPUSpatialContexts(
  id: string,
  source: GPUSpatialContext,
  target: GPUSpatialContext
): void {
  const compatibility = getGPUSpatialContextCompatibility(source, target);
  if ('reason' in compatibility) {
    throw new Error(`${id} incompatible spatial contexts: ${compatibility.reason}`);
  }
}

function isPositiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

function equalOptionalNumber(left: number | undefined, right: number | undefined): boolean {
  return left === undefined || right === undefined || left === right;
}

function equalOptionalEllipsoid(
  left: GPUSpatialEllipsoid | undefined,
  right: GPUSpatialEllipsoid | undefined
): boolean {
  return (
    !left ||
    !right ||
    (left.semiMajorAxis === right.semiMajorAxis && left.flattening === right.flattening)
  );
}
