// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {log} from '@luma.gl/core';
import {
  GPU_EARTH_AUTHALIC_RADIUS_METERS,
  GPU_EARTH_MEAN_RADIUS_METERS,
  GPU_WGS84_FLATTENING,
  GPU_WGS84_SEMI_MAJOR_AXIS_METERS
} from './earth';

/** Logical coordinate spaces consumed by spatial-analysis operations. */
export type GPUSpatialCoordinateSpace =
  | 'planar'
  | 'longitude-latitude'
  | 'discrete-grid'
  | 'topology-only';

/** Metric applied to coordinates, kept separate from their coordinate space. */
export type GPUSpatialMetric = 'none' | 'native' | 'great-circle' | 'rhumb' | 'ellipsoidal';

/** How polygon and line edges between longitude/latitude vertices are interpreted. */
export type GPUSpatialEdgeModel = 'linear' | 'great-circle';

/**
 * Units of metric results. `'native'` is meaningful only for a native planar metric; a planar
 * native context may also declare `'meters'` when its native unit is the meter (local frames do).
 */
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
  /**
   * Longitude/latitude only. Omitted means 'linear' (straight in degrees, RFC 7946).
   * 'great-circle' means minor great-circle arcs on the context sphere.
   */
  edgeModel?: GPUSpatialEdgeModel;
  /**
   * Geodetic datum of longitude/latitude positions or of a planar frame's source. Recommended
   * EPSG geographic codes: 'EPSG:4326' (WGS84), 'EPSG:4269' (NAD83), 'EPSG:4267' (NAD27),
   * 'EPSG:4258' (ETRS89). Free ids such as 'GCJ-02' and 'BD-09' are allowed.
   * Absent means undeclared, never defaulted. The library checks declarations where two contexts
   * meet and never converts; two-input operators check their second side through an optional
   * `<side>SpatialContext` prop (see `resolveGPUSecondarySpatialContext`).
   */
  datum?: string;
  /** Grid family when `coordinateSpace` is `discrete-grid`, for example `h3` or `quadbin`. */
  gridFamily?: string;
  /** Fixed grid resolution when required by an operation. */
  gridResolution?: number;
  /**
   * Identity of the planar frame positions were projected into, for example by
   * `createGPULocalFrame`. Contexts whose frames have different ids are incompatible, so data
   * projected around different origins is never mixed. Planar coordinate spaces only.
   */
  frame?: GPUSpatialFrame;
};

/** Kind of projected planar frame. */
export type GPUSpatialFrameKind =
  | 'local-azimuthal-equidistant'
  | 'local-lambert-equal-area'
  | 'crs';

/** Identity of a projected planar frame; compared by `id`. */
export type GPUSpatialFrame = {
  /** Stable identity; equal ids mean interchangeable coordinates. */
  id: string;
  kind: GPUSpatialFrameKind;
  /** Frame origin `[longitude, latitude]` in degrees for local frames. */
  origin?: [number, number];
  /** CRS identifier (for example `EPSG:32633`) when `kind` is `'crs'`. */
  crs?: string;
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
  if (context.edgeModel !== undefined) {
    if (context.edgeModel !== 'linear' && context.edgeModel !== 'great-circle') {
      throw new Error(`${id} edgeModel must be linear or great-circle`);
    }
    if (coordinateSpace !== 'longitude-latitude') {
      throw new Error(`${id} edgeModel needs a longitude-latitude coordinate space`);
    }
    if (metric === 'ellipsoidal') {
      throw new Error(`${id} ellipsoidal contexts take ellipsoidalEdgeModel, not edgeModel`);
    }
  }
  if (
    context.datum !== undefined &&
    (typeof context.datum !== 'string' || context.datum.trim() === '')
  ) {
    throw new Error(`${id} datum must be a non-empty string`);
  }
  if (
    metric === 'native' &&
    context.units &&
    context.units !== 'native' &&
    context.units !== 'meters'
  ) {
    throw new Error(`${id} native metrics must use native or meters units`);
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
  if (context.frame !== undefined && (coordinateSpace !== 'planar' || !context.frame.id)) {
    throw new Error(`${id} frame needs a planar coordinate space and an id`);
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

/** Quantity an operation measures, which decides the planar local frame kind it accepts. */
export type GPUSpatialMeasure = 'distance' | 'area';

/** Coordinate spaces and metrics an operation accepts, declared next to its implementation. */
export type GPUSpatialContextSupport = {
  coordinateSpaces: readonly GPUSpatialCoordinateSpace[];
  /** Accepted metrics. Omitted accepts every metric valid for an accepted coordinate space. */
  metrics?: readonly GPUSpatialMetric[];
  /** Accepted edge models. Omitted accepts any (point operators). */
  edgeModels?: readonly GPUSpatialEdgeModel[];
  /**
   * Accepted metrics per coordinate space, narrowing `metrics`. A space without an entry accepts
   * what `metrics` accepts. Failure reads `<id> <space> positions need the <list> metric`.
   */
  metricsByCoordinateSpace?: Partial<
    Record<GPUSpatialCoordinateSpace, readonly GPUSpatialMetric[]>
  >;
  /** Accepted result units. Omitted accepts any unit valid for the metric. */
  units?: readonly GPUSpatialUnit[];
  /**
   * What the operation measures. `'distance'` rejects contexts whose frame is
   * `'local-lambert-equal-area'`; `'area'` (densities included) rejects
   * `'local-azimuthal-equidistant'`. `'crs'` frames and contexts without a frame always pass.
   * Either value also requires the `'native'` metric for planar positions (planar `'none'` declares
   * that nothing is measured). Omit for operations that measure both or neither.
   */
  measure?: GPUSpatialMeasure;
};

/** The context assumed when an operation is given none: native planar coordinates. */
export const GPU_PLANAR_SPATIAL_CONTEXT: GPUSpatialContext = Object.freeze({
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'native'
});

/** Support declaration of operations that only do planar Euclidean math. */
export const GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar'] as const),
  metrics: Object.freeze(['none', 'native'] as const)
});

/** Support value for consumers that interpret polygon edges as straight in degrees. */
export const GPU_LINEAR_EDGE_MODELS = Object.freeze(['linear'] as const);

/** Returns the declared edge model, defaulting to linear (always linear for valid planar contexts). */
export function getGPUEdgeModel(context: GPUSpatialContext): GPUSpatialEdgeModel {
  return context.edgeModel ?? 'linear';
}

/**
 * Trims and uppercases a datum key, mapping WGS84, WGS 84, EPSG:4326 and EPSG:6326 to EPSG:4326.
 * No other aliases are applied: a loud mismatch beats a silent pass.
 */
export function getGPUDatumKey(datum: string): string {
  const key = datum.trim().toUpperCase();
  return ['WGS84', 'WGS 84', 'EPSG:4326', 'EPSG:6326'].includes(key) ? 'EPSG:4326' : key;
}

/**
 * Validates an operation's spatial context against what it supports and returns the context to use.
 * An omitted context means native planar coordinates. Unsupported contexts throw at construction:
 * spatial analysis never treats longitude/latitude degrees as planar coordinates.
 */
export function resolveGPUSpatialContext(
  id: string,
  context: GPUSpatialContext | undefined,
  support: GPUSpatialContextSupport = GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT
): GPUSpatialContext {
  const resolved = context ?? GPU_PLANAR_SPATIAL_CONTEXT;
  validateGPUSpatialContext(id, resolved);
  if (!support.coordinateSpaces.includes(resolved.coordinateSpace)) {
    const hint =
      resolved.coordinateSpace === 'longitude-latitude'
        ? '; project positions to a planar frame first'
        : '';
    throw new Error(`${id} does not support ${resolved.coordinateSpace} coordinates${hint}`);
  }
  if (support.metrics && !support.metrics.includes(resolved.metric)) {
    throw new Error(`${id} does not support the ${resolved.metric} metric`);
  }
  const spaceMetrics = support.metricsByCoordinateSpace?.[resolved.coordinateSpace];
  if (spaceMetrics && !spaceMetrics.includes(resolved.metric)) {
    throw new Error(
      `${id} ${resolved.coordinateSpace} positions need the ${spaceMetrics.join(' or ')} metric`
    );
  }
  if (support.units && resolved.units !== undefined && !support.units.includes(resolved.units)) {
    throw new Error(
      `${id} units ${resolved.units} are not supported; results are in ${support.units.join(' or ')}`
    );
  }
  if (support.measure && resolved.coordinateSpace === 'planar' && resolved.metric === 'none') {
    throw new Error(`${id} planar positions need the native metric`);
  }
  const frameKind = resolved.frame?.kind;
  if (support.measure === 'distance' && frameKind === 'local-lambert-equal-area') {
    throw new Error(
      `${id} measures distance; project into an azimuthal-equidistant frame, not lambert-equal-area`
    );
  }
  if (support.measure === 'area' && frameKind === 'local-azimuthal-equidistant') {
    throw new Error(
      `${id} measures area; project into a lambert-equal-area frame, not azimuthal-equidistant`
    );
  }
  if (
    support.edgeModels &&
    resolved.coordinateSpace === 'longitude-latitude' &&
    !support.edgeModels.includes(getGPUEdgeModel(resolved))
  ) {
    throw new Error(
      `${id} needs ${support.edgeModels.join(' or ')} edges; prepare polygons with GPULonLatPolygonPreparation`
    );
  }
  return resolved;
}

/**
 * How an area output is computed.
 * - `planar`: shoelace area in squared position units.
 * - `sphere-equal-area`: Lambert cylindrical equal-area (Chamberlain-Duquette) area on a sphere of
 *   the context radius.
 * - `wgs84-authalic`: equal-area mapping of the WGS84 ellipsoid onto its authalic sphere, with
 *   straight edges in the mapped coordinates.
 * - `sphere-polygon`: exact area of a spherical polygon with great-circle edges, on the WGS84
 *   authalic sphere (H3 cell areas).
 * - `geodesic`: like `wgs84-authalic`, but long edges follow ellipsoidal geodesics.
 */
export type GPUAreaModel =
  | 'planar'
  | 'sphere-equal-area'
  | 'sphere-polygon'
  | 'wgs84-authalic'
  | 'geodesic';

/** How a length output is computed. */
export type GPULengthModel = 'planar' | 'great-circle' | 'ellipsoidal-geodesic';

/** Units of an area output. */
export type GPUAreaUnit = 'native-squared' | 'square-meters' | 'square-kilometers';

/**
 * Machine-readable description of what an operation's length and area outputs mean, derived from
 * the resolved spatial context and options, so a legend or export never has to guess.
 */
export type GPUMeasureDescription = {
  lengthModel: GPULengthModel;
  areaModel: GPUAreaModel;
  lengthUnits: GPUSpatialUnit;
  areaUnits: GPUAreaUnit;
};

/** Returns the area unit that matches a length unit. */
export function getGPUAreaUnit(lengthUnits: GPUSpatialUnit): GPUAreaUnit {
  if (lengthUnits === 'meters') {
    return 'square-meters';
  }
  return lengthUnits === 'kilometers' ? 'square-kilometers' : 'native-squared';
}

/** Longitude/latitude positions with no metric: what coordinate-only operations (indexing, cover) accept. */
export const GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT: GPUSpatialContext = Object.freeze({
  coordinateSpace: 'longitude-latitude',
  metric: 'none'
});

/** Support declaration of operations that read longitude/latitude positions but measure nothing. */
export const GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport =
  Object.freeze({
    coordinateSpaces: Object.freeze(['longitude-latitude'] as const),
    metrics: Object.freeze(['none'] as const)
  });

/** Longitude/latitude context with great-circle metric in meters: what spherical operations default to. */
export const GPU_GREAT_CIRCLE_SPATIAL_CONTEXT: GPUSpatialContext = Object.freeze({
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters'
});

/**
 * Support declaration of operations that accept native planar coordinates, or longitude/latitude
 * with an explicit `'great-circle'` metric (degrees are never treated as planar units). Result
 * units are the context's own, so `units: 'kilometers'` is honored through {@link getGPUSphereRadius}.
 */
export const GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport =
  Object.freeze({
    coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
    metricsByCoordinateSpace: Object.freeze({
      planar: Object.freeze(['none', 'native'] as const),
      'longitude-latitude': Object.freeze(['great-circle'] as const)
    })
  });

/**
 * Like {@link GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT}, for kernels whose results are
 * in meters: `units: 'kilometers'` is rejected.
 */
export const GPU_PLANAR_OR_GREAT_CIRCLE_METERS_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport =
  Object.freeze({
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    units: Object.freeze(['native', 'meters'] as const)
  });

/**
 * Sphere radius of a spherical context in the context's length unit: `sphereRadius` when given,
 * else the mean Earth radius in meters (or kilometers when `units` is `'kilometers'`).
 */
export function getGPUSphereRadius(context: GPUSpatialContext): number {
  return (
    context.sphereRadius ??
    (context.units === 'kilometers'
      ? GPU_EARTH_MEAN_RADIUS_METERS / 1000
      : GPU_EARTH_MEAN_RADIUS_METERS)
  );
}

/**
 * Sphere radius for spherical *areas* in the context's length unit: `sphereRadius` when given,
 * else the WGS84 authalic (equal-area) radius in meters (or kilometers when `units` is
 * `'kilometers'`). Areas computed with it are in the context's length unit squared. Spherical
 * distances use {@link getGPUSphereRadius} (mean radius) instead.
 */
export function getGPUAuthalicSphereRadius(context: GPUSpatialContext): number {
  return (
    context.sphereRadius ??
    (context.units === 'kilometers'
      ? GPU_EARTH_AUTHALIC_RADIUS_METERS / 1000
      : GPU_EARTH_AUTHALIC_RADIUS_METERS)
  );
}

/** The WGS84 ellipsoid: semi-major axis in meters and flattening. */
export const GPU_WGS84_ELLIPSOID: GPUSpatialEllipsoid = Object.freeze({
  semiMajorAxis: GPU_WGS84_SEMI_MAJOR_AXIS_METERS,
  flattening: GPU_WGS84_FLATTENING
});

/**
 * Ellipsoid of an ellipsoidal context: `context.ellipsoid` when given, else WGS84. The semi-major
 * axis is in meters; ellipsoidal kernels bake both parameters into their shaders.
 */
export function getGPUEllipsoid(context: GPUSpatialContext): GPUSpatialEllipsoid {
  return context.ellipsoid ?? GPU_WGS84_ELLIPSOID;
}

/**
 * Worst-case float32 spacing of a lon/lat position as a ground length in the context's unit:
 * 2^-16 degrees (|longitude| >= 128) on getGPUSphereRadius (great-circle, rhumb) or the
 * ellipsoid semi-major axis (ellipsoidal; /1000 for kilometers). 0 for non-lon/lat or metric 'none'.
 */
export function getGPULongitudeLatitudeQuantum(context: GPUSpatialContext): number {
  if (context.coordinateSpace !== 'longitude-latitude') {
    return 0;
  }
  let radius: number;
  if (context.metric === 'great-circle' || context.metric === 'rhumb') {
    radius = getGPUSphereRadius(context);
  } else if (context.metric === 'ellipsoidal') {
    radius = getGPUEllipsoid(context).semiMajorAxis / (context.units === 'kilometers' ? 1000 : 1);
  } else {
    return 0;
  }
  return ((2 ** -16 * Math.PI) / 180) * radius;
}

/** Number of float32 longitude/latitude quanta below which a positive distance is short. */
export const GPU_SHORT_LONGITUDE_LATITUDE_DISTANCE_FACTOR = 10;

/** Pure: finite distance > 0 and < factor * quantum. */
export function isGPUShortLongitudeLatitudeDistance(
  context: GPUSpatialContext,
  distance: number
): boolean {
  const quantum = getGPULongitudeLatitudeQuantum(context);
  return (
    quantum > 0 &&
    Number.isFinite(distance) &&
    distance > 0 &&
    distance < GPU_SHORT_LONGITUDE_LATITUDE_DISTANCE_FACTOR * quantum
  );
}

/** log.warn once per operation (probe.gl dedups by message). */
export function warnGPUShortLongitudeLatitudeDistance(
  operation: string,
  context: GPUSpatialContext | undefined,
  distance: number | undefined
): void {
  if (
    context !== undefined &&
    distance !== undefined &&
    isGPUShortLongitudeLatitudeDistance(context, distance)
  ) {
    log.warn(`${operation} distance is too short for float32 lon/lat; use createGPULocalFrame`)();
  }
}

/**
 * Length unit of a context's metric results: `context.units` when declared, else `'native'` for
 * planar coordinates and `'meters'` for geographic metrics. Feeds {@link GPUMeasureDescription}.
 */
export function getGPULengthUnits(context: GPUSpatialContext): GPUSpatialUnit {
  return context.units ?? (context.coordinateSpace === 'planar' ? 'native' : 'meters');
}

/**
 * Sphere radius of a great-circle context in the context's length unit, or `undefined` for any
 * other metric (planar distances). The one place kernels derive their default radius.
 */
export function getGPUGreatCircleRadius(context: GPUSpatialContext): number | undefined {
  return context.metric === 'great-circle' ? getGPUSphereRadius(context) : undefined;
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
  if (
    source.datum !== undefined &&
    target.datum !== undefined &&
    getGPUDatumKey(source.datum) !== getGPUDatumKey(target.datum)
  ) {
    return {compatible: false, reason: `datums differ (${source.datum} and ${target.datum})`};
  }
  if (source.metric !== 'none' && target.metric !== 'none' && source.metric !== target.metric) {
    return {compatible: false, reason: `metrics differ (${source.metric} and ${target.metric})`};
  }
  if (
    source.coordinateSpace === 'longitude-latitude' &&
    target.coordinateSpace === 'longitude-latitude'
  ) {
    const sourceEdgeModel = getGPUEdgeModel(source);
    const targetEdgeModel = getGPUEdgeModel(target);
    if (sourceEdgeModel !== targetEdgeModel) {
      return {
        compatible: false,
        reason: `edge models differ (${sourceEdgeModel} and ${targetEdgeModel})`
      };
    }
  }
  if (
    source.units &&
    target.units &&
    source.units !== target.units &&
    !isDeclaredPlanarMeters(source, target)
  ) {
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
  if (source.frame && target.frame && source.frame.id !== target.frame.id) {
    return {
      compatible: false,
      reason: `planar frames differ (${source.frame.id} and ${target.frame.id})`
    };
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

/**
 * Resolves the optional context of an operation's second input (its polygon, feature or raster
 * side), named `<sideName>SpatialContext` after the operation's prop for that input. Omitted returns
 * `primary`, so one context still describes both inputs. When given, it is validated, checked
 * against `edgeModels` when it is longitude/latitude, then checked for compatibility with `primary`
 * (datum, coordinate space, metric, units, frame id). Messages start with
 * `${id} ${sideName}SpatialContext`.
 */
export function resolveGPUSecondarySpatialContext(
  id: string,
  sideName: string,
  context: GPUSpatialContext | undefined,
  primary: GPUSpatialContext,
  edgeModels?: readonly GPUSpatialEdgeModel[]
): GPUSpatialContext {
  if (context === undefined) {
    return primary;
  }
  const label = `${id} ${sideName}SpatialContext`;
  validateGPUSpatialContext(label, context);
  if (
    edgeModels &&
    context.coordinateSpace === 'longitude-latitude' &&
    !edgeModels.includes(getGPUEdgeModel(context))
  ) {
    throw new Error(
      `${label} needs ${edgeModels.join(' or ')} edges; prepare polygons with GPULonLatPolygonPreparation`
    );
  }
  assertCompatibleGPUSpatialContexts(label, context, primary);
  return context;
}

/** Planar `'native'` units are undeclared, so they match a planar frame declared in meters. */
function isDeclaredPlanarMeters(source: GPUSpatialContext, target: GPUSpatialContext): boolean {
  const units = [source.units, target.units];
  return (
    source.coordinateSpace === 'planar' && units.includes('native') && units.includes('meters')
  );
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
