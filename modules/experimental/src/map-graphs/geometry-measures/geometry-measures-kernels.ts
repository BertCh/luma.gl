// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createMapGraphKernelNode,
  getWGSLFloatLiteral,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import {
  GEODESIC_WGSL,
  GPU_GEODESIC_WGS84_FLATTENING,
  GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS
} from './geodesic-wgsl';
import {getVincentyWGSL, GPU_GEODESIC_DEFAULT_ITERATIONS} from './geodesic-kernels';

/** Coordinate interpretation of `GPUGeometryMeasures`. */
export type GPUGeometryCoordinateSystem = 'planar' | 'spherical' | 'wgs84';

/** How `GPUGeometryMeasures` combines the rings of one polygon feature. */
export type GPUGeometryHoleRule = 'winding' | 'first-ring-exterior';

/**
 * Words per feature or group in the internal stats column:
 * `[length, signedArea, area, centroidX, centroidY, minX, minY, maxX, maxY, vertexCount, featureCount]`
 * stored as u32 words (floats as their bit patterns), so the integer counts are never
 * reinterpreted as subnormal floats that a GPU may flush to zero.
 *
 * @internal
 */
export const GEOMETRY_STATS_STRIDE = 11;

/** WGS84-derived constants of the authalic sphere, computed in f64. @internal */
export function getAuthalicConstants(): {
  eccentricitySquared: number;
  authalicRadius: number;
  /** `2 (1 - e^2) / q_p`, the scale from `∫ (1 - e^2 s^2)^-2 ds` to `sin(authalic latitude)`. */
  sinAuthalicScale: number;
  /** `q_p` itself. */
  polarQ: number;
} {
  const flattening = GPU_GEODESIC_WGS84_FLATTENING;
  const eccentricitySquared = flattening * (2 - flattening);
  const eccentricity = Math.sqrt(eccentricitySquared);
  const polarQ =
    1 +
    ((1 - eccentricitySquared) / (2 * eccentricity)) *
      Math.log((1 + eccentricity) / (1 - eccentricity));
  return {
    eccentricitySquared,
    authalicRadius: GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS * Math.sqrt(polarQ / 2),
    sinAuthalicScale: (2 * (1 - eccentricitySquared)) / polarQ,
    polarQ
  };
}

function getMeasureConstantsSource(
  coordinateSystem: GPUGeometryCoordinateSystem,
  radius: number
): string {
  const authalic = getAuthalicConstants();
  const eccentricity = Math.sqrt(authalic.eccentricitySquared);
  const areaRadius = coordinateSystem === 'wgs84' ? authalic.authalicRadius : radius;
  return /* wgsl */ `
const RADIUS: f32 = ${getWGSLFloatLiteral(radius)};
const AREA_RADIUS: f32 = ${getWGSLFloatLiteral(areaRadius)};
const WGS84_A: f32 = ${getWGSLFloatLiteral(GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS)};
const WGS84_F: f32 = ${getWGSLFloatLiteral(GPU_GEODESIC_WGS84_FLATTENING)};
const WGS84_E2: f32 = ${getWGSLFloatLiteral(authalic.eccentricitySquared)};
const WGS84_E: f32 = ${getWGSLFloatLiteral(eccentricity)};
const SIN_AUTHALIC_SCALE: f32 = ${getWGSLFloatLiteral(authalic.sinAuthalicScale)};
const POLAR_Q: f32 = ${getWGSLFloatLiteral(authalic.polarQ)};
`;
}

/**
 * WGSL helpers of the WGS84 mode.
 *
 * - Edge lengths use the shared f32 Vincenty inverse (`getVincentyWGSL`); `getWgs84Length(a, b)`,
 *   the fallback for edges where Vincenty does not converge (near-antipodal), is Lambert's
 *   long-line formula on reduced latitudes β; `β2 - β1` comes
 *   from the midpoint derivative `dβ/dφ` for short spans and from `Δφ - (gap(φ2) - gap(φ1))` with
 *   the small closed-form gap `φ - β` otherwise, so no span subtracts two rounded β values.
 * - `getAuthalicSinDelta(sinOrigin, sinDelta)`: `sin ξ - sin ξ0` for authalic latitude ξ, as
 *   `SIN_AUTHALIC_SCALE * ∫ (1 - e² s²)^-2 ds` over `[s0, s0 + Δs]` by 8-interval composite Simpson
 *   (integrand is smooth; relative error below 1e-8 for any span), so small polygons never subtract
 *   two rounded authalic values.
 * - `getGeodeticFromAuthalic(xi)`: series inverse of the authalic latitude.
 */
const WGS84_WGSL = /* wgsl */ `
fn getWgs84Length(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let phi1 = a.y * GEODESIC_DEGREES_TO_RADIANS;
  let phi2 = b.y * GEODESIC_DEGREES_TO_RADIANS;
  let deltaPhi = (b.y - a.y) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicWrapLongitudeDelta(b.x - a.x) * GEODESIC_DEGREES_TO_RADIANS;
  let beta1 = atan2((1.0 - WGS84_F) * sin(phi1), cos(phi1));
  let beta2 = atan2((1.0 - WGS84_F) * sin(phi2), cos(phi2));
  // beta2 - beta1 without subtracting two rounded reduced latitudes: the midpoint derivative
  // dβ/dφ for short spans (error below 1e-10 rad), else deltaPhi minus the difference of the small
  // closed-form gaps phi - beta.
  var halfDifference = 0.5 * (deltaPhi - (getReducedLatitudeGap(phi2) - getReducedLatitudeGap(phi1)));
  if (abs(deltaPhi) < 0.01) {
    let sinMid = sin(0.5 * (phi1 + phi2));
    let cosMid = cos(0.5 * (phi1 + phi2));
    halfDifference = 0.5 * deltaPhi * (1.0 - WGS84_F) /
      (cosMid * cosMid + (1.0 - WGS84_F) * (1.0 - WGS84_F) * sinMid * sinMid);
  }
  let halfSum = 0.5 * (beta1 + beta2);
  let sinHalfDifference = sin(halfDifference);
  let sinHalfLambda = sin(0.5 * deltaLambda);
  let haversine = clamp(
    sinHalfDifference * sinHalfDifference + cos(beta1) * cos(beta2) * sinHalfLambda * sinHalfLambda,
    0.0,
    1.0
  );
  if (haversine <= 0.0) {
    return 0.0;
  }
  let sigma = 2.0 * atan2(sqrt(haversine), sqrt(1.0 - haversine));
  let sinSigma = sin(sigma);
  let sinP = sin(halfSum);
  let cosP = cos(halfSum);
  let cosQ = cos(halfDifference);
  let x = (sigma - sinSigma) * sinP * sinP * cosQ * cosQ / max(1.0 - haversine, 1e-12);
  let y = (sigma + sinSigma) * cosP * cosP * sinHalfDifference * sinHalfDifference / haversine;
  return WGS84_A * (sigma - 0.5 * WGS84_F * (x + y));
}

fn getAuthalicIntegrand(s: f32) -> f32 {
  let denominator = 1.0 - WGS84_E2 * s * s;
  return 1.0 / (denominator * denominator);
}

fn getAuthalicSinDelta(sinOrigin: f32, sinDelta: f32) -> f32 {
  let step = sinDelta / 8.0;
  var sum = getAuthalicIntegrand(sinOrigin) + getAuthalicIntegrand(sinOrigin + sinDelta);
  for (var interval = 1u; interval < 8u; interval++) {
    let weight = select(2.0, 4.0, (interval & 1u) == 1u);
    sum = sum + weight * getAuthalicIntegrand(sinOrigin + f32(interval) * step);
  }
  return SIN_AUTHALIC_SCALE * sum * step / 3.0;
}

fn getSinAuthalic(s: f32) -> f32 {
  let es = WGS84_E * s;
  let q = (1.0 - WGS84_E2) * (s / (1.0 - es * es) + log((1.0 + es) / (1.0 - es)) / (2.0 * WGS84_E));
  return q / POLAR_Q;
}

fn getGeodeticFromAuthalic(xi: f32) -> f32 {
  let e4 = WGS84_E2 * WGS84_E2;
  return xi + (WGS84_E2 / 3.0 + 31.0 * e4 / 180.0) * sin(2.0 * xi) +
    (17.0 * e4 / 360.0) * sin(4.0 * xi);
}
`;

/** Properties for {@link createFeatureMeasuresNode}. @internal */
export type FeatureMeasuresNodeProps = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  ringOffsets: GraphDataView<'uint32'>;
  featureRingOffsets?: GraphDataView<'uint32'>;
  featureCount: number;
  isPolygon: boolean;
  holeRule: GPUGeometryHoleRule;
  coordinateSystem: GPUGeometryCoordinateSystem;
  radius: number;
  /** `featureCount * GEOMETRY_STATS_STRIDE` words. */
  featureStats: GraphDataView<'uint32'>;
};

/**
 * Builds the per-feature measurement kernel: one invocation per feature walks its rings in row
 * order with Neumaier-compensated sums in local coordinates (relative to the feature's first
 * vertex), so results are deterministic and keep f32 precision far from the origin.
 *
 * @internal
 */
export function createFeatureMeasuresNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: FeatureMeasuresNodeProps
): GPUCommandNode<Parameters> {
  const {coordinateSystem} = props;
  const isGeographic = coordinateSystem !== 'planar';
  const bindings: MapGraphKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'}
  ];
  if (props.featureRingOffsets) {
    bindings.push({
      name: 'featureRingOffsets',
      view: props.featureRingOffsets,
      type: 'u32',
      access: 'read'
    });
  }
  bindings.push({
    name: 'featureStats',
    view: props.featureStats,
    type: 'u32',
    access: 'read_write'
  });
  const ringRange = props.featureRingOffsets
    ? `let ringBegin = min(featureRingOffsets[featureRingOffsetsOffset + index], RING_COUNT);
  let ringEnd = min(max(featureRingOffsets[featureRingOffsetsOffset + index + 1u], ringBegin), RING_COUNT);`
    : `let ringBegin = index;
  let ringEnd = index + 1u;`;
  const lengthSource = {
    planar: 'return length(b - a);',
    spherical: 'return geodesicCentralAngle(a, b) * RADIUS;',
    wgs84: `let inverse = vincentyInverse(a, b);
  if (inverse.converged) {
    return inverse.distance;
  }
  return getWgs84Length(a, b);`
  }[coordinateSystem];
  // Area coordinates: planar local x/y; geographic x = unwrapped longitude in radians,
  // y = sin(latitude) - sin(origin latitude) (authalic for WGS84). Shoelace in these coordinates
  // is the Lambert cylindrical equal-area area.
  const areaYSource = {
    planar: 'return position.y - origin.y;',
    spherical: 'return geodesicSinLatitudeDelta(position.y, origin.y);',
    wgs84:
      'return getAuthalicSinDelta(sin(origin.y * GEODESIC_DEGREES_TO_RADIANS), geodesicSinLatitudeDelta(position.y, origin.y));'
  }[coordinateSystem];
  const polygonCentroidY = {
    planar: 'origin.y + centroid.y',
    spherical:
      'asin(clamp(sin(origin.y * GEODESIC_DEGREES_TO_RADIANS) + centroid.y, -1.0, 1.0)) * GEODESIC_RADIANS_TO_DEGREES',
    wgs84:
      'getGeodeticFromAuthalic(asin(clamp(getSinAuthalic(sin(origin.y * GEODESIC_DEGREES_TO_RADIANS)) + centroid.y, -1.0, 1.0))) * GEODESIC_RADIANS_TO_DEGREES'
  }[coordinateSystem];
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'feature-measures',
    bindings,
    invocationCount: props.featureCount,
    declarations: /* wgsl */ `
const ROW_COUNT: u32 = ${props.positions.length}u;
const RING_COUNT: u32 = ${props.ringOffsets.length - 1}u;
const IS_POLYGON: bool = ${props.isPolygon};
const FIRST_RING_EXTERIOR: bool = ${props.holeRule === 'first-ring-exterior'};
const IS_GEOGRAPHIC: bool = ${isGeographic};
const STATS_STRIDE: u32 = ${GEOMETRY_STATS_STRIDE}u;
${GEODESIC_WGSL}
${getMeasureConstantsSource(coordinateSystem, props.radius)}
${
  coordinateSystem === 'wgs84'
    ? `${getVincentyWGSL(GPU_GEODESIC_DEFAULT_ITERATIONS)}
${WGS84_WGSL}`
    : ''
}

fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getEdgeLength(a: vec2<f32>, b: vec2<f32>) -> f32 {
  ${lengthSource}
}

fn getAreaY(position: vec2<f32>, origin: vec2<f32>) -> f32 {
  ${areaYSource}
}

// Neumaier compensated addition: sum.x holds the running sum, sum.y the compensation.
fn addCompensated(sum: ptr<function, vec2<f32>>, value: f32) {
  let total = (*sum).x + value;
  if (abs((*sum).x) >= abs(value)) {
    (*sum).y = (*sum).y + (((*sum).x - total) + value);
  } else {
    (*sum).y = (*sum).y + ((value - total) + (*sum).x);
  }
  (*sum).x = total;
}

fn getCompensated(sum: vec2<f32>) -> f32 {
  return sum.x + sum.y;
}

fn writeStat(feature: u32, slot: u32, value: f32) {
  featureStats[featureStatsOffset + feature * STATS_STRIDE + slot] = bitcast<u32>(value);
}
`,
    body: /* wgsl */ `
  ${ringRange}
  // Local origin: the first vertex of the first non-empty ring.
  var origin = vec2<f32>(0.0);
  var hasVertex = false;
  for (var ring = ringBegin; ring < ringEnd; ring++) {
    let rowBegin = min(ringOffsets[ringOffsetsOffset + ring], ROW_COUNT);
    let rowEnd = min(max(ringOffsets[ringOffsetsOffset + ring + 1u], rowBegin), ROW_COUNT);
    if (rowEnd > rowBegin) {
      origin = getPosition(rowBegin);
      hasVertex = true;
      break;
    }
  }
  // A runtime operand keeps the NaN out of const-expression evaluation, where it is an error.
  let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  if (!hasVertex) {
    writeStat(index, 0u, 0.0);
    writeStat(index, 1u, 0.0);
    writeStat(index, 2u, 0.0);
    for (var slot = 3u; slot < 9u; slot++) {
      writeStat(index, slot, nan);
    }
    featureStats[featureStatsOffset + index * STATS_STRIDE + 9u] = 0u;
    featureStats[featureStatsOffset + index * STATS_STRIDE + 10u] = 1u;
    return;
  }
  var lengthSum = vec2<f32>(0.0);
  var areaSum = vec2<f32>(0.0);
  var momentX = vec2<f32>(0.0);
  var momentY = vec2<f32>(0.0);
  var lineMomentX = vec2<f32>(0.0);
  var lineMomentY = vec2<f32>(0.0);
  var vertexSum = vec2<f32>(0.0);
  var boundsMin = vec2<f32>(3.4e38);
  var boundsMax = vec2<f32>(-3.4e38);
  var vertexCount = 0u;
  var isFirstRing = true;
  for (var ring = ringBegin; ring < ringEnd; ring++) {
    let rowBegin = min(ringOffsets[ringOffsetsOffset + ring], ROW_COUNT);
    let rowEnd = min(max(ringOffsets[ringOffsetsOffset + ring + 1u], rowBegin), ROW_COUNT);
    let rowCount = rowEnd - rowBegin;
    if (rowCount == 0u) {
      continue;
    }
    vertexCount += rowCount;
    var ringArea = vec2<f32>(0.0);
    var ringMomentX = vec2<f32>(0.0);
    var ringMomentY = vec2<f32>(0.0);
    let firstPosition = getPosition(rowBegin);
    // Local display coordinates (degrees for geographic, longitudes unwrapped along the ring).
    var local = firstPosition - origin;
    if (IS_GEOGRAPHIC) {
      local.x = geodesicWrapLongitudeDelta(firstPosition.x - origin.x);
    }
    let firstLocal = local;
    var areaX = local.x;
    if (IS_GEOGRAPHIC) {
      areaX = local.x * GEODESIC_DEGREES_TO_RADIANS;
    }
    var areaY = getAreaY(firstPosition, origin);
    var position = firstPosition;
    let edgeCount = select(rowCount - 1u, rowCount, IS_POLYGON);
    for (var vertex = 0u; vertex < rowCount; vertex++) {
      boundsMin = min(boundsMin, local);
      boundsMax = max(boundsMax, local);
      vertexSum += local;
      if (vertex >= edgeCount) {
        break;
      }
      let nextRow = select(rowBegin + vertex + 1u, rowBegin, vertex + 1u == rowCount);
      let nextPosition = getPosition(nextRow);
      var nextLocal = nextPosition - origin;
      if (IS_GEOGRAPHIC) {
        nextLocal = vec2<f32>(
          local.x + geodesicWrapLongitudeDelta(nextPosition.x - position.x),
          nextPosition.y - origin.y
        );
      }
      if (vertex + 1u == rowCount) {
        // Closing edge: return to the ring's first local coordinates.
        nextLocal = firstLocal;
      }
      var nextAreaX = nextLocal.x;
      if (IS_GEOGRAPHIC) {
        nextAreaX = nextLocal.x * GEODESIC_DEGREES_TO_RADIANS;
      }
      let nextAreaY = getAreaY(nextPosition, origin);
      let edgeLength = getEdgeLength(position, nextPosition);
      addCompensated(&lengthSum, edgeLength);
      addCompensated(&lineMomentX, edgeLength * 0.5 * (local.x + nextLocal.x));
      addCompensated(&lineMomentY, edgeLength * 0.5 * (local.y + nextLocal.y));
      let cross = areaX * nextAreaY - nextAreaX * areaY;
      addCompensated(&ringArea, cross);
      addCompensated(&ringMomentX, (areaX + nextAreaX) * cross);
      addCompensated(&ringMomentY, (areaY + nextAreaY) * cross);
      position = nextPosition;
      local = nextLocal;
      areaX = nextAreaX;
      areaY = nextAreaY;
    }
    var ringFactor = 1.0;
    if (FIRST_RING_EXTERIOR) {
      let ringSign = select(-1.0, 1.0, getCompensated(ringArea) >= 0.0);
      ringFactor = select(-1.0, 1.0, isFirstRing) * ringSign;
    }
    addCompensated(&areaSum, ringFactor * getCompensated(ringArea));
    addCompensated(&momentX, ringFactor * getCompensated(ringMomentX));
    addCompensated(&momentY, ringFactor * getCompensated(ringMomentY));
    isFirstRing = false;
  }
  let crossSum = getCompensated(areaSum);
  let totalLength = getCompensated(lengthSum);
  let areaScale = select(1.0, AREA_RADIUS * AREA_RADIUS, IS_GEOGRAPHIC);
  let signedArea = select(0.0, 0.5 * crossSum * areaScale, IS_POLYGON);
  var centroid = vertexSum / f32(vertexCount);
  var centroidX = origin.x + centroid.x;
  var centroidY = origin.y + centroid.y;
  if (IS_POLYGON && crossSum != 0.0) {
    centroid = vec2<f32>(getCompensated(momentX), getCompensated(momentY)) / (3.0 * crossSum);
    if (IS_GEOGRAPHIC) {
      centroidX = origin.x + centroid.x * GEODESIC_RADIANS_TO_DEGREES;
    } else {
      centroidX = origin.x + centroid.x;
    }
    centroidY = ${polygonCentroidY};
  } else if (!IS_POLYGON && totalLength > 0.0) {
    centroid = vec2<f32>(getCompensated(lineMomentX), getCompensated(lineMomentY)) / totalLength;
    centroidX = origin.x + centroid.x;
    centroidY = origin.y + centroid.y;
  }
  writeStat(index, 0u, totalLength);
  writeStat(index, 1u, signedArea);
  writeStat(index, 2u, abs(signedArea));
  writeStat(index, 3u, centroidX);
  writeStat(index, 4u, centroidY);
  writeStat(index, 5u, origin.x + boundsMin.x);
  writeStat(index, 6u, origin.y + boundsMin.y);
  writeStat(index, 7u, origin.x + boundsMax.x);
  writeStat(index, 8u, origin.y + boundsMax.y);
  featureStats[featureStatsOffset + index * STATS_STRIDE + 9u] = vertexCount;
  featureStats[featureStatsOffset + index * STATS_STRIDE + 10u] = 1u;`
  });
}

/** Optional per-row outputs written by {@link createMeasuresScatterNode}. @internal */
export type GeometryMeasureColumns = {
  lengths?: GraphDataView<'float32'>;
  areas?: GraphDataView<'float32'>;
  signedAreas?: GraphDataView<'float32'>;
  centroids?: GraphDataView<'float32x2'>;
  bounds?: GraphDataView<'float32x4'>;
  vertexCounts?: GraphDataView<'uint32'>;
  featureCounts?: GraphDataView<'uint32'>;
};

/**
 * Copies a packed stats column into the requested output columns, one invocation per row.
 *
 * @internal
 */
export function createMeasuresScatterNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rowCount: number;
    stats: GraphDataView<'uint32'>;
    columns: GeometryMeasureColumns;
  }
): GPUCommandNode<Parameters> {
  const {columns} = props;
  const bindings: MapGraphKernelBinding[] = [
    {name: 'stats', view: props.stats, type: 'u32', access: 'read'}
  ];
  const statements: string[] = [];
  const addColumn = (
    name: keyof GeometryMeasureColumns,
    type: 'f32' | 'u32',
    statement: string
  ) => {
    const view = columns[name];
    if (view) {
      bindings.push({name, view, type, access: 'read_write'});
      statements.push(statement);
    }
  };
  addColumn('lengths', 'f32', 'lengths[lengthsOffset + index] = getStat(0u);');
  addColumn('signedAreas', 'f32', 'signedAreas[signedAreasOffset + index] = getStat(1u);');
  addColumn('areas', 'f32', 'areas[areasOffset + index] = getStat(2u);');
  addColumn(
    'centroids',
    'f32',
    `centroids[centroidsOffset + 2u * index] = getStat(3u);
  centroids[centroidsOffset + 2u * index + 1u] = getStat(4u);`
  );
  addColumn(
    'bounds',
    'f32',
    `for (var corner = 0u; corner < 4u; corner++) {
    bounds[boundsOffset + 4u * index + corner] = getStat(5u + corner);
  }`
  );
  addColumn('vertexCounts', 'u32', 'vertexCounts[vertexCountsOffset + index] = getStatBits(9u);');
  addColumn(
    'featureCounts',
    'u32',
    'featureCounts[featureCountsOffset + index] = getStatBits(10u);'
  );
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'scatter',
    bindings,
    invocationCount: props.rowCount,
    declarations: `const STATS_STRIDE: u32 = ${GEOMETRY_STATS_STRIDE}u;
var<private> row: u32;

fn getStatBits(slot: u32) -> u32 {
  return stats[statsOffset + row * STATS_STRIDE + slot];
}

fn getStat(slot: u32) -> f32 {
  return bitcast<f32>(getStatBits(slot));
}`,
    body: `row = index;
  ${statements.join('\n  ')}`
  });
}

/**
 * Prepares group sort keys: `min(groupId, groupCount)` keys and feature-row values.
 *
 * @internal
 */
export function createGroupSortPrepareNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    groupIds: GraphDataView<'uint32'>;
    groupCount: number;
    sortKeys: GraphDataView<'uint32'>;
    sortValues: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'group-sort-prepare',
    bindings: [
      {name: 'groupIds', view: props.groupIds, type: 'u32', access: 'read'},
      {name: 'sortKeys', view: props.sortKeys, type: 'u32', access: 'read_write'},
      {name: 'sortValues', view: props.sortValues, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.groupIds.length,
    declarations: `const GROUP_COUNT: u32 = ${props.groupCount}u;`,
    body: `sortKeys[sortKeysOffset + index] = min(groupIds[groupIdsOffset + index], GROUP_COUNT);
  sortValues[sortValuesOffset + index] = index;`
  });
}

/**
 * Finds each group's first row in the sorted keys (lower bound), `groupCount + 1` invocations.
 *
 * @internal
 */
export function createGroupOffsetsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    sortedKeys: GraphDataView<'uint32'>;
    groupCount: number;
    groupOffsets: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'group-offsets',
    bindings: [
      {name: 'sortedKeys', view: props.sortedKeys, type: 'u32', access: 'read'},
      {name: 'groupOffsets', view: props.groupOffsets, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.groupCount + 1,
    declarations: `const KEY_COUNT: u32 = ${props.sortedKeys.length}u;`,
    body: `var low = 0u;
  var high = KEY_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (sortedKeys[sortedKeysOffset + middle] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  groupOffsets[groupOffsetsOffset + index] = low;`
  });
}

const GROUP_WORKGROUP_SIZE = 128;
const GROUP_SUM_SLOTS = 9;

/**
 * Reduces feature stats into group stats with one workgroup per group: strided per-thread partials
 * over the group's features in sorted (ascending feature row) order, then a fixed binary tree, so
 * results are bitwise reproducible. The group centroid is the weighted mean of feature centroids
 * (weights are areas for polygons, lengths for lines), falling back to the unweighted mean when
 * the total weight is zero.
 *
 * @internal
 */
export function createGroupReduceNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    isPolygon: boolean;
    groupCount: number;
    sortedIndices: GraphDataView<'uint32'>;
    groupOffsets: GraphDataView<'uint32'>;
    featureStats: GraphDataView<'uint32'>;
    groupStats: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createMapGraphKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'group-reduce',
    bindings: [
      {name: 'sortedIndices', view: props.sortedIndices, type: 'u32', access: 'read'},
      {name: 'groupOffsets', view: props.groupOffsets, type: 'u32', access: 'read'},
      {name: 'featureStats', view: props.featureStats, type: 'u32', access: 'read'},
      {name: 'groupStats', view: props.groupStats, type: 'u32', access: 'read_write'}
    ],
    workgroupSize: GROUP_WORKGROUP_SIZE,
    invocationCount: props.groupCount * GROUP_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const STATS_STRIDE: u32 = ${GEOMETRY_STATS_STRIDE}u;
const SUM_SLOTS: u32 = ${GROUP_SUM_SLOTS}u;
const IS_POLYGON: bool = ${props.isPolygon};
const ROW_COUNT: u32 = ${props.sortedIndices.length}u;
// Slots: length, signedArea, area, weight, weightedX, weightedY, meanX, meanY, centroidCount.
var<workgroup> partialSums: array<array<f32, ${GROUP_SUM_SLOTS}>, ${GROUP_WORKGROUP_SIZE}>;
var<workgroup> partialMinimum: array<vec2<f32>, ${GROUP_WORKGROUP_SIZE}>;
var<workgroup> partialMaximum: array<vec2<f32>, ${GROUP_WORKGROUP_SIZE}>;
var<workgroup> partialVertices: array<u32, ${GROUP_WORKGROUP_SIZE}>;

fn getFeatureStat(word: u32) -> f32 {
  return bitcast<f32>(featureStats[word]);
}`,
    // No early return: every invocation of a workgroup must reach the barriers.
    body: `let group = index / ${GROUP_WORKGROUP_SIZE}u;
  let isInRange = index < INVOCATION_COUNT;
  var sums: array<f32, ${GROUP_SUM_SLOTS}>;
  for (var slot = 0u; slot < SUM_SLOTS; slot++) {
    sums[slot] = 0.0;
  }
  var minimum = vec2<f32>(3.4e38);
  var maximum = vec2<f32>(-3.4e38);
  var vertices = 0u;
  var begin = 0u;
  var end = 0u;
  if (isInRange) {
    begin = groupOffsets[groupOffsetsOffset + group];
    end = min(groupOffsets[groupOffsetsOffset + group + 1u], ROW_COUNT);
    for (var row = begin + localInvocationIndex; row < end; row += ${GROUP_WORKGROUP_SIZE}u) {
      let base = featureStatsOffset + sortedIndices[sortedIndicesOffset + row] * STATS_STRIDE;
      let vertexCount = featureStats[base + 9u];
      let featureLength = getFeatureStat(base);
      let featureArea = getFeatureStat(base + 2u);
      sums[0] += featureLength;
      sums[1] += getFeatureStat(base + 1u);
      sums[2] += featureArea;
      if (vertexCount > 0u) {
        let centroid = vec2<f32>(getFeatureStat(base + 3u), getFeatureStat(base + 4u));
        let weight = select(featureLength, featureArea, IS_POLYGON);
        sums[3] += weight;
        sums[4] += weight * centroid.x;
        sums[5] += weight * centroid.y;
        sums[6] += centroid.x;
        sums[7] += centroid.y;
        sums[8] += 1.0;
        minimum = min(minimum, vec2<f32>(getFeatureStat(base + 5u), getFeatureStat(base + 6u)));
        maximum = max(maximum, vec2<f32>(getFeatureStat(base + 7u), getFeatureStat(base + 8u)));
        vertices += vertexCount;
      }
    }
  }
  for (var slot = 0u; slot < SUM_SLOTS; slot++) {
    partialSums[localInvocationIndex][slot] = sums[slot];
  }
  partialMinimum[localInvocationIndex] = minimum;
  partialMaximum[localInvocationIndex] = maximum;
  partialVertices[localInvocationIndex] = vertices;
  workgroupBarrier();
  for (var stride = ${GROUP_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      let other = localInvocationIndex + stride;
      for (var slot = 0u; slot < SUM_SLOTS; slot++) {
        partialSums[localInvocationIndex][slot] += partialSums[other][slot];
      }
      partialMinimum[localInvocationIndex] = min(partialMinimum[localInvocationIndex], partialMinimum[other]);
      partialMaximum[localInvocationIndex] = max(partialMaximum[localInvocationIndex], partialMaximum[other]);
      partialVertices[localInvocationIndex] += partialVertices[other];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    let base = groupStatsOffset + group * STATS_STRIDE;
    let nan = bitcast<f32>(0x7fc00000u | (group & 0u));
    let weight = partialSums[0][3];
    let centroidCount = partialSums[0][8];
    var centroid = vec2<f32>(nan);
    if (weight > 0.0) {
      centroid = vec2<f32>(partialSums[0][4], partialSums[0][5]) / weight;
    } else if (centroidCount > 0.0) {
      centroid = vec2<f32>(partialSums[0][6], partialSums[0][7]) / centroidCount;
    }
    groupStats[base] = bitcast<u32>(partialSums[0][0]);
    groupStats[base + 1u] = bitcast<u32>(partialSums[0][1]);
    groupStats[base + 2u] = bitcast<u32>(partialSums[0][2]);
    groupStats[base + 3u] = bitcast<u32>(centroid.x);
    groupStats[base + 4u] = bitcast<u32>(centroid.y);
    let hasBounds = centroidCount > 0.0;
    groupStats[base + 5u] = bitcast<u32>(select(nan, partialMinimum[0].x, hasBounds));
    groupStats[base + 6u] = bitcast<u32>(select(nan, partialMinimum[0].y, hasBounds));
    groupStats[base + 7u] = bitcast<u32>(select(nan, partialMaximum[0].x, hasBounds));
    groupStats[base + 8u] = bitcast<u32>(select(nan, partialMaximum[0].y, hasBounds));
    groupStats[base + 9u] = partialVertices[0];
    groupStats[base + 10u] = end - begin;
  }`
  });
}
