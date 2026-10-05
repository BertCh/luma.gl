// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {validateGraphViewsBelongToGraph} from '../map-graph-utils';
import {
  validateTerrainBuffersDistinct,
  validateTerrainSettings
} from '../terrain-analysis/terrain-analysis-utils';
import {getSolarTimeParameter} from './solar-position';
import {
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainIlluminationView
} from './terrain-illumination-utils';

/** Number of float32 values read from `GPUSolarPositionProps.settings`. */
export const GPU_SOLAR_POSITION_PARAMETER_LENGTH = 4;

/** Geometric sun-center altitude at sunrise and sunset (refraction plus solar radius), degrees. */
export const GPU_SOLAR_SUNRISE_ALTITUDE_DEGREES = -0.833;

/** CPU-side description packed by {@link getGPUSolarPositionParameterValues}. */
export type GPUSolarPositionSettings = {
  /** Unix epoch milliseconds (UTC) or a `Date`. */
  timestamp: number | Date;
  /**
   * Geometric altitude above which `daylight` is 1. Defaults to
   * {@link GPU_SOLAR_SUNRISE_ALTITUDE_DEGREES}; use -6, -12, or -18 for civil, nautical, or
   * astronomical twilight.
   */
  daylightAltitudeDegrees?: number;
};

/**
 * Packs settings into the 4-float layout read by {@link GPUSolarPosition}:
 * `[dayNumber, dayFraction, daylightAltitudeDegrees, 0]` (see `getSolarTimeParameter`).
 */
export function getGPUSolarPositionParameterValues(
  settings: GPUSolarPositionSettings,
  target: Float32Array = new Float32Array(GPU_SOLAR_POSITION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SOLAR_POSITION_PARAMETER_LENGTH) {
    throw new Error('Solar position settings target must hold 4 values');
  }
  const {dayNumber, dayFraction} = getSolarTimeParameter(settings.timestamp);
  target.set([
    dayNumber,
    dayFraction,
    settings.daylightAltitudeDegrees ?? GPU_SOLAR_SUNRISE_ALTITUDE_DEGREES,
    0
  ]);
  return target;
}

/**
 * Properties for {@link GPUSolarPosition}.
 *
 * Topology: row count, `refraction`, and which outputs exist. Per-frame: `settings` (time and
 * daylight threshold) and position contents.
 */
export type GPUSolarPositionProps = {
  /** Prefix for node IDs. Defaults to `'solar-position'`. */
  id?: string;
  /** Packed `[longitude, latitude]` degrees per row. */
  positions: GraphDataView<'float32x2'>;
  /** Per-frame settings with at least 4 float32 values, see {@link getGPUSolarPositionParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Optional sun azimuth per row, degrees clockwise from north in `[0, 360)`. */
  azimuth?: GraphDataView<'float32'>;
  /** Optional sun altitude per row in degrees (refracted when `refraction` is true). */
  altitude?: GraphDataView<'float32'>;
  /** Optional 1 when the geometric altitude exceeds the settings threshold, else 0. */
  daylight?: GraphDataView<'uint32'>;
  /** Apply the NOAA refraction approximation to `altitude`. Defaults to true. */
  refraction?: boolean;
};

/**
 * Computes the sun azimuth and altitude for every row of a longitude/latitude column at one
 * per-frame instant, with the NOAA solar calculator formulas evaluated in float32.
 *
 * Use it for per-feature daylight, sun-angle styling, or day/night terminators on point layers;
 * a time slider writes two floats and never recompiles. Results match `getSolarPosition` within
 * about 0.02 degree for dates within a few decades of 2000. Rows with non-finite coordinates or
 * `|latitude| > 90` receive NaN and daylight 0.
 */
export class GPUSolarPosition implements GPUMapGraphRecipe {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'solar-position';
  /** Validated properties. */
  readonly props: GPUSolarPositionProps;

  constructor(props: GPUSolarPositionProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rowCount = props.positions.length;
    if (!props.azimuth && !props.altitude && !props.daylight) {
      throw new Error(`${id} requires at least one output`);
    }
    validateTerrainIlluminationView(id, 'azimuth', props.azimuth, 'float32', rowCount);
    validateTerrainIlluminationView(id, 'altitude', props.altitude, 'float32', rowCount);
    validateTerrainIlluminationView(id, 'daylight', props.daylight, 'uint32', rowCount);
    validateTerrainSettings(id, props.settings, GPU_SOLAR_POSITION_PARAMETER_LENGTH);
    validateTerrainBuffersDistinct(
      id,
      [props.azimuth, props.altitude, props.daylight],
      [props.positions, props.settings]
    );
  }

  /** Returns one kernel node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.settings,
      props.azimuth,
      props.altitude,
      props.daylight
    ]);
    const bindings: MapGraphKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
    ];
    if (props.azimuth) {
      bindings.push({
        name: 'azimuthValues',
        view: props.azimuth,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.altitude) {
      bindings.push({
        name: 'altitudeValues',
        view: props.altitude,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.daylight) {
      bindings.push({
        name: 'daylightValues',
        view: props.daylight,
        type: 'u32',
        access: 'read_write'
      });
    }
    const refraction = props.refraction ?? true;
    return [
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-sun`,
        operation: 'GPUSolarPosition',
        variant: refraction ? 'refracted' : 'geometric',
        bindings,
        invocationCount: props.positions.length,
        declarations: `${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
fn wrapDegrees(value: f32, divisor: f32) -> f32 { return value - divisor * floor(value / divisor); }
fn getRefractionDegrees(elevation: f32) -> f32 {
  if (elevation > 85.0) { return 0.0; }
  let tangent = tan(elevation * DEGREES_TO_RADIANS);
  var arcseconds = 0.0;
  if (elevation > 5.0) {
    arcseconds = 58.1 / tangent - 0.07 / (tangent * tangent * tangent) +
      0.000086 / (tangent * tangent * tangent * tangent * tangent);
  } else if (elevation > -0.575) {
    arcseconds = 1735.0 + elevation * (-518.2 + elevation * (103.4 + elevation * (-12.79 + elevation * 0.711)));
  } else {
    arcseconds = -20.772 / tangent;
  }
  return arcseconds / 3600.0;
}`,
        body: `let longitude = positions[positionsOffset + index * 2u];
  let latitude = positions[positionsOffset + index * 2u + 1u];
  let dayNumber = settings[settingsOffset];
  let dayFraction = settings[settingsOffset + 1u];
  let isValid = isFiniteValue(longitude) && isFiniteValue(latitude) && abs(latitude) <= 90.0;
  // Angles that grow by ~1 degree per day are reduced from the whole-day part first so float32
  // keeps sub-arcsecond resolution: 0.98564736 deg/day is 36000.76983 deg per Julian century.
  let julianCentury = (dayNumber + dayFraction) / 36525.0;
  let meanLongitude = wrapDegrees(
    wrapDegrees(280.46646 + 0.98564736 * dayNumber, 360.0) + 0.98564736 * dayFraction +
      0.0003032 * julianCentury * julianCentury,
    360.0
  );
  let meanAnomalyDegrees = wrapDegrees(
    wrapDegrees(357.52911 + 0.98560028 * dayNumber, 360.0) + 0.98560028 * dayFraction -
      0.0001537 * julianCentury * julianCentury,
    360.0
  );
  let anomaly = meanAnomalyDegrees * DEGREES_TO_RADIANS;
  let eccentricity = 0.016708634 - julianCentury * (0.000042037 + 0.0000001267 * julianCentury);
  let equationOfCenter = sin(anomaly) * (1.914602 - julianCentury * (0.004817 + 0.000014 * julianCentury)) +
    sin(2.0 * anomaly) * (0.019993 - 0.000101 * julianCentury) + sin(3.0 * anomaly) * 0.000289;
  let omega = wrapDegrees(125.04 - 1934.136 * julianCentury, 360.0) * DEGREES_TO_RADIANS;
  let apparentLongitude =
    (meanLongitude + equationOfCenter - 0.00569 - 0.00478 * sin(omega)) * DEGREES_TO_RADIANS;
  let meanObliquity = 23.0 + (26.0 + (21.448 - julianCentury * (46.815 + julianCentury *
    (0.00059 - julianCentury * 0.001813))) / 60.0) / 60.0;
  let obliquity = (meanObliquity + 0.00256 * cos(omega)) * DEGREES_TO_RADIANS;
  let declination = asin(sin(obliquity) * sin(apparentLongitude));
  let y = tan(obliquity * 0.5) * tan(obliquity * 0.5);
  let longitudeRadians = meanLongitude * DEGREES_TO_RADIANS;
  let equationOfTime = 4.0 * RADIANS_TO_DEGREES * (y * sin(2.0 * longitudeRadians) -
    2.0 * eccentricity * sin(anomaly) +
    4.0 * eccentricity * y * sin(anomaly) * cos(2.0 * longitudeRadians) -
    0.5 * y * y * sin(4.0 * longitudeRadians) -
    1.25 * eccentricity * eccentricity * sin(2.0 * anomaly));
  let minutesUTC = fract(dayFraction + 0.5) * 1440.0;
  let trueSolarTime = wrapDegrees(minutesUTC + equationOfTime + 4.0 * longitude, 1440.0);
  let hourAngleDegrees = trueSolarTime / 4.0 - 180.0;
  let hourAngle = hourAngleDegrees * DEGREES_TO_RADIANS;
  let latitudeRadians = latitude * DEGREES_TO_RADIANS;
  let cosZenith = clamp(sin(latitudeRadians) * sin(declination) +
    cos(latitudeRadians) * cos(declination) * cos(hourAngle), -1.0, 1.0);
  let geometricAltitude = 90.0 - acos(cosZenith) * RADIANS_TO_DEGREES;
  let altitude = geometricAltitude${refraction ? ' + getRefractionDegrees(geometricAltitude)' : ''};
  let azimuth = wrapDegrees(atan2(sin(hourAngle),
    cos(hourAngle) * sin(latitudeRadians) - tan(declination) * cos(latitudeRadians)) *
    RADIANS_TO_DEGREES + 180.0, 360.0);
  let invalidValue = getNaN(index);
  ${props.azimuth ? 'azimuthValues[azimuthValuesOffset + index] = select(invalidValue, azimuth, isValid);' : ''}
  ${props.altitude ? 'altitudeValues[altitudeValuesOffset + index] = select(invalidValue, altitude, isValid);' : ''}
  ${
    props.daylight
      ? 'daylightValues[daylightValuesOffset + index] = select(0u, 1u, isValid && geometricAltitude > settings[settingsOffset + 2u]);'
      : ''
  }`
      })
    ];
  }
}
