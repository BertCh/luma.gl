// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {GPURasterBufferToTexture} from '../../gpu-raster/index';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  captureGraphCommandNodes,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  validateTerrainBuffersDistinct,
  validateTerrainGrid,
  validateTerrainSettings,
  validateTerrainTexture
} from '../terrain-analysis/terrain-analysis-utils';
import {
  GPU_TERRAIN_HORIZON_MAX_DIRECTION_COUNT,
  GPU_TERRAIN_HORIZON_MIN_DIRECTION_COUNT
} from './gpu-terrain-horizon';
import {GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES} from './gpu-solar-shadow-mask';
import {getSolarPosition} from './solar-position';
import {
  getTerrainSolarVisibilityWGSL,
  TERRAIN_ILLUMINATION_WGSL_CONSTANTS,
  validateTerrainHorizonFormat,
  validateTerrainHorizonView,
  validateTerrainIlluminationView,
  type GPUTerrainHorizonFormat
} from './terrain-illumination-utils';

/** Number of float32 values read from `GPUSolarIrradianceProps.settings`. */
export const GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH = 8;

/** Float32 values per sun table row: azimuth, altitude, duration, direct normal irradiance. */
export const GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE = 4;

/** Solar constant used by the Meinel clear-sky model in W/m^2. */
const SOLAR_CONSTANT = 1353;

/**
 * Direct normal irradiance of the Meinel clear-sky model in W/m^2: `1353 * 0.7^(AM^0.678)` with
 * the Kasten-Young (1989) air mass `AM = 1 / (cos(z) + 0.50572 * (96.07995 - z)^-1.6364)`, where
 * `z` is the zenith angle in degrees of the (refracted) sun altitude.
 *
 * @returns 0 when `altitudeDegrees <= 0`.
 */
export function getGPUSolarDirectNormalIrradiance(altitudeDegrees: number): number {
  if (!(altitudeDegrees > 0)) {
    return 0;
  }
  const zenithDegrees = 90 - altitudeDegrees;
  const airMass =
    1 /
    (Math.cos((zenithDegrees * Math.PI) / 180) + 0.50572 * (96.07995 - zenithDegrees) ** -1.6364);
  return SOLAR_CONSTANT * 0.7 ** (airMass ** 0.678);
}

/** Options for {@link getGPUSolarIrradianceSunTable}. */
export type GPUSolarIrradianceSunTableOptions = {
  /** Degrees east. */
  longitude: number;
  /** Degrees north in `[-90, 90]`. */
  latitude: number;
  /** Start of the period: Unix epoch milliseconds (UTC) or a `Date`. */
  start: number | Date;
  /** End of the period, after `start`. */
  end: number | Date;
  /** Sampling step in minutes. Defaults to 5. */
  stepMinutes?: number;
  /**
   * Direct normal irradiance in W/m^2 while the sun is up: a constant, or `'meinel'` for the
   * clear-sky model of {@link getGPUSolarDirectNormalIrradiance}. Defaults to `'meinel'`.
   */
  directNormalIrradiance?: number | 'meinel';
  /** Apply atmospheric refraction to the sun altitude. Defaults to true. */
  refraction?: boolean;
  /** Optional destination with at least `sampleCount * 4` values. */
  target?: Float32Array;
};

/**
 * Samples the sun over a period into the table read by {@link GPUSolarIrradiance}.
 *
 * Each row covers one step of `stepMinutes` and is evaluated at the midpoint of the interval it
 * covers; the last partial step is shortened and its `durationHours` is its real length. Rows are
 * `[azimuthDegrees, altitudeDegrees, durationHours, directNormalIrradiance]` (compass azimuth
 * clockwise from north, see `getSolarPosition`). The CPU float64 solar position keeps sunrise and
 * sunset timing exact; the table is then plain float32 data that one buffer write replaces.
 *
 * @returns The packed rows and their number. `values` has `sampleCount * 4` entries (the `target`
 * when it was passed).
 * @throws If a time is not finite, `end <= start`, the step is not positive, or `target` is short.
 */
export function getGPUSolarIrradianceSunTable(options: GPUSolarIrradianceSunTableOptions): {
  values: Float32Array;
  sampleCount: number;
} {
  const startMilliseconds =
    typeof options.start === 'number' ? options.start : options.start.getTime();
  const endMilliseconds = typeof options.end === 'number' ? options.end : options.end.getTime();
  if (!Number.isFinite(startMilliseconds) || !Number.isFinite(endMilliseconds)) {
    throw new Error('Solar irradiance sun table times must be finite');
  }
  if (!(endMilliseconds > startMilliseconds)) {
    throw new Error('Solar irradiance sun table end must be after start');
  }
  const stepMinutes = options.stepMinutes ?? 5;
  if (!Number.isFinite(stepMinutes) || stepMinutes <= 0) {
    throw new Error('Solar irradiance sun table stepMinutes must be positive');
  }
  const stepMilliseconds = stepMinutes * 60000;
  const sampleCount = Math.ceil((endMilliseconds - startMilliseconds) / stepMilliseconds - 1e-9);
  const stride = GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE;
  const values = options.target ?? new Float32Array(sampleCount * stride);
  if (values.length < sampleCount * stride) {
    throw new Error(`Solar irradiance sun table target must hold ${sampleCount * stride} values`);
  }
  const directNormal = options.directNormalIrradiance ?? 'meinel';
  if (directNormal !== 'meinel' && !(Number.isFinite(directNormal) && directNormal >= 0)) {
    throw new Error('Solar irradiance directNormalIrradiance must be meinel or non-negative');
  }
  for (let sample = 0; sample < sampleCount; sample++) {
    const intervalStart = startMilliseconds + sample * stepMilliseconds;
    const intervalEnd = Math.min(intervalStart + stepMilliseconds, endMilliseconds);
    const position = getSolarPosition(
      (intervalStart + intervalEnd) / 2,
      options.longitude,
      options.latitude,
      {refraction: options.refraction}
    );
    const irradiance =
      position.altitudeDegrees > 0
        ? directNormal === 'meinel'
          ? getGPUSolarDirectNormalIrradiance(position.altitudeDegrees)
          : directNormal
        : 0;
    values.set(
      [
        position.azimuthDegrees,
        position.altitudeDegrees,
        (intervalEnd - intervalStart) / 3600000,
        irradiance
      ],
      sample * stride
    );
  }
  return {values, sampleCount};
}

/** CPU-side description packed by {@link getGPUSolarIrradianceParameterValues}. */
export type GPUSolarIrradianceSettings = {
  /** Number of valid sun table rows, an integer in `[0, sampleCapacity]`. */
  sampleCount: number;
  /**
   * Angular radius of the solar disk in degrees: 0 gives hard shadows. Defaults to
   * {@link GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES}.
   */
  angularRadiusDegrees?: number;
  /**
   * Diffuse horizontal irradiance in W/m^2 added while the sun is up, scaled by the sky-view
   * factor when one is bound. Defaults to 0.
   */
  diffuseIrradiance?: number;
};

/**
 * Packs settings into the 8-float layout read by {@link GPUSolarIrradiance}:
 * `[sampleCount, angularRadiusDegrees, diffuseIrradiance, 0, 0, 0, 0, 0]`.
 *
 * @throws If `sampleCount` is not a non-negative integer, the radius or diffuse irradiance is
 * negative or not finite, or `target` is too short.
 */
export function getGPUSolarIrradianceParameterValues(
  settings: GPUSolarIrradianceSettings,
  target: Float32Array = new Float32Array(GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH) {
    throw new Error('Solar irradiance settings target must hold 8 values');
  }
  if (!Number.isSafeInteger(settings.sampleCount) || settings.sampleCount < 0) {
    throw new Error('Solar irradiance sampleCount must be a non-negative integer');
  }
  const angularRadius = settings.angularRadiusDegrees ?? GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES;
  if (!Number.isFinite(angularRadius) || angularRadius < 0) {
    throw new Error('Solar irradiance angular radius must be finite and non-negative');
  }
  const diffuse = settings.diffuseIrradiance ?? 0;
  if (!Number.isFinite(diffuse) || diffuse < 0) {
    throw new Error('Solar irradiance diffuseIrradiance must be finite and non-negative');
  }
  target.fill(0, 0, GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH);
  target.set([settings.sampleCount, angularRadius, diffuse]);
  return target;
}

/**
 * Properties for {@link GPUSolarIrradiance}.
 *
 * Topology: grid size, `directionCount`, `horizonFormat`, `sampleCapacity`, `samplesPerNode`, and
 * which inputs and outputs exist. Per-frame: `settings`, the contents of `sunTable`, and input
 * contents. The contributor works in pixel space and never reads cell sizes: all terrain geometry is
 * already in the horizon map, slope, and aspect.
 */
export type GPUSolarIrradianceProps = {
  /** Prefix for node and transient IDs. Defaults to `'solar-irradiance'`. */
  id?: string;
  /** Grid width in pixels. */
  width: number;
  /** Grid height in pixels. */
  height: number;
  /** Number of horizon sectors, an integer in `[4, 64]`, matching the `GPUTerrainHorizon`. */
  directionCount: number;
  /** Storage format of `horizon`. Defaults to `'float32'`; see `GPUSolarShadowMask`. */
  horizonFormat?: GPUTerrainHorizonFormat;
  /**
   * Pixel-major horizon angles in degrees from `GPUTerrainHorizon`: `width * height *
   * directionCount` float32 values, or for `'unorm16'` a `uint32` view of
   * `ceil(width * height * directionCount / 2)` words. A pixel is invalid when its sector 0
   * horizon is NaN (the horizon contributor writes NaN in every sector of an invalid pixel).
   */
  horizon: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /**
   * Sun rows, `sampleCapacity * 4` float32 values, see {@link getGPUSolarIrradianceSunTable}.
   * Rewrite it per frame to change location, date range, or step without recompiling.
   */
  sunTable: GraphDataView<'float32'>;
  /** Maximum number of sun table rows, an integer of at least 1 (topology). */
  sampleCapacity: number;
  /**
   * Rows processed per node (topology). Defaults to `sampleCapacity`. Smaller values split the
   * table across several nodes that accumulate into the outputs, which keeps each dispatch short
   * on large grids (GPU watchdog safety). Results are identical to the unchunked path.
   */
  samplesPerNode?: number;
  /** Per-frame settings with at least 8 float32 values, see {@link getGPUSolarIrradianceParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Optional slope in degrees, as written by `GPUTerrainDerivatives`. Used with `aspect` for `insolation`. */
  slope?: GraphDataView<'float32'>;
  /** Optional aspect in degrees clockwise from north, -1 when flat. Required with `slope`. */
  aspect?: GraphDataView<'float32'>;
  /** Optional sky-view factor scaling the diffuse term of `insolation`. */
  skyViewFactor?: GraphDataView<'float32'>;
  /** Optional hours the solar disk is visible, summed over the sun table. */
  sunHours?: GraphDataView<'float32'>;
  /** Optional energy in Wh/m^2 on the (inclined) surface summed over the sun table. */
  insolation?: GraphDataView<'float32'>;
  /** Optional per-pixel 1 where every output is valid, else 0. */
  validity?: GraphDataView<'uint32'>;
  /** Optional storage texture (`r32float` or `rgba32float`, channel 0) receiving `sunHours`. */
  sunHoursTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
  /** Optional storage texture (`r32float` or `rgba32float`, channel 0) receiving `insolation`. */
  insolationTexture?: GraphTextureView<'r32float' | 'rgba32float'>;
};

/**
 * Accumulates sun-hours and insolation over a day or date range from a precomputed horizon map.
 *
 * For every sun table row whose disk is at least partly above the astronomical horizon
 * (`altitude + radius > 0`) the kernel interpolates the horizon at the sun azimuth and takes the
 * solar-disk visibility, with code identical to `GPUSolarShadowMask`. Then
 *
 * ```
 * sunHours   = sum(visibility * duration)
 * insolation = sum(DNI * visibility * max(cosIncidence, 0) * duration)
 *            + sum(diffuse * svf * duration)           (rows with altitude > 0)
 * ```
 *
 * with `cosIncidence` from slope and aspect when they are bound (a flat aspect of -1 gives
 * `sin(altitude)`), else `sin(altitude)` for a horizontal surface; the diffuse term uses
 * `svf = 1` without a sky-view factor. Durations are hours and irradiances W/m^2, so `insolation` is in
 * Wh/m^2. Pixels with a NaN horizon, slope, aspect, or sky-view factor receive NaN and validity 0.
 *
 * One invocation per pixel loops over its chunk of table rows, so a day at 5 minute steps is 288
 * iterations with two horizon reads each; `samplesPerNode` splits long tables across nodes.
 */
export class GPUSolarIrradiance implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSolarIrradianceProps;
  /** Storage format of the horizon input. */
  readonly horizonFormat: GPUTerrainHorizonFormat;
  /** Number of sun table rows handled by each accumulation node. */
  readonly samplesPerNode: number;

  constructor(props: GPUSolarIrradianceProps) {
    this.id = props.id ?? 'solar-irradiance';
    this.props = props;
    const {id} = this;
    const pixelCount = validateTerrainGrid(id, props.width, props.height);
    const {directionCount, sampleCapacity} = props;
    if (
      !Number.isSafeInteger(directionCount) ||
      directionCount < GPU_TERRAIN_HORIZON_MIN_DIRECTION_COUNT ||
      directionCount > GPU_TERRAIN_HORIZON_MAX_DIRECTION_COUNT
    ) {
      throw new Error(`${id} directionCount must be an integer in [4, 64]`);
    }
    if (!Number.isSafeInteger(sampleCapacity) || sampleCapacity < 1) {
      throw new Error(`${id} sampleCapacity must be a positive integer`);
    }
    this.samplesPerNode = props.samplesPerNode ?? sampleCapacity;
    if (!Number.isSafeInteger(this.samplesPerNode) || this.samplesPerNode < 1) {
      throw new Error(`${id} samplesPerNode must be a positive integer`);
    }
    if (
      !props.sunHours &&
      !props.insolation &&
      !props.validity &&
      !props.sunHoursTexture &&
      !props.insolationTexture
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    if (Boolean(props.slope) !== Boolean(props.aspect)) {
      throw new Error(`${id} slope and aspect must be provided together`);
    }
    validateTerrainHorizonFormat(id, props.horizonFormat ?? 'float32');
    this.horizonFormat = props.horizonFormat ?? 'float32';
    validateTerrainHorizonView(
      id,
      'horizon',
      props.horizon,
      this.horizonFormat,
      pixelCount,
      directionCount
    );
    validateTerrainIlluminationView(
      id,
      'sunTable',
      props.sunTable,
      'float32',
      sampleCapacity * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE
    );
    for (const [name, view] of [
      ['slope', props.slope],
      ['aspect', props.aspect],
      ['skyViewFactor', props.skyViewFactor],
      ['sunHours', props.sunHours],
      ['insolation', props.insolation]
    ] as const) {
      validateTerrainIlluminationView(id, name, view, 'float32', pixelCount);
    }
    validateTerrainIlluminationView(id, 'validity', props.validity, 'uint32', pixelCount);
    validateTerrainSettings(id, props.settings, GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH);
    for (const [name, texture] of [
      ['sunHoursTexture', props.sunHoursTexture],
      ['insolationTexture', props.insolationTexture]
    ] as const) {
      validateTerrainTexture(
        id,
        name,
        texture,
        ['r32float', 'rgba32float'],
        props.width,
        props.height
      );
    }
    validateTerrainBuffersDistinct(
      id,
      [props.sunHours, props.insolation, props.validity],
      [
        props.horizon,
        props.sunTable,
        props.settings,
        props.slope,
        props.aspect,
        props.skyViewFactor
      ]
    );
  }

  /** Returns one accumulation node per sun table chunk, an optional validity node, and texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height, directionCount, sampleCapacity} = props;
    const pixelCount = width * height;
    validateGraphViewsBelongToGraph(id, graph, [
      props.horizon,
      props.sunTable,
      props.settings,
      props.slope,
      props.aspect,
      props.skyViewFactor,
      props.sunHours,
      props.insolation,
      props.validity
    ]);
    for (const texture of [props.sunHoursTexture, props.insolationTexture]) {
      if (texture && texture.texture.graph !== graph) {
        throw new Error(`${id} views must belong to the target graph`);
      }
    }
    const hoursTarget =
      props.sunHours ??
      (props.sunHoursTexture
        ? createTransientView(graph, `${id}-sun-hours`, 'float32', pixelCount)
        : undefined);
    const insolationTarget =
      props.insolation ??
      (props.insolationTexture
        ? createTransientView(graph, `${id}-insolation`, 'float32', pixelCount)
        : undefined);
    const horizonType = this.horizonFormat === 'unorm16' ? 'u32' : 'f32';
    const inputBindings = (): WGSLKernelBinding[] => {
      const bindings: WGSLKernelBinding[] = [
        {name: 'horizon', view: props.horizon, type: horizonType, access: 'read'}
      ];
      if (props.slope && props.aspect) {
        bindings.push(
          {name: 'slopeValues', view: props.slope, type: 'f32', access: 'read'},
          {name: 'aspectValues', view: props.aspect, type: 'f32', access: 'read'}
        );
      }
      if (props.skyViewFactor) {
        bindings.push({name: 'skyView', view: props.skyViewFactor, type: 'f32', access: 'read'});
      }
      return bindings;
    };
    const hasGeometry = Boolean(props.slope && props.aspect);
    // Sector 0 of an invalid pixel is NaN, as are slope, aspect, and svf.
    const validityWGSL = `var isValid = isFiniteValue(readHorizonAngle(index * DIRECTION_COUNT));
  ${
    hasGeometry
      ? `isValid = isValid && isFiniteValue(slopeValues[slopeValuesOffset + index]) &&
    isFiniteValue(aspectValues[aspectValuesOffset + index]);`
      : ''
  }
  ${props.skyViewFactor ? 'isValid = isValid && isFiniteValue(skyView[skyViewOffset + index]);' : ''}`;
    const visibilityDeclarations = `${TERRAIN_ILLUMINATION_WGSL_CONSTANTS}
${getTerrainSolarVisibilityWGSL(this.horizonFormat, 'horizon', directionCount)}`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (hoursTarget || insolationTarget) {
      for (
        let chunkStart = 0, chunk = 0;
        chunkStart < sampleCapacity;
        chunkStart += this.samplesPerNode, chunk++
      ) {
        const chunkEnd = Math.min(chunkStart + this.samplesPerNode, sampleCapacity);
        const bindings: WGSLKernelBinding[] = [
          ...inputBindings(),
          {name: 'sunTable', view: props.sunTable, type: 'f32', access: 'read'},
          {name: 'settings', view: props.settings, type: 'f32', access: 'read'}
        ];
        if (hoursTarget) {
          bindings.push({
            name: 'hoursValues',
            view: hoursTarget,
            type: 'f32',
            access: 'read_write'
          });
        }
        if (insolationTarget) {
          bindings.push({
            name: 'insolationValues',
            view: insolationTarget,
            type: 'f32',
            access: 'read_write'
          });
        }
        const first = chunk === 0;
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-sun-${chunk}`,
            operation: 'GPUSolarIrradiance',
            variant: `${this.horizonFormat}${insolationTarget ? '-insolation' : ''}`,
            bindings,
            invocationCount: pixelCount,
            declarations: `const CHUNK_START: u32 = ${chunkStart}u;
const CHUNK_END: u32 = ${chunkEnd}u;
const TABLE_STRIDE: u32 = ${GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE}u;
${visibilityDeclarations}`,
            body: `${validityWGSL}
  let radiusDegrees = settings[settingsOffset + 1u];
  let diffuseIrradiance = settings[settingsOffset + 2u];
  let sampleEnd = min(CHUNK_END, u32(max(settings[settingsOffset], 0.0)));
  var hours = 0.0;
  var energy = 0.0;
  if (isValid) {
    ${
      hasGeometry
        ? `let slopeRadians = slopeValues[slopeValuesOffset + index] * DEGREES_TO_RADIANS;
    let aspectDegrees = aspectValues[aspectValuesOffset + index];
    let sinSlope = sin(slopeRadians);
    let cosSlope = cos(slopeRadians);`
        : ''
    }
    ${props.skyViewFactor ? 'let skyViewFactor = skyView[skyViewOffset + index];' : 'let skyViewFactor = 1.0;'}
    for (var sampleIndex = CHUNK_START; sampleIndex < sampleEnd; sampleIndex++) {
      let row = sunTableOffset + sampleIndex * TABLE_STRIDE;
      let azimuthDegrees = sunTable[row];
      let altitudeDegrees = sunTable[row + 1u];
      let duration = sunTable[row + 2u];
      if (altitudeDegrees + radiusDegrees <= 0.0) {
        continue;
      }
      let visibility = getSolarDiskVisibility(
        altitudeDegrees, radiusDegrees, getHorizonAtAzimuth(index, azimuthDegrees));
      hours += visibility * duration;
      ${
        insolationTarget
          ? `let altitude = altitudeDegrees * DEGREES_TO_RADIANS;
      var cosIncidence = sin(altitude);
      ${
        hasGeometry
          ? `if (aspectDegrees >= 0.0) {
        cosIncidence = cosSlope * sin(altitude) +
          sinSlope * cos(altitude) * cos(azimuthDegrees * DEGREES_TO_RADIANS - aspectDegrees * DEGREES_TO_RADIANS);
      }`
          : ''
      }
      energy += sunTable[row + 3u] * visibility * max(cosIncidence, 0.0) * duration;
      if (altitudeDegrees > 0.0) {
        energy += diffuseIrradiance * skyViewFactor * duration;
      }`
          : ''
      }
    }
  }
  ${
    hoursTarget
      ? `hoursValues[hoursValuesOffset + index] = select(getNaN(index), ${first ? 'hours' : 'hoursValues[hoursValuesOffset + index] + hours'}, isValid);`
      : ''
  }
  ${
    insolationTarget
      ? `insolationValues[insolationValuesOffset + index] = select(getNaN(index), ${first ? 'energy' : 'insolationValues[insolationValuesOffset + index] + energy'}, isValid);`
      : ''
  }`
          })
        );
      }
    }
    if (props.validity) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-validity`,
          operation: 'GPUSolarIrradiance',
          variant: 'validity',
          bindings: [
            ...inputBindings(),
            {name: 'validityValues', view: props.validity, type: 'u32', access: 'read_write'}
          ],
          invocationCount: pixelCount,
          declarations: visibilityDeclarations,
          body: `${validityWGSL}
  validityValues[validityValuesOffset + index] = select(0u, 1u, isValid);`
        })
      );
    }
    for (const [name, target, texture] of [
      ['sun-hours', hoursTarget, props.sunHoursTexture],
      ['insolation', insolationTarget, props.insolationTexture]
    ] as const) {
      if (texture && target) {
        nodes.push(
          ...captureGraphCommandNodes(graph, () =>
            new GPURasterBufferToTexture({
              id: `${id}-${name}-texture`,
              input: {
                id: `${id}-${name}-band`,
                format: 'float32',
                storage: {kind: 'buffer', values: target}
              },
              output: texture,
              channel: 0
            }).addToGraph(graph)
          )
        );
      }
    }
    return nodes;
  }
}
