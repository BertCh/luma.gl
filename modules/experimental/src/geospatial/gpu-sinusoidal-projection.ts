// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuSpatial.

import {type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUFloat32Positions, GPUFloat64Positions} from './types';
import {
  GEOSPATIAL_WORKGROUP_SIZE,
  POSITION_FORMATS,
  RAW_POINT_WGSL,
  addGeospatialPass,
  assertGraphOwnership,
  getFloat32Literal,
  getGeospatialDispatchLayout,
  getGeospatialInvocationIndexSource,
  getPositionReadSource,
  getRawBinary64Literal,
  getRowChunks,
  isGraphVectorView,
  validateMatchingRows,
  validateRowView,
  validateSeparateBuffers
} from './geospatial-utils';

const CUSPATIAL_EARTH_CIRCUMFERENCE_KILOMETRES = 40_000;
const CUSPATIAL_KILOMETRES_PER_DEGREE = CUSPATIAL_EARTH_CIRCUMFERENCE_KILOMETRES / 360;

/** Properties for a cuSpatial-compatible sinusoidal projection. */
export type GPUSinusoidalProjectionProps = {
  /** Prefix for generated graph-node IDs. */
  id?: string;
  /** Longitude/latitude rows in degrees. */
  positions: GPUFloat32Positions | GPUFloat64Positions;
  /** Caller-owned projected XY rows in kilometres. */
  output: GPUFloat32Positions;
  /** Longitude and latitude origin in degrees. Defaults to `[0, 0]`. */
  origin?: readonly [number, number];
  /**
   * Wraps the longitude delta `originLongitude - longitude` into `[-180, 180)` degrees, so points
   * across the antimeridian from the origin project next to it instead of about 40,000 km away.
   * Defaults to `false`, which matches cuSpatial (it does not wrap). Compile-time.
   */
  wrapLongitude?: boolean;
};

/**
 * Applies cuSpatial's local longitude/latitude projection in kilometres.
 *
 * It is cuSpatial's `sinusoidal_projection` contract, kept exactly:
 * - The Earth is a sphere of circumference 40,000 km, 111.111 km per degree. That is not the
 *   6371008.8 m mean sphere of `GPUHaversineDistance` and the spatial-analysis operators
 *   (111.195 km per degree), so the two disagree by 0.075% for the same pair of points.
 * - Output `x` is `(originLongitude - longitude) * 111.111 * cos(midpointLatitude)` and `y` is
 *   `(originLatitude - latitude) * 111.111`, both from the point toward the origin: positive `x` is
 *   west of the origin and positive `y` is south of it, the reverse of an east/north projection.
 * - Longitude deltas are not wrapped unless `wrapLongitude` is set (cuSpatial never wraps).
 *
 * Raw Float64 coordinate deltas are rounded once to f32 before the remaining f32 arithmetic and
 * trigonometry. For new code, prefer the spatial-analysis local frames, which use the mean sphere
 * and east/north axes.
 */
export class GPUSinusoidalProjection {
  readonly id: string;
  readonly positions: GPUFloat32Positions | GPUFloat64Positions;
  readonly output: GPUFloat32Positions;
  readonly origin: readonly [number, number];
  /** Whether longitude deltas are wrapped into `[-180, 180)`. */
  readonly wrapLongitude: boolean;

  /** Creates a projection contributor without compiling or submitting GPU work. */
  constructor(props: GPUSinusoidalProjectionProps) {
    this.id = props.id ?? 'gpu-sinusoidal-projection';
    this.positions = props.positions;
    this.output = props.output;
    this.origin = props.origin ?? [0, 0];
    this.wrapLongitude = props.wrapLongitude ?? false;
    validateRowView(this.positions, POSITION_FORMATS, `${this.id} positions`);
    validateRowView(this.output, ['float32x2'], `${this.id} output`);
    validateMatchingRows(this.positions, this.output, `${this.id} positions and output`);
    validateSeparateBuffers(this.output, [this.positions], this.id);
    const [originLongitude, originLatitude] = this.origin;
    if (
      !Number.isFinite(originLongitude) ||
      !Number.isFinite(originLatitude) ||
      originLongitude < -180 ||
      originLongitude > 180 ||
      originLatitude < -90 ||
      originLatitude > 90
    ) {
      throw new Error(`${this.id} origin must be valid longitude/latitude degrees`);
    }
  }

  /** Adds one projection node per non-empty input chunk to the target graph. */
  addToGraph<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    assertGraphOwnership(graph, [this.positions, this.output], this.id);
    const inputChunks = getRowChunks(this.positions);
    const outputChunks = getRowChunks(this.output);
    for (let chunkIndex = 0; chunkIndex < inputChunks.length; chunkIndex++) {
      const input = inputChunks[chunkIndex];
      const output = outputChunks[chunkIndex];
      if (input.length === 0) continue;
      const inputSource = getPositionReadSource('positions', input);
      const dispatchLayout = getGeospatialDispatchLayout(
        input.length,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      );
      const outputOffset = (output.byteOffset % 256) / 4 / 2;
      const positionSetup = inputSource.precise
        ? `let rawLongitudeLatitude = ${inputSource.read('index')};
  let longitudeLatitude = rawPointToF32(rawLongitudeLatitude);
  var longitudeDelta = sub_fp64u32_to_f32(ORIGIN_LONGITUDE_BITS, rawLongitudeLatitude.x);
  let latitudeDelta = sub_fp64u32_to_f32(ORIGIN_LATITUDE_BITS, rawLongitudeLatitude.y);`
        : `let longitudeLatitude = ${inputSource.read('index')};
  var longitudeDelta = ORIGIN_LONGITUDE - longitudeLatitude.x;
  let latitudeDelta = ORIGIN_LATITUDE - longitudeLatitude.y;`;
      const source = /* wgsl */ `
${inputSource.precise ? RAW_POINT_WGSL : ''}
const ELEMENT_COUNT: u32 = ${input.length}u;
const OUTPUT_OFFSET: u32 = ${outputOffset}u;
const ORIGIN_LONGITUDE: f32 = ${getFloat32Literal(this.origin[0])};
const ORIGIN_LATITUDE: f32 = ${getFloat32Literal(this.origin[1])};
const ORIGIN_LONGITUDE_BITS: vec2u = ${getRawBinary64Literal(this.origin[0])};
const ORIGIN_LATITUDE_BITS: vec2u = ${getRawBinary64Literal(this.origin[1])};
const DEGREES_TO_RADIANS: f32 = ${getFloat32Literal(Math.PI / 180)};
const KILOMETRES_PER_DEGREE: f32 = ${getFloat32Literal(CUSPATIAL_KILOMETRES_PER_DEGREE)};
${inputSource.declaration}
@group(0) @binding(auto) var<storage, read_write> outputPositions: array<vec2f>;

@compute @workgroup_size(${GEOSPATIAL_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  ${getGeospatialInvocationIndexSource(dispatchLayout)}
  if (index >= ELEMENT_COUNT) { return; }
  ${positionSetup}
  ${
    this.wrapLongitude
      ? '// Wrap into [-180, 180) so antimeridian neighbours stay next to the origin.\n  longitudeDelta = longitudeDelta - 360.0 * floor((longitudeDelta + 180.0) / 360.0);'
      : ''
  }
  let midpointLatitudeRadians =
    (longitudeLatitude.y + ORIGIN_LATITUDE) * 0.5 * DEGREES_TO_RADIANS;
  outputPositions[OUTPUT_OFFSET + index] = vec2f(
    longitudeDelta * KILOMETRES_PER_DEGREE * cos(midpointLatitudeRadians),
    latitudeDelta * KILOMETRES_PER_DEGREE
  );
}`;
      addGeospatialPass(graph, {
        id: isGraphVectorView(this.positions) ? `${this.id}-chunk-${chunkIndex}` : this.id,
        source,
        resources: [
          {buffer: input, usage: 'storage-read'},
          {buffer: output, usage: 'storage-write'}
        ],
        bindings: {positions: input, outputPositions: output},
        dispatchLayout,
        precise: inputSource.precise
      });
    }
  }
}
