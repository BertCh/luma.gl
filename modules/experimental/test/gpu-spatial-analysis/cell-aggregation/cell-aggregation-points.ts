// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared point sets for the cell aggregation specs: random points plus pole, antimeridian, signed
// zero, subnormal, and exact tile-edge coordinates.

/** Deterministic xorshift in [0, 1). */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/** Longitude/latitude pairs on and around Quadbin edge cases at `resolution`. */
export function getEdgeCasePoints(resolution: number): [number, number][] {
  const points: [number, number][] = [];
  const longitudes = [
    -180,
    180,
    -179.99999,
    179.99999,
    0,
    -0,
    1e-30,
    -1e-30,
    1e-40,
    -1e-40,
    360,
    -360,
    Infinity,
    -Infinity
  ];
  const latitudes = [
    90, -90, 85.051129, -85.051129, 85.05, -85.05, 0, -0, 1e-30, -1e-40, 89.999, -89.999
  ];
  for (const longitude of longitudes) {
    for (const latitude of latitudes) {
      points.push([longitude, latitude]);
    }
  }
  // Exact tile-column edges and their f32 neighbours.
  const tileCount = 2 ** resolution;
  for (let step = 0; step <= 16; step++) {
    const column = Math.floor((step / 16) * tileCount);
    const longitude = Math.fround(-180 + (360 * column) / tileCount);
    for (const offset of [-1, 0, 1]) {
      points.push([nextFloat32(longitude, offset), 10]);
    }
  }
  // Tile-row edges (f64 inverse Mercator) and their f32 neighbours.
  for (let step = 1; step < 16; step++) {
    const row = Math.floor((step / 16) * tileCount);
    const latitude = (Math.atan(Math.sinh(Math.PI * (1 - (2 * row) / tileCount))) * 180) / Math.PI;
    for (const offset of [-1, 0, 1]) {
      points.push([20, nextFloat32(Math.fround(latitude), offset)]);
    }
  }
  return points;
}

/** The f32 `steps` ULPs away from `value` (an f32). */
export function nextFloat32(value: number, steps: number): number {
  if (steps === 0 || !Number.isFinite(value)) {
    return value;
  }
  const buffer = new DataView(new ArrayBuffer(4));
  buffer.setFloat32(0, value);
  let bits = buffer.getInt32(0);
  if (value === 0) {
    bits = steps > 0 ? 1 : -2147483647;
    buffer.setInt32(0, bits);
    return buffer.getFloat32(0);
  }
  bits += (value > 0 ? 1 : -1) * steps;
  buffer.setInt32(0, bits);
  return buffer.getFloat32(0);
}

/** Random f32 points over the whole globe plus clustered points, then the edge cases. */
export function createPointPositions(
  seed: number,
  count: number,
  resolution: number
): Float32Array {
  const random = createRandom(seed);
  const points: [number, number][] = [];
  for (let index = 0; index < count; index++) {
    if (index % 3 === 0) {
      // Clustered around a few centres so cells hold many rows.
      const centre = index % 9;
      points.push([-100 + centre * 25 + random() * 0.5, -40 + centre * 10 + random() * 0.5]);
    } else {
      points.push([random() * 360 - 180, random() * 180 - 90]);
    }
  }
  points.push(...getEdgeCasePoints(resolution));
  return Float32Array.from(points.flat());
}
