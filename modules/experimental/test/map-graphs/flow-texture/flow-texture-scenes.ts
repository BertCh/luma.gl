// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Packed test vector field. */
export type TestField = {
  velocities: Float32Array;
  width: number;
  height: number;
};

/** Rigid rotation with angular speed `omega` around the centre of a `size x size` unit-cell field. */
export function createVortexField(size: number, omega: number): TestField {
  const velocities = new Float32Array(size * size * 2);
  const centre = size / 2;
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      velocities[2 * (row * size + column)] = -omega * (row + 0.5 - centre);
      velocities[2 * (row * size + column) + 1] = omega * (column + 0.5 - centre);
    }
  }
  return {velocities, width: size, height: size};
}

/** Constant field `(u, v)` on a `width x height` grid. */
export function createUniformField(width: number, height: number, u: number, v: number): TestField {
  const velocities = new Float32Array(width * height * 2);
  for (let cell = 0; cell < width * height; cell++) {
    velocities[2 * cell] = u;
    velocities[2 * cell + 1] = v;
  }
  return {velocities, width, height};
}
