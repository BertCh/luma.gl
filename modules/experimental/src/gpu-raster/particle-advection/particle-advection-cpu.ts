// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {sampleFieldOnCPU, type FieldRaster} from './particle-advection-field';
import {getPhilox4x32, getPhiloxUnitFloat} from './particle-advection-random';

/** Philox key word 1 of the respawn position and staggered-age stream. */
export const PARTICLE_ADVECTION_SPAWN_PURPOSE = 1;
/** Philox key word 1 of the per-frame drop test stream. */
export const PARTICLE_ADVECTION_DROP_PURPOSE = 2;

/** Mutable CPU particle state, laid out like the GPU views. */
export type ParticleAdvectionCPUState = {
  /** `particleCount * 2` positions. */
  positions: Float32Array;
  /** `particleCount * 2` positions before the last step. */
  previousPositions: Float32Array;
  /** Frames since spawn. */
  ages: Uint32Array;
  /** Respawn count. */
  generations: Uint32Array;
  /** `|v| * |speedScale|` at the current position, 0 when the sample is invalid. */
  speeds: Float32Array;
  /** Optional `particleCount * trailLength * 2` trail ring. */
  trails?: Float32Array;
};

/** Creates zeroed CPU state for `particleCount` particles and an optional trail ring. */
export function createParticleAdvectionCPUState(
  particleCount: number,
  trailLength: number = 0
): ParticleAdvectionCPUState {
  return {
    positions: new Float32Array(particleCount * 2),
    previousPositions: new Float32Array(particleCount * 2),
    ages: new Uint32Array(particleCount),
    generations: new Uint32Array(particleCount),
    speeds: new Float32Array(particleCount),
    trails: trailLength > 0 ? new Float32Array(particleCount * trailLength * 2) : undefined
  };
}

const fround = Math.fround;

function getSpeed(sample: [number, number], speedScale: number): number {
  return fround(fround(Math.hypot(sample[0], sample[1])) * Math.abs(speedScale));
}

/**
 * CPU oracle of one `GPUParticleAdvection` frame. Updates `state` in place.
 *
 * Mirrors the WGSL with f32 rounding of every intermediate. Random decisions (drop tests, spawn
 * positions, staggered ages) are bit-identical to the GPU; positions agree to a few f32 ulps
 * because WGSL division, `length` and fused multiply-add may round differently.
 *
 * @param state Particle state, updated in place.
 * @param field Packed vector field.
 * @param parameters Float parameters from `getGPUParticleAdvectionParameterValues`.
 * @param words Word parameters from `getGPUParticleAdvectionWordParameterValues`.
 * @param trailLength Trail ring length `L`, or 0 without trails.
 */
export function advanceParticlesOnCPU(
  state: ParticleAdvectionCPUState,
  field: FieldRaster,
  parameters: ArrayLike<number>,
  words: ArrayLike<number>,
  trailLength: number = 0
): void {
  const extent = [parameters[0], parameters[1], parameters[2], parameters[3]].map(fround);
  const timeStep = fround(parameters[4]);
  const speedScale = fround(parameters[5]);
  const dropRate = fround(parameters[6]);
  const minimumSpeed = fround(parameters[7]);
  const spawnMin = [fround(parameters[8]), fround(parameters[9])];
  const spawnMax = [fround(parameters[10]), fround(parameters[11])];
  const seed = words[0] >>> 0;
  const frame = words[1] >>> 0;
  const maximumAge = words[2] >>> 0;
  const reset = words[3] !== 0;
  const step = fround(timeStep * speedScale);
  const halfStep = fround(0.5 * step);
  const particleCount = state.ages.length;

  for (let particle = 0; particle < particleCount; particle++) {
    let x = state.positions[2 * particle];
    let y = state.positions[2 * particle + 1];
    let previousX = x;
    let previousY = y;
    let age = state.ages[particle];
    let generation = state.generations[particle];
    let respawn = reset;
    let speed = 0;
    if (reset) {
      generation = 0;
    } else {
      const k1 = sampleFieldOnCPU(field, x, y, extent);
      const middleX = fround(x + fround(halfStep * (k1?.[0] ?? 0)));
      const middleY = fround(y + fround(halfStep * (k1?.[1] ?? 0)));
      const k2 = sampleFieldOnCPU(field, middleX, middleY, extent);
      const nextX = fround(x + fround(step * (k2?.[0] ?? 0)));
      const nextY = fround(y + fround(step * (k2?.[1] ?? 0)));
      const landing = sampleFieldOnCPU(field, nextX, nextY, extent);
      const dropWord = getPhilox4x32(
        [particle, frame, 0, 0],
        [seed, PARTICLE_ADVECTION_DROP_PURPOSE]
      )[0];
      const expired = maximumAge !== 0 && age + 1 >= maximumAge;
      const tooSlow = k2 !== undefined && getSpeed(k2, speedScale) < minimumSpeed;
      if (!k1 || !k2 || !landing || expired || tooSlow || getPhiloxUnitFloat(dropWord) < dropRate) {
        respawn = true;
        generation = (generation + 1) >>> 0;
      } else {
        x = nextX;
        y = nextY;
        age = (age + 1) >>> 0;
        speed = getSpeed(landing, speedScale);
      }
    }
    if (respawn) {
      const random = getPhilox4x32(
        [particle, generation, 0, 0],
        [seed, PARTICLE_ADVECTION_SPAWN_PURPOSE]
      );
      x = fround(
        spawnMin[0] + fround(getPhiloxUnitFloat(random[0]) * fround(spawnMax[0] - spawnMin[0]))
      );
      y = fround(
        spawnMin[1] + fround(getPhiloxUnitFloat(random[1]) * fround(spawnMax[1] - spawnMin[1]))
      );
      previousX = x;
      previousY = y;
      age = reset
        ? Math.min(
            Math.floor(fround(getPhiloxUnitFloat(random[2]) * fround(maximumAge))),
            Math.max(maximumAge, 1) - 1
          )
        : 0;
      const landing = sampleFieldOnCPU(field, x, y, extent);
      speed = landing ? getSpeed(landing, speedScale) : 0;
    }
    state.positions[2 * particle] = x;
    state.positions[2 * particle + 1] = y;
    state.previousPositions[2 * particle] = previousX;
    state.previousPositions[2 * particle + 1] = previousY;
    state.ages[particle] = age;
    state.generations[particle] = generation;
    state.speeds[particle] = speed;

    if (state.trails && trailLength > 0) {
      const base = particle * trailLength;
      if (reset || age === 0) {
        for (let slot = 0; slot < trailLength; slot++) {
          state.trails[2 * (base + slot)] = x;
          state.trails[2 * (base + slot) + 1] = y;
        }
      } else {
        const slot = frame % trailLength;
        state.trails[2 * (base + slot)] = x;
        state.trails[2 * (base + slot) + 1] = y;
      }
    }
  }
}
