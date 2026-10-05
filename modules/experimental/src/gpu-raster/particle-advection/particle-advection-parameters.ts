// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Number of float32 elements in a {@link GPUParticleAdvection} parameter view. */
export const GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH = 12;

/** Number of uint32 elements in a {@link GPUParticleAdvection} word parameter view. */
export const GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH = 4;

/** Per-frame float settings of {@link GPUParticleAdvection}. */
export type GPUParticleAdvectionSettings = {
  /**
   * Field raster placement `[originX, originY, cellWidth, cellHeight]`. Row 0 is the row with the
   * smallest y; cell `(i, j)` is centred at `origin + (i + 0.5, j + 0.5) * cellSize`.
   */
  fieldExtent: readonly [number, number, number, number];
  /** Time step of this frame, in the time unit of the field velocities. */
  timeStep: number;
  /** Multiplier on the time step, for example a playback speed. Defaults to 1. */
  speedScale?: number;
  /** Probability in `[0, 1]` that a particle respawns this frame regardless of age. Defaults to 0. */
  dropRate?: number;
  /** Particles slower than this (world units per time unit, after `speedScale`) respawn. Defaults to 0. */
  minimumSpeed?: number;
  /**
   * Respawn rectangle `[minX, minY, maxX, maxY]`. Defaults to the field bounds. Particles spawned
   * outside the field respawn again on the next frame.
   */
  spawnBounds?: readonly [number, number, number, number];
};

/** Per-frame integer settings of {@link GPUParticleAdvection}. */
export type GPUParticleAdvectionWordSettings = {
  /** Random seed. Every random decision is a pure function of the seed and particle counters. */
  seed: number;
  /** Frame counter. Keys the per-frame drop test and selects the trail ring slot `frame % L`. */
  frame: number;
  /** Particles respawn when their age reaches this many frames. `0` disables ageing. */
  maximumAge: number;
  /**
   * When true, every particle respawns with generation 0 and a staggered age, which initialises
   * the state buffers. Pass it on the first frame and whenever the scene is reset.
   */
  reset?: boolean;
};

/**
 * Packs float parameters of {@link GPUParticleAdvection}.
 *
 * Layout: `[originX, originY, cellWidth, cellHeight, timeStep, speedScale, dropRate,
 * minimumSpeed, spawnMinX, spawnMinY, spawnMaxX, spawnMaxY]` as float32.
 *
 * @param settings Per-frame settings.
 * @param fieldSize `[width, height]` of the field in cells, used for the default spawn bounds.
 * @param target Optional destination of at least {@link GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH} elements.
 */
export function getGPUParticleAdvectionParameterValues(
  settings: GPUParticleAdvectionSettings,
  fieldSize: readonly [number, number],
  target: Float32Array = new Float32Array(GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH) {
    throw new Error(
      `Particle advection parameter target must hold ${GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH} elements`
    );
  }
  const [originX, originY, cellWidth, cellHeight] = settings.fieldExtent;
  const spawnBounds = settings.spawnBounds ?? [
    originX,
    originY,
    originX + cellWidth * fieldSize[0],
    originY + cellHeight * fieldSize[1]
  ];
  target.set([
    originX,
    originY,
    cellWidth,
    cellHeight,
    settings.timeStep,
    settings.speedScale ?? 1,
    settings.dropRate ?? 0,
    settings.minimumSpeed ?? 0,
    spawnBounds[0],
    spawnBounds[1],
    spawnBounds[2],
    spawnBounds[3]
  ]);
  return target;
}

/**
 * Packs integer parameters of {@link GPUParticleAdvection}.
 *
 * Layout: `[seed, frame, maximumAge, reset]` as uint32.
 *
 * @param settings Per-frame integer settings.
 * @param target Optional destination of at least {@link GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH} elements.
 */
export function getGPUParticleAdvectionWordParameterValues(
  settings: GPUParticleAdvectionWordSettings,
  target: Uint32Array = new Uint32Array(GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH)
): Uint32Array {
  if (target.length < GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH) {
    throw new Error(
      `Particle advection word parameter target must hold ${GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH} elements`
    );
  }
  for (const [name, value] of [
    ['seed', settings.seed],
    ['frame', settings.frame],
    ['maximumAge', settings.maximumAge]
  ] as const) {
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
      throw new Error(`Particle advection ${name} must be a uint32 integer`);
    }
  }
  target[0] = settings.seed;
  target[1] = settings.frame;
  target[2] = settings.maximumAge;
  target[3] = settings.reset ? 1 : 0;
  return target;
}
