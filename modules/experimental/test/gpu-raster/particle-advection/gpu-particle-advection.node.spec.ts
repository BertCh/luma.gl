// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  advanceParticlesOnCPU,
  createParticleAdvectionCPUState,
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionWordParameterValues,
  GPUParticleAdvection,
  type GPUParticleAdvectionProps
} from '../../../src/gpu-raster/particle-advection';
import {getPhilox4x32} from '../../../src/gpu-raster/particle-advection/particle-advection-random';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUParticleAdvectionProps> = {}
): GPUParticleAdvectionProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    velocities: view('float32x2', 12),
    fieldWidth: 4,
    fieldHeight: 3,
    parameters: view('float32', 12),
    wordParameters: view('uint32', 4),
    state: {
      positions: view('float32x2', 10),
      ages: view('uint32', 10),
      generations: view('uint32', 10)
    },
    previousPositions: view('float32x2', 10),
    speeds: view('float32', 10),
    trails: {positions: view('float32x2', 30), length: 3},
    ...overrides
  };
}

it('GPUParticleAdvection validates props and builds deterministic nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
    id: 'particle-validation'
  });
  const recipe = new GPUParticleAdvection(createProps(graph, {id: 'wind'}));
  expect(recipe.recipe).toBe('particle-advection');
  expect(recipe.particleCount).toBe(10);
  expect(recipe.getCommandNodes(graph).map(node => node.id)).toEqual(['wind-step', 'wind-trails']);
  const withoutTrails = new GPUParticleAdvection(createProps(graph, {trails: undefined}));
  expect(withoutTrails.getCommandNodes(graph)).toHaveLength(1);

  expect(() => new GPUParticleAdvection(createProps(graph, {fieldWidth: 5}))).toThrow(
    /fieldWidth \* fieldHeight/
  );
  expect(() => new GPUParticleAdvection(createProps(graph, {fieldHeight: 0}))).toThrow(
    /fieldHeight/
  );
  expect(
    () =>
      new GPUParticleAdvection(
        createProps(graph, {
          parameters: createTransientView(graph, 'short', 'float32', 4)
        })
      )
  ).toThrow(/parameters must contain 12/);
  expect(
    () =>
      new GPUParticleAdvection(
        createProps(graph, {
          speeds: createTransientView(graph, 'speeds-short', 'float32', 9)
        })
      )
  ).toThrow(/one row per particle/);
  expect(
    () =>
      new GPUParticleAdvection(
        createProps(graph, {
          trails: {
            positions: createTransientView(graph, 'trail-short', 'float32x2', 20),
            length: 3
          }
        })
      )
  ).toThrow(/trails.positions/);
  const props = createProps(graph);
  expect(
    () =>
      new GPUParticleAdvection({
        ...props,
        speeds: props.parameters as never
      })
  ).toThrow();
  const otherGraph = new GPUCommandGraph(createNullWebGPUDevice(), {
    id: 'other'
  });
  const foreign = new GPUParticleAdvection(createProps(otherGraph));
  expect(() => foreign.getCommandNodes(graph)).toThrow(/target graph/);
});

it('GPUParticleAdvection parameter helpers pack the documented layout', () => {
  expect(
    Array.from(
      getGPUParticleAdvectionParameterValues(
        {fieldExtent: [10, 20, 2, 4], timeStep: 0.5, dropRate: 0.25},
        [8, 4]
      )
    )
  ).toEqual([10, 20, 2, 4, 0.5, 1, 0.25, 0, 10, 20, 26, 36]);
  expect(
    Array.from(
      getGPUParticleAdvectionWordParameterValues({
        seed: 5,
        frame: 9,
        maximumAge: 60,
        reset: true
      })
    )
  ).toEqual([5, 9, 60, 1]);
  expect(() =>
    getGPUParticleAdvectionWordParameterValues({
      seed: -1,
      frame: 0,
      maximumAge: 1
    })
  ).toThrow(/seed/);
  expect(() =>
    getGPUParticleAdvectionParameterValues(
      {fieldExtent: [0, 0, 1, 1], timeStep: 1},
      [1, 1],
      new Float32Array(3)
    )
  ).toThrow(/12 elements/);
});

it('advanceParticlesOnCPU moves exactly along a uniform field and keys spawns by generation', () => {
  const width = 8;
  const velocities = new Float32Array(width * width * 2);
  for (let cell = 0; cell < width * width; cell++) {
    velocities[2 * cell] = 0.5;
    velocities[2 * cell + 1] = -0.25;
  }
  const field = {velocities, width, height: width};
  const parameters = getGPUParticleAdvectionParameterValues(
    {
      fieldExtent: [0, 0, 1, 1],
      timeStep: 2,
      speedScale: 1,
      spawnBounds: [3, 3, 5, 5]
    },
    [width, width]
  );
  const state = createParticleAdvectionCPUState(64, 4);
  advanceParticlesOnCPU(
    state,
    field,
    parameters,
    getGPUParticleAdvectionWordParameterValues({
      seed: 11,
      frame: 0,
      maximumAge: 100,
      reset: true
    }),
    4
  );
  for (let particle = 0; particle < 64; particle++) {
    const x = state.positions[2 * particle];
    const y = state.positions[2 * particle + 1];
    expect(x >= 3 && x < 5 && y >= 3 && y < 5).toBe(true);
    expect(state.generations[particle]).toBe(0);
    expect(state.ages[particle]).toBeLessThan(100);
  }
  // Staggered ages are spread, not all equal.
  expect(new Set(state.ages).size).toBeGreaterThan(20);
  const before = state.positions.slice();
  advanceParticlesOnCPU(
    state,
    field,
    parameters,
    getGPUParticleAdvectionWordParameterValues({
      seed: 11,
      frame: 1,
      maximumAge: 1000
    }),
    4
  );
  for (let particle = 0; particle < 64; particle++) {
    expect(state.positions[2 * particle]).toBeCloseTo(before[2 * particle] + 1, 5);
    expect(state.positions[2 * particle + 1]).toBeCloseTo(before[2 * particle + 1] - 0.5, 5);
    expect(state.previousPositions[2 * particle]).toBe(before[2 * particle]);
    expect(state.speeds[particle]).toBeCloseTo(Math.hypot(0.5, 0.25), 5);
  }
  // Spawn positions depend only on (seed, particle, generation), not on the particle count.
  const small = createParticleAdvectionCPUState(4);
  advanceParticlesOnCPU(
    small,
    field,
    parameters,
    getGPUParticleAdvectionWordParameterValues({
      seed: 11,
      frame: 0,
      maximumAge: 100,
      reset: true
    })
  );
  expect(Array.from(small.positions)).toEqual(Array.from(before.slice(0, 8)));
});

it('advanceParticlesOnCPU drop rate is exact and independent between frames', () => {
  const field = {
    velocities: new Float32Array(2).fill(0.1),
    width: 1,
    height: 1
  };
  const parameters = getGPUParticleAdvectionParameterValues(
    {fieldExtent: [0, 0, 100, 100], timeStep: 1, dropRate: 0.3},
    [1, 1]
  );
  const state = createParticleAdvectionCPUState(4000);
  advanceParticlesOnCPU(
    state,
    field,
    parameters,
    getGPUParticleAdvectionWordParameterValues({
      seed: 1,
      frame: 0,
      maximumAge: 0,
      reset: true
    })
  );
  advanceParticlesOnCPU(
    state,
    field,
    parameters,
    getGPUParticleAdvectionWordParameterValues({
      seed: 1,
      frame: 1,
      maximumAge: 0
    })
  );
  // A spawn rectangle equal to the 100 x 100 field keeps every step inside except at the edges.
  const dropped = Array.from(state.generations).filter(generation => generation === 1).length;
  expect(dropped / 4000).toBeGreaterThan(0.27);
  expect(dropped / 4000).toBeLessThan(0.34);
  // Philox words are reproducible.
  expect(getPhilox4x32([1, 2, 3, 4], [5, 6])).toEqual(getPhilox4x32([1, 2, 3, 4], [5, 6]));
});
