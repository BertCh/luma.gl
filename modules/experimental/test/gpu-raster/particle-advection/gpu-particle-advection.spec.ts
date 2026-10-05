// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  advanceParticlesOnCPU,
  createParticleAdvectionCPUState,
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionWordParameterValues,
  GPUParticleAdvection,
  type GPUParticleAdvectionSettings,
  type GPUParticleAdvectionWordSettings,
  type ParticleAdvectionCPUState
} from '../../../src/gpu-raster/particle-advection';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';

/** Absolute position tolerance: a few f32 ulps at coordinates near 64. */
const POSITION_TOLERANCE = 2e-4;

type Field = {velocities: Float32Array; width: number; height: number};

type Fixture = {
  run(
    settings: GPUParticleAdvectionSettings,
    words: GPUParticleAdvectionWordSettings
  ): Promise<ParticleAdvectionCPUState>;
  writeState(state: ParticleAdvectionCPUState): void;
  compileCount: number;
  destroy(): void;
};

/** Rigid rotation around the field centre with angular speed `omega`. */
function createVortexField(size: number, omega: number): Field {
  const velocities = new Float32Array(size * size * 2);
  const centre = size / 2;
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      const x = column + 0.5 - centre;
      const y = row + 0.5 - centre;
      velocities[2 * (row * size + column)] = -omega * y;
      velocities[2 * (row * size + column) + 1] = omega * x;
    }
  }
  return {velocities, width: size, height: size};
}

function createFixture(
  device: Device,
  field: Field,
  particleCount: number,
  trailLength: number
): Fixture {
  const graph = new GPUCommandGraph(device, {id: 'particle-advection-graph'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'particle-parameters',
    format: 'float32',
    length: 12
  });
  const wordBuffer = new GPUParameterBuffer(device, {
    id: 'particle-words',
    format: 'uint32',
    length: 4
  });
  const outputs = {
    positions: track(createOutputBuffer(device, particleCount * 2)),
    previous: track(createOutputBuffer(device, particleCount * 2)),
    ages: track(createOutputBuffer(device, particleCount)),
    generations: track(createOutputBuffer(device, particleCount)),
    speeds: track(createOutputBuffer(device, particleCount)),
    trails: track(createOutputBuffer(device, Math.max(particleCount * trailLength * 2, 1)))
  };
  graph.add(
    new GPUParticleAdvection({
      id: 'particles',
      velocities: importGraphBuffer(
        graph,
        'field',
        track(createInputBuffer(device, field.velocities)),
        'float32x2',
        field.width * field.height
      ),
      fieldWidth: field.width,
      fieldHeight: field.height,
      parameters: parameterBuffer.importToGraph(graph),
      wordParameters: wordBuffer.importToGraph(graph),
      state: {
        positions: importGraphBuffer(
          graph,
          'positions',
          outputs.positions,
          'float32x2',
          particleCount
        ),
        ages: importGraphBuffer(graph, 'ages', outputs.ages, 'uint32', particleCount),
        generations: importGraphBuffer(
          graph,
          'generations',
          outputs.generations,
          'uint32',
          particleCount
        )
      },
      previousPositions: importGraphBuffer(
        graph,
        'previous',
        outputs.previous,
        'float32x2',
        particleCount
      ),
      speeds: importGraphBuffer(graph, 'speeds', outputs.speeds, 'float32', particleCount),
      trails:
        trailLength > 0
          ? {
              positions: importGraphBuffer(
                graph,
                'trails',
                outputs.trails,
                'float32x2',
                particleCount * trailLength
              ),
              length: trailLength
            }
          : undefined
    })
  );
  const compiled = graph.compile();
  const fixture: Fixture = {
    compileCount: 1,
    async run(settings, words) {
      parameterBuffer.write(
        getGPUParticleAdvectionParameterValues(settings, [field.width, field.height])
      );
      wordBuffer.write(getGPUParticleAdvectionWordParameterValues(words));
      submitGraph(device, compiled, undefined);
      const state = createParticleAdvectionCPUState(particleCount, trailLength);
      state.positions.set(await readFloat32(outputs.positions, particleCount * 2));
      state.previousPositions.set(await readFloat32(outputs.previous, particleCount * 2));
      state.ages.set(await readUint32(outputs.ages, particleCount));
      state.generations.set(await readUint32(outputs.generations, particleCount));
      state.speeds.set(await readFloat32(outputs.speeds, particleCount));
      if (state.trails) {
        state.trails.set(await readFloat32(outputs.trails, particleCount * trailLength * 2));
      }
      return state;
    },
    writeState(state) {
      outputs.positions.write(state.positions);
      outputs.ages.write(state.ages);
      outputs.generations.write(state.generations);
      if (state.trails) {
        outputs.trails.write(state.trails);
      }
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      wordBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
  return fixture;
}

function cloneState(state: ParticleAdvectionCPUState): ParticleAdvectionCPUState {
  return {
    positions: state.positions.slice(),
    previousPositions: state.previousPositions.slice(),
    ages: state.ages.slice(),
    generations: state.generations.slice(),
    speeds: state.speeds.slice(),
    trails: state.trails?.slice()
  };
}

function getMaximumDifference(left: ArrayLike<number>, right: ArrayLike<number>): number {
  let maximum = 0;
  for (let index = 0; index < left.length; index++) {
    maximum = Math.max(maximum, Math.abs(left[index] - right[index]));
  }
  return maximum;
}

function expectStateClose(
  actual: ParticleAdvectionCPUState,
  expected: ParticleAdvectionCPUState
): void {
  expect(Array.from(actual.ages)).toEqual(Array.from(expected.ages));
  expect(Array.from(actual.generations)).toEqual(Array.from(expected.generations));
  expect(getMaximumDifference(actual.positions, expected.positions)).toBeLessThan(
    POSITION_TOLERANCE
  );
  expect(getMaximumDifference(actual.previousPositions, expected.previousPositions)).toBeLessThan(
    POSITION_TOLERANCE
  );
  expect(getMaximumDifference(actual.speeds, expected.speeds)).toBeLessThan(1e-4);
  if (actual.trails && expected.trails) {
    expect(getMaximumDifference(actual.trails, expected.trails)).toBeLessThan(POSITION_TOLERANCE);
  }
}

const VORTEX_SETTINGS: GPUParticleAdvectionSettings = {
  fieldExtent: [0, 0, 1, 1],
  timeStep: 0.5,
  dropRate: 0.01,
  spawnBounds: [16, 16, 48, 48]
};

it('GPUParticleAdvection matches the CPU oracle frame by frame on a vortex', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const field = createVortexField(64, 0.05);
  const particleCount = 2048;
  const trailLength = 8;
  const fixture = createFixture(device, field, particleCount, trailLength);
  try {
    let previous: ParticleAdvectionCPUState | undefined;
    let respawnCount = 0;
    for (let frame = 0; frame < 24; frame++) {
      const words = {seed: 7, frame, maximumAge: 16, reset: frame === 0};
      const actual = await fixture.run(VORTEX_SETTINGS, words);
      // Feed the oracle the GPU's previous state so float drift never accumulates.
      const expected = previous
        ? cloneState(previous)
        : createParticleAdvectionCPUState(particleCount, trailLength);
      advanceParticlesOnCPU(
        expected,
        field,
        getGPUParticleAdvectionParameterValues(VORTEX_SETTINGS, [64, 64]),
        getGPUParticleAdvectionWordParameterValues(words),
        trailLength
      );
      expectStateClose(actual, expected);
      if (previous) {
        for (let particle = 0; particle < particleCount; particle++) {
          respawnCount += actual.generations[particle] !== previous.generations[particle] ? 1 : 0;
        }
      }
      previous = actual;
    }
    // Ageing and the 1% drop rate both fire over 24 frames.
    expect(respawnCount).toBeGreaterThan(particleCount);
    expect(fixture.compileCount).toBe(1);
  } finally {
    fixture.destroy();
  }
});

it('GPUParticleAdvection replays bitwise and changes parameters without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const field = createVortexField(32, 0.1);
  const runSequence = async () => {
    const fixture = createFixture(device, field, 512, 4);
    const snapshots: ParticleAdvectionCPUState[] = [];
    try {
      for (let frame = 0; frame < 10; frame++) {
        // Speed scale, time step and drop rate change every frame on the same compiled graph.
        const settings: GPUParticleAdvectionSettings = {
          fieldExtent: [0, 0, 1, 1],
          timeStep: 0.25 + 0.05 * frame,
          speedScale: frame % 2 === 0 ? 1 : 2,
          dropRate: frame % 3 === 0 ? 0.2 : 0
        };
        snapshots.push(
          await fixture.run(settings, {
            seed: 99,
            frame,
            maximumAge: 6,
            reset: frame === 0
          })
        );
      }
      expect(fixture.compileCount).toBe(1);
    } finally {
      fixture.destroy();
    }
    return snapshots;
  };
  const first = await runSequence();
  const second = await runSequence();
  for (let frame = 0; frame < first.length; frame++) {
    expect(new Uint32Array(second[frame].positions.buffer)).toEqual(
      new Uint32Array(first[frame].positions.buffer)
    );
    expect(second[frame].generations).toEqual(first[frame].generations);
    expect(new Uint32Array(second[frame].trails!.buffer)).toEqual(
      new Uint32Array(first[frame].trails!.buffer)
    );
  }
  // A different speed scale moves particles differently on the same graph.
  expect(first[1].positions).not.toEqual(first[2].positions);
});

it('GPUParticleAdvection respawns on NaN cells, field exit, age, slowness and fills trails', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // 16 x 16 field moving +x at 1 unit per time; column 8 rows 0-3 are NaN; rows 12-15 are still.
  const size = 16;
  const velocities = new Float32Array(size * size * 2);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      const base = 2 * (row * size + column);
      velocities[base] = row >= 12 ? 0 : 1;
      if (column === 8 && row < 4) {
        velocities[base] = Number.NaN;
        velocities[base + 1] = Number.NaN;
      }
    }
  }
  const field = {velocities, width: size, height: size};
  const trailLength = 3;
  const fixture = createFixture(device, field, 5, trailLength);
  try {
    const state = createParticleAdvectionCPUState(5, trailLength);
    // 0: free mover. 1: steps into the NaN column. 2: exits through x = 16.
    // 3: reaches the maximum age. 4: sits in the still rows.
    state.positions.set([2.25, 6.25, 6.75, 1.25, 15.25, 8.25, 3.25, 9.25, 4.25, 13.25]);
    state.ages.set([0, 0, 0, 9, 0]);
    state.generations.set([3, 3, 3, 3, 3]);
    state.trails!.fill(-1);
    fixture.writeState(state);
    const settings: GPUParticleAdvectionSettings = {
      fieldExtent: [0, 0, 1, 1],
      timeStep: 1,
      minimumSpeed: 0.5,
      spawnBounds: [1, 1, 2, 2]
    };
    const words = {seed: 3, frame: 4, maximumAge: 10};
    const actual = await fixture.run(settings, words);
    const expected = cloneState(state);
    advanceParticlesOnCPU(
      expected,
      field,
      getGPUParticleAdvectionParameterValues(settings, [size, size]),
      getGPUParticleAdvectionWordParameterValues(words),
      trailLength
    );
    expectStateClose(actual, expected);
    expect(Array.from(actual.generations)).toEqual([3, 4, 4, 4, 4]);
    expect(Array.from(actual.ages)).toEqual([1, 0, 0, 0, 0]);
    expect(actual.positions[0]).toBeCloseTo(3.25, 5);
    // Respawned particles land in the spawn rectangle and their previous position equals it.
    for (let particle = 1; particle < 5; particle++) {
      const x = actual.positions[2 * particle];
      const y = actual.positions[2 * particle + 1];
      expect(x >= 1 && x < 2 && y >= 1 && y < 2).toBe(true);
      expect(actual.previousPositions[2 * particle]).toBe(x);
      // The whole trail ring of a respawned particle holds the spawn position.
      for (let slot = 0; slot < trailLength; slot++) {
        expect(actual.trails![2 * (particle * trailLength + slot)]).toBe(x);
      }
    }
    // The mover wrote only slot frame % L = 1.
    expect(Array.from(actual.trails!.slice(0, 6))).toEqual([-1, -1, 3.25, 6.25, -1, -1]);
  } finally {
    fixture.destroy();
  }
});
