// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUParticleAdvection,
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionUint32ParameterValues
} from '../../../src/gpu-raster/particle-advection';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import fixtureJSON from '../../oracles/fixtures/parity-wp4-advection.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type AdvectionCase = {
  name: string;
  reference: string;
  width: number;
  height: number;
  extent: [number, number, number, number];
  spawnBounds: [number, number, number, number];
  field: number[];
  seeds: number[];
  timeStep: number;
  times: number[];
  positions: number[][];
  previousPositions: number[][];
  speeds: number[][];
  ages: number[][];
  generations: number[];
  trailLength: number;
};
const fixture = decodeOracleFixture<{cases: AdvectionCase[]}>(fixtureJSON);

for (const scene of fixture.cases) {
  it(`GPUParticleAdvection independent SciPy parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = (values: Float32Array | Uint32Array) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return buffer;
    };
    const output = (length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return buffer;
    };
    const particleCount = scene.seeds.length / 2;
    const positions = output(scene.seeds.length);
    positions.write(new Float32Array(scene.seeds));
    const ages = output(particleCount);
    const generations = output(particleCount);
    const previous = output(scene.seeds.length);
    const speeds = output(particleCount);
    const trails = output(scene.seeds.length * scene.trailLength);
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 12
    });
    const words = new GPUParameterBuffer(device, {id: 'words', format: 'uint32', length: 4});
    parameters.write(
      getGPUParticleAdvectionParameterValues(
        {fieldExtent: scene.extent, timeStep: scene.timeStep, spawnBounds: scene.spawnBounds},
        [scene.width, scene.height]
      )
    );
    graph.add(
      new GPUParticleAdvection({
        velocities: importGraphBuffer(
          graph,
          'velocities',
          input(new Float32Array(scene.field)),
          'float32x2'
        ),
        fieldWidth: scene.width,
        fieldHeight: scene.height,
        parameters: parameters.importToGraph(graph),
        uint32Parameters: words.importToGraph(graph),
        state: {
          positions: importGraphBuffer(graph, 'positions', positions, 'float32x2'),
          ages: importGraphBuffer(graph, 'ages', ages, 'uint32'),
          generations: importGraphBuffer(graph, 'generations', generations, 'uint32')
        },
        output: {
          previousPositions: importGraphBuffer(graph, 'previous', previous, 'float32x2'),
          speeds: importGraphBuffer(graph, 'speeds', speeds, 'float32'),
          trails: {
            positions: importGraphBuffer(graph, 'trails', trails, 'float32x2'),
            length: scene.trailLength
          }
        }
      })
    );
    const compiled = graph.compile();
    const expectedTrails = new Float64Array(scene.seeds.length * scene.trailLength);
    try {
      for (let frame = 0; frame < scene.times.length; frame++) {
        words.write(getGPUParticleAdvectionUint32ParameterValues({seed: 7, frame, maximumAge: 0}));
        submitGraph(device, compiled, undefined);
        const compare = (
          metric: string,
          actual: ArrayLike<number>,
          expected: ArrayLike<number>,
          exact = false
        ) =>
          compareRasterParity(
            'GPUParticleAdvection',
            `${scene.name}/t=${scene.times[frame]}`,
            scene.reference,
            metric,
            actual,
            expected,
            exact ? 'E' : 'L'
          );
        compare(
          'positions',
          await readFloat32(positions, scene.seeds.length),
          scene.positions[frame]
        );
        compare(
          'previousPositions',
          await readFloat32(previous, scene.seeds.length),
          scene.previousPositions[frame]
        );
        compare('landingSpeeds', await readFloat32(speeds, particleCount), scene.speeds[frame]);
        compare('ages', await readUint32(ages, particleCount), scene.ages[frame], true);
        compare(
          'generations',
          await readUint32(generations, particleCount),
          scene.generations,
          true
        );
        for (let particle = 0; particle < particleCount; particle++) {
          for (let slot = 0; slot < scene.trailLength; slot++) {
            if (scene.ages[frame][particle] === 0 || slot === frame % scene.trailLength) {
              expectedTrails.set(
                scene.positions[frame].slice(2 * particle, 2 * particle + 2),
                2 * (particle * scene.trailLength + slot)
              );
            }
          }
        }
        compare('trailRing', await readFloat32(trails, expectedTrails.length), expectedTrails);
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      words.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
