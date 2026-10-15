// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUParticleAdvection,
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionUint32ParameterValues
} from '../../../src/gpu-raster/particle-advection';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import fixture from '../../oracles/fixtures/parity-wp4-particle-topology.json';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
for (const {count, trailLength} of fixture.topologies) {
  it(`Wind particle topology independent full outputs: ${count}x${trailLength}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const owner = new ParityRasterGraph(device, 'particle-topology');
    const positions = owner.output('positions', 'float32x2', count);
    positions.buffer.write(new Float32Array(count * 2).fill(4));
    const ages = owner.output('ages', 'uint32', count);
    const generations = owner.output('generations', 'uint32', count);
    const speeds = owner.output('speeds', 'float32', count);
    const trails = owner.output('trails', 'float32x2', count * trailLength);
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 12
    });
    const words = new GPUParameterBuffer(device, {id: 'words', format: 'uint32', length: 4});
    owner.graph.add(
      new GPUParticleAdvection({
        fieldWidth: 8,
        fieldHeight: 8,
        velocities: owner.input(
          'field',
          'float32x2',
          Float32Array.from({length: 128}, (_, index) => (index % 2 === 0 ? 1 / 4096 : 0))
        ),
        parameters: parameters.importToGraph(owner.graph),
        uint32Parameters: words.importToGraph(owner.graph),
        state: {positions: positions.view, ages: ages.view, generations: generations.view},
        output: {speeds: speeds.view, trails: {positions: trails.view, length: trailLength}}
      })
    );
    const compiled = owner.graph.compile();
    const trailReference = new Float64Array(trailLength * 2);
    try {
      for (const [index, frame] of fixture.frames.entries()) {
        parameters.write(
          getGPUParticleAdvectionParameterValues(
            {
              fieldExtent: [0, 0, 1, 1],
              timeStep: frame.timeStep,
              spawnBounds: frame.viewport ? [2, 2, 6, 6] : undefined
            },
            [8, 8]
          )
        );
        words.write(
          getGPUParticleAdvectionUint32ParameterValues({seed: 5, frame: index, maximumAge: 120})
        );
        submitGraph(device, compiled, undefined);
        trailReference.set(frame.position, index * 2);
        const compare = (
          metric: string,
          actual: ArrayLike<number>,
          expected: ArrayLike<number>,
          profile: 'E' | 'L' = 'L'
        ) =>
          compareRasterParity(
            'GPUParticleAdvection',
            `count-${count}/trail-${trailLength}/viewport-${frame.viewport}/adaptive-${frame.adaptive}`,
            fixture.reference,
            metric,
            actual,
            expected,
            profile
          );
        compare(
          'all-positions',
          await readFloat32(positions.buffer, count * 2),
          Float64Array.from({length: count * 2}, (_, coordinate) => frame.position[coordinate % 2])
        );
        compare(
          'all-speeds',
          await readFloat32(speeds.buffer, count),
          new Float64Array(count).fill(frame.speed)
        );
        compare(
          'all-ages',
          await readUint32(ages.buffer, count),
          new Uint32Array(count).fill(index + 1),
          'E'
        );
        compare(
          'all-generations',
          await readUint32(generations.buffer, count),
          new Uint32Array(count),
          'E'
        );
        compare(
          'every-trail-slot',
          await readFloat32(trails.buffer, count * trailLength * 2),
          Float64Array.from(
            {length: count * trailLength * 2},
            (_, coordinate) => trailReference[coordinate % (trailLength * 2)]
          )
        );
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      words.destroy();
      owner.destroy();
    }
  });
}
