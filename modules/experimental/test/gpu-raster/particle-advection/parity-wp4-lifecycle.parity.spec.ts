// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUParticleAdvection,
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionUint32ParameterValues,
  type GPUParticleAdvectionSettings,
  type GPUParticleAdvectionUint32Settings
} from '../../../src/gpu-raster/particle-advection';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import first from '../../oracles/fixtures/parity-wp4-lifecycle-8-0.json';
import second from '../../oracles/fixtures/parity-wp4-lifecycle-8-1.json';
import ocean from '../../oracles/fixtures/parity-wp4-lifecycle-8-2.json';
import third from '../../oracles/fixtures/parity-wp4-lifecycle-16-0.json';
import fourth from '../../oracles/fixtures/parity-wp4-lifecycle-16-1.json';
import fifth from '../../oracles/fixtures/parity-wp4-lifecycle-32-0.json';
import sixth from '../../oracles/fixtures/parity-wp4-lifecycle-32-1.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Frame = {
  settings: GPUParticleAdvectionSettings;
  words: GPUParticleAdvectionUint32Settings;
  velocity: number[];
  positions: number[];
  previous: number[];
  speeds: number[];
  ages: number[];
  generations: number[];
  trails: number[];
};
type LifecycleCase = {
  name: string;
  width: number;
  height: number;
  initial: number[];
  trailLength: number;
  outputMode: string;
  frames: Frame[];
};
for (const encoded of [first, second, ocean, third, fourth, fifth, sixth]) {
  const fixture = decodeOracleFixture<{reference: string; cases: LifecycleCase[]}>(encoded);
  for (const scene of fixture.cases) {
    it(`GPUParticleAdvection independent lifecycle: ${scene.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const owner = new ParityRasterGraph(device, scene.name);
      const count = scene.initial.length / 2;
      const field = owner.output('field', 'float32x2', scene.width * scene.height);
      const positions = owner.output('positions', 'float32x2', count);
      positions.buffer.write(new Float32Array(scene.initial));
      const ages = owner.output('ages', 'uint32', count);
      const generations = owner.output('generations', 'uint32', count);
      const previous = owner.output('previous', 'float32x2', count);
      const speeds = owner.output('speeds', 'float32', count);
      const trails = owner.output('trails', 'float32x2', count * scene.trailLength);
      const parameters = new GPUParameterBuffer(device, {
        id: 'parameters',
        format: 'float32',
        length: 12
      });
      const words = new GPUParameterBuffer(device, {id: 'words', format: 'uint32', length: 4});
      owner.graph.add(
        new GPUParticleAdvection({
          velocities: field.view,
          fieldWidth: scene.width,
          fieldHeight: scene.height,
          parameters: parameters.importToGraph(owner.graph),
          uint32Parameters: words.importToGraph(owner.graph),
          state: {positions: positions.view, ages: ages.view, generations: generations.view},
          output:
            scene.outputMode === 'none'
              ? undefined
              : {
                  previousPositions: scene.outputMode === 'all' ? previous.view : undefined,
                  speeds: speeds.view,
                  trails: {positions: trails.view, length: scene.trailLength}
                }
        })
      );
      const compiled = owner.graph.compile();
      try {
        for (const frame of scene.frames) {
          field.buffer.write(
            Float32Array.from(
              {length: scene.width * scene.height * 2},
              (_, index) => frame.velocity[index % 2]
            )
          );
          parameters.write(
            getGPUParticleAdvectionParameterValues(frame.settings, [scene.width, scene.height])
          );
          words.write(getGPUParticleAdvectionUint32ParameterValues(frame.words));
          submitGraph(device, compiled, undefined);
          const compare = (
            metric: string,
            actual: ArrayLike<number>,
            expected: ArrayLike<number>,
            profile: 'E' | 'L' = 'L'
          ) =>
            compareRasterParity(
              'GPUParticleAdvection',
              `${scene.name}/${frame.words.frame}`,
              fixture.reference,
              metric,
              actual,
              expected,
              profile
            );
          compare('positions', await readFloat32(positions.buffer, count * 2), frame.positions);
          compare('ages', await readUint32(ages.buffer, count), frame.ages, 'E');
          compare(
            'generations',
            await readUint32(generations.buffer, count),
            frame.generations,
            'E'
          );
          if (scene.outputMode !== 'none') {
            compare('speeds', await readFloat32(speeds.buffer, count), frame.speeds);
            compare(
              'all-trail-slots',
              await readFloat32(trails.buffer, frame.trails.length),
              frame.trails
            );
          }
          if (scene.outputMode === 'all')
            compare('previous', await readFloat32(previous.buffer, count * 2), frame.previous);
        }
      } finally {
        compiled.destroy();
        parameters.destroy();
        words.destroy();
        owner.destroy();
      }
    });
  }
}
