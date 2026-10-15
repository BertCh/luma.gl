// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPURasterExtremaPyramid,
  getGPURasterExtremaPyramidLayout
} from '../../../src/gpu-raster/raster-pyramid';
import {pyramidCases} from '../../oracles/parity-wp4-round4-fixtures';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRoundFour} from '../../oracles/parity-wp4-round4-compare';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of pyramidCases) {
  it(`GPURasterExtremaPyramid NumPy parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const [values, validity, maximum, minimum] = await Promise.all([
      decodePackedNumbers(scene.values),
      decodePackedNumbers(scene.validity),
      decodePackedNumbers(scene.maximum),
      decodePackedNumbers(scene.minimum)
    ]);
    const owner = new ParityRasterGraph(device, scene.name);
    const grid = makeGPURasterMetadata(scene);
    const layout = getGPURasterExtremaPyramidLayout(grid, scene);
    const outputMaximum = owner.output('maximum', 'float32', scene.length);
    const outputMinimum = owner.output('minimum', 'float32', scene.length);
    const compare = (metric: string, actual: ArrayLike<number>, expected: ArrayLike<number>) =>
      compareRoundFour(
        'GPURasterExtremaPyramid',
        scene.name,
        scene.reference,
        metric,
        actual,
        expected,
        'E'
      );
    compare(
      'all-level-layouts',
      layout.levels.flatMap(level => [level.blockSize, level.width, level.height, level.offset]),
      scene.levels.flatMap(level => [level.blockSize, level.width, level.height, level.offset])
    );
    owner.graph.add(
      new GPURasterExtremaPyramid({
        grid,
        firstBlockSize: scene.firstBlockSize,
        footprint: scene.footprint,
        input: {
          id: 'elevation',
          format: 'float32',
          scale: scene.scale,
          offset: scene.offset,
          noDataValue: -9999,
          storage: {
            kind: 'buffer',
            values: owner.input('values', 'float32', new Float32Array(values))
          },
          validity: owner.input('validity', 'uint32', new Uint32Array(validity))
        },
        output: {maximum: outputMaximum.view, minimum: outputMinimum.view}
      })
    );
    const compiled = owner.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      compare(
        'every-level-maximum-including-empty-footprints',
        await readFloat32(outputMaximum.buffer, scene.length),
        maximum
      );
      compare(
        'every-level-minimum-including-empty-footprints',
        await readFloat32(outputMinimum.buffer, scene.length),
        minimum
      );
    } finally {
      compiled.destroy();
      owner.destroy();
    }
  });
}
