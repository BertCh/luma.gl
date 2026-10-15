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

for (const original of pyramidCases) {
  for (const mode of ['combined', 'minimum', 'maximum'] as const) {
    for (const truncate of [false, true]) {
      for (const firstLevel of original.firstBlockSize === 4
        ? [0, 1, 2].filter(level => level < original.levels.length)
        : [0]) {
        const sourceOffset = original.levels[firstLevel].offset;
        const selectedLevels = original.levels
          .slice(firstLevel, truncate ? firstLevel + 1 : undefined)
          .map((level, index) => ({...level, level: index, offset: level.offset - sourceOffset}));
        const last = selectedLevels[selectedLevels.length - 1];
        const scene = {
          ...original,
          firstBlockSize: selectedLevels[0].blockSize,
          name: `${original.name}/${mode}/block-${selectedLevels[0].blockSize}/${truncate ? 'first-level' : 'all-levels'}`,
          levels: selectedLevels,
          length: last.offset + last.width * last.height
        };
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
          const layout = getGPURasterExtremaPyramidLayout(grid, {
            ...scene,
            levelCount: selectedLevels.length
          });
          const combined = owner.output('combined', 'float32', 2 * scene.length);
          const outputMaximum = owner.output('maximum', 'float32', scene.length);
          const outputMinimum = owner.output('minimum', 'float32', scene.length);
          const compare = (
            metric: string,
            actual: ArrayLike<number>,
            expected: ArrayLike<number>
          ) =>
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
            layout.levels.flatMap(level => [
              level.blockSize,
              level.width,
              level.height,
              level.offset
            ]),
            scene.levels.flatMap(level => [
              level.blockSize,
              level.width,
              level.height,
              level.offset
            ])
          );
          owner.graph.add(
            new GPURasterExtremaPyramid({
              grid,
              firstBlockSize: scene.firstBlockSize,
              levelCount: selectedLevels.length,
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
              output:
                mode === 'combined'
                  ? {combined: combined.view}
                  : mode === 'maximum'
                    ? {maximum: outputMaximum.view}
                    : {minimum: outputMinimum.view}
            })
          );
          const compiled = owner.graph.compile();
          try {
            submitGraph(device, compiled, undefined);
            if (mode === 'combined') {
              compare(
                'combined-maximum-then-minimum',
                await readFloat32(combined.buffer, 2 * scene.length),
                [
                  ...maximum.slice(sourceOffset, sourceOffset + scene.length),
                  ...minimum.slice(sourceOffset, sourceOffset + scene.length)
                ]
              );
            } else {
              compare(
                mode,
                await readFloat32(
                  mode === 'maximum' ? outputMaximum.buffer : outputMinimum.buffer,
                  scene.length
                ),
                (mode === 'maximum' ? maximum : minimum).slice(
                  sourceOffset,
                  sourceOffset + scene.length
                )
              );
            }
          } finally {
            compiled.destroy();
            owner.destroy();
          }
        });
      }
    }
  }
}
