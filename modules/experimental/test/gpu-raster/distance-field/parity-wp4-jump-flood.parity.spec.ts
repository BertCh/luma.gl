// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture} from '@luma.gl/core';
import {GPURasterTextureToBuffer} from '../../../src/gpu-raster/gpu-raster-texture-to-buffer';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUDistanceField,
  getGPUDistanceFieldParameterValues
} from '../../../src/gpu-raster/distance-field';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {distanceCases, decodePackedRuns} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const NONE = 0xffffffff;
for (const scene of distanceCases) {
  for (const refinement of [-1, 0, 1, 2] as const) {
    for (const pointMode of [false, true]) {
      for (const reverseRows of [false, true]) {
        if ((pointMode || reverseRows || refinement === -1) && scene.width * scene.height > 10000)
          continue;
        it(`GPUDistanceField independent EDT quality: ${scene.name}/algorithm-${refinement}/points-${pointMode}/south-${reverseRows}`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const owner = new ParityRasterGraph(device, scene.name);
          const length = scene.width * scene.height;
          const [mask, effective, squared] = await Promise.all([
            decodePackedRuns(scene.mask, length),
            decodePackedRuns(scene.effectiveMask, length),
            decodePackedRuns(scene.squared, length)
          ]);
          const grid = makeGPURasterMetadata({
            width: scene.width,
            height: scene.height,
            cellSize: [scene.cellSize[0], scene.cellSize[1]],
            rowDirection: reverseRows ? 'south' : 'north'
          });
          // Re-express the independently rasterized seed set at exact cell centres.
          // Translation and row reflection are isometries, so the EDT reference is unchanged.
          const points: number[] = [];
          const identifiers: number[] = [];
          for (let cell = 0; cell < length; cell++)
            if (effective[cell]) {
              points.push(
                ((cell % scene.width) + 0.5) * scene.cellSize[0],
                (Math.floor(cell / scene.width) + 0.5) * scene.cellSize[1] * (reverseRows ? -1 : 1)
              );
              identifiers.push(effective[cell] - 1);
            }
          const texture = pointMode
            ? device.createTexture({
                format: 'r32float',
                width: scene.width,
                height: scene.height,
                usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST
              })
            : undefined;
          const textureView = texture
            ? owner.graph.createTextureView(
                owner.graph.importTexture(
                  {
                    id: 'distance-texture',
                    format: 'r32float',
                    width: scene.width,
                    height: scene.height,
                    usage: texture.props.usage
                  },
                  texture
                )
              )
            : undefined;
          const textureValues = owner.output('texture-values', 'float32', length);
          const textureValidity = owner.output('texture-validity', 'uint32', length);
          const distances = owner.output('distances', 'float32', length);
          const nearest = owner.output('nearest', 'uint32', length);
          const allocation = owner.output('allocation', 'uint32', length);
          owner.graph.add(
            new GPUDistanceField({
              grid,
              mode: refinement === -1 ? 'exact' : 'jump-flood',
              jumpFloodRefinementIterationCount: refinement === -1 ? undefined : refinement,
              parameters: owner.input(
                'parameters',
                'float32',
                getGPUDistanceFieldParameterValues({maximumDistance: scene.maximumDistance}, grid)
              ),
              seedMask: pointMode ? undefined : owner.input('mask', 'uint32', mask),
              ...(pointMode
                ? {
                    seedPositions: owner.input(
                      'points',
                      'float32x2',
                      new Float32Array(points.length ? points : [-1, -1])
                    ),
                    seedIds: owner.input(
                      'ids',
                      'uint32',
                      new Uint32Array(identifiers.length ? identifiers : [NONE])
                    ),
                    seedCount: owner.input('count', 'uint32', Uint32Array.of(identifiers.length))
                  }
                : scene.positions.length
                  ? {
                      seedPositions: owner.input(
                        'positions',
                        'float32x2',
                        Float32Array.from(scene.positions, (value, index) =>
                          reverseRows && index % 2 === 1 ? -value : value
                        )
                      ),
                      seedIds: owner.input('seed-ids', 'uint32', new Uint32Array(scene.ids)),
                      seedCount: owner.input('seed-count', 'uint32', Uint32Array.of(scene.count))
                    }
                  : {}),
              output: {
                distances: distances.view,
                texture: textureView,
                nearestCells: nearest.view,
                allocation: allocation.view
              }
            })
          );
          if (textureView)
            new GPURasterTextureToBuffer({
              input: {
                id: 'distance-texture',
                format: 'float32',
                storage: {kind: 'texture', view: textureView}
              },
              output: textureValues.view,
              outputValidity: textureValidity.view
            }).addToGraph(owner.graph);
          const compiled = owner.graph.compile();
          try {
            submitGraph(device, compiled, undefined);
            const actual = await readFloat32(distances.buffer, length);
            const cells = await readUint32(nearest.buffer, length);
            const labels = await readUint32(allocation.buffer, length);
            const witnesses = new Float64Array(length);
            const feasible = new Uint32Array(length);
            let maxAbs = 0;
            let maxRel = 0;
            let missedCutoff = 0;
            for (let index = 0; index < length; index++) {
              const optimum = squared[index] === NONE ? Infinity : Math.sqrt(squared[index]);
              const cell = cells[index];
              if (cell === NONE) {
                witnesses[index] = Infinity;
                feasible[index] = Number(
                  labels[index] === NONE &&
                    (optimum === Infinity ||
                      // No approximate seed can exceed the raster diameter. Thus a cutoff
                      // covering that diameter cannot justify an unpublished finite seed.
                      scene.maximumDistance <
                        Math.hypot(
                          (scene.width - 1) * scene.cellSize[0],
                          (scene.height - 1) * scene.cellSize[1]
                        ))
                );
                if (optimum <= scene.maximumDistance && Number.isFinite(optimum)) missedCutoff++;
                continue;
              }
              const horizontal = ((cell % scene.width) - (index % scene.width)) * scene.cellSize[0];
              const vertical =
                (Math.floor(cell / scene.width) - Math.floor(index / scene.width)) *
                scene.cellSize[1];
              const witnessSquared = horizontal ** 2 + vertical ** 2;
              witnesses[index] = Math.sqrt(witnessSquared);
              feasible[index] = Number(
                cell < length &&
                  effective[cell] !== 0 &&
                  labels[index] === effective[cell] - 1 &&
                  witnessSquared >= squared[index] &&
                  witnesses[index] <= scene.maximumDistance
              );
              const error = witnesses[index] - optimum;
              maxAbs = Math.max(maxAbs, error);
              maxRel = Math.max(maxRel, error / Math.max(1, optimum));
            }
            if (refinement === -1)
              compareRasterParity(
                'GPUDistanceField',
                `${scene.name}/exact/points-${pointMode}/south-${reverseRows}`,
                scene.reference,
                'exact-distances',
                actual,
                Float64Array.from(squared, value =>
                  value === NONE || Math.sqrt(value) > scene.maximumDistance
                    ? Infinity
                    : Math.sqrt(value)
                ),
                'L'
              );
            if (pointMode) {
              // The raster-band adapter maps nonfinite texture samples to NaN/invalid.
              // The direct distance buffer below retains the distance API's +Infinity.
              compareRasterParity(
                'GPUDistanceField',
                `${scene.name}/algorithm-${refinement}/south-${reverseRows}`,
                scene.reference,
                'texture-distance-to-every-reported-seed',
                await readFloat32(textureValues.buffer, length),
                Float64Array.from(witnesses, value => (Number.isFinite(value) ? value : NaN)),
                'L'
              );
              compareRasterParity(
                'GPUDistanceField',
                `${scene.name}/algorithm-${refinement}/south-${reverseRows}`,
                scene.reference,
                'texture-validity',
                await readUint32(textureValidity.buffer, length),
                Uint32Array.from(witnesses, value => Number(Number.isFinite(value))),
                'E'
              );
            }
            // Jump flooding is documented approximate. Report independent optimum gaps,
            // never relabel a quality report as exact EDT parity or widen R to hide error.
            console.log(
              'PARITY',
              JSON.stringify({
                class: 'GPUDistanceField',
                case: `${scene.name}/algorithm-${refinement}/points-${pointMode}/south-${reverseRows}`,
                reference: scene.reference,
                n: length,
                metric: 'approximation-gap-to-exact-EDT',
                maxAbs,
                maxRel,
                mismatches: missedCutoff,
                tolerance: {kind: 'quality report; no equality claim', missedCutoff}
              })
            );
            expect(maxAbs).toBeGreaterThanOrEqual(0);
            compareRasterParity(
              'GPUDistanceField',
              `${scene.name}/algorithm-${refinement}/points-${pointMode}/south-${reverseRows}`,
              scene.reference,
              'all-returned-seeds-feasible-and-no-underestimate',
              feasible,
              new Uint32Array(length).fill(1),
              'E'
            );
            compareRasterParity(
              'GPUDistanceField',
              `${scene.name}/algorithm-${refinement}/points-${pointMode}/south-${reverseRows}`,
              scene.reference,
              'distance-to-every-reported-seed',
              actual,
              witnesses,
              'L'
            );
          } finally {
            compiled.destroy();
            texture?.destroy();
            owner.destroy();
          }
        });
      }
    }
  }
}
