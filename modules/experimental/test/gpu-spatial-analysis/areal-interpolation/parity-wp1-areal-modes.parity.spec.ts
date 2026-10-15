// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUArealInterpolation} from '../../../src/gpu-spatial-analysis/areal-interpolation';
import fixtureJson from '../../oracles/fixtures/parity-wp1-areal-modes.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareReference} from '../../oracles/parity-wp1-round8-compare';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {requireTestDevice} from '../../utils/require-test-device';

type ArealCase = {
  name: string;
  reference: string;
  width: number;
  height: number;
  sourceZones: number[];
  targetZones: number[];
  cellWeights: number[];
  sourceCount: number;
  targetCount: number;
  categories: number[];
  offsets: number[];
  neighbors: number[];
  areas: number[];
  sourceAreas: number[];
  categoryShares: number[];
  references: Record<
    string,
    {extensiveWeights: number[]; intensiveWeights: number[]; fractions: number[]}
  >;
};
const fixture = decodeOracleFixture<{cases: ArealCase[]}>(fixtureJson);
for (const scene of fixture.cases) {
  for (const extensive of ['zone', 'overlap'] as const)
    for (const intensive of ['zone', 'overlap'] as const)
      for (const mode of ['extensive', 'intensive'] as const) {
        it(`GPUArealInterpolation ${scene.name}/${extensive}/${intensive}/${mode}`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const rig = new AnalysisRig(device);
          const offsets = rig.output('uint32', scene.targetCount + 1);
          const neighbors = rig.output('uint32', scene.neighbors.length);
          const weights = rig.output('float32', scene.neighbors.length);
          const alternateWeights = rig.output('float32', scene.neighbors.length);
          const areas = rig.output('float32', scene.neighbors.length);
          const sourceAreas = rig.output('float32', scene.sourceCount);
          const sourceTransferredFractions = rig.output('float32', scene.sourceCount);
          const categoryShares = rig.output('float32', scene.targetCount * 2);
          const overflow = rig.output('uint32', 1);
          const invalidCount = rig.output('uint32', 1);
          const requiredPairCount = rig.output('uint32', 1);
          try {
            rig.run(
              new GPUArealInterpolation({
                grid: {
                  width: scene.width,
                  height: scene.height,
                  affine: [1, 0, 0, 0, 1, 0],
                  pixelInterpretation: 'area'
                },
                sourceZones: rig.input(new Uint32Array(scene.sourceZones), 'uint32'),
                targetZones: rig.input(new Uint32Array(scene.targetZones), 'uint32'),
                sourceZoneCount: scene.sourceCount,
                targetZoneCount: scene.targetCount,
                denominator: {extensive, intensive},
                mode,
                ...(scene.cellWeights.length
                  ? {cellWeights: rig.input(new Float32Array(scene.cellWeights), 'float32')}
                  : {}),
                categories: {
                  sourceCategories: rig.input(new Uint32Array(scene.categories), 'uint32'),
                  categoryCount: 2
                },
                output: {
                  weights: {offsets, neighbors, weights},
                  alternateWeights,
                  areas,
                  sourceAreas,
                  sourceTransferredFractions,
                  categoryShares,
                  status: {overflow, invalidCount, requiredPairCount}
                }
              })
            );
            const compare = (metric: string, actual: number[], expected: number[], exact = false) =>
              compareReference(
                'GPUArealInterpolation',
                {...scene, name: `${scene.name}/${extensive}/${intensive}/${mode}`},
                metric,
                actual,
                expected,
                exact
              );
            const expected = scene.references[`${extensive}-${intensive}`];
            compare('offsets', await rig.readUint(offsets), scene.offsets, true);
            compare('neighbors', await rig.readUint(neighbors), scene.neighbors, true);
            compare('areas', await rig.readFloat(areas), scene.areas, true);
            compare('source-areas', await rig.readFloat(sourceAreas), scene.sourceAreas, true);
            compare(
              'weights',
              await rig.readFloat(weights),
              mode === 'extensive' ? expected.extensiveWeights : expected.intensiveWeights
            );
            compare(
              'alternate-weights',
              await rig.readFloat(alternateWeights),
              mode === 'extensive' ? expected.intensiveWeights : expected.extensiveWeights
            );
            compare(
              'fractions',
              await rig.readFloat(sourceTransferredFractions),
              expected.fractions
            );
            compare('category-shares', await rig.readFloat(categoryShares), scene.categoryShares);
            compare('overflow', await rig.readUint(overflow), [0], true);
            compare('invalid-count', await rig.readUint(invalidCount), [1], true);
            compare(
              'pair-count',
              await rig.readUint(requiredPairCount),
              [scene.neighbors.length],
              true
            );
          } finally {
            rig.destroy();
          }
        });
      }
}
