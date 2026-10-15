// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUArealInterpolation} from '../../../src/gpu-spatial-analysis/areal-interpolation';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';
import fixtureJson from '../../oracles/fixtures/parity-wp1-areal-interpolation.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {requireTestDevice} from '../../utils/require-test-device';
import {GraphRig} from './areal-interpolation-harness';

type ArealReference = {
  extensiveWeights: number[];
  intensiveWeights: number[];
  extensive: number[];
  intensive: number[];
  sourceTransferredFractions: number[];
  sourceLostFractions: number[];
  massTotal: number;
  massLost: number;
};
type ArealCase = {
  name: string;
  reference: string;
  width: number;
  height: number;
  sourceZoneCount: number;
  targetZoneCount: number;
  sourceZones: number[];
  targetZones: number[];
  sourceValues: number[];
  sourceRates: number[];
  offsets: number[];
  neighbors: number[];
  areas: number[];
  sourceAreas: number[];
  targetUncoveredFractions: number[];
  invalidCount: number;
  references: Record<string, ArealReference>;
};
const fixture = decodeOracleFixture<{cases: ArealCase[]}>(fixtureJson);

function compareValues(
  testCase: ArealCase,
  metric: string,
  actual: number[],
  expected: number[],
  exact = false
): void {
  const scale = Math.max(1, ...expected.filter(Number.isFinite).map(Math.abs));
  const absolute = exact ? 0 : 1e-6 * scale;
  const relative = exact ? 0 : 1e-4;
  let maxAbs = 0;
  let maxRel = 0;
  let mismatches = Math.abs(actual.length - expected.length);
  expected.forEach((value, row) => {
    if (!Number.isFinite(value) || !Number.isFinite(actual[row])) {
      if (!Object.is(actual[row], value)) mismatches++;
      return;
    }
    const error = Math.abs(actual[row] - value);
    maxAbs = Math.max(maxAbs, error);
    maxRel = Math.max(maxRel, error / Math.max(Math.abs(value), Number.MIN_VALUE));
    if (error > absolute + relative * Math.abs(value)) mismatches++;
  });
  console.log(
    'PARITY',
    JSON.stringify({
      class: 'GPUArealInterpolation',
      case: testCase.name,
      reference: testCase.reference,
      n: testCase.width * testCase.height,
      metric,
      maxAbs,
      maxRel,
      mismatches,
      tolerance: {profile: exact ? 'G/exact-grid' : 'S', absolute, relative}
    })
  );
  expect(mismatches, `${testCase.name}: ${metric}`).toBe(0);
}

for (const testCase of fixture.cases) {
  for (const extensive of ['zone', 'overlap'] as const) {
    for (const intensive of ['zone', 'overlap'] as const) {
      for (const mode of ['extensive', 'intensive'] as const) {
        it(`GPUArealInterpolation Tobler parity: ${testCase.name}/${extensive}-${intensive}/${mode}`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const rig = new GraphRig(device, 'parity-wp1-areal');
          try {
            const pairCount = testCase.neighbors.length;
            const capacity = pairCount + 3;
            const offsets = rig.output('uint32', testCase.targetZoneCount + 1);
            const neighbors = rig.output('uint32', capacity);
            const weights = rig.output('float32', capacity);
            const alternateWeights = rig.output('float32', capacity);
            const areas = rig.output('float32', capacity);
            const sourceAreas = rig.output('float32', testCase.sourceZoneCount);
            const sourceTransferredFractions = rig.output('float32', testCase.sourceZoneCount);
            const overflow = rig.output('uint32', 1);
            const requiredPairCount = rig.output('uint32', 1);
            const invalidCount = rig.output('uint32', 1);
            const transferredExtensive = rig.output('float32', testCase.targetZoneCount);
            const transferredIntensive = rig.output('float32', testCase.targetZoneCount);
            const primary = {
              offsets: offsets.view,
              neighbors: neighbors.view,
              weights: weights.view
            };
            const alternate = {...primary, weights: alternateWeights.view};
            const expected = testCase.references[`${extensive}-${intensive}`];
            const comparisonCase = {
              ...testCase,
              name: `${testCase.name}/${extensive}-${intensive}/${mode}`
            };
            rig.run(
              new GPUArealInterpolation({
                grid: makeGPURasterMetadata({width: testCase.width, height: testCase.height}),
                sourceZones: rig.input(Uint32Array.from(testCase.sourceZones), 'uint32'),
                targetZones: rig.input(Uint32Array.from(testCase.targetZones), 'uint32'),
                sourceZoneCount: testCase.sourceZoneCount,
                targetZoneCount: testCase.targetZoneCount,
                // Explicitly exercise the default per-kind normalization as well.
                denominator:
                  extensive === 'zone' && intensive === 'overlap' && mode === 'extensive'
                    ? undefined
                    : {extensive, intensive},
                mode,
                // Exercise integer counts and generic sorted-area accumulation on the same support.
                unweightedFastPath: mode === 'extensive',
                output: {
                  weights: primary,
                  alternateWeights: alternateWeights.view,
                  areas: areas.view,
                  sourceAreas: sourceAreas.view,
                  sourceTransferredFractions: sourceTransferredFractions.view,
                  status: {
                    overflow: overflow.view,
                    requiredPairCount: requiredPairCount.view,
                    invalidCount: invalidCount.view
                  }
                }
              }),
              new GPUSpatialLag({
                id: 'parity-wp1-extensive',
                weights: mode === 'extensive' ? primary : alternate,
                values: rig.input(Float32Array.from(testCase.sourceValues), 'float32'),
                output: transferredExtensive.view
              }),
              new GPUSpatialLag({
                id: 'parity-wp1-intensive',
                weights: mode === 'intensive' ? primary : alternate,
                values: rig.input(Float32Array.from(testCase.sourceRates), 'float32'),
                output: transferredIntensive.view
              })
            );
            const actualWeights = (await weights.readFloat32()).slice(0, pairCount);
            const actualAlternate = (await alternateWeights.readFloat32()).slice(0, pairCount);
            const actualFractions = await sourceTransferredFractions.readFloat32();
            const actualExtensive = await transferredExtensive.readFloat32();
            const actualOffsets = await offsets.readUint32();
            compareValues(comparisonCase, 'csr-offsets', actualOffsets, testCase.offsets, true);
            compareValues(
              comparisonCase,
              'csr-neighbors',
              (await neighbors.readUint32()).slice(0, pairCount),
              testCase.neighbors,
              true
            );
            compareValues(
              comparisonCase,
              'pair-areas',
              (await areas.readFloat32()).slice(0, pairCount),
              testCase.areas,
              true
            );
            compareValues(
              comparisonCase,
              'source-areas',
              await sourceAreas.readFloat32(),
              testCase.sourceAreas,
              true
            );
            compareValues(
              comparisonCase,
              'status-overflow-required-invalid',
              [
                ...(await overflow.readUint32()),
                ...(await requiredPairCount.readUint32()),
                ...(await invalidCount.readUint32())
              ],
              [0, pairCount, testCase.invalidCount],
              true
            );
            compareValues(
              comparisonCase,
              'extensive-weights',
              mode === 'extensive' ? actualWeights : actualAlternate,
              expected.extensiveWeights
            );
            compareValues(
              comparisonCase,
              'intensive-weights',
              mode === 'intensive' ? actualWeights : actualAlternate,
              expected.intensiveWeights
            );
            compareValues(comparisonCase, 'target-extensive', actualExtensive, expected.extensive);
            compareValues(
              comparisonCase,
              'target-intensive',
              await transferredIntensive.readFloat32(),
              expected.intensive
            );
            compareValues(
              comparisonCase,
              'source-transferred-fractions',
              actualFractions,
              expected.sourceTransferredFractions
            );
            compareValues(
              comparisonCase,
              'source-lost-fractions',
              actualFractions.map(value => 1 - value),
              expected.sourceLostFractions
            );
            const transferredTotal = actualExtensive.reduce((sum, value) => sum + value, 0);
            compareValues(
              comparisonCase,
              'mass-transferred-and-lost',
              [
                transferredTotal,
                testCase.sourceValues.reduce((sum, value) => sum + value, 0) - transferredTotal
              ],
              [expected.massTotal, expected.massLost]
            );
            if (intensive === 'zone') {
              const intensiveWeights = mode === 'intensive' ? actualWeights : actualAlternate;
              const uncovered = Array.from(
                {length: testCase.targetZoneCount},
                (_, row) =>
                  1 -
                  intensiveWeights
                    .slice(actualOffsets[row], actualOffsets[row + 1])
                    .reduce((sum, value) => sum + value, 0)
              );
              compareValues(
                comparisonCase,
                'target-uncovered-fractions',
                uncovered,
                testCase.targetUncoveredFractions
              );
            }
          } finally {
            rig.destroy();
          }
        }, 120000);
      }
    }
  }
}
