// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPUGeographicDistribution,
  getGPUGeographicDistributionParameterValues
} from '../../../src/gpu-spatial-analysis/geographic-distribution';
import allFixture from '../../oracles/fixtures/parity-wp1-geographic-all.json';
import weightedFixture from '../../oracles/fixtures/parity-wp1-geographic-weighted-masked.json';
import smallFixture from '../../oracles/fixtures/parity-wp1-geographic-small.json';
import {decodeRound4Fixture, type GeographicCase} from '../../oracles/parity-wp1-round4-fixture';
import {createRound4Harness} from '../../oracles/parity-wp1-round4-harness';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';

const VERTEX_COUNT = 16;
for (const reduction of ['chunked', 'sorted'] as const) {
  for (const ellipseConvention of ['standard', 'arcgis'] as const) {
    for (const fixtureName of ['all', 'weighted', 'small'] as const) {
      it(`GPUGeographicDistribution pointpats parity ${fixtureName} ${reduction} ${ellipseConvention}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const testCase =
          fixtureName === 'small'
            ? (await decodeRound4Fixture<{cases: GeographicCase[]}>(smallFixture)).cases[0]
            : await decodeRound4Fixture<GeographicCase>(
                fixtureName === 'all' ? allFixture : weightedFixture
              );
        const harness = createRound4Harness(device);
        try {
          const groupCount = testCase.groupCount;
          const standardDeviations = ellipseConvention === 'arcgis' ? 2 : 1;
          const ellipseScale =
            standardDeviations * (ellipseConvention === 'arcgis' ? Math.SQRT2 : 1);
          harness.graph.add(
            new GPUGeographicDistribution({
              positions: harness.input('positions', testCase.positions, 'float32x2'),
              // Exercise the actual implicit-unit-weight path as well as explicit weights.
              weights:
                fixtureName === 'all'
                  ? undefined
                  : harness.input('weights', testCase.weights, 'float32'),
              groupIds: harness.input('group-ids', testCase.groups, 'uint32'),
              mask: harness.input('mask', testCase.mask, 'uint32'),
              groupCount,
              reduction,
              parameters: harness.input(
                'parameters',
                Array.from(
                  getGPUGeographicDistributionParameterValues({
                    ellipseConvention,
                    standardDeviations
                  })
                ),
                'float32'
              ),
              polygonVertexCount: VERTEX_COUNT,
              output: {
                counts: harness.output('counts', groupCount, 'uint32'),
                weightSums: harness.output('weightSums', groupCount, 'float32'),
                meanCenters: harness.output('meanCenters', groupCount * 2, 'float32x2'),
                standardDistances: harness.output('standardDistances', groupCount, 'float32'),
                ellipses: harness.output('ellipses', groupCount * 3, 'float32'),
                circleVertices: harness.output(
                  'circleVertices',
                  groupCount * VERTEX_COUNT * 2,
                  'float32x2'
                ),
                ellipseVertices: harness.output(
                  'ellipseVertices',
                  groupCount * VERTEX_COUNT * 2,
                  'float32x2'
                )
              }
            })
          );
          harness.run();
          const context = {...testCase, name: `${testCase.name}-${reduction}-${ellipseConvention}`};
          const expected = testCase.expected;
          compareParity(
            'GPUGeographicDistribution',
            context,
            'counts',
            await harness.read('counts'),
            expected.counts,
            'E'
          );
          compareParity(
            'GPUGeographicDistribution',
            context,
            'weightSums',
            await harness.read('weightSums'),
            expected.weightSums
          );
          compareParity(
            'GPUGeographicDistribution',
            context,
            'meanCenters',
            await harness.read('meanCenters'),
            expected.meanCenters
          );
          compareParity(
            'GPUGeographicDistribution',
            context,
            'standardDistances',
            await harness.read('standardDistances'),
            expected.standardDistances.map(value => value * standardDeviations)
          );
          const ellipses = await harness.read('ellipses');
          const actualAngles: number[] = [];
          const expectedAngles: number[] = [];
          const expectedCircle: number[] = [];
          const expectedEllipse: number[] = [];
          for (let group = 0; group < groupCount; group++) {
            if (expected.angleDefined[group] || expected.counts[group] === 0) {
              // Ellipse orientation is axial: compare double angles, so +/- pi
              // representations cannot produce a spurious wrap mismatch. Empty
              // groups still compare NaN masks; zero-spread orientation is undefined.
              actualAngles.push(
                Math.cos(2 * ellipses[group * 3]),
                Math.sin(2 * ellipses[group * 3])
              );
              expectedAngles.push(
                Math.cos(2 * expected.ellipses[group * 3]),
                Math.sin(2 * expected.ellipses[group * 3])
              );
            }
            const angle = expected.ellipses[group * 3];
            const shortAxis = expected.ellipses[group * 3 + 1] * ellipseScale;
            const longAxis = expected.ellipses[group * 3 + 2] * ellipseScale;
            const centerX = expected.meanCenters[group * 2];
            const centerY = expected.meanCenters[group * 2 + 1];
            for (let vertex = 0; vertex < VERTEX_COUNT; vertex++) {
              const phase = (vertex * 2 * Math.PI) / VERTEX_COUNT;
              const cosine = Math.cos(phase);
              const sine = Math.sin(phase);
              expectedCircle.push(
                centerX + expected.standardDistances[group] * standardDeviations * cosine,
                centerY + expected.standardDistances[group] * standardDeviations * sine
              );
              expectedEllipse.push(
                centerX + shortAxis * cosine * Math.cos(angle) - longAxis * sine * Math.sin(angle),
                centerY + shortAxis * cosine * Math.sin(angle) + longAxis * sine * Math.cos(angle)
              );
            }
          }
          compareParity(
            'GPUGeographicDistribution',
            context,
            'axial-angle-cos-sin',
            actualAngles,
            expectedAngles
          );
          for (const component of [1, 2]) {
            compareParity(
              'GPUGeographicDistribution',
              context,
              component === 1 ? 'short-axes' : 'long-axes',
              ellipses.filter((_, index) => index % 3 === component),
              expected.ellipses
                .filter((_, index) => index % 3 === component)
                .map(value => value * ellipseScale)
            );
          }
          compareParity(
            'GPUGeographicDistribution',
            context,
            'circleVertices',
            await harness.read('circleVertices'),
            expectedCircle
          );
          compareParity(
            'GPUGeographicDistribution',
            context,
            'ellipseVertices',
            await harness.read('ellipseVertices'),
            expectedEllipse
          );
        } finally {
          harness.destroy();
        }
      }, 120000);
    }
  }
}
