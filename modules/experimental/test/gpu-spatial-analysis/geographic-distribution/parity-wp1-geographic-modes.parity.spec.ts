// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUGeographicDistribution,
  getGPUGeographicDistributionParameterValues
} from '../../../src/gpu-spatial-analysis/geographic-distribution';
import fixture from '../../oracles/fixtures/parity-wp1-round8-geographic-modes.json';
import {compareReference} from '../../oracles/parity-wp1-round8-compare';
import {createRound4Harness} from '../../oracles/parity-wp1-round4-harness';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of fixture.cases)
  for (const reduction of ['auto', 'chunked', 'sorted'] as const) {
    it(`GPUGeographicDistribution independent ${scene.name}/${reduction}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createRound4Harness(device);
      try {
        harness.graph.add(
          new GPUGeographicDistribution({
            positions: harness.input('positions', scene.positions, 'float32x2'),
            lineEnds: harness.input('lineEnds', scene.lineEnds, 'float32x2'),
            weights: harness.input('weights', scene.weights, 'float32'),
            parameters: harness.input(
              'parameters',
              Array.from(
                getGPUGeographicDistributionParameterValues({
                  orientationOnly: scene.orientationOnly,
                  medianTolerance: 1e-6
                })
              ),
              'float32'
            ),
            maximumMedianIterations: 256,
            reduction,
            output: {
              medianCenters: harness.output('median', 2, 'float32x2'),
              medianConverged: harness.output('converged', 1, 'uint32'),
              directionalMeans: harness.output('directional', 3, 'float32')
            }
          })
        );
        harness.run();
        const compare = (metric: string, actual: number[], expected: number[], exact = false) =>
          compareReference(
            'GPUGeographicDistribution',
            {...scene, name: `${scene.name}/${reduction}`},
            metric,
            actual,
            expected,
            exact
          );
        compare('median-location', await harness.read('median'), scene.median);
        compare('median-converged', await harness.read('converged'), [1], true);
        compare(
          'directional-mean-variance-length',
          await harness.read('directional'),
          scene.directional.map(value => value ?? NaN)
        );
      } finally {
        harness.destroy();
      }
    });
  }
