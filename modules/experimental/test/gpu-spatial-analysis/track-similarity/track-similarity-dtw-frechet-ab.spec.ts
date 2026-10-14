// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Time decision-only and distance-only graphs independently. The previous all-output graph
// included full Frechet and DTW, masking decision savings. Its jitter is less than sqrt(2), so
// both epsilon=1.5 and 1e6 are upper-bound accepts, not different free-space workloads.

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {createDynamicProgramHarness} from './track-similarity-dtw-frechet-harness';
import {createRandom} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import {requireTestDevice} from '../../utils/require-test-device';

it('GPUTrackSimilarity A/B: Frechet decision vs distance on 10k pairs of 200 vertices', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(17);
  const track = () => {
    let [x, y] = [random() * 1000, random() * 1000];
    return Array.from({length: 200}, () => {
      x += random() * 2;
      y += (random() - 0.5) * 2;
      return [x, y];
    });
  };
  const setA = Array.from({length: 10_000}, track);
  const setB = setA.map(points => points.map(([x, y]) => [x + random(), y + random()]));
  const lines: string[] = [];
  // Preserve the original accepted workload, then exercise endpoint and interior-box rejection.
  for (const scene of ['jitter', 'separated', 'excursion', 'dead-front'] as const) {
    // Equal boxes and endpoints, but opposite interior order: epsilon=1.5 has no path.
    const sources =
      scene === 'dead-front'
        ? setA.map(() =>
            Array.from({length: 200}, (_, vertex) =>
              vertex === 0 || vertex === 199 ? [0, 0] : vertex < 100 ? [1000, 0] : [0, 1000]
            )
          )
        : setA;
    const targets =
      scene === 'dead-front'
        ? sources.map(points => points.map(([x, y]) => [y, x]))
        : scene === 'jitter'
          ? setB
          : setA.map(points =>
              points.map(([x, y], vertex) => [
                x,
                y + (scene === 'separated' || (vertex > 0 && vertex < 199) ? 1000 : 0)
              ])
            );
    for (const outputMode of ['distance', 'decision'] as const) {
      const harness = createDynamicProgramHarness(device, sources, targets, {outputMode});
      try {
        for (const epsilon of outputMode === 'distance' ? [1.5] : [1e6, 1.5]) {
          await harness.time(epsilon);
          const times: number[] = [];
          for (let encoding = 0; encoding < 5; encoding++) {
            times.push(await harness.time(epsilon));
          }
          times.sort((left, right) => left - right);
          lines.push(
            `track-similarity-ab scene=${scene} output=${outputMode} epsilon=${epsilon} median=${times[2].toFixed(2)} ms`
          );
          if (outputMode === 'decision') {
            const result = await harness.run(epsilon);
            const expected = epsilon === 1e6 || scene === 'jitter' ? 1 : 0;
            expect(result.within.every(value => value === expected)).toBe(true);
          }
        }
      } finally {
        harness.destroy();
      }
    }
  }
  // eslint-disable-next-line no-console
  console.log(lines.join('\n'));
  expect(lines.length).toBe(12);
}, 600000);
