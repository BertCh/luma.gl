// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  getClassBreaksMomentNodes,
  getClassBreaksMomentTileCount,
  MOMENT_STRIDE,
  CLASS_BREAKS_STATE_LENGTH
} from '../../../src/gpu-dataframe/column-classification/class-breaks-kernels';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import {ParityReductionHarness} from '../../oracles/parity-wp6-reduction-harness';
import fixtureJson from '../../oracles/fixtures/parity-wp6-moment-diagnostics.json';

type Scene = {
  name: string;
  values: number[];
  mask: number[];
  expected: number[];
  tiles: number[][];
};
const fixture = decodeOracleFixture<{reference: string; cases: Scene[]}>(fixtureJson);
for (const scene of fixture.cases) {
  it(`GPUClassBreaks internal moments pandas parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new ParityReductionHarness(device);
    const values = harness.input('values', Float32Array.from(scene.values), 'float32');
    const mask = harness.input('mask', Uint32Array.from(scene.mask), 'uint32');
    const state = harness.input('state', new Uint32Array(CLASS_BREAKS_STATE_LENGTH), 'uint32');
    const partials = harness.output(
      'partials',
      getClassBreaksMomentTileCount(scene.values.length) * MOMENT_STRIDE,
      'float32'
    );
    const moments = harness.output('moments', MOMENT_STRIDE, 'float32');
    // Read the actual GPU moment kernel directly: absolute class edges at a large offset
    // could hide an inaccurate spread under an edge-relative tolerance.
    harness.graph.add({
      getCommandNodes: graph =>
        getClassBreaksMomentNodes(graph, {
          id: 'offset-moments',
          operation: 'GPUClassBreaks',
          values,
          mask,
          state,
          partials,
          moments,
          slot: 0,
          gate: {kind: 'method', methodCode: 0}
        })
    });
    let compiled: ReturnType<typeof harness.graph.compile> | undefined;
    try {
      compiled = harness.graph.compile();
      const translated = fixture.cases.find(
        candidate => candidate.name === `${scene.name}-translated`
      );
      // Reuse the same graph and buffers across an exact translation and back. A cached or
      // stale pivot must fail the independently generated tile/final means as well as M2.
      const encodings = translated ? [scene, translated, scene] : [scene, scene];
      for (const [encoding, activeScene] of encodings.entries()) {
        harness.buffers[0].write(Float32Array.from(activeScene.values));
        submitGraph(device, compiled, undefined);
        const outputs = await harness.read();
        const actual = outputs['moments'];
        // Tile M2 separates sequential/workgroup arithmetic from the final tile merge.
        // Expected moments always use the already-f32-rounded uploaded values; 2^24 input
        // quantization cannot justify widening the spread tolerance.
        for (const [tileIndex, expected] of activeScene.tiles.entries()) {
          const tile = outputs['partials'].slice(
            tileIndex * MOMENT_STRIDE,
            (tileIndex + 1) * MOMENT_STRIDE
          );
          tile[1] += tile[5];
          for (const [index, metric] of ['count', 'mean', 'M2', 'minimum', 'maximum'].entries()) {
            const exact = index === 0 || index >= 3;
            compareParity(
              'GPUClassBreaks',
              `${activeScene.name}/encoding=${encoding}/tile=${tileIndex}`,
              fixture.reference,
              scene.values.length,
              metric,
              [tile[index]],
              [exact ? Math.fround(expected[index]) : expected[index]],
              exact
            );
          }
        }
        for (const [index, metric] of ['count', 'mean', 'M2', 'minimum', 'maximum'].entries()) {
          const exact = index === 0 || index >= 3;
          compareParity(
            'GPUClassBreaks',
            `${activeScene.name}/encoding=${encoding}`,
            fixture.reference,
            scene.values.length,
            metric,
            [actual[index]],
            [exact ? Math.fround(activeScene.expected[index]) : activeScene.expected[index]],
            exact
          );
        }
      }
    } finally {
      compiled?.destroy();
      harness.destroy();
    }
  }, 120000);
}
