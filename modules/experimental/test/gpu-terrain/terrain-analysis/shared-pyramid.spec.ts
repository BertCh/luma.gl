// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPU_TERRAIN_VISIBILITY as V} from '../../../src/gpu-terrain/terrain-analysis/index';
import {
  createSharedPyramidScene,
  type SharedPyramidMode,
  type SharedPyramidOutputs
} from './shared-pyramid-fixture';

const WIDTH = 96;
const HEIGHT = 80;

async function runScene(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  mode: SharedPyramidMode
): Promise<{outputs: SharedPyramidOutputs; nodeCount: number}> {
  const scene = createSharedPyramidScene(device, {width: WIDTH, height: HEIGHT, mode});
  const encoding = scene.run();
  const outputs = await scene.read();
  const nodeCount = encoding.stats.nodeCount;
  scene.destroy();
  return {outputs, nodeCount};
}

it('shared pyramid results are bit-identical to march and to self-built pyramids', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const march = await runScene(device, 'march');
  const self = await runScene(device, 'self');
  const shared = await runScene(device, 'shared');
  const prebuilt = await runScene(device, 'prebuilt');

  // Non-trivial structure: a WGSL compile error would give silent zeros.
  const hidden = march.outputs.viewshed.filter(code => code === V.hidden).length;
  const visible = march.outputs.viewshed.filter(code => code === V.visible).length;
  expect(hidden).toBeGreaterThan(0);
  expect(visible).toBeGreaterThan(0);
  expect(march.outputs.cumulative.some(count => count > 0)).toBe(true);
  expect(march.outputs.profileDistance.some(distance => distance > 0)).toBe(true);
  expect(new Set(march.outputs.lineOfSightVisibility).size).toBeGreaterThan(1);

  expect(self.outputs).toEqual(march.outputs);
  expect(shared.outputs).toEqual(march.outputs);
  expect(shared.outputs).toEqual(self.outputs);
  expect(prebuilt.outputs).toEqual(march.outputs);

  // One pyramid instead of five: the shared graph schedules fewer nodes than five self-built ones,
  // and a prebuilt pyramid schedules fewer still.
  expect(shared.nodeCount).toBeLessThan(self.nodeCount);
  expect(prebuilt.nodeCount).toBeLessThan(shared.nodeCount);
});
